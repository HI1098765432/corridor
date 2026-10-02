"""Run every training variant to the same standard and report one table.

One run does not settle anything. The contrast-augmentation run changes several
things at once -- augmentation, ``min_train_masks=0``, ``rescale=False`` -- so a
gain from it is not attributable until the control has run with the identical
budget. This does the whole ladder unattended and prints the comparison.

**Two directions, because they are not interchangeable.**

*Train KK1, test KK2.* This is the published held-out comparison (KK1Model on
KK2 = F1 0.482), so any variant here is directly comparable with a number that
already exists. Only labelled data is used.

*Train KK2, test KK1.* Required for anything that touches the unlabelled
stacks: every one of them is named ``052924_*``, the same acquisition day as 31
of the 32 KK2 labels. Using them and then reporting a KK2 score would be
training on the test set by a side door. KK1 is four other days and is clean.

Each variant starts from the model trained on its own half, never from the
combined model -- that one saw all 71 images, scores 0.796 on KK2, and any
"improvement" measured from it would be measuring memorisation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
START_WEIGHTS = {
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
}

#: Published baselines for the two held-out directions, from
#: docs/model_evaluation.json. Every variant is measured against these.
PUBLISHED = {("KK1", "KK2"): 0.482, ("KK2", "KK1"): 0.303}


def load(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image.astype(np.float32)


def truth_of(path: Path) -> np.ndarray:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


def labelled(group: str) -> list[Path]:
    return [
        p for p in sorted((TRAIN / group).glob("*.tif"))
        if p.with_name(p.name.replace(".tif", "_seg.npy")).exists()
    ]


def contrast_variant(image: np.ndarray, masks: np.ndarray, scale: float) -> np.ndarray:
    """The same cells at a different contrast; the label stays exactly correct."""
    from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

    background = median_filter(image, size=(1, 41), mode="nearest")
    grown = binary_dilation(masks > 0, iterations=8).astype(np.float32)
    support = np.clip(gaussian_filter(grown, sigma=3.0) * 1.6, 0.0, 1.0)
    return image + (scale - 1.0) * support * (image - background)


def evaluate(model, paths: list[Path]) -> dict:
    from scipy.optimize import linear_sum_assignment

    tp = fp = fn = 0
    for path in paths:
        truth = truth_of(path)
        pred = np.asarray(
            model.eval(load(path), channels=[0, 0], diameter=None,
                       cellprob_threshold=0.0, flow_threshold=0.4)[0]
        ).astype(np.int32)
        t = [int(v) for v in np.unique(truth) if v]
        p = [int(v) for v in np.unique(pred) if v]
        if not t or not p:
            fp += len(p)
            fn += len(t)
            continue
        ti = {l: i for i, l in enumerate(t)}
        pi = {l: i for i, l in enumerate(p)}
        inter = np.zeros((len(t), len(p)), np.int64)
        both = (truth > 0) & (pred > 0)
        for tv, pv in zip(truth[both], pred[both]):
            inter[ti[int(tv)], pi[int(pv)]] += 1
        ta = np.array([(truth == l).sum() for l in t], np.int64)
        pa = np.array([(pred == l).sum() for l in p], np.int64)
        iou = inter / np.maximum(ta[:, None] + pa[None, :] - inter, 1)
        rows, cols = linear_sum_assignment(-iou)
        hits = sum(1 for r, c in zip(rows, cols) if iou[r, c] >= 0.5)
        tp += hits
        fp += len(p) - hits
        fn += len(t) - hits

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


#: (name, copies, contrast band). copies=0 is the control: identical harness
#: fixes, identical budget, no contrast randomisation, so the difference
#: between it and the others is attributable to the augmentation alone.
VARIANTS = [
    ("control (harness fixes only)", 0, None),
    ("contrast x0.5-2.0", 2, (0.5, 2.0)),
    ("contrast x0.35-2.6", 2, (0.35, 2.6)),
    ("contrast x0.25-4.0 wide", 3, (0.25, 4.0)),
]


def build(paths: list[Path], rng, copies: int, band):
    images, masks = [], []
    for path in paths:
        image = load(path)
        truth = truth_of(path)
        images.append(image)
        masks.append(truth)
        if copies and (truth > 0).any():
            lo, hi = band
            for _ in range(copies):
                scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                images.append(contrast_variant(image, truth, scale))
                masks.append(truth)
    return images, masks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-group", default="KK1")
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.0002)
    ap.add_argument("--only", default="", help="substring: run just one variant")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp, train as cp_train

    train_paths = labelled(args.train_group)
    test_paths = labelled(args.test_group)
    start = START_WEIGHTS[args.train_group]
    baseline = PUBLISHED.get((args.train_group, args.test_group))

    print(f"train {args.train_group} ({len(train_paths)} images) -> "
          f"test {args.test_group} ({len(test_paths)} images)")
    print(f"starting weights: {start.parent.parent.name}, which never saw {args.test_group}")
    if baseline:
        print(f"published baseline for this direction: F1 {baseline}")
    print()

    results = []
    for name, copies, band in VARIANTS:
        if args.only and args.only.lower() not in name.lower():
            continue
        rng = np.random.default_rng(0)
        images, masks = build(train_paths, rng, copies, band)

        model = cp.CellposeModel(pretrained_model=str(start), gpu=False)
        model.net.mkldnn = False  # cannot back-propagate through MKLDNN weights

        before = evaluate(model, test_paths)
        started = time.time()
        cp_train.train_seg(
            model.net,
            train_data=images,
            train_labels=masks,
            channels=[0, 0],
            normalize=True,
            min_train_masks=0,   # the default of 5 discards 68% of these images
            rescale=False,       # cell LENGTH would otherwise jitter the scale
            learning_rate=args.learning_rate,
            n_epochs=args.epochs,
            batch_size=8,
            weight_decay=1e-5,
            SGD=False,
            save_path=str(ROOT / "build" / "models"),
            model_name=f"ladder_{args.train_group}_{name.split()[0]}_{copies}",
            save_every=max(args.epochs, 1),
        )
        minutes = (time.time() - started) / 60.0
        after = evaluate(model, test_paths)

        row = {
            "variant": name, "copies": copies, "band": list(band) if band else None,
            "training_images": len(images), "minutes": round(minutes, 1),
            "before": before, "after": after,
            "delta_f1": round(after["f1"] - before["f1"], 4),
        }
        results.append(row)
        print(f"{name:30s} n={len(images):4d} {minutes:6.1f}min  "
              f"F1 {before['f1']:.4f} -> {after['f1']:.4f}  "
              f"(P {after['precision']:.3f} R {after['recall']:.3f})", flush=True)

        out = Path(args.out) if args.out else (
            ROOT / "docs" / f"train_ladder_{args.train_group}_to_{args.test_group}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"direction": f"{args.train_group}->{args.test_group}",
             "published_baseline": baseline,
             "epochs": args.epochs, "results": results}, indent=2), encoding="utf-8")

    print()
    print(f"{'variant':30s} {'F1 before':>10} {'F1 after':>9} {'vs published':>13}")
    for row in results:
        against = (f"{row['after']['f1'] - baseline:+.4f}" if baseline else "-")
        print(f"{row['variant']:30s} {row['before']['f1']:10.4f} "
              f"{row['after']['f1']:9.4f} {against:>13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
