"""Teach the model that a cell is a cell at any contrast.

The measurement this exists to act on (`scripts/experiment_contrast_response.py`)
is that recall peaks at exactly the contrast the model was trained on -- 0.829 at
native, 0.098 at 0.25x, 0.463 at 2.5x -- with the cells unchanged in position,
shape and noise, on images the model was trained on. The network learned
"a cell looks like *this*, at *this* brightness". It is the common cause of the
held-out collapse, of the flat probability field on a miss, and of why eight
inference-time strategies all stalled at F1 0.843: every one of them changed the
decision rule, and none changed what the network is tuned to.

So the intervention is to show it the same cells across a wide band of contrast
and let it discover that brightness is not what makes a cell. Three things are
fixed at once, because any of them alone would confound the result:

1. **Contrast randomisation**, by the same delta-field transform used to measure
   the problem. The cell's perturbation on its background is rescaled; nothing
   else moves. This is not generic augmentation -- it is aimed at a measured
   axis, and the augmented images are real images with real labels.
2. **`min_train_masks=0`**, because the default of 5 discards 48 of the 71
   labelled images and 46% of every hand-drawn instance, including all nine
   empty-channel negatives.
3. **Thread pinning**, because torch's default oversubscribes this CPU and costs
   roughly 6x wall-clock for identical arithmetic.

The evaluation is deliberately the harshest available one: train on KK1, score
on KK2, which is a different acquisition day the model never sees. The published
baseline for exactly that comparison is F1 0.482, and it was produced by the same
data without the contrast augmentation -- so the difference is attributable.
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
#: Starting weights per training group. The combined model is deliberately NOT
#: the default: it was trained on all 71 images, so evaluating it on KK2 scores
#: a model that memorised KK2 (F1 0.796, measured) and any "improvement" from
#: there would be meaningless. Initialising from the model trained on the SAME
#: half we train on keeps the test half genuinely unseen, and makes the result
#: directly comparable with the published KK1Model -> KK2 baseline of 0.482.
START_WEIGHTS = {
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
}
COMBI = TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"

#: The band to train across. The measured response is already down at 0.25x and
#: 2.5x, so covering [0.35, 2.6] asks the network to hold on to a cell roughly
#: 7x either side of where it currently peaks.
CONTRAST_RANGE = (0.35, 2.6)
#: Augmented copies per source image. Each is a different draw from the band, so
#: the same cell is seen at several brightnesses.
COPIES_PER_IMAGE = 3

#: The scale band to train across. Measured: sweeping the diameter Cellpose is
#: told to assume moves held-out F1 by +0.032 (KK1->KK2) and +0.036 (KK2->KK1),
#: both peaking at 36 against the checkpoint's own 31.2. The same optimum in the
#: direction that was not tuned on means the model is scale-sensitive as well as
#: contrast-sensitive, so the same remedy applies: show it the cells at several
#: sizes rather than hope the right one is guessed at inference.
SCALE_RANGE = (0.75, 1.35)


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


def scale_variant(image: np.ndarray, masks: np.ndarray, factor: float):
    """The same field at a different magnification, image and labels together.

    Resizing both with the same factor keeps every label exactly correct -- this
    is not pseudo-labelling and carries none of its risk. The image is resampled
    smoothly and the masks by nearest neighbour, because interpolating a label
    image invents instance ids that never existed.
    """
    import cv2

    height, width = image.shape[:2]
    new = (max(32, int(round(width * factor))), max(32, int(round(height * factor))))
    resized = cv2.resize(image, new, interpolation=cv2.INTER_LINEAR)
    labels = cv2.resize(masks.astype(np.int32), new, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32), labels.astype(np.int32)


def contrast_variant(image: np.ndarray, masks: np.ndarray, scale: float) -> np.ndarray:
    """The same cells, the same everywhere else, at a different contrast."""
    from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

    background = median_filter(image, size=(1, 41), mode="nearest")
    grown = binary_dilation(masks > 0, iterations=8).astype(np.float32)
    support = np.clip(gaussian_filter(grown, sigma=3.0) * 1.6, 0.0, 1.0)
    return image + (scale - 1.0) * support * (image - background)


def masks_for(path: Path, corrected: Path | None) -> np.ndarray:
    """The corrected labels for this image when they exist, else the originals."""
    if corrected is not None:
        candidate = corrected / f"{path.stem}_masks.npy"
        if candidate.exists():
            return np.load(candidate).astype(np.int32)
    return truth_of(path)


def build_training_set(paths: list[Path], rng, *, copies: int, corrected=None,
                       scale_range=None):
    """Original images plus contrast-randomised copies, labels unchanged.

    The labels are reused verbatim, which is the point: the transform moves no
    boundary, so a mask that was correct stays exactly correct. This is not
    pseudo-labelling and carries none of its risk.
    """
    images, masks = [], []
    for path in paths:
        image = load(path)
        truth = masks_for(path, corrected)
        images.append(image)
        masks.append(truth)
        if not (truth > 0).any():
            # An empty frame has no cell perturbation to rescale, so a copy
            # would be the identical image. Kept once, as a negative.
            continue
        for _ in range(copies):
            lo, hi = CONTRAST_RANGE
            scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            varied = contrast_variant(image, truth, scale)
            varied_masks = truth
            if scale_range is not None:
                slo, shi = scale_range
                factor = float(np.exp(rng.uniform(np.log(slo), np.log(shi))))
                varied, varied_masks = scale_variant(varied, truth, factor)
            images.append(varied)
            masks.append(varied_masks)
    return images, masks


def evaluate(model, paths: list[Path], iou_match: float = 0.5, normalize=True) -> dict:
    from scipy.optimize import linear_sum_assignment

    tp = fp = fn = 0
    for path in paths:
        truth = truth_of(path)
        pred = np.asarray(
            model.eval(load(path), channels=[0, 0], diameter=None,
                       cellprob_threshold=0.0, flow_threshold=0.4,
                       normalize=normalize)[0]
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
        hits = sum(1 for r, c in zip(rows, cols) if iou[r, c] >= iou_match)
        tp += hits
        fp += len(p) - hits
        fn += len(t) - hits

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-group", default="KK1")
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--copies", type=int, default=COPIES_PER_IMAGE)
    ap.add_argument("--learning-rate", type=float, default=0.0002)
    ap.add_argument("--no-augment", action="store_true",
                    help="the control: identical run without contrast randomisation")
    ap.add_argument("--scale-aug", action="store_true",
                    help="also vary magnification, the second measured axis")
    ap.add_argument("--window", default="",
                    help="percentile window used for BOTH training and evaluation, "
                         "e.g. 3,97. A model evaluated through a window it never "
                         "trained under is being asked about an input distribution "
                         "it has not seen.")
    ap.add_argument("--start-from", default="",
                    help="continue from a checkpoint instead of the original weights")
    ap.add_argument("--corrected-labels", default="",
                    help="directory of *_masks.npy with the annotator's missed cells "
                         "restored. Each added cell is positioned by interpolating two "
                         "human centroids and outlined with a human mask from an "
                         "adjacent frame, so training on them is not self-supervision.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)

    from cellpose import models as cp, train as cp_train

    rng = np.random.default_rng(0)
    train_paths = labelled(args.train_group)
    test_paths = labelled(args.test_group)

    copies = 0 if args.no_augment else args.copies
    corrected = Path(args.corrected_labels) if args.corrected_labels else None
    if corrected is not None:
        restored = len(list(corrected.glob("*_masks.npy")))
        print(f"using corrected labels from {corrected} ({restored} images)")
    images, masks = build_training_set(
        train_paths, rng, copies=copies, corrected=corrected,
        scale_range=SCALE_RANGE if args.scale_aug else None)
    if args.scale_aug:
        print(f"scale randomisation over {SCALE_RANGE[0]}x-{SCALE_RANGE[1]}x")
    print(f"training on {args.train_group}: {len(train_paths)} source images "
          f"-> {len(images)} after {copies} contrast copies each")
    print(f"evaluating on {args.test_group}: {len(test_paths)} images "
          f"(a different acquisition day, never seen)")

    window = True
    if args.window:
        lo, hi = (float(v) for v in args.window.split(","))
        window = {"percentile": (lo, hi)}
        print(f"normalisation window {lo}-{hi} applied to training AND evaluation")

    start = Path(args.start_from) if args.start_from else START_WEIGHTS.get(
        args.train_group, COMBI)
    print(f"starting from {start.parent.parent.name}, which never saw {args.test_group}")
    model = cp.CellposeModel(pretrained_model=str(start), gpu=False)
    # Cellpose enables MKL-DNN on CPU because it speeds up inference. It cannot
    # back-propagate through weights held as MKLDNN tensors, so training dies
    # with "The MKLDNN backend does not support weight as an MKLDNN tensor
    # during training". It is an eval-path optimisation and is simply switched
    # off while the weights are being changed.
    model.net.mkldnn = False

    print("\nbefore training:")
    before = evaluate(model, test_paths, normalize=window)
    print(f"  {before}")

    started = time.time()
    tag = "contrast_invariant" if copies else "control_no_augment"
    save_path = ROOT / "build" / "models"
    save_path.mkdir(parents=True, exist_ok=True)

    cp_train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        channels=[0, 0],
        normalize=window,
        # The default of 5 would discard 68% of these images, including every
        # empty-channel negative. See scripts/diag_training_harness.py.
        min_train_masks=0,
        # Cellpose rescales each image by the equivalent-disk diameter of its
        # masks, which for an elongated confined cell is dominated by length.
        # That jitters the one quantity the device holds constant -- the 12 px
        # width -- so it is switched off and the known scale used instead.
        rescale=False,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=8,
        weight_decay=1e-5,
        SGD=False,
        save_path=str(save_path),
        model_name=f"corridor_{tag}{'_w' + args.window.replace(',', '_') if args.window else ''}",
        save_every=max(args.epochs, 1),
    )
    elapsed = time.time() - started
    print(f"\ntrained in {elapsed / 60:.1f} min")

    print("\nafter training:")
    after = evaluate(model, test_paths, normalize=window)
    print(f"  {after}")

    report = {
        "train_group": args.train_group,
        "test_group": args.test_group,
        "augmented": bool(copies),
        "copies_per_image": copies,
        "contrast_range": list(CONTRAST_RANGE) if copies else None,
        "source_images": len(train_paths),
        "training_images": len(images),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "minutes": round(elapsed / 60, 1),
        "before": before,
        "after": after,
    }
    print()
    print(f"{'':10s} {'precision':>10} {'recall':>8} {'F1':>8}")
    print(f"{'before':10s} {before['precision']:10.4f} {before['recall']:8.4f} {before['f1']:8.4f}")
    print(f"{'after':10s} {after['precision']:10.4f} {after['recall']:8.4f} {after['f1']:8.4f}")
    delta = after["f1"] - before["f1"]
    print(f"{'change':10s} {'':10s} {'':8s} {delta:+8.4f}")

    out = Path(args.out) if args.out else ROOT / "docs" / f"train_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
