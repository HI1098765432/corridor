"""Move a new image onto the contrast the model can actually see.

The measured cause of the held-out collapse is that the model peaks at the
contrast it was trained on and falls away in both directions -- recall 0.829 at
native, 0.098 at 0.25x, 0.512 at 2.5x, with the cells otherwise untouched. The
two halves of this dataset sit at different points on that curve: KK1 cells have
median contrast 0.184 (54% below 0.20), KK2 cells 0.385 (15% below).

Retraining is the durable fix and it is running. But the same diagnosis suggests
something that needs no training at all: **if the model can only see cells in a
narrow band, put the image's cells in that band before asking.**

Four ways to try it, in increasing presumption:

1. ``baseline`` -- what cellpose does by default, percentiles (1, 99).
2. ``percentile`` -- a narrower window, already measured to lift KK1Model on KK2
   from 0.482 to 0.653. Included as the number to beat.
3. ``histogram`` -- match the whole intensity distribution to the training
   half's average. Classic domain adaptation, and it uses only the training
   images, which the model has already seen, so it leaks nothing.
4. ``gain`` -- estimate how much the image's local structure would have to be
   amplified for a typical cell to reach the model's peak contrast, and apply
   exactly that. This uses the contrast curve as a calibration rather than as an
   explanation.

Nothing here alters the labels or the model. If a transform helps, it helps
because the image was outside what the network can respond to, which is the
claim being tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
MODELS = {
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
    "combi": TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi",
}
#: Where the combined model's recall peaked, from the frozen-window sweep.
PEAK_CONTRAST = 0.158


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


def reference_histogram(paths: list[Path], bins: int = 512):
    """The pooled intensity distribution of a training half, as a CDF."""
    samples = []
    for path in paths:
        image = load(path)
        lo, hi = np.percentile(image, [0.1, 99.9])
        samples.append(np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1).ravel()[::7])
    pooled = np.concatenate(samples)
    counts, edges = np.histogram(pooled, bins=bins, range=(0, 1))
    cdf = np.cumsum(counts).astype(np.float64)
    cdf /= cdf[-1]
    return cdf, edges


def match_to(image: np.ndarray, cdf, edges) -> np.ndarray:
    """Re-map an image's intensities so its distribution matches the reference."""
    lo, hi = np.percentile(image, [0.1, 99.9])
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    source_counts, _ = np.histogram(scaled, bins=len(cdf), range=(0, 1))
    source_cdf = np.cumsum(source_counts).astype(np.float64)
    source_cdf /= source_cdf[-1]
    centres = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.interp(scaled.ravel(), centres, source_cdf)
    return np.interp(quantiles, cdf, centres).reshape(image.shape).astype(np.float32)


def local_structure_gain(image: np.ndarray, target: float = PEAK_CONTRAST) -> np.ndarray:
    """Amplify local structure until typical objects sit at the model's peak.

    The amount is estimated from the image alone -- how far the brightest local
    departures from the background already are -- so nothing about the labels
    or the test set is used.
    """
    from scipy.ndimage import median_filter

    background = median_filter(image, size=(1, 41), mode="nearest")
    delta = image - background
    lo, hi = np.percentile(image, [1, 99])
    span = max(float(hi - lo), 1e-6)
    # The typical strong departure, as a fraction of the image's range.
    present = float(np.percentile(np.abs(delta), 99.5)) / span
    if present <= 1e-6:
        return image
    gain = float(np.clip(target / present, 0.3, 4.0))
    return background + gain * delta


def evaluate(model, paths: list[Path], transform=None, normalize=True) -> dict:
    """Score a transform, telling cellpose whether it still needs to normalise.

    This distinction is the whole experiment. A transform that maps the image to
    [0, 1] has *already* normalised it; letting cellpose percentile-normalise
    the result a second time largely undoes the transform, and the measurement
    then says the idea failed when what failed was the plumbing. Measured: the
    (3, 97) window scores 0.134 when double-normalised and 0.653 when applied
    once -- the same transform, a factor of five apart.
    """
    from scipy.optimize import linear_sum_assignment

    tp = fp = fn = 0
    for path in paths:
        image = load(path)
        if transform is not None:
            image = transform(image)
        truth = truth_of(path)
        pred = np.asarray(
            model.eval(image, channels=[0, 0], diameter=None,
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
        hits = sum(1 for r, c in zip(rows, cols) if iou[r, c] >= 0.5)
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
    ap.add_argument("--model", default="KK1",
                    help="a name from MODELS, or a path to a trained checkpoint")
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--train-group", default="KK1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    weights = MODELS.get(args.model) or Path(args.model)
    if not Path(weights).exists():
        raise SystemExit(f"no such model: {weights}")
    model = cp.CellposeModel(pretrained_model=str(weights), gpu=False)
    test_paths = labelled(args.test_group)
    train_paths = labelled(args.train_group)
    cdf, edges = reference_histogram(train_paths)

    def percentile_window(image):
        lo, hi = np.percentile(image, [3, 97])
        return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    # (transform, does the transform already produce a normalised image?)
    # A percentile window goes through cellpose's OWN normalise step, never by
    # pre-clipping here. Clipping looks equivalent and is not: these cells sit
    # in the tail of the intensity distribution -- 8-21% of their pixels fall
    # outside [p1, p99] -- so clipping at p97 deletes the brightest part of
    # every cell. Measured, that is F1 0.134 against 0.653 for the same window.
    #
    # Each entry is (array transform, the value for cellpose's `normalize`).
    strategies = {
        "baseline (cellpose default)": (None, True),
        "percentile window (3, 97)": (None, {"percentile": (3.0, 97.0)}),
        "percentile window (2, 98)": (None, {"percentile": (2.0, 98.0)}),
        "percentile window (5, 95)": (None, {"percentile": (5.0, 95.0)}),
        "percentile window (10, 90)": (None, {"percentile": (10.0, 90.0)}),
        f"histogram matched to {args.train_group}": (
            lambda im: match_to(im, cdf, edges), True),
        "local gain to the model's peak": (local_structure_gain, True),
        "histogram + window (3, 97)": (
            lambda im: match_to(im, cdf, edges), {"percentile": (3.0, 97.0)}),
    }

    print(f"model {args.model} on {args.test_group} ({len(test_paths)} images), "
          f"no training, input transforms only\n")
    print(f"{'transform':36s} {'P':>7} {'R':>7} {'F1':>7}")

    results = {}
    for name, (transform, norm) in strategies.items():
        scores = evaluate(model, test_paths, transform, normalize=norm)
        results[name] = scores
        print(f"{name:36s} {scores['precision']:7.4f} {scores['recall']:7.4f} "
              f"{scores['f1']:7.4f}", flush=True)

    best = max(results.items(), key=lambda kv: kv[1]["f1"])
    base = results["baseline (cellpose default)"]["f1"]
    print()
    print(f"best: {best[0]} at F1 {best[1]['f1']:.4f} "
          f"({best[1]['f1'] - base:+.4f} over the default)")

    out = Path(args.out) if args.out else (
        ROOT / "docs" / f"input_matching_{args.model}_on_{args.test_group}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "test_group": args.test_group, "results": results},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
