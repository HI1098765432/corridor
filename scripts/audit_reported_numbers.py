"""Check the three things that would invalidate the reported 0.726.

An external review raised three objections, each of which would make the figure
wrong rather than merely imprecise. None is answered by argument; all three are
answered by running something.

**1. Does the saved checkpoint reproduce the score?** The number was measured on
the in-memory network straight after training. A model is only real once it has
been written to disk and loaded back: cellpose stores ``diam_labels`` and other
state in the checkpoint, and a reloaded model can rescale its input differently
from the object that produced the number.

**2. Does the instance matcher undercount?** Both the evaluator here and
``scripts/evaluate_models.py`` solve ``linear_sum_assignment`` on IoU and *then*
apply the 0.5 threshold. That maximises total IoU, which is not the same as
maximising the number of valid matches, so a high-overlap pair can capture a
prediction that a second truth needed. This measures how often the two
disagree **on the real data**, rather than arguing about whether a
counterexample exists.

**3. Is the gain just more gradient steps?** The augmented run trained on 112
images for 60 epochs; the unaugmented baseline it is compared against did not
train at all. Three times the updates is an alternative explanation for the
whole effect, and until a matched-budget control has run, the augmentation is
not the attributed cause.

This script settles 1 and 2. The control for 3 is a training run.
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
IOU_MATCH = 0.5


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


def iou_matrix(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = [int(v) for v in np.unique(truth) if v]
    p = [int(v) for v in np.unique(pred) if v]
    if not t or not p:
        return np.zeros((len(t), len(p)))
    ti = {l: i for i, l in enumerate(t)}
    pi = {l: i for i, l in enumerate(p)}
    inter = np.zeros((len(t), len(p)), np.int64)
    both = (truth > 0) & (pred > 0)
    for tv, pv in zip(truth[both], pred[both]):
        inter[ti[int(tv)], pi[int(pv)]] += 1
    ta = np.array([(truth == l).sum() for l in t], np.int64)
    pa = np.array([(pred == l).sum() for l in p], np.int64)
    return inter / np.maximum(ta[:, None] + pa[None, :] - inter, 1)


def matches_iou_first(ious: np.ndarray, threshold: float = IOU_MATCH) -> int:
    """What this project has been doing: maximise total IoU, then threshold."""
    from scipy.optimize import linear_sum_assignment

    if ious.size == 0:
        return 0
    rows, cols = linear_sum_assignment(-ious)
    return sum(1 for r, c in zip(rows, cols) if ious[r, c] >= threshold)


def matches_count_first(ious: np.ndarray, threshold: float = IOU_MATCH) -> int:
    """The correct rule: maximise the NUMBER of valid matches, ties by IoU.

    Zeroing every sub-threshold cell first makes the assignment unable to spend
    a prediction on a pair that could never count, so the optimum is the largest
    set of valid pairs. The surviving IoU values then break ties among the
    equally large solutions.
    """
    from scipy.optimize import linear_sum_assignment

    if ious.size == 0:
        return 0
    valid = np.where(ious >= threshold, ious, 0.0)
    # A large constant per valid pair makes pair-count dominate the objective;
    # the IoU term, being < 1, only orders solutions of equal size.
    score = np.where(valid > 0, 1000.0 + valid, 0.0)
    rows, cols = linear_sum_assignment(-score)
    return sum(1 for r, c in zip(rows, cols) if valid[r, c] > 0)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--window", default="3,97")
    ap.add_argument("--out", default=str(ROOT / "docs" / "audit_reported_numbers.json"))
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    weights = Path(args.model)
    if not weights.exists():
        raise SystemExit(f"no such checkpoint: {weights}")

    model = cp.CellposeModel(pretrained_model=str(weights), gpu=False)
    print(f"reloaded {weights.name}")
    print(f"  diam_mean   {getattr(model, 'diam_mean', None)}")
    print(f"  diam_labels {getattr(model, 'diam_labels', None)}")

    normalize: object = True
    if args.window:
        lo, hi = (float(v) for v in args.window.split(","))
        normalize = {"percentile": (lo, hi)}

    paths = labelled(args.test_group)
    tp_a = fp_a = fn_a = 0
    tp_b = fp_b = fn_b = 0
    disagreements = []

    for path in paths:
        truth = truth_of(path)
        pred = np.asarray(
            model.eval(load(path), channels=[0, 0], diameter=None,
                       cellprob_threshold=0.0, flow_threshold=0.4,
                       normalize=normalize)[0]
        ).astype(np.int32)
        ious = iou_matrix(truth, pred)
        n_truth = ious.shape[0] if ious.size else len([v for v in np.unique(truth) if v])
        n_pred = ious.shape[1] if ious.size else len([v for v in np.unique(pred) if v])

        a = matches_iou_first(ious)
        b = matches_count_first(ious)
        if a != b:
            disagreements.append({"image": path.name, "iou_first": a, "count_first": b})

        tp_a += a
        fp_a += n_pred - a
        fn_a += n_truth - a
        tp_b += b
        fp_b += n_pred - b
        fn_b += n_truth - b

    old = prf(tp_a, fp_a, fn_a)
    new = prf(tp_b, fp_b, fn_b)

    print(f"\nevaluated on {args.test_group}: {len(paths)} images, window {args.window}")
    print(f"{'matcher':28s} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"{'IoU-max then threshold':28s} {tp_a:5d} {fp_a:5d} {fn_a:5d} "
          f"{old[0]:8.4f} {old[1]:8.4f} {old[2]:8.4f}")
    print(f"{'max valid matches':28s} {tp_b:5d} {fp_b:5d} {fn_b:5d} "
          f"{new[0]:8.4f} {new[1]:8.4f} {new[2]:8.4f}")

    print()
    if disagreements:
        print(f"the two matchers disagree on {len(disagreements)} of {len(paths)} images:")
        for row in disagreements:
            print(f"   {row['image']}: {row['iou_first']} -> {row['count_first']}")
        print("The reported figures used the first rule and are therefore understated.")
    else:
        print("the two matchers agree on every image: the defect is real in principle")
        print("but does not occur in this data, so the reported figures stand.")

    report = {
        "checkpoint": str(weights),
        "test_group": args.test_group,
        "window": args.window,
        "reloaded_diam_labels": float(getattr(model, "diam_labels", 0) or 0),
        "iou_first": dict(zip(("precision", "recall", "f1"), old)),
        "count_first": dict(zip(("precision", "recall", "f1"), new)),
        "disagreements": disagreements,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
