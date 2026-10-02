"""Which recovery strategies actually raise detection recall?

The measured failure mode of this model is missing cells, not inventing them:
held out, recall is 0.20-0.44 while precision holds at 0.54-0.60. So the
question is not "is the model good" but "what can be added around it that finds
the cells it misses without inventing any".

This measures candidate strategies against the real manual labels, so the
answer is a number rather than an intuition. Every strategy is scored on the
same images with the same matching rule, and both recall *and* precision are
reported, because a strategy that finds everything by calling everything a cell
is worse than the baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
COMBI = TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"
KK1 = TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10"
KK2 = TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10"

IOU_MATCH = 0.5


def load_pairs(directory: Path) -> list[tuple[Path, Path]]:
    out = []
    for seg in sorted(directory.glob("*_seg.npy")):
        image = seg.with_name(seg.name.replace("_seg.npy", ".tif"))
        if image.exists():
            out.append((image, seg))
    return out


def truth_of(seg: Path) -> np.ndarray:
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


def iou_matrix(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t_labels = [int(v) for v in np.unique(truth) if v]
    p_labels = [int(v) for v in np.unique(pred) if v]
    if not t_labels or not p_labels:
        return np.zeros((len(t_labels), len(p_labels)))
    ti = {l: i for i, l in enumerate(t_labels)}
    pi = {l: i for i, l in enumerate(p_labels)}
    inter = np.zeros((len(t_labels), len(p_labels)), dtype=np.int64)
    both = (truth > 0) & (pred > 0)
    for tv, pv in zip(truth[both], pred[both]):
        inter[ti[int(tv)], pi[int(pv)]] += 1
    ta = np.array([(truth == l).sum() for l in t_labels], dtype=np.int64)
    pa = np.array([(pred == l).sum() for l in p_labels], dtype=np.int64)
    union = ta[:, None] + pa[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def score(truth: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    ious = iou_matrix(truth, pred)
    nt, npd = ious.shape
    if nt and npd:
        rows, cols = linear_sum_assignment(-ious)
        tp = sum(1 for r, c in zip(rows, cols) if ious[r, c] >= IOU_MATCH)
    else:
        tp = 0
    return tp, npd - tp, nt - tp


# --------------------------------------------------------------------------
# Merging several label images into one
# --------------------------------------------------------------------------


def merge_masks(masks: list[np.ndarray], overlap: float = 0.3) -> np.ndarray:
    """Union of instances across runs, dropping duplicates.

    Instances are added in order; a new instance is kept only if it does not
    substantially overlap one already accepted. That makes the first run in the
    list authoritative and later runs purely additive, which is what a recall
    fallback should be.
    """
    if not masks:
        return np.zeros((0, 0), dtype=np.int32)
    out = np.zeros_like(masks[0], dtype=np.int32)
    next_label = 1
    for mask in masks:
        for value in np.unique(mask):
            if value == 0:
                continue
            region = mask == value
            taken = out[region]
            if taken.any():
                # How much of this candidate is already claimed?
                if (taken > 0).sum() / region.sum() > overlap:
                    continue
                region = region & (out == 0)
                if region.sum() < 20:
                    continue
            out[region] = next_label
            next_label += 1
    return out


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def build_strategies(models: dict):
    combi = models["combi"]

    def single(cellprob=0.0, flow=0.4, augment=False, model=None):
        def run(image):
            m = model or combi
            return np.asarray(
                m.eval(image, channels=[0, 0], diameter=None,
                       cellprob_threshold=cellprob, flow_threshold=flow,
                       augment=augment)[0]
            ).astype(np.int32)
        return run

    def ensemble(settings, augment=False):
        runs = [single(c, f, augment) for c, f in settings]
        def run(image):
            return merge_masks([r(image) for r in runs])
        return run

    def multimodel(settings):
        runs = []
        for name, cellprob, flow in settings:
            runs.append(single(cellprob, flow, False, models.get(name)))
        def run(image):
            return merge_masks([r(image) for r in runs])
        return run

    # Test-time augmentation is omitted: Cellpose's augment=True runs four
    # rotated, tiled passes, which on a CPU costs more than running three
    # different models and measured no better in a trial run.
    return {
        "baseline (prob 0.0, flow 0.4)": single(),
        "prob -2": single(cellprob=-2.0),
        "flow 0.6": single(flow=0.6),
        "ensemble: prob 0 and -2": ensemble([(0.0, 0.4), (-2.0, 0.4)]),
        "ensemble: flow 0.4 and 0.6": ensemble([(0.0, 0.4), (0.0, 0.6)]),
        "ensemble: 4 settings": ensemble(
            [(0.0, 0.4), (-2.0, 0.4), (0.0, 0.6), (-2.0, 0.6)]
        ),
        "3 models, default settings": multimodel(
            [("combi", 0.0, 0.4), ("KK1", 0.0, 0.4), ("KK2", 0.0, 0.4)]
        ),
        "3 models x 2 settings": multimodel(
            [("combi", 0.0, 0.4), ("combi", -2.0, 0.6),
             ("KK1", 0.0, 0.4), ("KK1", -2.0, 0.6),
             ("KK2", 0.0, 0.4), ("KK2", -2.0, 0.6)]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "recall_experiment.json"))
    args = ap.parse_args()

    from cellpose import models as cp

    models = {
        "combi": cp.CellposeModel(pretrained_model=str(COMBI), gpu=False),
        "KK1": cp.CellposeModel(pretrained_model=str(KK1), gpu=False),
        "KK2": cp.CellposeModel(pretrained_model=str(KK2), gpu=False),
    }
    strategies = build_strategies(models)

    pairs = load_pairs(TRAIN / "KK1") + load_pairs(TRAIN / "KK2")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} labelled images\n")
    print(f"{'strategy':34s} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'s/img':>6}")

    results = {}
    for name, run in strategies.items():
        tp = fp = fn = 0
        started = time.time()
        for image_path, seg_path in pairs:
            image = tifffile.imread(image_path)
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                image = image[..., 0]
            a, b, c = score(truth_of(seg_path), run(image))
            tp, fp, fn = tp + a, fp + b, fn + c
        elapsed = (time.time() - started) / max(len(pairs), 1)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
            "seconds_per_image": round(elapsed, 2),
        }
        print(f"{name:34s} {precision:6.3f} {recall:6.3f} {f1:6.3f} "
              f"{tp:5d} {fp:5d} {fn:5d} {elapsed:6.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_images": len(pairs), "strategies": results}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
