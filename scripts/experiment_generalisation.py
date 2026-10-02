"""Can the held-out collapse be fixed at inference time?

`scripts/evaluate_models.py` established the problem: a model trained on one
half of this data recovers only 0.20-0.44 recall on the other half, against
0.82 on its own. The usual conclusion is "collect more data and retrain", which
is true and slow. This asks whether any of it is recoverable now.

**The diagnosis first, because it decides the treatment.** Measuring the two
halves directly:

    group   1st pct   99th pct   cell width   (cell-background)/range
    KK1        1530       2775      12.1 px            0.219
    KK2       20447      42670      12.2 px            0.378

The cells are the *same size* to within a tenth of a pixel, so this is not a
scale problem and a diameter override will not touch it. The absolute
intensities differ by a factor of fourteen, which percentile normalisation
already removes. What survives normalisation is the third column: **a cell in
KK1 stands 0.219 of the image's range above its background, and a cell in KK2
stands 0.378 -- nearly twice as far.**

That predicts the asymmetry in the failure, and the prediction matches:

*   KK2Model, trained only on high-contrast images, scores recall **0.203** on
    KK1. It has never been shown a faint cell.
*   KK1Model, whose training contrast spans 0.115-0.361 and therefore overlaps
    KK2's range, scores **0.435** the other way.

So the hypothesis is that a large part of the held-out loss is a *contrast*
domain shift, not a shape or scale one -- and contrast is something that can be
changed before the image reaches the network. Cellpose exposes the levers:
percentile window, local (tiled) normalisation, and sharpening.

This measures each of them on the genuinely held-out pairs. If a setting lifts
held-out recall without wrecking in-distribution precision, it is a real
generalisation improvement available today. If nothing does, that is worth
knowing too, and the answer is retraining.
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
MODELS = {
    "combi": TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi",
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
}

IOU_MATCH = 0.5

#: (model, image group, whether that group was held out of this model)
PAIRS = [
    ("KK2", "KK1", True),   # the worst case: recall 0.203
    ("KK1", "KK2", True),   # the milder one: recall 0.435
    ("KK1", "KK1", False),  # in-distribution control, to catch a setting that
    ("KK2", "KK2", False),  # buys held-out recall by ruining everything else
]

#: Every candidate is a Cellpose ``normalize`` argument. The default is first.
STRATEGIES: dict[str, object] = {
    "default (1, 99)": True,
    "percentile (3, 97)": {"percentile": (3.0, 97.0)},
    "percentile (5, 95)": {"percentile": (5.0, 95.0)},
    "percentile (10, 90)": {"percentile": (10.0, 90.0)},
    "percentile (20, 80)": {"percentile": (20.0, 80.0)},
    "tile norm 128": {"tile_norm_blocksize": 128},
    "tile norm 64": {"tile_norm_blocksize": 64},
    "sharpen 15": {"sharpen_radius": 15},
    "sharpen 30": {"sharpen_radius": 30},
    "tile 128 + sharpen 15": {"tile_norm_blocksize": 128, "sharpen_radius": 15},
    "percentile (5,95) + sharpen 15": {"percentile": (5.0, 95.0), "sharpen_radius": 15},
}


def load_pairs(directory: Path) -> list[tuple[Path, Path]]:
    out = []
    for seg in sorted(directory.glob("*_seg.npy")):
        image = seg.with_name(seg.name.replace("_seg.npy", ".tif"))
        if image.exists():
            out.append((image, seg))
    return out


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image


def score(truth: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    """One-to-one instance matching at IoU 0.5, as in every other script here."""
    t_labels = [int(v) for v in np.unique(truth) if v]
    p_labels = [int(v) for v in np.unique(pred) if v]
    if not t_labels or not p_labels:
        return 0, len(p_labels), len(t_labels)

    ti = {l: i for i, l in enumerate(t_labels)}
    pi = {l: i for i, l in enumerate(p_labels)}
    inter = np.zeros((len(t_labels), len(p_labels)), dtype=np.int64)
    both = (truth > 0) & (pred > 0)
    for tv, pv in zip(truth[both], pred[both]):
        inter[ti[int(tv)], pi[int(pv)]] += 1
    ta = np.array([(truth == l).sum() for l in t_labels], dtype=np.int64)
    pa = np.array([(pred == l).sum() for l in p_labels], dtype=np.int64)
    ious = inter / np.maximum(ta[:, None] + pa[None, :] - inter, 1)

    rows, cols = linear_sum_assignment(-ious)
    tp = sum(1 for r, c in zip(rows, cols) if ious[r, c] >= IOU_MATCH)
    return tp, len(p_labels) - tp, len(t_labels) - tp


def evaluate(model, images, normalize) -> dict:
    tp = fp = fn = 0
    started = time.time()
    for image, truth in images:
        pred = np.asarray(
            model.eval(
                image,
                channels=[0, 0],
                diameter=None,
                cellprob_threshold=0.0,
                flow_threshold=0.4,
                normalize=normalize,
            )[0]
        ).astype(np.int32)
        a, b, c = score(truth, pred)
        tp, fp, fn = tp + a, fp + b, fn + c

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "seconds_per_image": round((time.time() - started) / max(len(images), 1), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="images per group, 0 for all")
    ap.add_argument("--out", default=str(ROOT / "docs" / "generalisation_experiment.json"))
    args = ap.parse_args()

    from cellpose import models as cp

    loaded = {
        name: cp.CellposeModel(pretrained_model=str(path), gpu=False)
        for name, path in MODELS.items()
        if path.exists()
    }
    if not loaded:
        print("no models found")
        return 1

    groups: dict[str, list] = {}
    for group in ("KK1", "KK2"):
        pairs = load_pairs(TRAIN / group)
        if args.limit:
            pairs = pairs[: args.limit]
        groups[group] = [
            (read_image(img), np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32))
            for img, seg in pairs
        ]
        print(f"{group}: {len(groups[group])} labelled images")

    report: dict = {"iou_match": IOU_MATCH, "strategies": {}}
    print(f"\n{'strategy':32s} " + "  ".join(
        f"{m}->{g}{'*' if held else ' '}" for m, g, held in PAIRS
    ) + "     (* = held out; each cell is recall/F1)")

    for label, normalize in STRATEGIES.items():
        row = {}
        cells = []
        for model_name, group, held_out in PAIRS:
            if model_name not in loaded:
                cells.append("     -    ")
                continue
            result = evaluate(loaded[model_name], groups[group], normalize)
            result["held_out"] = held_out
            row[f"{model_name}->{group}"] = result
            cells.append(f"{result['recall']:.3f}/{result['f1']:.3f}")
        report["strategies"][label] = row
        print(f"{label:32s} " + "  ".join(f"{c:>11}" for c in cells), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # -- say plainly whether anything worked --------------------------------
    baseline = report["strategies"]["default (1, 99)"]
    print("\nheld-out recall against the default:")
    for key in ("KK2->KK1", "KK1->KK2"):
        if key not in baseline:
            continue
        base = baseline[key]["recall"]
        best_label, best = max(
            ((label, row[key]) for label, row in report["strategies"].items() if key in row),
            key=lambda item: item[1]["recall"],
        )
        change = best["recall"] - base
        print(f"  {key}: {base:.3f} -> {best['recall']:.3f} ({change:+.3f}) "
              f"with '{best_label}', precision {baseline[key]['precision']:.3f} -> "
              f"{best['precision']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
