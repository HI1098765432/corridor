"""Are the missed cells simply too long for the scale the model assumes?

The error breakdown says the 9 cells that are "visible but missed" -- normal
contrast, strong edges, normal width, not at a border -- have a median major
axis of 95.4 px, against 82.8 px for the objects the model does fire on, with an
inter-quartile range reaching 130 px. The model's ``diam_mean`` is 30.

Cellpose does not segment at native resolution. It rescales each image so that
the objects match the diameter it was trained at, either from the diameter it is
given or, when given none, from ``diam_labels`` stored in the checkpoint. For a
round cell "diameter" is unambiguous; for a confined cell 10 px wide and 130 px
long it is not, and the equivalent-disk diameter such a cell implies is
dominated by its **length**. A field containing an unusually long cell is
therefore rescaled differently from one containing ordinary ones, and the long
cell can end up at a scale the network was never trained on.

This sweeps the diameter directly. Nothing is retrained and no label is touched:
the only thing that changes is the size Cellpose believes a cell to be. If
recall on the long cells rises at a larger diameter while the rest holds, the
misses are a scale problem and the fix is scale handling rather than more
appearance training. If nothing moves, the hypothesis is wrong and that is worth
the twenty minutes it costs to find out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.metrics import iou_matrix, match, score_image  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
MODELS = {
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
    "combi": TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi",
    "trained": ROOT / "build" / "models" / "models" / "corridor_contrast_invariant",
}
#: None means "use the diameter stored in the checkpoint", which is what every
#: measurement so far has used.
DIAMETERS = (None, 25.0, 30.0, 36.0, 45.0, 60.0, 80.0)
#: A cell longer than this is in the group the error table flagged.
LONG_CELL_PX = 90.0


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


def major_axis(region: np.ndarray) -> float:
    from skimage.measure import regionprops

    props = regionprops(region.astype(np.int32))
    return float(props[0].axis_major_length) if props else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained", choices=sorted(MODELS))
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--window", default="3,97")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "docs" / "diameter_sweep.json"))
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(args.threads)
    from cellpose import models as cp

    weights = MODELS[args.model]
    if not Path(weights).exists():
        raise SystemExit(f"no such checkpoint: {weights}")
    model = cp.CellposeModel(pretrained_model=str(weights), gpu=False)

    normalize: object = True
    if args.window:
        lo, hi = (float(v) for v in args.window.split(","))
        normalize = {"percentile": (lo, hi)}

    paths = labelled(args.test_group)

    # Split the truth into long cells and the rest, so a gain on one is not
    # hidden by a loss on the other.
    long_cells: dict[str, set[int]] = {}
    n_long = n_short = 0
    for path in paths:
        truth = truth_of(path)
        ids = set()
        for label in np.unique(truth):
            if not label:
                continue
            if major_axis(truth == label) >= LONG_CELL_PX:
                ids.add(int(label))
                n_long += 1
            else:
                n_short += 1
        long_cells[path.name] = ids

    print(f"{args.model} on {args.test_group}: {len(paths)} images, "
          f"{n_long} cells longer than {LONG_CELL_PX:.0f} px, {n_short} shorter")
    print(f"checkpoint diam_labels = {getattr(model, 'diam_labels', None):.2f}\n")
    print(f"{'diameter':>10} {'F1':>8} {'P':>8} {'R':>8} "
          f"{'recall long':>12} {'recall short':>13}")

    rows = []
    for diameter in DIAMETERS:
        total = None
        found_long = found_short = 0
        for path in paths:
            truth = truth_of(path)
            prediction = np.asarray(
                model.eval(load(path), channels=[0, 0], diameter=diameter,
                           cellprob_threshold=0.0, flow_threshold=0.4,
                           normalize=normalize)[0]
            ).astype(np.int32)
            score = score_image(truth, prediction)
            total = score if total is None else total + score

            # Which of the matched truths were long ones?
            ious = iou_matrix(truth, prediction)
            truth_labels = [int(v) for v in np.unique(truth) if v]
            for r, _ in match(ious):
                label = truth_labels[r]
                if label in long_cells[path.name]:
                    found_long += 1
                else:
                    found_short += 1

        recall_long = found_long / n_long if n_long else 0.0
        recall_short = found_short / n_short if n_short else 0.0
        label = "from model" if diameter is None else f"{diameter:.0f}"
        rows.append({
            "diameter": diameter, "f1": round(total.f1, 4),
            "precision": round(total.precision, 4), "recall": round(total.recall, 4),
            "recall_long_cells": round(recall_long, 4),
            "recall_short_cells": round(recall_short, 4),
        })
        print(f"{label:>10} {total.f1:8.4f} {total.precision:8.4f} "
              f"{total.recall:8.4f} {recall_long:12.4f} {recall_short:13.4f}",
              flush=True)
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "group": args.test_group, "window": args.window,
             "long_cell_threshold_px": LONG_CELL_PX,
             "n_long": n_long, "n_short": n_short, "results": rows},
            indent=2), encoding="utf-8")

    best = max(rows, key=lambda r: r["f1"])
    baseline = next(r for r in rows if r["diameter"] is None)
    print()
    print(f"best F1 at diameter {best['diameter'] or 'from model'}: {best['f1']:.4f} "
          f"({best['f1'] - baseline['f1']:+.4f} against the checkpoint's own)")
    best_long = max(rows, key=lambda r: r["recall_long_cells"])
    print(f"best recall on the long cells at diameter "
          f"{best_long['diameter'] or 'from model'}: "
          f"{best_long['recall_long_cells']:.4f} "
          f"(against {baseline['recall_long_cells']:.4f})")
    if best_long["recall_long_cells"] > baseline["recall_long_cells"] + 0.05:
        print("\nThe long cells ARE a scale problem: they are found at a diameter the")
        print("checkpoint does not choose for itself. Scale handling is the fix.")
    else:
        print("\nChanging the diameter does not recover them, so their length is not")
        print("what defeats the model. The hypothesis is wrong and is recorded as such.")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
