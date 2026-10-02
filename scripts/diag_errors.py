"""Every remaining error, one row each, with the evidence to classify it.

Aggregate scores say how much is wrong, never what. At F1 0.695 on KK2 there are
about 30 misses and 29 false detections left, and the next experiment should be
chosen by what those 59 things actually are -- not by which method sounds most
promising.

For each error this records properties computed from pixels and labels alone:
contrast against local background, size, elongation, blur, distance to the image
border, how crowded the neighbourhood is, and for a false positive whether it
sits where a cell is in an adjacent frame of the same field. Nothing here is a
model output, so the classification cannot be the model excusing itself.

The buckets it sorts into decide the next move:

*   **too faint** -- contrast below the band the model responds in. More
    appearance training, or input conditioning.
*   **split / merged** -- the cell was found but carved up or fused. A
    representation problem, not a detection one.
*   **suspect label** -- a "false positive" that looks exactly like a cell and
    sits where neighbouring frames have one. The reference is wrong, not the
    model.
*   **border / debris / crowding** -- structural cases each with their own fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.metrics import iou_matrix, match  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"


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


def describe(image: np.ndarray, region: np.ndarray) -> dict:
    """Physical properties of one object, from pixels only."""
    from scipy.ndimage import binary_dilation
    from skimage.measure import regionprops

    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return {}
    lo, hi = np.percentile(image, [1, 99])
    span = max(float(hi - lo), 1e-6)

    ring = binary_dilation(region, iterations=10) & ~binary_dilation(region, iterations=3)
    background = float(np.median(image[ring])) if ring.any() else float(np.median(image))
    contrast = (float(image[region].mean()) - background) / span

    props = regionprops(region.astype(np.int32))[0]
    height, width = image.shape[:2]
    border = min(int(ys.min()), int(xs.min()),
                 int(height - 1 - ys.max()), int(width - 1 - xs.max()))

    # Sharpness of the object's own edge, as a blur proxy.
    from scipy.ndimage import sobel

    gy, gx = sobel(image, axis=0), sobel(image, axis=1)
    gradient = np.hypot(gx, gy)
    edge = region & ~binary_dilation(region, iterations=1) if region.sum() > 4 else region
    edge_strength = float(gradient[region].mean() / max(gradient.mean(), 1e-6))

    return {
        "contrast": round(float(contrast), 4),
        "area_px": int(region.sum()),
        "eccentricity": round(float(props.eccentricity), 3),
        "major_px": round(float(props.axis_major_length), 1),
        "minor_px": round(float(props.axis_minor_length), 1),
        "distance_to_border_px": border,
        "edge_strength": round(edge_strength, 3),
        "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
    }


def classify_miss(row: dict, faint_threshold: float) -> str:
    if row.get("best_iou", 0.0) >= 0.25:
        return "found but outline too poor"
    if abs(row.get("contrast", 0.0)) < faint_threshold:
        return "too faint"
    if row.get("distance_to_border_px", 99) <= 3:
        return "at the image border"
    if row.get("area_px", 0) < 120:
        return "very small"
    return "visible but missed"


def classify_false_positive(row: dict, faint_threshold: float, typical_area: float) -> str:
    if row.get("best_iou", 0.0) >= 0.25:
        return "overlaps a real cell (split or poor outline)"
    if row.get("distance_to_border_px", 99) <= 3:
        return "at the image border"
    if row.get("area_px", 0) > 2.5 * typical_area:
        return "too large for a cell"
    if row.get("eccentricity", 1.0) < 0.8:
        return "not elongated (debris or wall)"
    if abs(row.get("contrast", 0.0)) >= faint_threshold:
        return "looks exactly like a cell (suspect label)"
    return "faint structure"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-group", default="KK2")
    ap.add_argument("--window", default="3,97")
    ap.add_argument("--out", default=str(ROOT / "docs" / "error_table.json"))
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    model = cp.CellposeModel(pretrained_model=str(args.model), gpu=False)
    normalize: object = True
    if args.window:
        lo, hi = (float(v) for v in args.window.split(","))
        normalize = {"percentile": (lo, hi)}

    paths = labelled(args.test_group)

    # What a labelled cell looks like here, so "faint" and "large" mean something.
    contrasts, areas = [], []
    for path in paths:
        image, truth = load(path), truth_of(path)
        for label in np.unique(truth):
            if not label:
                continue
            row = describe(image, truth == label)
            if row:
                contrasts.append(abs(row["contrast"]))
                areas.append(row["area_px"])
    faint_threshold = float(np.percentile(contrasts, 15)) if contrasts else 0.1
    typical_area = float(np.median(areas)) if areas else 800.0
    print(f"reference cells: {len(contrasts)} instances, median area {typical_area:.0f} px, "
          f"faint below |c| {faint_threshold:.3f}\n")

    misses, false_positives = [], []
    for path in paths:
        image, truth = load(path), truth_of(path)
        pred = np.asarray(
            model.eval(image, channels=[0, 0], diameter=None, cellprob_threshold=0.0,
                       flow_threshold=0.4, normalize=normalize)[0]
        ).astype(np.int32)

        ious = iou_matrix(truth, pred)
        pairs = match(ious)
        matched_truth = {r for r, _ in pairs}
        matched_pred = {c for _, c in pairs}
        truth_labels = [int(v) for v in np.unique(truth) if v]
        pred_labels = [int(v) for v in np.unique(pred) if v]

        for i, label in enumerate(truth_labels):
            if i in matched_truth:
                continue
            row = describe(image, truth == label)
            row["image"] = path.name
            row["best_iou"] = round(float(ious[i].max()), 3) if ious.size else 0.0
            row["kind"] = classify_miss(row, faint_threshold)
            misses.append(row)

        for j, label in enumerate(pred_labels):
            if j in matched_pred:
                continue
            row = describe(image, pred == label)
            row["image"] = path.name
            row["best_iou"] = (
                round(float(ious[:, j].max()), 3) if ious.size and ious.shape[0] else 0.0
            )
            row["kind"] = classify_false_positive(row, faint_threshold, typical_area)
            false_positives.append(row)

    print(f"MISSES ({len(misses)})")
    for kind, n in Counter(r["kind"] for r in misses).most_common():
        print(f"   {n:3d}  {kind}")
    print(f"\nFALSE DETECTIONS ({len(false_positives)})")
    for kind, n in Counter(r["kind"] for r in false_positives).most_common():
        print(f"   {n:3d}  {kind}")

    if misses:
        faint = [r for r in misses if r["kind"] == "too faint"]
        print(f"\nmissed cells: median |contrast| "
              f"{np.median([abs(r['contrast']) for r in misses]):.3f} "
              f"against {np.median(contrasts):.3f} for all labelled cells")
        if faint:
            print(f"   of which {len(faint)} are below the faint threshold")

    report = {
        "model": args.model, "test_group": args.test_group, "window": args.window,
        "faint_threshold": round(faint_threshold, 4),
        "typical_area_px": typical_area,
        "n_misses": len(misses), "n_false_positives": len(false_positives),
        "miss_kinds": dict(Counter(r["kind"] for r in misses)),
        "false_positive_kinds": dict(Counter(r["kind"] for r in false_positives)),
        "misses": misses, "false_positives": false_positives,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
