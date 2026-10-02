"""Are the labels ground truth, or are they a noisy, partly-absent reference?

Every accuracy figure this project has published is an agreement score against
71 hand-labelled images. That is only a measure of the model if the labels are
right. Two things say they might not be:

*   9 of the 71 images contain **no labelled cells at all**, and several sit
    between frames of the same field that do have cells. Every detection a
    model makes in those images is scored as a false positive, whether or not
    a cell is there.
*   The mean IoU of *correctly matched* pairs is 0.696. Even when detection
    succeeds, the outlines only roughly agree.

This measures how much of the reported error is the model being wrong and how
much is the reference being incomplete. It does not assume the model is right:
it reports both numbers and lets the gap speak.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
COMBI = TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"
IOU_MATCH = 0.5


def load(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image


def truth_of(path: Path) -> np.ndarray:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


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


def score(truth: np.ndarray, pred: np.ndarray):
    ious = iou_matrix(truth, pred)
    nt, npd = ious.shape
    if nt and npd:
        rows, cols = linear_sum_assignment(-ious)
        matched = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= IOU_MATCH]
        tp = len(matched)
        overlaps = [float(ious[r, c]) for r, c in matched]
    else:
        tp, overlaps = 0, []
    return tp, npd - tp, nt - tp, overlaps


def labelled_files() -> list[Path]:
    out = []
    for group in ("KK1", "KK2"):
        for path in sorted((TRAIN / group).glob("*.tif")):
            if path.with_name(path.name.replace(".tif", "_seg.npy")).exists():
                out.append(path)
    return out


def elongated_objects(image: np.ndarray, reference_areas: list[float]) -> int:
    """Count bright, elongated, cell-sized objects using no network at all.

    This is the independent check. Asking the model whether an unlabelled frame
    contains a cell would be circular -- the model's opinion is what is on
    trial. Intensity and shape are not the model's opinion.
    """
    from skimage.measure import label, regionprops

    if not reference_areas:
        return 0
    lo, hi = np.percentile(image, [50, 99.5])
    if hi <= lo:
        return 0
    strong = image > (lo + 0.55 * (hi - lo))
    typical = float(np.median(reference_areas))
    count = 0
    for region in regionprops(label(strong)):
        if not (0.35 * typical <= region.area <= 3.0 * typical):
            continue
        if region.eccentricity < 0.85:
            continue
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "label_quality.json"))
    args = ap.parse_args()

    # Cellpose, OpenCV and torch each spawn their own thread pool. Run under
    # anything else heavy and they oversubscribe the machine, at which point
    # OpenCV raises an opaque "Unknown C++ exception" from inside resize. One
    # thread each is slower and finishes.
    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(1)

    from cellpose import models as cp

    model = cp.CellposeModel(pretrained_model=str(COMBI), gpu=False)
    files = labelled_files()

    # Typical labelled cell area, for the network-free object test.
    areas: list[float] = []
    for path in files:
        truth = truth_of(path)
        for lab in np.unique(truth):
            if lab:
                areas.append(float((truth == lab).sum()))

    empty_fp = 0
    rest_tp = rest_fp = rest_fn = 0
    all_overlaps: list[float] = []
    empty_rows = []

    for path in files:
        image = load(path)
        truth = truth_of(path)
        pred = np.asarray(
            model.eval(image, channels=[0, 0], diameter=None,
                       cellprob_threshold=0.0, flow_threshold=0.4)[0]
        ).astype(np.int32)
        tp, fp, fn, overlaps = score(truth, pred)
        all_overlaps += overlaps

        if not [l for l in np.unique(truth) if l]:
            empty_fp += fp
            empty_rows.append({
                "image": f"{path.parent.name}/{path.name}",
                "model_found": fp,
                "objects_without_a_network": elongated_objects(image, areas),
            })
        else:
            rest_tp += tp
            rest_fp += fp
            rest_fn += fn

    total_fp = empty_fp + rest_fp

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    as_published = prf(rest_tp, total_fp, rest_fn)
    excluding_empty = prf(rest_tp, rest_fp, rest_fn)

    report = {
        "n_images": len(files),
        "n_images_with_no_labels": len(empty_rows),
        "empty_labelled_images": empty_rows,
        "false_positives_from_empty_images": empty_fp,
        "false_positives_from_the_rest": rest_fp,
        "as_published": dict(zip(("precision", "recall", "f1"), as_published)),
        "excluding_unlabelled_images": dict(zip(("precision", "recall", "f1"), excluding_empty)),
        "mean_iou_of_matches": round(float(np.mean(all_overlaps)), 4) if all_overlaps else None,
        "median_iou_of_matches": round(float(np.median(all_overlaps)), 4) if all_overlaps else None,
    }

    print(f"{len(files)} labelled images, {len(empty_rows)} of them with no labels at all\n")
    print("In the images labelled empty, what is actually there:")
    print(f"  {'image':30s} {'model finds':>12} {'objects (no network)':>22}")
    for row in empty_rows:
        print(f"  {row['image']:30s} {row['model_found']:12d} "
              f"{row['objects_without_a_network']:22d}")

    print(f"\nfalse positives from those {len(empty_rows)} images : {empty_fp}")
    print(f"false positives from the other {len(files) - len(empty_rows)} images : {rest_fp}")
    share = 100 * empty_fp / max(total_fp, 1)
    print(f"they are {100 * len(empty_rows) / len(files):.0f}% of the images and "
          f"{share:.0f}% of all false positives")

    print(f"\nas published                  : P {as_published[0]}  R {as_published[1]}  F1 {as_published[2]}")
    print(f"excluding unlabelled images   : P {excluding_empty[0]}  R {excluding_empty[1]}  F1 {excluding_empty[2]}")
    print(f"\nmean IoU of matched pairs     : {report['mean_iou_of_matches']}")
    print(f"median IoU of matched pairs   : {report['median_iou_of_matches']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
