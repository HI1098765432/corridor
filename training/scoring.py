"""Scoring for research, built on ``corridor.core.metrics`` and nothing else.

Every count here comes from :func:`corridor.core.metrics.iou_matrix` and
:func:`corridor.core.metrics.match` (count-first matching), so a research
figure and an application figure mean the same thing. This module adds only
what the contract asks for around them (``docs/NEXT_GENERATION.md`` section 9):

- **Border-margin exclusion.** Instances whose pixels come within ``margin`` px
  of the image edge are dropped from truth *and* prediction, independently,
  **before** anything is counted -- ``score_image`` counts every non-zero label,
  so filtering afterwards would still count the excluded ones. The distance is
  the one ``scripts/diag_errors.py`` uses: ``min(ys.min(), xs.min(),
  H-1-ys.max(), W-1-xs.max())``, so 0 means touching the edge. Because the two
  sides are filtered independently, a truth just inside the margin whose
  prediction is just outside it becomes a false positive; that is the defined
  behaviour, and the reason the unmodified figure is always reported beside it.
- **Average precision over IoU 0.50:0.05:0.95**, in the Cellpose convention:
  ``AP(tau) = TP / (TP + FP + FN)`` pooled over images at threshold tau, and its
  mean over the ten thresholds. It is not the COCO precision-recall area.
- **Splits and merges** from the IoU matrix: intersections are recovered from
  IoU and the two areas, a prediction *belongs* to the truth holding at least
  half of its pixels and a truth to the prediction holding at least half of
  its pixels; a truth owning two or more predictions is split, a prediction
  owning two or more truths is a merge.
- **Breakdowns** of recall (truth instances) and precision (predictions) by any
  per-instance property: contrast quartile, size, border distance, experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import training  # noqa: F401  (puts src/ on the path)
from corridor.core.metrics import DEFAULT_IOU, Score, iou_matrix, match, score_image

#: IoU 0.50, 0.55, ..., 0.95.
AP_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
#: Border-distance bins, px: touching, the 3 px band diag_errors calls "at the
#: border", up to 10 (where 18 of the 64 round-1 errors sat), and beyond.
BORDER_EDGES_PX = (0, 3, 10, 30)
#: A prediction (truth) belongs to the instance holding at least this fraction
#: of its pixels.
OWNERSHIP_FRACTION = 0.5


def labels_of(mask: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(mask) if v]


def border_distance(region: np.ndarray) -> int:
    """Pixels between an instance and the nearest image edge (0 = touching)."""
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return -1
    height, width = region.shape[:2]
    return int(min(ys.min(), xs.min(), height - 1 - ys.max(), width - 1 - xs.max()))


def drop_border(labels: np.ndarray, margin_px: int | None) -> np.ndarray:
    """A copy of a label image without instances within ``margin_px`` of the edge.

    An instance at border distance ``<= margin_px`` is removed; ``None`` keeps
    everything. Remaining labels keep their ids.
    """
    out = np.asarray(labels).copy()
    if margin_px is None:
        return out
    for label in labels_of(out):
        region = out == label
        if border_distance(region) <= margin_px:
            out[region] = 0
    return out


def score(truth: np.ndarray, prediction: np.ndarray, *, threshold: float = DEFAULT_IOU,
          margin_px: int | None = None) -> Score:
    """``core.metrics.score_image`` with optional border-margin exclusion first."""
    return score_image(drop_border(truth, margin_px), drop_border(prediction, margin_px),
                       threshold)


def _intersections(truth: np.ndarray, prediction: np.ndarray, ious: np.ndarray):
    t_area = np.array([(truth == l).sum() for l in labels_of(truth)], dtype=float)
    p_area = np.array([(prediction == l).sum() for l in labels_of(prediction)], dtype=float)
    inter = ious * (t_area[:, None] + p_area[None, :]) / (1.0 + ious)
    return np.rint(inter), t_area, p_area


def split_merge(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Split and merge counts derived from the IoU matrix (see module docstring)."""
    ious = iou_matrix(truth, prediction)
    if ious.size == 0:
        return {"splits": 0, "merges": 0, "split_truths": [], "merging_predictions": []}
    inter, t_area, p_area = _intersections(truth, prediction, ious)
    pred_owner = np.where(inter.max(axis=0) >= OWNERSHIP_FRACTION * p_area,
                          inter.argmax(axis=0), -1)
    truth_owner = np.where(inter.max(axis=1) >= OWNERSHIP_FRACTION * t_area,
                           inter.argmax(axis=1), -1)
    t_labels, p_labels = labels_of(truth), labels_of(prediction)
    split_truths = [t_labels[i] for i in range(len(t_labels))
                    if int(np.count_nonzero(pred_owner == i)) >= 2]
    merging = [p_labels[j] for j in range(len(p_labels))
               if int(np.count_nonzero(truth_owner == j)) >= 2]
    return {"splits": len(split_truths), "merges": len(merging),
            "split_truths": split_truths, "merging_predictions": merging}


@dataclass
class ImageResult:
    """Everything about one image that the pooled figures are built from."""

    image: str
    n_truth: int
    n_pred: int
    #: Score at each AP threshold, keyed by the threshold as a string.
    by_threshold: dict
    splits: int
    merges: int

    @property
    def at_50(self) -> Score:
        return self.by_threshold["0.5"]


def evaluate_image(image_id: str, truth: np.ndarray, prediction: np.ndarray, *,
                   margin_px: int | None = None,
                   thresholds: tuple[float, ...] = AP_THRESHOLDS) -> ImageResult:
    truth = drop_border(truth, margin_px)
    prediction = drop_border(prediction, margin_px)
    by_threshold = {str(t): score_image(truth, prediction, t) for t in thresholds}
    sm = split_merge(truth, prediction)
    return ImageResult(image_id, len(labels_of(truth)), len(labels_of(prediction)),
                       by_threshold, sm["splits"], sm["merges"])


def pool(results: list[ImageResult], thresholds: tuple[float, ...] = AP_THRESHOLDS) -> dict:
    """Pooled AP curve, P/R/F1 at 0.5, per-image FP/FN, splits and merges."""
    ap = {}
    for t in thresholds:
        total = Score(0, 0, 0, [])
        for r in results:
            total = total + r.by_threshold[str(t)]
        denominator = total.tp + total.fp + total.fn
        ap[str(t)] = round(total.tp / denominator, 4) if denominator else None
    at_50 = Score(0, 0, 0, [])
    for r in results:
        at_50 = at_50 + r.by_threshold["0.5"]
    defined = [v for v in ap.values() if v is not None]
    return {
        "images": len(results),
        "truth_instances": sum(r.n_truth for r in results),
        "predicted_instances": sum(r.n_pred for r in results),
        "at_iou_0.5": at_50.to_row(),
        "ap_by_iou": ap,
        "ap_50_95": round(float(np.mean(defined)), 4) if len(defined) == len(ap) else None,
        "fp_per_image": round(at_50.fp / len(results), 3) if results else None,
        "fn_per_image": round(at_50.fn / len(results), 3) if results else None,
        "splits": sum(r.splits for r in results),
        "merges": sum(r.merges for r in results),
        "per_image": [{"image": r.image, "truth": r.n_truth, "pred": r.n_pred,
                       "tp": r.at_50.tp, "fp": r.at_50.fp, "fn": r.at_50.fn,
                       "splits": r.splits, "merges": r.merges} for r in results],
    }


# --------------------------------------------------------------------------
# Per-instance properties and breakdowns


@dataclass
class InstanceRow:
    image: str
    side: str  # "truth" or "pred"
    label: int
    matched: bool  # at IoU 0.5, by core.metrics.match
    best_iou: float
    area_px: int
    major_px: float
    minor_px: float
    #: (mean inside - median of a 3-10 px ring) / (p99 - p1): diag_errors' definition.
    contrast: float
    border_px: int
    group: str = ""  # experiment id, filled by the caller

    def to_dict(self) -> dict:
        return asdict(self)


def describe_instances(image_id: str, image: np.ndarray, truth: np.ndarray,
                       prediction: np.ndarray, *, threshold: float = DEFAULT_IOU,
                       group: str = "") -> list[InstanceRow]:
    """One row per truth and per predicted instance, with its match status."""
    from scipy.ndimage import binary_dilation
    from skimage.measure import regionprops

    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1, 99])
    span = max(float(hi - lo), 1e-6)
    ious = iou_matrix(truth, prediction)
    pairs = match(ious, threshold)
    matched_t = {r for r, _ in pairs}
    matched_p = {c for _, c in pairs}

    rows: list[InstanceRow] = []
    for side, mask, matched in (("truth", truth, matched_t), ("pred", prediction, matched_p)):
        for index, label in enumerate(labels_of(mask)):
            region = mask == label
            ring = binary_dilation(region, iterations=10) & ~binary_dilation(region, iterations=3)
            background = float(np.median(image[ring])) if ring.any() else float(np.median(image))
            props = regionprops(region.astype(np.int32))[0]
            if ious.size:
                best = float(ious[index].max() if side == "truth" else ious[:, index].max())
            else:
                best = 0.0
            rows.append(InstanceRow(
                image=image_id, side=side, label=label, matched=index in matched,
                best_iou=round(best, 4), area_px=int(region.sum()),
                major_px=round(float(props.axis_major_length), 1),
                minor_px=round(float(props.axis_minor_length), 1),
                contrast=round((float(image[region].mean()) - background) / span, 4),
                border_px=border_distance(region), group=group))
    return rows


def quartile_edges(values) -> list[float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return []
    return [float(v) for v in np.percentile(values, [25, 50, 75])]


def _bin_of(value: float, edges) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), value, side="left"))


def breakdown(rows: list[InstanceRow], field: str, edges=None) -> list[dict]:
    """Recall of truth and precision of predictions per bin of one property.

    ``edges`` are inclusive upper bounds (a value equal to an edge falls in the
    lower bin); ``None`` groups by the field's distinct values instead, which is
    how experiments are broken down.
    """
    def key_of(row):
        value = getattr(row, field)
        return value if edges is None else _bin_of(float(value), edges)

    keys = sorted({key_of(r) for r in rows}, key=lambda k: (str(type(k)), k))
    out = []
    for key in keys:
        truths = [r for r in rows if r.side == "truth" and key_of(r) == key]
        preds = [r for r in rows if r.side == "pred" and key_of(r) == key]
        if edges is None:
            label = str(key)
        else:
            lower = "-inf" if key == 0 else f"{edges[key - 1]:g}"
            upper = "inf" if key >= len(edges) else f"{edges[key]:g}"
            label = f"({lower}, {upper}]"
        out.append({
            "bin": label,
            "truth": len(truths),
            "recall": round(sum(r.matched for r in truths) / len(truths), 4) if truths else None,
            "pred": len(preds),
            "precision": round(sum(r.matched for r in preds) / len(preds), 4) if preds else None,
        })
    return out


def standard_breakdowns(rows: list[InstanceRow]) -> dict:
    """Experiment, contrast quartile, size quartile and border distance.

    Quartile edges come from the **truth** instances, so predictions are binned
    on the same scale the reference defines.
    """
    truths = [r for r in rows if r.side == "truth"]
    contrast_edges = quartile_edges(r.contrast for r in truths)
    size_edges = quartile_edges(r.area_px for r in truths)
    return {
        "experiment": breakdown(rows, "group"),
        "contrast_quartile": {"edges": [round(e, 4) for e in contrast_edges],
                              "bins": breakdown(rows, "contrast", contrast_edges)},
        "size_quartile_px": {"edges": [round(e, 1) for e in size_edges],
                             "bins": breakdown(rows, "area_px", size_edges)},
        "border_distance_px": {"edges": list(BORDER_EDGES_PX),
                               "bins": breakdown(rows, "border_px", BORDER_EDGES_PX)},
    }
