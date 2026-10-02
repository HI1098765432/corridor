"""One instance-matching implementation, so every score means the same thing.

This existed four times over -- in ``scripts/evaluate_models.py``, in each
training script, and in the recall experiments -- and all four copies shared a
subtle defect that an external review caught.

**The defect.** Matching predictions to truth is an assignment problem, and the
obvious solution is ``linear_sum_assignment`` on the IoU matrix followed by
dropping pairs below the threshold. That maximises *total IoU*, which is not the
question being asked. The question is how many objects were found, and a
high-overlap pair can capture a prediction that a second truth needed:

    truth A: IoU 1.00 with X, 0.50 with Y
    truth B: IoU 0.50 with X, 0.00 with Y

Maximising total IoU prefers (A->X, B->Y) at 1.00 over (A->Y, B->X) at 1.00 --
a tie the solver may break either way -- and the first gives one valid match
where the second gives two. Counted correctly the answer is two.

**The fix.** Zero every sub-threshold cell first, then add a large constant to
the survivors so that the number of valid pairs dominates the objective and the
IoU values only order solutions of equal size. The result maximises valid
matches, with IoU as the tie-break, which is what the metric is defined to be.

Measured on this project's data the two rules agree on every image, so no
published figure changes. It is fixed anyway: a metric that is right by luck on
the current data is not right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The threshold every headline figure in this project is quoted at.
DEFAULT_IOU = 0.5
#: Large enough that one extra valid pair always beats any IoU difference,
#: since IoU is bounded by 1.
_PAIR_WEIGHT = 1000.0


def iou_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Intersection over union for every (truth, prediction) instance pair."""
    truth_labels = [int(v) for v in np.unique(truth) if v]
    pred_labels = [int(v) for v in np.unique(prediction) if v]
    if not truth_labels or not pred_labels:
        return np.zeros((len(truth_labels), len(pred_labels)), dtype=float)

    ti = {label: i for i, label in enumerate(truth_labels)}
    pi = {label: i for i, label in enumerate(pred_labels)}
    intersection = np.zeros((len(truth_labels), len(pred_labels)), dtype=np.int64)
    overlap = (truth > 0) & (prediction > 0)
    for t_value, p_value in zip(truth[overlap], prediction[overlap]):
        intersection[ti[int(t_value)], pi[int(p_value)]] += 1

    truth_area = np.array([(truth == l).sum() for l in truth_labels], dtype=np.int64)
    pred_area = np.array([(prediction == l).sum() for l in pred_labels], dtype=np.int64)
    union = truth_area[:, None] + pred_area[None, :] - intersection
    return intersection / np.maximum(union, 1)


def match(ious: np.ndarray, threshold: float = DEFAULT_IOU) -> list[tuple[int, int]]:
    """The largest set of pairs that all clear the threshold, ties by IoU."""
    from scipy.optimize import linear_sum_assignment

    if ious.size == 0:
        return []
    valid = np.where(ious >= threshold, ious, 0.0)
    if not valid.any():
        return []
    score = np.where(valid > 0, _PAIR_WEIGHT + valid, 0.0)
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c] > 0]


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    ious: list[float]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_iou(self) -> float | None:
        return float(np.mean(self.ious)) if self.ious else None

    def to_row(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "mean_iou_of_matches": (
                round(self.mean_iou, 4) if self.mean_iou is not None else None
            ),
        }

    def __add__(self, other: "Score") -> "Score":
        return Score(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
                     self.ious + other.ious)


def score_image(
    truth: np.ndarray, prediction: np.ndarray, threshold: float = DEFAULT_IOU
) -> Score:
    """Score one image's instances against its labels."""
    ious = iou_matrix(truth, prediction)
    n_truth = len([v for v in np.unique(truth) if v])
    n_pred = len([v for v in np.unique(prediction) if v])
    pairs = match(ious, threshold)
    overlaps = [float(ious[r, c]) for r, c in pairs]
    return Score(len(pairs), n_pred - len(pairs), n_truth - len(pairs), overlaps)


def score_all(pairs, threshold: float = DEFAULT_IOU) -> Score:
    """Pool scores over an iterable of (truth, prediction) pairs."""
    total = Score(0, 0, 0, [])
    for truth, prediction in pairs:
        total = total + score_image(truth, prediction, threshold)
    return total
