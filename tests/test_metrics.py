"""Instance matching, including the case four copies of it got wrong.

An external review pointed out that solving the assignment on IoU and *then*
applying the threshold maximises total overlap rather than the number of objects
found. On this project's data the two rules happen to agree on every image, so
no published figure changes -- but a metric that is right by luck on today's
data is not right, and the counterexample is easy to write down.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.metrics import iou_matrix, match, score_image


def rect(shape, box, value=1):
    out = np.zeros(shape, dtype=np.int32)
    r0, c0, r1, c1 = box
    out[r0:r1, c0:c1] = value
    return out


def test_a_perfect_prediction_scores_one():
    truth = rect((40, 40), (5, 5, 25, 25))
    score = score_image(truth, truth.copy())
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.f1 == 1.0
    assert score.mean_iou == pytest.approx(1.0)


def test_a_missed_cell_is_a_false_negative():
    truth = rect((40, 40), (5, 5, 25, 25))
    score = score_image(truth, np.zeros_like(truth))
    assert (score.tp, score.fp, score.fn) == (0, 0, 1)
    assert score.recall == 0.0


def test_an_invented_cell_is_a_false_positive():
    prediction = rect((40, 40), (5, 5, 25, 25))
    score = score_image(np.zeros_like(prediction), prediction)
    assert (score.tp, score.fp, score.fn) == (0, 1, 0)
    assert score.precision == 0.0


def test_an_overlap_below_the_threshold_is_not_a_match():
    truth = rect((40, 40), (0, 0, 10, 20))
    prediction = rect((40, 40), (0, 12, 10, 32))  # 8 of 32 columns shared
    assert iou_matrix(truth, prediction)[0, 0] < 0.5
    score = score_image(truth, prediction)
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)


def test_matching_maximises_the_number_of_matches_not_the_total_overlap():
    """The defect the review found, as a matrix.

    Truth A overlaps X perfectly and Y at exactly the threshold; truth B
    overlaps X at the threshold and Y not at all. Maximising total IoU scores
    (A->X, B->Y) = 1.0, which is one valid pair. Maximising valid pairs finds
    (A->Y, B->X), which is two. Two is the right answer: both cells were found.
    """
    ious = np.array([[1.0, 0.5],
                     [0.5, 0.0]])

    pairs = match(ious, threshold=0.5)
    assert len(pairs) == 2, "both truths have a valid partner available"
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_ties_among_equally_large_solutions_are_broken_by_overlap():
    """With the same number of valid pairs available, prefer the better fit."""
    ious = np.array([[0.9, 0.6],
                     [0.6, 0.9]])
    pairs = match(ious, threshold=0.5)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_a_prediction_can_only_be_claimed_once():
    """One blob covering two cells is one match and one miss, never two matches."""
    truth = np.zeros((40, 60), dtype=np.int32)
    truth[10:30, 5:25] = 1
    truth[10:30, 35:55] = 2
    merged = np.zeros((40, 60), dtype=np.int32)
    merged[10:30, 5:55] = 1

    score = score_image(truth, merged)
    assert score.tp <= 1
    assert score.tp + score.fn == 2


def test_empty_against_empty_is_not_an_error():
    blank = np.zeros((20, 20), dtype=np.int32)
    score = score_image(blank, blank)
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    assert score.f1 == 0.0  # nothing to find and nothing found


def test_scores_add_across_images():
    truth = rect((40, 40), (5, 5, 25, 25))
    good = score_image(truth, truth.copy())
    missed = score_image(truth, np.zeros_like(truth))
    total = good + missed
    assert (total.tp, total.fp, total.fn) == (1, 0, 1)
    assert total.recall == 0.5
