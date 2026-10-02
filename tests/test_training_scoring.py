"""Research scoring: border margin, AP, splits/merges and breakdowns, on known answers."""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.metrics import score_image
from training import scoring

SHAPE = (100, 120)


def rect(mask, label, y0, y1, x0, x1):
    mask[y0:y1, x0:x1] = label
    return mask


def test_border_distance_is_diag_errors_definition():
    m = np.zeros(SHAPE, bool)
    m[0:10, 50:55] = True  # touches the top edge
    assert scoring.border_distance(m) == 0
    m = np.zeros(SHAPE, bool)
    m[20:30, 4:8] = True
    assert scoring.border_distance(m) == 4
    m = np.zeros(SHAPE, bool)
    m[40:96, 60:70] = True  # H-1-ys.max() = 99-95
    assert scoring.border_distance(m) == 4


def test_margin_drops_truth_and_prediction_before_counting():
    truth = rect(np.zeros(SHAPE, np.int32), 1, 0, 20, 10, 18)    # at the edge, missed
    rect(truth, 2, 40, 70, 50, 58)
    prediction = rect(np.zeros(SHAPE, np.int32), 7, 41, 70, 50, 58)
    rect(prediction, 9, 80, 100, 100, 110)                        # at the edge, spurious

    plain = scoring.score(truth, prediction)
    assert (plain.tp, plain.fp, plain.fn) == (1, 1, 1)
    excluded = scoring.score(truth, prediction, margin_px=0)
    assert (excluded.tp, excluded.fp, excluded.fn) == (1, 0, 0)
    # Without a margin this is exactly core.metrics.
    reference = score_image(truth, prediction)
    assert (plain.tp, plain.fp, plain.fn) == (reference.tp, reference.fp, reference.fn)
    # Labels that survive keep their ids.
    assert set(np.unique(scoring.drop_border(prediction, 0))) == {0, 7}


def test_ap_curve_counts_tp_over_tp_fp_fn_at_each_threshold():
    truth = rect(np.zeros(SHAPE, np.int32), 1, 10, 50, 10, 30)    # 40 x 20 = 800 px
    exact = truth.copy()
    # Shifted by 6 rows: intersection 34x20=680, union 920 -> IoU 0.739.
    shifted = rect(np.zeros(SHAPE, np.int32), 3, 16, 56, 10, 30)
    perfect = scoring.pool([scoring.evaluate_image("a", truth, exact)])
    assert set(perfect["ap_by_iou"].values()) == {1.0}
    assert perfect["ap_50_95"] == 1.0

    partial = scoring.pool([scoring.evaluate_image("a", truth, shifted)])
    ap = partial["ap_by_iou"]
    assert ap["0.7"] == 1.0 and ap["0.75"] == 0.0  # 1 TP; then 1 FP + 1 FN -> 0/2
    assert partial["ap_50_95"] == pytest.approx(0.5)
    assert partial["at_iou_0.5"]["f1"] == 1.0


def test_splits_and_merges_from_the_iou_matrix():
    truth = rect(np.zeros(SHAPE, np.int32), 1, 10, 60, 10, 20)
    rect(truth, 2, 10, 40, 40, 50)
    rect(truth, 3, 42, 70, 40, 50)
    prediction = rect(np.zeros(SHAPE, np.int32), 5, 10, 35, 10, 20)   # truth 1, top half
    rect(prediction, 6, 35, 60, 10, 20)                                 # truth 1, bottom half
    rect(prediction, 7, 10, 70, 40, 50)                                 # truths 2 and 3
    result = scoring.split_merge(truth, prediction)
    assert result["splits"] == 1 and result["split_truths"] == [1]
    assert result["merges"] == 1 and result["merging_predictions"] == [7]
    assert scoring.split_merge(truth, truth) == {"splits": 0, "merges": 0, "split_truths": [],
                                                 "merging_predictions": []}


def test_pool_reports_per_image_fp_and_fn():
    truth = rect(np.zeros(SHAPE, np.int32), 1, 10, 50, 10, 30)
    empty = np.zeros(SHAPE, np.int32)
    results = [scoring.evaluate_image("hit", truth, truth),
               scoring.evaluate_image("miss", truth, empty),
               scoring.evaluate_image("ghost", empty, truth)]
    pooled = scoring.pool(results)
    assert pooled["at_iou_0.5"]["tp"] == 1
    assert pooled["fp_per_image"] == pytest.approx(1 / 3, abs=1e-3)
    assert pooled["fn_per_image"] == pytest.approx(1 / 3, abs=1e-3)
    assert [r["image"] for r in pooled["per_image"]] == ["hit", "miss", "ghost"]


def test_breakdowns_bin_recall_and_precision():
    image = np.full(SHAPE, 100.0)
    truth = np.zeros(SHAPE, np.int32)
    rect(truth, 1, 0, 20, 10, 18)      # touching the edge, faint
    rect(truth, 2, 40, 70, 50, 58)     # interior, bright
    image[40:70, 50:58] = 400.0
    image[0:20, 10:18] = 120.0
    prediction = rect(np.zeros(SHAPE, np.int32), 4, 40, 70, 50, 58)
    rows = scoring.describe_instances("x", image, truth, prediction, group="20240101-s01")
    truths = {r.label: r for r in rows if r.side == "truth"}
    assert truths[1].border_px == 0 and not truths[1].matched
    assert truths[2].matched and truths[2].contrast > truths[1].contrast

    border = scoring.breakdown(rows, "border_px", scoring.BORDER_EDGES_PX)
    assert border[0]["bin"] == "(-inf, 0]" and border[0]["recall"] == 0.0
    assert border[-1]["recall"] == 1.0 and border[-1]["precision"] == 1.0
    by_experiment = scoring.breakdown(rows, "group")
    assert by_experiment == [{"bin": "20240101-s01", "truth": 2, "recall": 0.5,
                              "pred": 1, "precision": 1.0}]
    standard = scoring.standard_breakdowns(rows)
    assert set(standard) == {"experiment", "contrast_quartile", "size_quartile_px",
                             "border_distance_px"}
