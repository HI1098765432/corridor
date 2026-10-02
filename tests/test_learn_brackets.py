"""The time-aware bracket rule on synthetic label images with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corridor.learn.brackets import (
    BracketRule,
    bracket_of,
    ceiling,
    ceiling_interval,
    diagnose,
    distance_rule_partners,
    find_undrawn,
    leave_one_out,
)
from corridor.learn.sequences import Sequence

SHAPE = (240, 300)


def sequence(t_indices, *, interval_min=20.0, px=0.5, offsets=None) -> Sequence:
    n = len(t_indices)
    return Sequence(
        group="KK2", shape=SHAPE, paths=[Path(f"s_{i}.tif") for i in range(n)],
        ordering="time", experiment_id="20240101-s01", t_indices=list(t_indices),
        frame_interval_min=interval_min, pixel_size_um=px,
        offsets_px=list(offsets) if offsets else [(0.0, 0.0)] * n, link_ncc=[1.0] * n)


def cell(mask: np.ndarray, label: int, x: float, y: float, length=60, width=8) -> np.ndarray:
    """A long thin cell, as in the channels: an ellipse centred on (x, y)."""
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    inside = ((yy - y) / (length / 2)) ** 2 + ((xx - x) / (width / 2)) ** 2 <= 1.0
    mask[inside] = label
    return mask


def frames(*cells_per_frame) -> list[np.ndarray]:
    out = []
    for cells in cells_per_frame:
        m = np.zeros(SHAPE, np.int32)
        for label, (x, y) in enumerate(cells, start=1):
            cell(m, label, x, y)
        out.append(m)
    return out


def test_a_missed_cell_is_placed_by_elapsed_time_not_the_midpoint():
    seq = sequence([1, 2, 5])  # bracket t1-t5 = 80 min, target 1/4 of the way
    masks = frames([(100, 100)], [], [(100, 120)])
    claims, checks = find_undrawn(seq, masks)
    assert [c.checkable for c in checks] == [False, True, False]
    (claim,) = claims
    assert claim.target == 1 and claim.fraction == pytest.approx(0.25)
    assert claim.at == pytest.approx((100.0, 105.0), abs=0.6)  # midpoint would be 110
    assert claim.span_min == pytest.approx(80.0)
    # Evidence is never modified.
    assert not masks[1].any()


def test_a_bracket_longer_than_the_limit_is_refused():
    seq = sequence([1, 2, 6])  # 100 min > 90
    claims, checks = find_undrawn(seq, frames([(100, 100)], [], [(100, 104)]))
    assert claims == []
    assert not checks[1].checkable and "spans 100 min" in checks[1].reason
    assert "not checkable" in diagnose(seq, frames([(100, 100)], [], [(100, 104)]), 1, (100, 102))


def test_a_cell_in_the_next_channel_is_not_the_same_cell():
    seq = sequence([1, 2, 3])
    # Centroids 30 px apart -- inside the withdrawn 45 px rule -- but the masks
    # never touch, as with cells either side of a channel wall.
    claims, _ = find_undrawn(seq, frames([(100, 100)], [], [(130, 100)]))
    assert claims == []


def test_a_drawn_cell_touching_the_outline_accounts_for_it():
    seq = sequence([1, 2, 3])
    # The drawn cell at t2 is 40 px along from the interpolated position: its
    # centroid is far (the withdrawn rule would have added a cell) but its body
    # touches the interpolated outline.
    masks = frames([(100, 100)], [(100, 158)], [(100, 104)])
    claims, _ = find_undrawn(seq, masks)
    assert claims == []
    assert diagnose(seq, masks, 1, (100, 102)) == "a drawn cell touches the time-interpolated outline"


def test_an_ambiguous_far_side_makes_no_claim():
    seq = sequence([1, 2, 3])
    masks = frames([(100, 100)], [], [(100, 80), (100, 130)])  # both overlap the t1 cell
    assert find_undrawn(seq, masks)[0] == []


def test_registration_offsets_are_applied_both_ways():
    # Frame 1 is offset by (dy, dx) = (10, -6): a point at p in it is at
    # p + (10, -6) in the reference. Its cell is drawn in its own pixels.
    seq = sequence([1, 2, 3], offsets=[(0.0, 0.0), (10.0, -6.0), (0.0, 0.0)])
    masks = frames([(100, 100)], [], [(100, 100)])
    (claim,) = find_undrawn(seq, masks)[0]
    assert claim.at == pytest.approx((106.0, 90.0), abs=0.6)


def test_a_run_of_missing_frames_is_not_filled_by_chaining():
    seq = sequence([1, 2, 3, 4])
    claims, _ = find_undrawn(seq, frames([(100, 100)], [], [], [(100, 103)]))
    assert claims == []


def test_both_stills_of_one_time_point_are_targets_with_one_bracket():
    seq = sequence([1, 2, 2, 3])
    masks = frames([(100, 100)], [], [], [(100, 104)])
    claims, checks = find_undrawn(seq, masks)
    assert sorted(c.target for c in claims) == [1, 2]
    assert checks[1].bracket.before == 0 and checks[1].bracket.after == 3


def test_a_frame_that_is_mostly_outside_the_image_makes_no_claim():
    seq = sequence([1, 2, 3])
    # Cells leaving through the top edge: the interpolated outline has under
    # MIN_ADDED_PX pixels inside the frame.
    masks = frames([(100, -26)], [], [(100, -27)])
    assert find_undrawn(seq, masks)[0] == []


def test_first_and_last_frames_cannot_be_checked():
    seq = sequence([4, 9, 12])
    assert bracket_of(seq, 0).reason == "first labelled time of its crop"
    assert bracket_of(seq, 2).reason == "last labelled time of its crop"
    legacy = Sequence(group="KK2", shape=SHAPE, paths=[Path("a.tif")] * 3)
    assert bracket_of(legacy, 1).reason == "no true time order"


def test_a_frame_beside_an_unannotated_one_is_not_checkable():
    # t2 and t4 have cells; t3, between them, has nothing drawn at all. t2 and
    # t4 each have t3 on one side of their bracket, so no undrawn cell could
    # ever be found there, and their labelled cells must not enter a ceiling.
    seq = sequence([1, 2, 3, 4, 5])
    masks = frames([(100, 100)], [(100, 101)], [], [(100, 103)], [(100, 104)])
    claims, checks = find_undrawn(seq, masks)
    assert [c.checkable for c in checks] == [False, False, True, False, False]
    assert checks[1].reason == "nothing drawn at t3 (s_2.tif, the later side of the bracket)"
    assert checks[3].reason == "nothing drawn at t3 (s_2.tif, the earlier side of the bracket)"
    assert checks[1].labelled == 1 and checks[1].linked_pairs == 0
    assert [c.target for c in claims] == [2] and checks[2].linked_pairs == 1
    assert diagnose(seq, masks, 1, (100, 101)).startswith("not checkable: nothing drawn at t3")


def test_a_frame_whose_bracket_links_no_cell_is_not_checkable():
    # Cells drawn on both sides, but none overlaps across the bracket: the rule
    # had nothing to test, so the frame did not test anything.
    seq = sequence([1, 2, 3])
    _, checks = find_undrawn(seq, frames([(100, 100)], [(200, 100)], [(160, 100)]))
    assert not checks[1].checkable and "nothing to test" in checks[1].reason


def test_leave_one_out_counts_drawn_cells_where_predicted():
    seq = sequence([1, 2, 3])
    rows = leave_one_out(seq, frames([(100, 100)], [(100, 102)], [(100, 104)]))
    assert rows == [{"image": "s_1.tif", "span_min": 40.0, "target_drawn": 1,
                     "drawn_where_predicted": True}]
    # Into a frame with nothing drawn the pair always fails: that row is a
    # missing label, and carries target_drawn 0 so a calibration can drop it.
    rows = leave_one_out(seq, frames([(100, 100)], [], [(100, 104)]))
    assert rows[0]["drawn_where_predicted"] is False and rows[0]["target_drawn"] == 0
    rows = leave_one_out(seq, frames([(100, 100)], [(220, 100)], [(100, 104)]))
    assert rows[0]["drawn_where_predicted"] is False and rows[0]["target_drawn"] == 1


def test_identity_by_distance_alone_is_recorded():
    # 80 min at 3 um/min plus 5 um is a 245 um (490 px) radius: both t1 cells
    # are inside it for both t0 cells, so neither pairing is unique.
    seq = sequence([1, 2, 5])
    masks = frames([(100, 100), (160, 100)], [], [(100, 104), (160, 104)])
    assert distance_rule_partners(seq, masks) == {"bracketed_t0_cells": 2,
                                                  "with_a_unique_partner": 0}
    one = frames([(100, 100)], [], [(100, 104)])
    assert distance_rule_partners(seq, one)["with_a_unique_partner"] == 1


def test_the_ceiling_interval_is_exact_and_maps_to_f1():
    ci = ceiling_interval(43, 2)
    # scipy's exact binomial interval for 2 of 45.
    assert ci["missing_rate"] == [0.0054, 0.1515]
    assert ci["f1"][0] == pytest.approx(2 * (1 - 0.15149) / (2 - 0.15149), abs=1e-4)
    assert ci["f1"][0] < ceiling(43, 2)["f1"] < ci["f1"][1]
    assert ceiling_interval(0, 0) is None


def test_the_ceiling_formula_reproduces_the_withdrawn_figures():
    # tp = labelled, fp = undrawn; P = tp/(tp+fp); F1 = 2P/(P+1).
    assert ceiling(57, 7)["f1"] == 0.9421
    assert ceiling(138, 9)["f1"] == 0.9684
    assert ceiling(10, 0)["f1"] == 1.0
    assert ceiling(0, 0)["f1"] is None


def test_the_rule_is_recorded_in_full():
    record = BracketRule().to_dict()
    assert record["max_bracket_min"] == 90.0 and record["max_speed_um_per_min"] == 3.0
    assert "elapsed fraction" in record["position"]
