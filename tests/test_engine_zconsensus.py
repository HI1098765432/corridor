"""Ground-truth tests for Bot 4 (Z consensus), ENGINE_4D section 4.

Acceptance: a dropped Z slice is repaired, and two touching objects stay
separate.  Every answer here is known by construction, so the numbers the bot
produces are checked against the truth, not asserted.
"""

from __future__ import annotations

import numpy as np

from corridor.engine.z_consensus import (
    ZConsensusSettings,
    link_z,
)


def _disk(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def test_2d_stack_is_a_passthrough():
    # Z = 1: every component is its own single-slice object, no links.
    h, w = 40, 40
    labels = np.zeros((1, h, w), dtype=np.int32)
    labels[0][_disk(h, w, 12, 12, 5)] = 1
    labels[0][_disk(h, w, 28, 28, 5)] = 2

    res = link_z(labels)

    assert res.passthrough is True
    assert res.n_objects == 2
    assert res.n_links_accepted == 0
    assert res.n_repaired_slices == 0
    # the single slice carries two distinct object ids
    assert set(np.unique(res.object_labels_zyx)) == {0, 1, 2}


def test_dropped_slice_is_repaired():
    # A straight tube: a disk present on slices 0, 1, 3, 4 and MISSING on 2.
    depth, h, w = 5, 48, 48
    cy, cx, r = 24, 24, 7
    labels = np.zeros((depth, h, w), dtype=np.int32)
    true_disk = _disk(h, w, cy, cx, r)
    for z in (0, 1, 3, 4):
        labels[z][true_disk] = 1

    res = link_z(labels)

    # one object spanning all five slices, slice 2 repaired
    assert res.n_objects == 1
    obj = res.objects[0]
    assert obj.n_slices == 5
    assert 2 in obj.repaired_slices
    assert res.n_repaired_slices == 1

    # the repaired slice is present in the label volume and matches the truth,
    # because the two neighbours are identical disks
    repaired = res.object_labels_zyx[2] > 0
    assert repaired.any()
    assert _iou(repaired, true_disk) > 0.8

    # the bridge link is reported with its cost, and it is a strong (cheap) link
    repairs = [l for l in res.links if l.kind == "repair"]
    assert len(repairs) == 1
    assert repairs[0].cost < ZConsensusSettings().strong_link_cost
    assert repairs[0].iou > ZConsensusSettings().min_bridge_iou


def test_two_slice_gap_is_not_repaired():
    # Only a single missing slice is bridged; a two-slice gap stays two objects.
    depth, h, w = 6, 48, 48
    cy, cx, r = 24, 24, 7
    labels = np.zeros((depth, h, w), dtype=np.int32)
    disk = _disk(h, w, cy, cx, r)
    for z in (0, 1, 4, 5):  # slices 2 and 3 both empty
        labels[z][disk] = 1

    res = link_z(labels)

    assert res.n_objects == 2
    assert res.n_repaired_slices == 0


def test_two_touching_objects_stay_separate():
    # Two tangent tubes, present on every slice.  A naive linker could cross-
    # link them between slices; the assignment plus the IoU/centroid gate must
    # keep them apart.
    depth, h, w = 5, 48, 48
    r = 6
    cx_left, cx_right = 16, 16 + 2 * r  # tangent disks, 12 px apart
    labels = np.zeros((depth, h, w), dtype=np.int32)
    left = _disk(h, w, 24, cx_left, r)
    right = _disk(h, w, 24, cx_right, r)
    for z in range(depth):
        labels[z][left] = 1
        labels[z][right] = 2

    res = link_z(labels)

    assert res.n_objects == 2
    for obj in res.objects:
        assert obj.n_slices == depth
        assert obj.repaired_slices == []

    # the two objects never merged: each slice keeps both ids, and no accepted
    # link ever joined a left component to a right one
    vol = res.object_labels_zyx
    ids = sorted(int(v) for v in np.unique(vol) if v)
    assert ids == [1, 2]
    for z in range(depth):
        assert (vol[z] == 1).any() and (vol[z] == 2).any()
    cross = [l for l in res.links if l.accepted and l.label_from != l.label_to]
    assert cross == []


def test_a_forced_sideways_link_is_rejected_by_the_gate():
    # Column A occupies slices 0-1 and column B (a tangent, different cell)
    # occupies slices 2-3.  Between slice 1 and slice 2 the only possible
    # pairing is A->B; the assignment is forced to select it, and the gate must
    # reject it so the two cells stay separate.
    depth, h, w = 4, 48, 48
    r = 6
    a = _disk(h, w, 24, 16, r)
    b = _disk(h, w, 24, 16 + 2 * r, r)
    labels = np.zeros((depth, h, w), dtype=np.int32)
    labels[0][a] = 1
    labels[1][a] = 1
    labels[2][b] = 1
    labels[3][b] = 1

    res = link_z(labels)

    assert res.n_objects == 2
    assert res.n_repaired_slices == 0
    forced = [l for l in res.links if l.z_from == 1 and l.z_to == 2]
    assert len(forced) == 1, "the 1->2 pairing must be selected and reported"
    assert forced[0].accepted is False
    assert forced[0].cost > ZConsensusSettings().max_link_cost


def test_every_link_cost_is_reported_and_finite():
    depth, h, w = 4, 40, 40
    labels = np.zeros((depth, h, w), dtype=np.int32)
    disk = _disk(h, w, 20, 20, 6)
    for z in range(depth):
        labels[z][disk] = 1

    res = link_z(labels)

    assert res.links, "a multi-slice tube must produce adjacent links"
    for link in res.links:
        assert np.isfinite(link.cost)
        assert 0.0 <= link.iou <= 1.0
        d = link.to_dict()
        assert set(["iou", "d_centroid_px", "ln_area_ratio", "d_shape", "cost"]) <= set(d)
    # a straight tube links at near-zero cost on every accepted step
    accepted = [l for l in res.links if l.accepted]
    assert accepted
    assert max(l.cost for l in accepted) < 0.1
