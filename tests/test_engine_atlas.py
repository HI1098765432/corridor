"""Ground truth for Bot 2 (static atlas).

The acceptance in ``docs/ENGINE_4D.md`` section 4: "walls are marked static,
and a paused cell is not".  The critical invariant the module exists to
protect is that low temporal change ALONE never marks a region static -- a
paused or resting cell does not move, yet it is a cell.  These tests plant a
persistent, field-spanning wall (which must become ``CHANNEL_WALL``) and cells
that stop moving (which must stay ``VALID_CELL_REGION``), on fields sized like
the real KK2 crops (channels ~324 px tall), so a wall really does span more
than the longest plausible cell (205.7 px).
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane
from corridor.engine.static_atlas import (
    ARTIFACT,
    CHANNEL_WALL,
    STATIC_BACKGROUND,
    VALID_CELL_REGION,
    AtlasConfig,
    AtlasResult,
    build_atlas,
)

RNG_BACKGROUND = 100.0
WALL_AMP = 1200.0
CELL_AMP = 1000.0


def _background(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return RNG_BACKGROUND + rng.normal(0.0, 3.0, shape).astype(np.float32)


def _stamp_ellipse(frame, cy, cx, ry, rx, amp):
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    frame[mask] += amp


def _add_vertical_wall(stack, x0, width=3, amp=WALL_AMP):
    stack[:, :, x0 : x0 + width] += amp


# --------------------------------------------------------------------------
# A moving cell that pauses for several frames
# --------------------------------------------------------------------------

# The cell rests at y=130 for frames 3..6 (4 of 10 frames) -- "several", but a
# minority, so its persistence stays below the 0.9 floor.
_CELL_Y = [60, 90, 110, 130, 130, 130, 130, 160, 190, 220]
_CELL_X = 45
_PAUSE_Y = 130


def _moving_cell_stack(seed=0):
    shape = (10, 300, 90)
    stack = _background(shape, seed)
    _add_vertical_wall(stack, x0=10)
    for t, cy in enumerate(_CELL_Y):
        _stamp_ellipse(stack[t], cy, _CELL_X, ry=20, rx=5, amp=CELL_AMP)
    return stack


def test_persistent_wall_is_channel_wall():
    atlas = build_atlas(_moving_cell_stack())
    # the wall column, sampled well away from the frame ends
    for y in (50, 150, 250):
        assert atlas.class_map[y, 11] == CHANNEL_WALL


def test_paused_cell_is_not_static():
    atlas = build_atlas(_moving_cell_stack())
    # the paused cell body must be cell territory, never device
    yy, xx = np.ogrid[:300, :90]
    body = ((yy - _PAUSE_Y) / 20.0) ** 2 + ((xx - _CELL_X) / 5.0) ** 2 <= 1.0
    assert not np.any(atlas.class_map[body] == STATIC_BACKGROUND)
    assert not np.any(atlas.class_map[body] == CHANNEL_WALL)
    assert not np.any(atlas.class_map[body] == ARTIFACT)
    assert atlas.class_map[_PAUSE_Y, _CELL_X] == VALID_CELL_REGION


def test_empty_background_is_static_background():
    atlas = build_atlas(_moving_cell_stack())
    # a column the cell never visits (cell is at x in [40,50]) and off the wall
    assert atlas.class_map[250, 70] == STATIC_BACKGROUND


# --------------------------------------------------------------------------
# The invariant: persistence alone is not enough (the extent gate)
# --------------------------------------------------------------------------


def test_still_cell_embedded_in_background_is_not_static():
    # A cell that never moves is static for the WHOLE series (persistence ~1),
    # yet it is cell-sized and must stay VALID_CELL_REGION.  This is the case
    # persistence alone would get wrong; the size/extent gate catches it, and
    # splitting flat vs bright static keeps the bright cell out of the
    # field-spanning background component.
    shape = (10, 300, 90)
    stack = _background(shape)
    for t in range(shape[0]):
        _stamp_ellipse(stack[t], 150, 45, ry=20, rx=5, amp=CELL_AMP)
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    body = ((yy - 150) / 20.0) ** 2 + ((xx - 45) / 5.0) ** 2 <= 1.0
    atlas = build_atlas(stack)
    # nowhere in the cell body may read as device
    assert not np.any(atlas.class_map[body] == STATIC_BACKGROUND)
    assert not np.any(atlas.class_map[body] == CHANNEL_WALL)
    assert not np.any(atlas.class_map[body] == ARTIFACT)
    assert atlas.class_map[150, 45] == VALID_CELL_REGION


# --------------------------------------------------------------------------
# Geometry combination and the occupancy veto
# --------------------------------------------------------------------------


def test_geometry_lane_is_marked_channel_wall():
    # A flat, fully-persistent field with no bright wall: only the passed lane
    # geometry can make a region CHANNEL_WALL, proving the combine path.
    shape = (16, 300, 90)
    stack = _background(shape)
    lane = Lane(index=0, origin=(45.0, 0.0), direction=(0.0, 1.0), half_width_px=20.0)
    geometry = ChannelGeometry(lanes=[lane], source=GEOMETRY_FROM_RIDGES, image_shape=(300, 90))
    atlas = build_atlas(stack, geometry=geometry)
    assert np.all(atlas.class_map[:, 44:47] == CHANNEL_WALL)  # within 3 px of x=45
    # far from the lane the flat field is background (robust to a stray flicker)
    assert np.mean(atlas.class_map[100:200, 5:15] == STATIC_BACKGROUND) > 0.95


def test_occupancy_vetoes_device_classification():
    # A field-spanning bright persistent bar would be a wall, but an occupancy
    # mask (a tracked cell was here) forbids any device label.
    shape = (16, 300, 90)
    stack = _background(shape)
    _add_vertical_wall(stack, x0=40, width=6)
    occupancy = np.zeros(shape[1:], dtype=bool)
    occupancy[:, 40:46] = True
    atlas = build_atlas(stack, occupancy=occupancy)
    assert np.all(atlas.class_map[:, 40:46] == VALID_CELL_REGION)


# --------------------------------------------------------------------------
# Background export, residual, and the evidence contract
# --------------------------------------------------------------------------


def test_residual_removes_the_background():
    stack = _moving_cell_stack()
    atlas = build_atlas(stack)
    residual = atlas.residual(stack)
    assert residual.shape == stack.shape
    # a background voxel's residual is near zero; a cell-present voxel's is large
    assert abs(float(residual[0, 250, 70])) < 5 * 3.0  # a few noise sigmas
    assert float(residual[0, _CELL_Y[0], _CELL_X]) > 100.0  # the cell at t=0


def test_class_counts_and_fractions_are_consistent():
    atlas = build_atlas(_moving_cell_stack())
    counts = atlas.class_counts()
    assert sum(counts.values()) == atlas.class_map.size
    payload = atlas.to_dict()
    assert abs(sum(payload["class_fractions"].values()) - 1.0) < 1e-9
    assert payload["has_z"] is False
    assert atlas.z_continuity is None  # Z disappears cleanly in 2-D


# --------------------------------------------------------------------------
# 3-D: a wall that runs through Z
# --------------------------------------------------------------------------


def test_3d_wall_through_z_is_channel_wall():
    # Y spans 280 px (> the 260 px longest-cell bound), so the wall is
    # device-scale on the Y axis; it is present in every Z plane, so it is
    # Z-continuous.
    shape = (5, 10, 280, 48)
    rng = np.random.default_rng(0)
    stack = RNG_BACKGROUND + rng.normal(0.0, 3.0, shape).astype(np.float32)
    stack[:, :, :, 8:11] += WALL_AMP  # bright sheet at x in [8,11), all Z, all Y, all T
    atlas = build_atlas(stack)
    assert atlas.has_z
    assert atlas.z_continuity is not None
    assert atlas.class_map[5, 140, 9] == CHANNEL_WALL
    assert atlas.class_map[5, 140, 30] == STATIC_BACKGROUND


def test_bad_dimensionality_is_rejected():
    with pytest.raises(ValueError):
        build_atlas(np.zeros((5, 5)))
