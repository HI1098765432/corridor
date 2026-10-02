"""Napari layer data, built without opening Napari.

The layers are pure functions of the saved rows, so they are checked here
directly: 2-D tracks as ``[id, t, y, x]``, 3-D as ``[id, t, z, y, x]``, rows
that cannot be placed left out (never put on slice 0), unknown speeds NaN
(never 0, which would read as "stopped"), and no migration-axis layer.
"""

from __future__ import annotations

import inspect
import math

import numpy as np

from corridor.viz import napari_qc


ROWS = [
    {"track_id": 2, "frame": 1, "x_px": 5.0, "y_px": 6.0, "z": 2.0, "speed_um_per_min": 0.5},
    {"track_id": 1, "frame": 0, "x_px": 1.0, "y_px": 2.0, "z": 1.0, "speed_um_per_hr": 12.0},
    {"track_id": 2, "frame": 0, "x_px": 4.0, "y_px": 6.0, "z": None},
    {"track_id": None, "frame": 0, "x_px": 4.0, "y_px": 6.0},
]


def test_two_d_tracks_are_id_t_y_x_sorted():
    data, props = napari_qc.track_layer_data(ROWS, three_d=False)
    assert data.shape == (3, 4)
    assert data[:, :2].tolist() == [[1, 0], [2, 0], [2, 1]]
    assert data[0].tolist() == [1, 0, 2.0, 1.0]
    speeds = props["speed_um_per_hr"]
    assert speeds[0] == 12.0
    assert math.isnan(speeds[1]), "an unknown speed is NaN, not 0"
    assert speeds[2] == 30.0, "µm/min x 60"


def test_three_d_tracks_carry_z_and_drop_unplaceable_rows():
    data, _ = napari_qc.track_layer_data(ROWS, three_d=True)
    assert data.shape == (2, 5)
    assert data.tolist() == [[1, 0, 1.0, 2.0, 1.0], [2, 1, 2.0, 6.0, 5.0]]


def test_points_follow_the_stack_dimensionality():
    detections = [{"frame": 0, "x": 3.0, "y": 4.0, "z": 1.0}, {"frame": 1, "x": 3.0, "y": 4.0}]
    assert napari_qc.point_layer_data(detections, three_d=False).tolist() == [
        [0, 4.0, 3.0], [1, 4.0, 3.0]
    ]
    assert napari_qc.point_layer_data(detections, three_d=True).tolist() == [[0, 1.0, 4.0, 3.0]]
    assert napari_qc.point_layer_data([], three_d=True).shape == (0, 4)


def test_z_is_scaled_only_when_the_step_is_known():
    assert napari_qc.layer_scale(4, 3.2) == (1.0, 3.2, 1.0, 1.0)
    assert napari_qc.layer_scale(4, None) is None
    assert napari_qc.layer_scale(3, 3.2) is None
    assert napari_qc._anisotropy({"calibration": {"z_step_um": 2.0, "pixel_size_um": 0.5}}) == 4.0
    assert napari_qc._anisotropy({"calibration": {"pixel_size_um": 0.5}}) is None


def test_there_is_no_migration_axis_layer():
    source = inspect.getsource(napari_qc)
    assert "migration axis" not in source.replace("No migration-axis layer", "")
    assert "axis" not in inspect.signature(napari_qc.build_viewer).parameters
    assert np.asarray(napari_qc.track_layer_data([], three_d=True)[0]).shape == (0, 5)
