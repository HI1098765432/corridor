"""The orthogonal viewer on a synthetic TZYX volume with a known bright box.

Each pane must cut the volume where the cursor says, in the right
orientation: XY is (Y, X) at the cursor's Z, XZ is (Z, X) at its Y, YZ is
(Y, Z) at its X (transposed so it shares XY's vertical axis). A transposed
pane looks plausible and is wrong, so the box is placed asymmetrically and
its position is checked in every pane.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from corridor.ui.widgets.ortho_viewer import OrthoViewer  # noqa: E402

T, Z, Y, X = 3, 8, 32, 40
BOX = (slice(2, 5), slice(10, 15), slice(20, 27))  # z, y, x


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def volume():
    stack = np.full((T, Z, Y, X), 10.0, dtype=np.float32)
    masks = np.zeros((T, Z, Y, X), dtype=np.int32)
    stack[(1,) + BOX] = 200.0
    masks[(1,) + BOX] = 7
    return stack, masks


def _viewer(qt_app, volume, anisotropy=None) -> OrthoViewer:
    stack, masks = volume
    viewer = OrthoViewer()
    viewer.resize(800, 600)
    viewer.set_volume(stack, masks, anisotropy=anisotropy)
    qt_app.processEvents()
    return viewer


def test_planes_cut_the_volume_through_the_cursor(qt_app, volume):
    viewer = _viewer(qt_app, volume)
    viewer.set_frame(1)
    viewer.set_cursor(23, 12, 3)
    planes = viewer.plane_arrays()
    assert planes["xy"].shape == (Y, X)
    assert planes["xz"].shape == (Z, X)
    assert planes["yz"].shape == (Y, Z)
    assert planes["xy"][12, 23] == 200.0 and planes["xy"][12, 5] == 10.0
    assert planes["xz"][3, 23] == 200.0 and planes["xz"][6, 23] == 10.0
    assert planes["yz"][12, 3] == 200.0 and planes["yz"][12, 6] == 10.0
    masks = viewer.mask_planes()
    assert masks["xz"][3, 23] == 7 and masks["yz"][12, 3] == 7
    assert not viewer.grab().isNull()


def test_time_and_z_are_independent(qt_app, volume):
    viewer = _viewer(qt_app, volume)
    viewer.set_cursor(23, 12, 3)
    assert viewer.plane_arrays()["xy"][12, 23] == 10.0, "frame 0 has no box"
    viewer.set_frame(1)
    assert viewer.plane_arrays()["xy"][12, 23] == 200.0
    viewer.z_slider.setValue(6)
    assert viewer.cursor[2] == 6
    assert viewer.plane_arrays()["xy"][12, 23] == 10.0, "slice 6 is above the box"
    assert "Slice 7 of 8" in viewer.z_readout.text()


def test_z_is_drawn_to_scale_only_when_the_step_is_known(qt_app, volume):
    unknown = _viewer(qt_app, volume)
    assert "unknown" in unknown.z_readout.text()
    scaled = _viewer(qt_app, volume, anisotropy=3.0)
    assert scaled.xz._aspect == (1.0, 3.0)
    assert scaled.yz._aspect == (3.0, 1.0)
    assert "to scale" in scaled.z_readout.text()


def test_the_selected_cell_is_under_the_crosshair(qt_app, volume):
    viewer = _viewer(qt_app, volume)
    rows = {1: [{"track_id": 5, "frame": 1, "x_px": 23.0, "y_px": 12.0, "z": 3.0}]}
    viewer.rows_for_frame = lambda f: rows.get(f, [])
    viewer.set_frame(1)
    viewer.select_track(5)
    assert viewer.cursor == (23.0, 12.0, 3)
    # The track's own marker is drawn on all three panes through it.
    assert [m[4] for m in viewer.xz.markers] == ["5"]
    assert [m[4] for m in viewer.yz.markers] == ["5"]


def test_picking_reports_a_three_d_point(qt_app, volume):
    viewer = _viewer(qt_app, volume)
    picked = []
    viewer.point_picked.connect(lambda x, y, z: picked.append((x, y, z)))
    viewer.set_cursor(10, 10, 4)
    viewer.set_picking(True)
    viewer.xy.clicked.emit(30.0, 20.0)
    assert picked == [(30.0, 20.0, 4.0)]
    viewer.xz.clicked.emit(15.0, 2.0)
    assert picked[-1] == (15.0, 10.0, 2.0)


def test_clicking_a_cell_selects_its_track(qt_app, volume):
    viewer = _viewer(qt_app, volume)
    rows = {0: [{"track_id": 9, "frame": 0, "x_px": 23.0, "y_px": 12.0, "z": 4.0}]}
    viewer.rows_for_frame = lambda f: rows.get(f, [])
    viewer.set_cursor(0, 0, 4)
    clicked = []
    viewer.track_clicked.connect(clicked.append)
    viewer.xy.clicked.emit(24.0, 13.0)
    assert clicked == [9]


def test_pane_coordinates_round_trip(qt_app, volume):
    viewer = _viewer(qt_app, volume, anisotropy=2.0)
    viewer.xz.resize(400, 200)
    point = viewer.xz.to_widget(17.0, 5.0)
    u, v = viewer.xz.to_plane(QPointF(point))
    assert u == pytest.approx(17.0) and v == pytest.approx(5.0)


def test_a_two_d_stack_is_refused(qt_app):
    viewer = OrthoViewer()
    with pytest.raises(ValueError):
        viewer.set_volume(np.zeros((4, 5), dtype=np.float32))
