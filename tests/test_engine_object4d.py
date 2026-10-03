"""Bot 6a on synthetic ground truth (``docs/ENGINE_4D.md`` acceptance).

There is no 3-D ground truth in the supplied data, so -- as the 2.0 contract
promises -- the measurement bot is validated on digitised shapes with a known
answer: a growing sphere whose volume we compute exactly, and a translating
cell whose velocity we set.  Every tolerance is a number this test measured and
pinned, not a hope.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core.detections import extract_detections, extract_detections_3d
from corridor.engine.object4d import Calibration4D, assemble_tubes, tube_from_detections

# the supplied datasets' calibration, so the unit conversions are the real ones
PIXEL_UM = 0.467060342995564
INTERVAL_MIN = 20.006894938151042


def sphere_labels(radius_vox: float, shape=(48, 64, 64), centre=(24, 32, 32)) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    inside = (
        ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
        <= radius_vox**2
    )
    return inside.astype(np.int32)


def disk_mask(radius_px: float, shape=(80, 80), centre=(40, 40)) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) <= radius_px**2


# --------------------------------------------------------------------------
# Known volume growth is recovered within the stated tolerance.
# --------------------------------------------------------------------------


def test_known_volume_growth_recovered_within_3_percent():
    """A sphere grown 6 -> 7 -> 8 voxels at isotropic 0.5 um.

    Isotropic 0.5 um sampling puts 12-16 slices through these spheres, well
    above MIN_RELIABLE_Z_SLICES, so the volume-matched mesh reads the volume to
    within a few percent (the table in ``extract_detections_3d``).  Measured
    here: +2.2 %, -1.2 %, -1.7 % against 4/3 pi r^3.
    """
    spacing = (0.5, 0.5, 0.5)  # dz, dy, dx in um
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    radii_vox = [6.0, 7.0, 8.0]
    dets = []
    for frame, r in enumerate(radii_vox):
        d = extract_detections_3d(sphere_labels(r), frame=frame, spacing_zyx_um=spacing)[0]
        dets.append(d)

    tube = tube_from_detections(track_id=1, detections=dets, calibration=calib)
    assert tube.is_3d
    assert tube.n_points == 3

    for point, r in zip(tube.points, radii_vox):
        true_um3 = 4.0 / 3.0 * math.pi * (r * 0.5) ** 3
        assert point.volume_um3 is not None
        assert abs(point.volume_um3 / true_um3 - 1.0) < 0.03

    # dV/dt is a gap-aware finite difference of the measured volumes, in um^3/min
    assert tube.points[0].dV_dt_um3_per_min is None  # no previous point
    for prev, cur in zip(tube.points[:-1], tube.points[1:]):
        expected = (cur.volume_um3 - prev.volume_um3) / 10.0  # dt = 1 frame * 10 min
        assert cur.dV_dt_um3_per_min == pytest.approx(expected, rel=1e-9)
        assert cur.dV_dt_um3_per_min > 0  # the sphere is growing


def test_volume_rate_is_zero_for_a_static_object():
    """No growth, no dV/dt -- the bot does not manufacture a rate from noise."""
    spacing = (0.5, 0.5, 0.5)
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    dets = [
        extract_detections_3d(sphere_labels(7.0), frame=f, spacing_zyx_um=spacing)[0]
        for f in range(3)
    ]
    tube = tube_from_detections(1, dets, calib)
    for cur in tube.points[1:]:
        assert cur.dV_dt_um3_per_min == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Known 3-D translation is recovered as velocity; acceleration is a second diff.
# --------------------------------------------------------------------------


def test_known_translation_recovered_as_velocity_3d():
    """A sphere stepping (dx, dy, dz) = (2, 1, 1) voxels per 10-min frame."""
    spacing = (0.5, 0.5, 0.5)
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    dets = []
    for frame in range(3):
        centre = (24 + frame * 1, 32 + frame * 1, 32 + frame * 2)  # (z, y, x)
        d = extract_detections_3d(
            sphere_labels(8.0, centre=centre), frame=frame, spacing_zyx_um=spacing
        )[0]
        dets.append(d)
    tube = tube_from_detections(1, dets, calib)

    # velocity is (dx, dy, dz) * 0.5 um / 10 min = (0.1, 0.05, 0.05) um/min
    v = tube.points[1].velocity_um_per_min
    assert v is not None and len(v) == 3
    assert v[0] == pytest.approx(0.1, abs=5e-3)
    assert v[1] == pytest.approx(0.05, abs=5e-3)
    assert v[2] == pytest.approx(0.05, abs=5e-3)
    # constant velocity -> acceleration ~ 0
    a = tube.points[2].acceleration_um_per_min2
    assert a is not None
    assert tube.points[2].accel_magnitude_um_per_min2 == pytest.approx(0.0, abs=5e-3)


# --------------------------------------------------------------------------
# A 2-D movie is Z = 1 and runs through the same code (contract section 1).
# --------------------------------------------------------------------------


def test_2d_movie_reports_footprint_not_volume():
    """A moving 2-D disk: area and dA/dt filled, volume strictly None."""
    calib = Calibration4D(
        pixel_size_um=PIXEL_UM, z_step_um=None, frame_interval_min=INTERVAL_MIN
    )
    dets = []
    for frame in range(3):
        mask = disk_mask(12.0, centre=(40, 30 + frame * 6))  # steps +6 px in x
        d = extract_detections(mask.astype(np.int32), frame=frame)[0]
        dets.append(d)
    tube = tube_from_detections(1, dets, calib)

    assert tube.ndim == 2
    for point in tube.points:
        # volume is never guessed from a 2-D movie (the 2.0 law)
        assert point.volume_vox is None
        assert point.volume_um3 is None
        assert point.surface_area_um2 is None
        assert point.area_px > 0

    # velocity is +6 px/frame in x -> 6 * PIXEL_UM / INTERVAL_MIN um/min
    v = tube.points[1].velocity_um_per_min
    assert v is not None and len(v) == 2
    assert v[0] == pytest.approx(6.0 * PIXEL_UM / INTERVAL_MIN, rel=1e-6)
    assert v[1] == pytest.approx(0.0, abs=1e-9)
    # the disk does not change size
    assert tube.points[1].dA_dt_um2_per_min == pytest.approx(0.0, abs=1e-6)


def test_uncalibrated_tube_reports_pixels_only():
    """Without a pixel size, every um field is None but px kinematics survive."""
    calib = Calibration4D()  # nothing calibrated
    dets = [
        extract_detections(disk_mask(10.0, centre=(40, 30 + f * 5)).astype(np.int32), frame=f)[0]
        for f in range(2)
    ]
    tube = tube_from_detections(1, dets, calib)
    p = tube.points[1]
    assert p.centroid_um is None
    assert p.velocity_um_per_min is None
    assert p.speed_um_per_min is None
    assert p.velocity_px_per_frame is not None
    assert p.velocity_px_per_frame[0] == pytest.approx(5.0, abs=1e-6)


# --------------------------------------------------------------------------
# A gap of more than one frame is handled by elapsed time, not frame count.
# --------------------------------------------------------------------------


def test_gap_aware_velocity():
    """Frames 0 and 2 (one missing): velocity divides by the elapsed time."""
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    d0 = extract_detections_3d(sphere_labels(8.0, centre=(24, 32, 32)), frame=0,
                               spacing_zyx_um=(0.5, 0.5, 0.5))[0]
    d2 = extract_detections_3d(sphere_labels(8.0, centre=(24, 32, 36)), frame=2,
                               spacing_zyx_um=(0.5, 0.5, 0.5))[0]
    tube = tube_from_detections(1, [d0, d2], calib)
    p = tube.points[1]
    assert p.gap_frames == 2
    assert p.dt_min == pytest.approx(20.0)  # 2 frames * 10 min, not 10
    # moved +4 vox in x = 2 um over 20 min -> 0.1 um/min
    assert p.velocity_um_per_min[0] == pytest.approx(0.1, abs=5e-3)


def test_to_rows_flattens_vectors():
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    dets = [
        extract_detections_3d(sphere_labels(7.0, centre=(24, 32, 32 + f)), frame=f,
                              spacing_zyx_um=(0.5, 0.5, 0.5))[0]
        for f in range(2)
    ]
    rows = tube_from_detections(5, dets, calib).to_rows()
    assert rows[0]["track_id"] == 5
    assert {"x_px", "y_px", "z_px", "vx_um_per_min", "volume_um3"} <= set(rows[1])
    assert rows[1]["vx_um_per_min"] is not None


def test_assemble_tubes_from_tracks_keeps_id_order():
    """The Track-based entry point builds a tube per track, in id order."""
    from corridor.core.tracking import Track

    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    tracks = []
    for tid, x0 in ((3, 30), (1, 40)):
        tr = Track(id=tid)
        for frame in range(2):
            d = extract_detections_3d(
                sphere_labels(7.0, centre=(24, 32, x0 + frame * 2)),
                frame=frame, spacing_zyx_um=(0.5, 0.5, 0.5),
            )[0]
            tr.observe(d)
        tracks.append(tr)

    tubes = assemble_tubes(tracks, calib)
    assert [t.track_id for t in tubes] == [1, 3]
    assert all(t.n_points == 2 for t in tubes)
