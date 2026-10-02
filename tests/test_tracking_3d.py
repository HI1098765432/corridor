"""3-D tracking on synthetic label volumes with a known answer.

There is no 3-D data in the supplied set (contract §0), so 3-D tracking is
validated here: ellipsoids rendered into anisotropic voxel grids, measured by
``extract_detections_3d`` exactly as an imported label image would be, and
tracked.  Z is tracked as ``z * anisotropy`` pixels, never as one slice per
pixel unless the Z step is unknown -- and then every track says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.config import Scale, TrackingConfig
from corridor.core.detections import Detection, extract_detections_3d
from corridor.core.tracking import (
    FLAG_MERGE,
    FLAG_Z_UNCALIBRATED,
    GATE_SPEED,
    MotionModel,
    explain_unlinked_starts,
    pair_cost,
    start_track,
    track_detections,
)

from conftest import identity_of, make_detection

PIXEL_UM = 0.5
Z_STEP_UM = 1.5  # anisotropy 3: one slice spans three XY pixels
FRAME_MIN = 10.0
SHAPE_ZYX = (14, 64, 110)


def ellipsoid_volume(cells, shape=SHAPE_ZYX, anisotropy=Z_STEP_UM / PIXEL_UM):
    """Label volume with one ellipsoid per ``(label, cx, cy, cz, ax, by, cz_px)``.

    Semi-axes are in XY pixels; Z is in slices and scaled by ``anisotropy``,
    so the rendered body has the same physical shape in every frame.
    """
    z, y, x = np.indices(shape, dtype=float)
    out = np.zeros(shape, dtype=np.int32)
    for label, cx, cy, cz, ax, by, cz_px in cells:
        inside = (
            ((x - cx) / ax) ** 2 + ((y - cy) / by) ** 2 + ((z - cz) * anisotropy / cz_px) ** 2
        ) <= 1.0
        out[inside] = label
    return out


def moving_ellipsoids(n_frames: int = 10):
    """A moves +5 px/frame in x and +0.2 slice/frame in z; B moves -5 px/frame, 18 px away in y."""
    dets = []
    for t in range(n_frames):
        volume = ellipsoid_volume([
            (1, 15 + 5 * t, 22, 4 + 0.2 * t, 14, 6, 6),
            (2, 95 - 5 * t, 40, 8.0, 14, 6, 6),
        ])
        dets += extract_detections_3d(volume, t, spacing_zyx_um=(Z_STEP_UM, PIXEL_UM, PIXEL_UM))
    return dets


@pytest.fixture
def scale3d() -> Scale:
    return Scale.from_values(PIXEL_UM, FRAME_MIN, z_step_um=Z_STEP_UM)


def test_two_ellipsoids_passing_keep_identity(scale3d):
    dets = moving_ellipsoids()
    assert all(d.ndim == 3 and d.mask_crop is not None for d in dets)
    tracks, events = track_detections(dets, 10, scale3d, TrackingConfig())
    assert len(tracks) == 2
    for label in (1, 2):
        assert len({identity_of(tracks, t, label) for t in range(10)}) == 1
    a = next(t for t in tracks if t.id == identity_of(tracks, 0, 1))
    assert all(o.z is not None for o in a.observations)
    # Velocity comes back in image units: px/frame in x, slices/frame in z.
    assert a.velocity[0] == pytest.approx(5.0, abs=0.5)
    assert a.velocity[2] == pytest.approx(0.2, abs=0.1)
    assert all(FLAG_Z_UNCALIBRATED not in t.flags for t in tracks)
    assert all(o.link_margin is not None for o in a.observations[1:])
    # The overlap term had 3-D masks to compare.
    assert all(o.breakdown.iou is not None for o in a.observations[1:])


def test_measurement_noise_uses_the_3d_principal_axes(scale3d):
    det = moving_ellipsoids(1)[0]
    model = MotionModel.from_config(scale3d, TrackingConfig(), 3)
    R = model.measurement_cov(det)
    assert R.shape == (3, 3)
    # The body is longest in x (28 px), 12 px in y and in z once Z is scaled:
    # R = sigma_p^2 I + (0.06 * 28)^2 x x^T + (0.12 * 12)^2 (y y^T + z z^T).
    sigma_p2 = model.position_sigma_px**2
    assert R[0, 0] == pytest.approx(sigma_p2 + (0.06 * 28) ** 2, rel=0.1)
    assert R[1, 1] == pytest.approx(sigma_p2 + (0.12 * 12) ** 2, rel=0.15)
    # Five slices sample the Z extent coarsely (10.5 px measured against 12).
    assert R[2, 2] == pytest.approx(R[1, 1], rel=0.3)
    assert np.allclose(R, R.T)
    # Unscaled Z (one slice = one pixel) would make the body 4 px deep, not 12.
    flat = MotionModel.from_config(Scale.from_values(PIXEL_UM, FRAME_MIN), TrackingConfig(), 3)
    assert flat.measurement_cov(det)[2, 2] < R[2, 2]


def test_positions_are_tracked_in_isotropic_pixels(scale3d):
    model = MotionModel.from_config(scale3d, TrackingConfig(), 3)
    assert model.anisotropy == pytest.approx(3.0)
    det = _blob(0, 10.0, 20.0, 4.0)
    assert list(model.position(det)) == pytest.approx([10.0, 20.0, 12.0])
    assert not model.z_assumed


def test_a_z_jump_is_judged_with_the_real_z_step():
    """40 slices in one 10 min frame: 60 um with the real 1.5 um step, 20 um if a slice were a pixel.

    The speed limit is 5 um/min * 10 min = 50 um, so only the calibrated
    reading refuses it -- and it is the right one.
    """
    calibrated = Scale.from_values(PIXEL_UM, FRAME_MIN, z_step_um=Z_STEP_UM)
    unknown_z = Scale.from_values(PIXEL_UM, FRAME_MIN)
    cfg = TrackingConfig()
    here, there = _blob(0, 50.0, 50.0, 2.0), _blob(1, 50.0, 50.0, 42.0)
    refused = pair_cost(start_track(here, calibrated, cfg), there, calibrated, cfg)
    assert refused.gated == GATE_SPEED
    assumed = pair_cost(start_track(here, unknown_z, cfg), there, unknown_z, cfg)
    assert assumed.gated != GATE_SPEED


def test_an_uncalibrated_z_step_is_flagged_on_every_track():
    scale = Scale.from_values(PIXEL_UM, FRAME_MIN)
    tracks, events = track_detections(moving_ellipsoids(4), 4, scale, TrackingConfig())
    assert tracks and all(FLAG_Z_UNCALIBRATED in t.flags for t in tracks)
    assert any("anisotropy 1.0" in n for n in events[0].notes)
    assert any("anisotropy 1.0" in n for n in tracks.notes)


def test_a_merge_in_3d_needs_the_prediction_inside_the_box_in_z(scale3d):
    """The same XY, different planes: a big object in the upper planes did not swallow a cell in the lower ones."""
    cfg = TrackingConfig()
    low = [_blob(t, 50.0, 50.0, 2.0, label=1) for t in range(2)]
    high = [_blob(t, 50.0, 50.0, 12.0, label=2, z_half=1) for t in range(2)]
    big = _blob(2, 50.0, 50.0, 12.0, label=2, z_half=1, volume=2.0 * high[0].volume_vox)
    tracks, events = track_detections(low + high + [big], 3, scale3d, cfg)
    assert not events[2].merge_suspected
    assert all(FLAG_MERGE not in t.flags for t in tracks)

    # Now the big object spans the low cell's plane too.
    big_deep = _blob(2, 50.0, 50.0, 7.0, label=2, z_half=6, volume=2.0 * high[0].volume_vox)
    tracks, events = track_detections(low + high + [big_deep], 3, scale3d, cfg)
    assert events[2].merge_suspected


def test_the_audit_reports_dz_in_isotropic_pixels(scale3d):
    cfg = TrackingConfig(max_gap=0, global_gap_closing=False)
    dets = [_blob(0, 50.0, 50.0, 3.0, label=1), _blob(2, 50.0, 50.0, 5.0, label=1)]
    tracks, _ = track_detections(dets, 3, scale3d, cfg)
    (audit,) = explain_unlinked_starts(tracks, scale3d, cfg)
    assert audit.dz_px == pytest.approx(6.0)
    assert audit.distance_px == pytest.approx(6.0)


def test_2d_and_3d_detections_cannot_be_mixed(scale3d):
    with pytest.raises(ValueError, match="2-D and 3-D"):
        track_detections(
            [make_detection(0, 10, 10), _blob(1, 10, 10, 3.0)], 2, scale3d, TrackingConfig()
        )


def _blob(frame, x, y, z, *, label=1, z_half=2, volume=None) -> Detection:
    """A hand-built 3-D detection: a 12 x 12 px footprint, ``2 z_half`` slices deep."""
    return Detection(
        frame=frame, label=label, x=float(x), y=float(y), z=float(z),
        area_px=113.0,
        bbox=(int(z - z_half), int(y - 6), int(x - 6), int(z + z_half) + 1, int(y + 6), int(x + 6)),
        extent_px=12, eccentricity=0.2, orientation_rad=0.0,
        major_axis_px=12.0, minor_axis_px=11.0, solidity=0.95, touches_border=False,
        volume_vox=float(volume if volume is not None else 113.0 * 2 * z_half),
    )
