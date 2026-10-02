"""Shared fixtures and synthetic-data helpers.

The tracking tests deliberately never touch Cellpose. Segmentation and
tracking fail in different ways, and a test that needs a neural network to
reproduce an assignment bug is not a useful test.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from corridor.core.confinement import Channel, ConfinementAxis
from corridor.core.config import Scale, TrackingConfig
from corridor.core.detections import Detection

# The calibration of the supplied datasets, used so that the synthetic tests
# exercise the same unit conversions as real runs.
PIXEL_SIZE_UM = 0.467060342995564
FRAME_INTERVAL_MIN = 20.006894938151042

REPO_ROOT = Path(__file__).resolve().parents[1]
#: ``CORRIDOR_SAMPLE_DIR`` points a checkout without ``data/`` (an agent's
#: isolated copy, a CI runner) at the supplied sample TIFFs, read-only.
SAMPLE_DIR = Path(
    os.environ.get("CORRIDOR_SAMPLE_DIR")
    or REPO_ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
)
#: ``CORRIDOR_BASELINE_DIR`` does the same for the frozen v1.3.0 runs, which
#: live under ``build/`` and are therefore absent from a fresh checkout.
BASELINE_DIR = Path(
    os.environ.get("CORRIDOR_BASELINE_DIR") or REPO_ROOT / "build" / "baseline_v1.3.0"
)


def has_samples() -> bool:
    return SAMPLE_DIR.is_dir() and any(SAMPLE_DIR.glob("*.tif"))


requires_samples = pytest.mark.skipif(
    not has_samples(), reason="supplied sample TIFFs are not present"
)


@pytest.fixture
def scale() -> Scale:
    return Scale.from_values(PIXEL_SIZE_UM, FRAME_INTERVAL_MIN)


@pytest.fixture
def vertical_axis() -> ConfinementAxis:
    """Legacy v1: a perfectly vertical single channel, as in the narrow crops.

    Only the modules that still take a ``ConfinementAxis`` (measurements,
    QC, recovery, the manifest) use this, until the integration package
    removes the axis from them. The tracker accepts it in the v1 argument
    position and reads only its channels.
    """
    return ConfinementAxis(
        ux=0.0, uy=1.0, source="configured", confidence=1.0,
        angle_sigma_rad=math.radians(0.5),
        channels=[Channel(index=0, origin=(45.0, 0.0), half_width_px=45.0)],
    )


@pytest.fixture
def tracking_config() -> TrackingConfig:
    return TrackingConfig()


def make_detection(
    frame: int,
    x: float,
    y: float,
    *,
    label: int = 1,
    area: float = 800.0,
    minor: float = 11.0,
    major: float = 90.0,
    eccentricity: float = 0.99,
    orientation_rad: float = 0.0,
    channel: int = 0,
    solidity: float = 0.95,
    touches_border: bool = False,
    with_mask: bool = False,
) -> Detection:
    """A detection shaped like the cells in the supplied training data.

    Those labelled cells are 9-15 px wide and 46-201 px long, so the defaults
    here are a real cell, not a generic blob. ``orientation_rad = 0`` is a
    body along the image rows (vertical), scikit-image's convention.

    ``with_mask`` attaches an elliptical ``mask_crop`` of the same size and
    orientation, so the tracker's overlap term has pixels to compare.
    """
    half_h, half_w = major / 2.0, minor / 2.0
    det = Detection(
        frame=frame,
        label=label,
        x=float(x),
        y=float(y),
        area_px=float(area),
        bbox=(
            int(y - half_h), int(x - half_w), int(y + half_h), int(x + half_w)
        ),
        extent_px=int(max(major, minor)),
        eccentricity=eccentricity,
        orientation_rad=orientation_rad,
        major_axis_px=major,
        minor_axis_px=minor,
        solidity=solidity,
        touches_border=touches_border,
        channel=channel,
    )
    if with_mask:
        attach_ellipse_mask(det)
    return det


def make_body(
    frame: int,
    x: float,
    y: float,
    heading_rad: float,
    *,
    label: int = 1,
    major: float = 60.0,
    minor: float = 8.0,
    with_mask: bool = True,
) -> Detection:
    """An elongated cell whose body lies along ``heading_rad`` (image x/y, y down).

    The open-field case the axis-free tracker exists for: a polarised cell
    migrates along its own long axis, so when it turns, its body turns with
    it. 60 x 8 px is the reviewer's scene (aspect 7.5, about the supplied
    cells' median 48.2/4.7 um body at half the length). Masks on by default,
    because production detections carry them.
    """
    # scikit-image orientation is the angle from the row axis, so the body's
    # unit vector (sin o, cos o) in (x, y) must equal (cos h, sin h).
    o = math.atan2(math.cos(heading_rad), math.sin(heading_rad))
    o = (o + math.pi / 2) % math.pi - math.pi / 2
    return make_detection(
        frame, x, y, label=label, major=major, minor=minor,
        area=math.pi * major * minor / 4.0, orientation_rad=o,
        eccentricity=math.sqrt(1.0 - (minor / major) ** 2), with_mask=with_mask,
    )


def attach_ellipse_mask(det: Detection) -> Detection:
    """Give ``det`` an elliptical mask crop matching its axes and orientation."""
    a, b = det.major_axis_px / 2.0, det.minor_axis_px / 2.0
    reach = int(math.ceil(max(a, b))) + 1
    min_r, min_c = int(math.floor(det.y)) - reach, int(math.floor(det.x)) - reach
    rows = np.arange(min_r, min_r + 2 * reach + 2)[:, None]
    cols = np.arange(min_c, min_c + 2 * reach + 2)[None, :]
    u = np.array([math.sin(det.orientation_rad), math.cos(det.orientation_rad)])
    dx, dy = cols - det.x, rows - det.y
    along = dx * u[0] + dy * u[1]
    across = -dx * u[1] + dy * u[0]
    mask = (along / a) ** 2 + (across / b) ** 2 <= 1.0
    r_idx, c_idx = np.nonzero(mask)
    r0, r1, c0, c1 = r_idx.min(), r_idx.max() + 1, c_idx.min(), c_idx.max() + 1
    det.mask_crop = mask[r0:r1, c0:c1]
    det.bbox = (min_r + int(r0), min_c + int(c0), min_r + int(r1), min_c + int(c1))
    det.area_px = float(det.mask_crop.sum())
    return det


def straight_track(
    n_frames: int,
    *,
    x: float = 45.0,
    y0: float = 20.0,
    step: float = 20.0,
    start_frame: int = 0,
    label: int = 1,
    area: float = 800.0,
    skip: set[int] | None = None,
) -> list[Detection]:
    """A cell moving down the channel at a constant speed."""
    skip = skip or set()
    out = []
    for k in range(n_frames):
        frame = start_frame + k
        if frame in skip:
            continue
        out.append(
            make_detection(frame, x, y0 + step * k, label=label, area=area)
        )
    return out


def group_by_frame(detections) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for d in detections:
        out.setdefault(int(d.frame), []).append(d)
    return out


def identity_of(tracks, frame: int, label: int):
    """The id of the track holding detection ``(frame, label)``, or None."""
    for tr in tracks:
        for obs in tr.observations:
            if obs.frame == frame and obs.det_label == label:
                return tr.id
    return None
