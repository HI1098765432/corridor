"""Shared fixtures and synthetic-data helpers.

The tracking tests deliberately never touch Cellpose. Segmentation and
tracking fail in different ways, and a test that needs a neural network to
reproduce an assignment bug is not a useful test.
"""

from __future__ import annotations

import math
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
SAMPLE_DIR = (
    REPO_ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
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
    """A perfectly vertical single channel, as in the narrow sample crops."""
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
) -> Detection:
    """A detection shaped like the cells in the supplied training data.

    Those labelled cells are 9-15 px wide and 46-201 px long, so the defaults
    here are a real cell, not a generic blob.
    """
    half_h, half_w = major / 2.0, minor / 2.0
    return Detection(
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
        solidity=0.95,
        touches_border=False,
        channel=channel,
    )


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
