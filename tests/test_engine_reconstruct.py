"""The production overlap reconstructor is a drop-in for the Kalman tracker and
is identity-correct on the eye-verified real movie."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from corridor.core.config import Scale, TrackingConfig
from corridor.core.detections import Detection
from corridor.core.tracking import Track, TrackList
from corridor.engine.reconstruct import reconstruct_tracks

ROOT = Path(__file__).resolve().parents[1]


def _elongating_cell(lane_x, y0, vy, length_schedule, frames, label):
    """A vertical cell in a lane whose length changes each frame (so its
    centroid lurches -- the case overlap linking must survive)."""
    dets, masks = [], []
    for k, f in enumerate(frames):
        y = y0 + vy * k
        half = length_schedule[k] / 2
        dets.append((f, label, lane_x, y, half))
    return dets


def _render(cells, nframes, H=120, W=120):
    """Return (masks[T,H,W] int32, detections[list]). cells: list of per-cell
    lists of (frame, label, x, y, half_len)."""
    masks = np.zeros((nframes, H, W), np.int32)
    dets = []
    for cell in cells:
        for (f, label, x, y, half) in cell:
            y0, y1 = int(max(0, y - half)), int(min(H, y + half))
            x0, x1 = int(x - 3), int(x + 3)
            masks[f, y0:y1, x0:x1] = label
            area = float((y1 - y0) * (x1 - x0))
            dets.append(Detection(frame=f, label=label, x=float(x), y=float(y),
                        area_px=area, bbox=(y0, x0, y1, x1), extent_px=int(y1 - y0),
                        eccentricity=0.95, orientation_rad=0.0,
                        major_axis_px=float(y1 - y0), minor_axis_px=6.0,
                        solidity=1.0, touches_border=False, channel=0))
    return masks, dets


def test_returns_tracklist_and_events():
    cell = _elongating_cell(30, 20, 6, [10, 40, 70, 30, 20], [0, 1, 2, 3, 4], 1)
    masks, dets = _render([cell], 5)
    tracks, events = reconstruct_tracks(dets, masks.shape[1:], 5, Scale.from_values(0.5, 10.0), TrackingConfig())
    assert isinstance(tracks, TrackList)
    assert all(isinstance(t, Track) for t in tracks)
    assert len(events) == 5


def test_one_elongating_cell_is_one_track_not_split_by_centroid_jumps():
    # length swings 10->70->20 px: the centroid lurches, but overlap holds it.
    cell = _elongating_cell(30, 30, 2, [10, 40, 70, 50, 20, 15], range(6), 1)
    masks, dets = _render([cell], 6)
    tracks, _ = reconstruct_tracks(dets, masks.shape[1:], 6, Scale.from_values(0.5, 10.0), TrackingConfig())
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 6


def test_two_cells_in_two_lanes_keep_separate_identities():
    a = _elongating_cell(25, 20, 5, [30] * 6, range(6), 1)
    b = _elongating_cell(90, 80, -4, [30] * 6, range(6), 2)
    masks, dets = _render([a, b], 6)
    tracks, _ = reconstruct_tracks(dets, masks.shape[1:], 6, Scale.from_values(0.5, 10.0), TrackingConfig())
    assert len(tracks) == 2


@pytest.mark.skipif(
    not (ROOT / "build/baseline_v1.3.0/052924_t1/masks.npz").exists(),
    reason="real sample masks not present",
)
def test_real_t1_is_one_track_over_its_verified_lifetime():
    movie = ROOT / "build/baseline_v1.3.0/052924_t1"
    masks = np.load(movie / "masks.npz")["masks"]
    dets = []
    with open(movie / "detections.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("source", "primary") != "primary":
                continue
            dets.append(Detection(
                frame=int(r["frame"]), label=int(r["label"]), x=float(r["x"]), y=float(r["y"]),
                area_px=float(r["area_px"]),
                bbox=(int(float(r["bbox_min_y"])), int(float(r["bbox_min_x"])),
                      int(float(r["bbox_max_y"])), int(float(r["bbox_max_x"]))),
                extent_px=int(float(r["extent_px"])), eccentricity=float(r["eccentricity"]),
                orientation_rad=float(r["orientation_rad"]), major_axis_px=float(r["major_axis_px"]),
                minor_axis_px=float(r["minor_axis_px"]), solidity=float(r["solidity"]),
                touches_border=(r["touches_border"] == "true"), channel=int(float(r["channel"]))))
    tracks, _ = reconstruct_tracks(dets, masks.shape[1:], masks.shape[0],
                                   Scale.from_values(0.467060342995564, 20.006894938151042),
                                   TrackingConfig())
    assert len(tracks) == 1
    frames = [o.frame for o in tracks[0].observations]
    assert frames[0] == 4 and frames[-1] == 16 and len(frames) == 13
