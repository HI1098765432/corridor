"""Synthetic saved analyses for the interface tests: one v2 run, one v1 run.

The files are written by hand rather than by the pipeline so that a UI test
never needs Cellpose, and so every displayed number can be checked against a
value computed here from the same positions. Two tracks move in straight
lines; the v2 columns (Len, D2S, per-hour speeds, MSD) are derived exactly
as contract §6 defines them, so a test can assert the screen shows them
unchanged.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PIXEL_UM = 0.5
INTERVAL_MIN = 20.0
SHAPE_TYX = (6, 48, 64)
MODEL_SHA = "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"

#: track_id -> list of (frame, x_px, y_px). Track 2 skips frame 3, so it has a
#: gap; its frames 0, 1, 2, 4, 5 give the lag set {1, 2, 3, 4, 5} from actual
#: frame differences (not the {1, 2, 3, 4} its five observations would give
#: by index).
POSITIONS = {
    1: [(f, 20.0, 5.0 + 6.0 * f) for f in range(6)],
    2: [(f, 44.0 + 1.0 * f, 40.0 - 4.0 * f) for f in (0, 1, 2, 4, 5)],
}


#: The two lanes of the v2 run, as centre-line end points across the 64 x 48
#: image: one vertical at x = 20, one tilted from (44, 0) to (50, 48).
LANE_ENDS = {0: ((20.0, 0.0), (20.0, 48.0)), 1: ((44.0, 0.0), (50.0, 48.0))}


def channel_geometry_v2() -> dict[str, Any]:
    """``run.json["channel_geometry"]`` exactly as the pipeline writes it.

    Built with ``ChannelGeometry.to_dict`` -- origin plus unit direction per
    lane -- so the viewer is tested against the format a real run carries.
    """
    from corridor.core.geometry import GEOMETRY_FROM_RIDGES, ChannelGeometry, Lane

    lanes = []
    for index, ((x0, y0), (x1, y1)) in LANE_ENDS.items():
        norm = math.hypot(x1 - x0, y1 - y0)
        lanes.append(Lane(
            index=index, origin=(x0, y0), direction=((x1 - x0) / norm, (y1 - y0) / norm),
            half_width_px=8.0, support_rows=40,
        ))
    return ChannelGeometry(
        lanes=lanes, source=GEOMETRY_FROM_RIDGES, confidence=0.9, pitch_px=24.0,
        notes=["two lanes"], applied=True, image_shape=SHAPE_TYX[1:],
    ).to_dict()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _track_rows_v2(track_id: int, points) -> list[dict[str, Any]]:
    rows = []
    path_um = 0.0
    x0, y0 = points[0][1], points[0][2]
    previous = None
    for index, (frame, x, y) in enumerate(points):
        d2p = speed_min = None
        gap = 0
        if previous is not None:
            pf, px, py = previous
            gap = frame - pf
            d2p = math.hypot(x - px, y - py) * PIXEL_UM
            path_um += d2p
            speed_min = d2p / (gap * INTERVAL_MIN)
        rows.append(
            {
                "track_id": track_id,
                "frame": frame,
                "x_px": x,
                "y_px": y,
                "x_um": x * PIXEL_UM,
                "y_um": y * PIXEL_UM,
                "gap_frames": gap if previous is not None else "",
                "observation_index": index,
                "speed_um_per_min": speed_min,
                "speed_um_per_hr": speed_min * 60.0 if speed_min is not None else None,
                "cumulative_path_um": path_um,
                "distance_from_start_um": math.hypot(x - x0, y - y0) * PIXEL_UM,
                "distance_from_previous_um": d2p,
                "distance_from_reference_um": None,
                "link_margin_chi2": 4.0 if previous is not None else None,
            }
        )
        previous = (frame, x, y)
    return rows


def _msd_rows(track_id: int, points) -> list[dict[str, Any]]:
    """MSD over every pair, keyed by the actual frame difference (§6)."""
    sums: dict[int, list[float]] = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            lag = points[j][0] - points[i][0]
            d2 = (points[j][1] - points[i][1]) ** 2 + (points[j][2] - points[i][2]) ** 2
            sums.setdefault(lag, []).append(d2)
    rows = []
    for lag in sorted(sums):
        values = sums[lag]
        msd_px2 = sum(values) / len(values)
        rows.append(
            {
                "track_id": track_id,
                "lag_frames": lag,
                "lag_time_min": lag * INTERVAL_MIN,
                "lag_time_hr": lag * INTERVAL_MIN / 60.0,
                "n_pairs": len(values),
                "msd_um2": msd_px2 * PIXEL_UM**2,
                "msd_px2": msd_px2,
            }
        )
    return rows


def expected_len_um(track_id: int) -> float:
    points = POSITIONS[track_id]
    return sum(
        math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(points, points[1:])
    ) * PIXEL_UM


def expected_d2s_um(track_id: int) -> float:
    points = POSITIONS[track_id]
    return math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2]) * PIXEL_UM


def _summary_v2(track_id: int, rows) -> dict[str, Any]:
    speeds = [r["speed_um_per_min"] for r in rows if r["speed_um_per_min"] is not None]
    first, last = rows[0]["frame"], rows[-1]["frame"]
    missing = (last - first + 1) - len(rows)
    mean = sum(speeds) / len(speeds)
    return {
        "track_id": track_id,
        "channel": 0,
        "first_frame": first,
        "last_frame": last,
        "n_observations": len(rows),
        "n_gaps": 1 if missing else 0,
        "total_missing_frames": missing,
        "duration_min": (last - first) * INTERVAL_MIN,
        "net_displacement_um": expected_d2s_um(track_id),
        "path_length_um": expected_len_um(track_id),
        "straightness": expected_d2s_um(track_id) / expected_len_um(track_id),
        "mean_speed_um_per_min": mean,
        "mean_speed_um_per_hr": mean * 60.0,
        "max_speed_um_per_min": max(speeds),
        "max_speed_um_per_hr": max(speeds) * 60.0,
        # Straight-line motion at constant speed is ballistic: alpha = 2.
        "msd_alpha": 2.0 if track_id == 1 else None,
        "msd_alpha_r2": 1.0 if track_id == 1 else None,
        "flags": "",
    }


def write_v2_analysis(directory: Path, *, three_d: bool = False) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tracks, summaries, msd = [], [], []
    for track_id, points in POSITIONS.items():
        rows = _track_rows_v2(track_id, points)
        if three_d:
            for k, row in enumerate(rows):
                row["z"] = 2.0 + (k % 2)
        tracks.extend(rows)
        summaries.append(_summary_v2(track_id, rows))
        msd.extend(_msd_rows(track_id, points))
    _write_csv(directory / "tracks.csv", tracks)
    _write_csv(directory / "track_summary.csv", summaries)
    _write_csv(directory / "track_msd.csv", msd)
    _write_csv(
        directory / "qc_issues.csv",
        [
            {
                "severity": "warning", "code": "link_ambiguous", "title": "Ambiguous link",
                "detail": "margin 0.4", "frame": 2, "track_id": 2,
            }
        ],
    )
    shape = list(SHAPE_TYX) if not three_d else [SHAPE_TYX[0], 5, SHAPE_TYX[1], SHAPE_TYX[2]]
    manifest = {
        "schema_version": 2,
        "dimensionality": "3D" if three_d else "2D",
        "input": {
            "name": "synthetic.tif",
            "path": str(directory / "synthetic.tif"),
            "shape": shape,
            "axes": "TZYX" if three_d else "TYX",
            "axes_source": "imagej metadata",
        },
        "calibration": {
            "pixel_size_um": PIXEL_UM,
            "pixel_size_um_source": "nd2_info",
            "frame_interval_min": INTERVAL_MIN,
            "frame_interval_min_source": "nd2_info",
            **({"z_step_um": 1.5, "z_step_um_source": "imagej"} if three_d else {}),
        },
        "model": {
            "model_id": "jhu_confined_cp3_combi",
            "model_version": "1.0.0",
            "architecture": "cellpose3-cyto2-resnet",
            "sha256": MODEL_SHA,
            "cellpose_version": ">=3,<4",
            "training_dataset_version": "KK1KK2-combi",
            "developer_override": False,
        },
        "channel_geometry": channel_geometry_v2(),
        "segmentation": {
            "cellprob_threshold": 0.0,
            "flow_threshold": 0.4,
            "min_extent_px": 20,
            "ensemble": "thresholds",
            "ensemble_passes": 2,
            "detections_from_fallback": 1,
            "normalisation_mode": "whole_frame",
            "normalize_percentiles": [1.0, 99.0],
            "normalize_tile_px": 0,
            "normalize_sharpen_px": 0,
            "raw_instances_per_frame": [2] * 6,
            "kept_instances_per_frame": [2, 2, 2, 1, 2, 2],
        },
        "tracking": {"max_gap": 3, "max_delta_frames": 4, "max_speed_um_per_min": 5.0},
        "measurement": {"reference_point_px": None, "msd_min_pairs": 3},
        "environment": {"cellpose": "3.1.1.3"},
        "results": {
            "n_tracks": 2,
            "n_detections": len(tracks),
            "mean_speed_um_per_min": 0.1,
            "mean_speed_um_per_hr": 6.0,
            "elapsed_seconds": 12.5,
        },
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def write_v1_analysis(directory: Path) -> Path:
    """A 1.3.0-shaped run: no schema_version, an axis, along/across columns."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tracks, summaries = [], []
    for track_id, points in POSITIONS.items():
        previous = None
        for index, (frame, x, y) in enumerate(points):
            speed = None
            if previous is not None:
                speed = math.hypot(x - previous[1], y - previous[2]) * PIXEL_UM / (
                    (frame - previous[0]) * INTERVAL_MIN
                )
            tracks.append(
                {
                    "track_id": track_id, "frame": frame, "x_px": x, "y_px": y,
                    "x_um": x * PIXEL_UM, "y_um": y * PIXEL_UM,
                    "gap_frames": (frame - previous[0]) if previous else "",
                    "speed_um_per_min": speed, "v_along_um_per_min": speed,
                    "v_across_um_per_min": 0.0 if speed is not None else None,
                    "observation_index": index,
                }
            )
            previous = (frame, x, y)
        rows = [t for t in tracks if t["track_id"] == track_id]
        speeds = [r["speed_um_per_min"] for r in rows if r["speed_um_per_min"] is not None]
        summaries.append(
            {
                "track_id": track_id, "channel": 0,
                "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
                "n_observations": len(rows), "n_gaps": 0, "total_missing_frames": 0,
                "duration_min": (rows[-1]["frame"] - rows[0]["frame"]) * INTERVAL_MIN,
                "net_displacement_um": expected_d2s_um(track_id),
                "net_along_um": expected_d2s_um(track_id), "net_across_um": 0.0,
                "path_length_um": expected_len_um(track_id),
                "straightness": expected_d2s_um(track_id) / expected_len_um(track_id),
                "mean_speed_um_per_min": sum(speeds) / len(speeds),
                "max_speed_um_per_min": max(speeds),
                "flags": "",
            }
        )
    _write_csv(directory / "tracks.csv", tracks)
    _write_csv(directory / "track_summary.csv", summaries)
    manifest = {
        "input": {
            "name": "legacy.tif",
            "path": str(directory / "legacy.tif"),
            "shape_tyx": list(SHAPE_TYX),
            "axes_reported": "TYX",
            "axes_interpretation": "time, y, x",
        },
        "calibration": {
            "pixel_size_um": PIXEL_UM, "pixel_size_um_source": "nd2_info",
            "frame_interval_min": INTERVAL_MIN, "frame_interval_min_source": "nd2_info",
        },
        "segmentation": {
            "model_path": "C:/models/cyto2_phase_microfluidic_KK1KK2_combi",
            "model_sha256": MODEL_SHA,
            "cellprob_threshold": 0.0, "flow_threshold": 0.4, "min_extent_px": 20,
            "raw_instances_per_frame": [2] * 6, "kept_instances_per_frame": [2] * 6,
        },
        "tracking": {"max_gap": 3, "max_delta_frames": 4, "max_speed_um_per_min": 5.0},
        "confinement": {
            "ux": 0.0, "uy": 1.0, "tilt_from_vertical_deg": 0.0, "source": "walls",
            "n_channels": 1,
            "channels": [{"index": 0, "origin_x": 20.0, "origin_y": 0.0, "half_width_px": 10.0}],
        },
        "environment": {"cellpose": "3.1.1.3"},
        "results": {"n_tracks": 2, "n_detections": len(tracks), "mean_speed_um_per_min": 0.1},
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def load_saved(directory: Path):
    """Load through the store; attach MSD rows if this store predates them.

    The 2.0 store (work package C) reads ``track_msd.csv`` itself and exposes
    ``SavedAnalysis.msd``; until it is merged, the rows are attached here so
    the interface code under test is exercised the same way.
    """
    from corridor.store import project

    analysis = project.load_analysis(directory)
    if not hasattr(analysis, "msd"):
        analysis.msd = project.read_table(Path(directory) / "track_msd.csv")
    return analysis


def stack_2d() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(100.0, 5.0, size=SHAPE_TYX).astype(np.float32)


def stack_3d() -> np.ndarray:
    rng = np.random.default_rng(1)
    stack = rng.normal(100.0, 5.0, size=(SHAPE_TYX[0], 5, SHAPE_TYX[1], SHAPE_TYX[2]))
    return stack.astype(np.float32)
