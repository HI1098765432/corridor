"""Build the synthetic legacy-run fixtures in this folder.

Every number here is invented (the repository is public; no cell data is
copied).  What is real is the *shape* of each release's output: the file set,
the column headers of every CSV and the key structure of ``run.json``, read
from runs written by Corridor 1.0.0, 1.1.0, 1.2.0 and 1.3.0, and the v1
arithmetic that filled them (``measurements.py`` at the ``shipped-1.3.0``
tag), so the values are what that release would have written for these
positions.

1.2.0 is not one shape.  Its CSV headers equal 1.3.0's in every run on disk,
but its ``run.json`` came in three key sets as the 1.2.0 builds evolved:
``build/rec_*`` match 1.3.0 exactly; ``build/e2e_ensemble`` lacks
``segmentation/ensemble_passes_requested``, ``ensemble_models_unavailable``
and ``recovery/interior``/``trailing``; ``build/e2e_check`` also lacks every
normalisation and ensemble key and ``detections_from_fallback``.  The
``v1_2_0`` fixture takes the sparsest (``e2e_check``), because the 1.3.0
fixture already covers the full set.

1.0.0 wrote ``unlinked_starts.csv`` only when a track began mid-stack
(``data/_runs/052924_t3_dual`` has one, the other 1.0.0 runs none); the
fixture has one, and a test removes it to cover the other case.  Every real
run also holds ``masks_raw.npz`` (the masks before size filtering, same
shape and dtype); the fixtures carry one so the file set matches.

The synthetic movie: 6 frames of 60 x 40 px at 0.5 µm/px and 10 min/frame.

*   Track 1 moves straight down at 6 px/frame (0.3 µm/min = 18 µm/hr) and is
    missed at frame 3, so frames 0, 1, 2, 4, 5.
*   Track 2 wobbles at frames 1-3; from 1.1.0 its frame-3 position was found
    by the intensity recovery tier.
*   Track 3 is a one-observation fragment at frame 5.

Run from the repository root: ``python tests/fixtures/legacy/make_fixtures.py``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIXEL_UM = 0.5
INTERVAL_MIN = 10.0
N_FRAMES, HEIGHT, WIDTH = 6, 60, 40
SOURCE_FRAMES = [11, 12, 13, 14, 15, 16]
UX, UY = 0.0, 1.0  # the v1 axis: straight down the image

# -- headers, exactly as each release wrote them ------------------------------

TRACKS_1_0 = (
    "track_id,frame,source_frame,elapsed_min,x_px,y_px,x_um,y_um,area_px,area_um2,"
    "det_label,channel,gap_frames,match_cost_chi2,vx_px_per_frame,vy_px_per_frame,"
    "speed_px_per_frame,vx_um_per_min,vy_um_per_min,speed_um_per_min,v_along_um_per_min,"
    "v_across_um_per_min,step_px,step_um,observation_index,n_observations,track_flags"
).split(",")
TRACKS_1_1 = TRACKS_1_0 + ["detection_source", "detection_confidence"]
SUMMARY_1_0 = (
    "track_id,channel,first_frame,last_frame,first_source_frame,last_source_frame,"
    "n_observations,n_gaps,total_missing_frames,span_frames,duration_min,"
    "net_displacement_um,net_along_um,net_across_um,path_length_um,straightness,"
    "mean_speed_um_per_min,median_speed_um_per_min,max_speed_um_per_min,mean_area_px,flags"
).split(",")
SUMMARY_1_3 = SUMMARY_1_0[:19] + [
    "net_speed_um_per_min", "along_speed_um_per_min", "path_speed_um_per_min",
] + SUMMARY_1_0[19:]
DETECTIONS_1_0 = (
    "frame,source_frame,elapsed_min,label,channel,x,y,x_um,y_um,area_px,area_um2,"
    "extent_px,bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y,eccentricity,orientation_rad,"
    "major_axis_px,minor_axis_px,solidity,touches_border"
).split(",")
DETECTIONS_1_1 = DETECTIONS_1_0 + ["source", "confidence"]
DIAGNOSTICS = (
    "frame,raw_instances,kept_instances,removed_instances,removed_max_extent_px,"
    "removed_max_area_px,cellpose_message"
).split(",")
EVENTS = (
    "frame,detections,candidate_tracks,matched,new_tracks,dormant,terminated,"
    "merge_suspected_tracks,notes"
).split(",")
QC = "severity,code,title,detail,frame,track_id".split(",")
RECOVERY = (
    "track_id,frame,predicted_x,predicted_y,recovered,found_by,confidence,"
    "offset_from_prediction_px,detail"
).split(",")
UNLINKED = (
    "track_id,starts_at_frame,nearest_earlier_track,that_track_ended_at_frame,gap_frames,"
    "distance_px,along_channel_px,across_channel_px,implied_speed_um_per_min,"
    "would_have_cost_chi2,refused_because,explanation"
).split(",")

# -- the synthetic movie -------------------------------------------------------

#: (track, frame, x, y, area_px, label, match_cost, source, confidence)
OBSERVATIONS = [
    (1, 0, 20.0, 10.0, 300.0, 1, None, "primary", 1.0),
    (1, 1, 20.0, 16.0, 310.0, 1, 1.5, "primary", 1.0),
    (1, 2, 20.0, 22.0, 320.0, 1, 2.0, "primary", 1.0),
    (1, 4, 20.0, 34.0, 330.0, 1, 4.25, "primary", 1.0),
    (1, 5, 20.0, 40.0, 340.0, 1, 0.75, "primary", 1.0),
    (2, 1, 30.0, 50.0, 200.0, 2, None, "primary", 1.0),
    (2, 2, 31.0, 47.0, 210.0, 2, 3.0, "primary", 1.0),
    (2, 3, 30.0, 44.0, 190.0, 4, 6.5, "intensity", 0.4),
    (3, 5, 10.0, 55.0, 150.0, 3, None, "primary", 1.0),
]
#: Synthetic shape per label: (eccentricity, orientation_rad, major, minor, solidity)
SHAPES = {
    1: (0.98, 0.05, 30.0, 6.0, 0.93),
    2: (0.95, -0.10, 24.0, 7.0, 0.90),
    3: (0.90, 0.20, 18.0, 8.0, 0.88),
    4: (0.97, 0.00, 22.0, 6.5, 0.91),
}


def clean(value):
    """v1 ``export._clean``: empty for None, 9 decimals, true/false."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 9))
    return value


def write(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: clean(row.get(c)) for c in columns})


def tracks_by_id(version: str):
    out: dict[int, list[tuple]] = {}
    for o in OBSERVATIONS:
        if version == "1.0.0" and o[7] != "primary":
            # 1.0.0 had no recovery: that position was never found.
            continue
        out.setdefault(o[0], []).append(o)
    return out


def track_rows(version: str):
    rows = []
    for tid, obs in sorted(tracks_by_id(version).items()):
        flags = "fragment" if len(obs) < 2 else ""
        for k, (_, frame, x, y, area, label, cost, source, conf) in enumerate(obs):
            row = {
                "track_id": tid, "frame": frame, "source_frame": SOURCE_FRAMES[frame],
                "elapsed_min": frame * INTERVAL_MIN, "x_px": x, "y_px": y,
                "x_um": x * PIXEL_UM, "y_um": y * PIXEL_UM, "area_px": area,
                "area_um2": area * PIXEL_UM**2, "det_label": label, "channel": 0,
                "gap_frames": 0 if k == 0 else frame - obs[k - 1][1],
                "match_cost_chi2": cost, "observation_index": k,
                "n_observations": len(obs), "track_flags": flags,
                "detection_source": source, "detection_confidence": conf,
            }
            if k:
                prev = obs[k - 1]
                dt = frame - prev[1]
                dx, dy = x - prev[2], y - prev[3]
                vx, vy = dx * PIXEL_UM / (dt * INTERVAL_MIN), dy * PIXEL_UM / (dt * INTERVAL_MIN)
                row.update(
                    vx_px_per_frame=dx / dt, vy_px_per_frame=dy / dt,
                    speed_px_per_frame=math.hypot(dx / dt, dy / dt),
                    vx_um_per_min=vx, vy_um_per_min=vy, speed_um_per_min=math.hypot(vx, vy),
                    v_along_um_per_min=vx * UX + vy * UY,
                    v_across_um_per_min=vx * -UY + vy * UX,
                    step_px=math.hypot(dx, dy), step_um=math.hypot(dx, dy) * PIXEL_UM,
                )
            rows.append(row)
    return rows


def summary_rows(version: str):
    rows = []
    for tid, obs in sorted(tracks_by_id(version).items()):
        span = obs[-1][1] - obs[0][1]
        gaps = [b[1] - a[1] for a, b in zip(obs, obs[1:]) if b[1] - a[1] > 1]
        speeds, path_px = [], 0.0
        for a, b in zip(obs, obs[1:]):
            d = math.hypot(b[2] - a[2], b[3] - a[3])
            path_px += d
            speeds.append(d * PIXEL_UM / ((b[1] - a[1]) * INTERVAL_MIN))
        nx, ny = obs[-1][2] - obs[0][2], obs[-1][3] - obs[0][3]
        net_px = math.hypot(nx, ny)
        duration = span * INTERVAL_MIN
        along = (nx * UX + ny * UY) * PIXEL_UM
        row = {
            "track_id": tid, "channel": 0, "first_frame": obs[0][1], "last_frame": obs[-1][1],
            "first_source_frame": SOURCE_FRAMES[obs[0][1]],
            "last_source_frame": SOURCE_FRAMES[obs[-1][1]],
            "n_observations": len(obs), "n_gaps": len(gaps),
            "total_missing_frames": sum(g - 1 for g in gaps), "span_frames": span,
            "duration_min": duration, "net_displacement_um": net_px * PIXEL_UM,
            "net_along_um": along, "net_across_um": (nx * -UY + ny * UX) * PIXEL_UM,
            "path_length_um": path_px * PIXEL_UM,
            "straightness": net_px / path_px if path_px > 1e-9 else None,
            "mean_speed_um_per_min": float(np.mean(speeds)) if speeds else None,
            "median_speed_um_per_min": float(np.median(speeds)) if speeds else None,
            "max_speed_um_per_min": float(np.max(speeds)) if speeds else None,
            "net_speed_um_per_min": net_px * PIXEL_UM / duration if span else None,
            "along_speed_um_per_min": along / duration if span else None,
            "path_speed_um_per_min": path_px * PIXEL_UM / duration if span else None,
            "mean_area_px": float(np.mean([o[4] for o in obs])),
            "flags": "fragment" if len(obs) < 2 else "",
        }
        rows.append(row)
    return rows


def detection_rows(version: str):
    rows = []
    for o in sorted(OBSERVATIONS, key=lambda o: (o[1], o[5])):
        _, frame, x, y, area, label, _, source, conf = o
        if version == "1.0.0" and source != "primary":
            continue
        ecc, ori, major, minor, solidity = SHAPES[label]
        rows.append({
            "frame": frame, "source_frame": SOURCE_FRAMES[frame],
            "elapsed_min": frame * INTERVAL_MIN, "label": label, "channel": 0,
            "x": x, "y": y, "x_um": x * PIXEL_UM, "y_um": y * PIXEL_UM,
            "area_px": area, "area_um2": area * PIXEL_UM**2, "extent_px": int(major),
            "bbox_min_x": int(x - minor / 2), "bbox_min_y": int(y - major / 2),
            "bbox_max_x": int(x + minor / 2), "bbox_max_y": int(y + major / 2),
            "eccentricity": ecc, "orientation_rad": ori, "major_axis_px": major,
            "minor_axis_px": minor, "solidity": solidity, "touches_border": False,
            "source": source, "confidence": conf,
        })
    return rows


def masks(version: str) -> np.ndarray:
    out = np.zeros((N_FRAMES, HEIGHT, WIDTH), np.int32)
    for _, frame, x, y, _, label, _, source, _ in OBSERVATIONS:
        if source != "primary":
            continue  # masks.npz never held recovered objects
        out[frame, max(0, int(y) - 4):int(y) + 4, max(0, int(x) - 2):int(x) + 2] = label
    return out


def manifest(version: str) -> dict:
    n_by_frame = [sum(1 for o in OBSERVATIONS if o[1] == f and o[7] == "primary") for f in range(N_FRAMES)]
    segmentation = {
        "model_path": "C:/synthetic/models/synthetic_model",
        "model_sha256": "0" * 64,
        "use_custom_model": True,
        "diameter": None,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "channels": [0, 0],
        "normalize": True,
    }
    if version == "1.3.0":  # the 1.2.0 fixture is the e2e_check key set: none of these
        segmentation.update({
            "normalisation_mode": "whole_frame", "normalize_percentiles": [1.0, 99.0],
            "normalize_tile_px": 0, "normalize_sharpen_px": 0, "ensemble": "off",
            "ensemble_passes_requested": 1, "ensemble_passes": 1,
            "ensemble_model_paths": [], "ensemble_models_unavailable": [],
        })
    segmentation.update({
        "min_extent_px": 20, "min_area_px": 0, "drop_border_touching": False,
        "raw_instances_per_frame": n_by_frame, "kept_instances_per_frame": n_by_frame,
        "removed_instances_total": 0,
    })
    if version == "1.3.0":
        segmentation["detections_from_fallback"] = 0
    channel = {"index": 0, "origin_x": 20.0, "origin_y": 0.0, "half_width_px": 15.0}
    if version != "1.0.0":
        channel["detected"] = True
    data = {
        "application": {"name": "Corridor", "version": version},
        "environment": {
            "python": "3.12.0", "platform": "synthetic", "cellpose": "3.0.0",
            "numpy": "1.26.0", "gpu_available": False, "gpu_used": False,
        },
        "input": {
            "path": "C:/synthetic/movie.nd2", "name": "movie",
            "shape_tyx": [N_FRAMES, HEIGHT, WIDTH], "dtype": "uint16",
            "axes_reported": "TYX", "axes_interpretation": "time-first (T, Y, X)",
            "source_frames": SOURCE_FRAMES, "source_frame_total": 30,
            "acquisition": {
                k: "synthetic" for k in (
                    "SizeT", "SizeX", "SizeY", "SizeC", "SizeZ", "Time Loop", "dPeriod",
                    "dAvgPeriodDiff", "dMinPeriodDiff", "dMaxPeriodDiff", "dCalibration",
                    "sObjective", "dObjectiveNA", "dExposureTime", "ImageJ",
                )
            },
            "notes": ["Synthetic fixture.", "Original frame numbers 11-16 of 30."],
        },
        "calibration": {
            "pixel_size_um": PIXEL_UM, "pixel_size_um_source": "nd2_info",
            "frame_interval_min": INTERVAL_MIN, "frame_interval_min_source": "nd2_info",
            "spatially_calibrated": True, "temporally_calibrated": True,
        },
        "segmentation": segmentation,
        "confinement": {
            "ux": UX, "uy": UY, "angle_deg": 90.0, "tilt_from_vertical_deg": 0.0,
            "angle_sigma_deg": 0.5, "source": "channel_ridges", "confidence": 0.9,
            "n_channels": 1, "pitch_px": None, "channels": [channel],
            "notes": ["Synthetic channel."] + (["Synthetic note."] if version != "1.1.0" else []),
        },
        "tracking": {
            "max_speed_um_per_min": 5.0, "max_perp_um": 4.0, "area_ratio_min": 0.3,
            "area_ratio_max": 3.0, "sigma_along_um": 3.0, "speed_uncertainty_fraction": 0.7,
            "sigma_perp_um": 0.8, "perp_width_fraction": 0.35, "max_perp_widths": 1.6,
            "sigma_ln_area": 0.3, "w_reversal": 4.0, "w_orientation": 1.0,
            "orientation_min_eccentricity": 0.6, "direction_noise_floor_um": 1.0,
            "unmatched_chi2": 15.0, "gate_chi2": 30.0, "max_gap": 3, "min_observations": 2,
            "enforce_channel_identity": True, "max_delta_frames": 4,
        },
    }
    if version == "1.1.0":
        data["recovery"] = {
            "enabled": True, "window": True, "permissive": True, "intensity": True,
            "window_scale": 2.2, "window_min_px": 48, "permissive_cellprob": -2.0,
            "permissive_flow": 0.8, "intensity_snr": 4.0, "intensity_min_area_fraction": 0.3,
            "max_offset_lengths": 1.0, "attempted": 1, "recovered": 1,
            "by_tier": {"intensity": 1},
        }
    elif version in ("1.2.0", "1.3.0"):
        recovery = {
            "enabled": True, "interior": True, "trailing": False, "window": True,
            "permissive": True, "intensity": True, "window_scale": 2.2, "window_min_px": 48,
            "permissive_cellprob": -2.0, "permissive_flow": 0.8, "intensity_snr": 4.0,
            "intensity_min_area_fraction": 0.3, "intensity_edge_margin_px": 4,
            "intensity_min_eccentricity": 0.8, "intensity_max_channel_offset": 0.5,
            "max_offset_lengths": 1.0, "attempted": 1, "recovered": 1, "by_tier": {},
        }
        if version == "1.2.0":  # interior/trailing arrived within 1.2.0 (e2e_check lacks them)
            del recovery["interior"], recovery["trailing"]
            recovery["by_tier"] = {"intensity": 1}
        data["recovery"] = recovery
    tracks = tracks_by_id(version)
    data["results"] = {
        "n_detections": sum(n_by_frame), "n_tracks": len(tracks),
        "n_tracks_with_velocity": sum(1 for t in tracks.values() if len(t) > 1),
        "n_observations": sum(len(t) for t in tracks.values()),
        "mean_speed_um_per_min": 0.3, "output_dir": "C:/synthetic/out",
        "elapsed_seconds": 1.0,
    }
    return data


def build(version: str) -> None:
    out = HERE / ("v" + version.replace(".", "_"))
    later = version != "1.0.0"
    write(out / "tracks.csv", TRACKS_1_1 if later else TRACKS_1_0, track_rows(version))
    write(out / "track_summary.csv",
          SUMMARY_1_3 if version in ("1.2.0", "1.3.0") else SUMMARY_1_0,
          summary_rows(version))
    write(out / "detections.csv", DETECTIONS_1_1 if later else DETECTIONS_1_0,
          detection_rows(version))
    write(out / "segmentation_diagnostics.csv", DIAGNOSTICS, [
        {"frame": f, "raw_instances": n, "kept_instances": n, "removed_instances": 0,
         # The message column is free text.  "nan" is the text that the 1.x
         # reader turned into a float; it must come back as text.
         "cellpose_message": "nan" if f == 3 else ""}
        for f, n in enumerate(manifest(version)["segmentation"]["kept_instances_per_frame"])
    ])
    write(out / "tracking_events.csv", EVENTS, [
        {"frame": f, "detections": n, "candidate_tracks": 0, "matched": 0, "new_tracks": 0,
         "dormant": 0, "terminated": 0, "merge_suspected_tracks": "", "notes": ""}
        for f, n in enumerate(manifest(version)["segmentation"]["kept_instances_per_frame"])
    ])
    write(out / "qc_issues.csv", QC, [
        {"severity": "info", "code": "gap_bridged", "title": "Synthetic gap",
         "detail": "Synthetic: track 1 reacquired after 1 missing frame.", "frame": 4,
         "track_id": 1},
        {"severity": "info", "code": "fragment", "title": "Synthetic fragment",
         "detail": "Synthetic: track 3 has 1 observation.", "frame": 5, "track_id": 3},
    ])
    if later:
        write(out / "recovery_attempts.csv", RECOVERY, [
            {"track_id": 2, "frame": 3, "predicted_x": 30.5, "predicted_y": 44.0,
             "recovered": True, "found_by": "intensity", "confidence": 0.4,
             "offset_from_prediction_px": 0.5, "detail": "synthetic recovery"},
        ])
        # Track 3 starts at (10, 55) at frame 5; track 2 ended at (30, 44) at
        # frame 3.  Step (-20, 11): along u=(0,1) is 11, across n=(-1,0) is 20.
        write(out / "unlinked_starts.csv", UNLINKED, [
            {"track_id": 1, "starts_at_frame": 0, "explanation": "Synthetic: first frame."},
            {"track_id": 3, "starts_at_frame": 5, "nearest_earlier_track": 2,
             "that_track_ended_at_frame": 3, "gap_frames": 2,
             "distance_px": math.hypot(20.0, 11.0), "along_channel_px": 11.0,
             "across_channel_px": 20.0,
             "implied_speed_um_per_min": math.hypot(20.0, 11.0) * PIXEL_UM / 20.0,
             "would_have_cost_chi2": None, "refused_because": "lateral_jump",
             "explanation": "Synthetic: would have had to jump sideways."},
        ])
    else:
        # 1.0.0 has no recovered frame 3, so track 2 ends at (31, 47) at frame
        # 2.  Step to (10, 55): (-21, 8); along u=(0,1) is 8, across n=(-1,0) 21.
        write(out / "unlinked_starts.csv", UNLINKED, [
            {"track_id": 3, "starts_at_frame": 5, "nearest_earlier_track": 2,
             "that_track_ended_at_frame": 2, "gap_frames": 3,
             "distance_px": math.hypot(21.0, 8.0), "along_channel_px": 8.0,
             "across_channel_px": 21.0,
             "implied_speed_um_per_min": math.hypot(21.0, 8.0) * PIXEL_UM / 30.0,
             "would_have_cost_chi2": None, "refused_because": "lateral_jump",
             "explanation": "Synthetic: would have had to jump sideways."},
        ])
    (out / "run.json").write_text(json.dumps(manifest(version), indent=2), encoding="utf-8")
    np.savez_compressed(out / "masks.npz", masks=masks(version))
    # The unfiltered masks; nothing synthetic is filtered, so they are equal.
    np.savez_compressed(out / "masks_raw.npz", masks=masks(version))


def build_db_copies() -> None:
    """The database's ``config_json`` (``RunConfig.to_dict()`` at 1.3.0)."""
    config = {
        "input_path": "C:/synthetic/movie.nd2",
        "output_dir": "C:/synthetic/out",
        "segmentation": {
            "model_path": "C:/synthetic/models/synthetic_model", "builtin_model": "cyto3",
            "use_custom_model": True, "diameter": None, "cellprob_threshold": 0.0,
            "flow_threshold": 0.4, "channels": [0, 0], "normalize": True, "use_gpu": False,
            "normalize_percentiles": [1.0, 99.0], "normalize_tile_px": 0,
            "normalize_sharpen_px": 0, "min_extent_px": 20, "min_area_px": 0,
            "drop_border_touching": False, "ensemble": "models",
            "ensemble_model_paths": ["C:/synthetic/models/companion"],
            "ensemble_merge_overlap": 0.3, "ensemble_min_fragment_px": 20,
        },
        "tracking": {
            k: v for k, v in manifest("1.3.0")["tracking"].items() if k != "max_delta_frames"
        } | {"enforce_channel_identity": False},
        "confinement": {
            "mode": "angle", "angle_deg": 87.0, "detect_walls": False,
            "multichannel_warn_ratio": 1.6, "min_channel_pitch_um": 20.0,
            "min_channel_pitch_px": 40.0,
        },
        "calibration": {"pixel_size_um": PIXEL_UM, "frame_interval_min": INTERVAL_MIN},
        "recovery": {
            k: v for k, v in manifest("1.3.0")["recovery"].items()
            if k not in ("attempted", "recovered", "by_tier")
        },
    }
    (HERE / "db_config_json_v1_3_0.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for v in ("1.0.0", "1.1.0", "1.2.0", "1.3.0"):
        build(v)
    build_db_copies()
    print("fixtures written to", HERE)
