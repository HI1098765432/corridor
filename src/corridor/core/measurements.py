"""Velocities, speeds and per-track summaries, in explicit physical units.

Column names always carry their units.  A column called ``speed`` whose
meaning depends on whether a calibration happened to be supplied is a trap;
``speed_um_per_min`` and ``speed_px_per_frame`` are two different numbers and
both are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import Scale
from .confinement import ConfinementAxis
from .tracking import Track


@dataclass
class TrackSummary:
    track_id: int
    channel: int
    first_frame: int
    last_frame: int
    first_source_frame: int | None
    last_source_frame: int | None
    n_observations: int
    n_gaps: int
    total_missing_frames: int
    span_frames: int
    duration_min: float | None
    net_displacement_um: float | None
    net_along_um: float | None
    net_across_um: float | None
    path_length_um: float | None
    straightness: float | None
    mean_speed_um_per_min: float | None
    median_speed_um_per_min: float | None
    max_speed_um_per_min: float | None
    mean_area_px: float
    flags: str

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def frame_rows(
    tracks: Sequence[Track],
    axis: ConfinementAxis,
    scale: Scale,
    *,
    source_frames: Sequence[int] | None = None,
    min_observations: int = 2,
) -> list[dict[str, Any]]:
    """One row per (track, observation), with gap-aware instantaneous velocity."""
    rows: list[dict[str, Any]] = []
    u, n = axis.unit, axis.normal

    for tr in sorted(tracks, key=lambda t: t.id):
        flags = set(tr.flags)
        if tr.n_obs < min_observations:
            flags.add("fragment")
        for k, obs in enumerate(tr.observations):
            row: dict[str, Any] = {
                "track_id": tr.id,
                "frame": obs.frame,
                "source_frame": (
                    int(source_frames[obs.frame])
                    if source_frames is not None and obs.frame < len(source_frames)
                    else None
                ),
                "elapsed_min": (
                    scale.frames_to_min(obs.frame) if scale.calibrated_time else None
                ),
                "x_px": obs.x,
                "y_px": obs.y,
                "area_px": obs.area_px,
                "det_label": obs.det_label,
                "channel": obs.channel,
                "gap_frames": obs.gap_frames,
                "match_cost_chi2": obs.cost,
                "observation_index": k,
                "n_observations": tr.n_obs,
                "track_flags": ";".join(sorted(flags)),
            }
            if scale.calibrated_space:
                row["x_um"] = scale.px_to_um(obs.x)
                row["y_um"] = scale.px_to_um(obs.y)
                row["area_um2"] = obs.area_px * scale.pixel_size_um**2

            if k == 0:
                for key in (
                    "vx_px_per_frame", "vy_px_per_frame", "speed_px_per_frame",
                    "vx_um_per_min", "vy_um_per_min", "speed_um_per_min",
                    "v_along_um_per_min", "v_across_um_per_min",
                    "step_px", "step_um",
                ):
                    row[key] = None
            else:
                prev = tr.observations[k - 1]
                dt_frames = obs.frame - prev.frame
                step = np.array([obs.x - prev.x, obs.y - prev.y], dtype=float)
                step_px = float(np.linalg.norm(step))

                vx_pf = step[0] / dt_frames
                vy_pf = step[1] / dt_frames
                row["vx_px_per_frame"] = vx_pf
                row["vy_px_per_frame"] = vy_pf
                row["speed_px_per_frame"] = math.hypot(vx_pf, vy_pf)
                row["step_px"] = step_px

                if scale.calibrated:
                    dt_min = scale.frames_to_min(dt_frames)
                    vx = scale.px_to_um(step[0]) / dt_min
                    vy = scale.px_to_um(step[1]) / dt_min
                    row["vx_um_per_min"] = vx
                    row["vy_um_per_min"] = vy
                    row["speed_um_per_min"] = math.hypot(vx, vy)
                    row["v_along_um_per_min"] = float(np.array([vx, vy]) @ u)
                    row["v_across_um_per_min"] = float(np.array([vx, vy]) @ n)
                    row["step_um"] = scale.px_to_um(step_px)
                else:
                    for key in (
                        "vx_um_per_min", "vy_um_per_min", "speed_um_per_min",
                        "v_along_um_per_min", "v_across_um_per_min", "step_um",
                    ):
                        row[key] = None
            rows.append(row)
    return rows


def summarise(
    tracks: Sequence[Track],
    axis: ConfinementAxis,
    scale: Scale,
    *,
    source_frames: Sequence[int] | None = None,
    min_observations: int = 2,
) -> list[TrackSummary]:
    out: list[TrackSummary] = []
    u, n = axis.unit, axis.normal

    for tr in sorted(tracks, key=lambda t: t.id):
        obs = tr.observations
        if not obs:
            continue
        flags = set(tr.flags)
        if len(obs) < min_observations:
            flags.add("fragment")

        gaps = [o.gap_frames for o in obs[1:] if o.gap_frames > 1]
        span = obs[-1].frame - obs[0].frame

        speeds: list[float] = []
        path_px = 0.0
        for a, b in zip(obs[:-1], obs[1:]):
            d = math.hypot(b.x - a.x, b.y - a.y)
            path_px += d
            dt = b.frame - a.frame
            if scale.calibrated and dt > 0:
                speeds.append(scale.px_to_um(d) / scale.frames_to_min(dt))

        net = np.array([obs[-1].x - obs[0].x, obs[-1].y - obs[0].y], dtype=float)
        net_px = float(np.linalg.norm(net))

        net_um = scale.px_to_um(net_px) if scale.calibrated_space else None
        path_um = scale.px_to_um(path_px) if scale.calibrated_space else None

        out.append(
            TrackSummary(
                track_id=tr.id,
                channel=tr.channel,
                first_frame=obs[0].frame,
                last_frame=obs[-1].frame,
                first_source_frame=(
                    int(source_frames[obs[0].frame])
                    if source_frames is not None and obs[0].frame < len(source_frames)
                    else None
                ),
                last_source_frame=(
                    int(source_frames[obs[-1].frame])
                    if source_frames is not None and obs[-1].frame < len(source_frames)
                    else None
                ),
                n_observations=len(obs),
                n_gaps=len(gaps),
                total_missing_frames=sum(g - 1 for g in gaps),
                span_frames=span,
                duration_min=scale.frames_to_min(span) if scale.calibrated_time else None,
                net_displacement_um=net_um,
                net_along_um=scale.px_to_um(float(net @ u)) if scale.calibrated_space else None,
                net_across_um=scale.px_to_um(float(net @ n)) if scale.calibrated_space else None,
                path_length_um=path_um,
                straightness=(net_px / path_px) if path_px > 1e-9 else None,
                mean_speed_um_per_min=float(np.mean(speeds)) if speeds else None,
                median_speed_um_per_min=float(np.median(speeds)) if speeds else None,
                max_speed_um_per_min=float(np.max(speeds)) if speeds else None,
                mean_area_px=float(np.mean([o.area_px for o in obs])),
                flags=";".join(sorted(flags)),
            )
        )
    return out
