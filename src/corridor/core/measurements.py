"""Velocities, distances, MSD and per-track summaries, in explicit physical units.

Column names always carry their units.  A column called ``speed`` whose
meaning depends on whether a calibration happened to be supplied is a trap;
``speed_um_per_min`` and ``speed_px_per_frame`` are two different numbers and
both are reported.  A µm column is empty, never filled with pixels, when the
data are uncalibrated, and the ``_px`` columns stay so uncalibrated data still
has every quantity it can honestly carry.

**One canonical velocity.**  Each step's physical velocity is
``v = Δr_um / Δt_min``, where ``Δt`` is the elapsed time between the two
observations (so a step across missed frames is divided by the time it really
took).  Every per-hour value is that number ``× 60`` and is never recomputed
from geometry, so the µm/min and µm/hr columns cannot disagree.

**MTrackJ conventions** (per observation, ``tracks.csv``):

*   ``cumulative_path_um`` (Len) and ``distance_from_start_um`` (D2S) are 0 at
    the first point;
*   ``distance_from_previous_um`` (D2P) and every speed are empty at the first
    point;
*   ``distance_from_reference_um`` (D2R) is filled only when a reference point
    was set.

**Dimensionality.**  Positions are measured in isotropic pixel units:
``(x, y)`` in 2-D and ``(x, y, z * anisotropy)`` in 3-D, where ``z`` is in
slices.  Without a known Z step a 3-D distance has no meaning in any unit (a
slice is not a pixel), so every 3-D distance, speed and MSD stays empty rather
than being computed in the XY plane and reported as if it were 3-D.

Nothing here receives a migration direction.  The v1 along/across columns are
gone from the schema (``docs/NEXT_GENERATION.md`` §7).

The functions read tracks by duck typing -- ``id``, ``observations``,
``flags``, ``channel`` on a track; ``frame``, ``x``, ``y`` and the rest on an
observation, with ``z``, ``link_margin`` and ``detection`` read through
``getattr`` -- so the same code measures tracker output, tracks rebuilt from a
saved ``tracks.csv`` and hand-built test tracks.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .config import MeasurementConfig, Scale

if TYPE_CHECKING:  # pragma: no cover - typing only; measurement needs no tracker
    from .tracking import Track

#: Minutes per hour.  Named so the one place a per-hour value is derived is
#: greppable.
MIN_PER_HR = 60.0


# --------------------------------------------------------------------------
# Per-track summary
# --------------------------------------------------------------------------


@dataclass
class TrackSummary:
    """One row of ``track_summary.csv`` (schema v2).

    Speeds appear in both µm/min (continuity with v1) and µm/hr (the unit the
    lab reports); the hourly value is always the per-minute one ``× 60``.  The
    ``_px`` fields carry the same geometry for uncalibrated data.
    """

    track_id: int
    channel: int
    #: "2D" or "3D".
    dimensionality: str
    first_frame: int
    last_frame: int
    first_source_frame: int | None
    last_source_frame: int | None
    n_observations: int
    n_gaps: int
    total_missing_frames: int
    span_frames: int
    duration_min: float | None
    duration_hr: float | None
    net_displacement_px: float | None
    net_displacement_um: float | None
    path_length_px: float | None
    path_length_um: float | None
    #: The largest D2S along the track: how far the cell ever got from where
    #: it started, which a cell that went out and came back hides from the net
    #: displacement.
    max_distance_from_start_um: float | None
    #: Net over path length.  1 for a straight run, 0 for a return trip.
    straightness: float | None
    mean_speed_um_per_min: float | None
    median_speed_um_per_min: float | None
    max_speed_um_per_min: float | None
    mean_speed_um_per_hr: float | None
    median_speed_um_per_hr: float | None
    max_speed_um_per_hr: float | None
    #: Net displacement divided by elapsed time, and the *robust* estimator on
    #: this data.  With a fifth of the detections deleted at random it is
    #: exactly right in half of trials and within 4.6% at the 90th percentile,
    #: where the mean of instantaneous speeds is already 7.2% out at the median
    #: and 19.3% at the 90th (scripts/experiment_velocity_robustness.py).
    #:
    #: The reason is structural rather than lucky: this reads only the first and
    #: last observation, so a missed frame in between costs it nothing, whereas
    #: every missed frame merges two short intervals into one long one that the
    #: instantaneous estimate then averages in with equal weight.
    #:
    #: It measures net progress, not path speed; the two differ by exactly how
    #: much the cell reversed, which ``straightness`` reports.
    net_speed_um_per_min: float | None
    net_speed_um_per_hr: float | None
    #: Total path length over elapsed time.  Unlike the mean of instantaneous
    #: speeds this is not inflated by short noisy intervals, but it still grows
    #: when detections are dense and jittery, so it is reported beside the net
    #: figure rather than instead of it.
    path_speed_um_per_min: float | None
    path_speed_um_per_hr: float | None
    #: The pixel-unit counterparts, for data without a calibration.
    mean_speed_px_per_frame: float | None
    net_speed_px_per_frame: float | None
    #: Mean of the per-observation ``turning_angle_deg`` (0 straight on, 180 a
    #: full reversal).
    mean_turning_angle_deg: float | None
    #: Mean cosine between the directions of two one-frame steps one frame
    #: apart (see :func:`directional_autocorrelation`).  1 is a cell that keeps
    #: its heading, 0 a heading forgotten within a frame.
    directional_autocorrelation: float | None
    #: Decay time of the directional autocorrelation, ``C(t) = exp(-t / P)``,
    #: by log-linear least squares.  Reported only with its fit quality
    #: (``persistence_fit_r2``, ``persistence_fit_lags``); empty when too few
    #: lags qualify or when no decay was observed.  It is the persistence
    #: *time of the fitted model*: a high r² is not proof the cell performs a
    #: persistent random walk.
    persistence_time_min: float | None
    persistence_time_hr: float | None
    persistence_fit_r2: float | None
    persistence_fit_lags: int | None
    #: MSD exponent from a log-log fit (``msd_fit``); empty when the track has
    #: too few qualifying lags.  ~1 diffusive, ~2 ballistic, <1 confined.
    msd_alpha: float | None
    msd_alpha_r2: float | None
    msd_fit_lags: int | None
    mean_area_px: float | None
    mean_area_um2: float | None
    #: 3-D only, and only with a calibrated Z step.
    mean_volume_um3: float | None
    #: The weakest accepted link in the track.  Small means at least one
    #: assignment had a nearly-as-good alternative.  A margin in chi-square
    #: units, not a probability.
    min_link_margin_chi2: float | None
    flags: str

    def to_row(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # -- transition only ----------------------------------------------------
    # v1 quality control (``qc.lateral_drift``) and research scripts still read
    # the axis fields.  There is no axis in 2.0, so they read None and the v1
    # rule that compares them never fires.  These are properties, not fields:
    # they are not in ``to_row()`` and never reach a CSV.  Remove them when the
    # integration package has rewritten qc.py.
    @property
    def net_along_um(self) -> None:
        return None

    @property
    def net_across_um(self) -> None:
        return None

    @property
    def along_speed_um_per_min(self) -> None:
        return None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _resolve_scale(scale: Any, legacy: Sequence[Any]) -> Scale:
    """Accept ``(scale)`` and, during the transition, ``(axis, scale)``.

    The 1.x signatures took a migration axis before the scale.  Until the
    integration package rewrites pipeline.py and qc.py, callers may still pass
    one; it is ignored.  Anything else is a programming error, raised rather
    than guessed at.
    """
    if legacy:
        if len(legacy) != 1 or not isinstance(legacy[0], Scale):
            raise TypeError("expected (tracks, scale) or the legacy (tracks, axis, scale)")
        return legacy[0]
    if not isinstance(scale, Scale):
        raise TypeError(f"expected a Scale, got {type(scale).__name__}")
    return scale


def _obs_z(obs: Any) -> float | None:
    z = getattr(obs, "z", None)
    if z is None:
        det = getattr(obs, "detection", None)
        z = getattr(det, "z", None) if det is not None else None
    return None if z is None else float(z)


def _source_frame(source_frames: Sequence[int] | None, frame: int) -> int | None:
    if source_frames is not None and 0 <= frame < len(source_frames):
        value = source_frames[frame]
        return None if value is None else int(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _per_hr(per_min: float | None) -> float | None:
    return None if per_min is None else per_min * MIN_PER_HR


def _norm(v: np.ndarray) -> float:
    return float(math.sqrt(float(np.dot(v, v))))


@dataclass
class _TrackGeometry:
    """One track's positions, in every unit they can honestly be expressed in."""

    frames: np.ndarray  # (n,) int
    xy_px: np.ndarray  # (n, 2) raw x, y in pixels; always known
    z_slice: np.ndarray | None  # (n,) slices, 3-D only
    #: Isotropic pixel positions ((x, y) or (x, y, z * anisotropy)); None for a
    #: 3-D track whose anisotropy is unknown.
    px: np.ndarray | None
    #: Physical positions; None without the calibration they need.
    um: np.ndarray | None
    three_d: bool


def _geometry(observations: Sequence[Any], scale: Scale) -> _TrackGeometry:
    frames = np.array([int(o.frame) for o in observations], dtype=int)
    xy = np.array([[float(o.x), float(o.y)] for o in observations], dtype=float).reshape(-1, 2)
    zs = [_obs_z(o) for o in observations]
    three_d = any(z is not None for z in zs)
    if not three_d:
        px = xy
        um = xy * scale.pixel_size_um if scale.calibrated_space else None
        return _TrackGeometry(frames, xy, None, px, um, False)

    # A track either has Z everywhere or it is malformed; a missing Z becomes
    # NaN so every quantity that touches it is empty rather than planar.
    z = np.array([np.nan if v is None else v for v in zs], dtype=float)
    aniso = scale.anisotropy
    px = np.column_stack([xy, z * aniso]) if aniso is not None else None
    um = (
        np.column_stack([xy * scale.pixel_size_um, z * float(scale.z_step_um)])
        if (scale.calibrated_space and scale.calibrated_z and scale.z_step_um)
        else None
    )
    return _TrackGeometry(frames, xy, z, px, um, True)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# Morphology carried from the detection
# --------------------------------------------------------------------------

#: Copied unchanged from the detection: unit-free ratios, angles, pixel and
#: voxel counts.  Physical columns are derived below, only with calibration.
_MORPHOLOGY_PASSTHROUGH = (
    "perimeter_px", "major_axis_px", "minor_axis_px", "aspect_ratio",
    "eccentricity", "orientation_rad", "circularity", "solidity",
    "extent_fraction", "mean_intensity", "volume_vox",
    "sphericity", "elongation", "flatness",
)


def _morphology(obs: Any, scale: Scale, row: dict[str, Any]) -> None:
    """Fill the morphology columns from ``obs.detection``, when there is one.

    Only the detection is read.  The v1 ``Observation`` also carries
    ``eccentricity``/``minor_axis_px`` with placeholder defaults (0.0, 1.0)
    for hand-built observations; reporting those would print a placeholder
    as a measurement.
    """
    det = getattr(obs, "detection", None)
    for key in _MORPHOLOGY_PASSTHROUGH:
        row[key] = _finite(getattr(det, key, None)) if det is not None else None
    row["perimeter_um"] = row["major_axis_um"] = row["minor_axis_um"] = None
    row["volume_um3"] = row["surface_area_um2"] = None
    if det is None:
        return
    p = scale.pixel_size_um
    if scale.calibrated_space:
        for name in ("perimeter", "major_axis", "minor_axis"):
            value = row[f"{name}_px"]
            row[f"{name}_um"] = value * p if value is not None else None
    # Volume and surface need a real Z step as well: a slice is not a pixel.
    if scale.calibrated_space and scale.calibrated_z and scale.z_step_um:
        volume = _finite(getattr(det, "volume_um3", None))
        if volume is None and row["volume_vox"] is not None:
            volume = row["volume_vox"] * p * p * float(scale.z_step_um)
        row["volume_um3"] = volume
        row["surface_area_um2"] = _finite(getattr(det, "surface_area_um2", None))


# --------------------------------------------------------------------------
# Reference distance (MTrackJ D2R)
# --------------------------------------------------------------------------


def add_reference_distance(
    rows: Iterable[dict[str, Any]],
    reference_point_px: Sequence[float] | None,
    pixel_size_um: float | None,
    *,
    z_step_um: float | None = None,
) -> list[dict[str, Any]]:
    """Return copies of ``rows`` with ``distance_from_reference_px/_um`` set.

    The reference is in the same frame as ``Detection.position``: ``(x, y)``
    pixels, or ``(x, y, z)`` with z in slices.  A 2-D reference is used only
    for 2-D rows and a 3-D one only for 3-D rows: measuring a 3-D cell from a
    point with no Z would silently report an in-plane distance under a 3-D
    name.  A 3-D distance needs both the pixel size and the Z step (in any
    unit, a slice and a pixel are only comparable through them).

    ``reference_point_px=None`` clears both columns, which is what MTrackJ
    shows when no reference was set.  The px distance is kept even when
    ``pixel_size_um`` is unknown.
    """
    ok_p = bool(pixel_size_um and math.isfinite(pixel_size_um) and pixel_size_um > 0)
    ok_z = bool(z_step_um and math.isfinite(z_step_um) and z_step_um > 0)
    ref = None if reference_point_px is None else tuple(float(v) for v in reference_point_px)
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["distance_from_reference_px"] = None
        row["distance_from_reference_um"] = None
        x, y = row.get("x_px"), row.get("y_px")
        z = row.get("z_slice")
        if ref is None or x is None or y is None:
            out.append(row)
            continue
        dx, dy = float(x) - ref[0], float(y) - ref[1]
        if z is None and len(ref) == 2:
            d_px = math.hypot(dx, dy)
            row["distance_from_reference_px"] = d_px
            row["distance_from_reference_um"] = d_px * pixel_size_um if ok_p else None
        elif z is not None and len(ref) == 3 and ok_p and ok_z:
            dz_um = (float(z) - ref[2]) * z_step_um
            d_um = math.sqrt((dx * pixel_size_um) ** 2 + (dy * pixel_size_um) ** 2 + dz_um**2)
            row["distance_from_reference_um"] = d_um
            row["distance_from_reference_px"] = d_um / pixel_size_um
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Per-observation rows
# --------------------------------------------------------------------------


def frame_rows(
    tracks: Sequence["Track"],
    scale: Scale,
    *legacy: Any,
    source_frames: Sequence[int] | None = None,
    min_observations: int = 2,
    reference_point_px: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """One row per (track, observation), with gap-aware kinematics.

    Accepts the legacy ``frame_rows(tracks, axis, scale)`` during the
    transition; the axis is ignored.

    Definitions, for observation ``k`` at elapsed time ``t_k``:

    *   ``speed_um_per_min = |r_k - r_{k-1}|_um / (t_k - t_{k-1})``, empty at
        ``k = 0``; ``speed_um_per_hr`` is that ``× 60``.
    *   ``acceleration_um_per_hr2`` is the change of speed between the two
        steps that meet at ``k - 1``, divided by the time between the steps'
        elapsed-time midpoints, ``((t_k + t_{k-1}) - (t_{k-1} + t_{k-2})) / 2
        = (t_k - t_{k-2}) / 2``.  It is the signed rate of change of *speed*
        (positive = speeding up), empty for ``k < 2``.  Change of direction
        is reported separately by ``turning_angle_deg``, so it is not folded
        in here as well.  Dividing by midpoint spacing rather than by one
        frame keeps it correct across missed frames.
    *   ``turning_angle_deg`` is the unsigned angle, 0-180 degrees, between
        step ``k-1 -> k`` and the step before it, so it is filled from
        ``k = 2``.  It is empty when either step has zero length (no
        direction to compare).  Unsigned so 2-D and 3-D share one definition;
        a direction-free quantity needs no axis.
    """
    scale = _resolve_scale(scale, legacy)
    rows: list[dict[str, Any]] = []
    p = scale.pixel_size_um

    for tr in sorted(tracks, key=lambda t: t.id):
        obs_list = sorted(tr.observations, key=lambda o: o.frame)
        if not obs_list:
            continue
        n = len(obs_list)
        flags = set(getattr(tr, "flags", None) or ())
        if n < min_observations:
            flags.add("fragment")
        flag_text = ";".join(sorted(flags))
        g = _geometry(obs_list, scale)

        cumulative_px = 0.0
        cumulative_um = 0.0
        prev_speed_hr: float | None = None
        for k, obs in enumerate(obs_list):
            det = getattr(obs, "detection", None)
            frame = int(obs.frame)
            elapsed_min = scale.frames_to_min(frame) if scale.calibrated_time else None
            area_px = getattr(obs, "area_px", None)
            if area_px is None and det is not None:
                area_px = getattr(det, "area_px", None)
            source = getattr(obs, "source", None) or getattr(det, "source", None)
            confidence = getattr(obs, "confidence", None)
            if confidence is None and det is not None:
                confidence = getattr(det, "confidence", None)
            det_label = getattr(obs, "det_label", None)
            if det_label is None and det is not None:
                det_label = getattr(det, "label", None)
            channel = getattr(obs, "channel", None)
            if channel is None:
                channel = getattr(tr, "channel", -1)

            row: dict[str, Any] = {
                "track_id": tr.id,
                "frame": frame,
                "source_frame": _source_frame(source_frames, frame),
                "elapsed_min": elapsed_min,
                "elapsed_hr": None if elapsed_min is None else elapsed_min / MIN_PER_HR,
                "x_px": float(obs.x),
                "y_px": float(obs.y),
                "z_slice": None if g.z_slice is None else _finite(g.z_slice[k]),
                "x_um": float(obs.x) * p if scale.calibrated_space else None,
                "y_um": float(obs.y) * p if scale.calibrated_space else None,
                "z_um": (
                    _finite(g.um[k, 2]) if (g.three_d and g.um is not None) else None
                ),
                "det_label": det_label,
                "channel": channel,
                "observation_index": k,
                "n_observations": n,
                "gap_frames": 0 if k == 0 else frame - int(obs_list[k - 1].frame),
                "match_cost_chi2": _finite(getattr(obs, "cost", None)),
                "link_margin_chi2": _finite(getattr(obs, "link_margin", None)),
                "area_px": None if area_px is None else float(area_px),
                "area_um2": (
                    float(area_px) * p * p
                    if (area_px is not None and scale.calibrated_space)
                    else None
                ),
                "track_flags": flag_text,
                "detection_source": source,
                "segmentation_confidence": confidence,
            }
            _morphology(obs, scale, row)

            # -- distances (MTrackJ: Len and D2S are 0 at the first point) -----
            row["distance_from_start_px"] = (
                _finite(_norm(g.px[k] - g.px[0])) if g.px is not None else None
            )
            row["distance_from_start_um"] = (
                _finite(_norm(g.um[k] - g.um[0])) if g.um is not None else None
            )
            kinematics = dict.fromkeys(
                (
                    "distance_from_previous_px", "distance_from_previous_um",
                    "vx_px_per_frame", "vy_px_per_frame", "speed_px_per_frame",
                    "vx_um_per_min", "vy_um_per_min", "vz_um_per_min", "speed_um_per_min",
                    "vx_um_per_hr", "vy_um_per_hr", "vz_um_per_hr", "speed_um_per_hr",
                    "acceleration_um_per_hr2", "turning_angle_deg",
                )
            )
            speed_hr: float | None = None
            if k > 0:
                dt_frames = frame - int(obs_list[k - 1].frame)
                if dt_frames > 0:
                    dxy = g.xy_px[k] - g.xy_px[k - 1]
                    kinematics["vx_px_per_frame"] = float(dxy[0]) / dt_frames
                    kinematics["vy_px_per_frame"] = float(dxy[1]) / dt_frames
                    if g.px is not None:
                        d_px = _finite(_norm(g.px[k] - g.px[k - 1]))
                        kinematics["distance_from_previous_px"] = d_px
                        if d_px is not None:
                            cumulative_px += d_px
                            kinematics["speed_px_per_frame"] = d_px / dt_frames
                    if g.um is not None:
                        step_um = g.um[k] - g.um[k - 1]
                        d_um = _finite(_norm(step_um))
                        kinematics["distance_from_previous_um"] = d_um
                        if d_um is not None:
                            cumulative_um += d_um
                        if scale.calibrated_time and d_um is not None:
                            dt_min = scale.frames_to_min(dt_frames)
                            v = step_um / dt_min
                            kinematics["vx_um_per_min"] = float(v[0])
                            kinematics["vy_um_per_min"] = float(v[1])
                            if g.three_d:
                                kinematics["vz_um_per_min"] = float(v[2])
                            speed_min = d_um / dt_min
                            kinematics["speed_um_per_min"] = speed_min
                            for axis_name in ("vx", "vy", "vz"):
                                kinematics[f"{axis_name}_um_per_hr"] = _per_hr(
                                    kinematics[f"{axis_name}_um_per_min"]
                                )
                            speed_hr = speed_min * MIN_PER_HR
                            kinematics["speed_um_per_hr"] = speed_hr

            if k >= 2:
                f0, f2 = int(obs_list[k - 2].frame), frame
                if prev_speed_hr is not None and speed_hr is not None:
                    midpoint_gap_hr = scale.frames_to_hr(f2 - f0) / 2.0
                    if midpoint_gap_hr > 0:
                        kinematics["acceleration_um_per_hr2"] = (
                            speed_hr - prev_speed_hr
                        ) / midpoint_gap_hr
                if g.px is not None:
                    kinematics["turning_angle_deg"] = _turning_angle_deg(
                        g.px[k - 1] - g.px[k - 2], g.px[k] - g.px[k - 1]
                    )
            prev_speed_hr = speed_hr

            row.update(kinematics)
            row["cumulative_path_px"] = cumulative_px if g.px is not None else None
            row["cumulative_path_um"] = cumulative_um if g.um is not None else None
            rows.append(row)

    if reference_point_px is not None:
        rows = add_reference_distance(
            rows,
            reference_point_px,
            p if scale.calibrated_space else None,
            z_step_um=scale.z_step_um if scale.calibrated_z else None,
        )
    else:
        for row in rows:
            row["distance_from_reference_px"] = None
            row["distance_from_reference_um"] = None
    return rows


def _turning_angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = _norm(a), _norm(b)
    if not (na > 0 and nb > 0) or not (math.isfinite(na) and math.isfinite(nb)):
        return None
    cos = float(np.dot(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# --------------------------------------------------------------------------
# Mean squared displacement
# --------------------------------------------------------------------------


def _msd_by_lag(g: _TrackGeometry) -> dict[int, list[float]]:
    """``lag -> [n_pairs, sum_sq_px, sum_sq_um]`` over every observation pair.

    The lag of a pair is its *actual* frame difference.  A track seen at
    frames 1, 2, 5 contributes lags 1, 3 and 4 -- not 1, 2 and 3, which is
    what indexing by observation number would claim, and which compresses
    time exactly where detections were missed.
    """
    n = len(g.frames)
    out: dict[int, list[float]] = {}
    if n < 2:
        return out
    i, j = np.triu_indices(n, k=1)
    lags = g.frames[j] - g.frames[i]  # >= 0: frames are sorted
    # One pass per quantity (bincount), not one mask per lag: the latter is
    # cubic in the track length and took minutes on a 2000-point track.
    counts = np.bincount(lags)
    sum_px = (
        np.bincount(lags, weights=np.sum((g.px[j] - g.px[i]) ** 2, axis=1))
        if g.px is not None
        else None
    )
    sum_um = (
        np.bincount(lags, weights=np.sum((g.um[j] - g.um[i]) ** 2, axis=1))
        if g.um is not None
        else None
    )
    for lag in np.flatnonzero(counts):
        if lag <= 0:
            continue  # duplicate frames: not a time lag
        out[int(lag)] = [
            float(counts[lag]),
            float(sum_px[lag]) if sum_px is not None else math.nan,
            float(sum_um[lag]) if sum_um is not None else math.nan,
        ]
    return out


def msd_rows(tracks: Sequence["Track"], scale: Scale) -> list[dict[str, Any]]:
    """``track_msd.csv`` rows: the time-averaged MSD of every track, by lag.

    ``msd_um2`` (µm², never µm) needs the spatial calibration; ``msd_px2`` is
    kept for uncalibrated data.  Lag times need the frame interval.
    """
    rows: list[dict[str, Any]] = []
    for tr in sorted(tracks, key=lambda t: t.id):
        obs_list = sorted(tr.observations, key=lambda o: o.frame)
        if len(obs_list) < 2:
            continue
        g = _geometry(obs_list, scale)
        for lag, (n_pairs, sum_px, sum_um) in sorted(_msd_by_lag(g).items()):
            lag_min = scale.frames_to_min(lag) if scale.calibrated_time else None
            rows.append(
                {
                    "track_id": tr.id,
                    "lag_frames": lag,
                    "lag_time_min": lag_min,
                    "lag_time_hr": None if lag_min is None else lag_min / MIN_PER_HR,
                    "n_pairs": int(n_pairs),
                    "msd_um2": _finite(sum_um / n_pairs),
                    "msd_px2": _finite(sum_px / n_pairs),
                }
            )
    return rows


#: ``summarise`` takes a keyword argument named ``msd_rows`` (the contract's
#: name), which shadows the function inside it.
_msd_rows = msd_rows


def _ols_loglog(x: Sequence[float], y: Sequence[float]) -> tuple[float, float | None]:
    """Slope and r² of ``log y`` against ``log x``."""
    lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else None
    return float(slope), r2


def msd_fit(
    rows_for_one_track: Sequence[dict[str, Any]],
    measurement_config: MeasurementConfig | None = None,
) -> tuple[float, float | None, int] | None:
    """``(alpha, r2, n_lags)`` of ``MSD ~ lag^alpha``, or None when refused.

    A lag enters the fit only if it is averaged over at least
    ``msd_min_pairs`` pairs, is at most ``msd_max_lag_fraction`` of the
    track's span, and has a positive MSD (log 0 is undefined; a stationary
    track therefore has no exponent rather than a fabricated one).  The fit
    is refused unless ``msd_min_lags_for_fit`` lags qualify.

    The span is the largest lag present, because the pair (first, last
    observation) always exists.  µm² is fitted when present, px² otherwise;
    the exponent is the same in either unit and in frames or minutes, because
    rescaling either axis only moves the intercept.  ``r2`` is None when every
    qualifying MSD is equal (a horizontal line has no variance to explain).

    The time-averaged estimate of a short track is biased low, not just noisy:
    measured when this was written on 200 Gaussian random walks of 60 steps
    (true alpha 1), the median fitted alpha was 0.905-0.979 across 20 seeds,
    and the 5th-95th percentile of single tracks 0.62-1.26 for one seed.  A
    single track's alpha is a description of that track, not a classification
    of its motion.
    """
    cfg = measurement_config or MeasurementConfig()
    usable = [r for r in rows_for_one_track if r.get("lag_frames") is not None]
    if not usable:
        return None
    span = max(float(r["lag_frames"]) for r in usable)
    key = "msd_um2" if any(_finite(r.get("msd_um2")) is not None for r in usable) else "msd_px2"
    x: list[float] = []
    y: list[float] = []
    for r in usable:
        lag = float(r["lag_frames"])
        value = _finite(r.get(key))
        n_pairs = r.get("n_pairs") or 0
        if (
            lag > 0
            and n_pairs >= cfg.msd_min_pairs
            and lag <= cfg.msd_max_lag_fraction * span
            and value is not None
            and value > 0
        ):
            x.append(lag)
            y.append(value)
    if len(x) < max(2, cfg.msd_min_lags_for_fit):
        return None
    alpha, r2 = _ols_loglog(x, y)
    return alpha, r2, len(x)


# --------------------------------------------------------------------------
# Directional persistence
# --------------------------------------------------------------------------


def directional_autocorrelation(
    observations: Sequence[Any], scale: Scale
) -> dict[int, tuple[float, int]]:
    """``lag_frames -> (mean cos, n_pairs)`` between one-frame step directions.

    Only steps between observations one frame apart are used, labelled by the
    frame they end in, so every direction is a velocity over the same time
    and a lag is a real time difference.  A step across missed frames is an
    average over a longer interval and would bias the correlation upward; it
    is left out rather than mixed in.  Zero-length steps have no direction
    and are skipped.  Directions are taken in isotropic pixels, so a 3-D
    track needs its anisotropy.
    """
    obs_list = sorted(observations, key=lambda o: o.frame)
    if len(obs_list) < 3:
        return {}
    g = _geometry(obs_list, scale)
    if g.px is None:
        return {}
    ends: list[int] = []
    units: list[np.ndarray] = []
    for k in range(1, len(obs_list)):
        if g.frames[k] - g.frames[k - 1] != 1:
            continue
        step = g.px[k] - g.px[k - 1]
        length = _norm(step)
        if length > 0 and math.isfinite(length):
            ends.append(int(g.frames[k]))
            units.append(step / length)
    if len(units) < 2:
        return {}
    u = np.asarray(units)
    e = np.asarray(ends)
    i, j = np.triu_indices(len(units), k=1)
    lags = e[j] - e[i]
    counts = np.bincount(lags)
    sums = np.bincount(lags, weights=np.einsum("ij,ij->i", u[i], u[j]))
    return {
        int(lag): (float(sums[lag] / counts[lag]), int(counts[lag]))
        for lag in np.flatnonzero(counts)
        if lag > 0
    }


def _persistence_fit(
    curve: dict[int, tuple[float, int]], span_frames: int, cfg: MeasurementConfig
) -> tuple[float, float | None, int] | None:
    """``(persistence_frames, r2, n_lags)`` of ``C(t) = exp(-t / P)``, or None.

    Closed-form least squares on ``log C`` against lag, through the same lag
    qualification as the MSD fit (pairs, span fraction, minimum lag count).
    Only the initial decay is fitted: lags are taken in order and the fit
    stops at the first lag where ``C <= 0``.  Past that point the curve is
    noise around zero, and the logarithm of its positive half bends the fit
    flat.  Measured when this was written on synthetic heading-diffusion
    walks (true P = 2 / sigma^2 frames, 30 tracks each): with every
    positive lag, the median fitted P was 187 frames for a true 8 (2000-step
    tracks) and 57.6 for a true 22.2 (400 steps); with this rule, 8.65 and
    21.4.  No optimiser is involved, so nothing
    "converges" -- and the r² (against the mean of ``log C``, so it can be
    negative for a fit worse than a constant) is reported precisely because a
    fit existing says nothing about whether the exponential model is right.
    None when the fitted slope is not negative: no decay was observed, and an
    infinite persistence time is not a measurement.
    """
    x: list[float] = []
    y: list[float] = []
    for lag in sorted(curve):
        c, n_pairs = curve[lag]
        if c <= 0:
            break
        if n_pairs >= cfg.msd_min_pairs and lag <= cfg.msd_max_lag_fraction * span_frames:
            x.append(float(lag))
            y.append(math.log(c))
    if len(x) < max(2, cfg.msd_min_lags_for_fit):
        return None
    xs, ys = np.asarray(x), np.asarray(y)
    # C(0) = 1 by definition, so the model is log C = -t / P through the origin.
    denom = float(np.dot(xs, xs))
    slope = float(np.dot(xs, ys)) / denom if denom > 0 else 0.0
    if not slope < 0:
        return None
    resid = ys - slope * xs
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else None
    return -1.0 / slope, r2, len(x)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def summarise(
    tracks: Sequence["Track"],
    scale: Scale,
    *legacy: Any,
    source_frames: Sequence[int] | None = None,
    min_observations: int = 2,
    msd_rows: Sequence[dict[str, Any]] | None = None,
    measurement: MeasurementConfig | None = None,
) -> list[TrackSummary]:
    """One :class:`TrackSummary` per track.

    Accepts the legacy ``summarise(tracks, axis, scale)`` during the
    transition; the axis is ignored.  ``msd_rows`` may be the output of
    :func:`msd_rows` for the same tracks, to avoid computing it twice; it is
    computed here otherwise.
    """
    scale = _resolve_scale(scale, legacy)
    cfg = measurement or MeasurementConfig()
    msd_by_track: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    if msd_rows is None:
        msd_rows = _msd_rows(tracks, scale)
    for r in msd_rows:
        msd_by_track[r.get("track_id")].append(r)

    out: list[TrackSummary] = []
    for tr in sorted(tracks, key=lambda t: t.id):
        obs = sorted(tr.observations, key=lambda o: o.frame)
        if not obs:
            continue
        flags = set(getattr(tr, "flags", None) or ())
        if len(obs) < min_observations:
            flags.add("fragment")
        g = _geometry(obs, scale)

        gaps = [int(b) - int(a) for a, b in zip(g.frames[:-1], g.frames[1:]) if b - a > 1]
        span = int(g.frames[-1] - g.frames[0])

        speeds_min: list[float] = []
        speeds_px: list[float] = []
        path_px = 0.0 if g.px is not None else None
        path_um = 0.0 if g.um is not None else None
        for k in range(1, len(obs)):
            dt = int(g.frames[k] - g.frames[k - 1])
            if dt <= 0:
                continue
            if g.px is not None:
                d = _norm(g.px[k] - g.px[k - 1])
                if math.isfinite(d):
                    path_px += d
                    speeds_px.append(d / dt)
            if g.um is not None:
                d = _norm(g.um[k] - g.um[k - 1])
                if math.isfinite(d):
                    path_um += d
                    if scale.calibrated_time:
                        speeds_min.append(d / scale.frames_to_min(dt))

        net_px = _finite(_norm(g.px[-1] - g.px[0])) if g.px is not None else None
        net_um = _finite(_norm(g.um[-1] - g.um[0])) if g.um is not None else None
        max_d2s_um = (
            _finite(float(np.max(np.linalg.norm(g.um - g.um[0], axis=1))))
            if g.um is not None
            else None
        )

        # Divide by the elapsed time of the whole span, not by the number of
        # observations.  A track seen 8 times over 14 frames covers 14 frames of
        # elapsed time; dividing by 8 would report a cell moving twice as fast
        # as it did, and that error grows with exactly the missed detections
        # these estimators exist to survive.
        duration_min = scale.frames_to_min(span) if scale.calibrated_time else None
        moving_time = duration_min if (duration_min and span) else None
        net_speed = (net_um / moving_time) if (net_um is not None and moving_time) else None
        path_speed = (path_um / moving_time) if (path_um is not None and moving_time) else None

        turning = []
        for k in range(2, len(obs)):
            if g.px is None:
                break
            angle = _turning_angle_deg(g.px[k - 1] - g.px[k - 2], g.px[k] - g.px[k - 1])
            if angle is not None:
                turning.append(angle)

        curve = directional_autocorrelation(obs, scale)
        persistence = _persistence_fit(curve, span, cfg) if span else None
        p_min = (
            scale.frames_to_min(persistence[0])
            if (persistence is not None and scale.calibrated_time)
            else None
        )

        fit = msd_fit(msd_by_track.get(tr.id, []), cfg)

        areas = [
            float(a) for a in (getattr(o, "area_px", None) for o in obs) if a is not None
        ]
        mean_area_px = _mean(areas)
        volumes = []
        if g.three_d and scale.calibrated_space and scale.calibrated_z and scale.z_step_um:
            for o in obs:
                row: dict[str, Any] = {}
                _morphology(o, scale, row)
                if row["volume_um3"] is not None:
                    volumes.append(row["volume_um3"])
        margins = [
            m for m in (_finite(getattr(o, "link_margin", None)) for o in obs) if m is not None
        ]

        out.append(
            TrackSummary(
                track_id=tr.id,
                channel=getattr(tr, "channel", -1),
                dimensionality="3D" if g.three_d else "2D",
                first_frame=int(g.frames[0]),
                last_frame=int(g.frames[-1]),
                first_source_frame=_source_frame(source_frames, int(g.frames[0])),
                last_source_frame=_source_frame(source_frames, int(g.frames[-1])),
                n_observations=len(obs),
                n_gaps=len(gaps),
                total_missing_frames=sum(gap - 1 for gap in gaps),
                span_frames=span,
                duration_min=duration_min,
                duration_hr=None if duration_min is None else duration_min / MIN_PER_HR,
                net_displacement_px=net_px,
                net_displacement_um=net_um,
                path_length_px=path_px,
                path_length_um=path_um,
                max_distance_from_start_um=max_d2s_um,
                straightness=(
                    net_px / path_px
                    if (net_px is not None and path_px is not None and path_px > 1e-9)
                    else None
                ),
                mean_speed_um_per_min=_mean(speeds_min),
                median_speed_um_per_min=(
                    float(np.median(speeds_min)) if speeds_min else None
                ),
                max_speed_um_per_min=float(np.max(speeds_min)) if speeds_min else None,
                mean_speed_um_per_hr=_per_hr(_mean(speeds_min)),
                median_speed_um_per_hr=_per_hr(
                    float(np.median(speeds_min)) if speeds_min else None
                ),
                max_speed_um_per_hr=_per_hr(
                    float(np.max(speeds_min)) if speeds_min else None
                ),
                net_speed_um_per_min=net_speed,
                net_speed_um_per_hr=_per_hr(net_speed),
                path_speed_um_per_min=path_speed,
                path_speed_um_per_hr=_per_hr(path_speed),
                mean_speed_px_per_frame=_mean(speeds_px),
                net_speed_px_per_frame=(
                    net_px / span if (net_px is not None and span) else None
                ),
                mean_turning_angle_deg=_mean(turning),
                directional_autocorrelation=curve[1][0] if 1 in curve else None,
                persistence_time_min=p_min,
                persistence_time_hr=None if p_min is None else p_min / MIN_PER_HR,
                persistence_fit_r2=persistence[1] if persistence else None,
                persistence_fit_lags=persistence[2] if persistence else None,
                msd_alpha=fit[0] if fit else None,
                msd_alpha_r2=fit[1] if fit else None,
                msd_fit_lags=fit[2] if fit else None,
                mean_area_px=mean_area_px,
                mean_area_um2=(
                    mean_area_px * scale.pixel_size_um**2
                    if (mean_area_px is not None and scale.calibrated_space)
                    else None
                ),
                mean_volume_um3=_mean(volumes),
                min_link_margin_chi2=min(margins) if margins else None,
                flags=";".join(sorted(flags)),
            )
        )
    return out
