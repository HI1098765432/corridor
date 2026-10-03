"""Bot 6a -- 4D cell tubes and their kinematics (``docs/ENGINE_4D.md`` Bot 6).

A *tube* ``C_k(x, y, z, t)`` is one tracked object followed through time.  This
bot assembles the tube from the tracker's output and measures, per timepoint,
the sub-pixel centroid, the volume, the surface area, and -- by gap-aware
finite differences against the previous observation -- the rate of volume
change ``dV/dt``, the 3-D velocity and the acceleration.

Three invariants it must honour, all of them from the 2.0 contract:

*   **Volume is never guessed from a 2-D movie.** Volume and surface area are
    filled only when the detection carries a true 3-D mask; otherwise the tube
    reports the XY footprint area and its rate of change, and every ``_um3`` /
    ``_um2`` field stays ``None`` (``detections.extract_detections_3d``).
*   **A 2-D movie is Z = 1, handled without a special code path.** Position is
    a numpy vector of length 2 or 3 throughout; the same arithmetic runs for
    both.
*   **Velocity is a gap-aware finite difference**, ``Δr / Δt`` over the elapsed
    time between observations, so a one-frame gap and a three-frame gap are not
    confused (the 2.0 measurement law, ``core/measurements.py``).  This bot
    computes its own kinematics from the measured centroids rather than reading
    the tracker's Kalman velocity: the Kalman state is a smoothed estimate, and
    a measurement bot must report what the pixels say.

Units live in the field names: ``_px`` is XY pixels, ``_vox`` is voxels,
``_slices`` is Z planes, ``_um`` is micrometres, ``_min`` is minutes.  Z is
measured in slices, never assumed to be sampled like XY.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from corridor.core.detections import Detection, extract_detections_3d
from corridor.core.tracking import Track


@dataclass(frozen=True)
class Calibration4D:
    """Physical sampling of the data.  Any axis may be unknown.

    ``pixel_size_um`` is the square XY pixel (dx = dy); ``z_step_um`` the Z
    plane spacing (dz); ``frame_interval_min`` the time between frames (dt).
    Nothing is assumed isotropic, and nothing is assumed calibrated.
    """

    pixel_size_um: float | None = None
    z_step_um: float | None = None
    frame_interval_min: float | None = None

    @property
    def calibrated_xy(self) -> bool:
        return self.pixel_size_um is not None and self.pixel_size_um > 0

    @property
    def calibrated_z(self) -> bool:
        return self.z_step_um is not None and self.z_step_um > 0

    @property
    def calibrated_time(self) -> bool:
        return self.frame_interval_min is not None and self.frame_interval_min > 0

    @property
    def anisotropy(self) -> float | None:
        """``z_step / pixel_size``; ``None`` unless both are known."""
        if self.calibrated_xy and self.calibrated_z:
            return float(self.z_step_um) / float(self.pixel_size_um)  # type: ignore[arg-type]
        return None

    def spacing_zyx_um(self) -> tuple[float, float, float] | None:
        """``(dz, dy, dx)`` for ``extract_detections_3d``; ``None`` if not full."""
        if self.calibrated_xy and self.calibrated_z:
            p = float(self.pixel_size_um)  # type: ignore[arg-type]
            return (float(self.z_step_um), p, p)  # type: ignore[arg-type]
        return None


def _volume_surface(
    det: Detection | None, calib: Calibration4D
) -> tuple[float | None, float | None, float | None]:
    """``(volume_vox, volume_um3, surface_area_um2)`` for one detection.

    The detection measured itself once (``extract_detections_3d``); those
    numbers are reused.  A 3-D mask that was never measured in 3-D (for
    instance a mask carried on a 2-D-measured detection) is measured here, so
    volume and surface are recovered wherever a 3-D mask exists -- as the
    contract asks -- and never fabricated where one does not.
    """
    if det is None:
        return None, None, None
    volume_vox = det.volume_vox
    volume_um3 = det.volume_um3
    surface = det.surface_area_um2
    mask = det.mask_crop
    is_3d = mask is not None and mask.ndim == 3

    if is_3d and (surface is None or volume_vox is None):
        spacing = calib.spacing_zyx_um()
        measured = extract_detections_3d(
            mask.astype(np.int32), int(det.frame), spacing_zyx_um=spacing
        )
        if measured:
            m = measured[0]
            volume_vox = m.volume_vox if volume_vox is None else volume_vox
            volume_um3 = m.volume_um3 if volume_um3 is None else volume_um3
            surface = m.surface_area_um2 if surface is None else surface

    if is_3d and volume_vox is None and mask is not None:
        volume_vox = float(mask.sum())
    if volume_um3 is None and volume_vox is not None and calib.spacing_zyx_um() is not None:
        p = float(calib.pixel_size_um)  # type: ignore[arg-type]
        volume_um3 = float(volume_vox) * p * p * float(calib.z_step_um)  # type: ignore[arg-type]
    return volume_vox, volume_um3, surface


@dataclass(frozen=True)
class TubePoint:
    """One timepoint of a tube.  Every rate is against the previous point."""

    frame: int
    gap_frames: int  # frames since the previous tube point (0 for the first)
    elapsed_min: float | None  # time of this frame, relative to the tube start

    # -- position --------------------------------------------------------
    centroid_px: tuple[float, ...]  # (x, y) or (x, y, z); z in slices
    centroid_um: tuple[float, ...] | None  # (x, y[, z]) in micrometres

    # -- size ------------------------------------------------------------
    area_px: float  # XY footprint (always defined, 2-D and 3-D)
    volume_vox: float | None
    volume_um3: float | None
    surface_area_um2: float | None

    # -- kinematics (None at the first point, or when uncalibrated) ------
    dt_min: float | None  # elapsed time since the previous point
    velocity_px_per_frame: tuple[float, ...] | None
    velocity_um_per_min: tuple[float, ...] | None
    speed_um_per_min: float | None
    acceleration_um_per_min2: tuple[float, ...] | None
    accel_magnitude_um_per_min2: float | None
    dV_dt_um3_per_min: float | None
    dV_dt_vox_per_frame: float | None
    dA_dt_px_per_frame: float | None  # footprint-area rate (2-D-valid)
    dA_dt_um2_per_min: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "gap_frames": self.gap_frames,
            "elapsed_min": self.elapsed_min,
            "centroid_px": list(self.centroid_px),
            "centroid_um": None if self.centroid_um is None else list(self.centroid_um),
            "area_px": self.area_px,
            "volume_vox": self.volume_vox,
            "volume_um3": self.volume_um3,
            "surface_area_um2": self.surface_area_um2,
            "dt_min": self.dt_min,
            "velocity_px_per_frame": (
                None if self.velocity_px_per_frame is None else list(self.velocity_px_per_frame)
            ),
            "velocity_um_per_min": (
                None if self.velocity_um_per_min is None else list(self.velocity_um_per_min)
            ),
            "speed_um_per_min": self.speed_um_per_min,
            "acceleration_um_per_min2": (
                None
                if self.acceleration_um_per_min2 is None
                else list(self.acceleration_um_per_min2)
            ),
            "accel_magnitude_um_per_min2": self.accel_magnitude_um_per_min2,
            "dV_dt_um3_per_min": self.dV_dt_um3_per_min,
            "dV_dt_vox_per_frame": self.dV_dt_vox_per_frame,
            "dA_dt_px_per_frame": self.dA_dt_px_per_frame,
            "dA_dt_um2_per_min": self.dA_dt_um2_per_min,
        }


@dataclass(frozen=True)
class CellTube:
    """A tracked object through time: ``C_k(x, y, z, t)`` and its measurements."""

    track_id: int
    ndim: int  # 2 or 3; the dimensionality of the positions
    points: tuple[TubePoint, ...]
    calibration: Calibration4D

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def is_3d(self) -> bool:
        return self.ndim == 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "ndim": self.ndim,
            "n_points": self.n_points,
            "calibration": {
                "pixel_size_um": self.calibration.pixel_size_um,
                "z_step_um": self.calibration.z_step_um,
                "frame_interval_min": self.calibration.frame_interval_min,
            },
            "points": [p.to_dict() for p in self.points],
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """One flat row per timepoint, for ``tubes.csv`` (schema in the contract)."""
        rows: list[dict[str, Any]] = []
        for p in self.points:
            row: dict[str, Any] = {"track_id": self.track_id}
            d = p.to_dict()
            cpx = d.pop("centroid_px")
            cum = d.pop("centroid_um")
            names = ("x", "y", "z")
            for i, name in enumerate(names):
                row[f"{name}_px"] = cpx[i] if i < len(cpx) else None
                row[f"{name}_um"] = cum[i] if (cum is not None and i < len(cum)) else None
            for vec_key, comp in (
                ("velocity_px_per_frame", "v{}_px_per_frame"),
                ("velocity_um_per_min", "v{}_um_per_min"),
                ("acceleration_um_per_min2", "a{}_um_per_min2"),
            ):
                vec = d.pop(vec_key)
                for i, name in enumerate(names):
                    row[comp.format(name)] = vec[i] if (vec is not None and i < len(vec)) else None
            row.update(d)
            rows.append(row)
        return rows


def _finite_diff(
    current: np.ndarray, previous: np.ndarray, denom: float | None
) -> np.ndarray | None:
    if denom is None or denom <= 0:
        return None
    return (current - previous) / denom


def _tube_points(
    observations: Sequence[Any], calib: Calibration4D
) -> tuple[list[TubePoint], int]:
    """Build the per-time measurements for one ordered sequence of observations.

    ``observations`` is anything with ``.frame`` and ``.detection`` (the
    tracker's :class:`~corridor.core.tracking.Observation`), already in time
    order.  The ndim is taken from the first detection that carries a Z
    coordinate, so a 2-D and a 3-D movie run through the same code.
    """
    pts: list[TubePoint] = []
    first_frame = int(observations[0].frame) if observations else 0

    prev_pos_px: np.ndarray | None = None
    prev_pos_um: np.ndarray | None = None
    prev_volume_vox: float | None = None
    prev_volume_um3: float | None = None
    prev_area_px: float | None = None
    prev_vel_um: np.ndarray | None = None
    prev_frame: int | None = None
    ndim = 2

    for obs in observations:
        det = getattr(obs, "detection", None)
        frame = int(obs.frame)
        z = getattr(obs, "z", None)
        if z is None and det is not None:
            z = det.z
        pos_components = [float(obs.x), float(obs.y)]
        if z is not None:
            pos_components.append(float(z))
            ndim = 3
        pos_px = np.array(pos_components, dtype=float)

        area_px = float(det.area_px) if det is not None else float(getattr(obs, "area_px", 0.0))
        volume_vox, volume_um3, surface = _volume_surface(det, calib)

        # physical centroid: x, y scaled by the pixel, z scaled by the Z step
        pos_um: np.ndarray | None = None
        if calib.calibrated_xy:
            comps = [pos_px[0] * calib.pixel_size_um, pos_px[1] * calib.pixel_size_um]
            if len(pos_px) == 3 and calib.calibrated_z:
                comps.append(pos_px[2] * calib.z_step_um)
            elif len(pos_px) == 3:
                comps = []  # a 3-D position without a Z step has no µm vector
            if comps:
                pos_um = np.array(comps, dtype=float)

        gap = 0 if prev_frame is None else frame - prev_frame
        dt_min = None
        if calib.calibrated_time and prev_frame is not None:
            dt_min = gap * float(calib.frame_interval_min)
        elapsed_min = (
            (frame - first_frame) * float(calib.frame_interval_min)
            if calib.calibrated_time
            else None
        )

        vel_px = vel_um = accel = None
        speed_um = accel_mag = None
        dV_dt_um3 = dV_dt_vox = dA_dt_px = dA_dt_um2 = None
        if prev_pos_px is not None:
            vp = _finite_diff(pos_px, prev_pos_px, float(gap))
            vel_px = tuple(float(v) for v in vp) if vp is not None else None
            if dt_min is not None and dt_min > 0:
                if pos_um is not None and prev_pos_um is not None and len(pos_um) == len(prev_pos_um):
                    vu = (pos_um - prev_pos_um) / dt_min
                    vel_um = tuple(float(v) for v in vu)
                    speed_um = float(np.linalg.norm(vu))
                    if prev_vel_um is not None and len(prev_vel_um) == len(vu):
                        au = (vu - prev_vel_um) / dt_min
                        accel = tuple(float(a) for a in au)
                        accel_mag = float(np.linalg.norm(au))
                    prev_vel_um = vu
                if volume_um3 is not None and prev_volume_um3 is not None:
                    dV_dt_um3 = (volume_um3 - prev_volume_um3) / dt_min
                if prev_area_px is not None:
                    dA_dt_um2 = None
                    if calib.calibrated_xy:
                        p2 = float(calib.pixel_size_um) ** 2
                        dA_dt_um2 = (area_px - prev_area_px) * p2 / dt_min
            if volume_vox is not None and prev_volume_vox is not None and gap > 0:
                dV_dt_vox = (volume_vox - prev_volume_vox) / gap
            if prev_area_px is not None and gap > 0:
                dA_dt_px = (area_px - prev_area_px) / gap

        pts.append(
            TubePoint(
                frame=frame,
                gap_frames=gap,
                elapsed_min=elapsed_min,
                centroid_px=tuple(float(v) for v in pos_px),
                centroid_um=None if pos_um is None else tuple(float(v) for v in pos_um),
                area_px=area_px,
                volume_vox=volume_vox,
                volume_um3=volume_um3,
                surface_area_um2=surface,
                dt_min=dt_min,
                velocity_px_per_frame=vel_px,
                velocity_um_per_min=vel_um,
                speed_um_per_min=speed_um,
                acceleration_um_per_min2=accel,
                accel_magnitude_um_per_min2=accel_mag,
                dV_dt_um3_per_min=dV_dt_um3,
                dV_dt_vox_per_frame=dV_dt_vox,
                dA_dt_px_per_frame=dA_dt_px,
                dA_dt_um2_per_min=dA_dt_um2,
            )
        )
        prev_pos_px = pos_px
        prev_pos_um = pos_um
        prev_volume_vox = volume_vox
        prev_volume_um3 = volume_um3
        prev_area_px = area_px
        prev_frame = frame

    return pts, ndim


def tube_from_detections(
    track_id: int,
    detections: Iterable[Detection],
    calibration: Calibration4D | None = None,
) -> CellTube:
    """Assemble one tube directly from a time-ordered list of detections.

    The building block behind :func:`assemble_tubes`, and the one the
    ground-truth tests use: it needs no tracker and no Kalman state, only the
    measured detections (each carrying its own frame, centroid and mask).
    """
    calib = calibration or Calibration4D()
    dets = sorted(detections, key=lambda d: int(d.frame))

    class _Obs:  # a minimal stand-in exposing what _tube_points reads
        __slots__ = ("frame", "x", "y", "z", "area_px", "detection")

        def __init__(self, d: Detection) -> None:
            self.frame = int(d.frame)
            self.x = float(d.x)
            self.y = float(d.y)
            self.z = d.z
            self.area_px = float(d.area_px)
            self.detection = d

    pts, ndim = _tube_points([_Obs(d) for d in dets], calib)
    return CellTube(track_id=int(track_id), ndim=ndim, points=tuple(pts), calibration=calib)


def assemble_tubes(
    tracks: Sequence[Track],
    calibration: Calibration4D | None = None,
) -> list[CellTube]:
    """Assemble a 4-D tube for every tracked object (contract Bot 6).

    Reads the tracker's observations -- each carries the measured detection
    with its mask -- and never re-segments.  Tracks are kept in id order.
    """
    calib = calibration or Calibration4D()
    tubes: list[CellTube] = []
    for tr in sorted(tracks, key=lambda t: int(t.id)):
        if not tr.observations:
            continue
        obs = sorted(tr.observations, key=lambda o: int(o.frame))
        pts, ndim = _tube_points(obs, calib)
        tubes.append(
            CellTube(track_id=int(tr.id), ndim=ndim, points=tuple(pts), calibration=calib)
        )
    return tubes
