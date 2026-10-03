"""Bot 6c -- the precision floor on every displacement (``docs/ENGINE_4D.md``).

The whole engine exists to recover real sub-pixel motion, which makes it the
engine most able to invent motion that is not there.  So every displacement it
reports carries a floor, and a motion that does not clear its floor is reported
as *below-floor* -- an instrument limit, never biology.

The floor is built, in quadrature, from the independent sources of positional
uncertainty:

*   **Boundary uncertainty.**  A segmentation boundary is good to about
    ``+/-0.5`` px.  Averaged over an object of ``area_px`` pixels, that jitter
    shrinks the centroid uncertainty as ``sigma_centroid ~ 0.5 / sqrt(area_px)``
    (independent edge errors on a closed boundary; the contract's default).
*   **In-ROI noise.**  A noisier ROI localises each boundary pixel less well, so
    the effective edge jitter grows with the noise-to-signal ratio:
    ``edge_sigma = boundary_sigma * (1 + noise_to_signal)``.
*   **Registration error.**  Bot 1 aligns each frame to a reference with a
    residual ``sigma`` per axis; a displacement inherits it.

A **displacement** is a difference of two positions, so its variance is the sum
of the two positions' variances -- the floor is ``sqrt(2)`` larger than a
single-position sigma when both frames share the same precision.  A velocity
floor is the displacement floor divided by the elapsed time; a ``dV/dt`` floor
comes from the boundary jitter acting over the object's surface.

Units are in the names: ``_px`` is XY pixels, ``_um`` micrometres, ``_min``
minutes.  Nothing here assumes the data is calibrated; µm figures are ``None``
until a pixel size is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: A segmentation boundary is good to about this, in XY pixels.  The contract's
#: default, propagated to the centroid as ``sigma ~ 0.5 / sqrt(area_px)``.
DEFAULT_BOUNDARY_SIGMA_PX = 0.5
#: A displacement must exceed this many floors to count as real motion.  1.0
#: means "larger than the instrument can resolve"; raise it for a stricter bar.
DEFAULT_SIGNIFICANCE_K = 1.0


def centroid_sigma_px(area_px: float, boundary_sigma_px: float = DEFAULT_BOUNDARY_SIGMA_PX) -> float:
    """Centroid uncertainty from boundary jitter: ``boundary_sigma / sqrt(area)``.

    WHY this form: the centroid averages the positions of the object's pixels,
    and the only uncertain ones sit on the boundary.  Independent ``+/-sigma``
    errors there average down with the square root of the pixel count, which is
    why a big cell has a better-defined centre than a small one.
    """
    if area_px <= 0:
        return float("inf")
    return float(boundary_sigma_px) / math.sqrt(float(area_px))


@dataclass(frozen=True)
class PrecisionFloor:
    """The precision floor for one object's displacement between two frames.

    Build it with :func:`displacement_floor`; read ``floor_px`` /
    ``floor_um`` and the per-source components, then judge a measured
    displacement with :meth:`assess`.
    """

    area_px: float
    boundary_sigma_px: float
    noise_to_signal: float
    registration_sigma_px: float
    pixel_size_um: float | None

    # -- derived components (per single position, px) --------------------
    edge_sigma_px: float  # boundary jitter inflated by in-ROI noise
    centroid_sigma_px: float  # edge jitter averaged over the object
    position_sigma_px: float  # centroid + registration, in quadrature

    # -- the floor on a displacement (difference of two positions) -------
    floor_px: float
    floor_um: float | None

    def assess(
        self, magnitude_px: float, *, significance_k: float = DEFAULT_SIGNIFICANCE_K
    ) -> "DisplacementVerdict":
        """Judge a measured displacement magnitude against this floor."""
        floor = self.floor_px
        ratio = float(magnitude_px) / floor if floor > 0 else float("inf")
        return DisplacementVerdict(
            magnitude_px=float(magnitude_px),
            magnitude_um=(
                float(magnitude_px) * self.pixel_size_um if self.pixel_size_um else None
            ),
            floor_px=floor,
            floor_um=self.floor_um,
            ratio=ratio,
            significance_k=float(significance_k),
            below_floor=ratio < float(significance_k),
        )

    def velocity_floor_um_per_min(self, dt_min: float) -> float | None:
        """The floor divided by the elapsed time, in µm/min."""
        if self.floor_um is None or dt_min <= 0:
            return None
        return self.floor_um / float(dt_min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_px": self.area_px,
            "boundary_sigma_px": self.boundary_sigma_px,
            "noise_to_signal": self.noise_to_signal,
            "registration_sigma_px": self.registration_sigma_px,
            "pixel_size_um": self.pixel_size_um,
            "edge_sigma_px": self.edge_sigma_px,
            "centroid_sigma_px": self.centroid_sigma_px,
            "position_sigma_px": self.position_sigma_px,
            "floor_px": self.floor_px,
            "floor_um": self.floor_um,
        }


@dataclass(frozen=True)
class DisplacementVerdict:
    """Whether one displacement cleared its floor.  Evidence is inspectable."""

    magnitude_px: float
    magnitude_um: float | None
    floor_px: float
    floor_um: float | None
    ratio: float  # magnitude / floor; < k means below-floor
    significance_k: float
    below_floor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnitude_px": self.magnitude_px,
            "magnitude_um": self.magnitude_um,
            "floor_px": self.floor_px,
            "floor_um": self.floor_um,
            "ratio": self.ratio,
            "significance_k": self.significance_k,
            "below_floor": self.below_floor,
            "verdict": "below_floor" if self.below_floor else "resolved",
        }


def displacement_floor(
    area_px: float,
    *,
    boundary_sigma_px: float = DEFAULT_BOUNDARY_SIGMA_PX,
    noise_to_signal: float = 0.0,
    registration_sigma_px: float = 0.0,
    pixel_size_um: float | None = None,
) -> PrecisionFloor:
    """The precision floor on a displacement of an object of ``area_px`` pixels.

    ``noise_to_signal`` is the in-ROI noise standard deviation over the signal
    (dimensionless, ``>= 0``); ``registration_sigma_px`` is Bot 1's per-axis
    residual.  Both default to zero, giving the boundary-only floor.
    """
    if noise_to_signal < 0:
        raise ValueError(f"noise_to_signal must be >= 0, got {noise_to_signal}")
    if registration_sigma_px < 0:
        raise ValueError(f"registration_sigma_px must be >= 0, got {registration_sigma_px}")

    edge_sigma = float(boundary_sigma_px) * (1.0 + float(noise_to_signal))
    centroid_sigma = centroid_sigma_px(area_px, edge_sigma)
    position_sigma = math.hypot(centroid_sigma, float(registration_sigma_px))
    # a displacement is a difference of two independent positions
    floor_px = math.sqrt(2.0) * position_sigma
    floor_um = floor_px * float(pixel_size_um) if pixel_size_um else None
    return PrecisionFloor(
        area_px=float(area_px),
        boundary_sigma_px=float(boundary_sigma_px),
        noise_to_signal=float(noise_to_signal),
        registration_sigma_px=float(registration_sigma_px),
        pixel_size_um=pixel_size_um,
        edge_sigma_px=edge_sigma,
        centroid_sigma_px=centroid_sigma,
        position_sigma_px=position_sigma,
        floor_px=floor_px,
        floor_um=floor_um,
    )


def volume_rate_floor(
    surface_px: float,
    *,
    boundary_sigma_px: float = DEFAULT_BOUNDARY_SIGMA_PX,
    noise_to_signal: float = 0.0,
    dt_frames: float = 1.0,
    voxel_um3: float | None = None,
    frame_interval_min: float | None = None,
) -> dict[str, float | None]:
    """A floor on ``dV/dt`` from boundary jitter over the object's surface.

    The volume is uncertain by the boundary jitter acting over the whole
    surface: ``sigma_V ~ edge_sigma * surface`` (voxels), and a rate over two
    frames is ``sqrt(2) * sigma_V / dt``.  ``surface_px`` is the surface area
    in pixel/voxel units (perimeter in 2-D, surface voxels in 3-D).
    """
    edge_sigma = float(boundary_sigma_px) * (1.0 + float(noise_to_signal))
    sigma_v_vox = edge_sigma * float(surface_px)
    dt = max(float(dt_frames), 1.0)
    floor_vox_per_frame = math.sqrt(2.0) * sigma_v_vox / dt
    floor_um3_per_min: float | None = None
    if voxel_um3 and frame_interval_min and frame_interval_min > 0:
        floor_um3_per_min = (
            math.sqrt(2.0) * sigma_v_vox * float(voxel_um3) / (dt * float(frame_interval_min))
        )
    return {
        "sigma_volume_vox": sigma_v_vox,
        "floor_vox_per_frame": floor_vox_per_frame,
        "floor_um3_per_min": floor_um3_per_min,
    }
