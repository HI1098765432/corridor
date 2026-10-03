"""Bot 6b -- surface displacement, translation removed (``docs/ENGINE_4D.md``).

A migrating cell both *moves* and *deforms*.  Counting every boundary pixel
that changed as "deformation" charges the whole translation to the membrane,
which is wrong: a cell that slides one body-width without changing shape would
read as a complete resurfacing.  So this bot splits the motion in two:

1.  **Whole-cell translation**, estimated by sub-pixel phase correlation on the
    ROI (``skimage.registration.phase_cross_correlation``).  It is *removed*
    first -- the later mask is shifted back onto the earlier one.
2.  **Local surface displacement** ``u(s, t)`` of what is left, read from the
    signed-distance fields of the two aligned masks
    (``scipy.ndimage.distance_transform_edt`` on a mask and on its complement).
    On the earlier boundary ``s``, ``u(s) = -phi_{t+1}^aligned(s)``: positive
    where the surface pushed outward (extension), negative where it pulled in
    (retraction), in pixels along the outward normal.

The gained and lost regions are reported as area (2-D) or volume (3-D), and
each is localised relative to the direction of motion: **front** (ahead of the
centroid along the motion direction), **rear** (behind) or **flank** (to the
side).  That front/rear split is the biology the engine exists to recover -- a
leading-edge protrusion is extension at the front, a retracting tail is
retraction at the rear.

Sign and ordering conventions, stated once:

*   The signed-distance field ``phi`` is **positive outside** the cell and
    **negative inside** (``edt(~mask) - edt(mask)``), zero on the boundary.
*   numpy arrays are indexed ``[y, x]`` (2-D) or ``[z, y, x]`` (3-D); every
    vector this bot *reports* is reordered to ``(x, y)`` or ``(x, y, z)``.
*   Translation is the cell's displacement from ``t`` to ``t+1``.  Volumes use
    the full anisotropic voxel; the normal displacement ``u`` is reported in
    XY-pixel-equivalent units (Z scaled by the anisotropy), so one number has
    one meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

#: Gaussian sigma (px) applied to a *mask* before phase correlation, so the
#: sub-pixel refinement has an edge gradient to work on.  Not applied to a
#: supplied intensity ROI, which already has one.
_CORR_SMOOTH_SIGMA = 2.0


def _to_xy(vec_array_order: Sequence[float]) -> tuple[float, ...]:
    """Reorder an array-indexed vector ``[y, x]`` / ``[z, y, x]`` to ``(x, y[, z])``."""
    v = list(vec_array_order)
    if len(v) == 2:
        return (float(v[1]), float(v[0]))
    return (float(v[2]), float(v[1]), float(v[0]))


@dataclass(frozen=True)
class SurfaceDelta:
    """Extension and retraction between two masks, translation removed."""

    ndim: int
    # -- whole-cell translation (cell displacement t -> t+1) -------------
    translation_px: tuple[float, ...]  # (x, y[, z]); phase-correlation estimate
    translation_magnitude_px: float
    centroid_translation_px: tuple[float, ...]  # cross-check, centroid difference
    translation_method: str  # "phase_correlation" or "centroid_fallback"

    # -- gained / lost (after translation removed) -----------------------
    extension_px: float  # voxels/pixels gained
    retraction_px: float  # voxels/pixels lost
    net_change_px: float  # extension - retraction
    extension_um: float | None  # area (2-D) or volume (3-D) in µm^2 / µm^3
    retraction_um: float | None
    net_change_um: float | None

    # -- boundary normal displacement u(s) on the earlier surface --------
    n_boundary_samples: int
    mean_extension_px: float  # mean of positive u (0 if none)
    mean_retraction_px: float  # mean of |negative u| (0 if none)
    max_extension_px: float
    max_retraction_px: float
    median_abs_displacement_px: float

    # -- localisation relative to the motion direction -------------------
    motion_direction: tuple[float, ...] | None  # unit vector (x, y[, z])
    motion_direction_source: str  # explicit | translation | major_axis | none
    extension_by_location_px: dict[str, float]  # front / rear / flank
    retraction_by_location_px: dict[str, float]
    dominant_extension_location: str  # front | rear | flank | none
    dominant_retraction_location: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ndim": self.ndim,
            "translation_px": list(self.translation_px),
            "translation_magnitude_px": self.translation_magnitude_px,
            "centroid_translation_px": list(self.centroid_translation_px),
            "translation_method": self.translation_method,
            "extension_px": self.extension_px,
            "retraction_px": self.retraction_px,
            "net_change_px": self.net_change_px,
            "extension_um": self.extension_um,
            "retraction_um": self.retraction_um,
            "net_change_um": self.net_change_um,
            "n_boundary_samples": self.n_boundary_samples,
            "mean_extension_px": self.mean_extension_px,
            "mean_retraction_px": self.mean_retraction_px,
            "max_extension_px": self.max_extension_px,
            "max_retraction_px": self.max_retraction_px,
            "median_abs_displacement_px": self.median_abs_displacement_px,
            "motion_direction": (
                None if self.motion_direction is None else list(self.motion_direction)
            ),
            "motion_direction_source": self.motion_direction_source,
            "extension_by_location_px": dict(self.extension_by_location_px),
            "retraction_by_location_px": dict(self.retraction_by_location_px),
            "dominant_extension_location": self.dominant_extension_location,
            "dominant_retraction_location": self.dominant_retraction_location,
        }


def _centroid_array_order(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords.mean(axis=0) if coords.size else np.zeros(mask.ndim)


def _signed_distance(mask: np.ndarray, sampling: Sequence[float]) -> np.ndarray:
    """``phi``: positive outside, negative inside, zero on the boundary."""
    from scipy import ndimage as ndi

    inside = mask.astype(bool)
    outside = ~inside
    phi = ndi.distance_transform_edt(outside, sampling=sampling) - ndi.distance_transform_edt(
        inside, sampling=sampling
    )
    return phi


def _major_axis_array_order(mask: np.ndarray, sampling: Sequence[float]) -> np.ndarray | None:
    """Unit principal axis of the mask, in array order, metric-scaled by sampling."""
    coords = np.argwhere(mask).astype(float)
    if coords.shape[0] < 2:
        return None
    coords_metric = coords * np.asarray(sampling, dtype=float)
    cov = np.cov(coords_metric, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    # back to raw array-index direction (undo the metric scaling)
    axis = axis / np.asarray(sampling, dtype=float)
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else None


def _classify(offsets_metric: np.ndarray, direction_metric: np.ndarray) -> np.ndarray:
    """front / rear / flank for each offset, by parallel vs perpendicular size.

    Parallel component beyond the perpendicular magnitude is front (positive) or
    rear (negative); otherwise flank.  A 45-degree cone -- no tuned threshold.
    """
    d = direction_metric / (np.linalg.norm(direction_metric) or 1.0)
    parallel = offsets_metric @ d
    perp = offsets_metric - np.outer(parallel, d)
    perp_mag = np.linalg.norm(perp, axis=1)
    out = np.empty(len(offsets_metric), dtype=object)
    front = parallel > perp_mag
    rear = (-parallel) > perp_mag
    out[:] = "flank"
    out[front] = "front"
    out[rear] = "rear"
    return out


def _dominant(by_location: dict[str, float]) -> str:
    total = sum(by_location.values())
    if total <= 0:
        return "none"
    return max(by_location, key=lambda k: by_location[k])


def surface_delta(
    mask_t: np.ndarray,
    mask_tp1: np.ndarray,
    *,
    pixel_size_um: float | None = None,
    z_step_um: float | None = None,
    image_t: np.ndarray | None = None,
    image_tp1: np.ndarray | None = None,
    motion_direction: Sequence[float] | None = None,
    translation_floor_px: float = 0.5,
) -> SurfaceDelta:
    """Split the change between two aligned-ROI masks into translation + surface.

    ``mask_t`` and ``mask_tp1`` are boolean arrays of the **same shape** (a
    common ROI), 2-D or 3-D.  ``image_t`` / ``image_tp1`` are the matching
    intensity ROIs for phase correlation; without them the masks themselves are
    correlated.  ``motion_direction`` overrides the direction used for the
    front/rear split (given in ``(x, y[, z])``); when omitted it comes from the
    estimated translation, and if that is below ``translation_floor_px`` from
    the mask's major axis.
    """
    from scipy import ndimage as ndi
    from skimage.registration import phase_cross_correlation

    mask_t = np.asarray(mask_t).astype(bool)
    mask_tp1 = np.asarray(mask_tp1).astype(bool)
    if mask_t.shape != mask_tp1.shape:
        raise ValueError(
            f"surface_delta needs two masks in a common ROI; got {mask_t.shape} and "
            f"{mask_tp1.shape}"
        )
    if mask_t.ndim not in (2, 3):
        raise ValueError(f"surface_delta handles 2-D or 3-D masks, got {mask_t.ndim}-D")
    ndim = int(mask_t.ndim)

    # anisotropy only matters in 3-D; XY stays isotropic, Z scaled by dz/dxy
    anisotropy = 1.0
    if ndim == 3 and pixel_size_um and z_step_um and pixel_size_um > 0:
        anisotropy = float(z_step_um) / float(pixel_size_um)
    sampling = ([anisotropy, 1.0, 1.0] if ndim == 3 else [1.0, 1.0])

    # -- 1. whole-cell translation ---------------------------------------
    # Phase correlation reads sub-pixel shifts from the phase of the cross-power
    # spectrum, which needs a gradient to work on.  A hard binary mask has
    # gradient only on its one-pixel rim, so its correlation peak quantises to
    # whole pixels; a light Gaussian gives the DFT upsampling real edges to
    # refine against.  A supplied intensity ROI already has gradient and is
    # used as-is.
    have_intensity = image_t is not None and image_tp1 is not None
    if have_intensity:
        ref = np.asarray(image_t, dtype=np.float32)
        mov = np.asarray(image_tp1, dtype=np.float32)
    else:
        ref = ndi.gaussian_filter(mask_t.astype(np.float32), _CORR_SMOOTH_SIGMA)
        mov = ndi.gaussian_filter(mask_tp1.astype(np.float32), _CORR_SMOOTH_SIGMA)
    method = "phase_correlation"
    shift = np.zeros(ndim)
    if mask_t.any() and mask_tp1.any():
        try:
            shift, _err, _phase = phase_cross_correlation(ref, mov, upsample_factor=100)
            shift = np.asarray(shift, dtype=float)
        except Exception:  # pragma: no cover - degenerate ROI
            method = "centroid_fallback"
    else:
        method = "centroid_fallback"

    c_t = _centroid_array_order(mask_t)
    c_tp1 = _centroid_array_order(mask_tp1)
    centroid_shift = c_tp1 - c_t  # cell displacement t -> t+1 (array order)
    if method == "centroid_fallback":
        # ndi.shift(mov, s) aligns mov onto ref; to align t+1 onto t that is
        # -(centroid displacement).
        shift = -centroid_shift

    # ndi.shift(mask_tp1, shift) registers t+1 onto t; the cell's displacement
    # is therefore the negative of that alignment shift.
    translation_array = -shift
    aligned = ndi.shift(mask_tp1.astype(np.float32), shift, order=1, mode="constant", cval=0.0)
    aligned_mask = aligned >= 0.5

    # -- 2. gained / lost after translation removed ----------------------
    extension_region = aligned_mask & ~mask_t
    retraction_region = mask_t & ~aligned_mask
    extension_px = float(extension_region.sum())
    retraction_px = float(retraction_region.sum())

    extension_um = retraction_um = net_um = None
    if pixel_size_um and pixel_size_um > 0:
        if ndim == 2:
            unit = float(pixel_size_um) ** 2
        else:
            unit = float(pixel_size_um) ** 2 * float(z_step_um) if z_step_um else None
        if unit is not None:
            extension_um = extension_px * unit
            retraction_um = retraction_px * unit
            net_um = extension_um - retraction_um

    # -- 3. boundary normal displacement u(s) = phi_t(s) - phi_{t+1}^aligned(s)
    # The *difference* of the two signed-distance fields, not -phi_{t+1} alone:
    # a boundary pixel sampled on the grid sits ~1 px inside the true contour,
    # so phi_t(s) is about -1 there, not 0.  Taking the difference cancels that
    # discretisation offset, so an unchanged boundary reads exactly zero
    # displacement (verified by the integer-translation test).  Outward is
    # positive: where the cell extended, the old boundary point is now deeper
    # inside, phi_{t+1} is more negative, and phi_t - phi_{t+1} > 0.
    phi_t = _signed_distance(mask_t, sampling)
    phi_tp1 = _signed_distance(aligned_mask, sampling)
    boundary = mask_t & ~ndi.binary_erosion(mask_t)
    u = (phi_t - phi_tp1)[boundary]  # px (metric-scaled) along the outward normal
    n_boundary = int(boundary.sum())
    pos = u[u > 0]
    neg = u[u < 0]
    mean_ext = float(pos.mean()) if pos.size else 0.0
    mean_ret = float(-neg.mean()) if neg.size else 0.0
    max_ext = float(pos.max()) if pos.size else 0.0
    max_ret = float(-neg.min()) if neg.size else 0.0
    median_abs = float(np.median(np.abs(u))) if u.size else 0.0

    # -- 4. localisation relative to motion direction --------------------
    direction_source = "none"
    direction_array: np.ndarray | None = None
    if motion_direction is not None:
        md = np.asarray(motion_direction, dtype=float)
        if md.size == ndim and np.linalg.norm(md) > 0:
            # caller gives (x, y[, z]); convert to array order (z, y, x / y, x)
            direction_array = md[::-1].copy()
            direction_source = "explicit"
    if direction_array is None and np.linalg.norm(translation_array) >= translation_floor_px:
        direction_array = translation_array.astype(float)
        direction_source = "translation"
    if direction_array is None:
        axis = _major_axis_array_order(mask_t, sampling)
        if axis is not None:
            direction_array = axis
            direction_source = "major_axis"

    ext_loc = {"front": 0.0, "rear": 0.0, "flank": 0.0}
    ret_loc = {"front": 0.0, "rear": 0.0, "flank": 0.0}
    if direction_array is not None:
        direction_metric = direction_array * np.asarray(sampling)
        for region, bucket in ((extension_region, ext_loc), (retraction_region, ret_loc)):
            coords = np.argwhere(region).astype(float)
            if coords.shape[0] == 0:
                continue
            offsets = (coords - c_t) * np.asarray(sampling)  # metric-scaled offsets
            labels = _classify(offsets, direction_metric)
            for lab in ("front", "rear", "flank"):
                bucket[lab] = float(np.count_nonzero(labels == lab))

    return SurfaceDelta(
        ndim=ndim,
        translation_px=_to_xy(translation_array),
        translation_magnitude_px=float(np.linalg.norm(translation_array)),
        centroid_translation_px=_to_xy(centroid_shift),
        translation_method=method,
        extension_px=extension_px,
        retraction_px=retraction_px,
        net_change_px=extension_px - retraction_px,
        extension_um=extension_um,
        retraction_um=retraction_um,
        net_change_um=net_um,
        n_boundary_samples=n_boundary,
        mean_extension_px=mean_ext,
        mean_retraction_px=mean_ret,
        max_extension_px=max_ext,
        max_retraction_px=max_ret,
        median_abs_displacement_px=median_abs,
        motion_direction=(
            None if direction_array is None else _to_xy(direction_array / (np.linalg.norm(direction_array) or 1.0))
        ),
        motion_direction_source=direction_source,
        extension_by_location_px=ext_loc,
        retraction_by_location_px=ret_loc,
        dominant_extension_location=_dominant(ext_loc),
        dominant_retraction_location=_dominant(ret_loc),
    )
