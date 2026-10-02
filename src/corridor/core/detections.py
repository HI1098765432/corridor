"""Turning label images into measured detections.

A detection is measured once, here, from its own mask: everything downstream
(tracking, measurement, quality control, prediction research) reads these
numbers and never re-derives them from pixels.

Units follow the field names.  ``_px`` is XY pixels, ``_vox`` is voxels, and a
3-D length in µm exists only when the voxel spacing was given -- Z is never
assumed to be sampled like XY.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Sequence

import numpy as np
from skimage.measure import regionprops


#: Where a detection came from.  These live here, beside the field that holds
#: them, rather than in the segmentation module: quality control and export
#: both need to read provenance, and neither should have to import the
#: segmentation service to learn the name of a string.
SOURCE_PRIMARY = "primary"
#: Found only by an extra, more permissive segmentation pass.
SOURCE_ENSEMBLE = "ensemble"


@dataclass
class Detection:
    """One segmented object in one frame.

    Coordinates are in pixels, x = column, y = row, matching image display
    order.  ``regionprops`` reports centroids as (row, col); the conversion
    happens here, once.  A 3-D detection adds ``z`` in slices, and its
    ``bbox`` becomes ``(min_z, min_row, min_col, max_z, max_row, max_col)``.

    Every field after ``confidence`` is optional, so a detection built by hand
    (tests, recovery probes) or read back from a v1 CSV is still valid; a
    morphology value that was not measured is None, never a placeholder.
    """

    frame: int
    label: int
    x: float
    y: float
    #: XY area in pixels. For a 3-D detection, the area of its projection on
    #: the XY plane -- its footprint -- so that ``area_um2`` stays a true area.
    area_px: float
    bbox: tuple[int, ...]  # (min_row, min_col, max_row, max_col); 6 values in 3-D
    extent_px: int  # larger XY bounding-box dimension
    eccentricity: float
    orientation_rad: float  # angle of the major axis, image coords
    major_axis_px: float
    minor_axis_px: float
    solidity: float
    touches_border: bool
    channel: int = -1  # assigned later; -1 == unassigned
    #: How this detection was found. A detection the network asserted on its
    #: own is a stronger claim than one recovered where a track predicted it,
    #: and the two must never be indistinguishable downstream.
    source: str = SOURCE_PRIMARY
    confidence: float = 1.0

    # -- dimensionality ------------------------------------------------------
    #: Centroid plane, in slices (not µm). None for a 2-D detection.
    z: float | None = None

    # -- 2-D morphology ------------------------------------------------------
    perimeter_px: float | None = None
    #: 4 pi A / P^2, clipped to [0, 1]. 1 for a disk, about 0.26 for a
    #: supplied confined cell (an ellipse 100 x 11 px).
    circularity: float | None = None
    #: major / minor axis length; None for a one-pixel-wide object.
    aspect_ratio: float | None = None
    #: area / bounding-box area (regionprops' ``extent``). Named apart from
    #: ``extent_px``, which is a length.
    extent_fraction: float | None = None
    convex_area_px: float | None = None
    #: Diameter of the disk with the same area.
    equivalent_diameter_px: float | None = None

    # -- intensity (2-D and 3-D) ---------------------------------------------
    mean_intensity: float | None = None
    median_intensity: float | None = None

    # -- 3-D morphology ------------------------------------------------------
    volume_vox: float | None = None
    #: Only with a known voxel spacing.
    volume_um3: float | None = None
    surface_area_um2: float | None = None
    #: Full lengths of the equivalent ellipsoid's axes, longest first, in µm
    #: (regionprops' 3-D convention, ``sqrt(20 * eigenvalue)`` of the voxel
    #: covariance, which is exactly 2a, 2b, 2c for a solid ellipsoid).
    principal_axis_lengths: tuple[float, ...] | None = None
    #: l1 / l2 and l2 / l3 of the principal axes, so both are >= 1.
    elongation: float | None = None
    flatness: float | None = None
    #: pi^(1/3) (6 V)^(2/3) / A: 1 for a sphere, smaller for anything else.
    sphericity: float | None = None

    #: The object's own pixels, in bounding-box coordinates (``bool``; (Y, X)
    #: in 2-D, (Z, Y, X) in 3-D). Kept so that later stages -- the tracker's
    #: overlap term, morphology research -- can use the mask without a second
    #: pass over the label image. Not part of rows, repr or equality: an
    #: array is not a CSV cell, and two detections with equal measurements
    #: are equal.
    mask_crop: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def ndim(self) -> int:
        return 3 if self.z is not None else 2

    @property
    def position(self) -> np.ndarray:
        """``(x, y)`` or ``(x, y, z)``, in pixels and slices.

        Raw image units on purpose. Scaling Z by the anisotropy is the
        tracker's job, because only it knows whether the spacing is known.
        """
        if self.z is None:
            return np.array([self.x, self.y], dtype=float)
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def size(self) -> float:
        """What a size ratio compares: area in 2-D, voxel count in 3-D."""
        if self.ndim == 3 and self.volume_vox is not None:
            return float(self.volume_vox)
        return float(self.area_px)

    @property
    def axis_unit(self) -> np.ndarray:
        """Unit vector along the cell body's major axis, in (x, y)."""
        return np.array(
            [math.sin(self.orientation_rad), math.cos(self.orientation_rad)], dtype=float
        )

    def full_mask(self, shape: Sequence[int]) -> np.ndarray:
        """Paste ``mask_crop`` back into a frame (or volume) of ``shape``."""
        if self.mask_crop is None:
            raise ValueError(f"detection {self.frame}/{self.label} carries no mask")
        nd = len(self.bbox) // 2
        out = np.zeros(tuple(int(s) for s in shape), dtype=bool)
        window = tuple(slice(lo, hi) for lo, hi in zip(self.bbox[:nd], self.bbox[nd:]))
        out[window] = self.mask_crop
        return out

    def to_row(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _NOT_IN_ROWS}
        if len(self.bbox) == 6:
            bmin_z, bmin_r, bmin_c, bmax_z, bmax_r, bmax_c = self.bbox
        else:
            bmin_r, bmin_c, bmax_r, bmax_c = self.bbox
            bmin_z = bmax_z = None
        d.update(
            bbox_min_x=bmin_c,
            bbox_min_y=bmin_r,
            bbox_max_x=bmax_c,
            bbox_max_y=bmax_r,
            bbox_min_z=bmin_z,
            bbox_max_z=bmax_z,
        )
        lengths = tuple(self.principal_axis_lengths or ())
        for i in range(3):
            d[f"principal_axis_{i + 1}_um"] = lengths[i] if i < len(lengths) else None
        return d


#: Fields that never become CSV cells: the bbox is split into named columns,
#: the axis-length tuple into three, and the mask is not tabular at all.
_NOT_IN_ROWS = frozenset({"bbox", "mask_crop", "principal_axis_lengths"})


@dataclass
class FrameDiagnostics:
    """Per-frame record of what segmentation produced and what survived.

    Exists so that "the cell vanished" can always be attributed to either
    Cellpose or the post-filter, without rerunning anything.
    """

    frame: int
    raw_count: int = 0
    kept_count: int = 0
    removed_count: int = 0
    removed_extents: list[int] = field(default_factory=list)
    removed_areas: list[float] = field(default_factory=list)
    cellpose_message: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "raw_instances": self.raw_count,
            "kept_instances": self.kept_count,
            "removed_instances": self.removed_count,
            "removed_max_extent_px": max(self.removed_extents) if self.removed_extents else None,
            "removed_max_area_px": max(self.removed_areas) if self.removed_areas else None,
            "cellpose_message": self.cellpose_message,
        }


# --------------------------------------------------------------------------


def _orientation_to_xy_angle(props_orientation: float) -> float:
    """Keep scikit-image's convention and document it.

    ``regionprops.orientation`` is the angle between the 0th axis (rows, i.e.
    y) and the region's major axis, in ``[-pi/2, pi/2]``.  The corresponding
    unit vector in (x, y) is ``(sin(theta), cos(theta))``.  The sign is
    arbitrary -- a major axis has no head or tail -- so consumers must compare
    orientations with ``abs(dot)``.
    """
    return float(props_orientation)


def _intensity_stats(
    intensity: np.ndarray | None, window: tuple[slice, ...], crop: np.ndarray
) -> tuple[float | None, float | None]:
    """Mean and median of the object's own pixels, or (None, None).

    A multichannel intensity image is not reduced here: which channel is
    analysed is decided once, at import (``ImportConfig.channel_index``), and
    averaging channels would be a quantity nobody asked for.
    """
    if intensity is None or intensity.ndim != crop.ndim:
        return None, None
    values = np.asarray(intensity[window], dtype=float)[crop]
    if values.size == 0:
        return None, None
    return float(values.mean()), float(np.median(values))


def extract_detections(
    mask: np.ndarray,
    frame: int,
    intensity: np.ndarray | None = None,
) -> list[Detection]:
    """Measure every labelled instance in one 2-D frame.

    The perimeter is scikit-image's Crofton estimate rather than its default
    4-neighbour boundary walk. Measured when this was written over seven
    orientations and five random sub-pixel placements of digitised disks
    (radius 5-30 px) and ellipses (up to 100 x 11 px, the supplied cells'
    shape): the boundary walk reads -6.7 to +5.5 % and drifts with size, so
    a perfect disk's circularity is 0.92-0.93 at radius 30 px but 0.95-1.00
    at 10 px; Crofton reads -3.1 to +1.5 %, and 0.97-1.00 (after clipping)
    for every disk from radius 10 px. Its one weakness, axis-aligned
    rectangles (-4 % against the pixel-centre outline of 100 x 12 px), is not
    a cell shape.
    """
    if mask is None or mask.size == 0:
        return []
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"extract_detections measures one 2-D frame, got shape {mask.shape}; "
            "use extract_detections_3d for a Z stack"
        )
    if mask.max() <= 0:
        return []

    h, w = mask.shape[:2]
    out: list[Detection] = []
    for p in regionprops(mask.astype(np.int32), intensity_image=intensity):
        if p.label == 0:
            continue
        min_r, min_c, max_r, max_c = p.bbox
        bh, bw = max_r - min_r, max_c - min_c
        cy, cx = p.centroid
        # A single-pixel or perfectly symmetric region has no defined
        # eccentricity/orientation; scikit-image returns 0.0, which would be
        # silently read as "aligned with y". Flag it via eccentricity 0.
        try:
            ecc = float(p.eccentricity)
        except (ValueError, ZeroDivisionError):
            ecc = 0.0
        area = float(p.area)
        major = float(p.axis_major_length)
        minor = float(p.axis_minor_length)
        # Convex-hull properties are unreliable below three pixels, which is
        # why solidity has always been pinned to 1.0 there.
        tiny = area <= 2
        perimeter = float(p.perimeter_crofton)
        crop = np.asarray(p.image, dtype=bool)
        mean_i, median_i = _intensity_stats(intensity, p.slice, crop)
        out.append(
            Detection(
                frame=int(frame),
                label=int(p.label),
                x=float(cx),
                y=float(cy),
                area_px=area,
                bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                extent_px=int(max(bh, bw)),
                eccentricity=ecc,
                orientation_rad=_orientation_to_xy_angle(p.orientation),
                major_axis_px=major,
                minor_axis_px=minor,
                solidity=float(p.solidity) if not tiny else 1.0,
                touches_border=bool(
                    min_r == 0 or min_c == 0 or max_r >= h or max_c >= w
                ),
                perimeter_px=perimeter,
                circularity=(
                    float(np.clip(4.0 * math.pi * area / perimeter**2, 0.0, 1.0))
                    if perimeter > 0
                    else None
                ),
                aspect_ratio=major / minor if minor > 0 else None,
                extent_fraction=float(p.extent),
                convex_area_px=float(p.area_convex) if not tiny else area,
                equivalent_diameter_px=float(p.equivalent_diameter_area),
                mean_intensity=mean_i,
                median_intensity=median_i,
                mask_crop=crop,
            )
        )
    return out


# --------------------------------------------------------------------------
# 3-D
# --------------------------------------------------------------------------

#: Zero padding around a crop before meshing: three smoothing sigmas, so the
#: smoothed object falls to ~0 before the edge and the mesh always closes.
_MESH_PAD_VOX = 3
#: Gaussian sigma, in voxels *per axis*, applied before meshing. See
#: ``_volume_matched_mesh`` for why the binary mask is not meshed directly.
_MESH_SMOOTH_SIGMA_VOX = 1.0
#: The iso-level is searched until the mesh volume is within this fraction of
#: the voxel-count volume.
_MESH_VOLUME_TOLERANCE = 1e-3
_MESH_MAX_ITERATIONS = 30


def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _volume_matched_mesh(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """A closed surface of one object, in voxel units: ``(verts, faces, volume)``.

    Meshing the binary mask directly is the obvious method and it is biased.
    Measured when this was written, on digitised ellipsoids against Knud
    Thomsen's formula, marching cubes on a binary mask overestimates the area
    by 9 % at isotropic 0.5 µm voxels and by 12-22 % at 2.0 x 0.5 x 0.5 µm,
    because every Z step becomes a terrace; a sphere then reads a
    sphericity of 0.84.

    So the padded binary crop is smoothed by one voxel along each axis, which
    removes the terraces, and the iso-level is chosen so the mesh encloses
    the voxel-count volume, which smoothing alone would shrink (by 5-7 % of
    the area for an object two slices thick). The level is found by a
    bracketed secant search, because the enclosed volume falls monotonically
    as the level rises. Both steps are per voxel and per volume ratio, so the
    mesh does not depend on the spacing: scaling its vertices by the spacing
    is exactly marching cubes run *with* that spacing.

    None when no closed surface exists at any level (an object too small to
    survive the smoothing).
    """
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes

    target = float(crop.sum())
    smooth = ndi.gaussian_filter(
        np.pad(crop, _MESH_PAD_VOX).astype(np.float32), _MESH_SMOOTH_SIGMA_VOX
    )
    peak = float(smooth.max())
    if not target > 0 or not peak > 0:
        return None

    lo, hi = 1e-3 * peak, 0.999 * peak
    level = min(0.5, 0.5 * peak)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    previous: tuple[float, float] | None = None
    for _ in range(_MESH_MAX_ITERATIONS):
        try:
            verts, faces, _normals, _values = marching_cubes(smooth, level)
        except (ValueError, RuntimeError):
            break
        volume = _mesh_volume(verts, faces)
        best = (verts, faces, volume)
        if abs(volume / target - 1.0) <= _MESH_VOLUME_TOLERANCE:
            break
        if volume > target:
            lo = level
        else:
            hi = level
        step = None
        if previous is not None and previous[1] != volume:
            step = level - (volume - target) * (level - previous[0]) / (volume - previous[1])
        previous = (level, volume)
        level = step if step is not None and lo < step < hi else 0.5 * (lo + hi)
    return best


def _surface_area_um2(
    mesh: tuple[np.ndarray, np.ndarray, float], spacing_zyx_um: tuple[float, float, float]
) -> float:
    from skimage.measure import mesh_surface_area

    verts, faces, _volume = mesh
    return float(mesh_surface_area(verts * np.asarray(spacing_zyx_um), faces))


def _solidity_3d(crop: np.ndarray, mesh: tuple[np.ndarray, np.ndarray, float] | None) -> float:
    """Volume over convex-hull volume, in voxel units (no spacing needed).

    Measured when this was written, on digitised spheres, ellipsoids and a
    rod at 0.5 µm and at 2.0 x 0.5 x 0.5 µm voxels, all convex, against a
    20^3 cube with a 10 x 10 tunnel (true solidity 0.75):

        hull of                        convex shapes    tunnel cube
        voxel centres                  1.04 - 1.33      0.875
        voxel corners                  0.78 - 0.91      0.750
        regionprops convex image       0.84 - 0.95      0.750
        the volume-matched mesh        0.95 - 1.00      0.754

    Centre hulls cut every boundary voxel in half (solidity above 1); corner
    hulls and convex images keep the staircase, so a digitised sphere is
    never convex. The mesh is the same surface the area comes from, so it is
    the one used. Without a mesh (an object too small to survive smoothing),
    the corner hull: exact for a cuboid, and never coplanar, not even for a
    single voxel or a one-slice sheet. NaN only if Qhull still refuses.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import ConvexHull, QhullError

    if mesh is not None:
        points, volume = mesh[0], mesh[2]
    else:
        nz, ny, nx = crop.shape
        corners = np.zeros((nz + 1, ny + 1, nx + 1), dtype=bool)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    corners[dz:dz + nz, dy:dy + ny, dx:dx + nx] |= crop
        # Only corners on the lattice's own surface can be hull vertices.
        points = np.argwhere(corners & ~ndi.binary_erosion(corners)).astype(float)
        volume = float(crop.sum())
    try:
        hull_volume = float(ConvexHull(points).volume)
    except (QhullError, ValueError):
        return float("nan")
    return volume / hull_volume if hull_volume > 0 else float("nan")


def extract_detections_3d(
    mask_zyx: np.ndarray,
    frame: int,
    intensity: np.ndarray | None = None,
    spacing_zyx_um: Sequence[float] | None = None,
) -> list[Detection]:
    """Measure every labelled instance in one (Z, Y, X) label volume.

    Without ``spacing_zyx_um`` only spacing-free quantities are filled:
    centroid, bounding box, voxel count, the XY footprint, solidity and
    extent fraction (both ratios survive any per-axis scaling) and
    intensity. Every µm quantity -- volume, surface area, principal axes,
    elongation, flatness, sphericity -- stays None, because computing it would
    mean assuming the Z step equals the pixel size.

    The 2-D shape fields (``eccentricity``, ``orientation_rad``,
    ``major_axis_px``, ``minor_axis_px``) describe the footprint, the
    object's projection on the XY plane, so every consumer written for 2-D
    data still reads a meaningful in-plane shape. ``solidity`` is the 3-D
    one (see ``_solidity_3d``); the 2-D-only fields (perimeter, circularity,
    convex area, equivalent diameter, aspect ratio) stay None.

    Principal axes come from the covariance of voxel centres in µm, as
    ``sqrt(20 * eigenvalue)``: regionprops' own 3-D convention, which returns
    2a, 2b, 2c for a solid ellipsoid. (Its 2-D ``4 * sqrt(eigenvalue)`` would
    read 0.89 of the true length in 3-D.) Surface area comes from
    ``_volume_matched_mesh`` scaled by the spacing. Measured on digitised
    ellipsoids at 2.0 x 0.5 x 0.5 µm voxels when this was written: volume
    within 1.6 %, surface area +2.0 to +3.5 % against Knud Thomsen's formula,
    a sphere's sphericity 0.974 (0.997 at isotropic 0.5 µm).

    An object cut by the top or bottom slice is flagged ``touches_border``
    like one cut by the image edge: its mesh is closed across the cut, so its
    volume, area and sphericity describe the visible part only.
    """
    if mask_zyx is None or np.asarray(mask_zyx).size == 0:
        return []
    mask_zyx = np.asarray(mask_zyx)
    if mask_zyx.ndim != 3:
        raise ValueError(f"extract_detections_3d needs a (Z, Y, X) volume, got {mask_zyx.shape}")
    if intensity is not None:
        intensity = np.asarray(intensity)
        if intensity.shape != mask_zyx.shape:
            raise ValueError(
                f"intensity shape {intensity.shape} does not match labels {mask_zyx.shape}"
            )
    if mask_zyx.max() <= 0:
        return []

    spacing: tuple[float, float, float] | None = None
    if spacing_zyx_um is not None:
        spacing = tuple(float(s) for s in spacing_zyx_um)  # type: ignore[assignment]
        if len(spacing) != 3 or not all(math.isfinite(s) and s > 0 for s in spacing):
            raise ValueError(f"spacing_zyx_um must be three positive numbers, got {spacing_zyx_um}")
    voxel_um3 = float(np.prod(spacing)) if spacing else None

    depth, h, w = mask_zyx.shape
    out: list[Detection] = []
    for p in regionprops(mask_zyx.astype(np.int32)):
        if p.label == 0:
            continue
        min_z, min_r, min_c, max_z, max_r, max_c = (int(v) for v in p.bbox)
        cz, cy, cx = (float(v) for v in p.centroid)
        crop = np.asarray(p.image, dtype=bool)
        n_vox = float(crop.sum())

        footprint = crop.any(axis=0).astype(np.int32)
        fp = regionprops(footprint)[0]
        try:
            ecc = float(fp.eccentricity)
        except (ValueError, ZeroDivisionError):
            ecc = 0.0

        mesh = _volume_matched_mesh(crop)
        lengths: tuple[float, ...] | None = None
        elongation = flatness = sphericity = surface = volume_um3 = None
        if spacing is not None:
            coords = np.argwhere(crop) * np.asarray(spacing)
            if coords.shape[0] > 1:
                eig = np.linalg.eigvalsh(np.cov(coords, rowvar=False, bias=True))[::-1]
            else:
                eig = np.zeros(3)
            lengths = tuple(float(math.sqrt(20.0 * max(e, 0.0))) for e in eig)
            l1, l2, l3 = lengths
            elongation = l1 / l2 if l2 > 0 else None
            flatness = l2 / l3 if l3 > 0 else None
            volume_um3 = n_vox * voxel_um3
            surface = _surface_area_um2(mesh, spacing) if mesh is not None else None
            if surface:
                sphericity = math.pi ** (1.0 / 3.0) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface

        window = (slice(min_z, max_z), slice(min_r, max_r), slice(min_c, max_c))
        mean_i, median_i = _intensity_stats(intensity, window, crop)
        bbox_vox = float((max_z - min_z) * (max_r - min_r) * (max_c - min_c))
        out.append(
            Detection(
                frame=int(frame),
                label=int(p.label),
                x=cx,
                y=cy,
                z=cz,
                area_px=float(footprint.sum()),
                bbox=(min_z, min_r, min_c, max_z, max_r, max_c),
                extent_px=int(max(max_r - min_r, max_c - min_c)),
                eccentricity=ecc,
                orientation_rad=_orientation_to_xy_angle(fp.orientation),
                major_axis_px=float(fp.axis_major_length),
                minor_axis_px=float(fp.axis_minor_length),
                solidity=_solidity_3d(crop, mesh),
                touches_border=bool(
                    min_z == 0 or min_r == 0 or min_c == 0
                    or max_z >= depth or max_r >= h or max_c >= w
                ),
                extent_fraction=n_vox / bbox_vox if bbox_vox > 0 else None,
                mean_intensity=mean_i,
                median_intensity=median_i,
                volume_vox=n_vox,
                volume_um3=volume_um3,
                surface_area_um2=surface,
                principal_axis_lengths=lengths,
                elongation=elongation,
                flatness=flatness,
                sphericity=sphericity,
                mask_crop=crop,
            )
        )
    return out


def detections_to_rows(
    detections: Iterable[Detection],
    *,
    pixel_size_um: float | None = None,
    frame_interval_min: float | None = None,
    source_frames: list[int] | None = None,
    z_step_um: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in detections:
        row = d.to_row()
        row["source_frame"] = (
            source_frames[d.frame]
            if source_frames is not None and d.frame < len(source_frames)
            else None
        )
        row["elapsed_min"] = (
            d.frame * frame_interval_min if frame_interval_min else None
        )
        if pixel_size_um:
            row["x_um"] = d.x * pixel_size_um
            row["y_um"] = d.y * pixel_size_um
            row["area_um2"] = d.area_px * pixel_size_um * pixel_size_um
            row["major_axis_um"] = d.major_axis_px * pixel_size_um
            row["minor_axis_um"] = d.minor_axis_px * pixel_size_um
            for name in ("perimeter", "equivalent_diameter"):
                value = getattr(d, f"{name}_px")
                row[f"{name}_um"] = value * pixel_size_um if value is not None else None
            row["convex_area_um2"] = (
                d.convex_area_px * pixel_size_um * pixel_size_um
                if d.convex_area_px is not None
                else None
            )
        if z_step_um and d.z is not None:
            row["z_um"] = d.z * z_step_um
            if pixel_size_um and d.volume_vox is not None and d.volume_um3 is None:
                row["volume_um3"] = d.volume_vox * pixel_size_um * pixel_size_um * z_step_um
        rows.append(row)
    return rows
