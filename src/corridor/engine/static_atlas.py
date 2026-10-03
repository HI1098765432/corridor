"""Bot 2 -- the static device atlas (``docs/ENGINE_4D.md`` section 2).

On the registered stack this bot separates what belongs to the *device* (the
microfluidic channel walls, the empty background, anything outside the device,
fixed artefacts) from what belongs to *cells*.  Cellpose-SAM zero-shot fails on
this data precisely because it latches onto the channel walls as the object
(``F1 0.00``); an atlas that names the walls lets a proposer be handed the
residual ``V' - B`` with the device removed (Experiment B').

The method is a robust temporal model:

* ``B = median_t(V')`` -- the device without the cells.
* ``MAD = median_t |V' - B|`` -- the per-voxel temporal spread.
* ``D_t = |V' - B| / (1.4826 * MAD + eps)`` -- a robust z-score of change.
* ``persistence`` -- the fraction of timepoints with ``D_t < change_k`` (how
  much of the time a voxel looks like the background model).
* ``edge_stability`` -- the median over t of the gradient-orientation agreement
  with ``B`` (a wall's edge points the same way in every frame).
* ``z_continuity`` (3-D) -- the fraction of a voxel's Z neighbours that are also
  persistently static (a wall runs through Z; a cell may not).

The **critical invariant**, tested in ``tests/test_engine_atlas.py``: low
temporal change ALONE never marks a region static.  A paused cell does not
move, yet it is a cell.  A region is called static only with *structural*
evidence:

1. ``persistence >= persist_fraction`` -- static for most of the series, not
   just for the few frames a cell happened to rest; **and**
2. size/extent beyond plausible cell morphology -- it belongs to a connected
   static structure longer or larger than the largest real cell (the longest
   labelled cell in ``docs/error_table.json`` is 205.7 px, the largest 2147 px);
   **and**
3. no track passes through it -- an optional occupancy mask from the tracker
   vetoes any voxel a cell was ever seen in.

The bot never tracks and never segments.  It exports ``B`` so a proposer can be
given ``V' - B``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------
# Class labels
# --------------------------------------------------------------------------

STATIC_BACKGROUND = 0
CHANNEL_WALL = 1
OUTSIDE_DEVICE = 2
ARTIFACT = 3
VALID_CELL_REGION = 4
UNKNOWN = 5

#: code -> name, for evidence files and tests.
CLASS_NAMES: Mapping[int, str] = {
    STATIC_BACKGROUND: "STATIC_BACKGROUND",
    CHANNEL_WALL: "CHANNEL_WALL",
    OUTSIDE_DEVICE: "OUTSIDE_DEVICE",
    ARTIFACT: "ARTIFACT",
    VALID_CELL_REGION: "VALID_CELL_REGION",
    UNKNOWN: "UNKNOWN",
}
#: name -> code, the inverse of :data:`CLASS_NAMES`.
CLASS_CODES: Mapping[str, int] = {name: code for code, name in CLASS_NAMES.items()}


@dataclass(frozen=True)
class AtlasConfig:
    """Frozen settings for :func:`build_atlas`.

    Every morphology bound carries the measured fact it comes from, so a reader
    knows why the number is what it is rather than finding a tuned constant.
    """

    #: ``|V' - B|`` beyond this many robust MADs is a change event at that voxel
    #: and timepoint.  3.0 is the usual robust-z-score outlier line.
    change_k: float = 3.0
    #: A voxel is a *candidate* for static only if it looks like the background
    #: model in at least this fraction of timepoints.  0.9 means a cell resting
    #: for up to ~10% of the series cannot be mistaken for the device.
    persist_fraction: float = 0.9
    #: Longest connected static extent a *cell* could plausibly have, in px.
    #: The longest labelled cell in docs/error_table.json is 205.7 px; a static
    #: structure longer than this is the device, not a cell.
    max_cell_extent_px: float = 260.0
    #: Largest connected static area a *cell* could plausibly have, in px.
    #: The largest labelled cell in docs/error_table.json is 2147 px.
    max_cell_area_px: float = 3200.0
    #: ``B`` brighter than the field background by this many robust spreads is a
    #: bright structure -- a channel wall rather than empty background.
    bright_k: float = 3.0
    #: Median gradient-orientation agreement (cosine) for a stable device edge.
    #: Exported as corroborating evidence; the wall/artifact split uses shape.
    edge_stability_min: float = 0.5
    #: ``extent^2 / area`` above which a device-scale bright static component is
    #: a wall rather than a compact fixed artefact.  A channel wall is long and
    #: thin (a 3 x 300 px ridge scores ~100); compact debris scores ~1.
    wall_elongation_min: float = 4.0
    #: 3-D only: fraction of a voxel's Z neighbours that must also be
    #: persistently static for it to read as a through-Z wall.
    z_continuity_min: float = 0.5
    #: Half-thickness, in px, of the band around a passed geometry lane line
    #: that is counted as wall (the lane sits on a bright wall ridge).
    wall_halfwidth_px: float = 3.0
    #: ``B`` at or below this spatial percentile, in a border-connected region,
    #: is outside the device.
    outside_percentile: float = 2.0
    eps: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtlasResult:
    """The background model, the per-voxel evidence, and the class map.

    All arrays are spatial (``(Y, X)`` for a 2-D movie, ``(Z, Y, X)`` for a
    3-D one).  ``background`` is ``B``; :meth:`residual` subtracts it from a
    stack so a proposer sees the device-free image.  ``to_dict`` summarises the
    evidence (counts and global statistics) without the arrays.
    """

    background: np.ndarray
    mad: np.ndarray
    persistence: np.ndarray
    edge_stability: np.ndarray
    change_max: np.ndarray
    z_continuity: np.ndarray | None
    class_map: np.ndarray
    has_z: bool
    config: AtlasConfig

    def residual(self, stack: np.ndarray) -> np.ndarray:
        """``V' - B`` for a registered stack, broadcast over time (float32)."""
        arr = np.asarray(stack, dtype=np.float32)
        return arr - self.background.astype(np.float32)

    def class_counts(self) -> dict[str, int]:
        """Voxel count per class name (only classes that appear)."""
        values, counts = np.unique(self.class_map, return_counts=True)
        return {CLASS_NAMES[int(v)]: int(c) for v, c in zip(values, counts)}

    def mask_for(self, class_code: int) -> np.ndarray:
        """Boolean mask of voxels assigned ``class_code``."""
        return self.class_map == class_code

    def to_dict(self) -> dict[str, Any]:
        total = int(self.class_map.size)
        counts = self.class_counts()
        return {
            "has_z": self.has_z,
            "shape": list(self.class_map.shape),
            "n_voxels": total,
            "class_counts": counts,
            "class_fractions": {
                name: counts.get(name, 0) / total if total else 0.0
                for name in CLASS_NAMES.values()
            },
            "background_median": float(np.median(self.background)),
            "mad_median": float(np.median(self.mad)),
            "persistence_mean": float(self.persistence.mean()),
            "config": self.config.to_dict(),
        }


# --------------------------------------------------------------------------
# Evidence fields
# --------------------------------------------------------------------------


def _edge_stability(volume: np.ndarray, background: np.ndarray, eps: float) -> np.ndarray:
    """Median-over-t cosine agreement between ``grad V'_t`` and ``grad B``.

    A fixed device edge (a wall) has its intensity gradient pointing the same
    way in every frame, so the cosine is ~1; a cell that moves through a voxel
    swings the gradient around, so the median cosine is low.  The score is
    forced to 0 where ``B`` is locally flat (no edge, so no orientation to
    agree with), which keeps empty background from reading as a stable edge.
    """
    grad_b = np.gradient(background.astype(np.float64))
    if background.ndim == 1:  # np.gradient returns a bare array for 1-D
        grad_b = [grad_b]
    mag_b = np.sqrt(sum(g * g for g in grad_b))
    cosines = np.empty((volume.shape[0],) + background.shape, dtype=np.float64)
    for t in range(volume.shape[0]):
        grad_v = np.gradient(volume[t].astype(np.float64))
        if volume[t].ndim == 1:
            grad_v = [grad_v]
        dot = sum(gb * gv for gb, gv in zip(grad_b, grad_v))
        mag_v = np.sqrt(sum(gv * gv for gv in grad_v))
        cosines[t] = dot / (mag_b * mag_v + eps)
    edge = np.median(cosines, axis=0)
    # Flat background carries no edge orientation; a robust floor on |grad B|.
    flat = mag_b <= (np.median(mag_b[mag_b > 0]) if np.any(mag_b > 0) else 0.0)
    edge[flat] = 0.0
    return edge


def _z_continuity(static_candidate: np.ndarray) -> np.ndarray:
    """Fraction of each voxel's in-bounds Z neighbours that are static (3-D).

    Z is spatial axis 0 of a ``(Z, Y, X)`` volume.  A wall runs continuously
    through Z, so its voxels have static neighbours above and below; a flat
    debris fleck in one plane does not.
    """
    cand = static_candidate.astype(np.float64)
    total = np.zeros_like(cand)
    count = np.zeros_like(cand)
    total[1:] += cand[:-1]
    count[1:] += 1.0
    total[:-1] += cand[1:]
    count[:-1] += 1.0
    return total / np.maximum(count, 1.0)


def _geometry_wall_mask(
    geometry: Any, spatial_shape: tuple[int, ...], has_z: bool, halfwidth_px: float
) -> np.ndarray:
    """Band within ``halfwidth_px`` of any passed lane centre line.

    A ``ChannelGeometry`` lane's centre line sits on a bright wall ridge
    (``core/geometry.py``), so the band around it is wall.  The lanes live in
    the ``(Y, X)`` plane; for a 3-D volume the same plane mask is broadcast
    over Z, because a wall spans every plane.
    """
    plane_shape = spatial_shape[1:] if has_z else spatial_shape
    wall_plane = np.zeros(plane_shape, dtype=bool)
    lanes = list(getattr(geometry, "lanes", []) or [])
    if lanes:
        yy, xx = np.mgrid[0 : plane_shape[0], 0 : plane_shape[1]].astype(np.float64)
        nearest = np.full(plane_shape, np.inf, dtype=np.float64)
        for lane in lanes:
            ox, oy = lane.origin
            nx, ny = lane.normal  # unit normal (x, y)
            offset = np.abs((xx - ox) * nx + (yy - oy) * ny)
            nearest = np.minimum(nearest, offset)
        wall_plane = nearest <= halfwidth_px
    if has_z:
        return np.broadcast_to(wall_plane, spatial_shape).copy()
    return wall_plane


def _label_components(mask: np.ndarray):
    """Label a boolean mask with full (face+diagonal) connectivity.

    Returns ``(labels, n, sizes, objects)`` where ``sizes[k]`` is the voxel
    count of label ``k`` and ``objects[k-1]`` its bounding-box slices.
    """
    structure = ndi.generate_binary_structure(mask.ndim, mask.ndim)
    labels, n = ndi.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel()) if n else np.array([mask.size])
    objects = ndi.find_objects(labels) if n else []
    return labels, n, sizes, objects


def _component_mask(labels, n, sizes, objects, predicate) -> np.ndarray:
    """Union of the labelled components whose ``(extent, area)`` passes ``predicate``."""
    keep: list[int] = []
    for label in range(1, n + 1):
        sl = objects[label - 1]
        if sl is None:
            continue
        extent = max(s.stop - s.start for s in sl)
        area = int(sizes[label])
        if predicate(extent, area):
            keep.append(label)
    if not keep:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, keep)


def _outside_mask(background: np.ndarray, cfg: AtlasConfig) -> np.ndarray:
    """Border-connected regions at or below the low intensity percentile.

    Outside the device is dark and reaches the frame edge; an interior dark
    patch (a lumen) is not border-connected and is left alone.
    """
    threshold = float(np.percentile(background, cfg.outside_percentile))
    low = background <= threshold
    structure = ndi.generate_binary_structure(background.ndim, background.ndim)
    labels, n = ndi.label(low, structure=structure)
    if n == 0:
        return np.zeros_like(background, dtype=bool)
    border_labels: set[int] = set()
    for axis in range(background.ndim):
        border_labels.update(np.unique(np.take(labels, 0, axis=axis)).tolist())
        border_labels.update(np.unique(np.take(labels, -1, axis=axis)).tolist())
    border_labels.discard(0)
    if not border_labels:
        return np.zeros_like(background, dtype=bool)
    return np.isin(labels, list(border_labels))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def build_atlas(
    stack: np.ndarray,
    geometry: Any | None = None,
    occupancy: np.ndarray | None = None,
    config: AtlasConfig | None = None,
) -> AtlasResult:
    """Classify every voxel of a registered stack into a device/cell class.

    ``stack`` is the registered ``(T, Y, X)`` or ``(T, Z, Y, X)`` data.
    ``geometry`` is an optional ``core.geometry.ChannelGeometry`` whose lanes
    contribute wall voxels.  ``occupancy`` is an optional spatial boolean mask
    of voxels a tracker ever saw a cell in; those voxels can never be device
    (the "no track through it" rule).  The bot never tracks: it only consumes
    an occupancy mask if one is supplied.
    """
    cfg = config or AtlasConfig()
    arr = np.asarray(stack)
    if arr.ndim == 3:
        has_z = False
    elif arr.ndim == 4:
        has_z = True
    else:
        raise ValueError(
            "build_atlas expects a (T, Y, X) or (T, Z, Y, X) stack; "
            f"got ndim={arr.ndim}"
        )
    volume = arr.astype(np.float32)
    spatial_shape = volume.shape[1:]

    if occupancy is None:
        occupied = np.zeros(spatial_shape, dtype=bool)
    else:
        occupied = np.asarray(occupancy, dtype=bool)
        if occupied.shape != spatial_shape:
            raise ValueError(
                f"occupancy shape {occupied.shape} does not match the spatial "
                f"shape {spatial_shape}"
            )

    # --- robust temporal model ---------------------------------------------
    background = np.median(volume, axis=0)
    abs_dev = np.abs(volume - background)
    mad = np.median(abs_dev, axis=0)
    # A per-voxel MAD estimated from only a handful of timepoints is sometimes
    # far below the true noise, which inflates D and pokes spurious "change"
    # holes in a genuinely static structure (measured: wall voxels dropping to
    # persistence 0.7 on a 10-frame phantom).  The field's median MAD is the
    # typical temporal noise of a static voxel; a change must clear that floor
    # to count, so the scale is never allowed below it.  This keeps the robust
    # z-score D = |V'-B| / (1.4826 * MAD + eps) but floors the MAD at the
    # measured noise level.
    mad_floor = float(np.median(mad))
    scale = 1.4826 * np.maximum(mad, mad_floor) + cfg.eps
    change = abs_dev / scale  # D_t, shape (T, *spatial)
    low_change = change < cfg.change_k
    persistence = low_change.mean(axis=0)
    change_max = change.max(axis=0)

    # --- structural evidence -----------------------------------------------
    edge_stability = _edge_stability(volume, background, cfg.eps)
    static_candidate = (persistence >= cfg.persist_fraction) & (~occupied)
    z_continuity = _z_continuity(static_candidate) if has_z else None

    background_level = float(np.median(background))
    b_spread = 1.4826 * float(np.median(np.abs(background - background_level))) + cfg.eps
    bright = (background - background_level) > cfg.bright_k * b_spread
    outside = _outside_mask(background, cfg)

    geom_wall = (
        _geometry_wall_mask(geometry, spatial_shape, has_z, cfg.wall_halfwidth_px)
        if geometry is not None
        else np.zeros(spatial_shape, dtype=bool)
    )

    # Split the static set into flat (empty background / outside) and bright
    # (walls, fixed artefacts).  Running the extent gate on each set SEPARATELY
    # is what keeps a still, bright cell out of the background: a bright cell
    # sits in the bright set, where it forms its own cell-sized component, and
    # is never merged into the field-spanning flat-background component.
    flat_static = static_candidate & ~bright
    bright_static = static_candidate & bright

    def _device_scale(extent: float, area: float) -> bool:
        # The longest side beyond the longest plausible cell is the
        # dimension-robust test (a cell never exceeds ~206 px on any axis; a
        # wall spans the field).  The area term catches a compact-but-huge 2-D
        # blob and is applied in 2-D only: in 3-D a single cell already holds
        # far more voxels than the 2-D area bound, so volume cannot gate there.
        if extent > cfg.max_cell_extent_px:
            return True
        if not has_z and area > cfg.max_cell_area_px:
            return True
        return False

    flat_labels = _label_components(flat_static)
    flat_struct = _component_mask(*flat_labels, _device_scale)

    bright_labels = _label_components(bright_static)
    bright_struct = _component_mask(*bright_labels, _device_scale)
    # A wall is device-scale AND elongated; a device-scale compact bright blob
    # is a fixed artefact.
    wall_shape = _component_mask(
        *bright_labels,
        lambda e, a: _device_scale(e, a) and (e * e / max(a, 1) >= cfg.wall_elongation_min),
    )

    if has_z:
        z_ok = z_continuity >= cfg.z_continuity_min
    else:
        z_ok = np.ones(spatial_shape, dtype=bool)

    structural = flat_struct | bright_struct

    # --- classification -----------------------------------------------------
    # Anything not device-scale static can hold a cell.  This is where a paused
    # or resting cell lands: either its persistence is below the floor (it
    # moved in and out) or its static footprint is cell-sized (a component the
    # extent gate dropped).  Either way it is never device.
    class_map = np.full(spatial_shape, UNKNOWN, dtype=np.int8)
    class_map[~structural] = VALID_CELL_REGION

    # Flat device-scale static: outside the device if dark and border-reaching,
    # otherwise the empty background.
    class_map[flat_struct & outside] = OUTSIDE_DEVICE
    class_map[flat_struct & (class_map == UNKNOWN)] = STATIC_BACKGROUND

    # Bright device-scale static: an elongated, Z-continuous ridge is a wall;
    # the rest is a fixed artefact.
    wall = wall_shape & z_ok
    class_map[wall] = CHANNEL_WALL
    class_map[bright_struct & (class_map == UNKNOWN)] = ARTIFACT

    # Geometry asserts a wall on the lane band, combining the measured channel
    # geometry with the atlas (ENGINE_4D section 2).  A measured wall ridge is
    # device structure wherever it was found; the only override is a tracked
    # cell (occupancy), never a stray noise flicker in the band.
    class_map[geom_wall & (~occupied)] = CHANNEL_WALL

    # Nothing structural should remain UNKNOWN; keep the device-safe default.
    class_map[structural & (class_map == UNKNOWN)] = STATIC_BACKGROUND

    return AtlasResult(
        background=background,
        mad=mad,
        persistence=persistence,
        edge_stability=edge_stability,
        change_max=change_max,
        z_continuity=z_continuity,
        class_map=class_map,
        has_z=has_z,
        config=cfg,
    )
