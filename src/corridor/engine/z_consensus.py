"""Bot 4 -- Z consensus: link 2-D slice components into 3-D objects.

Design contract: ``docs/ENGINE_4D.md`` section 2.  The proposer segments each
Z slice independently, so the same physical cell wears a different label on
every slice and may be dropped on one.  This bot decides **only Z membership**:
which 2-D components belong to the same 3-D object.  It never segments, never
tracks across time, and never estimates a volume.

**The links.**  Between every pair of adjacent slices it solves a linear
assignment problem (``scipy.optimize.linear_sum_assignment``) on the cost

    cost = w1 (1 - IoU)
         + w2 d_centroid_px / sigma_centroid_px
         + w3 |ln(area_b / area_a)|
         + w4 d_shape

with ``d_shape`` a bounded, measured descriptor difference
``|Δeccentricity| + |Δfill_fraction|`` (both in [0, 1]; eccentricity from the
component's inertia tensor, fill fraction = area / bounding-box area).  IoU and
the centroid distance dominate; area and shape are tie-breakers.  A selected
pair whose cost exceeds ``max_link_cost`` is rejected, so two objects that merely
touch -- high centroid distance, near-zero IoU -- never link.  **The cost of
every selected link is reported** (:class:`ZLink`), accepted or not.

Why a forced rectangular assignment, then a gate (as in
``corridor.core.metrics.match``): the solver minimises total cost, so every
genuinely cheap tube link wins its partner; only a row *forced* onto an
expensive partner is produced, and the gate then drops it.  No cheap, correct
link is ever displaced by a bad one -- the opposite failure (a valid pair
dropped for a higher-overlap one) is specific to *maximising* overlap and cannot
occur here.

**The repair.**  A slice missing between two strongly matched neighbours
(``cost < strong_link_cost`` across the one-slice gap, and the neighbours
overlap by at least ``min_bridge_iou``) is repaired by signed-distance
interpolation of the two neighbour masks: the 0.5 level set of
``0.5 (phi_a + phi_b)``.  This is a morphological midpoint of the two real
masks, drawn only between components of what the evidence says is one object, so
it can never merge two distinct objects.

**2-D data (Z = 1) is a no-op passthrough**: every component becomes its own
single-slice object, with no links, and ``passthrough`` is True.  The held-out
Corridor stills are all 2-D, so on real data this bot does exactly that; its
3-D behaviour is validated on synthetic volumes with known answers
(``tests/test_engine_zconsensus.py``), as ``docs/NEXT_GENERATION.md`` section 0
requires of every 3-D claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class ZConsensusSettings:
    """Frozen weights and gates for one Z-consensus run.

    The defaults were chosen so a straight tube (IoU ~ 1, centroid and area
    stable) links at cost ~ 0, while a sideways jump to a touching neighbour
    (IoU ~ 0) costs at least ``w_iou`` and is rejected by ``max_link_cost``.
    """

    #: Weight on ``1 - IoU`` (w1 in the contract).  The dominant term.
    w_iou: float = 1.0
    #: Weight on the normalised centroid distance (w2).
    w_centroid: float = 1.0
    #: Weight on ``|ln(area ratio)|`` (w3).  A tie-breaker.
    w_area: float = 0.5
    #: Weight on the shape-descriptor difference (w4).  A tie-breaker.
    w_shape: float = 0.5
    #: Centroid distance that costs one unit of ``w_centroid``.  Roughly an
    #: in-slice object radius: KK2 cells are ~30 px across, so 15 px.
    sigma_centroid_px: float = 15.0
    #: A selected adjacent link above this cost is rejected (objects stay split).
    max_link_cost: float = 0.9
    #: A one-slice gap is repaired only when the skip-neighbour link is at least
    #: this cheap -- stricter than ``max_link_cost`` because it fabricates a mask.
    strong_link_cost: float = 0.5
    #: ...and only when the two neighbours overlap at least this much, so the
    #: interpolated midpoint is well defined.
    min_bridge_iou: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Comp:
    """One 2-D component measured from its own slice mask (internal)."""

    slice_z: int
    label: int
    mask: np.ndarray = field(repr=False, compare=False)  # bool (Y, X)
    cx_px: float
    cy_px: float
    area_px: float
    eccentricity: float
    fill_fraction: float


@dataclass
class ZLink:
    """One slice-to-slice pairing, every cost term exposed for the audit."""

    kind: str  # "adjacent" or "repair"
    z_from: int
    label_from: int
    z_to: int
    label_to: int
    iou: float
    d_centroid_px: float
    ln_area_ratio: float
    d_shape: float
    cost: float
    accepted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZObject:
    """A 3-D object: the set of (slice, label) components judged to be one cell."""

    object_id: int
    members: list[tuple[int, int]]  # (slice_z, label), including repaired slices
    repaired_slices: list[int]

    @property
    def n_slices(self) -> int:
        return len({z for z, _ in self.members})

    @property
    def slices(self) -> list[int]:
        return sorted({z for z, _ in self.members})

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "n_slices": self.n_slices,
            "slices": self.slices,
            "members": [list(m) for m in self.members],
            "repaired_slices": sorted(self.repaired_slices),
        }


@dataclass
class ZConsensusResult:
    """The decided Z membership plus every link's evidence."""

    passthrough: bool
    n_objects: int
    objects: list[ZObject]
    links: list[ZLink]
    settings: ZConsensusSettings
    #: (Z, Y, X) int32 label volume, one id per 3-D object, repaired slices
    #: painted with their object's id.  Excluded from :meth:`to_dict`.
    object_labels_zyx: np.ndarray = field(repr=False)

    @property
    def n_links_accepted(self) -> int:
        return sum(1 for l in self.links if l.accepted)

    @property
    def n_repaired_slices(self) -> int:
        return sum(len(o.repaired_slices) for o in self.objects)

    def to_dict(self) -> dict:
        return {
            "passthrough": self.passthrough,
            "n_objects": self.n_objects,
            "n_links_total": len(self.links),
            "n_links_accepted": self.n_links_accepted,
            "n_repaired_slices": self.n_repaired_slices,
            "settings": self.settings.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "links": [l.to_dict() for l in self.links],
        }


# -- geometry helpers (deterministic, no regionprops so there is no hidden
#    pixel-variance correction and the numbers are exactly reproducible) -------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _eccentricity(ys: np.ndarray, xs: np.ndarray) -> float:
    """Eccentricity from the inertia tensor of the pixel coordinates.

    ``sqrt(1 - lambda_minor / lambda_major)`` of the 2x2 coordinate covariance;
    0.0 when the component is a point or a single line of pixels (orientation
    undefined), matching how ``detections.extract_detections`` flags that case.
    """
    if ys.size < 2:
        return 0.0
    cov = np.cov(np.vstack([ys.astype(float), xs.astype(float)]))
    eig = np.linalg.eigvalsh(cov)
    lam_major = float(max(eig[1], 0.0))
    lam_minor = float(max(eig[0], 0.0))
    if lam_major <= 0.0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - lam_minor / lam_major)))


def _measure(mask: np.ndarray, slice_z: int, label: int) -> _Comp:
    ys, xs = np.nonzero(mask)
    area = float(ys.size)
    cy = float(ys.mean()) if area else 0.0
    cx = float(xs.mean()) if area else 0.0
    if area:
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill = area / float(bbox_area) if bbox_area else 1.0
    else:
        fill = 1.0
    return _Comp(slice_z, label, mask.astype(bool), cx, cy, area,
                 _eccentricity(ys, xs), float(fill))


def _components(labels_slice: np.ndarray, slice_z: int) -> list[_Comp]:
    out: list[_Comp] = []
    for lab in (int(v) for v in np.unique(labels_slice) if v):
        out.append(_measure(labels_slice == lab, slice_z, lab))
    return out


def _pair(a: _Comp, b: _Comp, s: ZConsensusSettings) -> ZLink:
    iou = _iou(a.mask, b.mask)
    d_centroid = float(np.hypot(a.cx_px - b.cx_px, a.cy_px - b.cy_px))
    ln_area = float(abs(np.log(max(b.area_px, 1.0) / max(a.area_px, 1.0))))
    d_shape = float(abs(a.eccentricity - b.eccentricity)
                    + abs(a.fill_fraction - b.fill_fraction))
    cost = (s.w_iou * (1.0 - iou)
            + s.w_centroid * d_centroid / s.sigma_centroid_px
            + s.w_area * ln_area
            + s.w_shape * d_shape)
    return ZLink("adjacent", a.slice_z, a.label, b.slice_z, b.label,
                 round(iou, 6), round(d_centroid, 4), round(ln_area, 6),
                 round(d_shape, 6), round(float(cost), 6),
                 accepted=cost <= s.max_link_cost)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance of a boolean mask: positive inside, negative outside."""
    from scipy.ndimage import distance_transform_edt

    m = mask.astype(bool)
    outside = distance_transform_edt(~m)
    if not m.any():
        return -outside
    inside = distance_transform_edt(m)
    return inside - outside


def _interpolate(mask_a: np.ndarray, mask_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Morphological midpoint: the ``alpha`` level set of the blended signed fields."""
    phi = (1.0 - alpha) * _signed_distance(mask_a) + alpha * _signed_distance(mask_b)
    return phi >= 0.0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def link_z(labels_zyx: np.ndarray,
           settings: ZConsensusSettings | None = None) -> ZConsensusResult:
    """Link per-slice 2-D components into 3-D objects; decide only Z membership.

    ``labels_zyx`` is an int label image per Z slice (shape ``(Z, Y, X)``, or a
    single ``(Y, X)`` slice).  Labels are independent per slice -- distinct
    positive integers are distinct components -- which is exactly the ambiguity
    this bot resolves.  Returns a :class:`ZConsensusResult` whose
    ``object_labels_zyx`` gives every voxel its object id and whose ``links``
    expose the cost of every decision.
    """
    s = settings or ZConsensusSettings()
    arr = np.asarray(labels_zyx)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"link_z needs (Z, Y, X) or (Y, X) labels, got {arr.shape}")
    depth, h, w = arr.shape

    comps: list[list[_Comp]] = [_components(arr[z], z) for z in range(depth)]
    links: list[ZLink] = []
    uf = _UnionFind()
    for z in range(depth):
        for c in comps[z]:
            uf.find((c.slice_z, c.label))

    passthrough = depth == 1

    # -- adjacent-slice assignment ------------------------------------------
    if not passthrough:
        from scipy.optimize import linear_sum_assignment

        for z in range(depth - 1):
            a_list, b_list = comps[z], comps[z + 1]
            if not a_list or not b_list:
                continue
            cost = np.zeros((len(a_list), len(b_list)))
            cand = [[None] * len(b_list) for _ in a_list]
            for i, a in enumerate(a_list):
                for j, b in enumerate(b_list):
                    link = _pair(a, b, s)
                    cand[i][j] = link
                    cost[i, j] = link.cost
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                link = cand[r][c]
                links.append(link)
                if link.accepted:
                    uf.union((a_list[r].slice_z, a_list[r].label),
                             (b_list[c].slice_z, b_list[c].label))

    # -- object membership from the accepted links --------------------------
    def members_of(root) -> set:
        return {k for k in uf.parent if uf.find(k) == root}

    def object_has_member_at(root, z: int) -> bool:
        return any(zz == z for zz, _ in members_of(root))

    # -- one-slice gap repair ------------------------------------------------
    synth_masks: dict[tuple[int, int], np.ndarray] = {}
    repaired_by_root: dict = {}
    if not passthrough:
        comp_lookup = {(c.slice_z, c.label): c for z in range(depth) for c in comps[z]}
        for z in range(depth - 2):
            mid = z + 1
            if comps[mid]:
                # the middle slice is occupied; a true gap needs it empty of the
                # candidate objects, which the per-pair check below enforces, but
                # if nothing sits there at all we still allow a bridge.
                pass
            for a in comps[z]:
                root_a = uf.find((a.slice_z, a.label))
                if object_has_member_at(root_a, mid):
                    continue
                for b in comps[z + 2]:
                    root_b = uf.find((b.slice_z, b.label))
                    if object_has_member_at(root_b, mid):
                        continue
                    probe = _pair(a, b, s)
                    iou = probe.iou
                    if probe.cost >= s.strong_link_cost or iou < s.min_bridge_iou:
                        continue
                    synth = _interpolate(a.mask, b.mask, 0.5)
                    if not synth.any():
                        continue
                    synth_label = max((int(v) for v in np.unique(arr[mid]) if v),
                                      default=0) + 1 + len(
                        [k for k in synth_masks if k[0] == mid])
                    key = (mid, synth_label)
                    synth_masks[key] = synth
                    uf.find(key)
                    uf.union((a.slice_z, a.label), key)
                    uf.union(key, (b.slice_z, b.label))
                    repaired_by_root.setdefault(uf.find(key), set()).add(mid)
                    links.append(ZLink(
                        "repair", a.slice_z, a.label, b.slice_z, b.label,
                        iou, probe.d_centroid_px, probe.ln_area_ratio,
                        probe.d_shape, probe.cost, accepted=True))

    # -- assemble objects and paint the label volume -----------------------
    roots: dict = {}
    for key in uf.parent:
        roots.setdefault(uf.find(key), []).append(key)

    object_labels = np.zeros((depth, h, w), dtype=np.int32)
    objects: list[ZObject] = []
    for oid, (root, keys) in enumerate(sorted(roots.items()), start=1):
        repaired = sorted(repaired_by_root.get(root, set()))
        objects.append(ZObject(oid, sorted(keys), repaired))
        for (zz, lab) in keys:
            if (zz, lab) in synth_masks:
                object_labels[zz][synth_masks[(zz, lab)]] = oid
            else:
                object_labels[zz][arr[zz] == lab] = oid

    objects.sort(key=lambda o: (-o.n_slices, o.object_id))
    for new_id, o in enumerate(objects, start=1):
        if o.object_id != new_id:
            object_labels[object_labels == o.object_id] = -new_id
            o.object_id = new_id
    object_labels[object_labels < 0] *= -1

    return ZConsensusResult(
        passthrough=passthrough,
        n_objects=len(objects),
        objects=objects,
        links=links,
        settings=s,
        object_labels_zyx=object_labels,
    )
