"""Building a reference more accurate than the single-frame human tracing.

The argument in one line: **a human tracing frame t sees frame t; a cell is one
object persisting through time.** Its outline at t is over-determined by its
outline at t-1 and t+1 together with smooth motion and a near-conserved area, so
a mask sequence solved jointly can be better than any frame traced alone -- not
because the algorithm is cleverer than the annotator, but because it is given
evidence the annotator never had.

Two things make this honest rather than wishful.

**It is not circular.** The supervising constraints are time, motion and image
gradient. None of them is the segmentation model's opinion, so a model trained
on the result is not marking its own homework. The same principle is already
measured one level down, at positions rather than outlines: track-guided
recovery fills 10 of 13 known holes with zero false positives purely by
interpolating between the observations either side of a gap.

**It is falsifiable.** :func:`boundary_sharpness` scores any mask -- human,
model or reconstructed -- by how well its boundary sits on the image's own
intensity gradient, which is a property of the photons and of nobody's opinion.
A boundary drawn in the right place runs along the edge of the cell, where the
gradient is steep. One drawn a few pixels out runs through flat background. That
gives an arbiter neither the annotator nor the network can argue with, and it is
perfectly capable of saying the reconstruction is worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How far, in pixels, a refinement may move a boundary from where the temporal
#: prior put it. Cells here are 9-15 px wide, so a couple of pixels is a
#: correction and ten would be a different object.
MAX_REFINE_PX = 3.0
#: Minimum overlap for two masks in consecutive frames to be the same cell.
LINK_IOU = 0.15


def boundary_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """How well a mask's boundary sits on the image's own intensity gradient.

    The arbiter. Every other measure in this project compares one opinion with
    another -- the model against the annotator, a reconstruction against a
    label. This compares a mask against the photons.

    A boundary in the right place runs along the cell's edge, where intensity
    changes fastest. A boundary a few pixels off runs through flat background or
    flat cytoplasm, where it does not. The score is the mean gradient magnitude
    on the boundary, normalised by the mean gradient over the whole
    neighbourhood so that a bright frame does not simply outscore a dim one.

    Returns 0.0 when the mask is empty, and is deliberately capable of ranking a
    reconstruction *below* a human tracing.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, sobel

    region = mask > 0
    if not region.any():
        return 0.0

    gy = sobel(image.astype(np.float32), axis=0)
    gx = sobel(image.astype(np.float32), axis=1)
    gradient = np.hypot(gx, gy)

    edge = region & ~binary_erosion(region)
    if not edge.any():
        return 0.0

    # Compare against the neighbourhood, not the whole frame: a cell sitting in
    # a busy corner of the device should not be rewarded for its surroundings.
    around = binary_dilation(region, iterations=6)
    local = gradient[around]
    scale = float(local.mean()) if local.size else 0.0
    if scale <= 1e-9:
        return 0.0
    return float(gradient[edge].mean() / scale)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class CellTrack:
    """One cell followed through a sequence of labelled frames."""

    #: frame index -> the label id of this cell in that frame's mask image
    members: dict[int, int] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return sorted(self.members)


def link_masks(masks: list[np.ndarray]) -> list[CellTrack]:
    """Follow each cell through the sequence by mask overlap.

    Overlap is used rather than centroid distance because these cells are long
    and thin: two cells in adjacent channels can have closer centroids than one
    cell has to itself two frames later, while overlap keeps them apart.
    """
    tracks: list[CellTrack] = []
    active: dict[int, tuple[CellTrack, np.ndarray]] = {}

    for index, mask in enumerate(masks):
        labels = [int(v) for v in np.unique(mask) if v]
        regions = {label: (mask == label) for label in labels}
        claimed: set[int] = set()

        for key, (track, previous) in list(active.items()):
            best, best_iou = None, LINK_IOU
            for label, region in regions.items():
                if label in claimed:
                    continue
                score = _iou(previous, region)
                if score > best_iou:
                    best, best_iou = label, score
            if best is not None:
                track.members[index] = best
                claimed.add(best)
                active[key] = (track, regions[best])
            else:
                del active[key]

        for label, region in regions.items():
            if label in claimed:
                continue
            track = CellTrack(members={index: label})
            tracks.append(track)
            active[max(active) + 1 if active else 0] = (track, region)

    return tracks


def _shift(region: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(region)
    h, w = region.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = region[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _centre_of(region: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(region)
    return float(ys.mean()), float(xs.mean())


def temporal_consensus(
    regions: dict[int, np.ndarray], target: int, *, span: int = 1
) -> np.ndarray | None:
    """What the neighbouring frames say this cell's shape is, moved into place.

    Each neighbour's mask is translated so that its centroid lands on where this
    cell's centroid is (or, for a frame with no mask of its own, is predicted to
    be), and the translated shapes are combined by majority vote. Translation
    alone is the right model here: the cells are confined to a straight channel
    and move along it, so between adjacent frames they shift far more than they
    deform.

    Returns None when there are not enough neighbours to form an opinion.
    """
    frames = sorted(regions)
    neighbours = [f for f in frames if f != target and abs(f - target) <= span]
    if len(neighbours) < 2:
        return None

    if target in regions:
        anchor = _centre_of(regions[target])
    else:
        # No mask here: put the shape where interpolation says the cell was.
        before = [f for f in neighbours if f < target]
        after = [f for f in neighbours if f > target]
        if not before or not after:
            return None
        b, a = max(before), min(after)
        cb, ca = _centre_of(regions[b]), _centre_of(regions[a])
        weight = (target - b) / (a - b)
        anchor = (cb[0] + (ca[0] - cb[0]) * weight, cb[1] + (ca[1] - cb[1]) * weight)

    votes = np.zeros(next(iter(regions.values())).shape, dtype=np.float32)
    for frame in neighbours:
        region = regions[frame]
        cy, cx = _centre_of(region)
        votes += _shift(region, int(round(anchor[0] - cy)), int(round(anchor[1] - cx)))

    return votes >= (len(neighbours) / 2.0)


def refine_to_image(
    image: np.ndarray, prior: np.ndarray, *, max_px: float = MAX_REFINE_PX
) -> np.ndarray:
    """Snap a temporally-derived mask onto the image's own edge.

    The temporal consensus gets the shape right and the position approximately
    right. This lets the boundary move by at most a couple of pixels to sit
    where the intensity actually changes -- the correction the annotator makes
    by eye, made here against the gradient instead.

    The cap matters: without it this becomes an ordinary region-grower and
    throws away the temporal evidence that justified the whole approach.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    if not prior.any():
        return prior

    frame = image.astype(np.float32)
    inner = binary_erosion(prior, iterations=int(max_px))
    outer = binary_dilation(prior, iterations=int(max_px))
    band = outer & ~inner
    if not band.any():
        return prior

    # Split the uncertain band by intensity: a threshold halfway between what
    # the mask's confident interior looks like and what surrounds it.
    interior = frame[inner] if inner.any() else frame[prior]
    exterior = frame[outer & ~prior]
    if interior.size == 0 or exterior.size == 0:
        return prior
    level = 0.5 * (float(np.median(interior)) + float(np.median(exterior)))
    bright = float(np.median(interior)) > float(np.median(exterior))

    refined = inner.copy()
    keep = (frame > level) if bright else (frame < level)
    refined |= band & keep

    # Keep it one connected object, and do not let it drift beyond the cap.
    from scipy.ndimage import label as cc_label

    labelled, count = cc_label(refined)
    if count > 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        refined = labelled == int(sizes.argmax())
    distance = distance_transform_edt(~prior)
    refined &= distance <= max_px
    return refined if refined.any() else prior
