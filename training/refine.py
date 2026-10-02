"""Per-instance boundary refinement of a predicted label image, two ways.

The target is the "carved wrong" errors: of the 64 round-1 errors on KK2, 17
were a cell found with a poor outline (best IoU 0.25-0.5, median 0.431), and
moving a boundary by a few pixels is exactly what would turn those into
matches. Both methods here are evaluated *beside* the unmodified prediction,
never instead of it (``training/evaluate_segmentation.py --refine``).

**Invariants, whatever the method** (tests pin each one):

- An instance is never created, deleted, merged or relabelled: the output has
  exactly the input's label ids, and an instance whose refinement comes out
  empty keeps its original mask.
- A boundary moves by at most ``max_px`` either way: the refined mask lies
  inside the prior dilated by ``max_px`` and contains the prior eroded by it.
- A pixel owned by another instance in the input is never taken, and a
  background pixel claimed by two refined instances goes to neither.
- Each refined instance is one connected component (unless the prediction
  was not, in which case it is left as predicted).
- A refinement that cannot meet all of these is declined for that instance,
  never forced.

Methods:

- ``image`` -- :func:`corridor.learn.reconstruct.refine_to_image`: re-threshold
  the +-``max_px`` band at the midpoint between the instance's interior and its
  surroundings. Model-free and cheap.
- ``contour`` -- a scikit-image morphological active contour initialised from
  the predicted mask in a padded bounding-box crop: ``geodesic`` (edge-driven,
  inverse Gaussian gradient, no balloon force) or ``chan_vese`` (region-mean
  driven). Few iterations, then clamped to the displacement band above.
"""

from __future__ import annotations

import numpy as np

import training  # noqa: F401  (puts src/ on the path)

#: Same cap as corridor.learn.reconstruct.MAX_REFINE_PX: cells here are 9-15 px
#: wide, so 3 px is a correction and 10 would be a different object.
MAX_REFINE_PX = 3
#: Context around each instance's bounding box. Must exceed max_px plus the
#: ring the methods compare against.
PAD_PX = 12
CONTOUR_ITERATIONS = 10
METHODS = ("image", "contour")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label as cc_label

    labelled, count = cc_label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def clamp_to_band(refined: np.ndarray, prior: np.ndarray, max_px: int) -> np.ndarray:
    """Keep a refinement within ``max_px`` of the prior boundary, both ways."""
    from scipy.ndimage import binary_dilation, binary_erosion

    outer = binary_dilation(prior, iterations=max_px)
    inner = binary_erosion(prior, iterations=max_px)
    return (refined & outer) | inner


def _accept(candidate: np.ndarray, prior: np.ndarray, max_px: int) -> np.ndarray:
    """The candidate if it keeps every invariant, else the prior unchanged.

    Taking the largest component could drop part of the eroded core when a
    refinement pinches a thin cell in two, which would move a boundary further
    than ``max_px``. Rather than repair that case, the instance is left as
    predicted: a refinement may only ever be declined, never forced.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    if not candidate.any():
        return prior
    candidate = _largest_component(candidate)
    inner = binary_erosion(prior, iterations=max_px)
    outer = binary_dilation(prior, iterations=max_px)
    if (inner & ~candidate).any() or (candidate & ~outer).any():
        return prior
    return candidate


def _crop_box(region: np.ndarray, pad: int):
    ys, xs = np.nonzero(region)
    h, w = region.shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def _refine_image(crop: np.ndarray, prior: np.ndarray, max_px: int) -> np.ndarray:
    from corridor.learn.reconstruct import refine_to_image

    return refine_to_image(crop, prior, max_px=float(max_px))


def _refine_contour(crop: np.ndarray, prior: np.ndarray, max_px: int, *,
                    variant: str = "geodesic",
                    iterations: int = CONTOUR_ITERATIONS) -> np.ndarray:
    from skimage.segmentation import (
        inverse_gaussian_gradient,
        morphological_chan_vese,
        morphological_geodesic_active_contour,
    )

    lo, hi = np.percentile(crop, [1, 99])
    norm = np.clip((crop.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    init = prior.astype(np.int8)
    if variant == "chan_vese":
        result = morphological_chan_vese(norm, iterations, init_level_set=init, smoothing=1)
    elif variant == "geodesic":
        gimage = inverse_gaussian_gradient(norm, alpha=100.0, sigma=1.5)
        result = morphological_geodesic_active_contour(
            gimage, iterations, init_level_set=init, smoothing=1, balloon=0)
    else:
        raise ValueError(f"unknown contour variant {variant!r}")
    return np.asarray(result, dtype=bool)


def refine_labels(image: np.ndarray, labels: np.ndarray, *, method: str = "image",
                  max_px: int = MAX_REFINE_PX, pad_px: int = PAD_PX,
                  contour_variant: str = "geodesic",
                  iterations: int = CONTOUR_ITERATIONS) -> np.ndarray:
    """Refine every instance of ``labels`` against ``image``; see the invariants above."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    image = np.asarray(image, dtype=np.float32)
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    claims = np.zeros(labels.shape, dtype=np.int32)
    refined_masks: dict[int, tuple[tuple, np.ndarray]] = {}

    for label in [int(v) for v in np.unique(labels) if v]:
        region = labels == label
        y0, y1, x0, x1 = _crop_box(region, pad_px)
        crop, prior = image[y0:y1, x0:x1], region[y0:y1, x0:x1]
        if method == "image":
            refined = _refine_image(crop, prior, max_px)
        else:
            refined = _refine_contour(crop, prior, max_px, variant=contour_variant,
                                      iterations=iterations)
        refined = clamp_to_band(np.asarray(refined, bool), prior, max_px)
        # Never take a pixel another instance owned in the input.
        refined &= (labels[y0:y1, x0:x1] == 0) | prior
        refined = _accept(refined, prior, max_px)
        refined_masks[label] = ((y0, y1, x0, x1), refined)
        claims[y0:y1, x0:x1] += refined.astype(np.int32)

    for label, ((y0, y1, x0, x1), refined) in refined_masks.items():
        own = labels[y0:y1, x0:x1] == label
        # Background contested by two refinements goes to neither; an instance's
        # own original pixels are never contested away from it.
        keep = _accept(refined & ((claims[y0:y1, x0:x1] == 1) | own), own, max_px)
        window = out[y0:y1, x0:x1]
        window[keep] = label
    return out


def displacement_px(before: np.ndarray, after: np.ndarray) -> float:
    """Largest distance any boundary pixel moved, by symmetric distance transform."""
    from scipy.ndimage import distance_transform_edt

    if not before.any() or not after.any():
        return 0.0
    gained = after & ~before
    lost = before & ~after
    worst = 0.0
    if gained.any():
        worst = max(worst, float(distance_transform_edt(~before)[gained].max()))
    if lost.any():
        worst = max(worst, float(distance_transform_edt(before)[lost].max()))
    return worst
