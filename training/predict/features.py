"""Three feature sets for one observation, kept apart so they cannot be confused.

1. **Strict morphology** (:func:`strict_morphology`): computed from a single
   binary mask and nothing else. No position, no velocity, no history, no
   intensity, no frame index. This is the only set a "morphology predicts
   migration" claim may rest on.
2. **Phenotype** (:func:`phenotype`): masked intensity texture. It is *not*
   morphology. In phase contrast the brightness and texture inside a cell
   depend on its thickness, focus, illumination and camera, so a result that
   needs these columns is a phenotype-or-imaging result and is reported under
   that name.
3. **Motion history** (:func:`motion_history`): previous speeds. A comparator
   only. Migration is persistent, so the past speed is the obvious predictor of
   the future speed, and the question that matters (directive section 49) is
   whether morphology adds anything *beyond* it.

Orientation is deliberately excluded from the strict set. In a confinement
device the channels fix the long axis of every cell (all 246 labelled cells in
the supplied data are vertical to within +/-20 degrees, ``build/maps/
research-data.md`` section 1), so orientation measures how the device was
mounted on the stage, not what the cell is doing. A model allowed to read it
would learn the acquisition, and Corridor 2.0 removed the migration axis from
everything for the same reason (design contract section 5). The same trap
hides in features that look innocent: ``regionprops.extent`` and the bounding
box height/width are computed in the image grid, so a tilted cell has a smaller
extent than the same cell upright. Extent here is measured in the cell's own
principal-axis frame, skeleton length is measured along a traced and smoothed
path (a pixel count would make a diagonal cell 1/sqrt(2) as long, and even a
chain code reads 8 % longer at 22.5 degrees than at 0), and the perimeter is
the sub-pixel contour of the lightly blurred mask rather than a pixel-edge
count. The tests rotate and mirror masks and check that the strict set does not
move; the experiment repeats that on every real mask and reports the residual.

Units: when the pixel size is known, lengths are in micrometres (``_um``),
areas in square micrometres (``_um2``) and curvature in 1/um (``_per_um``).
Without calibration the same features carry ``_px`` names, and
:mod:`training.predict.dataset` refuses to pool calibrated and uncalibrated
folders, because KK1 and KK2 differ by 0.639 vs 0.467 um/px and equal pixel
widths are different physical widths.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

#: Below this many pixels a mask has no meaningful contour curvature or
#: skeleton. Far below any cell in the supplied data (hundreds of pixels).
MIN_AREA_PX = 30
#: The outline is traced on the mask blurred by this Gaussian (pixels), not on
#: the raw binary mask. The marching-squares contour of a binary mask is a
#: staircase: on a digitised disc of radius 30 px it overstates the perimeter
#: enough to give circularity 0.906, against 0.990 after this blur. The noise
#: is a property of the grid, so its scale is the pixel, not the micrometre.
CONTOUR_BLUR_PX = 1.0
#: Further Gaussian smoothing along the arc length before curvature (pixels).
CONTOUR_SMOOTH_PX = 2.0
#: Contour resampling step (pixels). Fine enough that where the samples fall
#: relative to a cell tip (the start point of the traced outline) no longer
#: moves the curvature percentiles: at 1 px a rot90 of the same mask moved
#: the 95th percentile by 5 %.
CONTOUR_STEP_PX = 0.25
#: Curvature below -this (per pixel: a concavity tighter than a 20 px radius)
#: counts as an indentation. On digitised convex ellipses 11 px wide, tilted
#: 0-90 degrees, the smoothed outline dips to -0.049/px at worst (pixel
#: staircase, not shape), so a tolerance of 0.02 reported phantom concavities
#: on a convex shape and this one does not -- by a thin margin, so
#: ``concave_fraction`` is among the rotation-sensitive features (see the
#: real-mask audit in docs/prediction_experiment.json).
CONCAVE_TOLERANCE_PER_PX = 0.05
#: Hu invariants are floored here before the logarithm. Exactly symmetric
#: shapes have invariants 3-7 of exactly zero, whose log is unbounded; on 26
#: real masks (every third frame of 052924_1) the smallest magnitude was
#: 10**-10.9, so the floor bounds synthetic shapes and never clips those cells.
HU_FLOOR = 1e-12
#: Standardised crop for the learned embedding: the cell is centred, its major
#: axis turned vertical, and scaled so the major axis spans this many pixels.
CROP_SIZE_PX = 64
CROP_MAJOR_PX = 48.0

STRICT_SET = "strict_morphology"
PHENOTYPE_SET = "phenotype"
HISTORY_SET = "motion_history"

#: Names that must never appear in the strict set; checked by the tests.
FORBIDDEN_STRICT_TOKENS = ("orientation", "angle", "bbox", "centroid", "position",
                           "speed", "velocity", "intensity", "frame", "time",
                           "pheno", "hist")


def _units(pixel_size_um: float | None) -> tuple[float, str, str, str]:
    """Length scale and the suffixes that go with it."""
    if pixel_size_um is not None and pixel_size_um > 0:
        return float(pixel_size_um), "_um", "_um2", "_per_um"
    return 1.0, "_px", "_px2", "_per_px"


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The largest 8-connected piece of ``mask`` and how many pieces there were.

    A Cellpose label is almost always one piece, but a merge or a recovered
    detection can be two. Contour and skeleton need one closed outline, so the
    features describe the largest piece and the dataset records the count as a
    quality column (never as a feature).
    """
    mask = np.asarray(mask, dtype=bool)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask, int(n)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1), int(n)


def _principal_axes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Centroid (row, col), major/minor unit vectors (row, col) and lengths.

    Lengths use the definition of ``regionprops.axis_major_length``
    (4 * sqrt of the eigenvalue of the coordinate covariance).
    """
    coords = np.argwhere(mask).astype(float)
    centre = coords.mean(axis=0)
    d = coords - centre
    cov = (d.T @ d) / len(d)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    minor_vec, major_vec = evecs[:, 0], evecs[:, 1]
    major_len = 4.0 * math.sqrt(max(evals[1], 0.0))
    minor_len = 4.0 * math.sqrt(max(evals[0], 0.0))
    return centre, major_vec, minor_vec, major_len, minor_len


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Vertices of the convex hull of 2-D points, in order (Andrew's monotone chain).

    Not ``scipy.spatial.ConvexHull``: on this Windows install every Qhull call
    creates and opens a temporary file, which measured 3.5 ms per hull and made
    the hull the slowest step of the whole feature set (``skimage``'s
    ``area_convex`` calls it twice more).
    """
    # Plain Python floats: numpy scalar arithmetic in this loop was 20x slower.
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float).tolist())))
    if len(pts) < 3:
        return np.array(pts, dtype=float).reshape(-1, 2)

    def chain(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for q in seq:
            while len(out) >= 2:
                (ox, oy), (ax, ay) = out[-2], out[-1]
                if (ax - ox) * (q[1] - oy) - (ay - oy) * (q[0] - ox) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    lower = chain(pts)
    upper = chain(pts[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _closed_contour(mask: np.ndarray) -> np.ndarray:
    """Longest 0.5-level contour of the lightly blurred mask, (row, col), not repeated."""
    from skimage.measure import find_contours

    pad = 4
    soft = ndimage.gaussian_filter(np.pad(mask, pad).astype(float), CONTOUR_BLUR_PX)
    contours = find_contours(soft, 0.5)
    if not contours:
        raise ValueError("mask has no contour")
    c = max(contours, key=len) - float(pad)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    return c


def _polygon_perimeter_area(c: np.ndarray) -> tuple[float, float]:
    """Perimeter and *signed* shoelace area of a closed polygon (row, col)."""
    d = np.diff(np.vstack([c, c[:1]]), axis=0)
    perimeter = float(np.hypot(d[:, 0], d[:, 1]).sum())
    x, y = c[:, 1], c[:, 0]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return perimeter, signed


def contour_curvature(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed curvature (1/px) along a closed outline, convex = positive.

    The outline is resampled every ``CONTOUR_STEP_PX`` of arc length and
    smoothed with a periodic Gaussian before differentiating. The sign is
    normalised by the polygon's signed area, so the result does not depend on
    which way the contour tracer walked round. Returns (curvature samples,
    arc length each sample stands for *on the smoothed curve*).

    Statistics must be weighted by that arc length. Smoothing pulls the two
    sides of a thin tip together, so the smoothed curve moves slowly there and
    its samples crowd into the tip; unweighted, the 95th percentile of an
    11-px-wide ellipse moved by 3-5 % between exact 90-degree rotations of the
    same mask, purely from where the samples happened to fall.

    Measured limit: the tip of a cell 11 px wide has a radius of curvature of
    a few pixels, at the grid's resolution, so the *extreme* curvature moves by
    tens of percent with the cell's angle. The strict set therefore reports the
    5th/95th percentiles, not the minimum and maximum.
    """
    perimeter, signed = _polygon_perimeter_area(contour)
    closed = np.vstack([contour, contour[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(perimeter / CONTOUR_STEP_PX)), 8)
    step = perimeter / n
    t = np.arange(n) * step
    y = np.interp(t, s, closed[:, 0])
    x = np.interp(t, s, closed[:, 1])
    sigma = CONTOUR_SMOOTH_PX / step
    x = ndimage.gaussian_filter1d(x, sigma, mode="wrap")
    y = ndimage.gaussian_filter1d(y, sigma, mode="wrap")
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * step)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * step)
    ddx = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / step**2
    ddy = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / step**2
    speed = np.sqrt(dx * dx + dy * dy)
    kappa = (dx * ddy - dy * ddx) / np.maximum(speed**3, 1e-12)
    if signed < 0:
        kappa = -kappa
    return kappa, speed * step


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q / 100.0 * w.sum(), cum, v))


def _crossing_number(skel: np.ndarray) -> np.ndarray:
    """0->1 transitions round each pixel's 8-neighbourhood (endpoint 1, line 2, junction >=3).

    Counting neighbours instead would call every L-shaped corner of a thin
    8-connected line a junction; the crossing number does not.
    """
    p = np.pad(skel.astype(np.int8), 1)
    h, w = skel.shape
    # Clockwise from north.
    ring = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    nb = [p[r:r + h, c:c + w] for r, c in ring]
    cn = np.zeros(skel.shape, dtype=np.int16)
    for i in range(8):
        cn += (nb[i] == 0) & (nb[(i + 1) % 8] == 1)
    return np.where(skel, cn, 0)


_EIGHT = np.ones((3, 3), dtype=bool)
#: Skeleton pieces shorter than this (pixels) are not counted as branches.
SPUR_PX = 4.0


def _branch_length(pixels: np.ndarray) -> float:
    """Length of one skeleton branch, traced in order and lightly smoothed.

    Chain-code estimators are not rotation invariant: on digital straight lines
    80 px long, counting orthogonal steps as 1 and diagonal steps as sqrt(2)
    reads 80.0 at 0 degrees and 86.8 at 22.5 (Kulpa's weights trade that for
    75.8 vs 82.4). Cells in a device sit within about 20 degrees of one angle,
    so that error would be read as length. Tracing the branch and measuring a
    Gaussian-smoothed polyline follows the true path instead.
    """
    if len(pixels) < 2:
        return 0.0
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(pixels)}
    neighbours: list[list[int]] = [[] for _ in pixels]
    for i, (r, c) in enumerate(pixels):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = index.get((int(r) + dr, int(c) + dc))
                if j is not None and j != i:
                    neighbours[i].append(j)
    ends = [i for i, nb in enumerate(neighbours) if len(nb) == 1]
    start = ends[0] if ends else 0
    order = [start]
    seen = {start}
    current = start
    while True:
        nxt = [j for j in neighbours[current] if j not in seen]
        if not nxt:
            break
        # Prefer the orthogonal neighbour so an L-corner is walked, not cut.
        nxt.sort(key=lambda j: abs(pixels[j][0] - pixels[current][0])
                 + abs(pixels[j][1] - pixels[current][1]))
        current = nxt[0]
        order.append(current)
        seen.add(current)
    path = pixels[order].astype(float)
    if len(path) >= 5:
        path = ndimage.gaussian_filter1d(path, 1.5, axis=0, mode="nearest")
        path[0], path[-1] = pixels[order[0]], pixels[order[-1]]
    return float(np.hypot(*np.diff(path, axis=0).T).sum())


def skeleton_stats(mask: np.ndarray) -> tuple[float, int, int, int]:
    """(length_px, branches, endpoints, junctions) of the mask's skeleton.

    Branches are the pieces left after cutting out each junction's hub (a
    straight cell is 1, a Y is 3, a closed loop 1); their lengths are summed and
    each piece end beside a hub is joined to the junction's centre.
    The skeleton stops short of the tips by about half the cell's width; that
    bias is the same for every cell of a given width and is left in.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(np.pad(mask, 1).astype(bool))
    n_px = int(skel.sum())
    if n_px == 0:
        return 0.0, 0, 0, 0
    if n_px == 1:
        return 0.0, 1, 0, 0
    cn = _crossing_number(skel)
    endpoints = int((cn == 1).sum())
    junction = cn >= 3
    j_lab, n_junctions = ndimage.label(junction, structure=_EIGHT)
    # Removing only the junction pixel leaves its arms touching diagonally, so
    # the hub (junction plus its skeleton neighbours) is what is cut out.
    hub = ndimage.binary_dilation(junction, structure=_EIGHT) & skel
    b_lab, n_pieces = ndimage.label(skel & ~hub, structure=_EIGHT)
    j_centres = (np.array(ndimage.center_of_mass(junction, j_lab, range(1, n_junctions + 1)))
                 if n_junctions else np.zeros((0, 2)))
    near_hub = ndimage.binary_dilation(hub, structure=_EIGHT)
    length = 0.0
    branches = 0
    for b in range(1, n_pieces + 1):
        px = np.argwhere(b_lab == b)
        piece = _branch_length(px)
        if n_junctions:
            for r, c in px[near_hub[px[:, 0], px[:, 1]]]:
                piece += float(np.hypot(j_centres[:, 0] - r, j_centres[:, 1] - c).min())
        length += piece
        # A spur shorter than SPUR_PX is skeletonisation noise on a ragged
        # outline, not an arm; it adds its length but is not counted.
        if piece >= SPUR_PX or n_pieces == 1:
            branches += 1
    return float(length), int(max(branches, 1)), endpoints, int(n_junctions)


def _log_hu(mask: np.ndarray) -> list[float]:
    """The seven Hu invariants, log-scaled as -sign(h) * log10|h|.

    Hu 7 is odd under reflection: a mirrored cell flips its sign. Mirroring is
    a label-preserving augmentation in this project (design contract section 9),
    so only its magnitude is kept. Magnitudes are floored at ``HU_FLOOR``.
    """
    from skimage.measure import moments_central, moments_hu, moments_normalized

    m = mask.astype(float)
    hu = moments_hu(moments_normalized(moments_central(m, order=3), order=3))
    out = []
    for i, h in enumerate(hu):
        mag = max(abs(float(h)), HU_FLOOR)
        sign = 1.0 if (i == 6 or h >= 0) else -1.0
        out.append(-sign * math.log10(mag))
    return out


def strict_morphology(mask: np.ndarray, pixel_size_um: float | None = None) -> dict[str, float]:
    """Strict morphology of one binary mask (a crop containing one object).

    Raises ``ValueError`` for masks too small to have a shape. Everything is
    measured on the largest connected piece (see :func:`largest_component`).
    """
    m, _ = largest_component(mask)
    area_px = int(m.sum())
    if area_px < MIN_AREA_PX:
        raise ValueError(f"mask has {area_px} px; at least {MIN_AREA_PX} are needed")
    m = np.pad(m, 2)
    scale, lu, au, cu = _units(pixel_size_um)

    _, major_vec, minor_vec, major_px, minor_px = _principal_axes(m)
    contour = _closed_contour(m)
    perimeter_px, signed_area = _polygon_perimeter_area(contour)
    polygon_area_px = abs(signed_area)
    hull = convex_hull(contour)
    convex_perimeter_px, hull_signed = _polygon_perimeter_area(hull)
    convex_area_px = abs(hull_signed)
    # Extent in the cell's own principal-axis frame (see the module docstring).
    principal_box_px = np.ptp(contour @ major_vec) * np.ptp(contour @ minor_vec)
    kappa, ds = contour_curvature(contour)
    arc = float(ds.sum())
    skel_len_px, branches, endpoints, junctions = skeleton_stats(m)

    features: dict[str, float] = {
        f"area{au}": area_px * scale**2,
        f"perimeter{lu}": perimeter_px * scale,
        f"convex_area{au}": convex_area_px * scale**2,
        f"convex_perimeter{lu}": convex_perimeter_px * scale,
        f"major_axis{lu}": major_px * scale,
        f"minor_axis{lu}": minor_px * scale,
        f"equivalent_diameter{lu}": math.sqrt(4.0 * area_px / math.pi) * scale,
        "aspect_ratio": major_px / max(minor_px, 1e-6),
        "eccentricity": math.sqrt(max(0.0, 1.0 - (minor_px / max(major_px, 1e-9)) ** 2)),
        # Outline area over hull area, both from the same sub-pixel outline, so
        # a convex shape scores 1 whatever its angle to the grid.
        "solidity": polygon_area_px / max(convex_area_px, 1e-9),
        "extent_principal": polygon_area_px / max(principal_box_px, 1e-9),
        # Polygon area with polygon perimeter, so a disc scores ~1 at any size.
        "circularity": 4.0 * math.pi * polygon_area_px / max(perimeter_px**2, 1e-9),
        "roughness": perimeter_px / max(convex_perimeter_px, 1e-9),
        # All curvature statistics are per unit arc length (see contour_curvature).
        f"curvature_mean_abs{cu}": float(np.sum(np.abs(kappa) * ds) / arc) / scale,
        f"curvature_std{cu}": math.sqrt(float(np.sum(kappa**2 * ds) / arc
                                              - (np.sum(kappa * ds) / arc) ** 2)) / scale,
        f"curvature_p95{cu}": _weighted_percentile(kappa, ds, 95) / scale,
        f"curvature_p05{cu}": _weighted_percentile(kappa, ds, 5) / scale,
        "concave_fraction": float(np.sum(ds[kappa < -CONCAVE_TOLERANCE_PER_PX]) / arc),
        # Dimensionless: (closed integral of kappa^2 ds) * L / (2 pi)^2, = 1 for a circle.
        "bending_energy": float(np.sum(kappa**2 * ds)) * arc / (4.0 * math.pi**2),
        f"skeleton_length{lu}": skel_len_px * scale,
        "skeleton_branches": float(branches),
        "skeleton_endpoints": float(endpoints),
        "skeleton_junctions": float(junctions),
    }
    for i, v in enumerate(_log_hu(m), start=1):
        features[f"hu{i}_log"] = v
    return features


def strict_feature_names(pixel_size_um: float | None) -> list[str]:
    """Column order of :func:`strict_morphology` (computed from a reference disc)."""
    yy, xx = np.mgrid[-12:13, -12:13]
    return list(strict_morphology((xx**2 + yy**2) <= 100, pixel_size_um).keys())


# ---------------------------------------------------------------------------
# Phenotype: masked intensity texture. NOT morphology.

#: Background ring between these dilations of the mask, the same definition
#: ``scripts/diag_errors.py`` uses for contrast.
RING_INNER_PX = 3
RING_OUTER_PX = 10
GLCM_LEVELS = 16


def phenotype(intensity: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Intensity texture inside the mask, relative to the local background ring.

    ``intensity`` and ``mask`` are the same crop, padded by at least
    ``RING_OUTER_PX`` so the ring fits. Everything is expressed relative to the
    ring's median and robust spread, so a brighter lamp or a longer exposure
    does not masquerade as a different cell; that removes gain and offset but
    not focus or thickness, which is exactly why this set is not morphology.
    GLCM statistics are averaged over four directions so they carry no
    orientation.
    """
    from skimage.feature import graycomatrix, graycoprops

    img = np.asarray(intensity, dtype=float)
    m = np.asarray(mask, dtype=bool)
    inner = ndimage.binary_dilation(m, iterations=RING_INNER_PX)
    outer = ndimage.binary_dilation(m, iterations=RING_OUTER_PX)
    ring = outer & ~inner
    inside = img[m]
    bg = img[ring] if ring.any() else img[~m]
    bg_med = float(np.median(bg))
    bg_scale = float(1.4826 * np.median(np.abs(bg - bg_med)))
    if bg_scale <= 0:
        bg_scale = float(np.std(bg)) or 1.0
    z = (inside - bg_med) / bg_scale
    mean_z = float(np.mean(z))
    std_z = float(np.std(z))
    centred = z - mean_z
    skew = float(np.mean(centred**3) / max(std_z**3, 1e-12))
    kurt = float(np.mean(centred**4) / max(std_z**4, 1e-12) - 3.0)
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy) / bg_scale
    edge = m & ~ndimage.binary_erosion(m)
    lo, hi = np.percentile(inside, [1, 99])
    q = np.zeros(img.shape, dtype=np.uint8)
    if hi > lo:
        q[m] = 1 + np.clip(((img[m] - lo) / (hi - lo) * GLCM_LEVELS).astype(int), 0, GLCM_LEVELS - 1)
    else:
        q[m] = 1
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS + 1, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(float)  # drop "outside the mask"
    total = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = glcm / np.maximum(total, 1.0)
    props = {p: float(np.mean(graycoprops(glcm, p))) for p in
             ("contrast", "homogeneity", "energy", "correlation")}
    return {
        "pheno_contrast": mean_z,
        "pheno_std": std_z,
        "pheno_skew": skew,
        "pheno_kurtosis": kurt,
        "pheno_inside_gradient": float(np.mean(grad[m])),
        "pheno_edge_gradient": float(np.mean(grad[edge])) if edge.any() else 0.0,
        "pheno_glcm_contrast": props["contrast"],
        "pheno_glcm_homogeneity": props["homogeneity"],
        "pheno_glcm_energy": props["energy"],
        "pheno_glcm_correlation": props["correlation"] if np.isfinite(props["correlation"]) else 0.0,
    }


# ---------------------------------------------------------------------------
# Motion history: a COMPARATOR, never a morphology result.

#: How many past steps the history features look back over.
HISTORY_STEPS = 3


def motion_history(past_xy_um: np.ndarray, past_t_min: np.ndarray) -> dict[str, float]:
    """Previous-speed features from this track's own past, ending at time t.

    ``past_xy_um`` is (k+1, 2) positions in micrometres at times ``past_t_min``
    (ascending, last row = the observation being predicted from), k >= 1. Each
    step's speed is its distance over its *elapsed* time, so a gap is a longer
    step, not a faster one. Speeds are um/hr.
    """
    xy = np.asarray(past_xy_um, dtype=float)[-(HISTORY_STEPS + 1):]
    t = np.asarray(past_t_min, dtype=float)[-(HISTORY_STEPS + 1):]
    if len(xy) < 2:
        raise ValueError("motion history needs at least one previous observation")
    step = np.hypot(*np.diff(xy, axis=0).T)
    dt_hr = np.diff(t) / 60.0
    if np.any(dt_hr <= 0):
        raise ValueError("observation times must increase")
    speeds = step / dt_hr
    net = float(np.hypot(*(xy[-1] - xy[0]))) / float((t[-1] - t[0]) / 60.0)
    return {
        "hist_speed_last_um_per_hr": float(speeds[-1]),
        "hist_speed_mean_um_per_hr": float(np.mean(speeds)),
        "hist_net_rate_um_per_hr": net,
    }


# ---------------------------------------------------------------------------
# Standardised crop for the learned embedding (derived from the mask only).

def standardised_crop(mask: np.ndarray, size: int = CROP_SIZE_PX,
                      major_px: float = CROP_MAJOR_PX) -> np.ndarray:
    """The mask resampled into a ``size`` x ``size`` frame of its own.

    Centred on the centroid, the major axis turned vertical, scaled so the
    major axis spans ``major_px`` pixels, then flipped so the third moments
    along and across the axis are non-negative. Every one of those steps removes
    something the strict set must not see: position, orientation, the arbitrary
    head/tail sign of an axis, and mirror image. Absolute size is removed too
    (it is in the handcrafted set already), so the embedding can only learn
    *shape*. Returns float32 in {0, 1}.
    """
    m, _ = largest_component(mask)
    m = np.pad(m, 2)
    centre, major_vec, minor_vec, major_len, _ = _principal_axes(m)
    s = major_px / max(major_len, 1e-6)
    half = (size - 1) / 2.0
    v, u = np.mgrid[0:size, 0:size].astype(float)
    v -= half
    u -= half
    rows = centre[0] + (v / s) * major_vec[0] + (u / s) * minor_vec[0]
    cols = centre[1] + (v / s) * major_vec[1] + (u / s) * minor_vec[1]
    out = ndimage.map_coordinates(m.astype(np.float32), [rows, cols], order=1, cval=0.0) >= 0.5
    if out.any():
        rr, cc = np.nonzero(out)
        if np.mean((rr - rr.mean()) ** 3) < 0:
            out = out[::-1, :]
        if np.mean((cc - cc.mean()) ** 3) < 0:
            out = out[:, ::-1]
    return np.ascontiguousarray(out, dtype=np.float32)
