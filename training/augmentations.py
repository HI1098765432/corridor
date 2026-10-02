"""Recorded, orientation-free augmentations for training only.

**Recorded.** A policy is a dataclass that serialises to JSON, and every copy it
makes carries a record of every sampled parameter (``AugmentationRecord``), so a
training report can say exactly what the network saw and
:func:`replay` can rebuild any copy bit for bit from the source and its record.

**Orientation-free.** Nothing here knows which way the channels run. Flips and
90-degree rotations are drawn uniformly, and every directional effect
(anisotropic blur, illumination gradient) takes a uniformly random angle. The
contrast transform estimates its local background with an isotropic Gaussian.
The old ``median_filter(size=(1, 41))`` in ``scripts/train_contrast_invariant.py``
was a horizontal window that, with every labelled cell vertical, ran *across*
the channels -- it worked only because cells are ~11 px wide inside 41 px.

**Label-free photometrics.** Photometric operations never read or change the
labels, so they cannot move a boundary (tests pin this). That includes contrast:
the old transform rescaled the image only near labelled cells (a dilated,
blurred mask support), which paints a halo exactly where the labels are -- a
cue the network can learn instead of the cell. Here the whole field is rescaled
about its local background.

**Geometric operations** (flips, rotations, scale) transform image and labels
together: images by linear interpolation, labels by nearest neighbour so no
instance id is ever invented.

Operations, in the order applied: flip, rot90, scale | contrast, gain/offset,
gamma, illumination gradient, vignetting, defocus blur, anisotropic blur,
downsample-and-reinterpolate, Poisson shot noise, Gaussian read noise.

The default probabilities and ranges are starting points, not measured optima;
the ranges for contrast (0.35-2.6x) and scale (0.75-1.35x) are the ones the
contrast-invariant runs used, and 1.35x is also the KK1/KK2 pixel-size ratio
(0.639/0.467 = 1.37). Degradation strengths are kept inside what a 9-15 px wide
cell survives: downsampling at most 3x, defocus sigma at most 2 px.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Draw:
    """One sampled range: uniform, or log-uniform for multiplicative factors."""

    lo: float
    hi: float
    log: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))
        return float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class AugmentationPolicy:
    name: str = "orientation_free_v1"
    seed: int = 0
    copies_per_image: int = 2
    #: Empty frames are kept once, unaugmented, as negatives (a photometric copy
    #: of a frame with no cell teaches nothing new about cells).
    augment_empty_frames: bool = False
    p_flip: float = 0.5  # per axis
    rotate90: bool = True  # k uniform in 0..3
    p_scale: float = 0.5
    scale: Draw = Draw(0.75, 1.35, log=True)
    p_contrast: float = 0.8
    contrast: Draw = Draw(0.35, 2.6, log=True)
    #: Sigma of the isotropic Gaussian local background, px. Larger than a cell
    #: is wide (9-15 px), smaller than the field, and the same in every direction.
    background_sigma_px: float = 20.0
    p_gain: float = 0.5
    gain: Draw = Draw(0.7, 1.4, log=True)
    #: Offset as a fraction of the p1-p99 range.
    offset: Draw = Draw(-0.05, 0.05)
    p_gamma: float = 0.3
    gamma: Draw = Draw(0.7, 1.4, log=True)
    p_gradient: float = 0.3
    #: Peak-to-centre multiplicative change across the field.
    gradient: Draw = Draw(0.0, 0.3)
    p_vignette: float = 0.3
    vignette: Draw = Draw(0.0, 0.3)
    p_defocus: float = 0.3
    defocus_sigma_px: Draw = Draw(0.5, 2.0)
    p_anisotropic: float = 0.2
    anisotropic_sigma_px: Draw = Draw(1.0, 3.0)
    #: Across-direction sigma as a fraction of the along-direction one.
    anisotropic_ratio: Draw = Draw(0.1, 0.5)
    p_downsample: float = 0.2
    downsample_factors: tuple[int, ...] = (2, 3)
    p_poisson: float = 0.5
    #: Photons at the p99 level; fewer is noisier.
    poisson_photons: Draw = Draw(20.0, 500.0, log=True)
    p_read_noise: float = 0.5
    #: Read-noise sigma as a fraction of the p1-p99 range.
    read_noise: Draw = Draw(0.005, 0.04, log=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AugmentationPolicy":
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict) and set(value) <= {"lo", "hi", "log"}:
                kwargs[key] = Draw(**value)
            elif key == "downsample_factors":
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


@dataclass
class AugmentationRecord:
    source: str
    copy: int
    seed: list[int]
    ops: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Operations. Geometric ones take and return (image, labels); photometric ones
# take and return the image only, which is how "never moves a boundary" is
# enforced by construction rather than by care.


def _cv2():
    """OpenCV with its own thread pool switched off, before its first call here.

    Under memory pressure beside a training run, OpenCV's worker pool has
    crashed inside ``cv2.resize`` with Windows fatal exception 0xc000070a, after
    which every cv2 call in the process raised "Unknown C++ exception": the
    intermittent failures of the augmentation tests. Training sets are built
    before the trainers configure Cellpose, so the switch lives here.
    ``setNumThreads(0)`` runs OpenCV's loops on the calling thread; these images
    are small enough that nothing measurable is lost.
    """
    import cv2

    if cv2.getNumThreads() != 0:
        cv2.setNumThreads(0)
    return cv2


def _range_of(image: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(image, [1, 99])
    return float(lo), max(float(hi - lo), 1e-6)


def flip(image, labels, *, axis: int):
    return np.flip(image, axis=axis).copy(), np.flip(labels, axis=axis).copy()


def rot90(image, labels, *, k: int):
    return np.rot90(image, k).copy(), np.rot90(labels, k).copy()


def rescale(image, labels, *, factor: float):
    """Resize image (linear) and labels (nearest) onto the same pixel grid.

    ``INTER_NEAREST_EXACT``, not ``INTER_NEAREST``: OpenCV's plain nearest mode
    samples without the half-pixel centre offset that ``INTER_LINEAR`` uses, so
    labels land shifted against the image, always towards larger x and y. How
    far depends on where a cell sits on the pixel grid. Over cells 9-12 px wide
    at every sub-pixel phase (``test_nearest_exact_removes_most_of_the_label_
    shift``): plain nearest shifts them +0.38 px on average at 0.75x and +0.54 px
    at 1.35x, up to 0.75 and 0.91 px; the exact mode leaves +0.13 and -0.06 px
    on average, scattered either way. ``scale_variant`` in
    ``scripts/train_contrast_invariant.py`` uses the plain mode.
    """
    cv2 = _cv2()

    h, w = image.shape[:2]
    size = (max(32, int(round(w * factor))), max(32, int(round(h * factor))))
    resized = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    relabelled = cv2.resize(labels.astype(np.int32), size,
                            interpolation=cv2.INTER_NEAREST_EXACT)
    return resized, relabelled


def local_background(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Isotropic local background: a Gaussian of large sigma, same in every direction."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image.astype(np.float32), sigma_px, mode="nearest")


def contrast(image, *, factor: float, sigma_px: float):
    background = local_background(image, sigma_px)
    return background + factor * (image - background)


def gain_offset(image, *, gain: float, offset: float):
    _, span = _range_of(image)
    return gain * image + offset * span


def gamma(image, *, gamma: float):
    lo, span = _range_of(image)
    z = np.clip((image - lo) / span, 0.0, None)
    return lo + span * np.power(z, gamma)


def gradient(image, *, amplitude: float, angle_rad: float):
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    ramp = (xx - w / 2) * math.cos(angle_rad) + (yy - h / 2) * math.sin(angle_rad)
    ramp /= max(float(np.abs(ramp).max()), 1e-6)
    return image * (1.0 + amplitude * ramp)


def vignette(image, *, strength: float, cy: float, cx: float):
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r2 = ((yy - cy * h) / h) ** 2 + ((xx - cx * w) / w) ** 2
    return image * (1.0 - strength * r2 / max(float(r2.max()), 1e-6))


def defocus(image, *, sigma_px: float):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma_px, mode="nearest")


def anisotropic_blur(image, *, sigma_px: float, ratio: float, angle_rad: float):
    from scipy.ndimage import convolve

    s_along, s_across = sigma_px, max(sigma_px * ratio, 0.3)
    radius = int(math.ceil(3 * s_along))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    u = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
    v = -xx * math.sin(angle_rad) + yy * math.cos(angle_rad)
    kernel = np.exp(-0.5 * ((u / s_along) ** 2 + (v / s_across) ** 2))
    kernel /= kernel.sum()
    return convolve(image, kernel.astype(np.float32), mode="nearest")


def downsample(image, *, factor: int):
    """Blur (sigma 0.4 x factor, as Cellpose 3 trains), decimate, interpolate back."""
    from scipy.ndimage import gaussian_filter

    cv2 = _cv2()

    h, w = image.shape
    blurred = gaussian_filter(image, 0.4 * factor, mode="nearest")
    small = cv2.resize(blurred, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def poisson(image, *, photons: float, seed: int):
    lo, span = _range_of(image)
    z = np.clip((image - lo) / span, 0.0, None)
    rng = np.random.default_rng(seed)
    return lo + span * rng.poisson(z * photons).astype(np.float32) / photons


def read_noise(image, *, sigma: float, seed: int):
    _, span = _range_of(image)
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, sigma * span, image.shape).astype(np.float32)


GEOMETRIC = {"flip": flip, "rot90": rot90, "scale": rescale}
PHOTOMETRIC = {
    "contrast": contrast, "gain_offset": gain_offset, "gamma": gamma,
    "gradient": gradient, "vignette": vignette, "defocus": defocus,
    "anisotropic_blur": anisotropic_blur, "downsample": downsample,
    "poisson": poisson, "read_noise": read_noise,
}


# --------------------------------------------------------------------------


def _seed_for(policy: AugmentationPolicy, source: str, copy: int) -> list[int]:
    # crc32, not hash(): Python salts str hashes per process.
    return [int(policy.seed), zlib.crc32(source.encode("utf-8")), int(copy)]


def sample_ops(policy: AugmentationPolicy, rng: np.random.Generator) -> list[dict]:
    """Draw one copy's operations and every parameter they will use."""
    ops: list[dict] = []
    for axis in (0, 1):
        if rng.random() < policy.p_flip:
            ops.append({"op": "flip", "axis": axis})
    if policy.rotate90:
        k = int(rng.integers(0, 4))
        if k:
            ops.append({"op": "rot90", "k": k})
    if rng.random() < policy.p_scale:
        ops.append({"op": "scale", "factor": policy.scale.sample(rng)})
    if rng.random() < policy.p_contrast:
        ops.append({"op": "contrast", "factor": policy.contrast.sample(rng),
                    "sigma_px": policy.background_sigma_px})
    if rng.random() < policy.p_gain:
        ops.append({"op": "gain_offset", "gain": policy.gain.sample(rng),
                    "offset": policy.offset.sample(rng)})
    if rng.random() < policy.p_gamma:
        ops.append({"op": "gamma", "gamma": policy.gamma.sample(rng)})
    if rng.random() < policy.p_gradient:
        ops.append({"op": "gradient", "amplitude": policy.gradient.sample(rng),
                    "angle_rad": float(rng.uniform(0, 2 * math.pi))})
    if rng.random() < policy.p_vignette:
        ops.append({"op": "vignette", "strength": policy.vignette.sample(rng),
                    "cy": float(rng.uniform(0.3, 0.7)), "cx": float(rng.uniform(0.3, 0.7))})
    if rng.random() < policy.p_defocus:
        ops.append({"op": "defocus", "sigma_px": policy.defocus_sigma_px.sample(rng)})
    if rng.random() < policy.p_anisotropic:
        ops.append({"op": "anisotropic_blur",
                    "sigma_px": policy.anisotropic_sigma_px.sample(rng),
                    "ratio": policy.anisotropic_ratio.sample(rng),
                    "angle_rad": float(rng.uniform(0, math.pi))})
    if rng.random() < policy.p_downsample:
        ops.append({"op": "downsample",
                    "factor": int(rng.choice(np.asarray(policy.downsample_factors)))})
    if rng.random() < policy.p_poisson:
        ops.append({"op": "poisson", "photons": policy.poisson_photons.sample(rng),
                    "seed": int(rng.integers(0, 2**31 - 1))})
    if rng.random() < policy.p_read_noise:
        ops.append({"op": "read_noise", "sigma": policy.read_noise.sample(rng),
                    "seed": int(rng.integers(0, 2**31 - 1))})
    return ops


def replay(image: np.ndarray, labels: np.ndarray, ops: list[dict]):
    """Apply recorded operations in order. Labels pass only through geometric ones."""
    out = np.asarray(image, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32)
    for op in ops:
        name = op["op"]
        params = {k: v for k, v in op.items() if k != "op"}
        if name in GEOMETRIC:
            out, lab = GEOMETRIC[name](out, lab, **params)
        elif name in PHOTOMETRIC:
            out = PHOTOMETRIC[name](out, **params)
        else:
            raise ValueError(f"unknown augmentation {name!r}")
    return out.astype(np.float32), lab.astype(np.int32)


def augment(image: np.ndarray, labels: np.ndarray, policy: AugmentationPolicy, *,
            source: str, copy: int):
    """One augmented copy of (image, labels), and the record that rebuilds it."""
    seed = _seed_for(policy, source, copy)
    ops = sample_ops(policy, np.random.default_rng(seed))
    out, lab = replay(image, labels, ops)
    return out, lab, AugmentationRecord(source=source, copy=copy, seed=seed, ops=ops)


def build_training_set(items, policy: AugmentationPolicy):
    """Originals plus ``copies_per_image`` recorded copies of each.

    ``items`` yields ``(source_id, image, labels)``. Returns images, labels and
    one record per training image (``copy`` -1 marks an original).
    """
    images, masks, records = [], [], []
    for source, image, labels in items:
        images.append(np.asarray(image, dtype=np.float32))
        masks.append(np.asarray(labels, dtype=np.int32))
        records.append(AugmentationRecord(source=source, copy=-1, seed=[], ops=[]))
        if not (np.asarray(labels) > 0).any() and not policy.augment_empty_frames:
            continue
        for copy in range(policy.copies_per_image):
            out, lab, record = augment(image, labels, policy, source=source, copy=copy)
            images.append(out)
            masks.append(lab)
            records.append(record)
    return images, masks, records
