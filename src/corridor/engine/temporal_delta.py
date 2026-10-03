"""Bot 5 -- temporal delta: forward/backward consistency of a correspondence.

Design contract: ``docs/ENGINE_4D.md`` section 2.  Given an object ``A_t`` at
time ``t`` and a proposed correspondence ``A_{t+k}`` at a later (or earlier)
time, this bot asks one question with deterministic mathematics: **if these two
masks are the same cell, do they agree when each is projected onto the other?**

For the forward leg it measures the object's displacement by *ROI phase
correlation* on the registered images -- a windowed cross-correlation around
``A_t``, which reads the real local motion of the pixels inside that window --
then projects ``A_t`` forward by that displacement and compares the projection
with ``A_{t+k}``.  For the backward leg it does the mirror: the displacement in
``A_{t+k}``'s own window, ``A_{t+k}`` projected back, compared with ``A_t``.
The disagreement of each leg is

    d = (1 - IoU) + centroid_distance_px

and the forward/backward error is ``E_FB = d_forward + d_backward``
(``ENGINE_4D``:  ``d(A_t, back(A_{t+1})) + d(A_{t+1}, fwd(A_t))``).

A true correspondence has a low ``E_FB``: the window's own motion carries ``A_t``
onto ``A_{t+k}`` and back.  A swap -- ``A_t`` matched to the wrong object -- has a
high ``E_FB``, because the window around ``A_t`` still reports ``A_t``'s true
motion, which lands nowhere near the wrong object, and the wrong object's window
reports its own motion, which lands nowhere near ``A_t``.  Windows of +-1 and
+-2 frames are measured.  **No volume is ever estimated.**

The images must already be registered (Bot 1 removes whole-field drift); the ROI
phase correlation then reads only the object's own residual displacement.  The
held-out Corridor stills are not contiguous frames, so there is no real
one-frame correspondence to score among them; this bot is therefore validated on
synthetic sequences with a planted displacement and on the one real contiguous
sample movie, and said so (``docs/NEXT_GENERATION.md`` section 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSettings:
    """Frozen settings for one temporal-consistency measurement."""

    #: Padding added around an object's bounding box to form the phase-
    #: correlation ROI, in px.  Must comfortably exceed the expected one-frame
    #: displacement so the moved object stays inside the common window.
    pad_px: int = 16
    #: ``upsample_factor`` for sub-pixel phase correlation (skimage).
    upsample: int = 10
    #: Cross-power-spectrum normalisation passed to skimage.  ``None`` is classic
    #: normalised cross-correlation; ``"phase"`` is whitened phase correlation.
    #: The default is ``None`` because a cell is a smooth, low-texture blob and
    #: phase whitening then locks onto noise -- measured: on a blob translated by
    #: a known (2, 6) px, ``"phase"`` returns a (0, 0) shift with peak error
    #: 0.9995, while ``None`` recovers (2, 6) with error 0.12.
    normalization: str | None = None
    #: Frame offsets to measure, as magnitudes; each is applied as +k and -k.
    windows: tuple[int, ...] = (1, 2)
    #: An ROI smaller than this on a side cannot be correlated; the leg is
    #: marked invalid rather than guessed.
    min_roi_px: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalDelta:
    """One correspondence's forward/backward consistency, every number exposed."""

    t: int
    label_t: int
    target_t: int
    label_target: int
    window_k: int  # signed frame offset (target_t - t)
    displacement_fwd_px: tuple[float, float]  # (dy, dx), A_t -> target
    displacement_back_px: tuple[float, float]  # (dy, dx), target -> t
    phase_error_fwd: float
    phase_error_back: float
    iou_fwd: float  # IoU(A_target, fwd(A_t))
    centroid_fwd_px: float  # |centroid(A_target) - centroid(fwd(A_t))|
    iou_back: float  # IoU(A_t, back(A_target))
    centroid_back_px: float
    d_fwd: float
    d_back: float
    e_fb: float
    valid: bool
    note: str = ""
    best: bool = False  # set by measure_object for the lowest-E_FB candidate

    def to_dict(self) -> dict:
        d = asdict(self)
        d["displacement_fwd_px"] = list(self.displacement_fwd_px)
        d["displacement_back_px"] = list(self.displacement_back_px)
        return d


# -- helpers -----------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = _centroid(a), _centroid(b)
    if ca is None or cb is None:
        return float("inf")
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def _shift_mask(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate a boolean mask by an integer-rounded shift, zero-filled."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    sy, sx = int(round(dy)), int(round(dx))
    y0, y1 = max(0, sy), min(h, h + sy)
    x0, x1 = max(0, sx), min(w, w + sx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = mask[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    return out


def _roi_window(mask: np.ndarray, pad_px: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad_px)
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad_px)
    return slice(y0, y1), slice(x0, x1)


def _roi_shift(reference: np.ndarray, moving: np.ndarray,
               window: tuple[slice, slice], s: TemporalSettings) -> tuple[float, float, float]:
    """Phase-correlate ``moving`` onto ``reference`` inside ``window``.

    Returns ``(dy, dx, error)``: the shift (skimage's convention) that registers
    ``moving`` to ``reference``.  With ``reference`` the frame being projected
    *toward* and ``moving`` the frame projected *from*, this shift is the object
    displacement from the moving frame to the reference frame.
    """
    from skimage.registration import phase_cross_correlation

    ref = reference[window].astype(np.float64)
    mov = moving[window].astype(np.float64)
    if min(ref.shape) < 1 or ref.shape != mov.shape:
        return 0.0, 0.0, float("inf")
    shift, error, _ = phase_cross_correlation(
        ref, mov, upsample_factor=s.upsample, normalization=s.normalization)
    err = float(error) if np.isfinite(error) else float("inf")
    return float(shift[0]), float(shift[1]), err


def _leg(images: np.ndarray, src_frame: int, dst_frame: int,
         src_mask: np.ndarray, dst_mask: np.ndarray,
         s: TemporalSettings) -> tuple[float, float, float, float, float]:
    """Project ``src_mask`` (at ``src_frame``) toward ``dst_frame`` and score it.

    Returns ``(dy, dx, error, iou, centroid_px)`` where ``iou``/``centroid_px``
    compare the projected source mask with ``dst_mask``.
    """
    window = _roi_window(src_mask, s.pad_px)
    if (window[0].stop - window[0].start < s.min_roi_px
            or window[1].stop - window[1].start < s.min_roi_px):
        return 0.0, 0.0, float("inf"), 0.0, float("inf")
    # reference = destination frame (project toward it); moving = source frame.
    dy, dx, err = _roi_shift(images[dst_frame], images[src_frame], window, s)
    projected = _shift_mask(src_mask, dy, dx)
    return dy, dx, err, _iou(projected, dst_mask), _centroid_distance(projected, dst_mask)


def consistency(images: np.ndarray, masks: np.ndarray,
                t: int, label_t: int, target_t: int, label_target: int,
                settings: TemporalSettings | None = None) -> TemporalDelta:
    """Measure ``E_FB`` for one proposed correspondence ``(t, label_t) <-> (target_t, label_target)``.

    ``images`` is ``(T, Y, X)`` registered intensity and ``masks`` is ``(T, Y, X)``
    per-frame instance labels.  Returns a :class:`TemporalDelta` with the
    forward and backward displacement, IoU, centroid distance, and ``E_FB``.
    """
    s = settings or TemporalSettings()
    images = np.asarray(images)
    masks = np.asarray(masks)
    k = int(target_t) - int(t)
    mask_t = masks[t] == label_t
    mask_target = masks[target_t] == label_target

    note = ""
    if not mask_t.any() or not mask_target.any():
        note = "missing object"
        return TemporalDelta(
            t, label_t, target_t, label_target, k, (0.0, 0.0), (0.0, 0.0),
            float("inf"), float("inf"), 0.0, float("inf"), 0.0, float("inf"),
            float("inf"), float("inf"), float("inf"), valid=False, note=note)

    # forward: project A_t onto the target frame, compare with A_target
    fdy, fdx, ferr, iou_fwd, cen_fwd = _leg(
        images, t, target_t, mask_t, mask_target, s)
    # backward: project A_target onto t, compare with A_t
    bdy, bdx, berr, iou_back, cen_back = _leg(
        images, target_t, t, mask_target, mask_t, s)

    d_fwd = (1.0 - iou_fwd) + cen_fwd
    d_back = (1.0 - iou_back) + cen_back
    e_fb = d_fwd + d_back
    valid = np.isfinite(e_fb)
    return TemporalDelta(
        t, label_t, target_t, label_target, k,
        (round(fdy, 4), round(fdx, 4)), (round(bdy, 4), round(bdx, 4)),
        round(ferr, 6), round(berr, 6),
        round(iou_fwd, 6), round(cen_fwd, 4) if np.isfinite(cen_fwd) else cen_fwd,
        round(iou_back, 6), round(cen_back, 4) if np.isfinite(cen_back) else cen_back,
        round(d_fwd, 6) if np.isfinite(d_fwd) else d_fwd,
        round(d_back, 6) if np.isfinite(d_back) else d_back,
        round(e_fb, 6) if np.isfinite(e_fb) else e_fb,
        valid=bool(valid), note=note)


def _labels_in(frame_labels: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(frame_labels) if v]


def measure_object(images: np.ndarray, masks: np.ndarray,
                   t: int, label_t: int,
                   settings: TemporalSettings | None = None) -> list[TemporalDelta]:
    """Score ``(t, label_t)`` against every candidate in each +-window frame.

    For every frame offset in ``settings.windows`` (as +k and -k), this measures
    ``E_FB`` for every object present in that frame and marks the lowest-``E_FB``
    candidate per offset ``best``.  The caller (the referee, Bot 8) reads the
    full list as evidence; the ``best`` flag is only a convenience.
    """
    s = settings or TemporalSettings()
    masks = np.asarray(masks)
    n = masks.shape[0]
    out: list[TemporalDelta] = []
    for mag in s.windows:
        for k in (mag, -mag):
            target_t = t + k
            if target_t < 0 or target_t >= n:
                continue
            records = [consistency(images, masks, t, label_t, target_t, lab, s)
                       for lab in _labels_in(masks[target_t])]
            finite = [r for r in records if r.valid and np.isfinite(r.e_fb)]
            if finite:
                best = min(finite, key=lambda r: r.e_fb)
                best.best = True
            out.extend(records)
    return out
