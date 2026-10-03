"""Bot 1 -- whole-volume registration (``docs/ENGINE_4D.md`` section 2).

The engine's thesis is that accuracy is sitting unused in the data, including
*across time*.  Before time can be used -- before a cell at ``t`` can be matched
to the same cell at ``t+1``, before a static atlas can be built from a temporal
median -- the stage drift between timepoints has to be removed, or the device
itself appears to move and every later stage inherits that error.

This bot estimates one rigid translation per timepoint and undoes it.  The
design facts it honours:

* **One reference, not a chain.**  Every timepoint is registered against a
  single reference timepoint, never frame-to-frame, because chained shifts
  accumulate their individual errors into a growing drift.  The reference is
  the sharpest volume *near the temporal centre* (highest variance of the
  Laplacian), so the common content is crisp and the reference is not an
  end-of-run frame that only overlaps half the series.

* **Sub-pixel by upsampled DFT.**  ``skimage.registration.phase_cross_correlation``
  with ``upsample_factor`` well above the contract floor of 10.  Measured on
  synthetic ground truth (``tests/test_engine_registration.py``): a known
  sub-pixel drift is recovered to <=0.05 px in plane.

* **Measured, not assumed, pre-processing.**  Real microscopy drift is *not* a
  circular shift -- new content enters at the frame edge -- which is the one
  case plain phase-normalised correlation handles badly (it assumes
  periodicity).  Measured on non-periodic synthetic drift, a centred Hann
  window plus un-normalised (cross-correlation) matching recovers the drift to
  ~0.02-0.04 px, where phase normalisation alone leaves ~0.15-0.19 px.  Those
  are therefore the shipped defaults; both knobs stay on
  :class:`RegistrationConfig` because a band-limited (periodic) volume is best
  matched with phase normalisation and no window, and the 3-D ground-truth
  test uses exactly that configuration.

* **Per-axis uncertainty from the peak.**  The correlation peak is sharp along
  a well-constrained axis and broad along a poorly-constrained one (Z, with few
  planes, is typically the worst).  The reported ``sigma_*_px`` is the
  first-order parabolic width of the peak along each axis, in pixels -- a
  relative precision indicator, not a calibrated confidence interval, and
  labelled as such.

* **Linear resampling to apply the correction.**  ``scipy.ndimage.shift`` with
  ``order=1``, per the contract.

The bot never segments anything and never estimates volume.  Rotation is
estimated only if a setting enables it; that path is not implemented in this
version and is refused loudly rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

#: ``normalization`` choices for :class:`RegistrationConfig`.  ``"cross_correlation"``
#: maps to ``phase_cross_correlation(normalization=None)`` (un-normalised,
#: measured best for real non-periodic drift); ``"phase"`` maps to the
#: phase-normalised default (best for a band-limited periodic volume).
NORMALIZATION_CROSS_CORRELATION = "cross_correlation"
NORMALIZATION_PHASE = "phase"
_NORMALIZATIONS = (NORMALIZATION_CROSS_CORRELATION, NORMALIZATION_PHASE)


@dataclass(frozen=True)
class RegistrationConfig:
    """Frozen settings for :func:`register_stack`.

    Defaults are the measured best for the real data this project has (2-D
    phase-contrast movies with non-periodic stage drift); see the module
    docstring and ``tests/test_engine_registration.py`` for the numbers.
    """

    #: Sub-pixel grid is ``1 / upsample_factor`` px.  The contract floor is 10;
    #: 50 gives a 0.02 px grid, comfortably finer than the 0.05 px acceptance.
    upsample_factor: int = 50
    #: ``"cross_correlation"`` (un-normalised) or ``"phase"``.  See the module
    #: docstring: un-normalised wins on real non-periodic drift.
    normalization: str = NORMALIZATION_CROSS_CORRELATION
    #: Multiply both images by a centred separable Hann window before matching.
    #: Suppresses the frame-edge discontinuity that a non-periodic shift creates
    #: and that otherwise biases the sub-pixel estimate.
    window: bool = True
    #: Only timepoints within this fraction of the series half-span from the
    #: temporal centre are eligible to be the reference (0.34 ~= central third).
    reference_window_fraction: float = 0.34
    #: Interpolation order for applying the inverse shift (contract: linear).
    interpolation_order: int = 1
    #: Boundary handling for the applied shift.  ``"nearest"`` keeps edge
    #: intensities rather than fabricating zeros where content left the frame.
    shift_mode: str = "nearest"
    #: Largest plausible absolute drift per axis, in px.  A microfluidic field
    #: of evenly-spaced channel walls is periodic across the channels, so the
    #: correlation has secondary peaks one channel pitch apart and a frame can
    #: lock onto a pitch-multiple -- measured on 052924_1.tif, t6 and t16 came
    #: back at ~72-84 px cross-channel while their temporal neighbours were
    #: ~1-7 px (build/eng/registration-atlas).  When set, a shift exceeding this
    #: on any axis is recorded as the raw estimate, flagged, and NOT applied
    #: (the frame is left unregistered rather than mis-shifted by a pitch).
    #: None (the default) keeps the validated behaviour: every shift is applied.
    max_shift_px: float | None = None
    #: Rotation estimation is not implemented in this bot version.  True raises,
    #: so a caller is never told rotation was handled when it was not.
    estimate_rotation: bool = False
    eps: float = 1e-8

    def skimage_normalization(self) -> str | None:
        """The value to pass to ``phase_cross_correlation(normalization=...)``."""
        if self.normalization == NORMALIZATION_CROSS_CORRELATION:
            return None
        if self.normalization == NORMALIZATION_PHASE:
            return "phase"
        raise ValueError(
            f"normalization must be one of {_NORMALIZATIONS}, got {self.normalization!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrationRow:
    """One timepoint's estimated drift and its per-axis precision.

    ``dx/dy/dz_px`` is the translation *of the volume* relative to the
    reference, in image coordinates (x = column, y = row, z = plane), in
    pixels.  It is the quantity a reader wants ("how far did the stage drift at
    t?"); the correction applied to register the volume is its negation.
    ``dz_px`` is 0 for a 2-D (TYX) movie.  ``error`` is the translation-
    invariant normalised RMS error from the correlation (0 = identical).
    ``sigma_*_px`` is the per-axis peak width (see the module docstring).
    """

    t: int
    dx_px: float
    dy_px: float
    dz_px: float
    error: float
    sigma_x_px: float
    sigma_y_px: float
    sigma_z_px: float
    reference_t: int
    #: True when the estimated shift exceeded ``config.max_shift_px`` and was
    #: therefore recorded but not applied (a likely pitch-mislock).
    flagged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrationResult:
    """The whole registered stack plus the per-timepoint evidence.

    ``registered`` has the same shape as the input, as float32, with each
    timepoint shifted onto the reference.  ``rows`` carries one
    :class:`RegistrationRow` per timepoint (the reference's row is all zeros).
    ``to_dict`` deliberately omits the pixel array -- it is the inspectable
    evidence, written to ``registration.csv`` / an evidence JSON by the
    pipeline, not the image data.
    """

    reference_t: int
    rows: tuple[RegistrationRow, ...]
    registered: np.ndarray = field(repr=False)
    has_z: bool
    config: RegistrationConfig

    @property
    def n_timepoints(self) -> int:
        return len(self.rows)

    def registration_rows(self) -> list[dict[str, Any]]:
        """The evidence as plain dicts -- the rows of ``registration.csv``."""
        return [row.to_dict() for row in self.rows]

    def max_abs_shift_px(self) -> float:
        """Largest absolute drift component over all timepoints and axes, in px."""
        return max(
            (max(abs(r.dx_px), abs(r.dy_px), abs(r.dz_px)) for r in self.rows),
            default=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_t": self.reference_t,
            "has_z": self.has_z,
            "n_timepoints": self.n_timepoints,
            "max_abs_shift_px": self.max_abs_shift_px(),
            "config": self.config.to_dict(),
            "rows": self.registration_rows(),
        }


# --------------------------------------------------------------------------
# Reference selection
# --------------------------------------------------------------------------


def _laplacian_variance(volume: np.ndarray) -> float:
    """Sharpness score: the variance of the Laplacian of one volume.

    A crisp, in-focus image has strong second derivatives and so a high
    Laplacian variance; a blurred or empty one has a low variance.  This is the
    standard focus measure and needs no tuning.
    """
    return float(ndi.laplace(volume.astype(np.float64)).var())


def choose_reference_timepoint(
    stack: np.ndarray, reference_window_fraction: float = 0.34
) -> int:
    """Index of the sharpest timepoint near the temporal centre.

    Only timepoints within ``reference_window_fraction`` of the series
    half-span from the centre are eligible, so the reference overlaps the rest
    of the series rather than sitting at an end.  Among those, the highest
    Laplacian variance wins; ties break towards the centre, then to the lower
    index, so the choice is deterministic.
    """
    arr = np.asarray(stack)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("cannot choose a reference timepoint from an empty stack")
    centre = (n - 1) / 2.0
    half = max(0.0, reference_window_fraction * (n - 1) / 2.0)
    candidates = [t for t in range(n) if abs(t - centre) <= half + 1e-9]
    if not candidates:
        candidates = [int(round(centre))]
    # max sharpness; then nearest the centre; then lowest index -- all via one key.
    return max(
        candidates,
        key=lambda t: (_laplacian_variance(arr[t]), -abs(t - centre), -t),
    )


# --------------------------------------------------------------------------
# Correlation helpers
# --------------------------------------------------------------------------


def _hann_window(shape: tuple[int, ...]) -> np.ndarray:
    """A separable Hann window over every axis of ``shape`` (ones where n < 4).

    An axis with fewer than four samples (a near-degenerate Z) cannot carry a
    meaningful taper, so it is left flat rather than collapsed to near-zero.
    """
    w = np.ones(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        h = np.hanning(n) if n >= 4 else np.ones(n)
        w = w * h.reshape([-1 if i == axis else 1 for i in range(len(shape))])
    return w


def _prepare(image: np.ndarray, window: np.ndarray | None) -> np.ndarray:
    """Mean-subtract (remove DC) and optionally apply the window, as float64."""
    out = image.astype(np.float64)
    out = out - out.mean()
    if window is not None:
        out = out * window
    return out


def _peak_uncertainty(
    a_ref: np.ndarray, a_mov: np.ndarray, cfg: RegistrationConfig
) -> np.ndarray:
    """Per-axis peak width of the correlation surface, in px (array-axis order).

    The surface is the (same-normalisation) cross-correlation of the two
    prepared images.  Around the integer peak a 3-point parabola is fitted
    along each axis; its curvature, divided into the robust noise level of the
    surface, is a first-order standard error of the sub-pixel peak location.  A
    sharp peak (large curvature) gives a small number; a broad or degenerate
    peak gives a large one, capped at 1 px ("not localised to within a pixel").
    This is a relative precision indicator, not a calibrated interval.
    """
    fa = np.fft.fftn(a_ref)
    fb = np.fft.fftn(a_mov)
    r = fa * np.conj(fb)
    if cfg.skimage_normalization() == "phase":
        r = r / (np.abs(r) + cfg.eps)
    surface = np.fft.ifftn(r).real
    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    median = float(np.median(surface))
    noise = 1.4826 * float(np.median(np.abs(surface - median))) + cfg.eps
    sigmas = np.empty(surface.ndim, dtype=np.float64)
    v0 = float(surface[peak])
    for axis in range(surface.ndim):
        length = surface.shape[axis]
        if length < 3:
            sigmas[axis] = 1.0
            continue
        lo = list(peak)
        lo[axis] = (peak[axis] - 1) % length
        hi = list(peak)
        hi[axis] = (peak[axis] + 1) % length
        vm = float(surface[tuple(lo)])
        vp = float(surface[tuple(hi)])
        curvature = 2.0 * v0 - vm - vp  # > 0 at a maximum
        if curvature <= noise:
            sigmas[axis] = 1.0
        else:
            sigmas[axis] = min(1.0, noise / curvature)
    return sigmas


def _register_pair(
    a_ref: np.ndarray, a_mov: np.ndarray, cfg: RegistrationConfig
) -> tuple[np.ndarray, float]:
    """Array-order shift that registers ``a_mov`` onto ``a_ref`` and its error."""
    shift, error, _ = phase_cross_correlation(
        a_ref,
        a_mov,
        upsample_factor=cfg.upsample_factor,
        normalization=cfg.skimage_normalization(),
    )
    return np.asarray(shift, dtype=np.float64), float(error)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def register_stack(
    stack: np.ndarray, config: RegistrationConfig | None = None
) -> RegistrationResult:
    """Register every timepoint of ``stack`` onto one reference timepoint.

    ``stack`` is ``(T, Y, X)`` (a 2-D movie) or ``(T, Z, Y, X)`` (a 3-D movie).
    Z disappears cleanly when absent: the 2-D path records ``dz_px = 0`` and
    ``sigma_z_px = 0`` and runs exactly the same code otherwise.  The returned
    ``registered`` stack is float32 and the same shape as the input.
    """
    cfg = config or RegistrationConfig()
    if cfg.estimate_rotation:
        raise NotImplementedError(
            "rotation estimation is not implemented in registration4d; "
            "set estimate_rotation=False (translation only)."
        )
    arr = np.asarray(stack)
    if arr.ndim == 3:
        has_z = False
    elif arr.ndim == 4:
        has_z = True
    else:
        raise ValueError(
            "register_stack expects a (T, Y, X) or (T, Z, Y, X) stack; "
            f"got ndim={arr.ndim}"
        )

    volume = arr.astype(np.float32)
    n = volume.shape[0]
    reference_t = choose_reference_timepoint(volume, cfg.reference_window_fraction)
    reference_image = volume[reference_t]
    window = _hann_window(reference_image.shape) if cfg.window else None
    prepared_reference = _prepare(reference_image, window)

    registered = np.empty_like(volume)
    rows: list[RegistrationRow] = []
    for t in range(n):
        if t == reference_t:
            registered[t] = reference_image
            rows.append(
                RegistrationRow(
                    t=t,
                    dx_px=0.0,
                    dy_px=0.0,
                    dz_px=0.0,
                    error=0.0,
                    sigma_x_px=0.0,
                    sigma_y_px=0.0,
                    sigma_z_px=0.0,
                    reference_t=reference_t,
                )
            )
            continue

        moving = volume[t]
        prepared_moving = _prepare(moving, window)
        shift, error = _register_pair(prepared_reference, prepared_moving, cfg)
        sigma = _peak_uncertainty(prepared_reference, prepared_moving, cfg)
        # A shift larger than the plausible drift is a likely pitch-mislock on a
        # periodic field: record the raw estimate but do not apply it, so the
        # frame is left as-is rather than shifted by a whole channel pitch.
        flagged = cfg.max_shift_px is not None and float(np.max(np.abs(shift))) > cfg.max_shift_px
        # The correction that registers moving onto the reference is exactly the
        # shift phase_cross_correlation returns (verified empirically); the
        # drift of the volume is its negation.
        if flagged:
            registered[t] = moving
        else:
            registered[t] = ndi.shift(
                moving, shift, order=cfg.interpolation_order, mode=cfg.shift_mode
            )
        if has_z:
            dz_px, dy_px, dx_px = (-shift[0], -shift[1], -shift[2])
            sigma_z_px, sigma_y_px, sigma_x_px = (
                float(sigma[0]),
                float(sigma[1]),
                float(sigma[2]),
            )
        else:
            dy_px, dx_px = (-shift[0], -shift[1])
            dz_px = 0.0
            sigma_y_px, sigma_x_px = (float(sigma[0]), float(sigma[1]))
            sigma_z_px = 0.0
        rows.append(
            RegistrationRow(
                t=t,
                dx_px=float(dx_px),
                dy_px=float(dy_px),
                dz_px=float(dz_px),
                error=error,
                sigma_x_px=sigma_x_px,
                sigma_y_px=sigma_y_px,
                sigma_z_px=sigma_z_px,
                reference_t=reference_t,
                flagged=flagged,
            )
        )

    return RegistrationResult(
        reference_t=reference_t,
        rows=tuple(rows),
        registered=registered,
        has_z=has_z,
        config=cfg,
    )
