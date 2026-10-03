"""Ground truth for Bot 1 (registration): a known sub-pixel drift is recovered.

The acceptance in ``docs/ENGINE_4D.md`` section 4: "known subpixel drift
recovered within 0.05 px".  Two phantoms exercise the two regimes:

* a 2-D movie whose drift is applied with ``scipy.ndimage.shift`` -- a
  *non-periodic* translation, exactly the real-microscopy case the shipped
  default (Hann window + un-normalised matching) is tuned for; and
* a 3-D volume whose drift is a band-limited Fourier shift -- a clean periodic
  translation, matched with the phase-normalised, window-free configuration,
  which recovers all three axes including the hard Z axis to <=0.05 px.

Both numbers are produced by the code here and are reproducible by rerunning.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.ndimage import fourier_shift

from corridor.engine.registration4d import (
    NORMALIZATION_PHASE,
    RegistrationConfig,
    choose_reference_timepoint,
    register_stack,
)


def _texture_2d(shape=(120, 90), sigma=2.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = ndi.gaussian_filter(rng.normal(0.0, 1.0, shape), sigma)
    return (tex - tex.mean()) * 300.0 + 800.0


def _texture_3d(shape=(40, 64, 56), sigma=2.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = ndi.gaussian_filter(rng.normal(0.0, 1.0, shape), sigma)
    return (tex - tex.mean()) * 300.0 + 800.0


def _fourier_shift(image: np.ndarray, drift) -> np.ndarray:
    return np.fft.ifftn(fourier_shift(np.fft.fftn(image), drift)).real


# --------------------------------------------------------------------------
# Reference selection
# --------------------------------------------------------------------------


def test_reference_is_sharpest_near_centre():
    base = _texture_2d()
    frames = [ndi.gaussian_filter(base, 2.0) for _ in range(7)]  # all blurred
    frames[3] = base  # the crisp one, at the centre
    stack = np.stack(frames)
    assert choose_reference_timepoint(stack) == 3


def test_reference_ignores_a_sharp_end_frame():
    # A crisp frame at the very end must not be chosen: it is outside the
    # central window, so the drift it defines would not overlap the series.
    base = _texture_2d()
    frames = [ndi.gaussian_filter(base, 2.0) for _ in range(7)]
    frames[6] = base * 3.0  # very sharp, but at the end
    stack = np.stack(frames)
    assert choose_reference_timepoint(stack) != 6


# --------------------------------------------------------------------------
# 2-D: non-periodic drift, shipped default config
# --------------------------------------------------------------------------

# (dy, dx) in array order; the centre frame (index 3) is the un-shifted reference.
_DRIFTS_2D = [
    (1.3, -0.7),
    (-0.9, 1.1),
    (0.4, 0.3),
    (0.0, 0.0),
    (-1.2, 0.6),
    (0.8, -1.3),
    (1.5, 0.9),
]


def _build_2d_stack():
    base = _texture_2d()
    frames = [ndi.shift(base, d, order=1, mode="nearest") for d in _DRIFTS_2D]
    return np.stack(frames)


def test_2d_subpixel_drift_recovered_within_tolerance():
    stack = _build_2d_stack()
    result = register_stack(stack)
    assert result.reference_t == 3
    errors = []
    for row, drift in zip(result.rows, _DRIFTS_2D):
        # stored drift is the volume translation; dx = x-component = drift[1].
        errors.append(abs(row.dx_px - drift[1]))
        errors.append(abs(row.dy_px - drift[0]))
        assert row.dz_px == 0.0  # Z disappears cleanly in 2-D
        assert row.sigma_z_px == 0.0
    assert max(errors) <= 0.05, f"max in-plane error {max(errors):.4f} px"


def test_2d_registered_stack_aligns_to_reference():
    stack = _build_2d_stack().astype(np.float32)
    result = register_stack(stack)
    ref = result.registered[result.reference_t]
    interior = (slice(16, -16), slice(16, -16))
    for t in range(stack.shape[0]):
        if t == result.reference_t:
            continue
        before = np.sqrt(np.mean((stack[t][interior] - ref[interior]) ** 2))
        after = np.sqrt(np.mean((result.registered[t][interior] - ref[interior]) ** 2))
        assert after < before  # registration reduces the residual to the reference


# --------------------------------------------------------------------------
# 3-D: band-limited drift, all three axes within tolerance
# --------------------------------------------------------------------------

# (dz, dy, dx) in array order; the centre frame (index 2) is the reference.
_DRIFTS_3D = [
    (0.6, 1.3, -0.7),
    (-0.4, -1.1, 0.9),
    (0.0, 0.0, 0.0),
    (1.2, -0.3, 0.55),
    (-0.8, 0.9, 1.1),
]


def test_3d_subpixel_drift_recovered_within_tolerance():
    base = _texture_3d()
    stack = np.stack([_fourier_shift(base, d) for d in _DRIFTS_3D])
    cfg = RegistrationConfig(normalization=NORMALIZATION_PHASE, window=False, upsample_factor=50)
    result = register_stack(stack, cfg)
    assert result.reference_t == 2
    assert result.has_z
    errors = []
    for row, drift in zip(result.rows, _DRIFTS_3D):
        errors.append(abs(row.dx_px - drift[2]))
        errors.append(abs(row.dy_px - drift[1]))
        errors.append(abs(row.dz_px - drift[0]))
    assert max(errors) <= 0.05, f"max 3-D error {max(errors):.4f} px"


# --------------------------------------------------------------------------
# Per-axis uncertainty
# --------------------------------------------------------------------------


def test_uncertainty_is_finite_and_bounded():
    result = register_stack(_build_2d_stack())
    for row in result.rows:
        for sigma in (row.sigma_x_px, row.sigma_y_px, row.sigma_z_px):
            assert np.isfinite(sigma)
            assert 0.0 <= sigma <= 1.0


def test_poorly_constrained_axis_reports_larger_uncertainty():
    # A volume with rich in-plane texture but almost no variation along Z gives
    # a correlation peak that is sharp in X/Y and broad in Z; the reported
    # sigma_z must exceed the in-plane sigmas.
    rng = np.random.default_rng(1)
    plane = ndi.gaussian_filter(rng.normal(0, 1, (64, 64)), 2.0)
    base = np.stack([plane] * 6, axis=0)  # identical planes -> flat in Z
    base = (base - base.mean()) * 300.0 + 800.0
    drifts = [(0.0, 1.1, -0.6), (0.0, -0.7, 0.9), (0.0, 0.0, 0.0),
              (0.0, 0.4, 0.5), (0.0, -1.2, 0.3), (0.0, 0.8, -0.9)]
    stack = np.stack([_fourier_shift(base, d) for d in drifts])
    cfg = RegistrationConfig(normalization=NORMALIZATION_PHASE, window=False)
    result = register_stack(stack, cfg)
    moved = [r for r in result.rows if r.t != result.reference_t]
    assert all(r.sigma_z_px >= max(r.sigma_x_px, r.sigma_y_px) for r in moved)


# --------------------------------------------------------------------------
# Edge cases and the evidence contract
# --------------------------------------------------------------------------


def test_reference_row_is_zero():
    result = register_stack(_build_2d_stack())
    ref_row = result.rows[result.reference_t]
    assert (ref_row.dx_px, ref_row.dy_px, ref_row.dz_px, ref_row.error) == (0.0, 0.0, 0.0, 0.0)


def test_single_timepoint_is_handled():
    stack = _texture_2d()[np.newaxis]
    result = register_stack(stack)
    assert result.n_timepoints == 1
    assert result.reference_t == 0
    assert result.max_abs_shift_px() == 0.0


def test_rotation_flag_is_refused_not_ignored():
    with pytest.raises(NotImplementedError):
        register_stack(_build_2d_stack(), RegistrationConfig(estimate_rotation=True))


def test_to_dict_carries_rows_without_the_pixel_array():
    result = register_stack(_build_2d_stack())
    payload = result.to_dict()
    assert set(payload) >= {"reference_t", "has_z", "n_timepoints", "config", "rows"}
    assert len(payload["rows"]) == result.n_timepoints
    assert set(payload["rows"][0]) == {
        "t", "dx_px", "dy_px", "dz_px", "error",
        "sigma_x_px", "sigma_y_px", "sigma_z_px", "reference_t", "flagged",
    }
    assert "registered" not in payload  # the array is not in the evidence


def test_implausible_shift_is_flagged_and_not_applied():
    # One frame is given a large drift; with a plausibility bound set, its shift
    # is recorded but not applied, so that frame is left close to its original
    # (not shifted by the full amount) and is flagged.
    base = _texture_2d()
    # the centre frame (index 3) is clean, so it is chosen as the reference;
    # frame 0 carries an implausibly large drift.
    drifts = [(18.0, -15.0), (0.5, -0.3), (0.4, 0.2), (0.0, 0.0), (-0.6, 0.5),
              (0.3, 0.1), (0.2, -0.2)]
    stack = np.stack([ndi.shift(base, d, order=1, mode="nearest") for d in drifts])
    result = register_stack(stack, RegistrationConfig(max_shift_px=8.0))
    assert result.reference_t == 3
    flagged = [r for r in result.rows if r.flagged]
    assert len(flagged) == 1 and flagged[0].t == 0
    # the flagged frame was left unregistered (equal to the moving frame), not
    # shifted by a wrong amount
    assert np.array_equal(result.registered[0], stack[0].astype(np.float32))
    # unflagged small-drift frames are still applied
    assert all(not r.flagged for r in result.rows if r.t != 0)



def test_bad_dimensionality_is_rejected():
    with pytest.raises(ValueError):
        register_stack(np.zeros((5, 5)))  # a single 2-D image is not a movie
