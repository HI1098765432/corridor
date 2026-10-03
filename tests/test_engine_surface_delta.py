"""Bot 6b on synthetic ground truth (``docs/ENGINE_4D.md`` acceptance).

Three things the surface bot must get right, each with a planted answer:

*   a pure translation produces ~zero surface displacement once the translation
    is removed (it must not charge sliding to the membrane);
*   a one-sided extension is localised to the correct end relative to the
    motion direction (front vs rear is the biology);
*   the two compose -- a cell that both migrates and protrudes at its leading
    edge is read as translation plus a front extension.

The masks are digitised ellipses and ellipsoids, so the gained/lost areas are
known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from corridor.engine.surface_delta import surface_delta


def ellipse_mask(shape, cx, cy, a, b) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0


def ellipsoid_mask(shape, c, r) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2 <= 1.0


# --------------------------------------------------------------------------
# A pure translation leaves ~zero surface displacement after removal.
# --------------------------------------------------------------------------


def test_pure_integer_translation_removed_exactly():
    """Shift a cell by +6 px in x: after removal, nothing is gained or lost.

    An integer shift is exact, so the residual must be literally zero -- this
    pins that the translation estimate and its removal agree to the pixel.
    """
    shape = (120, 160)
    m_t = ellipse_mask(shape, 60, 60, 30, 10)
    m_tp1 = np.zeros_like(m_t)
    m_tp1[:, 6:] = m_t[:, :-6]  # +6 columns == +6 in x

    sd = surface_delta(m_t, m_tp1)
    assert sd.translation_method == "phase_correlation"
    assert sd.translation_px[0] == pytest.approx(6.0, abs=0.1)
    assert sd.translation_px[1] == pytest.approx(0.0, abs=0.1)
    assert sd.extension_px == 0.0
    assert sd.retraction_px == 0.0
    assert sd.mean_extension_px == 0.0
    assert sd.mean_retraction_px == 0.0


def test_subpixel_translation_recovered_from_intensity_roi():
    """A genuine 3.4 px sub-pixel shift encoded in the intensity ROI.

    Phase correlation reads sub-pixel shifts from the ROI intensity, so when a
    smooth intensity image is shifted by 3.4 px (no thresholding to quantise
    it), the estimate must be within 0.1 px -- the real-data path, where the
    cell's grey levels carry the fractional motion.  After removing it, the
    residual surface change is a thin resampling rim, well under a tenth of the
    cell area.
    """
    shape = (120, 160)
    m_t = ellipse_mask(shape, 60, 60, 30, 10)
    img_t = ndi.gaussian_filter(m_t.astype(float), 3.0)
    img_tp1 = ndi.shift(img_t, (0.0, 3.4), order=3)  # +3.4 px in x, no threshold
    m_tp1 = ndi.shift(m_t.astype(float), (0.0, 3.4), order=1) >= 0.5

    sd = surface_delta(m_t, m_tp1, image_t=img_t, image_tp1=img_tp1)
    assert sd.translation_px[0] == pytest.approx(3.4, abs=0.1)
    assert sd.translation_px[1] == pytest.approx(0.0, abs=0.1)
    assert (sd.extension_px + sd.retraction_px) / m_t.sum() < 0.10


def test_mask_only_translation_is_the_masks_own_shift():
    """Without an intensity ROI, a hard mask resolves to the shift it encodes.

    A mask shifted by 3.4 px and re-thresholded has a true centroid shift of
    3.0 px (the threshold quantises the boundary); the mask-only estimate
    reports that honest 3.0, not the 3.4 that was lost to thresholding.
    """
    shape = (120, 160)
    m_t = ellipse_mask(shape, 60, 60, 30, 10)
    m_tp1 = ndi.shift(m_t.astype(float), (0.0, 3.4), order=1) >= 0.5
    true_shift = float(np.argwhere(m_tp1).mean(0)[1] - np.argwhere(m_t).mean(0)[1])

    sd = surface_delta(m_t, m_tp1)
    assert sd.translation_px[0] == pytest.approx(true_shift, abs=0.2)
    assert sd.centroid_translation_px[0] == pytest.approx(true_shift, abs=0.01)
    assert (sd.extension_px + sd.retraction_px) / m_t.sum() < 0.08


def test_pure_translation_3d():
    """A sphere stepping (dx, dy, dz) = (4, 2, 1) voxels: ~zero surface change."""
    shape = (40, 60, 60)
    m_t = ellipsoid_mask(shape, (20, 30, 28), (8, 8, 8))
    m_tp1 = ellipsoid_mask(shape, (21, 32, 32), (8, 8, 8))  # +1 z, +2 y, +4 x

    sd = surface_delta(m_t, m_tp1)
    assert sd.ndim == 3
    assert sd.translation_px[0] == pytest.approx(4.0, abs=0.3)  # x
    assert sd.translation_px[1] == pytest.approx(2.0, abs=0.3)  # y
    assert sd.translation_px[2] == pytest.approx(1.0, abs=0.3)  # z
    # a rigid translation of a digitised sphere leaves only a thin resampling rim
    assert (sd.extension_px + sd.retraction_px) / m_t.sum() < 0.10


# --------------------------------------------------------------------------
# A one-sided extension is localised to the correct end.
# --------------------------------------------------------------------------


def test_one_sided_extension_localised_to_front():
    """No translation, a protrusion at the +x end, motion declared as +x.

    The gained region must land at the front, the lost region must be empty,
    and the gained area must match the planted protrusion.
    """
    shape = (120, 160)
    m_t = ellipse_mask(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy()
    m_tp1[55:66, 110:122] = True  # an 11 x 12 cap past the +x tip
    planted = int((m_tp1 & ~m_t).sum())

    sd = surface_delta(m_t, m_tp1, motion_direction=(1.0, 0.0))
    assert sd.motion_direction_source == "explicit"
    assert sd.dominant_extension_location == "front"
    assert sd.retraction_px == pytest.approx(0.0, abs=2.0)
    assert sd.extension_px == pytest.approx(planted, rel=0.05)
    assert sd.extension_by_location_px["front"] > 0.8 * sd.extension_px


def test_localisation_follows_the_declared_direction():
    """The same protrusion is 'rear' when the motion direction is reversed.

    Localisation is relative to the motion direction, not to the image, so
    flipping the direction must flip front and rear -- proof the label is real,
    not an artefact of where the pixels happen to sit.
    """
    shape = (120, 160)
    m_t = ellipse_mask(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy()
    m_tp1[55:66, 110:122] = True

    assert surface_delta(m_t, m_tp1, motion_direction=(1.0, 0.0)).dominant_extension_location == "front"
    assert surface_delta(m_t, m_tp1, motion_direction=(-1.0, 0.0)).dominant_extension_location == "rear"


def test_retraction_localised_to_rear():
    """A tail lost at the -x end while the cell migrates +x -> retraction, rear."""
    shape = (120, 160)
    m_t = ellipse_mask(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy()
    m_tp1[:, :58] = False  # erase the -x (trailing) tip
    lost = int((m_t & ~m_tp1).sum())

    sd = surface_delta(m_t, m_tp1, motion_direction=(1.0, 0.0))
    assert sd.dominant_retraction_location == "rear"
    assert sd.retraction_px == pytest.approx(lost, rel=0.05)
    assert sd.extension_px == pytest.approx(0.0, abs=2.0)


# --------------------------------------------------------------------------
# Translation and deformation compose: a migrating, protruding cell.
# --------------------------------------------------------------------------


def test_migration_plus_leading_edge_protrusion():
    """Translate +8 px in x AND grow a leading-edge protrusion.

    After the whole-cell translation is removed, only the protrusion remains,
    localised at the front -- the leading-edge biology the engine exists for.
    The direction here comes from the estimated translation, not a hand-set
    vector, so this also exercises the translation-derived direction.
    """
    shape = (140, 220)
    base = ellipse_mask(shape, 80, 70, 30, 10)
    # t+1: the same body translated +8 px in x, then a cap added at its new tip
    translated = np.zeros_like(base)
    translated[:, 8:] = base[:, :-8]
    m_tp1 = translated.copy()
    m_tp1[64:77, 120:132] = True  # protrusion past the translated +x tip
    protrusion = int((m_tp1 & ~translated).sum())

    sd = surface_delta(base, m_tp1)
    assert sd.motion_direction_source == "translation"
    assert sd.translation_px[0] == pytest.approx(8.0, abs=0.5)
    # the extension is the protrusion only -- the translated bulk is removed
    assert sd.extension_px == pytest.approx(protrusion, rel=0.25)
    assert sd.dominant_extension_location == "front"
    # net surface change is small compared with the whole cell that moved
    assert sd.extension_px < 0.3 * base.sum()


# --------------------------------------------------------------------------
# Evidence and guards.
# --------------------------------------------------------------------------


def test_boundary_displacement_field_signs():
    """u(s) is positive (outward) where the cell extended, negative where it pulled in."""
    shape = (120, 160)
    m_t = ellipse_mask(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy()
    m_tp1[55:66, 110:122] = True  # extend +x

    sd = surface_delta(m_t, m_tp1, motion_direction=(1.0, 0.0))
    assert sd.max_extension_px > 1.0  # the boundary pushed outward
    assert sd.n_boundary_samples > 0


def test_um_conversion_when_calibrated():
    shape = (120, 160)
    m_t = ellipse_mask(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy()
    m_tp1[55:66, 110:122] = True
    planted = int((m_tp1 & ~m_t).sum())

    sd = surface_delta(m_t, m_tp1, pixel_size_um=0.5, motion_direction=(1.0, 0.0))
    assert sd.extension_um == pytest.approx(planted * 0.25, rel=0.05)  # 0.5^2 um^2/px


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        surface_delta(np.zeros((10, 10), bool), np.zeros((10, 12), bool))
