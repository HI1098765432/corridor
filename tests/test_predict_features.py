"""Strict morphology must describe the shape and nothing else.

These pin the properties the morphology->migration claim depends on: the strict
set carries no orientation (rotating or mirroring a cell does not move it),
lengths scale with calibration, and the geometric primitives (curvature,
skeleton, hull) give the known answers on shapes whose answers are known.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# training/ is research code and deliberately not an installed package (the
# installer must never contain it), so the checkout root has to be importable
# whatever pytest.ini's pythonpath says.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.predict import features as F  # noqa: E402


def cell_mask(angle_rad: float = 0.0, length: float = 90.0, width: float = 11.0,
              wobble: float = 0.0, seed: int = 0) -> np.ndarray:
    """A confined-cell-shaped mask (an elongated, optionally wobbly ellipse)."""
    from skimage.draw import polygon

    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * math.pi, 240, endpoint=False)
    r = np.ones_like(theta)
    if wobble:
        for k in (2, 3, 5):
            r += wobble * rng.normal() * np.cos(k * theta + rng.uniform(0, 6.28))
    a = 0.5 * length * np.cos(theta) * r
    b = 0.5 * width * np.sin(theta) * r
    size = int(length + 20)
    c = size / 2.0
    rows = c + a * math.cos(angle_rad) - b * math.sin(angle_rad)
    cols = c + a * math.sin(angle_rad) + b * math.cos(angle_rad)
    m = np.zeros((size, size), dtype=bool)
    rr, cc = polygon(rows, cols, shape=m.shape)
    m[rr, cc] = True
    return m


#: Features whose value is set by the shape at the scale of the whole cell.
#: Curvature statistics and Hu invariants are excluded from the tight check:
#: they live at the scale of the pixel grid (see contour_curvature).
ROBUST = ("area_um2", "perimeter_um", "convex_area_um2", "convex_perimeter_um",
          "major_axis_um", "minor_axis_um", "equivalent_diameter_um", "aspect_ratio",
          "eccentricity", "solidity", "circularity", "roughness", "skeleton_length_um")


def test_strict_set_has_no_orientation_position_or_motion():
    names = F.strict_feature_names(0.467)
    for name in names:
        for token in F.FORBIDDEN_STRICT_TOKENS:
            assert token not in name, f"{name} looks like it carries {token}"
    assert "extent" not in names, "grid-aligned extent encodes orientation"


@pytest.mark.parametrize("angle_deg", [7, 20, 33, 45, 62, 90])
def test_rotating_a_cell_does_not_move_its_shape_features(angle_deg):
    ref = F.strict_morphology(cell_mask(0.0, wobble=0.04), 0.467)
    rot = F.strict_morphology(cell_mask(math.radians(angle_deg), wobble=0.04), 0.467)
    for k in ROBUST:
        assert rot[k] == pytest.approx(ref[k], rel=0.06), k
    # The principal-axis box of an 11-px-wide cell is ~11 px across, so a
    # +/-0.5 px quantisation of that side alone moves extent by ~5 %.
    assert rot["extent_principal"] == pytest.approx(ref["extent_principal"], rel=0.10)
    for k in ("skeleton_branches", "skeleton_junctions"):
        assert rot[k] == ref[k], k


def test_exact_grid_symmetries_leave_every_strict_feature_unchanged():
    m = cell_mask(0.3, wobble=0.05, seed=2)
    ref = F.strict_morphology(m, 0.467)
    for variant in (np.rot90(m), np.fliplr(m), np.flipud(m), np.rot90(m, 2)):
        got = F.strict_morphology(variant, 0.467)
        for k, v in ref.items():
            if k.startswith("curvature") or k in ("bending_energy", "concave_fraction"):
                # The contour is resampled from a different start point.
                assert got[k] == pytest.approx(v, rel=0.04, abs=2e-3), k
            elif k.startswith("skeleton_length"):
                # Branch tracing breaks neighbour ties in a grid-dependent order.
                assert got[k] == pytest.approx(v, rel=2e-3), k
            else:
                assert got[k] == pytest.approx(v, rel=1e-6, abs=1e-9), k


def test_calibration_scales_lengths_and_areas_and_nothing_else():
    m = cell_mask(0.2, wobble=0.03)
    px = F.strict_morphology(m, None)
    um = F.strict_morphology(m, 0.5)
    assert px["area_px2"] * 0.25 == pytest.approx(um["area_um2"])
    assert px["major_axis_px"] * 0.5 == pytest.approx(um["major_axis_um"])
    assert px["skeleton_length_px"] * 0.5 == pytest.approx(um["skeleton_length_um"])
    assert px["curvature_mean_abs_per_px"] / 0.5 == pytest.approx(um["curvature_mean_abs_per_um"])
    for k in ("aspect_ratio", "solidity", "circularity", "roughness", "hu1_log", "bending_energy"):
        assert px[k] == pytest.approx(um[k]), k


def test_a_disc_has_the_known_answers():
    from skimage.draw import disk

    m = np.zeros((101, 101), dtype=bool)
    rr, cc = disk((50, 50), 30)
    m[rr, cc] = True
    f = F.strict_morphology(m, None)
    assert f["circularity"] == pytest.approx(1.0, abs=0.02)
    assert f["curvature_mean_abs_per_px"] == pytest.approx(1 / 30, rel=0.05)
    assert f["concave_fraction"] == 0.0
    assert f["aspect_ratio"] == pytest.approx(1.0, abs=0.02)
    assert f["roughness"] == pytest.approx(1.0, abs=0.01)
    assert f["skeleton_branches"] == 1


def test_a_y_shape_has_three_branches():
    y = np.zeros((120, 120), dtype=bool)
    y[60:110, 57:63] = True
    for k in range(50):
        y[60 - k:63 - k, 57 - k:63 - k] = True
        y[60 - k:63 - k, 57 + k:63 + k] = True
    length, branches, endpoints, junctions = F.skeleton_stats(y)
    assert (branches, endpoints, junctions) == (3, 3, 1)


@pytest.mark.parametrize("angle_deg", [0, 10, 22.5, 30, 45, 70])
def test_skeleton_length_of_a_digital_line_does_not_depend_on_its_angle(angle_deg):
    from skimage.draw import line

    a = math.radians(angle_deg)
    r1, c1 = int(round(10 + 80 * math.sin(a))), int(round(10 + 80 * math.cos(a)))
    m = np.zeros((120, 120), dtype=bool)
    rr, cc = line(10, 10, r1, c1)
    m[rr, cc] = True
    true = math.hypot(r1 - 10, c1 - 10)
    assert F._branch_length(np.argwhere(m)) == pytest.approx(true, rel=0.02)


def test_convex_hull_and_concavity():
    square = np.array([[0, 0], [0, 2], [2, 2], [2, 0], [1, 1], [0, 1]], dtype=float)
    hull = F.convex_hull(square)
    assert len(hull) == 4
    perim, area = F._polygon_perimeter_area(hull)
    assert perim == pytest.approx(8.0) and abs(area) == pytest.approx(4.0)
    # A cell pinched in the middle (two lobes joined by a neck) is concave.
    m = np.zeros((60, 120), dtype=bool)
    from skimage.draw import disk
    for cc0 in (35, 85):
        rr, cc = disk((30, cc0), 20)
        m[rr, cc] = True
    m[25:35, 35:85] = True
    f = F.strict_morphology(m, None)
    assert f["concave_fraction"] > 0.05
    assert f["solidity"] < 0.9


def test_motion_history_uses_elapsed_time():
    xy = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 30.0]])
    t = np.array([0.0, 20.0, 80.0])  # second step spans a 2-frame gap
    h = F.motion_history(xy, t)
    assert h["hist_speed_last_um_per_hr"] == pytest.approx(20.0)  # 20 um in 1 h, not 2x faster
    assert h["hist_speed_mean_um_per_hr"] == pytest.approx((30.0 + 20.0) / 2)
    assert h["hist_net_rate_um_per_hr"] == pytest.approx(30.0 / (80 / 60))
    with pytest.raises(ValueError):
        F.motion_history(xy[:1], t[:1])


def test_phenotype_is_named_apart_and_finite():
    rng = np.random.default_rng(0)
    m = np.pad(cell_mask(0.1), 12)
    img = 1000 + rng.normal(0, 20, m.shape) - 300 * m + rng.normal(0, 40, m.shape) * m
    f = F.phenotype(img, m)
    assert all(k.startswith("pheno_") for k in f)
    assert all(np.isfinite(v) for v in f.values())
    assert f["pheno_contrast"] < -5  # darker than its ring, in ring-MAD units
    assert not set(f) & set(F.strict_feature_names(None))


def test_standardised_crop_removes_angle_and_mirror():
    a = F.standardised_crop(cell_mask(0.0, wobble=0.05, seed=4))
    for variant in (cell_mask(math.radians(40), wobble=0.05, seed=4),
                    np.fliplr(cell_mask(math.radians(75), wobble=0.05, seed=4))):
        b = F.standardised_crop(variant)
        iou = np.logical_and(a > 0, b > 0).sum() / np.logical_or(a > 0, b > 0).sum()
        assert iou > 0.8
    assert a.shape == (F.CROP_SIZE_PX, F.CROP_SIZE_PX) and a.dtype == np.float32
