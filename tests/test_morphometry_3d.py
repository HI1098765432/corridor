"""3-D measurement on synthetic volumes with known answers (contract §4).

There is no 3-D ground truth in the supplied data, so this is the validation
the contract promises instead: digitised ellipsoids sampled the way confocal
stacks are, 4x more coarsely in Z (2.0 µm) than in XY (0.5 µm).

Reference values and tolerances:

*   Volume: 4/3 pi a b c. Voxel counting measured within 1.6 % on these
    shapes; asserted within 3 %.
*   Surface area: Knud Thomsen's approximation (p = 1.6075), itself within
    1.061 % of the exact ellipsoid area. The volume-matched mesh measured
    +2.0 to +3.5 %; asserted within 5 %.
*   Sphericity of a sphere: measured 0.974 at 2.0 x 0.5 x 0.5 µm and 0.997
    at isotropic 0.5 µm; asserted within 0.05 and 0.01 of 1.
*   Principal axes: the full axes 2a, 2b, 2c within 2 % (4 % for an axis
    only four slices long), and identical to scikit-image's own regionprops
    given the same spacing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.measure import regionprops

from corridor.core.detections import detections_to_rows, extract_detections_3d

ANISOTROPIC = (2.0, 0.5, 0.5)  # dz, dy, dx in µm
ISOTROPIC = (0.5, 0.5, 0.5)


def thomsen_area(a: float, b: float, c: float, p: float = 1.6075) -> float:
    return 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)


def rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q


def ellipsoid_volume(a, b, c, spacing, *, rot=None, offset=(0.31, 0.17, 0.43), margin=3):
    """Labels of one solid ellipsoid (semi-axes a, b, c in µm along x, y, z
    before rotation), centred off the voxel grid so no symmetry helps."""
    reach = max(a, b, c)
    half = [int(math.ceil(reach / s)) + margin for s in spacing]
    axes = [(np.arange(-n, n + 1) + o) * s for n, o, s in zip(half, offset, spacing)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    if rot is not None:
        pts = pts @ rot
    inside = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2 + (pts[:, 2] / c) ** 2 <= 1.0
    return inside.reshape(zz.shape).astype(np.int32)


CASES = [
    pytest.param((8.0, 6.0, 10.0), None, id="ellipsoid"),
    pytest.param((8.0, 6.0, 10.0), 11, id="ellipsoid-rotated"),
    pytest.param((20.0, 4.0, 4.0), None, id="rod"),
    pytest.param((20.0, 4.0, 4.0), 5, id="rod-rotated"),
]


@pytest.mark.parametrize("semi_axes, seed", CASES)
def test_volume_and_surface_area_match_the_analytic_ellipsoid(semi_axes, seed):
    a, b, c = semi_axes
    labels = ellipsoid_volume(a, b, c, ANISOTROPIC, rot=None if seed is None else rotation(seed))
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=ANISOTROPIC)
    assert d.volume_um3 == pytest.approx(4 / 3 * math.pi * a * b * c, rel=0.03)
    assert d.surface_area_um2 == pytest.approx(thomsen_area(a, b, c), rel=0.05)


@pytest.mark.parametrize("semi_axes, seed", CASES)
def test_principal_axes_are_the_full_ellipsoid_axes(semi_axes, seed):
    a, b, c = semi_axes
    labels = ellipsoid_volume(a, b, c, ANISOTROPIC, rot=None if seed is None else rotation(seed))
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=ANISOTROPIC)
    expected = sorted((2 * a, 2 * b, 2 * c), reverse=True)
    # The unrotated rod's 8 µm Z axis spans four 2 µm slices; digitising it
    # measured 3.2 % long. Every axis sampled more finely is within 2 %.
    tol = 0.04 if 2 * c / ANISOTROPIC[0] <= 4 else 0.02
    assert d.principal_axis_lengths == pytest.approx(expected, rel=tol)
    assert d.elongation == pytest.approx(expected[0] / expected[1], rel=tol + 0.01)
    assert d.flatness == pytest.approx(expected[1] / expected[2], rel=tol + 0.01)

    reference = regionprops(labels, spacing=ANISOTROPIC)[0]
    assert d.principal_axis_lengths[0] == pytest.approx(reference.axis_major_length, rel=1e-6)
    assert d.principal_axis_lengths[-1] == pytest.approx(reference.axis_minor_length, rel=1e-6)


@pytest.mark.parametrize(
    "spacing, tolerance",
    [pytest.param(ANISOTROPIC, 0.05, id="dz=2.0"), pytest.param(ISOTROPIC, 0.01, id="isotropic")],
)
def test_a_sphere_is_spherical(spacing, tolerance):
    labels = ellipsoid_volume(8.0, 8.0, 8.0, spacing)
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=spacing)
    assert d.sphericity == pytest.approx(1.0, abs=tolerance)
    assert d.elongation == pytest.approx(1.0, abs=0.02)
    assert d.flatness == pytest.approx(1.0, abs=0.02)


def test_a_rod_is_far_from_spherical():
    (d,) = extract_detections_3d(ellipsoid_volume(20.0, 4.0, 4.0, ANISOTROPIC), 0,
                                 spacing_zyx_um=ANISOTROPIC)
    (sphere,) = extract_detections_3d(ellipsoid_volume(8.0, 8.0, 8.0, ANISOTROPIC), 0,
                                      spacing_zyx_um=ANISOTROPIC)
    assert d.sphericity < sphere.sphericity - 0.2
    assert d.elongation > 4.0


def test_without_spacing_nothing_physical_is_invented():
    labels = ellipsoid_volume(8.0, 6.0, 10.0, ANISOTROPIC)
    (d,) = extract_detections_3d(labels, 0)
    assert d.volume_vox == int((labels > 0).sum())
    for name in ("volume_um3", "surface_area_um2", "principal_axis_lengths",
                 "elongation", "flatness", "sphericity"):
        assert getattr(d, name) is None, name
    # Ratios survive per-axis scaling, so they are still reported.
    assert 0.9 < d.solidity <= 1.0
    assert 0 < d.extent_fraction < 1


def test_spacing_must_be_three_positive_numbers():
    labels = ellipsoid_volume(4.0, 4.0, 4.0, ISOTROPIC)
    for bad in ((1.0, 1.0), (0.0, 1.0, 1.0), (1.0, float("nan"), 1.0)):
        with pytest.raises(ValueError):
            extract_detections_3d(labels, 0, spacing_zyx_um=bad)


# --------------------------------------------------------------------------
# Solidity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("semi_axes, seed", CASES)
def test_convex_objects_read_nearly_solid(semi_axes, seed):
    """Hulls of voxel corners read 0.78-0.91 here, because the staircase of
    a digitised curve is never convex; the mesh hull reads 0.93-0.99."""
    a, b, c = semi_axes
    labels = ellipsoid_volume(a, b, c, ANISOTROPIC, rot=None if seed is None else rotation(seed))
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=ANISOTROPIC)
    assert 0.92 <= d.solidity <= 1.0


def test_a_tunnel_through_a_cube_has_the_right_solidity():
    """A 20^3 cube with a 10 x 10 tunnel: 6000 of 8000 voxels -> 0.75."""
    labels = np.zeros((24, 24, 24), np.int32)
    labels[2:22, 2:22, 2:22] = 1
    labels[2:22, 7:17, 7:17] = 0
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=ISOTROPIC)
    assert d.volume_vox == 6000
    assert d.solidity == pytest.approx(0.75, abs=0.02)


def test_degenerate_objects_do_not_crash():
    """A single voxel and a one-slice sheet: no coplanar hull, no NaN."""
    labels = np.zeros((5, 12, 12), np.int32)
    labels[2, 2, 2] = 1
    labels[3, 5:10, 5:10] = 2  # one slice thick
    dets = {d.label: d for d in extract_detections_3d(labels, 0, spacing_zyx_um=ANISOTROPIC)}
    assert dets[1].volume_vox == 1 and dets[1].volume_um3 == pytest.approx(0.5)
    assert dets[1].principal_axis_lengths == (0.0, 0.0, 0.0)
    assert dets[1].elongation is None and dets[1].flatness is None
    for d in dets.values():
        assert math.isfinite(d.solidity)


# --------------------------------------------------------------------------
# Geometry bookkeeping
# --------------------------------------------------------------------------


@pytest.fixture
def two_cells():
    labels = np.zeros((10, 40, 50), np.int32)
    labels[0:4, 5:15, 5:25] = 1  # touches the first slice
    labels[3:7, 20:30, 25:45] = 2  # interior
    intensity = np.random.default_rng(0).uniform(100, 200, labels.shape)
    return labels, intensity


def test_bbox_centroid_and_footprint(two_cells):
    labels, _ = two_cells
    dets = {d.label: d for d in extract_detections_3d(labels, 4)}
    d = dets[2]
    assert d.frame == 4 and d.ndim == 3
    assert d.bbox == (3, 20, 25, 7, 30, 45)
    assert (d.x, d.y, d.z) == pytest.approx((34.5, 24.5, 4.5))
    assert np.array_equal(d.position, [34.5, 24.5, 4.5])
    assert d.area_px == 10 * 20  # the XY footprint
    assert d.extent_px == 20
    assert d.size == d.volume_vox == 4 * 10 * 20
    assert d.extent_fraction == pytest.approx(1.0)


def test_touching_the_top_or_bottom_slice_is_touching_the_border(two_cells):
    labels, _ = two_cells
    dets = {d.label: d for d in extract_detections_3d(labels, 0)}
    assert dets[1].touches_border is True
    assert dets[2].touches_border is False


def test_mean_intensity_in_3d(two_cells):
    labels, intensity = two_cells
    for d in extract_detections_3d(labels, 0, intensity=intensity):
        assert d.mean_intensity == pytest.approx(intensity[labels == d.label].mean())
        assert d.median_intensity == pytest.approx(np.median(intensity[labels == d.label]))
    with pytest.raises(ValueError):
        extract_detections_3d(labels, 0, intensity=intensity[:, :, :-1])


def test_mask_crop_round_trips_in_3d(two_cells):
    labels, _ = two_cells
    for d in extract_detections_3d(labels, 0):
        z0, r0, c0, z1, r1, c1 = d.bbox
        assert d.mask_crop.shape == (z1 - z0, r1 - r0, c1 - c0)
        assert np.array_equal(d.full_mask(labels.shape), labels == d.label)


def test_rows_carry_z_and_split_the_axes(two_cells):
    labels, _ = two_cells
    dets = extract_detections_3d(labels, 0, spacing_zyx_um=ANISOTROPIC)
    rows = detections_to_rows(dets, pixel_size_um=0.5, z_step_um=2.0)
    d, row = dets[1], rows[1]
    assert "mask_crop" not in row and "principal_axis_lengths" not in row
    assert row["z"] == d.z and row["z_um"] == pytest.approx(d.z * 2.0)
    assert (row["bbox_min_z"], row["bbox_max_z"]) == (3, 7)
    assert row["principal_axis_1_um"] == d.principal_axis_lengths[0]
    assert row["volume_um3"] == pytest.approx(4 * 10 * 20 * 0.5)
    assert row["area_um2"] == pytest.approx(10 * 20 * 0.25)  # footprint, a true area


def test_rows_without_a_z_step_have_no_z_um(two_cells):
    labels, _ = two_cells
    dets = extract_detections_3d(labels, 0)
    rows = detections_to_rows(dets, pixel_size_um=0.5)
    assert all("z_um" not in r for r in rows)
    assert all(r["volume_um3"] is None for r in rows)
    late = detections_to_rows(dets, pixel_size_um=0.5, z_step_um=2.0)
    assert late[1]["volume_um3"] == pytest.approx(4 * 10 * 20 * 0.5 * 0.5 * 2.0)


def test_an_empty_volume_has_no_detections():
    assert extract_detections_3d(np.zeros((3, 4, 5), np.int32), 0) == []
    with pytest.raises(ValueError):
        extract_detections_3d(np.zeros((4, 5), np.int32), 0)
