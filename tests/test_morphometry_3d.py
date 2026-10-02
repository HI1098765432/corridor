"""3-D measurement on synthetic volumes with known answers (contract §4).

There is no 3-D ground truth in the supplied data, so this is the validation
the contract promises instead: digitised ellipsoids sampled the way confocal
stacks are, 4x more coarsely in Z (2.0 µm) than in XY (0.5 µm).

Accuracy depends on how many Z slices an object occupies, and on where the
slices fall, so every accuracy test runs each shape at four Z phases and
takes its tolerance from the slice count the detection itself reports
(``Detection.z_slices``).  A single fixed placement passed while hiding the
errors of objects a few slices tall.  The tolerances are the envelope
measured over 32 Z phases x 2-3 XY offsets per shape (the table in
``extract_detections_3d``) with a small margin; the placements used here are
points of that grid.

*   Volume: 4/3 pi a b c.  Surface area: Knud Thomsen's approximation
    (p = 1.6075), itself within 1.061 % of the exact ellipsoid area.
    From :data:`MIN_RELIABLE_Z_SLICES` slices, within 3 % and 5 %.
*   Principal axes: the full axes 2a, 2b, 2c within 3 % from 8 slices, 5 %
    from 6, 11 % from 4, and identical to scikit-image's own regionprops
    given the same spacing at any count.
*   Sphericity of a sphere: measured 0.973-0.980 at 2.0 x 0.5 x 0.5 µm and
    0.997 at isotropic 0.5 µm; asserted within 0.05 and 0.01 of 1.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from skimage.measure import regionprops

from corridor.core.detections import (
    MIN_RELIABLE_Z_SLICES,
    detections_to_rows,
    extract_detections_3d,
)

ANISOTROPIC = (2.0, 0.5, 0.5)  # dz, dy, dx in µm
ISOTROPIC = (0.5, 0.5, 0.5)

#: Sub-voxel placements (z, y, x), as fractions of a voxel: four Z phases.
OFFSETS = [
    pytest.param((0.0, 0.17, 0.43), id="z0"),
    pytest.param((0.25, 0.71, 0.05), id="z.25"),
    pytest.param((0.5, 0.17, 0.43), id="z.5"),
    pytest.param((0.75, 0.71, 0.05), id="z.75"),
]


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


#: Semi-axes (a, b, c) in µm along x, y, z before rotation, and the rotation
#: seed. Slice counts are what the four Z phases produce at dz = 2 µm.
SHAPES = [
    pytest.param((8.0, 6.0, 10.0), None, id="ellipsoid"),  # 9-10 slices
    pytest.param((8.0, 6.0, 10.0), 11, id="ellipsoid-rotated"),  # 8-9
    pytest.param((20.0, 4.0, 4.0), 5, id="rod-rotated"),  # 10-11
    pytest.param((24.0, 2.4, 2.4), 3, id="cell-oblique"),  # 16-17, confined-cell shape
    pytest.param((20.0, 4.0, 6.0), None, id="rod-6-slices"),  # 5-6
    pytest.param((20.0, 4.0, 4.0), None, id="rod-4-slices"),  # 3-4
    pytest.param((20.0, 5.0, 2.5), None, id="flat-cell"),  # 2-3: a confined cell's height
]


@functools.lru_cache(maxsize=None)
def measured(semi_axes, seed, offset, spacing=ANISOTROPIC):
    """Labels and the one detection of a placed ellipsoid, computed once."""
    a, b, c = semi_axes
    rot = None if seed is None else rotation(seed)
    labels = ellipsoid_volume(a, b, c, spacing, rot=rot, offset=offset)
    (d,) = extract_detections_3d(labels, 0, spacing_zyx_um=spacing)
    return labels, d


def volume_area_tolerance(slices: int) -> tuple[float, float]:
    """Measured envelope by slice count (volume %, area %), plus margin."""
    if slices >= MIN_RELIABLE_Z_SLICES:
        return 0.03, 0.05  # measured -2.3..+1.8 and -0.2..+4.2
    if slices >= 4:
        return 0.06, 0.05  # -5.3..+3.0 and -0.1..+4.7
    if slices == 3:
        return 0.12, 0.12  # -10.5..+6.9 and -10.2..+6.4
    return 0.22, 0.13  # 2 slices: -20.5..+13.0 and -11.9..+0.4


def axis_tolerance(slices: int) -> float | None:
    """Largest relative error of 2a, 2b, 2c; None below four slices, where
    the Z axis of an object two slices tall is off by up to 48 %."""
    if slices >= 8:
        return 0.03  # measured within 2.6 %
    if slices >= 6:
        return 0.05  # 4.7 %
    if slices >= 4:
        return 0.11  # 10.5 %
    return None


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("semi_axes, seed", SHAPES)
def test_volume_and_surface_area_match_the_analytic_ellipsoid(semi_axes, seed, offset):
    a, b, c = semi_axes
    _, d = measured(semi_axes, seed, offset)
    vol_tol, area_tol = volume_area_tolerance(d.z_slices)
    assert d.volume_um3 == pytest.approx(4 / 3 * math.pi * a * b * c, rel=vol_tol)
    assert d.surface_area_um2 == pytest.approx(thomsen_area(a, b, c), rel=area_tol)


def test_a_confined_cell_sampled_every_2_um_is_below_the_reliable_slice_count():
    """The realistic case: a cell ~5 µm tall at dz = 2 µm occupies 2-3
    planes, so its 3-D values are reported but must not pass as precise."""
    for offset in OFFSETS:
        _, d = measured((20.0, 5.0, 2.5), None, offset.values[0])
        assert d.z_slices in (2, 3)
        assert d.z_slices < MIN_RELIABLE_Z_SLICES
        assert d.volume_um3 is not None and d.surface_area_um2 is not None


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("semi_axes, seed", SHAPES)
def test_principal_axes_are_the_full_ellipsoid_axes(semi_axes, seed, offset):
    a, b, c = semi_axes
    labels, d = measured(semi_axes, seed, offset)
    lengths = d.principal_axis_lengths
    assert d.elongation == pytest.approx(lengths[0] / lengths[1])
    assert d.flatness == pytest.approx(lengths[1] / lengths[2])
    tol = axis_tolerance(d.z_slices)
    if tol is not None:
        expected = sorted((2 * a, 2 * b, 2 * c), reverse=True)
        assert lengths == pytest.approx(expected, rel=tol)

    reference = regionprops(labels, spacing=ANISOTROPIC)[0]
    assert lengths[0] == pytest.approx(reference.axis_major_length, rel=1e-6)
    assert lengths[-1] == pytest.approx(reference.axis_minor_length, rel=1e-6)


@pytest.mark.parametrize("offset", OFFSETS)
def test_a_sphere_is_spherical(offset):
    """Its three axes are held to the slice-count tolerance, not its axis
    ratios: a ratio carries the errors of two axes (flatness read 1.041 at
    Z phase 0, where the 16 µm sphere occupies 7 slices)."""
    _, d = measured((8.0, 8.0, 8.0), None, offset)
    assert d.sphericity == pytest.approx(1.0, abs=0.05)
    assert d.principal_axis_lengths == pytest.approx([16.0] * 3, rel=axis_tolerance(d.z_slices))


def test_a_sphere_is_spherical_at_isotropic_voxels():
    _, d = measured((8.0, 8.0, 8.0), None, (0.31, 0.17, 0.43), ISOTROPIC)
    assert d.sphericity == pytest.approx(1.0, abs=0.01)
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


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("semi_axes, seed", SHAPES)
def test_convex_objects_read_nearly_solid(semi_axes, seed, offset):
    """Hulls of voxel corners read 0.78-0.91 on convex shapes, because the
    staircase of a digitised curve is never convex. The mesh hull measured
    0.906-0.995 from four slices (the low end a thin rod oblique to Z) and
    down to 0.77 for objects two or three slices tall."""
    _, d = measured(semi_axes, seed, offset)
    floor = 0.90 if d.z_slices >= 4 else 0.75
    assert floor <= d.solidity <= 1.0


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
