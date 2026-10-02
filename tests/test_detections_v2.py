"""2-D morphology, the detection mask, and the v1 compatibility of Detection.

Tolerances are the measured behaviour of the estimators on digitised shapes,
not round numbers: Crofton perimeters of disks and ellipses read -3.1 to
+1.9 % of the exact value over orientation and sub-pixel placement, and a
pixel rectangle -2.9 to -4.0 % of its pixel-centre outline.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from corridor.core.detections import (
    Detection,
    detections_to_rows,
    extract_detections,
)

#: Every key a v1 row carried. They must all survive, unchanged in meaning.
V1_ROW_KEYS = {
    "frame", "label", "x", "y", "area_px", "extent_px", "eccentricity",
    "orientation_rad", "major_axis_px", "minor_axis_px", "solidity",
    "touches_border", "channel", "source", "confidence",
    "bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y",
}


def ramanujan_perimeter(a: float, b: float) -> float:
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def ellipse_mask(shape, centre, a, b, theta) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(float)
    dx, dy = xx - centre[0], yy - centre[1]
    c, s = math.cos(theta), math.sin(theta)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


@pytest.fixture
def scene():
    """A rectangle, a tilted ellipse, a disk and a cell touching the border."""
    labels = np.zeros((120, 160), np.int32)
    labels[10:50, 10:30] = 1  # 40 x 20 px rectangle
    labels[ellipse_mask(labels.shape, (80.3, 30.2), 20.0, 10.0, 0.4)] = 2
    labels[ellipse_mask(labels.shape, (125.4, 80.1), 20.0, 20.0, 0.0)] = 3
    labels[100:120, 60:70] = 4  # cut by the bottom edge
    intensity = (np.arange(labels.size, dtype=float).reshape(labels.shape) % 97) + 5.0
    return labels, intensity


def by_label(dets):
    return {d.label: d for d in dets}


# --------------------------------------------------------------------------
# 2-D morphology against known answers
# --------------------------------------------------------------------------


def test_rectangle_morphology(scene):
    labels, _ = scene
    d = by_label(extract_detections(labels, 0))[1]
    h, w = 40, 20
    assert d.area_px == h * w
    assert d.extent_fraction == pytest.approx(1.0)
    assert d.convex_area_px == pytest.approx(h * w)
    assert d.solidity == pytest.approx(1.0)
    assert d.equivalent_diameter_px == pytest.approx(math.sqrt(4 * h * w / math.pi))
    # regionprops' axis lengths of a pixel rectangle: 4 * sqrt((n^2 - 1) / 12)
    assert d.aspect_ratio == pytest.approx(math.sqrt((h * h - 1) / (w * w - 1)), rel=1e-9)
    assert d.perimeter_px == pytest.approx(2 * (h + w) - 4, rel=0.04)
    expected_circ = 4 * math.pi * d.area_px / d.perimeter_px**2
    assert d.circularity == pytest.approx(expected_circ)


def test_ellipse_perimeter_and_circularity(scene):
    labels, _ = scene
    d = by_label(extract_detections(labels, 0))[2]
    a, b = 20.0, 10.0
    true_p = ramanujan_perimeter(a, b)
    true_circ = 4 * math.pi * (math.pi * a * b) / true_p**2  # 0.841
    assert d.perimeter_px == pytest.approx(true_p, rel=0.035)
    assert d.circularity == pytest.approx(true_circ, abs=0.03)
    assert d.aspect_ratio == pytest.approx(a / b, rel=0.03)
    assert 0 < d.circularity < 1


def test_a_disk_reads_circular_whatever_its_size(scene):
    """The 4-neighbour perimeter gives this r = 20 disk 0.945 (0.927 at
    r = 30); Crofton gives 1.007 before clipping."""
    labels, _ = scene
    d = by_label(extract_detections(labels, 0))[3]
    assert 0.97 <= d.circularity <= 1.0
    assert d.aspect_ratio == pytest.approx(1.0, abs=0.02)


def test_circularity_is_clipped_to_one():
    """Small digitised disks are where 4 pi A / P^2 overshoots 1."""
    raws = []
    for radius in (2.0, 3.0, 4.0, 5.0):
        labels = ellipse_mask((16, 16), (7.6, 7.3), radius, radius, 0.0).astype(np.int32)
        d = extract_detections(labels, 0)[0]
        raw = 4 * math.pi * d.area_px / d.perimeter_px**2
        raws.append(raw)
        assert d.circularity == pytest.approx(min(1.0, raw))
        assert 0.0 <= d.circularity <= 1.0
    assert max(raws) > 1.0  # the clip was exercised, not just declared


def test_intensity_statistics_use_the_object_pixels_only(scene):
    labels, intensity = scene
    for d in extract_detections(labels, 0, intensity=intensity):
        pixels = intensity[labels == d.label]
        assert d.mean_intensity == pytest.approx(pixels.mean())
        assert d.median_intensity == pytest.approx(np.median(pixels))


def test_no_intensity_image_means_no_intensity_values(scene):
    labels, _ = scene
    for d in extract_detections(labels, 0):
        assert d.mean_intensity is None and d.median_intensity is None


def test_a_one_pixel_object_does_not_invent_shape():
    labels = np.zeros((5, 5), np.int32)
    labels[2, 2] = 1
    d = extract_detections(labels, 0)[0]
    assert d.area_px == 1
    assert d.solidity == 1.0
    assert d.aspect_ratio is None  # minor axis 0: no ratio, not infinity
    assert d.eccentricity == 0.0


def test_a_3d_label_image_is_refused_with_a_pointer():
    with pytest.raises(ValueError, match="extract_detections_3d"):
        extract_detections(np.ones((3, 4, 4), np.int32), 0)


# --------------------------------------------------------------------------
# The mask crop
# --------------------------------------------------------------------------


def test_mask_crop_round_trips_to_the_label(scene):
    labels, _ = scene
    for d in extract_detections(labels, 0):
        assert d.mask_crop.dtype == bool
        r0, c0, r1, c1 = d.bbox
        assert d.mask_crop.shape == (r1 - r0, c1 - c0)
        assert int(d.mask_crop.sum()) == int(d.area_px)
        assert np.array_equal(d.full_mask(labels.shape), labels == d.label)


def test_a_detection_without_a_mask_says_so():
    d = Detection(0, 1, 1.0, 1.0, 4.0, (0, 0, 2, 2), 2, 0.0, 0.0, 2.0, 2.0, 1.0, True)
    with pytest.raises(ValueError, match="no mask"):
        d.full_mask((4, 4))


def test_rows_never_carry_the_mask(scene):
    labels, intensity = scene
    dets = extract_detections(labels, 0, intensity=intensity)
    for d in dets:
        row = d.to_row()
        assert "mask_crop" not in row
        assert "bbox" not in row and "principal_axis_lengths" not in row
        assert not any(isinstance(v, np.ndarray) for v in row.values())
    for row in detections_to_rows(dets, pixel_size_um=0.5, frame_interval_min=20.0):
        assert "mask_crop" not in row


def test_mask_is_left_out_of_equality_and_repr(scene):
    labels, _ = scene
    d = by_label(extract_detections(labels, 0))[1]
    twin = Detection(**{f.name: getattr(d, f.name) for f in fields(d)})
    twin.mask_crop = ~d.mask_crop
    assert twin == d  # would raise "truth value of an array" if compared
    assert "mask_crop" not in repr(d)


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def test_rows_keep_every_v1_key_and_add_the_new_ones(scene):
    labels, intensity = scene
    d = by_label(extract_detections(labels, 0, intensity=intensity))[1]
    row = d.to_row()
    assert V1_ROW_KEYS <= set(row)
    assert (row["bbox_min_x"], row["bbox_min_y"], row["bbox_max_x"], row["bbox_max_y"]) == (
        10, 10, 30, 50,
    )
    for key in ("perimeter_px", "circularity", "aspect_ratio", "extent_fraction",
                "convex_area_px", "equivalent_diameter_px", "mean_intensity",
                "median_intensity", "z", "bbox_min_z", "bbox_max_z",
                "principal_axis_1_um", "principal_axis_2_um", "principal_axis_3_um"):
        assert key in row
    assert row["z"] is None and row["bbox_min_z"] is None
    assert row["principal_axis_1_um"] is None


def test_physical_columns_follow_the_pixel_size(scene):
    labels, _ = scene
    d = by_label(extract_detections(labels, 0))[1]
    px = 0.5
    row = detections_to_rows([d], pixel_size_um=px, frame_interval_min=20.0)[0]
    assert row["area_um2"] == pytest.approx(d.area_px * px * px)
    assert row["major_axis_um"] == pytest.approx(d.major_axis_px * px)
    assert row["minor_axis_um"] == pytest.approx(d.minor_axis_px * px)
    assert row["perimeter_um"] == pytest.approx(d.perimeter_px * px)
    assert row["equivalent_diameter_um"] == pytest.approx(d.equivalent_diameter_px * px)
    assert row["convex_area_um2"] == pytest.approx(d.convex_area_px * px * px)
    assert "z_um" not in row  # a 2-D detection has no Z, calibrated or not
    bare = detections_to_rows([d])[0]
    assert "area_um2" not in bare and "perimeter_um" not in bare


# --------------------------------------------------------------------------
# v1 compatibility
# --------------------------------------------------------------------------


def test_a_v1_constructed_detection_is_still_valid():
    """conftest.make_detection, tracking.py's probe and selftest build these."""
    d = Detection(
        frame=2, label=7, x=10.0, y=20.0, area_px=800.0, bbox=(0, 5, 90, 16),
        extent_px=90, eccentricity=0.99, orientation_rad=0.0, major_axis_px=90.0,
        minor_axis_px=11.0, solidity=0.95, touches_border=False, channel=0,
    )
    assert d.ndim == 2
    assert np.array_equal(d.position, [10.0, 20.0])
    assert np.array_equal(d.position, d.xy)
    assert d.size == 800.0
    assert d.z_slices is None
    assert d.mask_crop is None and d.perimeter_px is None and d.z is None
    assert V1_ROW_KEYS <= set(d.to_row())


def test_position_and_size_switch_with_dimensionality():
    d = Detection(0, 1, 1.0, 2.0, 30.0, (0, 0, 1, 4, 1, 1), 1, 0.0, 0.0, 1.0, 1.0, 1.0, False,
                  z=3.5, volume_vox=120.0)
    assert d.ndim == 3
    assert np.array_equal(d.position, [1.0, 2.0, 3.5])
    assert d.size == 120.0  # volume, not the footprint area
    assert d.z_slices == 4


def test_asdict_is_not_the_row_form(scene):
    """asdict copies the mask like any field; to_row is the tabular form."""
    from dataclasses import asdict

    d = extract_detections(scene[0], 0)[0]
    copied = asdict(d)["mask_crop"]
    assert np.array_equal(copied, d.mask_crop) and copied is not d.mask_crop
    assert "mask_crop" not in d.to_row()
