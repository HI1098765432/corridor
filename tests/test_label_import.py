"""Imported label images: the segmentation route that needs no model.

This is the only way 3-D data is measured while no 3-D model is validated
(contract §2, §4), so the synthetic objects here have known answers: a solid
ellipsoid's volume is 4/3 pi a b c, and its principal axes are 2a, 2b, 2c.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import tifffile

from corridor.core import segmentation as seg
from corridor.core.config import Scale
from corridor.core.imaging import UnsupportedStackError, load_stack, read_metadata
from corridor.core.segmentation import PROVENANCE_IMPORTED, load_label_stack

#: Voxel spacing of the synthetic volume: Z sampled twice as coarsely as XY,
#: as optical stacks routinely are, so an isotropy assumption would show.
Z_UM, XY_UM = 1.0, 0.5
#: Semi-axes in µm (z, y, x): 5 slices, 12 px and 16 px.
SEMI_UM = (5.0, 6.0, 8.0)
SHAPE_ZYX = (13, 30, 44)
CENTRES = [(6.0, 15.0, 18.0), (6.0, 15.0, 24.0)]  # moves 6 px in x


def _ellipsoid_labels(shape, centre, semi_um, spacing, label=1) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    inside = sum(
        ((g - c) * s / a) ** 2 for g, c, s, a in zip((zz, yy, xx), centre, spacing, semi_um)
    ) <= 1.0
    return np.where(inside, label, 0).astype(np.uint16)


@pytest.fixture
def volume_pair(tmp_path):
    """A TZYX ImageJ image with a Z step, and its labels saved without metadata."""
    spacing = (Z_UM, XY_UM, XY_UM)
    labels = np.stack(
        [_ellipsoid_labels(SHAPE_ZYX, c, SEMI_UM, spacing, label=7) for c in CENTRES]
    )
    image = (100 + 900 * (labels > 0)).astype(np.uint16)
    image_path = tmp_path / "cells.tif"
    tifffile.imwrite(
        image_path, image, imagej=True,
        metadata={"axes": "TZYX", "spacing": Z_UM, "unit": "micron", "finterval": 600.0},
        resolution=(1 / XY_UM, 1 / XY_UM),
    )
    label_path = tmp_path / "cells_labels.tif"
    tifffile.imwrite(label_path, labels, photometric="minisblack")  # axes 'QQYX'
    return image_path, label_path, labels


def _scale(meta, *, with_z=True) -> Scale:
    return Scale.from_values(
        meta.pixel_size_um.value,
        meta.frame_interval_min.value,
        z_step_um=meta.z_step_um.value if with_z else None,
    )


def test_imported_3d_labels_measure_the_synthetic_ellipsoid(volume_pair):
    image_path, label_path, labels = volume_pair
    meta = read_metadata(image_path)
    assert meta.axes == "TZYX" and meta.z_step_um.value == pytest.approx(Z_UM)
    stack = load_stack(image_path, meta)

    out = load_label_stack(label_path, meta.axes, image=stack, scale=_scale(meta))

    assert out.provenance == PROVENANCE_IMPORTED
    assert out.labels_sha256 == hashlib.sha256(label_path.read_bytes()).hexdigest()
    assert out.model is None and out.model_manifest() is None and out.passes_per_frame == 0
    assert out.dimensionality == "3D"
    assert out.masks.shape == stack.shape and np.array_equal(out.masks, labels)
    assert [d.kept_count for d in out.diagnostics] == [1, 1]
    assert any("cells_labels.tif" in note for note in out.notes)

    analytic = 4.0 / 3.0 * math.pi * SEMI_UM[0] * SEMI_UM[1] * SEMI_UM[2]
    assert len(out.detections) == 2
    for det, centre in zip(out.detections, CENTRES):
        assert det.label == 7
        assert (det.z, det.y, det.x) == pytest.approx(centre)
        assert det.volume_vox == float((labels[det.frame] > 0).sum())
        # The measurement is the voxel count at the real spacing...
        assert det.volume_um3 == pytest.approx(det.volume_vox * Z_UM * XY_UM * XY_UM)
        # ...and that is the ellipsoid's volume, to the digitisation error
        # (measured when this was written: -1.15 %; axes within 1.9 %).
        assert det.volume_um3 == pytest.approx(analytic, rel=0.03)
        assert det.principal_axis_lengths == pytest.approx(
            sorted((2 * a for a in SEMI_UM), reverse=True), rel=0.03
        )
        assert det.surface_area_um2 is not None and det.mean_intensity == pytest.approx(1000.0)


def test_without_a_z_step_imported_objects_are_counted_in_voxels(volume_pair):
    image_path, label_path, labels = volume_pair
    meta = read_metadata(image_path)
    stack = load_stack(image_path, meta)
    out = load_label_stack(label_path, meta.axes, image=stack, scale=_scale(meta, with_z=False))
    det = out.detections[0]
    assert det.volume_vox == float((labels[0] > 0).sum())
    assert det.volume_um3 is None and det.surface_area_um2 is None
    assert det.principal_axis_lengths is None
    assert any("voxels only" in note for note in out.notes)


def test_imported_2d_labels_become_2d_detections(tmp_path):
    labels = np.zeros((3, 40, 30), np.uint16)
    for t in range(3):
        labels[t, 5 + 4 * t:30 + 4 * t, 10:16] = 1
        labels[t, 2:8, 22:28] = 2
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, labels, imagej=True, metadata={"axes": "TYX"})
    image = np.where(labels > 0, 500.0, 50.0).astype(np.float32)

    out = load_label_stack(path, "TYX", image=image)
    assert out.dimensionality == "2D" and out.masks.shape == (3, 40, 30)
    assert len(out.detections) == 6
    assert all(d.z is None for d in out.detections)
    cell = [d for d in out.detections if d.label == 1]
    assert [d.y for d in cell] == pytest.approx([17.0, 21.0, 25.0])
    # Imported objects are measured as they are: the 6-px-wide square that
    # the 20 px extent filter would drop from a model's output stays.
    assert sum(d.label == 2 for d in out.detections) == 3


def test_labels_of_another_shape_are_refused(volume_pair):
    image_path, label_path, _ = volume_pair
    meta = read_metadata(image_path)
    stack = load_stack(image_path, meta)
    with pytest.raises(UnsupportedStackError, match="must match the image"):
        load_label_stack(label_path, meta.axes, image=stack[:1])
    with pytest.raises(UnsupportedStackError, match="must match the image"):
        load_label_stack(label_path, meta.axes, expected_shape=(2, 13, 30, 43))


def test_labels_that_say_z_are_not_matched_to_time(tmp_path):
    """Same pixel count, different meaning: a Z stack of labels is not 4 frames."""
    labels = np.zeros((4, 6, 8), np.uint16)
    path = tmp_path / "zlabels.tif"
    tifffile.imwrite(path, labels, imagej=True, metadata={"axes": "ZYX"})
    with pytest.raises(UnsupportedStackError, match="must match the image"):
        load_label_stack(path, "TYX", expected_shape=(4, 6, 8))


def test_an_explicit_label_axis_order_overrides_the_label_file(tmp_path):
    labels = np.zeros((4, 6, 8), np.uint16)
    labels[:, 1:5, 2:4] = 3
    path = tmp_path / "cyx_labels.tif"
    tifffile.imwrite(path, labels, imagej=True)  # tifffile records this as 4 channels
    with pytest.raises(UnsupportedStackError, match="4 channels"):
        load_label_stack(path, "TYX", expected_shape=(4, 6, 8))
    out = load_label_stack(path, "TYX", expected_shape=(4, 6, 8), label_axes="TYX")
    assert out.masks.shape == (4, 6, 8) and len(out.detections) == 4


@pytest.mark.parametrize(
    "values, message",
    [
        (np.array([[[0, 1], [1, 0]]], bool), "binary mask"),
        (np.array([[[0.0, 1.5], [1.0, 0.0]]], np.float32), "non-integer"),
        (np.array([[[0, -1], [1, 0]]], np.int16), "negative"),
    ],
    ids=["binary", "fractional", "negative"],
)
def test_what_is_not_a_label_image_is_refused(values, message):
    with pytest.raises(UnsupportedStackError, match=message):
        seg._as_labels(values, "labels.tif")


def test_whole_number_float_labels_are_accepted(tmp_path):
    """Fiji saves label images as 32-bit float."""
    labels = np.zeros((2, 30, 20), np.float32)
    labels[:, 3:28, 5:11] = 4.0
    path = tmp_path / "float_labels.tif"
    tifffile.imwrite(path, labels, imagej=True, metadata={"axes": "TYX"})
    out = load_label_stack(path, "TYX", expected_shape=(2, 30, 20))
    assert out.masks.dtype == np.int32 and {d.label for d in out.detections} == {4}


def test_without_the_image_a_label_of_the_wrong_dimensionality_is_refused(tmp_path):
    """The label file's own metadata says Z stack; the image is a 2-D movie."""
    path = tmp_path / "labels.tif"
    tifffile.imwrite(
        path, np.zeros((2, 3, 6, 8), np.uint16), imagej=True, metadata={"axes": "TZYX"}
    )
    with pytest.raises(UnsupportedStackError, match="dimensions"):
        load_label_stack(path, "TYX")


def test_label_import_can_be_cancelled(volume_pair):
    image_path, label_path, _ = volume_pair
    meta = read_metadata(image_path)
    stack = load_stack(image_path, meta)
    seen = []

    def progress(done, total):
        seen.append((done, total))
        return False

    with pytest.raises(KeyboardInterrupt):
        load_label_stack(label_path, meta.axes, image=stack, progress=progress)
    assert seen == [(1, 2)]


def test_the_two_argument_interface_call_works_and_checks_dimensionality(volume_pair):
    """``load_label_stack(path, axes)`` as the interface names it.

    Without the image there are no sizes to compare, so the dimensionality is
    checked and the missing comparison is recorded rather than skipped quietly.
    """
    _, label_path, labels = volume_pair
    out = load_label_stack(label_path, "TZYX")
    assert out.masks.shape == labels.shape and out.dimensionality == "3D"
    assert any("not compared with the image's size" in note for note in out.notes)
    with pytest.raises(UnsupportedStackError):
        load_label_stack(label_path, "TYX")  # a 4-D label file for a 2-D movie


def test_a_label_movie_saved_by_fiji_as_a_plain_stack_is_not_a_z_stack(tmp_path):
    """Fiji writes slices=N for a plain stack; the labels follow the image's T."""
    labels = np.zeros((4, 32, 32), np.uint16)
    for t in range(4):
        labels[t, 10:22, 4 + 2 * t:16 + 2 * t] = 3
    path = tmp_path / "fiji_labels.tif"
    tifffile.imwrite(
        path, labels, description="ImageJ=1.54f\nimages=4\nslices=4\nloop=false\n",
        metadata=None, photometric="minisblack",
    )
    out = load_label_stack(path, "TYX", expected_shape=(4, 32, 32))
    assert out.dimensionality == "2D" and len(out.detections) == 4
    assert [d.frame for d in out.detections] == [0, 1, 2, 3]
