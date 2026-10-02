"""Dimensional import: T and Z only from metadata that establishes them.

Every file here is synthetic and written with tifffile, the way ImageJ and
OME writers lay them out, because there is no 3-D data anywhere in the
supplied set (contract §0). What is tested is the reading of the metadata,
which is exactly where a Z stack used to become a time-lapse.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from corridor.core.config import CalibrationConfig, ImportConfig
from corridor.core.imaging import (
    SOURCE_IMAGEJ,
    SOURCE_MISSING,
    SOURCE_OME,
    SOURCE_USER,
    AmbiguousAxes,
    UnsupportedStackError,
    effective_z_step,
    load_stack,
    read_metadata,
)


def _ramp(shape) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)


# --------------------------------------------------------------------------
# ImageJ hyperstacks
# --------------------------------------------------------------------------


def test_an_imagej_tzyx_hyperstack_is_read_with_its_z_step(tmp_path):
    data = _ramp((3, 4, 6, 8))
    path = tmp_path / "tzyx.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "TZYX", "spacing": 2.0, "unit": "micron", "finterval": 600.0},
        resolution=(2.0, 2.0),
    )
    meta = read_metadata(path)
    assert meta.axes == "TZYX"
    assert meta.axes_source == SOURCE_IMAGEJ
    assert meta.dimensionality == "3D"
    assert (meta.n_frames, meta.n_slices, meta.height, meta.width) == (3, 4, 6, 8)
    assert meta.shape == meta.axes_shape == (3, 4, 6, 8)
    assert meta.z_step_um.value == pytest.approx(2.0)
    assert meta.z_step_um.source == SOURCE_IMAGEJ
    assert meta.pixel_size_um.value == pytest.approx(0.5)
    assert meta.frame_interval_min.value == pytest.approx(10.0)

    stack = load_stack(path, meta)
    assert stack.shape == (3, 4, 6, 8) and stack.dtype == np.float32
    assert np.array_equal(stack, data)


def test_a_slices_only_stack_is_zyx_not_tyx(tmp_path):
    """ImageJ ``slices=N`` with no ``frames`` is one Z stack, never N frames."""
    data = _ramp((5, 6, 8))
    path = tmp_path / "slices.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "ZYX", "spacing": 1.5, "unit": "um"},
        resolution=(4.0, 4.0),
    )
    meta = read_metadata(path)
    assert meta.axes == "ZYX"
    assert meta.n_frames == 1 and meta.n_slices == 5
    assert meta.axes_shape == (5, 6, 8)
    assert meta.shape == (1, 5, 6, 8)  # what load_stack returns: T[Z]YX
    assert meta.z_step_um.value == pytest.approx(1.5)
    assert meta.pixel_size_um.value == pytest.approx(0.25)
    assert any("no motion" in note for note in meta.notes)
    stack = load_stack(path, meta)
    assert stack.shape == (1, 5, 6, 8)
    assert np.array_equal(stack[0], data)


def test_a_hand_written_imagej_slices_header_with_a_z_spacing_is_z(tmp_path):
    """The header ImageJ writes for a Z stack whose voxel depth was set.

    ImageJ writes ``slices=N`` for every plain stack (an ImagePlus starts as
    c=1, z=N, t=1), so ``slices`` alone is a default, not a statement; the
    ``spacing`` it writes only for a set voxel depth is what makes it Z.
    """
    path = tmp_path / "handmade.tif"
    tifffile.imwrite(
        path, np.zeros((4, 6, 8), np.uint16),
        description="ImageJ=1.54f\nimages=4\nslices=4\nunit=micron\nspacing=2.5\n",
        metadata=None, photometric="minisblack",
    )
    meta = read_metadata(path)
    assert (meta.axes, meta.n_slices, meta.n_frames) == ("ZYX", 4, 1)
    assert meta.z_step_um.value == pytest.approx(2.5)


@pytest.mark.parametrize(
    "header",
    [
        # The adversarial review's probe: a Fiji time-lapse saved as a plain stack.
        "ImageJ=1.54f\nimages=5\nslices=5\nunit=micron\nfinterval=600\nloop=false\n",
        # An image sequence opened and saved: no calibration at all.
        "ImageJ=1.54f\nimages=5\nslices=5\nloop=false\n",
        # A Z spacing *and* a frame interval: the header contradicts itself.
        "ImageJ=1.54f\nimages=5\nslices=5\nunit=micron\nspacing=2.0\nfinterval=600\n",
    ],
    ids=["finterval", "uncalibrated", "both"],
)
def test_imagej_default_slices_do_not_establish_z(tmp_path, header):
    """ImageJ labels every plain stack 'slices', time-lapses included."""
    path = tmp_path / "plain_slices.tif"
    data = _ramp((5, 6, 8))
    tifffile.imwrite(path, data, description=header, metadata=None, photometric="minisblack")
    with tifffile.TiffFile(path) as tf:
        assert tf.series[0].axes == "ZYX"  # tifffile's reading, which is not trusted
    with pytest.raises(AmbiguousAxes) as info:
        read_metadata(path)
    assert info.value.choices[:2] == ("TYX", "ZYX")
    assert "slices" in str(info.value)
    assert info.value.axes_raw == "ZYX"

    meta = read_metadata(path, ImportConfig(axes="TYX"))
    assert (meta.axes, meta.n_frames, meta.axes_source) == ("TYX", 5, SOURCE_USER)
    if "finterval" in header:
        assert meta.frame_interval_min.value == pytest.approx(10.0)
    assert np.array_equal(load_stack(path, meta), data)
    assert read_metadata(path, ImportConfig(axes="ZYX")).axes == "ZYX"


def test_imagej_slices_with_frames_or_hyperstack_stand_as_written(tmp_path):
    """A header that names frames, channels or hyperstack=true was laid out on purpose."""
    path = tmp_path / "hyper.tif"
    tifffile.imwrite(
        path, np.zeros((4, 6, 8), np.uint16),
        description="ImageJ=1.54f\nimages=4\nslices=4\nhyperstack=true\n",
        metadata=None, photometric="minisblack",
    )
    assert read_metadata(path).axes == "ZYX"


def test_an_imagej_stack_without_frames_or_slices_is_ambiguous(tmp_path):
    """An ImageJ header with images=N and neither frames nor slices does not say T or Z."""
    path = tmp_path / "plain_ij.tif"
    tifffile.imwrite(
        path, np.zeros((4, 6, 8), np.uint16),
        description="ImageJ=1.54f\nimages=4\nunit=micron\n",
        metadata=None, photometric="minisblack",
    )
    with pytest.raises(AmbiguousAxes) as info:
        read_metadata(path)
    assert info.value.choices[:2] == ("TYX", "ZYX")
    assert info.value.shape == (4, 6, 8)
    assert isinstance(info.value, UnsupportedStackError)  # older handlers still show it


def test_a_hyperstack_with_channels_reduces_to_the_chosen_channel(tmp_path):
    data = np.zeros((3, 4, 2, 6, 8), np.uint16)
    data[:, :, 1] = 9
    path = tmp_path / "tzcyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TZCYX", "spacing": 2.0, "unit": "um"})
    meta = read_metadata(path, ImportConfig(channel_index=1))
    assert meta.axes == "TZYX" and meta.n_channels == 2 and meta.channel_index == 1
    stack = load_stack(path, meta)
    assert stack.shape == (3, 4, 6, 8) and np.all(stack == 9)


def test_without_a_z_step_the_step_is_missing_not_the_pixel_size(tmp_path):
    data = np.zeros((4, 6, 8), np.uint16)
    path = tmp_path / "nospacing.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "ZYX", "unit": "micron"},
                     resolution=(2.0, 2.0))
    meta = read_metadata(path)
    assert meta.pixel_size_um.known
    assert not meta.z_step_um.known and meta.z_step_um.source == SOURCE_MISSING
    assert any("Z step" in note for note in meta.notes)
    # A user-entered step wins and says so; it never applies to a 2-D file.
    entered = effective_z_step(meta, CalibrationConfig(z_step_um=3.0))
    assert (entered.value, entered.source) == (3.0, SOURCE_USER)
    assert not effective_z_step(meta, CalibrationConfig()).known


def test_a_2d_file_never_has_a_z_step(tmp_path):
    """ImageJ writes ``spacing`` into 2-D files too; it is not a Z step there."""
    data = np.zeros((3, 6, 8), np.uint16)
    path = tmp_path / "tyx_spacing.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TYX", "spacing": 2.0, "unit": "um"})
    meta = read_metadata(path)
    assert meta.dimensionality == "2D" and not meta.z_step_um.known
    assert not effective_z_step(meta, CalibrationConfig(z_step_um=3.0)).known


# --------------------------------------------------------------------------
# OME-TIFF
# --------------------------------------------------------------------------


def test_an_ome_tiff_is_read_from_its_sizes_and_physical_sizes(tmp_path):
    data = _ramp((3, 4, 6, 8))
    path = tmp_path / "tzyx.ome.tif"
    tifffile.imwrite(
        path, data, ome=True,
        metadata={
            "axes": "TZYX",
            "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5, "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 2.0, "PhysicalSizeZUnit": "µm",
            "TimeIncrement": 600.0, "TimeIncrementUnit": "s",
        },
    )
    meta = read_metadata(path)
    assert meta.axes == "TZYX" and meta.axes_source == SOURCE_OME
    assert meta.shape == (3, 4, 6, 8)
    assert meta.z_step_um.value == pytest.approx(2.0)
    assert meta.z_step_um.source == SOURCE_OME
    assert meta.pixel_size_um.value == pytest.approx(0.5)
    assert meta.pixel_size_um.source == SOURCE_OME
    assert meta.frame_interval_min.value == pytest.approx(10.0)
    assert meta.frame_interval_min.source == SOURCE_OME
    assert meta.acquisition["OME SizeZ"] == "4" and meta.acquisition["OME SizeT"] == "3"
    assert np.array_equal(load_stack(path, meta), data)


def test_ome_physical_sizes_are_converted_from_their_units(tmp_path):
    path = tmp_path / "nm.ome.tif"
    tifffile.imwrite(
        path, np.zeros((4, 6, 8), np.uint16), ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 250.0, "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 300.0, "PhysicalSizeYUnit": "nm",
            "PhysicalSizeZ": 0.001, "PhysicalSizeZUnit": "mm",
        },
    )
    meta = read_metadata(path)
    assert meta.axes == "ZYX" and meta.n_slices == 4
    assert meta.pixel_size_um.value == pytest.approx(0.25)
    assert meta.pixel_size_y_um.value == pytest.approx(0.30)
    assert meta.anisotropic_pixels
    assert meta.z_step_um.value == pytest.approx(1.0)


def test_an_ome_z_stack_without_physical_size_z_has_no_step(tmp_path):
    path = tmp_path / "noz.ome.tif"
    tifffile.imwrite(path, np.zeros((4, 6, 8), np.uint16), ome=True,
                     metadata={"axes": "ZYX", "PhysicalSizeX": 0.5})
    meta = read_metadata(path)
    assert meta.axes == "ZYX"
    assert not meta.z_step_um.known


# --------------------------------------------------------------------------
# Ambiguity and the explicit override
# --------------------------------------------------------------------------


def test_an_unlabelled_4d_stack_lists_every_reading(tmp_path):
    path = tmp_path / "q4.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 6, 8), np.uint16), photometric="minisblack")
    with tifffile.TiffFile(path) as tf:
        assert tf.series[0].axes == "QQYX"
    with pytest.raises(AmbiguousAxes) as info:
        read_metadata(path)
    assert {"TZYX", "ZTYX"} <= set(info.value.choices)


def test_the_override_reads_an_ambiguous_file_and_is_recorded(tmp_path):
    data = _ramp((3, 4, 6, 8))
    path = tmp_path / "q4.tif"
    tifffile.imwrite(path, data, photometric="minisblack")
    meta = read_metadata(path, ImportConfig(axes="TZYX"))
    assert (meta.axes, meta.axes_source, meta.axes_used) == ("TZYX", SOURCE_USER, "TZYX")
    assert np.array_equal(load_stack(path, meta), data)

    # An order naming the axes the other way round is moved, not misread.
    swapped = read_metadata(path, ImportConfig(axes="ZTYX"))
    assert swapped.axes == "TZYX" and swapped.shape == (4, 3, 6, 8)
    assert np.array_equal(load_stack(path, swapped), np.transpose(data, (1, 0, 2, 3)))


def test_an_override_that_contradicts_the_metadata_wins_and_says_so(tmp_path):
    """ImageJ users save frames as slices; the user's explicit word is recorded."""
    path = tmp_path / "zyx.tif"
    tifffile.imwrite(path, np.zeros((4, 6, 8), np.uint16), imagej=True, metadata={"axes": "ZYX"})
    meta = read_metadata(path, ImportConfig(axes="TYX"))
    assert meta.axes == "TYX" and meta.axes_source == SOURCE_USER
    assert any("'ZYX'" in note and "import settings" in note for note in meta.notes)


@pytest.mark.parametrize("bad", ["TQYX", "TYX", "ZZYX", ""])
def test_a_malformed_override_is_refused(tmp_path, bad):
    path = tmp_path / "q4.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 6, 8), np.uint16), photometric="minisblack")
    with pytest.raises(UnsupportedStackError):
        read_metadata(path, ImportConfig(axes=bad or "  "))


def test_a_shaped_tifffile_axes_string_is_trusted(tmp_path):
    """An axes string written with the file is metadata that establishes T/Z."""
    path = tmp_path / "shaped.tif"
    tifffile.imwrite(path, np.zeros((2, 3, 6, 8), np.uint16), metadata={"axes": "TZYX"})
    meta = read_metadata(path)
    assert meta.axes == "TZYX" and meta.shape == (2, 3, 6, 8)


# --------------------------------------------------------------------------
# Reopening a saved analysis reads the source as the saved run read it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "write",
    [
        lambda p, d: tifffile.imwrite(p, d, photometric="minisblack"),  # 'QYX'
        lambda p, d: tifffile.imwrite(p, d, imagej=True, metadata={"axes": "ZYX"}),  # 'ZYX'
        lambda p, d: tifffile.imwrite(  # ImageJ plain stack, now ambiguous
            p, d, description="ImageJ=1.54f\nimages=4\nslices=4\n", metadata=None,
            photometric="minisblack",
        ),
    ],
    ids=["unlabelled", "z", "imagej-plain"],
)
def test_a_v1_project_reopens_with_the_frames_it_was_analysed_with(tmp_path, write):
    """1.x read the one non-spatial axis as time; its masks are (T, Y, X)."""
    from corridor.core.imaging import saved_import_config

    data = _ramp((4, 6, 8))
    path = tmp_path / "v1_source.tif"
    write(path, data)
    v1_input = {"path": str(path), "shape_tyx": [4, 6, 8], "axes_reported": "QYX"}
    config = saved_import_config(path, v1_input)
    meta = read_metadata(path, config)
    assert meta.axes == "TYX" and meta.shape == (4, 6, 8)
    assert np.array_equal(load_stack(path, meta), data)


def test_a_v1_time_lapse_needs_no_replay(tmp_path):
    from corridor.core.imaging import saved_import_config

    path = tmp_path / "tyx.tif"
    tifffile.imwrite(path, np.zeros((3, 6, 8), np.uint16), imagej=True, metadata={"axes": "TYX"})
    assert saved_import_config(path, {"shape_tyx": [3, 6, 8]}) is None


def test_a_v2_project_replays_its_recorded_axes_and_channel(tmp_path):
    from corridor.core.imaging import saved_import_config

    data = np.zeros((3, 4, 2, 6, 8), np.uint16)
    data[:, :, 1] = 7
    path = tmp_path / "tzcyx.tif"
    tifffile.imwrite(path, data, photometric="minisblack")  # 'QQQYX': ambiguous alone
    original = read_metadata(path, ImportConfig(axes="TZCYX", channel_index=1))
    config = saved_import_config(path, original.to_dict())
    assert (config.axes, config.channel_index) == ("TZCYX", 1)
    meta = read_metadata(path, config)
    assert meta.axes == "TZYX" and np.all(load_stack(path, meta) == 7)


def test_metadata_dict_names_the_letters_of_its_shape(tmp_path):
    """``shape`` is load_stack's T[Z]YX; ``axes_shape`` goes with ``axes``."""
    path = tmp_path / "zyx.tif"
    tifffile.imwrite(path, np.zeros((5, 6, 8), np.uint16), imagej=True,
                     metadata={"axes": "ZYX", "spacing": 1.0, "unit": "um"})
    d = read_metadata(path).to_dict()
    assert dict(zip(d["axes"], d["axes_shape"])) == {"Z": 5, "Y": 6, "X": 8}
    assert dict(zip(d["stack_axes"], d["shape"])) == {"T": 1, "Z": 5, "Y": 6, "X": 8}


