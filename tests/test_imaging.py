"""Reading TIFFs and interpreting microscopy metadata."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from corridor.core.imaging import (
    SOURCE_IMAGEJ,
    SOURCE_MISSING,
    SOURCE_ND2_INFO,
    SOURCE_TIFF_TAG,
    SOURCE_USER,
    AmbiguousAxes,
    UnsupportedStackError,
    load_stack,
    parse_info_block,
    parse_source_frames,
    read_metadata,
)
from corridor.core.config import CalibrationConfig, ImportConfig
from corridor.core.pipeline import effective_calibration

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, SAMPLE_DIR, requires_samples

# The five supplied files and the shape each one actually contains, as opposed
# to the SizeT = 54 inherited from the parent acquisition.
EXPECTED = {
    "052924_1.tif": (20, 327, 525),
    "052924_2.tif": (20, 327, 525),
    "052924_t1.tif": (18, 324, 90),
    "052924_t2_empty.tif": (5, 324, 90),
    "052924_t3_dual.tif": (11, 324, 90),
}

EXPECTED_SOURCE_FRAMES = {
    "052924_t1.tif": (1, 18),
    "052924_t2_empty.tif": (23, 27),
    "052924_t3_dual.tif": (42, 52),
}


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def test_info_block_parsing():
    info = parse_info_block(
        "SizeT = 54\n dCalibration = 0.467060342995564\n"
        "sObjective = Plan Fluor 10x Ph1 DLL\njunk line\n"
    )
    assert info["SizeT"] == "54"
    assert info["dCalibration"] == "0.467060342995564"
    assert info["sObjective"] == "Plan Fluor 10x Ph1 DLL"


def test_source_frame_labels():
    labels = [f"t:{i}/54 - acquisition.nd2 (series 01)" for i in range(42, 53)]
    frames, total = parse_source_frames(labels)
    assert frames == list(range(42, 53))
    assert total == 54


def test_source_frames_absent_when_labels_are_not_time_labels():
    frames, total = parse_source_frames(["channel 1", "channel 2"])
    assert frames is None and total is None


# --------------------------------------------------------------------------
# Synthetic TIFFs
# --------------------------------------------------------------------------


def test_time_first_stack_is_read_unchanged(tmp_path):
    data = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    path = tmp_path / "tyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TYX"})
    meta = read_metadata(path)
    assert meta.shape == (3, 5, 7)
    assert np.array_equal(load_stack(path, meta), data)


def test_a_single_plane_is_one_frame(tmp_path):
    data = np.zeros((6, 8), dtype=np.uint16)
    path = tmp_path / "yx.tif"
    tifffile.imwrite(path, data)
    meta = read_metadata(path)
    assert meta.n_frames == 1
    assert load_stack(path, meta).shape == (1, 6, 8)


def test_singleton_channel_axis_is_collapsed(tmp_path):
    data = np.zeros((4, 1, 6, 8), dtype=np.uint16)
    path = tmp_path / "tcyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TCYX"})
    meta = read_metadata(path)
    assert meta.shape == (4, 6, 8)
    assert load_stack(path, meta).shape == (4, 6, 8)


def test_a_multichannel_stack_is_reduced_to_the_chosen_channel(tmp_path):
    """1.x refused every C > 1 file; 2.0 analyses one channel and records which."""
    data = np.zeros((4, 2, 6, 8), dtype=np.uint16)
    data[:, 1] = 7  # only the second channel is non-zero
    path = tmp_path / "tcyx2.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TCYX"})

    first = read_metadata(path)
    assert (first.axes, first.shape) == ("TYX", (4, 6, 8))
    assert (first.channel_index, first.n_channels) == (0, 2)
    assert first.to_dict()["channel_index"] == 0
    assert any("2 channels" in note for note in first.notes)
    assert load_stack(path, first).max() == 0

    second = read_metadata(path, ImportConfig(channel_index=1))
    assert second.channel_index == 1
    stack = load_stack(path, second)
    assert stack.shape == (4, 6, 8) and np.all(stack == 7)


def test_a_channel_that_does_not_exist_is_refused(tmp_path):
    data = np.zeros((4, 2, 6, 8), dtype=np.uint16)
    path = tmp_path / "tcyx2.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TCYX"})
    with pytest.raises(UnsupportedStackError, match="2 channels"):
        read_metadata(path, ImportConfig(channel_index=2))
    single = tmp_path / "tyx.tif"
    tifffile.imwrite(single, data[:, 0], imagej=True, metadata={"axes": "TYX"})
    with pytest.raises(UnsupportedStackError, match="1 channel"):
        read_metadata(single, ImportConfig(channel_index=1))


def test_the_stack_is_float32_and_holds_every_uint16_value_exactly(tmp_path):
    data = np.array([[[0, 1], [65534, 65535]]], dtype=np.uint16)
    path = tmp_path / "range.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TYX"})
    stack = load_stack(path, read_metadata(path))
    assert stack.dtype == np.float32
    assert np.array_equal(stack, data.astype(np.float64))


def test_unequal_pixel_sizes_are_recorded_not_averaged(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "rect.tif"
    tifffile.imwrite(
        path, data, imagej=True, metadata={"axes": "TYX", "unit": "micron"},
        resolution=(2.0, 2.5),
    )
    meta = read_metadata(path)
    assert meta.pixel_size_um.value == pytest.approx(0.5)
    assert meta.pixel_size_y_um.value == pytest.approx(0.4)
    assert meta.pixel_size_y_um.source == SOURCE_TIFF_TAG
    assert meta.anisotropic_pixels
    assert meta.to_dict()["anisotropic_pixels"] is True
    assert any("not square" in note for note in meta.notes)


def test_square_pixels_are_not_flagged(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "square.tif"
    tifffile.imwrite(
        path, data, imagej=True, metadata={"axes": "TYX", "unit": "micron"},
        resolution=(2.14105, 2.14105),
    )
    meta = read_metadata(path)
    assert meta.pixel_size_y_um.value == pytest.approx(meta.pixel_size_um.value)
    assert not meta.anisotropic_pixels


@pytest.mark.parametrize("unit", ["micron", "microns", "um", "\\u00B5m"])
def test_every_spelling_of_micrometre_is_a_calibration(tmp_path, unit):
    """ImageJ writes the micro sign several ways; 1.x lost the escaped one.

    The header is ASCII, so a literal µ never reaches it: ImageJ escapes it
    as ``\\u00B5m``. (A literal µ arrives through OME-XML, tested in
    test_imaging_3d.)
    """
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "unit.tif"
    tifffile.imwrite(
        path, data, imagej=True, metadata={"axes": "TYX", "unit": unit}, resolution=(2.0, 2.0)
    )
    meta = read_metadata(path)
    assert meta.pixel_size_um.value == pytest.approx(0.5)


def test_an_imagej_time_unit_is_honoured(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "tunit.tif"
    tifffile.imwrite(
        path, data, imagej=True, metadata={"axes": "TYX", "finterval": 20.0, "tunit": "min"}
    )
    meta = read_metadata(path)
    assert meta.frame_interval_min.value == pytest.approx(20.0)
    assert meta.frame_interval_min.source == SOURCE_IMAGEJ


def test_missing_calibration_is_reported_not_invented(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TYX"})
    meta = read_metadata(path)
    assert meta.pixel_size_um.source == SOURCE_MISSING
    assert meta.frame_interval_min.source == SOURCE_MISSING
    assert not meta.pixel_size_um.known


def test_imagej_frame_interval_is_converted_from_seconds(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "interval.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "TYX", "finterval": 1200.4136962890625, "unit": "micron"},
        resolution=(2.14105, 2.14105),
    )
    meta = read_metadata(path)
    assert meta.frame_interval_min.source == SOURCE_IMAGEJ
    assert meta.frame_interval_min.value == pytest.approx(20.00689494, rel=1e-8)
    assert meta.pixel_size_um.source == SOURCE_TIFF_TAG
    assert meta.pixel_size_um.value == pytest.approx(0.46706055, rel=1e-6)


def test_nd2_info_is_preferred_over_the_imagej_header(tmp_path):
    """The embedded acquisition block carries full precision; prefer it."""
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "nd2info.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={
            "axes": "TYX",
            "finterval": 1200.4136962890625,
            "unit": "micron",
            "Info": "dCalibration = 0.467060342995564\ndAvgPeriodDiff = 1200413.6719809477\n",
        },
        resolution=(2.14105, 2.14105),
    )
    meta = read_metadata(path)
    assert meta.pixel_size_um.source == SOURCE_ND2_INFO
    assert meta.pixel_size_um.value == pytest.approx(0.467060342995564, rel=1e-12)
    assert meta.frame_interval_min.source == SOURCE_ND2_INFO


def test_user_override_wins_and_is_recorded(tmp_path):
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "override.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "TYX", "finterval": 600.0, "unit": "micron"},
        resolution=(2.0, 2.0),
    )
    meta = read_metadata(path)
    pixel, interval, scale = effective_calibration(
        meta, CalibrationConfig(pixel_size_um=0.25, frame_interval_min=15.0)
    )
    assert pixel.value == 0.25 and pixel.source == SOURCE_USER
    assert interval.value == 15.0 and interval.source == SOURCE_USER
    assert scale.calibrated


def test_source_sizet_does_not_become_the_frame_count(tmp_path):
    """The defect: SizeT = 54 from the parent ND2 is not this file's length."""
    data = np.zeros((5, 6, 8), dtype=np.uint16)
    path = tmp_path / "cropped.tif"
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "TYX", "Info": "SizeT = 54\n", "unit": "micron"},
    )
    meta = read_metadata(path)
    assert meta.n_frames == 5
    assert meta.acquisition["SizeT"] == "54"
    assert any("54" in note for note in meta.notes)


# --------------------------------------------------------------------------
# The supplied data
# --------------------------------------------------------------------------


@requires_samples
@pytest.mark.parametrize("name,shape", sorted(EXPECTED.items()))
def test_supplied_stacks_have_the_shape_they_actually_contain(name, shape):
    meta = read_metadata(SAMPLE_DIR / name)
    assert meta.shape == shape
    assert meta.axes_raw == "TYX"
    assert meta.dtype == "uint16"
    # Time is established by the ImageJ header's frames=N, not assumed.
    assert (meta.axes, meta.axes_source, meta.dimensionality) == ("TYX", SOURCE_IMAGEJ, "2D")
    assert (meta.n_channels, meta.channel_index) == (1, 0)
    assert not meta.anisotropic_pixels
    assert not meta.z_step_um.known


@requires_samples
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_supplied_stacks_are_twenty_minutes_per_frame(name):
    """The central correction: about 20.0069 min, never 10."""
    meta = read_metadata(SAMPLE_DIR / name)
    assert meta.frame_interval_min.known
    assert meta.frame_interval_min.value == pytest.approx(20.006894938, rel=1e-6)
    assert meta.frame_interval_min.value != pytest.approx(10.0, rel=0.01)


@requires_samples
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_supplied_stacks_have_the_nd2_pixel_size(name):
    meta = read_metadata(SAMPLE_DIR / name)
    assert meta.pixel_size_um.known
    assert meta.pixel_size_um.value == pytest.approx(0.467060342995564, rel=1e-9)
    assert meta.pixel_size_um.source == SOURCE_ND2_INFO


@requires_samples
@pytest.mark.parametrize("name,span", sorted(EXPECTED_SOURCE_FRAMES.items()))
def test_original_frame_numbers_are_preserved(name, span):
    meta = read_metadata(SAMPLE_DIR / name)
    assert meta.source_frames is not None
    assert (meta.source_frames[0], meta.source_frames[-1]) == span
    assert meta.source_frame_total == 54
    assert len(meta.source_frames) == meta.n_frames


@requires_samples
@pytest.mark.parametrize("name,shape", sorted(EXPECTED.items()))
def test_supplied_stacks_load_with_the_declared_shape(name, shape):
    meta = read_metadata(SAMPLE_DIR / name)
    stack = load_stack(SAMPLE_DIR / name, meta)
    assert stack.shape == shape
    assert stack.flags["C_CONTIGUOUS"]


def test_channel_axis_is_never_read_as_time(tmp_path):
    """A 3-channel image is one image of one chosen channel, not 3 frames."""
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "cyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX"})
    meta = read_metadata(path)
    assert meta.axes == "YX"
    assert meta.n_frames == 1 and meta.n_channels == 3
    assert load_stack(path, meta).shape == (1, 6, 8)


def test_an_unlabelled_multipage_stack_is_ambiguous_not_time(tmp_path):
    """1.x read a plain multi-page stack as time with a note; it may be Z."""
    data = np.zeros((4, 6, 8), dtype=np.uint16)
    path = tmp_path / "plainstack.tif"
    tifffile.imwrite(path, data, photometric="minisblack")  # -> axes 'QYX'
    with pytest.raises(AmbiguousAxes) as info:
        read_metadata(path)
    assert info.value.choices[:2] == ("TYX", "ZYX")
    assert "TYX" in str(info.value) and "ZYX" in str(info.value)
    # The import settings answer the question, and the answer is recorded.
    meta = read_metadata(path, ImportConfig(axes="TYX"))
    assert (meta.axes, meta.n_frames, meta.axes_source) == ("TYX", 4, SOURCE_USER)
    assert load_stack(path, meta).shape == (4, 6, 8)


def test_z_slices_are_never_read_as_time(tmp_path):
    """The T-vs-Z trap: 1.x analysed a one-time-point Z stack as a time-lapse."""
    data = np.zeros((4, 6, 8), dtype=np.uint16)
    path = tmp_path / "zyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "ZYX"})
    meta = read_metadata(path)
    assert meta.axes == "ZYX" and meta.dimensionality == "3D"
    assert meta.n_frames == 1 and meta.n_slices == 4
    assert meta.shape == (1, 4, 6, 8)
    assert load_stack(path, meta).shape == (1, 4, 6, 8)


def test_8_bit_colour_is_one_image_not_three_time_points(tmp_path):
    """An interleaved 8-bit RGB image has 3 samples, not 3 time points."""
    data = np.zeros((6, 8, 3), dtype=np.uint8)
    path = tmp_path / "yxs.tif"
    tifffile.imwrite(path, data)  # -> axes 'YXS'
    meta = read_metadata(path)
    assert meta.axes == "YX" and meta.n_frames == 1
    assert meta.n_channels == 3 and meta.channel_index == 0


@pytest.mark.parametrize("planes", [3, 4])
def test_planar_16_bit_samples_are_ambiguous_not_colour(tmp_path, planes):
    """tifffile writes any (3|4, Y, X) 16-bit array as planar RGB by default.

    A 3- or 4-frame time-lapse saved with a bare ``imwrite`` therefore reads
    as one colour image; 1.x refused it, and reading it as channel 0 would
    silently drop all but one frame. It is a question for the user.
    """
    data = _ramp_u16((planes, 6, 8))
    path = tmp_path / "syx.tif"
    tifffile.imwrite(path, data)
    with tifffile.TiffFile(path) as tf:
        assert tf.series[0].axes == "SYX"
    with pytest.raises(AmbiguousAxes) as info:
        read_metadata(path)
    assert info.value.choices[:2] == ("TYX", "ZYX") and "CYX" in info.value.choices
    assert "planar colour samples" in str(info.value)
    meta = read_metadata(path, ImportConfig(axes="TYX"))
    assert (meta.axes, meta.n_frames, meta.axes_source) == ("TYX", planes, SOURCE_USER)
    assert np.array_equal(load_stack(path, meta), data)


def _ramp_u16(shape) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)


def test_an_entered_pixel_size_does_not_make_square_pixels_anisotropic(tmp_path):
    """pipeline.run_analysis overwrites metadata.pixel_size_um with the entry.

    The flag compared that with the file's Y size, so entering any pixel size
    raised a false critical "anisotropic pixels" issue on square pixels.
    """
    path = tmp_path / "square.tif"
    tifffile.imwrite(
        path, np.zeros((3, 6, 8), np.uint16), imagej=True,
        metadata={"axes": "TYX", "unit": "micron"}, resolution=(2.0, 2.0),
    )
    meta = read_metadata(path)
    assert not meta.anisotropic_pixels
    pixel, _, _ = effective_calibration(meta, CalibrationConfig(pixel_size_um=0.467))
    meta.pixel_size_um = pixel  # exactly what pipeline.run_analysis does
    assert not meta.anisotropic_pixels
    assert meta.to_dict()["anisotropic_pixels"] is False


def test_a_tiff_resolution_in_centimetres_is_a_calibration(tmp_path):
    path = tmp_path / "cm.tif"
    tifffile.imwrite(
        path, np.zeros((3, 6, 8), np.uint16), metadata={"axes": "TYX"},
        resolution=(10000, 8000), resolutionunit="CENTIMETER",
    )
    meta = read_metadata(path)
    assert meta.pixel_size_um.value == pytest.approx(1.0)
    assert meta.pixel_size_um.source == SOURCE_TIFF_TAG
    assert meta.pixel_size_y_um.value == pytest.approx(1.25)
    assert meta.anisotropic_pixels


def test_dots_per_inch_are_not_a_calibration(tmp_path):
    """tifffile writes INCH for a bare resolution=; 72 dpi would be 352.8 um/px."""
    path = tmp_path / "dpi.tif"
    tifffile.imwrite(
        path, np.zeros((3, 6, 8), np.uint16), metadata={"axes": "TYX"}, resolution=(72, 72)
    )
    meta = read_metadata(path)
    assert not meta.pixel_size_um.known and meta.pixel_size_um.source == SOURCE_MISSING
    assert any("dots per inch" in note for note in meta.notes)
