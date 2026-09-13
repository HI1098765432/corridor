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
    UnsupportedStackError,
    load_stack,
    parse_info_block,
    parse_source_frames,
    read_metadata,
)
from corridor.core.config import CalibrationConfig
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


def test_multichannel_stack_is_refused_clearly(tmp_path):
    data = np.zeros((4, 2, 6, 8), dtype=np.uint16)
    path = tmp_path / "tcyx2.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "TCYX"})
    with pytest.raises(UnsupportedStackError) as excinfo:
        read_metadata(path)
    assert "ambiguous" in str(excinfo.value).lower()


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
    """A 3-channel image must not be analysed as a 3-frame time-lapse."""
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "cyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX"})
    with pytest.raises(UnsupportedStackError):
        read_metadata(path)


def test_unlabelled_multipage_stack_is_read_as_time_with_a_note(tmp_path):
    """A plain multi-page stack is accepted, but the reading is stated out loud."""
    data = np.zeros((4, 6, 8), dtype=np.uint16)
    path = tmp_path / "plainstack.tif"
    tifffile.imwrite(path, data, photometric="minisblack")  # -> axes 'QYX'
    meta = read_metadata(path)
    assert meta.n_frames == 4
    assert any("read as time" in note for note in meta.notes)


def test_z_slices_are_read_as_time_with_a_note(tmp_path):
    data = np.zeros((4, 6, 8), dtype=np.uint16)
    path = tmp_path / "zyx.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "ZYX"})
    meta = read_metadata(path)
    assert meta.n_frames == 4
    assert any("'Z'" in note for note in meta.notes)


def test_rgb_samples_are_never_read_as_time(tmp_path):
    """A single colour image has 3 samples, not 3 time points."""
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    path = tmp_path / "syx.tif"
    tifffile.imwrite(path, data)  # -> axes 'SYX', one page
    with pytest.raises(UnsupportedStackError):
        read_metadata(path)
