"""The run manifest, and the join between detections.csv and tracks.csv.

Both of these were broken at some point today by changes that 193 other tests
happily accepted, because nothing fast built a manifest or checked that the two
CSVs refer to the same objects. A defect that makes every real run fail should
not need a real run to find.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from corridor.core import export
from corridor.core.config import RunConfig, Scale
from corridor.core.detections import SOURCE_ENSEMBLE, detections_to_rows
from corridor.core.imaging import SOURCE_USER, Calibrated, StackMetadata
from corridor.core.measurements import frame_rows, summarise
from corridor.core.pipeline import AnalysisResult, build_manifest
from corridor.core.recovery import SOURCE_WINDOW, RecoveryResult
from corridor.core.segmentation import SegmentationOutput
from corridor.core.tracking import track_detections

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, make_detection, straight_track


@pytest.fixture
def metadata(tmp_path) -> StackMetadata:
    return StackMetadata(
        path=tmp_path / "synthetic.tif",
        n_frames=8,
        height=324,
        width=90,
        dtype="uint16",
        axes_raw="TYX",
        axes_interpretation="TYX",
        pixel_size_um=Calibrated(PIXEL_SIZE_UM, SOURCE_USER),
        frame_interval_min=Calibrated(FRAME_INTERVAL_MIN, SOURCE_USER),
    )


def segmentation_output(detections) -> SegmentationOutput:
    return SegmentationOutput(
        masks=np.zeros((8, 324, 90), np.int32),
        raw_masks=np.zeros((8, 324, 90), np.int32),
        detections=list(detections),
        diagnostics=[],
        model_path="models/combi",
        model_sha256="b33bdbda",
        cellpose_version="3.1.1.3",
        used_gpu=False,
        passes_per_frame=1,
    )


def build(metadata, vertical_axis, scale, tracking_config, detections, recovery=None):
    tracks, events = track_detections(
        detections, metadata.n_frames, vertical_axis, scale, tracking_config
    )
    summaries = summarise(tracks, vertical_axis, scale)
    config = RunConfig()
    config.tracking = tracking_config
    segmentation = segmentation_output(
        [d for d in detections if d.source != SOURCE_WINDOW]
    )
    manifest = build_manifest(
        config, metadata, scale, vertical_axis, segmentation, tracks, summaries,
        elapsed_s=1.0, output_dir=None, recovery=recovery,
    )
    result = AnalysisResult(
        config=config, metadata=metadata, scale=scale, axis=vertical_axis,
        segmentation=segmentation, tracks=tracks, events=events,
        rows=frame_rows(tracks, vertical_axis, scale), summaries=summaries,
        issues=[], recovery=recovery, manifest=manifest,
    )
    return result


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_a_manifest_can_be_built_at_all(metadata, vertical_axis, scale, tracking_config):
    """The regression: a name error here failed every run and no fast test saw it."""
    result = build(metadata, vertical_axis, scale, tracking_config, straight_track(6))
    assert result.manifest["application"]["name"]
    assert result.manifest["segmentation"]["ensemble"] == "off"
    assert result.manifest["segmentation"]["ensemble_passes"] == 1


def test_the_manifest_is_json_with_no_bare_nan(
    metadata, vertical_axis, scale, tracking_config
):
    """``NaN`` is not JSON, and a reader that accepts it is not reading JSON.

    numpy scalars subclass float, so a naive encoder emits bare NaN without
    ever calling a ``default`` hook.
    """
    result = build(metadata, vertical_axis, scale, tracking_config, straight_track(6))
    text = json.dumps(result.manifest, default=str, allow_nan=False)
    assert "NaN" not in text


def test_the_manifest_records_the_configuration_that_ran(
    metadata, vertical_axis, scale, tracking_config
):
    result = build(metadata, vertical_axis, scale, tracking_config, straight_track(6))
    seg = result.manifest["segmentation"]
    # Both numbers, because they differ when an optional model is missing.
    assert seg["ensemble_passes_requested"] == 1
    assert seg["normalisation_mode"] == "whole_frame"


# --------------------------------------------------------------------------
# detections.csv and tracks.csv describe the same objects
# --------------------------------------------------------------------------


def test_every_track_row_finds_its_detection(
    metadata, vertical_axis, scale, tracking_config
):
    """The join a user will obviously try, on a result containing recovery.

    A recovered position carries a label from the crop it was found in, which
    routinely collides with a primary label in the same frame. If the two files
    disagree, the join silently returns another cell's measurements.
    """
    primary = straight_track(6, step=20.0)
    # A recovered position at frame 6, deliberately given a colliding label.
    recovered = make_detection(6, 45.0, 140.0, label=1)
    recovered.source = SOURCE_WINDOW
    recovered.confidence = 0.8

    # The pipeline relabels above whatever the frame already uses; do the same.
    highest = max((d.label for d in primary if d.frame == 6), default=0)
    recovered.label = highest + 1

    recovery = RecoveryResult(detections=[recovered])
    result = build(
        metadata, vertical_axis, scale, tracking_config,
        primary + [recovered], recovery=recovery,
    )

    assert len(result.all_detections) == len(primary) + 1

    rows = detections_to_rows(
        result.all_detections,
        pixel_size_um=scale.pixel_size_um,
        frame_interval_min=scale.frame_interval_min,
    )
    index = {(r["frame"], r["label"]) for r in rows}
    missing = [
        r for r in result.rows if (r["frame"], r["det_label"]) not in index
    ]
    assert missing == [], "a track row points at a detection that is not exported"


def test_no_frame_exports_two_detections_with_the_same_label(
    metadata, vertical_axis, scale, tracking_config
):
    primary = straight_track(6, step=20.0)
    recovered = make_detection(3, 45.0, 80.0, label=1)  # collides on purpose
    recovered.source = SOURCE_WINDOW
    highest = max((d.label for d in primary if d.frame == 3), default=0)
    recovered.label = highest + 1

    result = build(
        metadata, vertical_axis, scale, tracking_config,
        primary + [recovered], recovery=RecoveryResult(detections=[recovered]),
    )
    pairs = [(d.frame, d.label) for d in result.all_detections]
    assert len(pairs) == len(set(pairs))


def test_a_result_without_recovery_exports_exactly_the_primary_detections(
    metadata, vertical_axis, scale, tracking_config
):
    primary = straight_track(6, step=20.0)
    result = build(metadata, vertical_axis, scale, tracking_config, primary)
    assert len(result.all_detections) == len(primary)


def test_the_summary_columns_match_the_dataclass(
    metadata, vertical_axis, scale, tracking_config
):
    """A column list that drifts from the fields writes one column's value
    under another column's name, which no reader would ever suspect."""
    result = build(metadata, vertical_axis, scale, tracking_config, straight_track(6))
    row = result.summaries[0].to_row()
    assert set(export.SUMMARY_COLUMNS) == set(row), (
        set(export.SUMMARY_COLUMNS) ^ set(row)
    )
