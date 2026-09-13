"""Regression tests against the supplied microscopy.

These run the real pipeline, Cellpose included, so they are slow and are marked
``realdata``. Run them with ``pytest -m realdata``.

What they are for: locking in findings that were established by measurement
during development, so that a later change cannot quietly undo them. They are
*not* a model-generalisation benchmark -- every one of these files comes from
the same experimental collection as the model's training data, and several of
the training images are from neighbouring positions of the same device. Passing
them shows the software still behaves as measured; it says nothing about how
the model would perform on a different experiment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from corridor.core import pipeline
from corridor.core.config import RunConfig, SegmentationConfig
from corridor.core.imaging import read_metadata

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, SAMPLE_DIR, requires_samples

pytestmark = [pytest.mark.realdata, requires_samples]

MODEL = (
    SAMPLE_DIR.parent / "CellPose_TrainData" / "KK1KK2_combiModel" / "models"
    / "cyto2_phase_microfluidic_KK1KK2_combi"
)

#: Measured with Cellpose 3.1.1.3 at the researcher's own labelling settings
#: (cellprob 0.0, flow 0.4, channels [0, 0]). The raw and kept counts are equal
#: for every frame of every supplied file: the min_extent filter removes
#: nothing, so any missing detection came from Cellpose, not post-processing.
EXPECTED_COUNTS = {
    "052924_t1.tif": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t2_empty.tif": [0, 1, 1, 0, 0],
    "052924_t3_dual.tif": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}

EXPECTED_TRACKS = {
    "052924_t1.tif": 1,
    "052924_t2_empty.tif": 1,
    "052924_t3_dual.tif": 3,
}


@pytest.fixture(scope="module")
def model_available() -> bool:
    if not MODEL.exists():
        pytest.skip(f"custom model not present at {MODEL}")
    return True


def run(name: str, tmp_path, **overrides) -> pipeline.AnalysisResult:
    config = RunConfig(
        input_path=str(SAMPLE_DIR / name),
        output_dir=str(tmp_path / name.replace(".tif", "")),
        segmentation=SegmentationConfig(model_path=str(MODEL), use_custom_model=True),
    )
    for key, value in overrides.items():
        setattr(config.tracking, key, value)
    return pipeline.run_analysis(config, save=True)


# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
def test_segmentation_counts_are_stable(name, tmp_path, model_available):
    result = run(name, tmp_path)
    kept = [d.kept_count for d in result.segmentation.diagnostics]
    raw = [d.raw_count for d in result.segmentation.diagnostics]
    assert kept == EXPECTED_COUNTS[name]
    assert raw == kept, (
        "the minimum-size filter removed something; on this data it never has, "
        "so a detection that disappears is Cellpose's, not the filter's"
    )


@pytest.mark.parametrize("name,n_tracks", sorted(EXPECTED_TRACKS.items()))
def test_track_counts_are_stable(name, n_tracks, tmp_path, model_available):
    result = run(name, tmp_path)
    assert result.n_tracks == n_tracks


def test_t1_is_one_continuous_cell(tmp_path, model_available):
    """Verified by eye against the overlay: one cell, frames 4 to 16."""
    result = run("052924_t1.tif", tmp_path)
    assert len(result.tracks) == 1
    track = result.tracks[0]
    assert [o.frame for o in track.observations] == list(range(4, 17))
    assert all(o.gap_frames <= 1 for o in track.observations[1:])

    (summary,) = result.summaries
    assert summary.n_observations == 13
    assert summary.n_gaps == 0
    # The cell migrates down the channel, then stalls near the outlet.
    assert summary.net_along_um > 80
    assert abs(summary.net_across_um) < abs(summary.net_along_um) / 4
    assert 0.2 < summary.mean_speed_um_per_min < 1.5


def test_t3_keeps_two_cells_apart(tmp_path, model_available):
    """Both cells are visible in frames 0 and 1 and must not be merged."""
    result = run("052924_t3_dual.tif", tmp_path)
    frame0 = [t for t in result.tracks if t.observations[0].frame == 0]
    assert len(frame0) == 2, "the two cells present at frame 0 became one identity"

    positions = sorted(t.observations[0].y for t in frame0)
    assert positions[1] - positions[0] > 100, "the two identities are the same object"

    # No detection may be claimed by two tracks in the same frame.
    seen: set[tuple[int, int]] = set()
    for track in result.tracks:
        for obs in track.observations:
            key = (obs.frame, obs.det_label)
            assert key not in seen, f"detection {key} was used twice"
            seen.add(key)


def test_t3_does_not_bridge_the_six_frame_gap(tmp_path, model_available):
    """The object at frame 10 is 6 frames after the last observation.

    With max_gap = 3 the tracker may bridge at most 4 frames, so this becomes a
    separate single-observation track. Note that this is the *gap rule* acting,
    not a judgement that the object is a different cell: measured, the pairing
    would fit comfortably (see the unlinked-start test below). Raising max_gap
    to 5 or more would join them, and that is the researcher's call to make on
    the evidence, not a default the software should choose for them.
    """
    result = run("052924_t3_dual.tif", tmp_path)
    late = [t for t in result.tracks if t.observations[0].frame == 10]
    assert len(late) == 1
    assert late[0].n_obs == 1
    for track in result.tracks:
        for obs in track.observations[1:]:
            assert obs.gap_frames <= result.config.tracking.max_delta_frames()


def test_empty_stack_is_safe_and_writes_valid_files(tmp_path, model_available):
    result = run("052924_t2_empty.tif", tmp_path)
    directory = result.output_dir
    for name in (
        pipeline.F_TRACKS, pipeline.F_DETECTIONS, pipeline.F_SUMMARY,
        pipeline.F_DIAGNOSTICS, pipeline.F_QC, pipeline.F_MANIFEST,
    ):
        assert (directory / name).exists(), f"{name} was not written"
    # Two detections, in frames 1 and 2 only.
    assert len(result.segmentation.detections) == 2
    assert result.n_tracks == 1


def test_calibration_is_read_from_every_supplied_file(model_available):
    for name in sorted(EXPECTED_COUNTS):
        metadata = read_metadata(SAMPLE_DIR / name)
        assert metadata.frame_interval_min.value == pytest.approx(
            FRAME_INTERVAL_MIN, rel=1e-6
        )
        assert metadata.pixel_size_um.value == pytest.approx(PIXEL_SIZE_UM, rel=1e-9)


def test_wide_field_tracks_never_cross_a_channel(tmp_path, model_available):
    """The scientific claim that makes wide-field tracking defensible."""
    result = run("052924_1.tif", tmp_path)
    assert result.axis.is_multichannel
    assert len(result.axis.channels) == 6
    assert 75 < (result.axis.pitch_px or 0) < 92
    for track in result.tracks:
        channels = {o.channel for o in track.observations}
        assert len(channels) == 1, (
            f"track {track.id} spans channels {sorted(channels)}"
        )


def test_manifest_is_complete_and_reproducible(tmp_path, model_available):
    result = run("052924_t1.tif", tmp_path)
    m = result.manifest
    assert m["environment"]["cellpose"].startswith("3.")
    assert m["segmentation"]["model_sha256"] == (
        "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"
    )
    assert m["calibration"]["pixel_size_um_source"] == "nd2_info"
    assert m["calibration"]["frame_interval_min_source"] == "nd2_info"
    assert m["input"]["axes_reported"] == "TYX"
    assert m["input"]["source_frames"][0] == 1
    assert m["confinement"]["n_channels"] == 1
    assert -4 < m["confinement"]["tilt_from_vertical_deg"] < -1
    assert m["tracking"]["max_delta_frames"] == m["tracking"]["max_gap"] + 1
    assert m["segmentation"]["removed_instances_total"] == 0


def test_t3_late_object_is_reported_as_a_judgement_not_a_fact(tmp_path, model_available):
    """The frame-10 object is refused only by the gap limit, and says so.

    Measured: it sits about 6 px from where track 1 ended, in the same channel,
    and would fit at a cost well under the rejection threshold. The only thing
    refusing it is that 6 frames is longer than max_gap allows. That is a policy
    choice, and the software has to present it as one rather than implying the
    object is a different cell.
    """
    from corridor.core.tracking import explain_unlinked_starts

    result = run("052924_t3_dual.tif", tmp_path)
    unlinked = explain_unlinked_starts(
        result.tracks, result.axis, result.scale, result.config.tracking
    )
    late = [u for u in unlinked if u.frame == 10]
    assert len(late) == 1
    evidence = late[0]

    assert evidence.candidate_track_id is not None
    assert evidence.gap_frames == 6
    assert evidence.gap_frames > result.config.tracking.max_delta_frames()
    assert evidence.refused_because == "gap_too_long"
    assert evidence.distance_px is not None and evidence.distance_px < 20
    assert evidence.cost_chi2 is not None
    assert evidence.cost_chi2 < result.config.tracking.gate_chi2, (
        "the fit itself is acceptable; only the gap refuses it, and the "
        "software must not imply otherwise"
    )
    assert "gap" in evidence.describe().lower()

    # It must still be reported to the reviewer, at warning level.
    codes = {(i.code, i.severity) for i in result.issues}
    assert ("unlinked_start", "warning") in codes


def test_unlinked_starts_file_is_written(tmp_path, model_available):
    result = run("052924_t3_dual.tif", tmp_path)
    path = result.output_dir / pipeline.F_UNLINKED
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("track_id,starts_at_frame,")
    assert "gap_too_long" in text


def test_t1_has_no_unlinked_starts(tmp_path, model_available):
    """One continuous cell leaves nothing to explain."""
    from corridor.core.tracking import explain_unlinked_starts

    result = run("052924_t1.tif", tmp_path)
    unlinked = explain_unlinked_starts(
        result.tracks, result.axis, result.scale, result.config.tracking
    )
    assert unlinked == [] or all(u.candidate_track_id is None for u in unlinked)
