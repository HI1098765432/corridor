"""Regression tests against the supplied microscopy (schema 2).

These read the output of the real pipeline, Cellpose included, so they are
slow and are marked ``realdata``. Run them with ``pytest -m realdata``.

Each movie is analysed **once** per session and every test reads the files
that run wrote -- the files are the product, so they are what is checked.
Set ``CORRIDOR_V2_RUNS`` to a directory holding one finished result per
movie (``<dir>/<movie stem>/run.json``; for example the CLI runs in
``build/v2_runs``) to test those instead of analysing again; they are
accepted only if their ``run.json`` says they came from that sample movie and
from the validated model.

What they are for: locking in findings that were established by measurement
during development, so that a later change cannot quietly undo them. They are
*not* a model-generalisation benchmark -- every one of these files comes from
the same experimental collection as the model's training data, and several of
the training images are from neighbouring positions of the same device. Passing
them shows the software still behaves as measured; it says nothing about how
the model would perform on a different experiment. There is no tracking ground
truth for them either (contract §0): the track counts below are the measured
2.0 behaviour, compared with v1.3.0 in ``docs/v2_vs_v1_baseline.json``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

from corridor.core import export, model_registry, pipeline
from corridor.core.config import RunConfig, TrackingConfig
from corridor.core.imaging import read_metadata
from corridor.core.model_registry import ModelUnavailable
from corridor.store.project import SavedAnalysis, analysis_is_complete, load_analysis

from conftest import FRAME_INTERVAL_MIN, PIXEL_SIZE_UM, SAMPLE_DIR, requires_samples

pytestmark = [pytest.mark.realdata, requires_samples]

#: A directory of finished results to test instead of analysing again.
RUNS_ENV = "CORRIDOR_V2_RUNS"

#: The checksum pinned in the application since 1.0.0 (registry entry
#: ``jhu_confined_cp3_combi`` 1.0.0).  Written out here as well as read from
#: the registry, so a registry edit cannot move both sides of the check.
VALIDATED_SHA256 = "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"

#: Measured with Cellpose 3.1.1.3 at the researcher's own labelling settings
#: (cellprob 0.0, flow 0.4, channels [0, 0]). The raw and kept counts are equal
#: for every frame of every supplied file: the min_extent filter removes
#: nothing, so any missing detection came from Cellpose, not post-processing.
#: Identical in v1.3.0 and 2.0: segmentation did not change.
EXPECTED_COUNTS = {
    "052924_t1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t2_empty": [0, 1, 1, 0, 0],
    "052924_t3_dual": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}

#: Tracks in the 2.0 result (recovery and re-tracking included), measured on
#: the five sample movies at the 2.0 defaults.  The narrow crops are unchanged
#: from v1.3.0; the wide fields differ, and every difference is attributed in
#: docs/v2_vs_v1_baseline.json.
#:
#: 052924_1's 16 INCLUDES A KNOWN DEFECT (track 15, see KNOWN_DEFECTS): the
#: count is pinned so that any change is noticed, not because 16 is right.
#: Fixing the defect changes it, and test_no_step_is_faster_than_any_baseline_step
#: then XPASSes (strict) so both pins are revisited together.
EXPECTED_TRACKS = {
    "052924_1": 16,
    "052924_2": 9,
    "052924_t1": 1,
    "052924_t2_empty": 1,
    "052924_t3_dual": 3,
}

#: The two wide fields: six lanes each, measured from the walls.
WIDE_FIELDS = ("052924_1", "052924_2")

#: The fastest single step of any track in the v1.3.0 baseline of these five
#: movies (build/baseline_v1.3.0, tracks.csv ``speed_um_per_min``):
#: 052924_2 track 7, frame 13 -> 14, 85.0 px in one 20.01 min frame.  The
#: segmentation is bit-identical in 2.0, so a faster step is a link 1.x never
#: made and no reviewed track ever showed -- a new claim about a cell that has
#: to be looked at before it is pinned.  (TrackingConfig's 5 um/min gate is the
#: hard physical limit; this is the measured one.)
FASTEST_BASELINE_STEP_UM_PER_MIN = 1.984089566

#: Defects the shipped defaults still produce, pinned as known rather than as
#: correct.  Each XFAILs (strict) the guard that catches it.
KNOWN_DEFECTS = {
    "052924_1": (
        "track 15: frame 16 (44.4, 279.6) primary -> frame 17 (55.0, 117.1) intensity-tier "
        "recovery, a 162.9 px (76 um) step in one 20 min frame = 3.80 um/min, backwards up "
        "the lane; v1.3.0 kept this cell in its track 4. Cause: the first-pass tracker "
        "linked (44, 280)@16 to (57, 60)@18 across a 2-frame gap (220 px; owner: tracking, "
        "WP-B), and recovery then accepted an intensity object 52.8 px from that link's "
        "interpolated prediction (owner: recovery, E2). QC warns about the track "
        "(fallback_dependent_track, morphology_discontinuity, size_jump)."
    ),
}

#: The (track id, frame) of every step each known defect makes, so the xfail
#: above cannot hide a second, new implausible step in the same movie.
KNOWN_DEFECT_STEPS = {"052924_1": {(15, 17)}}

_CACHE: dict[str, SavedAnalysis] = {}


def _validated_sha() -> str:
    return model_registry.production_spec("2D").sha256


@pytest.fixture(scope="session")
def analysed(tmp_path_factory):
    """``analysed(stem)`` -> the SavedAnalysis of that movie, analysed at most once."""

    def get(stem: str) -> SavedAnalysis:
        if stem in _CACHE:
            return _CACHE[stem]
        runs = os.environ.get(RUNS_ENV)
        if runs:
            directory = Path(runs) / stem
            if not analysis_is_complete(directory):
                pytest.skip(f"{RUNS_ENV} has no finished result for {stem} in {directory}")
        else:
            try:
                model_registry.resolve_model("2D")
            except ModelUnavailable as exc:
                pytest.skip(str(exc))
            directory = tmp_path_factory.mktemp(stem)
            pipeline.run_analysis(
                RunConfig(input_path=str(SAMPLE_DIR / f"{stem}.tif"), output_dir=str(directory))
            )
        saved = load_analysis(directory)
        # A result handed in from outside must be the one it claims to be.
        assert saved.manifest["input"]["name"] == f"{stem}.tif"
        assert saved.manifest["model"]["sha256"] == _validated_sha()
        assert saved.manifest["model"]["developer_override"] is False
        _CACHE[stem] = saved
        return saved

    return get


def tracks_of(saved: SavedAnalysis) -> dict[int, list[dict]]:
    """tracks.csv rows by track id, each in frame order."""
    out: dict[int, list[dict]] = defaultdict(list)
    for row in saved.tracks:
        out[int(row["track_id"])].append(row)
    return {tid: sorted(rows, key=lambda r: r["frame"]) for tid, rows in out.items()}


# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(EXPECTED_COUNTS))
def test_segmentation_counts_are_stable(stem, analysed):
    saved = analysed(stem)
    kept = [d["kept_instances"] for d in saved.diagnostics]
    raw = [d["raw_instances"] for d in saved.diagnostics]
    assert kept == EXPECTED_COUNTS[stem]
    assert raw == kept, (
        "the minimum-size filter removed something; on this data it never has, "
        "so a detection that disappears is Cellpose's, not the filter's"
    )


@pytest.mark.parametrize("stem,n_tracks", sorted(EXPECTED_TRACKS.items()))
def test_track_counts_are_stable(stem, n_tracks, analysed):
    saved = analysed(stem)
    assert len(tracks_of(saved)) == n_tracks == saved.manifest["results"]["n_tracks"]


def test_t1_is_one_continuous_cell(analysed):
    """Verified by eye against the overlay: one cell, frames 4 to 16."""
    saved = analysed("052924_t1")
    (rows,) = tracks_of(saved).values()
    assert [r["frame"] for r in rows] == list(range(4, 17))
    assert all(r["gap_frames"] <= 1 for r in rows[1:])

    (summary,) = saved.summaries
    assert summary["n_observations"] == 13
    assert summary["n_gaps"] == 0
    # The cell migrates down the channel, then stalls near the outlet. 2.0
    # measures no direction, so the claim is the net displacement itself.
    assert summary["net_displacement_um"] > 80
    assert 0.2 < summary["mean_speed_um_per_min"] < 1.5
    assert summary["mean_speed_um_per_hr"] == pytest.approx(summary["mean_speed_um_per_min"] * 60)


def test_t3_keeps_two_cells_apart(analysed):
    """Both cells are visible in frames 0 and 1 and must not be merged."""
    saved = analysed("052924_t3_dual")
    frame0 = [rows for rows in tracks_of(saved).values() if rows[0]["frame"] == 0]
    assert len(frame0) == 2, "the two cells present at frame 0 became one identity"
    positions = sorted(rows[0]["y_px"] for rows in frame0)
    assert positions[1] - positions[0] > 100, "the two identities are the same object"


@pytest.mark.parametrize("stem", sorted(EXPECTED_TRACKS))
def test_no_detection_is_claimed_twice_and_every_row_joins(stem, analysed):
    """(frame, det_label) is unique in detections.csv and every track row finds its detection.

    Recovered detections are relabelled above the frame's primary labels, so
    the obvious join between the two files stays exact after recovery.
    """
    saved = analysed(stem)
    keys = [(d["frame"], d["label"]) for d in saved.detections]
    assert len(keys) == len(set(keys))
    assert len(saved.detections) == saved.manifest["results"]["n_detections"]
    index = set(keys)
    seen: set[tuple[int, int]] = set()
    for row in saved.tracks:
        key = (row["frame"], row["det_label"])
        assert key in index, f"track row {key} has no detection"
        assert key not in seen, f"detection {key} was used twice"
        seen.add(key)


@pytest.mark.parametrize("stem", sorted(EXPECTED_TRACKS))
def test_recovery_attempts_name_final_track_ids(stem, analysed):
    """Critique C4: recovery_attempts.csv joins tracks.csv on the FINAL id."""
    saved = analysed(stem)
    by_key = {(r["frame"], r["det_label"]): r["track_id"] for r in saved.tracks}
    ids = {r["track_id"] for r in saved.tracks}
    for attempt in saved.recovery:
        if attempt["track_id"] is not None:
            assert attempt["track_id"] in ids
        if attempt["recovered"]:
            assert by_key.get((attempt["frame"], attempt["det_label"])) == attempt["track_id"]


def test_t3_does_not_bridge_the_six_frame_gap(analysed):
    """The object at frame 10 is 6 frames after the last observation.

    With max_gap = 3 the tracker may bridge at most 4 frames (global gap
    closing included), so this becomes a separate single-observation track.
    This is the *gap rule* acting, not a judgement that the object is a
    different cell (see the unlinked-start test below).
    """
    saved = analysed("052924_t3_dual")
    late = [rows for rows in tracks_of(saved).values() if rows[0]["frame"] == 10]
    assert len(late) == 1 and len(late[0]) == 1
    limit = saved.manifest["tracking"]["max_delta_frames"]
    assert all(r["gap_frames"] <= limit for r in saved.tracks if r["observation_index"] > 0)


def test_empty_stack_is_safe_and_writes_valid_files(analysed):
    saved = analysed("052924_t2_empty")
    for name in (
        pipeline.F_TRACKS, pipeline.F_DETECTIONS, pipeline.F_SUMMARY, pipeline.F_MSD,
        pipeline.F_DIAGNOSTICS, pipeline.F_EVENTS, pipeline.F_QC, pipeline.F_RECOVERY,
        pipeline.F_UNLINKED, pipeline.F_MASKS, pipeline.F_MANIFEST,
    ):
        assert (saved.directory / name).exists(), f"{name} was not written"
    # Two detections, in frames 1 and 2 only.
    assert saved.manifest["results"]["n_detections_primary"] == 2
    assert sorted(d["frame"] for d in saved.detections) == [1, 2]
    assert len(tracks_of(saved)) == 1


def test_calibration_is_read_from_every_supplied_file():
    for stem in sorted(EXPECTED_TRACKS):
        metadata = read_metadata(SAMPLE_DIR / f"{stem}.tif")
        assert metadata.frame_interval_min.value == pytest.approx(FRAME_INTERVAL_MIN, rel=1e-6)
        assert metadata.pixel_size_um.value == pytest.approx(PIXEL_SIZE_UM, rel=1e-9)
        assert metadata.axes == "TYX" and metadata.dimensionality == "2D"


@pytest.mark.parametrize("stem", WIDE_FIELDS)
def test_wide_field_tracks_never_cross_a_lane(stem, analysed):
    """The scientific claim that makes wide-field tracking defensible.

    The lanes are measured from the walls and applied as a gate; no track of
    the result may visit two of them.
    """
    saved = analysed(stem)
    geo = saved.manifest["channel_geometry"]
    assert geo["source"] == "channel_ridges"
    assert geo["applied"] is True and geo["lane_gate_applied"] is True
    assert geo["n_lanes"] == 6
    assert 75 < (geo["pitch_px"] or 0) < 92
    for tid, rows in tracks_of(saved).items():
        lanes = {r["channel"] for r in rows}
        assert len(lanes) == 1, f"track {tid} spans lanes {sorted(lanes)}"
        assert lanes != {-1}, f"track {tid} is outside every lane"


def _steps(saved: SavedAnalysis):
    """(track id, frame, um/min) of every step, from tracks.csv's own speed column."""
    for tid, rows in tracks_of(saved).items():
        for row in rows[1:]:
            if row["speed_um_per_min"] is not None:
                yield tid, row["frame"], float(row["speed_um_per_min"])


@pytest.mark.parametrize(
    "stem",
    [
        pytest.param(stem, marks=pytest.mark.xfail(strict=True, reason=KNOWN_DEFECTS[stem]))
        if stem in KNOWN_DEFECTS else stem
        for stem in sorted(EXPECTED_TRACKS)
    ],
)
def test_no_step_is_faster_than_any_baseline_step(stem, analysed):
    """A guard the suite lacked: a track count can pin a physically implausible track.

    tracks.csv's speed is per step (distance from the previous observation /
    elapsed time), so a gap-bridging link is measured at its true rate.
    """
    saved = analysed(stem)
    fast = [
        (tid, frame, round(v, 3)) for tid, frame, v in _steps(saved)
        if v > FASTEST_BASELINE_STEP_UM_PER_MIN * (1 + 1e-9)
    ]
    assert not fast, (
        f"steps faster than any v1.3.0 step ({FASTEST_BASELINE_STEP_UM_PER_MIN} um/min): "
        f"(track, frame, um/min) {fast}"
    )


@pytest.mark.parametrize("stem", sorted(EXPECTED_TRACKS))
def test_the_only_implausible_steps_are_the_known_ones(stem, analysed):
    """A known defect excuses its own step, never another one."""
    saved = analysed(stem)
    fast = {
        (tid, frame) for tid, frame, v in _steps(saved)
        if v > FASTEST_BASELINE_STEP_UM_PER_MIN * (1 + 1e-9)
    }
    assert fast == KNOWN_DEFECT_STEPS.get(stem, set())


@pytest.mark.parametrize("stem", sorted(EXPECTED_TRACKS))
def test_an_implausible_step_is_never_silent(stem, analysed):
    """Until a known defect is fixed, the reviewer must at least be warned about its track."""
    saved = analysed(stem)
    warned = {
        int(i["track_id"]) for i in saved.issues
        if i["severity"] in ("warning", "critical") and i["track_id"] is not None
    }
    for tid, frame, v in _steps(saved):
        if v > FASTEST_BASELINE_STEP_UM_PER_MIN * (1 + 1e-9):
            assert tid in warned, f"track {tid}'s {v:.2f} um/min step at frame {frame} has no QC warning"


def test_manifest_is_complete_and_reproducible(analysed):
    saved = analysed("052924_t1")
    m = saved.manifest
    assert m["schema_version"] == 2 == export.SCHEMA_VERSION
    assert "confinement" not in m
    assert m["environment"]["cellpose"].startswith("3.")
    # The model, pinned through the registry and by its own literal checksum.
    assert m["model"]["sha256"] == _validated_sha() == VALIDATED_SHA256
    assert m["model"]["model_id"] == model_registry.production_spec("2D").model_id
    assert m["segmentation"]["model_sha256"] == VALIDATED_SHA256
    assert m["calibration"]["pixel_size_um_source"] == "nd2_info"
    assert m["calibration"]["frame_interval_min_source"] == "nd2_info"
    assert m["calibration"]["z_step_um"] is None and m["calibration"]["anisotropy"] is None
    assert m["dimensionality"] == "2D" == m["input"]["dimensionality"]
    assert m["input"]["axes"] == "TYX" and m["input"]["axes_reported"] == "TYX"
    assert m["input"]["shape"] == [18, 324, 90]
    assert m["input"]["source_frames"][0] == 1
    geo = m["channel_geometry"]
    assert geo["n_lanes"] == 1
    # The lane runs down the image, tilted a little: measured, never configured.
    assert -4 < geo["lanes"][0]["tilt_from_vertical_deg"] < -1
    assert m["tracking"]["max_delta_frames"] == m["tracking"]["max_gap"] + 1
    assert m["tracking"]["gate_chi2"] == 2 * m["tracking"]["unmatched_chi2"]
    assert m["segmentation"]["removed_instances_total"] == 0
    assert set(m["measurement"]) >= {"reference_point_px", "msd_min_pairs"}
    # Nothing overridden: what was used is what the file reports.
    cal, file = m["calibration"], m["calibration"]["reported_by_file"]
    assert cal["pixel_size_um"] == cal["pixel_size_y_um"] == file["pixel_size_x_um"]
    assert file["pixel_size_y_um"] == pytest.approx(file["pixel_size_x_um"], rel=1e-9)
    assert cal["anisotropic_pixels"] is False
    # The run-level speed called robust is the median over tracks.
    res = m["results"]
    assert res["speed_estimators"]["robust_estimator"] == "median_net_speed"
    assert res["median_net_speed_um_per_hr"] == pytest.approx(res["median_net_speed_um_per_min"] * 60)


def test_t3_late_object_is_reported_as_a_judgement_not_a_fact(analysed):
    """The frame-10 object is refused only by the gap limit, and says so.

    Measured: it sits a few px from where the earlier track ended, in the same
    lane, and would fit at a cost under the rejection threshold. The only
    thing refusing it is that 6 frames is longer than max_gap allows. That is
    a policy choice, and the software has to present it as one rather than
    implying the object is a different cell.
    """
    saved = analysed("052924_t3_dual")
    late = [u for u in saved.unlinked if u["starts_at_frame"] == 10]
    assert len(late) == 1
    evidence = late[0]
    tracking = saved.manifest["tracking"]

    assert evidence["nearest_earlier_track"] is not None
    assert evidence["gap_frames"] == 6
    assert evidence["gap_frames"] > tracking["max_delta_frames"]
    assert evidence["refused_because"] == "gap_too_long"
    assert evidence["distance_px"] is not None and evidence["distance_px"] < 20
    assert evidence["would_have_cost_chi2"] is not None
    assert evidence["would_have_cost_chi2"] < tracking["gate_chi2"], (
        "the fit itself is acceptable; only the gap refuses it, and the "
        "software must not imply otherwise"
    )
    assert "gap" in evidence["explanation"].lower()

    # It must still be reported to the reviewer, at warning level.
    codes = {(i["code"], i["severity"]) for i in saved.issues}
    assert ("unlinked_start", "warning") in codes


def test_unlinked_starts_file_is_written(analysed):
    saved = analysed("052924_t3_dual")
    text = (saved.directory / pipeline.F_UNLINKED).read_text(encoding="utf-8")
    assert text.startswith("track_id,starts_at_frame,")
    assert "gap_too_long" in text


def test_t1_has_no_unlinked_starts(analysed):
    """One continuous cell leaves nothing to explain."""
    saved = analysed("052924_t1")
    assert saved.unlinked == [] or all(u["nearest_earlier_track"] is None for u in saved.unlinked)


def test_the_defaults_tested_are_the_defaults_shipped(analysed):
    """A result produced with other tracking settings would test something else."""
    saved = analysed("052924_t1")
    tracking = saved.manifest["tracking"]
    defaults = TrackingConfig()
    for key in ("max_gap", "unmatched_chi2", "max_speed_um_per_min", "channel_constraint",
                "position_sigma_um", "global_gap_closing"):
        assert tracking[key] == getattr(defaults, key), key
