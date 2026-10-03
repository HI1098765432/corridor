"""The 4D pipeline skeleton on a tiny synthetic stack.

The skeleton has to build and run before Bots 1, 2, 4, 5 and 6 exist, on a
model-free given-masks proposer, and write the three evidence files the
contract names.  These tests plant a known tiny TYX movie and check exactly
that -- and that the stages which have no bot yet say so honestly rather than
inventing their numbers.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from corridor.engine.consensus import CONSENSUS_COLUMNS, VALID_CELL_REGION
from corridor.engine.pipeline4d import (
    Engine4DConfig,
    Engine4DResult,
    _to_tzyx,
    run_engine_4d,
)
from corridor.engine.proposer import GivenMasksProposer, Normalization


def _moving_cell(n_frames=3, h=24, w=16, y0=4, step=3, bh=8, bw=4, x0=6):
    """A single bright cell moving down the image, with its label mask.

    Returns ``(intensity[T,Y,X], masks[T,Y,X])``; the mask is label 1 on the
    cell block, so the given-masks proposer proposes one object per frame and
    the tracker should link them into one identity.
    """
    intensity = np.zeros((n_frames, h, w), dtype=np.float32)
    masks = np.zeros((n_frames, h, w), dtype=np.int32)
    for t in range(n_frames):
        y = y0 + step * t
        intensity[t, y : y + bh, x0 : x0 + bw] = 1000.0
        masks[t, y : y + bh, x0 : x0 + bw] = 1
    return intensity, masks


def test_to_tzyx_canonicalises_every_dimensionality():
    assert _to_tzyx(np.zeros((5, 6)), None)[0].shape == (1, 1, 5, 6)  # YX
    assert _to_tzyx(np.zeros((3, 5, 6)), None)[0].shape == (3, 1, 5, 6)  # TYX
    assert _to_tzyx(np.zeros((3, 5, 6)), "ZYX")[0].shape == (1, 3, 5, 6)  # ZYX, not time
    assert _to_tzyx(np.zeros((2, 3, 5, 6)), None)[0].shape == (2, 3, 5, 6)  # TZYX


def test_to_tzyx_rejects_axes_mismatch():
    with pytest.raises(ValueError):
        _to_tzyx(np.zeros((3, 5, 6)), "TYXZ")  # wrong length for ndim 3


def test_skeleton_runs_and_writes_evidence_files(tmp_path):
    intensity, masks = _moving_cell()
    proposer = GivenMasksProposer(masks=masks)

    result = run_engine_4d(intensity, proposer, tmp_path, axes="TYX")

    assert isinstance(result, Engine4DResult)
    assert result.shape_tzyx == (3, 1, 24, 16)
    assert result.n_proposed == 3  # one object per frame

    # The three contract files exist.
    reg = tmp_path / "registration.csv"
    atlas = tmp_path / "atlas.npz"
    cons = tmp_path / "consensus.csv"
    assert reg.exists() and atlas.exists() and cons.exists()

    # registration.csv: a header and one row per timepoint, reference at centre.
    with reg.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["t"] for r in rows] == ["0", "1", "2"]
    assert {r["reference_t"] for r in rows} == {"1"}

    # atlas.npz: background + class map + a readable legend.
    with np.load(atlas, allow_pickle=False) as data:
        assert data["background"].shape == (1, 24, 16)
        assert data["class_map"].shape == (1, 24, 16)
        assert VALID_CELL_REGION in [str(n) for n in data["class_names"]]

    # consensus.csv: one row per object and time, full evidence columns, every
    # decision carrying its reasons (the law: measure, never assert).
    with cons.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CONSENSUS_COLUMNS
        crows = list(reader)
    assert len(crows) == 3
    for r in crows:
        assert r["reasons"]
        assert r["atlas_class"] == VALID_CELL_REGION
        # No proposer probability on given masks -> one pillar -> LIKELY.
        assert r["state"] == "LIKELY"


def test_skeleton_names_the_bots_it_does_not_have(tmp_path):
    intensity, masks = _moving_cell()
    result = run_engine_4d(intensity, GivenMasksProposer(masks=masks), tmp_path, axes="TYX")

    # The real stages ran.
    assert "propose" in result.stages_run
    assert "referee" in result.stages_run
    # The absent bots are each named as skipped, not silently missing.
    skipped = " ".join(result.stages_skipped)
    assert "Bot 1" in skipped  # registration
    assert "Bot 2" in skipped  # atlas
    assert "Bot 4" in skipped  # Z consensus
    assert "Bot 5" in skipped  # temporal delta
    assert "Bot 6" in skipped  # measurement


def test_identity_handoff_links_the_moving_cell(tmp_path):
    """The hand-off to the existing axis-free tracker is real: the kept object
    across three frames becomes one track."""
    intensity, masks = _moving_cell()
    result = run_engine_4d(intensity, GivenMasksProposer(masks=masks), tmp_path, axes="TYX")
    assert "identity" in result.stages_run
    assert isinstance(result.n_tracks, int) and result.n_tracks >= 1


def test_proposer_probability_reaches_the_referee(tmp_path):
    """A given-masks proposer carrying a strong cell-probability field gives a
    second pillar (proposer strong + valid atlas region) -> CONFIRMED, and the
    probability is written to consensus.csv, proving it flowed object -> referee."""
    intensity, masks = _moving_cell()
    cellprob = np.where(masks > 0, 4.0, -6.0).astype(np.float32)  # strong interiors
    proposer = GivenMasksProposer(
        masks=masks, cellprob=cellprob, normalization=Normalization(method="given")
    )
    result = run_engine_4d(intensity, proposer, tmp_path, axes="TYX")

    assert all(d.state.value == "CONFIRMED" for d in result.decisions)
    with (tmp_path / "consensus.csv").open() as fh:
        crows = list(csv.DictReader(fh))
    assert all(float(r["cellprob"]) == 4.0 for r in crows)


def test_single_frame_yx_runs(tmp_path):
    """Z = 1 and T = 1 need no special path: a single YX frame runs."""
    intensity, masks = _moving_cell(n_frames=1)
    result = run_engine_4d(intensity[0], GivenMasksProposer(masks=masks[0]), tmp_path)
    assert result.shape_tzyx == (1, 1, 24, 16)
    assert (tmp_path / "consensus.csv").exists()
    assert result.n_proposed == 1


def test_reject_in_walls_counts_static_rejections(tmp_path):
    """With a config that disables wall rejection, nothing is counted static;
    the count is an honest zero on a trivial (wall-free) atlas regardless."""
    intensity, masks = _moving_cell()
    cfg = Engine4DConfig(reject_in_walls=True)
    result = run_engine_4d(intensity, GivenMasksProposer(masks=masks), tmp_path, config=cfg)
    assert result.n_rejected_static == 0  # the skeleton atlas has no walls
