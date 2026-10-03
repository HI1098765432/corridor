"""Ground-truth tests for the referee (Bot 8).

The referee turns evidence into a state by fixed rules, so the truth here is
planted in the evidence and the test asserts the state and -- because the
project's law is *measure, never assert* -- that the decision names the numbers
that produced it.  No model, no network: every value is constructed.
"""

from __future__ import annotations

from corridor.engine.consensus import (
    ARTIFACT,
    CHANNEL_WALL,
    OUTSIDE_DEVICE,
    VALID_CELL_REGION,
    ConsensusConfig,
    ConsensusState,
    Evidence,
    referee,
)


def test_false_positive_in_a_confident_wall_is_rejected():
    """A proposal that is mostly inside a confident CHANNEL_WALL is device
    structure, not a cell, and is rejected unconditionally -- even with an
    otherwise strong proposer, because the walls are exactly what a zero-shot
    proposer latches onto (ENGINE_4D SS0, cpsam_v2 F1 0.00 on the channels)."""
    ev = Evidence(
        object_id=7,
        t=3,
        cellprob=5.0,  # a strong proposer -- must not rescue a wall fragment
        atlas_class=CHANNEL_WALL,
        atlas_wall_fraction=0.82,
        atlas_class_confidence=0.90,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.REJECTED
    # The decision explains itself with its own measured values.
    joined = " ".join(decision.reasons)
    assert "wall_fraction=0.82" in joined and "CHANNEL_WALL" in joined


def test_wall_fragment_below_the_fraction_is_not_force_rejected():
    """A cell that merely grazes a wall (small wall fraction) is not rejected
    by the wall rule; the ordinary tally decides it."""
    ev = Evidence(
        object_id=1,
        t=0,
        cellprob=5.0,
        atlas_class=VALID_CELL_REGION,  # dominant class is a valid region
        atlas_wall_fraction=0.10,
        atlas_class_confidence=0.9,
        past_support=True,
        future_support=True,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.CONFIRMED


def test_missing_object_bracketed_in_time_is_a_recovery_candidate():
    """Absent at t, present at t-1 and t+1, structures matching above the
    matcher cliff and plausible motion -> TEMPORAL_RECOVERY_CANDIDATE."""
    ev = Evidence(
        object_id=12,
        t=5,
        present_now=False,
        past_support=True,
        future_support=True,
        structure_match=0.71,  # IoU of the two brackets, above 0.5
        motion_speed_px_per_frame=8.0,
        max_speed_px_per_frame=71.0,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.TEMPORAL_RECOVERY_CANDIDATE
    joined = " ".join(decision.reasons)
    assert "structure IoU 0.71" in joined and "t-1 and t+1" in joined


def test_missing_object_with_one_bracket_is_not_recovered():
    """Only one side present -> the recovery is explicitly declined, not
    silently invented."""
    ev = Evidence(
        object_id=12,
        t=5,
        present_now=False,
        past_support=True,
        future_support=False,
        structure_match=0.9,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.REJECTED
    assert "no t+1 match" in " ".join(decision.reasons)


def test_missing_object_with_poor_structure_match_is_not_recovered():
    """Both brackets present but their structures disagree (below the 0.5
    matcher cliff) -> no object to recover."""
    ev = Evidence(
        object_id=3,
        t=2,
        present_now=False,
        past_support=True,
        future_support=True,
        structure_match=0.30,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.REJECTED
    assert "structure IoU 0.30 < 0.50" in " ".join(decision.reasons)


def test_two_pillars_confirm():
    """Strong proposer plus a valid atlas region (two independent pillars),
    nothing against -> CONFIRMED."""
    ev = Evidence(
        object_id=1,
        t=0,
        cellprob=3.0,
        atlas_class=VALID_CELL_REGION,
        atlas_class_confidence=0.95,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.CONFIRMED


def test_one_pillar_is_only_likely():
    """A valid atlas region but a neutral proposer (no probability) and no
    temporal match -> one pillar -> LIKELY. This is the honest state of a
    given-masks object in a skeleton run."""
    ev = Evidence(
        object_id=1,
        t=0,
        cellprob=None,
        atlas_class=VALID_CELL_REGION,
        atlas_class_confidence=0.9,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.LIKELY


def test_support_and_against_is_ambiguous():
    """A strong proposer (support) but an implausible speed (against) ->
    AMBIGUOUS: the referee reports the conflict rather than resolving it by a
    hidden weight."""
    ev = Evidence(
        object_id=9,
        t=4,
        cellprob=4.0,
        atlas_class=VALID_CELL_REGION,  # +1 support as well
        atlas_class_confidence=0.9,
        motion_speed_px_per_frame=120.0,
        max_speed_px_per_frame=71.0,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.AMBIGUOUS
    joined = " ".join(decision.reasons)
    assert "120.0 px/frame > ceiling 71.0" in joined


def test_weak_proposer_alone_is_rejected():
    """A weak proposer with nothing supporting it -> REJECTED (one against,
    zero support)."""
    ev = Evidence(
        object_id=2,
        t=1,
        cellprob=-3.0,  # below cellprob_likely (0.0)
        atlas_class="UNKNOWN",
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.REJECTED


def test_confident_outside_device_is_rejected():
    ev = Evidence(
        object_id=5,
        t=0,
        cellprob=5.0,
        atlas_class=OUTSIDE_DEVICE,
        atlas_class_confidence=0.85,
    )
    decision = referee(ev)
    assert decision.state is ConsensusState.REJECTED
    assert "OUTSIDE_DEVICE" in " ".join(decision.reasons)


def test_confident_artifact_counts_against():
    """A confident ARTIFACT class is evidence-against; with one support pillar
    it lands AMBIGUOUS, with none it is REJECTED."""
    supported = Evidence(
        object_id=1, t=0, cellprob=3.0,  # strong proposer: one support
        atlas_class=ARTIFACT, atlas_class_confidence=0.9,
    )
    assert referee(supported).state is ConsensusState.AMBIGUOUS

    unsupported = Evidence(
        object_id=1, t=0, cellprob=1.0,  # between likely and confirm: neutral
        atlas_class=ARTIFACT, atlas_class_confidence=0.9,
    )
    assert referee(unsupported).state is ConsensusState.REJECTED


def test_config_rejects_inverted_strength_thresholds():
    import pytest

    with pytest.raises(ValueError):
        ConsensusConfig(cellprob_confirm=0.0, cellprob_likely=2.0)


def test_every_decision_lists_its_evidence_values():
    """The law: a decision must carry the numbers behind it. The tally line is
    always present and names the support/against counts."""
    ev = Evidence(object_id=1, t=0, cellprob=3.0, atlas_class=VALID_CELL_REGION,
                  atlas_class_confidence=0.9)
    decision = referee(ev)
    assert decision.reasons  # never empty
    assert any("tally:" in r for r in decision.reasons)
    # The row carries both the evidence and the verdict for consensus.csv.
    row = decision.to_row()
    assert row["state"] == "CONFIRMED"
    assert row["cellprob"] == 3.0 and row["atlas_class"] == VALID_CELL_REGION
