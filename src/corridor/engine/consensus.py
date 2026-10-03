"""Bot 8 -- the referee.

The proposer (Bot 3) asserts masks; the referee decides what to believe about
each one, by combining deterministic evidence into an explicit state.  No
language model, no learned weight: the rules are fixed thresholds, applied in a
fixed order, and **every decision carries the exact numbers that produced it**
(``Decision.reasons``), so a reviewer can audit any call without rerunning
anything.  This is the project's law in this module -- measure, never assert:
a state is a function of evidence values, not a vote with hidden coefficients.

The evidence per object (``docs/ENGINE_4D.md`` SS2, Bot 8):

* **proposer** -- its mean cell-probability over the object (backend-scaled);
* **static-atlas class** -- ``CHANNEL_WALL``, ``VALID_CELL_REGION`` etc. from
  Bot 2, with how much of the object lies in a wall and the class confidence;
* **Z support** -- how many slices the object links across (Bot 4);
* **past and future temporal support** -- matched at t-1 / t+1, with the
  forward-backward error ``E_FB`` (Bot 5);
* **motion plausibility** -- object speed against the physical ceiling;
* **shape plausibility** -- is this a cell-shaped object at all.

The states (``docs/ENGINE_4D.md`` SS2):

* ``CONFIRMED`` -- corroborated by at least two independent pillars, nothing
  against it;
* ``LIKELY`` -- one pillar supports it, nothing against;
* ``AMBIGUOUS`` -- support and evidence-against both present, or no evidence
  either way;
* ``REJECTED`` -- evidence against and nothing supporting; **and,
  unconditionally, a proposal inside a confident ``CHANNEL_WALL``** (the walls
  are the static structure a zero-shot proposer latches onto -- cpsam_v2 scored
  F1 0.00 by outlining the ~320x93 px channel, ``docs/ENGINE_4D.md`` SS0);
* ``TEMPORAL_RECOVERY_CANDIDATE`` -- the object is *absent* at t but present at
  t-1 and t+1 with matching pixel structure and plausible motion, so the gap is
  a review/recovery case rather than a true absence.

Why these pillars and this order recover real accuracy: of 64 held-out errors
(``docs/error_table.json``), ~13 are suspect labels (the model was right, the
human label was missing) and ~17 are boundary near-misses the model *found*
(median IoU 0.43 against the 0.5 cliff).  A proposer-strong, shape-plausible,
temporally corroborated object that a single-frame IoU-0.5 score would discard
is exactly what the referee is built to keep; a bright static wall fragment is
exactly what it is built to drop.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------
# Atlas classes (Bot 2 produces them; defined here, the referee's home, so the
# one consumer that must reason about them owns the names -- static_atlas.py
# imports these rather than redefining them, so the strings cannot drift apart
# into two modules, one filename apart).
# --------------------------------------------------------------------------

STATIC_BACKGROUND = "STATIC_BACKGROUND"
CHANNEL_WALL = "CHANNEL_WALL"
OUTSIDE_DEVICE = "OUTSIDE_DEVICE"
ARTIFACT = "ARTIFACT"
VALID_CELL_REGION = "VALID_CELL_REGION"
UNKNOWN = "UNKNOWN"

#: Every atlas class, and a small integer code for each so a class map can be
#: stored as int8 in ``atlas.npz`` with this legend beside it.
ATLAS_CLASSES = (
    STATIC_BACKGROUND,
    CHANNEL_WALL,
    OUTSIDE_DEVICE,
    ARTIFACT,
    VALID_CELL_REGION,
    UNKNOWN,
)
ATLAS_CLASS_CODES = {name: code for code, name in enumerate(ATLAS_CLASSES)}


class ConsensusState(str, Enum):
    """The referee's verdict on one object.  ``str`` so it writes to a CSV cell
    and compares against a plain string without ceremony."""

    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED = "REJECTED"
    TEMPORAL_RECOVERY_CANDIDATE = "TEMPORAL_RECOVERY_CANDIDATE"


# --------------------------------------------------------------------------
# Thresholds -- every one explicit, none learned.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusConfig:
    """The referee's fixed thresholds.

    The proposer-strength thresholds are the only backend-dependent numbers:
    they are on the proposer's cell-probability scale (for Cellpose, the
    ``cellprob`` field, whose mask threshold is 0.0 and whose interiors are
    positive), and they are config, to be calibrated per backend rather than
    asserted once for all.  The structural ones below are tied to measured
    facts and to the project's one matcher.
    """

    # -- proposer strength (backend-scaled; Cellpose cellprob convention) ----
    cellprob_confirm: float = 2.0
    cellprob_likely: float = 0.0

    # -- channel-wall rejection ----------------------------------------------
    #: A proposal with at least this fraction of its pixels inside a wall, where
    #: the atlas is at least ``wall_confidence_min`` sure it is a wall, is
    #: device structure, not a cell. Half the mask is the explicit line.
    wall_reject_fraction: float = 0.5
    wall_confidence_min: float = 0.7

    # -- other atlas classes that argue against a cell -----------------------
    outside_confidence_min: float = 0.7
    artifact_confidence_min: float = 0.7

    # -- temporal recovery ---------------------------------------------------
    #: IoU between the t-1 and t+1 structure bracketing the gap, required to
    #: call a missing object a recovery candidate. 0.5 is the project's one
    #: matcher cliff (``core.metrics.DEFAULT_IOU``): if the brackets would not
    #: even match each other, there is no object to recover.
    recovery_structure_min: float = 0.5

    def __post_init__(self) -> None:
        # Keep the invariant the two strength thresholds encode: "confirm" is a
        # stronger interior than "likely", or the LIKELY band would be empty and
        # the tally below would misclassify silently.
        if self.cellprob_confirm < self.cellprob_likely:
            raise ValueError(
                f"cellprob_confirm ({self.cellprob_confirm}) must be >= "
                f"cellprob_likely ({self.cellprob_likely})"
            )


# --------------------------------------------------------------------------
# Evidence and decision records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Every number behind one object's decision, at one (t, z).

    Fields default to "no information": a ``None`` probability, a neutral
    ``UNKNOWN`` atlas class, no temporal match, an infinite speed ceiling.  A
    bot that has not run (the pipeline builds before they all land) leaves its
    fields at these defaults, and the referee then simply has fewer pillars --
    it never invents a value, so a skeleton run reports honest, under-supported
    states rather than confident wrong ones.
    """

    object_id: int
    t: int
    z: int = 0
    #: False means the object is not in this frame's proposal; the referee then
    #: runs the recovery branch instead of the present-object rules.
    present_now: bool = True

    # proposer (Bot 3)
    cellprob: float | None = None
    proposer_backend: str = ""

    # static atlas (Bot 2)
    atlas_class: str = UNKNOWN
    atlas_wall_fraction: float = 0.0
    atlas_class_confidence: float = 0.0

    # Z consensus (Bot 4)
    z_support: int = 1
    z_link_cost: float | None = None

    # temporal delta (Bot 5)
    past_support: bool = False
    future_support: bool = False
    e_fb_past: float | None = None
    e_fb_future: float | None = None

    # motion plausibility
    motion_speed_px_per_frame: float | None = None
    max_speed_px_per_frame: float = float("inf")

    # shape plausibility
    shape_plausible: bool = True
    aspect_ratio: float | None = None
    area_px: float | None = None

    #: IoU of the t-1 and t+1 structure at a gap; read only when present_now is
    #: False (the recovery branch).
    structure_match: float | None = None

    def to_row(self) -> dict:
        return {
            "object_id": self.object_id,
            "t": self.t,
            "z": self.z,
            "present_now": self.present_now,
            "cellprob": self.cellprob,
            "proposer_backend": self.proposer_backend,
            "atlas_class": self.atlas_class,
            "atlas_wall_fraction": round(self.atlas_wall_fraction, 4),
            "atlas_class_confidence": round(self.atlas_class_confidence, 4),
            "z_support": self.z_support,
            "z_link_cost": self.z_link_cost,
            "past_support": self.past_support,
            "future_support": self.future_support,
            "e_fb_past": self.e_fb_past,
            "e_fb_future": self.e_fb_future,
            "motion_speed_px_per_frame": self.motion_speed_px_per_frame,
            "max_speed_px_per_frame": self.max_speed_px_per_frame,
            "shape_plausible": self.shape_plausible,
            "aspect_ratio": self.aspect_ratio,
            "area_px": self.area_px,
            "structure_match": self.structure_match,
        }


@dataclass(frozen=True)
class Decision:
    """The referee's verdict on one object, with the numbers that produced it.

    ``reasons`` is ordered: the rule that fixed the state is stated in terms of
    its own measured values (e.g. ``"REJECT: wall_fraction=0.82 >= 0.50 in a
    confident CHANNEL_WALL (conf 0.90 >= 0.70)"``), so a decision explains
    itself without the evidence table beside it.
    """

    object_id: int
    t: int
    z: int
    state: ConsensusState
    reasons: tuple[str, ...]
    evidence: Evidence

    def to_row(self) -> dict:
        row = self.evidence.to_row()
        row["state"] = self.state.value
        row["reasons"] = " ; ".join(self.reasons)
        return row


#: Column order for ``consensus.csv`` (one row per object and time), evidence
#: first, then the verdict and its reasons.
CONSENSUS_COLUMNS = [
    "t",
    "z",
    "object_id",
    "state",
    "present_now",
    "cellprob",
    "proposer_backend",
    "atlas_class",
    "atlas_wall_fraction",
    "atlas_class_confidence",
    "z_support",
    "z_link_cost",
    "past_support",
    "future_support",
    "e_fb_past",
    "e_fb_future",
    "motion_speed_px_per_frame",
    "max_speed_px_per_frame",
    "shape_plausible",
    "aspect_ratio",
    "area_px",
    "structure_match",
    "reasons",
]


# --------------------------------------------------------------------------
# The referee
# --------------------------------------------------------------------------


def referee(ev: Evidence, config: ConsensusConfig | None = None) -> Decision:
    """Decide one object's state from its evidence, deterministically.

    Order (first applicable wins for the guard rules; the tally decides the
    rest):

    1. **Absent object** -> recovery branch (``_referee_absent``).
    2. **Confident channel wall** -> ``REJECTED`` unconditionally.
    3. **Confidently outside the device** -> ``REJECTED``.
    4. Otherwise tally the support pillars and the evidence-against, and map the
       counts to ``CONFIRMED`` / ``LIKELY`` / ``AMBIGUOUS`` / ``REJECTED``.
    """
    cfg = config or ConsensusConfig()

    if not ev.present_now:
        return _referee_absent(ev, cfg)

    reasons: list[str] = []

    # -- Rule 2: a proposal inside a confident wall is device, not cell. ------
    if (
        ev.atlas_class == CHANNEL_WALL
        and ev.atlas_wall_fraction >= cfg.wall_reject_fraction
        and ev.atlas_class_confidence >= cfg.wall_confidence_min
    ):
        reasons.append(
            f"REJECT: wall_fraction={ev.atlas_wall_fraction:.2f} >= "
            f"{cfg.wall_reject_fraction:.2f} in a CHANNEL_WALL "
            f"(conf {ev.atlas_class_confidence:.2f} >= {cfg.wall_confidence_min:.2f})"
        )
        return _decide(ev, ConsensusState.REJECTED, reasons)

    # -- Rule 3: confidently outside the device. ------------------------------
    if (
        ev.atlas_class == OUTSIDE_DEVICE
        and ev.atlas_class_confidence >= cfg.outside_confidence_min
    ):
        reasons.append(
            f"REJECT: OUTSIDE_DEVICE at conf {ev.atlas_class_confidence:.2f} >= "
            f"{cfg.outside_confidence_min:.2f}"
        )
        return _decide(ev, ConsensusState.REJECTED, reasons)

    # -- Rule 4: tally pillars and evidence-against. --------------------------
    n_support = 0
    n_against = 0

    # Pillar A: proposer strength.
    if ev.cellprob is not None:
        if ev.cellprob >= cfg.cellprob_confirm:
            n_support += 1
            reasons.append(
                f"support: proposer cellprob {ev.cellprob:.2f} >= "
                f"{cfg.cellprob_confirm:.2f} (strong)"
            )
        elif ev.cellprob < cfg.cellprob_likely:
            n_against += 1
            reasons.append(
                f"against: proposer cellprob {ev.cellprob:.2f} < "
                f"{cfg.cellprob_likely:.2f} (weak)"
            )
        # between likely and confirm: present but not strong -- neutral.

    # Pillar B: static-atlas class.
    if ev.atlas_class == VALID_CELL_REGION:
        n_support += 1
        reasons.append("support: atlas class VALID_CELL_REGION")
    elif (
        ev.atlas_class == ARTIFACT
        and ev.atlas_class_confidence >= cfg.artifact_confidence_min
    ):
        n_against += 1
        reasons.append(
            f"against: atlas class ARTIFACT at conf "
            f"{ev.atlas_class_confidence:.2f} >= {cfg.artifact_confidence_min:.2f}"
        )

    # Pillar C: temporal corroboration (past and/or future).
    temporal = int(ev.past_support) + int(ev.future_support)
    if temporal >= 1:
        n_support += 1
        where = []
        if ev.past_support:
            where.append("t-1" + (f" (E_FB {ev.e_fb_past:.2f})" if ev.e_fb_past is not None else ""))
        if ev.future_support:
            where.append("t+1" + (f" (E_FB {ev.e_fb_future:.2f})" if ev.e_fb_future is not None else ""))
        reasons.append("support: temporal match at " + ", ".join(where))

    # Evidence-against that are not pillars: implausible motion, bad shape.
    if (
        ev.motion_speed_px_per_frame is not None
        and ev.motion_speed_px_per_frame > ev.max_speed_px_per_frame
    ):
        n_against += 1
        reasons.append(
            f"against: speed {ev.motion_speed_px_per_frame:.1f} px/frame > ceiling "
            f"{ev.max_speed_px_per_frame:.1f}"
        )
    if not ev.shape_plausible:
        n_against += 1
        reasons.append("against: shape implausible for a cell")

    state = _tally_state(n_support, n_against)
    reasons.append(f"tally: {n_support} support, {n_against} against -> {state.value}")
    return _decide(ev, state, reasons)


def _tally_state(n_support: int, n_against: int) -> ConsensusState:
    """Map the evidence counts to a state -- the only place the counts become a
    verdict, so the mapping is one readable table, not scattered branches."""
    if n_against >= 1:
        return ConsensusState.AMBIGUOUS if n_support >= 1 else ConsensusState.REJECTED
    if n_support >= 2:
        return ConsensusState.CONFIRMED
    if n_support == 1:
        return ConsensusState.LIKELY
    return ConsensusState.AMBIGUOUS  # no evidence either way


def _referee_absent(ev: Evidence, cfg: ConsensusConfig) -> Decision:
    """The object is not in this frame's proposal.  It becomes a recovery
    candidate only when both temporal brackets hold it, their structures match
    above the matcher cliff, and the implied motion is plausible; otherwise the
    recovery is explicitly declined (``REJECTED``), never silently created."""
    motion_ok = (
        ev.motion_speed_px_per_frame is None
        or ev.motion_speed_px_per_frame <= ev.max_speed_px_per_frame
    )
    structure_ok = (
        ev.structure_match is not None
        and ev.structure_match >= cfg.recovery_structure_min
    )
    if ev.past_support and ev.future_support and structure_ok and motion_ok:
        reasons = [
            "RECOVERY: absent at t but matched at t-1 and t+1",
            f"structure IoU {ev.structure_match:.2f} >= {cfg.recovery_structure_min:.2f}",
            (
                f"speed {ev.motion_speed_px_per_frame:.1f} <= "
                f"{ev.max_speed_px_per_frame:.1f} px/frame"
                if ev.motion_speed_px_per_frame is not None
                else "motion not constrained"
            ),
        ]
        return _decide(ev, ConsensusState.TEMPORAL_RECOVERY_CANDIDATE, reasons)

    missing = []
    if not ev.past_support:
        missing.append("no t-1 match")
    if not ev.future_support:
        missing.append("no t+1 match")
    if not structure_ok:
        sm = "none" if ev.structure_match is None else f"{ev.structure_match:.2f}"
        missing.append(f"structure IoU {sm} < {cfg.recovery_structure_min:.2f}")
    if not motion_ok:
        missing.append(
            f"speed {ev.motion_speed_px_per_frame:.1f} > {ev.max_speed_px_per_frame:.1f}"
        )
    reasons = ["REJECT: absent at t and recovery declined (" + "; ".join(missing) + ")"]
    return _decide(ev, ConsensusState.REJECTED, reasons)


def _decide(ev: Evidence, state: ConsensusState, reasons: Sequence[str]) -> Decision:
    return Decision(
        object_id=ev.object_id, t=ev.t, z=ev.z, state=state,
        reasons=tuple(reasons), evidence=ev,
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_consensus_csv(decisions: Sequence[Decision], path: str | Path) -> Path:
    """Write ``consensus.csv`` -- one row per object and time, every evidence
    value and the verdict with its reasons (``docs/ENGINE_4D.md`` SS3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONSENSUS_COLUMNS)
        writer.writeheader()
        for d in decisions:
            row = d.to_row()
            writer.writerow({k: row.get(k) for k in CONSENSUS_COLUMNS})
    return path


__all__ = [
    "STATIC_BACKGROUND",
    "CHANNEL_WALL",
    "OUTSIDE_DEVICE",
    "ARTIFACT",
    "VALID_CELL_REGION",
    "UNKNOWN",
    "ATLAS_CLASSES",
    "ATLAS_CLASS_CODES",
    "ConsensusState",
    "ConsensusConfig",
    "Evidence",
    "Decision",
    "CONSENSUS_COLUMNS",
    "referee",
    "write_consensus_csv",
]
