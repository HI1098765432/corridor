"""The 4D engine orchestrator (``docs/ENGINE_4D.md`` SS3).

::

    load T[Z]YX -> calibrate -> register -> atlas -> propose (on V' or V'-B)
    -> reject static/impossible -> Z consensus -> temporal consensus -> referee
    -> identity (tracker) -> 4D tubes -> measurements +/- uncertainty -> export

This is the **skeleton**: it wires the stages, the frozen config and the
evidence files, and it runs end to end on a given-masks proposer today.  The
bots it does not yet have -- registration (Bot 1), static atlas (Bot 2), Z
consensus (Bot 4), temporal delta (Bot 5), measurement (Bot 6) -- are imported
**lazily**, inside the stage that needs them, so this module builds and runs
before they land.  When a bot is absent the stage degrades to an honest, named
fallback (identity registration, an all-``VALID_CELL_REGION`` atlas, each 2-D
component its own object, no temporal signal) and records itself as skipped,
rather than fabricating the number the bot would have measured.  The referee
(Bot 8) and the proposer interface (Bot 3) are real and always run; the
identity hand-off to ``corridor.core.tracking`` is real too, because that
tracker already exists.

The files written under the output directory are exactly those the contract
names: ``registration.csv``, ``atlas.npz`` and ``consensus.csv``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .consensus import (
    ATLAS_CLASS_CODES,
    ATLAS_CLASSES,
    VALID_CELL_REGION,
    ConsensusConfig,
    ConsensusState,
    Decision,
    Evidence,
    referee,
    write_consensus_csv,
)
from .proposer import Proposer


# --------------------------------------------------------------------------
# Configuration (frozen -- one run's settings cannot drift mid-run)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Engine4DConfig:
    """Every knob the 4D engine reads, frozen for the run.

    Defaults describe the honest skeleton behaviour: register if the bot is
    present, keep the referee's own default thresholds, and a motion ceiling on
    the proposer's own calibration.  ``max_speed_px_per_frame`` is the fresh
    cell's prior scale noted in ``core.tracking`` (~71 px/frame at the KK2
    calibration): generous on purpose here, because the referee's job is to
    reject the *physically impossible*, not to re-impose the tracker's gate.
    """

    register: bool = True
    estimate_rotation: bool = False

    detect_walls: bool = True
    reject_in_walls: bool = True

    z_window: int = 1  # +/- slices linked by Z consensus
    temporal_window: int = 2  # +/- frames for E_FB (contract uses +/-1 and +/-2)

    max_speed_px_per_frame: float = 71.0
    subtract_background: bool = False  # propose on V'-B (Experiment B') when True

    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)

    # Output file names (ENGINE_4D SS3). Kept configurable, defaulted to the
    # contract's names.
    registration_csv: str = "registration.csv"
    atlas_npz: str = "atlas.npz"
    consensus_csv: str = "consensus.csv"


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class Engine4DResult:
    """What one run produced: shapes, counts, the decisions and the files.

    ``stages_run`` and ``stages_skipped`` make the skeleton honest about which
    bots actually contributed -- a reader never has to guess whether a figure
    came from Bot 5 or from the fallback that stood in for it.
    """

    shape_tzyx: tuple[int, int, int, int]
    n_proposed: int
    n_rejected_static: int
    decisions: list[Decision]
    registration_path: Path
    atlas_path: Path
    consensus_path: Path
    n_tracks: int | None
    stages_run: list[str]
    stages_skipped: list[str]

    def state_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ConsensusState}
        for d in self.decisions:
            counts[d.state.value] += 1
        return counts

    def summary(self) -> dict:
        return {
            "shape_tzyx": list(self.shape_tzyx),
            "n_proposed": self.n_proposed,
            "n_rejected_static": self.n_rejected_static,
            "n_tracks": self.n_tracks,
            "state_counts": self.state_counts(),
            "stages_run": list(self.stages_run),
            "stages_skipped": list(self.stages_skipped),
            "files": {
                "registration": str(self.registration_path),
                "atlas": str(self.atlas_path),
                "consensus": str(self.consensus_path),
            },
        }


# --------------------------------------------------------------------------
# Canonicalisation -- one array I[t, z, y, x], Z=1 handled without a special
# code path (ENGINE_4D SS1).
# --------------------------------------------------------------------------


def _to_tzyx(image: np.ndarray, axes: str | None) -> tuple[np.ndarray, str]:
    """Return ``(I[t, z, y, x], resolved_axes)``.

    A 2-D movie has Z = 1 and every stage handles that without a branch.  T and
    Z are taken from ``axes`` when given; otherwise 2-D is YX, 3-D is TYX (the
    supplied data is a movie, never a single Z stack by default) and 4-D is the
    canonical TZYX.  A 3-D stack that is really ZYX must say so via ``axes`` --
    a file that says Z is never read as time (``docs/NEXT_GENERATION.md`` SS4).
    """
    arr = np.asarray(image)
    if axes:
        axes = axes.upper()
        if len(axes) != arr.ndim:
            raise ValueError(f"axes {axes!r} does not match array ndim {arr.ndim}")
        resolved = axes
    elif arr.ndim == 2:
        resolved = "YX"
    elif arr.ndim == 3:
        resolved = "TYX"
    elif arr.ndim == 4:
        resolved = "TZYX"
    else:
        raise ValueError(f"cannot interpret a {arr.ndim}-D array as T[Z]YX")

    # Insert singleton T then Z so the result is always (T, Z, Y, X).
    if resolved == "YX":
        out = arr[None, None]
    elif resolved == "TYX":
        out = arr[:, None]
    elif resolved == "ZYX":
        out = arr[None]
    elif resolved == "TZYX":
        out = arr
    else:
        raise ValueError(f"unsupported axes {resolved!r}; use YX/TYX/ZYX/TZYX")
    return np.ascontiguousarray(out), resolved


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def run_engine_4d(
    image: np.ndarray,
    proposer: Proposer,
    output_dir: str | Path,
    *,
    scale=None,
    axes: str | None = None,
    intensity: np.ndarray | None = None,
    config: Engine4DConfig | None = None,
) -> Engine4DResult:
    """Run the skeleton end to end and write the evidence files.

    ``scale`` is a ``corridor.core.config.Scale`` (optional; the tracking
    hand-off builds a pixel/frame one when it is absent).  ``intensity`` is the
    raw image to measure from when it differs from ``image`` (e.g. a normalised
    input); it defaults to ``image``.
    """
    cfg = config or Engine4DConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stages_run: list[str] = []
    stages_skipped: list[str] = []

    vol, _resolved_axes = _to_tzyx(image, axes)  # (T, Z, Y, X)
    raw = vol if intensity is None else _to_tzyx(intensity, axes)[0]
    T, Z, Y, X = vol.shape

    # -- Stage: register (Bot 1) --------------------------------------------
    reference_t = T // 2
    shifts = _register(vol, cfg, reference_t, stages_run, stages_skipped)
    registration_path = _write_registration_csv(out / cfg.registration_csv, shifts, reference_t)

    # -- Stage: static atlas (Bot 2) ----------------------------------------
    background, class_map = _atlas(vol, cfg, stages_run, stages_skipped)  # (Z,Y,X), (Z,Y,X) int8
    atlas_path = _write_atlas_npz(out / cfg.atlas_npz, background, class_map)

    # -- Stage: propose (Bot 3) ---------------------------------------------
    # One proposal per (t, z). Z=1 is the common case and needs no branch.
    proposals: dict[tuple[int, int], object] = {}
    n_proposed = 0
    for t in range(T):
        for z in range(Z):
            frame = raw[t, z]
            prop = proposer.propose(frame, t=t, z=z)
            proposals[(t, z)] = prop
            n_proposed += prop.n_objects()
    stages_run.append("propose")

    # -- Stage: reject static / Z consensus / temporal consensus ------------
    # The real Bots 4 and 5 land later; here each 2-D object stands alone and
    # carries no temporal signal, which the referee reads as "no information".
    _z_consensus(cfg, stages_run, stages_skipped)
    _temporal_consensus(cfg, stages_run, stages_skipped)

    # -- Stage: referee (Bot 8) ---------------------------------------------
    decisions: list[Decision] = []
    n_rejected_static = 0
    backend = getattr(proposer, "backend", "")
    for (t, z), prop in sorted(proposals.items()):
        cls_plane = class_map[z]
        for label in prop.labels():
            ev = _evidence_for_object(
                prop, label, t, z, cls_plane, backend, cfg
            )
            decision = referee(ev, cfg.consensus)
            decisions.append(decision)
            if (
                cfg.reject_in_walls
                and decision.state is ConsensusState.REJECTED
                and ev.atlas_wall_fraction >= cfg.consensus.wall_reject_fraction
            ):
                n_rejected_static += 1
    stages_run.append("referee")
    consensus_path = write_consensus_csv(decisions, out / cfg.consensus_csv)

    # -- Stage: identity (hand off to core.tracking) ------------------------
    n_tracks = _track(proposals, raw, scale, Z, decisions, stages_run, stages_skipped)

    # -- Stage: measurement (Bot 6) -----------------------------------------
    _measure(stages_run, stages_skipped)

    return Engine4DResult(
        shape_tzyx=(T, Z, Y, X),
        n_proposed=n_proposed,
        n_rejected_static=n_rejected_static,
        decisions=decisions,
        registration_path=registration_path,
        atlas_path=atlas_path,
        consensus_path=consensus_path,
        n_tracks=n_tracks,
        stages_run=stages_run,
        stages_skipped=stages_skipped,
    )


# --------------------------------------------------------------------------
# Stage helpers -- each lazy-imports its bot and degrades honestly without it.
# --------------------------------------------------------------------------


def _register(vol, cfg, reference_t, run, skipped):
    """Per-timepoint ``(dx, dy, dz, error)`` against the reference frame."""
    if not cfg.register:
        skipped.append("register (disabled)")
        return [(0.0, 0.0, 0.0, 0.0) for _ in range(vol.shape[0])]
    try:
        from . import registration4d  # Bot 1
    except ImportError:
        skipped.append("register (Bot 1 not present -> identity)")
        return [(0.0, 0.0, 0.0, 0.0) for _ in range(vol.shape[0])]
    run.append("register")
    # register_stack takes the (T, Z, Y, X) stack directly and returns one row
    # per timepoint with the volume's translation and a correlation error.
    reg_cfg = registration4d.RegistrationConfig(estimate_rotation=cfg.estimate_rotation)
    result = registration4d.register_stack(vol, reg_cfg)
    return [(r.dx_px, r.dy_px, r.dz_px, r.error) for r in result.rows]


def _atlas(vol, cfg, run, skipped):
    """``(background[Z,Y,X], class_map[Z,Y,X] int8)``.

    Without Bot 2, the background is the temporal median (the device without
    the cells, the one piece of the atlas that is a plain measurement) and the
    class map is all ``VALID_CELL_REGION`` -- the honest default: nothing has
    been classified as wall or artifact yet, so nothing is.
    """
    T, Z, Y, X = vol.shape
    background = np.median(vol.astype(np.float32), axis=0)  # (Z, Y, X)
    try:
        from . import static_atlas  # Bot 2, lands later
    except ImportError:
        skipped.append("atlas (Bot 2 not present -> median background, all VALID_CELL_REGION)")
        class_map = np.full((Z, Y, X), ATLAS_CLASS_CODES[VALID_CELL_REGION], dtype=np.int8)
        return background, class_map
    run.append("atlas")
    # build_atlas(stack, geometry=None, occupancy=None, config): pass an atlas
    # config, not the pipeline config (the 2nd positional arg is geometry).
    # Its integer class codes match consensus.ATLAS_CLASS_CODES (0=background,
    # 1=wall, ..., 4=valid), so the class map is consumed directly.
    result = static_atlas.build_atlas(vol, config=static_atlas.AtlasConfig())
    return result.background, result.class_map


def _z_consensus(cfg, run, skipped):
    try:
        from . import z_consensus  # Bot 4, lands later  # noqa: F401
    except ImportError:
        skipped.append("z_consensus (Bot 4 not present -> each 2-D object stands alone)")
        return
    run.append("z_consensus")  # pragma: no cover - post-integration


def _temporal_consensus(cfg, run, skipped):
    try:
        from . import temporal_delta  # Bot 5, lands later  # noqa: F401
    except ImportError:
        skipped.append("temporal_consensus (Bot 5 not present -> no E_FB signal)")
        return
    run.append("temporal_consensus")  # pragma: no cover - post-integration


def _measure(run, skipped):
    try:
        from . import object4d  # Bot 6, lands later  # noqa: F401
    except ImportError:
        skipped.append("measure (Bot 6 not present)")
        return
    run.append("measure")  # pragma: no cover - post-integration


def _track(proposals, raw, scale, Z, decisions, run, skipped):
    """Hand the kept objects to the existing axis-free tracker (Bot 7).

    Real, because ``core.tracking`` exists.  It runs only on a 2-D movie
    (Z == 1): 3-D detection extraction differs (``extract_detections_3d``) and
    is wired at integration, so a Z stack records the hand-off as skipped
    rather than running the wrong extractor.  Any failure degrades to
    ``n_tracks = None`` with a named note -- the skeleton must not die on the
    hand-off.
    """
    if Z != 1:
        skipped.append("identity (Z>1: 3-D extraction wired at integration)")
        return None
    try:
        from corridor.core.config import Scale, TrackingConfig
        from corridor.core.detections import extract_detections
        from corridor.core.tracking import track_detections
    except Exception as exc:  # pragma: no cover - import guard
        skipped.append(f"identity (tracking import failed: {exc})")
        return None

    kept = {
        (d.t, d.object_id)
        for d in decisions
        if d.state in (ConsensusState.CONFIRMED, ConsensusState.LIKELY, ConsensusState.AMBIGUOUS)
    }
    try:
        detections = []
        n_frames = 0
        for (t, z), prop in sorted(proposals.items()):
            n_frames = max(n_frames, t + 1)
            kept_mask = _mask_of_kept(prop.masks, {lbl for (tt, lbl) in kept if tt == t})
            dets = extract_detections(kept_mask, frame=t, intensity=raw[t, z])
            detections.extend(dets)
        scale = scale or Scale.from_values(None, None)
        tracks, _events = track_detections(detections, n_frames, scale, TrackingConfig())
        run.append("identity")
        return len(tracks)
    except Exception as exc:  # pragma: no cover - defensive
        skipped.append(f"identity (tracking failed: {exc})")
        return None


def _mask_of_kept(masks: np.ndarray, keep_labels: set[int]) -> np.ndarray:
    """A label image holding only the objects the referee kept at this frame."""
    out = np.zeros_like(masks)
    for lbl in keep_labels:
        out[masks == lbl] = lbl
    return out


# --------------------------------------------------------------------------
# Evidence assembly
# --------------------------------------------------------------------------


def _evidence_for_object(prop, label, t, z, class_plane, backend, cfg) -> Evidence:
    """Assemble the evidence the present bots can supply for one object.

    Fields a missing bot would fill are left at ``Evidence``'s neutral defaults
    (no temporal support, neutral atlas unless the class map says otherwise):
    the referee then simply has fewer pillars.  The two signals real today are
    the proposer's cell-probability and the object's atlas class / wall
    fraction, both measured over the object's own footprint.
    """
    sel = prop.masks == int(label)
    area_px = float(sel.sum())
    dominant_class, confidence, wall_fraction = _atlas_class_of(class_plane, sel)
    return Evidence(
        object_id=int(label),
        t=t,
        z=z,
        present_now=True,
        cellprob=prop.mean_cellprob(label),
        proposer_backend=backend,
        atlas_class=dominant_class,
        atlas_wall_fraction=wall_fraction,
        atlas_class_confidence=confidence,
        max_speed_px_per_frame=cfg.max_speed_px_per_frame,
        area_px=area_px,
    )


def _atlas_class_of(class_plane: np.ndarray, sel: np.ndarray):
    """``(dominant_class_name, confidence, channel_wall_fraction)`` for one
    object, where confidence is the fraction of the object's pixels in its
    dominant class and the wall fraction is the fraction in ``CHANNEL_WALL``."""
    from .consensus import CHANNEL_WALL

    codes = class_plane[sel]
    if codes.size == 0:
        return VALID_CELL_REGION, 0.0, 0.0
    counts = np.bincount(codes.astype(np.int64), minlength=len(ATLAS_CLASSES))
    dominant_code = int(counts.argmax())
    confidence = float(counts[dominant_code] / codes.size)
    dominant = ATLAS_CLASSES[dominant_code]
    wall_fraction = float(counts[ATLAS_CLASS_CODES[CHANNEL_WALL]] / codes.size)
    return dominant, confidence, wall_fraction


# --------------------------------------------------------------------------
# File writers
# --------------------------------------------------------------------------


def _write_registration_csv(path: Path, shifts, reference_t: int) -> Path:
    """``registration.csv``: t, dx, dy, dz, error, reference_t (ENGINE_4D SS3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "dx", "dy", "dz", "error", "reference_t"])
        for t, (dx, dy, dz, error) in enumerate(shifts):
            writer.writerow([t, dx, dy, dz, error, reference_t])
    return path


def _write_atlas_npz(path: Path, background: np.ndarray, class_map: np.ndarray) -> Path:
    """``atlas.npz``: the background and the class map, with the class legend
    so the int8 codes are readable without this module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        background=background,
        class_map=class_map,
        class_names=np.array(list(ATLAS_CLASSES)),
        class_codes=np.array([ATLAS_CLASS_CODES[n] for n in ATLAS_CLASSES]),
    )
    return path


__all__ = [
    "Engine4DConfig",
    "Engine4DResult",
    "run_engine_4d",
]
