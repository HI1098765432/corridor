"""Does a higher-recall detector produce a better *trajectory*?

``scripts/experiment_recall.py`` scores single images, and by that score the
aggressive fallback rungs look bad: pushing recall from 0.846 to 0.902 drops
precision from 0.832 to 0.703, so F1 falls.  That would settle the question if
the product were a segmentation mask.  It is not -- it is a trajectory, and the
two are scored differently in a way that is easy to state and easy to check:

*   A **missed** cell is unrecoverable.  Nothing downstream can invent a
    position the detector never proposed, so every miss is a permanent hole.
*   A **false** detection gets a second examination it cannot pass by accident.
    To reach the output it must be linked, which means landing near where a
    track predicted a cell would be, inside the same lane, at a compatible
    area, for two frames running.  Debris and wall texture do not do that.

If that argument is right, the fallback rungs should convert to more trajectory
coverage without a matching rise in spurious tracked positions.  If it is
wrong -- if the extra detections link to each other and manufacture tracks --
then the conservative rung is correct and the ladder should stay off.

This measures it on stacks whose answer is known:

*   ``052924_t1``    one cell, present and eye-checked in frames 4-16.
*   ``052924_t3_dual`` two cells in adjacent channels, with a known hole where
    the probability field collapses.
*   ``052924_t2_empty`` frames 0, 3 and 4 contain no cell at all, so anything
    reported there is false by construction.
*   ``052924_1`` / ``052924_2`` wide fields; no per-cell truth, but any track
    that crosses between lanes is wrong on the device geometry alone.

**Corridor 2.0 keeps only the threshold rungs.**  ``off``, ``thresholds`` and
``wide`` re-run the one validated model at lower thresholds and are measured
here.  The companion-model rungs (``models`` and ``max_recall``, which ran the
KK1 and KK2 models beside the combined one) no longer exist -- only the
validated model may segment -- so they cannot be re-measured; asking for them
is refused rather than silently run as ``off``.  Their 1.x numbers stay in the
report this script wrote before.  There is no migration axis: lanes come from
``geometry.detect_channels``, and the tracker is the axis-free one.  The model
is the validated one (``resolve_model``) unless ``--model`` names a research
file, which every output then records as an override.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: ``CORRIDOR_SAMPLE_DIR`` points a checkout without ``data/`` at the samples.
SAMPLES = Path(
    os.environ.get("CORRIDOR_SAMPLE_DIR")
    or ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
)
#: The rungs that ran other models. Removed in 2.0; see the module docstring.
REMOVED_RUNGS = ("models", "max_recall")
#: The rungs 2.0 can measure: the validated model at its own and lower thresholds.
THRESHOLD_RUNGS = ("off", "thresholds", "wide")

PIXEL_UM = 0.467060342995564
INTERVAL_MIN = 20.006894938151042

#: Frames in which a cell is known to be present, verified by eye.
KNOWN_PRESENT = {"052924_t1.tif": set(range(4, 17))}
#: Frames in which no cell exists, so any reported position is false.
KNOWN_ABSENT = {"052924_t2_empty.tif": {0, 3, 4}}

STACKS = [
    "052924_t1.tif",
    "052924_t2_empty.tif",
    "052924_t3_dual.tif",
    "052924_1.tif",
    "052924_2.tif",
]


def resolve(model_path: str | None):
    """The validated model, or an explicitly named research model."""
    from corridor.core.model_registry import research_model, resolve_model

    if model_path:
        return research_model(Path(model_path), label=Path(model_path).name)
    return resolve_model("2D")


def parse_rungs(text: str) -> list[str]:
    """The requested rungs, refusing the removed ones and anything unknown."""
    rungs = [r.strip() for r in text.split(",") if r.strip()]
    removed = [r for r in rungs if r in REMOVED_RUNGS]
    if removed:
        # The service would run them as 'off', and the report would show
        # identical rungs that look like a measured result.
        raise ValueError(
            f"rung(s) {', '.join(removed)} were removed in Corridor 2.0 (they ran other "
            f"models); only {', '.join(THRESHOLD_RUNGS)} can be measured."
        )
    unknown = [r for r in rungs if r not in THRESHOLD_RUNGS]
    if unknown:
        raise ValueError(f"unknown rung(s): {', '.join(unknown)}")
    return rungs


def link(detections, n_frames: int, geometry):
    """Track and summarise one detection set, the way the pipeline does."""
    from corridor.core.config import Scale, TrackingConfig
    from corridor.core.measurements import summarise
    from corridor.core.tracking import track_detections

    scale = Scale.from_values(PIXEL_UM, INTERVAL_MIN)
    tracks, _ = track_detections(detections, n_frames, scale, TrackingConfig(), geometry=geometry)
    return tracks, summarise(tracks, scale)


def describe(detections, tracks, summaries, geometry) -> dict:
    """What a rung produced, before any ground truth is applied."""
    from corridor.core.detections import SOURCE_ENSEMBLE

    # A track of one observation is not a trajectory; it is a detection with a
    # number attached. Only multi-observation tracks reach a user's conclusions,
    # so they are what precision should be judged on.
    real = [t for t in tracks if len(t.observations) >= 2]
    tracked = {(o.frame, round(o.x, 3), round(o.y, 3)) for t in real for o in t.observations}
    every = {(d.frame, round(d.x, 3), round(d.y, 3)) for d in detections}
    return {
        "detections": len(detections),
        "ensemble_only": sum(1 for d in detections if d.source == SOURCE_ENSEMBLE),
        "tracks_all": len(tracks),
        "tracks_real": len(real),
        "tracked_observations": len(tracked),
        "rejected_by_tracker": len(every - tracked),
        "lanes": geometry.n_lanes if geometry is not None else 0,
        "lane_gate_applied": bool(geometry is not None and geometry.applied),
        "cross_lane_tracks": sum(
            1 for t in real if len({o.channel for o in t.observations if o.channel >= 0}) > 1
        ),
        "longest_observations": max((len(t.observations) for t in real), default=0),
        "speeds": [
            round(s.net_speed_um_per_min, 4)
            for s in summaries
            if s.net_speed_um_per_min is not None and s.n_observations >= 2
        ],
        "_detections": detections,
        "_tracked": tracked,
    }


def analyse(stack: np.ndarray, rung: str, model) -> dict:
    """Run the real pipeline stages for one fallback rung."""
    from corridor.core.config import GeometryConfig, SegmentationConfig
    from corridor.core.geometry import assign_lanes, detect_channels
    from corridor.core.segmentation import SegmentationService

    service = SegmentationService(SegmentationConfig(ensemble=rung), model=model)
    started = time.time()
    output = service.run_stack(stack)
    seconds = time.time() - started

    geometry = detect_channels(stack, GeometryConfig(), output.detections,
                               pixel_size_um=PIXEL_UM)
    assign_lanes(output.detections, geometry)
    tracks, summaries = link(output.detections, int(stack.shape[0]), geometry)
    result = describe(output.detections, tracks, summaries, geometry)
    result["seconds"] = round(seconds, 1)
    return result


def judge(name: str, result: dict) -> dict:
    """Apply whatever ground truth exists for this stack."""
    verdict: dict = {}
    detections = result.pop("_detections")
    tracked = result.pop("_tracked")

    if name in KNOWN_PRESENT:
        wanted = KNOWN_PRESENT[name]
        covered = {f for f, _, _ in tracked} & wanted
        verdict["known_present_frames"] = len(wanted)
        verdict["covered"] = len(covered)
        verdict["missed"] = sorted(wanted - covered)

    if name in KNOWN_ABSENT:
        wrong = KNOWN_ABSENT[name]
        verdict["known_absent_frames"] = len(wrong)
        # Two numbers, deliberately: what the detector proposed in those frames,
        # and what survived into a trajectory. The gap between them is the
        # tracker's contribution to precision, and it is the whole question.
        verdict["false_detections"] = sum(1 for d in detections if d.frame in wrong)
        verdict["false_tracked"] = sum(1 for f, _, _ in tracked if f in wrong)

    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rungs", default=",".join(THRESHOLD_RUNGS),
                    help=f"comma-separated, from {', '.join(THRESHOLD_RUNGS)}")
    ap.add_argument("--stacks", default=",".join(STACKS))
    ap.add_argument("--samples", default=str(SAMPLES), help="directory holding the sample TIFFs")
    ap.add_argument("--model", default=None,
                    help="research model file (default: the validated model)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "fallback_experiment.json"))
    args = ap.parse_args()

    try:
        rungs = parse_rungs(args.rungs)
    except ValueError as exc:
        ap.error(str(exc))
    stacks = [s.strip() for s in args.stacks.split(",") if s.strip()]

    import tifffile

    model = resolve(args.model)
    report: dict = {"model": model.to_manifest(), "rungs": {}}
    for rung in rungs:
        print(f"\n=== {rung} ===", flush=True)
        print(f"{'stack':22s} {'det':>5} {'ens':>5} {'trk':>4} {'obs':>5} "
              f"{'rej':>5} {'xln':>4} {'sec':>7}  verdict")
        per_stack = {}
        for name in stacks:
            path = Path(args.samples) / name
            if not path.exists():
                continue
            result = analyse(tifffile.imread(path), rung, model)
            verdict = judge(name, result)
            result.update(verdict)
            per_stack[name] = result
            print(f"{name:22s} {result['detections']:5d} {result['ensemble_only']:5d} "
                  f"{result['tracks_real']:4d} {result['tracked_observations']:5d} "
                  f"{result['rejected_by_tracker']:5d} {result['cross_lane_tracks']:4d} "
                  f"{result['seconds']:7.1f}  "
                  + (", ".join(f"{k}={v}" for k, v in verdict.items()) or "-"),
                  flush=True)
        report["rungs"][rung] = per_stack

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
