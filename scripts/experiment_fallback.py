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
    track predicted a cell would be, inside the same channel, at a compatible
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
    that crosses between channels is wrong on the device geometry alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "confinedmig_cellTrack"
SAMPLES = DATA / "sample_data"
TRAIN = DATA / "CellPose_TrainData"
MODEL = TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"
COMPANIONS = (
    str(TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10"),
    str(TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10"),
)

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


def analyse(stack: np.ndarray, rung: str):
    """Run the real pipeline stages for one fallback rung."""
    from corridor.core.config import (
        ConfinementConfig, Scale, SegmentationConfig, TrackingConfig,
    )
    from corridor.core.confinement import assign_channels, resolve_axis
    from corridor.core.measurements import summarise
    from corridor.core.segmentation import SOURCE_ENSEMBLE, SegmentationService
    from corridor.core.tracking import track_detections

    cfg = SegmentationConfig(
        model_path=str(MODEL),
        use_custom_model=True,
        ensemble=rung,
        ensemble_model_paths=COMPANIONS,
    )
    service = SegmentationService(cfg)
    started = time.time()
    output = service.run_stack(stack)
    seconds = time.time() - started

    axis = resolve_axis(stack, ConfinementConfig(), output.detections,
                        pixel_size_um=PIXEL_UM)
    assign_channels(output.detections, axis)
    scale = Scale.from_values(PIXEL_UM, INTERVAL_MIN)
    tracks, _ = track_detections(
        output.detections, int(stack.shape[0]), axis, scale, TrackingConfig()
    )
    summaries = summarise(tracks, axis, scale)

    # A track of one observation is not a trajectory; it is a detection with a
    # number attached. Only multi-observation tracks reach a user's conclusions,
    # so they are what precision should be judged on.
    real = [t for t in tracks if len(t.observations) >= 2]
    tracked = {(o.frame, round(o.x, 3), round(o.y, 3)) for t in real for o in t.observations}
    every = {(d.frame, round(d.x, 3), round(d.y, 3)) for d in output.detections}

    return {
        "seconds": round(seconds, 1),
        "detections": len(output.detections),
        "ensemble_only": sum(1 for d in output.detections if d.source == SOURCE_ENSEMBLE),
        "tracks_all": len(tracks),
        "tracks_real": len(real),
        "tracked_observations": len(tracked),
        "rejected_by_tracker": len(every - tracked),
        "channels": len(axis.channels),
        "cross_channel_tracks": sum(
            1 for t in real if len({o.channel for o in t.observations if o.channel >= 0}) > 1
        ),
        "longest_observations": max((len(t.observations) for t in real), default=0),
        "speeds": [
            round(s.net_speed_um_per_min, 4)
            for s in summaries
            if s.net_speed_um_per_min is not None and s.n_observations >= 2
        ],
        "_detections": output.detections,
        "_tracked": tracked,
    }


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="off,thresholds,wide,models,max_recall")
    ap.add_argument("--stacks", default=",".join(STACKS))
    ap.add_argument("--out", default=str(ROOT / "docs" / "fallback_experiment.json"))
    args = ap.parse_args()

    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    stacks = [s.strip() for s in args.stacks.split(",") if s.strip()]

    report: dict = {"rungs": {}}
    for rung in rungs:
        print(f"\n=== {rung} ===", flush=True)
        print(f"{'stack':22s} {'det':>5} {'ens':>5} {'trk':>4} {'obs':>5} "
              f"{'rej':>5} {'xch':>4} {'sec':>7}  verdict")
        per_stack = {}
        for name in stacks:
            path = SAMPLES / name
            if not path.exists():
                continue
            result = analyse(tifffile.imread(path), rung)
            verdict = judge(name, result)
            result.update(verdict)
            per_stack[name] = result
            print(f"{name:22s} {result['detections']:5d} {result['ensemble_only']:5d} "
                  f"{result['tracks_real']:4d} {result['tracked_observations']:5d} "
                  f"{result['rejected_by_tracker']:5d} {result['cross_channel_tracks']:4d} "
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
