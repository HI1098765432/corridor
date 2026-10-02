"""How much does a missed detection actually change the reported velocity?

Instance F1 is the usual segmentation score, but it is not the quantity this
software exists to produce. A researcher acts on migration speed, and speed is
computed from positions over time. Those two accuracies are not the same
number, and conflating them would either overstate or understate how usable the
result is.

Specifically: a missed frame does not bias a velocity if the gap is handled
correctly, because the displacement is divided by the elapsed time rather than
by one frame. What a missed frame costs is *precision* -- fewer, longer
intervals mean a noisier estimate -- and, if enough are missed in a row, the
track breaks and the trajectory is truncated instead of wrong.

This measures that directly, and it measures every estimator the software
reports rather than one, because they do not degrade at the same rate.
Detections are deleted at random from a stack whose complete answer is known,
and each estimator is compared against its own complete-data value. The output
says which number a user can quote at what detection quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "confinedmig_cellTrack"
SAMPLES = DATA / "sample_data"
MODEL = (
    DATA / "CellPose_TrainData" / "KK1KK2_combiModel" / "models"
    / "cyto2_phase_microfluidic_KK1KK2_combi"
)
PIXEL_UM = 0.467060342995564
INTERVAL_MIN = 20.006894938151042

#: Every estimator worth quoting, in the order they are printed.
ESTIMATORS = [
    ("mean_speed", "mean speed"),
    ("median_speed", "median spd"),
    ("path_speed", "path speed"),
    ("net_speed", "net speed"),
    ("along_speed", "along speed"),
    ("net_along_um", "net along"),
]


def measure(detections, n_frames, axis, scale, tracking):
    """Every reported quantity for the longest track in a detection set."""
    from corridor.core.measurements import summarise
    from corridor.core.tracking import track_detections

    tracks, _ = track_detections(detections, n_frames, axis, scale, tracking)
    summaries = summarise(tracks, axis, scale)
    usable = [s for s in summaries if s.mean_speed_um_per_min is not None]
    if not usable:
        return None
    longest = max(usable, key=lambda s: s.n_observations)
    return {
        "n_tracks": len(summaries),
        "longest_observations": longest.n_observations,
        "longest_span": longest.span_frames,
        "mean_speed": longest.mean_speed_um_per_min,
        "median_speed": longest.median_speed_um_per_min,
        "path_speed": longest.path_speed_um_per_min,
        "net_speed": longest.net_speed_um_per_min,
        "along_speed": longest.along_speed_um_per_min,
        "net_along_um": longest.net_along_um,
        "path_length_um": longest.path_length_um,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="052924_t1.tif")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "docs" / "velocity_robustness.json"))
    args = ap.parse_args()

    from corridor.core.config import (
        ConfinementConfig, Scale, SegmentationConfig, TrackingConfig,
    )
    from corridor.core.confinement import assign_channels, resolve_axis
    from corridor.core.segmentation import SegmentationService

    stack = tifffile.imread(SAMPLES / args.stack)
    service = SegmentationService(
        SegmentationConfig(model_path=str(MODEL), use_custom_model=True)
    )
    output = service.run_stack(stack)
    axis = resolve_axis(stack, ConfinementConfig(), output.detections, pixel_size_um=PIXEL_UM)
    assign_channels(output.detections, axis)
    scale = Scale.from_values(PIXEL_UM, INTERVAL_MIN)
    tracking = TrackingConfig()
    n_frames = int(stack.shape[0])

    full = measure(output.detections, n_frames, axis, scale, tracking)
    if full is None:
        print("no usable track in the complete data; cannot run this experiment")
        return 1
    print(f"{args.stack}: {len(output.detections)} detections")
    print(f"complete data -> {full['longest_observations']} observations")
    for key, label in ESTIMATORS:
        print(f"  {label:12s} {full[key]:8.4f}")
    print()

    rng = np.random.default_rng(0)
    report = {"stack": args.stack, "complete": full, "loss": []}

    header = "  ".join(f"{label:>11}" for _, label in ESTIMATORS)
    print("error % against the complete-data answer, as median/p90\n")
    print(f"{'loss':>5} {'obs':>5}  {header}  {'broken':>6}")
    for rate in (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
        errors = {key: [] for key, _ in ESTIMATORS}
        obs_counts = []
        broken = 0
        trials = 1 if rate == 0.0 else args.trials
        for _ in range(trials):
            keep = [d for d in output.detections if rng.random() >= rate]
            got = measure(keep, n_frames, axis, scale, tracking)
            if got is None:
                broken += 1
                continue
            obs_counts.append(got["longest_observations"])
            for key, _ in ESTIMATORS:
                reference, value = full.get(key), got.get(key)
                if reference and value is not None:
                    errors[key].append(abs(value - reference) / abs(reference) * 100)
            # "Broken" means the trajectory no longer spans most of the original,
            # which is a different failure from being inaccurate: the answer is
            # not wrong, it is about a shorter piece of the experiment.
            if got["longest_span"] < full["longest_span"] * 0.6:
                broken += 1

        row = {
            "loss_rate": rate,
            "trials": trials,
            "median_observations": float(np.median(obs_counts)) if obs_counts else 0,
            "broken_fraction": round(broken / trials, 3),
        }
        for key, _ in ESTIMATORS:
            values = errors[key]
            row[f"{key}_median_err_pct"] = round(float(np.median(values)), 2) if values else None
            row[f"{key}_p90_err_pct"] = round(float(np.percentile(values, 90)), 2) if values else None
        report["loss"].append(row)

        def cell(key):
            median = row[f"{key}_median_err_pct"]
            if median is None:
                return f"{'-':>11}"
            return f"{median:5.1f}/{row[f'{key}_p90_err_pct']:5.1f}"

        cells = "  ".join(cell(key) for key, _ in ESTIMATORS)
        print(f"{rate:5.0%} {row['median_observations']:5.1f}  {cells}  {row['broken_fraction']:6.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
