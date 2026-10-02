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

Corridor 2.0: there is no migration axis, so 1.x's ``along_speed`` and
``net_along_um`` estimators are gone. Net displacement and the maximum
distance from the start (MTrackJ's D2S) stand in for them: both are
direction-free, and the net figure is the robust one the summary documents.
The tracker takes lanes measured from the walls (``geometry.detect_channels``)
instead of an axis. The model is the validated one (``resolve_model``) unless
``--model`` names a research file, which every output then records as an
override.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: ``CORRIDOR_SAMPLE_DIR`` points a checkout without ``data/`` at the samples.
SAMPLES = Path(
    os.environ.get("CORRIDOR_SAMPLE_DIR")
    or ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
)
PIXEL_UM = 0.467060342995564
INTERVAL_MIN = 20.006894938151042

#: Every estimator worth quoting, in the order they are printed.
ESTIMATORS = [
    ("mean_speed", "mean speed"),
    ("median_speed", "median spd"),
    ("path_speed", "path speed"),
    ("net_speed", "net speed"),
    ("net_displacement_um", "net displ"),
    ("max_d2s_um", "max D2S"),
]


def resolve(model_path: str | None):
    """The validated model, or an explicitly named research model."""
    from corridor.core.model_registry import research_model, resolve_model

    if model_path:
        return research_model(Path(model_path), label=Path(model_path).name)
    return resolve_model("2D")


def measure(detections, n_frames, scale, tracking, geometry=None):
    """Every reported quantity for the longest track in a detection set."""
    from corridor.core.measurements import summarise
    from corridor.core.tracking import track_detections

    tracks, _ = track_detections(detections, n_frames, scale, tracking, geometry=geometry)
    summaries = summarise(tracks, scale)
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
        "net_displacement_um": longest.net_displacement_um,
        "max_d2s_um": longest.max_distance_from_start_um,
        "path_length_um": longest.path_length_um,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stack", default="052924_t1.tif")
    ap.add_argument("--samples", default=str(SAMPLES), help="directory holding the sample TIFFs")
    ap.add_argument("--model", default=None,
                    help="research model file (default: the validated model)")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "docs" / "velocity_robustness.json"))
    args = ap.parse_args()

    import tifffile

    from corridor.core.config import GeometryConfig, Scale, SegmentationConfig, TrackingConfig
    from corridor.core.geometry import assign_lanes, detect_channels
    from corridor.core.segmentation import SegmentationService

    stack = tifffile.imread(Path(args.samples) / args.stack)
    service = SegmentationService(SegmentationConfig(), model=resolve(args.model))
    output = service.run_stack(stack)
    geometry = detect_channels(stack, GeometryConfig(), output.detections, pixel_size_um=PIXEL_UM)
    assign_lanes(output.detections, geometry)
    scale = Scale.from_values(PIXEL_UM, INTERVAL_MIN)
    tracking = TrackingConfig()
    n_frames = int(stack.shape[0])

    full = measure(output.detections, n_frames, scale, tracking, geometry)
    if full is None:
        print("no usable track in the complete data; cannot run this experiment")
        return 1
    print(f"{args.stack}: {len(output.detections)} detections, {geometry.n_lanes} lane(s), "
          f"lane gate {'applied' if geometry.applied else 'not applied'}")
    print(f"complete data -> {full['longest_observations']} observations")
    for key, label in ESTIMATORS:
        value = full[key]
        print(f"  {label:12s} {value:8.4f}" if value is not None else f"  {label:12s}        -")
    print()

    rng = np.random.default_rng(0)
    report = {
        "stack": args.stack,
        "model": service.resolved_model.to_manifest() if service.resolved_model else None,
        "complete": full,
        "loss": [],
    }

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
            got = measure(keep, n_frames, scale, tracking, geometry)
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
