"""Does recovery find real cells, or does it manufacture positions?

Counting recoveries proves nothing: a detector that fires everywhere recovers
everything. What is needed is ground truth for frames where a cell is known to
be present and known to be absent, and there is a way to get both from the
supplied data without any new labelling.

**Known present.** In 052924_t1 the model finds the cell in all of frames 4-16,
and that was confirmed by eye against the overlay. Deleting a primary detection
from one of those frames creates a hole whose correct answer is known exactly:
the detection that was removed. Recovery is then scored on whether it fills the
hole, and how far from the true centroid it lands.

**Known absent.** In 052924_t2_empty the model finds nothing in frames 0, 3 and
4, and the cell-probability field there is flat. Any recovery in those frames is
a false positive.

Both numbers matter. Recall alone would reward a detector that answers "yes"
everywhere, which is precisely the failure this experiment exists to catch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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

#: Frames where a cell is known present (found by the model and eye-checked).
PRESENT = {"052924_t1.tif": list(range(4, 17))}
#: Frames where a cell is known absent (no masks, and a flat probability field).
ABSENT = {"052924_t2_empty.tif": [0, 3, 4]}


def setup(name: str):
    from corridor.core.config import ConfinementConfig, Scale, SegmentationConfig
    from corridor.core.confinement import assign_channels, resolve_axis, static_projection
    from corridor.core.segmentation import SegmentationService

    stack = tifffile.imread(SAMPLES / name)
    service = SegmentationService(
        SegmentationConfig(model_path=str(MODEL), use_custom_model=True)
    )
    output = service.run_stack(stack)
    axis = resolve_axis(stack, ConfinementConfig(), output.detections,
                        pixel_size_um=PIXEL_UM)
    assign_channels(output.detections, axis)
    scale = Scale.from_values(PIXEL_UM, INTERVAL_MIN)
    return stack, service, output, axis, scale, static_projection(stack)


def run_case(name: str, hidden_frame: int | None, cfg, tracking):
    """Track with one frame's detections removed, then try to recover them."""
    from corridor.core.recovery import recover
    from corridor.core.tracking import track_detections

    stack, service, output, axis, scale, background = CACHE[name]
    detections = [d for d in output.detections if d.frame != hidden_frame]
    truth = [d for d in output.detections if d.frame == hidden_frame]

    tracks, _ = track_detections(detections, stack.shape[0], axis, scale, tracking)
    if not tracks:
        return None, truth
    result = recover(stack, tracks, service, axis, scale, tracking, cfg,
                     background=background)
    return result, truth


def evaluate(cfg, tracking, label: str) -> dict:
    hits = misses = 0
    offsets: list[float] = []
    tiers: dict[str, int] = {}
    false_positives = 0
    fp_tiers: dict[str, int] = {}

    # --- frames where a cell is known to be there -------------------------
    for name, frames in PRESENT.items():
        for frame in frames:
            result, truth = run_case(name, frame, cfg, tracking)
            if result is None or not truth:
                continue
            recovered = [d for d in result.detections if d.frame == frame]
            if not recovered:
                misses += 1
                continue
            best = min(
                recovered,
                key=lambda d: min(float(np.hypot(d.x - t.x, d.y - t.y)) for t in truth),
            )
            offset = min(float(np.hypot(best.x - t.x, best.y - t.y)) for t in truth)
            # Within one cell width of the true centroid counts as found.
            if offset <= max(truth[0].minor_axis_px * 1.5, 12.0):
                hits += 1
                offsets.append(offset)
                tiers[best.source] = tiers.get(best.source, 0) + 1
            else:
                misses += 1

    # --- frames where a cell is known NOT to be there ---------------------
    for name, frames in ABSENT.items():
        result, _ = run_case(name, None, cfg, tracking)
        if result is None:
            continue
        for d in result.detections:
            if d.frame in frames:
                false_positives += 1
                fp_tiers[d.source] = fp_tiers.get(d.source, 0) + 1

    attempted = hits + misses
    recall = hits / attempted if attempted else 0.0
    return {
        "strategy": label,
        "holes": attempted,
        "recovered": hits,
        "missed": misses,
        "recovery_rate": round(recall, 3),
        "median_offset_px": round(float(np.median(offsets)), 2) if offsets else None,
        "max_offset_px": round(float(np.max(offsets)), 2) if offsets else None,
        "by_tier": tiers,
        "false_positives": false_positives,
        "false_positive_tiers": fp_tiers,
    }


CACHE: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "recovery_experiment.json"))
    args = ap.parse_args()

    from corridor.core.config import TrackingConfig
    from corridor.core.recovery import RecoveryConfig

    for name in list(PRESENT) + list(ABSENT):
        print(f"preparing {name} ...", flush=True)
        CACHE[name] = setup(name)

    tracking = TrackingConfig()

    # Every variant states enabled=True explicitly. RecoveryConfig defaults to
    # disabled, so a variant that merely omits it measures nothing at all and
    # reports a confident row of zeros -- which is exactly what this script did
    # once the default was flipped, silently turning the experiment into a test
    # of whether "off" finds cells. An instrument that can quietly measure its
    # own silence has to be made to say what it is measuring.
    def variant(**kwargs) -> RecoveryConfig:
        kwargs.setdefault("enabled", True)
        return RecoveryConfig(**kwargs)

    # The interior/trailing split is the interesting axis. An interior gap is a
    # cell known present on both sides; a trailing frame is a cell that may
    # simply have gone. Measuring them together hides which one the errors come
    # from, which is the whole question.
    variants = [
        ("interior: window only", variant(permissive=False, intensity=False)),
        ("interior: + permissive", variant(intensity=False)),
        ("interior: intensity only", variant(window=False, permissive=False)),
        ("interior: all three tiers", variant()),
        ("interior + trailing: all tiers", variant(trailing=True)),
        ("trailing only: all tiers", variant(interior=False, trailing=True)),
        ("interior, tighter offset", variant(max_offset_lengths=0.6)),
        ("interior, intensity 8 sigma", variant(intensity_snr=8.0)),
        ("disabled (the shipped default)", RecoveryConfig(enabled=False)),
    ]
    for label, cfg in variants:
        if cfg.enabled and not (cfg.window or cfg.permissive or cfg.intensity):
            raise SystemExit(f"variant {label!r} has every tier switched off")

    print(f"\n{'strategy':30s} {'holes':>6} {'found':>6} {'rate':>6} "
          f"{'medOff':>7} {'maxOff':>7} {'FP':>4}  tiers")
    report = []
    for label, cfg in variants:
        row = evaluate(cfg, tracking, label)
        report.append(row)
        print(f"{label:30s} {row['holes']:6d} {row['recovered']:6d} "
              f"{row['recovery_rate']:6.3f} "
              f"{str(row['median_offset_px']):>7} {str(row['max_offset_px']):>7} "
              f"{row['false_positives']:4d}  {row['by_tier']}"
              + (f"  FP:{row['false_positive_tiers']}" if row['false_positives'] else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
