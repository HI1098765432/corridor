"""Does the temporal reconstruction beat the human tracing? Two honest tests.

This is the claim everything downstream rests on. If a mask sequence solved
jointly across frames is *better* than the frame-by-frame tracing, then training
on it can carry the model past the annotator, and the 0.843 "ceiling" was a
property of the reference. If it is not better, the whole approach dies here and
that is a perfectly good outcome to record.

Two tests, neither of which lets a party mark its own work.

**Test 1 - leave-one-frame-out.** Hide a frame's human labels entirely.
Reconstruct that frame from its neighbours' labels plus its own image. Compare
the reconstruction with the hidden label. This measures whether temporal
evidence carries real information about a frame nobody looked at, against two
baselines: copying the previous frame's mask unchanged, and what the
segmentation model predicts there.

**Test 2 - boundary sharpness.** Score every mask by how well its boundary sits
on the image's own intensity gradient (``boundary_sharpness``). This is the
arbiter: it is a property of the photons, not of the annotator or the network,
and it is free to rank the reconstruction below the human. If the reconstruction
scores *higher* on frames where a human label also exists, the reconstruction is
placing boundaries better than the person did.

Test 1 alone would be circular-ish -- it rewards agreeing with the annotator, so
its ceiling is the annotator. Test 2 is what can exceed them. Both are reported
because a method that wins test 2 and loses test 1 badly is probably drifting
rather than improving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.reconstruct import (  # noqa: E402
    boundary_sharpness,
    link_masks,
    refine_to_image,
    temporal_consensus,
)
from corridor.learn.sequences import (  # noqa: E402
    find_sequences,
    load_image,
    load_masks,
)

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "reconstruction_experiment.json"))
    ap.add_argument("--with-model", action="store_true",
                    help="also score what the segmentation model predicts (slow)")
    args = ap.parse_args()

    model = None
    if args.with_model:
        import cv2
        cv2.setNumThreads(0)
        import torch
        torch.set_num_threads(1)
        from cellpose import models as cp
        combi = (TRAIN / "KK1KK2_combiModel" / "models"
                 / "cyto2_phase_microfluidic_KK1KK2_combi")
        model = cp.CellposeModel(pretrained_model=str(combi), gpu=False)

    held_out = {"reconstruction": [], "copy_previous": [], "model": []}
    sharpness = {"human": [], "reconstruction": [], "consensus_only": []}
    per_sequence = []

    for sequence in find_sequences(TRAIN):
        if not sequence.looks_like_a_movie:
            continue
        images = [load_image(p) for p in sequence.paths]
        masks = [load_masks(p) for p in sequence.paths]
        tracks = link_masks(masks)

        seq_recon, seq_copy, seq_sharp_h, seq_sharp_r, seq_sharp_c = [], [], [], [], []

        for track in tracks:
            frames = track.frames
            if len(frames) < 3:
                continue
            regions = {f: (masks[f] == track.members[f]) for f in frames}

            for target in frames[1:-1]:
                truth = regions[target]

                # -- test 1: rebuild this frame without its own label ---------
                without = {f: r for f, r in regions.items() if f != target}
                prior = temporal_consensus(without, target, span=1)
                if prior is None or not prior.any():
                    continue
                rebuilt = refine_to_image(images[target], prior)
                seq_recon.append(iou(rebuilt, truth))

                previous = max((f for f in frames if f < target), default=None)
                if previous is not None:
                    seq_copy.append(iou(regions[previous], truth))

                if model is not None:
                    predicted = np.asarray(
                        model.eval(images[target], channels=[0, 0], diameter=None,
                                   cellprob_threshold=0.0, flow_threshold=0.4)[0]
                    ).astype(np.int32)
                    best = 0.0
                    for label in np.unique(predicted):
                        if label:
                            best = max(best, iou(predicted == label, truth))
                    held_out["model"].append(best)

                # -- test 2: which boundary sits on the image's own edge? -----
                # Here the reconstruction may use every frame, this one too:
                # the question is no longer "can it guess a hidden label" but
                # "given all the evidence, who places the boundary better".
                full = temporal_consensus(regions, target, span=1)
                if full is not None and full.any():
                    refined = refine_to_image(images[target], full)
                    seq_sharp_h.append(boundary_sharpness(images[target], truth))
                    seq_sharp_r.append(boundary_sharpness(images[target], refined))
                    # The honest control on our own arbiter. refine_to_image
                    # places its boundary at an intensity threshold, and a
                    # threshold crossing on a smooth edge sits near the gradient
                    # peak -- so sharpness may simply be rewarding the thing
                    # that method optimises. temporal_consensus never looks at
                    # the image at all: it is neighbouring masks and centroid
                    # interpolation, nothing else. If ITS boundaries also beat
                    # the human on gradient, the result is not an artefact of
                    # the metric.
                    seq_sharp_c.append(boundary_sharpness(images[target], full))

        if not seq_recon:
            continue
        held_out["reconstruction"] += seq_recon
        held_out["copy_previous"] += seq_copy
        sharpness["human"] += seq_sharp_h
        sharpness["reconstruction"] += seq_sharp_r
        sharpness["consensus_only"] += seq_sharp_c

        per_sequence.append({
            "sequence": sequence.name,
            "frames": sequence.n_frames,
            "cells_followed": sum(1 for t in tracks if len(t.frames) >= 3),
            "held_out_iou_reconstruction": round(float(np.median(seq_recon)), 4),
            "held_out_iou_copy_previous": round(float(np.median(seq_copy)), 4) if seq_copy else None,
            "sharpness_human": round(float(np.median(seq_sharp_h)), 4) if seq_sharp_h else None,
            "sharpness_reconstruction": round(float(np.median(seq_sharp_r)), 4) if seq_sharp_r else None,
        })

    def summarise(values):
        if not values:
            return None
        return {
            "n": len(values),
            "median": round(float(np.median(values)), 4),
            "mean": round(float(np.mean(values)), 4),
        }

    report = {
        "test_1_held_out_frame_iou": {k: summarise(v) for k, v in held_out.items()},
        "test_2_boundary_sharpness": {k: summarise(v) for k, v in sharpness.items()},
        "per_sequence": per_sequence,
    }

    print("TEST 1 - rebuild a frame whose label was hidden, compare with that label")
    print(f"  {'method':24s} {'n':>5} {'median IoU':>12} {'mean':>8}")
    for key in ("reconstruction", "copy_previous", "model"):
        s = report["test_1_held_out_frame_iou"].get(key)
        if s:
            print(f"  {key:24s} {s['n']:5d} {s['median']:12.4f} {s['mean']:8.4f}")

    print()
    print("TEST 2 - whose boundary sits on the image's own gradient")
    print(f"  {'mask':24s} {'n':>5} {'median sharpness':>18} {'mean':>8}")
    for key in ("human", "consensus_only", "reconstruction"):
        s = report["test_2_boundary_sharpness"].get(key)
        if s:
            print(f"  {key:24s} {s['n']:5d} {s['median']:18.4f} {s['mean']:8.4f}")

    h = report["test_2_boundary_sharpness"].get("human")
    r = report["test_2_boundary_sharpness"].get("reconstruction")
    if h and r:
        wins = sum(1 for a, b in zip(sharpness["reconstruction"], sharpness["human"]) if a > b)
        share = 100.0 * wins / len(sharpness["human"])
        print()
        print(f"  the reconstruction places the boundary better in {wins}/{len(sharpness['human'])} "
              f"cases ({share:.0f}%)")
        verdict = ("the reconstruction is the better reference"
                   if r["median"] > h["median"] else
                   "the human tracing is still the better reference")
        print(f"  VERDICT: {verdict}")
        report["verdict"] = verdict
        report["reconstruction_wins_share"] = round(share / 100.0, 4)

    print()
    print(f"{'sequence':16s} {'cells':>6} {'recon IoU':>10} {'copy IoU':>9} "
          f"{'sharp H':>8} {'sharp R':>8}")
    for row in per_sequence:
        print(f"{row['sequence']:16s} {row['cells_followed']:6d} "
              f"{row['held_out_iou_reconstruction']:10.4f} "
              f"{str(row['held_out_iou_copy_previous']):>9} "
              f"{str(row['sharpness_human']):>8} {str(row['sharpness_reconstruction']):>8}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
