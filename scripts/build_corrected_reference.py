"""A reference with the missing cells put back, so a score can exceed 0.94.

Measured model-free (`scripts/measure_label_completeness.py`): at least 10.9 % of
the cells in KK2's verified movie carry no label, and 6.1 % in KK1. Because a
correct detection of an unlabelled cell scores as a false positive, **a perfect
detector cannot exceed F1 0.942 on KK2 against the labels as supplied.** No
amount of training reaches 99 % against a reference that is 11 % incomplete. The
reference has to be repaired first, and the repair has to be defensible.

**Where each added cell comes from.** A cell the annotator drew at *t-1* and
again at *t+1*, within one plausible step, but not at *t*. The evidence is
entirely the annotator's own work on adjacent frames: the model is not consulted
about whether the cell exists, and is not consulted about where it is. The
position is the interpolation of the two human centroids; the outline is the
human mask from the nearer frame, translated onto it.

**What is deliberately not done.** Cells missing from a *run* of consecutive
frames cannot be recovered this way, because there is no far side to interpolate
from -- so the corrected reference is still incomplete, and its remaining
incompleteness still caps any score. And nothing is added on the strength of
looking like a cell, however convincing, because that is the judgement under
test.

Scores against this reference must always be reported **beside** the score
against the original labels, never instead of it. The original is what every
published figure used; this one is what the images actually contain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.sequences import find_sequences, load_masks  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
SAME_CELL_PX = 45.0


def regions_of(masks: np.ndarray):
    for label in np.unique(masks):
        if label:
            region = masks == label
            ys, xs = np.nonzero(region)
            yield region, (float(xs.mean()), float(ys.mean()))


def shift_to(region: np.ndarray, centre) -> np.ndarray:
    ys, xs = np.nonzero(region)
    dy = int(round(centre[1] - ys.mean()))
    dx = int(round(centre[0] - xs.mean()))
    out = np.zeros_like(region)
    h, w = region.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = region[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="KK2")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "build" / "corrected_reference" / args.group)
    out_dir.mkdir(parents=True, exist_ok=True)

    added_total = 0
    per_image = []
    covered = 0

    for sequence in find_sequences(TRAIN, groups=(args.group,)):
        masks = [load_masks(p).copy() for p in sequence.paths]
        if not sequence.looks_like_a_movie:
            # Written through unchanged: without a verified frame order there is
            # no temporal evidence, and guessing would corrupt the reference.
            for path, mask in zip(sequence.paths, masks):
                np.save(out_dir / f"{path.stem}_masks.npy", mask.astype(np.int32))
            print(f"{sequence.name}: left unchanged (correlation "
                  f"{sequence.frame_correlation:.3f}, not a verified movie)")
            continue

        covered += sequence.n_frames
        for index in range(1, sequence.n_frames - 1):
            here = [c for _, c in regions_of(masks[index])]
            for region_before, centre_before in regions_of(masks[index - 1]):
                # The same cell must be drawn on the far side too.
                best = None
                for region_after, centre_after in regions_of(masks[index + 1]):
                    distance = float(np.hypot(centre_after[0] - centre_before[0],
                                              centre_after[1] - centre_before[1]))
                    if distance <= SAME_CELL_PX and (best is None or distance < best[0]):
                        best = (distance, region_after, centre_after)
                if best is None:
                    continue
                # ...and must NOT be drawn here.
                if any(np.hypot(x - centre_before[0], y - centre_before[1]) <= SAME_CELL_PX
                       for x, y in here):
                    continue

                centre = ((centre_before[0] + best[2][0]) / 2.0,
                          (centre_before[1] + best[2][1]) / 2.0)
                # The outline is a human mask, moved -- never machine-drawn.
                source = region_before if best[0] > 0 else best[1]
                recovered = shift_to(source, centre) & (masks[index] == 0)
                if recovered.sum() < 80:
                    continue
                masks[index][recovered] = int(masks[index].max()) + 1
                added_total += 1
                per_image.append({
                    "image": sequence.paths[index].name,
                    "at": [round(centre[0], 1), round(centre[1], 1)],
                })
                here.append(centre)

        for path, mask in zip(sequence.paths, masks):
            np.save(out_dir / f"{path.stem}_masks.npy", mask.astype(np.int32))

    original = sum(
        len([v for v in np.unique(load_masks(p)) if v])
        for s in find_sequences(TRAIN, groups=(args.group,)) for p in s.paths
    )
    print()
    print(f"{args.group}: {original} labelled instances originally")
    print(f"          +{added_total} recovered from the annotator's adjacent frames")
    print(f"          = {original + added_total} in the corrected reference")
    for row in per_image:
        print(f"   {row['image']:22s} at {row['at']}")

    report = {
        "group": args.group,
        "original_instances": original,
        "recovered": added_total,
        "corrected_instances": original + added_total,
        "frames_covered_by_temporal_evidence": covered,
        "per_image": per_image,
        "out_dir": str(out_dir),
    }
    (ROOT / "docs" / f"corrected_reference_{args.group}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote masks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
