"""How many labels are missing, measured without the model proposing anything.

The recovery script lets the model propose candidates and the annotator's own
neighbouring frames accept or reject them. That is sound for *training* -- a
label is only added where a human drew one nearby -- but it cannot be used to
judge the reference, because cells the model also misses never become
candidates. The bias runs one way and it flatters the model.

This uses no model at all. For every cell the annotator labelled, it asks
whether that cell was also labelled in the adjacent frames of the same field. A
cell that exists at t-1 and t+1 but is absent at t did not leave and come back:
these cells move roughly 20-30 px per frame inside a sealed channel. It was not
drawn.

That gives a lower bound on how many labels are missing, and therefore an upper
bound on the F1 any detector can score against this reference -- because a
perfect detector finds the unlabelled cells too, and every one of them is
counted as a false positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.sequences import find_sequences, load_image, load_masks  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
#: Two labelled cells this close in adjacent frames are the same cell.
SAME_CELL_PX = 45.0


def centroids(masks: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for label in np.unique(masks):
        if label:
            ys, xs = np.nonzero(masks == label)
            out.append((float(xs.mean()), float(ys.mean())))
    return out


def nearest(point, others) -> float:
    if not others:
        return float("inf")
    return min(float(np.hypot(x - point[0], y - point[1])) for x, y in others)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="KK2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    total_labelled = 0
    gaps = []

    for sequence in find_sequences(TRAIN, groups=(args.group,)):
        if not sequence.looks_like_a_movie:
            print(f"skipping {sequence.name}: correlation "
                  f"{sequence.frame_correlation:.3f}, not a verified movie")
            continue
        masks = [load_masks(p) for p in sequence.paths]
        images = [load_image(p) for p in sequence.paths]
        points = [centroids(m) for m in masks]
        total_labelled += sum(len(p) for p in points)

        # A cell labelled at t-1 AND t+1, absent at t, was not drawn at t.
        for index in range(1, sequence.n_frames - 1):
            before, here, after = points[index - 1], points[index], points[index + 1]
            for bx, by in before:
                distance_after = nearest((bx, by), after)
                if distance_after > SAME_CELL_PX:
                    continue  # not the same cell on the far side
                if nearest((bx, by), here) <= SAME_CELL_PX:
                    continue  # it was drawn here too
                # Interpolate where it should have been, and check the image.
                ax, ay = min(after, key=lambda q: np.hypot(q[0] - bx, q[1] - by))
                mx, my = (bx + ax) / 2.0, (by + ay) / 2.0
                frame = images[index]
                lo, hi = np.percentile(frame, [1, 99])
                span = max(float(hi - lo), 1e-6)
                patch = frame[max(0, int(my) - 10):int(my) + 10,
                              max(0, int(mx) - 10):int(mx) + 10]
                if patch.size == 0:
                    continue
                contrast = (float(patch.max()) - float(np.median(frame))) / span
                gaps.append({
                    "sequence": sequence.name,
                    "image": sequence.paths[index].name,
                    "at": [round(mx, 1), round(my, 1)],
                    "contrast_there": round(contrast, 3),
                })

    print(f"{args.group}: {total_labelled} labelled instances in verified movies")
    print(f"cells labelled either side of a frame but not in it: {len(gaps)}")
    for row in gaps:
        print(f"   {row['image']:22s} at {row['at']}  contrast {row['contrast_there']:.3f}")

    if total_labelled:
        missing_rate = len(gaps) / (total_labelled + len(gaps))
        print()
        print(f"at least {100 * missing_rate:.1f}% of the cells in these sequences "
              f"carry no label")
        # A perfect detector finds every cell, so each unlabelled one is scored
        # as a false positive: precision caps below 1 while recall stays 1.
        tp = total_labelled
        fp = len(gaps)
        precision = tp / (tp + fp)
        f1_cap = 2 * precision * 1.0 / (precision + 1.0)
        print(f"a PERFECT detector scored against this reference would reach")
        print(f"   precision {precision:.4f}, recall 1.0000, F1 {f1_cap:.4f}")
        print()
        print("That is the ceiling the reference imposes, and it is a lower bound:")
        print("this counts only cells the annotator drew on BOTH neighbours, so")
        print("runs of consecutive missed frames are invisible to it.")

    report = {
        "group": args.group,
        "labelled_instances": total_labelled,
        "missing_labels_found": len(gaps),
        "gaps": gaps,
    }
    out = Path(args.out) if args.out else (
        ROOT / "docs" / f"label_completeness_{args.group}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
