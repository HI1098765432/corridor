"""Is there a cell in the frames nobody labelled? Answered without the network.

Nine of the 71 labelled images contain no labelled cells, and five of those sit
*inside* a sequence with labelled cells on both sides. Either the cells left and
came back, or the annotator stopped. Which one it is decides whether those
frames are legitimate negatives or holes in the reference.

The model must not be the one to answer. Its opinion is exactly what is on
trial: if we accept "the model found a cell there, so a cell is there", we have
assumed the conclusion. So this uses only evidence the model had no part in:

*   **Where** to look comes from the *neighbouring frames' human labels* -- take
    the cells the annotator drew on frame t-1 and t+1 and ask what the image
    contains between them.
*   **What counts as a cell** comes from the intensity statistics of the cells
    the annotator drew elsewhere in the same sequence.

A cell that is genuinely present in an unlabelled frame will sit between its own
labelled positions, at the brightness the annotator's own cells have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.sequences import (  # noqa: E402
    find_sequences,
    load_image,
    load_masks,
)

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"


def cell_statistics(images, masks):
    """How bright a labelled cell is, relative to its own frame."""
    contrasts, areas = [], []
    for image, mask in zip(images, masks):
        labels = [l for l in np.unique(mask) if l]
        if not labels:
            continue
        frame = image.astype(np.float32)
        lo, hi = np.percentile(frame, [1, 99])
        span = max(hi - lo, 1e-6)
        background = float(np.median(frame[mask == 0]))
        for label in labels:
            region = mask == label
            contrasts.append((float(frame[region].mean()) - background) / span)
            areas.append(float(region.sum()))
    return contrasts, areas


def bridge(mask_before: np.ndarray, mask_after: np.ndarray) -> np.ndarray:
    """Where a cell would have to be, if it travelled between two labelled ones.

    The straight-line hull between a cell's position before and after. Any cell
    present in the missing frame lies inside it, because the alternative -- that
    it left the field and returned to the same track -- is not something these
    confined cells do.
    """
    from scipy.ndimage import binary_dilation

    both = (mask_before > 0) | (mask_after > 0)
    if not both.any():
        return both
    # Dilating the union by roughly half a cell length covers the path between.
    return binary_dilation(both, iterations=12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "label_gaps.json"))
    args = ap.parse_args()

    report = {"sequences": []}
    print(f"{'sequence':16s} {'frame':>6} {'contrast here':>14} {'labelled cells':>16}  verdict")

    for sequence in find_sequences(TRAIN):
        if not sequence.looks_like_a_movie:
            continue
        images = [load_image(p) for p in sequence.paths]
        masks = [load_masks(p) for p in sequence.paths]
        contrasts, areas = cell_statistics(images, masks)
        if not contrasts:
            continue
        typical = float(np.median(contrasts))
        floor = float(np.percentile(contrasts, 10))

        rows = []
        for index in sequence.unlabelled_interior_frames:
            before = next((masks[i] for i in range(index - 1, -1, -1)
                           if (masks[i] > 0).any()), None)
            after = next((masks[i] for i in range(index + 1, len(masks))
                          if (masks[i] > 0).any()), None)
            if before is None or after is None:
                continue

            frame = images[index].astype(np.float32)
            lo, hi = np.percentile(frame, [1, 99])
            span = max(hi - lo, 1e-6)
            corridor = bridge(before, after)
            background = float(np.median(frame[~corridor]))

            # The brightest cell-sized patch anywhere along the path.
            from scipy.ndimage import uniform_filter

            window = max(3, int(round(np.sqrt(np.median(areas)))))
            smoothed = uniform_filter(frame, size=window)
            inside = smoothed[corridor]
            peak = float(np.percentile(inside, 99.5)) if inside.size else background
            contrast_here = (peak - background) / span

            verdict = (
                "a cell is there, unlabelled" if contrast_here >= floor
                else "plausibly empty"
            )
            rows.append({
                "frame_index": index,
                "image": sequence.paths[index].name,
                "contrast_found": round(contrast_here, 4),
                "typical_labelled_contrast": round(typical, 4),
                "weakest_labelled_contrast": round(floor, 4),
                "verdict": verdict,
            })
            print(f"{sequence.name:16s} {index:6d} {contrast_here:14.3f} "
                  f"{typical:16.3f}  {verdict}")

        report["sequences"].append({
            "name": sequence.name,
            "n_frames": sequence.n_frames,
            "frame_correlation": round(sequence.frame_correlation, 3),
            "typical_labelled_contrast": round(typical, 4),
            "unlabelled_interior": rows,
        })

    found = [r for s in report["sequences"] for r in s["unlabelled_interior"]
             if r["verdict"].startswith("a cell")]
    total = [r for s in report["sequences"] for r in s["unlabelled_interior"]]
    report["unlabelled_frames_examined"] = len(total)
    report["unlabelled_frames_containing_a_cell"] = len(found)

    print()
    print(f"interior frames with no labels examined : {len(total)}")
    print(f"of those, a cell is visibly present in  : {len(found)}")
    if total:
        print()
        print("Every detection a model makes in those frames is currently scored as a")
        print("false positive, and every one of them is used in training as background.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
