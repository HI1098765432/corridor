"""Are the 'false positives' that look like cells actually unlabelled cells?

The error table classes 13 of 29 false detections on KK2 as "looks exactly like
a cell": elongated, the right size, at the brightness of the annotator's own
cells, away from the border. Either the model is inventing objects that are
indistinguishable from cells, or the reference is missing them.

That question cannot be settled by the model -- it is the model's claim under
test. It can be settled by **time**. These images are frames of a movie. A cell
present in frame t is present in frames t-1 and t+1, a short distance away. A
piece of debris or a wall artefact is also present in the neighbours, but it
does not move. And a genuine hallucination is present in neither.

So each suspect detection is checked against its own sequence's *human labels*:

*   a labelled cell in a neighbouring frame within a plausible step ->
    **the annotator labelled this cell elsewhere and missed it here**
*   nothing labelled nearby, but the image shows the same object in the
    neighbours at the same place -> **stationary structure**, a true false
    positive
*   nothing in the neighbours at all -> **invented**, a true false positive

Only the first verdict rescues the model, and it is the one supported by the
human's own annotations on adjacent frames rather than by anything the network
produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.sequences import find_sequences, load_image, load_masks  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
#: A cell moves a median of 20-30 px between labelled frames in these sequences,
#: so a labelled cell within this distance in a neighbour is plausibly the same
#: one. Wider than the median, because the tail matters and a wrong "same cell"
#: verdict here would flatter the model.
PLAUSIBLE_STEP_PX = 45.0


def centroids_of(masks: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for label in np.unique(masks):
        if not label:
            continue
        ys, xs = np.nonzero(masks == label)
        out.append((float(xs.mean()), float(ys.mean())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default=str(ROOT / "docs" / "error_table.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "suspect_labels.json"))
    args = ap.parse_args()

    table = json.loads(Path(args.errors).read_text(encoding="utf-8"))
    suspects = [
        row for row in table["false_positives"]
        if row["kind"] == "looks exactly like a cell (suspect label)"
    ]
    print(f"{len(suspects)} suspect false detections to check\n")

    # Index every sequence by image name so a detection can find its neighbours.
    frame_of: dict[str, tuple] = {}
    for sequence in find_sequences(TRAIN):
        for index, path in enumerate(sequence.paths):
            frame_of[path.name] = (sequence, index)

    verdicts = []
    for row in suspects:
        name = row["image"]
        if name not in frame_of:
            verdicts.append({**row, "verdict": "no sequence for this image"})
            continue
        sequence, index = frame_of[name]
        x, y = row["centroid"]

        # 1. Did the annotator label a cell near here in a neighbouring frame?
        nearest = None
        for offset in (-1, 1, -2, 2):
            j = index + offset
            if not (0 <= j < sequence.n_frames):
                continue
            for cx, cy in centroids_of(load_masks(sequence.paths[j])):
                distance = float(np.hypot(cx - x, cy - y))
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, j, offset)

        if nearest and nearest[0] <= PLAUSIBLE_STEP_PX:
            verdict = "a labelled cell is here in a neighbouring frame"
        else:
            # 2. Is the same object physically present in the neighbours,
            #    unmoved? Then it is device structure, not a cell.
            here = load_image(sequence.paths[index])
            patch = (slice(max(0, int(y) - 12), int(y) + 12),
                     slice(max(0, int(x) - 12), int(x) + 12))
            same = []
            for offset in (-1, 1):
                j = index + offset
                if 0 <= j < sequence.n_frames:
                    other = load_image(sequence.paths[j])
                    if other.shape == here.shape:
                        a, b = here[patch].ravel(), other[patch].ravel()
                        if a.std() > 1e-6 and b.std() > 1e-6:
                            same.append(float(np.corrcoef(a, b)[0, 1]))
            if same and float(np.mean(same)) > 0.85:
                verdict = "unmoved structure (device, not a cell)"
            else:
                verdict = "no support in neighbouring frames"

        verdicts.append({
            **row,
            "nearest_labelled_in_neighbour_px": (
                round(nearest[0], 1) if nearest else None
            ),
            "verdict": verdict,
        })

    counts = Counter(v["verdict"] for v in verdicts)
    print(f"{'verdict':48s} {'n':>4}")
    for verdict, n in counts.most_common():
        print(f"{verdict:48s} {n:4d}")

    rescued = counts["a labelled cell is here in a neighbouring frame"]
    if rescued:
        tp = 73
        fp_before, fn = 29, 35
        fp_after = fp_before - rescued
        def prf(tp, fp, fn):
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            return p, r, 2 * p * r / (p + r)
        before = prf(tp, fp_before, fn)
        after = prf(tp, fp_after, fn)
        print()
        print(f"{rescued} of the suspect detections sit where the annotator labelled a")
        print("cell in an adjacent frame of the same field. Counting those as correct")
        print("rather than as errors:")
        print(f"   as scored : P {before[0]:.4f}  R {before[1]:.4f}  F1 {before[2]:.4f}")
        print(f"   corrected : P {after[0]:.4f}  R {after[1]:.4f}  F1 {after[2]:.4f}")
        print()
        print("That is an upper bound on the label-quality effect, not a new score:")
        print("a detection near where a cell was is not proof a cell is here. It")
        print("bounds how much of the remaining error is the reference.")

    out = Path(args.out)
    out.write_text(json.dumps(
        {"n_suspects": len(suspects), "counts": dict(counts), "verdicts": verdicts},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
