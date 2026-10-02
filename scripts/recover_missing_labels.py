"""Put back the cells the annotator missed, using the annotator's own evidence.

Measured on KK2: of 29 false detections, 13 look exactly like cells, and 11 of
those sit where the annotator labelled a cell in an adjacent frame of the same
field. They are cells that were missed, not objects that were invented.

That matters far more for **training** than for scoring. A missed label is not a
neutral absence: the pixels of a real cell are handed to the optimiser as
certain background, so the network is actively taught that a cell there is not a
cell. With 46 % of instances already discarded by ``min_train_masks`` and 9
images carrying no labels at all, the training signal has been contradicting
itself.

**Where the new label comes from, and why it is not the model's opinion.**
A candidate is accepted only when the *annotator* drew a cell at that position in
an adjacent frame of the same sequence, close enough to be the same cell moving.
The model proposes where to look; the human's own neighbouring annotations
decide. The outline is taken from the adjacent frame's human-drawn mask,
translated onto the candidate's centroid -- so every recovered mask is a human
mask, moved, never a machine-drawn one.

**What it refuses.** A candidate with no labelled neighbour is rejected, however
convincing. A candidate whose surroundings are identical in the neighbouring
frames is stationary device structure and is rejected. Nothing is accepted on
the strength of looking like a cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.metrics import iou_matrix, match  # noqa: E402
from corridor.learn.sequences import find_sequences, load_image, load_masks  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
#: A labelled cell this close in a neighbouring frame is plausibly the same one.
PLAUSIBLE_STEP_PX = 45.0
#: Above this correlation the neighbourhood did not change between frames, so
#: whatever is there is fixed to the device rather than migrating through it.
STATIONARY_CORRELATION = 0.85


def shift_to(region: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return region
    dy = int(round(centre[1] - ys.mean()))
    dx = int(round(centre[0] - xs.mean()))
    out = np.zeros_like(region)
    h, w = region.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = region[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def labelled_regions(masks: np.ndarray):
    for label in np.unique(masks):
        if label:
            region = masks == label
            ys, xs = np.nonzero(region)
            yield region, (float(xs.mean()), float(ys.mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group", default="KK1")
    ap.add_argument("--window", default="3,97")
    ap.add_argument("--out-dir", default=str(ROOT / "build" / "recovered_labels"))
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    model = cp.CellposeModel(pretrained_model=str(args.model), gpu=False)
    normalize: object = True
    if args.window:
        lo, hi = (float(v) for v in args.window.split(","))
        normalize = {"percentile": (lo, hi)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted = rejected_no_neighbour = rejected_stationary = 0
    per_image = []

    for sequence in find_sequences(TRAIN, groups=(args.group,)):
        if not sequence.looks_like_a_movie:
            print(f"skipping {sequence.name}: frame correlation "
                  f"{sequence.frame_correlation:.3f} is too low to trust as a movie")
            continue

        images = [load_image(p) for p in sequence.paths]
        masks = [load_masks(p).copy() for p in sequence.paths]

        for index, path in enumerate(sequence.paths):
            pred = np.asarray(
                model.eval(images[index], channels=[0, 0], diameter=None,
                           cellprob_threshold=0.0, flow_threshold=0.4,
                           normalize=normalize)[0]
            ).astype(np.int32)

            ious = iou_matrix(masks[index], pred)
            claimed = {c for _, c in match(ious)}
            pred_labels = [int(v) for v in np.unique(pred) if v]
            added_here = 0

            for j, label in enumerate(pred_labels):
                if j in claimed:
                    continue
                candidate = pred == label
                ys, xs = np.nonzero(candidate)
                centre = (float(xs.mean()), float(ys.mean()))

                # The annotator must have drawn this cell in a neighbour.
                support = None
                for offset in (-1, 1, -2, 2):
                    k = index + offset
                    if not (0 <= k < sequence.n_frames):
                        continue
                    for region, (cx, cy) in labelled_regions(masks[k]):
                        distance = float(np.hypot(cx - centre[0], cy - centre[1]))
                        if distance <= PLAUSIBLE_STEP_PX and (
                            support is None or distance < support[0]
                        ):
                            support = (distance, region)
                if support is None:
                    rejected_no_neighbour += 1
                    continue

                # ...and the neighbourhood must have changed, or it is device.
                patch = (slice(max(0, int(centre[1]) - 12), int(centre[1]) + 12),
                         slice(max(0, int(centre[0]) - 12), int(centre[0]) + 12))
                stationary = []
                for offset in (-1, 1):
                    k = index + offset
                    if 0 <= k < sequence.n_frames and images[k].shape == images[index].shape:
                        a = images[index][patch].ravel()
                        b = images[k][patch].ravel()
                        if a.std() > 1e-6 and b.std() > 1e-6:
                            stationary.append(float(np.corrcoef(a, b)[0, 1]))
                if stationary and float(np.mean(stationary)) > STATIONARY_CORRELATION:
                    rejected_stationary += 1
                    continue

                # The outline is the HUMAN mask from the neighbour, moved here.
                recovered = shift_to(support[1], centre)
                recovered &= masks[index] == 0
                if recovered.sum() < 80:
                    rejected_no_neighbour += 1
                    continue
                masks[index][recovered] = masks[index].max() + 1
                accepted += 1
                added_here += 1

            if added_here:
                per_image.append({"image": path.name, "added": added_here})

        for path, mask in zip(sequence.paths, masks):
            np.save(out_dir / f"{path.stem}_masks.npy", mask.astype(np.int32))

    print()
    print(f"accepted   {accepted:4d}  a cell the annotator drew in a neighbouring frame")
    print(f"rejected   {rejected_no_neighbour:4d}  nothing labelled nearby in any neighbour")
    print(f"rejected   {rejected_stationary:4d}  unmoved between frames (device structure)")
    if per_image:
        print(f"\n{len(per_image)} images gained a label:")
        for row in per_image[:20]:
            print(f"   {row['image']:24s} +{row['added']}")

    report = {
        "group": args.group, "model": args.model,
        "accepted": accepted,
        "rejected_no_labelled_neighbour": rejected_no_neighbour,
        "rejected_stationary": rejected_stationary,
        "per_image": per_image,
        "out_dir": str(out_dir),
    }
    (ROOT / "docs" / f"recovered_labels_{args.group}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote masks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
