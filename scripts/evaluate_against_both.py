"""Score every model against both the original labels and the corrected ones.

Round 2 appeared to fail: F1 0.653 -> 0.645. But recall rose (0.602 -> 0.648)
while precision fell (0.714 -> 0.642). Against a reference measured to be at
least 11 % incomplete, that is exactly what a model that got *better* looks
like: it found more cells, and the cells it found were scored as false positives
because nobody had drawn them.

Which of the two it is cannot be decided from the original labels alone. So
every model is scored twice -- against the labels as supplied, and against the
same labels with the provably-missing cells put back by
``scripts/build_corrected_reference.py``, where each addition is positioned by
interpolating two human centroids and outlined with a human mask from the
adjacent frame.

Both columns are always reported. The original keeps every figure comparable
with what has already been published; the corrected one is what the images
actually contain. A model that improves on one and not the other is telling you
something, and which way round it goes is the whole question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.metrics import score_image  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
CORRECTED = ROOT / "build" / "corrected_reference"

MODELS = {
    "KK1Model (original)":
        TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "round 1 (contrast aug)":
        ROOT / "build" / "models" / "models" / "corridor_contrast_invariant",
    "round 2 (aug + trained at 3,97)":
        ROOT / "build" / "models" / "models" / "corridor_contrast_invariant_w3_97",
}
WINDOWS = {
    "default": True,
    "(3, 97)": {"percentile": (3.0, 97.0)},
    "(5, 95)": {"percentile": (5.0, 95.0)},
}


def load(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image.astype(np.float32)


def original_masks(path: Path) -> np.ndarray:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


def corrected_masks(path: Path, group: str) -> np.ndarray | None:
    candidate = CORRECTED / group / f"{path.stem}_masks.npy"
    if candidate.exists():
        return np.load(candidate).astype(np.int32)
    return None


def labelled(group: str) -> list[Path]:
    return [
        p for p in sorted((TRAIN / group).glob("*.tif"))
        if p.with_name(p.name.replace(".tif", "_seg.npy")).exists()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="KK2")
    ap.add_argument("--out", default=str(ROOT / "docs" / "evaluation_grid.json"))
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    paths = labelled(args.group)
    have_corrected = all(corrected_masks(p, args.group) is not None for p in paths)
    if not have_corrected:
        print(f"no corrected reference for {args.group}; run "
              f"scripts/build_corrected_reference.py --group {args.group}")
        return 1

    n_original = sum(len([v for v in np.unique(original_masks(p)) if v]) for p in paths)
    n_corrected = sum(
        len([v for v in np.unique(corrected_masks(p, args.group)) if v]) for p in paths
    )
    print(f"{args.group}: {len(paths)} images, {n_original} labelled instances, "
          f"{n_corrected} in the corrected reference "
          f"(+{n_corrected - n_original})\n")

    rows = []
    print(f"{'model':32s} {'window':9s} "
          f"{'original F1':>12} {'corrected F1':>13} {'corr. P':>8} {'corr. R':>8}")

    for model_name, weights in MODELS.items():
        if not Path(weights).exists():
            print(f"{model_name:32s} -- not found, skipped")
            continue
        model = cp.CellposeModel(pretrained_model=str(weights), gpu=False)

        for window_name, normalize in WINDOWS.items():
            total_original = None
            total_corrected = None
            for path in paths:
                prediction = np.asarray(
                    model.eval(load(path), channels=[0, 0], diameter=None,
                               cellprob_threshold=0.0, flow_threshold=0.4,
                               normalize=normalize)[0]
                ).astype(np.int32)
                a = score_image(original_masks(path), prediction)
                b = score_image(corrected_masks(path, args.group), prediction)
                total_original = a if total_original is None else total_original + a
                total_corrected = b if total_corrected is None else total_corrected + b

            row = {
                "model": model_name, "window": window_name,
                "original": total_original.to_row(),
                "corrected": total_corrected.to_row(),
            }
            rows.append(row)
            print(f"{model_name:32s} {window_name:9s} "
                  f"{total_original.f1:12.4f} {total_corrected.f1:13.4f} "
                  f"{total_corrected.precision:8.4f} {total_corrected.recall:8.4f}",
                  flush=True)

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"group": args.group, "original_instances": n_original,
                 "corrected_instances": n_corrected, "results": rows},
                indent=2), encoding="utf-8")

    if rows:
        best_original = max(rows, key=lambda r: r["original"]["f1"])
        best_corrected = max(rows, key=lambda r: r["corrected"]["f1"])
        print()
        print(f"best against the ORIGINAL labels : {best_original['model']} "
              f"{best_original['window']} at F1 {best_original['original']['f1']:.4f}")
        print(f"best against the CORRECTED labels: {best_corrected['model']} "
              f"{best_corrected['window']} at F1 {best_corrected['corrected']['f1']:.4f}")
        if best_original["model"] != best_corrected["model"]:
            print()
            print("The two references disagree about which model is better. The")
            print("corrected one counts cells the annotator missed; the original")
            print("scores a detector for finding them.")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
