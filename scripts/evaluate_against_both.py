"""Score every model against the corrected reference first, and the original beside it.

Round 2 appeared to fail: F1 0.653 -> 0.645. But recall rose (0.602 -> 0.648)
while precision fell (0.714 -> 0.642). Against a reference that is missing
cells, that is what a model that got *better* can look like: it found more
cells, and the cells it found were scored as false positives because nobody had
drawn them.

Which of the two it is cannot be decided from the original labels alone. So
every model is scored twice -- against the corrected reference (by default
version 2, ``scripts/build_corrected_reference.py``: cells the annotator drew at
the nearest labelled times either side, on true time order, positioned by
elapsed-time interpolation and outlined with a human mask) and against the
labels as supplied. The corrected figure leads the printout and the undrawn-cell
count is printed with it; the original is always printed beside it, never
dropped. ``--reference legacy`` scores against the withdrawn version 1
(filename adjacency, ``build/corrected_reference``) to reproduce
``docs/evaluation_grid.json``.

``--border-margin N`` drops truth and predicted instances within N px of the
image edge before counting (``training.scoring``; the distance is
``scripts/diag_errors.py``'s). It is a separate figure, reported as such.

The output never overwrites an existing file unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from training.scoring import drop_border, score as score_with_margin  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
REFERENCES = {
    "v2": ROOT / "build" / "corrected_reference_v2",
    "legacy": ROOT / "build" / "corrected_reference",
}

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


def corrected_masks(path: Path, group: str, reference: str = "v2") -> np.ndarray | None:
    candidate = REFERENCES[reference] / group / f"{path.stem}_masks.npy"
    if candidate.exists():
        return np.load(candidate).astype(np.int32)
    return None


def labelled(group: str) -> list[Path]:
    return [
        p for p in sorted((TRAIN / group).glob("*.tif"))
        if p.with_name(p.name.replace(".tif", "_seg.npy")).exists()
    ]


def count(masks: np.ndarray, margin: int | None) -> int:
    return len([v for v in np.unique(drop_border(masks, margin)) if v])


def default_out(reference: str, margin: int | None) -> Path:
    if reference == "legacy" and margin is None:
        return ROOT / "docs" / "evaluation_grid.json"
    suffix = f"_{reference}" + (f"_margin{margin}px" if margin is not None else "")
    return ROOT / "docs" / f"evaluation_grid{suffix}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="KK2")
    ap.add_argument("--reference", default="v2", choices=tuple(REFERENCES),
                    help="corrected reference: v2 (true time order, default) or the "
                         "withdrawn legacy version 1")
    ap.add_argument("--border-margin", type=int, default=None,
                    help="exclude truth and predicted instances within this many px of "
                         "the edge before counting")
    ap.add_argument("--out", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else default_out(args.reference, args.border_margin)
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite {out}; pass --out or --overwrite")
        return 2

    paths = labelled(args.group)
    have_corrected = all(corrected_masks(p, args.group, args.reference) is not None
                         for p in paths)
    if not have_corrected:
        print(f"no {args.reference} corrected reference for {args.group}; run "
              f"scripts/build_corrected_reference.py --group {args.group}"
              f"{' --legacy' if args.reference == 'legacy' else ''}")
        return 1

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    margin = args.border_margin
    n_original = sum(count(original_masks(p), None) for p in paths)
    n_corrected = sum(count(corrected_masks(p, args.group, args.reference), None)
                      for p in paths)
    undrawn = n_corrected - n_original
    print(f"{args.group}: {len(paths)} images, {n_corrected} instances in the corrected "
          f"reference {args.reference} = {n_original} labelled + {undrawn} the annotator "
          f"did not draw")
    if margin is not None:
        kept_corrected = sum(count(corrected_masks(p, args.group, args.reference), margin)
                             for p in paths)
        kept_original = sum(count(original_masks(p), margin) for p in paths)
        print(f"border margin {margin} px (this changes the measurement, not the model): "
              f"{kept_corrected} corrected and {kept_original} original instances remain")
    print()

    rows = []
    print(f"{'model':32s} {'window':9s} {'corrected F1':>13} {'corr. P':>8} {'corr. R':>8} "
          f"{'original F1':>12}")

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
                a = score_with_margin(original_masks(path), prediction, margin_px=margin)
                b = score_with_margin(corrected_masks(path, args.group, args.reference),
                                      prediction, margin_px=margin)
                total_original = a if total_original is None else total_original + a
                total_corrected = b if total_corrected is None else total_corrected + b

            row = {
                "model": model_name, "window": window_name,
                "corrected": total_corrected.to_row(),
                "original": total_original.to_row(),
            }
            rows.append(row)
            print(f"{model_name:32s} {window_name:9s} "
                  f"{total_corrected.f1:13.4f} {total_corrected.precision:8.4f} "
                  f"{total_corrected.recall:8.4f} {total_original.f1:12.4f}", flush=True)

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(
                {"group": args.group, "reference": args.reference,
                 "corrected_reference_dir": REFERENCES[args.reference].relative_to(ROOT).as_posix(),
                 "border_margin_px": margin,
                 "border_margin_note": None if margin is None else
                 "changes the measurement, not the model",
                 "original_instances": n_original, "corrected_instances": n_corrected,
                 "undrawn_cells": undrawn, "results": rows},
                indent=2), encoding="utf-8")

    if rows:
        best_corrected = max(rows, key=lambda r: r["corrected"]["f1"])
        best_original = max(rows, key=lambda r: r["original"]["f1"])
        print()
        print(f"best against the CORRECTED labels ({args.reference}, +{undrawn} undrawn): "
              f"{best_corrected['model']} {best_corrected['window']} at F1 "
              f"{best_corrected['corrected']['f1']:.4f}")
        print(f"best against the ORIGINAL labels : {best_original['model']} "
              f"{best_original['window']} at F1 {best_original['original']['f1']:.4f}")
        if best_original["model"] != best_corrected["model"]:
            print()
            print("The two references disagree about which model is better. The")
            print("corrected one counts cells the annotator missed; the original")
            print("scores a detector for finding them.")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
