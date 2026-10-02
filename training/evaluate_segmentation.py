"""Evaluate one Cellpose 3 checkpoint on one locked split, the way the contract says.

    python -m training.evaluate_segmentation --checkpoint build/models/models/<name> \\
        --split test --window 3,97 --diameter 36 \\
        --border-margin 3 --border-margin 10 --refine image --refine contour

What it reports, per variant (unmodified first, then each ``--border-margin``
and each ``--refine`` method, each beside the unmodified figure):

- AP at IoU 0.50:0.05:0.95 (``TP/(TP+FP+FN)``, pooled) and its mean;
  precision, recall and F1 at IoU 0.5; FP and FN per image; splits and merges.
- Breakdowns of recall and precision by experiment, contrast quartile, size
  quartile and distance to the border.
- **Against the corrected reference v2 first** (``build/corrected_reference_v2``,
  ``scripts/build_corrected_reference.py``), and against the original labels
  beside it, with the number of cells v2 adds stated. Scores against the
  corrected reference are never reported instead of the original ones.

**Whether the split is held out from this checkpoint** is checked, not
assumed: the checkpoint's SHA-256 is looked up in the research model registry
and its whole parent chain must be recorded as never trained on the split's
experiments (:func:`training.registry.held_out`). Otherwise the report says
``held_out: false`` with the reason and the run prints a warning. Every pre-v2
checkpoint, round 4 of ``scripts/train_contrast_invariant.py`` included, was
trained on all of KK1 and therefore on the locked validation experiment: its
``--split val`` figure is a training-set figure.

Scoring is ``corridor.core.metrics`` through :mod:`training.scoring` only.
The checkpoint is an explicit path and its SHA-256 is recorded; there is no
default model and no name lookup, so nothing can silently fall back to a
built-in (``docs/NEXT_GENERATION.md`` section 2 explains why that matters with
Cellpose). ``--dry-run`` checks the arguments, the split, both references and
the checkpoint file without importing torch or Cellpose. ``--predictions``
rescores saved label images without a model at all.

The output never overwrites an existing file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from training import ROOT
from training.datasets import REGISTRY_PATH, load_registry, sha256_file
from training.refine import METHODS as REFINE_METHODS, refine_labels
from training.scoring import (
    AP_THRESHOLDS,
    describe_instances,
    drop_border,
    evaluate_image,
    pool,
    standard_breakdowns,
)
from training.registry import MODELS_PATH, held_out
from training.splits import SPLITS_PATH, load_splits

from corridor.learn.sequences import load_image, load_masks  # noqa: E402

CORRECTED_V2 = ROOT / "build" / "corrected_reference_v2"
EVAL_DIR = ROOT / "build" / "eval"


def parse_window(text: str):
    """``default`` -> True (Cellpose's own 1-99), ``3,97`` -> a percentile window.

    Never ``{"lowhigh": ...}``: it returns zero masks in cellpose 3.1.1.3.
    """
    if text in ("", "default", "true", "True"):
        return True
    lo, hi = (float(v) for v in text.split(","))
    if not 0.0 <= lo < hi <= 100.0:
        raise argparse.ArgumentTypeError(f"bad percentile window {text!r}")
    return {"percentile": (lo, hi)}


def parse_diameter(text: str) -> float | None:
    return None if text in ("", "none", "None", "checkpoint") else float(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="",
                    help="explicit path to a Cellpose 3 checkpoint (required unless --predictions)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--splits", default=str(SPLITS_PATH))
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--models", default=str(MODELS_PATH),
                    help="research model registry, for what the checkpoint was trained on")
    ap.add_argument("--window", default="3,97", type=str,
                    help="normalisation percentile window 'lo,hi', or 'default'")
    ap.add_argument("--diameter", default="none", type=str,
                    help="diameter in px, or 'none' for the checkpoint's own diam_labels")
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--border-margin", type=int, action="append", default=[],
                    help="also report with instances within this many px of the edge "
                         "excluded from truth and prediction (repeatable)")
    ap.add_argument("--refine", action="append", default=[], choices=REFINE_METHODS,
                    help="also report with per-instance boundary refinement (repeatable)")
    ap.add_argument("--refine-max-px", type=int, default=3)
    ap.add_argument("--reference-dir", default=str(CORRECTED_V2),
                    help="corrected reference v2 folder (holds <group>/<stem>_masks.npy)")
    ap.add_argument("--predictions", default="",
                    help="score saved label images (.npz keyed like --save-predictions) "
                         "instead of running a model")
    ap.add_argument("--save-predictions", default="",
                    help="write the raw predicted label images to this .npz")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="", help="report path (default build/eval/...); "
                                              "never overwritten")
    ap.add_argument("--dry-run", action="store_true",
                    help="check arguments, data and checkpoint; load no model")
    return ap


def npz_key(image_id: str) -> str:
    return image_id.replace("/", "__")


def load_inputs(args) -> dict:
    """Everything except the model: images, both references, metadata."""
    splits = load_splits(Path(args.splits))
    registry = load_registry(Path(args.registry))
    if splits["dataset_registry_sha256"] != registry["content_sha256"]:
        raise SystemExit(f"{args.splits} was made from a different dataset registry "
                         f"than {args.registry}; rebuild the splits or pass the right registry")
    records = {r["image_id"]: r for r in registry["records"]}
    image_ids = list(splits["splits"][args.split]["labelled_images"])
    reference_dir = Path(args.reference_dir)
    items, missing = [], []
    for image_id in image_ids:
        record = records[image_id]
        path = ROOT / record["path"]
        corrected = reference_dir / record["group"] / f"{path.stem}_masks.npy"
        if not corrected.exists():
            missing.append(str(corrected))
            continue
        items.append({"id": image_id, "path": path, "experiment": record["experiment_id"],
                      "group": record["group"], "corrected_path": corrected})
    if missing:
        raise SystemExit(f"{len(missing)} corrected-reference files missing, e.g. {missing[0]}; "
                         f"run scripts/build_corrected_reference.py for that group")
    for item in items:
        item["original"] = load_masks(item["path"])
        item["corrected"] = np.load(item["corrected_path"]).astype(np.int32)
        if item["corrected"].shape != item["original"].shape:
            raise SystemExit(f"{item['id']}: reference shapes differ")
    n_original = sum(len([v for v in np.unique(i["original"]) if v]) for i in items)
    n_corrected = sum(len([v for v in np.unique(i["corrected"]) if v]) for i in items)
    return {"splits": splits, "registry": registry, "items": items,
            "n_original": n_original, "n_corrected": n_corrected}


def cellpose_version() -> str | None:
    try:
        return metadata.version("cellpose")
    except metadata.PackageNotFoundError:
        return None


def predict(checkpoint: Path, items: list[dict], *, window, diameter, cellprob, flow,
            threads: int) -> dict[str, np.ndarray]:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(threads)
    from cellpose import models as cp

    model = cp.CellposeModel(pretrained_model=str(checkpoint), gpu=False)
    out = {}
    for item in items:
        image = load_image(item["path"]).astype(np.float32)
        out[item["id"]] = np.asarray(model.eval(
            image, channels=[0, 0], diameter=diameter, cellprob_threshold=cellprob,
            flow_threshold=flow, normalize=window)[0]).astype(np.int32)
    return out


def score_variant(items: list[dict], predictions: dict[str, np.ndarray], *,
                  margin_px: int | None = None) -> dict:
    out = {}
    for reference in ("corrected", "original"):
        results, rows = [], []
        for item in items:
            truth = item[reference]
            prediction = predictions[item["id"]]
            results.append(evaluate_image(item["id"], truth, prediction, margin_px=margin_px))
            rows += describe_instances(item["id"], load_image(item["path"]),
                                       drop_border(truth, margin_px),
                                       drop_border(prediction, margin_px),
                                       group=item["experiment"])
        pooled = pool(results)
        pooled["breakdowns"] = standard_breakdowns(rows)
        out["corrected_v2" if reference == "corrected" else "original"] = pooled
    return out


def run(args, inputs: dict, predictions: dict[str, np.ndarray]) -> dict:
    items = inputs["items"]
    variants = [("unmodified", predictions, None)]
    for margin in args.border_margin:
        variants.append((f"border_margin_{margin}px", predictions, margin))
    for method in args.refine:
        refined = {i["id"]: refine_labels(load_image(i["path"]), predictions[i["id"]],
                                          method=method, max_px=args.refine_max_px)
                   for i in items}
        variants.append((f"refine_{method}_{args.refine_max_px}px", refined, None))
    results = []
    for name, preds, margin in variants:
        started = time.time()
        results.append({"variant": name, "border_margin_px": margin,
                        **score_variant(items, preds, margin_px=margin),
                        "seconds": round(time.time() - started, 1),
                        "note": ("changes the measurement, not the model" if margin is not None
                                 else "changes the prediction; the references are untouched"
                                 if name.startswith("refine") else "")})
    return {"results": results}


def default_out(args, checkpoint: Path | None) -> Path:
    stem = checkpoint.name if checkpoint else Path(args.predictions).stem
    window = args.window.replace(",", "-")
    return EVAL_DIR / f"{stem}_{args.split}_w{window}_d{args.diameter}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    window = parse_window(args.window)
    diameter = parse_diameter(args.diameter)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None and not args.predictions:
        print("--checkpoint is required (an explicit path; there is no default model)")
        return 2
    if checkpoint is not None and not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}")
        return 2
    version = cellpose_version()
    if checkpoint is not None and (version is None or not version.startswith("3.")):
        print(f"this evaluates Cellpose 3 checkpoints; installed cellpose is {version}")
        return 2

    out = Path(args.out) if args.out else default_out(args, checkpoint)
    if out.exists():
        print(f"refusing to overwrite {out}; pass a new --out")
        return 2

    inputs = load_inputs(args)
    items = inputs["items"]
    digest = sha256_file(checkpoint) if checkpoint else None
    split_experiments = sorted({i["experiment"] for i in items})
    provenance = (held_out(digest, split_experiments, path=Path(args.models)) if digest else
                  {"held_out": False, "lineage": [],
                   "reason": "scored from saved predictions: the model behind them is not "
                             "identified"})
    plan = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": digest,
        "held_out": provenance["held_out"],
        "held_out_reason": provenance["reason"],
        "lineage": provenance["lineage"],
        "predictions_file": args.predictions or None,
        "cellpose_version": version,
        "split": args.split,
        "splits_version": inputs["splits"]["version"],
        "dataset_registry": inputs["registry"]["version"],
        "dataset_registry_sha256": inputs["registry"]["content_sha256"],
        "settings": {"normalize": window, "diameter": diameter, "cellprob_threshold":
                     args.cellprob, "flow_threshold": args.flow, "channels": [0, 0]},
        "variants": ["unmodified"] + [f"border_margin_{m}px" for m in args.border_margin]
                    + [f"refine_{m}_{args.refine_max_px}px" for m in args.refine],
        "images": len(items),
        "experiments": sorted({i["experiment"] for i in items}),
        "reference": {
            "corrected_v2_dir": Path(args.reference_dir).as_posix(),
            "original_instances": inputs["n_original"],
            "corrected_v2_instances": inputs["n_corrected"],
            "undrawn_cells_added_in_v2": inputs["n_corrected"] - inputs["n_original"],
        },
        "ap_thresholds": list(AP_THRESHOLDS),
    }
    print(f"{args.split}: {len(items)} images from {', '.join(plan['experiments'])}; "
          f"{inputs['n_original']} labelled instances, {inputs['n_corrected']} in the "
          f"corrected reference v2 (+{plan['reference']['undrawn_cells_added_in_v2']} undrawn)")
    print(f"checkpoint {checkpoint} sha256 {digest}" if checkpoint
          else f"predictions {args.predictions}")
    print(f"window {window}, diameter {diameter}, variants {plan['variants']}")
    if not plan["held_out"]:
        print(f"WARNING: not shown to be held out from this model -- {plan['held_out_reason']}. "
              f"The report records held_out: false; do not quote it as a held-out figure.")
    if args.dry_run:
        print(f"dry run: inputs load; would write {out}")
        return 0

    if args.predictions:
        with np.load(args.predictions) as saved:
            predictions = {i["id"]: saved[npz_key(i["id"])].astype(np.int32) for i in items}
    else:
        started = time.time()
        predictions = predict(checkpoint, items, window=window, diameter=diameter,
                              cellprob=args.cellprob, flow=args.flow, threads=args.threads)
        plan["inference_minutes"] = round((time.time() - started) / 60.0, 2)
        if args.save_predictions:
            Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_predictions,
                                **{npz_key(k): v for k, v in predictions.items()})

    report = {**plan, **run(args, inputs, predictions)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    undrawn = plan["reference"]["undrawn_cells_added_in_v2"]
    print(f"\n{'variant':28s} {'reference':13s} {'F1@.5':>7} {'P':>7} {'R':>7} "
          f"{'AP.5:.95':>9} {'FP/img':>7} {'FN/img':>7} {'split':>6} {'merge':>6}")
    for result in report["results"]:
        for key, label in (("corrected_v2", f"v2 (+{undrawn})"), ("original", "original")):
            r = result[key]
            row = r["at_iou_0.5"]
            print(f"{result['variant']:28s} {label:13s} {row['f1']:7.4f} {row['precision']:7.4f} "
                  f"{row['recall']:7.4f} {r['ap_50_95'] if r['ap_50_95'] is not None else '-':>9} "
                  f"{r['fp_per_image']:7.3f} {r['fn_per_image']:7.3f} {r['splits']:6d} "
                  f"{r['merges']:6d}")
    if args.border_margin:
        print("border-margin rows change the measurement, not the model.")
    if not plan["held_out"]:
        print(f"NOT HELD OUT: {plan['held_out_reason']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
