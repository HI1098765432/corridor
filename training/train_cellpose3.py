"""Fine-tune a Cellpose 3 checkpoint on a locked split with a recorded policy.

    python -m training.train_cellpose3 --start <checkpoint> --name <new name> \\
        --epochs 60 --window 3,97 [--labels corrected_v2] [--policy policy.json]

The generalisation of ``scripts/train_contrast_invariant.py``, with the parts
that made its results hard to trace removed:

- **Data from the split, not from a folder.** Training images are the
  ``train`` split of ``build/registry/splits_v1.json``; the run refuses to start
  if any validation or test image, or any experiment the split marks
  never-train (the sample movies' 20240529-s01 among them), would be used. Of
  each set of pixel-identical stills only the first is kept (``041824_2`` and
  ``_7`` are one frame labelled twice; keeping both double-weights it with two
  disagreeing outlines). ``--keep-duplicates`` keeps them.
- **Labels named.** ``--labels corrected_v2`` (default) reads
  ``build/corrected_reference_v2``; ``original`` reads the ``_seg.npy`` files.
- **An explicit start checkpoint**, hashed; no default model and no fallback.
- **A recorded augmentation policy** (:mod:`training.augmentations`): the full
  policy and every sampled parameter of every copy go into the report.
- **The checkpoint is scored after reloading it from disk**, on the
  validation split, against corrected v2 and the original labels. In-memory
  and reloaded scores of the same run have differed here before (round 1
  default window 0.5389 against 0.5297), and the file is what anyone else gets.
  That score is marked held out only if the start checkpoint's registered
  lineage never saw the validation experiment (:func:`training.registry.
  held_out`); starting from any pre-v2 checkpoint, it did. The run's own
  training experiments are recorded in the report and the model registry.
  The locked test split is scored only by ``training.evaluate_segmentation``.

Training settings carried over because each was measured to matter:
``min_train_masks=0`` (the default 5 drops 48 of 71 images and every negative),
``rescale=False`` (the equivalent-disk diameter of a confined cell is set by its
length), MKL-DNN off for training, and pinned threads. The checkpoint goes to
``build/models/models/<name>`` and an existing file is refused, never replaced
(Cellpose's ``train_seg`` overwrites silently).

``--dry-run`` builds the training set and writes nothing; it imports no torch.
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
from training.augmentations import AugmentationPolicy, build_training_set
from training.datasets import REGISTRY_PATH, load_registry, sha256_file
from training.registry import held_out
from training.splits import NEVER_TRAIN_EXPERIMENTS, SPLITS_PATH, load_splits

from corridor.learn.sequences import load_image, load_masks  # noqa: E402

MODELS_DIR = ROOT / "build" / "models"
REPORTS_DIR = ROOT / "build" / "models" / "reports"
CORRECTED_V2 = ROOT / "build" / "corrected_reference_v2"


def select_images(registry: dict, splits: dict, split_names: list[str], *,
                  keep_duplicates: bool = False) -> tuple[list[dict], list[str]]:
    """Registry records of the chosen splits, guarded against leakage."""
    records = {r["image_id"]: r for r in registry["records"]}
    chosen = [records[i] for name in split_names for i in splits["splits"][name]["labelled_images"]]
    forbidden = {i for name in ("train", "val", "test") if name not in split_names
                 for i in splits["splits"][name]["images"]}
    leaks = [r["image_id"] for r in chosen if r["image_id"] in forbidden
             or r["experiment_id"] in NEVER_TRAIN_EXPERIMENTS]
    if leaks:
        raise SystemExit(f"refusing to train on held-out or never-train images: {leaks}")
    dropped: list[str] = []
    if not keep_duplicates:
        kept: list[dict] = []
        seen: set[str] = set()
        for r in chosen:
            if r["image_id"] in seen:
                dropped.append(r["image_id"])
                continue
            seen.update(r["duplicate_of"])
            kept.append(r)
        chosen = kept
    return chosen, dropped


def labels_for(record: dict, labels: str, reference_dir: Path) -> np.ndarray:
    path = ROOT / record["path"]
    if labels == "original":
        return load_masks(path)
    corrected = reference_dir / record["group"] / f"{path.stem}_masks.npy"
    if not corrected.exists():
        raise SystemExit(f"no corrected label for {record['image_id']} at {corrected}; "
                         f"run scripts/build_corrected_reference.py --group {record['group']}")
    return np.load(corrected).astype(np.int32)


def parse_window(text: str):
    if text in ("", "default"):
        return True
    lo, hi = (float(v) for v in text.split(","))
    return {"percentile": (lo, hi)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True, help="explicit Cellpose 3 checkpoint path")
    ap.add_argument("--name", required=True, help="new checkpoint name (never overwritten)")
    ap.add_argument("--splits", default=str(SPLITS_PATH))
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--train-on", default="train", choices=("train", "train+val"),
                    help="train+val only for a final model, after choices are made on val")
    ap.add_argument("--labels", default="corrected_v2", choices=("corrected_v2", "original"))
    ap.add_argument("--reference-dir", default=str(CORRECTED_V2))
    ap.add_argument("--policy", default="", help="AugmentationPolicy JSON (default policy if empty)")
    ap.add_argument("--copies", type=int, default=None, help="override copies_per_image")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--window", default="3,97", help="normalisation window for training and scoring")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    start = Path(args.start)
    if not start.is_file():
        print(f"start checkpoint not found: {start} (there is no default)")
        return 2
    destination = MODELS_DIR / "models" / args.name
    report_path = REPORTS_DIR / f"{args.name}.json"
    for existing in (destination, report_path):
        if existing.exists():
            print(f"refusing to overwrite {existing}; choose a new --name")
            return 2

    registry = load_registry(Path(args.registry))
    splits = load_splits(Path(args.splits))
    if splits["dataset_registry_sha256"] != registry["content_sha256"]:
        print("the splits were made from a different dataset registry")
        return 2
    split_names = ["train"] if args.train_on == "train" else ["train", "val"]
    records, dropped = select_images(registry, splits, split_names,
                                     keep_duplicates=args.keep_duplicates)

    policy = (AugmentationPolicy.from_dict(json.loads(Path(args.policy).read_text()))
              if args.policy else AugmentationPolicy())
    if args.no_augment:
        policy = AugmentationPolicy.from_dict({**policy.to_dict(), "copies_per_image": 0})
    elif args.copies is not None:
        policy = AugmentationPolicy.from_dict({**policy.to_dict(), "copies_per_image": args.copies})
    window = parse_window(args.window)
    reference_dir = Path(args.reference_dir)
    items = [(r["image_id"], load_image(ROOT / r["path"]).astype(np.float32),
              labels_for(r, args.labels, reference_dir)) for r in records]
    import cv2

    # Before the first cv2 call anywhere in the run, augmentations included:
    # OpenCV's worker pool has crashed the process under memory pressure
    # (training.augmentations._cv2).
    cv2.setNumThreads(0)
    images, masks, aug_records = build_training_set(items, policy)
    print(f"training on {', '.join(split_names)}: {len(records)} images "
          f"({len(dropped)} duplicate(s) dropped: {dropped}) -> {len(images)} with "
          f"{policy.copies_per_image} recorded copies each; labels {args.labels}")
    print(f"start {start} -> {destination}")
    trained_experiments = sorted({r["experiment_id"] for r in records})
    start_sha = sha256_file(start)
    # The validation score below is only a held-out figure if the start
    # checkpoint's lineage never saw the validation experiment. Every pre-v2
    # checkpoint was trained on all of KK1, so this is the common case.
    validation_provenance = held_out(start_sha, splits["splits"]["val"]["experiments"])
    if args.train_on == "train" and not validation_provenance["held_out"]:
        print(f"WARNING: the start checkpoint is not shown to be held out from val -- "
              f"{validation_provenance['reason']}. The validation score will be recorded "
              f"as not held out.")
    if args.dry_run:
        print("dry run: nothing trained or written")
        return 0

    version = metadata.version("cellpose")
    if not version.startswith("3."):
        print(f"this trains Cellpose 3 checkpoints; installed cellpose is {version}")
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from cellpose import models as cp, train as cp_train

    model = cp.CellposeModel(pretrained_model=str(start), gpu=False)
    # MKL-DNN tensors cannot be trained through ("does not support weight as an
    # MKLDNN tensor during training"); it is an inference optimisation.
    model.net.mkldnn = False
    destination.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cp_train.train_seg(
        model.net, train_data=images, train_labels=masks, channels=[0, 0],
        normalize=window, min_train_masks=0, rescale=False,
        learning_rate=args.learning_rate, n_epochs=args.epochs,
        batch_size=args.batch_size, weight_decay=args.weight_decay, SGD=False,
        save_path=str(MODELS_DIR), model_name=args.name, save_every=max(args.epochs, 1))
    minutes = (time.time() - started) / 60.0
    if not destination.exists():
        print(f"training finished but {destination} was not written")
        return 1
    out_sha = sha256_file(destination)

    validation = None
    if args.train_on == "train":
        from training.evaluate_segmentation import predict, score_variant

        val_records, _ = select_images(registry, splits, ["val"], keep_duplicates=True)
        val_items = []
        for r in val_records:
            path = ROOT / r["path"]
            val_items.append({"id": r["image_id"], "path": path, "experiment": r["experiment_id"],
                              "original": load_masks(path),
                              "corrected": labels_for(r, "corrected_v2", reference_dir)})
        predictions = predict(destination, val_items, window=window, diameter=None,
                              cellprob=0.0, flow=0.4, threads=args.threads)
        validation = score_variant(val_items, predictions)
        mark = "" if validation_provenance["held_out"] else "  (NOT held out)"
        for key in ("corrected_v2", "original"):
            row = validation[key]["at_iou_0.5"]
            print(f"val vs {key:12s}: F1 {row['f1']:.4f}  P {row['precision']:.4f}  "
                  f"R {row['recall']:.4f}  AP50:95 {validation[key]['ap_50_95']}{mark}")

    report = {
        "name": args.name,
        "architecture": "cellpose3",
        "cellpose_version": version,
        "torch_version": torch.__version__,
        "start_checkpoint": str(start), "start_sha256": start_sha,
        "checkpoint": str(destination), "checkpoint_sha256": out_sha,
        "dataset_registry": registry["version"],
        "dataset_registry_sha256": registry["content_sha256"],
        "splits": splits["version"], "trained_on": split_names,
        "trained_on_experiments": trained_experiments,
        "labels": args.labels, "reference_dir": reference_dir.as_posix(),
        "images": [r["image_id"] for r in records], "dropped_duplicates": dropped,
        "training_images": len(images),
        "settings": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                     "batch_size": args.batch_size, "weight_decay": args.weight_decay,
                     "normalize": window, "min_train_masks": 0, "rescale": False,
                     "channels": [0, 0], "SGD": False},
        "augmentation_policy": policy.to_dict(),
        "augmentation_records": [r.to_dict() for r in aug_records],
        "minutes": round(minutes, 1),
        "validation_reloaded_from_disk": validation,
        "validation_held_out": validation_provenance["held_out"] if validation else None,
        "validation_held_out_reason": validation_provenance["reason"] if validation else None,
        "command": sys.argv,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {destination} ({out_sha[:12]}) and {report_path}")

    if not args.no_register:
        from training.registry import load as load_models, find, register

        parent = find(load_models(), start_sha)
        register(args.name, destination, parent=parent["id"] if parent else start_sha,
                 training_report=report_path, dataset_registry=registry["version"],
                 dataset_registry_sha256=registry["content_sha256"],
                 splits=splits["version"], augmentation_policy=policy.name,
                 architecture="cellpose3", trained_on=split_names,
                 trained_on_experiments=trained_experiments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
