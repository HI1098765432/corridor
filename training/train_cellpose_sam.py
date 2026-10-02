"""Fine-tune Cellpose-SAM (Cellpose 4, ``cpsam_v2``) on a locked split -- Experiment C.

Runs only in the separate ``.venv-cp4`` (cellpose 4.2.1.1, torch 2.6.0 CPU),
never in the application's environment:

    .venv-cp4/Scripts/python.exe -m training.train_cellpose_sam \\
        --start build/cp4_models/cpsam_v2 --name cpsam_v2_kk1_ft1 --smoke

Defaults follow the maintainers' own fine-tuning recipe
(``build/maps/external-cellpose4.md`` section 6; ``paper/cpsam/train_subsets.py``):
AdamW, learning rate 1e-5, weight decay 0.1, batch size 1, 256 px tiles (the
only size SAM accepts), 100 epochs, fp32 weights (``use_bfloat16=False``;
cellpose >= 4.0.9 trains in fp32 anyway), ``rescale=False`` with Cellpose's
own 0.75-1.25 scale jitter, ``min_train_masks=0``. The public CP4 ``train_seg``
adds rotation, flip and scale only; the recorded policy of
:mod:`training.augmentations` adds the photometric and degradation copies.

Guards, because Cellpose 4 fails quietly in exactly the ways that matter:

- **An explicit start path, hashed.** Given a missing path or a model name,
  Cellpose 4 logs a warning and loads -- downloading 1.2 GB if needed -- its
  default ``cpsam_v2``. A file named ``cpsam_v2`` must also match the Hugging
  Face LFS SHA-256.
- **cellpose >= 4.0.8.** 4.0.1-4.0.7 load a Cellpose 3 file into a randomly
  initialised ViT without an error.
- **The checkpoint is never overwritten**, and is saved under
  ``build/models/models/<name>`` like the Cellpose 3 runs.

``--smoke`` is the feasibility probe the contract asks for before any full CPU
run is attempted: 2 epochs on 4 training images, each cut to one 256x256 tile
around its labelled cells, no copies. It measures seconds per epoch; whether a
full run is feasible is then a number, not an assumption. ``--dry-run`` builds
the set and imports neither torch nor cellpose.
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
from training.splits import SPLITS_PATH, load_splits
from training.train_cellpose3 import (
    CORRECTED_V2,
    MODELS_DIR,
    REPORTS_DIR,
    labels_for,
    parse_window,
    select_images,
)

from corridor.learn.sequences import load_image  # noqa: E402

#: huggingface.co/mouseland/cellpose-sam, file cpsam_v2 (LFS record).
CPSAM_V2_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
TILE_PX = 256
SMOKE_EPOCHS = 2
SMOKE_IMAGES = 4


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split(".")[:3] if p.isdigit())


def tile_around_cells(image: np.ndarray, labels: np.ndarray, size: int = TILE_PX):
    """One size x size tile centred on the labelled cells (padded if the image is smaller)."""
    h, w = labels.shape
    ys, xs = np.nonzero(labels)
    cy, cx = (int(ys.mean()), int(xs.mean())) if ys.size else (h // 2, w // 2)
    y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    img = image[y0:y0 + size, x0:x0 + size]
    lab = labels[y0:y0 + size, x0:x0 + size]
    pad = ((0, size - img.shape[0]), (0, size - img.shape[1]))
    return (np.pad(img, pad, mode="reflect") if any(p[1] for p in pad) else img,
            np.pad(lab, pad) if any(p[1] for p in pad) else lab)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True, help="explicit cpsam_v2 (or other CP4) weights path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", default=str(SPLITS_PATH))
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--labels", default="corrected_v2", choices=("corrected_v2", "original"))
    ap.add_argument("--reference-dir", default=str(CORRECTED_V2))
    ap.add_argument("--policy", default="")
    ap.add_argument("--copies", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--window", default="default",
                    help="normalisation percentile window 'lo,hi', or 'default' (1-99)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_EPOCHS} epochs, {SMOKE_IMAGES} images, one {TILE_PX}px tile each")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    start = Path(args.start)
    if not start.is_file():
        print(f"start weights not found: {start}. Cellpose 4 would silently load (and "
              f"download) its default model instead, so this stops here.")
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
    records, dropped = select_images(registry, splits, ["train"])
    policy = (AugmentationPolicy.from_dict(json.loads(Path(args.policy).read_text()))
              if args.policy else AugmentationPolicy())
    epochs = args.epochs
    if args.smoke:
        epochs = SMOKE_EPOCHS
        records = [r for r in records if r["n_instances"] > 0][:SMOKE_IMAGES]
        policy = AugmentationPolicy.from_dict({**policy.to_dict(), "copies_per_image": 0})
    elif args.copies is not None:
        policy = AugmentationPolicy.from_dict({**policy.to_dict(), "copies_per_image": args.copies})
    reference_dir = Path(args.reference_dir)
    import cv2

    # Before the first cv2 call of the run (augmentations, then Cellpose's own
    # rotate-and-resize): see training.augmentations._cv2.
    cv2.setNumThreads(0)
    items = []
    for r in records:
        image = load_image(ROOT / r["path"]).astype(np.float32)
        labels = labels_for(r, args.labels, reference_dir)
        if args.smoke:
            image, labels = tile_around_cells(image, labels)
        items.append((r["image_id"], image, labels))
    images, masks, aug_records = build_training_set(items, policy)
    window = parse_window(args.window)
    normalize = {"normalize": True, **window} if isinstance(window, dict) else True
    print(f"{'SMOKE: ' if args.smoke else ''}{len(records)} images -> {len(images)} "
          f"training images, {epochs} epochs, lr {args.learning_rate}, wd {args.weight_decay}, "
          f"batch {args.batch_size}, bsize {TILE_PX}, labels {args.labels}")
    if args.dry_run:
        print("dry run: nothing trained or written")
        return 0

    version = metadata.version("cellpose")
    if _version_tuple(version) < (4, 0, 8):
        print(f"cellpose {version}: Cellpose-SAM fine-tuning needs >= 4.0.8 (earlier 4.x "
              f"load incompatible weights without an error)")
        return 2
    start_sha = sha256_file(start)
    if start.name == "cpsam_v2" and start_sha != CPSAM_V2_SHA256:
        print(f"{start} does not match the published cpsam_v2 SHA-256")
        return 2

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from cellpose import models as cp, train as cp_train

    model = cp.CellposeModel(gpu=False, pretrained_model=str(start), use_bfloat16=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _, train_losses, _ = cp_train.train_seg(
        model.net, train_data=images, train_labels=masks, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, SGD=False,
        n_epochs=epochs, normalize=normalize, rescale=False, bsize=TILE_PX,
        min_train_masks=0, save_path=str(MODELS_DIR), model_name=args.name,
        save_every=max(epochs, 1))
    seconds = time.time() - started
    if not destination.exists():
        print(f"training finished but {destination} was not written")
        return 1
    out_sha = sha256_file(destination)

    report = {
        "name": args.name, "architecture": "cellpose4-cpsam",
        "cellpose_version": version, "torch_version": torch.__version__,
        "smoke": args.smoke,
        "start_checkpoint": str(start), "start_sha256": start_sha,
        "checkpoint": str(destination), "checkpoint_sha256": out_sha,
        "dataset_registry": registry["version"],
        "dataset_registry_sha256": registry["content_sha256"],
        "splits": splits["version"], "trained_on": ["train"],
        "trained_on_experiments": sorted({r["experiment_id"] for r in records}),
        "labels": args.labels,
        "images": [r["image_id"] for r in records], "dropped_duplicates": dropped,
        "training_images": len(images),
        "settings": {"epochs": epochs, "learning_rate": args.learning_rate,
                     "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                     "bsize": TILE_PX, "normalize": normalize, "rescale": False,
                     "min_train_masks": 0, "use_bfloat16": False,
                     "tiles": f"one {TILE_PX}px tile per image" if args.smoke else "full images"},
        "augmentation_policy": policy.to_dict(),
        "augmentation_records": [r.to_dict() for r in aug_records],
        "seconds": round(seconds, 1),
        "seconds_per_epoch": round(seconds / max(epochs, 1), 1),
        "train_losses": [float(v) for v in np.asarray(train_losses).ravel()],
        "command": sys.argv,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{seconds / 60:.1f} min ({report['seconds_per_epoch']} s/epoch); wrote "
          f"{destination} and {report_path}")
    if not args.no_register:
        from training.registry import register

        register(args.name, destination, parent=start_sha, training_report=report_path,
                 dataset_registry=registry["version"],
                 dataset_registry_sha256=registry["content_sha256"], splits=splits["version"],
                 augmentation_policy=policy.name, architecture="cellpose4-cpsam",
                 trained_on=["train"], trained_on_experiments=report["trained_on_experiments"],
                 notes="smoke run" if args.smoke else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
