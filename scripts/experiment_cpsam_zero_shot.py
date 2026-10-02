"""Experiment B: Cellpose-SAM (``cpsam_v2``) with no training, on the labelled stills.

The directive orders the Cellpose-SAM work as four separable experiments so
that a gain can be attributed: A, the existing custom CP3 model (frozen
baseline); B, the foundation model with no training at all; C, the
foundation model fine-tuned on our data; D, any architectural change. This is
B. It answers one question: how much of our held-out problem does a model
trained on 22,826 *other* images already solve?

Nothing here has seen a KK1 or KK2 image, so both groups are held out. The
fair comparison for KK2 is the published held-out figure for the CP3 work,
which was trained on KK1 (0.4821 at Cellpose's default normalisation; 0.7273
for the round-1 contrast-augmented checkpoint read through (3, 97) at d=36).

Runs ONLY in the separate Cellpose 4 environment, never the app's::

    OMP_NUM_THREADS=4 ./.venv-cp4/Scripts/python.exe scripts/experiment_cpsam_zero_shot.py

The weights are passed as an explicit, hash-checked path. Cellpose 4 silently
substitutes its default model -- and downloads 1.2 GB -- for a name or path it
does not recognise, so a bare name is never used here either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.metrics import Score, score_image  # noqa: E402  (numpy-only module)

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
WEIGHTS = ROOT / "build" / "cp4_models" / "cpsam_v2"
#: SHA-256 of huggingface.co/mouseland/cellpose-sam cpsam_v2 (LFS record).
WEIGHTS_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def labelled(group: str) -> list[Path]:
    return [p for p in sorted((TRAIN / group).glob("*.tif"))
            if p.with_name(p.stem + "_seg.npy").exists()]


def load(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if image.ndim == 3:
        image = image[..., 0] if image.shape[-1] in (3, 4) else image[0]
    return image.astype(np.float32)


def truth_of(path: Path) -> np.ndarray:
    seg = np.load(path.with_name(path.stem + "_seg.npy"), allow_pickle=True).item()
    return np.asarray(seg["masks"]).astype(np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="KK2,KK1")
    ap.add_argument("--windows", default="default,3:97",
                    help="comma list; 'default' = Cellpose's own (1, 99)")
    ap.add_argument("--diameters", default="none,30",
                    help="comma list; 'none' = no rescaling")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N images per group (timing probe)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "cpsam_zero_shot.json"))
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}; run build/logs/setup_cp4.sh")
    digest = sha256(WEIGHTS)
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"{WEIGHTS} has sha256 {digest}, expected {WEIGHTS_SHA256}")

    import torch
    torch.set_num_threads(args.threads)
    import cellpose
    from cellpose import models

    model = models.CellposeModel(gpu=False, pretrained_model=str(WEIGHTS))
    version = getattr(cellpose, "version", None) or getattr(cellpose, "__version__", "unknown")

    windows = []
    for w in args.windows.split(","):
        windows.append(("default", True) if w == "default"
                       else (w, {"percentile": tuple(float(v) for v in w.split(":"))}))
    diameters = [None if d == "none" else float(d) for d in args.diameters.split(",")]

    rows = []
    for group in args.groups.split(","):
        paths = labelled(group)
        if args.limit:
            paths = paths[: args.limit]
        images = {p: load(p) for p in paths}
        truths = {p: truth_of(p) for p in paths}
        for wname, normalize in windows:
            for diameter in diameters:
                started = time.time()
                total = Score(0, 0, 0, [])
                for p in paths:
                    masks = model.eval(images[p], normalize=normalize, diameter=diameter,
                                       cellprob_threshold=0.0, flow_threshold=0.4)[0]
                    total = total + score_image(truths[p], np.asarray(masks).astype(np.int32))
                seconds = (time.time() - started) / max(len(paths), 1)
                row = {"group": group, "window": wname, "diameter": diameter,
                       "n_images": len(paths), "seconds_per_image": round(seconds, 1),
                       **total.to_row()}
                rows.append(row)
                print(json.dumps(row), flush=True)

    report = {
        "experiment": "B: cpsam_v2 zero-shot (no training on any KK image)",
        "weights": "huggingface.co/mouseland/cellpose-sam cpsam_v2",
        "weights_sha256": digest,
        "cellpose_version": str(version),
        "torch": torch.__version__,
        "matcher": "corridor.core.metrics (count-first, IoU >= 0.5)",
        "reference": "original human labels (_seg.npy)",
        "comparison": {
            "cp3_kk1_model_on_kk2_default_normalisation": 0.4821,
            "cp3_round1_contrast_invariant_on_kk2_window_3_97_d36": 0.7273,
        },
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
