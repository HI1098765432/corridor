"""Diagnostic: raw Cellpose output versus the min_extent post-filter.

Answers one question only: when a detection disappears, was it never produced
by Cellpose, or was it produced and then removed by post-processing?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "confinedmig_cellTrack"
MODEL = DATA / "CellPose_TrainData" / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"
SAMPLES = DATA / "sample_data"


def instance_stats(mask: np.ndarray) -> list[dict]:
    out = []
    labels = np.unique(mask)
    for lab in labels[labels != 0]:
        ys, xs = np.where(mask == lab)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        out.append(
            {
                "label": int(lab),
                "area": int(ys.size),
                "h": h,
                "w": w,
                "extent": max(h, w),
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stacks", nargs="*", default=None)
    ap.add_argument("--min-extent", type=int, default=20)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "_diag"))
    args = ap.parse_args()

    stacks = args.stacks or [
        "052924_t3_dual.tif",
        "052924_t1.tif",
        "052924_t2_empty.tif",
    ]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from cellpose import models

    model = models.CellposeModel(pretrained_model=str(MODEL), gpu=False)
    print(f"cellpose model: diam_mean={float(model.diam_mean)} "
          f"diam_labels={float(model.diam_labels)}")
    print(f"params: cellprob={args.cellprob} flow={args.flow} "
          f"diameter={args.diameter} min_extent={args.min_extent}")

    report = {}
    for name in stacks:
        path = SAMPLES / name if not Path(name).exists() else Path(name)
        stack = tifffile.imread(path)
        if stack.ndim == 2:
            stack = stack[None]
        print("\n" + "=" * 78)
        print(f"{path.name}  shape={stack.shape}")
        print(f"{'frame':>5} {'raw':>4} {'kept':>5} {'removed':>8}  removed extents / areas")
        frames = []
        raw_masks = []
        for t in range(stack.shape[0]):
            res = model.eval(
                stack[t],
                channels=[0, 0],
                diameter=args.diameter,
                cellprob_threshold=args.cellprob,
                flow_threshold=args.flow,
            )
            mask = np.asarray(res[0]).astype(np.int32)
            raw_masks.append(mask)
            stats = instance_stats(mask)
            kept = [s for s in stats if s["extent"] >= args.min_extent]
            removed = [s for s in stats if s["extent"] < args.min_extent]
            frames.append(
                {
                    "frame": t,
                    "raw": len(stats),
                    "kept": len(kept),
                    "removed": len(removed),
                    "instances": stats,
                }
            )
            detail = ", ".join(f"ext={s['extent']} area={s['area']}" for s in removed[:6])
            print(f"{t:>5} {len(stats):>4} {len(kept):>5} {len(removed):>8}  {detail}")
        np.save(outdir / f"{path.stem}_rawmasks.npy", np.stack(raw_masks))
        report[path.name] = {
            "shape": list(stack.shape),
            "frames": frames,
            "raw_counts": [f["raw"] for f in frames],
            "kept_counts": [f["kept"] for f in frames],
        }
        print(f"  RAW  : {[f['raw'] for f in frames]}")
        print(f"  KEPT : {[f['kept'] for f in frames]}")

    (outdir / "segmentation_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {outdir / 'segmentation_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
