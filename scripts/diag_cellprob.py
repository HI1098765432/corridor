"""Why are frames 5-9 of 052924_t3_dual empty?

Counting masks only tells you that Cellpose produced nothing. It does not say
whether Cellpose *saw* nothing. This reads the cell-probability field itself,
which is what the mask threshold is applied to, and asks a sharper question:
inside the confinement channel, how close did the network come to calling a
cell in the frames that came back empty, compared with the frames that did not?

If the probability inside the channel collapses in those frames, the object is
absent from the network's view. If it stays high and only the mask assembly
fails, the threshold is the problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "confinedmig_cellTrack"
MODEL = (
    DATA / "CellPose_TrainData" / "KK1KK2_combiModel" / "models"
    / "cyto2_phase_microfluidic_KK1KK2_combi"
)
STACK = DATA / "sample_data" / "052924_t3_dual.tif"
OUT = ROOT / "data" / "_diag"


def main() -> int:
    from cellpose import models

    stack = tifffile.imread(STACK)
    model = models.CellposeModel(pretrained_model=str(MODEL), gpu=False)

    # The channel occupies a narrow band of columns; measure inside it only, so
    # that bright device structure elsewhere cannot mask a collapse.
    projection = np.median(stack.astype(np.float32), axis=0)
    column_profile = projection.mean(axis=0)
    centre = int(np.argmax(column_profile - np.median(column_profile)))
    lo, hi = max(0, centre - 12), min(stack.shape[2], centre + 13)
    print(f"channel columns {lo}-{hi} (centre {centre})")
    print()
    print(f"{'frame':>5} {'masks':>6} {'p_max':>8} {'p_p99':>8} {'px>0':>7} {'px>-1':>7} "
          f"{'px>-2':>7}  where p is highest")
    rows = []
    for t in range(stack.shape[0]):
        masks, flows, _ = model.eval(
            stack[t], channels=[0, 0], diameter=None,
            cellprob_threshold=0.0, flow_threshold=0.4,
        )
        # flows[2] is the cell-probability map the mask threshold is applied to.
        prob = np.asarray(flows[2], dtype=np.float32)
        band = prob[:, lo:hi]
        best_row = int(np.argmax(band.max(axis=1)))
        row = {
            "frame": t,
            "n_masks": int(np.asarray(masks).max()),
            "prob_max": float(band.max()),
            "prob_p99": float(np.percentile(band, 99)),
            "px_above_0": int((band > 0.0).sum()),
            "px_above_minus1": int((band > -1.0).sum()),
            "px_above_minus2": int((band > -2.0).sum()),
            "peak_row": best_row,
        }
        rows.append(row)
        print(
            f"{t:>5} {row['n_masks']:>6} {row['prob_max']:>8.2f} {row['prob_p99']:>8.2f} "
            f"{row['px_above_0']:>7} {row['px_above_minus1']:>7} {row['px_above_minus2']:>7}"
            f"  y={best_row}"
        )

    empty = [r for r in rows if r["n_masks"] == 0]
    found = [r for r in rows if r["n_masks"] > 0]
    print()
    print("summary inside the channel:")
    for label, group in (("frames WITH a mask", found), ("frames WITHOUT a mask", empty)):
        if not group:
            continue
        print(
            f"  {label:24s} n={len(group):>2}  "
            f"prob_max {np.mean([r['prob_max'] for r in group]):+6.2f} mean, "
            f"{max(r['prob_max'] for r in group):+6.2f} best   "
            f"px>0 {np.mean([r['px_above_0'] for r in group]):7.1f} mean"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cellprob_diagnostic.json").write_text(
        json.dumps({"channel_columns": [lo, hi], "frames": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'cellprob_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
