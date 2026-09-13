"""Cellpose v3 parameter sensitivity on the supplied narrow-channel stacks.

Purpose: decide whether missing detections are a thresholding artefact that
reasonable parameters recover, or a genuine limit of the trained model.
Retraining is only justified if no reasonable parameter set helps.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "confinedmig_cellTrack"
MODEL = DATA / "CellPose_TrainData" / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi"
SAMPLES = DATA / "sample_data"

STACKS = ["052924_t3_dual.tif", "052924_t1.tif", "052924_t2_empty.tif"]
CELLPROB = [0.0, -1.0, -2.0, -3.0, -4.0, -6.0]
FLOW = [0.4, 0.8]
DIAMETER = [None, 31.9]


def main() -> None:
    from cellpose import models

    model = models.CellposeModel(pretrained_model=str(MODEL), gpu=False)
    results = {}
    for name in STACKS:
        stack = tifffile.imread(SAMPLES / name)
        if stack.ndim == 2:
            stack = stack[None]
        print("=" * 88)
        print(f"{name}  {stack.shape}")
        print(f"{'cellprob':>9} {'flow':>5} {'diam':>6} | per-frame instance counts")
        results[name] = {}
        for cp, fl, dm in itertools.product(CELLPROB, FLOW, DIAMETER):
            counts = []
            for t in range(stack.shape[0]):
                res = model.eval(
                    stack[t],
                    channels=[0, 0],
                    diameter=dm,
                    cellprob_threshold=cp,
                    flow_threshold=fl,
                )
                counts.append(int(np.asarray(res[0]).max()))
            key = f"cp{cp}_fl{fl}_d{dm}"
            results[name][key] = counts
            dmtxt = "auto" if dm is None else f"{dm:g}"
            print(f"{cp:>9g} {fl:>5g} {dmtxt:>6} | {counts}  total={sum(counts)}")

    out = ROOT / "data" / "_diag" / "param_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
