"""Is the cell present in t3_dual frames 5-9 with inverted phase contrast?

If Cellpose finds objects in the inverted image at the same location, the cell
is present but its appearance left the model's training distribution. If it
finds nothing either way, the object is simply not there.
"""
from pathlib import Path
import numpy as np, tifffile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "confinedmig_cellTrack"
MODEL = D / "CellPose_TrainData/KK1KK2_combiModel/models/cyto2_phase_microfluidic_KK1KK2_combi"

from cellpose import models
m = models.CellposeModel(pretrained_model=str(MODEL), gpu=False)
stack = tifffile.imread(D / "sample_data/052924_t3_dual.tif")

for name, xform in (("original", lambda f: f),
                    ("inverted", lambda f: (f.max() - f).astype(f.dtype))):
    counts, where = [], []
    for t in range(stack.shape[0]):
        mask = np.asarray(m.eval(xform(stack[t]), channels=[0, 0], diameter=None,
                                 cellprob_threshold=0.0, flow_threshold=0.4)[0])
        counts.append(int(mask.max()))
        ys = [float(np.where(mask == l)[0].mean()) for l in range(1, int(mask.max()) + 1)]
        where.append([round(v) for v in ys])
    print(f"{name:9s} counts={counts}")
    print(f"{'':9s} centroid_y={where}")
