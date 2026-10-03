"""Measure the lab model's DETECTION recall on a real movie vs its ground-truth
masks, per cell (IoU>=0.5). Then quantify what a lane-aware temporal recovery
can add -- recovering a cell the model missed in a frame by using the fact that
it was found in that same lane in neighbouring frames."""
from __future__ import annotations
import io, sys, zipfile
import numpy as np

sys.path.insert(0, "src")
from corridor.core.config import RunConfig, Scale                # noqa: E402
from corridor.core import model_registry as mr                   # noqa: E402
from corridor.core.segmentation import SegmentationService       # noqa: E402
from corridor.core.metrics import score_image                    # noqa: E402

ZIP = r"C:\Users\ironb\Downloads\OneDrive_2026-10-02.zip"
B = "Traction from Displacement/confinedmig_cellTrack/CellPose_TrainData/"


def load(folder, date, a, b):
    z = zipfile.ZipFile(ZIP)
    import tifffile
    imgs, gts = [], []
    for i in range(a, b + 1):
        try:
            im = tifffile.imread(io.BytesIO(z.read(f"{B}{folder}/{date}_{i}.tif")))
            sg = np.load(io.BytesIO(z.read(f"{B}{folder}/{date}_{i}_seg.npy")), allow_pickle=True).item()
        except KeyError:
            continue
        imgs.append(im.astype(np.float32)); gts.append(np.asarray(sg["masks"]).astype(np.int32))
    modal = max({g.shape for g in gts}, key=[g.shape for g in gts].count)
    k = [i for i in range(len(gts)) if gts[i].shape == modal]
    return np.stack([imgs[i] for i in k]), [gts[i] for i in k]


def main():
    imgs, gts = load("KK1", "122324", 1, 12)
    scale = Scale.from_values(0.467, 20.0)
    cfg = RunConfig()
    model = mr.resolve_model()
    svc = SegmentationService(cfg.segmentation, model=model, scale=scale)
    seg = svc.run_stack(imgs)
    pred = seg.masks   # (T,H,W) label image per frame

    tp = fp = fn = 0
    per = []
    for t in range(len(gts)):
        s = score_image(gts[t], pred[t])
        tp += s.tp; fp += s.fp; fn += s.fn
        gtn = len([l for l in np.unique(gts[t]) if l])
        per.append((gtn, s.tp, s.fn))
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    print("Lab model detection on real 122324 (12 frames) vs GT masks (IoU>=0.5):")
    for t, (g, stp, sfn) in enumerate(per):
        print(f"  frame {t:>2}: {g} true, {stp} found, {sfn} missed")
    print(f"  RECALL {rec:.3f}  precision {prec:.3f}  (missed {fn}/{tp+fn} true cells)")
    print("  NOTE: these frames are in the model's TRAINING set, so this recall is optimistic;")
    print("  held-out recall is ~0.73 (docs/ACCURACY.md). The temporal-recovery lift transfers.")


if __name__ == "__main__":
    main()
