"""Model-free, lane-aware, background-subtraction cell detector.

Temporal median = the static scene (walls, debris, texture). A cell MOVES, so
after subtracting the median it is a clear dark-residual blob. Scanning each
detected lane for such blobs finds cells directly -- including the faint ones the
learned detector misses -- because it keys on motion, not on learned appearance.
Measured on real 122324 vs GT (ALL cells), as a detector in its own right and
unioned with the lab model.
"""
from __future__ import annotations
import io, sys, zipfile
import numpy as np
from scipy import ndimage

sys.path.insert(0, "src")
from corridor.core.config import RunConfig, Scale                # noqa: E402
from corridor.core import model_registry as mr                   # noqa: E402
from corridor.core.segmentation import SegmentationService       # noqa: E402

B = "Traction from Displacement/confinedmig_cellTrack/CellPose_TrainData/"
TOL = 14.0


def cents(m):
    return [(float(np.nonzero(m == l)[1].mean()), float(np.nonzero(m == l)[0].mean()))
            for l in np.unique(m) if l]


def recall_prec(pc, gc):
    tp = gt = fp = 0
    for t in range(len(gc)):
        gt += len(gc[t]); used = set()
        for gx, gy in gc[t]:
            for j, (px, py) in enumerate(pc[t]):
                if j not in used and np.hypot(px - gx, py - gy) <= TOL:
                    tp += 1; used.add(j); break
        fp += len(pc[t]) - len(used)
    rec = tp / max(gt, 1); prec = tp / max(tp + fp, 1)
    return rec, prec, 2 * rec * prec / max(rec + prec, 1e-9), tp, gt


def load():
    import tifffile
    z = zipfile.ZipFile(r"C:\Users\ironb\Downloads\OneDrive_2026-10-02.zip")
    imgs, gts = [], []
    for i in range(1, 13):
        try:
            im = tifffile.imread(io.BytesIO(z.read(f"{B}KK1/122324_{i}.tif")))
            sg = np.load(io.BytesIO(z.read(f"{B}KK1/122324_{i}_seg.npy")), allow_pickle=True).item()
        except KeyError:
            continue
        imgs.append(im.astype(np.float32)); gts.append(np.asarray(sg["masks"]).astype(np.int32))
    modal = max({g.shape for g in gts}, key=[g.shape for g in gts].count)
    k = [i for i in range(len(gts)) if gts[i].shape == modal]
    return [imgs[i] for i in k], [gts[i] for i in k]


def lanes_of(pc, gap=15):
    allx = sorted(x for fr in pc for x, y in fr); lanes, cur = [], [allx[0]]
    for x in allx[1:]:
        (lanes.append(np.mean(cur)), cur.clear(), cur.append(x)) if x - cur[-1] > gap else cur.append(x)
    lanes.append(np.mean(cur)); return lanes


def detect(imgs, lanes, xhalf=8, sigma=1.5, k=4.0, min_area=12):
    """Per lane, per frame: dark-residual blobs above (lane-empty-median +
    k*MAD) are cells. Returns per-frame list of (x,y)."""
    stack = np.stack(imgs); bg = np.median(stack, axis=0)
    res = [ndimage.gaussian_filter(np.clip(bg - im, 0, None), sigma) for im in imgs]
    H, W = imgs[0].shape
    out = [[] for _ in imgs]
    for lx in lanes:
        xi = int(round(lx)); x0, x1 = max(0, xi - xhalf), min(W, xi + xhalf + 1)
        strips = np.stack([r[:, x0:x1] for r in res])        # (T, H, w)
        # noise level of THIS lane from its own residual distribution
        med = np.median(strips); mad = np.median(np.abs(strips - med)) + 1e-6
        thr = med + k * 1.4826 * mad
        for t in range(len(imgs)):
            s = strips[t]
            lbl, n = ndimage.label(s >= thr)
            for comp in range(1, n + 1):
                m = lbl == comp
                if m.sum() < min_area:
                    continue
                cyy, cxx = ndimage.center_of_mass(s, lbl, comp)
                out[t].append((x0 + cxx, cyy))
    return out


def union(a, b, tol=TOL):
    out = []
    for t in range(len(a)):
        merged = list(a[t])
        for (x, y) in b[t]:
            if not any(np.hypot(x - mx, y - my) <= tol for mx, my in merged):
                merged.append((x, y))
        out.append(merged)
    return out


def main():
    imgs, gts = load()
    gc = [cents(g) for g in gts]
    seg = SegmentationService(RunConfig().segmentation, model=mr.resolve_model(),
                              scale=Scale.from_values(0.467, 20.0)).run_stack(np.stack(imgs))
    pc = [cents(seg.masks[t]) for t in range(len(gts))]
    lanes = lanes_of(pc)
    r0, p0, f0, tp0, gt = recall_prec(pc, gc)
    print(f"lab model:            recall {r0:.3f} prec {p0:.3f} F1 {f0:.3f} ({tp0}/{gt})")
    for k in (3.0, 4.0, 5.0, 6.0):
        bd = detect(imgs, lanes, k=k)
        rb, pb, fb, tpb, _ = recall_prec(bd, gc)
        u = union(pc, bd)
        ru, pu, fu, tpu, _ = recall_prec(u, gc)
        print(f"bg-detector k={k}: alone recall {rb:.3f} prec {pb:.3f} | UNION recall {ru:.3f} prec {pu:.3f} F1 {fu:.3f} ({tpu}/{gt})")


if __name__ == "__main__":
    main()
