"""A model-free cell detector ("the model can be zero"), measured against the
71 human-labelled stills -- real ground truth, real sample size.

Deterministic classical CV + the confined-cell prior: subtract a large-scale
background, enhance elongated bright ridges (Sato tubeness from the Hessian),
threshold, and keep components with cell-like size and elongation. No training,
no network. Scored per image with corridor.core.metrics against the human masks,
head-to-head with the numbers Cellpose gets on the same data.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import sato, threshold_otsu
from skimage.measure import label, regionprops

sys.path.insert(0, "src")
from corridor.core.metrics import score_image  # noqa: E402

ROOT = Path(".")
TRAIN = ROOT / "data/confinedmig_cellTrack/CellPose_TrainData"


def labelled(group):
    out = []
    for p in sorted((TRAIN / group).glob("*.tif")):
        seg = p.with_name(p.stem + "_seg.npy")
        if seg.exists():
            out.append(p)
    return out


def load_img(p):
    im = tifffile.imread(p).astype(np.float32)
    if im.ndim == 3:
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    return im


def truth(p):
    seg = np.load(p.with_name(p.stem + "_seg.npy"), allow_pickle=True).item()
    return np.asarray(seg["masks"]).astype(np.int32)


def detect(img, min_area=200, max_area=6000, min_ecc=0.9):
    """Model-free detection. Returns an int32 label image."""
    H, W = img.shape
    lo, hi = np.percentile(img, [1, 99])
    g = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    bg = ndimage.gaussian_filter(g, sigma=25)
    flat = np.clip(g - bg, 0, None)
    # walls: full-height vertical lines -> columns that are bright over most rows.
    col_profile = (flat > 0.05).mean(axis=0)                 # fraction of rows lit per column
    wall_cols = ndimage.binary_dilation(col_profile > 0.6, iterations=3)
    ridge = sato(flat, sigmas=range(1, 6), black_ridges=False)
    if ridge.max() <= 0:
        return np.zeros(img.shape, np.int32)
    ridge = ridge / ridge.max()
    try:
        thr = max(threshold_otsu(ridge[ridge > 0.02]), 0.08)
    except ValueError:
        thr = 0.1
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(ridge > thr, iterations=1))
    mask[:, wall_cols] = False                               # erase wall columns
    lbl = label(mask)
    out = np.zeros(img.shape, np.int32)
    nxt = 1
    for r in regionprops(lbl):
        minr, minc, maxr, maxc = r.bbox
        width = maxc - minc
        height = maxr - minr
        if not (min_area <= r.area <= max_area and r.eccentricity >= min_ecc):
            continue
        if not (4 <= width <= 22):                           # cell width ~11 px
            continue
        if height >= 0.75 * H:                               # a wall spans the frame
            continue
        if r.solidity < 0.6:
            continue
        out[lbl == r.label] = nxt
        nxt += 1
    return out


if __name__ == "__main__":
    print("MODEL-FREE detector vs 71 human-labelled stills (corridor.core.metrics, IoU>=0.5)")
    print(f"{'group':>6} {'images':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'prec':>6} {'rec':>6} {'F1':>6}")
    for group in ("KK1", "KK2"):
        paths = labelled(group)
        tp = fp = fn = 0
        for p in paths:
            pred = detect(load_img(p))
            s = score_image(truth(p), pred)
            tp += s.tp; fp += s.fp; fn += s.fn
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"{group:>6} {len(paths):>7} {tp:>5} {fp:>5} {fn:>5} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print("\nReference (Cellpose, same data): in-distribution F1 0.839; held-out KK2 ~0.73 (docs/ACCURACY.md)")
