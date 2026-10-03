"""Make my own controlled problems on a REAL movie and test robustness.

Take a real movie whose cell counts are eye-verified (052924_t3_dual), then
inject known perturbations and re-run the full pipeline, measuring how detection
holds up:
  * additive Gaussian NOISE at rising levels (sensor noise);
  * STATIC cell-shaped artifacts (fixed debris) added to every frame -- a good
    system should not report MORE migrating cells than are really there.
Because I add the perturbations, I know the ground truth exactly.
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
import tifffile

sys.path.insert(0, "src")
from corridor.core.config import RunConfig, Scale                # noqa: E402
from corridor.core import model_registry as mr                   # noqa: E402
from corridor.core.segmentation import SegmentationService       # noqa: E402

MOVIE = "data/confinedmig_cellTrack/sample_data/052924_t3_dual.tif"
EXPECTED = [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1]   # eye-verified per-frame cells


def count_per_frame(masks):
    return [len([l for l in np.unique(masks[t]) if l]) for t in range(masks.shape[0])]


def score(counts, exp):
    n = min(len(counts), len(exp))
    tp = sum(min(counts[t], exp[t]) for t in range(n))
    over = sum(max(counts[t] - exp[t], 0) for t in range(n))
    gt = sum(exp[:n])
    rec = tp / max(gt, 1); prec = tp / max(tp + over, 1)
    return rec, prec


def run(img, label):
    seg = SegmentationService(RunConfig().segmentation, model=mr.resolve_model(),
                              scale=Scale.from_values(0.467, 20.0)).run_stack(img.astype(np.float32))
    c = count_per_frame(seg.masks)
    rec, prec = score(c, EXPECTED)
    print(f"  {label:>34}: recall {rec:.3f} prec {prec:.3f}  counts {c}")
    return rec, prec


def main():
    base = tifffile.imread(MOVIE).astype(np.float32)[:len(EXPECTED)]
    rng = np.random.default_rng(0)
    sd = base.std()
    print(f"{Path(MOVIE).name}: shape {base.shape}, intensity std {sd:.0f}")
    print("NOISE robustness (additive Gaussian, sigma as x the image std):")
    for f in (0.0, 0.25, 0.5, 1.0, 2.0):
        img = base + rng.normal(0, f * sd, base.shape)
        run(np.clip(img, 0, None), f"noise {f:.2f}x std")
    print("STATIC artifact robustness (fixed cell-shaped blobs in empty lanes):")
    art = base.copy()
    # add 3 static dark cell-shaped blobs at fixed positions (every frame)
    H, W = base.shape[1:]
    for (cx, cy) in [(20, 120), (45, 200), (70, 80)]:
        art[:, max(0, cy - 15):cy + 15, max(0, cx - 3):cx + 4] -= 0.4 * sd
    run(np.clip(art, 0, None), "3 static artifacts added")


if __name__ == "__main__":
    main()
