"""Measure global MCF association on real data vs eye-verified truth + physics."""
from __future__ import annotations
import sys
import numpy as np
import tifffile
sys.path.insert(0, "build/traj")
from mcf import mcf_tracks  # noqa: E402

EXPECTED_COUNTS = {
    "052924_t1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t3_dual": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}
EXPECTED_TRACKS = {"052924_t1": 1, "052924_t3_dual": 3}


def dets_from_masks(masks, nf):
    out = []  # (t,x,y)
    for t in range(nf):
        for lab in np.unique(masks[t]):
            if not lab:
                continue
            ys, xs = np.nonzero(masks[t] == lab)
            out.append((t, float(xs.mean()), float(ys.mean()), masks[t]==lab))
    return out


def present_sets(tracks, dets):
    sets = []
    for tr in tracks:
        ts = sorted(dets[i][0] for i in tr)
        sets.append(set(range(ts[0], ts[-1] + 1)))
    return sets


def score(sets, expected):
    n = len(expected); counts = [0] * n
    for ps in sets:
        for t in ps:
            if 0 <= t < n:
                counts[t] += 1
    tp = sum(min(counts[t], expected[t]) for t in range(n))
    over = sum(max(counts[t] - expected[t], 0) for t in range(n))
    gt = sum(expected)
    rec = tp / max(gt, 1); prec = tp / max(tp + over, 1)
    return rec, prec, 2 * rec * prec / max(rec + prec, 1e-9)


print("GLOBAL MCF on real data")
print(f"{'movie':>16} {'drop':>5} {'trk(exp)':>9} {'rec':>6} {'prec':>6} {'F1':>6}")
for movie, expected in EXPECTED_COUNTS.items():
    n = len(expected); et = EXPECTED_TRACKS[movie]
    images = tifffile.imread(f"data/confinedmig_cellTrack/sample_data/{movie}.tif").astype(np.float32)[:n]
    masks = np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"][:n]
    for drop in (0.0, 0.3, 0.5):
        recs, precs, f1s, ntr = [], [], [], []
        for seed in range(15):
            rng = np.random.default_rng(seed)
            m = masks.copy()
            if drop:
                for t in range(n):
                    for lab in np.unique(m[t]):
                        if lab and rng.random() < drop:
                            m[t][m[t] == lab] = 0
            dets = dets_from_masks(m, n)
            tracks = mcf_tracks(dets, images)
            sets = present_sets(tracks, dets)
            r, p, f = score(sets, expected)
            recs.append(r); precs.append(p); f1s.append(f); ntr.append(len(tracks))
        print(f"{movie:>16} {drop:>5.1f} {np.mean(ntr):>4.1f}/{et:<4} "
              f"{np.mean(recs):>6.3f} {np.mean(precs):>6.3f} {np.mean(f1s):>6.3f}")

# physics breadth on the wide fields
print("\nwide fields (physics): tracks and wall-crossings")
for movie in ("052924_1", "052924_2"):
    masks = np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"]
    images = tifffile.imread(f"data/confinedmig_cellTrack/sample_data/{movie}.tif").astype(np.float32)
    n = masks.shape[0]
    dets = dets_from_masks(masks, n)
    tracks = mcf_tracks(dets, images)
    xr = []
    for tr in tracks:
        xs = [dets[i][1] for i in tr]
        xr.append(max(xs) - min(xs) if len(xs) > 1 else 0.0)
    xr = np.array(xr)
    print(f"  {movie}: {len(tracks)} tracks, wall-crossings(x-range>40): {int((xr>40).sum())}, max x-range {xr.max():.1f}px")
