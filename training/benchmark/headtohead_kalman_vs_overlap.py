"""Head-to-head on the REAL movies: the shipped Kalman tracker
(``track_detections``) vs the overlap reconstructor (``reconstruct_tracks``),
both fed the SAME detections built from the eye-verified label masks and scored
against the eye-verified per-frame counts. This is the measured evidence for
whether ``tracking.reconstructor == "overlap"`` should be the production default
on confined, elongating cells -- not a memory of an earlier run.
"""
from __future__ import annotations
import sys
import numpy as np
import tifffile

sys.path.insert(0, "src")
from corridor.core.config import Scale, TrackingConfig           # noqa: E402
from corridor.core.detections import Detection                   # noqa: E402
from corridor.core.tracking import track_detections              # noqa: E402
from corridor.engine.reconstruct import reconstruct_tracks       # noqa: E402

EXPECTED_COUNTS = {
    "052924_t1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t3_dual": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}


def load_masks(movie, nframes):
    masks = np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"]
    return masks[:nframes]


def detections_from_masks(masks, drop=0.0, seed=0):
    """One Detection per labelled instance per frame, with the fields both
    trackers read (centroid, bbox, axes, eccentricity, mask_crop, channel).
    ``drop`` randomly removes that fraction of detections (detector dropout)."""
    rng = np.random.default_rng(seed)
    dets = []
    label = 1
    for t, mt in enumerate(masks):
        for lab in np.unique(mt):
            if not lab:
                continue
            if drop and rng.random() < drop:
                continue
            m = mt == lab
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            minr, minc, maxr, maxc = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            major = float(maxr - minr)
            minor = float(max(maxc - minc, 1))
            ecc = float(np.sqrt(max(1.0 - (minor / max(major, 1)) ** 2, 0.0)))
            dets.append(Detection(
                frame=int(t), label=int(label),
                x=float(xs.mean()), y=float(ys.mean()),
                area_px=float(m.sum()),
                bbox=(int(minr), int(minc), int(maxr), int(maxc)),
                extent_px=int(max(major, minor)), eccentricity=ecc,
                orientation_rad=0.0, major_axis_px=major, minor_axis_px=minor,
                solidity=1.0, touches_border=False, channel=0,
                mask_crop=m[minr:maxr, minc:maxc].copy()))
            label += 1
    return dets


def score_counts(tracks, expected):
    """Count-based F1 against eye-verified per-frame counts: recall rewards
    covering the frames a cell is truly present; precision penalises tracks
    that put a cell where there is none (the elongation-induced phantom)."""
    n = len(expected)
    counts = [0] * n
    for tr in tracks:
        for o in tr.observations:
            if 0 <= o.frame < n:
                counts[o.frame] += 1
    tp = sum(min(counts[t], expected[t]) for t in range(n))
    over = sum(max(counts[t] - expected[t], 0) for t in range(n))
    gt = sum(expected)
    rec = tp / max(gt, 1)
    prec = tp / max(tp + over, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return rec, prec, f1


def main():
    scale = Scale.from_values(0.467, 20.0)   # KK2 sample movies
    cfg = TrackingConfig()
    print("HEAD-TO-HEAD on real movies (same detections, scored vs eye-verified counts)")
    print(f"{'movie':>15} {'drop':>5} {'backend':>8} {'rec':>6} {'prec':>6} {'F1':>6}")
    # drop 0.0 is deterministic (1 seed); degraded regimes averaged over seeds.
    for movie, expected in EXPECTED_COUNTS.items():
        n = len(expected)
        masks = load_masks(movie, n)
        shape = masks.shape[1:]
        for drop in (0.0, 0.3, 0.5):
            seeds = range(1) if drop == 0.0 else range(20)
            fk, fo = [], []
            for s in seeds:
                dets = detections_from_masks(masks, drop=drop, seed=s)
                tk, _ = track_detections(dets, n, scale, cfg)
                fk.append(score_counts(list(tk), expected))
                to, _ = reconstruct_tracks(dets, shape, n, scale, cfg)
                fo.append(score_counts(list(to), expected))
            rk, pk, f_k = np.mean(fk, axis=0)
            ro, po, f_o = np.mean(fo, axis=0)
            print(f"{movie:>15} {drop:>5.1f} {'kalman':>8} {rk:>6.3f} {pk:>6.3f} {f_k:>6.3f}")
            print(f"{movie:>15} {drop:>5.1f} {'overlap':>8} {ro:>6.3f} {po:>6.3f} {f_o:>6.3f}")


if __name__ == "__main__":
    main()
