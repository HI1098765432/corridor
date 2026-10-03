"""GENUINE benchmark, v2: link REAL cell masks by OVERLAP (IoU), not centroid.

The genuine real-data test (real_bench.py) revealed that confined cells
elongate, so their centroids jump 30-49 px frame-to-frame even when they are
one cell. Overlap is the stable feature. This links the real per-frame masks
by IoU across small temporal gaps, completes, and scores against the
developer's eye-verified per-frame counts. Still real images, real masks,
human truth.

Stress test: drop a fraction of real frames' detections and check overlap-gap
linking + completion rebuild the verified track (the recovery claim on real
data).
"""
from __future__ import annotations
import numpy as np

EXPECTED_COUNTS = {
    "052924_t1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t3_dual": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}
EXPECTED_TRACKS = {"052924_t1": 1, "052924_t3_dual": 3}


def load_masks(movie):
    return np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"]


def frame_objects(mask_t):
    """{label: boolean mask} for one frame."""
    out = {}
    for lab in np.unique(mask_t):
        if lab:
            out[int(lab)] = mask_t == lab
    return out


def overlap(a, b):
    """Elongation-robust overlap: max(IoU, containment). When a confined cell
    doubles in length its IoU with the previous frame drops, but the shorter
    mask stays *contained* in the longer one, so intersection/min-area stays
    high. Centroid distance sees a 49 px jump; this sees the same cell."""
    inter = np.count_nonzero(a & b)
    if not inter:
        return 0.0
    union = np.count_nonzero(a | b)
    min_area = min(np.count_nonzero(a), np.count_nonzero(b))
    return max(inter / union, inter / max(min_area, 1))


def link_by_overlap(objects_per_frame, nframes, max_gap=3, min_iou=0.1, min_support=1):
    """Greedy-but-gap-aware overlap linking. Each object node used once; link to
    the best-IoU object within max_gap frames ahead. Overlap is translation- and
    elongation-robust, which centroid distance is not."""
    # node = (t, label); track = list of (t, label)
    nodes = [(t, lab) for t in range(nframes) for lab in objects_per_frame.get(t, {})]
    used = set()
    tracks = []
    for start in nodes:
        if start in used:
            continue
        used.add(start)
        track = [start]
        t, lab = start
        cur = objects_per_frame[t][lab]
        while True:
            best, best_iou = None, min_iou
            for dt in range(1, max_gap + 1):
                tt = t + dt
                for lab2, m2 in objects_per_frame.get(tt, {}).items():
                    if (tt, lab2) in used:
                        continue
                    ov = overlap(cur, m2)
                    if ov > best_iou:
                        best, best_iou = (tt, lab2), ov
                if best is not None:
                    break
            if best is None:
                break
            used.add(best)
            track.append(best)
            t, lab = best
            cur = objects_per_frame[t][lab]
        if len(track) >= min_support:
            tracks.append(track)
    return tracks


def present_frames(track, nframes):
    """Frames the track occupies, completing interior gaps (the cell did not
    leave and come back within its own span)."""
    ts = sorted(t for t, _ in track)
    return set(range(ts[0], ts[-1] + 1))


def per_frame_counts(tracks, nframes):
    counts = [0] * nframes
    for tr in tracks:
        for t in present_frames(tr, nframes):
            if 0 <= t < nframes:
                counts[t] += 1
    return counts


def score(tracks, expected):
    n = len(expected)
    counts = per_frame_counts(tracks, n)
    tp = sum(min(counts[t], expected[t]) for t in range(n))
    over = sum(max(counts[t] - expected[t], 0) for t in range(n))
    gt = sum(expected)
    recall = tp / max(gt, 1)
    precision = tp / max(tp + over, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    return recall, precision, f1, counts


print("GENUINE real-data benchmark v2: overlap-linked real masks vs eye-verified counts")
print(f"{'movie':>16} {'drop':>5} {'tracks(exp)':>12} {'recall':>7} {'prec':>7} {'F1':>7}")
for movie, expected in EXPECTED_COUNTS.items():
    masks = load_masks(movie)
    nframes = len(expected)
    objs_full = {t: frame_objects(masks[t]) for t in range(nframes)}
    for drop in (0.0, 0.3, 0.5):
        recs, precs, f1s, ntr = [], [], [], []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            objs = {t: ({} if drop and rng.random() < drop else dict(o))
                    for t, o in objs_full.items()} if drop else objs_full
            tracks = link_by_overlap(objs, nframes)
            r, p, f1, _ = score(tracks, expected)
            recs.append(r); precs.append(p); f1s.append(f1); ntr.append(len(tracks))
        exp_tr = EXPECTED_TRACKS.get(movie, "-")
        print(f"{movie:>16} {drop:>5.1f} {np.mean(ntr):>5.1f}/{str(exp_tr):>5} "
              f"{np.mean(recs):>7.3f} {np.mean(precs):>7.3f} {np.mean(f1s):>7.3f}")
