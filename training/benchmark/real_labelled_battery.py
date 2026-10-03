"""A REAL tracking battery on REAL labelled cells -- no synthetic anything.

Source: five short runs of consecutive hand-labelled frames in the lab's
CellPose_TrainData (each frame has a human ground-truth mask, _seg.npy). The
tracker is fed those ground-truth masks as detections, so this tests the
TRACKING ARCHITECTURE on real cell shapes and motion (the segmentation model
never enters, so there is no train/test contamination).

Truth trajectories are built by strict overlap-linking of the GT masks. Then,
over many random seeds, a fraction of the GT detections is dropped (the real
segmentation misses ~27% of cells) and we measure how well the architecture
reconstructs the true trajectories -- identity-aware recall / precision / F1.
"""
from __future__ import annotations
import io
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from corridor.core.config import Scale, TrackingConfig          # noqa: E402
from corridor.core.detections import Detection                  # noqa: E402
from corridor.core.tracking import track_detections             # noqa: E402
from corridor.engine.reconstruct import reconstruct_tracks      # noqa: E402

ZIP = r"C:\Users\ironb\Downloads\OneDrive_2026-10-02.zip"
B = "Traction from Displacement/confinedmig_cellTrack/CellPose_TrainData/"
RUNS = [   # (folder, date, first, last) -- the consecutive labelled runs
    ("KK1", "041824", 5, 12), ("KK1", "041824", 16, 20),
    ("KK1", "061523", 1, 5), ("KK1", "122324", 1, 12),
    ("KK2", "052924", 22, 26),
]
TOL = 8.0
SCALE = Scale.from_values(0.5, 10.0)


def load_runs():
    z = zipfile.ZipFile(ZIP)
    movies = []
    for folder, date, a, b in RUNS:
        frames = []
        for i in range(a, b + 1):
            seg = f"{B}{folder}/{date}_{i}_seg.npy"
            try:
                d = np.load(io.BytesIO(z.read(seg)), allow_pickle=True).item()
            except KeyError:
                continue
            frames.append(np.asarray(d["masks"]).astype(np.int32))
        # keep only the leading frames that share the modal shape (a clean stack)
        if not frames:
            continue
        shapes = [f.shape for f in frames]
        modal = max(set(shapes), key=shapes.count)
        stack = [f for f in frames if f.shape == modal]
        if len(stack) >= 4:
            movies.append((f"{folder}/{date}_{a}-{b}", stack))
    return movies


def instances(mask):
    out = []
    for lab in np.unique(mask):
        if lab:
            out.append(mask == lab)
    return out


def det_from_mask(frame, label, m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    minr, minc, maxr, maxc = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    major = float(maxr - minr); minor = float(max(maxc - minc, 1))
    ecc = float(np.sqrt(max(1 - (minor / max(major, 1)) ** 2, 0)))
    return Detection(frame=int(frame), label=int(label), x=float(xs.mean()), y=float(ys.mean()),
                     area_px=float(m.sum()), bbox=(int(minr), int(minc), int(maxr), int(maxc)),
                     extent_px=int(max(major, minor)), eccentricity=ecc, orientation_rad=0.0,
                     major_axis_px=major, minor_axis_px=minor, solidity=1.0,
                     touches_border=False, channel=0, mask_crop=m[minr:maxr, minc:maxc].copy())


def overlap(a, b):
    inter = np.count_nonzero(a & b)
    if not inter:
        return 0.0
    return max(inter / np.count_nonzero(a | b), inter / max(min(a.sum(), b.sum()), 1))


def truth_tracks(stack):
    """Strict one-to-one overlap-linking of GT masks -> true trajectories
    {cell_id: {frame: (x,y)}}. Only unambiguous links (best and > 0.2) are made."""
    nf = len(stack)
    per = [instances(m) for m in stack]
    cents = [[(_c(m)) for m in fr] for fr in per]
    nextid = 0
    track_of = [dict() for _ in range(nf)]   # frame -> {inst_idx: cell_id}
    truth = defaultdict(dict)
    for j, m in enumerate(per[0]):
        truth[nextid][0] = cents[0][j]; track_of[0][j] = nextid; nextid += 1
    for t in range(1, nf):
        for j, m in enumerate(per[t]):
            best, bo = None, 0.2
            for k, pm in enumerate(per[t - 1]):
                o = overlap(m, pm)
                if o > bo:
                    best, bo = k, o
            if best is not None and best in track_of[t - 1]:
                cid = track_of[t - 1][best]
            else:
                cid = nextid; nextid += 1
            truth[cid][t] = cents[t][j]; track_of[t][j] = cid
    # keep cells seen in >=3 frames (a trajectory, not a blip)
    return {c: tk for c, tk in truth.items() if len(tk) >= 3}


def _c(m):
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def score(tracks, truth):
    recon = [{o.frame: (o.x, o.y) for o in tr.observations} for tr in tracks]
    total = correct = 0
    for c, tk in truth.items():
        best, bh = None, -1
        for i, r in enumerate(recon):
            h = sum(1 for t, (x, y) in tk.items() if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
            if h > bh:
                best, bh = i, h
        total += len(tk)
        if best is not None:
            r = recon[best]
            correct += sum(1 for t, (x, y) in tk.items() if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
    rec = correct / max(total, 1)
    rtot = rhit = 0
    for r in recon:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in tk and np.hypot(tk[t][0] - x, tk[t][1] - y) <= TOL for tk in truth.values()):
                rhit += 1
    prec = rhit / max(rtot, 1)
    return rec, prec, 2 * rec * prec / max(rec + prec, 1e-9)


def build_dets(stack, drop, seed):
    rng = np.random.default_rng(seed)
    dets, label = [], 1
    for t, m in enumerate(stack):
        for inst in instances(m):
            if drop and rng.random() < drop:
                continue
            d = det_from_mask(t, label, inst)
            if d is not None:
                dets.append(d); label += 1
    return dets


def main():
    movies = load_runs()
    print(f"REAL labelled mini-movies loaded: {len(movies)}")
    for name, stack in movies:
        tr = truth_tracks(stack)
        print(f"  {name:>22}: {len(stack)} frames, {len(tr)} true cell trajectories")
    print()
    print("Operating point (GT masks as detections, no extra drop):")
    for backend in ("kalman", "overlap"):
        fs = []
        for name, stack in movies:
            truth = truth_tracks(stack)
            dets = build_dets(stack, 0.0, 0)
            tr = _track(dets, stack, backend)
            fs.append(score(tr, truth)[2])
        print(f"  {backend:>8}: mean F1 {np.mean(fs):.3f} over {len(movies)} real movies")
    print()
    n = 54   # 54 seeds x 5 movies = 270 real trials
    print(f"REAL battery: {n * len(movies)} trials (GT detections, random drop 0-0.3 per trial):")
    print(f"  {'backend':>8} {'trials':>7} {'meanF1':>7} {'median':>7} {'min':>6} {'>=0.98':>8}")
    for backend in ("kalman", "overlap"):
        fs = []
        for name, stack in movies:
            truth = truth_tracks(stack)
            for seed in range(n):
                drop = float(np.random.default_rng(seed * 13 + 1).choice([0.0, 0.1, 0.2, 0.3]))
                dets = build_dets(stack, drop, seed)
                tr = _track(dets, stack, backend)
                fs.append(score(tr, truth))
        fs = np.array([f[2] for f in fs])
        print(f"  {backend:>8} {len(fs):>7} {fs.mean():>7.3f} {np.median(fs):>7.3f} {fs.min():>6.3f} "
              f"{int((fs >= 0.98).sum()):>4}/{len(fs)}")


def _track(dets, stack, backend):
    nf = len(stack); shape = stack[0].shape
    if backend == "overlap":
        return list(reconstruct_tracks(dets, shape, nf, SCALE, TrackingConfig())[0])
    return list(track_detections(dets, nf, SCALE, TrackingConfig())[0])


if __name__ == "__main__":
    main()
