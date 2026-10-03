"""NEW architecture: kymograph reconstruction. A confined channel is a 1-D tube;
collapse it to a position-along-channel x time image (a kymograph) built from RAW
intensity, not detections. A moving cell is a continuous bright curve there, so a
frame the detector missed is still present as intensity -- the missing-detection
fragmentation that caps detection-centric tracking at ~0.88 cannot occur.

Measured head-to-head with the detection-centric reconstructor under detector
dropout, on rendered intensity movies with planted per-cell truth. "The model
can be zero": the kymograph uses NO detector at all.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, "src")
from corridor.core.config import Scale, TrackingConfig          # noqa: E402
from corridor.core.detections import Detection                  # noqa: E402
from corridor.engine.reconstruct import reconstruct_tracks      # noqa: E402

H, W = 160, 200
TOL = 6.0


def render(seed):
    """Intensity movie + planted truth. Cells are bright elongated blobs moving
    along channels; channel walls are static bright bands; plus noise."""
    rng = np.random.default_rng(seed)
    nframes = int(rng.integers(14, 24))
    n_lanes = int(rng.integers(1, 6))
    lane_x = np.linspace(28, W - 28, n_lanes)
    truth = {}
    cell_masks = {}
    cid = 0
    for lx in lane_x:
        for _ in range(int(rng.integers(1, 2) + 1)):
            birth = int(rng.integers(0, max(1, nframes // 3)))
            death = int(min(nframes, birth + rng.integers(nframes // 2, nframes)))
            y = float(rng.uniform(30, H - 30)); vy = float(rng.normal(0, 3.0))
            length = float(rng.uniform(16, 36))
            tk, mk = {}, {}
            for t in range(birth, death):
                y += vy + rng.normal(0, 0.6); vy = 0.82 * vy + rng.normal(0, 0.9)
                length = float(np.clip(length + rng.normal(0, 2), 12, 50))
                y = float(np.clip(y, 10, H - 10)); x = float(lx + rng.normal(0, 1.0))
                tk[t] = (x, y)
                y0, y1 = int(max(0, y - length / 2)), int(min(H, y + length / 2))
                m = np.zeros((H, W), bool); m[y0:y1, int(x - 3):int(x + 4)] = True
                mk[t] = m
            if len(tk) >= 4:
                truth[cid] = tk; cell_masks[cid] = mk; cid += 1
    # render intensity: walls (static) + cells (bright) + noise
    images = np.zeros((nframes, H, W), np.float32)
    for lx in lane_x:                       # static wall bands either side of lumen
        for wx in (int(lx - 12), int(lx + 12)):
            if 0 <= wx < W:
                images[:, :, wx:wx + 2] += 0.5
    for c, mk in cell_masks.items():
        for t, m in mk.items():
            images[t][m] += 1.0
    images += rng.normal(0, 0.3, images.shape).astype(np.float32)
    return images, truth, lane_x, cell_masks, nframes


# ---- detection-centric path (for comparison), with detector dropout ----
def detections_with_dropout(cell_masks, dropout, seed, nframes):
    rng = np.random.default_rng(seed + 9999)
    dets = []; label = 1
    for c, mk in cell_masks.items():
        for t, m in mk.items():
            if dropout and rng.random() < dropout:
                continue
            ys, xs = np.nonzero(m)
            minr, minc, maxr, maxc = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            dets.append(Detection(frame=t, label=label, x=float(xs.mean()), y=float(ys.mean()),
                area_px=float(m.sum()), bbox=(int(minr), int(minc), int(maxr), int(maxc)),
                extent_px=int(maxr - minr), eccentricity=0.95, orientation_rad=0.0,
                major_axis_px=float(maxr - minr), minor_axis_px=float(maxc - minc),
                solidity=1.0, touches_border=False, channel=0,
                mask_crop=m[minr:maxr, minc:maxc].copy())); label += 1
    return dets


# ---- NEW kymograph path (no detector) ----
def _runs(row_mask, min_run=6, max_run=70):
    """Contiguous True runs in a 1-D boolean profile -> list of (center, length)."""
    out = []
    idx = np.nonzero(row_mask)[0]
    if len(idx) == 0:
        return out
    splits = np.where(np.diff(idx) > 2)[0]  # allow a 1-2 px gap inside a cell
    groups = np.split(idx, splits + 1)
    for g in groups:
        length = g[-1] - g[0] + 1
        if min_run <= length <= max_run:
            out.append((float(g.mean()), int(length)))
    return out


def kymo_reconstruct(images, lane_x, half=6, min_len=4, thr_sigma=3.0,
                     v_max=18.0, max_gap=4):
    """Trajectories from raw intensity, per channel, NO detections. Each frame's
    lumen profile is segmented into 1-D bright runs (a cell = a contiguous bright
    run: centre = position, length = the cell), and runs are linked across frames
    -- in the dropout-free kymograph a cell's run is present every frame it lives,
    so there are no gaps to bridge and two cells stay separate by their centres."""
    nframes, Hh, Ww = images.shape
    tracks = []
    for lx in lane_x:
        x0, x1 = int(max(0, lx - half)), int(min(Ww, lx + half + 1))
        K = images[:, :, x0:x1].max(axis=2)                  # (nframes, H) kymograph
        bg = np.median(K)
        mad = np.median(np.abs(K - bg)) + 1e-6
        thr = bg + thr_sigma * 1.4826 * mad
        per_frame = [ _runs(K[t] > thr) for t in range(nframes) ]
        # link runs across frames by centre proximity (1-D, velocity-gated)
        used = {t: [False] * len(per_frame[t]) for t in range(nframes)}
        for t0 in range(nframes):
            for i0, (c0, L0) in enumerate(per_frame[t0]):
                if used[t0][i0]:
                    continue
                used[t0][i0] = True
                tk = {t0: (float(lx), c0)}
                t, c = t0, c0
                while True:
                    best, bj, bt, bd = None, None, None, v_max * max_gap
                    for dt in range(1, max_gap + 1):
                        tt = t + dt
                        if tt >= nframes:
                            break
                        for j, (cj, Lj) in enumerate(per_frame[tt]):
                            if used[tt][j]:
                                continue
                            d = abs(cj - c)
                            if d / dt <= v_max and d < bd:
                                best, bj, bt, bd = (tt, cj), j, tt, d
                        if best is not None:
                            break
                    if best is None:
                        break
                    used[bt][bj] = True
                    tk[bt] = (float(lx), best[1])
                    t, c = bt, best[1]
                if len(tk) >= min_len:
                    tracks.append(tk)
    return tracks


def _kymo_runs_per_frame(images, lx, half, thr_sigma, nframes):
    x0, x1 = int(max(0, lx - half)), int(min(images.shape[2], lx + half + 1))
    K = images[:, :, x0:x1].max(axis=2)
    bg = np.median(K); mad = np.median(np.abs(K - bg)) + 1e-6
    thr = bg + thr_sigma * 1.4826 * mad
    return [_runs(K[t] > thr) for t in range(nframes)]


def _run_near(runs_t, y, tol):
    best, bd = None, tol
    for (c, L) in runs_t:
        if abs(c - y) < bd:
            best, bd = c, abs(c - y)
    return best


def hybrid_reconstruct(images, dets, lane_x, scale, cfg, half=6, thr_sigma=3.0,
                       fill_tol=10.0, confirm_frac=0.6):
    """HYBRID: the detector-based reconstructor is the high-precision backbone;
    the kymograph (raw intensity, gap-proof) supplies independent, POSITION-
    specific evidence to (a) stitch two fragments of one cell when the cell is
    actually visible between them, and (b) fill the frames the detector dropped.
    The detector runs to its own ability; the kymograph covers only its gaps."""
    nframes = images.shape[0]
    trk, _ = reconstruct_tracks(dets, (images.shape[1], images.shape[2]), nframes, scale, cfg)
    segs = [{o.frame: (o.x, o.y) for o in t.observations} for t in trk]
    # per-channel kymograph runs
    runs = {lx: _kymo_runs_per_frame(images, lx, half, thr_sigma, nframes) for lx in lane_x}

    def chan(seg):
        mx = np.mean([p[0] for p in seg.values()])
        return min(lane_x, key=lambda lx: abs(lx - mx))

    # (a) kymograph-evidenced stitch: merge same-channel fragments when the raw
    #     intensity shows the cell continuously between them
    changed = True
    while changed:
        changed = False
        segs = [s for s in segs if s]
        segs.sort(key=lambda s: min(s))
        for i, a in enumerate(segs):
            if a is None:
                continue
            lx = chan(a); R = runs[lx]
            ta = max(a); ya = a[ta]
            for j, b in enumerate(segs):
                if b is None or j == i or chan(b) != lx:
                    continue
                tb = min(b); yb = b[tb]
                gap = tb - ta
                if not (1 <= gap <= 10):
                    continue
                inner = list(range(ta + 1, tb))
                ok = sum(1 for t in inner
                         if _run_near(R[t], ya[1] + (yb[1] - ya[1]) * (t - ta) / gap, fill_tol) is not None)
                if inner and ok >= confirm_frac * len(inner):
                    for t in inner:
                        yp = ya[1] + (yb[1] - ya[1]) * (t - ta) / gap
                        c = _run_near(R[t], yp, fill_tol)
                        if c is not None:
                            a[t] = (lx, c)
                    a.update(b); segs[j] = None; changed = True
            # end-extension while the kymograph keeps showing the cell
        segs = [s for s in segs if s is not None]

    # (b) fill interior gaps + extend ends of each track from the kymograph
    out = []
    for s in segs:
        lx = chan(s); R = runs[lx]
        fs = sorted(s)
        full = dict(s)
        for a, bb in zip(fs, fs[1:]):
            for t in range(a + 1, bb):
                yp = s[a][1] + (s[bb][1] - s[a][1]) * (t - a) / (bb - a)
                c = _run_near(R[t], yp, fill_tol)
                if c is not None:
                    full[t] = (lx, c)
        # end-extension: keep the cell while the kymograph still shows it nearby
        for direction in (-1, +1):
            t = (min(full) if direction < 0 else max(full))
            yprev = full[t][1]
            while 0 <= t + direction < nframes:
                t += direction
                c = _run_near(R[t], yprev, fill_tol)
                if c is None:
                    break
                full[t] = (lx, c); yprev = c
        out.append(full)

    # (c) kymograph-only tracks: cells the detector missed ENTIRELY. Add a kymo
    #     track only where no hybrid track already occupies that channel+frames.
    covered = {}
    for tr in out:
        lx = chan(tr)
        for t, (_, y) in tr.items():
            covered.setdefault((round(lx), t), []).append(y)
    for ktr in kymo_reconstruct(images, lane_x):
        lx = chan(ktr)
        novel = sum(1 for t, (_, y) in ktr.items()
                    if all(abs(y - yc) > fill_tol for yc in covered.get((round(lx), t), [])))
        if novel >= 0.6 * len(ktr):      # mostly new -> a genuinely missed cell
            out.append(ktr)
    return out


def score(tracks, truth):
    total = correct = 0
    for c, tk in truth.items():
        best = -1
        for r in tracks:
            hits = sum(1 for t, (x, y) in tk.items()
                       if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
            best = max(best, hits)
        total += len(tk); correct += max(best, 0)
    recall = correct / max(total, 1)
    rtot = rhit = 0
    for r in tracks:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in tk and np.hypot(tk[t][0] - x, tk[t][1] - y) <= TOL for tk in truth.values()):
                rhit += 1
    prec = rhit / max(rtot, 1)
    return recall, prec, 2 * recall * prec / max(recall + prec, 1e-9), len(tracks)


def main():
    scale = Scale.from_values(0.5, 10.0); cfg = TrackingConfig()
    print("detector dropout cannot touch the kymograph (it uses raw intensity, no detections)")
    print(f"{'dropout':>8} {'method':>11} {'recall':>7} {'prec':>7} {'F1':>7} {'trk-err':>8}")
    for dropout in (0.0, 0.3, 0.5, 0.7):
        for method in ("detection", "kymograph", "HYBRID"):
            rs, ps, fs, errs = [], [], [], []
            for seed in range(40):
                images, truth, lane_x, cell_masks, nf = render(seed)
                if not truth:
                    continue
                if method == "detection":
                    dets = detections_with_dropout(cell_masks, dropout, seed, nf)
                    trk, _ = reconstruct_tracks(dets, (H, W), nf, scale, cfg)
                    tracks = [{o.frame: (o.x, o.y) for o in t.observations} for t in trk]
                elif method == "kymograph":
                    tracks = kymo_reconstruct(images, lane_x)   # no detector at all
                else:
                    dets = detections_with_dropout(cell_masks, dropout, seed, nf)
                    tracks = hybrid_reconstruct(images, dets, lane_x, scale, cfg)
                r, p, f, ntr = score(tracks, truth)
                rs.append(r); ps.append(p); fs.append(f); errs.append(ntr - len(truth))
            print(f"{dropout:>8.1f} {method:>11} {np.mean(rs):>7.3f} {np.mean(ps):>7.3f} "
                  f"{np.mean(fs):>7.3f} {np.mean(errs):>+8.2f}")


if __name__ == "__main__":
    main()
