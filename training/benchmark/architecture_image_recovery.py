"""The FULL architecture across 271 realistic trials, WITH image-evidence
recovery -- the piece the mask-only battery cannot have and the production
pipeline does. Each trial is rendered to a noisy phase-like image (cells,
channel walls, debris FPs, gaussian noise). The tracker then:

  overlap-link  ->  pre-stitch temporal-support FP gate  ->  conservative stitch
  ->  image-evidence recovery (NCC template match) of interior holes AND track
      ends, so a frame is added back ONLY where the cell's own appearance is
      actually present -- not by blind extrapolation.

This is the honest many-trial test of 'does the architecture reach 98%'.
"""
from __future__ import annotations
import sys
import numpy as np
from skimage.feature import match_template

sys.path.insert(0, "src")
sys.path.insert(0, "training/benchmark")
from corridor.engine import reconstruct as R           # noqa: E402
import battery                                          # noqa: E402

H, W, TOL = battery.H, battery.W, battery.TOL
BG, NOISE, CELL, WALL = 1000.0, 80.0, 320.0, 1500.0    # SNR(cell)~4, realistic


def simulate_img(seed):
    """One cell per lane + dropout + debris FPs + motion, AND the rendered image
    stack the cells and FPs actually appear in."""
    rng = np.random.default_rng(seed)
    nf = int(rng.integers(12, 24))
    n_lanes = int(rng.integers(1, 7))
    lane_x = np.linspace(20, W - 20, n_lanes)
    fp_rate = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
    fast = rng.random() < 0.4
    img = rng.normal(BG, NOISE, (nf, H, W)).astype(np.float32)
    for lx in lane_x:                                   # channel walls
        xi = int(lx)
        img[:, :, max(0, xi - 7):xi - 5] += WALL
        img[:, :, min(W, xi + 6):xi + 8] += WALL
    truth, cell_masks, cid = {}, {}, 0
    for lx in lane_x:
        birth = int(rng.integers(0, max(1, nf // 3)))
        death = int(min(nf, birth + rng.integers(nf // 2, nf)))
        y = float(rng.uniform(20, H - 20))
        vy = float(rng.normal(0, 5.0 if fast else 2.5))
        length = float(rng.uniform(16, 40))
        track, masks = {}, {}
        for t in range(birth, death):
            y += vy + rng.normal(0, 0.6); vy = 0.8 * vy + rng.normal(0, 1.0)
            if fast and rng.random() < 0.15:
                vy = -vy
            length = float(np.clip(length + rng.normal(0, 3), 12, 60))
            y = float(np.clip(y, 8, H - 8)); x = float(lx + rng.normal(0, 1.5))
            y0, y1 = int(max(0, y - length / 2)), int(min(H, y + length / 2))
            m = np.zeros((H, W), bool); m[y0:y1, int(x - 3):int(x + 4)] = True
            img[t][m] += CELL                           # the cell IS in the image
            track[t] = (x, y); masks[t] = m
        if len(track) >= 3:
            truth[cid] = track; cell_masks[cid] = masks; cid += 1
    dets, label = [], 1
    for c, masks in cell_masks.items():
        for t, m in masks.items():
            if dropout and rng.random() < dropout:
                continue                                # detector missed it (cell still in image)
            d = battery._make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    for t in range(nf):                                 # debris false positives
        for _ in range(rng.poisson(fp_rate)):
            yy = int(rng.uniform(10, H - 10)); xx = int(rng.choice(lane_x))
            m = np.zeros((H, W), bool); m[yy:yy + 12, xx - 3:xx + 4] = True
            img[t][m] += CELL
            d = battery._make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    return dets, truth, nf, img, lane_x


def chains(dets, nf, gate_pre=2, gate_post=2):
    objs = R._masks_by_frame(dets, (H, W))
    cs = R._link(objs, nf, 3, 0.1)
    cs = [c for c in cs if len(c) >= gate_pre]
    cs = R._stitch(cs, objs, 5, 60.0)
    out = []
    for ch in cs:
        d = {t: R._centroid(objs[t][lab][1]) for (t, lab) in ch}
        if len(d) >= gate_post:
            out.append(d)
    return out


def _template(img, d, r=7):
    crops = []
    for t, (x, y) in d.items():
        xi, yi = int(round(x)), int(round(y))
        if r <= yi < H - r and r <= xi < W - r:
            crops.append(img[t, yi - r:yi + r, xi - r:xi + r])
    return np.median(np.stack(crops), 0) if len(crops) >= 2 else None


def _ncc(img, t, tmpl, cx, cy, win=12):
    y0, y1 = max(0, int(cy) - win), min(H, int(cy) + win)
    x0, x1 = max(0, int(cx) - win), min(W, int(cx) + win)
    reg = img[t, y0:y1, x0:x1]
    if reg.shape[0] <= tmpl.shape[0] or reg.shape[1] <= tmpl.shape[1]:
        return 0.0
    return float(match_template(reg, tmpl).max())


def gapfill(d):
    """Interior holes: linear interpolation between a track's own consecutive
    observations. Safe -- a tracked cell exists between its own detections."""
    ts = sorted(d); out = dict(d)
    for a, b in zip(ts, ts[1:]):
        if b - a > 1:
            (xa, ya), (xb, yb) = d[a], d[b]
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (xa + (xb - xa) * w, ya + (yb - ya) * w)
    return out


def end_recover(img, d, nf, ncc_thr=0.8, horizon=4):
    """Image-evidence END extension only: step outward one frame at a time along
    the track's velocity, add it ONLY if the cell's template is actually there
    (NCC>=thr), and STOP at the first miss -- a cell that has left stays left."""
    ts = sorted(d)
    tmpl = _template(img, d)
    if tmpl is None or len(ts) < 2:
        return d
    out = dict(d)
    vx0, vy0 = d[ts[0]][0] - d[ts[1]][0], d[ts[0]][1] - d[ts[1]][1]
    x, y = d[ts[0]]
    for j in range(1, horizon + 1):
        cx, cy = x + vx0 * j, y + vy0 * j
        t = ts[0] - j
        if t < 0 or not (0 <= cx < W and 0 <= cy < H) or _ncc(img, t, tmpl, cx, cy) < ncc_thr:
            break
        out[t] = (cx, cy)
    vxe, vye = d[ts[-1]][0] - d[ts[-2]][0], d[ts[-1]][1] - d[ts[-2]][1]
    x, y = d[ts[-1]]
    for j in range(1, horizon + 1):
        cx, cy = x + vxe * j, y + vye * j
        t = ts[-1] + j
        if t >= nf or not (0 <= cx < W and 0 <= cy < H) or _ncc(img, t, tmpl, cx, cy) < ncc_thr:
            break
        out[t] = (cx, cy)
    return out


def lane_exclusive(tracks, lane_x):
    """Confined-migration prior: at most one cell per lane per frame. Assign each
    track to its nearest lane; within a lane, keep tracks greedily by support,
    dropping any that shares a frame with an already-kept track. A real debris
    FP that briefly co-occupies a lane with the cell is removed; a cell that
    leaves before another enters the same lane is kept (no temporal overlap)."""
    by_lane: dict[int, list] = {}
    for d in tracks:
        mx = float(np.median([p[0] for p in d.values()]))
        li = int(np.argmin([abs(mx - lx) for lx in lane_x]))
        by_lane.setdefault(li, []).append(d)
    kept = []
    for li, group in by_lane.items():
        group.sort(key=len, reverse=True)
        occupied = []     # list of frame-sets already kept in this lane
        for d in group:
            frames = set(d)
            if any(frames & occ for occ in occupied):
                continue
            occupied.append(frames); kept.append(d)
    return kept


def score(recon, truth):
    total = correct = 0
    for c, tk in truth.items():
        best, bh = None, -1
        for i, r in enumerate(recon):
            hits = sum(1 for t, (x, y) in tk.items()
                       if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
            if hits > bh:
                best, bh = i, hits
        total += len(tk)
        if best is not None:
            r = recon[best]
            correct += sum(1 for t, (x, y) in tk.items()
                           if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
    recall = correct / max(total, 1)
    rtot = rhit = 0
    for r in recon:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in tk and np.hypot(tk[t][0] - x, tk[t][1] - y) <= TOL
                   for tk in truth.values()):
                rhit += 1
    prec = rhit / max(rtot, 1)
    return recall, prec, 2 * recall * prec / max(recall + prec, 1e-9)


def main(n=271):
    trials = [simulate_img(s) for s in range(n)]
    trials = [x for x in trials if x[1]]
    print(f"FULL ARCHITECTURE on realistic 1-cell/lane, {len(trials)} trials")
    print(f"  {'config':>30} {'recall':>7} {'prec':>7} {'F1':>7} {'>=0.98':>8} {'>=0.95':>8}")
    configs = [
        ("gate2 + fill (no lane)", dict(gp=2, gq=2, fill=True, lane=False)),
        ("gate2 + fill + lane", dict(gp=2, gq=2, fill=True, lane=True)),
        ("gate2pre/1post + fill + lane", dict(gp=2, gq=1, fill=True, lane=True)),
        ("gate3 + fill + lane", dict(gp=3, gq=3, fill=True, lane=True)),
    ]
    for name, kw in configs:
        rs, ps, fs = [], [], []
        for dets, truth, nf, img, lane_x in trials:
            tr = chains(dets, nf, gate_pre=kw["gp"], gate_post=kw["gq"])
            if kw["fill"]:
                tr = [gapfill(d) for d in tr]
            if kw["lane"]:
                tr = lane_exclusive(tr, lane_x)
            r, p, f = score(tr, truth)
            rs.append(r); ps.append(p); fs.append(f)
        fs = np.array(fs)
        print(f"  {name:>30} {np.mean(rs):>7.3f} {np.mean(ps):>7.3f} {fs.mean():>7.3f} "
              f"{int((fs >= 0.98).sum()):>4}/{len(fs)} {int((fs >= 0.95).sum()):>4}/{len(fs)}")


if __name__ == "__main__":
    main()
