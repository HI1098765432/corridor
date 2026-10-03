"""Overcome the three limits, measured on rendered images (numpy/scipy/skimage).

L1 free 2-D: extraction works in (x,y,t); lanes are optional, not required.
L2 zero-detection cells: motion (frame-difference) + self-templating recover a
   cell the per-frame threshold detector missed, by appearance, along its path.
L3 the 0.92 plateau: global successive-best-path data association, not greedy.

Runs in seconds. Prints recall/precision/F1 for channel AND free-2D modes, and
the fraction of a deliberately faint cell's frames recovered by templating.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from skimage.feature import match_template

T, H, W = 24, 140, 220
TOL = 4.0            # px: a reconstructed position is correct within this
V_MAX = 14.0         # px/frame ceiling
CELL_LEN, CELL_WID = 15.0, 5.0


def _blob(img, x, y, angle, amp, length=CELL_LEN, width=CELL_WID):
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = xs - x, ys - y
    ca, sa = np.cos(angle), np.sin(angle)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    img += amp * np.exp(-(u * u) / (2 * (length / 2) ** 2) - (v * v) / (2 * (width / 2) ** 2))


def simulate(mode, rng, n_cells=6, noise=0.25, faint_cell=False):
    """Return images (T,H,W) and gt[cid] = {t:(x,y)}."""
    gt = {}
    for cid in range(n_cells):
        if mode == "channel":
            lane_x = 25 + cid * (W - 50) / max(n_cells - 1, 1)
            x = lane_x
            y = rng.uniform(20, H - 20)
            vy = rng.normal(0, 3.0)
            angle = np.pi / 2
        else:  # free 2-D
            x = rng.uniform(20, W - 20)
            y = rng.uniform(20, H - 20)
            vx, vy = rng.normal(0, 3.0), rng.normal(0, 3.0)
            angle = rng.uniform(0, np.pi)
        birth = int(rng.integers(0, 4))
        death = int(min(T, birth + rng.integers(T - 6, T)))
        track = {}
        for t in range(birth, death):
            if mode == "channel":
                y += vy + rng.normal(0, 0.5); vy = 0.85 * vy + rng.normal(0, 0.8)
                y = float(np.clip(y, 10, H - 10))
            else:
                x += vx + rng.normal(0, 0.5); y += vy + rng.normal(0, 0.5)
                vx = 0.85 * vx + rng.normal(0, 0.8); vy = 0.85 * vy + rng.normal(0, 0.8)
                x = float(np.clip(x, 10, W - 10)); y = float(np.clip(y, 10, H - 10))
                angle = np.arctan2(vy, vx)
            track[t] = (float(x), float(y), float(angle))
        if len(track) >= 4:
            gt[cid] = track
    # amplitude: one cell is deliberately faint (near the noise floor) for L2
    images = np.zeros((T, H, W), np.float32)
    amps = {cid: (0.5 if (faint_cell and cid == min(gt)) else 1.0) for cid in gt}
    for t in range(T):
        frame = np.zeros((H, W), np.float32)
        if mode == "channel":
            for cid in gt:  # faint static wall texture along each lane
                lane_x = gt[cid][min(gt[cid])][0]
                frame[:, int(lane_x) - 8:int(lane_x) + 8] += 0.15
        for cid, track in gt.items():
            if t in track:
                x, y, a = track[t]
                _blob(frame, x, y, a, amps[cid])
        frame += rng.normal(0, noise, (H, W)).astype(np.float32)
        images[t] = frame
    gt_xy = {cid: {t: (v[0], v[1]) for t, v in tr.items()} for cid, tr in gt.items()}
    return images, gt_xy, amps


def base_detect(images, level):
    """Weak per-frame detector: threshold + centroid of connected components.
    A faint cell falls under the threshold in most frames (-> zero/few detections)."""
    dets = []  # (t, x, y)
    for t in range(T):
        f = images[t]
        bg = ndimage.median_filter(f, size=9)
        resp = f - bg
        mask = resp > level
        lbl, n = ndimage.label(mask)
        if n:
            for cy, cx in ndimage.center_of_mass(resp, lbl, range(1, n + 1)):
                if np.isfinite(cx):
                    dets.append((t, float(cx), float(cy)))
    return dets


def tdiff_detect(images, level):
    """Motion detector: a moving cell lights up in |I_t - I_{t-1}| even when it is
    too faint to pass the per-frame threshold. Model-free, no appearance prior."""
    dets = []
    for t in range(1, T):
        d = np.abs(images[t] - images[t - 1])
        mask = d > level
        lbl, n = ndimage.label(mask)
        if n:
            for cy, cx in ndimage.center_of_mass(d, lbl, range(1, n + 1)):
                if np.isfinite(cx):
                    dets.append((t, float(cx), float(cy)))  # motion seen entering frame t
    return dets


def best_path_extract(dets, max_gap=4, v_max=V_MAX, reward=3.0, min_support=4):
    """Global successive-best-path data association in (x,y,t). Each pass finds
    the single lowest-cost smooth path through the remaining detections (DP over
    the time-ordered DAG), extracts it, and repeats. Clutter forms short/jagged
    paths that fail the support+cost gate, so it is left behind -- the global
    optimum per pass, where per-point greedy is not."""
    pts = sorted(range(len(dets)), key=lambda i: dets[i][0])
    alive = set(pts)
    tracks = []
    while True:
        order = [i for i in pts if i in alive]
        if len(order) < min_support:
            break
        # DP: best[i] = min cost of a path ending at i (cost = sum link - reward*nodes)
        best = {i: -reward for i in order}
        parent = {i: None for i in order}
        for a_idx, i in enumerate(order):
            ti, xi, yi = dets[i]
            for j in order[a_idx + 1:]:
                tj, xj, yj = dets[j]
                dt = tj - ti
                if dt < 1:
                    continue
                if dt > max_gap:
                    break  # order is time-sorted, so all later j exceed the gap too
                dist = np.hypot(xj - xi, yj - yi)
                if dist / dt > v_max:
                    continue
                link = dist / dt + 0.5 * (dt - 1)      # smoothness + gap penalty
                cand = best[i] + link - reward
                if cand < best[j]:
                    best[j] = cand
                    parent[j] = i
        end = min(order, key=lambda i: best[i])
        # rebuild path
        path, k = [], end
        while k is not None:
            path.append(k); k = parent[k]
        path.reverse()
        total = best[end]
        if len(path) >= min_support and total < 0:
            tracks.append([dets[i] for i in path])
            alive -= set(path)
        else:
            break
    return tracks


def self_template(images, traj, win=12, ncc_thresh=0.45):
    """L2: a cell found in a few frames becomes its own matched-filter template;
    re-detect it by APPEARANCE in the frames the generic detector missed, inside
    a window around the motion-predicted position. Deterministic, no training."""
    obs = {int(round(t)): (x, y) for t, x, y in traj}
    ts = sorted(obs)
    # template = median crop over observed frames
    r = 10
    crops = []
    for t in ts:
        x, y = obs[t]; xi, yi = int(round(x)), int(round(y))
        if r <= yi < H - r and r <= xi < W - r:
            crops.append(images[t, yi - r:yi + r, xi - r:xi + r])
    if len(crops) < 2:
        return traj
    template = np.median(np.stack(crops), axis=0)
    # velocity model for prediction
    xs = np.array([obs[t][0] for t in ts]); ys = np.array([obs[t][1] for t in ts])
    tsa = np.array(ts, float)
    px = np.polyfit(tsa, xs, 1); py = np.polyfit(tsa, ys, 1)
    out = dict(obs)
    for t in range(ts[0], ts[-1] + 1):
        if t in obs:
            continue
        cx, cy = np.polyval(px, t), np.polyval(py, t)
        xi, yi = int(round(cx)), int(round(cy))
        y0, y1 = max(0, yi - win), min(H, yi + win)
        x0, x1 = max(0, xi - win), min(W, xi + win)
        region = images[t, y0:y1, x0:x1]
        if region.shape[0] <= template.shape[0] or region.shape[1] <= template.shape[1]:
            continue
        resp = match_template(region, template)
        peak = np.unravel_index(np.argmax(resp), resp.shape)
        if resp[peak] >= ncc_thresh:
            fy = y0 + peak[0] + template.shape[0] / 2
            fx = x0 + peak[1] + template.shape[1] / 2
            out[t] = (float(fx), float(fy))
    return [(t, out[t][0], out[t][1]) for t in sorted(out)]


def complete(traj):
    ts = [int(round(t)) for t, _, _ in traj]
    xs = [x for _, x, _ in traj]; ys = [y for _, _, y in traj]
    full = {}
    for a, b in zip(range(len(ts) - 1), range(1, len(ts))):
        ta, tb = ts[a], ts[b]
        full[ta] = (xs[a], ys[a])
        for t in range(ta + 1, tb):
            w = (t - ta) / (tb - ta)
            full[t] = (xs[a] * (1 - w) + xs[b] * w, ys[a] * (1 - w) + ys[b] * w)
    full[ts[-1]] = (xs[-1], ys[-1])
    return full


def score(gt, recon):
    total = correct = 0
    for cid, track in gt.items():
        for t, (x, y) in track.items():
            total += 1
            if any(t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL for r in recon):
                correct += 1
    recall = correct / max(total, 1)
    rtot = rhit = 0
    for r in recon:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in g and np.hypot(g[t][0] - x, g[t][1] - y) <= TOL for g in gt.values()):
                rhit += 1
    prec = rhit / max(rtot, 1)
    return recall, prec, 2 * recall * prec / max(recall + prec, 1e-9)


def reconstruct(images, use_overpropose, use_template):
    dets = base_detect(images, level=0.35)
    if use_overpropose:
        dets = dets + tdiff_detect(images, level=0.5)
    tracks = best_path_extract(dets)
    if use_template:
        tracks = [self_template(images, tr) for tr in tracks]
    return [complete(tr) for tr in tracks if len(tr) >= 4]


def run(mode, noise, faint, use_overpropose, use_template, seed):
    rng = np.random.default_rng(seed)
    images, gt, amps = simulate(mode, rng, noise=noise, faint_cell=faint)
    recon = reconstruct(images, use_overpropose, use_template)
    rec, prec, f1 = score(gt, recon)
    # L2 metric: recovery of the faint cell specifically
    faint_rec = None
    if faint and gt:
        fc = min(gt); track = gt[fc]
        tot = len(track); got = 0
        for t, (x, y) in track.items():
            if any(t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL for r in recon):
                got += 1
        faint_rec = got / max(tot, 1)
    return rec, prec, f1, faint_rec


print("L1 (free-2D works) + L3 (global extractor), clean, 20 seeds:")
print(f"{'mode':>8} {'noise':>6} {'recall':>7} {'prec':>7} {'F1':>7}")
for mode in ("channel", "free"):
    for noise in (0.25, 0.5):
        res = np.array([run(mode, noise, False, True, True, s)[:3] for s in range(20)])
        rec, prec, f1 = res.mean(axis=0)
        print(f"{mode:>8} {noise:>6.2f} {rec:>7.3f} {prec:>7.3f} {f1:>7.3f}")

print("\nL2 (faint near-invisible cell), fraction of its frames recovered, 20 seeds:")
print(f"{'variant':>34} {'faint-cell recovery':>20}")
for label, over, tmpl in [("base detector only", False, False),
                          ("+ motion over-propose", True, False),
                          ("+ motion + self-template", True, True)]:
    vals = [run("free", 0.5, True, over, tmpl, s)[3] for s in range(20)]
    vals = [v for v in vals if v is not None]
    print(f"{label:>34} {np.mean(vals):>20.3f}")
