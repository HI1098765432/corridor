"""Close the two measured gaps in the overlap backbone, the 'fill the gaps with
the complementary method' approach -- without switching trackers:

  +gapfill : interior missing frames interpolated along the chain  -> recall
  +gate    : chains with too little support dropped (phantom FP blobs) -> precision
  +extend  : a track's ends extended a few frames by its own velocity -> recall

A/B/C/D against the plain backbone on the SAME 271 adversarial-synthetic trials,
reporting recall / precision / F1 and the count that clear 0.98, so each
component's lift is measured, not assumed.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "training/benchmark")
from corridor.engine import reconstruct as R           # noqa: E402
import battery                                          # noqa: E402

TOL = battery.TOL
H, W = battery.H, battery.W


def simulate_one(seed):
    """The REAL confined-migration regime: exactly ONE cell per microfluidic
    lane (the device isolates cells), no division, with the degradations that
    actually occur -- segmentation dropout, occasional debris false positives,
    and ordinary stall/surge/reversal motion. Mirrors battery.simulate's cell
    and detection model, with ncell fixed at 1."""
    rng = np.random.default_rng(seed)
    nframes = int(rng.integers(12, 24))
    n_lanes = int(rng.integers(1, 7))
    lane_x = np.linspace(20, W - 20, n_lanes)
    fp_rate = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3]))   # real recall is ~0.73
    fast = rng.random() < 0.4
    truth, cell_masks, cid = {}, {}, 0
    for lx in lane_x:
        birth = int(rng.integers(0, max(1, nframes // 3)))
        death = int(min(nframes, birth + rng.integers(nframes // 2, nframes)))
        y = float(rng.uniform(20, H - 20))
        vy = float(rng.normal(0, 5.0 if fast else 2.5))
        length = float(rng.uniform(16, 40))
        track, masks = {}, {}
        for t in range(birth, death):
            y += vy + rng.normal(0, 0.6); vy = 0.8 * vy + rng.normal(0, 1.0)
            if fast and rng.random() < 0.15:
                vy = -vy
            length = float(np.clip(length + rng.normal(0, 3), 12, 60))
            y = float(np.clip(y, 8, H - 8))
            x = float(lx + rng.normal(0, 1.5))
            y0, y1 = int(max(0, y - length / 2)), int(min(H, y + length / 2))
            m = np.zeros((H, W), bool); m[y0:y1, int(x - 3):int(x + 4)] = True
            track[t] = (x, y); masks[t] = m
        if len(track) >= 3:
            truth[cid] = track; cell_masks[cid] = masks; cid += 1
    dets, label = [], 1
    for c, masks in cell_masks.items():
        for t, m in masks.items():
            if dropout and rng.random() < dropout:
                continue
            d = battery._make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    for t in range(nframes):
        for _ in range(rng.poisson(fp_rate)):
            yy = int(rng.uniform(10, H - 10)); xx = int(rng.choice(lane_x))
            m = np.zeros((H, W), bool); m[yy:yy + 12, xx - 3:xx + 4] = True
            d = battery._make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    return dets, truth, nframes


def chains_for(dets, nf, gate_pre=0, stitch=True):
    """Link by overlap; optionally drop unsupported chains BEFORE stitching, so
    a false-positive fragment can never be merged into a real track (and then
    gap-filled into phantom points). Returns per-track position dicts."""
    objs = R._masks_by_frame(dets, (H, W))
    chains = R._link(objs, nf, max_gap=3, min_overlap=0.1)
    if gate_pre:
        chains = [c for c in chains if len(c) >= gate_pre]
    if stitch:
        chains = R._stitch(chains, objs, 5, 60.0)
    out = []
    for ch in chains:
        d = {t: R._centroid(objs[t][lab][1]) for (t, lab) in ch}
        out.append(d)
    return out


def gapfill(d):
    ts = sorted(d)
    out = dict(d)
    for a, b in zip(ts, ts[1:]):
        if b - a > 1:
            (xa, ya), (xb, yb) = d[a], d[b]
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (xa + (xb - xa) * w, ya + (yb - ya) * w)
    return out


def extend(d, nf, k=2):
    """Extrapolate up to k frames at each end along the local velocity, kept in
    frame. Conservative: only when there are >=2 points to define a velocity."""
    ts = sorted(d)
    if len(ts) < 2:
        return d
    out = dict(d)
    (x0, y0), (x1, y1) = d[ts[0]], d[ts[1]]
    vx, vy = x0 - x1, y0 - y1                       # backward velocity (toward earlier)
    for j in range(1, k + 1):
        t = ts[0] - j
        if t < 0:
            break
        x, y = x0 + vx * j, y0 + vy * j
        if 0 <= x < W and 0 <= y < H:
            out[t] = (x, y)
    (xe, ye), (xp, yp) = d[ts[-1]], d[ts[-2]]
    fx, fy = xe - xp, ye - yp
    for j in range(1, k + 1):
        t = ts[-1] + j
        if t >= nf:
            break
        x, y = xe + fx * j, ye + fy * j
        if 0 <= x < W and 0 <= y < H:
            out[t] = (x, y)
    return out


def build(dets, nf, *, fill=False, gate=0, gate_pre=0, stitch=True, ext=0):
    recon = chains_for(dets, nf, gate_pre=gate_pre, stitch=stitch)
    if gate:
        recon = [d for d in recon if len(d) >= gate]
    if ext:
        recon = [extend(d, nf, ext) for d in recon]
    if fill:
        recon = [gapfill(d) for d in recon]
    return recon


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


CONFIGS = [
    ("plain", dict()),
    ("pre2+post3+fill", dict(gate_pre=2, gate=3, fill=True)),
    ("pre2+post4+fill", dict(gate_pre=2, gate=4, fill=True)),
    ("pre3+post3+fill", dict(gate_pre=3, gate=3, fill=True)),
    ("pre2+post3+fill+ext1", dict(gate_pre=2, gate=3, fill=True, ext=1)),
    ("pre3+post3+fill+ext1", dict(gate_pre=3, gate=3, fill=True, ext=1)),
]


def run(hard=False, n=271, realistic=False):
    trials = []
    for seed in range(n):
        if realistic:
            dets, truth, nf = simulate_one(seed)
        else:
            dets, truth, nf = battery.simulate(seed, hard=hard)
        if truth:
            trials.append((dets, truth, nf))
    label = ("REALISTIC (1 cell/lane, real confined migration)" if realistic
             else ("ADVERSARIAL" if hard else "WELL-POSED (2 cells/lane)"))
    print(f"\n{label}  ({len(trials)} trials)")
    print(f"  {'config':>18} {'recall':>7} {'prec':>7} {'F1':>7} {'>=0.98':>8}")
    for name, kw in CONFIGS:
        rs, ps, fs = [], [], []
        for dets, truth, nf in trials:
            recon = build(dets, nf, **kw)
            r, p, f = score(recon, truth)
            rs.append(r); ps.append(p); fs.append(f)
        fs = np.array(fs)
        print(f"  {name:>18} {np.mean(rs):>7.3f} {np.mean(ps):>7.3f} "
              f"{fs.mean():>7.3f} {int((fs >= 0.98).sum()):>5}/{len(fs)}")


if __name__ == "__main__":
    run(realistic=True)
    run(hard=False)
    run(hard=True)
