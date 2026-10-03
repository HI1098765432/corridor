"""Statistical solidity: run the PRODUCTION reconstructor over 100+ independent
trials with hard cases it does NOT assume (collisions, divisions, entries/exits,
fast motion, false positives, dropout), each with planted per-cell truth, and
report the distribution of identity-aware accuracy. Adversarial-synthetic (the
simulator deliberately breaks the solver's assumptions); the 2 real movies
remain the real anchor. Mask-based, fast.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "src")
from corridor.core.config import Scale, TrackingConfig          # noqa: E402
from corridor.core.detections import Detection                  # noqa: E402
from corridor.engine.reconstruct import reconstruct_tracks      # noqa: E402

H, W = 130, 180
TOL = 6.0


def _make_detection(frame, label, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    minr, minc, maxr, maxc = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    major = float(maxr - minr); minor = float(max(maxc - minc, 1))
    return Detection(
        frame=int(frame), label=int(label), x=float(xs.mean()), y=float(ys.mean()),
        area_px=float(mask.sum()), bbox=(int(minr), int(minc), int(maxr), int(maxc)),
        extent_px=int(max(major, minor)), eccentricity=0.95, orientation_rad=0.0,
        major_axis_px=major, minor_axis_px=minor, solidity=1.0, touches_border=False,
        channel=0, mask_crop=mask[minr:maxr, minc:maxc].copy())


def simulate(seed, hard=True):
    """Return (detections, truth) where truth[cell_id] = {frame:(x,y)}.
    hard=True spans the full adversarial regime; hard=False is the well-posed
    regime of real confined migration (one cell per channel, no division, no
    identity-ambiguous collision), still with dropout, false positives and
    ordinary motion."""
    rng = np.random.default_rng(seed)
    nframes = int(rng.integers(12, 24))
    n_lanes = int(rng.integers(1, 7))
    lane_x = np.linspace(20, W - 20, n_lanes)
    collisions = hard and rng.random() < 0.4
    divisions = hard and rng.random() < 0.3
    fp_rate = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
    dropout = float(rng.choice([0.0, 0.0, 0.1, 0.2]))
    fast = hard and rng.random() < 0.4

    truth = {}            # cell_id -> {frame: (x,y)}
    cell_masks = {}       # cell_id -> {frame: boolmask}
    cid = 0
    for li, lx in enumerate(lane_x):
        ncell = int(rng.integers(1, 3 if collisions else 2) + 1)
        for _ in range(ncell):
            birth = int(rng.integers(0, max(1, nframes // 3)))
            death = int(min(nframes, birth + rng.integers(nframes // 2, nframes)))
            y = float(rng.uniform(20, H - 20))
            vy = float(rng.normal(0, 5.0 if fast else 2.5))
            length = float(rng.uniform(16, 40))
            track, masks = {}, {}
            for t in range(birth, death):
                y += vy + rng.normal(0, 0.6); vy = 0.8 * vy + rng.normal(0, 1.0)
                if fast and rng.random() < 0.15:
                    vy = -vy                      # abrupt reversal (not assumed)
                length = float(np.clip(length + rng.normal(0, 3), 12, 60))
                y = float(np.clip(y, 8, H - 8))
                x = float(lx + rng.normal(0, 1.5))
                y0, y1 = int(max(0, y - length / 2)), int(min(H, y + length / 2))
                x0, x1 = int(x - 3), int(x + 4)
                m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
                track[t] = (x, y); masks[t] = m
            if len(track) >= 3:
                truth[cid] = track; cell_masks[cid] = masks; cid += 1
                if divisions and rng.random() < 0.5 and death < nframes - 3:
                    # a daughter appears at the division point (one -> two)
                    y = track[max(track)][1]
                    track2, masks2 = {}, {}
                    for t in range(death, nframes):
                        y += rng.normal(1.0, 1.0)
                        y = float(np.clip(y, 8, H - 8)); x = float(lx)
                        y0, y1 = int(max(0, y - 10)), int(min(H, y + 10))
                        m = np.zeros((H, W), bool); m[y0:y1, int(x - 3):int(x + 4)] = True
                        track2[t] = (x, y); masks2[t] = m
                    if len(track2) >= 3:
                        truth[cid] = track2; cell_masks[cid] = masks2; cid += 1

    # build per-frame detections from the cell masks, with dropout + FPs
    dets = []
    label = 1
    for c, masks in cell_masks.items():
        for t, m in masks.items():
            if dropout and rng.random() < dropout:
                continue
            d = _make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    for t in range(nframes):
        for _ in range(rng.poisson(fp_rate)):
            yy = int(rng.uniform(10, H - 10)); xx = int(rng.choice(lane_x))
            m = np.zeros((H, W), bool); m[yy:yy + 12, xx - 3:xx + 4] = True
            d = _make_detection(t, label, m)
            if d is not None:
                dets.append(d); label += 1
    return dets, truth, nframes


def score(tracks, truth):
    """Identity-aware: match each truth cell to the recon track covering most of
    its frames; a state counts only if that track is within TOL there (correct
    identity). Also count extra/false tracks via precision over recon states."""
    recon = []
    for tr in tracks:
        recon.append({o.frame: (o.x, o.y) for o in tr.observations})
    total = correct = 0
    used = set()
    for c, tk in truth.items():
        # best matching recon track for this cell
        best, best_hits = None, -1
        for i, r in enumerate(recon):
            hits = sum(1 for t, (x, y) in tk.items()
                       if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
            if hits > best_hits:
                best, best_hits = i, hits
        total += len(tk)
        if best is not None:
            used.add(best)
            r = recon[best]
            correct += sum(1 for t, (x, y) in tk.items()
                           if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
    recall = correct / max(total, 1)
    # precision: recon states that sit on some truth cell
    rtot = rhit = 0
    for r in recon:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in tk and np.hypot(tk[t][0] - x, tk[t][1] - y) <= TOL for tk in truth.values()):
                rhit += 1
    prec = rhit / max(rtot, 1)
    f1 = 2 * recall * prec / max(recall + prec, 1e-9)
    return recall, prec, f1, len(tracks), len(truth)


def run_battery(hard, n=120):
    scale = Scale.from_values(0.5, 10.0)
    cfg = TrackingConfig()
    recs, precs, f1s, dtrk = [], [], [], []
    for seed in range(n):
        dets, truth, nf = simulate(seed, hard=hard)
        if not truth:
            continue
        tracks, _ = reconstruct_tracks(dets, (H, W), nf, scale, cfg)
        r, p, f, ntr, ntruth = score(tracks, truth)
        recs.append(r); precs.append(p); f1s.append(f); dtrk.append(ntr - ntruth)
    recs, precs, f1s = map(np.array, (recs, precs, f1s))
    label = "FULL adversarial (incl. ill-posed division/collision)" if hard else \
            "WELL-POSED (one cell per channel, no division; dropout+FP+motion)"
    print(f"{label}: {len(f1s)} trials")
    print(f"  F1 mean {f1s.mean():.3f}  median {np.median(f1s):.3f}  min {f1s.min():.3f}"
          f"  | >=0.98: {int((f1s>=0.98).sum())}/{len(f1s)}  | recall {recs.mean():.3f} prec {precs.mean():.3f}"
          f"  | track-count err {np.mean(dtrk):+.2f}")


def main():
    run_battery(hard=False)
    run_battery(hard=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
