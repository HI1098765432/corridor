"""Hard-code detection, step 1, made precise by TIME.

Background subtraction finds the faint cells (recall ~0.98) but floods noise
blobs (low precision). A single frame cannot tell a faint cell from noise -- they
are the same brightness. TIME can: a real cell is a connected object that moves
smoothly along its lane across many frames; a noise blob is not. So link every
bg-detection into per-lane tracks and keep only tracks that are temporally
coherent cells (enough frames, smooth motion). Measure whether that recovers
precision without losing the faint cells. Real 122324 vs GT.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "training/benchmark")
import bg_detector as BD                                          # noqa: E402


def link(dets, lanes, max_gap=2):
    lane_of = lambda x: int(np.argmin([abs(x - l) for l in lanes]))
    by = {}
    for t, fr in enumerate(dets):
        for x, y in fr:
            by.setdefault(lane_of(x), {}).setdefault(t, []).append((x, y))
    tracks = []
    for L, byf in by.items():
        active = []
        for t in sorted(byf):
            cands = sorted(byf[t], key=lambda p: p[1]); used = set()
            for rec in active:
                if t - rec[1] > max_gap:
                    continue
                best, bd = None, None
                for j, (x, y) in enumerate(cands):
                    if j in used:
                        continue
                    d = abs(y - rec[2])
                    if bd is None or d < bd:
                        bd, best = d, j
                if best is not None:
                    x, y = cands[best]; rec[0][t] = (x, y); rec[1] = t; rec[2] = y; used.add(best)
            for j, (x, y) in enumerate(cands):
                if j not in used:
                    active.append([{t: (x, y)}, t, y])
        tracks.extend(rec[0] for rec in active)
    return tracks


def features(tr):
    ts = sorted(tr); ys = np.array([tr[t][1] for t in ts]); xs = np.array([tr[t][0] for t in ts])
    n = len(ts); span = ts[-1] - ts[0] + 1
    net = abs(ys[-1] - ys[0]); path = np.abs(np.diff(ys)).sum() if n > 1 else 0
    # jitter: RMS residual of y from a straight line in time (a migrating cell is near-linear)
    if n >= 3:
        fit = np.polyval(np.polyfit(ts, ys, 1), ts); jit = float(np.sqrt(np.mean((ys - fit) ** 2)))
    else:
        jit = 0.0
    return dict(n=n, span=span, net=net, path=path, straight=net / max(path, 1), jitter=jit, xstd=float(xs.std()))


def is_real(tr, gc, tol=14.0):
    hits = 0
    for t, (x, y) in tr.items():
        if any(np.hypot(x - gx, y - gy) <= tol for gx, gy in gc[t]):
            hits += 1
    return hits >= max(2, 0.5 * len(tr))


def main():
    imgs, gts = BD.load()
    gc = [BD.cents(g) for g in gts]
    from corridor.core.config import RunConfig, Scale
    from corridor.core import model_registry as mr
    from corridor.core.segmentation import SegmentationService
    seg = SegmentationService(RunConfig().segmentation, model=mr.resolve_model(),
                              scale=Scale.from_values(0.467, 20.0)).run_stack(np.stack(imgs))
    pc = [BD.cents(seg.masks[t]) for t in range(len(gts))]
    lanes = BD.lanes_of(pc)
    bd = BD.detect(imgs, lanes, k=4.0)
    u = BD.union(pc, bd)
    tracks = [t for t in link(u, lanes) if len(t) >= 2]
    real = [tr for tr in tracks if is_real(tr, gc)]; fake = [tr for tr in tracks if not is_real(tr, gc)]
    print(f"bg+model union tracks: {len(tracks)}  ({len(real)} real, {len(fake)} noise)")
    def summ(name, grp, key):
        v = [features(t)[key] for t in grp]
        return f"{name} {key}: median {np.median(v):.1f} [{np.percentile(v,10):.1f}-{np.percentile(v,90):.1f}]"
    for key in ("n", "span", "net", "straight", "jitter"):
        print("   ", summ("REAL", real, key), "||", summ("noise", fake, key))
    # try a coherence filter and measure recall/precision of the KEPT detections
    print("\n filter sweep (keep tracks passing the rule):")
    for name, rule in [
        ("n>=3", lambda f: f["n"] >= 3),
        ("n>=4", lambda f: f["n"] >= 4),
        ("n>=3 & net>=20", lambda f: f["n"] >= 3 and f["net"] >= 20),
        ("n>=3 & straight>=0.5", lambda f: f["n"] >= 3 and f["straight"] >= 0.5),
        ("n>=4 & straight>=0.6 & net>=15", lambda f: f["n"] >= 4 and f["straight"] >= 0.6 and f["net"] >= 15),
    ]:
        kept = [tr for tr in tracks if rule(features(tr))]
        pts = [[] for _ in range(len(gts))]
        for tr in kept:
            for t, (x, y) in tr.items():
                pts[t].append((x, y))
        r, p, f1, tp, gt = BD.recall_prec(pts, gc)
        print(f"   {name:>28}: recall {r:.3f} prec {p:.3f} F1 {f1:.3f} ({tp}/{gt})")


if __name__ == "__main__":
    main()
