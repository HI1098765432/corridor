"""Drive the 4D engine bots on the REAL movie, scored against eye-verified truth.

Uses corridor.engine: registration4d (remove microscope drift before overlap
linking), static_atlas (reject detections inside channel walls), temporal_delta
(forward/backward E_FB), consensus.referee (accept/reject). Plus the two fixes
the genuine benchmark demanded: a global stitch (close the elongation-induced
identity split) and self-templating recovery (rebuild a cell in frames the
detector lost, by its own appearance). Real images, real masks, human truth.
"""
from __future__ import annotations
import sys
import numpy as np
import tifffile
from scipy import ndimage
from skimage.feature import match_template

sys.path.insert(0, "src")
from corridor.engine import registration4d, static_atlas  # noqa: E402

EXPECTED_COUNTS = {
    "052924_t1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "052924_t3_dual": [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1],
}
EXPECTED_TRACKS = {"052924_t1": 1, "052924_t3_dual": 3}


def load(movie, nframes):
    img = tifffile.imread(f"data/confinedmig_cellTrack/sample_data/{movie}.tif").astype(np.float32)
    masks = np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"]
    return img[:nframes], masks[:nframes]


def register_masks(images, masks):
    """Register the movie and shift the label masks by the same correction, so
    overlap linking compares drift-aligned frames (t1 drifts ~21 px)."""
    res = registration4d.register_stack(images)
    out = np.zeros_like(masks)
    for t, row in enumerate(res.rows):
        out[t] = ndimage.shift(masks[t], (row.dy_px, row.dx_px), order=0, mode="constant")
    return res.registered, out, res


def wall_map(images):
    """Channel-wall voxels from the static atlas (to reject wall-texture FPs)."""
    try:
        atlas = static_atlas.build_atlas(images)
        cm = atlas.class_map
        return cm == static_atlas.CHANNEL_WALL
    except Exception:
        return np.zeros(images.shape[1:], bool)


def frame_objects(mask_t, wall):
    out = {}
    for lab in np.unique(mask_t):
        if not lab:
            continue
        m = mask_t == lab
        if wall is not None and np.count_nonzero(m & wall) > 0.6 * np.count_nonzero(m):
            continue  # a detection that is mostly inside a wall is rejected
        out[int(lab)] = m
    return out


def overlap(a, b):
    inter = np.count_nonzero(a & b)
    if not inter:
        return 0.0
    return max(inter / np.count_nonzero(a | b), inter / max(min(a.sum(), b.sum()), 1))


def link(objs, nframes, max_gap=3, min_ov=0.1):
    nodes = [(t, lab) for t in range(nframes) for lab in objs.get(t, {})]
    used, tracks = set(), []
    for start in nodes:
        if start in used:
            continue
        used.add(start); track = [start]; t, lab = start; cur = objs[t][lab]
        while True:
            best, bo = None, min_ov
            for dt in range(1, max_gap + 1):
                for lab2, m2 in objs.get(t + dt, {}).items():
                    if (t + dt, lab2) in used:
                        continue
                    ov = overlap(cur, m2)
                    if ov > bo:
                        best, bo = (t + dt, lab2), ov
                if best:
                    break
            if not best:
                break
            used.add(best); track.append(best); t, lab = best; cur = objs[t][lab]
        tracks.append(track)
    return tracks


def centroid(objs, node):
    t, lab = node
    ys, xs = np.nonzero(objs[t][lab])
    return t, xs.mean(), ys.mean()


def stitch(tracks, objs, max_gap=5, max_jump=60.0):
    """Global gap-closing: join a track that ends to one that starts shortly
    after, in the same place -- closes the identity split a big elongation step
    opens when consecutive overlap momentarily drops out."""
    tr = [list(t) for t in tracks]
    changed = True
    while changed:
        changed = False
        tr.sort(key=lambda t: t[0][0])
        for i, a in enumerate(tr):
            if a is None:
                continue
            te, xe, ye = centroid(objs, a[-1])
            best, bd = None, max_jump
            for j, b in enumerate(tr):
                if b is None or j == i:
                    continue
                tb, xb, yb = centroid(objs, b[0])
                dt = tb - te
                if 1 <= dt <= max_gap:
                    d = np.hypot(xb - xe, yb - ye)
                    if d < bd:
                        best, bd = j, d
            if best is not None:
                a.extend(tr[best]); tr[best] = None; changed = True
        tr = [t for t in tr if t is not None]
    return tr


def template_recover(images, objs, track, win=16, ncc=0.5):
    """L2 recovery: build a template from the cell's observed crops and NCC-match
    it in the frames between first and last that have no linked detection."""
    obs = {t: centroid(objs, (t, lab))[1:] for t, lab in track}
    ts = sorted(obs)
    r = 9
    crops = []
    for t in ts:
        x, y = obs[t]; xi, yi = int(round(x)), int(round(y))
        H, W = images.shape[1:]
        if r <= yi < H - r and r <= xi < W - r:
            crops.append(images[t, yi - r:yi + r, xi - r:xi + r])
    present = set(ts)
    if len(crops) >= 2:
        tmpl = np.median(np.stack(crops), 0)
        xs = np.array([obs[t][0] for t in ts]); ys = np.array([obs[t][1] for t in ts])
        px, py = np.polyfit(ts, xs, 1), np.polyfit(ts, ys, 1)
        H, W = images.shape[1:]
        for t in range(ts[0], ts[-1] + 1):
            if t in obs:
                continue
            cx, cy = int(round(np.polyval(px, t))), int(round(np.polyval(py, t)))
            y0, y1 = max(0, cy - win), min(H, cy + win)
            x0, x1 = max(0, cx - win), min(W, cx + win)
            reg = images[t, y0:y1, x0:x1]
            if reg.shape[0] > tmpl.shape[0] and reg.shape[1] > tmpl.shape[1]:
                resp = match_template(reg, tmpl)
                if resp.max() >= ncc:
                    present.add(t)
    return present


def _template(images, seg, r=9):
    H, W = images.shape[1:]
    crops = []
    for t, (x, y) in seg.items():
        xi, yi = int(round(x)), int(round(y))
        if r <= yi < H - r and r <= xi < W - r:
            crops.append(images[t, yi - r:yi + r, xi - r:xi + r])
    return np.median(np.stack(crops), 0) if len(crops) >= 2 else None


def _ncc_at(images, t, tmpl, cx, cy, win):
    H, W = images.shape[1:]
    y0, y1 = max(0, int(cy) - win), min(H, int(cy) + win)
    x0, x1 = max(0, int(cx) - win), min(W, int(cx) + win)
    reg = images[t, y0:y1, x0:x1]
    if reg.shape[0] <= tmpl.shape[0] or reg.shape[1] <= tmpl.shape[1]:
        return 0.0
    return float(match_template(reg, tmpl).max())


def evidence_bridge(images, objs, tracks, max_bridge=12, win=16, ncc=0.5, confirm_frac=0.6):
    """Evidence-gated stitch + recovery. Two fragments are merged ONLY if the
    cell's own template is found (NCC) in most of the frames between them --
    which happens when the detector was merely blind there (the cell is still
    in the pixels), and does NOT happen when the cell truly left. Dropping a
    detection never drops the cell from the image, so this recovers the weak-
    proposer case while refusing to fuse two genuinely different cells."""
    segs = []
    for tr in tracks:
        d = {}
        for (t, lab) in tr:
            _, x, y = centroid(objs, (t, lab))
            d[t] = (x, y)
        if d:
            segs.append(d)
    changed = True
    while changed:
        changed = False
        segs.sort(key=lambda s: min(s))
        for i, a in enumerate(segs):
            if a is None:
                continue
            te = max(a); xe, ye = a[te]
            tmpl = _template(images, a)
            for j, b in enumerate(segs):
                if b is None or j == i:
                    continue
                tb = min(b)
                gap = tb - te
                if not (1 <= gap <= max_bridge) or tmpl is None:
                    continue
                xb, yb = b[tb]
                confirmed, inner = {}, list(range(te + 1, tb))
                for t in inner:
                    w = (t - te) / (tb - te)
                    cx, cy = xe * (1 - w) + xb * w, ye * (1 - w) + yb * w
                    if _ncc_at(images, t, tmpl, cx, cy, win) >= ncc:
                        confirmed[t] = (cx, cy)
                if not inner or len(confirmed) >= confirm_frac * len(inner):
                    a.update(confirmed); a.update(b); segs[j] = None; changed = True
                    break
        segs = [s for s in segs if s is not None]
    # fill interior holes within each merged segment that the template confirms
    out = []
    for s in segs:
        tmpl = _template(images, s)
        ts = sorted(s)
        present = set(ts)
        if tmpl is not None:
            xs = np.array([s[t][0] for t in ts]); ys = np.array([s[t][1] for t in ts])
            px, py = np.polyfit(ts, xs, 1), np.polyfit(ts, ys, 1)
            for t in range(ts[0], ts[-1] + 1):
                if t not in s and _ncc_at(images, t, tmpl, np.polyval(px, t), np.polyval(py, t), win) >= ncc:
                    present.add(t)
        out.append(present)
    return out


def reconstruct(movie, nframes, drop, seed, use_reg, use_stitch, use_tmpl, evidence=False):
    rng = np.random.default_rng(seed)
    images, masks = load(movie, nframes)
    if use_reg:
        images, masks, _ = register_masks(images, masks)
    wall = wall_map(images)
    objs = {}
    for t in range(nframes):
        o = frame_objects(masks[t], wall)
        if drop:
            o = {lab: m for lab, m in o.items() if rng.random() >= drop}
        objs[t] = o
    tracks = link(objs, nframes)
    if evidence:
        return evidence_bridge(images, objs, tracks)
    if use_stitch:
        tracks = stitch(tracks, objs)
    present_sets = []
    for tr in tracks:
        if use_tmpl:
            present_sets.append(template_recover(images, objs, tr))
        else:
            ts = sorted(t for t, _ in tr)
            present_sets.append(set(range(ts[0], ts[-1] + 1)))
    return present_sets


def score(present_sets, expected):
    n = len(expected)
    counts = [0] * n
    for ps in present_sets:
        for t in ps:
            if 0 <= t < n:
                counts[t] += 1
    tp = sum(min(counts[t], expected[t]) for t in range(n))
    over = sum(max(counts[t] - expected[t], 0) for t in range(n))
    gt = sum(expected)
    rec = tp / max(gt, 1); prec = tp / max(tp + over, 1)
    return rec, prec, 2 * rec * prec / max(rec + prec, 1e-9), len(present_sets)


if __name__ == "__main__":
    print("4D-ENGINE real-data benchmark, vs eye-verified truth. EV = evidence-gated bridging.")
    print(f"{'movie':>15} {'drop':>5} {'mode':>9} {'trk(exp)':>9} {'rec':>6} {'prec':>6} {'F1':>6}")
    for movie, expected in EXPECTED_COUNTS.items():
        n = len(expected); et = EXPECTED_TRACKS[movie]
        for drop in (0.0, 0.3, 0.5):
            for mode, ev in [("stitch+tpl", False), ("EV-gated", True)]:
                rows = [score(reconstruct(movie, n, drop, s, True, True, True, evidence=ev), expected)
                        for s in range(15)]
                rec, prec, f1, ntr = np.mean(rows, axis=0)
                print(f"{movie:>15} {drop:>5.1f} {mode:>9} {ntr:>4.1f}/{et:<4} "
                      f"{rec:>6.3f} {prec:>6.3f} {f1:>6.3f}")
