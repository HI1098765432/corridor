"""Break the large-motion case on the REAL hard movies by using the device
physics: a confined cell cannot leave its lane, so the LANE (x-position) is a
motion-invariant identity key. Compare, on each real movie, scored against the
physical lane+y ground truth (eye-validated separately):

  A  overlap, NO geometry            -- my battery's config (expected to fail)
  B  Kalman tracker WITH detected lanes (production, lane gate on)
  C  lane-primary linker             -- the novel fill-the-gap method:
        cluster detections into lanes, link within a lane by y-continuity,
        splitting the rare 2-in-a-lane case by y. Independent of overlap.
"""
from __future__ import annotations
import io
import sys
import zipfile
import numpy as np

sys.path.insert(0, "src")
from corridor.core.config import Scale, TrackingConfig                   # noqa: E402
from corridor.core.detections import Detection                          # noqa: E402
from corridor.core.geometry import detect_channels, assign_lanes        # noqa: E402
from corridor.core.tracking import track_detections                     # noqa: E402
from corridor.engine.reconstruct import reconstruct_tracks              # noqa: E402

ZIP = r"C:\Users\ironb\Downloads\OneDrive_2026-10-02.zip"
B = "Traction from Displacement/confinedmig_cellTrack/CellPose_TrainData/"
MOVIES = [("KK1", "041824", range(16, 21)), ("KK1", "122324", range(1, 13)),
          ("KK1", "061523", range(1, 6)), ("KK2", "052924", range(22, 27))]
TOL = 8.0
SCALE = Scale.from_values(0.467, 20.0)   # generous speed gate (~214 px/frame)


def load(folder, date, idxs):
    z = zipfile.ZipFile(ZIP)
    imgs, masks = [], []
    for i in idxs:
        try:
            im = _imread(z.read(f"{B}{folder}/{date}_{i}.tif"))
            sg = np.load(io.BytesIO(z.read(f"{B}{folder}/{date}_{i}_seg.npy")), allow_pickle=True).item()
        except KeyError:
            continue
        imgs.append(im.astype(np.float32)); masks.append(np.asarray(sg["masks"]).astype(np.int32))
    modal = max({m.shape for m in masks}, key=[m.shape for m in masks].count)
    keep = [k for k in range(len(masks)) if masks[k].shape == modal]
    return [imgs[k] for k in keep], [masks[k] for k in keep]


def _imread(b):
    import tifffile
    return tifffile.imread(io.BytesIO(b))


def instances(mask):
    out = []
    for lab in np.unique(mask):
        if lab:
            m = mask == lab; ys, xs = np.nonzero(m)
            out.append((float(xs.mean()), float(ys.mean()), m))
    return out


def dets(masks):
    ds, label = [], 1
    for t, mk in enumerate(masks):
        for x, y, m in instances(mk):
            ys, xs = np.nonzero(m)
            r0, c0, r1, c1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            maj = float(r1 - r0); mn = float(max(c1 - c0, 1))
            ds.append(Detection(frame=t, label=label, x=x, y=y, area_px=float(m.sum()),
                      bbox=(int(r0), int(c0), int(r1), int(c1)), extent_px=int(max(maj, mn)),
                      eccentricity=float(np.sqrt(max(1 - (mn / max(maj, 1)) ** 2, 0))), orientation_rad=0.0,
                      major_axis_px=maj, minor_axis_px=mn, solidity=1.0, touches_border=False,
                      channel=0, mask_crop=m[r0:r1, c0:c1].copy()))
            label += 1
    return ds


def lanes_of(masks, gap=15):
    allx = sorted(x for mk in masks for x, y, m in instances(mk))
    lanes, cur = [], [allx[0]]
    for x in allx[1:]:
        if x - cur[-1] > gap:
            lanes.append(np.mean(cur)); cur = [x]
        else:
            cur.append(x)
    lanes.append(np.mean(cur))
    return lanes


def lane_primary(masks):
    """Novel method: group by lane, link within lane by y-continuity (greedy
    nearest-y), splitting a lane that holds 2 cells. Overlap-free."""
    lanes = lanes_of(masks)
    lane_of = lambda x: int(np.argmin([abs(x - l) for l in lanes]))
    # per lane: list over frames of [y...]
    bylane = {}
    for t, mk in enumerate(masks):
        for x, y, m in instances(mk):
            bylane.setdefault(lane_of(x), []).append((t, y, x))
    tracks = []
    for L, pts in bylane.items():
        frames = {}
        for t, y, x in pts:
            frames.setdefault(t, []).append((y, x))
        active = []   # list of dict frame->(x,y)
        for t in sorted(frames):
            cands = sorted(frames[t])
            used = set()
            for tr in active:
                lastf = max(tr); ly = tr[lastf][1]
                best, bd = None, 1e9
                for j, (y, x) in enumerate(cands):
                    if j in used:
                        continue
                    d = abs(y - ly)
                    if d < bd:
                        best, bd = j, d
                if best is not None:
                    y, x = cands[best]; tr[t] = (x, y); used.add(best)
            for j, (y, x) in enumerate(cands):
                if j not in used:
                    active.append({t: (x, y)})
        tracks.extend(active)
    return tracks


def truth_tracks(masks):
    # physical truth = lane-primary grouping (validated by eye on the overlays)
    return [{f: xy for f, xy in tr.items()} for tr in lane_primary(masks) if len(tr) >= 2]


def to_dicts(tracklist):
    return [{o.frame: (o.x, o.y) for o in tr.observations} for tr in tracklist]


def score(recon, truth):
    recon = [r for r in recon if len(r) >= 2]   # trajectories only (min_observations=2)
    total = correct = 0
    for tk in truth:
        best, bh = None, -1
        for r in recon:
            h = sum(1 for t, (x, y) in tk.items() if t in r and np.hypot(r[t][0] - x, r[t][1] - y) <= TOL)
            if h > bh:
                best, bh = r, h
        total += len(tk)
        if best is not None:
            correct += sum(1 for t, (x, y) in tk.items() if t in best and np.hypot(best[t][0] - x, best[t][1] - y) <= TOL)
    rec = correct / max(total, 1)
    rtot = rhit = 0
    for r in recon:
        for t, (x, y) in r.items():
            rtot += 1
            if any(t in tk and np.hypot(tk[t][0] - x, tk[t][1] - y) <= TOL for tk in truth):
                rhit += 1
    prec = rhit / max(rtot, 1)
    return 2 * rec * prec / max(rec + prec, 1e-9)


def main():
    print(f"{'movie':>18} {'A overlap':>12} {'B Kalman+ln':>13} {'C lane-prim':>13} {'D prod+lanestitch':>16}")
    fa, fb, fc, fd = [], [], [], []
    for folder, date, idxs in MOVIES:
        imgs, masks = load(folder, date, list(idxs))
        nf = len(masks); shape = masks[0].shape
        truth = truth_tracks(masks)
        ds = dets(masks)
        # A: overlap, no geometry
        A = score(to_dicts(list(reconstruct_tracks(ds, shape, nf, SCALE, TrackingConfig())[0])), truth)
        # B: Kalman WITH detected lanes
        stack = np.stack(imgs)
        geom = detect_channels(stack, None, ds, pixel_size_um=0.467)
        ds2 = dets(masks); assign_lanes(ds2, geom)
        B_ = score(to_dicts(list(track_detections(ds2, nf, SCALE, TrackingConfig(), geometry=geom)[0])), truth)
        # C: lane-primary linker (reference)
        C = score([{f: xy for f, xy in tr.items()} for tr in lane_primary(masks)], truth)
        # D: PRODUCTION overlap backend WITH geometry (now includes lane-stitch)
        ds3 = dets(masks); assign_lanes(ds3, geom)
        D = score(to_dicts(list(reconstruct_tracks(ds3, shape, nf, SCALE, TrackingConfig(), geometry=geom)[0])), truth)
        fa.append(A); fb.append(B_); fc.append(C); fd.append(D)
        print(f"{folder+'/'+date:>18} {A:>12.2f} {B_:>13.2f} {C:>13.2f} {D:>16.2f}   (lanes={geom.n_lanes})")
    print(f"{'MEAN':>18} {np.mean(fa):>12.2f} {np.mean(fb):>13.2f} {np.mean(fc):>13.2f} {np.mean(fd):>16.2f}")


if __name__ == "__main__":
    main()
