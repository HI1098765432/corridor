"""Production trajectory reconstruction: a drop-in for ``track_detections``.

The accuracy benchmark (``docs/ENGINE_ACCURACY.md``) established that for
confined, *elongating* cells the stable link feature is mask **overlap with
containment**, not the centroid (a stretching cell's centroid lurches up to
~50 px while it is one cell). This reconstructs trajectories by overlap
linking across small gaps, closes identity splits with a conservative stitch,
and returns the same ``(TrackList, events)`` the Kalman tracker returns, so the
pipeline can select it without any downstream change.

On the eye-verified real movies this reaches F1 1.00 with exact identity at the
detector's normal operating point (t1 1/1, t3_dual 3/3), precision 1.00.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ..core.detections import Detection
from ..core.config import Scale, TrackingConfig
from ..core.geometry import ChannelGeometry
from ..core.tracking import FrameEvent, Track, TrackList, TrackState


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """IoU, floored by containment: a short mask fully inside a long one scores
    high even when IoU is low, which is what an elongating cell does."""
    inter = int(np.count_nonzero(a & b))
    if not inter:
        return 0.0
    union = int(np.count_nonzero(a | b))
    smaller = min(int(a.sum()), int(b.sum()))
    return max(inter / union, inter / max(smaller, 1))


def _detection_mask(d: Detection, shape) -> np.ndarray:
    """A boolean full-frame mask for one detection, from its own ``mask_crop``
    (so a recovered detection, absent from the saved label image, still links);
    a detection without a crop falls back to its bounding box."""
    if d.mask_crop is not None:
        return np.asarray(d.full_mask(shape), dtype=bool)
    m = np.zeros(shape, bool)
    r0, c0, r1, c1 = d.bbox
    m[int(r0):int(r1), int(c0):int(c1)] = True
    return m


def _masks_by_frame(detections, shape):
    """{frame: {det_label: (Detection, boolmask)}} built from the detections."""
    out: dict[int, dict[int, tuple]] = {}
    for d in detections:
        out.setdefault(int(d.frame), {})[int(d.label)] = (d, _detection_mask(d, shape))
    return out


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _link(objs, n_frames, max_gap, min_overlap):
    nodes = [(t, lab) for t in range(n_frames) for lab in objs.get(t, {})]
    used: set = set()
    chains = []
    for start in nodes:
        if start in used:
            continue
        used.add(start)
        chain = [start]
        t, lab = start
        cur = objs[t][lab][1]
        while True:
            best, best_ov = None, min_overlap
            for dt in range(1, max_gap + 1):
                for lab2, (_, m2) in objs.get(t + dt, {}).items():
                    if (t + dt, lab2) in used:
                        continue
                    ov = _overlap(cur, m2)
                    if ov > best_ov:
                        best, best_ov = (t + dt, lab2), ov
                if best is not None:
                    break
            if best is None:
                break
            used.add(best)
            chain.append(best)
            t, lab = best
            cur = objs[t][lab][1]
        chains.append(chain)
    return chains


def _stitch(chains, objs, max_gap, max_jump):
    """Close an identity split opened by a momentary overlap dropout: join a
    chain that ends to one that starts shortly after, in the same place. Held
    conservative (small gap) so two genuinely distinct cells are never fused."""
    chains = [list(c) for c in chains]
    changed = True
    while changed:
        changed = False
        chains.sort(key=lambda c: c[0][0])
        for i, a in enumerate(chains):
            if a is None:
                continue
            te, le = a[-1]
            xe, ye = _centroid(objs[te][le][1])
            best, bd = None, max_jump
            for j, b in enumerate(chains):
                if b is None or j == i:
                    continue
                tb, lb = b[0]
                dt = tb - te
                if 1 <= dt <= max_gap:
                    xb, yb = _centroid(objs[tb][lb][1])
                    d = float(np.hypot(xb - xe, yb - ye))
                    if d < bd:
                        best, bd = j, d
            if best is not None:
                a.extend(chains[best])
                chains[best] = None
                changed = True
        chains = [c for c in chains if c is not None]
    return chains


def _chain_channel(chain, objs) -> int | None:
    """The channel a chain sits in: the majority of its detections' channels,
    ignoring the unassigned ones. None when no detection is in a lane."""
    chans = [objs[t][lab][0].channel for (t, lab) in chain]
    chans = [int(c) for c in chans if c is not None and int(c) >= 0]
    if not chans:
        return None
    return max(set(chans), key=chans.count)


def _lane_exclusive(chains, objs, max_victim: int = 3):
    """Confined-migration prior: at most one cell per lane per frame. Keeping
    chains longest-first, drop a chain that shares a frame with an already-kept
    chain in the same lane -- but *only* a brief one (at most ``max_victim``
    observations), so a genuine second cell tracked over many frames is never
    removed, while a brief debris false positive that co-occupies a lane with
    the real cell is. Measured on 271 realistic trials to raise precision
    0.94 -> 0.95 with no recall cost (docs/ENGINE_ACCURACY.md). Only ever applied
    when real lanes exist; a chain in no lane is always kept."""
    kept: list = []
    occupied: dict[int, list[set]] = {}
    for chain in sorted(chains, key=len, reverse=True):
        ch = _chain_channel(chain, objs)
        if ch is None:
            kept.append(chain)
            continue
        frames = {t for (t, _) in chain}
        occ = occupied.setdefault(ch, [])
        if len(chain) <= max_victim and any(frames & seen for seen in occ):
            continue
        occ.append(frames)
        kept.append(chain)
    return kept


def reconstruct_tracks(
    detections: Iterable[Detection],
    image_shape,
    n_frames: int,
    scale: Scale,
    config: TrackingConfig,
    *,
    geometry: ChannelGeometry | None = None,
    max_gap: int = 3,
    min_overlap: float = 0.1,
    stitch_gap: int = 5,
    stitch_jump_px: float = 60.0,
    lane_exclusive: bool = False,
    min_support: int = 1,
) -> tuple[TrackList, list[FrameEvent]]:
    """Overlap-linked reconstruction, returned as a Kalman-tracker-shaped
    ``(TrackList, events)`` so it is a drop-in for ``track_detections``.

    ``image_shape`` is the frame's ``(H, W)`` (or ``(Z, Y, X)``); masks are
    built from each detection's own ``mask_crop``, so both primary and recovered
    detections link without needing the saved label image.

    ``lane_exclusive`` (opt-in, default off) applies the confined-migration
    prior -- at most one cell per lane per frame -- and only when ``geometry``
    is present. Measured on a 271-trial realistic study (``docs/ENGINE_ACCURACY.md``)
    to raise precision 0.94 -> 0.95 on debris-heavy synthetic data; it is **off
    by default because on real data it can drop a genuine brief second cell that
    shares a lane** (it took the eye-verified t3_dual from 3 tracks to 2). It is
    offered for datasets the user knows isolate one cell per lane. ``min_support``
    (default 1, off) optionally drops tracks still shorter than it *after*
    stitching -- an opt-in for debris; the stitch runs first so a real cell's
    terminal fragment is never lost. With both at their defaults the backend is a
    faithful drop-in: it reproduces the Kalman tracker's counts on the real
    movies (t1 1 track, t3_dual 3 tracks).
    """
    detections = list(detections)
    shape = tuple(int(s) for s in image_shape)
    objs = _masks_by_frame(detections, shape)
    chains = _link(objs, n_frames, max_gap, min_overlap)
    chains = _stitch(chains, objs, stitch_gap, stitch_jump_px)
    if lane_exclusive and geometry is not None:
        chains = _lane_exclusive(chains, objs)
    if min_support > 1:
        chains = [c for c in chains if len(c) >= min_support]
    chains = [c for c in chains if c]
    chains.sort(key=lambda c: (c[0][0], _centroid(objs[c[0][0]][c[0][1]][1])[0]))

    tracks: list[Track] = []
    id_map: dict[int, int] = {}
    for new_id, chain in enumerate(chains, start=1):
        track = Track(id=new_id)
        for t, lab in chain:
            det = objs[t][lab][0]
            track.observe(det, lane=det.channel)
        track.state = TrackState.TERMINATED
        if track.observations:
            track.channel = track.observations[0].channel
        tracks.append(track)

    # events: one per frame, with the full FrameEvent surface the exporter and
    # QC read. n_matched + n_new == n_detections, as the tracker guarantees.
    events = []
    for f in range(n_frames):
        present = [tr for tr in tracks if any(o.frame == f for o in tr.observations)]
        new = sum(1 for tr in present if tr.observations[0].frame == f)
        terminated = sum(1 for tr in present if tr.observations[-1].frame == f)
        events.append(FrameEvent(
            frame=f,
            n_detections=len(objs.get(f, {})),
            n_candidates=len(present),
            n_matched=len(present) - new,
            n_new=new,
            n_dormant=0,
            n_terminated=terminated,
        ))
    return TrackList(tracks, id_map), events
