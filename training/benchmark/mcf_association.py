"""Global min-cost-flow data association for confined cells (successive shortest
paths). Each detection is used at most once (uniqueness enforced by capacity 1),
so two distinct cells are kept separate when that explains the data more cheaply
than fusing them -- which is what blind/appearance bridging got wrong.

Transition cost is confinement-aware: same channel (x stable) + smooth y-motion
+ appearance (1-NCC). Reward/enter/exit set the minimum worthwhile track length.
Measured on real masks vs eye-verified truth, plus physics (wall-crossings).
"""
from __future__ import annotations
import numpy as np

V_MAX = 30.0        # px/frame along channel
LANE_TOL = 25.0     # px: same-channel gate on x (cells are x-stable)
# Cost balance: starting a track must cost MORE than including a detection, so a
# lone/short track is unprofitable and the flow prefers few long tracks; linking
# must be cheap, and appearance is only a weak tie-breaker (a confined cell's
# look changes as it elongates, so a low NCC must NOT block a link).
R_DET = 6.0         # reward for including a real detection
ENTER = EXIT = 5.0  # cost to start/end a track (lone det: 5+5-6 = +4 > 0, rejected)
W_MOVE, W_APP, W_GAP = 0.02, 0.5, 1.0
INF = 1e9


def _crop(images, t, x, y, r=9):
    H, W = images.shape[1:]
    xi, yi = int(round(x)), int(round(y))
    if r <= yi < H - r and r <= xi < W - r:
        return images[t, yi - r:yi + r, xi - r:xi + r]
    return None


def _ncc(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _overlap(a, b):
    inter = np.count_nonzero(a & b)
    if not inter:
        return 0.0
    return max(inter / np.count_nonzero(a | b), inter / max(min(a.sum(), b.sum()), 1))


def transition(d_i, d_j, images):
    ti, xi, yi, mi = d_i
    tj, xj, yj, mj = d_j
    dt = tj - ti
    if dt < 1 or dt > 4:
        return None
    if abs(xi - xj) > LANE_TOL:              # must stay in the same channel
        return None
    if dt == 1:
        # adjacent frames: link by OVERLAP (containment-robust to elongation),
        # NOT centroid -- an elongating cell's centroid lurches up to ~50 px.
        ov = _overlap(mi, mj)
        if ov < 0.1:
            return None
        return W_MOVE * (1.0 - ov)
    # a real gap (dropout): overlap is impossible, so gate on channel + speed and
    # pay a gap price; appearance is a weak tie-break only.
    if abs(yi - yj) / dt > V_MAX:
        return None
    app = 1.0 - max(_ncc(_crop(images, ti, xi, yi), _crop(images, tj, xj, yj)), 0.0)
    return W_MOVE * abs(yi - yj) / dt + W_APP * app + W_GAP * (dt - 1)


def bellman_ford(n, edges, src, dst):
    """Shortest path with negative edges. edges: list of (u,v,cost,cap,flow,rev_idx)."""
    dist = [INF] * n
    pre = [-1] * n
    dist[src] = 0.0
    adj = [[] for _ in range(n)]
    for idx, (u, v, c, cap, flow, rev) in enumerate(edges):
        adj[u].append(idx)
    for _ in range(n):
        changed = False
        for u in range(n):
            if dist[u] >= INF:
                continue
            for idx in adj[u]:
                v, c, cap, flow = edges[idx][1], edges[idx][2], edges[idx][3], edges[idx][4]
                if cap - flow > 0 and dist[u] + c < dist[v] - 1e-12:
                    dist[v] = dist[u] + c
                    pre[v] = idx
                    changed = True
        if not changed:
            break
    return dist, pre


def mcf_tracks(dets, images):
    """dets: list of (t, x, y, boolmask). Returns list of tracks, each a list of
    detection indices, via successive shortest paths."""
    D = list(dets)
    nd = len(D)
    if nd == 0:
        return []
    S, T = 2 * nd, 2 * nd + 1
    n = 2 * nd + 2
    edges = []

    def add(u, v, c, cap):
        edges.append([u, v, c, cap, 0, len(edges) + 1])
        edges.append([v, u, -c, 0, 0, len(edges) - 1])

    for i in range(nd):
        add(S, 2 * i, ENTER, 1)              # enter
        add(2 * i, 2 * i + 1, -R_DET, 1)     # include detection i (reward)
        add(2 * i + 1, T, EXIT, 1)           # exit
    for i in range(nd):
        for j in range(nd):
            if i == j:
                continue
            c = transition(D[i], D[j], images)
            if c is not None and D[j][0] > D[i][0]:
                add(2 * i + 1, 2 * j, c, 1)   # i -> j
    # successive shortest paths while the cheapest S->T path is negative
    while True:
        dist, pre = bellman_ford(n, edges, S, T)
        if dist[T] >= -1e-9:
            break
        v = T
        while v != S:
            idx = pre[v]
            edges[idx][4] += 1
            edges[idx ^ 1][4] -= 1
            v = edges[idx][0]
    # reconstruct trajectories from detection-edge flow and i->j flow
    nxt = {}
    for idx in range(0, len(edges), 2):
        u, v, c, cap, flow, rev = edges[idx]
        if flow > 0 and u % 2 == 1 and v % 2 == 0 and v not in (S, T):
            nxt[(u - 1) // 2] = v // 2
    used_as_next = set(nxt.values())
    starts = []
    for i in range(nd):
        # i is included if its detection edge carries flow
        det_edge = [e for e in edges if e[0] == 2 * i and e[1] == 2 * i + 1][0]
        if det_edge[4] > 0 and i not in used_as_next:
            starts.append(i)
    tracks = []
    for s in starts:
        tr = [s]
        while tr[-1] in nxt:
            tr.append(nxt[tr[-1]])
        tracks.append(tr)
    return tracks
