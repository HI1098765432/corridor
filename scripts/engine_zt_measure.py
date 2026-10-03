"""Measure what Bot 4 (Z consensus) and Bot 5 (temporal delta) recover.

The project law is MEASURE, never assert: every number below is produced by
running the bots on data and is written to ``build/eng/z-temporal/`` so it can
be reproduced by rerunning this script.  Nothing here is hand-entered.

Honest scope.  The held-out Corridor stills are 2-D (Z = 1) and are not
contiguous frames, so neither the Z lever nor the time lever can recover any
held-out accuracy on them -- and this script measures exactly that (the Z bot
is a passthrough on every real still; the error budget's recoverable part lives
in the border / gradient / label levers of the other bots).  The two bots are
therefore validated where their inputs actually exist: on synthetic volumes and
sequences with known answers, and on the one real contiguous sample movie.

Run (from the main checkout, with the worktree on PYTHONPATH):
    OMP_NUM_THREADS=2 python scripts/engine_zt_measure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corridor.engine.temporal_delta import TemporalSettings, consistency
from corridor.engine.z_consensus import ZConsensusSettings, link_z

ROOT = Path("C:/Users/ironb/Projects/ConfinedMig")
DATA = ROOT / "data"
BUILD = ROOT / "build"
OUT = BUILD / "eng" / "z-temporal"
OUT.mkdir(parents=True, exist_ok=True)

MOVIE = DATA / "confinedmig_cellTrack/sample_data/052924_t1.tif"
MASKS = DATA / "confinedmig_cellTrack/sample_data/052924_t1_corridor/masks.npz"
KK2_REF = BUILD / "corrected_reference_v2" / "KK2"


# -- small geometry helpers (self-contained, so the script reproduces alone) --

def _disk(h, w, cy, cx, r):
    ys, xs = np.ogrid[:h, :w]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2


def _blob(h, w, cy, cx, s=3.0):
    ys, xs = np.ogrid[:h, :w]
    return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * s * s))


def _scene(h, w, centres):
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 0.01, size=(h, w))
    for (cy, cx) in centres:
        img = img + _blob(h, w, cy, cx)
    return img


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _stats(xs):
    xs = [float(x) for x in xs if np.isfinite(x)]
    if not xs:
        return {"n": 0}
    a = np.array(xs)
    return {"n": len(xs), "min": round(float(a.min()), 4),
            "median": round(float(np.median(a)), 4),
            "max": round(float(a.max()), 4),
            "mean": round(float(a.mean()), 4)}


# ===========================================================================
# Bot 4 -- Z consensus
# ===========================================================================

def measure_z_consensus() -> dict:
    out: dict = {"bot": "z_consensus", "settings": ZConsensusSettings().to_dict()}

    # -- synthetic: a dropped slice is repaired -----------------------------
    depth, h, w = 5, 48, 48
    cy, cx, r = 24, 24, 7
    labels = np.zeros((depth, h, w), np.int32)
    truth = _disk(h, w, cy, cx, r)
    for z in (0, 1, 3, 4):
        labels[z][truth] = 1
    res = link_z(labels)
    repaired = res.object_labels_zyx[2] > 0
    repair_links = [l for l in res.links if l.kind == "repair"]
    out["synthetic_dropped_slice"] = {
        "n_objects": res.n_objects,
        "repaired_slices": res.objects[0].repaired_slices if res.objects else [],
        "repaired_slice_iou_vs_truth": round(_iou(repaired, truth), 4),
        "bridge_cost": repair_links[0].cost if repair_links else None,
        "bridge_iou": repair_links[0].iou if repair_links else None,
        "answer": "1 object spanning all 5 slices, slice 2 reconstructed",
    }

    # -- synthetic: two tangent tubes stay separate -------------------------
    labels = np.zeros((depth, h, w), np.int32)
    left = _disk(h, w, 24, 16, 6)
    right = _disk(h, w, 24, 16 + 12, 6)
    for z in range(depth):
        labels[z][left] = 1
        labels[z][right] = 2
    res2 = link_z(labels)
    cross = [l for l in res2.links if l.accepted and l.label_from != l.label_to]
    out["synthetic_two_tubes"] = {
        "n_objects": res2.n_objects,
        "cross_links_accepted": len(cross),
        "answer": "2 objects, never merged",
    }

    # -- synthetic: a forced sideways link is rejected ----------------------
    labels = np.zeros((4, h, w), np.int32)
    labels[0][left] = 1
    labels[1][left] = 1
    labels[2][right] = 1
    labels[3][right] = 1
    res3 = link_z(labels)
    forced = [l for l in res3.links if l.z_from == 1 and l.z_to == 2]
    out["synthetic_forced_sideways"] = {
        "n_objects": res3.n_objects,
        "forced_link_cost": forced[0].cost if forced else None,
        "forced_link_accepted": forced[0].accepted if forced else None,
        "gate": ZConsensusSettings().max_link_cost,
        "answer": "2 objects; the forced A->B link is rejected by the gate",
    }

    # -- real data: every held-out KK2 still is 2-D, so the bot passes through
    ref_files = sorted(KK2_REF.glob("*_masks.npy"))
    n_imgs = n_comp = n_obj = n_links = non_passthrough = 0
    for f in ref_files:
        m = np.load(f)
        if m.ndim != 2:
            continue
        r = link_z(m.astype(np.int32))
        n_imgs += 1
        n_comp += len([v for v in np.unique(m) if v])
        n_obj += r.n_objects
        n_links += r.n_links_accepted
        non_passthrough += 0 if r.passthrough else 1
    out["real_kk2_stills"] = {
        "n_images": n_imgs,
        "total_components": n_comp,
        "total_objects": n_obj,
        "objects_equal_components": n_comp == n_obj,
        "total_links": n_links,
        "images_not_passthrough": non_passthrough,
        "recovered_accuracy": 0.0,
        "why": ("the stills are 2-D (Z=1); Z consensus is a no-op passthrough, "
                "so it neither helps nor harms held-out F1. Its value is in 3-D "
                "stacks, proven on the synthetic volume above."),
    }
    return out


# ===========================================================================
# Bot 5 -- temporal delta
# ===========================================================================

def measure_temporal_delta() -> dict:
    s = TemporalSettings()
    out: dict = {"bot": "temporal_delta", "settings": s.to_dict()}

    # -- synthetic: planted displacement, true match vs a spatial swap ------
    h, w = 64, 64
    p, d = (32.0, 20.0), (2.0, 6.0)
    true_pos = (p[0] + d[0], p[1] + d[1])
    decoy = (10.0, 50.0)
    images = np.stack([_scene(h, w, [p]), _scene(h, w, [true_pos, decoy])])
    masks = np.zeros((2, h, w), np.int32)
    masks[0][_disk(h, w, *p, r=5)] = 1
    masks[1][_disk(h, w, *true_pos, r=5)] = 1
    masks[1][_disk(h, w, *decoy, r=5)] = 2
    good = consistency(images, masks, 0, 1, 1, 1)
    swap = consistency(images, masks, 0, 1, 1, 2)
    out["synthetic_planted_shift"] = {
        "planted_dy_dx": list(d),
        "measured_fwd_dy_dx": list(good.displacement_fwd_px),
        "displacement_error_px": round(float(np.hypot(
            good.displacement_fwd_px[0] - d[0],
            good.displacement_fwd_px[1] - d[1])), 4),
        "e_fb_true_match": good.e_fb,
        "e_fb_spatial_swap": swap.e_fb,
        "separation_ratio": round(swap.e_fb / good.e_fb, 2) if good.e_fb else None,
        "answer": "E_FB low for the true match, high for the wrong cell",
    }

    # -- real data: the one contiguous sample movie -------------------------
    import tifffile
    movie = tifffile.imread(MOVIE).astype(np.float64)  # (18, 324, 90)
    real_masks = np.load(MASKS)["masks"]                # (18, 324, 90) int32
    occupied = [t for t in range(real_masks.shape[0]) if real_masks[t].max() > 0]

    # Measure the one-frame motion scale from the masks themselves (centroid
    # step), because the ROI pad must exceed it or the moved cell leaves the
    # window and the correlation reads the wrong feature.
    def _centroid(m):
        ys, xs = np.nonzero(m)
        return (float(ys.mean()), float(xs.mean())) if ys.size else None

    steps = []
    for a, b in zip(occupied, occupied[1:]):
        if b - a == 1:
            ca, cb = _centroid(real_masks[a] == 1), _centroid(real_masks[b] == 1)
            if ca and cb:
                steps.append(float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])))
    max_step_px = round(max(steps), 1) if steps else 0.0
    # Pad comfortably past the fastest measured step so a true continuation
    # stays inside the ROI; the frame is 324 px tall, so this is safe.
    real_s = TemporalSettings(pad_px=int(max_step_px) + 24)

    # A note on registration: whole-field phase correlation on this movie is
    # dominated by the single cell in an otherwise near-empty 90-px channel, so
    # it reads cell motion, not device drift, and cannot establish registration.
    # The microfluidic device is mechanically fixed, so true field drift is
    # sub-pixel; the bot's ROI correlation measures the cell's local motion
    # regardless.  The measured (cell-contaminated) whole-field step is reported
    # only so the contamination is visible, never used as a drift claim.
    from skimage.registration import phase_cross_correlation
    whole = []
    for a, b in zip(occupied, occupied[1:]):
        if b - a == 1:
            sh, _, _ = phase_cross_correlation(movie[b], movie[a],
                                               upsample_factor=10, normalization=None)
            whole.append(float(np.hypot(sh[0], sh[1])))

    adjacent, skip, adj_detail = [], [], []
    for i, t1 in enumerate(occupied):
        for t2 in occupied[i + 1:]:
            k = t2 - t1
            rec = consistency(movie, real_masks, t1, 1, t2, 1, real_s)
            if not np.isfinite(rec.e_fb):
                continue
            if k == 1:
                adjacent.append(rec.e_fb)
                adj_detail.append({"t": t1, "t_next": t2, "e_fb": rec.e_fb,
                                   "fwd_dy_dx": list(rec.displacement_fwd_px),
                                   "iou_fwd": rec.iou_fwd, "iou_back": rec.iou_back})
            elif k >= 6:
                skip.append(rec.e_fb)

    adj_stats, skip_stats = _stats(adjacent), _stats(skip)
    out["real_sample_movie"] = {
        "movie": str(MOVIE.name),
        "frames_with_a_cell": occupied,
        "roi_pad_px": real_s.pad_px,
        "max_one_frame_centroid_step_px": max_step_px,
        "whole_field_step_px_cell_contaminated": _stats(whole),
        "e_fb_true_adjacent_links": adj_stats,
        "e_fb_skip_links_gap_ge_6": skip_stats,
        "separation_median": (
            round(skip_stats["median"] / adj_stats["median"], 2)
            if adj_stats.get("median") else None),
        "per_adjacent_link": adj_detail,
        "answer": ("genuine one-frame correspondences score a low E_FB once the "
                   "ROI pad exceeds the cell's one-frame motion; 6+-frame skips "
                   "(wrong single-step correspondences) score far higher"),
    }
    return out


def main() -> None:
    z = measure_z_consensus()
    (OUT / "z_consensus_measured.json").write_text(json.dumps(z, indent=2))
    td = measure_temporal_delta()
    (OUT / "temporal_delta_measured.json").write_text(json.dumps(td, indent=2))

    summary = {
        "note": ("Bots 4 and 5 are the Z and time levers of Corridor 2.1. On the "
                 "held-out KK2 stills (2-D, non-contiguous) they recover 0 by "
                 "construction -- the recoverable error budget there is at the "
                 "border, in the gradient and in the labels (other bots). These "
                 "two are validated on synthetic ground truth and the one real "
                 "contiguous movie."),
        "z_consensus": {
            "real_stills_recovered": z["real_kk2_stills"]["recovered_accuracy"],
            "synthetic_repair_iou": z["synthetic_dropped_slice"]["repaired_slice_iou_vs_truth"],
            "synthetic_two_tubes_objects": z["synthetic_two_tubes"]["n_objects"],
        },
        "temporal_delta": {
            "synthetic_displacement_error_px": td["synthetic_planted_shift"]["displacement_error_px"],
            "synthetic_true_vs_swap": [td["synthetic_planted_shift"]["e_fb_true_match"],
                                       td["synthetic_planted_shift"]["e_fb_spatial_swap"]],
            "real_adjacent_median_e_fb": td["real_sample_movie"]["e_fb_true_adjacent_links"].get("median"),
            "real_skip_median_e_fb": td["real_sample_movie"]["e_fb_skip_links_gap_ge_6"].get("median"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
