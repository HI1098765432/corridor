"""The v2 axis-free tracker against the frozen v1.3.0 tracks, without Cellpose.

There is no tracking ground truth for the supplied movies (contract §0), so
this is not an accuracy measurement.  It answers a narrower question that can
be answered: given *exactly* the detections v1.3.0 tracked, where does the v2
tracker decide differently, and why?

For each of the five baseline runs in ``build/baseline_v1.3.0/<movie>/``:

*   ``detections.csv`` is rebuilt into ``Detection`` objects (primary and
    recovered; v1's second tracking pass ran on both), and each primary
    detection gets its mask crop from ``masks.npz`` -- which holds primary
    detections only, so recovered ones have no crop and the overlap term is
    skipped for them, as it would be in a real run.
*   The lanes are re-measured from the movie stack with
    ``geometry.detect_channels``.
*   The v2 tracker runs with the default ``TrackingConfig`` and the run's own
    calibration.
*   Every baseline track is matched to the v2 track that holds the majority of
    its observations (joined on ``(frame, det_label)``, which is unique per
    frame: recovered detections were relabelled above the primary labels).

Each disagreement is listed link by link: a link v1 made that v2 did not
(with v2's cost of that link, computed from a v2 track rebuilt on the v1
track's own history), and a link v2 made that v1 did not (with v2's cost and
margin, and v1's own recorded reason from ``unlinked_starts.csv``).  Every v2
track is checked against the re-measured lanes: in the wide fields
``052924_1`` and ``052924_2`` no track may visit two lanes.

Usage::

    python scripts/compare_tracking_to_baseline.py \\
        [--baseline build/baseline_v1.3.0] [--samples data/.../sample_data] \\
        [--out docs/tracking_v2_vs_v1_baseline.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.core.config import GeometryConfig, Scale, TrackingConfig  # noqa: E402
from corridor.core.detections import Detection  # noqa: E402
from corridor.core.geometry import ChannelGeometry, detect_channels  # noqa: E402
from corridor.core.tracking import (  # noqa: E402
    MotionModel,
    Track,
    pair_cost,
    track_detections,
)

MOVIES = ["052924_1", "052924_2", "052924_t1", "052924_t2_empty", "052924_t3_dual"]
WIDE_FIELDS = {"052924_1", "052924_2"}

_DUPLICATES_052924_2 = (
    "Lane 4, frames 11-14: v1's recovery (windowed tier) added a detection at "
    "each frame 20.7, 7.9, 1.2 and 0.6 px from the primary detection of the same "
    "cell -- the cell detected twice. Both trackers therefore carry two tracks "
    "along one cell for those frames, and differ only in which track gets which "
    "copy: v2's choice at frame 11 has a link margin of 0.16 chi2, a tie. A "
    "recovery artefact, not a tracking disagreement."
)

#: Every disagreement on the five baseline movies, read link by link against
#: the detections. Keyed by (movie, kind, link); the script reports any
#: disagreement missing from here as UNREVIEWED, and any entry here that no
#: longer occurs as STALE, so this cannot silently drift from the numbers.
REVIEWED: dict[tuple[str, str, tuple[tuple[int, int], tuple[int, int]]], str] = {
    ("052924_1", "v1_link_not_in_v2", ((8, 4), (9, 5))): (
        "Lane 3. v1 track 2 was moving down the lane (y 207 -> 247, +40 px/frame); "
        "(9,5) is an intensity-tier recovered detection 50.6 px back up the lane at "
        "half the area. v2 refuses it as a motion outlier (d2 25.5 > 13.8: a "
        "90 px/frame reversal against the prediction) and instead starts a track at "
        "(9,5) that continues (10,6), (11,5), (12,2) on a steady ~-50 px/frame "
        "trajectory. v1's reading makes the cell reverse and then become another "
        "cell one frame later; v2's is kinematically coherent."
    ),
    ("052924_1", "v2_link_not_in_v1", ((9, 5), (10, 6))): (
        "Lane 3, the same scene: the second step of the steady upward trajectory "
        "(47 px, area ratio 1.01, cost 0.51). v1 had already given (9,5) to "
        "track 2 and refused this link (above_cost_gate)."
    ),
    ("052924_1", "v2_link_not_in_v1", ((14, 1), (15, 4))): (
        "Lane 1. At frame 15 the cell (122 px long at y 61 in frame 14) appears as "
        "two pieces, 221 px at y 34 and 454 px at y 131. v2 continues the track "
        "with the larger piece (cost 15.1: motion 10.5, size 3.6) and flags "
        "split_suspected on both tracks; v1 refused both continuations "
        "(above_cost_gate). Neither piece is seen after frame 15, so this is one "
        "observation either way; which reading is right is not decidable from "
        "the detections."
    ),
    ("052924_1", "v2_link_not_in_v1", ((15, 3), (16, 2))): (
        "Lane 3. The cell is accelerating down the lane (y 46, 67, 89, 127: +21, "
        "+22, +38 px/frame); the next step is +42 px at area ratio 0.98. v2 links "
        "it (motion d2 9.6, cost 10.7, margin 19.3); v1 refused it "
        "(above_cost_gate) and started a new track at frame 16."
    ),
    ("052924_2", "v1_link_not_in_v2", ((10, 1), (11, 3))): _DUPLICATES_052924_2,
    ("052924_2", "v1_link_not_in_v2", ((15, 1), (16, 1))): _DUPLICATES_052924_2,
    ("052924_2", "v2_link_not_in_v1", ((10, 1), (11, 2))): _DUPLICATES_052924_2,
    ("052924_2", "v2_link_not_in_v1", ((14, 4), (16, 1))): _DUPLICATES_052924_2,
}


def _bool(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


def load_detections(run_dir: Path) -> tuple[list[Detection], dict[str, int]]:
    """Rebuild every detection of a v1 run, with a mask crop for the primary ones."""
    masks = np.load(run_dir / "masks.npz")["masks"]
    dets: list[Detection] = []
    counts = Counter()
    with open(run_dir / "detections.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            min_r, min_c = int(row["bbox_min_y"]), int(row["bbox_min_x"])
            max_r, max_c = int(row["bbox_max_y"]), int(row["bbox_max_x"])
            det = Detection(
                frame=int(row["frame"]),
                label=int(row["label"]),
                x=float(row["x"]),
                y=float(row["y"]),
                area_px=float(row["area_px"]),
                bbox=(min_r, min_c, max_r, max_c),
                extent_px=int(float(row["extent_px"])),
                eccentricity=float(row["eccentricity"]),
                orientation_rad=float(row["orientation_rad"]),
                major_axis_px=float(row["major_axis_px"]),
                minor_axis_px=float(row["minor_axis_px"]),
                solidity=float(row["solidity"]),
                touches_border=_bool(row["touches_border"]),
                channel=-1,
                source=row["source"],
                confidence=float(row["confidence"] or 1.0),
            )
            counts[f"source_{det.source}"] += 1
            if det.source == "primary":
                crop = masks[det.frame, min_r:max_r, min_c:max_c] == det.label
                if int(crop.sum()) == int(round(det.area_px)):
                    det.mask_crop = crop
                    counts["primary_with_mask"] += 1
                else:
                    counts["primary_mask_mismatch"] += 1
            dets.append(det)
    return dets, dict(counts)


def load_baseline_tracks(run_dir: Path) -> dict[int, list[tuple[int, int]]]:
    tracks: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with open(run_dir / "tracks.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tracks[int(row["track_id"])].append((int(row["frame"]), int(row["det_label"])))
    return {tid: sorted(obs) for tid, obs in tracks.items()}


def load_unlinked(run_dir: Path) -> dict[int, dict[str, str]]:
    path = run_dir / "unlinked_starts.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["track_id"]): r for r in csv.DictReader(fh)}


def scale_of(run_dir: Path) -> Scale:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    cal = manifest["calibration"]
    return Scale.from_values(
        cal["pixel_size_um"] if cal.get("spatially_calibrated") else None,
        cal["frame_interval_min"] if cal.get("temporally_calibrated") else None,
    )


def n_frames_of(run_dir: Path) -> int:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return int(manifest["input"]["shape_tyx"][0])


def _rebuilt_track(history: list[Detection], scale: Scale, cfg: TrackingConfig,
                   geometry: ChannelGeometry) -> Track:
    """A v2 track filtered over ``history`` (a v1 track's observations up to a link)."""
    track = Track(id=0, model=MotionModel.from_config(scale, cfg, 2))
    for det in history:
        track.observe(det, cost=0.0, lane=geometry.lane_of(det.x, det.y))
    return track


def _breakdown_dict(b) -> dict[str, Any]:
    if b is None:
        return {}
    out = {
        "gated": b.gated,
        "total": None if b.gated else round(b.total, 3),
        "mahalanobis": None if b.mahalanobis is None else round(b.mahalanobis, 3),
    }
    if not b.gated:
        for name in ("motion", "size", "shape", "orientation", "direction", "overlap", "gap"):
            out[name] = round(getattr(b, name), 3)
        if b.motion_forward is not None:
            out["motion_forward"] = round(b.motion_forward, 3)
            out["motion_backward"] = round(b.motion_backward, 3)
    return out


def _v2_choice(obs, previous) -> Any:
    """What v2 linked ``obs`` to instead: its previous observation, or a track start."""
    if previous is None:
        return "track start"
    return {
        "previous": [previous.frame, previous.det_label],
        "cost": None if obs.cost is None else round(obs.cost, 3),
        "link_margin": None if obs.link_margin is None else round(obs.link_margin, 3),
    }


def compare_movie(name: str, baseline_dir: Path, samples: Path) -> dict[str, Any]:
    run_dir = baseline_dir / name
    scale = scale_of(run_dir)
    cfg = TrackingConfig()
    n_frames = n_frames_of(run_dir)
    dets, det_counts = load_detections(run_dir)
    by_key = {(d.frame, d.label): d for d in dets}
    assert len(by_key) == len(dets), "(frame, label) must be unique"

    stack = tifffile.imread(samples / f"{name}.tif")
    primary = [d for d in dets if d.source == "primary"]
    geometry = detect_channels(
        stack, GeometryConfig(), primary,
        pixel_size_um=scale.pixel_size_um if scale.calibrated_space else None,
        channel_constraint=cfg.channel_constraint,
    )

    tracks, events = track_detections(dets, n_frames, scale, cfg, geometry=geometry)
    new_of: dict[tuple[int, int], int] = {}
    for tr in tracks:
        for o in tr.observations:
            new_of[(o.frame, o.det_label)] = tr.id
    by_new = {tr.id: tr for tr in tracks}

    baseline = load_baseline_tracks(run_dir)
    unlinked = load_unlinked(run_dir)
    v1_of = {key: tid for tid, obs in baseline.items() for key in obs}

    # -- identity agreement ---------------------------------------------------
    per_track = []
    agreed = total = 0
    for tid, obs in sorted(baseline.items()):
        holders = Counter(new_of.get(key) for key in obs)
        majority, count = holders.most_common(1)[0]
        agreed += count
        total += len(obs)
        per_track.append({
            "baseline_track": tid,
            "frames": [f for f, _ in obs],
            "n_obs": len(obs),
            "majority_v2_track": majority,
            "agreement": round(count / len(obs), 4),
            "v2_tracks_holding_it": {str(k): v for k, v in holders.items()},
        })
    reverse = []
    for tr in tracks:
        holders = Counter(v1_of.get((o.frame, o.det_label)) for o in tr.observations)
        reverse.append({
            "v2_track": tr.id,
            "frames": [o.frame for o in tr.observations],
            "baseline_tracks_in_it": {str(k): v for k, v in holders.items()},
            "flags": sorted(tr.flags),
            "lanes": sorted({o.channel for o in tr.observations}),
        })

    # -- links: the unit at which the two trackers can disagree ---------------
    def links(groups: dict[int, list[tuple[int, int]]]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        out = set()
        for obs in groups.values():
            for a, b in zip(obs[:-1], obs[1:]):
                out.add((a, b))
        return out

    v2_groups = {tr.id: [(o.frame, o.det_label) for o in tr.observations] for tr in tracks}
    v1_links, v2_links = links(baseline), links(v2_groups)
    only_v1 = sorted(v1_links - v2_links)
    only_v2 = sorted(v2_links - v1_links)

    disagreements = []
    for a, b in only_v1:
        tid = v1_of[a]
        history = [by_key[k] for k in baseline[tid] if k[0] <= a[0]]
        rebuilt = _rebuilt_track(history, scale, cfg, geometry)
        v2_cost = pair_cost(rebuilt, by_key[b], scale, cfg, geometry=geometry)
        holder_b = by_new[new_of[b]]
        idx = next(i for i, o in enumerate(holder_b.observations) if (o.frame, o.det_label) == b)
        v2_prev = holder_b.observations[idx - 1] if idx > 0 else None
        disagreements.append({
            "kind": "v1_link_not_in_v2",
            "link": [list(a), list(b)],
            "baseline_track": tid,
            "v2_tracks": [new_of[a], new_of[b]],
            "v2_cost_of_the_v1_link": _breakdown_dict(v2_cost),
            "v2_instead": _v2_choice(holder_b.observations[idx], v2_prev),
            "lanes": [geometry.lane_of(by_key[a].x, by_key[a].y), geometry.lane_of(by_key[b].x, by_key[b].y)],
            "step_px": round(float(np.hypot(by_key[b].x - by_key[a].x, by_key[b].y - by_key[a].y)), 1),
            "area_ratio": round(by_key[b].area_px / by_key[a].area_px, 3),
            "sources": [by_key[a].source, by_key[b].source],
        })
    for a, b in only_v2:
        tr = by_new[new_of[b]]
        obs_b = next(o for o in tr.observations if (o.frame, o.det_label) == b)
        v1_start = unlinked.get(v1_of.get(b, -1)) if v1_of.get(b) and baseline[v1_of[b]][0] == b else None
        disagreements.append({
            "kind": "v2_link_not_in_v1",
            "link": [list(a), list(b)],
            "v2_track": tr.id,
            "baseline_tracks": [v1_of.get(a), v1_of.get(b)],
            "v2_cost": _breakdown_dict(obs_b.breakdown),
            "v2_link_margin": None if obs_b.link_margin is None else round(obs_b.link_margin, 3),
            "closed_by_stage_2": bool(obs_b.breakdown is not None and obs_b.breakdown.motion_forward is not None),
            "v1_reason": (
                {"refused_because": v1_start["refused_because"], "explanation": v1_start["explanation"]}
                if v1_start else None
            ),
            "lanes": [geometry.lane_of(by_key[a].x, by_key[a].y), geometry.lane_of(by_key[b].x, by_key[b].y)],
            "step_px": round(float(np.hypot(by_key[b].x - by_key[a].x, by_key[b].y - by_key[a].y)), 1),
            "area_ratio": round(by_key[b].area_px / by_key[a].area_px, 3),
            "gap_frames": b[0] - a[0],
            "sources": [by_key[a].source, by_key[b].source],
        })

    def crossings(groups) -> list[dict[str, Any]]:
        out = []
        for tid, keys in groups.items():
            lanes = {geometry.lane_of(by_key[k].x, by_key[k].y) for k in keys}
            lanes.discard(-1)
            if len(lanes) > 1:
                out.append({"track": tid, "lanes": sorted(lanes)})
        return out

    def nearest_primary_px(key: tuple[int, int]) -> float | None:
        det = by_key[key]
        if det.source == "primary":
            return None
        others = [p for p in dets if p.frame == det.frame and p.source == "primary"]
        if not others:
            return None
        return round(min(float(np.hypot(p.x - det.x, p.y - det.y)) for p in others), 1)

    seen = set()
    for d in disagreements:
        a, b = (tuple(k) for k in d["link"])
        d["recovered_detection_to_nearest_primary_px"] = [nearest_primary_px(a), nearest_primary_px(b)]
        key = (name, d["kind"], (a, b))
        seen.add(key)
        d["explanation"] = REVIEWED.get(key, "UNREVIEWED")
    stale = [list(k[2]) for k in REVIEWED if k[0] == name and k not in seen]

    closures = [
        {"track": tr.id, "frame": o.frame, "det_label": o.det_label, "gap_frames": o.gap_frames,
         "in_v1": (lambda prev: v1_of.get(prev) is not None and v1_of.get(prev) == v1_of.get((o.frame, o.det_label)))(
             (tr.observations[i - 1].frame, tr.observations[i - 1].det_label)),
         **_breakdown_dict(o.breakdown),
         "link_margin": round(o.link_margin, 3)}
        for tr in tracks for i, o in enumerate(tr.observations)
        if o.breakdown is not None and o.breakdown.motion_forward is not None
    ]
    margins = [o.link_margin for tr in tracks for o in tr.observations if o.link_margin is not None]
    return {
        "movie": name,
        "n_frames": n_frames,
        "detections": {"total": len(dets), **det_counts},
        "geometry": {
            "source": geometry.source, "applied": geometry.applied,
            "n_lanes": geometry.n_lanes,
            "pitch_px": None if geometry.pitch_px is None else round(geometry.pitch_px, 2),
            "lane_tilts_deg": [round(lane.tilt_from_vertical_deg, 2) for lane in geometry.lanes],
            "notes": geometry.notes,
        },
        "n_baseline_tracks": len(baseline),
        "n_v2_tracks": len(tracks),
        "baseline_observations": total,
        "identity_agreement": round(agreed / total, 4) if total else None,
        "links": {"v1": len(v1_links), "v2": len(v2_links), "shared": len(v1_links & v2_links),
                  "only_v1": len(only_v1), "only_v2": len(only_v2)},
        "lane_crossings": {"v2": crossings(v2_groups), "v1": crossings(baseline)},
        "stage2_gap_closures": closures,
        "stale_reviewed_explanations": stale,
        "v2_flags": dict(Counter(f for tr in tracks for f in tr.flags)),
        "v2_link_margin_chi2": (
            {"n": len(margins), "min": round(min(margins), 3),
             "median": round(statistics.median(margins), 3)} if margins else None
        ),
        "per_baseline_track": per_track,
        "per_v2_track": reverse,
        "disagreements": disagreements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=ROOT / "build" / "baseline_v1.3.0")
    parser.add_argument(
        "--samples", type=Path, default=ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "tracking_v2_vs_v1_baseline.json")
    args = parser.parse_args(argv)

    results = [compare_movie(m, args.baseline, args.samples) for m in MOVIES]
    agreed = sum(r["identity_agreement"] * r["baseline_observations"] for r in results)
    total = sum(r["baseline_observations"] for r in results)
    report = {
        "what_this_is": (
            "v2 axis-free tracker run on the exact detections of the frozen v1.3.0 "
            "baseline, compared track by track. The baseline is v1's answer, not ground "
            "truth: agreement measures consistency, not accuracy."
        ),
        "tracking_config": TrackingConfig().__dict__,
        "summary": {
            "baseline_tracks": sum(r["n_baseline_tracks"] for r in results),
            "v2_tracks": sum(r["n_v2_tracks"] for r in results),
            "baseline_observations": total,
            "identity_agreement_obs_weighted": round(agreed / total, 4) if total else None,
            "links_only_in_v1": sum(r["links"]["only_v1"] for r in results),
            "links_only_in_v2": sum(r["links"]["only_v2"] for r in results),
            "v2_lane_crossings_in_wide_fields": sum(
                len(r["lane_crossings"]["v2"]) for r in results if r["movie"] in WIDE_FIELDS
            ),
            "stage2_gap_closures": sum(len(r["stage2_gap_closures"]) for r in results),
            "unreviewed_disagreements": sum(
                1 for r in results for d in r["disagreements"] if d["explanation"] == "UNREVIEWED"
            ),
            "stale_reviewed_explanations": sum(len(r["stale_reviewed_explanations"]) for r in results),
        },
        "movies": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    for r in results:
        print(
            f"{r['movie']:16s} tracks v1 {r['n_baseline_tracks']:3d}  v2 {r['n_v2_tracks']:3d}  "
            f"agreement {r['identity_agreement']:.3f}  links only-v1 {r['links']['only_v1']} "
            f"only-v2 {r['links']['only_v2']}  v2 lane crossings {len(r['lane_crossings']['v2'])}"
        )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
