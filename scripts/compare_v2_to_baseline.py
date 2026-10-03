"""Corridor 2.0 end to end against the frozen v1.3.0 baseline, movie by movie.

There is no tracking ground truth for the supplied movies (contract §0), so
this is a consistency comparison, not an accuracy measurement: it says where
2.0 answers differently from 1.3.0 on the same movie, and which stage the
difference comes from.

Inputs (both produced by the CLI at its defaults, one run per movie):

*   ``build/baseline_v1.3.0/<movie>/`` -- frozen at commit ``eebd6bd``.
*   ``build/v2_runs/<movie>/`` -- the 2.0 pipeline.

For each movie:

1.  **Segmentation.** ``masks.npz`` and ``masks_raw.npz`` are compared
    array for array (dtype, shape, every pixel), the model checksum and every
    segmentation setting in the two ``run.json`` files are compared, and the
    primary rows of ``detections.csv`` are compared on every column both
    schemas have.  Same model, same settings and identical masks is the
    proof that segmentation did not change.
2.  **Recovery.** Recovered detections (``source != primary``) are matched
    across versions by frame and centroid (within ``--match-px``).  One that
    exists in only one version is a recovery difference.
3.  **Tracking.** Every observation gets a version-independent key: a primary
    detection is ``(frame, label)`` -- identical in both versions, because the
    masks are -- and a recovered one is its cross-version match.  Each v1
    track is paired with the v2 track holding most of its observations
    (identity agreement, observation-weighted), and every link (consecutive
    observations of one track) present in only one version is listed.  A link
    difference touching a detection only one version recovered is attributed
    to recovery; any other is the tracker's.
4.  **Measurement.** For every paired track, net speed and median step speed
    (µm/min) from both ``track_summary.csv`` files.  A speed difference
    between tracks made of the *same* observations is a measurement
    difference; otherwise it is attributed to whatever changed the
    observations.
5.  **Plausibility.** Every v2 step faster than the fastest step of any
    v1.3.0 track (measured from the baseline's own ``tracks.csv``) is listed
    as a KNOWN DEFECT, never as a neutral difference: the segmentation is
    bit-identical, so such a step is a link 1.x never made and no reviewed
    track ever showed.  Each is traced to its cause -- the first-pass link
    across the gap (its bracket, and how far apart the bracket's two
    observations are) and the recovery attempt that filled it -- and every
    link and track comparison that touches it is re-attributed to it.

Usage::

    python scripts/compare_v2_to_baseline.py \\
        [--baseline build/baseline_v1.3.0] [--v2 build/v2_runs] \\
        [--out docs/v2_vs_v1_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.store.project import read_table  # noqa: E402

MOVIES = ["052924_1", "052924_2", "052924_t1", "052924_t2_empty", "052924_t3_dual"]
PRIMARY = "primary"

#: Segmentation settings that decide the masks; all must be equal for the
#: masks to be expected identical.
SEG_SETTINGS = (
    "diameter", "cellprob_threshold", "flow_threshold", "channels", "normalize",
    "normalisation_mode", "normalize_percentiles", "normalize_tile_px",
    "normalize_sharpen_px", "ensemble", "ensemble_passes", "min_extent_px",
    "min_area_px", "drop_border_touching",
)

#: A difference smaller than this (relative) is float formatting, not a change.
REL_TOL = 1e-6


def _close(a: Any, b: Any, rel: float = REL_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)
    return a == b


def _masks(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        return data[data.files[0]]


# --------------------------------------------------------------------------
# 1. Segmentation
# --------------------------------------------------------------------------


def compare_segmentation(v1: Path, v2: Path, m1: dict, m2: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("masks.npz", "masks_raw.npz"):
        a, b = _masks(v1 / name), _masks(v2 / name)
        if a is None or b is None:
            out[name] = {"compared": False, "reason": "missing in one version"}
            continue
        same_shape = a.shape == b.shape
        out[name] = {
            "shape_v1": list(a.shape), "shape_v2": list(b.shape),
            "dtype_v1": str(a.dtype), "dtype_v2": str(b.dtype),
            "identical": bool(same_shape and np.array_equal(a, b)),
            "pixels_differing": int(np.count_nonzero(a != b)) if same_shape else None,
            "labelled_pixels": int(np.count_nonzero(a)),
        }
    s1, s2 = m1["segmentation"], m2["segmentation"]
    sha1 = s1.get("model_sha256")
    sha2 = (m2.get("model") or {}).get("sha256")
    out["model_sha256_v1"] = sha1
    out["model_sha256_v2"] = sha2
    out["same_model"] = sha1 == sha2 and sha1 is not None
    differing = {}
    for key in SEG_SETTINGS:
        a = s1.get(key)
        b = s2.get(key)
        if isinstance(a, list) or isinstance(b, list):
            a, b = list(a or []), list(b or [])
        if a != b:
            differing[key] = {"v1": a, "v2": b}
    out["settings_differing"] = differing
    out["kept_per_frame_identical"] = (
        s1.get("kept_instances_per_frame") == s2.get("kept_instances_per_frame")
    )
    out["raw_per_frame_identical"] = (
        s1.get("raw_instances_per_frame") == s2.get("raw_instances_per_frame")
    )
    return out


def compare_primary_detections(d1: list[dict], d2: list[dict]) -> dict[str, Any]:
    p1 = {(r["frame"], r["label"]): r for r in d1 if r.get("source") == PRIMARY}
    p2 = {(r["frame"], r["label"]): r for r in d2 if r.get("source") == PRIMARY}
    common_cols = sorted(
        (set(d1[0]) & set(d2[0])) - {"source", "confidence", "channel"} if d1 and d2 else set()
    )
    differing: Counter = Counter()
    for key in p1.keys() & p2.keys():
        for col in common_cols:
            if not _close(p1[key].get(col), p2[key].get(col)):
                differing[col] += 1
    lane_changes = sum(
        1 for key in p1.keys() & p2.keys() if p1[key].get("channel") != p2[key].get("channel")
    )
    return {
        "n_primary_v1": len(p1),
        "n_primary_v2": len(p2),
        "same_keys": p1.keys() == p2.keys(),
        "columns_compared": common_cols,
        "values_differing_by_column": dict(differing),
        "identical": p1.keys() == p2.keys() and not differing,
        # Lane ids are not segmentation: v1 numbered channels from its axis,
        # 2.0 from the measured lanes (-1 outside every lane).
        "lane_id_changes": lane_changes,
    }


# --------------------------------------------------------------------------
# 2. Recovery: match recovered detections across versions
# --------------------------------------------------------------------------


def _int(value: Any) -> int | None:
    """``read_table`` returns ids outside its integer list as floats (3.0)."""
    return None if value is None else int(value)


def _attempt_for(det: dict, attempts: list[dict]) -> dict | None:
    """The found attempt that produced a recovered detection.

    v2 attempts name the detection (``det_label``); v1 attempts do not, so
    the v1 one is the found attempt of that frame whose recorded offset best
    matches the detection's distance from its prediction.
    """
    found = [a for a in attempts if a["frame"] == det["frame"] and a.get("recovered")]
    named = [a for a in found if a.get("det_label") is not None]
    if named:
        return next((a for a in named if _int(a["det_label"]) == det["label"]), None)
    if not found:
        return None
    return min(found, key=lambda a: abs(
        math.hypot(det["x"] - a["predicted_x"], det["y"] - a["predicted_y"])
        - float(a.get("offset_from_prediction_px") or 0.0)
    ))


def _why_one_sided(
    det: dict, own: list[dict], other: list[dict], other_name: str, same_px: float
) -> tuple[str, str]:
    """``(stage, reason)`` for a recovered detection only one version has.

    Recovery searches only where a FIRST-PASS track bridged a gap, so the
    question is whether the other version searched the same place at all.
    If it did not, its first-pass tracker made no link across that gap and the
    difference is the tracker's; if it did, recovery itself decided otherwise.
    """
    attempt = _attempt_for(det, own)
    if attempt is None:
        return "recovery", "no attempt record found for this detection"
    px, py = attempt["predicted_x"], attempt["predicted_y"]
    near = [
        a for a in other
        if a["frame"] == det["frame"] and a.get("predicted_x") is not None
        and math.hypot(a["predicted_x"] - px, a["predicted_y"] - py) <= same_px
    ]
    if not near:
        return "tracking", (
            f"{other_name}'s first pass made no link across this gap, so its recovery never "
            "searched here"
        )
    a = near[0]
    if a.get("duplicate_of_label") is not None:
        return "recovery", (
            f"{other_name} searched here and dropped the candidate as a duplicate of primary "
            f"label {_int(a['duplicate_of_label'])}"
        )
    if a.get("recovered"):
        return "recovery", (
            f"{other_name} searched here and found a different object "
            f"({a.get('found_by')}, {float(a.get('offset_from_prediction_px') or 0):.1f} px from "
            "the prediction)"
        )
    return "recovery", f"{other_name} searched here and found nothing: {a.get('detail')}"


def match_recovered(
    d1: list[dict], d2: list[dict], match_px: float,
    attempts1: list[dict], attempts2: list[dict], same_px: float,
) -> dict[str, Any]:
    r1 = [r for r in d1 if r.get("source") != PRIMARY]
    r2 = [r for r in d2 if r.get("source") != PRIMARY]
    primary2 = defaultdict(list)
    for r in d2:
        if r.get("source") == PRIMARY:
            primary2[r["frame"]].append(r)
    pairs: list[tuple[dict, dict, float]] = []
    for a in r1:
        for b in r2:
            if a["frame"] == b["frame"]:
                dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                if dist <= match_px:
                    pairs.append((a, b, dist))
    pairs.sort(key=lambda p: p[2])
    used1, used2, matched = set(), set(), []
    for a, b, dist in pairs:
        k1, k2 = (a["frame"], a["label"]), (b["frame"], b["label"])
        if k1 in used1 or k2 in used2:
            continue
        used1.add(k1)
        used2.add(k2)
        matched.append((k1, k2, dist, a.get("source"), b.get("source")))

    def nearest_primary(row: dict) -> float | None:
        cands = primary2.get(row["frame"], [])
        if not cands:
            return None
        return round(min(math.hypot(row["x"] - c["x"], row["y"] - c["y"]) for c in cands), 1)

    causes: dict[tuple, str] = {}
    only1, only2 = [], []
    for r in r1:
        if (r["frame"], r["label"]) in used1:
            continue
        stage, reason = _why_one_sided(r, attempts1, attempts2, "v2", same_px)
        causes[("r_v1_only", r["frame"], r["label"])] = stage
        only1.append({
            "frame": r["frame"], "label_v1": r["label"], "x": round(r["x"], 1),
            "y": round(r["y"], 1), "tier": r.get("source"),
            "nearest_primary_px": nearest_primary(r), "attributed_to": stage, "why": reason,
        })
    for r in r2:
        if (r["frame"], r["label"]) in used2:
            continue
        stage, reason = _why_one_sided(r, attempts2, attempts1, "v1", same_px)
        causes[("r_v2_only", r["frame"], r["label"])] = stage
        only2.append({
            "frame": r["frame"], "label_v2": r["label"], "x": round(r["x"], 1),
            "y": round(r["y"], 1), "tier": r.get("source"),
            "nearest_primary_px": nearest_primary(r), "attributed_to": stage, "why": reason,
        })
    return {
        "n_recovered_v1": len(r1),
        "n_recovered_v2": len(r2),
        "matched": [
            {"frame": k1[0], "label_v1": k1[1], "label_v2": k2[1], "distance_px": round(d, 2),
             "tier_v1": s1, "tier_v2": s2}
            for k1, k2, d, s1, s2 in matched
        ],
        "only_v1": only1,
        "only_v2": only2,
        "_map_v1_to_v2": {k1: k2 for k1, k2, *_ in matched},
        "_causes": causes,
    }


# --------------------------------------------------------------------------
# 3. Tracking
# --------------------------------------------------------------------------


def _keyed_tracks(rows: list[dict], key_of) -> dict[int, list[tuple]]:
    tracks: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for r in rows:
        tracks[int(r["track_id"])].append((r["frame"], key_of(r)))
    return {tid: [k for _, k in sorted(obs)] for tid, obs in tracks.items()}


def _links(tracks: dict[int, list[tuple]]) -> set[tuple[tuple, tuple]]:
    return {(a, b) for obs in tracks.values() for a, b in zip(obs[:-1], obs[1:])}


def compare_tracking(
    t1_rows: list[dict], t2_rows: list[dict], rec: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, int | None], dict[int, list], dict[int, list]]:
    mapping = rec["_map_v1_to_v2"]

    def key1(r: dict) -> tuple:
        k = (r["frame"], r["det_label"])
        if r.get("detection_source") == PRIMARY:
            return ("p", *k)
        return ("r", *mapping[k]) if k in mapping else ("r_v1_only", *k)

    matched_v2 = set(mapping.values())

    def key2(r: dict) -> tuple:
        k = (r["frame"], r["det_label"])
        if r.get("detection_source") == PRIMARY:
            return ("p", *k)
        return ("r", *k) if k in matched_v2 else ("r_v2_only", *k)

    tr1 = _keyed_tracks(t1_rows, key1)
    tr2 = _keyed_tracks(t2_rows, key2)
    owner2 = {k: tid for tid, obs in tr2.items() for k in obs}

    pairing: dict[int, int | None] = {}
    agree = total = agree_p = total_p = 0
    for tid, obs in tr1.items():
        votes = Counter(owner2[k] for k in obs if k in owner2)
        best, n = votes.most_common(1)[0] if votes else (None, 0)
        pairing[tid] = best
        agree += n
        total += len(obs)
        prim = [k for k in obs if k[0] == "p"]
        total_p += len(prim)
        agree_p += sum(1 for k in prim if best is not None and owner2.get(k) == best)

    links1, links2 = _links(tr1), _links(tr2)

    causes = rec["_causes"]

    def cause(link: tuple[tuple, tuple]) -> str:
        """A link touching a one-sided recovered detection inherits that detection's cause."""
        stages = {causes.get(k, "recovery") for k in link if k[0] in ("r_v1_only", "r_v2_only")}
        return " + ".join(sorted(stages)) if stages else "tracking"

    def describe(link):
        return {"from": list(link[0]), "to": list(link[1]), "attributed_to": cause(link)}

    only1 = sorted(links1 - links2)
    only2 = sorted(links2 - links1)
    paired_v2 = {v for v in pairing.values() if v is not None}
    return (
        {
            "n_tracks_v1": len(tr1),
            "n_tracks_v2": len(tr2),
            "n_observations_v1": total,
            "n_observations_v2": sum(len(o) for o in tr2.values()),
            "identity_agreement_obs_weighted": round(agree / total, 4) if total else None,
            "identity_agreement_primary_obs_only": round(agree_p / total_p, 4) if total_p else None,
            "v1_to_v2_track": {str(k): v for k, v in sorted(pairing.items())},
            "v2_tracks_with_no_v1_majority": sorted(set(tr2) - paired_v2),
            "links_v1": len(links1),
            "links_v2": len(links2),
            "links_only_in_v1": [describe(l) for l in only1],
            "links_only_in_v2": [describe(l) for l in only2],
        },
        pairing, tr1, tr2,
    )


# --------------------------------------------------------------------------
# 4. Measurement
# --------------------------------------------------------------------------


def compare_speeds(
    s1_rows: list[dict], s2_rows: list[dict], pairing: dict[int, int | None],
    tr1: dict[int, list], tr2: dict[int, list], causes: dict[tuple, str],
) -> list[dict[str, Any]]:
    s1 = {int(r["track_id"]): r for r in s1_rows}
    s2 = {int(r["track_id"]): r for r in s2_rows}
    out = []
    for tid1, tid2 in sorted(pairing.items()):
        a = s1.get(tid1, {})
        b = s2.get(tid2, {}) if tid2 is not None else {}
        obs1, obs2 = set(tr1[tid1]), set(tr2.get(tid2, []))
        same_obs = obs1 == obs2
        row = {
            "v1_track": tid1, "v2_track": tid2,
            "n_obs_v1": len(obs1), "n_obs_v2": len(obs2), "same_observations": same_obs,
        }
        differs = False
        for name in ("net_speed_um_per_min", "median_speed_um_per_min", "mean_speed_um_per_min"):
            va, vb = a.get(name), b.get(name)
            row[f"{name}_v1"] = va
            row[f"{name}_v2"] = vb
            row[f"{name}_diff"] = (
                None if va is None or vb is None else round(float(vb) - float(va), 6)
            )
            if not _close(va, vb):
                differs = True
        if not differs:
            row["attributed_to"] = "none (identical)"
        elif same_obs and any(k[0] == "r" for k in obs1):
            # The same recovered cell in both, but each version measured its
            # own crop of it: the positions differ by up to --match-px.
            row["attributed_to"] = "recovery (same recovered cell, different centroid)"
        elif same_obs:
            row["attributed_to"] = "measurement"
        else:
            stages: set[str] = set()
            for k in obs1 ^ obs2:
                if k[0] in ("r_v1_only", "r_v2_only"):
                    stages.add(causes.get(k, "recovery"))
                else:
                    stages.add("tracking")
            row["attributed_to"] = " + ".join(sorted(stages)) or "tracking"
            row["observations_only_v1"] = sorted(list(k) for k in obs1 - obs2)
            row["observations_only_v2"] = sorted(list(k) for k in obs2 - obs1)
        out.append(row)
    return out


# --------------------------------------------------------------------------
# 5. Plausibility
# --------------------------------------------------------------------------


def fastest_step(baseline: Path) -> dict[str, Any]:
    """The fastest single step of any track in the baseline, all movies."""
    best: dict[str, Any] = {"speed_um_per_min": 0.0}
    for movie in MOVIES:
        for r in read_table(baseline / movie / "tracks.csv"):
            v = r.get("speed_um_per_min")
            if v is not None and float(v) > best["speed_um_per_min"]:
                best = {"speed_um_per_min": float(v), "movie": movie,
                        "track_id": _int(r["track_id"]), "frame": _int(r["frame"])}
    return best


def implausible_steps(
    t2_rows: list[dict], bound: float, attempts2: list[dict], d2: list[dict]
) -> list[dict[str, Any]]:
    """v2 steps faster than ``bound`` (µm/min), each traced to its cause."""
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in t2_rows:
        by_track[int(r["track_id"])].append(r)
    position = {(r["frame"], r["label"]): (r["x"], r["y"]) for r in d2}
    out = []
    for tid, rows in sorted(by_track.items()):
        rows.sort(key=lambda r: r["frame"])
        for a, b in zip(rows[:-1], rows[1:]):
            v = b.get("speed_um_per_min")
            if v is None or float(v) <= bound * (1 + 1e-9):
                continue
            step = {
                "track_id": tid,
                "from": {"frame": a["frame"], "det_label": a["det_label"],
                         "x": round(a["x_px"], 1), "y": round(a["y_px"], 1),
                         "source": a.get("detection_source")},
                "to": {"frame": b["frame"], "det_label": b["det_label"],
                       "x": round(b["x_px"], 1), "y": round(b["y_px"], 1),
                       "source": b.get("detection_source")},
                "step_px": round(math.hypot(b["x_px"] - a["x_px"], b["y_px"] - a["y_px"]), 1),
                "speed_um_per_min": round(float(v), 3),
                "times_fastest_baseline_step": round(float(v) / bound, 2),
                "track_observations": len(rows),
            }
            causes = []
            for end in (a, b):
                if end.get("detection_source") == PRIMARY:
                    continue
                attempt = next(
                    (x for x in attempts2 if x.get("recovered") and x["frame"] == end["frame"]
                     and _int(x.get("det_label")) == end["det_label"]), None,
                )
                if attempt is None:
                    continue
                before = (_int(attempt["bracket_frame_before"]), _int(attempt["bracket_label_before"]))
                after = (_int(attempt["bracket_frame_after"]), _int(attempt["bracket_label_after"]))
                jump = None
                if before in position and after in position:
                    (x0, y0), (x1, y1) = position[before], position[after]
                    jump = round(math.hypot(x1 - x0, y1 - y0), 1)
                causes.append({
                    "recovered_frame": end["frame"],
                    "found_by": attempt.get("found_by"),
                    "offset_from_prediction_px": round(float(attempt["offset_from_prediction_px"]), 1),
                    "first_pass_track_id": _int(attempt["first_pass_track_id"]),
                    "first_pass_bracket": {"before": list(before), "after": list(after)},
                    "first_pass_link_px": jump,
                    "owners": (
                        "tracking (WP-B): the first pass linked the bracket across the gap; "
                        "recovery (E2): accepted a candidate this far from the prediction"
                    ),
                })
            step["cause"] = causes or "a link between two primary detections (tracking, WP-B)"
            out.append(step)
    return out


def _defect_label(step: dict[str, Any]) -> str:
    return (
        f"KNOWN DEFECT: track {step['track_id']} steps {step['step_px']} px in "
        f"{step['to']['frame'] - step['from']['frame']} frame(s) = {step['speed_um_per_min']} "
        f"um/min ({step['times_fastest_baseline_step']}x the fastest v1.3.0 step)"
    )


def mark_defects(
    tracking: dict[str, Any], speeds: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> None:
    """Re-attribute every comparison that touches an implausible step to it.

    Link and observation keys end in ``(frame, det_label)`` of the version
    they come from, so a v2 step is found among them by its two endpoints.
    """
    for step in steps:
        ends = {(step["from"]["frame"], step["from"]["det_label"]),
                (step["to"]["frame"], step["to"]["det_label"])}
        label = _defect_label(step)
        for link in tracking["links_only_in_v2"]:
            if {tuple(link["from"][1:]), tuple(link["to"][1:])} == ends:
                link["neutral_attribution"] = link["attributed_to"]
                link["attributed_to"] = "known defect"
                link["known_defect"] = label
        for row in speeds:
            touched = [tuple(k[1:]) for k in row.get("observations_only_v1", [])
                       + row.get("observations_only_v2", [])]
            if row["v2_track"] == step["track_id"] or any(k in ends for k in touched):
                row["neutral_attribution"] = row["attributed_to"]
                row["attributed_to"] = "known defect"
                row["known_defect"] = label


# --------------------------------------------------------------------------


def compare_movie(
    baseline: Path, v2: Path, movie: str, match_px: float, same_px: float,
    fastest_baseline_step: float,
) -> dict[str, Any]:
    b, n = baseline / movie, v2 / movie
    m1 = json.loads((b / "run.json").read_text(encoding="utf-8"))
    m2 = json.loads((n / "run.json").read_text(encoding="utf-8"))
    d1, d2 = read_table(b / "detections.csv"), read_table(n / "detections.csv")
    seg = compare_segmentation(b, n, m1, m2)
    prim = compare_primary_detections(d1, d2)
    attempts1 = read_table(b / "recovery_attempts.csv")
    attempts2 = read_table(n / "recovery_attempts.csv")
    rec = match_recovered(d1, d2, match_px, attempts1, attempts2, same_px)
    tracking, pairing, tr1, tr2 = compare_tracking(
        read_table(b / "tracks.csv"), read_table(n / "tracks.csv"), rec
    )
    speeds = compare_speeds(
        read_table(b / "track_summary.csv"), read_table(n / "track_summary.csv"),
        pairing, tr1, tr2, rec["_causes"],
    )
    t2_rows = read_table(n / "tracks.csv")
    defects = implausible_steps(t2_rows, fastest_baseline_step, attempts2, d2)
    mark_defects(tracking, speeds, defects)
    rec.pop("_map_v1_to_v2")
    rec.pop("_causes")
    rec["attempts_v1"] = m1.get("recovery", {}).get("attempted")
    rec["attempts_v2"] = m2.get("recovery", {}).get("attempted")
    rec["duplicates_dropped_v2"] = [
        {"frame": a["frame"], "first_pass_track_id": _int(a["first_pass_track_id"]),
         "final_track_id": _int(a["track_id"]),
         "duplicate_of_label": _int(a["duplicate_of_label"])}
        for a in attempts2 if a.get("duplicate_of_label") is not None
    ]
    attribution = Counter(r["attributed_to"] for r in speeds)
    for link in tracking["links_only_in_v1"] + tracking["links_only_in_v2"]:
        attribution[f"link:{link['attributed_to']}"] += 1
    segmentation_identical = bool(
        seg["same_model"] and not seg["settings_differing"]
        and seg["masks.npz"].get("identical") and seg["masks_raw.npz"].get("identical")
        and prim["identical"]
    )
    return {
        "segmentation_bit_identical": segmentation_identical,
        "segmentation": seg,
        "primary_detections": prim,
        "recovery": rec,
        "tracking": tracking,
        "speeds_per_track": speeds,
        "known_defects": defects,
        "attribution_counts": dict(attribution),
        "results_v1": {k: m1["results"].get(k) for k in ("n_detections", "n_tracks", "mean_speed_um_per_min")},
        "results_v2": {
            k: m2["results"].get(k)
            for k in ("n_detections", "n_detections_primary", "n_detections_recovered",
                      "n_tracks", "mean_speed_um_per_min", "mean_net_speed_um_per_min",
                      "median_net_speed_um_per_min", "median_net_speed_um_per_hr")
        },
        "channel_geometry_v2": {
            k: m2["channel_geometry"].get(k)
            for k in ("source", "n_lanes", "pitch_px", "applied", "lane_gate_applied")
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", type=Path, default=ROOT / "build" / "baseline_v1.3.0")
    p.add_argument("--v2", type=Path, default=ROOT / "build" / "v2_runs")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "v2_vs_v1_baseline.json")
    p.add_argument(
        "--match-px", type=float, default=5.0,
        help="centroid distance (px) within which a recovered detection is the same in both",
    )
    p.add_argument(
        "--same-prediction-px", type=float, default=20.0,
        help="predicted positions (px) this close are the same recovery search in both versions",
    )
    p.add_argument(
        "--tracking-only", type=Path, default=None,
        help="output of compare_tracking_to_baseline.py at the same commit: the tracker "
        "alone on v1's exact detections, embedded as the reference that separates tracking "
        "from recovery in the identity agreement",
    )
    args = p.parse_args(argv)

    fastest = fastest_step(args.baseline)
    movies = {
        m: compare_movie(
            args.baseline, args.v2, m, args.match_px, args.same_prediction_px,
            fastest["speed_um_per_min"],
        )
        for m in MOVIES
    }
    total_obs = sum(r["tracking"]["n_observations_v1"] for r in movies.values())
    agreed = sum(
        (r["tracking"]["identity_agreement_obs_weighted"] or 0) * r["tracking"]["n_observations_v1"]
        for r in movies.values()
    )
    attribution: Counter = Counter()
    for r in movies.values():
        attribution.update(r["attribution_counts"])
    summary = {
        "segmentation_bit_identical_all_movies": all(
            r["segmentation_bit_identical"] for r in movies.values()
        ),
        "primary_detections_v1": sum(r["primary_detections"]["n_primary_v1"] for r in movies.values()),
        "primary_detections_v2": sum(r["primary_detections"]["n_primary_v2"] for r in movies.values()),
        "recovered_detections_v1": sum(r["recovery"]["n_recovered_v1"] for r in movies.values()),
        "recovered_detections_v2": sum(r["recovery"]["n_recovered_v2"] for r in movies.values()),
        "recovered_in_both": sum(len(r["recovery"]["matched"]) for r in movies.values()),
        "tracks_v1": sum(r["tracking"]["n_tracks_v1"] for r in movies.values()),
        "tracks_v2": sum(r["tracking"]["n_tracks_v2"] for r in movies.values()),
        "identity_agreement_obs_weighted": round(agreed / total_obs, 4) if total_obs else None,
        "links_only_in_v1": sum(len(r["tracking"]["links_only_in_v1"]) for r in movies.values()),
        "links_only_in_v2": sum(len(r["tracking"]["links_only_in_v2"]) for r in movies.values()),
        "attribution_counts": dict(attribution),
        "fastest_v1_step": fastest,
        "known_defects": [
            {"movie": m, **{k: d[k] for k in ("track_id", "step_px", "speed_um_per_min",
                                              "times_fastest_baseline_step", "cause")}}
            for m, r in movies.items() for d in r["known_defects"]
        ],
        "per_movie_tracks": {
            m: [r["tracking"]["n_tracks_v1"], r["tracking"]["n_tracks_v2"]] for m, r in movies.items()
        },
    }
    report = {
        "what_this_is": (
            "Corridor 2.0 (build/v2_runs, CLI defaults) against the frozen v1.3.0 baseline "
            "(build/baseline_v1.3.0, CLI defaults), movie by movie. The baseline is v1's answer, "
            "not ground truth: agreement measures consistency, not accuracy. Every difference is "
            "attributed to segmentation, recovery (the re-segmentation of windows a track "
            "predicts), tracking or measurement -- except a step faster than any v1.3.0 step, "
            "which is attributed to 'known defect' (summary.known_defects), with the stage it "
            "would otherwise have been given kept as neutral_attribution."
        ),
        "how": __doc__.split("For each movie:")[1].split("Usage::")[0].strip(),
        "match_px": args.match_px,
        "same_prediction_px": args.same_prediction_px,
        "summary": summary,
        "movies": movies,
    }
    if args.tracking_only is not None and args.tracking_only.exists():
        alone = json.loads(args.tracking_only.read_text(encoding="utf-8"))
        report["tracker_alone_on_v1_detections"] = {
            "what": (
                "scripts/compare_tracking_to_baseline.py at this commit's tracking defaults: the "
                "v2 tracker given exactly v1.3.0's detections (primary and recovered), so "
                "neither segmentation nor recovery can differ. Its agreement against the "
                "end-to-end figure above separates the tracker's share from recovery's."
            ),
            "tracking_config": alone.get("tracking_config"),
            "summary": alone.get("summary"),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for m, r in movies.items():
        t = r["tracking"]
        print(
            f"{m:16s} seg identical={r['segmentation_bit_identical']} "
            f"primary {r['primary_detections']['n_primary_v1']}/{r['primary_detections']['n_primary_v2']} "
            f"recovered {r['recovery']['n_recovered_v1']}/{r['recovery']['n_recovered_v2']} "
            f"(both {len(r['recovery']['matched'])}) tracks {t['n_tracks_v1']}/{t['n_tracks_v2']} "
            f"identity {t['identity_agreement_obs_weighted']} "
            f"links only v1 {len(t['links_only_in_v1'])} only v2 {len(t['links_only_in_v2'])}"
        )
        for d in r["known_defects"]:
            print(f"  {_defect_label(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
