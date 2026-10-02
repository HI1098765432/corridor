"""Rank unlabelled movie frames for annotation, from Corridor's own results.

Input: Corridor result folders (``masks.npz`` with ``masks`` as T x Y x X, and
``tracks.csv``; ``run.json`` when present). Each frame gets the signals the
contract names (``docs/NEXT_GENERATION.md`` section 9), every one reported so a
reviewer can re-rank:

- ``zero_between_occupied`` -- no detection in a frame whose nearest earlier
  and later frames both have one. A cell population does not vanish for one
  frame; a segmentation can.
- ``track_gaps`` -- tracks observed before and after the frame but not in it
  (the tracker bridged a gap there): a likely missed detection.
- ``count_jump`` -- how far the detection count departs from the mean of the
  two neighbouring frames.
- ``broken_masks`` -- instances made of more than one connected piece.
- ``oversized_masks`` -- instances larger than ``OVERSIZE_FACTOR`` times the
  movie's median instance area: merge candidates.
- ``inconsistency`` -- the fraction of the frame's instances that overlap
  nothing (IoU < ``LINK_IOU``) in either neighbouring frame.

The score is a weighted sum (:data:`WEIGHTS`). It is an ordering heuristic for
choosing what a person should look at first, not a probability of error.

**Leakage guard.** A result folder whose input belongs to an experiment the
locked split marks never-train -- the five sample movies are 20240529-s01, the
test experiment -- is refused unless ``--allow-held-out`` is given, and then
the output says so. Mining hard examples from the test experiment and labelling
them would put test cells into training.

The guard fails closed. A result whose experiment cannot be established -- no
``run.json``, an input path that no longer resolves on this machine, an input
with no ImageJ time label -- is treated as held out: refused unless
``--allow-unknown-experiment`` is given, and listed under
``unknown_experiment`` either way. An unknown experiment is exactly what a
moved copy of a sample movie looks like.

    python -m training.hard_examples <result_dir> [<result_dir> ...] --out ranking.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from training import ROOT
from training.splits import NEVER_TRAIN_EXPERIMENTS

#: Same as corridor.learn.reconstruct.LINK_IOU: overlap that links a cell across frames.
LINK_IOU = 0.15
#: diag_errors calls a false positive "too large for a cell" above 2.5x the
#: typical area.
OVERSIZE_FACTOR = 2.5
WEIGHTS = {
    "zero_between_occupied": 3.0,
    "track_gaps": 1.0,
    "count_jump": 0.5,
    "broken_masks": 1.0,
    "oversized_masks": 1.0,
    "inconsistency": 2.0,
}


def _regions(frame: np.ndarray) -> dict[int, np.ndarray]:
    return {int(v): frame == v for v in np.unique(frame) if v}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _pieces(region: np.ndarray) -> int:
    from scipy.ndimage import label as cc_label

    return int(cc_label(region)[1])


def track_gaps(tracks_csv: Path, n_frames: int) -> list[int]:
    """Per frame, how many tracks are seen before and after it but not in it."""
    frames_of: dict[str, set[int]] = defaultdict(set)
    if tracks_csv.exists():
        with open(tracks_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frames_of[row["track_id"]].add(int(float(row["frame"])))
    gaps = [0] * n_frames
    for frames in frames_of.values():
        if len(frames) < 2:
            continue
        for t in range(min(frames) + 1, max(frames)):
            if t not in frames and 0 <= t < n_frames:
                gaps[t] += 1
    return gaps


def identify(result_dir: Path) -> tuple[str | None, str]:
    """The experiment id of a result's input, or None and why it is unknown."""
    from corridor.learn.sequences import read_still_meta

    run = result_dir / "run.json"
    if not run.exists():
        return None, "no run.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    text = data.get("input", {}).get("path", "")
    if not text:
        return None, "run.json names no input"
    path = Path(text)
    for candidate in (path, ROOT / path):
        if candidate.is_file():
            meta = read_still_meta(candidate, "result")
            if meta is None:
                return None, f"input {text} has no ImageJ time label"
            if meta.experiment_id.startswith("unknown"):
                return None, f"input {text} carries no date"
            return meta.experiment_id, ""
    return None, f"input {text} does not exist here"


def experiment_of(result_dir: Path) -> str | None:
    """The experiment id of a result's input, from the input file's ImageJ label."""
    return identify(result_dir)[0]


def frame_signals(masks: np.ndarray, gaps: list[int]) -> list[dict]:
    n = masks.shape[0]
    regions = [_regions(masks[t]) for t in range(n)]
    counts = [len(r) for r in regions]
    areas = [int(r.sum()) for frame in regions for r in frame.values()]
    typical = float(np.median(areas)) if areas else 0.0

    rows = []
    for t in range(n):
        neighbours = ([counts[t - 1]] if t > 0 else []) + ([counts[t + 1]] if t + 1 < n else [])
        zero_between = int(counts[t] == 0 and len(neighbours) == 2 and min(neighbours) > 0)
        jump = abs(counts[t] - float(np.mean(neighbours))) if neighbours else 0.0
        broken = sum(_pieces(r) > 1 for r in regions[t].values())
        oversized = sum(r.sum() > OVERSIZE_FACTOR * typical for r in regions[t].values()) \
            if typical else 0
        isolated = 0
        for region in regions[t].values():
            linked = any(_iou(region, other) >= LINK_IOU
                         for k in (t - 1, t + 1) if 0 <= k < n
                         for other in regions[k].values())
            isolated += int(not linked)
        inconsistency = isolated / counts[t] if counts[t] else 0.0
        signals = {
            "zero_between_occupied": zero_between,
            "track_gaps": int(gaps[t]),
            "count_jump": round(jump, 2),
            "broken_masks": int(broken),
            "oversized_masks": int(oversized),
            "inconsistency": round(inconsistency, 3),
        }
        score = sum(WEIGHTS[k] * float(v) for k, v in signals.items())
        rows.append({"frame": t, "detections": counts[t], "score": round(score, 3),
                     "signals": signals})
    return rows


def rank(result_dirs: list[Path], *, allow_held_out: bool = False,
         allow_unknown_experiment: bool = False) -> dict:
    ranked, refused, held_out_used, unknown = [], [], [], []
    for result_dir in result_dirs:
        experiment, why = identify(result_dir)
        if experiment is None:
            unknown.append({"result": result_dir.as_posix(), "why": why,
                            "included": allow_unknown_experiment})
            if not allow_unknown_experiment:
                continue
        elif experiment in NEVER_TRAIN_EXPERIMENTS:
            if not allow_held_out:
                refused.append({"result": result_dir.as_posix(), "experiment": experiment})
                continue
            held_out_used.append(result_dir.as_posix())
        with np.load(result_dir / "masks.npz") as saved:
            masks = np.asarray(saved["masks"])
        if masks.ndim != 3:
            raise ValueError(f"{result_dir}: expected T x Y x X masks, got {masks.shape}")
        gaps = track_gaps(result_dir / "tracks.csv", masks.shape[0])
        for row in frame_signals(masks, gaps):
            ranked.append({"result": result_dir.as_posix(), "experiment": experiment, **row})
    ranked.sort(key=lambda r: (-r["score"], r["result"], r["frame"]))
    return {
        "weights": WEIGHTS,
        "link_iou": LINK_IOU,
        "oversize_factor": OVERSIZE_FACTOR,
        "refused_held_out": refused,
        "held_out_included": held_out_used,
        "unknown_experiment": unknown,
        "frames": ranked,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--allow-held-out", action="store_true",
                    help="include results from never-train experiments (marked in the output)")
    ap.add_argument("--allow-unknown-experiment", action="store_true",
                    help="include results whose experiment cannot be established "
                         "(marked in the output)")
    args = ap.parse_args(argv)

    report = rank([Path(p) for p in args.results], allow_held_out=args.allow_held_out,
                  allow_unknown_experiment=args.allow_unknown_experiment)
    for row in report["refused_held_out"]:
        print(f"refused {row['result']}: experiment {row['experiment']} is held out")
    for row in report["unknown_experiment"]:
        print(f"{'INCLUDED' if row['included'] else 'refused'} {row['result']}: "
              f"experiment unknown ({row['why']})")
    for row in report["frames"][:args.top]:
        print(f"{row['score']:7.2f}  {row['result']}  frame {row['frame']:3d}  "
              f"n={row['detections']:2d}  {row['signals']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    refused_any = report["refused_held_out"] or any(
        not row["included"] for row in report["unknown_experiment"])
    return 0 if report["frames"] or not refused_any else 3


if __name__ == "__main__":
    raise SystemExit(main())
