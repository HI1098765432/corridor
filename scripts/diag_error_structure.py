"""What distinguishes the errors from the cells that work? Read from the table.

`scripts/diag_errors.py` already recorded contrast, area, elongation, edge
strength, border distance and best-IoU for every miss and every false detection.
That file is enough to answer several questions without running the network
again, which matters when the machine is busy.

Three questions decide what to build next:

1. **Are the 17 "carved wrong" errors splits, merges, or bad boundaries?** A
   split is one cell answered by two predictions; a merge is two cells answered
   by one; a bad boundary is one-to-one with poor overlap. Each has a different
   fix and guessing wrong wastes a training run.

2. **What is different about the 9 "visible but missed"?** They have normal
   contrast and normal size and are not at a border, so something else explains
   them. If nothing in the recorded properties separates them from cells that
   are found, then the recorded properties are the wrong ones and the next step
   is a better measurement rather than a better model.

3. **Are the errors concentrated in a few images?** Sixty-four errors spread
   evenly over 31 images is a different problem from sixty-four errors in four
   images, and only the second is worth attacking image-by-image.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarise(values, name: str, unit: str = "") -> str:
    if not values:
        return f"{name:26s}  (none)"
    a = np.asarray(values, dtype=float)
    return (f"{name:26s}  median {np.median(a):7.3f}{unit}  "
            f"iqr {np.percentile(a, 25):6.3f}-{np.percentile(a, 75):6.3f}  n={len(a)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default=str(ROOT / "docs" / "error_table.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "error_structure.json"))
    args = ap.parse_args()

    table = json.loads(Path(args.errors).read_text(encoding="utf-8"))
    misses = table["misses"]
    false_positives = table["false_positives"]

    report: dict = {}

    # ---------------------------------------------------------------- 1
    print("=" * 74)
    print("1. THE 17 CARVED-WRONG ERRORS: SPLIT, MERGED, OR JUST BADLY OUTLINED?")
    print("=" * 74)

    poor_misses = [m for m in misses if m["kind"] == "found but outline too poor"]
    overlapping_fp = [f for f in false_positives
                      if f["kind"] == "overlaps a real cell (split or poor outline)"]

    # Group by image: several predictions over one truth in the same image is a
    # split; several truths under one prediction is a merge.
    by_image: dict[str, dict[str, int]] = defaultdict(lambda: {"miss": 0, "fp": 0})
    for row in poor_misses:
        by_image[row["image"]]["miss"] += 1
    for row in overlapping_fp:
        by_image[row["image"]]["fp"] += 1

    splits = merges = boundaries = 0
    for image, counts in sorted(by_image.items()):
        if counts["fp"] > counts["miss"]:
            kind = "SPLIT (more predictions than truths)"
            splits += counts["fp"] - counts["miss"]
        elif counts["miss"] > counts["fp"]:
            kind = "MERGE (more truths than predictions)"
            merges += counts["miss"] - counts["fp"]
        else:
            kind = "one-to-one, boundary too poor"
            boundaries += counts["miss"]
        print(f"   {image:24s} {counts['miss']} miss / {counts['fp']} fp   {kind}")

    print()
    print(f"   splits          {splits:3d}   one cell answered by several predictions")
    print(f"   merges          {merges:3d}   several cells answered by one prediction")
    print(f"   bad boundaries  {boundaries:3d}   one-to-one, overlap below 0.5")
    report["carving"] = {"splits": splits, "merges": merges, "boundaries": boundaries}

    print()
    print("   overlap of these near-misses with their truth:")
    print("   " + summarise([m["best_iou"] for m in poor_misses], "misses, best IoU"))
    print("   " + summarise([f["best_iou"] for f in overlapping_fp], "false det, best IoU"))
    # How far from 0.5 are they? A pile just below the threshold is a different
    # problem from a pile at 0.3.
    near = [m["best_iou"] for m in poor_misses if m["best_iou"] >= 0.4]
    print(f"   {len(near)} of {len(poor_misses)} misses are within 0.1 of the 0.5 threshold")
    report["near_threshold_misses"] = len(near)

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 74)
    print("2. THE 9 'VISIBLE BUT MISSED': WHAT IS DIFFERENT ABOUT THEM?")
    print("=" * 74)

    invisible = [m for m in misses if m["kind"] == "visible but missed"]
    faint = [m for m in misses if m["kind"] == "too faint"]

    for name, group in (("visible but missed", invisible), ("too faint", faint)):
        print(f"\n   {name} (n={len(group)}):")
        for field, unit in (("contrast", ""), ("area_px", " px"),
                            ("eccentricity", ""), ("major_px", " px"),
                            ("minor_px", " px"), ("edge_strength", ""),
                            ("distance_to_border_px", " px")):
            print("      " + summarise([abs(r[field]) if field == "contrast" else r[field]
                                        for r in group if field in r], field, unit))

    report["visible_but_missed"] = [
        {k: r.get(k) for k in ("image", "contrast", "area_px", "eccentricity",
                               "major_px", "minor_px", "edge_strength")}
        for r in invisible
    ]

    # A comparison group: the false detections that look exactly like cells are
    # the model's own idea of a cell, so their properties show what it responds
    # to. If the missed ones sit outside that envelope, that is the signal.
    looks_like = [f for f in false_positives
                  if f["kind"] == "looks exactly like a cell (suspect label)"]
    if looks_like and invisible:
        print("\n   for comparison, objects the model DID fire on:")
        for field in ("contrast", "area_px", "eccentricity", "major_px"):
            got = [abs(r[field]) if field == "contrast" else r[field]
                   for r in looks_like if field in r]
            missed = [abs(r[field]) if field == "contrast" else r[field]
                      for r in invisible if field in r]
            if got and missed:
                print(f"      {field:22s} fired on {np.median(got):7.2f}   "
                      f"missed {np.median(missed):7.2f}")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 74)
    print("3. ARE THE ERRORS CONCENTRATED?")
    print("=" * 74)
    counts = Counter(r["image"] for r in misses + false_positives)
    total_images = 31
    print(f"   {len(counts)} of {total_images} images contain at least one error")
    print(f"   worst offenders:")
    for image, n in counts.most_common(8):
        print(f"      {image:24s} {n:2d} errors")
    top5 = sum(n for _, n in counts.most_common(5))
    share = 100 * top5 / max(sum(counts.values()), 1)
    print(f"\n   the worst 5 images hold {top5} of {sum(counts.values())} errors ({share:.0f}%)")
    report["error_concentration"] = {
        "images_with_errors": len(counts),
        "worst_five_share": round(share / 100, 3),
        "per_image": dict(counts.most_common()),
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
