"""How many labels are missing, measured without the model proposing anything.

The recovery script lets the model propose candidates and the annotator's own
neighbouring frames accept or reject them. That is sound for *training* -- a
label is only added where a human drew one nearby -- but it cannot be used to
judge the reference, because cells the model also misses never become
candidates. The bias runs one way and it flatters the model.

This uses no model at all. For every cell the annotator labelled, it asks
whether that cell was also labelled at the nearest earlier and later labelled
times of the same crop, and whether it is absent in between. The rule is the
time-aware bracket rule of :mod:`corridor.learn.brackets` -- the same function
``build_corrected_reference.py`` adds cells with, so the count here and the
additions there are the same cells by construction.

That gives a lower bound on how many labels are missing **in the frames that
could be checked**, and therefore an upper bound on the F1 any detector can
score against this reference: a perfect detector finds the unlabelled cells
too, and every one of them is counted as a false positive. With
tp = labelled and fp = undrawn, P = tp / (tp + fp), recall = 1 and
F1 = 2P / (P + 1). A frame is "checked" only where the rule could have found
an undrawn cell (:class:`corridor.learn.brackets.FrameCheck`); a frame beside
an unannotated one could not, and counting its cells as tp would inflate the
ceiling. With two events per group the exact interval
(``ceiling_interval_95``), not the point estimate, is the result.

Version 1 (``--legacy``) took filename neighbours as consecutive frames and is
withdrawn; it is kept, unchanged, so its 0.968 (KK1) and 0.942 (KK2) can be
reproduced and seen to come from filename adjacency. It never writes over the
published ``docs/label_completeness_<G>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.brackets import (  # noqa: E402
    BracketRule,
    ceiling,
    ceiling_interval,
    distance_rule_partners,
    find_undrawn,
    leave_one_out,
)
from corridor.learn.sequences import (  # noqa: E402
    find_sequences,
    has_labels,
    load_image,
    load_masks,
)

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
#: Two labelled cells this close in adjacent frames are the same cell (v1 only).
SAME_CELL_PX = 45.0
LEGACY_DIR = ROOT / "build" / "corrected_reference_legacy"
#: Span bins, in minutes, for the leave-one-out table that sets the rule's
#: longest bracket. The edges sit between the spans that occur in the data.
LOO_BINS = ((0.0, 85.0), (85.0, 125.0), (125.0, float("inf")))


def centroids(masks: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for label in np.unique(masks):
        if label:
            ys, xs = np.nonzero(masks == label)
            out.append((float(xs.mean()), float(ys.mean())))
    return out


def nearest(point, others) -> float:
    if not others:
        return float("inf")
    return min(float(np.hypot(x - point[0], y - point[1])) for x, y in others)


def contrast_at(frame: np.ndarray, x: float, y: float) -> float | None:
    """Brightest pixel in a 20 px box at (x, y) over the frame median, in p1-p99 units."""
    lo, hi = np.percentile(frame, [1, 99])
    span = max(float(hi - lo), 1e-6)
    patch = frame[max(0, int(y) - 10):int(y) + 10, max(0, int(x) - 10):int(x) + 10]
    if patch.size == 0:
        return None
    return (float(patch.max()) - float(np.median(frame))) / span


# --------------------------------------------------------------------------
# Version 1, kept verbatim behind --legacy


def legacy(group: str, out: Path) -> int:
    total_labelled = 0
    gaps = []

    for sequence in find_sequences(TRAIN, groups=(group,), order="filename"):
        if not sequence.looks_like_a_movie:
            print(f"skipping {sequence.name}: correlation "
                  f"{sequence.frame_correlation:.3f}, not a verified movie")
            continue
        masks = [load_masks(p) for p in sequence.paths]
        images = [load_image(p) for p in sequence.paths]
        points = [centroids(m) for m in masks]
        total_labelled += sum(len(p) for p in points)

        for index in range(1, sequence.n_frames - 1):
            before, here, after = points[index - 1], points[index], points[index + 1]
            for bx, by in before:
                distance_after = nearest((bx, by), after)
                if distance_after > SAME_CELL_PX:
                    continue
                if nearest((bx, by), here) <= SAME_CELL_PX:
                    continue
                ax, ay = min(after, key=lambda q: np.hypot(q[0] - bx, q[1] - by))
                mx, my = (bx + ax) / 2.0, (by + ay) / 2.0
                frame = images[index]
                lo, hi = np.percentile(frame, [1, 99])
                span = max(float(hi - lo), 1e-6)
                patch = frame[max(0, int(my) - 10):int(my) + 10,
                              max(0, int(mx) - 10):int(mx) + 10]
                if patch.size == 0:
                    continue
                contrast = (float(patch.max()) - float(np.median(frame))) / span
                gaps.append({
                    "sequence": sequence.name,
                    "image": sequence.paths[index].name,
                    "at": [round(mx, 1), round(my, 1)],
                    "contrast_there": round(contrast, 3),
                })

    print(f"{group}: {total_labelled} labelled instances in verified movies")
    print(f"cells labelled either side of a frame but not in it: {len(gaps)}")
    if total_labelled:
        tp, fp = total_labelled, len(gaps)
        precision = tp / (tp + fp)
        print(f"version-1 ceiling: precision {precision:.4f}, "
              f"F1 {2 * precision / (precision + 1.0):.4f}")

    report = {
        "group": group,
        "labelled_instances": total_labelled,
        "missing_labels_found": len(gaps),
        "gaps": gaps,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    published = ROOT / "docs" / f"label_completeness_{group}.json"
    if published.exists():
        print(f"byte-identical to the published {published.name}: "
              f"{published.read_text(encoding='utf-8') == text}")
    return 0


# --------------------------------------------------------------------------
# Version 2


def measure_v2(group: str, rule: BracketRule) -> dict:
    labelled = [p for p in sorted((TRAIN / group).glob("*.tif")) if has_labels(p)]
    group_instances = sum(len([v for v in np.unique(load_masks(p)) if v]) for p in labelled)

    gaps, frames, loo = [], [], []
    distance_only = {"bracketed_t0_cells": 0, "with_a_unique_partner": 0}
    in_a_sequence: set[str] = set()
    for sequence in find_sequences(TRAIN, groups=(group,), order="time"):
        masks = [load_masks(p) for p in sequence.paths]
        claims, checks = find_undrawn(sequence, masks, rule)
        loo += [dict(row, sequence=sequence.name) for row in leave_one_out(sequence, masks, rule)]
        for key, value in distance_rule_partners(sequence, masks, rule).items():
            distance_only[key] += value
        in_a_sequence.update(p.name for p in sequence.paths)
        for check in checks:
            frames.append({"image": check.image, "sequence": sequence.name,
                           "t": check.t_index, "labelled": check.labelled,
                           "checkable": check.checkable, "reason": check.reason,
                           "linked_pairs": check.linked_pairs,
                           "span_min": None if check.bracket is None
                           else round(check.bracket.span_min, 1)})
        for claim in claims:
            contrast = contrast_at(load_image(sequence.paths[claim.target]), *claim.at)
            gaps.append({
                "sequence": claim.sequence, "image": claim.image, "at": list(claim.at),
                "t": claim.t_index, "t0": claim.t0_index, "t1": claim.t1_index,
                "span_min": claim.span_min, "fraction": claim.fraction,
                "displacement_um": claim.displacement_um, "link_iou": claim.link_iou,
                "contrast_there": None if contrast is None else round(contrast, 3),
            })
    for path in labelled:
        if path.name not in in_a_sequence:
            frames.append({"image": path.name, "sequence": None, "t": None,
                           "labelled": len([v for v in np.unique(load_masks(path)) if v]),
                           "checkable": False,
                           "reason": "not part of any crop with three or more labelled times",
                           "linked_pairs": 0, "span_min": None})

    checked = [f for f in frames if f["checkable"]]
    labelled_checked = sum(f["labelled"] for f in checked)
    linked_checked = sum(f["linked_pairs"] for f in checked)
    calibration = []
    for lo, hi in LOO_BINS:
        rows = [r for r in loo if lo < r["span_min"] <= hi]
        # A pair into a frame with nothing drawn fails whatever the span: that
        # is a missing label, the thing being measured, not an interpolation
        # error, so it is counted beside the calibration and not in it.
        annotated = [r for r in rows if r["target_drawn"]]
        calibration.append({
            "span_min": [lo, None if hi == float("inf") else hi],
            "linked_pairs": len(annotated),
            "drawn_where_predicted": sum(r["drawn_where_predicted"] for r in annotated),
            "spans_present": sorted({r["span_min"] for r in annotated}),
            "excluded_pairs_into_frames_with_nothing_drawn": len(rows) - len(annotated),
            "excluded_frames": sorted({r["image"] for r in rows if not r["target_drawn"]}),
        })

    legacy_report = ROOT / "docs" / f"label_completeness_{group}.json"
    legacy_numbers = None
    if legacy_report.exists():
        old = json.loads(legacy_report.read_text(encoding="utf-8"))
        legacy_numbers = {
            "labelled_instances": old["labelled_instances"],
            "missing_labels_found": old["missing_labels_found"],
            "ceiling": ceiling(old["labelled_instances"], old["missing_labels_found"]),
            "status": "withdrawn: derived by filename adjacency (see docs/RESEARCH_V2.md)",
        }

    return {
        "group": group,
        "version": "v2",
        "ordering": "true time index from the ImageJ label; crops by registration",
        "rule": rule.to_dict(),
        "formula": "tp = labelled, fp = undrawn, recall = 1, P = tp/(tp+fp), F1 = 2P/(P+1)",
        "images": len(labelled),
        "labelled_instances_in_group": group_instances,
        "checked_images": len(checked),
        "checked_image_names": sorted(f["image"] for f in checked),
        "checkable_means": ("bracket <= max_bracket_min, a drawn cell on both sides, and at "
                            "least one cell linked across it: the rule could have found an "
                            "undrawn cell there"),
        "labelled_instances_in_checked_images": labelled_checked,
        "linked_pairs_tested_in_checked_images": linked_checked,
        "missing_labels_found": len(gaps),
        "ceiling_on_checked_images": ceiling(labelled_checked, len(gaps)),
        "ceiling_interval_95": ceiling_interval(labelled_checked, len(gaps)),
        "bound_on_whole_group": ceiling(group_instances, len(gaps)),
        "bound_note": ("The whole-group figure counts only the cells found in the checked "
                       "images; the unchecked images may hide more, so it is an upper bound "
                       "on F1 and a weak one."),
        "calibration_leave_one_out": calibration,
        "distance_only_identity": dict(distance_only, radius=(
            f"{rule.position_tolerance_um} um + {rule.max_speed_um_per_min} um/min x span, "
            f"registered; every bracketed frame, any span")),
        "frames": frames,
        "gaps": gaps,
        "version_1": legacy_numbers,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", default="KK2", choices=("KK1", "KK2"))
    ap.add_argument("--out", default="")
    ap.add_argument("--legacy", action="store_true",
                    help="rerun the withdrawn version 1 into build/corrected_reference_legacy/")
    args = ap.parse_args()

    if args.legacy:
        out = Path(args.out) if args.out else LEGACY_DIR / f"label_completeness_{args.group}.json"
        if out.resolve() == (ROOT / "docs" / f"label_completeness_{args.group}.json").resolve():
            print(f"refusing to overwrite the published {out}")
            return 2
        return legacy(args.group, out)

    out = Path(args.out) if args.out else ROOT / "docs" / f"label_completeness_v2_{args.group}.json"
    published_v1 = ROOT / "docs" / f"label_completeness_{args.group}.json"
    if out.resolve() == published_v1.resolve():
        print(f"refusing to overwrite the published version-1 {out}")
        return 2
    report = measure_v2(args.group, BracketRule())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    c, b = report["ceiling_on_checked_images"], report["bound_on_whole_group"]
    ci = report["ceiling_interval_95"]
    print(f"{args.group}: {report['images']} labelled images, "
          f"{report['labelled_instances_in_group']} labelled instances")
    print(f"   checkable (a bracket of <= {report['rule']['max_bracket_min']:.0f} min, cells drawn "
          f"on both sides and linked): {report['checked_images']} images, "
          f"{report['labelled_instances_in_checked_images']} labelled instances, "
          f"{report['linked_pairs_tested_in_checked_images']} linked pairs tested")
    print(f"   cells drawn either side and not in the frame: {report['missing_labels_found']}")
    for row in report["gaps"]:
        print(f"      {row['image']:16s} t{row['t']} (t{row['t0']}-t{row['t1']}, "
              f"{row['span_min']:.0f} min) at {row['at']}  contrast {row['contrast_there']}")
    print(f"   ceiling on the checked images: tp {c['tp']}, fp {c['fp']}, "
          f"P {c['precision']}, F1 {c['f1']}")
    if ci:
        print(f"      95% interval ({ci['method']}): missing {ci['missing_rate']}, F1 {ci['f1']}")
    print(f"   bound on the whole group:      tp {b['tp']}, fp {b['fp']}, "
          f"P {b['precision']}, F1 {b['f1']}")
    if report["version_1"]:
        v1 = report["version_1"]["ceiling"]
        print(f"   version 1 (withdrawn):         tp {v1['tp']}, fp {v1['fp']}, F1 {v1['f1']}")
    print("   leave-one-out (linked pairs drawn where interpolation puts them), frames with "
          "something drawn:")
    for row in report["calibration_leave_one_out"]:
        print(f"      span {row['span_min']}: {row['drawn_where_predicted']} of "
              f"{row['linked_pairs']}  (+{row['excluded_pairs_into_frames_with_nothing_drawn']} "
              f"into frames with nothing drawn, excluded: {row['excluded_frames']})")
    d = report["distance_only_identity"]
    print(f"   identity by distance alone: {d['with_a_unique_partner']} of "
          f"{d['bracketed_t0_cells']} bracketed t0 cells have a unique partner")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
