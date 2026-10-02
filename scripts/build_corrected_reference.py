"""A reference with the provably missing cells put back -- version 2, on true time.

**Where each added cell comes from.** A cell the annotator drew at the nearest
earlier labelled time t0 and again at the nearest later labelled time t1 of the
same crop, but not at t. The evidence is entirely the annotator's own work: the
model is not consulted about whether the cell exists or where it is. The
position is the two human centroids interpolated by elapsed-time fraction; the
outline is the human mask from the nearer labelled time, translated onto it.
The rule itself, and the measurements that set its limits, live in
:mod:`corridor.learn.brackets` and ``docs/RESEARCH_V2.md``.

**What changed from version 1, and why version 1 is withdrawn.** Version 1
took filename neighbours to be consecutive frames and any two centroids within
45 px to be one cell. The ImageJ labels say otherwise: two groups are out of
time order, one KK2 group holds two crops, frames of a crop are offset by up to
38 px, and its "adjacent" frames were up to 11 h apart. ``--legacy`` reruns
version 1 exactly (into a separate folder) so the withdrawn figures stay
reproducible and the comparison is made on the record, not from memory.

**What is deliberately not done.** Cells missing from a run of consecutive
labelled times cannot be recovered this way (no far side), frames whose bracket
is longer than the rule allows are left as drawn, and nothing is added on the
strength of looking like a cell, because that is the judgement under test. The
corrected reference is still incomplete, and its remaining incompleteness still
caps any score.

Scores against this reference must always be reported **beside** the score
against the original labels, never instead of it.

Outputs (version 2): ``build/corrected_reference_v2/<G>/<stem>_masks.npy`` for
**every** labelled still of the group (stills outside any sequence are written
through unchanged, so a scorer never finds a gap), and
``docs/corrected_reference_v2_<G>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corridor.learn.brackets import BracketRule, diagnose, find_undrawn  # noqa: E402
from corridor.learn.sequences import find_sequences, has_labels, load_masks  # noqa: E402

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
SAME_CELL_PX = 45.0
V2_DIR = ROOT / "build" / "corrected_reference_v2"
LEGACY_DIR = ROOT / "build" / "corrected_reference_legacy"
PUBLISHED_V1_DIR = ROOT / "build" / "corrected_reference"
#: An addition of the withdrawn rule "survives" when version 2 claims a cell in
#: the same image within this distance of it.
SURVIVES_PX = 10.0


def regions_of(masks: np.ndarray):
    for label in np.unique(masks):
        if label:
            region = masks == label
            ys, xs = np.nonzero(region)
            yield region, (float(xs.mean()), float(ys.mean()))


def shift_to(region: np.ndarray, centre) -> np.ndarray:
    ys, xs = np.nonzero(region)
    dy = int(round(centre[1] - ys.mean()))
    dx = int(round(centre[0] - xs.mean()))
    out = np.zeros_like(region)
    h, w = region.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = region[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def n_instances(mask: np.ndarray) -> int:
    return len([v for v in np.unique(mask) if v])


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------
# Version 1, kept verbatim behind --legacy


def legacy(group: str, out_dir: Path, report_path: Path) -> int:
    """The withdrawn rule, unchanged: filename adjacency and a 45 px same-cell cut.

    The only differences from the published script are where it writes (never
    over the published files) and the comparison printed at the end.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    added_total = 0
    per_image = []
    covered = 0

    for sequence in find_sequences(TRAIN, groups=(group,), order="filename"):
        masks = [load_masks(p).copy() for p in sequence.paths]
        if not sequence.looks_like_a_movie:
            for path, mask in zip(sequence.paths, masks):
                np.save(out_dir / f"{path.stem}_masks.npy", mask.astype(np.int32))
            print(f"{sequence.name}: left unchanged (correlation "
                  f"{sequence.frame_correlation:.3f}, not a verified movie)")
            continue

        covered += sequence.n_frames
        for index in range(1, sequence.n_frames - 1):
            here = [c for _, c in regions_of(masks[index])]
            for region_before, centre_before in regions_of(masks[index - 1]):
                best = None
                for region_after, centre_after in regions_of(masks[index + 1]):
                    distance = float(np.hypot(centre_after[0] - centre_before[0],
                                              centre_after[1] - centre_before[1]))
                    if distance <= SAME_CELL_PX and (best is None or distance < best[0]):
                        best = (distance, region_after, centre_after)
                if best is None:
                    continue
                if any(np.hypot(x - centre_before[0], y - centre_before[1]) <= SAME_CELL_PX
                       for x, y in here):
                    continue

                centre = ((centre_before[0] + best[2][0]) / 2.0,
                          (centre_before[1] + best[2][1]) / 2.0)
                source = region_before if best[0] > 0 else best[1]
                recovered = shift_to(source, centre) & (masks[index] == 0)
                if recovered.sum() < 80:
                    continue
                masks[index][recovered] = int(masks[index].max()) + 1
                added_total += 1
                per_image.append({
                    "image": sequence.paths[index].name,
                    "at": [round(centre[0], 1), round(centre[1], 1)],
                })
                here.append(centre)

        for path, mask in zip(sequence.paths, masks):
            np.save(out_dir / f"{path.stem}_masks.npy", mask.astype(np.int32))

    original = sum(
        len([v for v in np.unique(load_masks(p)) if v])
        for s in find_sequences(TRAIN, groups=(group,), order="filename") for p in s.paths
    )
    print()
    print(f"{group}: {original} labelled instances originally")
    print(f"          +{added_total} recovered from the annotator's adjacent frames")
    print(f"          = {original + added_total} in the corrected reference")
    for row in per_image:
        print(f"   {row['image']:22s} at {row['at']}")

    report = {
        "group": group,
        "original_instances": original,
        "recovered": added_total,
        "corrected_instances": original + added_total,
        "frames_covered_by_temporal_evidence": covered,
        "per_image": per_image,
        "out_dir": str(out_dir),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote masks to {out_dir} and {report_path}")

    # Show, rather than assert, that this is what was published.
    published_json = ROOT / "docs" / f"corrected_reference_{group}.json"
    published_dir = PUBLISHED_V1_DIR / group
    if published_json.exists() and published_dir.is_dir():
        old = json.loads(published_json.read_text(encoding="utf-8"))
        same_json = ({k: v for k, v in old.items() if k != "out_dir"}
                     == {k: v for k, v in report.items() if k != "out_dir"})
        files = sorted(p.name for p in out_dir.glob("*_masks.npy"))
        same_files = [n for n in files if (published_dir / n).exists()
                      and (published_dir / n).read_bytes() == (out_dir / n).read_bytes()]
        print(f"against the published version 1: report identical (except out_dir): "
              f"{same_json}; mask files byte-identical: {len(same_files)}/{len(files)} "
              f"(published folder holds {len(list(published_dir.glob('*_masks.npy')))})")
    return 0


# --------------------------------------------------------------------------
# Version 2


def build_v2(group: str, out_dir: Path, report_path: Path, rule: BracketRule) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labelled = [p for p in sorted((TRAIN / group).glob("*.tif")) if has_labels(p)]
    corrected: dict[str, np.ndarray] = {p.name: load_masks(p) for p in labelled}
    original_instances = sum(n_instances(m) for m in corrected.values())

    per_image, sequences_out, checks_out = [], [], []
    in_a_sequence: set[str] = set()
    sequences = find_sequences(TRAIN, groups=(group,), order="time")
    for sequence in sequences:
        masks = [corrected[p.name].copy() for p in sequence.paths]
        claims, checks = find_undrawn(sequence, masks, rule)
        for claim in claims:
            target = corrected[claim.image]
            region = claim.region & (target == 0)
            target[region] = int(target.max()) + 1
            per_image.append({
                "image": claim.image, "sequence": claim.sequence,
                "at": list(claim.at), "t": claim.t_index,
                "t0": claim.t0_index, "t1": claim.t1_index,
                "span_min": claim.span_min, "fraction": claim.fraction,
                "displacement_um": claim.displacement_um, "link_iou": claim.link_iou,
                "added_px": int(region.sum()),
            })
        in_a_sequence.update(p.name for p in sequence.paths)
        checks_out += [{"image": c.image, "sequence": sequence.name, "t": c.t_index,
                        "labelled": c.labelled, "checkable": c.checkable, "reason": c.reason,
                        "linked_pairs": c.linked_pairs,
                        "span_min": None if c.bracket is None else round(c.bracket.span_min, 1)}
                       for c in checks]
        sequences_out.append({
            "name": sequence.name, "experiment": sequence.experiment_id,
            "crop": sequence.crop, "shape": list(sequence.shape),
            "frame_interval_min": round(sequence.frame_interval_min or 0.0, 4),
            "pixel_size_um": sequence.pixel_size_um,
            "frame_correlation": round(sequence.frame_correlation, 3),
            "frames": [{"image": p.name, "t": t, "offset_px": [round(o[0], 1), round(o[1], 1)],
                        "link_ncc": round(l, 3)}
                       for p, t, o, l in zip(sequence.paths, sequence.t_indices,
                                             sequence.offsets_px, sequence.link_ncc)],
        })
    for path in labelled:
        if path.name not in in_a_sequence:
            checks_out.append({"image": path.name, "sequence": None, "t": None,
                               "labelled": n_instances(corrected[path.name]), "checkable": False,
                               "reason": "not part of any crop with three or more labelled times",
                               "linked_pairs": 0, "span_min": None})
        np.save(out_dir / f"{path.stem}_masks.npy", corrected[path.name].astype(np.int32))

    # Every addition of the withdrawn rule, judged one by one.
    legacy_rows = []
    published = ROOT / "docs" / f"corrected_reference_{group}.json"
    if published.exists():
        by_name = {p.name: (s, k) for s in sequences for k, p in enumerate(s.paths)}
        for row in json.loads(published.read_text(encoding="utf-8"))["per_image"]:
            image, at = row["image"], tuple(row["at"])
            survivors = [r for r in per_image if r["image"] == image
                         and np.hypot(r["at"][0] - at[0], r["at"][1] - at[1]) <= SURVIVES_PX]
            if survivors:
                verdict = "survives"
            elif image in by_name:
                s, k = by_name[image]
                verdict = diagnose(s, [load_masks(p) for p in s.paths], k, at, rule)
            else:
                verdict = "not part of any crop with three or more labelled times"
            legacy_rows.append({"image": image, "at": list(at), "survives": bool(survivors),
                                "why_not": "" if survivors else verdict})

    added = len(per_image)
    report = {
        "group": group,
        "reference_version": "v2",
        "ordering": "true time index from the ImageJ label; crops by registration",
        "rule": rule.to_dict(),
        "images": len(labelled),
        "original_instances": original_instances,
        "added": added,
        "corrected_instances": original_instances + added,
        "checked_images": sorted(c["image"] for c in checks_out if c["checkable"]),
        "frames": checks_out,
        "sequences": sequences_out,
        "per_image": per_image,
        "legacy_additions": legacy_rows,
        "legacy_survivors": sum(r["survives"] for r in legacy_rows),
        "out_dir": _rel(out_dir),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", default="KK2", choices=("KK1", "KK2"))
    ap.add_argument("--out-dir", default="", help="mask folder (default by version)")
    ap.add_argument("--report", default="", help="JSON report path (default by version)")
    ap.add_argument("--legacy", action="store_true",
                    help="rerun the withdrawn version 1 (filename adjacency) into "
                         "build/corrected_reference_legacy/, never over the published files")
    args = ap.parse_args()

    forbidden = (PUBLISHED_V1_DIR / args.group,
                 ROOT / "docs" / f"corrected_reference_{args.group}.json")

    def refuses(*paths: Path) -> bool:
        for path in paths:
            for published in forbidden:
                if path.resolve() == published.resolve():
                    print(f"refusing to overwrite the published version-1 {published}")
                    return True
        return False

    if args.legacy:
        out_dir = Path(args.out_dir) if args.out_dir else LEGACY_DIR / args.group
        report = Path(args.report) if args.report else (
            LEGACY_DIR / f"corrected_reference_{args.group}.json")
        if refuses(out_dir, report):
            return 2
        return legacy(args.group, out_dir, report)

    out_dir = Path(args.out_dir) if args.out_dir else V2_DIR / args.group
    report_path = Path(args.report) if args.report else (
        ROOT / "docs" / f"corrected_reference_v2_{args.group}.json")
    if refuses(out_dir, report_path):
        return 2
    report = build_v2(args.group, out_dir, report_path, BracketRule())

    print(f"{args.group}: {report['images']} labelled images, "
          f"{report['original_instances']} labelled instances")
    print(f"   checkable frames (bracket <= {report['rule']['max_bracket_min']:.0f} min, cells "
          f"drawn on both sides and linked across it): {len(report['checked_images'])}")
    print(f"   +{report['added']} cells the annotator drew either side and not here")
    for row in report["per_image"]:
        print(f"      {row['image']:16s} t{row['t']} (t{row['t0']}-t{row['t1']}, "
              f"{row['span_min']:.0f} min, f={row['fraction']:.2f}) at {row['at']}")
    print(f"   = {report['corrected_instances']} in the corrected reference")
    if report["legacy_additions"]:
        print(f"   version-1 additions surviving: {report['legacy_survivors']} of "
              f"{len(report['legacy_additions'])}")
        for row in report["legacy_additions"]:
            mark = "survives" if row["survives"] else row["why_not"]
            print(f"      {row['image']:16s} at {row['at']}: {mark}")
    print(f"\nwrote masks to {out_dir} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
