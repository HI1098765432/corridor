"""What actually reaches the network, and what the held-out split actually holds out.

Two claims surfaced by the research pass, both of which would invalidate any
retraining number before it was measured. Both are checked here directly against
the files rather than taken on trust.

**1. Cellpose silently discards most of the labelled data.**
``train.train_seg`` defaults to ``min_train_masks=5``: an image with fewer than
five instances is dropped from training without a word. These are confined
cells, a handful per field. If the default holds, a "retrained on 71 images"
model was retrained on a fraction of them, and would be compared against a
baseline that was not.

**2. The held-out split may not be held out.**
The two halves are named KK1 and KK2, which reads like two conditions. If they
are really several acquisition days unevenly divided, then "KK2Model scores
0.303 on KK1" is partly a statement about how many days each model saw, not
about a domain gap. And every unlabelled stack shares the 052924 prefix with the
KK2 labels, so using those stacks to improve a model evaluated on KK2 would be
training on the test set by another route.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
SAMPLES = ROOT / "data" / "confinedmig_cellTrack" / "sample_data"

#: Cellpose 3's default. Anything below this many instances never trains.
CELLPOSE_MIN_TRAIN_MASKS = 5

_DAY = re.compile(r"^(\d{6})")


def day_of(path: Path) -> str:
    match = _DAY.match(path.stem)
    return match.group(1) if match else "unknown"


def mask_count(path: Path) -> int:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    masks = np.asarray(np.load(seg, allow_pickle=True).item()["masks"])
    return len([v for v in np.unique(masks) if v])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "training_harness.json"))
    args = ap.parse_args()

    groups: dict[str, list[Path]] = {}
    for name in ("KK1", "KK2"):
        groups[name] = [
            p for p in sorted((TRAIN / name).glob("*.tif"))
            if p.with_name(p.name.replace(".tif", "_seg.npy")).exists()
        ]

    report: dict = {"groups": {}, "days": {}, "contamination": {}}

    print("=" * 72)
    print("1. WHAT REACHES THE NETWORK")
    print("=" * 72)
    total_images = total_kept = total_instances = kept_instances = 0
    for name, paths in groups.items():
        counts = [mask_count(p) for p in paths]
        histogram = dict(sorted(Counter(counts).items()))
        kept = [c for c in counts if c >= CELLPOSE_MIN_TRAIN_MASKS]
        total_images += len(counts)
        total_kept += len(kept)
        total_instances += sum(counts)
        kept_instances += sum(kept)
        report["groups"][name] = {
            "images": len(counts),
            "instances": int(sum(counts)),
            "mean_masks_per_image": round(float(np.mean(counts)), 2),
            "histogram_masks_per_image": {str(k): v for k, v in histogram.items()},
            "images_surviving_default": len(kept),
            "instances_surviving_default": int(sum(kept)),
        }
        print(f"{name}: {len(counts)} images, {sum(counts)} instances, "
              f"mean {np.mean(counts):.2f} masks/image")
        print(f"     masks-per-image histogram: {histogram}")
        print(f"     survive min_train_masks={CELLPOSE_MIN_TRAIN_MASKS}: "
              f"{len(kept)} images, {sum(kept)} instances")

    dropped_images = total_images - total_kept
    dropped_instances = total_instances - kept_instances
    report["dropped_at_default"] = {
        "images": dropped_images,
        "instances": dropped_instances,
        "share_of_images": round(dropped_images / max(total_images, 1), 4),
        "share_of_instances": round(dropped_instances / max(total_instances, 1), 4),
    }
    print()
    print(f"AT THE CELLPOSE DEFAULT, {dropped_images} of {total_images} images "
          f"({100 * dropped_images / total_images:.0f}%) never reach the network,")
    print(f"taking {dropped_instances} of {total_instances} hand-drawn instances "
          f"({100 * dropped_instances / total_instances:.0f}%) with them.")
    print("Any fine-tune that does not pass min_train_masks=0 trains on a fraction")
    print("of the data and is then compared against a model that did not.")

    print()
    print("=" * 72)
    print("2. WHAT THE SPLIT ACTUALLY SPLITS")
    print("=" * 72)
    for name, paths in groups.items():
        by_day: dict[str, list[int]] = defaultdict(list)
        for path in paths:
            by_day[day_of(path)].append(mask_count(path))
        report["days"][name] = {
            day: {"images": len(counts), "instances": int(sum(counts))}
            for day, counts in sorted(by_day.items())
        }
        parts = ", ".join(
            f"{day} ({len(c)} img / {sum(c)} inst)" for day, c in sorted(by_day.items())
        )
        print(f"{name}: {len(by_day)} acquisition day(s) -> {parts}")

    kk1_days = set(report["days"].get("KK1", {}))
    kk2_days = set(report["days"].get("KK2", {}))
    report["days_shared"] = sorted(kk1_days & kk2_days)
    print()
    if len(kk1_days) != len(kk2_days):
        print(f"The halves are NOT symmetric: KK1 spans {len(kk1_days)} day(s), "
              f"KK2 spans {len(kk2_days)}.")
        print("A model trained on fewer days has seen less variation, so the")
        print("difference between the two held-out scores is partly a difference")
        print("in training breadth rather than purely a domain gap.")

    print()
    print("=" * 72)
    print("3. CONTAMINATION BETWEEN THE UNLABELLED STACKS AND THE LABELS")
    print("=" * 72)
    stack_days = sorted({day_of(p) for p in SAMPLES.glob("*.tif")})
    report["contamination"]["unlabelled_stack_days"] = stack_days
    overlap = {}
    for name in groups:
        shared = sorted(set(stack_days) & set(report["days"][name]))
        overlap[name] = shared
        print(f"unlabelled stacks are from {stack_days}; {name} labels include {shared or 'none'}")
    report["contamination"]["overlap"] = overlap

    contaminated = [n for n, s in overlap.items() if s]
    if contaminated:
        print()
        print(f"The unlabelled time-lapse shares its acquisition day with {', '.join(contaminated)}.")
        print("Improving the model with those stacks and then reporting a score on")
        print("that half is not a held-out result. Any unlabelled-data method must")
        print("be evaluated on the half that does NOT share the day.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
