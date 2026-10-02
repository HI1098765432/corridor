"""Experiment-level splits, with the test set frozen before anything is tuned.

**The locked test set is every KK2 experiment** (20240529-s01, 20240529-s02 and
the unlabelled 20240409-s01). That is the published held-out direction -- train
on KK1, score on KK2, the comparison behind the 0.482 baseline and every figure
since -- and KK2 is a different instrument (Flash4.0, 16-bit, 0.467 um/px) as
well as a different day, so it tests the transfer the application actually
faces. It is fixed here, in code, before any v2 training run exists.

**Validation is 20230615-s04** (KK1, 5 stills, 17 instances). It is the only
choice that is

- labelled (20230704-s02, the ``34-34`` still, is one empty frame);
- its own acquisition: 20240418-s01 and -s42 are two stage positions of one
  .nd2 file -- one dish, one day, one condition -- so either as validation
  would leak into the other in training;
- small enough to leave most of KK1 to train on: 20241223-s01 would take 56 of
  KK1's 138 instances (41 %), 20230615-s04 takes 17 (12 %).

**Train is the rest of KK1**: 20240418-s01, 20240418-s42, 20241223-s01 and
20230704-s02.

**The five sample movies are 20240529-s01**, the same experiment (and series)
as half the KK2 stills, t1-t52 of the same 54-frame movie. While KK2 is the test
set they must never be used for training, pseudo-labelling or hard-example
mining: they would put the test experiment's own cells into training. The same
holds for any Corridor result folder made from them.

Splits are by experiment and, through the acquisition rule above, never cut an
acquisition either. Frames of one experiment never straddle splits; the
pixel-identical ``041824_2`` / ``041824_7`` are in the same split by
construction.

    python -m training.splits          # writes build/registry/splits_v1.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from training import REGISTRY_DIR

SPLIT_VERSION = "splits_v1"
SPLITS_PATH = REGISTRY_DIR / f"{SPLIT_VERSION}.json"

#: Frozen. Changing any of these is a new split version, never an edit.
TEST_GROUPS = ("KK2",)
VALIDATION_EXPERIMENTS = ("20230615-s04",)
#: Experiments that must never feed training while the test set above stands.
#: The sample movies are 20240529-s01 (verified from their ImageJ labels by
#: :func:`training.datasets.sample_movies`).
NEVER_TRAIN_EXPERIMENTS = ("20240529-s01", "20240529-s02", "20240409-s01")

JUSTIFICATION = {
    "test": ("All KK2 experiments: the published held-out direction (KK1 -> KK2) and a "
             "different instrument (Flash4.0 16-bit 0.467 um/px against DS-Qi1Mc 12-bit "
             "0.639 um/px) as well as a different day. Frozen before any v2 training."),
    "validation": ("20230615-s04: the only labelled KK1 experiment that is its own "
                   "acquisition (20240418-s01 and -s42 share one .nd2 file) and small "
                   "enough (17 of 138 KK1 instances) to leave training most of KK1."),
    "train": "The remaining KK1 experiments.",
    "sample_movies": ("sample_data/*.tif are 20240529-s01, the test experiment itself. Never "
                      "train on them, pseudo-label them or mine hard examples from them while "
                      "KK2 is the test set; the same applies to any Corridor results made "
                      "from them."),
}


def split_of_record(record) -> str | None:
    """The split of one registry record (a dict or a StillRecord)."""
    get = record.get if isinstance(record, dict) else lambda k: getattr(record, k)
    if get("group") in TEST_GROUPS:
        return "test"
    if get("experiment_id") in VALIDATION_EXPERIMENTS:
        return "val"
    if get("experiment_id") is None:
        return None  # no metadata: no defensible split, so never used
    return "train"


def assign(records) -> dict[str, str | None]:
    out = {}
    for record in records:
        image_id = record["image_id"] if isinstance(record, dict) else record.image_id
        out[image_id] = split_of_record(record)
    _check(records, out)
    return out


def _check(records, split_of: dict) -> None:
    """Refuse a split that cuts an experiment or an acquisition."""
    by_experiment, by_acquisition = defaultdict(set), defaultdict(set)
    for record in records:
        get = record.get if isinstance(record, dict) else lambda k, r=record: getattr(r, k)
        split = split_of[get("image_id")]
        by_experiment[get("experiment_id")].add(split)
        by_acquisition[(get("group"), get("acquisition_id"))].add(split)
    for key, splits in {**by_experiment, **by_acquisition}.items():
        if len(splits) > 1:
            raise ValueError(f"{key} straddles splits {sorted(map(str, splits))}")


def build_splits(registry: dict) -> dict:
    records = registry["records"]
    split_of = assign(records)
    experiments: dict[str, dict] = {}
    for r in records:
        e = experiments.setdefault(r["experiment_id"], {
            "experiment_id": r["experiment_id"], "group": r["group"],
            "acquisition_id": r["acquisition_id"], "split": split_of[r["image_id"]],
            "images": [], "labelled_images": 0, "instances": 0,
        })
        e["images"].append(r["image_id"])
        e["labelled_images"] += int(r["has_label"])
        e["instances"] += r["n_instances"]

    splits = {}
    for name in ("train", "val", "test"):
        rows = [r for r in records if split_of[r["image_id"]] == name]
        splits[name] = {
            "experiments": sorted({r["experiment_id"] for r in rows}),
            "images": sorted(r["image_id"] for r in rows),
            "labelled_images": sorted(r["image_id"] for r in rows if r["has_label"]),
            "instances": sum(r["n_instances"] for r in rows),
        }
    movies = registry.get("sample_movies", [])
    return {
        "version": SPLIT_VERSION,
        "locked": True,
        "dataset_registry": registry["version"],
        "dataset_registry_sha256": registry["content_sha256"],
        "policy": {
            "test_groups": list(TEST_GROUPS),
            "validation_experiments": list(VALIDATION_EXPERIMENTS),
            "never_train_experiments": list(NEVER_TRAIN_EXPERIMENTS),
        },
        "justification": JUSTIFICATION,
        "splits": splits,
        "experiments": sorted(experiments.values(), key=lambda e: e["experiment_id"]),
        "sample_movies": [
            {"movie_id": m["movie_id"], "experiment_id": m["experiment_id"],
             "t_first": m["t_first"], "t_last": m["t_last"],
             "usable_for_training": m["experiment_id"] not in NEVER_TRAIN_EXPERIMENTS}
            for m in movies
        ],
    }


def load_splits(path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def images_of(splits: dict, split: str, *, labelled_only: bool = True) -> list[str]:
    key = "labelled_images" if labelled_only else "images"
    return list(splits["splits"][split][key])


def main() -> int:
    from training.datasets import (
        REGISTRY_PATH,
        build_registry,
        load_registry,
        models_using,
        write_versioned,
    )

    ap = argparse.ArgumentParser(description="Write the locked experiment-level splits.")
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--out", default=str(SPLITS_PATH))
    ap.add_argument("--supersede", action="store_true",
                    help="replace a split file whose train/val or registry hash changed, keeping "
                         "the old one under superseded/ (never for a changed test set, never "
                         "once a registered model used it)")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    registry = load_registry(registry_path) if registry_path.exists() else build_registry()
    result = build_splits(registry)

    out = Path(args.out)
    users: list[str] = []
    changed = False
    if out.exists():
        existing = load_splits(out)
        if existing["splits"]["test"]["images"] != result["splits"]["test"]["images"]:
            print(f"refusing: {out} is locked and its test set differs from this policy. "
                  f"A changed test set is a new split version, never an edit.")
            return 2
        # Locked means all of it: a moved validation image changes every
        # figure selected on val just as surely.
        changed = existing != json.loads(json.dumps(result))
        users = [m for m in models_using("splits", existing["version"])
                 if m in models_using("dataset_registry_sha256",
                                      existing["dataset_registry_sha256"])]
    refusal = write_versioned(out, json.dumps(result, indent=2), changed=changed, users=users,
                              supersede=args.supersede)
    if refusal:
        print(f"refusing: {refusal}")
        return 2
    for name, s in result["splits"].items():
        print(f"{name:5s} {len(s['labelled_images']):2d} labelled images, "
              f"{s['instances']:3d} instances: {', '.join(s['experiments'])}")
    for m in result["sample_movies"]:
        print(f"movie {m['movie_id']}: {m['experiment_id']} -> "
              f"{'usable' if m['usable_for_training'] else 'NEVER for training'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
