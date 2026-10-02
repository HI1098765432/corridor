"""The research model registry: every checkpoint a research run produced, by hash.

``build/registry/models.json`` (gitignored, like the weights) holds one entry per
checkpoint: id, path, SHA-256, parent (the checkpoint it was trained from, by
id or by hash), training report, dataset registry and split versions, the
augmentation policy, and any metrics recorded against it later. This is the
research side only. The application's validated model lives in
``src/corridor/assets/model_registry.json`` and is locked by hash
(``docs/NEXT_GENERATION.md`` section 2); nothing here can put a model there.

Ids are never reused and an entry's hash never changes: a retrained model is a
new entry. ``verify`` rehashes every file, so a checkpoint overwritten in place
(Cellpose's ``train_seg`` silently replaces ``save_path/models/<name>``) is
caught rather than scored under its old name.

**What a model was trained on travels with it.** Each entry records the
experiments it saw (``trained_on_experiments``; ``None`` when unknown, ``[]``
for a public model that saw none of ours). :func:`held_out` walks the parent
chain, because a fine-tune inherits everything its start checkpoint saw: every
pre-v2 checkpoint (``corridor_contrast_invariant``, the round-4 run of
``scripts/train_contrast_invariant.py``) was trained on all of KK1, the locked
validation experiment 20230615-s04 included, and so is everything fine-tuned
from one. A model whose lineage is not fully recorded is never called held out.

    python -m training.registry list
    python -m training.registry verify
    python -m training.registry add --id <id> --path <checkpoint> [--parent <id-or-sha>] \\
        --trained-on-experiments 20230615-s04,20240418-s01,...   # or 'none'
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training import REGISTRY_DIR, ROOT
from training.datasets import sha256_file

MODELS_PATH = REGISTRY_DIR / "models.json"


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def load(path: Path = MODELS_PATH) -> dict:
    if not Path(path).exists():
        return {"version": 1, "models": []}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(registry: dict, path: Path = MODELS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp.replace(path)


def find(registry: dict, key: str) -> dict | None:
    """An entry by id, or by full or prefix SHA-256 (at least 12 characters)."""
    for entry in registry["models"]:
        if entry["id"] == key:
            return entry
    if len(key) >= 12:
        hits = [e for e in registry["models"] if e["sha256"].startswith(key)]
        if len(hits) == 1:
            return hits[0]
    return None


def register(model_id: str, checkpoint: Path, *, parent: str | None = None,
             training_report: Path | None = None, dataset_registry: str | None = None,
             dataset_registry_sha256: str | None = None, splits: str | None = None,
             augmentation_policy: str | None = None, architecture: str = "",
             trained_on: list[str] | None = None,
             trained_on_experiments: list[str] | None = None,
             notes: str = "", path: Path = MODELS_PATH) -> dict:
    """Add a checkpoint. Refuses a reused id and a hash already registered.

    ``trained_on`` names the splits, ``trained_on_experiments`` the experiments
    this checkpoint's own training saw -- not its parent's, which
    :func:`held_out` follows separately.
    """
    registry = load(path)
    if any(e["id"] == model_id for e in registry["models"]):
        raise ValueError(f"model id {model_id!r} is already registered; ids are never reused")
    digest = sha256_file(Path(checkpoint))
    same = [e["id"] for e in registry["models"] if e["sha256"] == digest]
    if same:
        raise ValueError(f"{checkpoint} has the same SHA-256 as {same[0]!r}")
    if parent is not None and find(registry, parent) is None and len(parent) != 64:
        raise ValueError(f"parent {parent!r} is neither a registered id nor a full SHA-256")
    entry = {
        "id": model_id,
        "path": _rel(Path(checkpoint)),
        "sha256": digest,
        "architecture": architecture,
        "parent": parent,
        "training_report": _rel(training_report) if training_report else None,
        "dataset_registry": dataset_registry,
        "dataset_registry_sha256": dataset_registry_sha256,
        "splits": splits,
        "augmentation_policy": augmentation_policy,
        "trained_on": list(trained_on) if trained_on is not None else None,
        "trained_on_experiments": (sorted(trained_on_experiments)
                                   if trained_on_experiments is not None else None),
        "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {},
        "notes": notes,
    }
    registry["models"].append(entry)
    save(registry, path)
    return entry


def _acquisition(experiment: str) -> str:
    return experiment.rsplit("-s", 1)[0]


def held_out(checkpoint: str, experiments: list[str], *, path: Path = MODELS_PATH) -> dict:
    """Is a checkpoint, and every checkpoint it descends from, recorded as never
    trained on any of ``experiments`` (or on another series of their acquisitions)?

    ``checkpoint`` is a registered id or SHA-256. Returns ``held_out`` True only
    when the whole lineage is registered with its training experiments and none
    overlaps; otherwise False with the reason. Acquisitions count because the
    locked splits never cut one: two series of one .nd2 file share a dish, a
    day and a condition.
    """
    registry = load(path)
    targets = set(experiments)
    target_acquisitions = {_acquisition(e) for e in targets}
    lineage: list[str] = []
    key: str | None = checkpoint
    while key is not None:
        entry = find(registry, key)
        if entry is None:
            return {"held_out": False, "lineage": lineage,
                    "reason": (f"{key[:16]} is not in {_rel(path)}: what it was trained on "
                               f"is unknown")}
        if entry["id"] in lineage:
            return {"held_out": False, "lineage": lineage,
                    "reason": f"the parent chain loops at {entry['id']}"}
        lineage.append(entry["id"])
        seen = entry.get("trained_on_experiments")
        if seen is None:
            return {"held_out": False, "lineage": lineage,
                    "reason": f"{entry['id']} does not record what it was trained on"}
        overlap = sorted(e for e in seen if e in targets or _acquisition(e) in target_acquisitions)
        if overlap:
            return {"held_out": False, "lineage": lineage,
                    "reason": f"{entry['id']} was trained on {', '.join(overlap)}"}
        key = entry.get("parent")
    return {"held_out": True, "lineage": lineage, "reason": ""}


def add_metrics(key: str, name: str, metrics: dict, *, path: Path = MODELS_PATH) -> dict:
    """Attach a named result (e.g. ``test_w3-97_d36``) to a registered model."""
    registry = load(path)
    entry = find(registry, key)
    if entry is None:
        raise KeyError(f"no registered model {key!r}")
    if name in entry["metrics"]:
        raise ValueError(f"{entry['id']} already has metrics {name!r}; use a new name")
    entry["metrics"][name] = metrics
    save(registry, path)
    return entry


def verify(path: Path = MODELS_PATH) -> list[dict]:
    """Rehash every registered checkpoint; returns one status row per entry."""
    rows = []
    for entry in load(path)["models"]:
        file = ROOT / entry["path"]
        if not file.exists():
            rows.append({"id": entry["id"], "status": "missing"})
            continue
        status = "ok" if sha256_file(file) == entry["sha256"] else "CHANGED"
        rows.append({"id": entry["id"], "status": status})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Research model registry.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("verify")
    add = sub.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--path", required=True)
    add.add_argument("--parent", default=None)
    add.add_argument("--training-report", default=None)
    add.add_argument("--architecture", default="")
    add.add_argument("--trained-on-experiments", default=None,
                     help="comma-separated experiment ids this checkpoint's own training saw, "
                          "or 'none' for a model trained on none of ours; omitted = unknown")
    add.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.command == "list":
        for e in load()["models"]:
            print(f"{e['id']:40s} {e['sha256'][:12]}  parent={e['parent']}  {e['path']}")
        return 0
    if args.command == "verify":
        rows = verify()
        for row in rows:
            print(f"{row['id']:40s} {row['status']}")
        return 0 if all(r["status"] == "ok" for r in rows) else 1
    seen = args.trained_on_experiments
    experiments = None if seen is None else (
        [] if seen.strip().lower() == "none" else [e.strip() for e in seen.split(",") if e.strip()])
    entry = register(args.id, Path(args.path), parent=args.parent,
                     training_report=Path(args.training_report) if args.training_report else None,
                     architecture=args.architecture, trained_on_experiments=experiments,
                     notes=args.notes)
    print(f"registered {entry['id']} {entry['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
