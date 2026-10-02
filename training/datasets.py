"""The dataset registry: what every supplied still is, from its own metadata.

One record per still under ``CellPose_TrainData/{KK1,KK2}``, labelled or not:
experiment (``<yyyymmdd>-s<series>``), true time index, frame interval, pixel
size, camera bit depth, crop, instance count and the SHA-256 of the image and of
its labels, plus the split it belongs to under the locked policy of
:mod:`training.splits`. Every training and evaluation report names the registry
version it used, so a figure can always be traced to the exact files behind it.

Facts it records rather than assumes (all read from the files):

- **Two instruments.** KK1 is a DS-Qi1Mc, 12-bit (max 4095), 0.639 um/px; KK2 a
  Flash4.0, 16-bit, 0.467 um/px. Equal widths in pixels are different widths in
  micrometres.
- **Duplicates.** ``041824_2`` and ``041824_7`` are pixel-identical (both t67 of
  20240418-s01) and were labelled twice, differently: a free intra-annotator
  agreement sample, and a double weight in any training set that keeps both.
  ``052924_6`` and ``052924_14`` are both t1 of 20240529-s01 but different
  crops, so they are not duplicates.
- **The sample movies** (``sample_data/*.tif``) are 20240529-s01, the same
  experiment as half the KK2 stills.

Output: ``build/registry/dataset_v1.json`` (gitignored). It is the only place
the source names are kept (``source_name_private``): they carry experimental
conditions and the repository is public, so nothing committed may copy them.

    python -m training.datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training import REGISTRY_DIR, ROOT, SAMPLE_ROOT, TRAIN_ROOT

from corridor.learn.sequences import (  # noqa: E402  (training/__init__ sets the path)
    CANONICAL_GROUPS,
    _index_of,
    load_image,
    read_still_meta,
    seg_path,
    split_into_crops,
)

DATASET_VERSION = "dataset_v1"
REGISTRY_PATH = REGISTRY_DIR / f"{DATASET_VERSION}.json"
#: A 12-bit camera never exceeds this; the KK1 stills peak at 3793-4095 and the
#: KK2 stills at 55938-65535, so the split is unambiguous on this data.
MAX_12_BIT = 4095


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_pixels(image: np.ndarray) -> str:
    """Hash of the pixel array alone, so a re-saved copy still matches."""
    array = np.ascontiguousarray(image)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class StillRecord:
    image_id: str
    group: str
    path: str
    experiment_id: str | None
    acquisition_id: str | None
    date_source: str | None
    series: int | None
    t_index: int | None
    n_t: int | None
    frame_interval_s: float | None
    frame_interval_min: float | None
    pixel_size_um: float | None
    shape: list[int]
    dtype: str
    max_value: int
    bits_per_pixel_metadata: int | None
    bit_depth_guess: int
    has_label: bool
    n_instances: int
    sha256_tif: str
    sha256_pixels: str
    sha256_seg: str | None
    crop_id: str | None = None
    #: Other stills that are the same pixels (by ``sha256_pixels``).
    duplicate_of: list[str] = field(default_factory=list)
    #: Other stills of the same experiment and time index (duplicates or not).
    same_time_as: list[str] = field(default_factory=list)
    split: str | None = None
    acquired_utc: str | None = None
    source_name_private: str | None = None


def _source_name(path: Path) -> str | None:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        labels = (tf.imagej_metadata or {}).get("Labels")
    label = labels[0] if isinstance(labels, (list, tuple)) and labels else labels
    match = re.match(r"^t:\d+/\d+\s+-\s+(.*?)\s*\(series\s+\d+\)\s*$", str(label or ""))
    return match.group(1) if match else None


def record_for(path: Path, group: str) -> StillRecord:
    image = load_image(path)
    meta = read_still_meta(path, group)
    seg = seg_path(path)
    n_instances = 0
    if seg.exists():
        masks = np.asarray(np.load(seg, allow_pickle=True).item()["masks"])
        n_instances = len([v for v in np.unique(masks) if v])
    max_value = int(image.max())
    return StillRecord(
        image_id=f"{group}/{path.stem}",
        group=group,
        path=_rel(path),
        experiment_id=meta.experiment_id if meta else None,
        acquisition_id=meta.acquisition_id if meta else None,
        date_source=meta.date_source if meta else None,
        series=meta.series if meta else None,
        t_index=meta.t_index if meta else None,
        n_t=meta.n_t if meta else None,
        frame_interval_s=meta.frame_interval_s if meta else None,
        frame_interval_min=meta.frame_interval_min if meta else None,
        pixel_size_um=meta.pixel_size_um if meta else None,
        shape=[int(v) for v in image.shape[:2]],
        dtype=str(image.dtype),
        max_value=max_value,
        bits_per_pixel_metadata=meta.bits_per_pixel if meta else None,
        bit_depth_guess=12 if max_value <= MAX_12_BIT else 16,
        has_label=seg.exists(),
        n_instances=n_instances,
        sha256_tif=sha256_file(path),
        sha256_pixels=sha256_pixels(image),
        sha256_seg=sha256_file(seg) if seg.exists() else None,
        acquired_utc=meta.acquired_utc if meta else None,
        source_name_private=_source_name(path),
    )


def _assign_crops(records: list[StillRecord]) -> None:
    """Crop ids by registration, within each experiment and image size.

    Members are ordered exactly as :func:`corridor.learn.sequences.find_sequences`
    orders them (time index, then trailing filename number, then name), so
    ``c0`` here is ``c0`` in the corrected-reference reports. Crop numbers follow
    the earliest member, and a crop's earliest member can depend on that
    tie-break: 20240529-s01 has a still of each crop at t1 (``052924_6``,
    ``052924_14``). The two agree as long as every still of a set is labelled,
    which holds for every set of the supplied data with more than one still.
    """
    by_set: dict[tuple, list[StillRecord]] = defaultdict(list)
    for record in records:
        by_set[(record.experiment_id, tuple(record.shape))].append(record)
    for (experiment, shape), members in by_set.items():
        members.sort(key=lambda r: (r.t_index or 0, _index_of(Path(r.path)), Path(r.path).name))
        tag = f"{experiment}-{shape[0]}x{shape[1]}"
        if len(members) == 1:
            members[0].crop_id = f"{tag}-c0"
            continue
        crops = split_into_crops([load_image(ROOT / r.path) for r in members])
        for number, crop in enumerate(crops):
            for index in crop.members:
                members[index].crop_id = f"{tag}-c{number}"


def _link_duplicates(records: list[StillRecord]) -> None:
    by_pixels: dict[str, list[str]] = defaultdict(list)
    by_time: dict[tuple, list[str]] = defaultdict(list)
    for r in records:
        by_pixels[r.sha256_pixels].append(r.image_id)
        if r.experiment_id and r.t_index is not None:
            by_time[(r.experiment_id, r.t_index)].append(r.image_id)
    for r in records:
        r.duplicate_of = sorted(i for i in by_pixels[r.sha256_pixels] if i != r.image_id)
        if r.experiment_id and r.t_index is not None:
            r.same_time_as = sorted(i for i in by_time[(r.experiment_id, r.t_index)]
                                    if i != r.image_id)


def annotation_agreement(a: Path, b: Path) -> dict:
    """How two labellings of the same pixels agree, scored by core.metrics."""
    from corridor.core.metrics import iou_matrix, score_image

    ma = np.asarray(np.load(seg_path(a), allow_pickle=True).item()["masks"]).astype(np.int32)
    mb = np.asarray(np.load(seg_path(b), allow_pickle=True).item()["masks"]).astype(np.int32)
    ious = iou_matrix(ma, mb)
    score = score_image(ma, mb)
    return {
        "a": a.stem, "b": b.stem,
        "instances": [int(ious.shape[0]), int(ious.shape[1])],
        "best_iou_per_instance_of_a": [round(float(v), 3) for v in ious.max(axis=1)]
        if ious.size else [],
        "f1_at_0.5": round(score.f1, 4),
    }


def sample_movies() -> list[dict]:
    """The unlabelled time-lapse stacks, with the experiment each came from."""
    import tifffile

    out = []
    for path in sorted(SAMPLE_ROOT.glob("*.tif")):
        with tifffile.TiffFile(path) as tf:
            ij = tf.imagej_metadata or {}
            labels = ij.get("Labels")
            shape = list(tf.series[0].shape)
        labels = labels if isinstance(labels, (list, tuple)) else [labels]
        times = [int(m.group(1)) for m in (re.match(r"^t:(\d+)/", str(l)) for l in labels) if m]
        meta = read_still_meta(path, "sample_data")
        out.append({
            "movie_id": f"sample_data/{path.stem}",
            "path": _rel(path),
            "shape": shape,
            "experiment_id": meta.experiment_id if meta else None,
            "t_first": min(times) if times else None,
            "t_last": max(times) if times else None,
            "frame_interval_min": meta.frame_interval_min if meta else None,
            "pixel_size_um": meta.pixel_size_um if meta else None,
            "sha256": sha256_file(path),
        })
    return out


def build_registry(groups: tuple[str, ...] = CANONICAL_GROUPS) -> dict:
    from training.splits import assign

    records: list[StillRecord] = []
    for group in groups:
        for path in sorted((TRAIN_ROOT / group).glob("*.tif")):
            records.append(record_for(path, group))
    _assign_crops(records)
    _link_duplicates(records)
    split_of = assign(records)
    for r in records:
        r.split = split_of.get(r.image_id)

    duplicates = []
    seen: set[str] = set()
    for r in records:
        if r.duplicate_of and r.image_id not in seen:
            group = [r.image_id, *r.duplicate_of]
            seen.update(group)
            entry = {"images": group, "kind": "pixel-identical"}
            labelled = [x for x in records if x.image_id in group and x.has_label]
            if len(labelled) >= 2:
                entry["annotation_agreement"] = annotation_agreement(
                    ROOT / labelled[0].path, ROOT / labelled[1].path)
            duplicates.append(entry)
    same_time = []
    for r in records:
        if r.same_time_as and r.image_id < min(r.same_time_as):
            group = [r.image_id, *r.same_time_as]
            kind = "pixel-identical" if set(r.same_time_as) <= set(r.duplicate_of) \
                else "same time point, different pixels"
            crops = sorted({x.crop_id for x in records if x.image_id in group})
            same_time.append({"images": group, "experiment_id": r.experiment_id,
                              "t_index": r.t_index, "kind": kind, "crops": crops})

    body = [asdict(r) for r in records]
    content_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {
        "version": DATASET_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": content_hash,
        "root": _rel(TRAIN_ROOT),
        "privacy": "source_name_private carries condition text: never copy it into "
                   "committed files; identify experiments by experiment_id only.",
        "n_stills": len(records),
        "n_labelled": sum(r.has_label for r in records),
        "n_instances": sum(r.n_instances for r in records),
        "duplicates": duplicates,
        "same_time_points": same_time,
        "sample_movies": sample_movies(),
        "records": body,
    }


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def models_using(key: str, value: str) -> list[str]:
    """Registered research models whose entry has ``key == value``."""
    from training.registry import MODELS_PATH, load

    if not MODELS_PATH.exists():
        return []
    return [e["id"] for e in load(MODELS_PATH)["models"] if e.get(key) == value]


def write_versioned(out: Path, text: str, *, changed: bool, users: list[str],
                    supersede: bool) -> str | None:
    """Write a versioned registry file; None on success, else why it was refused.

    A version name is a promise that its content never changes. Different
    content under an existing name is refused, unless ``supersede`` is given
    *and* no registered model was trained or scored against the old content;
    the old file is then kept under ``superseded/`` with its hash in the name,
    never deleted.
    """
    if out.exists() and changed:
        if users:
            return (f"{out} differs and models were made with it ({', '.join(users)}): "
                    f"write a new version instead")
        if not supersede:
            return (f"{out} exists with different content: a changed {out.stem} is a new "
                    f"version (pass --out), or --supersede while no model uses it")
        archive = out.parent / "superseded"
        archive.mkdir(parents=True, exist_ok=True)
        old = out.read_bytes()
        digest = hashlib.sha256(old).hexdigest()[:16]
        (archive / f"{out.stem}.{digest}{out.suffix}").write_bytes(old)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return None


def labelled_records(registry: dict, split: str | None = None) -> list[dict]:
    rows = [r for r in registry["records"] if r["has_label"]]
    return [r for r in rows if split is None or r["split"] == split]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the dataset registry.")
    ap.add_argument("--out", default=str(REGISTRY_PATH))
    ap.add_argument("--supersede", action="store_true",
                    help="replace an existing registry with different content, keeping the old "
                         "one under superseded/ (refused if any registered model used it)")
    args = ap.parse_args()

    registry = build_registry()
    out = Path(args.out)
    old_sha = load_registry(out)["content_sha256"] if out.exists() else None
    refusal = write_versioned(
        out, json.dumps(registry, indent=2),
        changed=old_sha is not None and old_sha != registry["content_sha256"],
        users=models_using("dataset_registry_sha256", old_sha) if old_sha else [],
        supersede=args.supersede)
    if refusal:
        print(f"refusing: {refusal}")
        return 2

    print(f"{registry['version']}: {registry['n_stills']} stills, "
          f"{registry['n_labelled']} labelled, {registry['n_instances']} instances")
    by_experiment: dict[str, list[dict]] = defaultdict(list)
    for r in registry["records"]:
        by_experiment[r["experiment_id"]].append(r)
    for experiment, rows in sorted(by_experiment.items()):
        crops = sorted({r["crop_id"] for r in rows})
        print(f"  {experiment:14s} {rows[0]['group']} split={rows[0]['split']:5s} "
              f"stills={len(rows):2d} labelled={sum(r['has_label'] for r in rows):2d} "
              f"instances={sum(r['n_instances'] for r in rows):3d} "
              f"t={sorted(r['t_index'] for r in rows)} "
              f"px={rows[0]['pixel_size_um']:.4f}um "
              f"dt={rows[0]['frame_interval_min']:.2f}min "
              f"bits={rows[0]['bit_depth_guess']} crops={len(crops)}")
    for d in registry["duplicates"]:
        print(f"  duplicate: {d['images']} {d.get('annotation_agreement', '')}")
    for d in registry["same_time_points"]:
        print(f"  same time point: {d['images']} t{d['t_index']} ({d['kind']}, crops {d['crops']})")
    for m in registry["sample_movies"]:
        print(f"  movie {m['movie_id']}: {m['experiment_id']} t{m['t_first']}-{m['t_last']}")
    print(f"wrote {out}  (content sha256 {registry['content_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
