"""Dataset registry records, duplicate detection, and the locked split policy."""

from __future__ import annotations

import numpy as np
import pytest

from training import datasets, splits
from test_learn_sequences import write_still


def test_a_record_carries_metadata_hashes_and_a_bit_depth_guess(tmp_path):
    image = np.full((20, 30), 4000, np.uint16)
    masks = np.zeros((20, 30), np.int32)
    masks[2:8, 2:5] = 1
    masks[10:18, 20:24] = 2
    path = write_still(tmp_path / "041824_1.tif", image, t=44, n_t=80, date="20240418",
                       masks=masks, info=" BitsPerPixel = 12\n")
    record = datasets.record_for(path, "KK1")
    assert (record.experiment_id, record.t_index, record.n_t) == ("20240418-s01", 44, 80)
    assert record.bit_depth_guess == 12 and record.bits_per_pixel_metadata == 12
    assert record.n_instances == 2 and record.has_label
    assert record.sha256_tif == datasets.sha256_file(path)
    assert record.sha256_seg is not None and len(record.sha256_pixels) == 64
    assert record.source_name_private.startswith("20240418")

    bright = write_still(tmp_path / "052924_1.tif", np.full((20, 30), 60000, np.uint16), t=1)
    assert datasets.record_for(bright, "KK2").bit_depth_guess == 16


def test_pixel_identical_stills_are_linked_and_same_time_stills_are_listed(tmp_path):
    a = np.arange(600, dtype=np.uint16).reshape(20, 30)
    paths = [write_still(tmp_path / "s_2.tif", a, t=67, date="20240418"),
             write_still(tmp_path / "s_7.tif", a, t=67, date="20240418"),
             write_still(tmp_path / "s_8.tif", a[::-1].copy(), t=67, date="20240418")]
    records = [datasets.record_for(p, "KK1") for p in paths]
    datasets._link_duplicates(records)
    assert records[0].duplicate_of == ["KK1/s_7"] and records[2].duplicate_of == []
    assert records[2].same_time_as == ["KK1/s_2", "KK1/s_7"]


def _record(image_id, group, experiment, acquisition):
    return {"image_id": image_id, "group": group, "experiment_id": experiment,
            "acquisition_id": acquisition}


def test_the_locked_policy_puts_kk2_in_test_and_one_kk1_experiment_in_val():
    records = [_record("KK2/a", "KK2", "20240529-s01", "20240529"),
               _record("KK1/b", "KK1", "20230615-s04", "20230615"),
               _record("KK1/c", "KK1", "20240418-s01", "20240418"),
               _record("KK1/d", "KK1", "20240418-s42", "20240418"),
               _record("KK1/e", "KK1", None, None)]
    assert splits.assign(records) == {"KK2/a": "test", "KK1/b": "val", "KK1/c": "train",
                                      "KK1/d": "train", "KK1/e": None}


def test_a_split_that_cuts_an_acquisition_is_refused():
    records = [_record("KK1/c", "KK1", "20240418-s01", "20240418"),
               _record("KK1/d", "KK1", "20240418-s42", "20240418")]
    with pytest.raises(ValueError, match="straddles"):
        splits._check(records, {"KK1/c": "train", "KK1/d": "val"})


def test_the_sample_movies_experiment_is_never_train():
    assert "20240529-s01" in splits.NEVER_TRAIN_EXPERIMENTS


def _full(image_id, group, experiment, *, n=2, duplicate_of=()):
    return {**_record(image_id, group, experiment, experiment.rsplit("-s", 1)[0]),
            "has_label": True, "n_instances": n, "duplicate_of": list(duplicate_of)}


def _registry():
    return {"version": "dataset_test", "content_sha256": "f" * 64, "records": [
        _full("KK1/041824_2", "KK1", "20240418-s01", duplicate_of=["KK1/041824_7"]),
        _full("KK1/041824_7", "KK1", "20240418-s01", duplicate_of=["KK1/041824_2"]),
        _full("KK1/041824_8", "KK1", "20240418-s01"),
        _full("KK1/061523_1", "KK1", "20230615-s04", n=1),
        _full("KK2/052924_6", "KK2", "20240529-s01", n=3),
        _full("KK2/052924_15", "KK2", "20240529-s02", n=0)]}


def test_build_splits_partitions_by_experiment_and_counts():
    result = splits.build_splits(_registry())
    s = result["splits"]
    assert s["train"]["experiments"] == ["20240418-s01"]
    assert s["train"]["labelled_images"] == ["KK1/041824_2", "KK1/041824_7", "KK1/041824_8"]
    assert s["val"]["experiments"] == ["20230615-s04"] and s["val"]["instances"] == 1
    assert s["test"]["experiments"] == ["20240529-s01", "20240529-s02"]
    assert result["dataset_registry_sha256"] == "f" * 64 and result["locked"]
    every = [i for name in ("train", "val", "test") for i in s[name]["images"]]
    assert sorted(every) == sorted(r["image_id"] for r in _registry()["records"])


def test_select_images_refuses_held_out_data_and_drops_pixel_duplicates():
    from training.train_cellpose3 import select_images

    registry = _registry()
    locked = splits.build_splits(registry)
    chosen, dropped = select_images(registry, locked, ["train"])
    assert [r["image_id"] for r in chosen] == ["KK1/041824_2", "KK1/041824_8"]
    assert dropped == ["KK1/041824_7"]
    kept, none = select_images(registry, locked, ["train"], keep_duplicates=True)
    assert len(kept) == 3 and none == []
    # The test experiments are never-train, whatever a caller asks for.
    with pytest.raises(SystemExit, match="refusing"):
        select_images(registry, locked, ["test"])
    # A split file that lists a validation image under train too is refused.
    tampered = {**locked, "splits": {**locked["splits"], "train": {
        **locked["splits"]["train"],
        "labelled_images": locked["splits"]["train"]["labelled_images"] + ["KK1/061523_1"]}}}
    with pytest.raises(SystemExit, match="KK1/061523_1"):
        select_images(registry, tampered, ["train"])


def test_a_versioned_file_is_never_silently_rewritten(tmp_path):
    out = tmp_path / "splits_v1.json"
    assert datasets.write_versioned(out, "one", changed=False, users=[], supersede=False) is None
    assert "new version" in datasets.write_versioned(out, "two", changed=True, users=[],
                                                     supersede=False)
    assert out.read_text() == "one"
    assert "models were made with it" in datasets.write_versioned(
        out, "two", changed=True, users=["round5"], supersede=True)
    assert datasets.write_versioned(out, "two", changed=True, users=[], supersede=True) is None
    assert out.read_text() == "two"
    (kept,) = (tmp_path / "superseded").glob("splits_v1.*.json")
    assert kept.read_text() == "one"
