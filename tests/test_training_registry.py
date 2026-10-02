"""The research model registry: ids never reused, hashes checked, changes caught."""

from __future__ import annotations

import pytest

from training import registry


def test_register_find_add_metrics_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    store = tmp_path / "models.json"
    parent = tmp_path / "parent_ckpt"
    parent.write_bytes(b"parent")
    child = tmp_path / "child_ckpt"
    child.write_bytes(b"child")

    first = registry.register("parent", parent, path=store, architecture="cellpose3")
    entry = registry.register("child", child, parent="parent", path=store,
                              dataset_registry="dataset_v1", splits="splits_v1")
    assert entry["path"] == "child_ckpt" and entry["parent"] == "parent"
    assert registry.find(registry.load(store), first["sha256"][:12])["id"] == "parent"

    with pytest.raises(ValueError, match="never reused"):
        registry.register("child", child, path=store)
    copy = tmp_path / "copy"
    copy.write_bytes(b"child")
    with pytest.raises(ValueError, match="same SHA-256"):
        registry.register("copy", copy, path=store)
    orphan = tmp_path / "orphan"
    orphan.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="neither a registered id"):
        registry.register("orphan", orphan, parent="nobody", path=store)

    registry.add_metrics("child", "val_w3-97", {"f1": 0.5}, path=store)
    with pytest.raises(ValueError):
        registry.add_metrics("child", "val_w3-97", {"f1": 0.6}, path=store)
    assert registry.find(registry.load(store), "child")["metrics"] == {"val_w3-97": {"f1": 0.5}}

    assert {r["status"] for r in registry.verify(store)} == {"ok"}
    child.write_bytes(b"overwritten in place")
    assert {r["id"]: r["status"] for r in registry.verify(store)}["child"] == "CHANGED"


def test_held_out_follows_the_whole_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    store = tmp_path / "models.json"
    files = {}
    for name in ("public", "legacy", "v2", "unknown_parent_child", "silent"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    val, test = ["20230615-s04"], ["20240529-s01", "20240529-s02"]

    registry.register("public", files["public"], trained_on_experiments=[], path=store)
    # A pre-v2 checkpoint: all of KK1, the validation experiment included.
    registry.register("legacy", files["legacy"], parent="public", path=store,
                      trained_on_experiments=["20230615-s04", "20240418-s01", "20241223-s01"])
    # A v2 fine-tune of it on the train split only still inherits val.
    v2 = registry.register("v2", files["v2"], parent="legacy", path=store, trained_on=["train"],
                           trained_on_experiments=["20240418-s01", "20241223-s01"])
    verdict = registry.held_out(v2["sha256"], val, path=store)
    assert verdict == {"held_out": False, "lineage": ["v2", "legacy"],
                       "reason": "legacy was trained on 20230615-s04"}
    assert registry.held_out("v2", test, path=store) == {
        "held_out": True, "lineage": ["v2", "legacy", "public"], "reason": ""}
    # Another series of a held-out acquisition counts as the acquisition.
    assert not registry.held_out("v2", ["20240418-s42"], path=store)["held_out"]

    # An unregistered parent, or a model that does not say what it saw, is unknown.
    registry.register("unknown_parent_child", files["unknown_parent_child"], parent="a" * 64,
                      trained_on_experiments=[], path=store)
    verdict = registry.held_out("unknown_parent_child", test, path=store)
    assert not verdict["held_out"] and "is not in" in verdict["reason"]
    registry.register("silent", files["silent"], path=store)
    assert registry.held_out("silent", test, path=store)["reason"] == (
        "silent does not record what it was trained on")
    assert not registry.held_out("b" * 64, test, path=store)["held_out"]
