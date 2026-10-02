"""The evaluation CLI end to end on a synthetic split, without Cellpose."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from training import evaluate_segmentation as ev

SHAPE = (80, 100)


def cell(mask, label, y0, x0):
    mask[y0:y0 + 40, x0:x0 + 8] = label
    return mask


@pytest.fixture
def dataset(tmp_path):
    group = tmp_path / "KK2"
    group.mkdir()
    ref_dir = tmp_path / "reference" / "KK2"
    ref_dir.mkdir(parents=True)
    records, predictions = [], {}
    for i in range(2):
        image = np.full(SHAPE, 300, np.uint16)
        labels = cell(cell(np.zeros(SHAPE, np.int32), 1, 10, 20), 2, 20, 60)
        image[labels > 0] = 900
        path = group / f"img_{i}.tif"
        tifffile.imwrite(path, image)
        np.save(group / f"img_{i}_seg.npy", {"masks": labels}, allow_pickle=True)
        corrected = labels.copy()
        if i == 0:
            cell(corrected, 3, 20, 40)  # one cell the annotator did not draw
        np.save(ref_dir / f"img_{i}_masks.npy", corrected)
        predictions[f"KK2__img_{i}"] = labels
        records.append({"image_id": f"KK2/img_{i}", "path": str(path), "group": "KK2",
                        "experiment_id": "20240101-s01", "has_label": True})
    (tmp_path / "registry.json").write_text(json.dumps(
        {"version": "dataset_test", "content_sha256": "abc", "records": records}))
    (tmp_path / "splits.json").write_text(json.dumps(
        {"version": "splits_test", "dataset_registry_sha256": "abc",
         "splits": {"test": {"labelled_images": [r["image_id"] for r in records]}}}))
    np.savez_compressed(tmp_path / "pred.npz", **predictions)
    return tmp_path


def common(tmp_path):
    return ["--registry", str(tmp_path / "registry.json"), "--splits",
            str(tmp_path / "splits.json"), "--reference-dir", str(tmp_path / "reference"),
            "--models", str(tmp_path / "models.json")]


def test_scores_against_the_corrected_reference_first_and_the_original_beside_it(dataset):
    out = dataset / "report.json"
    argv = common(dataset) + ["--predictions", str(dataset / "pred.npz"), "--border-margin",
                              "2", "--refine", "image", "--out", str(out)]
    assert ev.main(argv) == 0
    report = json.loads(out.read_text())
    assert report["reference"] == {"corrected_v2_dir": (dataset / "reference").as_posix(),
                                   "original_instances": 4, "corrected_v2_instances": 5,
                                   "undrawn_cells_added_in_v2": 1}
    unmodified = report["results"][0]
    assert unmodified["variant"] == "unmodified"
    assert list(unmodified)[2:4] == ["corrected_v2", "original"]
    assert unmodified["original"]["at_iou_0.5"]["f1"] == 1.0
    assert unmodified["corrected_v2"]["at_iou_0.5"]["recall"] == 0.8  # 4 of 5
    assert unmodified["corrected_v2"]["ap_50_95"] == 0.8
    assert unmodified["corrected_v2"]["fn_per_image"] == 0.5
    assert set(unmodified["corrected_v2"]["breakdowns"]) == {
        "experiment", "contrast_quartile", "size_quartile_px", "border_distance_px"}
    assert [r["variant"] for r in report["results"]] == [
        "unmodified", "border_margin_2px", "refine_image_3px"]
    # A report is never overwritten.
    assert ev.main(argv) == 2


def test_dry_run_hashes_the_checkpoint_and_loads_no_model(dataset, capsys):
    checkpoint = dataset / "fake_checkpoint"
    checkpoint.write_bytes(b"not a model")
    argv = common(dataset) + ["--checkpoint", str(checkpoint), "--dry-run",
                              "--window", "3,97", "--diameter", "36"]
    assert ev.main(argv) == 0
    printed = capsys.readouterr().out
    assert "dry run" in printed and "+1 undrawn" in printed
    import hashlib

    assert hashlib.sha256(b"not a model").hexdigest() in printed


def test_a_split_the_checkpoint_may_have_seen_is_reported_as_not_held_out(
        dataset, capsys, monkeypatch):
    from training import registry

    monkeypatch.setattr(registry, "ROOT", dataset)
    models = dataset / "models.json"
    unknown = dataset / "unregistered_checkpoint"
    unknown.write_bytes(b"pre-v2")
    seen = dataset / "trained_on_it"
    seen.write_bytes(b"trained on 20240101-s01")
    registry.register("seen", seen, trained_on_experiments=["20240101-s01"], path=models)
    clean = dataset / "never_saw_it"
    clean.write_bytes(b"trained elsewhere")
    registry.register("clean", clean, trained_on_experiments=["20230615-s04"], path=models)

    for checkpoint, held, reason in (
            (unknown, False, "is not in"), (seen, False, "seen was trained on 20240101-s01"),
            (clean, True, "")):
        out = dataset / f"{checkpoint.name}.json"
        # Inference is not what is under test: score the saved predictions as if
        # they came from this checkpoint.
        monkeypatch.setattr(ev, "predict", lambda *a, **k: {
            f"KK2/img_{i}": np.load(dataset / "pred.npz")[f"KK2__img_{i}"] for i in range(2)})
        monkeypatch.setattr(ev, "cellpose_version", lambda: "3.1.1.3")
        assert ev.main(common(dataset) + ["--checkpoint", str(checkpoint), "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["held_out"] is held and reason in report["held_out_reason"]
        printed = capsys.readouterr().out
        assert ("NOT HELD OUT" in printed) is (not held)

    out = dataset / "from_predictions.json"
    assert ev.main(common(dataset) + ["--predictions", str(dataset / "pred.npz"),
                                      "--out", str(out)]) == 0
    assert json.loads(out.read_text())["held_out"] is False


def test_there_is_no_default_model(dataset):
    assert ev.main(common(dataset)) == 2
    assert ev.main(common(dataset) + ["--checkpoint", str(dataset / "missing")]) == 2


def test_a_missing_corrected_reference_stops_the_run(dataset):
    (dataset / "reference" / "KK2" / "img_1_masks.npy").unlink()
    with pytest.raises(SystemExit):
        ev.main(common(dataset) + ["--predictions", str(dataset / "pred.npz"),
                                   "--out", str(dataset / "r.json")])


def test_windows_and_diameters_parse_strictly():
    assert ev.parse_window("default") is True
    assert ev.parse_window("3,97") == {"percentile": (3.0, 97.0)}
    with pytest.raises(Exception):
        ev.parse_window("97,3")
    assert ev.parse_diameter("none") is None and ev.parse_diameter("36") == 36.0
