"""Hard-example ranking on a synthetic Corridor result with planted defects."""

from __future__ import annotations

import csv
import json

import numpy as np
import tifffile

from training import hard_examples as hx

T, SHAPE = 7, (80, 120)


def cell(frame, label, x, y, length=30, width=6):
    frame[y:y + length, x:x + width] = label


def write_result(folder, *, input_label: str | None = None):
    """Two cells moving down; frame 3 empty, frame 5 merged, frame 1 broken."""
    folder.mkdir()
    masks = np.zeros((T,) + SHAPE, np.int32)
    rows = []
    for t in range(T):
        if t == 3:
            continue  # nothing detected: the planted hole
        if t == 5:
            masks[t, 10 + 2 * t:60 + 2 * t, 20:70] = 1  # one blob over both cells
            continue
        cell(masks[t], 1, 20, 10 + 2 * t)
        cell(masks[t], 2, 60, 15 + 2 * t)
        if t == 1:
            masks[t, 70:74, 100:104] = 1  # a second, detached piece of label 1
        for track, x in ((1, 20), (2, 60)):
            rows.append({"track_id": track, "frame": t, "x_px": x, "y_px": 10 + 2 * t})
    np.savez_compressed(folder / "masks.npz", masks=masks)
    with open(folder / "tracks.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "frame", "x_px", "y_px"])
        writer.writeheader()
        writer.writerows(rows)
    if input_label is not None:
        movie = folder / "movie.tif"
        tifffile.imwrite(movie, np.zeros((T,) + SHAPE, np.uint16), imagej=True,
                         metadata={"axes": "TYX", "Labels": [input_label] * T})
        (folder / "run.json").write_text(json.dumps({"input": {"path": str(movie)}}))
    return folder


def by_frame(report):
    return {row["frame"]: row for row in report["frames"]}


def test_planted_defects_are_found_and_the_hole_ranks_first(tmp_path):
    folder = write_result(tmp_path / "r",
                          input_label="t:1/80 - 20240418 movie.nd2 (series 01)")
    report = hx.rank([folder])
    frames = by_frame(report)
    assert report["frames"][0]["frame"] == 3
    assert frames[3]["signals"]["zero_between_occupied"] == 1
    assert frames[3]["signals"]["track_gaps"] == 2
    assert frames[5]["signals"]["oversized_masks"] == 1
    assert frames[1]["signals"]["broken_masks"] == 1
    assert frames[0]["signals"]["zero_between_occupied"] == 0
    # A frame whose cells continue into the next one, at a steady count, scores nothing.
    assert frames[0]["score"] == 0.0


def test_results_from_the_test_experiment_are_refused(tmp_path):
    held = write_result(tmp_path / "held",
                        input_label="t:1/54 - 20240529 movie.nd2 (series 01)")
    report = hx.rank([held])
    assert report["frames"] == []
    assert report["refused_held_out"] == [{"result": held.as_posix(),
                                           "experiment": "20240529-s01"}]
    allowed = hx.rank([held], allow_held_out=True)
    assert allowed["held_out_included"] == [held.as_posix()] and allowed["frames"]


def test_a_result_of_unknown_experiment_is_refused_unless_allowed(tmp_path):
    no_run = write_result(tmp_path / "no_run")
    moved = write_result(tmp_path / "moved",
                         input_label="t:1/54 - 20240529 movie.nd2 (series 01)")
    (moved / "movie.tif").unlink()  # the input no longer resolves on this machine
    report = hx.rank([no_run, moved])
    assert report["frames"] == []
    assert [(r["result"], r["why"], r["included"]) for r in report["unknown_experiment"]] == [
        (no_run.as_posix(), "no run.json", False),
        (moved.as_posix(), f"input {moved / 'movie.tif'} does not exist here", False)]
    allowed = hx.rank([no_run], allow_unknown_experiment=True)
    assert allowed["frames"] and allowed["unknown_experiment"][0]["included"]
    assert hx.main([str(no_run)]) == 3


def test_other_experiments_are_ranked(tmp_path):
    folder = write_result(tmp_path / "ok",
                          input_label="t:1/80 - 20240418 movie.nd2 (series 01)")
    report = hx.rank([folder])
    assert report["refused_held_out"] == [] and report["frames"][0]["experiment"] == "20240418-s01"
