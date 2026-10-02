"""Sequences are built from ImageJ metadata, not from filenames or image size.

Every still here is synthetic and written with real ImageJ metadata
(``Labels`` ``t:N/M - <name>.nd2 (series S)``, ``finterval``, ``XResolution``),
so the tests exercise the same parser the supplied data goes through. The
failure modes pinned are the ones measured on the real stills: filename order
that is not time order, two crops of one experiment sharing a size, frames of
one crop offset by several pixels, and two stills at the same time point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from corridor.learn import sequences as sq

PIXEL_UM = 0.467
INTERVAL_S = 1200.0
CONDITION = "secretdrug 5uM"


def _field(seed: int = 0, shape=(260, 900)) -> np.ndarray:
    """A static device field: periodic walls plus fixed debris landmarks."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = np.full(shape, 1000.0)
    field[:, ::40] += 400.0  # walls, one every 40 px
    debris = np.zeros(shape)
    ys = rng.integers(0, shape[0], 120)
    xs = rng.integers(0, shape[1], 120)
    debris[ys, xs] = rng.choice([-1.0, 1.0], 120) * 3000.0
    return field + gaussian_filter(debris, 2.0) * 6.0


def _still(field: np.ndarray, y0: int, x0: int, shape=(200, 300), seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crop = field[y0:y0 + shape[0], x0:x0 + shape[1]].copy()
    return np.clip(crop + rng.normal(0, 15, crop.shape), 0, 65535).astype(np.uint16)


def write_still(path: Path, image: np.ndarray, *, t: int, n_t: int = 54, series: int = 1,
                date: str | None = "20240529", masks: np.ndarray | None = None,
                info: str = " BitsPerPixel = 16\ndTimeAbsolute = 2460460.377812998\n") -> Path:
    name = f"{date} {CONDITION}" if date else CONDITION
    tifffile.imwrite(
        path, image, imagej=True, resolution=(1 / PIXEL_UM, 1 / PIXEL_UM),
        metadata={"unit": "micron", "finterval": INTERVAL_S,
                  "Labels": [f"t:{t}/{n_t} - {name}.nd2 (series {series:02d})"],
                  "Info": info})
    if masks is None:
        masks = np.zeros(image.shape, np.int32)
    np.save(path.with_name(path.name.replace(".tif", "_seg.npy")), {"masks": masks},
            allow_pickle=True)
    return path


def test_metadata_gives_time_experiment_and_calibration(tmp_path):
    path = write_still(tmp_path / "052924_3.tif", np.zeros((20, 30), np.uint16), t=7, series=2)
    meta = sq.read_still_meta(path, "KK2")
    assert meta.t_index == 7 and meta.n_t == 54 and meta.series == 2
    assert meta.experiment_id == "20240529-s02" and meta.date_source == "label"
    assert meta.pixel_size_um == pytest.approx(PIXEL_UM, rel=1e-6)
    assert meta.frame_interval_min == pytest.approx(20.0)
    assert meta.time_min == pytest.approx(6 * 20.0)
    assert meta.bits_per_pixel == 16
    # The condition text is parsed past and never kept.
    assert CONDITION.split()[0] not in repr(meta)


def test_date_falls_back_to_the_filename_then_the_acquisition_start(tmp_path):
    named = write_still(tmp_path / "122324_1.tif", np.zeros((8, 8), np.uint16), t=3, date=None)
    meta = sq.read_still_meta(named, "KK1")
    assert (meta.experiment_id, meta.date_source) == ("20241223-s01", "filename")

    anonymous = write_still(tmp_path / "34-34.tif", np.zeros((8, 8), np.uint16), t=34,
                            series=2, date=None)
    meta = sq.read_still_meta(anonymous, "KK1")
    # dTimeAbsolute 2460460.3778 is 2024-05-29 21:04 UTC.
    assert (meta.experiment_id, meta.date_source) == ("20240529-s02", "nd2_start_utc")


def test_a_still_without_a_time_label_is_not_guessed_into_a_sequence(tmp_path):
    path = tmp_path / "x_1.tif"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint16), imagej=True)
    assert sq.read_still_meta(path) is None


def test_time_order_replaces_filename_order(tmp_path):
    group = tmp_path / "KK2"
    group.mkdir()
    field = _field()
    # Filename order 1..4, time order t10, t20, t30, t40 scrambled across names.
    for name, t in (("052924_1", 30), ("052924_2", 10), ("052924_3", 20), ("052924_4", 40)):
        write_still(group / f"{name}.tif", _still(field, 20, 100, seed=t), t=t)

    (seq,) = sq.find_sequences(tmp_path, groups=("KK2",), order="time")
    assert seq.ordering == "time"
    assert [p.stem for p in seq.paths] == ["052924_2", "052924_3", "052924_1", "052924_4"]
    assert seq.t_indices == [10, 20, 30, 40]
    assert seq.experiment_id == "20240529-s01"
    assert seq.elapsed_min(0, 1) == pytest.approx(10 * INTERVAL_S / 60.0)
    assert seq.elapsed_min(3, 0) == pytest.approx(-30 * INTERVAL_S / 60.0)
    assert seq.name == "KK2-20240529-s01-c0"

    (legacy,) = sq.find_sequences(tmp_path, groups=("KK2",), order="filename")
    assert [p.stem for p in legacy.paths] == ["052924_1", "052924_2", "052924_3", "052924_4"]
    assert legacy.name == "KK2-200x300"
    with pytest.raises(ValueError):
        legacy.elapsed_min(0, 1)

    # Unported callers name no order: they keep what they published, and are told.
    with pytest.warns(FutureWarning, match="withdrawn filename ordering"):
        (default,) = sq.find_sequences(tmp_path, groups=("KK2",))
    assert default.ordering == "filename" and default.paths == legacy.paths


def test_two_crops_of_one_experiment_are_two_sequences(tmp_path):
    group = tmp_path / "KK2"
    group.mkdir()
    field = _field(seed=3)
    for i, t in enumerate((1, 5, 9)):
        write_still(group / f"a_{i}.tif", _still(field, 20, 40, seed=10 + t), t=t)
        write_still(group / f"b_{i}.tif", _still(field, 20, 560, seed=20 + t), t=t + 1)

    sequences = sq.find_sequences(tmp_path, groups=("KK2",), order="time")
    assert len(sequences) == 2
    members = sorted(sorted(p.stem[0] for p in s.paths) for s in sequences)
    assert members == [["a", "a", "a"], ["b", "b", "b"]]
    # Shape grouping would have called this one six-frame movie.
    (legacy,) = sq.find_sequences(tmp_path, groups=("KK2",), order="filename")
    assert legacy.n_frames == 6


def test_offsets_register_a_jittered_crop(tmp_path):
    group = tmp_path / "KK1"
    group.mkdir()
    field = _field(seed=5)
    jitter = {1: (0, 0), 2: (6, -10), 3: (-4, 8), 4: (10, 14)}
    for t, (dy, dx) in jitter.items():
        write_still(group / f"041824_{t}.tif", _still(field, 30 + dy, 200 + dx, seed=t), t=t,
                    date="20240418")
    (seq,) = sq.find_sequences(tmp_path, groups=("KK1",), order="time")
    for k, t in enumerate(seq.t_indices):
        dy, dx = jitter[t]
        # A point at p in frame t is at p + (dy, dx) in the reference (t1) frame.
        assert seq.offsets_px[k][0] == pytest.approx(dy, abs=2)
        assert seq.offsets_px[k][1] == pytest.approx(dx, abs=2)
    x, y = seq.to_reference(2, (50.0, 60.0))
    assert seq.from_reference(2, (x, y)) == pytest.approx((50.0, 60.0))
    assert seq.frame_correlation > sq.MIN_FRAME_CORRELATION
    assert seq.looks_like_a_movie


def test_two_stills_at_one_time_point_share_it(tmp_path):
    group = tmp_path / "KK1"
    group.mkdir()
    field = _field(seed=7)
    image = _still(field, 20, 300, seed=1)
    write_still(group / "041824_2.tif", image, t=67, date="20240418")
    write_still(group / "041824_7.tif", image, t=67, date="20240418")
    write_still(group / "041824_8.tif", _still(field, 20, 300, seed=2), t=69, date="20240418")
    (seq,) = sq.find_sequences(tmp_path, groups=("KK1",), order="time")
    assert seq.t_indices == [67, 67, 69]
    assert [p.stem for p in seq.paths] == ["041824_2", "041824_7", "041824_8"]
    assert seq.elapsed_min(0, 1) == 0.0


def test_crop_split_uses_landmarks_not_walls():
    """Two crops one wall pitch apart look identical to a wall-dominated score."""
    field = _field(seed=11)
    a = _still(field, 20, 40, seed=1).astype(np.float32)
    b = _still(field, 20, 280, seed=2).astype(np.float32)  # 6 pitches over
    same = _still(field, 24, 44, seed=3).astype(np.float32)
    score_same, shift = sq.register(sq.structure_image(a), sq.structure_image(same))
    score_other, _ = sq.register(sq.structure_image(a), sq.structure_image(b))
    assert score_same > sq.SAME_CROP_NCC > score_other
    assert shift == pytest.approx((4.0, 4.0), abs=2)
