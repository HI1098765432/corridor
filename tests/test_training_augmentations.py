"""Augmentations keep labels exact, never move a boundary photometrically, and replay."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from training import augmentations as aug

SHAPE = (90, 130)


def scene():
    labels = np.zeros(SHAPE, np.int32)
    labels[10:70, 20:30] = 1
    labels[30:85, 80:89] = 2
    labels[5:15, 100:125] = 3  # horizontal: nothing here assumes vertical cells
    rng = np.random.default_rng(0)
    image = 1000.0 + 600.0 * (labels > 0) + rng.normal(0, 20, SHAPE)
    return image.astype(np.float32), labels


PHOTOMETRIC_CASES = [
    ("contrast", {"factor": 2.0, "sigma_px": 20.0}),
    ("gain_offset", {"gain": 1.3, "offset": 0.04}),
    ("gamma", {"gamma": 0.7}),
    ("gradient", {"amplitude": 0.3, "angle_rad": 1.0}),
    ("vignette", {"strength": 0.3, "cy": 0.4, "cx": 0.6}),
    ("defocus", {"sigma_px": 2.0}),
    ("anisotropic_blur", {"sigma_px": 3.0, "ratio": 0.2, "angle_rad": 0.7}),
    ("downsample", {"factor": 3}),
    ("poisson", {"photons": 30.0, "seed": 1}),
    ("read_noise", {"sigma": 0.04, "seed": 2}),
]


@pytest.mark.parametrize("name,params", PHOTOMETRIC_CASES)
def test_photometric_operations_never_move_a_boundary(name, params):
    image, labels = scene()
    before = labels.copy()
    out, lab = aug.replay(image, labels, [{"op": name, **params}])
    assert np.array_equal(lab, before)
    assert out.shape == image.shape and out.dtype == np.float32
    assert not np.array_equal(out, image)  # it did something to the image
    # ...and what it did does not depend on where the labels are: the old
    # contrast transform rescaled only near labelled cells, a halo the network
    # could learn instead of the cell. Different labels, the same image out.
    moved = np.roll(labels, 17, axis=1)
    moved[0:20, 0:20] = 9
    other, _ = aug.replay(image, moved, [{"op": name, **params}])
    assert np.array_equal(out, other)
    # And no photometric operation can be handed the labels at all.
    import inspect

    assert "labels" not in inspect.signature(aug.PHOTOMETRIC[name]).parameters


def _label_shift_px(mode: int, factor: float) -> np.ndarray:
    """Label centroid minus the cell's true resized centre, over sub-pixel phases.

    Cells 9-12 px wide at eight positions each, resized on OpenCV's pixel-centre
    grid, where a source span [x0, x0 + w) has its centre at (x0 + w/2) f - 1/2.
    """
    cv2 = aug._cv2()
    shifts = []
    for x0 in range(20, 28):
        for width in (9, 10, 11, 12):
            labels = np.zeros((96, 96), np.int32)
            labels[30:70, x0:x0 + width] = 1
            size = int(round(96 * factor))
            resized = cv2.resize(labels, (size, size), interpolation=mode) > 0
            xs = np.nonzero(resized)[1]
            shifts.append(xs.mean() - ((x0 + width / 2) * size / 96 - 0.5))
    return np.asarray(shifts)


@pytest.mark.parametrize("factor", [0.75, 1.35])
def test_nearest_exact_removes_most_of_the_label_shift(factor):
    """The plain nearest mode of scale_variant (scripts/train_contrast_invariant.py)
    shifts every label the same way; the exact mode used here does not."""
    cv2 = aug._cv2()
    plain = _label_shift_px(cv2.INTER_NEAREST, factor)
    exact = _label_shift_px(cv2.INTER_NEAREST_EXACT, factor)
    assert plain.min() > -1e-6 and plain.mean() > 0.35  # one direction, ~0.4-0.5 px
    assert abs(exact.mean()) < 0.15 and exact.min() < 0 < exact.max()
    assert abs(exact.mean()) < plain.mean() / 2.5


@pytest.mark.parametrize("op", [{"op": "flip", "axis": 0}, {"op": "flip", "axis": 1},
                                {"op": "rot90", "k": 1}, {"op": "rot90", "k": 3}])
def test_flips_and_rotations_carry_labels_exactly(op):
    image, labels = scene()
    out, lab = aug.replay(image, labels, [op])
    assert set(np.unique(lab)) == set(np.unique(labels))
    for v in (1, 2, 3):
        assert (lab == v).sum() == (labels == v).sum()
    # Image and labels still coincide pixel for pixel.
    assert np.array_equal(out > 1300, lab > 0)


@pytest.mark.parametrize("factor", [0.75, 1.35])
def test_scaling_keeps_image_and_labels_aligned_and_invents_no_id(factor):
    image, labels = scene()
    out, lab = aug.replay(image, labels, [{"op": "scale", "factor": factor}])
    assert out.shape == lab.shape
    assert set(np.unique(lab)) <= set(np.unique(labels))
    bright = out > 1300
    overlap = np.logical_and(bright, lab > 0).sum() / np.logical_or(bright, lab > 0).sum()
    assert overlap > 0.97


def test_a_copy_replays_bit_for_bit_from_its_record():
    image, labels = scene()
    policy = aug.AugmentationPolicy(seed=7, p_poisson=1.0, p_read_noise=1.0)
    out, lab, record = aug.augment(image, labels, policy, source="KK1/041824_1", copy=1)
    replayed, relab = aug.replay(image, labels, json.loads(json.dumps(record.to_dict()))["ops"])
    assert np.array_equal(out, replayed) and np.array_equal(lab, relab)
    again, _, record2 = aug.augment(image, labels, policy, source="KK1/041824_1", copy=1)
    assert np.array_equal(out, again) and record2.ops == record.ops
    other, _, _ = aug.augment(image, labels, policy, source="KK1/041824_1", copy=2)
    assert not np.array_equal(out, other)


def test_every_sampled_parameter_is_recorded():
    policy = aug.AugmentationPolicy(p_flip=1, p_scale=1, p_contrast=1, p_gain=1, p_gamma=1,
                                    p_gradient=1, p_vignette=1, p_defocus=1,
                                    p_anisotropic=1, p_downsample=1, p_poisson=1,
                                    p_read_noise=1)
    ops = aug.sample_ops(policy, np.random.default_rng(3))
    names = [o["op"] for o in ops]
    assert {"flip", "scale", "contrast", "gain_offset", "gamma", "gradient", "vignette",
            "defocus", "anisotropic_blur", "downsample", "poisson",
            "read_noise"} <= set(names)
    for o in ops:
        assert len(o) > 1  # an op without its parameters could not be replayed


def test_the_policy_round_trips_through_json():
    policy = aug.AugmentationPolicy(seed=4, copies_per_image=3)
    assert aug.AugmentationPolicy.from_dict(json.loads(policy.to_json())) == policy


def test_the_local_background_has_no_preferred_direction():
    image, _ = scene()
    assert np.allclose(aug.local_background(image.T, 20.0),
                       aug.local_background(image, 20.0).T, atol=1e-3)


def test_directional_effects_take_random_angles():
    policy = aug.AugmentationPolicy(p_gradient=1, p_anisotropic=1)
    angles = [o["angle_rad"] for seed in range(200)
              for o in aug.sample_ops(policy, np.random.default_rng(seed))
              if o["op"] == "gradient"]
    quadrants = np.histogram(np.mod(angles, 2 * math.pi), bins=4, range=(0, 2 * math.pi))[0]
    assert quadrants.min() > 20
    ks = [o["k"] for seed in range(200)
          for o in aug.sample_ops(policy, np.random.default_rng(seed)) if o["op"] == "rot90"]
    assert set(ks) == {1, 2, 3}


def test_a_smoke_tile_is_centred_on_the_cells_and_padded_to_size():
    from training.train_cellpose_sam import tile_around_cells

    image, labels = scene()
    tile, tile_labels = tile_around_cells(image, labels, size=64)
    assert tile.shape == tile_labels.shape == (64, 64)
    assert set(np.unique(tile_labels)) - {0}  # some cell is inside
    # Smaller than the tile along one axis: reflected image, zero labels below.
    small, small_labels = tile_around_cells(image[:40], labels[:40], size=64)
    assert small.shape == small_labels.shape == (64, 64)
    assert not small_labels[40:].any() and small[40:].std() > 0


def test_training_set_keeps_originals_and_empty_frames_once():
    image, labels = scene()
    empty = np.zeros(SHAPE, np.int32)
    policy = aug.AugmentationPolicy(copies_per_image=2)
    images, masks, records = aug.build_training_set(
        [("a", image, labels), ("b", image, empty)], policy)
    assert len(images) == len(masks) == len(records) == 4  # a, a+2 copies, b
    assert [r.copy for r in records] == [-1, 0, 1, -1]
    assert np.array_equal(masks[0], labels) and np.array_equal(images[0], image)
