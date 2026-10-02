"""Boundary refinement moves boundaries a little and never touches instance identity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter

from training import refine

SHAPE = (120, 160)


def scene():
    """Two bright cells on a dark field, and predictions drawn 2 px too fat."""
    truth = np.zeros(SHAPE, np.int32)
    truth[20:80, 40:50] = 1
    truth[30:100, 100:110] = 2
    image = np.where(truth > 0, 900.0, 300.0)
    image = gaussian_filter(image, 1.0) + np.random.default_rng(0).normal(0, 10, SHAPE)
    fat = np.zeros(SHAPE, np.int32)
    for label in (1, 2):
        fat[binary_dilation(truth == label, iterations=2)] = label
    return image, truth, fat


def iou(a, b):
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


@pytest.mark.parametrize("method,variant", [(m, "geodesic") for m in refine.METHODS]
                         + [("contour", "chan_vese")])
def test_invariants_hold_for_every_method(method, variant):
    image, truth, fat = scene()
    out = refine.refine_labels(image, fat, method=method, max_px=3, contour_variant=variant)
    assert set(np.unique(out)) == set(np.unique(fat))  # nothing created or deleted
    for label in (1, 2):
        before, after = fat == label, out == label
        assert refine.displacement_px(before, after) <= 3.0 + 1e-9
        from scipy.ndimage import label as cc

        assert cc(after)[1] == 1


def test_image_refinement_pulls_a_fat_outline_onto_the_cell():
    image, truth, fat = scene()
    out = refine.refine_labels(image, fat, method="image", max_px=3)
    for label in (1, 2):
        assert iou(out == label, truth == label) > iou(fat == label, truth == label)


def test_an_instance_never_takes_another_instances_pixels():
    image, _, _ = scene()
    labels = np.zeros(SHAPE, np.int32)
    labels[20:80, 40:50] = 1
    labels[20:80, 50:56] = 2  # touching neighbour, drawn over part of the bright cell
    out = refine.refine_labels(image, labels, method="image", max_px=3)
    assert not np.any((labels == 2) & (out == 1))
    assert not np.any((labels == 1) & (out == 2))


def test_background_contested_by_two_refinements_goes_to_neither():
    image = np.full(SHAPE, 300.0)
    image[30:70, 60:71] = 900.0  # one bright cell, half under each prediction
    labels = np.zeros(SHAPE, np.int32)
    labels[30:70, 57:65] = 1
    labels[30:70, 66:74] = 2
    # Column 65 is bright, unlabelled and inside both 3 px bands.
    out = refine.refine_labels(image, labels, method="image", max_px=3)
    assert not np.any(out[30:70, 65])
    assert not np.any((labels == 2) & (out == 1)) and not np.any((labels == 1) & (out == 2))
    assert set(np.unique(out)) == {0, 1, 2}


def test_an_instance_the_method_empties_keeps_its_original_mask():
    image = np.full(SHAPE, 300.0)  # featureless: nothing to snap to
    labels = np.zeros(SHAPE, np.int32)
    labels[50:52, 50:52] = 3  # too small to erode
    out = refine.refine_labels(image, labels, method="contour", max_px=3)
    assert np.array_equal(out == 3, labels == 3)


def test_clamp_keeps_the_displacement_band():
    prior = np.zeros(SHAPE, bool)
    prior[40:80, 40:60] = True
    wild = np.zeros(SHAPE, bool)
    wild[10:110, 10:150] = True
    clamped = refine.clamp_to_band(wild, prior, 3)
    assert refine.displacement_px(prior, clamped) <= 3.0
    shrunk = refine.clamp_to_band(np.zeros(SHAPE, bool), prior, 3)
    assert refine.displacement_px(prior, shrunk) <= 3.0


def test_unknown_methods_are_refused():
    with pytest.raises(ValueError):
        refine.refine_labels(np.zeros(SHAPE), np.zeros(SHAPE, np.int32), method="magic")
