"""Ground-truth tests for Bot 5 (temporal delta), ENGINE_4D section 4.

Acceptance: E_FB is low for a true temporal match and high for a swapped pair.
A displacement is planted so the phase-correlation sign and magnitude are
checked against the truth, not asserted.
"""

from __future__ import annotations

import numpy as np

from corridor.engine.temporal_delta import (
    TemporalSettings,
    consistency,
    measure_object,
)


def _blob(h: int, w: int, cy: float, cx: float, sigma: float = 3.0) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2))


def _disk(h: int, w: int, cy: float, cx: float, r: float = 5.0) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2


def _scene(h: int, w: int, centres, amp: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 0.01, size=(h, w))
    for (cy, cx) in centres:
        img = img + amp * _blob(h, w, cy, cx)
    return img


def test_phase_displacement_recovers_the_planted_shift():
    # A single blob moves by a known (dy, dx); the forward displacement must
    # recover it in sign and magnitude.
    h, w = 64, 64
    p = (32.0, 20.0)
    d = (2.0, 6.0)
    q = (p[0] + d[0], p[1] + d[1])
    images = np.stack([_scene(h, w, [p]), _scene(h, w, [q])])
    masks = np.zeros((2, h, w), dtype=np.int32)
    masks[0][_disk(h, w, *p)] = 1
    masks[1][_disk(h, w, *q)] = 1

    rec = consistency(images, masks, t=0, label_t=1, target_t=1, label_target=1)

    assert rec.valid
    assert abs(rec.displacement_fwd_px[0] - d[0]) < 0.5
    assert abs(rec.displacement_fwd_px[1] - d[1]) < 0.5
    # the backward leg reports the opposite motion
    assert abs(rec.displacement_back_px[0] + d[0]) < 0.5
    assert abs(rec.displacement_back_px[1] + d[1]) < 0.5


def test_true_match_is_low_and_a_swap_is_high():
    h, w = 64, 64
    p = (32.0, 20.0)
    d = (2.0, 6.0)
    true_pos = (p[0] + d[0], p[1] + d[1])
    decoy = (10.0, 50.0)
    images = np.stack([_scene(h, w, [p]), _scene(h, w, [true_pos, decoy])])
    masks = np.zeros((2, h, w), dtype=np.int32)
    masks[0][_disk(h, w, *p)] = 1
    masks[1][_disk(h, w, *true_pos)] = 1  # label 1 = the real continuation
    masks[1][_disk(h, w, *decoy)] = 2     # label 2 = a different cell

    good = consistency(images, masks, t=0, label_t=1, target_t=1, label_target=1)
    swap = consistency(images, masks, t=0, label_t=1, target_t=1, label_target=2)

    assert good.valid and swap.valid
    # the true correspondence projects onto itself both ways
    assert good.e_fb < 2.0
    assert good.iou_fwd > 0.8 and good.iou_back > 0.8
    # the swap lands nowhere near the wrong cell
    assert swap.e_fb > 10.0 * good.e_fb + 10.0
    assert swap.iou_fwd == 0.0


def test_measure_object_marks_the_true_continuation_best():
    h, w = 64, 64
    p = (32.0, 20.0)
    d = (2.0, 6.0)
    true_pos = (p[0] + d[0], p[1] + d[1])
    decoy = (10.0, 50.0)
    images = np.stack([_scene(h, w, [p]), _scene(h, w, [true_pos, decoy])])
    masks = np.zeros((2, h, w), dtype=np.int32)
    masks[0][_disk(h, w, *p)] = 1
    masks[1][_disk(h, w, *true_pos)] = 1
    masks[1][_disk(h, w, *decoy)] = 2

    # only window +1 exists (two frames); windows=(1,) keeps the test focused
    records = measure_object(images, masks, t=0, label_t=1,
                             settings=TemporalSettings(windows=(1,)))

    forward = [r for r in records if r.window_k == 1]
    assert len(forward) == 2
    best = [r for r in forward if r.best]
    assert len(best) == 1
    assert best[0].label_target == 1  # the true continuation, not the decoy


def test_windows_cover_plus_and_minus_two():
    # Five frames of a blob drifting by (0, 4) per frame; +-1 and +-2 from the
    # middle frame must all be measured, and the one-frame step recovered.
    h, w = 64, 96
    r = 5.0
    cy = 32.0
    step = 4.0
    centres = [(cy, 20.0 + step * k) for k in range(5)]
    images = np.stack([_scene(h, w, [c]) for c in centres])
    masks = np.zeros((5, h, w), dtype=np.int32)
    for k, c in enumerate(centres):
        masks[k][_disk(h, w, c[0], c[1], r)] = 1

    records = measure_object(images, masks, t=2, label_t=1,
                             settings=TemporalSettings(windows=(1, 2)))

    offsets = sorted({r.window_k for r in records})
    assert offsets == [-2, -1, 1, 2]
    # every single object per frame: each window has exactly one candidate,
    # and each is a true match -> low E_FB and the right per-step displacement
    for rec in records:
        assert rec.valid
        assert rec.e_fb < 2.0
        # forward displacement magnitude matches |k| * step in x
        assert abs(abs(rec.displacement_fwd_px[1]) - abs(rec.window_k) * step) < 0.6
