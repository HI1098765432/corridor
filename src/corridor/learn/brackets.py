"""The time-aware bracket rule: which cells did the annotator provably not draw.

A cell drawn at the nearest earlier labelled time *t0* and again at the nearest
later labelled time *t1* of the same crop, and not drawn at *t* in between, was
missed at *t* -- if, and only if, "the same cell" means something across that
bracket. The withdrawn rule (``order="filename"``) took filename neighbours as
consecutive frames and called two centroids within 45 px the same cell. On the
true time order neither holds: brackets span 51 min to 12 h, and the rule has to
know how far apart in time its evidence is.

**Why identity is by overlap, not by distance.** Measured on the stills
(``docs/RESEARCH_V2.md``): one-frame steps reach 2.77 um/min (1.98 um/min in the
114 tracked steps of the sample movies), the shortest bracket in the data is
51 min, and the nearest other cell in the same frame is a median 58 um away on
KK2 (5th percentile 37 um, about one channel pitch). A same-cell radius of
``5 um + 3 um/min x span`` is therefore at least 158 um, and with it **not one**
of the 198 bracketed cells (109 KK1, 89 KK2) has a unique partner on the far
side (:func:`distance_rule_partners`, recomputed in every label-completeness
report). Distance alone cannot tell a cell from its neighbour here. Overlap can: a cell (median length
46 um on KK2, 59 um on KK1) that moved less than its own length still covers
part of its old footprint,
while a cell in the next channel never touches it, whatever the orientation of
the channels. So *t0* and *t1* cells are the same cell when their registered
masks overlap at IoU >= :data:`LINK_IOU` and the pairing is unique in both
directions; the speed bound stays as a plausibility gate.

**Why brackets longer than 90 min are refused.** Leave-one-out on the cells that
*were* drawn: for every overlap-linked t0/t1 pair, is there a drawn cell where
the time-interpolated outline says it should be? Pooled over KK1 and KK2, into
frames where something is drawn: for brackets of 51-80 min, 25 of 26; for
91-122 min, 6 of 12; for 153-540 min, 10 of 17. (Pairs into frames with nothing
drawn -- 3 short and 4 long -- are left out: there every pair "fails" because
the frame was not annotated, which is the missing label being measured, not an
interpolation error.) The spans in the data fall into those clusters, so any
limit from 80.0 up to, not including, 91.3 min gives the same result. The case
rests on 12 pairs in the middle cluster; it is a measured cut, not a precise
one, and :func:`leave_one_out` recomputes it with every report.

**What can be checked at all.** A frame is checkable only if the rule could
have found an undrawn cell there: a short enough bracket, something drawn on
both sides of it, and at least one cell linked across it (:class:`FrameCheck`).
A frame beside an unannotated one fails the second test, and its labelled cells
stay out of a ceiling's denominator, where they could only have inflated it.

**What counts as drawn.** The cell's outline -- the human mask from the nearer
labelled time, moved to the position interpolated by **elapsed-time fraction**
(not the midpoint) -- grown by :data:`POSITION_TOLERANCE_UM`. Any drawn mask at
*t* touching it accounts for the cell, which makes the claim conservative: a
neighbour in the same channel can veto a missed cell, never invent one.

Evidence is always the original labels. A cell added at *t* never becomes
evidence for another addition (the withdrawn script mutated its masks in place,
so its additions could chain).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sequences import Sequence

#: Upper bound on centroid speed, um/min. The fastest step measured anywhere in
#: the supplied data is 2.77 um/min (two KK1 stills 10 min apart); the 114
#: one-frame steps of the v1.3.0 baseline tracks reach 1.98 um/min with a
#: 5 um/min tracker gate that never bound.
MAX_SPEED_UM_PER_MIN = 3.0
#: Centroid noise that is not motion: two annotations of the same image
#: (041824_2 and _7, pixel-identical) put the same three cells 1.7-2.5 um apart,
#: and registration is quantised to 2 px (1.3 um on KK1).
POSITION_TOLERANCE_UM = 5.0
#: Longest bracket, in minutes, across which an absent cell is evidence of a
#: missed label. See the module docstring for the measurement.
MAX_BRACKET_MIN = 90.0
#: Two masks at t0 and t1 are one cell when they overlap at least this much.
#: The same threshold :mod:`corridor.learn.reconstruct` links frames with.
LINK_IOU = 0.15
#: An interpolated outline left with fewer pixels than this inside the frame is
#: mostly outside it: the cell may have left the field, so nothing is claimed.
MIN_ADDED_PX = 80


@dataclass(frozen=True)
class BracketRule:
    max_speed_um_per_min: float = MAX_SPEED_UM_PER_MIN
    position_tolerance_um: float = POSITION_TOLERANCE_UM
    max_bracket_min: float = MAX_BRACKET_MIN
    link_iou: float = LINK_IOU
    min_added_px: int = MIN_ADDED_PX

    def to_dict(self) -> dict:
        return {
            "max_speed_um_per_min": self.max_speed_um_per_min,
            "position_tolerance_um": self.position_tolerance_um,
            "max_bracket_min": self.max_bracket_min,
            "link_iou": self.link_iou,
            "min_added_px": self.min_added_px,
            "identity": "unique mutual mask overlap (IoU) between t0 and t1, registered",
            "position": "t0 centroid + (t1 - t0 centroid) * elapsed fraction",
            "outline": "human mask from the nearer labelled time, translated",
            "drawn_here": "any drawn mask at t touching the outline grown by the tolerance",
        }


@dataclass
class Bracket:
    target: int
    before: int
    after: int
    span_min: float
    #: Elapsed-time fraction of the target between before (0) and after (1).
    fraction: float


@dataclass
class FrameCheck:
    """Whether one frame could be checked at all, and why not.

    Checkable means the rule could have found an undrawn cell here: the bracket
    is short enough, both of its sides have drawn cells, and at least one cell
    links across it. A frame next to an unannotated one (``041824_9``,
    ``052924_7`` and ``052924_26`` have nothing drawn) is not checkable, and its
    labelled cells must not enter a ceiling's denominator, because no outcome of
    the test could have counted against them.
    """

    image: str
    t_index: int
    checkable: bool
    reason: str = ""
    bracket: Bracket | None = None
    labelled: int = 0
    #: t0/t1 cell pairs the rule tested in this frame (0 when not checkable).
    linked_pairs: int = 0


@dataclass
class Undrawn:
    """A cell drawn either side of a frame and provably not drawn in it."""

    image: str
    sequence: str
    target: int
    #: Interpolated centroid in the target frame's own pixels, (x, y).
    at: tuple[float, float]
    t_index: int
    t0_index: int
    t1_index: int
    span_min: float
    fraction: float
    displacement_um: float
    link_iou: float
    #: The added outline in the target frame (boolean), for the corrected
    #: reference. Not serialised.
    region: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]


def bracket_of(sequence: Sequence, k: int, rule: BracketRule = BracketRule()) -> FrameCheck:
    """The nearest earlier and later labelled times of member k, or why there are none.

    Members sharing a time index (the pixel-identical 041824_2 and _7) are one
    time point: the earlier-named of them is the evidence, and each is still a
    target in its own right.
    """
    image = sequence.paths[k].name
    if sequence.ordering != "time" or sequence.frame_interval_min is None:
        return FrameCheck(image, -1, False, "no true time order")
    t = sequence.t_indices[k]
    earlier = [i for i, ti in enumerate(sequence.t_indices) if ti < t]
    later = [i for i, ti in enumerate(sequence.t_indices) if ti > t]
    if not earlier:
        return FrameCheck(image, t, False, "first labelled time of its crop")
    if not later:
        return FrameCheck(image, t, False, "last labelled time of its crop")
    t0 = max(sequence.t_indices[i] for i in earlier)
    t1 = min(sequence.t_indices[i] for i in later)
    before = min(i for i in earlier if sequence.t_indices[i] == t0)
    after = min(i for i in later if sequence.t_indices[i] == t1)
    span = sequence.elapsed_min(before, after)
    fraction = sequence.elapsed_min(before, k) / span
    bracket = Bracket(k, before, after, span, fraction)
    if span > rule.max_bracket_min:
        return FrameCheck(image, t, False,
                          f"bracket t{t0}-t{t1} spans {span:.0f} min > {rule.max_bracket_min:.0f}",
                          bracket)
    return FrameCheck(image, t, True, "", bracket)


def _shift(region: np.ndarray, dy: float, dx: float) -> np.ndarray:
    out = np.zeros_like(region)
    h, w = region.shape
    sy, sx = int(round(dy)), int(round(dx))
    y0, y1 = max(0, sy), min(h, h + sy)
    x0, x1 = max(0, sx), min(w, w + sx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = region[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    return out


def _into(sequence: Sequence, region: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Member src's mask, moved into member dst's pixels by the registration."""
    if not sequence.offsets_px:
        return region
    dy = sequence.offsets_px[src][0] - sequence.offsets_px[dst][0]
    dx = sequence.offsets_px[src][1] - sequence.offsets_px[dst][1]
    return _shift(region, dy, dx)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _regions(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    return [(int(v), mask == v) for v in np.unique(mask) if v]


def _centre(region: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(region)
    return float(xs.mean()), float(ys.mean())


def find_undrawn(
    sequence: Sequence,
    masks: list[np.ndarray],
    rule: BracketRule = BracketRule(),
) -> tuple[list[Undrawn], list[FrameCheck]]:
    """Every cell the bracket rule says was not drawn, and every frame's status.

    ``masks`` are the original labels of ``sequence.paths`` in order; they are
    not modified. Returns the claims and one :class:`FrameCheck` per member.
    """
    from scipy.ndimage import binary_dilation

    if sequence.pixel_size_um is None:
        raise ValueError(f"{sequence.name}: no pixel size, so no physical rule")
    um = sequence.pixel_size_um
    grow_px = max(1, int(round(rule.position_tolerance_um / um)))
    structure = np.ones((3, 3), bool)

    regions = [_regions(m) for m in masks]
    working = [m.copy() for m in masks]
    claims: list[Undrawn] = []
    checks: list[FrameCheck] = []

    for k in range(sequence.n_frames):
        check = bracket_of(sequence, k, rule)
        check.labelled = len(regions[k])
        checks.append(check)
        if not check.checkable:
            continue
        b = check.bracket
        assert b is not None
        pairs = list(_linked_pairs(sequence, regions, b, rule))
        reason = _no_evidence(sequence, regions, b) or (
            "" if pairs else "no cell drawn on one side links to one on the other: nothing to test")
        if reason:
            check.checkable, check.reason = False, reason
            continue
        check.linked_pairs = len(pairs)
        for r0_own, r1_own, c0, c1, link in pairs:
            displacement_px = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            at_ref = (c0[0] + (c1[0] - c0[0]) * b.fraction,
                      c0[1] + (c1[1] - c0[1]) * b.fraction)
            at = sequence.from_reference(k, at_ref)

            # The outline is a human mask, from the nearer labelled time.
            source = r0_own if b.fraction <= 0.5 else r1_own
            sx, sy = _centre(source)
            outline = _shift(source, at[1] - sy, at[0] - sx)
            if outline.sum() < rule.min_added_px:
                continue
            grown = binary_dilation(outline, structure=structure, iterations=grow_px)
            if np.any(working[k][grown]):
                continue  # something drawn here accounts for it
            region = outline & (working[k] == 0)
            if region.sum() < rule.min_added_px:
                continue
            working[k][region] = int(working[k].max()) + 1
            claims.append(Undrawn(
                image=sequence.paths[k].name,
                sequence=sequence.name,
                target=k,
                at=(round(at[0], 1), round(at[1], 1)),
                t_index=sequence.t_indices[k],
                t0_index=sequence.t_indices[b.before],
                t1_index=sequence.t_indices[b.after],
                span_min=round(b.span_min, 1),
                fraction=round(b.fraction, 3),
                displacement_um=round(displacement_px * um, 2),
                link_iou=round(link, 3),
                region=region,
            ))
    return claims, checks


def _no_evidence(sequence: Sequence, regions: list, b: Bracket) -> str:
    """Why a short enough bracket still cannot show a missed cell, or ``""``.

    With nothing drawn on one side there is no cell that could have been
    missed in between: an unannotated frame is not evidence that a cell left.
    """
    for side, which in ((b.before, "earlier"), (b.after, "later")):
        if not regions[side]:
            return (f"nothing drawn at t{sequence.t_indices[side]} "
                    f"({sequence.paths[side].name}, the {which} side of the bracket)")
    return ""


def _linked_pairs(sequence: Sequence, regions: list, b: Bracket, rule: BracketRule):
    """Unique mutual-overlap t0/t1 pairs of a bracket, with the speed gate."""
    um = sequence.pixel_size_um
    max_step_px = (rule.position_tolerance_um + rule.max_speed_um_per_min * b.span_min) / um
    before_in_after = [_into(sequence, reg, b.before, b.after) for _, reg in regions[b.before]]
    after_regs = [reg for _, reg in regions[b.after]]
    if not before_in_after or not after_regs:
        return
    overlap = np.array([[_iou(r0, r1) for r1 in after_regs] for r0 in before_in_after])
    for i in range(len(before_in_after)):
        partners = np.flatnonzero(overlap[i] >= rule.link_iou)
        if len(partners) != 1:
            continue
        j = int(partners[0])
        if int(np.count_nonzero(overlap[:, j] >= rule.link_iou)) != 1:
            continue
        r0, r1 = regions[b.before][i][1], regions[b.after][j][1]
        c0 = sequence.to_reference(b.before, _centre(r0))
        c1 = sequence.to_reference(b.after, _centre(r1))
        if float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) > max_step_px:
            continue  # faster than any cell measured here: not one cell
        yield r0, r1, c0, c1, float(overlap[i, j])


def diagnose(
    sequence: Sequence,
    masks: list[np.ndarray],
    k: int,
    at: tuple[float, float],
    rule: BracketRule = BracketRule(),
) -> str:
    """Why the rule makes no claim at ``at`` (x, y in member k's pixels).

    For explaining, one by one, why an addition of the withdrawn rule does not
    survive: it names the first condition that fails for the t0 cell nearest to
    that position.
    """
    from scipy.ndimage import binary_dilation

    check = bracket_of(sequence, k, rule)
    if not check.checkable:
        return f"not checkable: {check.reason}"
    b = check.bracket
    assert b is not None
    regions = [_regions(m) for m in masks]
    reason = _no_evidence(sequence, regions, b)
    if reason:
        return f"not checkable: {reason}"
    at_ref = sequence.to_reference(k, at)
    nearest = min(regions[b.before], key=lambda lr: np.hypot(
        *(np.subtract(sequence.to_reference(b.before, _centre(lr[1])), at_ref))))
    for r0, r1, c0, c1, _ in _linked_pairs(sequence, regions, b, rule):
        if r0 is nearest[1]:
            p = sequence.from_reference(k, (c0[0] + (c1[0] - c0[0]) * b.fraction,
                                            c0[1] + (c1[1] - c0[1]) * b.fraction))
            source = r0 if b.fraction <= 0.5 else r1
            sx, sy = _centre(source)
            outline = _shift(source, p[1] - sy, p[0] - sx)
            if outline.sum() < rule.min_added_px:
                return "the interpolated outline is mostly outside the frame"
            grow = max(1, int(round(rule.position_tolerance_um / sequence.pixel_size_um)))
            grown = binary_dilation(outline, structure=np.ones((3, 3), bool), iterations=grow)
            if np.any(masks[k][grown]):
                return "a drawn cell touches the time-interpolated outline"
            return "claimed"
    r0_in_after = _into(sequence, nearest[1], b.before, b.after)
    partners = sum(_iou(r0_in_after, reg) >= rule.link_iou for _, reg in regions[b.after])
    if partners == 0:
        return (f"the t{sequence.t_indices[b.before]} cell overlaps nothing at "
                f"t{sequence.t_indices[b.after]}: no evidence it stayed")
    if partners > 1:
        return "ambiguous: overlaps more than one cell on the far side"
    return "its far-side partner overlaps another cell too, or moved faster than the speed bound"


def leave_one_out(
    sequence: Sequence, masks: list[np.ndarray], rule: BracketRule = BracketRule()
) -> list[dict]:
    """The calibration behind :data:`MAX_BRACKET_MIN`, rerunnable on any data.

    For every bracketed frame -- whatever its span -- and every overlap-linked
    t0/t1 pair, is a drawn mask at t where the time-interpolated outline says
    the cell is (IoU >= ``link_iou``)? Where the rule's identity and
    interpolation hold this is true for nearly every pair; where it falls well
    below that, an empty spot is no longer evidence.

    Each row carries ``target_drawn``, the number of cells drawn at t. A pair
    into a frame with nothing drawn always "fails", but that is the missing
    label the rule exists to find, not an interpolation error, so callers must
    leave those rows out of the calibration (and report them separately).
    """
    regions = [_regions(m) for m in masks]
    rows = []
    for k in range(sequence.n_frames):
        check = bracket_of(sequence, k, BracketRule(
            max_speed_um_per_min=rule.max_speed_um_per_min,
            position_tolerance_um=rule.position_tolerance_um,
            max_bracket_min=float("inf"), link_iou=rule.link_iou,
            min_added_px=rule.min_added_px))
        if check.bracket is None:
            continue
        b = check.bracket
        for r0, r1, c0, c1, _ in _linked_pairs(sequence, regions, b, rule):
            at = sequence.from_reference(k, (c0[0] + (c1[0] - c0[0]) * b.fraction,
                                             c0[1] + (c1[1] - c0[1]) * b.fraction))
            source = r0 if b.fraction <= 0.5 else r1
            sx, sy = _centre(source)
            outline = _shift(source, at[1] - sy, at[0] - sx)
            best = max((_iou(outline, reg) for _, reg in regions[k]), default=0.0)
            rows.append({"image": sequence.paths[k].name, "span_min": round(b.span_min, 1),
                         "target_drawn": len(regions[k]),
                         "drawn_where_predicted": bool(best >= rule.link_iou)})
    return rows


def distance_rule_partners(
    sequence: Sequence, masks: list[np.ndarray], rule: BracketRule = BracketRule()
) -> dict:
    """What identity by distance alone would have given, for the record.

    For every bracketed frame (any span) and every cell drawn at t0: how many
    t1 cells lie within ``position_tolerance_um + max_speed_um_per_min x span``
    of it (registered), and is the pairing unique both ways? This is the
    measurement that rules distance out as the same-cell test (module
    docstring); it is recomputed with every report rather than quoted.
    """
    um = sequence.pixel_size_um
    regions = [_regions(m) for m in masks]
    t0_cells = unique = 0
    for k in range(sequence.n_frames):
        b = bracket_of(sequence, k, BracketRule(max_bracket_min=float("inf"))).bracket
        if b is None or um is None:
            continue
        radius_px = (rule.position_tolerance_um + rule.max_speed_um_per_min * b.span_min) / um
        p0 = [sequence.to_reference(b.before, _centre(r)) for _, r in regions[b.before]]
        p1 = [sequence.to_reference(b.after, _centre(r)) for _, r in regions[b.after]]
        near = np.array([[np.hypot(a[0] - c[0], a[1] - c[1]) <= radius_px for c in p1]
                         for a in p0], bool).reshape(len(p0), len(p1))
        t0_cells += len(p0)
        for i in range(len(p0)):
            partners = np.flatnonzero(near[i])
            if len(partners) == 1 and int(near[:, partners[0]].sum()) == 1:
                unique += 1
    return {"bracketed_t0_cells": t0_cells, "with_a_unique_partner": unique}


def ceiling_interval(labelled: int, undrawn: int, confidence: float = 0.95) -> dict | None:
    """Exact (Clopper-Pearson) interval on the missing rate, and the F1 it implies.

    The missing rate is ``undrawn / (labelled + undrawn)``; F1 = 2(1 - m)/(2 - m)
    falls as it rises, so the F1 interval is the missing-rate interval mapped
    and reversed. With a handful of events this interval, not the point
    estimate, is the result.
    """
    from scipy.stats import binomtest

    n = labelled + undrawn
    if n == 0:
        return None
    ci = binomtest(undrawn, n).proportion_ci(confidence_level=confidence, method="exact")

    def f1(m: float) -> float:
        return 2.0 * (1.0 - m) / (2.0 - m)

    return {
        "confidence": confidence,
        "method": "Clopper-Pearson exact, scipy.stats.binomtest",
        "missing_rate": [round(float(ci.low), 4), round(float(ci.high), 4)],
        "f1": [round(f1(float(ci.high)), 4), round(f1(float(ci.low)), 4)],
    }


def ceiling(labelled: int, undrawn: int) -> dict:
    """The F1 a perfect detector reaches against a reference missing cells.

    A perfect detector finds every cell, so each undrawn one is scored as a
    false positive: tp = labelled, fp = undrawn, recall 1, P = tp / (tp + fp),
    F1 = 2P / (P + 1).
    """
    if labelled + undrawn == 0:
        return {"tp": labelled, "fp": undrawn, "precision": None, "f1": None}
    precision = labelled / (labelled + undrawn)
    return {
        "tp": labelled,
        "fp": undrawn,
        "precision": round(precision, 4),
        "f1": round(2 * precision / (precision + 1.0), 4),
        "missing_rate": round(undrawn / (labelled + undrawn), 4),
    }
