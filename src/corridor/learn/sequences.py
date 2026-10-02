"""Recovering the time-lapse sequences hidden inside the labelled still images.

The supplied training folder looks like a pile of unrelated pictures:
``041824_1.tif`` through ``041824_40.tif`` and so on, each with its own
``_seg.npy``. They are not unrelated. Grouping them by image dimensions and
ordering them by their trailing number recovers **six contiguous runs of frames
from the same field of view**, which is what the numbering meant all along:

    KK1  (264, 727)  frames 1-12
    KK1  (290, 762)  frames 1-12
    KK1  (309, 649)  frames 13-22
    KK1  (381, 864)  frames 1-5
    KK2  (303, 573)  frames 15-30
    KK2  (306, 621)  frames 1-14

Shape is a sound grouping key here because each field was cropped once and every
frame of it inherits that crop; two different fields agreeing on both dimensions
to the pixel would be a coincidence, and the frame-to-frame image correlation
reported by :func:`describe` is what confirms it rather than assumes it.

Nothing downstream should trust a grouping this module could not justify, so
every sequence carries the evidence for its own existence: the median
correlation between consecutive frames and the median distance a cell centroid
moves between them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

#: Consecutive frames of one field look alike. Below this correlation a group is
#: reported as questionable rather than silently treated as a movie.
MIN_FRAME_CORRELATION = 0.25
#: A run shorter than this cannot support a temporal argument: with two frames
#: there is no interior gap and nothing to interpolate between.
MIN_SEQUENCE_LENGTH = 3

_TRAILING_NUMBER = re.compile(r"_(\d+)$")


def _index_of(path: Path) -> int:
    """The trailing number in a filename, or -1 when there is not one.

    ``052924.tif`` has no index and sorts first; it is the odd one out in the
    supplied data and is kept rather than dropped, because a frame with no
    number is still a frame.
    """
    match = _TRAILING_NUMBER.search(path.stem)
    return int(match.group(1)) if match else -1


def load_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image


def load_masks(path: Path) -> np.ndarray:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


def has_labels(path: Path) -> bool:
    return path.with_name(path.name.replace(".tif", "_seg.npy")).exists()


@dataclass
class Sequence:
    """One field of view, as the run of frames it actually is."""

    group: str
    shape: tuple[int, int]
    paths: list[Path]
    #: Median Pearson correlation between consecutive frames. High values are
    #: what make "these are frames of one movie" a measurement.
    frame_correlation: float = 0.0
    #: Median distance a matched cell centroid travels between frames, in px.
    centroid_step_px: float = 0.0
    #: Labelled cell count per frame, in order. A zero in the middle of a run is
    #: the signature of a frame nobody annotated.
    cells_per_frame: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.group}-{self.shape[0]}x{self.shape[1]}"

    @property
    def n_frames(self) -> int:
        return len(self.paths)

    @property
    def looks_like_a_movie(self) -> bool:
        return (
            self.n_frames >= MIN_SEQUENCE_LENGTH
            and self.frame_correlation >= MIN_FRAME_CORRELATION
        )

    @property
    def unlabelled_interior_frames(self) -> list[int]:
        """Positions with no labelled cell that sit between frames that have one.

        These are the frames where the annotator stopped, not where the cells
        did. A frame at the start or end of a run with no cells may genuinely be
        empty; one in the middle, with cells on both sides, is a hole in the
        reference rather than a hole in the biology.
        """
        counts = self.cells_per_frame
        out = []
        for i, n in enumerate(counts):
            if n:
                continue
            if any(counts[:i]) and any(counts[i + 1:]):
                out.append(i)
        return out

    def images(self) -> Iterator[np.ndarray]:
        for path in self.paths:
            yield load_image(path)

    def masks(self) -> Iterator[np.ndarray]:
        for path in self.paths:
            yield load_masks(path)


def _centroids(mask: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for label in np.unique(mask):
        if label == 0:
            continue
        ys, xs = np.where(mask == label)
        out.append((float(xs.mean()), float(ys.mean())))
    return out


def describe(sequence: Sequence) -> Sequence:
    """Fill in the evidence that this group really is a run of frames."""
    from scipy.optimize import linear_sum_assignment

    images = [load_image(p).astype(np.float32) for p in sequence.paths]
    masks = [load_masks(p) for p in sequence.paths]
    sequence.cells_per_frame = [len([l for l in np.unique(m) if l]) for m in masks]

    correlations = []
    for a, b in zip(images[:-1], images[1:]):
        if a.shape != b.shape:
            continue
        flat_a, flat_b = a.ravel(), b.ravel()
        if flat_a.std() < 1e-6 or flat_b.std() < 1e-6:
            continue
        correlations.append(float(np.corrcoef(flat_a, flat_b)[0, 1]))
    sequence.frame_correlation = float(np.median(correlations)) if correlations else 0.0

    steps = []
    for a, b in zip(masks[:-1], masks[1:]):
        pa, pb = _centroids(a), _centroids(b)
        if not pa or not pb:
            continue
        cost = np.array([[np.hypot(p[0] - q[0], p[1] - q[1]) for q in pb] for p in pa])
        rows, cols = linear_sum_assignment(cost)
        # Anything further than a few cell lengths is a different cell, not a
        # step, and would turn this diagnostic into a measure of cell density.
        steps += [float(cost[r, c]) for r, c in zip(rows, cols) if cost[r, c] < 60.0]
    sequence.centroid_step_px = float(np.median(steps)) if steps else 0.0
    return sequence


#: The folders holding the original images. The supplied tree also contains
#: KK1_KK2_combi/, which is KK1 and KK2 copied into one folder as the combined
#: model's training set, and three *Model/ folders holding checkpoints. Walking
#: those would return every sequence twice over -- which is not cosmetic: it
#: would double-weight a field during training and make a "held-out" split
#: silently contain its own training images. The canonical groups are named
#: rather than guessed at, because a heuristic that is wrong here fails silently.
CANONICAL_GROUPS = ("KK1", "KK2")


def find_sequences(
    train_root: Path,
    *,
    groups: tuple[str, ...] = CANONICAL_GROUPS,
    describe_each: bool = True,
) -> list[Sequence]:
    """Group the labelled stills back into the runs of frames they came from."""
    sequences: list[Sequence] = []
    for name in groups:
        group_dir = train_root / name
        if not group_dir.is_dir():
            continue
        labelled = [p for p in sorted(group_dir.glob("*.tif")) if has_labels(p)]
        if not labelled:
            continue
        by_shape: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for path in labelled:
            shape = load_image(path).shape[:2]
            by_shape[(int(shape[0]), int(shape[1]))].append(path)
        for shape, paths in by_shape.items():
            if len(paths) < MIN_SEQUENCE_LENGTH:
                continue
            ordered = sorted(paths, key=_index_of)
            sequences.append(Sequence(group=group_dir.name, shape=shape, paths=ordered))

    if describe_each:
        sequences = [describe(s) for s in sequences]
    return sorted(sequences, key=lambda s: -s.n_frames)
