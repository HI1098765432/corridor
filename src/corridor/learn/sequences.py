"""Recovering the time-lapse sequences hidden inside the labelled still images.

The supplied training folder looks like a pile of unrelated pictures:
``041824_1.tif`` through ``041824_22.tif`` and so on, each with its own
``_seg.npy``. They are not unrelated, but they are not contiguous frames either,
which is what this module used to assume.

**What each still actually is.** Every still was exported from ImageJ and its
``Labels`` field records the source movie and time point, for example
``t:44/80 - <name>.nd2 (series 01)``; ``finterval`` gives the frame interval and
``XResolution`` the pixel size. Read that way the stills are sparse, irregular
samples of a few movies:

- filename order is **not** time order for two groups (KK1 ``041824_1-12``:
  t44, 67, 23, 1, 45, ...; KK2 ``052924``, ``_1-14``: t14, 15, 29, 6, ...);
- the steps between time-adjacent stills range from one frame to sixty, and the
  KK2 series-02 group jumps from t29 to t48 (6.3 h at 20 min per frame);
- one experiment can hold **more than one crop of the same size**. KK2 series
  01 is two 306x621 crops of different parts of the field (``052924_6`` and
  ``052924_14`` are both t1 with different pixels), which is why the old
  shape grouping saw a frame correlation of 0.10 there and called it "several
  fields";
- frames of one crop are **not pixel-registered**: they sit up to 38 px from
  their crop's first frame on KK2 20240529-s02 and up to 19 px on KK1
  20240418-s01 (whose t1 frame is the outlier) -- the same order as the
  distance a cell moves between two stills.

So sequences are now built from the metadata: group by experiment
(``<yyyymmdd>-s<series>``) and shape, split each such set into crops by image
registration, order each crop by its true time index, and carry the registration
offsets and the elapsed time between members so a temporal rule can work in one
coordinate frame and in minutes rather than in "neighbouring filenames".

The previous behaviour (group by shape, order by the trailing filename number)
is kept, unchanged, behind ``order="filename"``, because the published label
ceilings and corrected reference were derived with it and have to stay
reproducible to be withdrawn honestly (``docs/RESEARCH_V2.md``). It is also
what a caller that names no order still gets, with a ``FutureWarning``: the
scripts not yet ported to the time-aware rule compare raw pixel positions and
would turn true-time neighbours hours apart into new, meaningless published
numbers (:func:`find_sequences`).

Nothing downstream should trust a grouping this module could not justify, so
every sequence carries the evidence for its own existence: the registration
score that put each frame in the crop, the median correlation between
time-consecutive registered frames and the median distance a cell centroid
moves between them.

Privacy: the source name inside ``Labels`` carries experimental condition text.
It is parsed for the date and series and then dropped; no field of
:class:`StillMeta` holds it.
"""

from __future__ import annotations

import datetime as _dt
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

#: Two stills of one experiment and shape are the same crop when their
#: wall-suppressed structure images correlate at least this well at the best
#: shift within :data:`MAX_REGISTRATION_SHIFT_PX`. It is a single-linkage
#: threshold, so a frame needs one good link into its crop. Measured on the
#: supplied stills with this implementation: the weakest link any crop needs is
#: 0.237 (20230615-s04 t67, 29 frames after its nearest labelled neighbour), and
#: no pair across the two 306x621 crops of KK2 20240529-s01 scores above 0.153.
#: 0.20 sits in that gap; the gap is narrow, which is why every link score is
#: carried on the sequence (``link_ncc``) and into the reports.
SAME_CROP_NCC = 0.20
#: Largest offset searched for along each axis, in full-resolution px. The
#: largest measured component is 28 px (KK2 20240529-s02, t17 against t1).
#: Staying well under the channel pitch (about 82 px on KK2) also keeps the
#: search from locking onto the periodic walls of a neighbouring lane.
MAX_REGISTRATION_SHIFT_PX = 40

_TRAILING_NUMBER = re.compile(r"_(\d+)$")
#: ``t:44/80 - <source name>.nd2 (series 01)``. The source name is matched only
#: to find where it ends; it is never stored.
_LABEL = re.compile(r"^t:(\d+)/(\d+)\s+-\s+(.*?)\s*\(series\s+(\d+)\)\s*$")
_LEADING_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})(?!\d)")
_FILENAME_MMDDYY = re.compile(r"^(\d{2})(\d{2})(\d{2})(?:_|$)")
_JULIAN_UNIX_EPOCH = 2440587.5


def _index_of(path: Path) -> int:
    """The trailing number in a filename, or -1 when there is not one.

    ``052924.tif`` has no index and sorts first; it is the odd one out in the
    supplied data and is kept rather than dropped, because a frame with no
    number is still a frame. It orders the legacy grouping, and only breaks
    ties between stills of one time point in the time ordering.
    """
    match = _TRAILING_NUMBER.search(path.stem)
    return int(match.group(1)) if match else -1


def load_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image


def seg_path(path: Path) -> Path:
    return path.with_name(path.name.replace(".tif", "_seg.npy"))


def load_masks(path: Path) -> np.ndarray:
    return np.asarray(np.load(seg_path(path), allow_pickle=True).item()["masks"]).astype(np.int32)


def has_labels(path: Path) -> bool:
    return seg_path(path).exists()


# --------------------------------------------------------------------------
# ImageJ metadata


@dataclass(frozen=True)
class StillMeta:
    """What a still's own metadata says it is. No condition text is kept."""

    path: Path
    group: str
    #: ``<yyyymmdd>-s<series>``: one stage position of one acquisition.
    experiment_id: str
    #: ``<yyyymmdd>``: all series of one .nd2 file share a dish, a day and a
    #: condition, so splits keep them together. Two files acquired on the same
    #: day would share this id too, which errs towards keeping them together.
    acquisition_id: str
    #: Where the date came from: ``label`` (leading yyyymmdd of the source
    #: name), ``filename`` (mmddyy prefix of the still's own name) or
    #: ``nd2_start_utc`` (Nikon ``dTimeAbsolute``, a Julian day in UTC).
    date_source: str
    series: int
    #: 1-based time index in the source movie, and its length.
    t_index: int
    n_t: int
    #: ImageJ ``finterval``: Nikon's measured mean period (``dAvgPeriodDiff``),
    #: not the nominal one -- 915.4 s for a nominal 900 s on 20240418.
    frame_interval_s: float | None
    pixel_size_um: float | None
    shape: tuple[int, int]
    #: ``BitsPerPixel`` from the Bio-Formats info block, when present.
    bits_per_pixel: int | None
    #: Acquisition start from ``dTimeAbsolute``, ISO 8601 UTC.
    acquired_utc: str | None

    @property
    def frame_interval_min(self) -> float | None:
        return None if self.frame_interval_s is None else self.frame_interval_s / 60.0

    @property
    def time_min(self) -> float | None:
        """Minutes since the source movie's first frame (t1 is 0)."""
        if self.frame_interval_min is None:
            return None
        return (self.t_index - 1) * self.frame_interval_min


def _first_label(ij: dict) -> str | None:
    labels = ij.get("Labels")
    if isinstance(labels, (list, tuple)):
        return labels[0] if labels else None
    return labels


def _info_value(info: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", info)
    return match.group(1) if match else None


def _julian_to_utc(jd: float) -> _dt.datetime:
    seconds = (jd - _JULIAN_UNIX_EPOCH) * 86400.0
    return _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=seconds)


def _valid_date(y: int, m: int, d: int) -> str | None:
    try:
        return _dt.date(y, m, d).strftime("%Y%m%d")
    except ValueError:
        return None


def read_still_meta(path: Path, group: str | None = None) -> StillMeta | None:
    """Parse a still's ImageJ metadata. None when there is no ``t:N/M`` label.

    The date is taken, in order of preference, from a leading ``yyyymmdd`` in
    the source name, from an ``mmddyy_`` prefix of the file's own name, and from
    the acquisition start. On the supplied data the three agree wherever more
    than one exists, except that ``dTimeAbsolute`` is UTC and the 20230615
    acquisition started at 23:40 local time, i.e. on 0616 UTC -- which is why it
    is the last resort.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        ij = tf.imagej_metadata or {}
        page = tf.pages[0]
        shape = tuple(int(v) for v in page.shape[:2])
        xres = page.tags.get("XResolution")
        xres = xres.value if xres is not None else None
    label = _first_label(ij)
    match = _LABEL.match(label.strip()) if isinstance(label, str) else None
    if not match:
        return None
    t_index, n_t, source, series = (int(match.group(1)), int(match.group(2)),
                                    match.group(3), int(match.group(4)))
    info = str(ij.get("Info", ""))

    acquired = None
    jd = _info_value(info, "dTimeAbsolute")
    if jd:
        try:
            acquired = _julian_to_utc(float(jd))
        except ValueError:
            acquired = None

    date, source_of_date = None, ""
    lead = _LEADING_DATE.match(source)
    if lead:
        date = _valid_date(int(lead.group(1)), int(lead.group(2)), int(lead.group(3)))
        source_of_date = "label"
    if date is None:
        prefix = _FILENAME_MMDDYY.match(path.stem)
        if prefix:
            mm, dd, yy = (int(v) for v in prefix.groups())
            date = _valid_date(2000 + yy, mm, dd)
            source_of_date = "filename"
    if date is None and acquired is not None:
        date = acquired.strftime("%Y%m%d")
        source_of_date = "nd2_start_utc"
    if date is None:
        date, source_of_date = "unknown", "none"
    del source  # condition text: parsed for its date, never kept

    pixel_size = None
    if xres and isinstance(xres, tuple) and xres[0]:
        unit = str(ij.get("unit", "")).lower()
        if unit in ("micron", "um", "µm", "\\u00b5m"):
            pixel_size = float(xres[1]) / float(xres[0])
    interval = ij.get("finterval")
    bits = _info_value(info, "BitsPerPixel")

    return StillMeta(
        path=path,
        group=group or path.parent.name,
        experiment_id=f"{date}-s{series:02d}",
        acquisition_id=date,
        date_source=source_of_date,
        series=series,
        t_index=t_index,
        n_t=n_t,
        frame_interval_s=float(interval) if interval else None,
        pixel_size_um=pixel_size,
        shape=shape,  # type: ignore[arg-type]
        bits_per_pixel=int(bits) if bits and bits.isdigit() else None,
        acquired_utc=acquired.isoformat(timespec="seconds") if acquired else None,
    )


# --------------------------------------------------------------------------
# Crop identity and registration


def structure_image(image: np.ndarray) -> np.ndarray:
    """The static landmarks of a frame, with straight walls suppressed.

    Half resolution, high-passed (minus a Gaussian of sigma 8 px), then minus a
    41-px running median along each image axis. The medians remove any line
    running straight along either axis -- the channel walls -- symmetrically,
    so no migration direction is assumed. What remains is debris, bubbles,
    channel ends and cells. Without this step the periodic walls make two crops
    one channel pitch apart look identical; with it, the two KK2 s01 crops
    separate (see :data:`SAME_CROP_NCC`).
    """
    from scipy.ndimage import gaussian_filter, median_filter

    small = np.asarray(image, dtype=np.float32)[::2, ::2]
    high = small - gaussian_filter(small, 4.0)
    high = high - median_filter(high, size=(21, 1))
    high = high - median_filter(high, size=(1, 21))
    high = gaussian_filter(high, 0.5)
    high -= high.mean()
    return high


def register(
    a: np.ndarray, b: np.ndarray, *, max_shift_px: float = MAX_REGISTRATION_SHIFT_PX
) -> tuple[float, tuple[float, float]]:
    """Best overlap-normalised correlation of two structure images, and its shift.

    Returns ``(ncc, (dy, dx))`` in full-resolution px, where a point at ``p`` in
    frame *b* sits at ``p + (dy, dx)`` in frame *a*. The correlation at each
    shift is normalised by the energy of the overlapping parts only, so a large
    shift is not favoured just because less background overlaps.
    """
    from scipy.signal import fftconvolve

    h, w = a.shape
    ones = np.ones_like(a)
    num = fftconvolve(a, b[::-1, ::-1], mode="full")
    ea = fftconvolve(a * a, ones[::-1, ::-1], mode="full")
    eb = fftconvolve(ones, (b * b)[::-1, ::-1], mode="full")
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = num / np.sqrt(np.clip(ea, 1e-12, None) * np.clip(eb, 1e-12, None))
    radius = int(max_shift_px // 2)
    cy, cx = h - 1, w - 1
    window = ncc[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
    window = np.nan_to_num(window, nan=-1.0)
    iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)
    return float(window[iy, ix]), (float(2 * (iy - radius)), float(2 * (ix - radius)))


@dataclass
class Crop:
    """Frames of one experiment that show the same field, and how they align."""

    members: list[int]
    #: ``offsets[k]`` added to a position in frame ``members[k]`` gives the
    #: position in the crop's reference frame (its earliest member).
    offsets: list[tuple[float, float]]
    #: The registration score of the link that attached each member (1.0 for
    #: the reference).
    link_ncc: list[float]


def split_into_crops(
    images: list[np.ndarray],
    *,
    same_crop_ncc: float = SAME_CROP_NCC,
    max_shift_px: float = MAX_REGISTRATION_SHIFT_PX,
) -> list[Crop]:
    """Cluster same-sized frames into crops and register each crop.

    Single-linkage on the pairwise registration score, then a maximum spanning
    tree from the first frame (callers pass frames in time order) so every
    frame's offset is chained through its strongest links rather than forced
    against one reference it may correlate poorly with five hours later.
    """
    n = len(images)
    structures = [structure_image(im) for im in images]
    score = np.full((n, n), -1.0)
    shift: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n):
        score[i, i] = 1.0
        for j in range(i + 1, n):
            if structures[i].shape != structures[j].shape:
                continue
            value, (dy, dx) = register(structures[i], structures[j], max_shift_px=max_shift_px)
            score[i, j] = score[j, i] = value
            shift[(i, j)] = (dy, dx)  # j -> i
            shift[(j, i)] = (-dy, -dx)  # i -> j

    unvisited = set(range(n))
    crops: list[Crop] = []
    while unvisited:
        root = min(unvisited)
        unvisited.discard(root)
        offset = {root: (0.0, 0.0)}
        link = {root: 1.0}
        frontier = [root]
        # Prim's algorithm restricted to links above threshold: always attach
        # the outside frame with the strongest link to any frame inside.
        while True:
            best = None
            for i in offset:
                for j in unvisited:
                    if score[i, j] >= same_crop_ncc and (best is None or score[i, j] > best[0]):
                        best = (score[i, j], i, j)
            if best is None:
                break
            value, i, j = best
            dy, dx = shift[(i, j)]
            offset[j] = (offset[i][0] + dy, offset[i][1] + dx)
            link[j] = float(value)
            unvisited.discard(j)
            frontier.append(j)
        members = sorted(offset)
        crops.append(Crop(members=members, offsets=[offset[m] for m in members],
                          link_ncc=[link[m] for m in members]))
    return crops


# --------------------------------------------------------------------------
# Sequences


@dataclass
class Sequence:
    """One field of view, as the run of frames it actually is."""

    group: str
    shape: tuple[int, int]
    paths: list[Path]
    #: Median Pearson correlation between consecutive (registered) frames. High
    #: values are what make "these are frames of one movie" a measurement.
    frame_correlation: float = 0.0
    #: Median distance a matched cell centroid travels between frames, in px.
    centroid_step_px: float = 0.0
    #: Labelled cell count per frame, in order. A zero in the middle of a run is
    #: the signature of a frame nobody annotated.
    cells_per_frame: list[int] = field(default_factory=list)
    #: ``"time"`` (metadata) or ``"filename"`` (the legacy, withdrawn ordering).
    ordering: str = "filename"
    experiment_id: str = ""
    crop: int = 0
    #: True time index of each member, from its ImageJ label. Empty for the
    #: legacy ordering, which never read it.
    t_indices: list[int] = field(default_factory=list)
    frame_interval_min: float | None = None
    pixel_size_um: float | None = None
    #: Added to a position in member k gives its position in the reference
    #: frame. Zeros for the legacy ordering, which never registered anything.
    offsets_px: list[tuple[float, float]] = field(default_factory=list)
    link_ncc: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.ordering == "time":
            return f"{self.group}-{self.experiment_id}-c{self.crop}"
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
        reference rather than a hole in the biology. With the time ordering,
        "between" means between in time; nothing here says how far apart.
        """
        counts = self.cells_per_frame
        out = []
        for i, n in enumerate(counts):
            if n:
                continue
            if any(counts[:i]) and any(counts[i + 1:]):
                out.append(i)
        return out

    def elapsed_min(self, i: int, j: int) -> float:
        """Minutes from member i to member j (negative if j is earlier).

        Uses the source movie's measured mean period. The Nikon period jitter
        recorded in the metadata is under 1 % for 20240418 and 20240529 and
        under 4 % for 20241223, so this is the elapsed time to that accuracy.
        """
        if self.ordering != "time" or self.frame_interval_min is None:
            raise ValueError(f"{self.name}: elapsed time needs the time ordering")
        return (self.t_indices[j] - self.t_indices[i]) * self.frame_interval_min

    def to_reference(self, k: int, xy: tuple[float, float]) -> tuple[float, float]:
        """An (x, y) position in member k, in the crop's reference frame."""
        if not self.offsets_px:
            return xy
        dy, dx = self.offsets_px[k]
        return (xy[0] + dx, xy[1] + dy)

    def from_reference(self, k: int, xy: tuple[float, float]) -> tuple[float, float]:
        """A reference-frame (x, y) position, in member k's own pixels."""
        if not self.offsets_px:
            return xy
        dy, dx = self.offsets_px[k]
        return (xy[0] - dx, xy[1] - dy)

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


def _registered_correlation(a: np.ndarray, b: np.ndarray, dy: float, dx: float) -> float | None:
    """Pearson correlation of a and b over their overlap, b shifted by (dy, dx)."""
    h, w = a.shape
    sy, sx = int(round(dy)), int(round(dx))
    ya0, ya1 = max(0, sy), min(h, h + sy)
    xa0, xa1 = max(0, sx), min(w, w + sx)
    if ya1 - ya0 < 8 or xa1 - xa0 < 8:
        return None
    pa = a[ya0:ya1, xa0:xa1].ravel()
    pb = b[ya0 - sy:ya1 - sy, xa0 - sx:xa1 - sx].ravel()
    if pa.std() < 1e-6 or pb.std() < 1e-6:
        return None
    return float(np.corrcoef(pa, pb)[0, 1])


def describe(sequence: Sequence) -> Sequence:
    """Fill in the evidence that this group really is a run of frames."""
    from scipy.optimize import linear_sum_assignment

    images = [load_image(p).astype(np.float32) for p in sequence.paths]
    masks = [load_masks(p) for p in sequence.paths]
    sequence.cells_per_frame = [len([l for l in np.unique(m) if l]) for m in masks]

    registered = sequence.ordering == "time" and bool(sequence.offsets_px)
    correlations = []
    for k, (a, b) in enumerate(zip(images[:-1], images[1:])):
        if a.shape != b.shape:
            continue
        if registered:
            # Shift that maps member k+1 onto member k.
            dy = sequence.offsets_px[k + 1][0] - sequence.offsets_px[k][0]
            dx = sequence.offsets_px[k + 1][1] - sequence.offsets_px[k][1]
            value = _registered_correlation(a, b, dy, dx)
            if value is not None:
                correlations.append(value)
            continue
        flat_a, flat_b = a.ravel(), b.ravel()
        if flat_a.std() < 1e-6 or flat_b.std() < 1e-6:
            continue
        correlations.append(float(np.corrcoef(flat_a, flat_b)[0, 1]))
    sequence.frame_correlation = float(np.median(correlations)) if correlations else 0.0

    steps = []
    for k, (a, b) in enumerate(zip(masks[:-1], masks[1:])):
        pa = [sequence.to_reference(k, p) for p in _centroids(a)]
        pb = [sequence.to_reference(k + 1, p) for p in _centroids(b)]
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


def _find_by_filename(group_dir: Path, labelled: list[Path]) -> list[Sequence]:
    """The withdrawn rule: group by shape, order by the trailing filename number."""
    out = []
    by_shape: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in labelled:
        shape = load_image(path).shape[:2]
        by_shape[(int(shape[0]), int(shape[1]))].append(path)
    for shape, paths in by_shape.items():
        if len(paths) < MIN_SEQUENCE_LENGTH:
            continue
        ordered = sorted(paths, key=_index_of)
        out.append(Sequence(group=group_dir.name, shape=shape, paths=ordered))
    return out


def _find_by_time(group_dir: Path, labelled: list[Path]) -> list[Sequence]:
    out = []
    by_experiment: dict[tuple[str, tuple[int, int]], list[StillMeta]] = defaultdict(list)
    for path in labelled:
        meta = read_still_meta(path, group_dir.name)
        if meta is None:
            # No time label: there is no defensible place for it in any
            # sequence, so it is left out rather than guessed into one.
            continue
        by_experiment[(meta.experiment_id, meta.shape)].append(meta)

    for (experiment, shape), metas in sorted(by_experiment.items()):
        if len(metas) < MIN_SEQUENCE_LENGTH:
            continue
        metas.sort(key=lambda m: (m.t_index, _index_of(m.path), m.path.name))
        images = [load_image(m.path) for m in metas]
        crops = split_into_crops(images)
        for number, crop in enumerate(crops):
            if len(crop.members) < MIN_SEQUENCE_LENGTH:
                continue
            members = [metas[i] for i in crop.members]
            intervals = {round(m.frame_interval_s or 0.0, 3) for m in members}
            pixel_sizes = {m.pixel_size_um for m in members}
            out.append(Sequence(
                group=group_dir.name,
                shape=shape,
                paths=[m.path for m in members],
                ordering="time",
                experiment_id=experiment,
                crop=number,
                t_indices=[m.t_index for m in members],
                frame_interval_min=members[0].frame_interval_min if len(intervals) == 1 else None,
                pixel_size_um=members[0].pixel_size_um if len(pixel_sizes) == 1 else None,
                offsets_px=list(crop.offsets),
                link_ncc=list(crop.link_ncc),
            ))
    return out


def find_sequences(
    train_root: Path,
    *,
    groups: tuple[str, ...] = CANONICAL_GROUPS,
    describe_each: bool = True,
    order: str | None = None,
) -> list[Sequence]:
    """Group the labelled stills back into the runs of frames they came from.

    ``order="time"`` uses the ImageJ metadata: experiment, crop and true time
    index; every caller ported to :mod:`corridor.learn.brackets` asks for it.
    ``order="filename"`` reproduces the withdrawn grouping bit for bit.

    **No order given means ``"filename"``, with a warning.** Four scripts
    (``verify_suspect_labels``, ``diag_label_gaps``, ``recover_missing_labels``,
    ``experiment_reconstruction``) were written for filename-adjacent frames:
    they compare raw pixel positions against fixed px thresholds, with no
    registration and no time limit, and by default write over published
    ``docs/*.json``. Handed true-time neighbours up to 12 h apart they would
    publish numbers from neither the withdrawn rule nor the v2 one. Until each
    is ported, a default rerun reproduces what it published.
    """
    if order is None:
        import warnings

        warnings.warn(
            "find_sequences() without order= gives the withdrawn filename ordering "
            "(docs/RESEARCH_V2.md); pass order='time' for true time order",
            FutureWarning, stacklevel=2)
        order = "filename"
    if order not in ("time", "filename"):
        raise ValueError(f"order must be 'time' or 'filename', not {order!r}")
    sequences: list[Sequence] = []
    for name in groups:
        group_dir = train_root / name
        if not group_dir.is_dir():
            continue
        labelled = [p for p in sorted(group_dir.glob("*.tif")) if has_labels(p)]
        if not labelled:
            continue
        if order == "filename":
            sequences += _find_by_filename(group_dir, labelled)
        else:
            sequences += _find_by_time(group_dir, labelled)

    if describe_each:
        sequences = [describe(s) for s in sequences]
    return sorted(sequences, key=lambda s: -s.n_frames)
