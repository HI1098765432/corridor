"""Per-observation samples for "does shape now predict migration next?".

One sample = one tracked observation at frame t whose own mask is in
``masks.npz``. Its predictors are the feature sets of :mod:`.features`; its
targets are what the same track does over the next h frames (h = 1, 3, 6):

- ``disp_um_h{h}``: straight-line displacement |r(t+h) - r(t)| in micrometres.
- ``speed_um_per_hr_h{h}``: mean speed over the horizon, path length along the
  observed positions divided by the *elapsed* time (so a frame gap is a longer
  interval, never a faster cell).
- ``migrating_h{h}``: 1 if the net displacement rate over the horizon is at
  least ``MIGRATING_MIN_NET_RATE_UM_PER_HR``, else 0 (stalled).

A target exists only if the track has an observation exactly h frames later.

Who may be a feature source, and why the others are excluded (each exclusion
is counted, never silent):

- Only ``detection_source == "primary"`` observations. Recovered detections
  (``intensity``, ``windowed``, ``permissive``) have no pixels in
  ``masks.npz`` (16 of 155 observations in the v1.3.0 baseline; their
  ``det_label`` does not exist in that frame's label image), and their
  outline came from a threshold-and-flood-fill, not from the model. Their
  *positions* are still used for targets and history.
- The mask pixel count must agree with the track's ``area_px`` (to 1 px) when
  the column exists, so a mis-joined label can never pass as the cell.
- Masks touching the image border are cut off by the frame: their shape is the
  crop's, not the cell's.

Splits: every sample carries ``acquisition``, ``experiment``, ``field``,
``movie`` and ``track_uid`` (``movie/track_id``). Cross-validation groups are
fields, or acquisitions as soon as there are two *resolved* ones
(:func:`outer_split`), so a track is always wholly inside one group;
:func:`assert_no_track_straddles` enforces it on every split the evaluation
makes. Fragments of one physical cell that the tracker split into two tracks
are inside the same movie, so they stay together too.

``experiment`` is the design contract's id, date + series
(``training.datasets``), and is recorded as such. It is *not* the unit of
independence: the supplied nd2 is ``T(54) x XY(57)``, so a series is one of 57
stage positions in one dish on one day. ``acquisition`` is that unit (see
:func:`load_result_folder`); two series of one acquisition are compared by
pixels like two crops, never assumed independent.

A *field* is a set of movies that show the same pixels. Movies are crops, and
two crops of one acquisition can overlap: in the supplied data ``052924_t1``
is a pixel-identical sub-crop of ``052924_1`` over the same 18 frames, and its
only track is ``052924_1`` track 4 segmented a second time. Splitting by movie
would put that cell in training and in test at once. :func:`find_region_overlaps`
detects such overlaps from the images (identical pixels at the same source
frame), the duplicated tracks of the smaller crop are dropped (counted), and
the overlapping movies form one field. Crops that share no source frame
cannot be compared this way (different instants), and are kept apart; the
experiment reports the evidence for those pairs instead of guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import features as F

HORIZONS_FRAMES = (1, 3, 6)
#: "Migrating" means the net displacement over the horizon, divided by its
#: elapsed time, is at least this. 12 um/hr is a cell moving its own width
#: every 20 minutes: the width is the 4.0 um constant that ``corridor.core.qc``
#: (``STATIONARY_NET_UM``) uses for "has not moved its own width" (cells 9-15
#: px wide at 0.467 um/px), and 20 minutes is the supplied frame interval. Only
#: the number is shared: the QC rule judges a whole track's net displacement
#: and only once it spans ``STATIONARY_MIN_MINUTES`` (30), so it never fires on
#: one 20-minute step. It is a *rate*, so it means the same
#: at every horizon and every frame interval. Fixed before any model was run;
#: the distribution of the outcome had been looked at (median net rate about
#: 19 um/hr), the features and models had not.
MIGRATING_MIN_NET_RATE_UM_PER_HR = 12.0
#: Crops are cut with this margin so the phenotype ring (10 px) fits.
CROP_PAD_PX = 12
#: Two crops show the same pixels where a tile of one matches the other at
#: this normalised cross-correlation. Identical pixels give 1.000 (052924_t1
#: inside 052924_1); different regions of this device photographed at the same
#: instant reach 0.95 (a 108x30 tile of 052924_t1 found in 052924_2), because
#: the channel walls repeat. Hence also why pixels alone cannot say whether two
#: crops taken at *different* instants show the same lane.
SAME_PIXELS_NCC = 0.99
#: An observation in the smaller of two overlapping crops is the same cell as
#: one in the larger crop if, at the same source frame, their centroids
#: (mapped through the measured offset) are this close. The 13 duplicated
#: observations of 052924_t1 agree with 052924_1 to within 3.1 px; two distinct
#: cells 9-15 px wide cannot have centroids this close without overlapping.
DUPLICATE_MAX_PX = 5.0
#: Prefix of a placeholder id: nothing said which acquisition or experiment
#: the movie came from, so it is a name for "unknown", never a group.
UNRESOLVED = "unknown:"
#: Condition of a movie nobody labelled. One shared value, so the per-condition
#: baseline is honestly the population median rather than a per-movie median.
UNSPECIFIED_CONDITION = "unspecified"
REPO_ROOT = Path(__file__).resolve().parents[2]


class LeakageError(AssertionError):
    """A split put observations of one track on both sides."""


# ---------------------------------------------------------------------------
# Reading result folders


@dataclass
class ResultFolder:
    path: Path
    movie: str
    #: The contract's id, date + series (one stage position); see the module docstring.
    experiment: str
    #: The unit of independence (:func:`load_result_folder`); ``UNRESOLVED``-prefixed if unknown.
    acquisition: str
    pixel_size_um: float | None
    frame_interval_min: float | None
    tracks: pd.DataFrame
    image_path: Path | None
    run: dict = field(repr=False, default_factory=dict)
    #: Experimental condition, only if someone said so (run.json or a mapping).
    condition: str | None = None

    def masks(self) -> np.ndarray:
        with np.load(self.path / "masks.npz") as z:
            return np.asarray(z["masks"])


#: ``t:1/54 - <source name>.nd2 (series 01)``; the same pattern as
#: ``corridor.learn.sequences``. The source name is matched only for its date.
_LABEL_SOURCE = re.compile(r"^t:\s*\d+\s*/\s*\d+\s*-\s*(.*?)\s*\(series\s+(\d+)\)\s*$")
_LEADING_DATE = re.compile(r"^(\d{8})(?!\d)")
#: The contract's experiment id, as ``_experiment_from_tiff`` writes it.
_EXPERIMENT_ID = re.compile(r"^(\d{8})-s\d+$")


def _source_from_tiff(path: Path) -> tuple[str | None, int | None]:
    """(date ``yyyymmdd``, series) of the source named in the ImageJ ``Labels`` of frame 1.

    The rest of the source name carries experimental conditions and the
    repository is public, so it is parsed for its date and dropped, never
    returned (``training/README.md``, *Privacy*). (None, None) if the file or
    its labels cannot be read; a source name without a leading date gives no
    date rather than a guessed one.
    """
    try:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            labels = (tf.imagej_metadata or {}).get("Labels")
    except Exception:  # noqa: BLE001 - metadata is optional evidence
        return None, None
    if not labels:
        return None, None
    first = labels[0] if isinstance(labels, (list, tuple)) else labels
    match = _LABEL_SOURCE.match(str(first).strip())
    if not match:
        return None, None
    date = _LEADING_DATE.match(match.group(1).strip())
    return (date.group(1) if date else None), int(match.group(2))


def _experiment_from_tiff(path: Path) -> str | None:
    """The contract's experiment id ``<yyyymmdd>-s<series>`` (the form ``training.datasets`` uses).

    None without a date: an id such as ``unknown-s01`` would be shared by every
    undated source of series 1 and pass for one real experiment.
    """
    date, series = _source_from_tiff(path)
    return f"{date}-s{series:02d}" if date and series is not None else None


def is_resolved(group_id) -> bool:
    """False for a placeholder (``UNRESOLVED`` prefix) or an empty id."""
    return bool(group_id) and not str(group_id).startswith(UNRESOLVED)


def _acquisition_of(run: dict, experiment: str | None, image: Path | None) -> str | None:
    """The unit an experiment-level split may hold out, or None if nothing says.

    In order: ``run.json["acquisition"]``; the date of a contract experiment id
    (``yyyymmdd-sNN``, from run.json or the TIFF); any other id run.json names
    as its experiment (taken at its word, e.g. the synthetic sets); else the
    TIFF source's date.

    The *day*, not the nd2 file: the file name cannot be committed (*Privacy*),
    and a date can be. Two files of one day are merged into one acquisition,
    which can only make the split coarser, never leakier: dishes imaged the
    same day usually share the cells' passage and the medium too. Someone who
    knows two same-day dishes are independent says so with run.json's
    ``acquisition``.
    """
    explicit = run.get("acquisition")
    if explicit:
        return str(explicit)
    if experiment:
        m = _EXPERIMENT_ID.match(str(experiment))
        if m:
            return m.group(1)
        if run.get("experiment"):
            return str(experiment)
    if image is not None:
        date, _ = _source_from_tiff(image)
        return date
    return None


def _rel(path: Path | None) -> str | None:
    """Repository-relative path with forward slashes (committed reports carry no home directory)."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _resolve_image(run: dict, folder: Path) -> Path | None:
    raw = (run.get("input") or {}).get("path")
    if not raw:
        return None
    p = Path(str(raw).replace("\\", "/"))
    for candidate in (p, REPO_ROOT / p, folder / p.name):
        if candidate.is_file():
            return candidate
    return None


def load_result_folder(path: str | Path, conditions: dict[str, str] | None = None) -> ResultFolder:
    """Read one Corridor result folder (schema v1 or v2 column names).

    ``experiment`` and ``acquisition`` that nothing resolves become
    ``unknown:<folder name>`` (``UNRESOLVED``): a name for "unknown", which
    :func:`outer_split` refuses to treat as a group of its own. The condition
    comes from ``conditions`` (keyed by movie, then acquisition, then
    experiment), else ``run.json["condition"]``, else stays None. Nothing
    derives it from an id: a condition that is just the experiment id can never
    appear in the training fold of a held-out experiment, and the per-condition
    baseline would silently be the population median.
    """
    path = Path(path)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    cal = run.get("calibration") or {}
    px = cal.get("pixel_size_um")
    fi = cal.get("frame_interval_min")
    if cal.get("spatially_calibrated") is False:
        px = None
    if cal.get("temporally_calibrated") is False:
        fi = None
    tracks = pd.read_csv(path / "tracks.csv")
    image = _resolve_image(run, path)
    experiment = run.get("experiment")
    if not experiment and image is not None:
        experiment = _experiment_from_tiff(image)
    acquisition = _acquisition_of(run, experiment, image)
    condition = None
    for key in (path.name, acquisition, experiment):
        if conditions and key and key in conditions:
            condition = str(conditions[key])
            break
    if condition is None and run.get("condition"):
        condition = str(run["condition"])
    return ResultFolder(
        path=path, movie=path.name,
        experiment=str(experiment) if experiment else f"{UNRESOLVED}{path.name}",
        acquisition=str(acquisition) if acquisition else f"{UNRESOLVED}{path.name}",
        pixel_size_um=float(px) if px else None,
        frame_interval_min=float(fi) if fi else None,
        tracks=tracks, image_path=image, run=run, condition=condition,
    )


def _positions_um(tr: pd.DataFrame, px: float | None) -> np.ndarray:
    if {"x_um", "y_um"} <= set(tr.columns) and tr["x_um"].notna().all():
        return tr[["x_um", "y_um"]].to_numpy(float)
    if px is None:
        raise ValueError("tracks have no micrometre positions and no pixel size")
    return tr[["x_px", "y_px"]].to_numpy(float) * px


def _times_min(tr: pd.DataFrame, fi: float | None) -> np.ndarray:
    if "elapsed_min" in tr.columns and tr["elapsed_min"].notna().all():
        return tr["elapsed_min"].to_numpy(float)
    if fi is None:
        raise ValueError("tracks have no elapsed time and no frame interval")
    return tr["frame"].to_numpy(float) * fi


# ---------------------------------------------------------------------------
# Crops that show the same pixels


@dataclass
class RegionOverlap:
    """``small`` pixel (r, c) is ``large`` pixel (r + dr, c + dc)."""

    small: str
    large: str
    offset_rc: tuple[int, int]
    ncc: list[float]
    source_frames_checked: list[int]


def _source_frames(lf: ResultFolder, n_frames: int) -> list[int] | None:
    src = (lf.run.get("input") or {}).get("source_frames")
    if src and len(src) == n_frames:
        return [int(v) for v in src]
    return None


def _best_tile_match(small: np.ndarray, large: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best NCC of any of 3x3 tiles of ``small`` slid over ``large``, and its offset.

    Tiles, not the whole frame: two crops can overlap partially, and a crop
    that overhangs the other by even a few rows (052924_t1 starts 6 rows above
    052924_1) cannot be matched whole.
    """
    from skimage.feature import match_template

    h, w = small.shape
    th, tw = h // 3, w // 3
    best, offset = -1.0, (0, 0)
    for i in range(3):
        for j in range(3):
            r0, c0 = i * th, j * tw
            tile = small[r0:r0 + th, c0:c0 + tw]
            if tile.std() == 0 or tile.shape[0] > large.shape[0] or tile.shape[1] > large.shape[1]:
                continue
            r = match_template(large, tile)
            k = int(np.argmax(r))
            if r.flat[k] > best:
                rr, cc = np.unravel_index(k, r.shape)
                best, offset = float(r.flat[k]), (int(rr) - r0, int(cc) - c0)
    return best, offset


def find_region_overlaps(loaded: list[ResultFolder], stacks: dict[str, np.ndarray]
                         ) -> tuple[list[RegionOverlap], list[dict]]:
    """Pixel-identical overlaps between movies, plus the evidence for every pair.

    Only frames with the same *source* frame index are compared (ImageJ frame
    numbers recorded in ``run.json``), so "identical" means the same pixels at
    the same instant. A pair is an overlap if the best tile matches at
    ``SAME_PIXELS_NCC`` or better at one offset in every checked frame. Two
    series (stage positions) of one acquisition are compared the same way:
    their time point k is the same imaging cycle, and positions in one dish
    can overlap.

    Every pair gets an evidence entry. A pair that could not be compared (a
    movie without its image) says so, ``"not checked: no image"``; it is never
    dropped, because a missing row there is exactly how a duplicated cell
    would go unnoticed.
    """
    overlaps: list[RegionOverlap] = []
    evidence: list[dict] = []
    for a_i, a in enumerate(loaded):
        for b in loaded[a_i + 1:]:
            if is_resolved(a.acquisition) and is_resolved(b.acquisition) \
                    and a.acquisition != b.acquisition:
                # Frame 1 of two acquisitions is two instants, so equal source
                # numbers prove nothing here, and two dishes share no cell.
                evidence.append({"small": a.movie, "large": b.movie,
                                 "verdict": "different acquisitions"})
                continue
            if a.movie not in stacks or b.movie not in stacks:
                evidence.append({"small": a.movie, "large": b.movie,
                                 "verdict": "not checked: no image",
                                 "missing_image": sorted(m for m in (a.movie, b.movie)
                                                         if m not in stacks)})
                continue
            sa, sb = stacks[a.movie], stacks[b.movie]
            small, large = (a, b) if sa[0].size <= sb[0].size else (b, a)
            ss, sl = stacks[small.movie], stacks[large.movie]
            src_s, src_l = _source_frames(small, len(ss)), _source_frames(large, len(sl))
            entry: dict = {"small": small.movie, "large": large.movie}
            if src_s is None or src_l is None:
                entry["verdict"] = "not checked: no source frame numbers"
                evidence.append(entry)
                continue
            shared = sorted(set(src_s) & set(src_l))
            entry["shared_source_frames"] = len(shared)
            if not shared:
                # Different instants: report the nearest frames' best match, which
                # cannot separate "same lane later" from "a lane that looks alike".
                pairs = [(abs(x - y), x, y) for x in src_s for y in src_l]
                gap, x, y = min(pairs)
                ncc, off = _best_tile_match(ss[src_s.index(x)].astype(np.float32),
                                            sl[src_l.index(y)].astype(np.float32))
                entry.update({"nearest_frames": [x, y], "frame_gap": gap,
                              "best_tile_ncc": ncc, "offset_rc": list(off),
                              "verdict": "undecidable from pixels (no shared instant)"})
                evidence.append(entry)
                continue
            picks = sorted({shared[0], shared[len(shared) // 2], shared[-1]})
            nccs, offs = [], []
            for t in picks:
                ncc, off = _best_tile_match(ss[src_s.index(t)].astype(np.float32),
                                            sl[src_l.index(t)].astype(np.float32))
                nccs.append(ncc)
                offs.append(off)
            same = all(n >= SAME_PIXELS_NCC for n in nccs) and len(set(offs)) == 1
            entry.update({"source_frames_checked": picks, "best_tile_ncc": nccs,
                          "offsets_rc": [list(o) for o in offs],
                          "verdict": "same pixels" if same else "different regions"})
            evidence.append(entry)
            if same:
                overlaps.append(RegionOverlap(small.movie, large.movie, offs[0], nccs, picks))
    return overlaps, evidence


def _duplicate_tracks(overlap: RegionOverlap, small: ResultFolder, large: ResultFolder,
                      n_small: int, n_large: int) -> set[int]:
    """Tracks of the smaller crop that are the larger crop's cells seen again."""
    src_s = _source_frames(small, n_small)
    src_l = _source_frames(large, n_large)
    dr, dc = overlap.offset_rc
    lt = large.tracks.assign(source=[src_l[int(f)] for f in large.tracks["frame"]])
    dupes = set()
    for track_id, tr in small.tracks.groupby("track_id"):
        hits = 0
        for _, o in tr.iterrows():
            cand = lt[lt["source"] == src_s[int(o["frame"])]]
            if len(cand) and np.min(np.hypot(cand["x_px"] - (o["x_px"] + dc),
                                             cand["y_px"] - (o["y_px"] + dr))) <= DUPLICATE_MAX_PX:
                hits += 1
        if hits * 2 >= len(tr):
            dupes.add(int(track_id))
    return dupes


#: Overlap verdicts that prove two movies hold different cells.
ESTABLISHED_DISTINCT = ("different regions", "different acquisitions")


def pixel_established_fields(folders: list[dict], evidence: list[dict]) -> list[str]:
    """The largest set of fields that the pixels prove pairwise distinct, greedily by size.

    Two fields are established as distinct only if every pair of their movies
    was compared at a shared instant and found to be different regions, or
    belongs to two resolved acquisitions (two dishes share no cell). A pair
    with no evidence entry, or one ``"not checked"``, is not established. A crop
    that shares no source frame with another (``052924_t3_dual`` is frames
    42-52, ``052924_1`` frames 1-20) may show the same lanes, and so possibly
    the same cell, hours later; pixels cannot say (``SAME_PIXELS_NCC``), so it
    is left out of this set rather than assumed independent. Fields are taken
    largest first (by observations), each kept only if established against
    every field already kept. Used for a sensitivity analysis that bounds what
    an undecidable crop could contribute, not for the primary split.
    """
    field_of = {f["movie"]: f["field"] for f in folders}
    size: dict[str, int] = {}
    for f in folders:
        size[f["field"]] = size.get(f["field"], 0) + int(f["n_observations"])
    verdict: dict[frozenset, bool] = {}
    for e in evidence:
        fa, fb = field_of.get(e["small"]), field_of.get(e["large"])
        if fa is None or fb is None or fa == fb:
            continue
        key = frozenset((fa, fb))
        verdict[key] = verdict.get(key, True) and e.get("verdict") in ESTABLISHED_DISTINCT
    kept: list[str] = []
    for fld in sorted(size, key=lambda k: (-size[k], k)):
        if all(verdict.get(frozenset((fld, k)), False) for k in kept):
            kept.append(fld)
    return kept


def _fields(movies: list[str], overlaps: list[RegionOverlap]) -> dict[str, str]:
    parent = {m: m for m in movies}

    def root(m: str) -> str:
        while parent[m] != m:
            m = parent[m]
        return m

    for o in overlaps:
        parent[root(o.small)] = root(o.large)
    members: dict[str, list[str]] = {}
    for m in movies:
        members.setdefault(root(m), []).append(m)
    return {m: "+".join(sorted(members[root(m)])) for m in movies}


# ---------------------------------------------------------------------------
# The dataset


@dataclass
class PredictionDataset:
    samples: pd.DataFrame
    crops: np.ndarray                 # (n, 64, 64) float32, aligned with samples
    strict_columns: list[str]
    phenotype_columns: list[str]
    history_columns: list[str]
    exclusions: dict[str, int]
    folders: list[dict]
    #: Every primary, non-border mask of every folder (for the unsupervised
    #: embedding, which may learn from masks that have no future target).
    all_crops: np.ndarray = field(default_factory=lambda: np.zeros((0, F.CROP_SIZE_PX, F.CROP_SIZE_PX), np.float32))
    all_crop_movies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    all_crop_experiments: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    all_crop_acquisitions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    all_crop_fields: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    #: Pixel-identical overlaps found between movies, and the evidence for every pair.
    overlaps: list = field(default_factory=list)
    overlap_evidence: list = field(default_factory=list)
    #: What could and could not be checked (``build_dataset``): a guard that
    #: did not run must be visible, not just produce fewer rows.
    checks: dict = field(default_factory=dict)

    def group_ids(self, by: str) -> np.ndarray:
        """``all_crops``' group ids for a grouping of :meth:`Task.groups` (the embedding pool)."""
        pools = {"movie": self.all_crop_movies, "field": self.all_crop_fields,
                 "experiment": self.all_crop_experiments,
                 "acquisition": self.all_crop_acquisitions}
        if by not in pools:
            raise ValueError(f"no embedding pool for grouping {by!r}")
        return pools[by]

    def target_names(self) -> list[str]:
        return [c for c in self.samples.columns if re.match(r"^(disp_um|speed_um_per_hr|migrating)_h\d+$", c)]

    def task(self, target: str, *, require_history: bool = True,
             require_phenotype: bool = False) -> "Task":
        """Rows with a finite ``target`` (and history), sorted by track then time."""
        df = self.samples
        ok = np.array(df[target].notna(), dtype=bool)
        if require_history:
            ok &= df[self.history_columns].notna().all(axis=1).to_numpy()
        if require_phenotype and self.phenotype_columns:
            ok &= df[self.phenotype_columns].notna().all(axis=1).to_numpy()
        ok &= df[self.strict_columns].notna().all(axis=1).to_numpy()
        idx = np.flatnonzero(ok)
        sub = df.iloc[idx]
        order = np.lexsort((sub["frame"].to_numpy(), sub["track_uid"].to_numpy()))
        idx = idx[order]
        sub = df.iloc[idx].reset_index(drop=True)
        return Task(
            target=target,
            y=sub[target].to_numpy(float),
            blocks={
                "strict": sub[self.strict_columns].to_numpy(float),
                "phenotype": (sub[self.phenotype_columns].to_numpy(float)
                              if self.phenotype_columns else np.zeros((len(sub), 0))),
                "history": sub[self.history_columns].to_numpy(float),
            },
            crops=self.crops[idx],
            track=sub["track_uid"].to_numpy(),
            movie=sub["movie"].to_numpy(),
            field=sub["field"].to_numpy(),
            experiment=sub["experiment"].to_numpy(),
            acquisition=sub["acquisition"].to_numpy(),
            condition=sub["condition"].to_numpy(),
            frame=sub["frame"].to_numpy(int),
            classification=target.startswith("migrating"),
            columns={"strict": list(self.strict_columns),
                     "phenotype": list(self.phenotype_columns),
                     "history": list(self.history_columns)},
            rows=sub,
        )


@dataclass
class Task:
    """Arrays for one target, rows sorted by (track_uid, frame)."""

    target: str
    y: np.ndarray
    blocks: dict[str, np.ndarray]
    crops: np.ndarray
    track: np.ndarray
    movie: np.ndarray
    field: np.ndarray
    experiment: np.ndarray
    acquisition: np.ndarray
    condition: np.ndarray
    frame: np.ndarray
    classification: bool
    columns: dict[str, list[str]]
    rows: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.y)

    def groups(self, by: str) -> np.ndarray:
        if by == "track":
            return self.track
        if by == "movie":
            return self.movie
        if by == "field":
            return self.field
        if by == "experiment":
            return self.experiment
        if by == "acquisition":
            return self.acquisition
        raise ValueError(f"unknown grouping {by!r}")

    def summary(self) -> dict:
        out = {
            "n_observations": int(self.n),
            "n_tracks": int(len(np.unique(self.track))),
            "n_movies": int(len(np.unique(self.movie))),
            "n_fields": int(len(np.unique(self.field))),
            "n_experiments": int(len(np.unique(self.experiment))),
            "n_acquisitions": int(len(np.unique(self.acquisition))),
            "observations_per_field": {str(k): int(v) for k, v in
                                       zip(*np.unique(self.field, return_counts=True))},
            "observations_per_movie": {str(k): int(v) for k, v in
                                       zip(*np.unique(self.movie, return_counts=True))},
            "tracks_per_movie": {str(m): int(len(np.unique(self.track[self.movie == m])))
                                 for m in np.unique(self.movie)},
        }
        if self.classification:
            out["n_migrating"] = int(np.sum(self.y == 1))
            out["n_stalled"] = int(np.sum(self.y == 0))
        else:
            out["target_mean"] = float(np.mean(self.y)) if self.n else None
            out["target_median"] = float(np.median(self.y)) if self.n else None
        return out


def build_dataset(folders: list[str | Path], *, horizons: tuple[int, ...] = HORIZONS_FRAMES,
                  with_phenotype: bool = True, exclude_border: bool = True,
                  detect_overlaps: bool = True,
                  conditions: dict[str, str] | None = None) -> PredictionDataset:
    """Samples from Corridor result folders (see the module docstring).

    ``conditions`` maps a movie, acquisition or experiment id to its
    experimental condition (:func:`load_result_folder`); unlabelled movies get
    ``UNSPECIFIED_CONDITION``. ``ds.checks`` records which movies had no image,
    whether every pair of movies was compared for overlap, and whether the
    phenotype set was computed or dropped, and why.
    """
    loaded = [load_result_folder(f, conditions) for f in folders]
    calibrated = {lf.pixel_size_um is not None for lf in loaded}
    if len(calibrated) > 1:
        raise ValueError(
            "Some folders are spatially calibrated and some are not. Pixel features "
            "from different instruments are different physical sizes and cannot be pooled."
        )
    px_for_names = loaded[0].pixel_size_um if loaded else None
    strict_cols = F.strict_feature_names(px_for_names)
    hist_cols = ["hist_speed_last_um_per_hr", "hist_speed_mean_um_per_hr", "hist_net_rate_um_per_hr"]
    pheno_cols: list[str] = []
    want_pheno = with_phenotype and all(lf.image_path is not None for lf in loaded)
    stacks: dict[str, np.ndarray] = {}
    if want_pheno or detect_overlaps:
        import tifffile

        for lf in loaded:
            if lf.image_path is not None:
                stacks[lf.movie] = tifffile.imread(lf.image_path)
    overlaps, overlap_evidence = (find_region_overlaps(loaded, stacks)
                                  if detect_overlaps else ([], []))
    without_image = sorted(lf.movie for lf in loaded if lf.image_path is None)
    unchecked = [e for e in overlap_evidence if str(e.get("verdict", "")).startswith("not checked")]
    checks = {
        "movies_without_image": without_image,
        "overlap_check": ("not requested" if not detect_overlaps
                          else "complete" if not unchecked
                          else f"incomplete: {len(unchecked)} of {len(overlap_evidence)} movie pairs "
                               "not compared"),
        "overlap_pairs_not_checked": len(unchecked),
        "phenotype": ("not requested" if not with_phenotype
                      else "computed" if want_pheno
                      else "dropped: no image for " + ", ".join(without_image)),
        "unresolved_acquisitions": sorted(lf.movie for lf in loaded if not is_resolved(lf.acquisition)),
        "n_movies_with_condition": int(sum(lf.condition is not None for lf in loaded)),
    }
    by_movie = {lf.movie: lf for lf in loaded}
    duplicates: dict[str, set[int]] = {}
    for o in overlaps:
        small, large = by_movie[o.small], by_movie[o.large]
        duplicates.setdefault(o.small, set()).update(_duplicate_tracks(
            o, small, large, len(stacks[o.small]), len(stacks[o.large])))
    field_of = _fields([lf.movie for lf in loaded], overlaps)

    rows: list[dict] = []
    crops: list[np.ndarray] = []
    all_crops: list[np.ndarray] = []
    all_movies: list[str] = []
    all_exps: list[str] = []
    all_acqs: list[str] = []
    all_fields: list[str] = []
    excl = {"recovered_no_mask": 0, "label_area_mismatch": 0, "touches_border": 0,
            "too_small_or_no_contour": 0, "duplicate_of_overlapping_movie": 0,
            "observations_total": 0}
    folder_info: list[dict] = []

    for lf in loaded:
        masks = lf.masks()
        stack = stacks.get(lf.movie) if want_pheno else None
        if stack is not None and stack.shape != masks.shape:
            raise ValueError(f"{lf.movie}: image {stack.shape} and masks {masks.shape} differ")
        tr_all = lf.tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
        excl["observations_total"] += len(tr_all)
        n_tracks_in_folder, n_obs_in_folder = int(tr_all["track_id"].nunique()), int(len(tr_all))
        dropped = duplicates.get(lf.movie, set())
        is_dupe = tr_all["track_id"].isin(dropped)
        excl["duplicate_of_overlapping_movie"] += int(is_dupe.sum())
        tr_all = tr_all[~is_dupe].reset_index(drop=True)
        folder_info.append({
            "movie": lf.movie, "experiment": lf.experiment, "acquisition": lf.acquisition,
            "path": _rel(lf.path),
            "field": field_of[lf.movie],
            # Whether a condition was given, never the label itself: conditions
            # are what the privacy rule keeps out of committed files.
            "condition_known": lf.condition is not None,
            "source_frames": _source_frames(lf, len(masks)),
            "duplicate_tracks_dropped": sorted(dropped),
            "pixel_size_um": lf.pixel_size_um, "frame_interval_min": lf.frame_interval_min,
            # As written by Corridor, then after dropping the duplicated tracks.
            "n_tracks_in_folder": n_tracks_in_folder, "n_observations_in_folder": n_obs_in_folder,
            "n_tracks": int(tr_all["track_id"].nunique()), "n_observations": int(len(tr_all)),
            "masks_shape": list(masks.shape),
            "phenotype_image": _rel(lf.image_path) if (want_pheno and lf.image_path) else None,
        })
        objects = {f: ndimage.find_objects(masks[f]) for f in np.unique(tr_all["frame"].astype(int))}
        _, height, width = masks.shape
        for track_id, tr in tr_all.groupby("track_id", sort=True):
            tr = tr.reset_index(drop=True)
            frames = tr["frame"].to_numpy(int)
            xy = _positions_um(tr, lf.pixel_size_um)
            t_min = _times_min(tr, lf.frame_interval_min)
            source = (tr["detection_source"].fillna("primary").astype(str).to_numpy()
                      if "detection_source" in tr.columns else np.array(["primary"] * len(tr)))
            for i in range(len(tr)):
                f, label = int(frames[i]), int(tr.loc[i, "det_label"])
                if source[i] != "primary":
                    excl["recovered_no_mask"] += 1
                    continue
                slices = objects[f]
                sl = slices[label - 1] if 0 < label <= len(slices) else None
                if sl is None:
                    excl["recovered_no_mask"] += 1
                    continue
                r0 = max(sl[0].start - CROP_PAD_PX, 0)
                r1 = min(sl[0].stop + CROP_PAD_PX, height)
                c0 = max(sl[1].start - CROP_PAD_PX, 0)
                c1 = min(sl[1].stop + CROP_PAD_PX, width)
                crop_mask = masks[f, r0:r1, c0:c1] == label
                n_px = int(crop_mask.sum())
                if "area_px" in tr.columns and pd.notna(tr.loc[i, "area_px"]) \
                        and abs(n_px - float(tr.loc[i, "area_px"])) > 1.0:
                    excl["label_area_mismatch"] += 1
                    continue
                touches = (sl[0].start == 0 or sl[1].start == 0
                           or sl[0].stop >= height or sl[1].stop >= width)
                if touches and exclude_border:
                    excl["touches_border"] += 1
                    continue
                try:
                    strict = F.strict_morphology(crop_mask, lf.pixel_size_um)
                except ValueError:
                    excl["too_small_or_no_contour"] += 1
                    continue
                crop = F.standardised_crop(crop_mask)
                all_crops.append(crop)
                all_movies.append(lf.movie)
                all_exps.append(lf.experiment)
                all_acqs.append(lf.acquisition)
                all_fields.append(field_of[lf.movie])
                _, n_components = F.largest_component(crop_mask)
                row: dict = {
                    "experiment": lf.experiment,
                    "acquisition": lf.acquisition,
                    "condition": lf.condition if lf.condition is not None else UNSPECIFIED_CONDITION,
                    "field": field_of[lf.movie],
                    "movie": lf.movie,
                    "track_id": int(track_id),
                    "track_uid": f"{lf.movie}/{int(track_id)}",
                    "frame": f,
                    "det_label": label,
                    "elapsed_min": float(t_min[i]),
                    "mask_components": n_components,
                    "touches_border": bool(touches),
                    "observation_index": i,
                    "track_length": len(tr),
                }
                row.update(strict)
                if stack is not None:
                    row.update(F.phenotype(stack[f, r0:r1, c0:c1], crop_mask))
                if i >= 1:
                    k0 = max(0, i - F.HISTORY_STEPS)
                    row.update(F.motion_history(xy[k0:i + 1], t_min[k0:i + 1]))
                else:
                    row.update({c: np.nan for c in hist_cols})
                for h in horizons:
                    _add_targets(row, h, frames, xy, t_min, source, i)
                rows.append(row)
                crops.append(crop)

    samples = pd.DataFrame(rows)
    if want_pheno and len(samples):
        pheno_cols = [c for c in samples.columns if c.startswith("pheno_")]
    for h in horizons:
        for col in (f"disp_um_h{h}", f"speed_um_per_hr_h{h}", f"migrating_h{h}"):
            if col not in samples.columns:
                samples[col] = np.nan
    ds = PredictionDataset(
        samples=samples,
        crops=np.stack(crops) if crops else np.zeros((0, F.CROP_SIZE_PX, F.CROP_SIZE_PX), np.float32),
        strict_columns=strict_cols,
        phenotype_columns=pheno_cols,
        history_columns=hist_cols,
        exclusions=excl,
        folders=folder_info,
        all_crops=np.stack(all_crops) if all_crops else np.zeros((0, F.CROP_SIZE_PX, F.CROP_SIZE_PX), np.float32),
        all_crop_movies=np.array(all_movies, dtype=object),
        all_crop_experiments=np.array(all_exps, dtype=object),
        all_crop_acquisitions=np.array(all_acqs, dtype=object),
        all_crop_fields=np.array(all_fields, dtype=object),
        overlaps=[{"small": o.small, "large": o.large, "offset_rc": list(o.offset_rc),
                   "ncc": o.ncc, "source_frames_checked": o.source_frames_checked}
                  for o in overlaps],
        overlap_evidence=overlap_evidence,
        checks=checks,
    )
    if len(ds.samples):
        for col in ("movie", "field", "acquisition"):
            assert_tracks_within_groups(ds.samples, col)
    return ds


def _add_targets(row: dict, h: int, frames: np.ndarray, xy: np.ndarray, t_min: np.ndarray,
                 source: np.ndarray, i: int) -> None:
    j = np.flatnonzero(frames == frames[i] + h)
    if len(j) == 0:
        row[f"disp_um_h{h}"] = np.nan
        row[f"speed_um_per_hr_h{h}"] = np.nan
        row[f"migrating_h{h}"] = np.nan
        row[f"target_gap_h{h}"] = np.nan
        row[f"target_recovered_h{h}"] = np.nan
        return
    j = int(j[0])
    elapsed_hr = (t_min[j] - t_min[i]) / 60.0
    disp = float(np.hypot(*(xy[j] - xy[i])))
    path = float(np.hypot(*np.diff(xy[i:j + 1], axis=0).T).sum())
    row[f"disp_um_h{h}"] = disp
    row[f"speed_um_per_hr_h{h}"] = path / elapsed_hr
    row[f"migrating_h{h}"] = float(disp / elapsed_hr >= MIGRATING_MIN_NET_RATE_UM_PER_HR)
    # Fewer observations than frames in between = a gap was bridged; the path
    # through it is a straight line and so a lower bound.
    row[f"target_gap_h{h}"] = float(j - i < h)
    row[f"target_recovered_h{h}"] = float(np.any(source[i + 1:j + 1] != "primary"))


# ---------------------------------------------------------------------------
# Splits that never cut a track


def assert_tracks_within_groups(samples: pd.DataFrame, group_col: str) -> None:
    """Every track_uid must map to exactly one value of ``group_col``."""
    per_track = samples.groupby("track_uid")[group_col].nunique()
    bad = per_track[per_track > 1]
    if len(bad):
        raise LeakageError(f"tracks span several {group_col} groups: {list(bad.index)[:5]}")


def assert_no_track_straddles(track: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    track = np.asarray(track)
    both = np.intersect1d(track[train_idx], track[test_idx])
    if len(both):
        raise LeakageError(f"{len(both)} track(s) on both sides of a split, e.g. {list(both[:3])}")


def outer_split(acquisitions) -> tuple[str, str]:
    """(unit, why): ``"acquisition"`` from two resolved acquisitions up, else ``"field"``.

    The design contract asks for experiment-level splits. Cells of one dish,
    day and device share everything a new experiment would not (medium,
    coating, temperature, focus, the segmentation's error pattern), so holding
    out a field of the same dish measures less than generalisation to a new
    one. The unit is therefore the acquisition, not the contract's date +
    series id, which names one stage position of a dish (module docstring):
    a second series of the same nd2 adds fields, never a second experiment.

    A placeholder id (``UNRESOLVED``, a movie whose run.json names nothing and
    whose image could not be read) is refused: it is a different string for
    every movie, so counting it would turn leave-one-movie-out into an
    "experiment-level" split, and in exactly the case where the overlap check
    could not run either. With fewer than two resolved acquisitions, a field
    (crops proven or assumed to share no pixels) is the strongest split there
    is. The switch is automatic so that the first second acquisition changes
    the claim's level without anyone having to remember to.
    """
    ids = sorted({str(a) for a in acquisitions})
    unresolved = [a for a in ids if not is_resolved(a)]
    if unresolved:
        return "field", (f"{len(unresolved)} movie(s) with no resolved acquisition "
                         f"({', '.join(unresolved[:3])}{', ...' if len(unresolved) > 3 else ''}); "
                         "an unknown is not a group, so no acquisition-level split is claimed")
    if len(ids) >= 2:
        return "acquisition", f"{len(ids)} resolved acquisitions"
    return "field", (f"one acquisition ({ids[0]}): leave-one-field-out is the strongest split"
                     if ids else "no acquisitions")


def outer_grouping(acquisitions) -> str:
    """The unit of :func:`outer_split`, without its reason."""
    return outer_split(acquisitions)[0]


def leave_one_group_out(groups: np.ndarray, track: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """LOGO splits; refuses any grouping finer than a track."""
    groups = np.asarray(groups)
    splits = []
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        if len(train) == 0:
            continue
        assert_no_track_straddles(track, train, test)
        splits.append((train, test))
    return splits


def group_kfold(groups: np.ndarray, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds of whole groups, balanced by size (largest groups placed first).

    Used for inner model selection with groups = tracks, so a hyperparameter is
    never chosen by predicting a track from its own neighbouring frames.
    """
    uniq, codes, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    k = int(min(k, len(uniq)))
    if k < 2:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k)
    fold_of = np.zeros(len(uniq), dtype=int)
    for gi in order:
        f = int(np.argmin(load))
        fold_of[gi] = f
        load[f] += counts[gi]
    fold = fold_of[codes]
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]
