"""TIFF reading and microscopy metadata interpretation.

Design rules this module exists to enforce:

1.  The axis order is *read*, never assumed.  Time (``T``) and depth (``Z``)
    are taken only from metadata that establishes them: ImageJ's
    ``frames``/``slices``, OME's ``SizeT``/``SizeZ`` (as tifffile turns them
    into ``series.axes``), an explicit axes string written with the file, or
    ``ImportConfig.axes``.  A file that says ``Z`` is never read as time, and
    a stack whose planes are unlabelled (tifffile's ``Q`` or ImageJ's ``I``)
    raises :class:`AmbiguousAxes` naming the choices instead of being
    guessed.  1.x read ``Z``, ``Q`` and ``I`` as time with only a note, so a
    single-timepoint Z stack was analysed as a time-lapse -- every velocity a
    distance between slices divided by a frame interval that never happened.
2.  The number of frames is the number of frames actually present in this
    file -- not the ``SizeT`` inherited from the original acquisition.  The
    supplied sample crops carry ``SizeT = 54`` from the parent ND2 while
    containing 5, 11, 18 or 20 frames.
3.  Every physical quantity records *where it came from*.  A calibration
    that was guessed and a calibration that was read from the file must not
    look the same downstream.  Z spacing is never assumed equal to XY.

Canonical orders (contract §4): the file is described by ``axes`` in
``YX``, ``TYX``, ``ZYX`` or ``TZYX``; a channel (``C``, or RGB samples
``S``) is reduced to ``ImportConfig.channel_index`` and recorded.
:func:`load_stack` always returns ``T[Z]YX`` float32 -- a single image or a
single Z stack gains a length-1 ``T`` -- so every consumer indexes time the
same way.
"""

from __future__ import annotations

import itertools
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import tifffile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import CalibrationConfig, ImportConfig

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

#: Ordered best-to-worst.  Used for display and for the run manifest.
SOURCE_ND2_INFO = "nd2_info"
SOURCE_IMAGEJ = "imagej_header"
SOURCE_TIFF_TAG = "tiff_tag"
SOURCE_OME = "ome_xml"
#: An axes string tifffile read from the file's own description (its
#: "shaped" metadata, or a format such as LSM or STK that defines its axes).
SOURCE_SERIES = "tiff_series_axes"
SOURCE_USER = "user_override"
SOURCE_DEFAULT = "fallback_default"
SOURCE_MISSING = "unavailable"

_SOURCE_LABELS = {
    SOURCE_ND2_INFO: "embedded ND2 metadata",
    SOURCE_IMAGEJ: "ImageJ header",
    SOURCE_TIFF_TAG: "TIFF resolution tag",
    SOURCE_OME: "OME-XML",
    SOURCE_SERIES: "axes recorded in the TIFF",
    SOURCE_USER: "entered by you",
    SOURCE_DEFAULT: "assumed default",
    SOURCE_MISSING: "not available",
}

#: The orders a file is described in after its channel axis is reduced.
CANONICAL_AXES = ("YX", "TYX", "ZYX", "TZYX")

#: Unequal X and Y pixel sizes are recorded (quality control raises them as
#: critical). Below this relative difference the two are the same number
#: written twice with float rounding: 1e-4 of 1000 px is 0.1 px.
ANISOTROPIC_PIXEL_TOLERANCE = 1e-4


def source_label(source: str) -> str:
    return _SOURCE_LABELS.get(source, source)


class UnsupportedStackError(ValueError):
    """Raised when a file cannot be interpreted as a 2-D or 3-D time-lapse."""


class AmbiguousAxes(UnsupportedStackError):
    """The file's metadata does not establish which axis is time and which Z.

    ``choices`` are complete axis orders for this file (``"TYX"``,
    ``"ZYX"``, ...), each of which would read it; the UI asks which one is
    true and the CLI takes ``--axes``.  A subclass of
    :class:`UnsupportedStackError` so that code which only knows that one
    still shows the message rather than a traceback.
    """

    def __init__(
        self,
        choices: Sequence[str],
        message: str,
        *,
        axes_raw: str = "",
        shape: tuple[int, ...] = (),
    ) -> None:
        self.choices: tuple[str, ...] = tuple(choices)
        self.axes_raw = axes_raw
        self.shape = tuple(shape)
        super().__init__(message)


@dataclass
class Calibrated:
    """A physical quantity plus the provenance of its value."""

    value: float | None
    source: str

    @property
    def known(self) -> bool:
        return self.value is not None and math.isfinite(self.value) and self.value > 0

    def describe(self, unit: str = "", digits: int = 6) -> str:
        if not self.known:
            return f"unknown ({source_label(self.source)})"
        return f"{self.value:.{digits}g}{unit} ({source_label(self.source)})"


def _missing() -> Calibrated:
    return Calibrated(None, SOURCE_MISSING)


@dataclass
class StackMetadata:
    """Everything known about one image file.

    ``axes`` is the canonical description of the file (``YX``, ``TYX``,
    ``ZYX`` or ``TZYX``) and ``axes_shape`` its sizes in that order.
    ``shape`` is what :func:`load_stack` returns, ``T[Z]YX``: identical for
    ``TYX``/``TZYX`` files, with a length-1 ``T`` in front for a single image
    or a single Z stack.  The 1.x fields keep their meaning (``n_frames`` is
    the number of time points), and every 2.0 field has a default so a
    hand-built 2-D metadata object (tests, v1 projects) stays valid.
    """

    path: Path
    n_frames: int
    height: int
    width: int
    dtype: str
    axes_raw: str
    axes_interpretation: str
    pixel_size_um: Calibrated
    frame_interval_min: Calibrated
    source_frames: list[int] | None = None
    source_frame_total: int | None = None
    acquisition: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- 2.0: dimensions --------------------------------------------------
    axes: str = "TYX"
    #: Where the T/Z reading came from: ImageJ header, OME-XML, the TIFF's
    #: own axes string, or the user's ``ImportConfig.axes``.
    axes_source: str = SOURCE_MISSING
    #: The full axis string applied to the file's pixel series, one letter per
    #: stored dimension (``TZCYX`` for a hyperstack). :func:`load_stack`
    #: replays it so the pixels are read exactly as the metadata was.
    axes_used: str = ""
    #: Z planes per time point; 1 for a 2-D file.
    n_slices: int = 1
    z_step_um: Calibrated = field(default_factory=_missing)
    #: Recorded separately so unequal X/Y pixel sizes cannot hide behind one
    #: number. ``pixel_size_um`` is the X size.
    pixel_size_y_um: Calibrated = field(default_factory=_missing)
    #: The analysed channel, and how many the file holds.
    channel_index: int = 0
    n_channels: int = 1

    @property
    def dimensionality(self) -> str:
        return "3D" if "Z" in self.axes else "2D"

    @property
    def stack_axes(self) -> str:
        """The axes of :func:`load_stack`'s array: ``TYX`` or ``TZYX``."""
        return "TZYX" if "Z" in self.axes else "TYX"

    @property
    def shape(self) -> tuple[int, ...]:
        """``(T, [Z,] Y, X)``, exactly what :func:`load_stack` returns."""
        if "Z" in self.axes:
            return (self.n_frames, self.n_slices, self.height, self.width)
        return (self.n_frames, self.height, self.width)

    @property
    def axes_shape(self) -> tuple[int, ...]:
        """Sizes in the order of ``axes`` (no invented length-1 T)."""
        sizes = {"T": self.n_frames, "Z": self.n_slices, "Y": self.height, "X": self.width}
        return tuple(sizes[a] for a in self.axes)

    @property
    def anisotropic_pixels(self) -> bool:
        """True when X and Y pixel sizes are both known and differ."""
        x, y = self.pixel_size_um, self.pixel_size_y_um
        if not (x.known and y.known):
            return False
        return abs(float(x.value) - float(y.value)) > ANISOTROPIC_PIXEL_TOLERANCE * float(x.value)

    @property
    def duration_min(self) -> float | None:
        if not self.frame_interval_min.known or self.n_frames < 2:
            return None
        return (self.n_frames - 1) * float(self.frame_interval_min.value)

    def elapsed_min(self, frame: int) -> float | None:
        if not self.frame_interval_min.known:
            return None
        return frame * float(self.frame_interval_min.value)

    def source_frame(self, frame: int) -> int | None:
        if self.source_frames is None or frame >= len(self.source_frames):
            return None
        return self.source_frames[frame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.path.name,
            "n_frames": self.n_frames,
            "n_slices": self.n_slices,
            "height": self.height,
            "width": self.width,
            "dtype": self.dtype,
            "axes": self.axes,
            "axes_shape": list(self.axes_shape),
            "shape": list(self.shape),
            "dimensionality": self.dimensionality,
            "axes_source": self.axes_source,
            "axes_raw": self.axes_raw,
            "axes_used": self.axes_used,
            "axes_interpretation": self.axes_interpretation,
            "channel_index": self.channel_index,
            "n_channels": self.n_channels,
            "pixel_size_um": self.pixel_size_um.value,
            "pixel_size_um_source": self.pixel_size_um.source,
            "pixel_size_y_um": self.pixel_size_y_um.value,
            "pixel_size_y_um_source": self.pixel_size_y_um.source,
            "anisotropic_pixels": self.anisotropic_pixels,
            "z_step_um": self.z_step_um.value,
            "z_step_um_source": self.z_step_um.source,
            "frame_interval_min": self.frame_interval_min.value,
            "frame_interval_min_source": self.frame_interval_min.source,
            "source_frames": self.source_frames,
            "source_frame_total": self.source_frame_total,
            "acquisition": self.acquisition,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# ND2 "Info" block parsing
# --------------------------------------------------------------------------

_INFO_NUM = re.compile(r"^\s*([A-Za-z0-9_ .#()\-]+?)\s*=\s*(.+?)\s*$")


def parse_info_block(info: str) -> dict[str, str]:
    """Parse the ``key = value`` lines ImageJ keeps from the source ND2."""
    out: dict[str, str] = {}
    for line in info.splitlines():
        m = _INFO_NUM.match(line)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            # Later duplicates are usually per-channel repeats; keep the first.
            out.setdefault(key, value)
    return out


def _as_float(text: Any) -> float | None:
    if text is None:
        return None
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError):
        return None


_LABEL_RE = re.compile(r"\bt:(\d+)\s*/\s*(\d+)")


def parse_source_frames(labels: list[str] | None) -> tuple[list[int] | None, int | None]:
    """Recover original acquisition frame numbers from ImageJ slice labels.

    ImageJ writes labels such as ``t:42/54 - <acquisition>.nd2 (series 01)``
    when a subset of a longer time-lapse is exported.  Preserving these makes
    a result traceable back to the original experiment.
    """
    if not labels:
        return None, None
    frames: list[int] = []
    total: int | None = None
    for label in labels:
        m = _LABEL_RE.search(label or "")
        if not m:
            return None, None
        frames.append(int(m.group(1)))
        total = int(m.group(2))
    return frames, total


# --------------------------------------------------------------------------
# Axis interpretation
# --------------------------------------------------------------------------

_SPATIAL = ("Y", "X")
#: A channel: fluorescence channels (C) and RGB samples (S) alike are reduced
#: to one, never read as time or depth.
_CHANNEL_LETTERS = ("C", "S")
_KNOWN = ("T", "Z", "Y", "X") + _CHANNEL_LETTERS
#: Letters an override may use. Q and I are tifffile's and ImageJ's words
#: for "unlabelled", which is exactly what an override exists to replace.
_OVERRIDE_LETTERS = frozenset("TZCSYX")


@dataclass(frozen=True)
class _Layout:
    """How one stored pixel series maps onto a canonical order."""

    axes_used: str  # one letter per stored dimension
    canonical: str  # YX | TYX | ZYX | TZYX
    order: tuple[int, ...]  # stored-dimension indices, in canonical order
    channel_axis: int | None  # stored-dimension index of the channel, if any
    channel_index: int
    n_channels: int
    dropped: tuple[str, ...]  # singleton letters collapsed


def _layout(axes: str, shape: Sequence[int], channel_index: int = 0) -> _Layout:
    """Map an axes string onto a canonical order, or refuse.

    Raises :class:`AmbiguousAxes` (via the caller, which knows the choices)
    only through :func:`_unknown_axes`; here an unlabelled non-singleton axis
    is a programming error of the caller and raises UnsupportedStackError.
    """
    axes = axes.upper()
    shape = tuple(int(n) for n in shape)
    if len(axes) != len(shape):
        raise UnsupportedStackError(
            f"The axis order '{axes}' names {len(axes)} axes but the image has "
            f"{len(shape)} ({shape})."
        )
    for letter in ("Y", "X"):
        if axes.count(letter) != 1:
            raise UnsupportedStackError(
                f"This file reports axes '{axes}' with shape {shape}; it does not contain "
                "the two spatial image axes this analysis needs."
            )

    kept: dict[str, int] = {}
    dropped: list[str] = []
    channel_axis: int | None = None
    for i, (letter, n) in enumerate(zip(axes, shape)):
        if letter in _SPATIAL:
            kept[letter] = i
            continue
        if n == 1:
            dropped.append(letter)
            continue
        if letter in _CHANNEL_LETTERS:
            if channel_axis is not None:
                raise UnsupportedStackError(
                    f"This file reports axes '{axes}' with shape {shape}: it has two "
                    "channel axes, and Corridor analyses one channel of one image."
                )
            channel_axis = i
            continue
        if letter not in ("T", "Z"):
            raise UnsupportedStackError(
                f"This file reports axes '{axes}' with shape {shape}. Corridor cannot "
                f"tell what its '{letter}' axis is."
            )
        if letter in kept:
            raise UnsupportedStackError(
                f"This file reports axes '{axes}' with shape {shape}: '{letter}' appears twice."
            )
        kept[letter] = i

    n_channels = int(shape[channel_axis]) if channel_axis is not None else 1
    if not 0 <= int(channel_index) < n_channels:
        raise UnsupportedStackError(
            f"Channel {channel_index} was requested, but this file has "
            f"{n_channels} channel{'s' if n_channels != 1 else ''} "
            f"(axes '{axes}', shape {shape}). Channels are numbered from 0."
        )

    canonical = "".join(a for a in "TZYX" if a in kept)
    return _Layout(
        axes_used=axes,
        canonical=canonical,
        order=tuple(kept[a] for a in canonical),
        channel_axis=channel_axis,
        channel_index=int(channel_index),
        n_channels=n_channels,
        dropped=tuple(dropped),
    )


def _unknown_axes(axes: str, shape: Sequence[int]) -> list[int]:
    """Indices of non-singleton axes whose meaning the file does not state."""
    return [i for i, (a, n) in enumerate(zip(axes.upper(), shape)) if n > 1 and a not in _KNOWN]


def _choices(axes: str, shape: Sequence[int]) -> list[str]:
    """Every complete reading of a file whose unlabelled axes are filled in.

    Time first, then Z, then channel, so the list reads in order of how often
    each is the truth for this application.
    """
    axes = axes.upper()
    unknown = _unknown_axes(axes, shape)
    taken = {a for a, n in zip(axes, shape) if n > 1}
    pool = [a for a in "TZC" if a not in taken]
    out: list[str] = []
    for letters in itertools.permutations(pool, len(unknown)):
        candidate = list(axes)
        for i, letter in zip(unknown, letters):
            candidate[i] = letter
        text = "".join(candidate)
        try:
            _layout(text, shape)
        except UnsupportedStackError:
            continue
        if text not in out:
            out.append(text)
    return out


def _apply_override(override: str, series_axes: str, shape: Sequence[int]) -> str:
    """Fit an explicit axis order onto the stored dimensions.

    The order may name every stored dimension, or only the non-singleton ones
    (tifffile keeps length-1 axes that the user never sees); the singletons
    keep the file's own letters and are collapsed later.
    """
    text = re.sub(r"\s+", "", str(override)).upper()
    bad = sorted(set(text) - _OVERRIDE_LETTERS)
    if not text or bad:
        raise UnsupportedStackError(
            f"The axis order '{override}' is not valid: use the letters T, Z, C, Y and X "
            "(for example TYX, ZYX or TZCYX)."
        )
    shape = tuple(int(n) for n in shape)
    if len(text) == len(shape):
        return text
    big = [i for i, n in enumerate(shape) if n > 1]
    if len(text) == len(big):
        letters = list(series_axes.upper())
        for i, letter in zip(big, text):
            letters[i] = letter
        return "".join(letters)
    raise UnsupportedStackError(
        f"The axis order '{override}' names {len(text)} axes, but this file stores "
        f"{len(shape)} ({'x'.join(str(n) for n in shape)})."
    )


def _describe(layout: _Layout) -> str:
    base = {
        "YX": "single 2D image (treated as one frame)",
        "TYX": "time-lapse (T, Y, X)",
        "ZYX": "single Z stack (Z, Y, X)",
        "TZYX": "time-lapse of Z stacks (T, Z, Y, X)",
    }[layout.canonical]
    if layout.channel_axis is not None:
        base += f", channel {layout.channel_index} of {layout.n_channels}"
    return base


def _layout_notes(layout: _Layout) -> list[str]:
    notes: list[str] = []
    if layout.dropped:
        notes.append(f"Collapsed singleton axes: {', '.join(layout.dropped)}.")
    if layout.channel_axis is not None:
        notes.append(
            f"The file holds {layout.n_channels} channels; channel {layout.channel_index} "
            "(numbered from 0) was analysed."
        )
    if layout.canonical == "YX":
        notes.append("This file holds a single image, so no motion can be measured.")
    if layout.canonical == "ZYX":
        notes.append(
            "This file holds one Z stack (one time point), so no motion can be measured."
        )
    if list(layout.order) != sorted(layout.order):
        notes.append(
            f"The stored axis order '{layout.axes_used}' was rearranged to "
            f"'{layout.canonical}'."
        )
    return notes


def _resolve_layout(
    tf: tifffile.TiffFile,
    series: Any,
    import_config: "ImportConfig | None",
) -> tuple[_Layout, str, list[str]]:
    """(layout, axes source, notes) for a file, honouring ``ImportConfig``."""
    series_axes = str(series.axes).upper()
    shape = tuple(int(n) for n in series.shape)
    channel_index = int(getattr(import_config, "channel_index", 0) or 0)
    override = getattr(import_config, "axes", None)
    notes: list[str] = []

    file_source = (
        SOURCE_OME if tf.is_ome else SOURCE_IMAGEJ if tf.is_imagej else SOURCE_SERIES
    )
    established = not _unknown_axes(series_axes, shape)

    if override:
        axes_used = _apply_override(override, series_axes, shape)
        layout = _layout(axes_used, shape, channel_index)
        if established:
            try:
                file_canonical = _layout(series_axes, shape, 0).canonical
            except UnsupportedStackError:
                file_canonical = series_axes
            if file_canonical != layout.canonical:
                notes.append(
                    f"The file's metadata reads as '{file_canonical}'; it was read as "
                    f"'{layout.canonical}' because the import settings say so."
                )
        else:
            notes.append(
                f"The file does not say what its axes are ('{series_axes}'); "
                f"'{layout.canonical}' was taken from the import settings."
            )
        return layout, SOURCE_USER, notes

    if not established:
        choices = _choices(series_axes, shape)
        unknown = [series_axes[i] for i in _unknown_axes(series_axes, shape)]
        sizes = "x".join(str(n) for n in shape)
        raise AmbiguousAxes(
            choices,
            f"This file's metadata does not say whether its "
            f"{', '.join(str(shape[i]) for i in _unknown_axes(series_axes, shape))} "
            f"planes (axis '{''.join(unknown)}', shape {sizes}) are time points, Z slices "
            f"or channels, and guessing would turn depth into motion. Choose the axis "
            f"order: {', '.join(choices) or 'none fits'} (T = time, Z = depth, "
            "C = channel) in the import settings, or with --axes on the command line.",
            axes_raw=series_axes,
            shape=shape,
        )

    layout = _layout(series_axes, shape, channel_index)
    return layout, file_source, notes


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

_MICRO = ("µ", "μ", "\\u00b5", "\\u03bc", "u")


def _um_per_unit(unit: Any) -> float | None:
    """Micrometres per one ``unit`` of length, or None for a non-length unit.

    ImageJ writes the micro sign three ways in the wild (``µm``, ``um`` and
    the escaped ``\\u00B5m``); 1.x matched only some of them and silently
    lost the calibration for the rest.
    """
    text = str(unit or "").strip().lower()
    if not text:
        return None
    for micro in _MICRO:
        if text.startswith(micro):
            rest = text[len(micro):]
            if rest in ("m", "meter", "metre", "meters", "metres"):
                return 1.0
    if text in ("micron", "microns", "micrometer", "micrometre", "micrometers", "micrometres"):
        return 1.0
    if text in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
        return 1e-3
    if text in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        return 1e3
    if text in ("cm", "centimeter", "centimetre"):
        return 1e4
    if text in ("m", "meter", "metre"):
        return 1e6
    return None


def _min_per_unit(unit: Any) -> float | None:
    """Minutes per one ``unit`` of time, or None when it is not a time unit."""
    text = str(unit or "").strip().lower()
    for micro in _MICRO:
        if text.startswith(micro) and text[len(micro):] in ("s", "sec"):
            return 1.0 / 60e6
    return {
        "s": 1.0 / 60.0,
        "sec": 1.0 / 60.0,
        "secs": 1.0 / 60.0,
        "second": 1.0 / 60.0,
        "seconds": 1.0 / 60.0,
        "ms": 1.0 / 60e3,
        "msec": 1.0 / 60e3,
        "millisecond": 1.0 / 60e3,
        "milliseconds": 1.0 / 60e3,
        "min": 1.0,
        "mins": 1.0,
        "minute": 1.0,
        "minutes": 1.0,
        "h": 60.0,
        "hr": 60.0,
        "hrs": 60.0,
        "hour": 60.0,
        "hours": 60.0,
    }.get(text)


# --------------------------------------------------------------------------
# OME-XML
# --------------------------------------------------------------------------


def _ome_pixels(tf: tifffile.TiffFile) -> dict[str, str]:
    """Attributes of the first ``Pixels`` element (series 0), or {}."""
    if not tf.is_ome:
        return {}
    try:
        root = ET.fromstring(tf.ome_metadata or "")
    except ET.ParseError:
        return {}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Pixels":
            return dict(element.attrib)
    return {}


def _ome_length(pixels: dict[str, str], key: str) -> float | None:
    """A ``PhysicalSize*`` in µm. The OME schema's default unit is µm."""
    value = _as_float(pixels.get(key))
    if not value or value <= 0:
        return None
    factor = _um_per_unit(pixels.get(f"{key}Unit", "µm"))
    return value * factor if factor else None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def read_metadata(path: str | Path, import_config: "ImportConfig | None" = None) -> StackMetadata:
    """Inspect a TIFF without loading its pixels.

    Raises :class:`AmbiguousAxes` when the metadata cannot establish T and Z
    and ``import_config.axes`` does not say; :class:`UnsupportedStackError`
    when the file cannot be read as a 2-D or 3-D image series at all.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise UnsupportedStackError(f"{path.name} contains no readable image series.")
        series = tf.series[0]
        layout, axes_source, notes = _resolve_layout(tf, series, import_config)
        notes = notes + _layout_notes(layout)
        shape = tuple(int(n) for n in series.shape)
        sizes = {a: shape[i] for a, i in zip(layout.canonical, layout.order)}

        ij = dict(tf.imagej_metadata or {})
        info = parse_info_block(ij.get("Info", "") or "")
        ome = _ome_pixels(tf)

        pixel_x, pixel_y = _resolve_pixel_size(tf, ij, info, ome)
        interval = _resolve_frame_interval(ij, info, ome, notes)
        z_step = (
            _resolve_z_step(ij, ome) if "Z" in layout.canonical else Calibrated(None, SOURCE_MISSING)
        )
        src_frames, src_total = parse_source_frames(ij.get("Labels"))
        n_frames = sizes.get("T", 1)
        if src_frames is not None and len(src_frames) != n_frames:
            # ImageJ labels every plane; in a hyperstack that is T x Z x C
            # labels, not one per time point, and they do not number frames.
            src_frames, src_total = None, None

        acquisition = {
            k: info[k]
            for k in (
                "SizeT",
                "SizeX",
                "SizeY",
                "SizeC",
                "SizeZ",
                "Time Loop",
                "dPeriod",
                "dAvgPeriodDiff",
                "dMinPeriodDiff",
                "dMaxPeriodDiff",
                "dCalibration",
                "sObjective",
                "dObjectiveNA",
                "dExposureTime",
            )
            if k in info
        }
        if ij.get("ImageJ"):
            acquisition["ImageJ"] = ij["ImageJ"]
        for key in ("SizeT", "SizeZ", "SizeC", "DimensionOrder"):
            if key in ome:
                acquisition[f"OME {key}"] = ome[key]

        declared_t = _as_float(info.get("SizeT"))
        if declared_t and int(declared_t) != n_frames:
            notes.append(
                f"The source acquisition had {int(declared_t)} frames; this file contains "
                f"{n_frames}. Analysis uses the {n_frames} frames present."
            )
        if src_frames:
            notes.append(
                f"Original frame numbers {src_frames[0]}-{src_frames[-1]} of "
                f"{src_total} were preserved from the acquisition."
            )
        if "Z" in layout.canonical and not z_step.known:
            notes.append(
                "The file does not record its Z step, so 3-D sizes are reported in voxels "
                "and every µm³/µm² value stays empty until a Z step is entered."
            )

        metadata = StackMetadata(
            path=path,
            n_frames=int(n_frames),
            height=int(sizes["Y"]),
            width=int(sizes["X"]),
            dtype=str(series.dtype),
            axes_raw=str(series.axes),
            axes_interpretation=_describe(layout),
            pixel_size_um=pixel_x,
            frame_interval_min=interval,
            source_frames=src_frames,
            source_frame_total=src_total,
            acquisition=acquisition,
            notes=notes,
            axes=layout.canonical,
            axes_source=axes_source,
            axes_used=layout.axes_used,
            n_slices=int(sizes.get("Z", 1)),
            z_step_um=z_step,
            pixel_size_y_um=pixel_y,
            channel_index=layout.channel_index,
            n_channels=layout.n_channels,
        )
        if metadata.anisotropic_pixels:
            notes.append(
                f"The pixels are not square: {pixel_x.value:.6g} µm in X but "
                f"{pixel_y.value:.6g} µm in Y. Distances use the X size."
            )
        return metadata


def _resolution_um(tag: Any, factor: float | None) -> float | None:
    if tag is None or not factor:
        return None
    try:
        num, den = tag.value
        if not num:
            return None
        per_unit = float(num) / float(den)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return factor / per_unit if per_unit > 0 else None


def _resolve_pixel_size(
    tf: tifffile.TiffFile, ij: dict, info: dict, ome: dict[str, str]
) -> tuple[Calibrated, Calibrated]:
    """(X, Y) pixel size in micrometres, best source first.

    Both come from the same source, so a difference between them is a fact
    about the pixels rather than about two metadata blocks disagreeing.
    """
    # 1. The ND2 calibration carries full double precision, and ND2 pixels
    #    are square by definition of that one number.
    value = _as_float(info.get("dCalibration"))
    if value and value > 0:
        return Calibrated(value, SOURCE_ND2_INFO), Calibrated(value, SOURCE_ND2_INFO)

    # 2. OME-XML PhysicalSizeX/Y, with their units.
    x = _ome_length(ome, "PhysicalSizeX")
    if x:
        y = _ome_length(ome, "PhysicalSizeY")
        return Calibrated(x, SOURCE_OME), (
            Calibrated(y, SOURCE_OME) if y else Calibrated(None, SOURCE_MISSING)
        )

    # 3. The TIFF resolution tags, if ImageJ says the unit is a length.
    page = tf.pages[0]
    unit = ij.get("unit", "")
    x = _resolution_um(page.tags.get("XResolution"), _um_per_unit(unit))
    if x:
        y = _resolution_um(page.tags.get("YResolution"), _um_per_unit(ij.get("yunit", unit)))
        return Calibrated(x, SOURCE_TIFF_TAG), (
            Calibrated(y, SOURCE_TIFF_TAG) if y else Calibrated(None, SOURCE_MISSING)
        )

    return Calibrated(None, SOURCE_MISSING), Calibrated(None, SOURCE_MISSING)


def _resolve_z_step(ij: dict, ome: dict[str, str]) -> Calibrated:
    """Z step in micrometres from OME ``PhysicalSizeZ`` or ImageJ ``spacing``.

    Never from the XY size: optical stacks are routinely sampled more
    coarsely in Z, and a missing step stays missing.
    """
    z = _ome_length(ome, "PhysicalSizeZ")
    if z:
        return Calibrated(z, SOURCE_OME)
    spacing = _as_float(ij.get("spacing"))
    factor = _um_per_unit(ij.get("zunit", ij.get("unit", "")))
    if spacing and spacing > 0 and factor:
        return Calibrated(spacing * factor, SOURCE_IMAGEJ)
    return Calibrated(None, SOURCE_MISSING)


def _resolve_frame_interval(
    ij: dict, info: dict, ome: dict[str, str], notes: list[str]
) -> Calibrated:
    """Frame interval in minutes, best source first.

    ``dAvgPeriodDiff`` is the mean measured period in milliseconds and is the
    most accurate description of what actually happened at the microscope.
    ``finterval`` is ImageJ's float32 echo of the same number, in ``tunit``
    (seconds unless the header names another unit).  OME's
    ``TimeIncrement`` is in ``TimeIncrementUnit`` (seconds by default).
    ``dPeriod`` is only the nominal setpoint that was requested.
    """
    ms = _as_float(info.get("dAvgPeriodDiff"))
    if ms and ms > 0:
        return Calibrated(ms / 1000.0 / 60.0, SOURCE_ND2_INFO)

    value = _as_float(ij.get("finterval"))
    if value and value > 0:
        tunit = ij.get("tunit", "sec")
        factor = _min_per_unit(tunit)
        if factor:
            return Calibrated(value * factor, SOURCE_IMAGEJ)
        notes.append(
            f"The ImageJ frame interval is in an unrecognised unit ('{tunit}') and was not used."
        )

    value = _as_float(ome.get("TimeIncrement"))
    if value and value > 0:
        unit = ome.get("TimeIncrementUnit", "s")
        factor = _min_per_unit(unit)
        if factor:
            return Calibrated(value * factor, SOURCE_OME)
        notes.append(
            f"The OME time increment is in an unrecognised unit ('{unit}') and was not used."
        )

    ms = _as_float(info.get("dPeriod"))
    if ms and ms > 0:
        return Calibrated(ms / 1000.0 / 60.0, SOURCE_ND2_INFO)

    return Calibrated(None, SOURCE_MISSING)


def effective_z_step(
    metadata: StackMetadata, calibration: "CalibrationConfig | None" = None
) -> Calibrated:
    """The Z step to use: a user override wins and says so, else the file's.

    The 2-D calibration merge lives in ``pipeline.effective_calibration``;
    this is its Z counterpart, kept here so it cannot read a 2-D file's
    stray ImageJ ``spacing`` as a Z step.
    """
    override = getattr(calibration, "z_step_um", None)
    if metadata.dimensionality == "3D" and override and math.isfinite(override) and override > 0:
        return Calibrated(float(override), SOURCE_USER)
    return metadata.z_step_um


def _to_canonical(arr: np.ndarray, layout: _Layout) -> np.ndarray:
    """Reduce channel and singleton axes, then permute to the canonical order."""
    selector = tuple(
        slice(None)
        if i in layout.order
        else (layout.channel_index if i == layout.channel_axis else 0)
        for i in range(arr.ndim)
    )
    arr = arr[selector]
    remaining = [i for i in range(len(layout.axes_used)) if i in layout.order]
    return np.transpose(arr, [remaining.index(i) for i in layout.order])


def _with_time(arr: np.ndarray, canonical: str) -> np.ndarray:
    """Give a single image or a single Z stack its length-1 time axis."""
    return arr[np.newaxis, ...] if not canonical.startswith("T") else arr


def load_stack(
    path: str | Path,
    metadata: StackMetadata | None = None,
    *,
    import_config: "ImportConfig | None" = None,
) -> np.ndarray:
    """Load pixels as a contiguous float32 ``(T, Y, X)`` or ``(T, Z, Y, X)`` array.

    The layout is the one :func:`read_metadata` established (replayed from
    ``metadata.axes_used`` and ``metadata.channel_index``), so pixels are
    never read differently from how the metadata was described.  Float32
    because every consumer computes in floating point and Cellpose converts
    anyway; it holds every uint16 value exactly.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise UnsupportedStackError(f"{path.name} contains no readable image series.")
        series = tf.series[0]
        shape = tuple(int(n) for n in series.shape)
        if metadata is not None and len(metadata.axes_used) == len(shape):
            layout = _layout(metadata.axes_used, shape, metadata.channel_index)
        else:
            layout = _resolve_layout(tf, series, import_config)[0]
        arr = series.asarray()

    arr = _with_time(_to_canonical(np.asarray(arr), layout), layout.canonical)
    if metadata is not None and arr.shape != metadata.shape:
        raise UnsupportedStackError(
            f"{path.name} loaded as {arr.shape} but its metadata described {metadata.shape}."
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_label_array(
    path: str | Path,
    axes: str,
    *,
    label_axes: str | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    """Read a label TIFF as ``T[Z]YX`` without judging its values.

    ``axes`` is the canonical order of the image the labels belong to
    (``YX``, ``TYX``, ``ZYX`` or ``TZYX``).  The label file's own metadata
    decides its layout when it establishes one; otherwise -- the usual case,
    since label images are mostly written without any -- ``label_axes``, or
    failing that ``axes``, is applied.  A channel axis is refused: a label
    image with channels is not one segmentation.

    Returns ``(array, canonical axes of the label file, notes)``.
    """
    path = Path(path)
    target = str(axes or "").strip().upper()
    if target not in CANONICAL_AXES:
        raise ValueError(f"axes must be one of {CANONICAL_AXES}, got {axes!r}")
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise UnsupportedStackError(f"{path.name} contains no readable image series.")
        series = tf.series[0]
        series_axes = str(series.axes).upper()
        shape = tuple(int(n) for n in series.shape)
        notes: list[str] = []
        if label_axes:
            layout = _layout(_apply_override(label_axes, series_axes, shape), shape)
        elif not _unknown_axes(series_axes, shape):
            layout = _layout(series_axes, shape)
        else:
            layout = _layout(_apply_override(target, series_axes, shape), shape)
            notes.append(
                f"The label image does not say what its axes are ('{series_axes}'); "
                f"they were read as '{layout.canonical}', the image's order."
            )
        if layout.channel_axis is not None:
            raise UnsupportedStackError(
                f"{path.name} has {layout.n_channels} channels (axes '{series_axes}', shape "
                f"{shape}); a label image must hold one segmentation."
            )
        arr = series.asarray()
    arr = _with_time(_to_canonical(np.asarray(arr), layout), layout.canonical)
    return np.ascontiguousarray(arr), layout.canonical, notes


def frame_to_display(frame: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Percentile-stretch one frame to uint8 for on-screen display only.

    Never used for measurement -- Cellpose sees the original pixels.
    """
    f = np.asarray(frame, dtype=np.float32)
    if f.size == 0:
        return np.zeros_like(f, dtype=np.uint8)
    lo, hi = np.percentile(f, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(f.shape, dtype=np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
