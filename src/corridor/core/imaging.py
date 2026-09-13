"""TIFF reading and microscopy metadata interpretation.

Design rules this module exists to enforce:

1.  The axis order is *read*, never assumed.  A spatial axis must never be
    silently interpreted as time.
2.  The number of frames is the number of frames actually present in this
    file -- not the ``SizeT`` inherited from the original acquisition.  The
    supplied sample crops carry ``SizeT = 54`` from the parent ND2 while
    containing 5, 11, 18 or 20 frames.
3.  Every physical quantity records *where it came from*.  A calibration
    that was guessed and a calibration that was read from the file must not
    look the same downstream.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

#: Ordered best-to-worst.  Used for display and for the run manifest.
SOURCE_ND2_INFO = "nd2_info"
SOURCE_IMAGEJ = "imagej_header"
SOURCE_TIFF_TAG = "tiff_tag"
SOURCE_OME = "ome_xml"
SOURCE_USER = "user_override"
SOURCE_DEFAULT = "fallback_default"
SOURCE_MISSING = "unavailable"

_SOURCE_LABELS = {
    SOURCE_ND2_INFO: "embedded ND2 metadata",
    SOURCE_IMAGEJ: "ImageJ header",
    SOURCE_TIFF_TAG: "TIFF resolution tag",
    SOURCE_OME: "OME-XML",
    SOURCE_USER: "entered by you",
    SOURCE_DEFAULT: "assumed default",
    SOURCE_MISSING: "not available",
}


def source_label(source: str) -> str:
    return _SOURCE_LABELS.get(source, source)


class UnsupportedStackError(ValueError):
    """Raised when a file cannot be interpreted as a 2D time-lapse."""


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


@dataclass
class StackMetadata:
    """Everything known about one time-lapse file."""

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

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_frames, self.height, self.width)

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
            "height": self.height,
            "width": self.width,
            "dtype": self.dtype,
            "axes_raw": self.axes_raw,
            "axes_interpretation": self.axes_interpretation,
            "pixel_size_um": self.pixel_size_um.value,
            "pixel_size_um_source": self.pixel_size_um.source,
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


def _as_float(text: str | None) -> float | None:
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


def _interpret_axes(axes: str, shape: tuple[int, ...]) -> tuple[tuple[int, ...], str, list[str]]:
    """Return (transpose order to TYX, human description, notes).

    Raises :class:`UnsupportedStackError` when the layout cannot be resolved
    unambiguously.  Guessing here would silently corrupt every velocity.
    """
    notes: list[str] = []
    axes = (axes or "").upper()

    # Drop length-1 axes that carry no information (single channel, single Z).
    keep = [i for i, (a, n) in enumerate(zip(axes, shape)) if not (n == 1 and a not in _SPATIAL)]
    dropped = [axes[i] for i in range(len(axes)) if i not in keep]
    if dropped:
        notes.append(f"Collapsed singleton axes: {', '.join(dropped)}.")
    axes_k = "".join(axes[i] for i in keep)
    shape_k = tuple(shape[i] for i in keep)

    if "Y" not in axes_k or "X" not in axes_k:
        raise UnsupportedStackError(
            f"This file reports axes '{axes}' with shape {shape}; it does not contain "
            "the two spatial image axes this analysis needs."
        )

    non_spatial = [a for a in axes_k if a not in _SPATIAL]
    if len(non_spatial) > 1:
        raise UnsupportedStackError(
            f"This file reports axes '{axes}' with shape {shape}. It has more than one "
            "non-spatial dimension, so the time axis is ambiguous. Split it into a "
            "single-channel, single-plane time-lapse first."
        )

    yi, xi = axes_k.index("Y"), axes_k.index("X")

    if not non_spatial:
        # A single 2D image. Valid, but only one frame: no motion to measure.
        order = tuple(keep[i] for i in (yi, xi))
        return order, "single 2D image (treated as one frame)", notes + [
            "This file holds a single image, so no motion can be measured."
        ]

    ti = axes_k.index(non_spatial[0])
    time_axis = non_spatial[0]
    if time_axis == "T":
        desc = "time-first (T, Y, X)" if ti < min(yi, xi) else f"time axis '{time_axis}' moved to front"
    elif time_axis in ("Z", "I", "Q"):
        # ImageJ writes 'Q'/'I' for unlabelled stacks and 'Z' when frames were
        # saved as slices. Accept, but say so out loud in the run manifest.
        desc = f"axis '{time_axis}' interpreted as time"
        notes.append(
            f"The file labels its third axis '{time_axis}' rather than 'T'; it was read as time."
        )
    else:
        raise UnsupportedStackError(
            f"This file reports axes '{axes}'. Corridor cannot tell which axis is time."
        )

    order = tuple(keep[i] for i in (ti, yi, xi))
    if ti > max(yi, xi):
        notes.append("The time axis was last in the file and has been moved to the front.")
    return order, desc, notes


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def read_metadata(path: str | Path) -> StackMetadata:
    """Inspect a TIFF without loading its pixels."""
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        if not tf.series:
            raise UnsupportedStackError(f"{path.name} contains no readable image series.")
        series = tf.series[0]
        order, desc, notes = _interpret_axes(series.axes, tuple(series.shape))
        shape = tuple(series.shape[i] for i in order)
        if len(shape) == 2:
            n_frames, height, width = 1, shape[0], shape[1]
        else:
            n_frames, height, width = shape

        ij = dict(tf.imagej_metadata or {})
        info = parse_info_block(ij.get("Info", "") or "")

        pixel = _resolve_pixel_size(tf, ij, info)
        interval = _resolve_frame_interval(ij, info)
        src_frames, src_total = parse_source_frames(ij.get("Labels"))

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

        return StackMetadata(
            path=path,
            n_frames=int(n_frames),
            height=int(height),
            width=int(width),
            dtype=str(series.dtype),
            axes_raw=str(series.axes),
            axes_interpretation=desc,
            pixel_size_um=pixel,
            frame_interval_min=interval,
            source_frames=src_frames,
            source_frame_total=src_total,
            acquisition=acquisition,
            notes=notes,
        )


def _resolve_pixel_size(tf: tifffile.TiffFile, ij: dict, info: dict) -> Calibrated:
    """Pixel size in micrometres, best source first."""
    # 1. The ND2 calibration carries full double precision.
    value = _as_float(info.get("dCalibration"))
    if value and value > 0:
        return Calibrated(value, SOURCE_ND2_INFO)

    # 2. The TIFF resolution tag, if the unit is a real length unit.
    unit = str(ij.get("unit", "")).lower()
    page = tf.pages[0]
    tag = page.tags.get("XResolution")
    if tag is not None:
        try:
            num, den = tag.value
            if num:
                per_unit = float(num) / float(den)
                if per_unit > 0:
                    um_per_px = 1.0 / per_unit
                    if unit in ("micron", "um", "microns", "µm", "micrometer", "micrometre"):
                        return Calibrated(um_per_px, SOURCE_TIFF_TAG)
                    if unit in ("mm", "millimeter", "millimetre"):
                        return Calibrated(um_per_px * 1000.0, SOURCE_TIFF_TAG)
                    if unit in ("nm", "nanometer", "nanometre"):
                        return Calibrated(um_per_px / 1000.0, SOURCE_TIFF_TAG)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return Calibrated(None, SOURCE_MISSING)


def _resolve_frame_interval(ij: dict, info: dict) -> Calibrated:
    """Frame interval in minutes, best source first.

    ``dAvgPeriodDiff`` is the mean measured period in milliseconds and is the
    most accurate description of what actually happened at the microscope.
    ``finterval`` is ImageJ's float32 echo of the same number in seconds.
    ``dPeriod`` is only the nominal setpoint that was requested.
    """
    ms = _as_float(info.get("dAvgPeriodDiff"))
    if ms and ms > 0:
        return Calibrated(ms / 1000.0 / 60.0, SOURCE_ND2_INFO)

    seconds = _as_float(ij.get("finterval"))
    if seconds and seconds > 0:
        return Calibrated(seconds / 60.0, SOURCE_IMAGEJ)

    ms = _as_float(info.get("dPeriod"))
    if ms and ms > 0:
        return Calibrated(ms / 1000.0 / 60.0, SOURCE_ND2_INFO)

    return Calibrated(None, SOURCE_MISSING)


def load_stack(path: str | Path, metadata: StackMetadata | None = None) -> np.ndarray:
    """Load pixels as a contiguous ``(T, Y, X)`` array.

    The transpose is derived from the file's own axis labels, so a ``(Y, X, T)``
    file is corrected rather than misread.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        order, _, _ = _interpret_axes(series.axes, tuple(series.shape))
        arr = series.asarray()

    # Take index 0 of every axis the interpretation discarded (singleton C/Z),
    # then permute what is left into the order the interpretation chose.
    selector = tuple(slice(None) if i in order else 0 for i in range(arr.ndim))
    arr = arr[selector]
    remaining = [i for i in range(len(series.shape)) if i in order]
    arr = np.transpose(arr, [remaining.index(i) for i in order])

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise UnsupportedStackError(
            f"{path.name} resolved to shape {arr.shape}; a (T, Y, X) stack was expected."
        )
    if metadata is not None and arr.shape != metadata.shape:
        raise UnsupportedStackError(
            f"{path.name} loaded as {arr.shape} but its metadata described {metadata.shape}."
        )
    return np.ascontiguousarray(arr)


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
