"""Turning label images into measured detections."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
from skimage.measure import regionprops


@dataclass
class Detection:
    """One segmented object in one frame.

    Coordinates are in pixels, x = column, y = row, matching image display
    order.  ``regionprops`` reports centroids as (row, col); the conversion
    happens here, once.
    """

    frame: int
    label: int
    x: float
    y: float
    area_px: float
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    extent_px: int  # larger bounding-box dimension
    eccentricity: float
    orientation_rad: float  # angle of the major axis, image coords
    major_axis_px: float
    minor_axis_px: float
    solidity: float
    touches_border: bool
    channel: int = -1  # assigned later; -1 == unassigned

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def axis_unit(self) -> np.ndarray:
        """Unit vector along the cell body's major axis, in (x, y)."""
        return np.array(
            [math.sin(self.orientation_rad), math.cos(self.orientation_rad)], dtype=float
        )

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("bbox")
        bmin_r, bmin_c, bmax_r, bmax_c = self.bbox
        d.update(
            bbox_min_x=bmin_c,
            bbox_min_y=bmin_r,
            bbox_max_x=bmax_c,
            bbox_max_y=bmax_r,
        )
        return d


@dataclass
class FrameDiagnostics:
    """Per-frame record of what segmentation produced and what survived.

    Exists so that "the cell vanished" can always be attributed to either
    Cellpose or the post-filter, without rerunning anything.
    """

    frame: int
    raw_count: int = 0
    kept_count: int = 0
    removed_count: int = 0
    removed_extents: list[int] = field(default_factory=list)
    removed_areas: list[float] = field(default_factory=list)
    cellpose_message: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "raw_instances": self.raw_count,
            "kept_instances": self.kept_count,
            "removed_instances": self.removed_count,
            "removed_max_extent_px": max(self.removed_extents) if self.removed_extents else None,
            "removed_max_area_px": max(self.removed_areas) if self.removed_areas else None,
            "cellpose_message": self.cellpose_message,
        }


# --------------------------------------------------------------------------


def _orientation_to_xy_angle(props_orientation: float) -> float:
    """Keep scikit-image's convention and document it.

    ``regionprops.orientation`` is the angle between the 0th axis (rows, i.e.
    y) and the region's major axis, in ``[-pi/2, pi/2]``.  The corresponding
    unit vector in (x, y) is ``(sin(theta), cos(theta))``.  The sign is
    arbitrary -- a major axis has no head or tail -- so consumers must compare
    orientations with ``abs(dot)``.
    """
    return float(props_orientation)


def extract_detections(
    mask: np.ndarray,
    frame: int,
    intensity: np.ndarray | None = None,
) -> list[Detection]:
    """Measure every labelled instance in one frame."""
    if mask is None or mask.size == 0:
        return []
    mask = np.asarray(mask)
    if mask.max() <= 0:
        return []

    h, w = mask.shape[:2]
    out: list[Detection] = []
    for p in regionprops(mask.astype(np.int32), intensity_image=intensity):
        if p.label == 0:
            continue
        min_r, min_c, max_r, max_c = p.bbox
        bh, bw = max_r - min_r, max_c - min_c
        cy, cx = p.centroid
        # A single-pixel or perfectly symmetric region has no defined
        # eccentricity/orientation; scikit-image returns 0.0, which would be
        # silently read as "aligned with y". Flag it via eccentricity 0.
        try:
            ecc = float(p.eccentricity)
        except (ValueError, ZeroDivisionError):
            ecc = 0.0
        out.append(
            Detection(
                frame=int(frame),
                label=int(p.label),
                x=float(cx),
                y=float(cy),
                area_px=float(p.area),
                bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                extent_px=int(max(bh, bw)),
                eccentricity=ecc,
                orientation_rad=_orientation_to_xy_angle(p.orientation),
                major_axis_px=float(p.axis_major_length),
                minor_axis_px=float(p.axis_minor_length),
                solidity=float(p.solidity) if p.area > 2 else 1.0,
                touches_border=bool(
                    min_r == 0 or min_c == 0 or max_r >= h or max_c >= w
                ),
            )
        )
    return out


def detections_to_rows(
    detections: Iterable[Detection],
    *,
    pixel_size_um: float | None = None,
    frame_interval_min: float | None = None,
    source_frames: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in detections:
        row = d.to_row()
        row["source_frame"] = (
            source_frames[d.frame]
            if source_frames is not None and d.frame < len(source_frames)
            else None
        )
        row["elapsed_min"] = (
            d.frame * frame_interval_min if frame_interval_min else None
        )
        if pixel_size_um:
            row["x_um"] = d.x * pixel_size_um
            row["y_um"] = d.y * pixel_size_um
            row["area_um2"] = d.area_px * pixel_size_um * pixel_size_um
        rows.append(row)
    return rows
