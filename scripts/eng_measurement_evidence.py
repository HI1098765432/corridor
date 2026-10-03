"""Reproducible evidence for Bot 6 (measurement), on synthetic ground truth.

This is the inspectable record the engine contract asks for: the numbers behind
each decision, produced by the bots on digitised shapes whose answer is known
by construction, and reproducible by rerunning.  It writes
``build/eng/measurement/evidence.json`` under the real project tree (data/ and
build/ live there, not in an isolated worktree).

No model, no torch, no cellpose -- numpy, scipy and scikit-image only.

Run (from a checkout with the venv):

    PYTHONPATH=src OMP_NUM_THREADS=2 python scripts/eng_measurement_evidence.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from corridor.core.detections import extract_detections, extract_detections_3d
from corridor.engine.object4d import Calibration4D, tube_from_detections
from corridor.engine.surface_delta import surface_delta
from corridor.engine.uncertainty import displacement_floor

OUT = Path(__file__).resolve().parents[1] / "build" / "eng" / "measurement" / "evidence.json"


def sphere(radius_vox, shape=(48, 64, 64), centre=(24, 32, 32)):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    inside = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= radius_vox**2
    return inside.astype(np.int32)


def ellipse(shape, cx, cy, a, b):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0


def volume_growth_case() -> dict:
    spacing = (0.5, 0.5, 0.5)
    calib = Calibration4D(pixel_size_um=0.5, z_step_um=0.5, frame_interval_min=10.0)
    radii = [6.0, 7.0, 8.0]
    dets = [extract_detections_3d(sphere(r), frame=f, spacing_zyx_um=spacing)[0]
            for f, r in enumerate(radii)]
    tube = tube_from_detections(1, dets, calib)
    frames = []
    for p, r in zip(tube.points, radii):
        true_um3 = 4.0 / 3.0 * math.pi * (r * 0.5) ** 3
        frames.append({
            "frame": p.frame,
            "radius_um": r * 0.5,
            "true_volume_um3": round(true_um3, 4),
            "measured_volume_um3": round(p.volume_um3, 4),
            "error_pct": round(100.0 * (p.volume_um3 / true_um3 - 1.0), 3),
            "dV_dt_um3_per_min": None if p.dV_dt_um3_per_min is None else round(p.dV_dt_um3_per_min, 4),
        })
    return {"description": "growing sphere, isotropic 0.5um, dt=10min", "frames": frames,
            "tolerance_pct": 3.0, "max_abs_error_pct": round(max(abs(f["error_pct"]) for f in frames), 3)}


def translation_cases() -> dict:
    # 2-D integer shift
    shape = (120, 160)
    m_t = ellipse(shape, 60, 60, 30, 10)
    m_tp1 = np.zeros_like(m_t); m_tp1[:, 6:] = m_t[:, :-6]
    sd_int = surface_delta(m_t, m_tp1)
    # 2-D sub-pixel via intensity ROI
    img_t = ndi.gaussian_filter(m_t.astype(float), 3.0)
    img_tp1 = ndi.shift(img_t, (0.0, 3.4), order=3)
    m_sp = ndi.shift(m_t.astype(float), (0.0, 3.4), order=1) >= 0.5
    sd_sp = surface_delta(m_t, m_sp, image_t=img_t, image_tp1=img_tp1)
    return {
        "integer_2d": {"true_shift_px_xy": [6.0, 0.0],
                       "recovered_px_xy": [round(v, 4) for v in sd_int.translation_px],
                       "residual_frac_of_area": round((sd_int.extension_px + sd_int.retraction_px) / m_t.sum(), 5)},
        "subpixel_2d_intensity": {"true_shift_px_xy": [3.4, 0.0],
                                  "recovered_px_xy": [round(v, 4) for v in sd_sp.translation_px],
                                  "residual_frac_of_area": round((sd_sp.extension_px + sd_sp.retraction_px) / m_t.sum(), 5)},
    }


def extension_case() -> dict:
    shape = (120, 160)
    m_t = ellipse(shape, 80, 60, 30, 10)
    m_tp1 = m_t.copy(); m_tp1[55:66, 110:122] = True
    planted = int((m_tp1 & ~m_t).sum())
    front = surface_delta(m_t, m_tp1, motion_direction=(1.0, 0.0))
    rear = surface_delta(m_t, m_tp1, motion_direction=(-1.0, 0.0))
    return {
        "planted_extension_px": planted,
        "measured_extension_px": front.extension_px,
        "measured_retraction_px": front.retraction_px,
        "dominant_location_motion_+x": front.dominant_extension_location,
        "dominant_location_motion_-x": rear.dominant_extension_location,
        "by_location_+x": front.extension_by_location_px,
    }


def uncertainty_case() -> dict:
    floor = displacement_floor(area_px=800.0, noise_to_signal=0.2,
                               registration_sigma_px=0.1, pixel_size_um=0.5)
    sub = floor.assess(0.05)
    supra = floor.assess(1.0)
    return {
        "floor_px": round(floor.floor_px, 5),
        "floor_um": round(floor.floor_um, 5),
        "centroid_sigma_px": round(floor.centroid_sigma_px, 5),
        "sub_floor_move_0.05px": sub.to_dict(),
        "supra_floor_move_1.0px": supra.to_dict(),
    }


def main() -> None:
    evidence = {
        "bot": "6 measurement (object4d + surface_delta + uncertainty)",
        "principle": "MEASURE never assert; numbers below are produced on synthetic ground truth",
        "no_model_no_torch": True,
        "volume_growth": volume_growth_case(),
        "translation": translation_cases(),
        "one_sided_extension": extension_case(),
        "precision_floor": uncertainty_case(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
