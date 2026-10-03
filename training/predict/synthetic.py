"""Synthetic confined-migration movies with a known answer.

The prediction pipeline is only believable if it (a) finds a dependence of
future speed on shape when one is planted, and (b) does *not* find one when
there is none. This module writes Corridor-shaped result folders
(``masks.npz``, ``tracks.csv``, ``run.json``) for both cases so the whole path
-- folder reading, mask features, targets, grouped CV, permutation test -- is
exercised exactly as on real data, not a shortcut through it.

The two cases are built to differ in **one** thing only:

- *planted*: each step's speed is ``base + k * (aspect_ratio(t) - 4.75) +
  movie offset + noise``, so a cell that is more elongated *now* moves faster
  over the next frames.
- *null*: the same formula, but driven by a second, independent copy of the
  aspect-ratio process (same mean, persistence and spread), so speeds have the
  identical distribution and temporal persistence and simply no link to the
  shape that is drawn.

Both cases share the ingredients that make real data hard for a naive test:
speeds persist along a track (autocorrelation), cells differ from each other
(a per-cell mean), and movies differ (a per-movie offset). A permutation test
that shuffled single observations would ignore the first two, and the null case
is what shows whether the track-block test in ``evaluate`` does.

Targets are measured from mask *centroids*, so the outline is drawn
point-symmetric about the cell's position (even wobble modes, drawn once per
cell): the measured position is the true position by construction, and in the
null nothing about the drawn shape can reach a target. The first version of
this generator redrew a lopsided outline (odd and even modes) every frame,
whose centroid jitters by a fraction of the cell's length -- a shape-dependent
*measurement* error even when the motion ignores shape. Its one null dataset
gave the classifier p = 0.04 and 0.01 at +1 and +3 frames, so it is kept as
``lopsided_outline`` and the experiment measures the false-positive rate of
both outlines over independent datasets rather than guessing which explanation
(chance or the centroid) is right.

Geometry follows the supplied data: one cell per vertical lane, cells 9-23 px
wide and 22-87 px long, never touching the image border.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ASPECT_CENTRE = 4.75


@dataclass(frozen=True)
class SyntheticConfig:
    n_movies: int = 5
    tracks_per_movie: int = 6
    n_frames: int = 14
    #: True = speed follows the drawn aspect ratio; False = an independent copy.
    planted: bool = True
    pixel_size_um: float = 1.0
    frame_interval_min: float = 10.0
    speed_base_um_per_hr: float = 24.0
    speed_per_aspect_um_per_hr: float = 8.0
    speed_noise_um_per_hr: float = 6.0
    movie_offset_sd_um_per_hr: float = 3.0
    #: AR(1) coefficient of the aspect-ratio process, per frame.
    aspect_persistence: float = 0.8
    aspect_innovation_sd: float = 0.6
    #: Low-order radial wobble of the outline (fraction of the radius).
    outline_wobble: float = 0.03
    #: False: even wobble modes drawn once per cell, so the outline is
    #: point-symmetric and its centroid *is* the cell's position. True: the
    #: first version of this generator -- odd and even modes redrawn every
    #: frame, so the centroid jitters with cell length (module docstring).
    lopsided_outline: bool = False
    lane_pitch_px: int = 40
    height_px: int = 360
    n_experiments: int = 1
    seed: int = 0


def _aspect_process(rng: np.random.Generator, cfg: SyntheticConfig, n: int) -> np.ndarray:
    mean = rng.uniform(2.5, 7.0)
    rho, sd = cfg.aspect_persistence, cfg.aspect_innovation_sd
    stationary_sd = sd / math.sqrt(1.0 - rho**2)
    a = np.empty(n)
    a[0] = mean + rng.normal(0.0, stationary_sd)
    for t in range(1, n):
        a[t] = mean + rho * (a[t - 1] - mean) + rng.normal(0.0, sd)
    return np.clip(a, 1.5, 10.0)


def _wobble_terms(rng: np.random.Generator, wobble: float, modes: tuple[int, ...]
                  ) -> list[tuple[int, float, float]]:
    return [(k, rng.normal(0.0, wobble), rng.uniform(0, 2 * math.pi)) for k in modes]


def _cell_polygon(cy: float, cx: float, major: float, minor: float, tilt_rad: float,
                  terms: list[tuple[int, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    r = np.ones_like(theta)
    for k, amp, phase in terms:
        r += amp * np.cos(k * theta + phase)
    along = 0.5 * major * np.cos(theta) * r
    across = 0.5 * minor * np.sin(theta) * r
    rows = cy + along * math.cos(tilt_rad) - across * math.sin(tilt_rad)
    cols = cx + along * math.sin(tilt_rad) + across * math.cos(tilt_rad)
    return rows, cols


def make_movie(cfg: SyntheticConfig, movie_index: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, list[dict], dict]:
    """One movie: (masks T,Y,X int32, track rows, truth)."""
    from skimage.draw import polygon

    n_t, n_cells = cfg.n_frames, cfg.tracks_per_movie
    h, w = cfg.height_px, cfg.lane_pitch_px * n_cells
    masks = np.zeros((n_t, h, w), dtype=np.int32)
    dt_hr = cfg.frame_interval_min / 60.0
    movie_offset = rng.normal(0.0, cfg.movie_offset_sd_um_per_hr)
    rows: list[dict] = []
    truth = {"aspect": [], "speed_um_per_hr": []}
    for c in range(n_cells):
        aspect = _aspect_process(rng, cfg, n_t)
        driver = aspect if cfg.planted else _aspect_process(rng, cfg, n_t)
        speed = np.maximum(
            0.0,
            cfg.speed_base_um_per_hr
            + cfg.speed_per_aspect_um_per_hr * (driver - ASPECT_CENTRE)
            + movie_offset
            + rng.normal(0.0, cfg.speed_noise_um_per_hr, n_t),
        )
        area_mean = rng.uniform(250.0, 600.0)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        cx = cfg.lane_pitch_px * (c + 0.5)
        y = 70.0 if direction > 0 else h - 70.0
        # Even modes are invariant under theta -> theta + pi: point symmetry.
        # (Not drawn at all for the lopsided outline, so that variant replays
        # the first generator's random stream exactly.)
        cell_terms = None if cfg.lopsided_outline else _wobble_terms(rng, cfg.outline_wobble, (2, 4))
        for t in range(n_t):
            area = area_mean * math.exp(rng.normal(0.0, 0.05))
            major = math.sqrt(4.0 * area * aspect[t] / math.pi)
            minor = major / aspect[t]
            # Keep the whole cell inside the image: a cell that would leave
            # simply stops at the margin (rare at these speeds).
            margin = 0.5 * major + 6.0
            y = float(np.clip(y, margin, h - margin))
            tilt = math.radians(rng.normal(0.0, 3.0))
            terms = (_wobble_terms(rng, cfg.outline_wobble, (2, 3, 4, 5))
                     if cfg.lopsided_outline else cell_terms)
            rr, cc = _cell_polygon(y, cx, major, minor, tilt, terms)
            pr, pc = polygon(rr, cc, shape=(h, w))
            masks[t, pr, pc] = c + 1
            ys, xs = pr.mean(), pc.mean()
            rows.append({
                "track_id": c + 1,
                "frame": t,
                "elapsed_min": t * cfg.frame_interval_min,
                "x_px": float(xs), "y_px": float(ys),
                "x_um": float(xs) * cfg.pixel_size_um, "y_um": float(ys) * cfg.pixel_size_um,
                "area_px": float(len(pr)),
                "det_label": c + 1,
                "detection_source": "primary",
            })
            # Step taken *after* this frame, at this frame's speed.
            y += direction * speed[t] * dt_hr / cfg.pixel_size_um
        truth["aspect"].append(aspect.tolist())
        truth["speed_um_per_hr"].append(speed.tolist())
    return masks, rows, truth


def write_dataset(cfg: SyntheticConfig, out_dir: Path) -> list[Path]:
    """Write ``cfg.n_movies`` result folders under ``out_dir``; return their paths."""
    import pandas as pd

    out_dir = Path(out_dir)
    rng = np.random.default_rng(cfg.seed)
    folders = []
    for m in range(cfg.n_movies):
        masks, rows, truth = make_movie(cfg, m, rng)
        kind = ("planted" if cfg.planted else "null") + ("_lopsided" if cfg.lopsided_outline else "")
        folder = out_dir / f"synthetic_{kind}_{m:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "masks.npz", masks=masks)
        pd.DataFrame(rows).to_csv(folder / "tracks.csv", index=False)
        run = {
            "synthetic": asdict(cfg),
            "experiment": f"synthetic-experiment-{m % max(cfg.n_experiments, 1)}",
            "calibration": {
                "pixel_size_um": cfg.pixel_size_um,
                "frame_interval_min": cfg.frame_interval_min,
            },
            "input": {"name": folder.name, "shape_tyx": list(masks.shape)},
            "truth": truth,
        }
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        folders.append(folder)
    return folders
