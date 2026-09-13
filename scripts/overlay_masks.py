"""Render image + mask-outline montages so segmentation can be judged by eye."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
DIAG = ROOT / "data" / "_diag"

PALETTE = [
    (255, 64, 64), (64, 200, 255), (120, 255, 120), (255, 200, 64),
    (220, 120, 255), (255, 140, 200), (150, 255, 220), (255, 255, 120),
]


def stretch(frame: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    f = frame.astype(np.float32)
    lo, hi = np.percentile(f, [lo_p, hi_p])
    return np.clip((f - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def outline(mask_2d: np.ndarray, label: int) -> np.ndarray:
    m = mask_2d == label
    if not m.any():
        return m
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


def render(stack_name: str, masks: np.ndarray, scale: int, cols: int, out: Path) -> None:
    stack = tifffile.imread(SAMPLES / stack_name)
    if stack.ndim == 2:
        stack = stack[None]
    T, H, W = stack.shape
    rows = (T + cols - 1) // cols
    pad = 6
    tile_w, tile_h = W * scale, H * scale
    canvas = Image.new("RGB", (cols * (tile_w + pad), rows * (tile_h + pad + 14)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for t in range(T):
        rgb = np.dstack([stretch(stack[t])] * 3)
        m = masks[t] if t < len(masks) else np.zeros((H, W), np.int32)
        for lab in np.unique(m):
            if lab == 0:
                continue
            col = PALETTE[(int(lab) - 1) % len(PALETTE)]
            rgb[outline(m, int(lab))] = col
        img = Image.fromarray(rgb).resize((tile_w, tile_h), Image.NEAREST)
        r, c = divmod(t, cols)
        x = c * (tile_w + pad)
        y = r * (tile_h + pad + 14) + 14
        canvas.paste(img, (x, y))
        n = int(m.max()) if m.size else 0
        draw.text((x + 2, y - 13), f"t={t}  n={n}", fill=(20, 20, 20))

    canvas.save(out)
    print("wrote", out, canvas.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="052924_t3_dual.tif")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = Path(args.stack).stem
    masks_path = Path(args.masks) if args.masks else DIAG / f"{stem}_rawmasks.npy"
    masks = np.load(masks_path)
    out = Path(args.out) if args.out else DIAG / f"{stem}_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(args.stack, masks, args.scale, args.cols, out)


if __name__ == "__main__":
    main()
