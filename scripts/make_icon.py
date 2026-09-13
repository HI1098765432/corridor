"""Generate the application icon.

The mark is the subject matter: two channel walls with an elongated cell
squeezed between them, which is exactly what the app measures.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "corridor" / "assets"

TEAL = (15, 110, 99, 255)
WHITE = (255, 255, 255, 255)
WALL = (255, 255, 255, 110)


def draw(size: int) -> Image.Image:
    # Supersample, then downscale: gives clean edges at every icon size.
    scale = 8
    s = size * scale
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)

    radius = int(s * 0.22)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=radius, fill=TEAL)

    # Two channel walls.
    wall_w = max(1, int(s * 0.045))
    inset_y = int(s * 0.17)
    for x in (int(s * 0.33), int(s * 0.67)):
        d.rounded_rectangle(
            [x - wall_w // 2, inset_y, x + wall_w // 2, s - inset_y],
            radius=wall_w // 2, fill=WALL,
        )

    # The cell: an elongated capsule between the walls, mid-migration.
    cell_w = int(s * 0.155)
    cx = s // 2
    top = int(s * 0.245)
    bottom = int(s * 0.735)
    d.rounded_rectangle(
        [cx - cell_w // 2, top, cx + cell_w // 2, bottom],
        radius=cell_w // 2, fill=WHITE,
    )

    return image.resize((size, size), Image.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    images = [draw(n) for n in sizes]
    ico = ASSETS / "corridor.ico"
    images[-1].save(ico, format="ICO", sizes=[(n, n) for n in sizes])
    draw(512).save(ASSETS / "corridor.png")
    print("wrote", ico)
    print("wrote", ASSETS / "corridor.png")


if __name__ == "__main__":
    main()
