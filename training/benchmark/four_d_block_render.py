"""Render a movie as a 4D block (x, y, time-as-depth) with each tracked cell
drawn as a tube/mesh threading through it -- what the loading screen shows as the
hard code + model build the reconstruction. Isometric projection, numpy + PIL."""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src"); sys.path.insert(0, "build/traj")
import hard_movie_solve as S
from corridor.core.config import Scale, TrackingConfig
from corridor.core.geometry import detect_channels, assign_lanes
from corridor.engine.reconstruct import reconstruct_tracks

COLORS = [(255,90,90),(90,180,255),(90,255,130),(255,215,60),(225,110,255),(255,160,70),
          (120,255,230),(255,120,180),(165,220,95),(100,165,255),(255,255,255),(255,185,120)]


def norm(a):
    lo, hi = np.percentile(a, [1, 99]); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def render(folder, date, idxs, out, dx=34, dy=20, done_frac=1.0):
    imgs, masks = S.load(folder, date, list(idxs)); nf = len(masks); H, W = masks[0].shape
    stack = np.stack(imgs); geom = detect_channels(stack, None, S.dets(masks), pixel_size_um=0.467)
    ds = S.dets(masks); assign_lanes(ds, geom)
    tl = [tr for tr in reconstruct_tracks(ds, masks[0].shape, nf, Scale.from_values(0.467,20.0),
          TrackingConfig(), geometry=geom)[0] if len(tr.observations) >= 2]
    n_done = max(1, int(nf * done_frac))
    cw, ch = W + dx * nf + 40, H + dy * nf + 40
    canvas = Image.new("RGB", (cw, ch), (12, 14, 18))
    # draw slices back-to-front (far frame first)
    for t in range(nf - 1, -1, -1):
        ox, oy = 20 + dx * t, 20 + dy * (nf - 1 - t)
        g = (norm(imgs[t]) * 150).astype(np.uint8)
        done = t < n_done
        base = np.stack([g, g, g], -1).astype(np.float32)
        if not done:
            base *= 0.35   # not-yet-processed frames dim
        sl = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        sl.putalpha(150)
        canvas.paste(sl, (ox, oy), sl)
        d = ImageDraw.Draw(canvas)
        d.rectangle([ox, oy, ox + W, oy + H], outline=(60, 70, 85), width=1)
        if done:   # overlay the segmented cell mesh for processed frames
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            for ti, tr in enumerate(tl):
                o = [ob for ob in tr.observations if ob.frame == t]
                if o:
                    x, y = o[0].x, o[0].y; c = COLORS[ti % len(COLORS)]
                    od.ellipse([x-5, y-11, x+5, y+11], fill=(*c, 180))
            canvas.paste(ov, (ox, oy), ov)
    # draw the track tubes threading through the block (connect centroids across depth)
    d = ImageDraw.Draw(canvas, "RGBA")
    for ti, tr in enumerate(tl):
        c = COLORS[ti % len(COLORS)]; pts = []
        for ob in tr.observations:
            if ob.frame < n_done:
                ox, oy = 20 + dx * ob.frame, 20 + dy * (nf - 1 - ob.frame)
                pts.append((ox + ob.x, oy + ob.y))
        for k in range(1, len(pts)):
            d.line([pts[k-1], pts[k]], fill=(*c, 230), width=3)
    ImageDraw.Draw(canvas).text((20, ch - 18), f"{folder}/{date}  x-y-time block  "
                                f"{n_done}/{nf} frames meshed  {len(tl)} cell tubes",
                                fill=(200, 210, 220))
    canvas.save(out); print("wrote", out, canvas.size)


if __name__ == "__main__":
    base = r"C:\Users\ironb\AppData\Local\Temp\claude\C--Users-ironb-Projects-ConfinedMig\82918ebc-86d9-426f-86ba-add2826e6787\scratchpad"
    render("KK1", "122324", range(1, 13), base + r"\block_full.png", done_frac=1.0)
    render("KK1", "122324", range(1, 13), base + r"\block_mid.png", done_frac=0.5)
