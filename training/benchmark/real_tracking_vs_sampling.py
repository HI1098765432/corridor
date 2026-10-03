"""The real-data result: tracking F1 vs time-sampling (motion/size) across ALL
7 real movies we have ground truth for -- the 5 newly-found labelled runs plus
the 2 eye-verified sample movies. Real cells, real masks, no synthetic. Shows
the architecture's valid regime: it reaches ~1.0 when a cell moves <~1 body
length per frame, and degrades when the movie is undersampled."""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")
sys.path.insert(0, "build/traj")
import real_labelled_battery as R                               # noqa: E402

OUT = "build/traj/ui_shots/00_real_tracking_vs_sampling.png"


def sample_movie(movie):
    masks = np.load(f"build/baseline_v1.3.0/{movie}/masks.npz")["masks"]
    return [masks[t] for t in range(masks.shape[0])]


def motion_over_size(stack):
    sizes = [m.sum() ** 0.5 for t in range(len(stack)) for m in R.instances(stack[t])]
    disp = []
    for t in range(len(stack) - 1):
        A = R.instances(stack[t]); B = R.instances(stack[t + 1])
        for ma in A:
            ca = R._c(ma)
            disp.append(min((np.hypot(ca[0] - R._c(mb)[0], ca[1] - R._c(mb)[1]) for mb in B), default=np.nan))
    if not disp or not sizes:
        return np.nan
    return float(np.nanmedian(disp) / np.nanmedian(sizes))


def f1_overlap(stack):
    truth = R.truth_tracks(stack)
    if not truth:
        return np.nan
    return R.score(R._track(R.build_dets(stack, 0, 0), stack, "overlap"), truth)[2]


def font(sz):
    for nm in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nm, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    pts = []
    for name, stack in R.load_runs():                    # 5 labelled runs
        pts.append((name.replace("KK1/", "").replace("KK2/", ""), motion_over_size(stack), f1_overlap(stack), "labelled"))
    for movie in ("052924_t1", "052924_t3_dual"):        # 2 eye-verified samples
        stack = sample_movie(movie)
        pts.append((movie.replace("052924_", ""), motion_over_size(stack), f1_overlap(stack), "sample"))
    pts = [p for p in pts if np.isfinite(p[1]) and np.isfinite(p[2])]
    for name, r, f, kind in sorted(pts, key=lambda p: p[1]):
        print(f"  {name:>16} motion/size {r:4.1f}x  overlap F1 {f:.2f}  ({kind})")

    W, H = 980, 620
    img = Image.new("RGB", (W, H), (248, 249, 250)); d = ImageDraw.Draw(img)
    d.text((28, 18), "Real-data tracking accuracy vs time-sampling (7 real movies, ground-truth masks)",
           fill=(15, 25, 35), font=font(20))
    d.text((28, 48), "Each point is a real movie. X = how far a cell moves per frame, in body-lengths. "
           "Y = trajectory F1 at perfect detections.", fill=(90, 100, 110), font=font(13))
    x0, y0, pw, ph = 90, 110, 840, 420
    xmax = 4.0
    # axes + gridlines
    for f in (0.25, 0.5, 0.75, 0.9, 1.0):
        y = y0 + ph - f * ph
        d.line([x0, y, x0 + pw, y], fill=(222, 226, 230)); d.text((x0 - 38, y - 7), f"{f:.2f}", fill=(150, 160, 170), font=font(12))
    for xv in (0, 1, 2, 3, 4):
        x = x0 + (xv / xmax) * pw
        d.line([x, y0, x, y0 + ph], fill=(236, 238, 240)); d.text((x - 4, y0 + ph + 6), str(xv), fill=(150, 160, 170), font=font(12))
    d.text((x0 + pw // 2 - 90, y0 + ph + 26), "cell motion per frame (body-lengths)", fill=(90, 100, 110), font=font(13))
    # the 1-body-length tracking limit
    xlim = x0 + (1.0 / xmax) * pw
    d.line([xlim, y0, xlim, y0 + ph], fill=(200, 120, 60), width=2)
    d.text((xlim + 6, y0 + 6), "~1 body-length: the tracking limit", fill=(176, 96, 42), font=font(12))
    for name, r, f, kind in pts:
        x = x0 + min(r, xmax) / xmax * pw
        y = y0 + ph - f * ph
        col = (29, 120, 90) if kind == "sample" else (41, 110, 180)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col)
        d.text((x + 9, y - 7), f"{name} ({f:.2f})", fill=(50, 60, 70), font=font(12))
    d.text((x0 + 8, y0 + 8), "green = eye-verified sample movies   blue = newly-found labelled runs",
           fill=(110, 120, 130), font=font(12))
    img.save(OUT); print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
