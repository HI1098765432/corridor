"""How wide is the band of contrast in which this model can see a cell at all?

The stated failure mode is that a missed cell leaves a *flat* probability field
(-2.7 to +0.2) rather than one that merely dips below threshold. That rules out
thresholds as a fix, but it does not say why the field is flat. This measures
the why, and it does so by changing exactly one thing.

The trick is to alter a cell's **contrast** without altering anything else. An
image is decomposed into a background and the perturbation the cell makes on it:

    B     = median filter along the channel axis   (background varies slowly
                                                    along a straight channel)
    delta = I - B                                  (everything the cell did)
    I(s)  = B + s * delta   inside a soft support around the labelled cell

At s = 1 this reconstructs the original image. At s = 0.5 the cell is half as
far above its background, in the same place, the same shape, with the same noise
and the same device around it. Nothing has changed except how visible it is.

Running the model across s gives a response curve, and the question it answers is
sharp: **is the model contrast-invariant?** If it is, recall stays flat and the
domain gap must be something else. If the curve has a knee, then the width of
the usable band is a measured property of the model, the labelled data sits
somewhere inside it, and any image whose cells fall outside it is invisible for
a reason no threshold can reach.

It is run on images the model was *trained on*, deliberately. A drop there cannot
be blamed on unfamiliar cells, a different device, or a different day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
MODELS = {
    "combi": TRAIN / "KK1KK2_combiModel" / "models" / "cyto2_phase_microfluidic_KK1KK2_combi",
    "KK1": TRAIN / "KK1Model" / "models" / "cyto2_phase_microfluidic_d10",
    "KK2": TRAIN / "KK2Model" / "models" / "cyto2_phase_microfluidic_d10",
}
SCALES = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.8, 2.5)


def load(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    return image.astype(np.float32)


def truth_of(path: Path) -> np.ndarray:
    seg = path.with_name(path.name.replace(".tif", "_seg.npy"))
    return np.asarray(np.load(seg, allow_pickle=True).item()["masks"]).astype(np.int32)


def background_of(image: np.ndarray) -> np.ndarray:
    """The image without its cells: a long median filter along the channel axis."""
    from scipy.ndimage import median_filter

    return median_filter(image, size=(1, 41), mode="nearest")


def support_of(masks: np.ndarray) -> np.ndarray:
    """A soft window around the cells, so rescaling leaves no hard seam.

    A hard mask edge would introduce a step that is itself a strong gradient,
    and the model would then be responding to an artefact of the experiment
    rather than to the cell's contrast.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    grown = binary_dilation(masks > 0, iterations=8).astype(np.float32)
    return np.clip(gaussian_filter(grown, sigma=3.0) * 1.6, 0.0, 1.0)


def rescaled(image: np.ndarray, background: np.ndarray, support: np.ndarray,
             scale: float) -> np.ndarray:
    delta = image - background
    return image + (scale - 1.0) * support * delta


def contrast_of(image: np.ndarray, masks: np.ndarray) -> float:
    """Mean cell-to-background separation, in units of the image's own range."""
    lo, hi = np.percentile(image, [1, 99])
    span = max(float(hi - lo), 1e-6)
    inside = masks > 0
    if not inside.any():
        return 0.0
    background = float(np.median(image[~inside]))
    return abs(float(image[inside].mean()) - background) / span


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def recall_at(model, image: np.ndarray, masks: np.ndarray,
              window: tuple[float, float] | None = None) -> tuple[int, int]:
    """Recall on one image, optionally with the normalisation window frozen.

    Freezing matters more than it sounds. Cellpose percentile-normalises every
    image it is given, and these cells sit in the tail of that distribution:
    they cover 1-2% of the frame, and 8-21% of their pixels fall outside the
    [p1, p99] window. So rescaling a cell MOVES the window, which rescales the
    whole image including the background -- and narrowing the window is itself
    a measured intervention that lifts held-out F1 from 0.482 to 0.653.

    Left free, a contrast sweep therefore varies two things at once, with the
    second one pushing the opposite way at low contrast. Passing the window
    computed from the native image holds it still, so the cell's amplitude is
    the only thing that changes.
    """
    # cellpose 3.1.1.3 accepts normalize={"lowhigh": (lo, hi)} but it does not
    # work: passing the image's OWN percentiles returns zero masks where
    # normalize=True returns four. Rather than depend on a parameter that is
    # broken in this version, the window is frozen by doing the normalisation
    # here and telling cellpose not to do it again. Verified equivalent on the
    # native image: cellpose-normalised and pre-normalised both give 4 masks.
    if window is not None:
        lo, hi = window
        image = (image - lo) / max(hi - lo, 1e-6)
    predicted = np.asarray(
        model.eval(image, channels=[0, 0], diameter=None,
                   cellprob_threshold=0.0, flow_threshold=0.4,
                   normalize=(window is None))[0]
    ).astype(np.int32)
    found = 0
    labels = [int(v) for v in np.unique(masks) if v]
    for label in labels:
        region = masks == label
        best = 0.0
        for other in np.unique(predicted):
            if other:
                best = max(best, iou(region, predicted == other))
        if best >= 0.5:
            found += 1
    return found, len(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="combi", choices=sorted(MODELS))
    ap.add_argument("--group", default="KK1")
    ap.add_argument("--limit", type=int, default=8, help="images to sweep")
    ap.add_argument("--free-window", action="store_true",
                    help="let cellpose renormalise per image (the confounded version)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(4)
    from cellpose import models as cp

    model = cp.CellposeModel(pretrained_model=str(MODELS[args.model]), gpu=False)

    files = [
        p for p in sorted((TRAIN / args.group).glob("*.tif"))
        if p.with_name(p.name.replace(".tif", "_seg.npy")).exists()
    ]
    # Images with the most cells first: more instances per Cellpose call.
    files.sort(key=lambda p: -len([v for v in np.unique(truth_of(p)) if v]))
    files = [p for p in files if len([v for v in np.unique(truth_of(p)) if v]) > 0]
    files = files[: args.limit]

    prepared = []
    for path in files:
        image = load(path)
        masks = truth_of(path)
        lo, hi = np.percentile(image, [1, 99])
        prepared.append((path, image, masks, background_of(image), support_of(masks),
                         (float(lo), float(hi))))

    print(f"model {args.model} on {len(prepared)} of its {args.group} images "
          f"({sum(len([v for v in np.unique(m) if v]) for _, _, m, _, _, _ in prepared)} instances)")
    print("Only the cells' contrast changes. Same place, same shape, same noise.\n")
    print(f"{'scale':>6} {'contrast |c|':>13} {'recall':>8} {'found/total':>12}")

    rows = []
    for scale in SCALES:
        found = total = 0
        contrasts = []
        for _, image, masks, background, support, window in prepared:
            altered = rescaled(image, background, support, scale)
            contrasts.append(contrast_of(altered, masks))
            f, t = recall_at(model, altered, masks,
                             None if args.free_window else window)
            found += f
            total += t
        recall = found / total if total else 0.0
        contrast = float(np.median(contrasts))
        rows.append({
            "scale": scale,
            "contrast": round(contrast, 4),
            "recall": round(recall, 4),
            "found": found,
            "total": total,
        })
        print(f"{scale:6.2f} {contrast:13.3f} {recall:8.3f} {found:7d}/{total:<5d}", flush=True)

    usable = [r for r in rows if r["recall"] >= 0.8]
    report = {
        "normalisation_window": "free (confounded)" if args.free_window else "frozen at native",
        "model": args.model,
        "group": args.group,
        "n_images": len(prepared),
        "curve": rows,
        "usable_contrast_band": (
            [usable[0]["contrast"], usable[-1]["contrast"]] if usable else None
        ),
    }
    print()
    if usable:
        lo, hi = usable[0]["contrast"], usable[-1]["contrast"]
        print(f"recall >= 0.8 only for contrast in [{lo:.2f}, {hi:.2f}] "
              f"-- a band {hi / max(lo, 1e-6):.1f}x wide")
        print("Outside it the cells are still there, unchanged in shape and position.")
        report["band_width_ratio"] = round(hi / max(lo, 1e-6), 2)
    else:
        print("no scale reached recall 0.8; the sweep may be too coarse")

    out = Path(args.out) if args.out else (
        ROOT / "docs" / (f"contrast_response_{args.model}_on_{args.group}"
                         f"{'_freewindow' if args.free_window else ''}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
