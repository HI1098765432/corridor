"""Open a saved analysis in Napari and screenshot every layer.

Run with the napari environment:
    .venv-napari/Scripts/python.exe scripts/napari_check.py <results dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/_runs/052924_t3_dual"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/_ui/napari.png"

    import napari
    from corridor.store.project import load_analysis
    from corridor.viz.napari_qc import open_saved_in_napari

    analysis = load_analysis(directory)
    print(f"analysis: {directory.name}")
    print(f"  tracks={len(analysis.summaries)} detections={len(analysis.detections)}")

    viewer = open_saved_in_napari(analysis, block=False)
    print(f"  viewer returned: {type(viewer).__name__}")
    print("  layers:")
    for layer in viewer.layers:
        print(f"    - {layer.name:16s} {type(layer).__name__:12s} shape={getattr(layer.data, 'shape', 'n/a')}")

    names = {layer.name for layer in viewer.layers}
    required = {"microscopy", "cell masks", "detections", "trajectories"}
    missing = required - names
    if missing:
        print(f"  MISSING LAYERS: {sorted(missing)}")

    # Show a frame that actually contains cells, otherwise the overlays have
    # nothing to draw and the screenshot proves nothing.
    frames = sorted({int(r["frame"]) for r in analysis.tracks if r.get("frame") is not None})
    busiest = max(frames, key=lambda f: len(analysis.rows_for_frame(f))) if frames else 0
    print(f"  showing frame {busiest} ({len(analysis.rows_for_frame(busiest))} tracked cells)")
    viewer.dims.set_point(0, busiest)
    # Select the image layer so the panel is not filled with shape-editing tools.
    viewer.layers.selection.active = viewer.layers["microscopy"]
    viewer.window.resize(1280, 860)
    from qtpy.QtWidgets import QApplication
    for _ in range(30):
        QApplication.processEvents()
    import time
    time.sleep(2)
    for _ in range(30):
        QApplication.processEvents()

    out.parent.mkdir(parents=True, exist_ok=True)
    viewer.screenshot(str(out), canvas_only=False)
    print(f"  saved {out}")
    viewer.close()
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
