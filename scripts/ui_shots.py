"""Drive the application through each state and save a screenshot of it.

Widget.grab() is used rather than a desktop screen capture so the images are
exact, unaffected by display scaling, and reproducible.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "_ui"
SAMPLES = ROOT / "data" / "confinedmig_cellTrack" / "sample_data"
RUNS = ROOT / "data" / "_runs"


def shoot(widget, name: str, size: tuple[int, int] | None = None) -> None:
    from PySide6.QtWidgets import QApplication

    if size:
        widget.resize(*size)
    QApplication.processEvents()
    for _ in range(6):
        QApplication.processEvents()
    pixmap = widget.grab()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    pixmap.save(str(path))
    print(f"  {path.name}  {pixmap.width()}x{pixmap.height()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma separated state names")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    def want(name: str) -> bool:
        return not wanted or name in wanted

    # Keep this run's projects out of the user's real database.
    tmp = Path(tempfile.mkdtemp(prefix="corridor_ui_"))
    os.environ["CORRIDOR_DATA_DIR"] = str(tmp)

    from corridor.store import db
    from corridor.store.project import load_analysis
    from corridor.ui.app import create_application
    from corridor.ui.dialogs import AboutDialog, ErrorDialog, SettingsDialog
    from corridor.ui.main_window import MainWindow

    app = create_application([])
    store = db.Store()
    window = MainWindow(store)
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()

    print("states:")

    # 1. Empty first launch.
    if want("home_empty"):
        shoot(window, "home_empty")

    # 2. A dataset open, metadata interpreted.
    from corridor.core.imaging import load_stack, read_metadata

    sample = SAMPLES / "052924_t1.tif"
    if sample.exists():
        window.open_file(str(sample))
        for _ in range(400):
            app.processEvents()
            if window._metadata is not None:
                break
            import time

            time.sleep(0.01)
        app.processEvents()
        if want("dataset"):
            shoot(window, "dataset")
        if want("dataset_advanced"):
            window.dataset.advanced_button.setChecked(True)
            shoot(window, "dataset_advanced")
            window.dataset.advanced_button.setChecked(False)
        if want("dataset_busy"):
            window.dataset.set_busy(True, "Segmenting")
            window.dataset.set_stage("Segmenting", "18 frames")
            window.dataset.set_progress(7, 18)
            shoot(window, "dataset_busy")
            window.dataset.set_busy(False)

    # 3. Results, loaded from a completed run.
    for run_name, label in (
        ("052924_t1", "results_t1"),
        ("052924_t3_dual", "results_t3"),
        ("052924_t2_empty", "results_empty"),
    ):
        directory = RUNS / run_name
        if not (want(label) or want("results")) or not (directory / "run.json").exists():
            continue
        analysis = load_analysis(directory)
        source = analysis.source_path
        if source is None or not source.exists():
            continue
        from corridor.ui.workers import import_config_for, read_metadata_for

        # With the axis order the run recorded, as the window itself reads it.
        metadata = read_metadata_for(source, import_config_for(analysis.manifest))
        stack = load_stack(source, metadata)
        window.results.load(analysis, stack)
        window._analysis = analysis
        window.stack.setCurrentIndex(2)
        app.processEvents()
        window.results.canvas.fit_to_view()
        shoot(window, label)

        if label == "results_t1":
            if want("results_selected"):
                ids = analysis.track_ids()
                if ids:
                    window.results.select_track(ids[0])
                    window.results.canvas.set_frame(8)
                    shoot(window, "results_selected")
            if want("results_checks"):
                window.results.tabs.setCurrentIndex(1)
                shoot(window, "results_checks")
            if want("results_run"):
                window.results.tabs.setCurrentIndex(2)
                shoot(window, "results_run")
                window.results.tabs.setCurrentIndex(0)
            if want("results_lanes"):
                if window.results.toggle_lanes.isEnabled():
                    window.results.toggle_lanes.setChecked(True)
                    shoot(window, "results_lanes")
                    window.results.toggle_lanes.setChecked(False)
            if want("results_reference"):
                height, width = stack.shape[-2:]
                # Shown, not saved: a screenshot must not change the run it shows.
                window.results._set_reference((width / 2.0, height / 3.0), persist=False)
                shoot(window, "results_reference")
                window.results._set_reference(None, persist=False)
            if want("results_export_menu"):
                menu = window.results.export_menu
                menu.adjustSize()
                shoot(menu, "results_export_menu")

    # 3b. The orthogonal viewer, on a synthetic TZYX volume. There is no 3-D
    # data in the supplied set (contract §0), so the state is drawn from a
    # volume with two known boxes rather than from a run.
    if want("ortho_3d"):
        import numpy as np

        from corridor.ui.widgets.ortho_viewer import OrthoViewer

        volume = np.full((3, 12, 96, 128), 20.0, dtype=np.float32)
        labels = np.zeros(volume.shape, dtype=np.int32)
        for label_id, (z, y, x) in ((1, (3, 30, 40)), (2, (8, 60, 90))):
            box = (slice(None), slice(z - 2, z + 3), slice(y - 6, y + 7), slice(x - 10, x + 11))
            volume[box] = 180.0
            labels[box] = label_id
        rows = [
            {"track_id": 1, "frame": t, "x_px": 40.0, "y_px": 30.0, "z": 3.0} for t in range(3)
        ] + [
            {"track_id": 2, "frame": t, "x_px": 90.0, "y_px": 60.0, "z": 8.0} for t in range(3)
        ]
        viewer = OrthoViewer()
        viewer.rows_for_frame = lambda f: [r for r in rows if r["frame"] == f]
        viewer.set_volume(volume, labels, anisotropy=4.0)
        viewer.select_track(1)
        shoot(viewer, "ortho_3d", size=(1000, 760))

    # 4. Home with recent projects.
    if want("home_recent"):
        window.go_home()
        app.processEvents()
        shoot(window, "home_recent")

    # 5. Narrow window, to check nothing clips.
    if want("home_narrow"):
        shoot(window, "home_narrow", size=(960, 640))
        window.resize(args.width, args.height)

    # 6. Dialogs.
    if want("settings"):
        dialog = SettingsDialog(store, window)
        dialog.show()
        app.processEvents()
        shoot(dialog, "settings")
        dialog.close()
    if want("about"):
        dialog = AboutDialog(window)
        dialog.show()
        app.processEvents()
        shoot(dialog, "about")
        dialog.close()
    if want("error"):
        dialog = ErrorDialog(
            "This file could not be read as a time-lapse.\n\n"
            "It reports axes 'CYX', which means its three planes are colour "
            "channels rather than time points.",
            "Traceback (most recent call last):\n  ...\n"
            "corridor.core.imaging.UnsupportedStackError: ...",
            window,
        )
        dialog.show()
        app.processEvents()
        shoot(dialog, "error")
        dialog.close()

    window.close()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
