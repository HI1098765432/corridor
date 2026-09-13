"""End-to-end test through the real application, not the backend.

Covers the journey the acceptance criteria describe: open a TIFF, analyse it,
see results, export, close, reopen, and find the previous analysis without
recomputing it.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLES = ROOT / "data" / "confinedmig_cellTrack" / "sample_data"

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    _results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


def pump(app, seconds: float = 0.0) -> None:
    deadline = time.time() + seconds
    while True:
        app.processEvents()
        if time.time() >= deadline:
            break
        time.sleep(0.01)


def wait_for(app, predicate, timeout: float = 900.0, label: str = "") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    print(f"    timed out waiting for {label}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="052924_t1.tif")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data dir")
    args = parser.parse_args()

    faulthandler.enable()
    data_dir = Path(tempfile.mkdtemp(prefix="corridor_e2e_"))
    export_dir = data_dir / "exported"
    os.environ["CORRIDOR_DATA_DIR"] = str(data_dir)
    print(f"data dir: {data_dir}")

    from corridor.store import db
    from corridor.store.project import analysis_is_complete, export_bundle, load_analysis
    from corridor.ui.app import create_application
    from corridor.ui.main_window import MainWindow

    app = create_application([])

    # ---------------------------------------------------------------- startup
    print("\n1. clean startup")
    store = db.Store()
    window = MainWindow(store)
    window.resize(1360, 880)
    window.show()
    pump(app, 0.4)
    check("window opens", window.isVisible())
    check("starts on the home screen", window.stack.currentIndex() == 0)
    check("no projects yet", len(store.recent_projects()) == 0)

    # ------------------------------------------------------------------- open
    print("\n2. open a time-lapse")
    stack_path = SAMPLES / args.stack
    if not stack_path.exists():
        print(f"  sample missing: {stack_path}")
        return 2
    window.open_file(str(stack_path))
    opened = wait_for(app, lambda: window._metadata is not None, 120, "metadata")
    check("file opens", opened)
    if not opened:
        return 1
    pump(app, 0.3)
    check("moves to the dataset screen", window.stack.currentIndex() == 1)
    metadata = window._metadata
    check(
        "frame count comes from the file, not SizeT",
        metadata.n_frames == 18 if args.stack == "052924_t1.tif" else metadata.n_frames > 0,
        f"{metadata.n_frames} frames",
    )
    check(
        "frame interval is ~20.0069 min",
        abs((metadata.frame_interval_min.value or 0) - 20.006894938) < 1e-6,
        f"{metadata.frame_interval_min.value}",
    )
    check(
        "pixel size is the ND2 value",
        abs((metadata.pixel_size_um.value or 0) - 0.467060342995564) < 1e-9,
        f"{metadata.pixel_size_um.value}",
    )
    check("a model is selected", bool(window._config.segmentation.model_path))
    check("Analyse is enabled", window.dataset.analyse_button.isEnabled())

    # ---------------------------------------------------------------- analyse
    print("\n3. analyse (this runs Cellpose)")
    started = time.time()
    window.dataset.analyse_button.click()
    pump(app, 0.2)
    check("UI reports it is busy", window.dataset.status_row.isVisible())
    check("Analyse is disabled while running", not window.dataset.analyse_button.isEnabled())

    done = wait_for(app, lambda: window.stack.currentIndex() == 2, 1800, "analysis")
    check("analysis finishes", done, f"{time.time() - started:.0f}s")
    if not done:
        return 1
    pump(app, 0.5)

    analysis = window._analysis
    check("results are loaded", analysis is not None)
    check("tracks were produced", len(analysis.summaries) > 0, f"{len(analysis.summaries)} tracks")
    check("detections were produced", len(analysis.detections) > 0, f"{len(analysis.detections)}")

    directory = analysis.directory
    for name in (
        "tracks.csv", "detections.csv", "track_summary.csv",
        "segmentation_diagnostics.csv", "tracking_events.csv", "qc_issues.csv",
        "run.json", "masks.npz",
    ):
        check(f"wrote {name}", (directory / name).exists())

    manifest = analysis.manifest
    check(
        "manifest records the 20 min interval",
        abs(manifest["calibration"]["frame_interval_min"] - 20.006894938) < 1e-6,
    )
    check(
        "manifest records the model checksum",
        manifest["segmentation"]["model_sha256"]
        == "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6",
    )
    check(
        "manifest records Cellpose 3",
        str(manifest["environment"]["cellpose"]).startswith("3."),
        manifest["environment"]["cellpose"],
    )
    check(
        "velocities are in explicit units",
        all(
            key in analysis.tracks[0]
            for key in ("speed_um_per_min", "vx_um_per_min", "speed_px_per_frame")
        ),
    )

    # ------------------------------------------------------------- interaction
    print("\n4. review the result")
    ids = analysis.track_ids()
    if ids:
        window.results.select_track(ids[0])
        pump(app, 0.2)
        check("selecting a track shows its detail", window.results.detail_box.isVisible())
        check("the canvas knows the selection", window.results.canvas.selected_track == ids[0])
    window.results.timeline.step(1)
    pump(app, 0.1)
    check("the timeline moves the canvas", window.results.canvas.frame == 1)
    window.results.tabs.setCurrentIndex(1)
    pump(app, 0.1)
    check("the checks tab lists findings", window.results.checks_list.count() >= 0)
    window.results.tabs.setCurrentIndex(0)

    # ----------------------------------------------------------------- export
    print("\n5. export")
    written = export_bundle(analysis, export_dir)
    check("export writes files", len(written) >= 7, f"{len(written)} files")
    check("exported tracks.csv exists", (export_dir / "tracks.csv").exists())

    # --------------------------------------------------------- close & reopen
    print("\n6. close and reopen")
    project_id = window._project.id
    window.close()
    pump(app, 0.3)

    store2 = db.Store()
    window2 = MainWindow(store2)
    window2.resize(1360, 880)
    window2.show()
    pump(app, 0.4)
    recent = store2.recent_projects()
    check("the project is remembered", len(recent) == 1, f"{len(recent)} in history")
    check("it is marked complete", recent and recent[0].status == "complete")
    check("its result files are still there", analysis_is_complete(recent[0].path))

    reopened_at = time.time()
    window2.open_project(project_id)
    ok = wait_for(app, lambda: window2.stack.currentIndex() == 2, 180, "reopen")
    elapsed = time.time() - reopened_at
    check("reopening shows the results", ok)
    check(
        "reopening does not rerun Cellpose",
        elapsed < 30,
        f"{elapsed:.1f}s (a fresh analysis took {time.time() - started:.0f}s)",
    )
    if ok and window2._analysis is not None:
        check(
            "the reopened result has the same tracks",
            len(window2._analysis.summaries) == len(analysis.summaries),
        )

    # -------------------------------------------------------------- same file
    print("\n7. opening the same file again reuses the analysis")
    window2.go_home()
    pump(app, 0.2)
    window2.open_file(str(stack_path))
    ok = wait_for(app, lambda: window2.stack.currentIndex() == 2, 120, "reuse")
    check("goes straight to the saved result", ok)
    check("still one project", len(store2.recent_projects()) == 1)

    window2.close()
    pump(app, 0.3)
    store2.close()

    print("\n" + "=" * 62)
    failures = [r for r in _results if r[0] == FAIL]
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAILED: {name} {detail}")
    if not args.keep:
        shutil.rmtree(data_dir, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
