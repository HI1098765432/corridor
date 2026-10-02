"""The provenance panel must describe *this* run, or admit it does not know.

This panel answers "what actually produced these numbers", so a row still
holding a value from the previously inspected analysis is not a cosmetic bug --
it is the panel asserting something false about a result. Every row is written
on every populate for exactly that reason.

The failure is invisible until two analyses are opened in the same session, and
invisible again unless one of them predates a field. Both cases are covered
here, because both happen the first time a user upgrades.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# A test must never open a window on the developer's desktop.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="the interface is not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from corridor.ui.screens.results import ResultsScreen  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def manifest(**segmentation):
    """A manifest with only the keys the provenance panel reads."""
    seg = {
        "model_path": "models/combi",
        "model_sha256": "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6",
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "min_extent_px": 20,
        "raw_instances_per_frame": [1, 1],
        "kept_instances_per_frame": [1, 1],
    }
    seg.update(segmentation)
    return {
        "input": {"name": "x.tif", "shape_tyx": [2, 10, 10]},
        "calibration": {},
        "segmentation": seg,
        "tracking": {},
        "confinement": {},
        "environment": {"cellpose": "3.1.1.3"},
        "results": {},
    }


def populate(screen, man):
    screen.analysis = SimpleNamespace(manifest=man)
    screen._populate_run()
    return {key: field._value.full_text() for key, field in screen.run_fields.items()}


def field_text(screen, key):
    return screen.run_fields[key]._value.full_text()


def test_a_modern_analysis_reports_what_ran(qt_app):
    screen = ResultsScreen()
    values = populate(
        screen,
        manifest(
            ensemble="thresholds",
            ensemble_passes=2,
            detections_from_fallback=3,
            normalisation_mode="local",
            normalize_percentiles=[1.0, 99.0],
            normalize_tile_px=128,
            normalize_sharpen_px=0,
        ),
    )
    assert "2 passes" in values["detection_effort"]
    assert "3 extra detection(s)" in values["detection_effort"]
    assert "Local contrast" in values["normalisation"]


def test_an_older_analysis_does_not_inherit_the_previous_ones_settings(qt_app):
    """The regression this file exists for.

    Open a v1.2.0 analysis, then a v1.1.0 one saved before these fields existed.
    The second must not still be showing the first one's detection effort.
    """
    screen = ResultsScreen()

    populate(
        screen,
        manifest(
            ensemble="max_recall",
            ensemble_passes=6,
            detections_from_fallback=9,
            normalisation_mode="stretch",
            normalize_percentiles=[3.0, 97.0],
            normalize_tile_px=0,
            normalize_sharpen_px=0,
        ),
    )
    assert "6 passes" in field_text(screen, "detection_effort")

    # Now an analysis saved before any of those keys existed.
    populate(screen, manifest())

    assert field_text(screen, "detection_effort") == "—"
    assert field_text(screen, "normalisation") == "—"
    # ...while the rows that *are* present still describe the new analysis.
    assert field_text(screen, "min_extent") == "20 px"


def test_an_older_analysis_shows_no_misleading_tooltip(qt_app):
    """An absent setting must not render as 'percentiles None, tile None px'."""
    screen = ResultsScreen()
    populate(screen, manifest())
    hint = screen.run_fields["normalisation"]._value.toolTip()
    assert "None" not in hint


def test_an_unknown_rung_name_is_shown_rather_than_swallowed(qt_app):
    """A manifest from a future version must not render as a blank row."""
    screen = ResultsScreen()
    populate(screen, manifest(ensemble="something_new", ensemble_passes=1))
    assert "something_new" in field_text(screen, "detection_effort")
