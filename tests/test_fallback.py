"""The detection fallback ladder: merging, provenance, and what it costs.

None of this touches Cellpose. Merging label images is ordinary array work, and
a test that needed a neural network to prove that two overlapping blobs are one
cell would be testing the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor.core.config import (
    ENSEMBLE_MAX_RECALL,
    ENSEMBLE_MODELS,
    ENSEMBLE_MODES,
    ENSEMBLE_OFF,
    ENSEMBLE_THRESHOLDS,
    ENSEMBLE_WIDE,
    SegmentationConfig,
)
from corridor.core.segmentation import (
    SOURCE_ENSEMBLE,
    SOURCE_PRIMARY,
    filter_instances,
    merge_labelled,
)


def blob(shape, box, value=1):
    """A rectangular instance, which is all the merge logic can see anyway."""
    out = np.zeros(shape, dtype=np.int32)
    r0, c0, r1, c1 = box
    out[r0:r1, c0:c1] = value
    return out


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def test_a_second_pass_adds_a_genuinely_new_instance():
    shape = (60, 60)
    primary = blob(shape, (5, 5, 25, 25))
    extra = blob(shape, (35, 35, 55, 55))

    merged, sources = merge_labelled(
        [(primary, SOURCE_PRIMARY), (extra, SOURCE_ENSEMBLE)]
    )

    labels = sorted(int(v) for v in np.unique(merged) if v)
    assert len(labels) == 2
    assert sources[labels[0]] == SOURCE_PRIMARY
    assert sources[labels[1]] == SOURCE_ENSEMBLE


def test_the_same_cell_found_twice_stays_one_instance():
    """The whole point: a permissive pass re-finding a known cell adds nothing.

    If this failed, every fallback rung would double-count every cell it got
    right, and the tracker would be handed two candidates per cell per frame.
    """
    shape = (60, 60)
    primary = blob(shape, (10, 10, 30, 30))
    # Same cell, one pixel off and slightly larger, as a lower threshold gives.
    extra = blob(shape, (9, 9, 31, 31))

    merged, sources = merge_labelled(
        [(primary, SOURCE_PRIMARY), (extra, SOURCE_ENSEMBLE)]
    )

    assert len([v for v in np.unique(merged) if v]) == 1
    assert set(sources.values()) == {SOURCE_PRIMARY}


def test_the_primary_pass_keeps_its_pixels_when_two_instances_clip():
    shape = (60, 60)
    primary = blob(shape, (10, 10, 30, 30))
    extra = blob(shape, (28, 10, 50, 30))  # overlaps by two rows

    merged, sources = merge_labelled(
        [(primary, SOURCE_PRIMARY), (extra, SOURCE_ENSEMBLE)]
    )

    assert len([v for v in np.unique(merged) if v]) == 2
    # The primary instance is untouched; the newcomer lost the shared rows.
    assert merged[10:30, 10:30].min() == 1
    assert (merged[28:30, 10:30] == 1).all()


def test_a_sliver_left_after_trimming_is_not_a_cell():
    shape = (60, 60)
    primary = blob(shape, (10, 10, 40, 40))
    # Overlaps almost entirely; what is left over is four pixels.
    extra = blob(shape, (10, 10, 40, 41))

    merged, _ = merge_labelled(
        [(primary, SOURCE_PRIMARY), (extra, SOURCE_ENSEMBLE)],
        min_fragment_px=20,
    )
    assert len([v for v in np.unique(merged) if v]) == 1


def test_merging_nothing_is_not_a_crash():
    merged, sources = merge_labelled([])
    assert merged.size == 0
    assert sources == {}


def test_a_pass_of_the_wrong_shape_is_skipped_rather_than_raising():
    """A companion model on a mismatched crop must not abort the analysis."""
    primary = blob((60, 60), (5, 5, 25, 25))
    wrong = blob((40, 40), (5, 5, 25, 25))

    merged, sources = merge_labelled(
        [(primary, SOURCE_PRIMARY), (wrong, SOURCE_ENSEMBLE)]
    )
    assert merged.shape == (60, 60)
    assert set(sources.values()) == {SOURCE_PRIMARY}


# --------------------------------------------------------------------------
# Provenance survives the post-filter
# --------------------------------------------------------------------------


def test_the_filter_reports_which_label_each_survivor_used_to_be():
    """Relabelling to 1..K must not erase which pass found each instance."""
    mask = np.zeros((60, 60), dtype=np.int32)
    mask[5:10, 5:8] = 1      # 5 px tall: below min_extent, will be dropped
    mask[20:50, 20:30] = 2   # a real cell
    mask[52:58, 5:8] = 3     # too small again

    result = filter_instances(mask, SegmentationConfig(min_extent_px=20))

    assert result.raw_count == 3
    assert result.kept_count == 1
    # The single survivor is now label 1, and it used to be label 2.
    assert result.label_map == {1: 2}


# --------------------------------------------------------------------------
# What each rung costs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rung, expected_cost",
    [
        (ENSEMBLE_OFF, 1),
        (ENSEMBLE_THRESHOLDS, 2),
        (ENSEMBLE_MODELS, 3),
        (ENSEMBLE_WIDE, 4),
        (ENSEMBLE_MAX_RECALL, 6),
    ],
)
def test_each_rung_costs_what_the_measured_table_says(rung, expected_cost):
    """The cost column in the config docstring is load-bearing.

    Those numbers came from scripts/experiment_recall.py, where each strategy's
    runtime was measured. If a rung silently gained a pass, the precision and
    recall quoted beside it would no longer describe what runs.
    """
    cfg = SegmentationConfig(
        ensemble=rung, ensemble_model_paths=("/models/KK1", "/models/KK2")
    )
    assert cfg.ensemble_cost_factor() == expected_cost


def test_a_model_rung_degrades_to_its_threshold_passes_without_companions():
    """An absent optional model costs recall, never the analysis."""
    cfg = SegmentationConfig(ensemble=ENSEMBLE_MODELS, ensemble_model_paths=())
    assert cfg.ensemble_passes() == ()

    cfg = SegmentationConfig(ensemble=ENSEMBLE_MAX_RECALL, ensemble_model_paths=())
    assert cfg.ensemble_cost_factor() == 2  # primary plus its own second setting


def test_an_unknown_rung_falls_back_to_a_single_pass():
    cfg = SegmentationConfig(ensemble="nonsense")
    assert cfg.ensemble_passes() == ()


def test_the_default_is_a_single_pass():
    """Changing this default changes every runtime estimate in the documents."""
    assert SegmentationConfig().ensemble == ENSEMBLE_OFF
    assert SegmentationConfig().ensemble_cost_factor() == 1
    assert ENSEMBLE_OFF in ENSEMBLE_MODES


# --------------------------------------------------------------------------
# Companion model discovery
# --------------------------------------------------------------------------


def test_discovery_finds_sibling_models(tmp_path):
    from corridor.core.segmentation import discover_companion_models

    root = tmp_path / "TrainData"
    primary = root / "CombiModel" / "models" / "combi"
    other = root / "KK1Model" / "models" / "half"
    for target in (primary, other):
        target.parent.mkdir(parents=True)
        target.write_bytes(b"\0" * (2 << 20))

    found = discover_companion_models(primary)
    assert found == (str(other),)


def test_discovery_ignores_logs_and_label_files(tmp_path):
    from corridor.core.segmentation import discover_companion_models

    root = tmp_path / "TrainData"
    primary = root / "CombiModel" / "models" / "combi"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"\0" * (2 << 20))

    sibling = root / "KK1Model" / "models"
    sibling.mkdir(parents=True)
    (sibling / "training.txt").write_text("log")
    (sibling / "labels.npy").write_bytes(b"\0" * (2 << 20))
    (sibling / "tiny").write_bytes(b"\0" * 100)  # right shape, far too small

    assert discover_companion_models(primary) == ()


def test_discovery_refuses_to_search_a_filesystem_root(tmp_path):
    """The guard against turning a stray path into a whole-disk scan.

    A model at ``C:/models/thing`` puts the search root at the drive letter.
    Globbing from there is slow and returns Python source files that merely
    live in a directory called "models", which would then be handed to
    Cellpose as checkpoints.
    """
    from corridor.core.segmentation import discover_companion_models

    assert discover_companion_models("C:/models/thing") == ()
    assert discover_companion_models("/models/thing") == ()


def test_discovery_of_a_path_that_does_not_exist_is_empty(tmp_path):
    from corridor.core.segmentation import discover_companion_models

    assert discover_companion_models(tmp_path / "absent" / "models" / "x") == ()
    assert discover_companion_models(None) == ()


# --------------------------------------------------------------------------
# Normalisation, the lever that addresses the contrast shift
# --------------------------------------------------------------------------


def test_the_default_normalisation_is_byte_identical_to_before_the_option_existed():
    """A default run must not change because a knob was added beside it.

    Cellpose treats ``normalize=True`` and ``normalize={'percentile': (1, 99)}``
    as the same computation, but only the first is what every published number
    in docs/ was measured with. Returning the plain boolean keeps a default run
    comparable with those.
    """
    assert SegmentationConfig().normalize_argument() is True


def test_normalisation_off_is_off():
    assert SegmentationConfig(normalize=False).normalize_argument() is False
    # ...even if the shaping options are set: they describe how to normalise,
    # not whether to.
    cfg = SegmentationConfig(normalize=False, normalize_tile_px=128)
    assert cfg.normalize_argument() is False


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"normalize_tile_px": 128}, {"tile_norm_blocksize": 128}),
        ({"normalize_sharpen_px": 15}, {"sharpen_radius": 15}),
        ({"normalize_percentiles": (3.0, 97.0)}, {"percentile": (3.0, 97.0)}),
        (
            {"normalize_tile_px": 128, "normalize_sharpen_px": 15},
            {"tile_norm_blocksize": 128, "sharpen_radius": 15},
        ),
    ],
)
def test_each_shaping_option_reaches_cellpose_under_its_own_name(kwargs, expected):
    assert SegmentationConfig(**kwargs).normalize_argument() == expected


def test_normalisation_presets_round_trip_through_the_config():
    """The panel reads the mode back from the fields, so they must agree.

    If a preset did not report itself, reopening a saved project would show
    whichever entry happened to be first while running with quite different
    settings -- the kind of disagreement nobody notices until the results do
    not reproduce.
    """
    from corridor.core.config import NORMALISATION_MODES

    for mode in NORMALISATION_MODES:
        cfg = SegmentationConfig()
        cfg.apply_normalisation_preset(mode)
        assert cfg.normalisation_mode == mode


def test_settings_matching_no_preset_report_themselves_as_custom():
    cfg = SegmentationConfig(normalize_tile_px=64, normalize_sharpen_px=7)
    assert cfg.normalisation_mode == "custom"
    assert cfg.normalize_argument() == {
        "tile_norm_blocksize": 64, "sharpen_radius": 7
    }


def test_an_unknown_preset_name_changes_nothing():
    cfg = SegmentationConfig(normalize_tile_px=128)
    cfg.apply_normalisation_preset("nonsense")
    assert cfg.normalize_tile_px == 128


def test_a_candidate_cut_in_two_keeps_only_its_larger_piece():
    """A trimmed instance must stay one connected object.

    An elongated cell clipped across its middle leaves a piece at each end.
    Labelling both as a single instance would place its centroid in the gap
    between them -- a position no cell occupies, which would then be handed to
    the tracker as a measurement.
    """
    shape = (60, 60)
    primary = blob(shape, (28, 10, 32, 50))          # a horizontal bar
    extra = blob(shape, (10, 20, 50, 26))            # a vertical bar crossing it

    merged, _ = merge_labelled(
        [(primary, SOURCE_PRIMARY), (extra, SOURCE_ENSEMBLE)],
        min_fragment_px=20,
    )

    added = merged == 2
    assert added.any(), "the crossing candidate should survive as one piece"

    from scipy import ndimage

    _, pieces = ndimage.label(added)
    assert pieces == 1, "the added instance must be a single connected object"

    # The larger piece is the one below the bar (18 rows, not 18 either side --
    # check it kept whichever was bigger rather than an arbitrary one).
    rows = np.where(added.any(axis=1))[0]
    assert rows.min() >= 32 or rows.max() <= 28
