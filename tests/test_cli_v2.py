"""The 2.0 command line: options, refusals and exit codes.

The exit codes are part of the interface a batch script relies on: 3 means
the validated model is unavailable, 4 that the file's axis order must be given
with --axes.  Neither may collapse into the generic 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corridor import _version, cli
from corridor.core import pipeline
from corridor.core.config import CHANNEL_CONSTRAINT_OFF
from corridor.core.imaging import AmbiguousAxes
from corridor.core.model_registry import MODEL_UNAVAILABLE_MESSAGE, ModelUnavailable

from test_pipeline_v2 import CELLS, synthetic_movie


def parse(*argv: str):
    return cli.config_from_args(cli.build_parser().parse_args(["in.tif", "-o", "out", *argv]))


def test_the_new_options_reach_the_configuration():
    config = parse(
        "--axes", "tzyx", "--channel-index", "2", "--labels", "labels.tif",
        "--reference-point", "10,20.5", "--no-channel-constraint",
        "--position-sigma", "0.7", "--z-step", "2.5",
    )
    assert config.import_.axes == "TZYX"
    assert config.import_.channel_index == 2
    assert Path(config.import_.labels_path) == Path("labels.tif")
    assert config.measurement.reference_point_px == (10.0, 20.5)
    assert config.tracking.channel_constraint == CHANNEL_CONSTRAINT_OFF
    assert config.tracking.enforce_channel_identity is False
    assert config.tracking.position_sigma_um == 0.7
    assert config.calibration.z_step_um == 2.5


def test_defaults_leave_the_validated_settings_alone():
    config = parse()
    assert config.tracking.channel_constraint == "auto"
    assert config.measurement.reference_point_px is None
    assert config.import_.labels_path is None
    assert config.segmentation.model_path is None


def test_a_3d_reference_point_is_accepted_and_a_malformed_one_refused(capsys):
    assert parse("--reference-point", "1,2,3").measurement.reference_point_px == (1.0, 2.0, 3.0)
    with pytest.raises(SystemExit) as info:
        cli.build_parser().parse_args(["in.tif", "--reference-point", "1;2"])
    assert info.value.code == 2
    assert "X,Y" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag", [["--axis", "vertical"], ["--axis-angle", "90"], ["--sigma-along", "3"],
             ["--sigma-across", "0.8"]],
)
def test_the_axis_options_are_refused_with_a_reason(flag, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["in.tif", "-o", "out", *flag])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "removed in Corridor 2.0" in err and flag[0] in err
    with pytest.raises(ValueError, match="migration axis"):
        parse(*flag)


def test_the_removed_options_are_not_advertised():
    text = cli.build_parser().format_help()
    for flag in cli.REMOVED_OPTIONS:
        assert f"{flag} " not in text and f"{flag}\n" not in text, flag
    for flag in ("--axes", "--channel-index", "--labels", "--reference-point",
                 "--no-channel-constraint", "--position-sigma"):
        assert flag in text


def test_version_prints_the_single_source_version(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip().endswith(_version.__version__)


def test_a_missing_model_exits_3_with_the_contract_message(monkeypatch, capsys):
    def unavailable(config, progress=None, **kwargs):
        raise ModelUnavailable(MODEL_UNAVAILABLE_MESSAGE, [(Path("C:/models/x"), "missing")])

    monkeypatch.setattr(pipeline, "run_analysis", unavailable)
    assert cli.main(["in.tif", "-o", "out", "--quiet"]) == cli.EXIT_MODEL_UNAVAILABLE == 3
    err = capsys.readouterr().err
    assert MODEL_UNAVAILABLE_MESSAGE in err and "missing" in err


def test_ambiguous_axes_exit_4_listing_the_choices(monkeypatch, capsys):
    def ambiguous(config, progress=None, **kwargs):
        raise AmbiguousAxes(("TYX", "ZYX", "CYX"), "This file does not say which axis is time.")

    monkeypatch.setattr(pipeline, "run_analysis", ambiguous)
    assert cli.main(["in.tif", "-o", "out", "--quiet"]) == cli.EXIT_AMBIGUOUS_AXES == 4
    err = capsys.readouterr().err
    assert "TYX, ZYX, CYX" in err and "--axes" in err


def test_a_missing_input_exits_2(tmp_path, capsys):
    assert cli.main([str(tmp_path / "absent.tif"), "-o", str(tmp_path / "o"), "--quiet"]) == 2


def test_a_labelled_run_from_the_command_line(tmp_path, capsys):
    movie, labels, _ = synthetic_movie(tmp_path)
    out = tmp_path / "cli_out"
    code = cli.main([str(movie), "-o", str(out), "--labels", str(labels), "--quiet"])
    assert code == 0
    text = capsys.readouterr().out
    assert "imported from labels.tif" in text and "no model ran" in text
    assert "lane(s) from channel_ridges" in text
    assert f"{len(CELLS)} tracks" in text
    assert (out / pipeline.F_MANIFEST).exists() and (out / pipeline.F_MSD).exists()


def test_the_model_line_names_id_version_and_checksum():
    from corridor.core import model_registry
    from corridor.core.model_registry import ResolvedModel

    spec = model_registry.production_spec("2D")
    result = type("R", (), {})()
    result.model = ResolvedModel(spec=spec, path=Path("m"), sha256=spec.sha256)
    result.segmentation = None
    line = cli.describe_model(result)
    assert spec.model_id in line and spec.model_version in line and spec.sha256[:12] in line
    assert "OVERRIDE" not in line
    result.model = ResolvedModel(spec=spec, path=Path("m"), sha256="0" * 64, developer_override=True)
    assert "DEVELOPER OVERRIDE" in cli.describe_model(result)
