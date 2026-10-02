"""The segmentation service loads only the verified model, and says which.

Cellpose itself is replaced by a recording fake installed as the ``cellpose``
package, so these tests never import torch and never run a network: what is
under test is which file reaches ``CellposeModel`` and with what arguments,
not what the network does with it.  The argument names the fake receives are
checked against the *installed* Cellpose's own signatures, read from its
source with ``ast`` (importing it would import torch).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from corridor import resources
from corridor.core import model_registry as mr
from corridor.core import segmentation as seg
from corridor.core.config import (
    ENSEMBLE_MAX_RECALL,
    ENSEMBLE_MODELS,
    ENSEMBLE_THRESHOLDS,
    Scale,
    SegmentationConfig,
)
from corridor.core.imaging import UnsupportedStackError
from corridor.core.model_registry import (
    ENV_DEVELOPER,
    ENV_DEVELOPER_MODEL,
    MODEL_UNAVAILABLE_MESSAGE,
    ModelUnavailable,
)
from corridor.core.segmentation import (
    PROVENANCE_MODEL,
    CellposeUnavailableError,
    SegmentationService,
)

CONTRACT_TEXT = (
    "The validated Corridor segmentation model is missing or does not match the "
    "expected checksum."
)
GOOD = b"pretend these are the validated weights"


# --------------------------------------------------------------------------
# Fixtures: a registry of one fake model, and a recording fake Cellpose
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name in (ENV_DEVELOPER, ENV_DEVELOPER_MODEL, "CORRIDOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    mr._HASH_CACHE.clear()
    yield
    mr._HASH_CACHE.clear()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """One registered 2-D model whose only candidate location is ``bundle/``."""
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "test_model",
                        "model_version": "1.0.0",
                        "architecture": "cellpose3-cyto2-resnet",
                        "cellpose_version": ">=3,<4",
                        "filename": "weights",
                        "sha256": hashlib.sha256(GOOD).hexdigest(),
                        "training_dataset_version": "synthetic",
                        "dimensions": ["2D"],
                        "pixel_size_range_um": None,
                        "validated_app_version": "2.0.0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mr, "registry_path", lambda: path)
    weights = tmp_path / "bundle" / "weights"
    weights.parent.mkdir()
    monkeypatch.setattr(mr, "candidate_paths", lambda spec: [weights])

    class Registry:
        model_file = weights
        digest = hashlib.sha256(GOOD).hexdigest()

        @staticmethod
        def install(content: bytes | None = GOOD) -> Path:
            if content is None:
                weights.unlink(missing_ok=True)
            else:
                weights.write_bytes(content)
            mr._HASH_CACHE.clear()
            return weights

    return Registry


def _blob_2d(shape) -> np.ndarray:
    """One elongated cell, long enough (30 px) to survive the 20 px filter."""
    out = np.zeros(shape, np.int32)
    h, w = shape
    r0, c0 = h // 2 - 15, w // 2 - 3
    out[max(r0, 0):r0 + 30, max(c0, 0):c0 + 6] = 1
    return out


def _ellipsoid(shape, centre, radii_vox) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    inside = sum(((g - c) / r) ** 2 for g, c, r in zip((zz, yy, xx), centre, radii_vox)) <= 1.0
    return inside.astype(np.int32)


@pytest.fixture
def cellpose(monkeypatch):
    """A fake ``cellpose`` package recording every model built and every eval."""
    calls: dict[str, list] = {"init": [], "eval": []}
    behaviour = {"loaded_path": None}

    class FakeCellposeModel:
        def __init__(self, gpu=False, pretrained_model=False, model_type=None, **kwargs):
            calls["init"].append(
                {"gpu": gpu, "pretrained_model": pretrained_model, "model_type": model_type, **kwargs}
            )
            # Real Cellpose 3.1 stores what it loaded here -- which is
            # cyto3's path when it silently fell back.
            self.pretrained_model = behaviour["loaded_path"] or pretrained_model

        def eval(self, x, **kwargs):
            calls["eval"].append({"shape": np.shape(x), **kwargs})
            x = np.asarray(x)
            if kwargs.get("do_3D"):
                z, h, w = x.shape
                masks = _ellipsoid(x.shape, (z // 2, h // 2, w // 2), (max(z // 3, 1), h // 3, w // 3))
            else:
                masks = _blob_2d(x.shape)
            return masks, None, None

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = FakeCellposeModel
    package = types.ModuleType("cellpose")
    package.models = models
    package.version = "fake"
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)

    class Fake:
        init = calls["init"]
        evals = calls["eval"]
        cls = FakeCellposeModel

        @staticmethod
        def fall_back_to(path: str) -> None:
            behaviour["loaded_path"] = path

    return Fake


def _stack(n=2, h=64, w=48) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(1000, 20, size=(n, h, w)).astype(np.float32)


def _assert_only_verified(cellpose, path: Path) -> None:
    """Every model Cellpose was asked to build is the verified absolute path."""
    assert cellpose.init, "no model was loaded"
    for call in cellpose.init:
        assert call["model_type"] is None, "a model_type (bare name) reached Cellpose"
        assert Path(call["pretrained_model"]).is_absolute()
        assert Path(call["pretrained_model"]) == path.resolve()


# --------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------


def test_production_loads_only_the_verified_absolute_path(registry, cellpose):
    weights = registry.install()
    out = SegmentationService(SegmentationConfig()).run_stack(_stack())

    _assert_only_verified(cellpose, weights)
    assert len(cellpose.init) == 1  # loaded once for the whole stack
    assert out.model_path == str(weights.resolve())
    assert out.model_sha256 == registry.digest
    assert out.model.spec.model_id == "test_model"
    assert out.provenance == PROVENANCE_MODEL and not out.developer_override
    assert out.model_manifest() == {
        "model_id": "test_model",
        "model_version": "1.0.0",
        "architecture": "cellpose3-cyto2-resnet",
        "sha256": registry.digest,
        "cellpose_version": ">=3,<4",
        "training_dataset_version": "synthetic",
        "developer_override": False,
    }
    assert out.masks.shape == (2, 64, 48) and len(out.detections) == 2


@pytest.mark.parametrize("content", [b"some other model", None], ids=["mismatch", "missing"])
def test_a_missing_or_mismatching_model_is_never_loaded(registry, cellpose, content):
    registry.install(content)
    with pytest.raises(ModelUnavailable) as info:
        SegmentationService(SegmentationConfig()).run_stack(_stack())
    assert str(info.value).startswith(CONTRACT_TEXT)
    assert str(registry.model_file) in str(info.value)
    assert cellpose.init == [] and cellpose.evals == []


def test_a_file_swapped_after_resolution_is_refused_before_loading(registry, cellpose):
    """The registry hashes at resolution; the service hashes again at load."""
    weights = registry.install()
    resolved = mr.resolve_model("2D")
    weights.write_bytes(b"swapped in afterwards")
    with pytest.raises(ModelUnavailable, match="checksum mismatch"):
        SegmentationService(SegmentationConfig(), model=resolved).run_stack(_stack())
    assert cellpose.init == []


@pytest.mark.parametrize(
    "legacy",
    [
        {"model_path": "IMPOSTOR", "use_custom_model": True},
        {"model_path": None, "use_custom_model": False, "builtin_model": "cyto3"},
        {"model_path": "IMPOSTOR", "use_custom_model": False, "builtin_model": "nuclei"},
        {"ensemble_model_paths": ("IMPOSTOR",), "ensemble": ENSEMBLE_MODELS},
    ],
    ids=["model_path", "builtin-cyto3", "builtin-nuclei", "companions"],
)
def test_legacy_model_settings_do_not_change_the_model(registry, cellpose, tmp_path, legacy):
    weights = registry.install()
    impostor = tmp_path / "impostor_weights"
    impostor.write_bytes(b"an unvalidated checkpoint")

    def real(value):
        if value == "IMPOSTOR":
            return str(impostor)
        if isinstance(value, tuple):
            return tuple(real(v) for v in value)
        return value

    fields = {key: real(value) for key, value in legacy.items()}
    out = SegmentationService(SegmentationConfig(**fields)).run_stack(_stack())

    _assert_only_verified(cellpose, weights)
    assert len(cellpose.init) == 1
    assert out.model_path == str(weights.resolve())
    assert any("ignored" in note for note in out.notes)


def test_the_cellpose_version_is_checked_before_loading(registry, cellpose, monkeypatch):
    """CP4's refusal of CP3 weights is an assert, gone under ``python -O``."""
    registry.install()
    monkeypatch.setattr(seg, "cellpose_version", lambda: "4.0.1")
    with pytest.raises(CellposeUnavailableError, match="4.0.1"):
        SegmentationService(SegmentationConfig()).run_stack(_stack())
    assert cellpose.init == []


def test_the_version_check_does_not_import_cellpose():
    """Reading the version must not cost a torch import (seconds, hundreds of MB).

    Run in a fresh interpreter: in this one torch or the fake cellpose may
    already be imported, and the check would pass without testing anything.
    """
    import os
    import subprocess

    src = str(Path(seg.__file__).resolve().parents[2])
    code = (
        "import sys; from corridor.core import segmentation as s; v = s.cellpose_version(); "
        "assert v and v != 'unavailable', v; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'cellpose' not in sys.modules, 'cellpose imported'; print(v)"
    )
    env = dict(os.environ, PYTHONPATH=src, OMP_NUM_THREADS="1")
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.returncode == 0, done.stderr


def test_a_hand_built_model_claiming_to_be_production_is_refused(registry, cellpose, tmp_path):
    """``ResolvedModel`` is public; its sha256 is the caller's claim, not the registry's.

    The adversarial review loaded an impostor this way, recorded as the
    production model with developer_override False and no note.
    """
    registry.install()
    spec = mr.load_registry()[0]
    impostor = tmp_path / "impostor"
    impostor.write_bytes(b"an unvalidated checkpoint")
    forged = mr.ResolvedModel(
        spec=spec, path=impostor.resolve(), sha256=mr.sha256_file(impostor)
    )
    with pytest.raises(ModelUnavailable, match="checksum mismatch") as info:
        SegmentationService(SegmentationConfig(), model=forged).run_stack(_stack())
    assert str(info.value).startswith(CONTRACT_TEXT)
    assert cellpose.init == [] and cellpose.evals == []


def test_a_hand_built_spec_that_matches_the_impostor_is_refused(registry, cellpose, tmp_path):
    """A forged spec carrying the impostor's own checksum is not a registry entry."""
    import dataclasses

    registry.install()
    impostor = tmp_path / "impostor"
    impostor.write_bytes(b"an unvalidated checkpoint")
    digest = mr.sha256_file(impostor)
    spec = dataclasses.replace(mr.load_registry()[0], sha256=digest)
    forged = mr.ResolvedModel(spec=spec, path=impostor.resolve(), sha256=digest)
    with pytest.raises(ModelUnavailable, match="not an entry of the model registry"):
        SegmentationService(SegmentationConfig(), model=forged).run_stack(_stack())
    assert cellpose.init == []


def test_an_explicit_registry_resolution_still_loads(registry, cellpose):
    weights = registry.install()
    resolved = mr.resolve_model("2D")
    out = SegmentationService(SegmentationConfig(), model=resolved).run_stack(_stack(n=1))
    _assert_only_verified(cellpose, weights)
    assert not out.developer_override and out.model_sha256 == registry.digest


def test_a_silent_cellpose_fallback_to_another_model_is_caught(registry, cellpose, tmp_path):
    """Cellpose 3.1 swaps in cyto3 with only a warning when a path is wrong."""
    registry.install()
    cellpose.fall_back_to(str(tmp_path / ".cellpose" / "models" / "cyto3_cp3"))
    with pytest.raises(ModelUnavailable, match="instead of the verified file"):
        SegmentationService(SegmentationConfig()).run_stack(_stack())


def test_a_developer_override_runs_and_is_recorded(registry, cellpose, tmp_path, monkeypatch):
    registry.install()
    dev = tmp_path / "dev" / "experimental"
    dev.parent.mkdir()
    dev.write_bytes(b"an experimental checkpoint")
    monkeypatch.setenv(ENV_DEVELOPER, "1")
    monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(dev))
    out = SegmentationService(SegmentationConfig()).run_stack(_stack())
    _assert_only_verified(cellpose, dev)
    assert out.developer_override is True
    assert out.model_manifest()["developer_override"] is True
    assert out.model_manifest()["model_id"] == mr.DEVELOPER_OVERRIDE_ID


def test_a_research_model_is_marked_as_an_override(registry, cellpose, tmp_path):
    research = tmp_path / "round4"
    research.write_bytes(b"research weights")
    resolved = mr.research_model(research, label="round4")
    out = SegmentationService(SegmentationConfig(), model=resolved).run_stack(_stack(n=1))
    _assert_only_verified(cellpose, research)
    assert out.developer_override is True


# --------------------------------------------------------------------------
# The fallback ladder: only rungs that re-run the same model survive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rung", [ENSEMBLE_MODELS, ENSEMBLE_MAX_RECALL])
def test_a_removed_rung_runs_as_off_and_says_so(registry, cellpose, rung):
    registry.install()
    out = SegmentationService(SegmentationConfig(ensemble=rung)).run_stack(_stack(n=3))
    assert out.passes_per_frame == 1
    assert len(cellpose.evals) == 3  # one pass per frame
    assert any(rung in note and "removed" in note for note in out.notes)


def test_the_threshold_rung_reuses_the_one_validated_model(registry, cellpose):
    weights = registry.install()
    out = SegmentationService(SegmentationConfig(ensemble=ENSEMBLE_THRESHOLDS)).run_stack(_stack(n=2))
    assert out.passes_per_frame == 2
    assert len(cellpose.evals) == 4
    assert len(cellpose.init) == 1
    _assert_only_verified(cellpose, weights)
    assert {(e["cellprob_threshold"], e["flow_threshold"]) for e in cellpose.evals} == {
        (0.0, 0.4), (-2.0, 0.4)
    }


def test_companion_discovery_is_gone(tmp_path):
    root = tmp_path / "TrainData"
    primary = root / "CombiModel" / "models" / "combi"
    other = root / "KK1Model" / "models" / "half"
    for target in (primary, other):
        target.parent.mkdir(parents=True)
        target.write_bytes(b"\0" * (2 << 20))
    assert seg.discover_companion_models(primary) == ()


# --------------------------------------------------------------------------
# 3-D
# --------------------------------------------------------------------------


def _stack_3d(t=1, z=9, h=24, w=30) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(1000, 20, size=(t, z, h, w)).astype(np.float32)


CALIBRATED_3D = Scale.from_values(0.5, 10.0, z_step_um=1.5)


def test_3d_with_the_2d_only_production_model_is_refused(registry, cellpose):
    registry.install()
    with pytest.raises(ModelUnavailable, match="No 3D-validated segmentation model") as info:
        SegmentationService(SegmentationConfig(), scale=CALIBRATED_3D).run_stack(_stack_3d())
    # The likeliest way here by mistake is a time-lapse labelled as slices.
    assert "--axes TYX" in str(info.value)
    assert cellpose.init == []


def test_3d_with_an_explicit_2d_model_is_refused(registry, cellpose):
    registry.install()
    resolved = mr.resolve_model("2D")
    service = SegmentationService(SegmentationConfig(), model=resolved, scale=CALIBRATED_3D)
    with pytest.raises(ModelUnavailable, match="validated for 2D only"):
        service.run_stack(_stack_3d())
    assert cellpose.init == []


def test_3d_never_runs_on_a_2d_model_already_loaded(registry, cellpose):
    registry.install()
    service = SegmentationService(SegmentationConfig(), scale=CALIBRATED_3D)
    service.run_stack(_stack())
    with pytest.raises(ModelUnavailable, match="3-D"):
        service.run_stack(_stack_3d())


@pytest.fixture
def dev_override(tmp_path, monkeypatch):
    dev = tmp_path / "dev" / "experimental_3d"
    dev.parent.mkdir()
    dev.write_bytes(b"an experimental 3-D checkpoint")
    monkeypatch.setenv(ENV_DEVELOPER, "1")
    monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(dev))
    return dev


def test_3d_under_a_developer_override_uses_cellpose_3d_arguments(registry, cellpose, dev_override):
    registry.install()
    stack = _stack_3d(t=2)
    out = SegmentationService(SegmentationConfig(), scale=CALIBRATED_3D).run_stack(stack)

    _assert_only_verified(cellpose, dev_override)
    assert len(cellpose.evals) == 2
    for call in cellpose.evals:
        assert call["shape"] == stack.shape[1:]
        assert call["do_3D"] is True and call["z_axis"] == 0
        assert call["anisotropy"] == pytest.approx(3.0)  # 1.5 um / 0.5 um
        assert call["channels"] == [0, 0]
    assert out.dimensionality == "3D" and out.developer_override
    assert out.masks.shape == stack.shape and out.raw_masks.shape == stack.shape
    assert len(out.detections) == 2
    det = out.detections[0]
    assert det.z is not None and det.volume_vox > 0
    assert det.volume_um3 == pytest.approx(det.volume_vox * 0.5 * 0.5 * 1.5)


def test_3d_without_a_z_step_is_refused_not_assumed_isotropic(registry, cellpose, dev_override):
    """Cellpose reads anisotropy=None as 1.0; the contract forbids assuming that."""
    registry.install()
    service = SegmentationService(SegmentationConfig(), scale=Scale.from_values(0.5, 10.0))
    with pytest.raises(UnsupportedStackError, match="Z step"):
        service.run_stack(_stack_3d())
    assert cellpose.init == []


def test_every_argument_passed_exists_in_the_installed_cellpose(registry, cellpose, dev_override):
    """The fake accepts anything; the real 3.1.1.3 signatures must too."""
    # PathFinder searches sys.path itself; importlib.util.find_spec would
    # return the fake already sitting in sys.modules.
    spec = importlib.machinery.PathFinder.find_spec("cellpose")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("cellpose is not installed")
    source = Path(list(spec.submodule_search_locations)[0]) / "models.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CellposeModel")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def names(fn):
        return {a.arg for a in fn.args.args + fn.args.kwonlyargs}

    registry.install()
    SegmentationService(SegmentationConfig(), scale=CALIBRATED_3D).run_stack(_stack_3d())
    service = SegmentationService(SegmentationConfig())
    service.run_stack(_stack(n=1))
    service.segment_crop(np.ones((16, 16), np.float32))
    for call in cellpose.init:
        assert set(call) - {"model_type"} <= names(methods["__init__"])
    for call in cellpose.evals:
        assert set(call) - {"shape"} <= names(methods["eval"])


# --------------------------------------------------------------------------
# resources.bundled_model_path delegates to the registry
# --------------------------------------------------------------------------


def test_bundled_model_path_returns_only_the_verified_file(registry, tmp_path, monkeypatch):
    impostor = tmp_path / "impostor"
    impostor.write_bytes(b"not validated")
    monkeypatch.setenv("CORRIDOR_MODEL", str(impostor))

    registry.install(None)
    assert resources.bundled_model_path() is None  # never the impostor

    registry.install(b"tampered")
    assert resources.bundled_model_path() is None

    weights = registry.install()
    assert resources.bundled_model_path() == weights.resolve()


def test_bundled_model_path_is_the_production_model_even_under_an_override(
    registry, tmp_path, monkeypatch
):
    weights = registry.install()
    dev = tmp_path / "dev_model"
    dev.write_bytes(b"developer weights")
    monkeypatch.setenv(ENV_DEVELOPER, "1")
    monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(dev))
    assert resources.bundled_model_path() == weights.resolve()


def test_the_unavailable_message_is_the_contract_text():
    assert MODEL_UNAVAILABLE_MESSAGE == CONTRACT_TEXT


# --------------------------------------------------------------------------
# The 1.x command line cannot silently lose its model choice
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag", [["--model", "x.pt"], ["--builtin-model", "cyto3"]], ids=["model", "builtin"]
)
def test_the_removed_model_options_are_refused_not_ignored(flag, capsys):
    from corridor import cli

    argv = ["in.tif", "-o", "out", *flag]
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    assert info.value.code == 2
    assert "removed in Corridor 2.0" in capsys.readouterr().err
    with pytest.raises(ValueError, match="removed in Corridor 2.0"):
        cli.config_from_args(cli.build_parser().parse_args(argv))


def test_the_command_line_names_the_axes_and_channel():
    from corridor import cli

    args = cli.build_parser().parse_args(
        ["in.tif", "-o", "out", "--axes", "TYX", "--channel", "1"]
    )
    config = cli.config_from_args(args)
    assert (config.import_.axes, config.import_.channel_index) == ("TYX", 1)
    # Nothing in the segmentation settings for the service to record as ignored.
    assert config.segmentation.model_path is None and config.segmentation.use_custom_model
