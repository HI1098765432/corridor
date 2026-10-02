"""The model registry: one validated model per dimensionality, locked by hash.

Every test builds its own registry and its own "model" (a few bytes whose
SHA-256 is known), so nothing here loads Cellpose or needs the real weights.
The one test that does read the real weights skips where they are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from corridor import resources
from corridor.core import model_registry as mr
from corridor.core.model_registry import (
    DEVELOPER_OVERRIDE_ID,
    ENV_DEVELOPER,
    ENV_DEVELOPER_MODEL,
    MODEL_UNAVAILABLE_MESSAGE,
    ModelUnavailable,
)

from conftest import REPO_ROOT

COMBI = "cyto2_phase_microfluidic_KK1KK2_combi"
COMBI_SHA256 = "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"

#: The exact text the contract (§2) requires, so a reworded message fails here.
CONTRACT_TEXT = (
    "The validated Corridor segmentation model is missing or does not match the "
    "expected checksum."
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """No developer variables from the shell, and no hashes from other tests."""
    for name in (ENV_DEVELOPER, ENV_DEVELOPER_MODEL, "CORRIDOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    mr._HASH_CACHE.clear()
    yield
    mr._HASH_CACHE.clear()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A registry naming one 2-D model, and the bytes that model must have."""
    content = b"pretend these are Cellpose weights"
    sha = hashlib.sha256(content).hexdigest()
    registry = tmp_path / "model_registry.json"
    registry.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "test_model",
                        "model_version": "1.0.0",
                        "architecture": "cellpose3-cyto2-resnet",
                        "cellpose_version": ">=3,<4",
                        "filename": "weights.bin",
                        "sha256": sha,
                        "training_dataset_version": "synthetic",
                        "dimensions": ["2D"],
                        "pixel_size_range_um": [0.46, 0.65],
                        "validated_app_version": "1.0.0",
                        "some_future_key": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mr, "registry_path", lambda: registry)

    class Fake:
        good = content
        digest = sha
        root = tmp_path

        @staticmethod
        def place(*files: tuple[str, bytes | None]) -> list[Path]:
            """Write each (name, bytes) and make them the candidate list."""
            paths = []
            for name, data in files:
                path = tmp_path / name / "weights.bin"
                if data is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                paths.append(path)
            monkeypatch.setattr(mr, "candidate_paths", lambda spec: list(paths))
            return paths

    return Fake


# --------------------------------------------------------------------------
# The shipped registry
# --------------------------------------------------------------------------


def test_the_shipped_registry_names_the_lab_model():
    specs = mr.load_registry()
    assert [s.model_id for s in specs] == ["jhu_confined_cp3_combi"]
    spec = mr.production_spec("2D")
    assert spec.filename == COMBI
    assert spec.sha256 == COMBI_SHA256
    assert spec.sha256 == resources.BUNDLED_MODEL_SHA256  # the v1 pin agrees
    assert spec.architecture == "cellpose3-cyto2-resnet"
    assert spec.cellpose_version == ">=3,<4"
    assert spec.dimensions == ("2D",)
    assert spec.pixel_size_range_um == (0.46, 0.65)
    assert spec.validated_app_version == "1.0.0"
    assert "71" in spec.training_dataset_version and "246" in spec.training_dataset_version


def test_the_shipped_registry_is_where_the_bundle_looks():
    """The spec bundles every file in assets/ into the frozen app's assets/."""
    path = mr.registry_path()
    assert path == resources.asset("model_registry.json")
    assert path.is_file()


def test_the_lab_model_accepts_only_cellpose_3():
    spec = mr.production_spec()
    assert spec.accepts_cellpose("3.1.1.3")
    assert spec.accepts_cellpose("3.0")
    assert not spec.accepts_cellpose("4.0.1")
    assert not spec.accepts_cellpose("2.2.3")
    assert not spec.accepts_cellpose("unavailable")


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_a_matching_file_resolves(fake):
    (path,) = fake.place(("bundle", fake.good))
    resolved = mr.resolve_model()
    assert resolved.path == path.resolve()
    assert resolved.sha256 == fake.digest
    assert resolved.developer_override is False
    assert resolved.spec.model_id == "test_model"


def test_a_mismatching_file_is_refused_with_the_contract_text(fake):
    (path,) = fake.place(("bundle", b"some other model entirely"))
    with pytest.raises(ModelUnavailable) as info:
        mr.resolve_model()
    message = str(info.value)
    assert message.startswith(CONTRACT_TEXT)
    assert MODEL_UNAVAILABLE_MESSAGE == CONTRACT_TEXT
    assert str(path) in message and "checksum mismatch" in message
    assert info.value.tried[0][0] == path


def test_a_missing_file_is_refused_and_every_path_is_named(fake):
    paths = fake.place(("bundle", None), ("localappdata", None), ("checkout", None))
    with pytest.raises(ModelUnavailable) as info:
        mr.resolve_model()
    assert str(info.value).startswith(CONTRACT_TEXT)
    assert [p for p, _ in info.value.tried] == paths
    assert all(why == "missing" for _, why in info.value.tried)
    for path in paths:
        assert str(path) in str(info.value)


def test_a_mismatch_earlier_in_the_order_is_skipped_never_used(fake):
    paths = fake.place(("bundle", b"tampered"), ("localappdata", None), ("checkout", fake.good))
    resolved = mr.resolve_model()
    assert resolved.path == paths[2].resolve()
    assert mr.sha256_file(resolved.path) == fake.digest


def test_no_fallback_ever_returns_a_mismatching_model(fake):
    """Every candidate present, none matching: nothing is returned."""
    fake.place(("bundle", b"a"), ("localappdata", b"b"), ("checkout", b"c"))
    with pytest.raises(ModelUnavailable) as info:
        mr.resolve_model()
    assert len(info.value.tried) == 3
    assert all(why.startswith("checksum mismatch") for _, why in info.value.tried)


def test_the_legacy_model_variable_is_ignored(fake, monkeypatch, tmp_path):
    impostor = tmp_path / "impostor.bin"
    impostor.write_bytes(b"not validated")
    monkeypatch.setenv("CORRIDOR_MODEL", str(impostor))
    (path,) = fake.place(("bundle", fake.good))
    assert mr.resolve_model().path == path.resolve()


def test_the_hash_is_cached_by_path_size_and_mtime(fake, monkeypatch):
    (path,) = fake.place(("bundle", fake.good))
    calls = []
    real = mr.sha256_file
    monkeypatch.setattr(mr, "sha256_file", lambda p: calls.append(p) or real(p))
    mr.resolve_model()
    mr.resolve_model()
    assert len(calls) == 1
    path.write_bytes(fake.good + b"!")  # a changed file is hashed again...
    with pytest.raises(ModelUnavailable):  # ...and refused
        mr.resolve_model()
    assert len(calls) == 2


# --------------------------------------------------------------------------
# Developer override
# --------------------------------------------------------------------------


@pytest.fixture
def dev_model(tmp_path):
    path = tmp_path / "dev" / "experimental.bin"
    path.parent.mkdir()
    path.write_bytes(b"an experimental checkpoint")
    return path


@pytest.mark.parametrize(
    "flag, model",
    [
        pytest.param("1", None, id="flag-only"),
        pytest.param(None, "set", id="model-only"),
        pytest.param("true", "set", id="flag-not-1"),
        pytest.param("0", "set", id="flag-0"),
    ],
)
def test_the_override_needs_both_variables(fake, dev_model, monkeypatch, flag, model):
    (path,) = fake.place(("bundle", fake.good))
    if flag is not None:
        monkeypatch.setenv(ENV_DEVELOPER, flag)
    if model is not None:
        monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(dev_model))
    resolved = mr.resolve_model()
    assert resolved.developer_override is False
    assert resolved.path == path.resolve()


def test_both_variables_select_the_developer_model_and_say_so(fake, dev_model, monkeypatch):
    fake.place(("bundle", fake.good))
    monkeypatch.setenv(ENV_DEVELOPER, "1")
    monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(dev_model))
    resolved = mr.resolve_model()
    assert resolved.developer_override is True
    assert resolved.path == dev_model.resolve()
    assert resolved.spec.model_id == DEVELOPER_OVERRIDE_ID
    assert resolved.sha256 == hashlib.sha256(dev_model.read_bytes()).hexdigest()
    manifest = resolved.to_manifest()
    assert manifest["developer_override"] is True
    assert manifest["model_id"] == DEVELOPER_OVERRIDE_ID


def test_a_developer_model_that_does_not_exist_is_an_error_not_a_fallback(fake, monkeypatch, tmp_path):
    fake.place(("bundle", fake.good))
    monkeypatch.setenv(ENV_DEVELOPER, "1")
    monkeypatch.setenv(ENV_DEVELOPER_MODEL, str(tmp_path / "nowhere.bin"))
    with pytest.raises(ModelUnavailable, match="does not exist"):
        mr.resolve_model()


def test_a_research_model_is_always_marked(dev_model):
    resolved = mr.research_model(dev_model, label="round4")
    assert resolved.developer_override is True
    assert resolved.spec.model_id == "research:round4"
    assert resolved.sha256 == hashlib.sha256(dev_model.read_bytes()).hexdigest()
    with pytest.raises(ModelUnavailable):
        mr.research_model(dev_model.parent / "missing.bin", label="x")


def test_manifest_block_has_exactly_the_contract_keys(fake):
    fake.place(("bundle", fake.good))
    assert set(mr.resolve_model().to_manifest()) == {
        "model_id", "model_version", "architecture", "sha256", "cellpose_version",
        "training_dataset_version", "developer_override",
    }


# --------------------------------------------------------------------------
# 3-D and malformed registries
# --------------------------------------------------------------------------


def test_3d_is_refused_because_no_model_is_validated_for_it(fake):
    fake.place(("bundle", fake.good))
    for call in (mr.production_spec, mr.resolve_model):
        with pytest.raises(ModelUnavailable, match="No 3D-validated segmentation model"):
            call("3D")


def test_the_shipped_registry_also_refuses_3d():
    with pytest.raises(ModelUnavailable, match="3D"):
        mr.production_spec("3d")


def test_an_unknown_dimensionality_is_a_programming_error():
    with pytest.raises(ValueError):
        mr.production_spec("4D")


def _write_registry(tmp_path, entries) -> Path:
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"models": entries}), encoding="utf-8")
    return path


def _entry(**overrides):
    entry = {
        "model_id": "m", "model_version": "1", "architecture": "a", "cellpose_version": ">=3,<4",
        "filename": "f", "sha256": "0" * 64, "training_dataset_version": "t",
        "dimensions": ["2D"], "pixel_size_range_um": None, "validated_app_version": "1.0.0",
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_entry(sha256="abc")], id="short-sha"),
        pytest.param([_entry(dimensions=["4D"])], id="bad-dimension"),
        pytest.param([_entry(dimensions=[])], id="no-dimension"),
        pytest.param([_entry(pixel_size_range_um=[0.7, 0.4])], id="inverted-range"),
        pytest.param([{k: v for k, v in _entry().items() if k != "sha256"}], id="missing-sha"),
        pytest.param([_entry(), _entry()], id="duplicate-id"),
    ],
)
def test_a_malformed_registry_is_refused(tmp_path, entries):
    with pytest.raises(ValueError):
        mr.load_registry(_write_registry(tmp_path, entries))


def test_two_production_models_for_one_dimensionality_is_refused(tmp_path, monkeypatch):
    path = _write_registry(tmp_path, [_entry(model_id="a"), _entry(model_id="b")])
    monkeypatch.setattr(mr, "registry_path", lambda: path)
    with pytest.raises(ModelUnavailable, match="exactly one"):
        mr.production_spec("2D")


def test_an_unreadable_registry_is_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "registry_path", lambda: tmp_path / "absent.json")
    with pytest.raises(ModelUnavailable) as info:
        mr.resolve_model()
    assert str(info.value).startswith(CONTRACT_TEXT)


# --------------------------------------------------------------------------
# Search order
# --------------------------------------------------------------------------


def test_search_order_in_a_source_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("CORRIDOR_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "local"))
    spec = mr.production_spec()
    paths = mr.candidate_paths(spec)
    repo = Path(resources.__file__).resolve().parents[2]
    data = repo / "data" / "confinedmig_cellTrack"
    assert paths == [
        resources.asset("models", COMBI),
        tmp_path / "local" / "Corridor" / "models" / COMBI,
        data / "cp_custom_model" / COMBI,
        data / "CellPose_TrainData" / "KK1KK2_combiModel" / "models" / COMBI,
    ]


def test_search_order_in_a_frozen_bundle(monkeypatch, tmp_path):
    spec = mr.production_spec()  # read before pretending to be frozen
    bundle = tmp_path / "_internal"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("CORRIDOR_DATA_DIR", str(tmp_path / "userdata"))
    assert mr.registry_path() == bundle / "assets" / "model_registry.json"
    paths = mr.candidate_paths(spec)
    assert paths == [
        bundle / "assets" / "models" / COMBI,
        tmp_path / "userdata" / "models" / COMBI,
    ]  # no source-tree paths inside an installed application


# --------------------------------------------------------------------------
# The real weights, where they exist
# --------------------------------------------------------------------------

REAL_COPIES = [
    REPO_ROOT / "src" / "corridor" / "assets" / "models" / COMBI,
    REPO_ROOT / "data" / "confinedmig_cellTrack" / "cp_custom_model" / COMBI,
    REPO_ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData" / "KK1KK2_combiModel"
    / "models" / COMBI,
]


@pytest.mark.skipif(
    not any(p.is_file() for p in REAL_COPIES), reason="the combi model weights are not present"
)
def test_the_real_model_file_matches_the_registry():
    spec = mr.production_spec()
    for path in (p for p in REAL_COPIES if p.is_file()):
        assert mr.sha256_file(path) == spec.sha256, path
    resolved = mr.resolve_model()
    assert resolved.sha256 == COMBI_SHA256
    assert resolved.developer_override is False
    assert os.path.basename(resolved.path) == COMBI
