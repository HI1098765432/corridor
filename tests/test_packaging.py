"""What the frozen bundle contains, checked without running PyInstaller.

``packaging/corridor.spec`` builds from ``packaging/bundle_plan.py``; these
tests read the same plan. The version tests that used to live here (the
Windows resource and the Inno Setup define, which pinned literal formats of
``APP_VERSION``) are in ``test_version.py`` now, derived from
``corridor._version``.

The Cellpose hidden imports are a measured list
(``packaging/cellpose_runtime_modules.txt``, from
``scripts/measure_runtime_imports.py``). The frozen self-test's real
segmentation remains the final guard; these tests make sure the spec cannot
drift away from the measurement before anyone gets that far.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from corridor import app_meta

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
SPEC = PACKAGING / "corridor.spec"
INNO_SCRIPT = PACKAGING / "corridor.iss"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
REQUIREMENTS = ROOT / "requirements"
LOCK = PACKAGING / "requirements-locked.txt"
sys.path.insert(0, str(PACKAGING))

import bundle_plan as plan  # noqa: E402

PRODUCTION_SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


def read(path: Path) -> str:
    # PowerShell writes UTF-8 with a BOM; utf-8-sig reads both kinds.
    return path.read_text(encoding="utf-8-sig")


@pytest.fixture(scope="module")
def measurement() -> plan.Measurement:
    return plan.load_measurement()


# --------------------------------------------------------------------------
# Install identity
# --------------------------------------------------------------------------


@pytest.mark.skipif(not INNO_SCRIPT.exists(), reason="packaging assets absent")
def test_the_install_identity_is_stable_across_versions():
    """Changing the GUID orphans the previous installation's uninstaller."""
    assert app_meta.APP_GUID in read(INNO_SCRIPT)


# --------------------------------------------------------------------------
# Hidden imports: a superset of what was measured
# --------------------------------------------------------------------------


def test_the_measurement_is_a_real_one(measurement):
    """The list must come from a run that loaded the inference path; an empty
    or hand-trimmed file would make the superset test below vacuous."""
    modules = set(measurement.cellpose_modules)
    for needed in ("cellpose.models", "cellpose.core", "cellpose.resnet_torch",
                   "cellpose.dynamics", "cellpose.transforms"):
        assert needed in modules
    assert all(m == "cellpose" or m.startswith("cellpose.") for m in modules)
    assert measurement.cellpose_version and measurement.cellpose_version.startswith("3.")


def test_hidden_imports_cover_every_measured_cellpose_module(measurement):
    """The plan's list is the same with or without Napari; the spec only
    appends Napari's own modules to it."""
    hidden = set(plan.hidden_imports(measurement))
    missing = set(measurement.cellpose_modules) - hidden
    assert not missing, f"the spec would not ship measured modules: {sorted(missing)}"


@pytest.mark.parametrize("napari", [False, True])
def test_no_measured_module_is_excluded(measurement, napari):
    excluded = plan.excludes(measurement, napari=napari)
    for module in measurement.cellpose_modules:
        assert not any(module == e or module.startswith(e + ".") for e in excluded), module
    loaded = measurement.third_party_top_level or frozenset()
    guarded = set(plan.OPTIONAL_THIRD_PARTY_CANDIDATES) | set(plan.NAPARI_STACK)
    assert not (set(excluded) & guarded & loaded)


def test_unused_cellpose_trees_are_excluded_only_because_they_were_not_loaded(measurement):
    excluded = set(plan.excludes(measurement, napari=False))
    for candidate in plan.CELLPOSE_EXCLUSION_CANDIDATES:
        measured_under = [
            m for m in measurement.cellpose_modules
            if m == candidate or m.startswith(candidate + ".")
        ]
        assert (candidate in excluded) == (not measured_under), candidate
    # And the guard works: pretend training had been loaded.
    pretend = plan.Measurement(
        measurement.cellpose_modules + ("cellpose.train",),
        measurement.third_party_top_level,
        measurement.cellpose_version,
    )
    assert "cellpose.train" not in plan.excludes(pretend, napari=False)
    assert "cellpose.train" in plan.hidden_imports(pretend)


def test_the_spec_builds_from_the_plan():
    text = read(SPEC)
    assert 'collect_submodules("cellpose")' not in text
    assert "plan.hidden_imports(measurement)" in text
    assert "plan.excludes(measurement" in text
    assert "plan.model_datas(model)" in text
    assert "check_measurement_current" in text


def test_a_stale_measurement_stops_the_build(measurement):
    plan.check_measurement_current(measurement, measurement.cellpose_version)
    with pytest.raises(plan.BundleError, match="Re-run"):
        plan.check_measurement_current(measurement, "3.9.9")
    with pytest.raises(plan.BundleError):
        plan.check_measurement_current(plan.Measurement(("cellpose",)), "3.1.1.3")


# --------------------------------------------------------------------------
# Never bundled: research code and data
# --------------------------------------------------------------------------


def test_training_and_data_are_never_collected(measurement, tmp_path):
    for napari in (False, True):
        excluded = plan.excludes(measurement, napari=napari)
        assert "training" in excluded
        hidden = plan.hidden_imports(measurement)
        assert not [h for h in hidden if h.split(".")[0] in ("training", "data")]

    datas = [
        (str(ROOT / "src" / "corridor" / "assets" / "corridor.ico"), "assets"),
        (str(ROOT / "data" / "confinedmig_cellTrack" / "sample_data" / "x.tif"), "x"),
        (str(ROOT / "training" / "datasets.py"), "training"),
        (str(ROOT / "build" / "models" / "checkpoint"), "assets/models"),
        (str(ROOT / "database_notes.txt"), "assets"),  # a prefix, not the dir
    ]
    flagged = plan.forbidden_sources(datas)
    assert flagged == [d[0] for d in datas[1:4]]


def test_the_spec_takes_the_model_only_from_the_staged_assets():
    """1.x fell back to the copy under data/, which bypassed the hash check."""
    text = read(SPEC)
    assert '"data"' not in text and "'data'" not in text
    assert "forbidden_sources(datas)" in text


@pytest.mark.skipif(
    not (ROOT / "src" / "corridor" / "assets" / "model_registry.json").exists(),
    reason="the model registry is created by another work package",
)
def test_the_real_asset_datas_carry_the_registry_and_nothing_forbidden():
    datas = plan.asset_datas()
    sources = {Path(s).name for s, _ in datas}
    assert "model_registry.json" in sources
    assert plan.forbidden_sources(datas) == []


# --------------------------------------------------------------------------
# Napari is opt-in; pandas goes when nothing needs it
# --------------------------------------------------------------------------


def test_napari_is_opt_in():
    assert plan.napari_requested({}) is False
    assert plan.napari_requested({plan.NAPARI_ENV: "0"}) is False
    assert plan.napari_requested({plan.NAPARI_ENV: "1"}) is True
    with pytest.raises(plan.BundleError):
        plan.napari_requested({plan.NAPARI_ENV: "ture"})


def test_napari_is_excluded_unless_requested(measurement):
    without = set(plan.excludes(measurement, napari=False))
    with_napari = set(plan.excludes(measurement, napari=True))
    assert "napari" in without and "napari" not in with_napari
    # Napari genuinely requires pandas and Dask.
    assert not ({"pandas", "dask"} & with_napari)


def test_pandas_is_excluded_only_when_nothing_needs_it(measurement, tmp_path):
    loaded = measurement.third_party_top_level
    assert loaded is not None, "runtime_imports.json is missing"
    assert "pandas" not in loaded, "re-measured: Cellpose now loads pandas"
    assert plan.app_imports("pandas") == []
    assert "pandas" in plan.excludes(measurement, napari=False)

    loads_pandas = plan.Measurement(
        measurement.cellpose_modules, loaded | {"pandas"}, measurement.cellpose_version
    )
    assert "pandas" not in plan.excludes(loads_pandas, napari=False)

    app = tmp_path / "corridor"
    app.mkdir()
    (app / "tables.py").write_text("def f():\n    import pandas\n", encoding="utf-8")
    assert "pandas" not in plan.excludes(measurement, napari=False, source=app)

    unknown = plan.Measurement(measurement.cellpose_modules)
    assert "pandas" not in plan.excludes(unknown, napari=False)


def test_pyproject_does_not_require_what_the_app_never_imports():
    tomllib = pytest.importorskip("tomllib")
    deps = tomllib.loads(read(ROOT / "pyproject.toml"))["project"]["dependencies"]
    names = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower() for d in deps}
    assert "pandas" not in names
    assert any(d.replace(" ", "") == "cellpose>=3,<4" for d in deps)


# --------------------------------------------------------------------------
# The production model, from the registry
# --------------------------------------------------------------------------


def _registry(tmp_path: Path, payload) -> Path:
    path = tmp_path / "model_registry.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                    encoding="utf-8")
    return path


GOOD = {
    "models": [
        {"model_id": "other", "filename": "other_model", "sha256": "0" * 64},
        {"model_id": "prod", "filename": "prod_model", "sha256": "AB" * 32},
    ],
    "production": {"2D": "prod"},
}


def test_the_registry_names_one_production_model(tmp_path):
    model = plan.production_model(_registry(tmp_path, GOOD))
    assert model == plan.ProductionModel("prod", "prod_model", "ab" * 32)


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        [],
        {"models": GOOD["models"]},
        {"models": GOOD["models"], "production": {"3D": "prod"}},
        {"models": "x", "production": {"2D": "prod"}},
        {"models": [], "production": {"2D": "prod"}},
        {"models": GOOD["models"] * 2, "production": {"2D": "prod"}},
        {"models": [{"model_id": "prod", "filename": "../evil", "sha256": "a" * 64}],
         "production": {"2D": "prod"}},
        {"models": [{"model_id": "prod", "filename": "C:model", "sha256": "a" * 64}],
         "production": {"2D": "prod"}},
        {"models": [{"model_id": "prod", "filename": "m", "sha256": "abc"}],
         "production": {"2D": "prod"}},
        {"models": [{"model_id": "prod", "filename": "m"}], "production": {"2D": "prod"}},
    ],
)
def test_a_malformed_registry_stops_the_build(tmp_path, payload):
    with pytest.raises(plan.RegistryError):
        plan.production_model(_registry(tmp_path, payload))


def test_a_missing_registry_stops_the_build(tmp_path):
    with pytest.raises(plan.RegistryError, match="missing"):
        plan.production_model(tmp_path / "absent.json")


def test_the_staged_model_is_hash_checked(tmp_path):
    weights = b"not really a network"
    model = plan.ProductionModel("prod", "prod_model", hashlib.sha256(weights).hexdigest())
    with pytest.raises(plan.BundleError, match="not staged"):
        plan.verify_staged_model(model, tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "prod_model").write_bytes(weights + b"!")
    with pytest.raises(plan.BundleError, match="validated"):
        plan.verify_staged_model(model, tmp_path)
    (tmp_path / "models" / "prod_model").write_bytes(weights)
    assert plan.model_datas(model, tmp_path) == [
        (str(tmp_path / "models" / "prod_model"), "assets/models")
    ]


@pytest.mark.skipif(not plan.REGISTRY_PATH.exists(), reason="registry not yet created")
def test_the_build_and_the_application_agree_on_the_production_model():
    """Two parsers of one file: this one (stdlib-only, for the spec and CI) and
    the application's. They must name the same weights."""
    registry = pytest.importorskip("corridor.core.model_registry")
    ours = plan.production_model()
    theirs = registry.production_spec("2D")
    assert (ours.filename, ours.sha256) == (theirs.filename, theirs.sha256.lower())


# --------------------------------------------------------------------------
# The release workflow
# --------------------------------------------------------------------------


@pytest.mark.skipif(not WORKFLOW.exists(), reason="workflow absent")
def test_the_workflow_hard_codes_no_model_and_checks_the_version():
    text = read(WORKFLOW)
    assert not PRODUCTION_SHA256.search(text), "a SHA-256 is written into release.yml"
    assert "cyto2_phase_microfluidic" not in text
    assert "bundle_plan.py model" in text
    assert "sync_version.py --check" in text
    assert "--release" in text and "--tag" in text
    assert "requirements-locked.txt" in text


# --------------------------------------------------------------------------
# Requirements
# --------------------------------------------------------------------------

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for raw in read(path).splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-r ", "--")):
            continue
        match = _PIN.match(line)
        assert match, f"{path.name}: not an exact pin: {line!r}"
        pins[_canonical(match.group(1))] = match.group(2)
    return pins


def test_the_lock_agrees_with_the_app_requirements():
    app = _pins(REQUIREMENTS / "app.txt")
    lock = _pins(LOCK)
    disagree = {n: (v, lock.get(n)) for n, v in app.items() if lock.get(n) != v}
    assert not disagree, f"requirements/app.txt vs requirements-locked.txt: {disagree}"


def test_the_lock_pins_no_commit_and_ships_no_unused_package():
    text = read(LOCK)
    assert not re.search(r"^\s*-e\s+git\+", text, re.MULTILINE)
    app = _pins(REQUIREMENTS / "app.txt")
    assert app["cellpose"].startswith("3.")
    for absent in ("pandas", "torchvision", "napari"):
        assert absent not in app


def test_cellpose_4_stays_out_of_the_application_environment():
    cp4 = _pins(REQUIREMENTS / "cp4.txt")
    assert cp4["cellpose"].startswith("4.")
    for name in ("app.txt", "research.txt", "napari.txt"):
        text = read(REQUIREMENTS / name)
        assert not re.search(r"^\s*-r\s+\S*cp4\.txt", text, re.MULTILINE), name
        assert _pins(REQUIREMENTS / name).get("cellpose", "3.").startswith("3.")
    assert not re.search(r"^\s*-r\s+app\.txt", read(REQUIREMENTS / "cp4.txt"), re.MULTILINE)


def test_research_and_napari_build_on_the_app():
    for name in ("research.txt", "napari.txt"):
        assert re.search(r"^-r app\.txt$", read(REQUIREMENTS / name), re.MULTILINE)
    assert "scikit-learn" in _pins(REQUIREMENTS / "research.txt")
    assert "napari" in _pins(REQUIREMENTS / "napari.txt")
