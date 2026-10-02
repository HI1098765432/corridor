"""Every place that states Corridor's version agrees with ``_version.py``.

By 1.3.0 the version was written in five places and they had drifted:
``app_meta`` said 1.3.0, ``pyproject.toml`` 1.1.0, the installed metadata
1.0.0, and the GitHub tags v1.2.0 and v1.3.0 both pointed at 1.1.0 code. Now it
is written once, in ``src/corridor/_version.py``. The application imports it,
setuptools reads it through ``attr``, and ``scripts/sync_version.py``
generates the two copies that tools outside Python need. These tests fail if
any of them stops deriving from it.

The release notes are the one location deliberately *not* required to name the
version on every commit. They are the GitHub release body, and between
releases they describe the release users have: during the 2.0 work they
describe 1.3.0, and making them say "New in 2.0.0" before 2.0 exists would
publish claims about features not yet built. What must never happen is a
release whose notes describe another version, so that is enforced where a
release is made (``sync_version.py --check --release``, run by the release
workflow on every tag). Here the tests check that the gate works, and that the
notes are never ahead of the code and never contradict themselves.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

import corridor
from corridor import _version, app_meta
from corridor.core import updates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_version  # noqa: E402

VERSION = _version.__version__
PYPROJECT = ROOT / "pyproject.toml"


def read(path: Path) -> str:
    # PowerShell writes UTF-8 with a BOM; utf-8-sig reads both kinds.
    return path.read_text(encoding="utf-8-sig")


def as_tuple(version: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in version.split("."))  # type: ignore[return-value]


# --------------------------------------------------------------------------
# The source
# --------------------------------------------------------------------------


def test_the_version_is_three_plain_integers():
    """``is_newer`` compares three components and the Windows resource holds
    integers, so ``2.0.0rc1`` would compare equal to 2.0.0 and cannot be
    written into the executable at all."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION)


def test_the_script_reads_the_same_version_without_importing():
    assert sync_version.read_version(ROOT) == VERSION


def test_the_application_reports_the_single_source():
    assert app_meta.APP_VERSION == VERSION
    assert corridor.__version__ == VERSION
    assert corridor.APP_VERSION == VERSION


def test_no_second_literal_version_exists_in_app_meta():
    """``APP_VERSION`` is re-exported, never re-typed. A pasted literal would
    pass the equality test above on the day it is pasted and drift after."""
    tree = ast.parse(read(ROOT / "src" / "corridor" / "app_meta.py"))
    literal_targets = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "APP_VERSION" not in literal_targets
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "_version"
    ]
    assert imports, "app_meta no longer imports the version from _version.py"


# --------------------------------------------------------------------------
# pyproject.toml
# --------------------------------------------------------------------------


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads(read(PYPROJECT))


def test_pyproject_takes_the_version_from_the_package():
    data = _pyproject()
    project = data["project"]
    assert "version" not in project, "pyproject.toml has a literal version again"
    assert "version" in project.get("dynamic", [])
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "corridor._version.__version__"


def test_pyproject_dynamic_version_resolves_to_this_version():
    """Resolved the way a build resolves it, through setuptools itself."""
    attr = _pyproject()["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    try:
        from setuptools.config.expand import read_attr
    except ImportError:  # pragma: no cover - setuptools absent
        module_name, _, name = attr.rpartition(".")
        resolved = getattr(importlib.import_module(module_name), name)
    else:
        resolved = read_attr(attr, package_dir={"": "src"}, root_dir=ROOT)
    assert resolved == VERSION


# --------------------------------------------------------------------------
# The generated copies
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (ROOT / sync_version.VERSION_INFO).exists(),
                    reason="packaging assets absent")
def test_the_windows_version_resource_matches():
    text = read(ROOT / sync_version.VERSION_INFO)
    major, minor, patch = as_tuple(VERSION)
    tuple_form = f"({major}, {minor}, {patch}, 0)"
    dotted = f"{major}.{minor}.{patch}.0"
    assert f"filevers={tuple_form}" in text
    assert f"prodvers={tuple_form}" in text
    assert f"'FileVersion', '{dotted}'" in text
    assert f"'ProductVersion', '{dotted}'" in text


@pytest.mark.skipif(not (ROOT / sync_version.INNO_SCRIPT).exists(),
                    reason="packaging assets absent")
def test_the_installer_script_matches():
    text = read(ROOT / sync_version.INNO_SCRIPT)
    defines = re.findall(r'^#define\s+AppVersion\s+"([^"]+)"', text, re.MULTILINE)
    assert defines == [VERSION]


def test_the_generated_files_are_exactly_what_sync_version_writes():
    """The in-process form of ``sync_version.py --check``."""
    assert sync_version.out_of_sync(ROOT) == []


def _scratch_root(tmp_path: Path, version: str) -> Path:
    """A minimal tree for sync_version to work on, never the real one."""
    for rel in (sync_version.APP_META, sync_version.INNO_SCRIPT):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
    (tmp_path / sync_version.VERSION_FILE).write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    return tmp_path


def test_sync_is_idempotent(tmp_path):
    root = _scratch_root(tmp_path, "7.8.9")
    first = sync_version.sync(root)
    assert set(first) == {sync_version.VERSION_INFO, sync_version.INNO_SCRIPT}
    snapshot = {rel: (root / rel).read_bytes() for rel in first}
    assert sync_version.sync(root) == []
    assert {rel: (root / rel).read_bytes() for rel in first} == snapshot
    assert sync_version.main(["--check", "--root", str(root)]) == 0
    assert "filevers=(7, 8, 9, 0)" in read(root / sync_version.VERSION_INFO)
    assert '"7.8.9"' in read(root / sync_version.INNO_SCRIPT)


def test_sync_keeps_crlf_and_the_rest_of_the_installer_script(tmp_path):
    root = _scratch_root(tmp_path, "7.8.9")
    iss = root / sync_version.INNO_SCRIPT
    original = read(iss).replace("\r\n", "\n")
    iss.write_bytes(original.replace("\n", "\r\n").encode("utf-8"))
    sync_version.sync(root)
    raw = iss.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    after = raw.decode("utf-8").replace("\r\n", "\n")
    changed = [
        (a, b) for a, b in zip(original.splitlines(), after.splitlines()) if a != b
    ]
    assert len(changed) == 1 and "AppVersion" in changed[0][1]


def test_check_mode_reports_and_never_writes(tmp_path, capsys):
    root = _scratch_root(tmp_path, "7.8.9")
    before = (root / sync_version.INNO_SCRIPT).read_bytes()
    assert sync_version.main(["--check", "--root", str(root)]) == 1
    assert (root / sync_version.INNO_SCRIPT).read_bytes() == before
    assert not (root / sync_version.VERSION_INFO).exists()
    assert "out of date" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["2.0", "2.0.0rc1", "v2.0.0", "2.0.0.1"])
def test_a_malformed_version_is_refused(tmp_path, bad):
    root = _scratch_root(tmp_path, bad)
    with pytest.raises(sync_version.VersionError):
        sync_version.read_version(root)
    assert sync_version.main(["--check", "--root", str(root)]) == 2


# --------------------------------------------------------------------------
# Release notes and tags: enforced at release time
# --------------------------------------------------------------------------

RELEASE_NOTES = ROOT / sync_version.RELEASE_NOTES


@pytest.mark.skipif(not RELEASE_NOTES.exists(), reason="release notes absent")
def test_the_release_notes_are_never_ahead_of_the_code():
    newest = sync_version.newest_notes_version(read(RELEASE_NOTES))
    assert newest is not None, "the release notes have no '## New in X.Y.Z' heading"
    assert as_tuple(newest) <= as_tuple(VERSION)


@pytest.mark.skipif(not RELEASE_NOTES.exists(), reason="release notes absent")
def test_the_release_notes_agree_with_themselves():
    """Every installer the notes link is the release their newest heading
    describes. A download link for another version is worse than none."""
    text = read(RELEASE_NOTES)
    newest = sync_version.newest_notes_version(text)
    names = set(re.findall(r"Corridor-(\d+\.\d+\.\d+)-Setup\.exe", text))
    assert names == {newest}, f"installer names {sorted(names)} vs heading {newest}"


def test_the_release_gate_requires_notes_for_this_version():
    good = (
        "Download **Corridor-2.4.1-Setup.exe** below.\n\n"
        "## New in 2.4.1\n\n- things\n\n## New in 2.4.0\n"
    )
    assert sync_version.release_notes_problems("2.4.1", good) == []

    stale = good.replace("2.4.1", "2.4.0")
    problems = sync_version.release_notes_problems("2.4.1", stale)
    assert any("heading is 2.4.0" in p for p in problems)
    assert any("never name Corridor-2.4.1-Setup.exe" in p for p in problems)

    mixed = good + "\nGet-FileHash Corridor-2.4.0-Setup.exe\n"
    assert any("other versions" in p for p in sync_version.release_notes_problems("2.4.1", mixed))


def test_a_release_tag_must_name_this_version():
    assert sync_version.tag_problems(f"v{VERSION}", VERSION) == []
    assert sync_version.tag_problems(VERSION, VERSION)  # the v is part of the convention
    assert sync_version.tag_problems("v1.3.0", "2.0.0")


# --------------------------------------------------------------------------
# Consumers
# --------------------------------------------------------------------------


def test_the_update_check_compares_against_this_version():
    """``is_newer`` binds the version as a default at import time, so it must
    see the single source -- and the semantics must not move with it."""
    default = inspect.signature(updates.is_newer).parameters["current"].default
    assert default == VERSION
    major, minor, patch = as_tuple(VERSION)
    assert updates.is_newer(VERSION) is False
    assert updates.is_newer(f"v{major}.{minor}.{patch + 1}") is True
    assert updates.is_newer(f"{major}.{minor + 1}.0") is True
    assert updates.is_newer(f"{major + 1}.0.0") is True
    # 1.3.0 is the last shipped release (tag shipped-1.3.0): never "newer"
    # than this one, and this one is announced to everybody still on it.
    assert updates.is_newer("1.3.0") is False
    assert updates.is_newer(VERSION, "1.3.0") is True


def test_the_upload_script_defaults_to_this_release():
    spec = importlib.util.spec_from_file_location(
        "upload_asset_under_test", ROOT / "scripts" / "upload_asset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_TAG == f"v{VERSION}"


def test_the_build_names_its_installer_from_this_version():
    spec = importlib.util.spec_from_file_location(
        "build_release_under_test", ROOT / "scripts" / "build_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app_version() == VERSION
