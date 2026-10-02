"""The version number lives in three files, and they must agree.

PyInstaller reads the Windows version resource, Inno Setup reads its own
define, and the application reports ``app_meta.APP_VERSION``. Nothing makes
them consistent automatically, so an installer can cheerfully advertise 1.1.0
while the program inside it says 1.2.0 -- and the first person to notice is a
user comparing the About box with the file they downloaded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from corridor import app_meta

ROOT = Path(__file__).resolve().parents[1]
VERSION_INFO = ROOT / "packaging" / "version_info.txt"
INNO_SCRIPT = ROOT / "packaging" / "corridor.iss"
RELEASE_NOTES = ROOT / "docs" / "RELEASE_NOTES.md"


def read(path: Path) -> str:
    # PowerShell writes UTF-8 with a BOM; utf-8-sig reads both kinds.
    return path.read_text(encoding="utf-8-sig")


def test_the_version_is_a_three_part_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", app_meta.APP_VERSION)


@pytest.mark.skipif(not VERSION_INFO.exists(), reason="packaging assets absent")
def test_the_windows_version_resource_matches():
    text = read(VERSION_INFO)
    major, minor, patch = app_meta.APP_VERSION.split(".")
    tuple_form = f"({major}, {minor}, {patch}, 0)"
    dotted = f"{major}.{minor}.{patch}.0"

    assert f"filevers={tuple_form}" in text
    assert f"prodvers={tuple_form}" in text
    assert f"'FileVersion', '{dotted}'" in text
    assert f"'ProductVersion', '{dotted}'" in text


@pytest.mark.skipif(not INNO_SCRIPT.exists(), reason="packaging assets absent")
def test_the_installer_script_matches():
    text = read(INNO_SCRIPT)
    match = re.search(r'#define\s+AppVersion\s+"([^"]+)"', text)
    assert match, "corridor.iss no longer defines AppVersion"
    assert match.group(1) == app_meta.APP_VERSION


@pytest.mark.skipif(not INNO_SCRIPT.exists(), reason="packaging assets absent")
def test_the_install_identity_is_stable_across_versions():
    """Changing the GUID orphans the previous installation's uninstaller."""
    text = read(INNO_SCRIPT)
    assert app_meta.APP_GUID in text


@pytest.mark.skipif(not RELEASE_NOTES.exists(), reason="release notes absent")
def test_the_release_notes_name_this_version():
    text = read(RELEASE_NOTES)
    assert f"## New in {app_meta.APP_VERSION}" in text
    assert f"Corridor-{app_meta.APP_VERSION}-Setup.exe" in text
    # A download link for a version that is not this one is worse than none.
    stale = re.findall(r"Corridor-(\d+\.\d+\.\d+)-Setup\.exe", text)
    assert set(stale) == {app_meta.APP_VERSION}, f"stale installer names: {set(stale)}"
