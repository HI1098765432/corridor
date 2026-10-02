"""Write the version from ``src/corridor/_version.py`` into the files that cannot import it.

PyInstaller's Windows version resource (``packaging/version_info.txt``) and
Inno Setup's ``#define AppVersion`` (``packaging/corridor.iss``) are read by
tools that never run Python, so they have to hold copies, and copies drift: by
1.3.0 the application, ``pyproject.toml`` and the installed metadata said three
different versions, and the GitHub tags v1.2.0 and v1.3.0 both pointed at 1.1.0
code. Both files are now generated from the one source and committed, so a
reviewer sees the change and the tests can compare them without a build.

    python scripts/sync_version.py          rewrite whatever is out of date
    python scripts/sync_version.py --check  exit 1 if anything is; writes nothing
    python scripts/sync_version.py --check --release --tag v2.0.0
        additionally require the release notes and the tag to name this version

``build_release.py`` runs the first form before it builds; the release workflow
runs the last. Idempotent: a second run changes nothing. A byte-order mark or
CRLF line endings already in a file are kept, and comparisons ignore both,
because git's ``autocrlf`` decides them per checkout, not the content.

Reads ``_version.py`` and ``app_meta.py`` with ``ast`` rather than importing
them, so it runs before the package is installed and cannot execute anything.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSION_FILE = Path("src/corridor/_version.py")
APP_META = Path("src/corridor/app_meta.py")
VERSION_INFO = Path("packaging/version_info.txt")
INNO_SCRIPT = Path("packaging/corridor.iss")
RELEASE_NOTES = Path("docs/RELEASE_NOTES.md")

#: Three integers and nothing else; see ``_version.py`` for why.
PLAIN_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_INNO_DEFINE = re.compile(r'^(#define[ \t]+AppVersion[ \t]+)"[^"\r\n]*"', re.MULTILINE)
_NOTES_HEADING = re.compile(r"^## New in (\d+\.\d+\.\d+)[ \t]*$", re.MULTILINE)
_INSTALLER_NAME = re.compile(r"Corridor-(\d+\.\d+\.\d+)-Setup\.exe")


class VersionError(ValueError):
    """The single source is missing, malformed, or cannot be written out."""


def _string_assignments(path: Path) -> dict[str, str]:
    """Top-level ``NAME = "literal"`` assignments, without executing the file."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value.value
    return values


def read_version(root: Path = ROOT) -> str:
    version = _string_assignments(root / VERSION_FILE).get("__version__")
    if version is None:
        raise VersionError(f"{VERSION_FILE.as_posix()} does not assign __version__ a string")
    if not PLAIN_VERSION.fullmatch(version):
        raise VersionError(
            f"__version__ = {version!r} is not three plain integers (MAJOR.MINOR.PATCH)"
        )
    return version


def windows_version(version: str) -> tuple[int, int, int, int]:
    """``2.0.0`` -> ``(2, 0, 0, 0)``; each field of a Windows version is 16 bits."""
    match = PLAIN_VERSION.fullmatch(version)
    if not match:
        raise VersionError(f"{version!r} is not MAJOR.MINOR.PATCH")
    parts = tuple(int(p) for p in match.groups()) + (0,)
    if any(p > 0xFFFF for p in parts):
        raise VersionError(f"{version!r}: a Windows version field cannot exceed 65535")
    return parts  # type: ignore[return-value]


def read_identity(root: Path = ROOT) -> dict[str, str]:
    values = _string_assignments(root / APP_META)
    wanted = ("APP_NAME", "APP_TAGLINE", "APP_PUBLISHER")
    missing = [name for name in wanted if name not in values]
    if missing:
        raise VersionError(f"{APP_META.as_posix()} no longer defines {', '.join(missing)}")
    return {name: values[name] for name in wanted}


def render_version_info(version: str, identity: dict[str, str]) -> str:
    """The PyInstaller version resource. PyInstaller ``eval``s this file."""
    four = windows_version(version)
    tuple_form = "(" + ", ".join(str(p) for p in four) + ")"
    dotted = ".".join(str(p) for p in four)
    name = identity["APP_NAME"]
    strings = [
        ("CompanyName", identity["APP_PUBLISHER"]),
        ("FileDescription", f"{name} - {identity['APP_TAGLINE']}"),
        ("FileVersion", dotted),
        ("InternalName", name),
        ("LegalCopyright", "MIT licensed"),
        ("OriginalFilename", f"{name}.exe"),
        ("ProductName", name),
        ("ProductVersion", dotted),
    ]
    table = "\n".join(
        f"            StringStruct({key!r}, {value!r})," for key, value in strings
    )
    return (
        "# Windows version resource. GENERATED by scripts/sync_version.py from\n"
        "# src/corridor/_version.py: edit that file and re-run the script, not this one.\n"
        "VSVersionInfo(\n"
        "  ffi=FixedFileInfo(\n"
        f"    filevers={tuple_form},\n"
        f"    prodvers={tuple_form},\n"
        "    mask=0x3f,\n"
        "    flags=0x0,\n"
        "    OS=0x40004,\n"
        "    fileType=0x1,\n"
        "    subtype=0x0,\n"
        "    date=(0, 0),\n"
        "  ),\n"
        "  kids=[\n"
        "    StringFileInfo(\n"
        "      [\n"
        "        StringTable(\n"
        "          '040904B0',\n"
        "          [\n"
        f"{table}\n"
        "          ],\n"
        "        )\n"
        "      ]\n"
        "    ),\n"
        "    VarFileInfo([VarStruct('Translation', [1033, 1200])]),\n"
        "  ],\n"
        ")\n"
    )


def render_inno(text: str, version: str) -> str:
    """``text`` with its one ``#define AppVersion`` line set to ``version``.

    Only that line: the rest of the script is hand-written, and the define is
    what ``AppVersion``, ``AppVerName``, ``VersionInfoVersion`` and the
    installer's file name all expand from.
    """
    found = len(_INNO_DEFINE.findall(text))
    if found != 1:
        raise VersionError(
            f"{INNO_SCRIPT.as_posix()} must have exactly one '#define AppVersion' line; "
            f"found {found}"
        )
    return _INNO_DEFINE.sub(lambda m: f'{m.group(1)}"{version}"', text)


def _read(path: Path) -> str | None:
    # utf-8-sig drops a BOM; text mode turns CRLF into LF.
    return path.read_text(encoding="utf-8-sig") if path.exists() else None


def desired_files(root: Path = ROOT, version: str | None = None) -> dict[Path, str]:
    """Relative path -> the content it should have, LF line endings."""
    version = version or read_version(root)
    inno = _read(root / INNO_SCRIPT)
    if inno is None:
        raise VersionError(f"{INNO_SCRIPT.as_posix()} is missing")
    return {
        VERSION_INFO: render_version_info(version, read_identity(root)),
        INNO_SCRIPT: render_inno(inno, version),
    }


def out_of_sync(root: Path = ROOT, version: str | None = None) -> list[Path]:
    return [
        rel for rel, desired in desired_files(root, version).items()
        if _read(root / rel) != desired
    ]


def sync(root: Path = ROOT, version: str | None = None) -> list[Path]:
    """Rewrite every out-of-date file; returns the ones written."""
    written = []
    for rel, desired in desired_files(root, version).items():
        path = root / rel
        if _read(path) == desired:
            continue
        raw = path.read_bytes() if path.exists() else b""
        bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
        body = desired.replace("\n", "\r\n") if b"\r\n" in raw else desired
        path.write_bytes(bom + body.encode("utf-8"))
        written.append(rel)
    return written


def newest_notes_version(text: str) -> str | None:
    """The first ``## New in X.Y.Z`` heading: the release the notes describe."""
    match = _NOTES_HEADING.search(text)
    return match.group(1) if match else None


def release_notes_problems(version: str, text: str) -> list[str]:
    """Why ``text`` cannot be the body of the ``version`` release, if it cannot.

    The notes are the GitHub release body (``publish_release.py`` and the
    workflow both upload them), so notes that describe another version, or
    link another version's installer, are what a user downloading this one
    reads.
    """
    problems = []
    newest = newest_notes_version(text)
    if newest != version:
        problems.append(
            f"the newest '## New in' heading is {newest or 'missing'}, not {version}"
        )
    names = set(_INSTALLER_NAME.findall(text))
    if version not in names:
        problems.append(f"the notes never name Corridor-{version}-Setup.exe")
    if names - {version}:
        problems.append(f"installer names for other versions: {sorted(names - {version})}")
    return problems


def tag_problems(tag: str, version: str) -> list[str]:
    expected = f"v{version}"
    if tag == expected:
        return []
    return [f"the tag {tag!r} does not name this version; expected {expected!r}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report, do not write; exit 1 if anything is out of date")
    parser.add_argument("--release", action="store_true",
                        help="also require the release notes to describe this version")
    parser.add_argument("--tag", help="also require this git tag to be v<version>")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        version = read_version(root)
        changed = out_of_sync(root, version) if args.check else sync(root, version)
    except VersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems = []
    if args.check:
        problems += [
            f"{rel.as_posix()} is out of date; run python scripts/sync_version.py"
            for rel in changed
        ]
    elif changed:
        for rel in changed:
            print(f"version {version}: rewrote {rel.as_posix()}")
    if args.release:
        notes = _read(root / RELEASE_NOTES)
        if notes is None:
            problems.append(f"{RELEASE_NOTES.as_posix()} is missing")
        else:
            problems += release_notes_problems(version, notes)
    if args.tag is not None:
        problems += tag_problems(args.tag, version)

    for problem in problems:
        print(f"version {version}: {problem}", file=sys.stderr)
    if not problems and not changed:
        print(f"version {version}: every generated location is in sync")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
