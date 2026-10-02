"""Which licences are actually being shipped, read from the installed packages.

A hand-written licence list is accurate on the day it is written. Dependencies
arrive transitively, change licence between versions, and pull in new packages
of their own, so the only list worth trusting is one derived from what is
installed now.

This reads the environment that the installer is built from and reports every
distribution's licence, separating them into three groups:

*   **permissive** -- BSD, MIT, Apache and friends. Attribution only.
*   **weak copyleft** -- LGPL and MPL. Distributable inside a combined work,
    but only on conditions the packaging has to satisfy. These are listed
    individually because each one is a claim that needs checking against
    ``docs/THIRD_PARTY.md``.
*   **strong copyleft** -- GPL and AGPL. These would require publishing this
    application under the same terms, which is a decision nobody should make by
    accident through a transitive dependency.

Exits non-zero when something appears that is not already accounted for, so it
can be run as a gate before a release rather than read as a report afterwards.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def canonical(name: str) -> str:
    """PEP 503 normalisation. ``PySide6_Essentials`` and ``pyside6-essentials``
    are the same project, and a lookup table that does not know that silently
    reports a documented dependency as an undocumented one."""
    return re.sub(r"[-_.]+", "-", name).lower()


#: Weak-copyleft packages already documented in docs/THIRD_PARTY.md, with the
#: reason the packaging satisfies them. Anything else in this class is news.
KNOWN_WEAK_COPYLEFT = {
    "pyside6": "one-folder build; Qt DLLs ship uncompressed and replaceable",
    "pyside6-essentials": "one-folder build; Qt DLLs ship uncompressed and replaceable",
    "pyside6-addons": "one-folder build; Qt DLLs ship uncompressed and replaceable",
    "shiboken6": "one-folder build; binding ships uncompressed and replaceable",
    "fastremap": "one-folder build; extension module is replaceable",
    "fill-voids": "one-folder build; extension module is replaceable",
    "certifi": "MPL-2.0 is file-level copyleft; the CA bundle ships unmodified",
    "tqdm": "MPL-2.0 is file-level copyleft; ships unmodified",
}

#: Tools used to *produce* the installer and never shipped inside it. PyInstaller
#: is GPLv2, which would be alarming if any of it ended up in the application --
#: it does not. Only its bootloader is linked into the executable, and that
#: carries an explicit exception permitting distribution of the result under any
#: licence. The safe way to keep this honest is not to trust the list below but
#: to audit the built tree, which --dist does.
BUILD_ONLY = {
    "pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile",
    "pywin32-ctypes", "setuptools", "wheel", "pip", "build", "packaging",
    "pytest", "pytest-qt", "coverage", "iniconfig", "pluggy", "debugpy",
    "twine", "ruff", "mypy",
}

_STRONG = re.compile(r"\bagpl|affero|(?<!l)gplv?[23]|\bgpl\b|general public license", re.I)
_WEAK = re.compile(r"\blgpl|lesser general public|\bmpl\b|mozilla public", re.I)
_PERMISSIVE = re.compile(
    r"\bbsd|\bmit\b|apache|isc|python software foundation|psf|zlib|unlicense|"
    r"public domain|historical permission|cc0|hpnd",
    re.I,
)


def licence_of(dist: md.Distribution) -> str:
    """Best available licence string for a distribution.

    Packages state this three different ways and none of them is reliable
    alone: a modern one uses the ``License-Expression`` field, an older one
    puts free text in ``License``, and many put nothing useful in either and
    rely on a Trove classifier.
    """
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return str(expression).strip()

    classifiers = [
        c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")
    ]
    if classifiers:
        # "License :: OSI Approved :: BSD License" -> "BSD License"
        return "; ".join(sorted({c.split("::")[-1].strip() for c in classifiers}))

    text = (meta.get("License") or "").strip()
    if text:
        # Some projects paste the entire licence into this field.
        first = text.splitlines()[0].strip()
        return first if len(first) <= 60 else first[:57] + "..."
    return "UNSTATED"


def classify(licence: str) -> str:
    # Order matters: "LGPL" contains "GPL", so weak copyleft is tested first.
    if _WEAK.search(licence):
        return "weak copyleft"
    if _STRONG.search(licence):
        return "strong copyleft"
    if _PERMISSIVE.search(licence):
        return "permissive"
    return "unclear"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--dist",
        default=str(ROOT / "build" / "dist" / "Corridor" / "_internal"),
        help="the built application tree; audited instead of this environment "
             "when it exists, because it is what actually ships",
    )
    args = ap.parse_args()

    shipped: set[str] | None = None
    coverage_note = ""
    dist_tree = Path(args.dist)
    if dist_tree.is_dir():
        # The authoritative answer to "what is bundled" is the bundle.
        shipped = {
            canonical(info.name.split("-")[0])
            for info in dist_tree.glob("*.dist-info")
        }
        source = f"the built application ({len(shipped)} packages in {dist_tree})"
        # PyInstaller keeps a package's dist-info only when something asks for
        # it, so the bundle declares far fewer packages than it contains: torch
        # and numpy are in there without metadata. Auditing only what declares
        # itself would quietly pass a bundle whose unlabelled half is the risk.
        environment = {canonical(d.metadata["Name"]) for d in md.distributions()
                       if d.metadata.get("Name")} - BUILD_ONLY
        undeclared = len(environment) - len(shipped & environment)
        if undeclared > 0:
            coverage_note = (
                f"{undeclared} package(s) present in the build environment carry no "
                "metadata in the bundle, so their licences are checked against the "
                "environment below rather than the bundle. Run without --dist to "
                "audit all of them."
            )
    else:
        source = "this build environment (no built application found to audit)"

    found: dict[str, dict] = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = canonical(name)
        if shipped is not None and key not in shipped:
            continue
        if shipped is None and key in BUILD_ONLY:
            continue
        licence = licence_of(dist)
        found[key] = {
            "name": name,
            "version": dist.version,
            "licence": licence,
            "class": classify(licence),
        }

    groups: dict[str, list[dict]] = {}
    for row in found.values():
        groups.setdefault(row["class"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["name"].lower())

    problems: list[str] = []

    for row in groups.get("strong copyleft", []):
        problems.append(
            f"{row['name']} {row['version']} is {row['licence']}. Shipping it inside "
            "this application would put the whole application under those terms."
        )
    for row in groups.get("weak copyleft", []):
        if canonical(row["name"]) not in KNOWN_WEAK_COPYLEFT:
            problems.append(
                f"{row['name']} {row['version']} is {row['licence']} and is not "
                "accounted for in docs/THIRD_PARTY.md. Either document how the "
                "packaging satisfies it, or remove it."
            )

    report = {
        "audited": source,
        "coverage_note": coverage_note,
        "total": len(found),
        "by_class": {k: len(v) for k, v in sorted(groups.items())},
        "groups": groups,
        "problems": problems,
    }

    if args.as_json or args.out:
        text = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    else:
        print(f"auditing {source}")
        if coverage_note:
            print(f"NOTE: {coverage_note}")
        print()
        for group in ("strong copyleft", "weak copyleft", "unclear", "permissive"):
            rows = groups.get(group, [])
            if not rows:
                continue
            print(f"-- {group} ({len(rows)}) " + "-" * (56 - len(group)))
            for row in rows:
                note = KNOWN_WEAK_COPYLEFT.get(canonical(row["name"]), "")
                suffix = f"   [documented: {note}]" if note else ""
                print(f"  {row['name']:28s} {row['version']:14s} {row['licence']}{suffix}")
            print()

    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("No undocumented copyleft dependencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
