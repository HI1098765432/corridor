"""Build the publishable source archive, with the exclusions enforced.

The archive is published, so what it must *not* contain matters more than what
it does. Two things stay out, and neither is a judgement call made at packing
time:

*   **The research images.** 71 hand-labelled frames and the sample
    time-lapses. Publishing somebody's unpublished data is not a packaging
    decision.
*   **The trained model.** It belongs to the researchers who trained it.
    Redistributing it is their call; the installer carries it because that is
    what an installer is for, and the source archive does not.

Both are excluded by an allow-list rather than a deny-list. A deny-list fails
open -- a new directory of data added next month would be published by default
and nobody would notice until it was downloaded. An allow-list fails closed.

The archive is verified after packing: every entry is re-read and checked
against the rules, so a mistake in the rules shows up here rather than on a
release page.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Only these top-level entries are published. Anything else is not, including
#: anything added later.
INCLUDE = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "pytest.ini",
    "docs",
    "packaging",
    "scripts",
    "src",
    "tests",
)

#: Paths that must never appear, checked again after packing.
FORBIDDEN_PARTS = ("data", "__pycache__", ".git", ".venv", "build", "dist")
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".nd2", ".npy", ".npz", ".pyc", ".exe", ".zip")
#: Cellpose checkpoints have no extension, so they are caught by size instead.
MAX_FILE_BYTES = 4 << 20


def wanted(path: Path) -> tuple[bool, str]:
    """Should this file be published, and if not, why not."""
    parts = path.relative_to(ROOT).parts
    if any(part in FORBIDDEN_PARTS for part in parts):
        return False, "excluded directory"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False, f"excluded file type {path.suffix}"
    if path.stat().st_size > MAX_FILE_BYTES:
        return False, f"too large ({path.stat().st_size / (1 << 20):.1f} MB)"
    return True, ""


def collect() -> list[Path]:
    out: list[Path] = []
    skipped: dict[str, int] = {}
    for entry in INCLUDE:
        target = ROOT / entry
        if not target.exists():
            continue
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            keep, why = wanted(path)
            if keep:
                out.append(path)
            else:
                skipped[why] = skipped.get(why, 0) + 1
    for why, count in sorted(skipped.items()):
        print(f"  skipped {count:4d}  {why}")
    return out


def verify(archive: Path) -> list[str]:
    """Re-read the finished archive and re-apply the rules to every entry."""
    problems: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts[1:]  # drop the top-level folder
            if any(part in FORBIDDEN_PARTS for part in parts):
                problems.append(f"{info.filename}: excluded directory")
            if Path(info.filename).suffix.lower() in FORBIDDEN_SUFFIXES:
                problems.append(f"{info.filename}: excluded file type")
            if info.file_size > MAX_FILE_BYTES:
                problems.append(f"{info.filename}: {info.file_size} bytes")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "build"))
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from corridor import app_meta

    name = f"Corridor-{app_meta.APP_VERSION}-source"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{name}.zip"

    print(f"packing {name}")
    files = collect()
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, Path(name) / path.relative_to(ROOT))

    problems = verify(archive)
    if problems:
        archive.unlink()
        print("\nREFUSING to publish this archive:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out_dir / f"{name}.zip.sha256").write_text(
        f"{digest}  {name}.zip\n", encoding="utf-8"
    )
    print(f"\n{len(files)} files")
    print(f"archive: {archive}")
    print(f"size:    {archive.stat().st_size:,} bytes")
    print(f"sha256:  {digest}")
    print("verified: no research data, no model, no build output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
