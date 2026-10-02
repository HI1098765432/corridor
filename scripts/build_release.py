"""Build the Windows application and its installer, then checksum both.

One command so that a release is reproducible and nobody has to remember the
order of the steps:

1. ``sync_version.py`` writes the version from ``src/corridor/_version.py``
   into the Windows version resource and the Inno Setup define. If it had to
   change either, the build goes ahead but says so: those files are committed,
   and a release must be built from what is committed (CI runs ``--check``).
2. The production model named by ``src/corridor/assets/model_registry.json``
   is staged into the assets and its SHA-256 checked.
3. PyInstaller (``packaging/corridor.spec``, which checks all of that again,
   because it can be run on its own), then Inno Setup.
4. ``build/build_manifest.json`` records what was built, including whether
   Napari is in it and how the build venv departs from
   ``packaging/requirements-locked.txt``. PyInstaller follows optional
   try-imports, so a package installed but not pinned can reach the bundle:
   a local build matches CI's only when that record is empty.

``--napari`` bundles Napari (it sets ``CORRIDOR_BUNDLE_NAPARI=1`` for the
spec); the default bundle has none.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD = ROOT / "build"
DIST = BUILD / "dist"
WORK = BUILD / "work"
INSTALLER_DIR = BUILD / "installer"
ASSETS = SRC / "corridor" / "assets"

for helper_dir in (ROOT / "scripts", ROOT / "packaging"):
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

import bundle_plan  # noqa: E402
import sync_version  # noqa: E402

#: Where a source checkout keeps the researchers' copies of the weights. Both
#: are checked against the registry's SHA-256 after staging, so the order only
#: decides which copy is read, never which model ships.
MODEL_SOURCE_DIRS = (
    ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
    / "KK1KK2_combiModel" / "models",
    ROOT / "data" / "confinedmig_cellTrack" / "cp_custom_model",
)

ISCC_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
    Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
]


def sha256(path: Path) -> str:
    return bundle_plan.sha256_file(path)


def run(command: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def app_version() -> str:
    """The one source, read without importing the package."""
    return sync_version.read_version(ROOT)


def sync_generated_files() -> str:
    version = app_version()
    written = sync_version.sync(ROOT, version)
    for rel in written:
        print(f"version {version}: rewrote {rel.as_posix()} -- commit it; "
              "CI builds only from committed files")
    notes = ROOT / sync_version.RELEASE_NOTES
    if notes.exists():
        problems = sync_version.release_notes_problems(
            version, notes.read_text(encoding="utf-8-sig")
        )
        for problem in problems:
            # Not fatal here: a local build is often a test build. The release
            # workflow runs the same check with --release and refuses.
            print(f"note: release notes are not ready for {version}: {problem}")
    return version


def stage_model() -> tuple[bundle_plan.ProductionModel, Path]:
    """Copy the production model into the assets the bundle will include."""
    try:
        model = bundle_plan.production_model()
    except bundle_plan.RegistryError as exc:
        raise SystemExit(str(exc)) from exc
    target = bundle_plan.staged_model_path(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        sources = [d / model.filename for d in MODEL_SOURCE_DIRS]
        source = next((s for s in sources if s.is_file()), None)
        if source is None:
            raise SystemExit(
                f"The Cellpose model {model.filename} is missing. Looked in:\n  "
                + "\n  ".join(str(s) for s in sources)
            )
        shutil.copy2(source, target)
    try:
        bundle_plan.verify_staged_model(model)
    except bundle_plan.BundleError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"model staged and verified: {model.model_id} {target.name}  "
          f"{model.sha256[:16]}...")
    return model, target


def report_environment_drift(napari: bool) -> dict[str, dict[str, str]]:
    """Say, without stopping a test build, what in this venv the lock does not pin."""
    installed = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    try:
        drift = bundle_plan.environment_drift(installed, napari=napari)
    except bundle_plan.BundleError as exc:
        raise SystemExit(str(exc)) from exc
    for key, what in (("outside_lock", "installed but not pinned"),
                      ("differs_from_lock", "not at the pinned version")):
        if drift[key]:
            print(f"note: {len(drift[key])} package(s) {what}; this bundle may differ "
                  "from one built from the lock: "
                  + ", ".join(f"{n} {v}" for n, v in drift[key].items()))
    return drift


def find_iscc() -> Path | None:
    for candidate in ISCC_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-app", action="store_true", help="reuse an existing build")
    parser.add_argument("--skip-installer", action="store_true")
    parser.add_argument("--napari", action="store_true",
                        help=f"bundle Napari (sets {bundle_plan.NAPARI_ENV}=1)")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.napari:
        env[bundle_plan.NAPARI_ENV] = "1"
    try:
        napari_bundled = bundle_plan.napari_requested(env)
    except bundle_plan.BundleError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        version = sync_generated_files()
    except sync_version.VersionError as exc:
        raise SystemExit(f"version: {exc}") from exc
    print(f"Corridor {version}" + (" (with Napari)" if napari_bundled else ""))
    drift = report_environment_drift(napari_bundled)

    python = sys.executable
    run([python, str(ROOT / "scripts" / "make_icon.py")])
    model, _ = stage_model()

    if not args.skip_app:
        if DIST.exists():
            shutil.rmtree(DIST, ignore_errors=True)
        run([
            python, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--distpath", str(DIST), "--workpath", str(WORK),
            str(ROOT / "packaging" / "corridor.spec"),
        ], cwd=str(ROOT), env=env)

    exe = DIST / "Corridor" / "Corridor.exe"
    if not exe.exists():
        raise SystemExit(f"The application was not produced: {exe}")
    total = sum(p.stat().st_size for p in (DIST / "Corridor").rglob("*") if p.is_file())
    print(f"\napplication: {exe}  ({total / 1e6:.0f} MB unpacked)")

    installer: Path | None = None
    if not args.skip_installer:
        iscc = find_iscc()
        if iscc is None:
            print("\nInno Setup was not found; skipping the installer.")
        else:
            INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
            run([str(iscc), str(ROOT / "packaging" / "corridor.iss")],
                cwd=str(ROOT / "packaging"))
            installer = INSTALLER_DIR / f"Corridor-{version}-Setup.exe"

    manifest = {
        "name": "Corridor",
        "version": version,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "unpacked_bytes": total,
        "model": {
            "model_id": model.model_id,
            "name": model.filename,
            "sha256": model.sha256,
        },
        "napari_bundled": napari_bundled,
        "environment_drift": drift,
    }
    if installer and installer.exists():
        digest = sha256(installer)
        manifest["installer"] = {
            "filename": installer.name,
            "bytes": installer.stat().st_size,
            "sha256": digest,
        }
        checksum_file = installer.with_suffix(".exe.sha256")
        checksum_file.write_text(f"{digest} *{installer.name}\n", encoding="utf-8")
        print(f"\ninstaller:   {installer}")
        print(f"size:        {installer.stat().st_size:,} bytes "
              f"({installer.stat().st_size / 1e6:.0f} MB)")
        print(f"sha256:      {digest}")

    (BUILD / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {BUILD / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
