"""Build the Windows application and its installer, then checksum both.

One command so that a release is reproducible and nobody has to remember the
order of the steps.
"""

from __future__ import annotations

import argparse
import hashlib
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

MODEL_NAME = "cyto2_phase_microfluidic_KK1KK2_combi"
MODEL_SHA256 = "b33bdbdab395a27051b1bf10897b66888abcc24da3b3ddd41814fea970177cd6"
MODEL_SOURCE = (
    ROOT / "data" / "confinedmig_cellTrack" / "CellPose_TrainData"
    / "KK1KK2_combiModel" / "models" / MODEL_NAME
)

ISCC_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
    Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def app_version() -> str:
    sys.path.insert(0, str(SRC))
    from corridor.app_meta import APP_VERSION

    return APP_VERSION


def stage_model() -> Path:
    """Copy the trained model into the assets the bundle will include."""
    target_dir = ASSETS / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / MODEL_NAME
    if not target.exists():
        if not MODEL_SOURCE.exists():
            raise SystemExit(
                f"The Cellpose model is missing.\nExpected it at: {MODEL_SOURCE}"
            )
        shutil.copy2(MODEL_SOURCE, target)
    digest = sha256(target)
    if digest != MODEL_SHA256:
        raise SystemExit(
            f"The staged model has checksum {digest}, expected {MODEL_SHA256}. "
            "Refusing to ship a model that is not the one this was validated against."
        )
    print(f"model staged and verified: {target.name}  {digest[:16]}...")
    return target


def find_iscc() -> Path | None:
    for candidate in ISCC_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-app", action="store_true", help="reuse an existing build")
    parser.add_argument("--skip-installer", action="store_true")
    args = parser.parse_args()

    version = app_version()
    print(f"Corridor {version}")

    python = sys.executable
    run([python, str(ROOT / "scripts" / "make_icon.py")])
    stage_model()

    if not args.skip_app:
        if DIST.exists():
            shutil.rmtree(DIST, ignore_errors=True)
        run([
            python, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--distpath", str(DIST), "--workpath", str(WORK),
            str(ROOT / "packaging" / "corridor.spec"),
        ], cwd=str(ROOT))

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
        "model": {"name": MODEL_NAME, "sha256": MODEL_SHA256},
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
