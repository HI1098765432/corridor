"""Record the signed installer's real checksum in the build manifest.

Signing rewrites the binary, so the checksum computed at build time describes a
file that no longer exists. The published checksum has to be the one a user can
reproduce from the file they downloaded, which is the signed one.

Without this step the manifest and the artefact disagree, and there are only
two ways that ends: the publish step refuses to run (what happens today), or
somebody "fixes" it by relaxing the check and the published checksum stops
meaning anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "build" / "build_manifest.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbprint", default="")
    ap.add_argument("--subject", default="")
    ap.add_argument("--self-signed", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"No build manifest at {MANIFEST}.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))

    installer = ROOT / "build" / "installer" / manifest["installer"]["filename"]
    if not installer.exists():
        raise SystemExit(f"Installer not found: {installer}")

    before = manifest["installer"]["sha256"]
    after = sha256(installer)
    manifest["installer"]["bytes"] = installer.stat().st_size
    manifest["installer"]["sha256"] = after
    if args.thumbprint:
        manifest["signing"] = {
            "subject": args.subject,
            "thumbprint": args.thumbprint,
            "self_signed": bool(args.self_signed),
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"installer: {installer.name}")
    print(f"  before signing: {before}")
    print(f"  after signing:  {after}")
    print(f"  bytes:          {manifest['installer']['bytes']:,}")
    if args.thumbprint:
        print(f"  signed by:      {args.subject} ({args.thumbprint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
