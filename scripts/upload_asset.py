"""Add or replace one asset on an existing release, without touching the rest.

``publish_release.py`` deletes and recreates the whole release, which means
re-uploading a 265 MB installer to change a 340 KB source archive. When only one
file has changed, this replaces just that file: the installer keeps its URL, its
checksum and its download count.

The upload is verified by fetching the published file back unauthenticated and
comparing it byte for byte with what was on disk. An asset that was uploaded but
never fetched is an assumption, not a download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def credential(host: str = "github.com") -> str:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input=f"protocol=https\nhost={host}\n\n",
        capture_output=True, text=True, check=True,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return values.get("password", "")


def api(token: str, method: str, url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    if data:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--repo", default="HI1098765432/corridor")
    ap.add_argument("--tag", default="v1.2.0")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    token = credential()
    if not token:
        raise SystemExit("No GitHub credential available.")

    release = api(token, "GET", f"{API}/repos/{args.repo}/releases/tags/{args.tag}")
    print(f"release: {release['html_url']}")

    # Replacing means deleting the old one first; GitHub will otherwise store a
    # second asset with a mangled name and serve whichever it likes.
    for asset in release.get("assets", []):
        if asset["name"] == path.name:
            print(f"removing the previous {asset['name']} ({asset['size']:,} bytes)")
            api(token, "DELETE", f"{API}/repos/{args.repo}/releases/assets/{asset['id']}")

    upload_url = release["upload_url"].split("{")[0] + f"?name={path.name}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    request = urllib.request.Request(upload_url, data=path.read_bytes(), method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request, timeout=600) as response:
        asset = json.loads(response.read())
    print(f"uploaded: {asset['browser_download_url']}")

    print("\nverifying by fetching it back...")
    fetch = urllib.request.Request(
        asset["browser_download_url"], headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(fetch, timeout=600) as response:
        fetched = response.read()
    local = path.read_bytes()
    ok = fetched == local
    print(f"  downloaded {len(fetched):,} bytes, local {len(local):,} bytes")
    print(f"  sha256 {hashlib.sha256(fetched).hexdigest()}")
    print(f"  match: {'YES' if ok else 'NO'}")

    print(f"\nDOWNLOAD:\n  {asset['browser_download_url']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
