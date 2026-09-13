"""Publish the built installer as a GitHub release, then verify the download.

Publishing without verifying is how a corrupt or truncated asset reaches a
user. This uploads the installer and then fetches it back over plain HTTPS,
exactly as a browser would, and compares the checksum of what came down with
the checksum of what was built.

The GitHub token is taken from the git credential helper, so no secret is
stored in the repository or passed on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def credential(host: str = "github.com") -> tuple[str, str]:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input=f"protocol=https\nhost={host}\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return values.get("username", ""), values.get("password", "")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def api(token: str, method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def upload_asset(token: str, upload_url: str, path: Path, content_type: str) -> dict:
    url = upload_url.split("{", 1)[0] + f"?name={path.name}"
    payload = path.read_bytes()
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(payload)))
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def download(url: str, destination: Path) -> int:
    """Fetch over plain HTTPS with no credentials, the way a browser would."""
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "Mozilla/5.0 (corridor release verification)")
    total = 0
    with urllib.request.urlopen(request, timeout=1800) as response, open(
        destination, "wb"
    ) as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="HI1098765432/corridor")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--notes", default=str(ROOT / "docs" / "RELEASE_NOTES.md"))
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()

    manifest_path = BUILD / "build_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("No build manifest. Run scripts/build_release.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    version = manifest["version"]
    tag = args.tag or f"v{version}"

    installer = BUILD / "installer" / manifest["installer"]["filename"]
    if not installer.exists():
        raise SystemExit(f"Installer not found: {installer}")
    local_digest = sha256(installer)
    if local_digest != manifest["installer"]["sha256"]:
        raise SystemExit("The installer on disk does not match the build manifest.")
    print(f"installer: {installer.name}")
    print(f"size:      {installer.stat().st_size:,} bytes")
    print(f"sha256:    {local_digest}")

    checksum_file = installer.with_suffix(".exe.sha256")
    checksum_file.write_text(f"{local_digest} *{installer.name}\n", encoding="utf-8")

    _, token = credential()
    if not token:
        raise SystemExit("No GitHub credential available.")

    notes_path = Path(args.notes)
    notes = notes_path.read_text(encoding="utf-8-sig") if notes_path.exists() else ""

    # Replace an existing release for this tag so re-publishing is safe.
    try:
        existing = api(token, "GET", f"{API}/repos/{args.repo}/releases/tags/{tag}")
        print(f"replacing existing release {tag}")
        api(token, "DELETE", f"{API}/repos/{args.repo}/releases/{existing['id']}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    release = api(
        token, "POST", f"{API}/repos/{args.repo}/releases",
        {
            "tag_name": tag,
            "name": f"Corridor {version}",
            "body": notes,
            "draft": bool(args.draft),
            "prerelease": False,
            "generate_release_notes": False,
        },
    )
    print(f"release:   {release['html_url']}")

    print("uploading installer...")
    asset = upload_asset(
        token, release["upload_url"], installer, "application/octet-stream"
    )
    upload_asset(token, release["upload_url"], checksum_file, "text/plain")
    print(f"uploaded:  {asset['browser_download_url']}")

    # -- verify by downloading it back, unauthenticated ---------------------
    print("\nverifying the published download...")
    scratch = BUILD / "verify"
    scratch.mkdir(parents=True, exist_ok=True)
    fetched = scratch / installer.name
    size = download(asset["browser_download_url"], fetched)
    fetched_digest = sha256(fetched)

    print(f"  downloaded bytes: {size:,}")
    print(f"  expected bytes:   {installer.stat().st_size:,}")
    print(f"  downloaded sha256: {fetched_digest}")
    print(f"  built sha256:      {local_digest}")
    ok = size == installer.stat().st_size and fetched_digest == local_digest
    print(f"  match: {'YES' if ok else 'NO'}")

    print("\nDESKTOP APP DOWNLOAD:")
    print(f"  {asset['browser_download_url']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
