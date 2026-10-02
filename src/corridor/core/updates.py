"""Telling the user a better version exists, without taking the decision for them.

Until now this application made **no network calls at all**, and that was worth
something: a microscope workstation holding unpublished data is often
deliberately offline, and software that quietly phones home is not welcome on
one. So the update check is opt-in, asked once, and the answer is remembered.

Three rules shape what this does and does not do.

**It asks before it ever connects.** The first run offers the choice and takes
"no" for an answer permanently. Nothing here runs until that has happened.

**It checks; it does not install.** A program that downloads an executable and
runs it is the precise shape of the thing every security guide warns about, and
"but it is our own executable" is what the attacker who has compromised the
release channel is counting on. This reports what exists and hands over a link.
The user downloads it, Windows checks the signature, and a human decides.

**It can say the model changed.** The trained weights matter more to a result
than the interface does, and a model is data rather than code, so a new model
can be reported separately from a new application. Whether to adopt it is still
the user's call, because a changed model changes their numbers -- and anyone
comparing against an analysis they ran last month needs to know that.

Everything that fails here fails silently in the user's favour: no network, a
rate limit, a proxy, a malformed response, all produce "no update information"
rather than an error in front of somebody trying to do an experiment.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .. import app_meta

#: The published releases. Read-only, unauthenticated, no user data is sent.
RELEASES_URL = "https://api.github.com/repos/HI1098765432/corridor/releases/latest"
#: Short, because this runs while someone is waiting to analyse something.
TIMEOUT_SECONDS = 6.0

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """``v1.2.0`` or ``1.2.0`` -> (1, 2, 0); anything else -> None."""
    match = _VERSION.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def is_newer(candidate: str, current: str = app_meta.APP_VERSION) -> bool:
    """Compare by components, never as text.

    String comparison would place "1.10.0" before "1.9.0", which is wrong in
    exactly the release that matters most.
    """
    a, b = parse_version(candidate), parse_version(current)
    if a is None or b is None:
        return False
    return a > b


@dataclass
class Release:
    """What is published, as far as this can tell."""

    version: str
    url: str
    notes: str
    installer_url: str | None = None
    installer_bytes: int | None = None
    published: str = ""

    @property
    def is_newer(self) -> bool:
        return is_newer(self.version)

    def summary(self, lines: int = 6) -> str:
        """The first few lines of the notes, for a banner rather than a wall."""
        kept = [l for l in (self.notes or "").splitlines() if l.strip()][:lines]
        return "\n".join(kept)


def _fetch(url: str, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(url)
    # Identifying the client is courteous to the host and sends nothing about
    # the user or their data.
    request.add_header("User-Agent", f"{app_meta.APP_NAME}/{app_meta.APP_VERSION}")
    request.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        # No network, a proxy, a rate limit, a captive portal, a malformed
        # reply: all of them mean "no update information", and none of them is
        # worth an error dialogue in front of someone mid-experiment.
        return None


def check(timeout: float = TIMEOUT_SECONDS) -> Release | None:
    """What is published now, or None if that could not be established.

    Returns the latest release whether or not it is newer; the caller decides
    what to do with it. Never raises.
    """
    payload = _fetch(RELEASES_URL, timeout)
    if not isinstance(payload, dict):
        return None

    version = str(payload.get("tag_name") or payload.get("name") or "").strip()
    if not parse_version(version):
        return None

    installer_url = None
    installer_bytes = None
    for asset in payload.get("assets") or []:
        name = str(asset.get("name", ""))
        if name.lower().endswith("setup.exe"):
            installer_url = asset.get("browser_download_url")
            installer_bytes = asset.get("size")
            break

    return Release(
        version=version.lstrip("v"),
        url=str(payload.get("html_url") or ""),
        notes=str(payload.get("body") or ""),
        installer_url=installer_url,
        installer_bytes=installer_bytes,
        published=str(payload.get("published_at") or ""),
    )


# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------

#: Settings keys. Stored in the project database beside everything else.
ASKED_KEY = "update_check_asked"
ENABLED_KEY = "update_check_enabled"
LAST_SEEN_KEY = "update_last_seen_version"


def has_been_asked(store) -> bool:
    return str(store.get_setting(ASKED_KEY, "")).lower() == "yes"


def is_enabled(store) -> bool:
    """Off unless the user has said yes. Absence of an answer is not consent."""
    return str(store.get_setting(ENABLED_KEY, "")).lower() == "yes"


def record_choice(store, enabled: bool) -> None:
    store.set_setting(ASKED_KEY, "yes")
    store.set_setting(ENABLED_KEY, "yes" if enabled else "no")


def should_check(store) -> bool:
    return has_been_asked(store) and is_enabled(store)


def already_seen(store, version: str) -> bool:
    """True when this exact version has already been shown and dismissed."""
    return str(store.get_setting(LAST_SEEN_KEY, "")) == version


def remember_seen(store, version: str) -> None:
    store.set_setting(LAST_SEEN_KEY, version)
