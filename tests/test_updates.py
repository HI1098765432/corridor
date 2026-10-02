"""Update checking: version comparison, consent, and failing quietly.

Two properties matter more than the feature itself. The check must never run
without the user having said yes, because this application otherwise makes no
network calls at all and a microscope workstation holding unpublished data is
often offline on purpose. And every failure must be silent: no network, a proxy,
a rate limit and a malformed reply all mean "no update information", never an
error in front of somebody mid-experiment.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from corridor.core import updates


class _Store:
    """The settings interface the real store exposes, in memory."""

    def __init__(self, initial=None):
        self._values = dict(initial or {})

    def get_setting(self, key, default=None):
        return self._values.get(key, default)

    def set_setting(self, key, value):
        self._values[key] = value


# --------------------------------------------------------------------------
# Version comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate, current, expected",
    [
        ("1.3.0", "1.2.0", True),
        ("v1.3.0", "1.2.0", True),
        ("1.2.1", "1.2.0", True),
        ("1.2.0", "1.2.0", False),
        ("1.1.9", "1.2.0", False),
        ("2.0.0", "1.9.9", True),
    ],
)
def test_version_comparison(candidate, current, expected):
    assert updates.is_newer(candidate, current) is expected


def test_versions_compare_numerically_not_as_text():
    """The release where a string comparison would be wrong.

    "1.10.0" sorts before "1.9.0" as text, so a tenth minor release would be
    silently treated as older than the ninth -- in exactly the release that has
    been shipping longest.
    """
    assert updates.is_newer("1.10.0", "1.9.0") is True
    assert updates.is_newer("1.9.0", "1.10.0") is False


def test_unparseable_versions_are_never_newer():
    """An unreadable tag must not be announced as an upgrade."""
    assert updates.is_newer("nightly", "1.2.0") is False
    assert updates.is_newer("", "1.2.0") is False
    assert updates.is_newer("1.2.0", "not a version") is False


# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------


def test_checking_is_off_until_the_user_says_yes():
    """Absence of an answer is not consent."""
    store = _Store()
    assert updates.is_enabled(store) is False
    assert updates.should_check(store) is False
    assert updates.has_been_asked(store) is False


def test_declining_is_remembered_and_stays_declined():
    store = _Store()
    updates.record_choice(store, enabled=False)
    assert updates.has_been_asked(store) is True
    assert updates.is_enabled(store) is False
    assert updates.should_check(store) is False


def test_accepting_enables_the_check():
    store = _Store()
    updates.record_choice(store, enabled=True)
    assert updates.should_check(store) is True


def test_a_dismissed_version_is_not_announced_again():
    store = _Store()
    updates.remember_seen(store, "1.3.0")
    assert updates.already_seen(store, "1.3.0") is True
    assert updates.already_seen(store, "1.4.0") is False


# --------------------------------------------------------------------------
# Every failure is silent
# --------------------------------------------------------------------------


def _raise(exc):
    def _fail(*args, **kwargs):
        raise exc
    return _fail


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("no network"),
        urllib.error.HTTPError("u", 403, "rate limited", {}, None),
        TimeoutError("slow"),
        OSError("a proxy ate it"),
    ],
)
def test_a_failed_check_returns_nothing_rather_than_raising(monkeypatch, failure):
    monkeypatch.setattr(updates.urllib.request, "urlopen", _raise(failure))
    assert updates.check(timeout=0.1) is None


def test_a_malformed_reply_returns_nothing(monkeypatch):
    class _Response:
        def read(self):
            return b"this is not json"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *a, **k: _Response())
    assert updates.check(timeout=0.1) is None


def test_a_reply_without_a_version_returns_nothing(monkeypatch):
    payload = json.dumps({"name": "some release", "body": "notes"}).encode()

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *a, **k: _Response())
    assert updates.check(timeout=0.1) is None


def test_a_good_reply_is_parsed(monkeypatch):
    payload = json.dumps({
        "tag_name": "v1.3.0",
        "html_url": "https://example.invalid/releases/v1.3.0",
        "body": "line one\n\nline two",
        "published_at": "2026-10-01T00:00:00Z",
        "assets": [
            {"name": "notes.txt", "browser_download_url": "https://x/n", "size": 10},
            {"name": "Corridor-1.3.0-Setup.exe",
             "browser_download_url": "https://x/setup", "size": 123},
        ],
    }).encode()

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *a, **k: _Response())
    release = updates.check(timeout=0.1)

    assert release is not None
    assert release.version == "1.3.0"
    assert release.installer_url == "https://x/setup"
    assert release.installer_bytes == 123
    assert release.summary(lines=2) == "line one\nline two"


def test_the_module_never_downloads_or_runs_anything():
    """The deliberate refusal, pinned as a test.

    A program that fetches an executable and launches it is the exact shape of
    the thing every security guide warns about, and "but it is our own
    executable" is what an attacker who has compromised the release channel is
    relying on. This module reports what exists and hands over a link; a human
    downloads it, Windows checks the signature, and a human decides.
    """
    source = (updates.__file__ or "")
    text = open(source, encoding="utf-8").read() if source else ""
    for forbidden in ("subprocess", "os.startfile", "ShellExecute", "urlretrieve"):
        assert forbidden not in text, f"{forbidden} must not appear in the updater"
