"""What the interface says about the segmentation model.

In 2.0 the model is not a setting (contract §2), so the interface never offers
a picker; it only *reports* which validated model production would use and
whether the file on this machine is the one the registry vouches for. Every
screen that mentions the model reads it from here, so a screen cannot drift
back to describing a path somebody once typed.

Two levels of cost, on purpose:

*   :func:`registered_model` reads the registry only (no hashing). It is what
    the dataset screen shows while a file is being inspected.
*   :func:`verified_model` hashes the 26.6 MB file through
    ``model_registry.resolve_model`` (cached by size and mtime there). It is
    what Settings and About show, and what an analysis checks before it starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core import model_registry

#: Characters of the SHA-256 shown in compact rows. Twelve hex digits are
#: enough for a reader to compare two runs by eye; the full value is always in
#: the tooltip.
SHA_PREFIX_CHARS = 12


@dataclass(frozen=True)
class ModelStatus:
    """A displayable description of the production model for one dimensionality."""

    dimensionality: str
    model_id: str | None
    model_version: str | None
    sha256: str | None
    #: True only when the file was found *and* its hash matched.
    verified: bool
    #: The resolved file, when one was verified.
    path: Path | None
    #: Why it is not verified (the ModelUnavailable text), or "" when it is.
    message: str
    developer_override: bool = False

    @property
    def sha_prefix(self) -> str | None:
        return f"{self.sha256[:SHA_PREFIX_CHARS]}…" if self.sha256 else None

    @property
    def status_text(self) -> str:
        if self.developer_override:
            return "developer override (not validated)"
        if self.verified:
            return "verified"
        return "missing or does not match"


def registered_model(dimensionality: str = "2D") -> ModelStatus:
    """The registry's entry, without touching the model file.

    ``verified`` is False here by construction: nothing was hashed.
    """
    try:
        spec = model_registry.production_spec(dimensionality)
    except model_registry.ModelUnavailable as exc:
        return ModelStatus(dimensionality, None, None, None, False, None, str(exc))
    return ModelStatus(
        dimensionality,
        spec.model_id,
        spec.model_version,
        spec.sha256,
        False,
        None,
        "",
    )


def verified_model(dimensionality: str = "2D") -> ModelStatus:
    """Resolve and hash-check the production model. Never raises.

    A ModelUnavailable is turned into a status carrying its message verbatim:
    the dialog that shows it must say exactly what the contract says, plus
    the paths that were tried, and never suggest another model.
    """
    try:
        resolved = model_registry.resolve_model(dimensionality)
    except model_registry.ModelUnavailable as exc:
        spec_status = registered_model(dimensionality)
        return ModelStatus(
            dimensionality,
            spec_status.model_id,
            spec_status.model_version,
            spec_status.sha256,
            False,
            None,
            str(exc),
        )
    spec = resolved.spec
    return ModelStatus(
        dimensionality,
        spec.model_id,
        spec.model_version,
        resolved.sha256,
        True,
        resolved.path,
        "",
        developer_override=bool(resolved.developer_override),
    )
