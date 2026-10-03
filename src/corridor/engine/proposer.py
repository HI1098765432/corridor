"""Bot 3 -- the proposer interface.

One interface, two intended backends (``docs/ENGINE_4D.md`` SS2, Bot 3):

* the locked CP3 lab model (``jhu_confined_cp3_combi``), **in-process**,
  resolved and SHA-256-verified through ``corridor.core.model_registry`` -- a
  bare model name is never passed to Cellpose, only a verified absolute path
  (``docs/NEXT_GENERATION.md`` SS2);
* Cellpose 4 ``cpsam_v2``, as a **worker process** in its own ``.venv-cp4``,
  exchanging arrays through files, because CP3 and CP4 cannot share one Python
  environment (different major versions; a CP3 checkpoint does not load in
  CP4).

**This module imports neither torch nor cellpose, by design.**  It defines the
data a proposer returns (``Proposal``), the protocol a backend satisfies
(``Proposer``), and a model-free ``GivenMasksProposer`` that returns masks
computed elsewhere -- an imported label image, a fixture, a unit test -- so the
engine and its whole test suite run without any neural network present.  The
two model-backed proposers are declared here as the interface; their bodies,
which must import Cellpose, live in the integration package
(``cellpose4_worker.py`` and the CP3 in-process backend) and raise
:class:`ProposerUnavailable` until wired.

A proposer *proposes*; it decides nothing.  It retains the mask, the
cell-probability field and the flows, with the normalisation recorded so a
reviewer can reproduce exactly what the model saw.  Every judgement about those
proposals is the referee's (Bot 8, ``consensus``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


class ProposerBackend(str, Enum):
    """Which proposer produced a :class:`Proposal`.  Recorded on every one, so
    a consensus decision can name the evidence's origin (a strong CP3 interior
    is a different claim from a zero-shot cpsam_v2 one)."""

    CP3_INPROCESS = "cp3_inprocess"
    CELLPOSE4_WORKER = "cellpose4_worker"
    GIVEN_MASKS = "given_masks"


#: Convenience alias for the model-free backend's name, re-exported at package
#: level because tests and the pipeline reference it directly.
GIVEN_MASKS = ProposerBackend.GIVEN_MASKS.value


class ProposerUnavailable(RuntimeError):
    """A model-backed proposer was asked to run before it was wired, or its
    model/weights could not be resolved.  Mirrors
    ``core.model_registry.ModelUnavailable``: a proposer never silently falls
    back to a different model, so the failure is explicit and names what was
    missing."""


# --------------------------------------------------------------------------
# What a proposer returns
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Normalization:
    """How the image was scaled before the model saw it.

    Recorded, not re-applied downstream: it exists so a reviewer can reproduce
    the model's input and so Experiment B' (``cpsam_v2`` on the atlas residual
    ``V' - B``) is distinguishable from a raw run by a field, not by guesswork.
    ``lower_value``/``upper_value`` are the raw intensities mapped to 0 and 1;
    with a percentile method they are the measured percentile values, so the
    exact mapping is on the record even when the percentiles are the default.
    """

    method: str = "none"  # "percentile" | "none" | "given"
    lower_percentile: float | None = None
    upper_percentile: float | None = None
    lower_value: float | None = None
    upper_value: float | None = None
    #: True when the proposer ran on ``V' - B`` (the static atlas background
    #: subtracted) rather than the raw registered image -- Experiment B'.
    background_subtracted: bool = False

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_value": self.lower_value,
            "upper_value": self.upper_value,
            "background_subtracted": self.background_subtracted,
        }


@dataclass(frozen=True)
class Proposal:
    """One proposer's output for one 2-D frame or one Z volume.

    ``masks`` is an integer label image (0 = background), shape ``(Y, X)`` for
    a frame or ``(Z, Y, X)`` for a volume.  ``cellprob`` is the cell-probability
    field at the same shape, or ``None`` for a proposer that has none (a given
    label image).  ``flows`` is the backend's flow field, kept but never
    interpreted here (its layout is the backend's; the referee does not read
    it).  ``weights_sha256`` records the hash-checked weights that produced the
    masks, or ``None`` for ``GIVEN_MASKS`` -- again so the evidence names its
    origin and a bare model name can never masquerade as a validated one.
    """

    masks: np.ndarray
    cellprob: np.ndarray | None
    flows: np.ndarray | None
    normalization: Normalization
    backend: str = GIVEN_MASKS
    weights_sha256: str | None = None

    def __post_init__(self) -> None:
        masks = np.asarray(self.masks)
        if masks.ndim not in (2, 3):
            raise ValueError(
                f"Proposal.masks must be (Y, X) or (Z, Y, X); got shape {masks.shape}"
            )
        if self.cellprob is not None and np.asarray(self.cellprob).shape != masks.shape:
            raise ValueError(
                "Proposal.cellprob must match masks shape "
                f"{masks.shape}; got {np.asarray(self.cellprob).shape}"
            )
        object.__setattr__(self, "masks", masks)

    @property
    def is_volume(self) -> bool:
        return self.masks.ndim == 3

    def labels(self) -> list[int]:
        """Every object id present, background excluded."""
        return [int(v) for v in np.unique(self.masks) if v]

    def n_objects(self) -> int:
        return len(self.labels())

    def mean_cellprob(self, label: int) -> float | None:
        """Mean cell-probability over one object's pixels, or ``None`` when the
        proposer carried no probability field.  This is the single scalar the
        referee reads as the proposer's strength for that object; it is a
        measurement over the object's own footprint, not a field-wide number."""
        if self.cellprob is None:
            return None
        sel = self.masks == int(label)
        if not sel.any():
            return None
        return float(np.asarray(self.cellprob)[sel].mean())

    def to_dict(self) -> dict:
        """Evidence form: shapes, counts, normalisation and origin -- never the
        arrays themselves, which belong in ``masks.npz`` / the proposal store,
        not in a CSV cell."""
        return {
            "backend": self.backend,
            "weights_sha256": self.weights_sha256,
            "mask_shape": list(self.masks.shape),
            "n_objects": self.n_objects(),
            "has_cellprob": self.cellprob is not None,
            "has_flows": self.flows is not None,
            "normalization": self.normalization.to_dict(),
        }


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


@runtime_checkable
class Proposer(Protocol):
    """A segmenter that proposes masks for one timepoint (and, with Z, one
    volume).  ``backend`` names which one, for the evidence record.

    ``t`` and ``z`` identify the slice being proposed so a proposer backed by
    precomputed results (``GivenMasksProposer``, an imported label stack) can
    look the right one up; a model-backed proposer segments ``image`` and may
    ignore them.
    """

    backend: str

    def propose(
        self, image: np.ndarray, *, t: int = 0, z: int | None = None
    ) -> Proposal: ...


# --------------------------------------------------------------------------
# The model-free proposer the engine and its tests run on
# --------------------------------------------------------------------------


@dataclass
class GivenMasksProposer:
    """Return masks produced elsewhere, touching no model.

    ``masks`` is a stack indexed by time (and optionally Z): ``(Y, X)`` for a
    single frame, ``(T, Y, X)`` for a movie, or ``(T, Z, Y, X)`` for a 4-D
    stack.  ``cellprob`` may be given at the same leading shape so a test can
    exercise the referee's proposer-strength rule; without it, ``mean_cellprob``
    is ``None`` and the referee treats the proposer as neutral (it gave a mask
    but asserts no probability).  This is also the production path for an
    imported label image (``ImportConfig.labels_path``): the engine's
    verification stages still run on masks a human, or another tool, supplied.
    """

    masks: np.ndarray
    cellprob: np.ndarray | None = None
    normalization: Normalization = field(default_factory=lambda: Normalization(method="given"))
    backend: str = GIVEN_MASKS

    def __post_init__(self) -> None:
        self.masks = np.asarray(self.masks)
        if self.masks.ndim not in (2, 3, 4):
            raise ValueError(
                "GivenMasksProposer.masks must be (Y,X), (T,Y,X) or (T,Z,Y,X); "
                f"got shape {self.masks.shape}"
            )
        if self.cellprob is not None:
            self.cellprob = np.asarray(self.cellprob)
            if self.cellprob.shape != self.masks.shape:
                raise ValueError(
                    "GivenMasksProposer.cellprob must match masks shape "
                    f"{self.masks.shape}; got {self.cellprob.shape}"
                )

    def _slice(self, arr: np.ndarray | None, t: int, z: int | None):
        if arr is None:
            return None
        if arr.ndim == 2:  # (Y, X): one frame, t and z ignored
            return arr
        if arr.ndim == 3:  # (T, Y, X)
            return arr[t]
        # (T, Z, Y, X): a volume per timepoint, or one Z plane of it
        return arr[t] if z is None else arr[t, z]

    def propose(
        self, image: np.ndarray, *, t: int = 0, z: int | None = None
    ) -> Proposal:
        masks = self._slice(self.masks, t, z)
        cellprob = self._slice(self.cellprob, t, z)
        return Proposal(
            masks=np.asarray(masks),
            cellprob=None if cellprob is None else np.asarray(cellprob),
            flows=None,
            normalization=self.normalization,
            backend=self.backend,
            weights_sha256=None,
        )


# --------------------------------------------------------------------------
# The two model-backed backends -- interface only, bodies in the integration
# package (they import Cellpose, which this module must not).
# --------------------------------------------------------------------------


@dataclass
class CP3LabProposer:
    """The locked CP3 lab model, run in-process.

    The model id and dimensionality resolve through
    ``corridor.core.model_registry`` (SHA-256 verified before loading, no
    fallback).  The body imports Cellpose and so is not written here; this
    class fixes the constructor and the :class:`Proposer` signature, and
    refuses to run until the integration package provides the backend.
    """

    model_id: str = "jhu_confined_cp3_combi"
    dimensionality: str = "2D"
    backend: str = ProposerBackend.CP3_INPROCESS.value

    def propose(
        self, image: np.ndarray, *, t: int = 0, z: int | None = None
    ) -> Proposal:
        raise ProposerUnavailable(
            "The in-process CP3 proposer is the declared interface only; its body "
            "imports Cellpose and lives in the integration package. Resolve and "
            "verify the model via corridor.core.model_registry there, never a bare "
            "model name."
        )


@dataclass
class Cellpose4WorkerProposer:
    """Cellpose 4 ``cpsam_v2`` as an out-of-process worker.

    CP3 and CP4 cannot share one environment, so this backend launches a worker
    in ``.venv-cp4`` and exchanges arrays through files; ``weights_path`` is an
    explicit, hash-checked absolute path, never a bare ``cpsam_v2`` name.  The
    engine process never imports Cellpose.  Body in ``cellpose4_worker.py``;
    this class is the interface and refuses until that is wired.
    """

    weights_path: str | None = None
    weights_sha256: str | None = None
    venv: str = ".venv-cp4"
    #: Propose on ``V' - B`` (atlas residual) -- Experiment B' -- when True.
    background_subtracted: bool = False
    backend: str = ProposerBackend.CELLPOSE4_WORKER.value

    def propose(
        self, image: np.ndarray, *, t: int = 0, z: int | None = None
    ) -> Proposal:
        raise ProposerUnavailable(
            "The Cellpose-4 worker proposer is the declared interface only; its "
            "body launches a .venv-cp4 worker and lives in cellpose4_worker.py. A "
            "hash-checked absolute weights path is required, never a bare name."
        )


__all__ = [
    "ProposerBackend",
    "GIVEN_MASKS",
    "ProposerUnavailable",
    "Normalization",
    "Proposal",
    "Proposer",
    "GivenMasksProposer",
    "CP3LabProposer",
    "Cellpose4WorkerProposer",
]
