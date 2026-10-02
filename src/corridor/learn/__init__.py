"""Learning from the data already in hand, rather than from more of it.

Everything this package does rests on one observation. The 71 hand-labelled
images are not 71 independent pictures: they are **six time-lapse sequences**,
contiguous frames of the same field, and nobody has ever used the fact that
they are sequences. Neither has anyone used the 74 unlabelled frames in
``sample_data``.

That matters because of what it makes available. A human tracing frame *t* sees
frame *t* alone. A cell, however, is one object persisting through time: its
outline at *t* is over-determined by its outline at *t-1* and *t+1* together
with smooth motion and a near-conserved area. Solving for the mask *sequence*
that best explains the image evidence under those constraints yields a
per-frame mask that can be **better than the single-frame tracing** -- not
because the algorithm is cleverer than the annotator, but because it is given
evidence the annotator never had.

This is the whole argument, and it is also what keeps the method honest. The
constraint supplying the supervision is time and physics, not the model's own
opinion, so training on the result is not the model marking its own homework.
The same principle already holds at the level of positions rather than
outlines: track-guided recovery fills 10 of 13 known holes with zero false
positives purely by interpolating between the observations either side of a gap
(``scripts/experiment_recovery.py``).
"""

from __future__ import annotations

__all__ = ["sequences", "reconstruct"]
