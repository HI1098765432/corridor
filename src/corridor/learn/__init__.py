"""Learning from the data already in hand, rather than from more of it.

The 71 hand-labelled images are not 71 unrelated pictures. Each still's ImageJ
label records the movie and time point it was cut from, and read that way they
are **sparse, irregular samples of eight fields in seven experiments**: a few
stills per field, from one to sixty frames apart, two groups out of time order
in filename order, and one experiment holding two crops of the same size. An
earlier version of this docstring called them "six time-lapse sequences,
contiguous frames"; they are not, and the methods built on that reading were
re-derived on the true order (``docs/RESEARCH_V2.md``).

That matters because of what it makes available. A human tracing frame *t* sees
frame *t* alone. A cell, however, is one object persisting through time: drawn
at the nearest earlier and later labelled times of the same field, close enough
in time that "the same cell" still means something, it is evidence about frame
*t* that the annotator never used. The constraint supplying that evidence is
time and physics, not the model's own opinion, so using it is not the model
marking its own homework.

- :mod:`.sequences` -- stills grouped by experiment and crop, in true time
  order, registered, with elapsed minutes between them.
- :mod:`.brackets` -- the time-aware rule for which cells were provably not
  drawn, shared by the corrected reference and the label-completeness ceiling.
- :mod:`.reconstruct` -- linking, temporal consensus and boundary refinement.
"""

from __future__ import annotations

__all__ = ["sequences", "brackets", "reconstruct"]
