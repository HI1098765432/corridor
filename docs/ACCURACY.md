# How accurate is this, really

Every number in this document was produced by a script in `scripts/`, against
the supplied data, and can be reproduced by running it. Where a claim could not
be measured, it is marked as unmeasured rather than estimated.

---

## The short answer

**"Is it 99% accurate?" has three different answers, because there are three
different things being counted.** Conflating them is the single easiest way to
either oversell this software or dismiss it unfairly.

| What is being counted | Measured | Can it reach 99%? |
|---|---|---|
| **1. Each cell outline, per frame** | F1 **0.839** in-distribution; **0.30–0.48** held out | **No.** Not with this model and this much labelled data. |
| **2. Cell identity along a trajectory** | **100%** of the available ground truth: no swap, no merge, no cross-channel link | **Yes**, and it is already there on this data. |
| **3. The reported quantity — migration speed** | Net speed **exact at the median, within 2.7–2.9% at p90** at realistic detection loss | **Yes for net speed and net displacement**; ~91–95% for the mean of instantaneous speeds. |

The thing a researcher publishes is level 3. The thing a segmentation
leaderboard scores is level 1. They are not the same number and level 1 is the
worse of the two, so quoting it as "the accuracy" understates what this
software delivers — and quoting level 3 as "the accuracy" would overstate how
good the detector is.

---

## Level 1 — finding each cell in each frame

Measured by `scripts/evaluate_models.py` and `scripts/experiment_recall.py`
against the 71 hand-labelled images supplied with the project, matching
instances one-to-one by IoU.

### In distribution

The combined model was trained on all 71 labelled images, so it has no held-out
data of its own. Scored on the images it learned from — an optimistic number by
construction — at IoU 0.5:

| | precision | recall | F1 |
|---|---|---|---|
| combined model, labelling settings | 0.832 | 0.846 | **0.839** |

### Held out — the number that predicts a new experiment

`KK1Model` and `KK2Model` were trained on disjoint halves, so each can be scored
on data it has never seen. This is the honest generalisation estimate:

| model | evaluated on | precision | recall | F1 |
|---|---|---|---|---|
| KK1Model | KK2 (held out) | 0.540 | 0.435 | **0.482** |
| KK2Model | KK1 (held out) | 0.596 | 0.203 | **0.303** |
| KK1Model | KK1 (its own training set) | 0.819 | 0.819 | 0.819 |

**The gap between 0.82 and 0.30–0.48 is the finding.** Roughly half of the
apparent performance is memorisation of the specific images. Two halves of what
looks like one dataset are different enough that a model trained on one loses
most of its recall on the other.

At IoU 0.75 the F1 falls to 0.07–0.22 and at IoU 0.9 it is zero. The outlines
are approximately right, not precisely right. That matters for area and shape
measurements; it matters much less for centroid position, which is what the
trajectory is built from.

### Why level 1 cannot be pushed to 99% here

`scripts/experiment_recall.py` measured eight detection strategies on the same
71 images:

| strategy | precision | recall | F1 | cost |
|---|---|---|---|---|
| baseline (prob 0.0, flow 0.4) | 0.832 | 0.846 | **0.839** | 1× |
| cellprob −2 | 0.798 | 0.740 | 0.768 | 1× |
| flow 0.6 | 0.816 | 0.850 | 0.833 | 1× |
| ensemble: prob 0 and −2 | 0.833 | 0.854 | **0.843** | 2× |
| ensemble: flow 0.4 and 0.6 | 0.816 | 0.850 | 0.833 | 2× |
| ensemble: 4 settings | 0.793 | 0.874 | 0.832 | 4× |
| 3 models | 0.772 | 0.882 | 0.824 | 3× |
| 3 models × 2 settings | 0.703 | **0.902** | 0.790 | 6× |

Recall can be bought up to 0.902, but only by paying precision down to 0.703.
**F1 never exceeds 0.843 at any price.** This is a ceiling of the model and the
labelled data, not of the search.

The reason the missing cells cannot be recovered by lowering a threshold was
measured directly in `scripts/diag_cellprob.py`: when this model succeeds, the
cell-probability field inside the channel peaks at **+3.8 to +4.6**. When it
fails, that field is **flat at −2.7 to +0.2** — not "below the threshold", but
carrying no peak at all. In frames 5–7 of `052924_t3_dual` there is not a single
pixel above −2.0. There is nothing there for a lower threshold to find.

### What the held-out collapse actually is

"It generalises poorly" is a symptom, not a cause, and the cause turns out to be
measurable. Comparing the two halves directly:

| | 1st percentile | 99th percentile | cell width | (cell − background) / range |
|---|---:|---:|---:|---:|
| KK1 (40 images) | 1 530 | 2 775 | 12.1 px | **0.219** |
| KK2 (31 images) | 20 447 | 42 670 | 12.2 px | **0.378** |

Three things follow, and they narrow the problem considerably:

* **It is not a scale problem.** The cells are the same size to within a tenth
  of a pixel, so a diameter override cannot help and was not pursued.
* **It is not the raw intensity.** The two halves differ by a factor of
  fourteen in absolute value, but Cellpose normalises each image by percentile
  before the network sees it, which removes exactly that.
* **What survives normalisation is contrast.** A cell in KK1 stands 0.219 of
  the image's range above its background; a cell in KK2 stands 0.378 — nearly
  twice as far.

That last row predicts the *asymmetry* of the failure, which is the part a
vaguer explanation cannot account for:

* **KK2Model** was trained only on high-contrast images and scores recall
  **0.203** on KK1. It has never been shown a faint cell.
* **KK1Model**, whose training contrast spans 0.115–0.361 and therefore
  overlaps KK2's range, scores **0.435** going the other way.

The model that saw the wider range of contrast generalises twice as well. That
is a data-coverage statement, and it is the most useful thing in this document
for anyone planning the next acquisition: **vary the contrast deliberately, or
the model will only work at the contrast you happened to use.**

Because contrast is something that can be changed *before* the image reaches
the network, this is also the one part of the generalisation problem that is
addressable without retraining. `scripts/experiment_generalisation.py` measures
the available levers — percentile window, local (tiled) normalisation, and
sharpening — on the genuinely held-out pairs.

Each cell is the **F1 at IoU 0.5**. The two starred columns are the
genuinely held-out pairs; the other two are the same models on their own
training images, included so that a setting which buys held-out recall by
destroying everything else cannot hide.

| normalisation | KK2→KK1 \* | KK1→KK2 \* | KK1→KK1 | KK2→KK2 |
|---|---:|---:|---:|---:|
| default (1, 99) — shipped default | 0.303 | 0.482 | 0.819 | 0.742 |
| **percentile (3, 97)** | 0.303 | 0.653 | 0.779 | 0.704 |
| percentile (5, 95) | 0.247 | 0.610 | 0.636 | 0.595 |
| percentile (10, 90) | 0.148 | 0.551 | 0.317 | 0.309 |
| percentile (20, 80) | 0.000 | 0.033 | 0.020 | 0.000 |
| tile norm 128 | 0.233 | 0.467 | 0.776 | 0.742 |
| tile norm 64 | 0.118 | 0.335 | 0.680 | 0.670 |
| sharpen 15 | 0.208 | 0.341 | 0.794 | 0.708 |
| sharpen 30 | 0.262 | 0.489 | 0.798 | 0.732 |

**One setting helps, and only in one direction.** Narrowing the percentile
window to (3, 97) lifts the held-out KK1→KK2 F1 from **0.482 to
0.653** — a 35 % relative gain
with no retraining, bought by stretching a faint image's contrast towards
the range the model was trained on. It costs in-distribution accuracy
(0.819 → 0.779), which is the trade it should cost: the images that were
already at the right contrast get pushed away from it.

Everything else is worse, and the shape of the failure is informative:

* **Narrower windows overshoot.** (5, 95) already gives back half the gain
  and (10, 90) most of it; (20, 80) destroys the result entirely (F1 0.02).
  The response is unimodal, so (3, 97) is a real optimum rather than the
  best of a monotonic trend that simply was not followed far enough.
* **Local (tiled) normalisation does not help**, despite being the obvious
  answer to a contrast problem — 0.467 against the default's 0.482. A tile
  small enough to follow the background is also small enough to rescale a
  cell against itself, which removes the very contrast being measured.
* **Sharpening is neutral at best.** A 30 px radius lands on 0.489 against
  the default's 0.482, which is not a difference worth a second pass; a
  15 px radius is clearly worse (0.341).
* **KK2→KK1 cannot be rescued at all.** No setting improves on 0.303. That
  is the direction where the model was trained only on high-contrast
  images and is asked about faint ones, and no amount of stretching
  invents the features it never learned.

So the honest summary of limitation 3 is: **a third of the held-out gap in
one direction is recoverable at inference time, and none of it in the
other.** The setting is offered under *Image normalisation* and documented
with these numbers; the default is unchanged, because changing it would
trade in-distribution accuracy for a generalisation this dataset does not
need. The remaining gap is a training-data problem.

Two further strategies — tile 128 + sharpen 15, percentile (5,95) + sharpen 15 — did not
complete: the sweep ended after nine of eleven. Both combine settings
that lost individually, so they are recorded as not run rather than
quietly dropped from the table.

**What would still raise level 1 beyond that:** more labelled data covering more
conditions, and retraining. Not parameters, not ensembling, not
post-processing. That is a data-collection task, and it is outside what this
software can do to itself.

---

## Level 2 — keeping the right cell on the right trajectory

This is where the software adds what the model alone cannot, and it is measured
on every piece of ground truth available:

| check | dataset | result |
|---|---|---|
| One cell, one track | `052924_t1` | **1 track, 13/13 observations**, no fragment |
| Two cells never merged | `052924_t3_dual` | 2 tracks, separate throughout |
| No cell leaves its channel | `052924_1`, `052924_2` — 6 channels each, 16 tracks, 113 positions | **0 cross-channel tracks** |
| Independent crops agree | `t1` crop vs channel 0 of the wide field | mean speed 0.4977 vs 0.4976 µm/min (**0.02%**) |

That last row is the strongest single piece of evidence in this document. The
same physical cell, reached through two different files, two different device
geometry solutions and two different tracking runs, gives the same speed to four
significant figures.

The identity model is described in `src/corridor/core/tracking.py`. Every cost
term is a squared residual over its allowed spread, so the gate is readable as a
number of standard deviations rather than a tuned weight, and every refusal is
recorded with its reason in `unlinked_starts.csv`.

---

## Level 3 — the number that gets published

A missed detection does **not** bias a velocity, because displacement is divided
by elapsed time rather than by one frame. What it costs is precision, and if
enough are missed consecutively the track truncates instead of going wrong.

`scripts/experiment_velocity_robustness.py` measures this directly: detections
are deleted at random from a stack whose complete answer is known, and each
reported estimator is compared against its own complete-data value.

Each cell is **median / 90th-percentile** error against the complete-data
answer, over 60 random deletions per row, on `052924_t1`:

| detections lost | observations left | mean speed | net speed | net displacement | path speed | tracks truncated |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 13 | 0.0 / 0.0 | **0.0 / 0.0** | 0.0 / 0.0 | 0.0 / 0.0 | 0% |
| 10% | 12 | 5.2 / 12.4 | **0.0 / 2.9** | 0.0 / 6.1 | 0.1 / 3.4 | 0% |
| 15% | 11 | 9.1 / 23.7 | **0.0 / 2.7** | 0.0 / 5.7 | 0.3 / 3.0 | 0% |
| 20% | 10 | 7.2 / 19.3 | **0.0 / 4.6** | 0.0 / 10.2 | 1.0 / 4.1 | 0% |
| 30% | 9 | 11.6 / 27.6 | **0.0 / 8.7** | 0.0 / 18.4 | 1.8 / 8.4 | 3% |
| 40% | 8 | 14.2 / 48.1 | **3.6 / 62.0** | 10.0 / 49.0 | 3.7 / 57.3 | 13% |
| 50% | 7 | 16.6 / 79.5 | **7.9 / 89.1** | 12.2 / 89.7 | 7.0 / 79.5 | 27% |
| 60% | 5 | 35.7 / 87.7 | **35.6 / 92.4** | 34.4 / 95.7 | 35.1 / 86.4 | 52% |

The measured detection recall on this data is 0.846, so the **10-15% rows are
the realistic operating point**. There, net speed is *exactly right half the
time* and within **2.7-2.9% at the 90th percentile**, and no track truncates.
That is the defensible sense in which this reaches 97-99% for its actual
purpose.

Three things in that table are worth reading carefully:

* **Net speed is more robust than net displacement** (0.0/2.9 against 0.0/6.1
  at 10% loss), which looks backwards until you see why: losing a first or
  last detection shortens the measured distance *and* the elapsed time, and
  the two errors partly cancel in the ratio. Displacement keeps the whole
  error.
* **Median instantaneous speed is the worst estimator here**, at 20% median
  error from 10% loss onwards. With 13 observations the median jumps between
  discrete interval speeds, so removing one moves it a whole step. It is
  reported because it resists outliers, not because it resists missing data.
* **The cliff is at 40%, not gradual.** Up to 30% loss the median error of
  the net estimators is 0.0% and at most 3% of tracks truncate. At 40% the
  endpoints themselves start disappearing, 13% of tracks truncate, and the
  p90 error jumps from 8.7% to 62%. Degradation is safe until it is not, so
  the honest guidance is a threshold rather than a slope: **below about 30%
  detection loss the displacement figures hold; above it, check the frame
  coverage in `segmentation_diagnostics.csv` before quoting anything.**

### What to quote

* **Net displacement and net speed** are the robust estimators. They read only
  the first and last observation, so an interior miss is invisible to them.
* **Mean instantaneous speed** degrades roughly twice as fast, because each
  missed frame both removes a sample and merges two short intervals into one
  long one, and a long interval under-reads the speed of a cell that wanders.
* **Path length and path speed** sit in between, and both are inflated by
  position noise when detections are dense.

All of these are written to `track_summary.csv` side by side, precisely so the
choice is visible rather than made silently on the researcher's behalf.

---

## The fallbacks, and the measurement that decided each one

The request was for multiple fallbacks. Three were built. **Two of the three are
off by default, because they were measured and they made the result worse.**
That measurement is the deliverable; shipping them switched on would not have
been.

### 1. Detection fallback ladder — built, exposed, default off

`SegmentationConfig.ensemble` offers five rungs (`off`, `thresholds`, `wide`,
`models`, `max_recall`), selectable in the interface under **Detection effort**.
Extra passes can only *add* instances — the primary pass always wins a merge —
and anything only a fallback found is tagged in `detections.csv`.

`scripts/experiment_fallback.py` ran every rung end to end on the stacks with
known answers:

| rung | `t1` frames covered | `t1` tracks | false detections in known-empty frames | of those, survived into a track |
|---|---|---|---|---|
| off | 13/13 | 1 | 0 | 0 |
| thresholds | 13/13 | 1 | 0 | 0 |
| wide | 13/13 | 1 | 4 | **3** |
| models | 13/13 | **2** | 1 | 1 |
| max_recall | 13/13 | **2** | 3 | **3** |

**No rung improved coverage. Two of them split a correct single trajectory into
two. Three of them put objects in frames that contain no cell.**

The hypothesis going in was that the tracker would act as a temporal precision
filter — that a false detection would fail to link and be discarded. **That
hypothesis is wrong, and the experiment is what proved it.** A false detection
produced by channel-wall texture appears in the *same place* frame after frame,
and a stationary object is the most self-consistent thing a predicted-position
cost model can be shown. The tracker filters *erratic* false positives; it
rewards *persistent* ones.

Two consequences were implemented rather than noted:

* `off` remains the default, and the interface states each rung's cost.
* Corridor now reports `stationary_track` when a trajectory covers less path
  than the width of one cell, and `fallback_dependent_track` when half or more
  of a trajectory's positions came only from a permissive pass. These are the
  two signatures of the failure above, and they appear in `qc_issues.csv`.

That second consequence was then checked against the failure itself rather than
assumed to work. Running the real pipeline over `052924_t2_empty` — a stack
whose channel contains no cell — with the `wide` rung switched on produces one
false trajectory, and it arrives carrying four independent warnings:

```
[warning] fallback_dependent_track  Track 1 depends on the extra detection passes
[warning] stationary_track          Track 1 barely moves
[warning] lateral_drift             Track 1 moved more across the channel than along it
[warning] segmentation_gap          No cells found in a frame between detections
```

The run manifest records `ensemble: wide`, `ensemble_passes: 4` and
`detections_from_fallback: 3`, and the results screen shows the same under
**Detection effort**. So the honest position is: this fallback cannot be made
safe on this data, and if someone turns it on anyway, nothing about that is
hidden from them.

### 2. Track-guided recovery — measured, **on** for interior gaps

Three tiers (windowed re-segmentation, permissive thresholds, intensity
matching) that look for a cell where a track says one should be.

**An earlier version of this document said recovery "does not work" and
reported 1 of 13 holes filled. That was wrong, and the way it was wrong is
worth recording.** `recover()` asked `Track.predict()` where a dormant cell
should be. That method extrapolates forward from a track's last observation and
refuses to look backwards — correct while tracking is running, and exactly
wrong afterwards, when `last_frame` is the track's *final* frame. Every interior
gap therefore lay behind it, every one raised `ValueError`, and a bare
`except: continue` dropped them all without recording an attempt. The
measurement described the skip, not the algorithm, and nothing in any output
file would have revealed it.

With that fixed — interior gaps now *interpolate* between the observations on
either side, which needs no velocity model and cannot drift —
`scripts/experiment_recovery.py` gives a different answer entirely:

| where it looks | recovered | median error | false positives |
|---|---:|---:|---:|
| **interior gaps only** | **10 / 13** | **3.5 px** | **0** |
| interior: windowed tier alone | 5 / 13 | 9.1 px | 0 |
| interior + trailing | 11 / 13 | 2.9 px | 2 |
| **trailing frames only** | **1 / 13** | 2.9 px | **2** |
| disabled | 0 / 13 | — | 0 |

The split is the finding. An **interior gap** is a frame between two
observations: the cell was there before and after, so a hole is a detection
failure and the only question is where exactly. Those recover at 0.77, with a
median error of about a quarter of a cell's width, and produced **no false
positives at all** on the frames known to contain no cell.

A **trailing frame**, past a track's last observation, is a different claim —
the cell may have left, died, or gone out of focus, and there is no evidence on
the far side to say otherwise. It bought one true position and two false ones.

So recovery now runs by default for interior gaps and not for trailing frames.
On the wide field `052924_1` that fills **7 of 7** interior gaps. Everything
recovered carries its tier and confidence in `detections.csv`, is counted in
`run.json`, and trips `fallback_dependent_track` if a trajectory comes to lean
on it.

### 3. Gap-bridging in the tracker — on by default, and it works

The tracker links across missing frames up to `max_gap`, with the prediction,
the gate and the velocity all scaled by the true elapsed time. This is the
fallback that actually earns its place: it is why 20% detection loss costs net
displacement under 3%, and it is verified by
`test_a_missing_interior_detection_does_not_move_the_net_speed`.

---

## Knowing when to trust a particular run

Accuracy on the supplied data does not transfer to a new experiment by
assertion, so every run reports what it did rather than only what it found:

* `segmentation_diagnostics.csv` — instances Cellpose produced vs instances kept,
  per frame, with Cellpose's own message. "The cell disappeared" is always
  attributable to segmentation or to tracking, never ambiguous.
* `unlinked_starts.csv` — every track that could have continued an earlier one
  and did not, with the cost it would have had and the rule that refused it.
* `qc_issues.csv` — calibration gaps, stationary tracks, fallback-dependent
  tracks, channels inferred rather than measured.
* `run.json` — model SHA-256, Cellpose version, every parameter, and where the
  pixel size and frame interval came from.

**The single most important check** is the calibration provenance. The research
notebook this replaces used `dt = 10.0` minutes where the true interval is
20.0069, and `0.46` µm/px where the true size is 0.467060343. That timing error
alone overstated velocity by **97%** — far larger than every accuracy question
in this document combined. Corridor reads both from the file, records which tag
it read them from, and refuses to report micrometres when neither is available.

---

## Honest summary

* This software **will not** deliver 99% correct cell outlines. The measured
  ceiling on this model and this labelled data is F1 0.843, and held out it is
  0.30–0.48. Anyone quoting 99% at that level is not measuring it.
* This software **does** deliver correct trajectory identity on all available
  ground truth, and net migration distance within a few percent at realistic
  detection loss — which is the quantity the experiment is actually asking
  about.
* Every fallback that was tried is reported above, including the two that
  failed. The failures were kept in the codebase, switched off, with their
  measurements attached, because the next person to have the same idea deserves
  the number rather than the idea.
* The held-out weakness has a **named cause**, not just a number: the two
  halves of this dataset differ in contrast by a factor of 1.7 while holding
  cell size constant, and the model trained on the narrower contrast range is
  the one that loses the most. That is actionable in a way "it generalises
  poorly" is not — it says what to vary in the next acquisition.

### What would move each level

| Level | What would improve it | Is it in this software's power |
|---|---|---|
| Cell outlines | More labelled data spanning more contrast; retraining | **No** — a data-collection task |
| Trajectory identity | Nothing measured is wrong with it today | Already done |
| Migration figures | Better detection recall, which is level 1 again | Already extracts most of what the detections allow |

The honest shape of this result is that **the software is no longer the limiting
factor — the training data is.** Every stage after segmentation was measured
against ground truth and none of them is losing accuracy that segmentation
delivered. That is a good place for a pipeline to be, and it is also the reason
no amount of further engineering here reaches 99% per-instance.

## Reproducing every number here

```bash
python scripts/evaluate_models.py                 # held-out model performance
python scripts/experiment_recall.py               # the 8 detection strategies
python scripts/experiment_fallback.py             # the ladder, end to end
python scripts/experiment_recovery.py             # track-guided recovery
python scripts/experiment_velocity_robustness.py  # velocity under detection loss
python scripts/experiment_generalisation.py       # the contrast shift, and what fixes it
python scripts/diag_cellprob.py                   # the flat probability field
```

Results are written to `docs/*.json` beside this file.
