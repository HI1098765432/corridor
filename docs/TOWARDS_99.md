# Getting to 99 %: changing what the model is trained to agree with

This document is a research log, written as the work happens. It records what
was tried, what the numbers were, and which ideas died. Nothing here is a claim
about the shipped application until it appears in `docs/ACCURACY.md`.

---

## The mistake that was blocking it

Every accuracy figure this project has published is an **agreement score against
71 hand-labelled images**. The reported ceiling — "F1 never exceeds 0.843 at any
price", measured across eight inference strategies — was therefore a statement
about *agreement with those labels*, and it was quietly treated as a statement
about the cell.

Those are not the same thing, and the difference is measurable:

| | measured | what it implies |
|---|---|---|
| Mean IoU of **correctly matched** pairs | **0.696** | Even when detection succeeds, model and annotator only roughly agree on the outline |
| F1 at IoU 0.75 | 0.07–0.22 | The outlines are approximately right, not precisely right |
| F1 at IoU 0.90 | **0.00** | Nothing agrees with the labels that closely — including, plausibly, a second annotator |
| Images containing **no labels at all** | **9 of 71** | Every detection there scores as a false positive whether or not a cell is present |

A model that became *more accurate than the labels* would score **worse** by this
metric. Optimising against it therefore cannot exceed the annotator; it can only
converge to them. That is why more parameters, more ensembling and more
thresholds all stalled at 0.843 — the wall is the yardstick, not the learner.

**So the route to 99 % is not a better optimiser. It is a better reference.**

---

## The data nobody had used

The supplied training folder looks like 71 unrelated pictures. It is not.
Grouping by image dimensions and ordering by the trailing number in each
filename recovers **six contiguous runs of frames from the same field of view**
(`src/corridor/learn/sequences.py`):

| sequence | frames | frame-to-frame correlation | median centroid step | unlabelled interior frames |
|---|---:|---:|---:|---|
| KK2 303×573 | 16 | 0.593 | 20.5 px | 1 |
| KK2 306×621 | 15 | **0.101** | 30.8 px | 2 |
| KK1 264×727 | 12 | 0.770 | 25.3 px | 2 |
| KK1 290×762 | 12 | **0.853** | 23.0 px | — |
| KK1 309×649 | 10 | 0.678 | 7.5 px | — |
| KK1 381×864 | 5 | 0.562 | 13.9 px | — |

Five of the six are genuine movies. The sixth (correlation 0.101) is flagged
rather than used: consecutive frames of one field look alike, and that one does
not, so it is probably several fields sharing a crop size. The grouping reports
the evidence for its own validity instead of asserting it.

On top of these there are **74 unlabelled frames** in `sample_data/`, including
two wide fields with six channels each.

### Why this changes the problem

A human tracing frame *t* sees frame *t* alone. **A cell is one object persisting
through time.** Its outline at *t* is over-determined by its outline at *t−1* and
*t+1*, together with smooth motion and a near-conserved area. Solving for the
mask *sequence* that best explains the image evidence under those constraints
gives a per-frame mask that can be **better than the single-frame tracing** — not
because the algorithm is cleverer than the annotator, but because it is given
evidence the annotator never had.

That is also what keeps it honest. The supervising constraint is **time and
physics, not the model's own opinion**, so training on the result is not the
model marking its own homework. The principle is already proven on this data one
level down, at positions rather than outlines: track-guided recovery fills
**10 of 13** known holes with **zero** false positives, median error 3.5 px,
purely by interpolating between the observations either side of a gap.

### The unlabelled interior frames

Five frames sit *inside* a sequence with no labels while the frames on both
sides have cells. The cells did not leave; the annotator stopped. Those frames
currently teach the model that cells are background, and penalise it at
evaluation for finding them. They are the clearest single instance of the
reference being wrong rather than the model.

---

## Programme

Each stage has to produce a number before the next one is believed.

1. **Measure how much of the reported error is the reference, not the model.**
   Re-score with the unlabelled images excluded, and check what is actually
   present in them using intensity and shape alone — never the model's opinion,
   which is what is on trial. (`scripts/diag_label_quality.py`)
2. **Build the temporal reconstruction.** Per-sequence, solve for mask sequences
   that are consistent across frames.
3. **Prove the reconstruction is better than the labels**, without assuming it.
   The test that matters: does the reconstruction predict a *held-out* human
   label better than the labels' own frame-to-frame jitter?
4. **Retrain** on the refined masks, with contrast randomisation aimed at the
   measured 1.7× domain shift (KK1 0.219 vs KK2 0.378 cell-to-background).
5. **Re-measure** on the untouched held-out split and report the real number.

From here every result is reported **twice**: against the human labels — so it
stays comparable with everything already published — and against the
reconstructed reference. Where the two disagree, the image decides.

---

## Results

*Failures are recorded here too; two of the three fallbacks already shipped in
this project were kept precisely because their measurements said no.*

### 1. The reference is demonstrably incomplete — measured without the network

`scripts/diag_label_gaps.py`. For each frame that has no labels but sits between
frames that do, it asks what the image contains along the path the cell must
have taken. The model plays no part in the answer: **where** to look comes from
the neighbouring frames' *human* labels, and **what counts as a cell** comes from
the intensity of the cells the annotator drew elsewhere in that same sequence.
Accepting "the model found something, so something is there" would assume the
conclusion, because the model is precisely what is on trial.

| sequence | frame | contrast found | the annotator's own cells | verdict |
|---|---:|---:|---:|---|
| KK2 303×573 | 11 | **0.398** | 0.404 | a cell is there, unlabelled |
| KK1 264×727 | 8 | **0.357** | 0.419 | a cell is there, unlabelled |
| KK1 264×727 | 3 | 0.125 | 0.419 | plausibly empty |

**Two of the three contain an object at essentially the brightness of the
annotator's own cells.** The test is not merely agreeable — frame 3 of
KK1 264×727 came back genuinely empty — so it discriminates rather than
confirming whatever it is asked.

Those two frames do damage twice over. In **training** they teach the network
that a cell is background. In **evaluation** every correct detection there is
counted as a false positive, depressing the precision that the published F1 is
built from.

This is the first hard evidence that the 0.843 ceiling is partly a property of
the reference rather than of the model. How *much* of the gap it explains needs
the model re-scored with those frames excluded, which is measured next.

### 2. Temporal reconstruction carries real information — and one of my claims for it was wrong

`scripts/experiment_reconstruction.py`. Cells are followed through each labelled
sequence by mask overlap, and each frame's mask is rebuilt from its neighbours.

**Test 1, rebuilding a frame whose label was hidden.** Clean, because it
predicts a human label the method never saw:

| method | n | median IoU with the hidden label |
|---|---:|---:|
| temporal reconstruction | 41 | **0.492** |
| copy the previous frame's mask | 41 | 0.359 |

Temporal evidence is worth **+37 % over the obvious baseline**. A frame nobody
labelled is not a blank: its neighbours substantially determine it.

**Test 2, boundary sharpness — and the control that falsified it.** The intent
was an arbiter made of photons rather than opinions: score a boundary by how
well it sits on the image's own intensity gradient. On that measure the
reconstruction beat the human tracing in 37 of 41 cases.

That result does not survive its own control:

| mask | median boundary sharpness |
|---|---:|
| human tracing | 1.241 |
| **temporal consensus alone** (never looks at the image) | **1.196** |
| consensus + image refinement | 1.688 |

`refine_to_image` places its boundary at an intensity threshold, and a threshold
crossing on a smooth edge sits near the gradient peak — so the metric was
substantially rewarding the thing that method optimises. **Pure temporal
evidence does not beat the annotator on boundary placement**; it is slightly
worse. The 90 % figure measures my refinement step agreeing with my metric, and
it is withdrawn as evidence for the central claim.

Recorded rather than deleted, because the failure is instructive: an arbiter
has to be independent of *every* method it ranks, and "it is a property of the
image" was not sufficient — the method also optimised a property of the image.
The surviving evidence for the approach is test 1, which predicts a held-out
human label and cannot be gamed that way.

### 3. The reconstruction is **not** a better teacher than the model — where the model works

Adding the model to test 1 settles the question the wrong way for the original
plan:

| method | n | median IoU with the hidden label |
|---|---:|---:|
| **the segmentation model** | 41 | **0.725** |
| temporal reconstruction | 41 | 0.492 |
| copy the previous frame | 41 | 0.359 |

On a frame whose label was hidden, the model reproduces that label far better
than the temporal reconstruction does. **Retraining on reconstructed masks would
therefore make the model worse**, not better, everywhere the model already
works. The naive form of this plan is dead, and it died on a number rather than
on an opinion.

#### What that leaves, and why it is the more useful half

The comparison above is taken only over cells that **are** labelled and **are**
detected — that is the only place all three methods can be compared. It is
therefore a measurement of *outline quality on the easy cases*, and it says the
network is better at outlines than a translation-based consensus. That is not
surprising in hindsight; the network was trained for exactly that.

The model's measured failure mode is not bad outlines. It is **producing nothing
at all**: recall 0.846 in-distribution and 0.20–0.44 held out, with a
probability field that goes *flat* on a miss rather than merely dipping below
threshold. Where the model outputs nothing, a mask at IoU 0.49 with the right
existence and the right position is not competing with 0.725. It is competing
with **zero**, and with a training target that currently says *background*.

So the programme narrows to its defensible core:

* Reconstruction must **not** replace labels or predictions where the model
  already succeeds — measured, 0.492 against 0.725.
* Reconstruction is worth exactly what it fills: cells the model missed, in
  frames the annotator skipped. That is the same signal that fills 10 of 13
  known holes with zero false positives.
* The training signal becomes: **human labels where they exist, plus
  reconstructed masks only at holes**, weighted below the human ones.

That is targeted pseudo-labelling aimed precisely at the recall failure, rather
than a wholesale replacement of the reference — and it is the version of the
idea that the measurements actually support.

### 4. Three defects in the harness, any one of which would have invalidated a retraining result

`scripts/diag_training_harness.py`, all verified directly against the files.

**Cellpose silently discards most of the labelled data.** `train.train_seg`
defaults to `min_train_masks=5`, dropping any image with fewer than five
instances without a word. These are confined cells, a few per field:

| | images | instances | survive the default |
|---|---:|---:|---|
| KK1 | 40 | 138 | 12 images, 73 instances |
| KK2 | 31 | 108 | 11 images, 61 instances |

**48 of 71 images (68 %) and 112 of 246 hand-drawn instances (46 %) never reach
the network.** Every one of the nine zero-mask images goes too — and those are
precisely the empty-channel negatives a model needs in order to learn what *not*
to fire on. Any fine-tune must pass `min_train_masks=0`.

**The split is not symmetric.** KK1 is four acquisition days (041824, 061523,
122324, plus one junk file); KK2 is a single day, 052924. So "KK2Model scores
0.303 on KK1" is partly a **one-day training set**, not purely a domain gap —
which revises the reading of the contrast finding in `docs/ACCURACY.md` rather
than overturning it.

**The unlabelled stacks share a day with KK2.** Every stack in `sample_data` is
`052924_*`, as are 31 of 32 KK2 labels. Improving the model with those stacks
and reporting a KK2 score would be training on the test set by another route.
Unlabelled-data methods must be evaluated on **KK1**.

### 5. The model is not contrast-invariant, and that explains everything

`scripts/experiment_contrast_response.py`. One thing is varied and nothing else:
each image is split into a slowly-varying background and the perturbation the
cell makes on it, and only that perturbation is rescaled. Same positions, same
shapes, same noise, same device. Run on images the model was **trained on**, so
a drop cannot be blamed on unfamiliar data.

| contrast scale | achieved &#124;c&#124; | recall |
|---:|---:|---:|
| 0.25× | 0.059 | **0.098** |
| 0.40× | 0.079 | 0.512 |
| 0.55× | 0.100 | 0.732 |
| 0.85× | 0.139 | 0.780 |
| **1.00× (native)** | **0.158** | **0.829** |
| 1.30× | 0.193 | 0.756 |
| 1.80× | 0.243 | 0.634 |
| 2.50× | 0.292 | **0.463** |

**Recall peaks exactly at the contrast the model was trained on and falls away in
both directions.** Making a cell *twice as visible* takes recall from 0.83 to
0.46. The model is not detecting a cell; it is detecting a cell at the contrast
it was shown.

This is the common cause behind every stalled measurement in this project:

* **the held-out collapse** — KK2's cells sit at 0.378 and KK1's at 0.219, which
  are different points on this curve, so a model fitted at one is off-peak at
  the other;
* **the flat probability field on a miss** — off-peak the network does not
  respond weakly, it does not respond;
* **why no threshold ever helped** — a flat field has nothing to threshold;
* **why the eight-strategy search stalled at F1 0.843** — every one of those
  changed the *decision rule*; none changed the network's contrast tuning.

And unlike the ceiling it explains, **contrast invariance is learnable**. The
transform written to measure this is exactly the augmentation needed to fix it:
the instrument and the treatment are the same operation.

#### The confound this survived

Adversarial review found a real flaw in the experiment as first run. Cellpose
percentile-normalises every image it is given, and **these cells live in the
tail of that distribution**: masks cover 1–2 % of the frame, and 8–21 % of cell
pixels fall outside the [p1, p99] window. Rescaling a cell therefore *moves the
window*, which rescales the whole image — and narrowing the window is itself a
measured intervention that lifts held-out F1 from 0.482 to 0.653. The sweep was
varying two things at once, with the second pushing the opposite way at low
contrast.

Freezing the window took two attempts. Cellpose 3.1.1.3 accepts
`normalize={"lowhigh": (lo, hi)}` but it **does not work**: handed an image's own
percentiles it returns zero masks where `normalize=True` returns four. The
window is instead frozen by normalising here and passing `normalize=False`,
verified equivalent on the native image.

| contrast scale | free window (confounded) | **frozen window** |
|---:|---:|---:|
| 0.25× | 0.098 | **0.098** |
| 0.55× | 0.732 | **0.732** |
| 1.00× native | 0.829 | **0.829** |
| 1.80× | 0.634 | **0.683** |
| 2.50× | 0.463 | **0.512** |

The confound was real in principle and small in effect. **The curve stands.**

Two independent measurements corroborate it. The per-instance contrast
distributions differ exactly as the curve predicts they must: KK1 median
&#124;c&#124; 0.184 with **54 % of instances below 0.20**, against KK2 median
0.385 with only **15 %** below. KK2Model was trained on bright cells and never
shown a dim one — and it scores recall 0.203 on KK1, whose cells are mostly
dim. The domain gap and the contrast curve are the same fact seen twice.

### 6. Acting on it: held-out F1 0.482 → 0.726

Two interventions, both following directly from the curve, measured on KK2 —
a different acquisition day, never seen, starting from KK1Model which never
trained on it.

**Input side, no training at all.** If the model only responds inside a narrow
band, put the image's cells in that band before asking
(`scripts/experiment_input_matching.py`):

| transform | precision | recall | **F1** |
|---|---:|---:|---:|
| baseline (cellpose default) | 0.540 | 0.435 | **0.482** |
| **percentile window (3, 97)** | 0.714 | 0.602 | **0.653** |
| (2, 98) | 0.667 | 0.574 | 0.617 |
| (5, 95) | 0.663 | 0.565 | 0.610 |
| (10, 90) | 0.614 | 0.500 | 0.551 |
| histogram matched to KK1 | 0.417 | 0.093 | 0.152 |
| local gain to the model's peak | 0.275 | 0.259 | 0.267 |

The two cleverer ideas both failed badly. Matching the whole intensity
distribution to the training half collapsed to 0.152, and calibrating a local
gain so typical structure lands on the measured peak reached only 0.267. A plain
percentile window beat both by a factor of two.

**Weight side, 106 minutes of CPU.** Fine-tuning on the same 40 KK1 images with
contrast randomisation over [0.35×, 2.6×], `min_train_masks=0` and
`rescale=False`: **F1 0.482 → 0.539**.

**Together**, which is the point:

| | default window | best window |
|---|---:|---:|
| original model | 0.482 | 0.653 |
| contrast-trained model | 0.530 | **0.726** |

**Recall, the actual failure mode, went 0.435 → 0.722**; precision 0.540 → 0.729.

Two honest qualifications. The (5, 95) window was chosen by looking at the test
set, which is mild test-set selection; at the **pre-committed** (3, 97) the same
model scores **0.695**, so 0.70 is the defensible figure and 0.726 the best
observed. The gain does not depend on that choice — every window from (2, 98) to
(10, 90) lands between 0.61 and 0.73, all far above the 0.53 default.

And the optimum **moved** from (3, 97) to (5, 95) after training, which is what a
widening response band should do: the model no longer needs the image pushed as
hard to reach a contrast it can see.

### 7. The errors, classified one by one — and what they turned out to be

`scripts/diag_errors.py` and `scripts/diag_error_structure.py`. At F1 0.695 on
KK2 there are 64 errors left. Aggregate scores say how much is wrong, never
what, and the classification redirected the work twice.

| misses (35) | | false detections (29) | |
|---|---:|---|---:|
| too faint | 14 | looks exactly like a cell (suspect label) | 13 |
| found but outline too poor | 9 | overlaps a real cell | 8 |
| visible but missed | 9 | faint structure | 7 |
| at the image border | 3 | at the image border | 1 |

**The 17 "carved wrong" are near-misses, not splits.** Zero splits, one merge,
seven bad boundaries — and **8 of 9 misses sit within 0.1 of the 0.5
threshold**, median IoU 0.431. Those cells are found, in the right place, with
roughly the right shape; they fail on a few percent of overlap. Recovering them
would convert an error in *both* columns at once.

**The 9 "visible but missed" are longer than the model's scale.** Median major
axis 95.4 px (IQR 94.9–129.9) against 82.8 px for objects it does fire on, at
normal contrast (0.391), strong edges (4.67) and normal width (10.4 px).

**The 14 "too faint" are border fragments.** Median 3.5 px from the image edge
(IQR 1–8) and half the usual area (386 px against 868). Clipped cells, partly
outside the field, inherently ambiguous.

The errors are also not concentrated: 28 of 31 images contain at least one, the
worst five holding 36 %. There is no small set of pathological images to fix.

### 8. Scale is the second axis, and diameter 36 is free — validated both ways

Cellpose rescales every image so objects match the diameter it was trained at.
For a round cell that diameter is unambiguous; for a confined cell 10 px wide
and 130 px long it is not, and the equivalent-disk diameter such a cell implies
is dominated by its **length**. So a field containing one unusually long cell is
rescaled differently from a field of ordinary ones.

Sweeping the diameter directly, retraining nothing
(`scripts/experiment_diameter.py`):

| diameter | KK1→KK2 F1 | KK2→KK1 F1 |
|---|---:|---:|
| from the checkpoint (31.2) | 0.6952 | 0.3027 |
| **36** | **0.7273** | **0.3388** |
| 45 | 0.6935 | 0.2924 |
| 60 | 0.5412 | 0.0417 |
| 80 | 0.1639 | 0.0000 |

**The same optimum, and a similar gain, in the direction that was not tuned
on.** That is what separates a real finding from test-set fitting, and it is why
this one is believed: +0.032 and +0.036 for a parameter, not a retraining.

The long-cell hypothesis came out **half right**. Long-cell recall moved
0.794 → 0.841 on KK2, below the threshold set in advance for calling it
confirmed, so the script recorded the hypothesis as wrong there. In the other
direction it moved 0.182 → 0.260 and cleared it. What the sweep revealed instead
is a sharper asymmetry: at the best diameter, **recall on short cells is 0.511
against 0.841 on long ones**. The hard group is the small border fragments — the
opposite of what was being chased.

Two axes now have the same shape: the model is tuned to the contrast it saw and
to the scale it saw, and in both cases the remedy is to show it more than one.

#### A defect this exposed in my own harness

The first run of the input-side experiment reported 0.134 for the (3, 97) window
— against 0.653 for what should have been the identical transform. The cause is
the same fact as everything else here: **these cells live in the tail of the
intensity distribution**, 8–21 % of their pixels beyond p99. My implementation
clipped to [0, 1] after rescaling, which deletes the brightest part of every
cell. Cellpose's own percentile normalisation does not clip. A five-fold
difference in the measured result, from one call to `np.clip`.
