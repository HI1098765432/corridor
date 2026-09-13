# Segmentation model evaluation

Reproduce with `python scripts/evaluate_models.py`. Raw numbers in
`model_evaluation.json`.

## Why this needed a specific construction

The combined model (`cyto2_phase_microfluidic_KK1KK2_combi`) was trained on all
71 labelled images — KK1 (40) plus KK2 (31). **There is no held-out data for it
in the supplied material.** Any score it gets on those images measures how well
it fits its own training set, not how it would behave on a new experiment.

The supplied data does support a genuine held-out test, because two further
models were trained on disjoint halves:

```
KK1Model  trained on KK1 (40 images, 138 cells)  ->  evaluated on KK2 (31 images, 108 cells)
KK2Model  trained on KK2 (31 images, 108 cells)  ->  evaluated on KK1 (40 images, 138 cells)
```

Each of those pairings is a real generalisation test. Both are reported below,
separately and clearly labelled, alongside the fit numbers.

Instances are matched one-to-one between prediction and ground truth by
intersection over union, using optimal assignment. Settings are the ones the
researcher used when labelling, read from the `_seg.npy` files themselves:
`cellprob_threshold = 0.0`, `flow_threshold = 0.4`, `channels = [0, 0]`.

## Results at IoU ≥ 0.5

| Model | Evaluated on | Kind | Precision | Recall | F1 | mean IoU |
|---|---|---|---:|---:|---:|---:|
| combi | KK1 | training-set fit | 0.859 | 0.884 | **0.871** | 0.705 |
| combi | KK2 | training-set fit | 0.796 | 0.796 | **0.796** | 0.727 |
| KK1Model | KK1 | training-set fit | 0.819 | 0.819 | 0.819 | 0.696 |
| KK2Model | KK2 | training-set fit | 0.752 | 0.731 | 0.742 | 0.687 |
| KK1Model | **KK2** | **held out** | 0.540 | 0.435 | **0.482** | 0.659 |
| KK2Model | **KK1** | **held out** | 0.596 | 0.203 | **0.303** | 0.616 |

## What this says

**Generalisation is roughly half of fit.** Held-out F1 is 0.30–0.48 against
0.74–0.87 on training data. Anyone quoting ~0.87 as this model's accuracy would
be quoting a fit number.

**The failure mode is missing cells, not inventing them.** Held out, precision
holds up (0.54–0.60) while recall collapses (0.44 and 0.20). The model is
conservative: what it finds is usually right, but it finds well under half of
what is there in data it was not trained on. For migration measurements this
matters in a specific way — trajectories will be *fragmented* rather than
*wrong*, which is the safer of the two failure modes but still biases any
statistic computed over track lengths.

**More training data generalised better.** KK1Model (40 images) reaching KK2 at
F1 0.482 clearly beats KK2Model (31 images) reaching KK1 at 0.303. That is one
comparison, not a curve, but it points the obvious direction if more labels
become available.

**Even in-distribution the model misses cells.** The combined model's recall on
its own training images is 0.796–0.884, so 12–20 % of labelled cells are missed
under the settings used to label them. The model has a real sensitivity floor
independent of any generalisation question.

## Bearing on the missing detections in `052924_t3_dual`

Frames 5–9 of that stack produce no masks. Two measurements together explain it:

1. **The post-filter is innocent.** Raw and kept instance counts are identical
   for every frame of every supplied file; `min_extent = 20` removed nothing.
2. **The network's own evidence is absent, not merely below threshold.** Inside
   the confinement channel the cell-probability field reads +3.76 to +4.59 in
   frames that produce a mask, and −2.74 to +0.17 in frames 5–9 — a collapse of
   about six units. Frames 5, 6 and 7 have *zero* pixels above −2.0, far below
   any usable threshold. See `scripts/diag_cellprob.py`.

So lowering the threshold cannot recover those frames: there is nothing there
to recover. A parameter sweep confirms this from the other direction — no
reasonable Cellpose v3 setting recovers them, and the settings that try
(`flow_threshold = 0.8`) make the near-empty control stack report a cell in
every frame.

What remains genuinely unresolved is *why* the network sees nothing: whether
the cells left the imaged region, changed appearance beyond what the model was
trained on, or were never there. The measured recall limit above makes a
sensitivity failure entirely plausible, but the data cannot distinguish these.

## What this evaluation does not establish

- It says nothing about a **different experiment**: a different device, cell
  line, objective, or illumination. Every image here comes from one collection.
- KK1 and KK2 are adjacent conditions from the same study, so even the held-out
  numbers are a *mild* generalisation test. A genuinely new dataset would
  likely score lower, not higher.
- Ground truth is one person's manual labelling. Inter-annotator agreement is
  unknown, and the mean IoU of correct matches (0.62–0.73) is close to what
  annotation variability alone could produce for objects this elongated.
