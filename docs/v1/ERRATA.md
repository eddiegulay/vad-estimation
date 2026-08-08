# v1 Errata

Corrections to [`ROADMAP.md`](./ROADMAP.md), which is archived verbatim and never edited.
Every entry quotes the original claim, states the correction, and cites the audit finding in
[`AUDIT.md`](./AUDIT.md) that establishes it.

Established 2026-08-08 by a five-agent audit of the completed v1 run. Corrected numbers are
collected in [`RESULTS.md`](./RESULTS.md).

---

### E1 — The RTF figures are a measurement artifact, and the ordering is inverted

> "~5x faster inference (RTF 0.006 vs 0.030)" — ROADMAP lines 6, 30, 56

The evaluation timer wraps the *first* forward pass on each new sequence length, so every
MPSGraph shape-specialisation compile is billed to a 261.9 s audio denominator (the first
example alone costs 1713 ms for the CRNN, 538 ms for the TCN).

Re-measured warm `forward_full`: **CRNN 0.0031, TCN 0.00037** — the TCN is 8.3× *faster*, the
opposite of what was reported.

The CRNN-over-TCN *decision* survives, but on different evidence: per-chunk streaming latency
on CPU, where the CRNN runs p50 0.59 ms / p95 1.85 ms and the TCN runs p50 17.8 ms / p95 103 ms
against a 32 ms budget. The TCN as implemented is not shippable on CPU. Root cause is
PyTorch's grouped-convolution dispatch (~2 µs fixed overhead per group), not the architecture.

*Finding: I8.*

---

### E2 — The CRNN-vs-TCN validation comparison compares two corruption patterns

> "TCN reaching a higher raw val_f1 (0.79 vs 0.73)" / "the TCN fits the val split better" — ROADMAP lines 26–30, 56

`nn.Conv1d` on the MPS backend (torch 2.13.0) silently returns wrong values for every batch row
with index ≥ 65536. The model reshapes to `[batch × num_chunks, 1, 640]` before the conv stack,
and validation batches reach 162,368 rows — so up to 60% of every validation batch was scored
from corrupted activations. No error is raised.

True validation F1, re-scored under the row limit: **CRNN 0.9211, TCN 0.9226 — a wash.**

Training was unaffected: the 625-chunk crop caps training at 40,000 rows. Gradients are fine;
only the metrics were wrong. The margin was thin, though — `batch_size ≥ 105` would have
corrupted training too.

*Finding: B1.*

---

### E3 — There is no overfitting

> The implicit narrative behind selecting `best.pt` at epoch 28 of 50

Re-scored safely, validation loss *falls* and AUROC *rises* from epoch 28 to epoch 49 for the
CRNN, and from epoch 19 to 49 for the TCN. The rising val-loss curve in TensorBoard is the same
corruption artifact as E2.

Consequence for v2: do not add early stopping, dropout, or extra weight decay on the basis of
that curve. Roughly half of each run was *not* wasted.

*Finding: B1.*

---

### E4 — `best.pt` is the worse checkpoint, and it is what was exported

> `train.sh` evaluates and exports `best.pt`

`best.pt` is worse than `last.pt` on every threshold-free metric, for both architectures,
because selection ran on the corrupted val F1 from E2. **The shipped ONNX models are the
mis-selected ones.** Prefer `last.pt` from `runs/v1/`.

*Finding: B1.*

---

### E5 — The calibrated-F1 difference between architectures is not resolvable

> "marginally higher calibrated F1 (0.947 vs 0.942)"

Cluster bootstrap over the 30 TEN files (frames are heavily autocorrelated; a naive iid CI is
3.2× too narrow): F1 difference **+0.0013, 95% CI [−0.0087, +0.0111], p = 0.78**. Pure noise.
Detecting an effect that size at 80% power would need ~3,300 files.

The AUROC difference (+0.0169, CI [−0.0004, +0.0348], p = 0.057) is suggestive but not
significant; it would need ~67 files.

*Finding: I9.*

---

### E6 — Hysteresis smoothing degrades event-level metrics; it does not stabilise them

> "Hysteresis smoothing costs ~1-2pt of F1 … expected, since smoothing optimizes event-level stability, not frame accuracy" — ROADMAP line 30

The rationale is wrong and the event-level metrics got *worse*. Calibration set `theta_on` from
a val sweep (0.15) while `theta_off` stayed at its config default (0.35), so `theta_off >
theta_on` and the Schmitt trigger inverts: any frame in [0.15, 0.35) satisfies the enter and
exit conditions simultaneously, degenerating the state machine into a fixed 8-on/6-off
oscillator that carries no information about the probability.

Paired file-level bootstrap, shipped minus raw: **onset F1 −0.1209 [−0.1966, −0.0551],
p < 0.001**; frame F1 −0.0117, p = 0.050; turn cost +27%; median endpoint dead-air 608 ms →
1552 ms.

The inversion is only ~10% of the damage. The dominant cause is that `theta_off = 0.35` sits
far below where the models place their silence probability — only 28 of TEN's 96 interior true
pauses ever produce 6 consecutive frames below it.

*Finding: B4.*

---

### E7 — The training crop discarded most of the supervision

> "Resolved in Phase 6: Trainer random-crops training sequences … eval sees full sequences" — ROADMAP line 47

The crop offset is drawn **once per batch** and indexed against the batch's *padded* length,
which is set by the longest example (up to 94 s). Any example shorter than the crop start lands
entirely inside its own zero padding and contributes no gradient.

Measured over the real manifest: **27.6% of labelled frames survive per epoch; 44.9% of
examples contribute zero gradient; 32.7% of forward-pass cells are supervised.** "50 epochs"
delivered roughly 14 epochs of data at full compute cost.

*Finding: B2.*

---

### E8 — Per-epoch augmentation re-randomisation never happened

> "resolved fresh per epoch via `VADDataset.set_epoch`" — ROADMAP line 64

`set_epoch` mutates the dataset in the main process, but `persistent_workers=True` with the
spawn start method means the four worker processes hold pickled copies whose epoch is frozen at
0 for the entire run. Reproduced empirically.

The run was 50 passes over **one static augmentation realisation** — 9,318 fixed
(speech, noise, SNR, RIR, gain) pairings instead of the intended ~466,000. Frozen: noise clip
choice, SNR, RIR presence and identity, noise crop offset, gain, and the gap silence/noise coin
flip.

This matters doubly because neither model has any other regulariser — no normalisation, no
dropout, no residuals.

*Finding: B3.*

---

### E9 — The AMI occupancy investigation was incomplete

> "AMI occupancy … investigated and accepted as expected, not a bug" — ROADMAP line 49

**20.7% of AMI training windows (1,306 windows, 7.26 h) contain zero negative frames** — they
are pure positives carrying full compute cost and no discriminative signal. The declared
`speech_occupancy_sanity_band: [0.20, 0.80]` is read by nothing.

*Finding: I5.*

---

### E10 — The FLEURS cross-lingual sanity set was never evaluated

> "FLEURS Swahili used as a cross-lingual sanity-eval set" — ROADMAP line 41

`sanity_fleurs.jsonl` (200 examples, 3.49 h) is rebuilt on every pipeline run and evaluated
never. No report for it exists in any checkpoint directory.

*Finding: I6d.*

---

### E11 — The headline F1 is reported without its trivial baseline

> "TEN F1 0.8974"

A constant "always speech" predictor scores **F1 0.8586** on TEN, which is 75.2% speech. The
real lift is **+3.9 points**. Not false, but materially incomplete — and the 20-step smoke
checkpoint already scored 0.8903, within 0.7 points of the full 2.5-hour run.

AUROC is the metric that actually separates them: 0.267 (random init) → 0.867 (smoke) → 0.905.

*Finding: audit §v1.*

---

### E12 — Threshold calibration on validation and reporting on TEN is invalid as set up

> The calibrated-threshold methodology

Validation is 87.2% speech; TEN is 75.2%. The F1-maximising threshold is strongly
prevalence-dependent, so the procedure transfers a systematically too-permissive threshold.
Measured: validation optima land at θ 0.15–0.25 while TEN-oracle optima are at θ 0.65–0.90 —
opposite ends of the grid. Every val-optimised post-processing configuration except a
turn-cost-optimised one transfers *worse than no post-processing at all*.

Separately, the "calibrated F1 0.9466" figure quoted alongside TEN test numbers is a
**validation-split** sweep result, never computed on TEN. The comparison table header labels it
correctly; the prose does not.

*Finding: I12.*
