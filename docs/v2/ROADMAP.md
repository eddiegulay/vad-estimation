# VAD Models — v2 Roadmap

Second iteration. v1 is archived at tag `v1.0`: artifacts in `runs/v1/`, record in
[`../v1/`](../v1/), corrected results in [`../v1/RESULTS.md`](../v1/RESULTS.md), audit in
[`../v1/AUDIT.md`](../v1/AUDIT.md).

Started 2026-08-08. Status: **Phase 0 — not started.**

---

## What v1 taught, in one paragraph

v1 shipped two models that beat a parameterless baseline by 3.9 points of F1, scored *at or
below* that baseline on the only conversational data in the kit, never released the floor on a
third of turn-ends, and reported a set of numbers that a five-agent audit found were largely
measuring corrupted activations. Every one of the six blocking defects was invisible in the
metrics being reported. **The binding constraint on v1 was never the model or the data — it was
that the project could not tell a real improvement from noise.** v2 is organised around fixing
that first.

## The v2 thesis

> Build the measurement foundation before training anything. Then land the changes whose effect
> is already measured, in an order where each one is provable.

Three consequences that shape every phase below:

1. **No training run happens until the benchmark can resolve the changes it is meant to
   evaluate.** v1's benchmark could not resolve a +0.027 AUROC effect; v2's log-compression
   change *is* +0.027 AUROC. Phase 1 exists because of this.
2. **Correctness fixes are not ablations.** B1, B6, B7, seeding, and the loss denominator have
   no accuracy hypothesis. They are verified by unit test, not by spending training runs proving
   that correct code beats broken code.
3. **Frame F1 is retired as an objective.** It is 3.9 points above trivial on the test set, it
   selects adversarial thresholds during calibration, and it hid every failure that matters. The
   headline objective is a turn-taking cost; AUROC is the model-quality metric.

---

## Phase checklist

| Phase | Name | Gate |
|---|---|---|
| **0** | Correctness and versioning | Every gate in `docs/v2/GATES.md` exists and fails against `v1.0`; v1 re-scored and pinned |
| **1** | Measurement foundation | Pooled paired ΔAUROC 80%-power MDE ≤ 0.010 |
| **2** | Label rebuild | Teacher-vs-TEN F1 ≥ 0.93; train transitions/min ≥ 45 |
| **3** | Data distribution and splits | Zero speaker/room/noise leakage; val calibration slice matches TEN (KS p > 0.10) |
| **4** | Supervision and training loop | ≥ 95% of labelled frames reach the loss; zero examples with no gradient |
| **5** | Model | Zero structurally dead parameters; CPU streaming p95 ≤ 10 ms |
| **6** | Ablation ladder | Positive control reproduces +0.027 AUROC ± harness noise |
| **7** | Operating point | Cost-calibrated beats raw with a paired CI excluding zero |
| **8** | Ship | All six ship-gate criteria met against pinned v1 `last.pt` |

---

## Phase 0 — Correctness and versioning

No accuracy hypothesis. Pure engineering, all of it verified by test.

**0.1 The MPS row-limit fix.** `nn.Conv1d` on MPS silently corrupts rows ≥ 65536 and the model
reshapes to `[batch × num_chunks, 1, 640]` before the conv stack. Chunk the encode call into
32768-row slices — *not* a batch-size cap, which couples an unrelated hyperparameter to a driver
bug, and not eval-on-CPU, which would run training and evaluation on different kernels and
destroy the ability to detect the next silent divergence. Still required after Phase 4: fixed
8 s training crops bring training to 16,000 rows, but validation on full sequences still reaches
188,032.

**0.2 Cross-backend guards.** A test reproducing the 65537-row corruption, and a once-per-run
startup check scoring 32 validation examples on both backends and aborting if AUROC differs by
more than 1e-4. The real lesson of B1 is that there was no such guard.

**0.3 Loss correctness.** Move the sigmoid out of the offline forward path and use
`binary_cross_entropy_with_logits`; keep the sigmoid in the streaming and ONNX paths so the
public contract is unchanged. Drop the `clamp(1e-7, 1-1e-7)`, which has zero gradient outside
its range — 3.63% of TCN test frames saturate to exactly 1.0, so confidently-wrong frames
currently produce *no* gradient. Change the denominator to the weight-mass sum so the loss is
prevalence-invariant across batches, and accumulate numerator and denominator across validation
batches rather than averaging per-batch losses.

**0.4 Seeding.** Nothing is seeded today. Seed python/numpy/torch before model construction,
pass an explicit generator to the DataLoader, seed workers, and log a startup seed report.
Bit-reproducibility is claimed for CPU only; MPS lacks deterministic kernels for several ops.

**0.5 Warmup off-by-one.** The LR lambda returns exactly 0.0 at step 0 and `LambdaLR` applies it
at construction, so the first optimizer step runs at zero learning rate.

**0.6 Config liveness.** At least nine declared config keys are read by nobody
(`precision`, `loss.type`, `loss.class_weighting`, `checkpoint.save_every_epochs`,
`checkpoint.keep_best_metric`, `env.pytorch_enable_mps_fallback`, three `dataloader.*` keys,
plus `target_duration_s`, `speech_occupancy_sanity_band`, `esc50_holdout_fold`,
`gain_range_db`). They currently agree with the hardcoded values, which makes them *worse than
absent* — editing them silently does nothing. Wire or delete each, then make the class of bug
impossible: a config wrapper that records key access, and a smoke test asserting no orphan keys.

**0.7 Run bookkeeping.** Per-epoch weight-only checkpoints, a `metrics.jsonl` per run, `--resume`,
and a `run.json` sidecar recording git commit, config hashes, manifest set id, seeds, device and
versions. v1 overwrote `last.pt` every epoch, so its true validation curve exists nowhere.

**0.8 Config and manifest versioning.** Rename `configs/{train,eval,data}/default.yaml` to
`v1.yaml` and branch `v2.yaml` from each — a mutable file called "default" is exactly what made
v1 unreconstructible from configs alone. Namespace manifests under `data_cache/manifests/v<N>/`
with a `manifest_meta.json` carrying a `manifest_set_id` stamped into every checkpoint and eval
report, so a v1/v2 mix-up is a detectable error rather than a silent one. Full scheme in
[`../CONVENTIONS.md`](../CONVENTIONS.md).

**0.9 Re-score and pin v1.** On CPU, for both architectures × both checkpoints × every eval set.
**Freeze the raw frame probabilities to `.npz`** — this is the single most valuable artifact of
the phase, because every future paired bootstrap, calibration re-sweep and metric redefinition
can then be recomputed against v1 without re-running it. Pin with hashes; add a regression test
asserting the harness recomputes each pinned metric to 1e-9, which turns the baseline into a
guard on the measurement code itself.

*Acceptance: all gates in `GATES.md` exist and fail against `v1.0`; the five predicted
corrections in `../v1/RESULTS.md` reproduce; `runs/v1/` probabilities pinned.*

---

## Phase 1 — Measurement foundation

**This phase gates every accuracy claim in the project.** v1's benchmark resolves ±0.043 AUROC;
the frontend change worth +0.027 is invisible on it.

**1.1 Pair everything.** The cheapest resolution available and v1 never used it. Per-cluster σ
for a *paired* estimator is 1.7× smaller than for an absolute one on AUROC and 2.1× on F1, so a
paired bootstrap on the same 30 files already detects +0.027 at 80% power. Same files, same
resample indices, both systems scored, CI on the per-cluster difference. Free.

**1.2 Expand the cluster count.** TEN-30 is complete and cannot be expanded from source, so it
is frozen as the external-comparability anchor and non-inferiority guard — never split, never
calibrated on. Build TEN-*distribution-matched* sets from held-out data already on disk:
LibriSpeech concat over 48 held-out speakers (48 clusters), FLEURS Swahili test (200 clusters),
AMI at series level (9 meeting clusters, diagnostic only — 9 clusters gives ±0.046 and must be
labelled as such), and a silence-only hard-negative set from ESC-50 fold 5. Pooled TEN-matched
composite: **278 clusters, paired ΔAUROC MDE 0.0082** — a 3.7× improvement.

**1.3 Match on measured statistics, not intuition.** Clip durations from TEN's empirical ECDF,
gap durations from TEN's (p10 202 / p50 463 / p90 926 ms, hard cap 1.4 s), occupancy 0.742
± 0.05. Acceptance is a two-sample KS test against TEN at p > 0.10 on both occupancy and gap
duration.

**1.4 Materialise and hash the benchmark.** ~950 MB of WAV plus a `BENCHMARK.lock` of sha256s. A
benchmark defined by a generator silently changes when the generator changes.

**1.5 Hand-label 50 TEN-like clips.** ~7.5 minutes of audio, 5–7 hours of human work, bootstrapped
from teacher boundaries and hand-corrected. It takes the human-labelled cluster count from 30 to
80 and single-handedly satisfies the ±0.02 AUROC requirement on real labels. Highest
effort-to-leverage item in the project.

**1.6 The metric set.** Headline: turn cost per turn, AUROC, onset **and offset** F1 as a pair,
endpoint latency p50/p90/p99 plus never-released count. Diagnostic: both-class P/R/F1, FAR/MISS,
balanced accuracy, MCC, AP for both classes, per-chunk streaming latency p50/p95/p99 with a
budget-violation rate, and the trivial-predictor floor with its lift — computed inside the same
bootstrap resample so the lift carries a CI. **No metric may appear as a bare float**; every
reported quantity is `{value, ci95}` or a named distribution, enforced by a schema test. Delete
the `rtf` field; it is an artifact, not a number to fix.

**1.7 Slice by tagging.** Every record carries corpus, SNR, noise category, RIR room, RT60,
occupancy, language and duration; every event carries its gap length. All slices are then a
groupby. Required: per-corpus, per-SNR ladder including a mandatory *clean* bin, per-gap-duration,
per-RIR-condition.

**1.8 Cluster bootstrap, B = 10000.** Frame-level resampling must be structurally unreachable —
a naive iid CI over 8,184 autocorrelated frames is 3.2× too narrow. NaN handling is zero-fill,
not drop: v1 silently excluded files where the model predicted no onsets from the precision mean
while still counting them as zero in recall.

*Acceptance: pooled paired ΔAUROC MDE ≤ 0.010; ≤ 0.025 on any headline slice; KS p > 0.10 on all
matched sets; realised per-cluster σ measured from the pinned v1 probabilities.*

---

## Phase 2 — Label rebuild

**The audit's own ordering was wrong here, and the correction reorders the work.** The synthetic
gap retune moves training transitions/min from 11.1 to 13.7 against a TEN target of 57.5. The
teacher relabel moves it to 45.4. **The label fix *is* the fix for the timescale mismatch**; the
gap retune's only real job is killing the >2 s tail. Doing the retune first would measure almost
nothing and invite the conclusion that the diagnosis was wrong.

**2.1 The energy trim cannot do this job — retire it.** Measured, the short pauses that need
recovering sit at a median of −21.5 dB below their own example's speech on AMI, and 63% are
above −25 dB. An RMS gate cannot reach them without clipping unvoiced fricatives and stop
closures. Every trim setting reachable without that damage stops 6.5+ points short of the
teacher. Delete the function and its call site rather than retuning it.

**2.2 Teacher-VAD pseudo-labelling.** Silero VAD via ONNX (weights ship inside the wheel; no
torch hub, no torchaudio). **Raw frame probabilities at a fixed threshold of 0.5, never the
timestamps API** — its internal `min_silence_duration_ms` hysteresis re-erases exactly the short
pauses being bought. Reset state per utterance (LibriSpeech) or per meeting (AMI), never per
window. Threshold 0.5 is within 0.0012 F1 of the sweep peak against TEN's human labels and
matches TEN's human occupancy to four decimals; prevalence fidelity matters more than the fourth
decimal of F1 for a label generator.

**2.3 AMI is a hybrid, not a replacement.** AMI's near-field multi-channel union is authoritative
for *presence*; the teacher works from a mixdown and could miss a quiet distant speaker. So start
from the human union and only *delete* speech where the teacher disagrees, capped at runs ≤ 2.0 s.
Never add speech where humans wrote silence. The cap costs almost nothing (only 3.8% of deleted
mass exceeds 2 s) and guarantees no whole utterance is ever removed by a teacher blackout.

The finding that justifies this: AMI's `transcriber_start`/`end` annotation is utterance-level
and **swallows intra-utterance pauses**. AMI human labels run 10.25 transitions/min; the teacher
on the same audio runs 63.5, against TEN's 57.5. The teacher's disagreement is 22.7% miss and
1.5% false-alarm, and 96.2% of the missed mass is in runs under 2 s. It is recovering pauses, not
dropping speakers.

**2.4 Four validation checks, all build gates.** Teacher-vs-TEN F1 ≥ 0.93 (measured 0.9383 —
against v1's 0.8974 and a trivial floor of 0.8586, the teacher is +8.0 over trivial where v1
managed +3.9). Teacher-vs-AMI FAR ≤ 0.03 (measured 0.015). Deletion-run p99 < 2.0 s and max
< 4.0 s (measured p90 0.755 s, max 3.30 s). And hold one AMI series out of teacher labelling
entirely, reporting under both label sets — any conclusion that only holds under teacher labels
is a distillation artifact.

**2.5 Provenance on every record.** `label_source` distinguishing human, teacher, and hybrid. The
tradeoff is stated plainly: this caps label quality near the teacher's (F1 ≈ 0.94 on TEN) and
compromises the "labels from scratch" framing — not the model's. The AMI hybrid is intersected
with human annotation, so those labels are strictly no worse than AMI's own on presence.

*Acceptance: teacher-vs-TEN F1 ≥ 0.93; AMI FAR ≤ 0.03; deletion-run p99 < 2 s; pooled train
transitions/min ≥ 45.*

---

## Phase 3 — Data distribution and splits

**3.1 Retarget the gap distribution.** Typical gaps [0.15, 1.2] s, long-pause probability 0.05
into [1.5, 4.0] s — down from [0.2, 3.0] with 8% into [3.0, 8.0]. Make assembly duration-driven
rather than count-driven, which finally makes `target_duration_s` load-bearing and gives concat
and AMI examples matching length statistics.

**3.2 The resulting mix.** Roughly 42 h at **0.697 speech and ~52 transitions/min**, against
v1's 0.838 and 11.0 and TEN's 57.5. Two residuals to accept rather than tune away: >2 s
non-speech mass lands near 0.28 versus TEN's 0.00 (TEN's zero is an artifact of its 8.7 s files,
and a model that never sees long silence will drift in deployment), and <0.5 s mass reaches 0.21
versus 0.31 (read speech is inherently more fluent; only more conversational source audio closes
this).

**3.3 AMI splits at series level.** Each `a`–`d` series is a fixed four-person group in a fixed
room, so a per-meeting split leaks all four speakers *and* the room. Six series train, two
validate, one tests — and the test series stays out of teacher labelling as the audit slice.

**3.4 The v1 validation bug.** Meeting discovery ran against the annotation directory (171
meetings) while only 33 have audio, so three of four validation meetings were drawn from
meetings with no audio and silently skipped. Restrict discovery to meetings with both, and
assert.

**3.5 LibriSpeech splits by speaker, using all four official splits.** They are mutually
speaker-disjoint; v1 used two and discarded the other 10.5 h for no reason.

**3.6 Wire the ESC-50 fold holdout.** Three augmentation templates, not one — v1 built a single
template from folds 1–4 and handed it to train, validation and the sanity set alike, so
validation noise was training noise. Assert the fold sets are disjoint.

**3.7 A validation slice that can set a threshold.** Blending to match TEN's prevalence is a
trap — the algebra forces the AMI fraction to ~1% and destroys the conversational signal.
Instead select 10 s windows by occupancy band from the validation series: measured, that yields
~1.2 h at occupancy 0.767 and 67 transitions/min against TEN's 0.752 and 57.5, a 16× larger
calibration set than TEN itself. It runs slightly *more* agile than TEN, which is the safe
direction.

**3.8 Augmentation, ranked.** A noise-probability gate (the model has never seen clean speech
while both eval sets are comparatively clean); an SNR mixture reshaped toward realistic
close-mic levels with the floor raised from −5 dB; RIR pool restricted to meeting/office/booth/
lecture — the auditorium impulse responses leak speech energy at −15.3 dB into frames labelled
silence, which is label corruption dressed as augmentation — plus a direct-path trim, since the
current convolution shifts audio late against unshifted labels by up to 90 ms, concentrated
exactly where boundary metrics are measured. Then wider gain and clipping (the ±3 dB default is
a no-op against 264× per-example loudness variation, and matters more once the frontend is
log-compressed). Babble synthesised from LibriSpeech and labelled **speech**, since both AMI and
TEN label overlap as speech. Speed perturbation deferred past v2.1 — it is the one change that
breaks the length-preserving invariant the whole dataset relies on.

**3.9 Hard negatives.** A `noise_only` manifest kind at ~4% of frames, short examples (6–10 s,
not 20 s, so they do not re-pollute the run-length distribution), with vocal confusers
oversampled ~3.4× and given their own higher SNR floor — at the global floor a laughing clip
does not confuse the model, it *dominates* a frame labelled silence, which is a mislabel rather
than a hard negative. FAR at the calibrated threshold is the number that must move.

**3.10 Do not build the occupancy reweighting.** The teacher relabel dissolves the problem it
was designed to solve: pure-positive AMI windows go from 20.7% to 0.0%, and the resulting
occupancy histogram is healthy. Weight everything equally and replace the dead sanity band with
a build-time assertion.

*Acceptance: zero speaker/room/noise-fold leakage across splits; calibration slice KS p > 0.10
against TEN; pooled train occupancy in [0.66, 0.72]; zero-lag test passes for every RIR in the
pool.*

---

## Phase 4 — Supervision and training loop

**4.1 Fixed-length crops, per example.** 8 s (250 chunks), drawn in the dataset from the
per-item RNG. This is the structural decision that unlocks the rest: every example contributes,
the training loss mask becomes all-ones so the tail-misalignment bug leaves the training path
entirely, batch shapes become identical so MPS compiles once, and — critically — **normalisation
statistics are computed only over real audio**, which is a hard prerequisite for adding any
normalisation at all. With ragged batches, padded chunks feed a large negative constant into the
batch statistics.

Rejected: a length-bucketed sampler. Length correlates almost perfectly with example kind (AMI
windows are all exactly 20 s), so bucketed batches would be nearly pure-AMI or nearly
pure-synthetic — biasing both the gradient and any normalisation statistics. And with fixed-length
crops there is no padding left to bucket away.

Accounting: **100% of labelled frames supervised at 42% of v1's compute per epoch** — roughly
3.1× the supervision per unit compute.

**4.2 A loss warm-up mask.** The recurrent state is zero at each crop start, a regime that is
3.2% of training frames at 250 chunks but 0.08% of a five-minute deployment stream — a 40×
over-representation. Exclude the first 8 chunks of each crop from the loss.

**4.3 Sampler-carried epoch.** A sampler yielding `(epoch, index)` tuples. The sampler lives in
the main process and *is* re-iterated every epoch even under persistent workers, so the epoch
rides an existing always-fresh channel. This removes the mutable-state bug class permanently
rather than papering over it, and it is what makes any epoch-dependent schedule possible at all.

**4.4 Model selection on AUROC, and ship `last`.** The class-weighted loss makes the model output
a *reweighted* posterior whose calibrated threshold is the weight ratio, not 0.5 — v1's weights
put it at 0.162, and the observed swept optima were 0.15/0.15/0.20/0.10. Selecting on F1 at a
fixed 0.5 costs ~2.5 points and selects on noise. AUROC is threshold-free and is the only metric
that showed signal. Add weight EMA and evaluate both; it removes selection variance entirely for
the cost of one extra parameter copy and zero extra runs.

**4.5 The schedule.** Report supervised frames, not epochs — epochs are not comparable across
v1 and v2. Roughly 40 epochs at ~155 steps delivers v1's total supervision at 42% of its compute.
**Hold batch size, LR, warmup, decay and clipping identical to v1**, so that when v2 wins the
cause is attributable; the MPS fix makes batch size a free variable again, and the right move is
not to spend it. Validate cheaply before committing: overfit-one-batch, the supervision-density
assertion, an LR range test, the A/B harness, and a quarter-length dry run — under an hour total.

*Acceptance: ≥ 95% of labelled frames reach the loss; zero examples with no gradient; augmentation
fingerprints differ across epochs for ≥ 90% of examples under persistent workers.*

---

## Phase 5 — Model

**5.1 Log-compress the frontend.** `log(mag + 1e-5)`. **+0.027 AUROC and −13% validation loss**,
measured in a controlled paired A/B — the largest single accuracy change available and one line.
Mechanism: inputs span four decades with 264× per-example variation and a 61× tilt across
frequency, and frequency is the *channel* dimension of an unnormalised convolution, so the first
layer receives 15× less gradient than it should.

Freeze both epsilons exactly as they are. The `1e-9` inside the magnitude sqrt already floors the
input above the log epsilon, and the +0.027 was measured with it present.

Explicitly rejected, both measured negative: per-chunk instance normalisation (−0.112 — it
destroys the absolute-level cue VAD depends on) and per-bin affine standardisation (−0.011 — it
removes spectral tilt, which is itself a speech cue).

**5.2 BatchNorm in the conv stack, not on the raw bins.** +0.003 on top of log, folds into the
preceding convolution at export so inference cost is zero. Its real job is preventing the dead-
channel pathology, not fixing input scale — log already does that.

**5.3 Kill the dead parameters.** 20,480 parameters (9.7% of the CRNN) sit in kernel taps that
only ever multiply zero padding, verified as gradient-zero and still at initialisation RMS in the
trained checkpoint. A further 19 of 128 encoder output channels are permanently zero, so the
recurrent layer's effective input width is 109. Replace padded strided convolutions over a
4-frame axis with valid convolutions that consume the four frames exactly. Keep frequency-as-
channels with a `k=1` first layer — that is a *dense* map over frequency and strictly more
expressive than convolving across it. Target: param-matched to v1, so "v2 is better" is never
confounded with "v2 is bigger".

**5.4 Reimplement the depthwise convolution.** Unfold/multiply/sum instead of `groups=C`:
bit-identical, and it takes the TCN streaming step from 11.33 ms to 0.084 ms, converting a 3.2×
budget miss into headroom. Keep the module as the parameter holder so state dicts and the ONNX
export path are unchanged. Add `set_num_threads(1)` to every streaming entry point — worth 7.6×
on its own, because threading is pure overhead at these tensor sizes.

**5.5 Frontend implementation split.** `rfft` for training and PyTorch streaming (5.5× faster,
42% of the CRNN's single-thread step); the convolution-as-DFT retained for ONNX export, where
maximum runtime compatibility matters more than speed. Gate the two against each other on
log-magnitude, probability, and end-to-end AUROC — the reported 1.9e-4 parity is on *linear*
magnitude, which is 1.6% relative and becomes a 0.016 shift after the log.

**5.6 Both architectures, co-trained off one dataloader.** They are statistically
indistinguishable on the current evidence and the ranking reverses between slices, so running the
ladder twice buys nothing. The pipeline is dataloader-bound, so a second model on the same batch
stream is nearly free — and it gives perfectly paired comparisons, which is exactly the
resolution lever from Phase 1. Run the ladder on the CRNN alone; bring the TCN in at the final
config.

**5.7 Two speculative arms.** A dense (non-separable) dilated temporal stack, since the
separable factorisation's only justification is a FLOP count that is irrelevant at this size. And
a future-frame auxiliary head — nothing architectural fixes a zero-lookahead offset except
supervising the future, and offsets are 20 points worse than onsets while endpoint dead-air
dominates the turn cost. Judge it on offset F1 and dead-air percentiles, never on frame F1.

*Acceptance: zero structurally dead parameters; ≤ 2% dead output channels; CPU streaming p95
≤ 10 ms and p99 ≤ 16 ms for every shipped architecture.*

---

## Phase 6 — Ablation ladder

Two-tier: a 500-step paired screen for every arm (~10 min), full runs only for survivors (~4 h).
That is the difference between ~40 h and ~130 h of compute.

| Rung | Arm | Expected | Judge on |
|---|---|---|---|
| R0 | v1 architecture + Phase 0 fixes | — | reference |
| R1 | + supervision bundle (Phase 4) | largest single delta | AUROC |
| R2 | + log compression | **+0.027 (known)** | **positive control** |
| R3 | + BatchNorm | +0.003 | AUROC, dead-channel count |
| R4 | + encoder reshape | ~0 ± 0.005 | params, dead params, latency |
| R5 | frequency spacing (3 arms) | ≤ ±0.005 | sample efficiency |
| R6 | temporal layer (4 arms) | wash | **latency p95, params**, then AUROC |
| R7 | future-frame aux head | unknown | **offset F1, dead-air p50/p90** |
| R8 | selection metric, EMA, operating point | — | post-hoc, zero training runs |

**R2 is not there to learn something new — it is there to validate the harness.** If a paired
three-seed screen does not reproduce roughly +0.027 for a change already measured at +0.027 under
the same conditions, the harness is broken and every rung below it is worthless. Stop and fix it.

Bundled without ablation, because they have no accuracy hypothesis: all Phase 0 correctness work,
and all implementation-level latency work (bit-identical by construction — ablating it for
accuracy would be a category error). The supervision bundle is one rung, full-run only; a
500-step proxy cannot adjudicate a change whose benefit accrues across epochs.

Pre-register the accept threshold for each rung before running it. With eleven arms at the
achievable resolution, post-hoc selection will find a spurious winner.

*Acceptance: R2 reproduces its known effect; every rung pre-registered; every number a cluster
bootstrap.*

---

## Phase 7 — Operating point

**7.1 Calibrate against a turn-taking cost, not frame F1.** Measured on v1: every val-F1- and
val-onset-F1-optimised configuration transfers to the test set *worse than no post-processing at
all*. Frame F1 on an 85%-speech validation set is dragged toward the majority class — the sweep
is monotone-decreasing from its low end, which is the signature.

**7.2 Fix the cost constants independently.** The only real constant is the ratio — "one
interruption is worth 1.67 s of extra silence" — elicited as a policy statement and then held
fixed, never fitted to the VAD. Write the elicitation into the config as a comment; that sentence
is the specification.

**7.3 Search all four parameters jointly** under a hard `theta_off ≤ theta_on` constraint, on
cached probabilities so the full grid runs in seconds. Include a zero onset-debounce option: every
good configuration found sits at the grid floor, so the grid must be able to say the debounce is
worth nothing. Break ties within one standard error by lower debounce, then lower dead-air p90.

**7.4 Report calibration error, not just test error.** Nested resampling — re-run the argmin on
each bootstrap of the calibration set, apply each winner to the full test set, and report the
spread. Plus an oracle gap (test metric at the test-set argmin minus at the calibration-selected
one) and the selection stability of each parameter. v1's within-benchmark split-half showed a
7-point oracle gap and a θ spread of 0.30; if v2's stability fails, the calibration set is too
small and no operating point should be believed until it grows.

**7.5 A mandatory control arm.** Every report emits the cost-calibrated operating point *and* raw
thresholding side by side. Shipping requires the former to beat the latter with a paired CI
excluding zero. v1 fails this gate — that is precisely the regression it exists to prevent.

*Acceptance: oracle gap ≤ 0.03 on onset F1; θ selection spread ≤ 0.15; calibrated beats raw with
a paired CI excluding zero.*

---

## Phase 8 — Ship

A v2 candidate ships only if, against the pinned v1 `last.pt`:

1. Paired ΔAUROC on the pooled TEN-matched composite: 95% CI lower bound **> 0**.
2. Paired Δ turn-cost-per-turn: 95% CI upper bound **< 0**.
3. TEN-30 paired ΔAUROC: CI lower bound **> −0.005** — non-inferiority on the external
   human-labelled anchor.
4. **Conversational slice frame F1 exceeds its always-speech floor with a CI lower bound > 0.**
   v1 fails this. It is the single criterion that says whether v2 solved the actual problem.
5. CPU streaming p95 within budget for every shipped architecture.
6. False alarms per hour on the silence-only set strictly below v1's.

---

## Sequencing

```
Phase 0 ──┬── Phase 1 ──┬── Phase 6 ── Phase 7 ── Phase 8
          │             │
Phase 2 ──┴── Phase 3 ──┤
                        │
          Phase 4 ──────┴── Phase 5
```

Phases 0 and 2 can start in parallel. **Phase 1 gates Phases 6–8 absolutely** — running the
ladder before the benchmark exists means reading noise, which is exactly how v1 concluded that
one architecture fit the validation set better than the other from two corruption patterns.

Phase 2 must precede Phase 3.1: the label rebuild is what moves the timescale, and doing the gap
retune first would measure almost nothing.

## What would make v2 fail

Stated in advance, so it is recognisable:

- **Shipping before Phase 1.** Every accuracy claim would be unfalsifiable, exactly as in v1.
- **Trusting the teacher.** It caps label quality near its own and is validated on 30 human-
  labelled files. The held-out never-relabelled series exists for this reason.
- **Tuning the residual distribution mismatches to zero.** TEN having no non-speech runs over 2 s
  is an artifact of its file length, not a property of conversation.
- **Reading frame F1 as progress.** It is 3.9 points above trivial and hid every failure that
  mattered.
- **Assuming ~19 h of real conversational speech is enough.** It is the binding constraint on
  everything above, and nothing in this roadmap fixes it.
