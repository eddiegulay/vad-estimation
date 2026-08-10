# v2 Decisions Log

Append-only, dated. Each entry records what was decided and why, so a later reader can tell a
deliberate choice from an accident. v1's decisions log was embedded in its roadmap and is
archived at [`../v1/ROADMAP.md`](../v1/ROADMAP.md).

---

**2026-08-08 — v1 is preserved as artifacts, not as a recipe.** v1 training was unseeded (model
init, dataloader shuffle and crop offset), and MPS adds its own nondeterminism, so re-running
v1's code produces *a* model, not *the* model. `runs/v1/` commits the weights, exports, reports,
manifests, curves and environment — ~7.3 MB, once. Optimizer state is stripped (61% of each
checkpoint); v1 is frozen, so resume is moot.

**2026-08-08 — Code evolves in place on `master`; versions are tags plus artifact directories.**
Versioned source packages were rejected: v2 changes every subpackage, so it would mean permanently
maintaining a duplicate of a library the audit measured as defective. Config *directory*
namespaces were rejected as the primary mechanism because they promise a reproducibility that
stops holding the moment code changes; version-stamped config *filenames* give the coexistence
that is actually needed.

**2026-08-08 — Archived docs are never edited; corrections are additive.** `docs/v1/ROADMAP.md`
is byte-identical to what shipped, plus an errata banner. Twelve now-known-wrong claims are
recorded in `ERRATA.md` against the audit findings that correct them.

**2026-08-08 — The measurement foundation is built before anything is trained.** v1's benchmark
resolves ±0.043 AUROC and the best available accuracy change is +0.027, so v1 could not have
detected its own best improvement. Phase 1 gates Phases 6–8 absolutely.

**2026-08-08 — Paired estimators become the primary decision instrument.** Per-cluster σ for a
paired comparison is 1.7× smaller than for an absolute one on AUROC. This is free resolution that
v1 never used, and it is why a 30-file benchmark was not as hopeless as the absolute CIs implied.

**2026-08-08 — The teacher relabel precedes the gap retune, reversing the audit's suggested
order.** Measured: the gap retune alone moves training transitions/min from 11.1 to 13.7 against a
target of 57.5; the teacher relabel moves it to 45.4. The label fix *is* the fix for the timescale
mismatch. Doing the retune first would measure almost nothing and invite the wrong conclusion.

**2026-08-08 — AMI's human annotations are authoritative for presence, not for timing.** AMI's
utterance-level `transcriber_start`/`end` swallows intra-utterance pauses: human labels run 10.25
transitions/min against the teacher's 63.5 on identical audio, with TEN at 57.5. So AMI labels are
a hybrid — start from the human union, delete only where the teacher disagrees, cap deletions at
2.0 s, never add speech where humans wrote silence.

**2026-08-08 — Accept label distillation, with an audit slice.** Teacher pseudo-labelling caps
label quality near Silero's (F1 ≈ 0.94 on TEN, against v1's 0.897 and a trivial floor of 0.859)
and compromises the "labels from scratch" framing — not the model's. One AMI series stays
teacher-free so any conclusion that only holds under teacher labels is identifiable as an
artifact.

**2026-08-08 — Frame F1 is retired as an objective.** It sits 3.9 points above a parameterless
predictor on the test set, it selects adversarial thresholds during calibration (every
val-F1-optimised operating point transferred worse than no post-processing at all), and it hid a
0.50 false-alarm rate and a third of turn-ends never being released. Headline objective is a
turn-taking cost; model quality is AUROC.

**2026-08-08 — Fixed-length 8 s crops rather than a length-bucketed sampler.** Bucketing sorts by
length, and length correlates almost perfectly with example kind, so bucketed batches would be
nearly pure-AMI or nearly pure-synthetic — biasing the gradient and any normalisation statistics.
Fixed crops also make normalisation statistics clean, which is a prerequisite for adding
BatchNorm at all.

**2026-08-08 — Batch size, LR, warmup and clipping stay at v1's values.** The MPS fix makes batch
size a free variable again; the right move is not to spend it. Holding them fixed keeps any v2
win attributable.

**2026-08-08 — Both architectures continue, co-trained off one dataloader.** They are
statistically indistinguishable and their ranking reverses between slices, so running the ablation
ladder twice buys nothing. The pipeline is dataloader-bound, so a second model on the same batch
stream is nearly free and gives perfectly paired comparisons.

**2026-08-08 — v2 is a fresh build, not a patch series.** `src/vad/` and `tests/` are replaced
wholesale in one cutover commit; v1 survives as the tag, the `runs/v1/` artifacts, and measured
knowledge. Carried modules (interval algebra, TEN parser, asset converters, augment primitives,
hysteresis machine, config loader) enter via `git checkout v1.0 -- <path>` with their test
files, and the review diff must be import-paths only. Everything else is rewritten from spec
with the v1 tests imported as the spec's executable half. An expected-divergence register
(DESIGN-NOTES §9) closes at cutover: seam proofs match v1's pinned bytes, match the register, or
stop the line.

**2026-08-08 — Pin before delete.** The v1 probability files that anchor the new measurement
harness are generated by v1's own code in a worktree at the tag, before the cutover. They do not
exist yet (`runs/v1/` holds checkpoints and reports, not probabilities), which makes WP0 the
first work package and a blocker for everything.

**2026-08-08 — Correction: "dataloader-bound" is withdrawn.** Measured (DESIGN-NOTES §3): the
pipeline is model-bound in every regime, and co-training the TCN costs ~1.8× per step, not
"nearly free". Co-training is retained for its perfect pairing with the real cost budgeted. This
supersedes the co-training rationale in the 2026-08-08 entry above.

**2026-08-08 — A named 48-speaker LibriSpeech benchmark reservation.** The earlier claim that
held-out speakers existed was false — v1 used all 146. The reservation is excluded from every
training manifest and enforced by the leakage gate.

**2026-08-08 — The teacher is a pinned, fingerprinted dependency.** silero-vad version and
weights sha256 are recorded in pyproject, folded into `manifest_set_id`, and stamped into
`run.json` — labels are teacher-derived artifacts, so a silent wheel upgrade must not be able to
change every label under an unchanged manifest identity.

**2026-08-08 — R2 becomes a two-stage protocol.** A bare "reproduce +0.027 or the harness is
broken" control is circular: non-reproduction could equally mean the effect shrank under fixed
labels and the new mix. Stage 1 replicates linear-vs-log on v1 data (validates the harness);
stage 2 measures the effect on rebuilt data, and that number — not +0.027 — is R2's
pre-registered expectation.

**2026-08-08 — Ship gate 3 widened to −0.015.** The −0.005 margin was below TEN-30's measured
paired resolution (±0.013) and silently demanded a +0.014 win while claiming non-inferiority.

**2026-08-08 — R1 split into R1a/R1b.** One rung confounded the supervision fixes with the data
rebuild, so the label rebuild's own effect would never have been measured. R1a = new supervision
on v1 labels; R1b = + rebuilt labels.

**2026-08-08 — No decision on a class-B bet before its spike.** The bet register (BETS.md) is
normative: claims are A (measured), B (spike specified), C (rung named), or D (bet with named
fallback). A falsified claim left unamended in the docs is a correctness bug. Two of the plan's
own premises were measured-false before execution began — that is the rule paying for itself.

**2026-08-10 — No new data is acquired for v2.** Confirmed against the filesystem: the corpora
on disk are the corpora v2 gets. This makes bet D3 (~19 h of conversational speech in 2 acoustic
regimes) permanent rather than provisional, caps the achievable data quality, and redirects
every data improvement toward protocol and preparation. Recorded so no later reader mistakes the
ceiling for an oversight. (Noted for a future version: 171 AMI meetings are annotated on disk but
only 33 have audio — the cheapest possible expansion if the premise is ever relaxed.)

**2026-08-10 — The augmentation policy is fitted, not chosen.** The v2 policy in DESIGN-NOTES §7
was measured to be an acoustic *regression* against v1's: v1 is statistically indistinguishable
from TEN on speech/non-speech contrast (KS p=0.68), the v2 draft is not (p=0.005), missing toward
easier-than-reality. `noise_prob 0.85` emitted −100 dBFS digital silence, the SNR mixture sat
10 dB above v1's centre, and the gain range's mean shifted levels below the test set. WP1.5 now
fits the level-axis parameters against `qc_contrast.py`, and G16 makes the realised match a
launch condition. This is the first gate in the project whose failing baseline is the plan's own
draft rather than v1.

**2026-08-10 — Pool membership is decided by measurement, not curation.** The ESC-50
vocal-confuser category list found 35 of the 79 most speech-contaminated clips; the RIR
room-name list was a proxy for acoustic properties that are directly measurable. Both are
replaced by per-file measured criteria, stored in `asset_qc.json` and enforced by G15.

**2026-08-10 — Scarcity is answered with protocol.** Two changes that would not otherwise earn
their compute: ship gate 4 is adjudicated leave-one-series-out across all 9 AMI series rather
than on a 3-series holdout, and the FLEURS benchmark samples from all 3,441 usable files rather
than a fixed 200. Both convert compute into statistical resolution, which is the only exchange
rate available when acquisition is off the table. LOSO applies to the conclusion, not the search
— rungs are still screened on the single split.
