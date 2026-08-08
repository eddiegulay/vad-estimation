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
