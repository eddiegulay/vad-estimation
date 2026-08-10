# v2 Design Notes — measured inputs

The numbers the roadmap depends on, committed so they are never conversation-only folklore.
Where a measurement was produced this session, its script is in
[`measurements/`](./measurements/); where it must be re-produced under a pinned protocol, that
is a WP0.5 spike (see [`BETS.md`](./BETS.md)). Sources: the v1 audit
([`../v1/AUDIT.md`](../v1/AUDIT.md)), the four v2 design reports (2026-08-08 session), and the
independent verification pass (scripts `m1`–`m7`).

---

## 1. Statistical resolution (m1_paired_sigma.py, m1_per_file.npz)

Measured on CPU from v1 `last.pt` (both architectures) over the 30 TEN files:

| quantity | value |
|---|---|
| σ_absolute per file, AUROC | 0.0714 (CRNN) / 0.0694 (TCN) |
| **σ_paired per file, ΔAUROC** | **0.0367** |
| absolute/paired ratio | **1.95×** AUROC, 1.69× F1 (the roadmap's first draft had these transposed) |
| paired MDE, 80% power, n=30 | **0.0188** — so +0.027 is detectable on TEN-30 alone, paired |
| paired MDE at n=278 (if σ transfers) | ≈ 0.0062 |

Caveats: σ_paired is specific to the system pair being compared; the n=278 projection assumes σ
transfers across cluster types and must be re-measured from the pinned v1 probabilities on the
real composite at WP4 exit (that re-measurement is the Phase 1 acceptance).

## 2. TEN statistics, with the convention pinned (m2_manifest_stats.py)

Two conventions coexist and both are correct; the plan's first draft mixed them silently:

| quantity | pooled | per-file macro |
|---|---|---|
| TEN speech occupancy | 0.7522 | 0.7421 |
| TEN transitions/min | 54.2 | 57.3 |

**Convention for all v2 gates: pooled.** The Phase 2/3 gate "train transitions/min ≥ 45" is
pooled. TEN interior gaps: p10/p50/p90 = 202/463/926 ms, max 1.31 s (hence the 1.4 s generator
cap); non-speech mass < 0.5 s = 0.39; > 2 s = 0.00 (an artifact of 8.7 s mean file length — do
not tune training to it). v1 train, pooled: occupancy 0.8378, 10.96 transitions/min.

## 3. Compute shape (m7_bound.py) — falsifies "dataloader-bound"

| quantity | measured |
|---|---|
| per-item `__getitem__` with full augmentation | ~12 ms → ~0.19 s per batch-64 across 4 workers |
| MPS train step, CRNN, 625 chunks | 0.54 s |
| MPS train step, CRNN, 250 chunks (v2 regime) | 0.22 s |
| MPS train step, TCN, 250 chunks | 0.18 s |

The pipeline is **model-bound** in every regime. Co-training both architectures costs ~1.8× per
step (0.22 → 0.40 s), not "nearly free". Kept for perfect pairing; cost budgeted. The 42%
per-step compute ratio for 8 s crops (0.22/0.54) is confirmed; the "40 epochs ≈ v1's total
supervision" half is arithmetic pending the rebuilt manifest's example count.

## 4. Cold-start transient (m5_warmup.py) — resizes the warm-up mask

Cold-start suffix probabilities vs full context, per chunks-since-start, 141 probes/chunk on the
trained v1 checkpoints: CRNN median |Δp| at chunk 8 is 0.036 **but p95 is still 0.64**, staying
above 0.3 through chunk ~19; TCN p95 first drops below 0.05 at chunk **29** (its receptive
field). The 8-chunk mask under-covers the transient it targets. v2 derives k from the curve
(smallest k with median |Δp| < 0.02) and re-measures on the trained v2 model — a model trained
with masked starts may shorten its own transient.

## 5. Benchmark composition (m4_clusters.py) — corrected

What exists on disk: LibriSpeech 146 speakers total (**zero currently held out** — v1 trained on
all of dev-* and validated on all of test-*); FLEURS sw_ke test 487 source utterances; AMI 9
audio-bearing series (EN2001×1, ES2002–5×4 each, IS1000–1×4 each, TS3003–4×4 each); ESC-50
fold 5 = 400 clips; TEN = 30 files, complete, inexpandable.

The pooled TEN-matched composite for paired scoring:

| set | clusters | cluster key | labels | role |
|---|---|---|---|---|
| TEN-30 | 30 | file | human | frozen anchor; never split, never calibrated on |
| LS-CONV | 48 | speaker (named reservation, excluded from all training) | teacher + exact synthetic gaps | CI workhorse |
| FLEURS-CONV | 200 | source utterance | teacher + exact synthetic gaps (**re-cut** — raw FLEURS is 93.1% speech and cannot pass the KS gate as-is) | CI workhorse |
| **composite** | **278** | | | ship gates 1–2 |

Honest caveat, stated wherever the composite is reported: it is 72% Swahili read speech by
cluster count. The per-set breakdown is always reported alongside, and the conversational claim
rides on the AMI diagnostic + ship gate 4, not the composite. The AMI diagnostic set uses the
**val + test series only** (3 series — using all 9 would put the six training series in a test
set). Teacher-labelled sets may adjudicate v2-vs-v2 paired comparisons; superiority claims over
any teacher-derived system come from human-labelled sets only (TEN-30, AMI, HUMAN-50).

## 6. Teacher-label protocol (design-session numbers; WP0.5 spike (a) re-pins them)

Protocol: silero-vad (pinned version + weights sha256 folded into `manifest_set_id`), ONNX
backend, raw frame probabilities at threshold 0.5, state reset per utterance (LibriSpeech) / per
meeting (AMI), never the timestamps API (its internal hysteresis re-erases the short pauses
being recovered).

| measurement | value |
|---|---|
| teacher vs TEN human labels, F1 @ 0.5 | 0.9383 (threshold sweep: 0.3 → 0.9395, 0.4 → 0.9395, 0.5 → 0.9383, 0.7 → 0.9297) |
| teacher occupancy on TEN @ 0.5 | 0.7520 vs human 0.7522 — prevalence-faithful |
| teacher vs AMI human union, FAR / MISS | 0.015 / 0.227 |
| missed-mass shape | 96.2% of deleted mass in runs < 2 s; deletion runs median 0.192 s, p90 0.755 s, max 3.30 s |
| AMI human vs teacher transitions/min | 10.25 vs 63.5 (TEN: 57.3 macro) — AMI's utterance-level annotation swallows intra-utterance pauses |
| gap-retune alone vs teacher alone, LS concat transitions/min | 13.7 vs 45.4 (v1: 11.1) — the label fix IS the timescale fix |
| hybrid deletion-cap sensitivity | cap 0.5 s → occ 0.726; 1.0 s → 0.660; **2.0 s → 0.630**; 3.0 s → 0.626 |
| pure-positive AMI train windows after relabel | 20.7% → 0.0%; ≥0.95-occupancy windows 36.9% → 0.4% |

## 7. Augmentation targets (design-session; concrete values the roadmap references)

> **PARTIALLY FALSIFIED 2026-08-10 — see [`DATA-QC.md`](./DATA-QC.md) §F10.** Measured against
> TEN, this policy is a *regression* on acoustic match versus v1's (KS p=0.005 vs p=0.68 on
> speech/non-speech contrast). `noise_prob 0.85` leaves 15% of examples with −100 dBFS digital
> silence; the SNR mixture's 18 dB centre is 10 dB above v1's and was never measured; the gain
> range's −6 dB mean shifts training below the test distribution. The RIR room-name list and the
> named vocal-confuser categories are superseded by measured filters. The values below stand only
> where DATA-QC.md's amendment table does not override them; the level-axis values must be
> re-derived by fitting `scripts/qc/qc_contrast.py` before WP6.

- `noise_prob: 0.85` (v1: unconditional — the model never saw clean speech).
- SNR mixture: 55% normal(18, 6) clipped [10, 35] dB; 32% uniform [3, 10]; 13% uniform [−2, 3].
  Floor raised from −5 dB; vocal-confuser clips get a separate floor of **+6 dB** (at −5 dB a
  laugh does not confuse the model, it dominates a frame labelled silence).
- Gain: **[−18, +6] dB**, plumbed from YAML (v1's ±3 dB was an unreachable constructor default);
  clipping p 0.10, ceiling uniform [0.7, 1.0] of peak.
- RIR pool: meeting/office/booth/lecture (86 of 214; RT60 median drops 1.30 → 0.35 s;
  aula_carolina leaked speech at −15.3 dB into silence-labelled frames); `rir_prob 0.35`;
  direct-path trim mandatory (uncompensated delay median 5.9 ms, max 90.2 ms).
- Hard negatives: `noise_only` kind, 6–10 s examples, ~4% of training frames, vocal confusers
  oversampled ~3.4×, 60% of noise-only examples routed to confuser classes.
- Babble: 4–8 summed LibriSpeech train-speaker utterances, SNR ≥ 5 dB, labelled **speech**
  (AMI and TEN both label overlap as speech), 10% of concat examples.
- Speed perturbation: deferred past v2.1 (breaks the length-preserving invariant).

## 8. Row-limit derivations

MPS `nn.Conv1d` corrupts rows ≥ 65536 (torch 2.13.0; reproduced twice independently). v1 val's
longest padded batch: 81.2 s → 2,537 chunks × 64 = **162,368 rows** (the 188,032 figure in an
earlier draft was the train manifest's longest example — same conclusion). v2 training at
64 × 250 = 16,000 rows has 4× margin; v2 full-sequence eval still exceeds the limit, so the
chunked encode is not optional. Chunk size 32768.

## 9. Expected-divergence register

Seam proofs against v1's pinned bytes must match to tolerance **or** match one of these
pre-registered, deliberate divergences. A third outcome stops the line. Closed at WP1; nothing
joins after its seam proof has run.

| # | divergence | direction | why |
|---|---|---|---|
| D1 | frame count `floor(n/512)` vs v1's `round()` | v2 has ≤ v1 frames on 149/563 val items | fixes the unmasked tail pairing labels with zero-padding probabilities |
| D2 | NaN handling zero-fill vs drop | v2 precision ≤ v1 on affected files (0/30 today) | removes the silent upward bias |
| D3 | sweep uses `>=` matching the state machine's `>=` | boundary-value decisions may differ | operator consistency |
| D4 | `trim_internal_silence` removed | concat occupancy differs from v1's post-trim values | superseded by teacher labels |
| D5 | AMI val meeting selection | v2 val ≠ {EN2001a} alone | v1's draw included meetings with no audio |

## 10. FSTTM cost constants

C_cutin = 5000 is literature-anchored (Raux & Eskenazi 2009). c_deadair = 3000/s is, per the
source note itself, "tuned so 6 cues ≈ human modal gap; **calibrate per deploy**" — a knob, not
a constant. The only real parameter is the ratio (1.67 s of silence per avoided interruption).
WP0.5 spike (d) sweeps the ratio over [1.0, 2.5] s on cached v1 probabilities: if the selected
operating point is stable across the range, the knob is immaterial; if not, the ratio becomes a
reported sensitivity and the fallback is shipping the operating-point frontier rather than a
point.
