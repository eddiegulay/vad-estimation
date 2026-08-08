# v1 Results — corrected

Standalone statement of what v1 actually achieved, as understood after the 2026-08-08 audit.
Read this rather than the tables in [`ROADMAP.md`](./ROADMAP.md), which are archived verbatim
and contain the errors catalogued in [`ERRATA.md`](./ERRATA.md).

Artifacts: `runs/v1/`. Audit: [`AUDIT.md`](./AUDIT.md).

---

## Headline

Two models, 50 epochs each, on a ~40 h local kit. Both learned something real. **Neither is
usable for conversational turn-taking, and the reason is visible only in metrics v1 did not
compute.**

| | CRNN | TCN | always-speech baseline |
|---|---|---|---|
| Trainable parameters | 210,561 | 179,585 | 0 |
| **TEN F1 @ 0.5** | 0.8974 | 0.8961 | **0.8586** |
| TEN precision / recall | 0.852 / 0.948 | 0.843 / 0.956 | 0.752 / 1.000 |
| **TEN AUROC** | **0.9046** | 0.8876 | 0.500 |
| Silence-class F1 | 0.6030 | 0.5773 | — |
| False-alarm rate | **0.5005** | 0.5404 | 1.000 |
| Miss rate | 0.0518 | 0.0437 | 0.000 |
| Balanced accuracy | 0.7238 | — | 0.500 |
| MCC | 0.524 | — | 0.000 |
| Onset F1 @ 200 ms | 0.5903 | 0.5320 | — |
| **Offset F1 @ 200 ms** | **0.3819** | 0.3367 | — |

**The F1 lift over a parameterless baseline is +3.9 points.** AUROC is the only frame metric
that separates a trained model from a constant one (random init scores 0.267). Report both, or
neither.

**Half of all true-silence frames are classified as speech** (FAR 0.5005), which the reported
speech-class F1 of 0.897 completely hides.

---

## Validation, corrected

The numbers reported during training were computed from MPS-corrupted activations (E2). These
are the true values, re-scored under the row limit:

| | val loss | val F1 @ 0.5 | val AUROC | calibrated F1 (θ) |
|---|---|---|---|---|
| CRNN ep28 (`best.pt`) | 0.2939 | 0.9211 | 0.9441 | 0.9466 (0.15) |
| **CRNN ep49 (`last.pt`)** | **0.2781** | 0.9198 | **0.9489** | **0.9493** (0.15) |
| TCN ep19 (`best.pt`) | 0.3387 | 0.9226 | 0.9323 | 0.9421 (0.20) |
| **TCN ep49 (`last.pt`)** | **0.2987** | 0.9071 | **0.9415** | **0.9446** (0.10) |

Three consequences: there is no overfitting (E3); `last.pt` beats `best.pt` on every
threshold-free metric for both architectures (E4); and the architecture comparison that
selected the CRNN was reading two different corruption patterns (E2).

Validation is 87.2% speech, so its always-speech floor is F1 0.9315 — the corrected val F1 of
0.9211 at threshold 0.5 is *below* it.

---

## Per-slice: where the models actually fail

| val slice | prior | always-speech floor | CRNN | TCN |
|---|---|---|---|---|
| `concat_synthetic` (n=300) | 0.828 | 0.9059 | 0.9145 | **0.9294** |
| `ami_window` (n=263) | 0.875 | **0.9334** | 0.9322 | 0.9110 |

**Both models score at or below the trivial floor on the AMI slice** — the only genuinely
conversational data in the kit. The ranking also reverses between slices, while the pooled val
F1 (0.9212 vs 0.9227) shows nothing. This is the most important single result in v1 and it was
invisible until the audit.

---

## Latency, corrected

The reported RTF figures are artifacts (E1). Re-measured:

| | offline `forward_full` (warm) | streaming p50 | p95 | p99 | budget |
|---|---|---|---|---|---|
| CRNN, CPU | 0.0031 | 0.59 ms | 1.85 ms | 3.92 ms | 32 ms ✓ |
| CRNN, MPS | — | 1.25 ms | 7.80 ms | 14.1 ms | 32 ms ✓ |
| TCN, CPU | 0.00037 | 17.8 ms | **103 ms** | **179 ms** | 32 ms ✗ |
| TCN, MPS | — | 2.21 ms | 11.1 ms | 20.1 ms | 32 ms ✓ |

The TCN is faster offline and **not shippable on CPU**. Root cause is PyTorch's
grouped-convolution dispatch, ~2 µs per group — a dense `groups=1` convolution doing 128× the
arithmetic runs 3.5× faster, and an unfold/multiply/sum rewrite is bit-identical and 61× faster.

**The decision to ship the CRNN was correct. Every reason given for it in the roadmap was
wrong.**

---

## Turn-taking cost

Under the FSTTM cost policy from the companion research vault (C_cutin 5000, c_deadair 3000/s):

| configuration | frame F1 | onset F1 | cost/turn | endpoint dead-air p50 |
|---|---|---|---|---|
| raw `p > 0.5`, no post-processing | 0.8974 | 0.5903 | 5099 | 608 ms |
| **as shipped** (θ 0.15/0.35, 256/192 ms) | 0.8857 | 0.4694 | **6491** | **1552 ms** |
| TEN-oracle parameters | 0.8972 | 0.7200 | **2585** | — |

**The shipped post-processing is a statistically significant regression against doing nothing**
(E6). And of 114 observable turn-ends, the raw model never releases the floor at all on 37 of
them; the shipped configuration fails on 59. A model that never signals end-of-turn on a third
of turns is unusable for the application, and F1 0.8974 says nothing about it.

---

## Statistical resolution

Cluster bootstrap over the 30 TEN files:

| | CRNN | TCN | difference | p |
|---|---|---|---|---|
| F1 | 0.8974 [0.8748, 0.9172] | 0.8961 [0.8765, 0.9130] | +0.0013 [−0.0087, +0.0111] | 0.78 |
| AUROC | 0.9046 [0.8722, 0.9332] | 0.8876 [0.8538, 0.9198] | +0.0169 [−0.0004, +0.0348] | 0.057 |

Nothing below roughly ±0.03 AUROC is measurable on this benchmark. For scale, the +0.027 AUROC
that log-compressing the frontend delivers in a controlled A/B is *below the resolution of the
test set v1 reported on*.

---

## Export

ONNX matched PyTorch to four decimal places on every metric, with streaming parity within
4.2e-4 against a 1e-3 tolerance. The export mechanism is sound. The exported weights are from
the mis-selected checkpoint (E4).

---

## What v1 established

1. The pipeline works end to end: manifests → training → evaluation → ONNX, with a verified
   streaming contract and a real parity gate.
2. ~40 h, of which ~19 h is conversational, is enough to beat a constant predictor by ~4 points
   of F1 and to reach 0.90 AUROC — and not enough to beat it on conversational speech at all.
3. Six defects that invalidated large parts of the measurement (see `ERRATA.md`), all of which
   are cheap to fix and none of which were visible in the metrics being reported. That is the
   real lesson, and it is why v2 builds the measurement foundation before it trains anything.
