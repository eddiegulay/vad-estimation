# v2 Correctness Gates

Every gate must **exist and demonstrably fail against tag `v1.0`** before WP10 (first training)
begins. They turn green only as the fresh build lands. A gate that has never failed has not been
shown to work.

Mechanics: gates live in `tests/gates/test_g*.py`. Behavioral gates (G1, G2, G3, G5, G8, G9,
G11, G12, G16, G17) are run once against a `v1.0` worktree via a thin driver; artifact gates
(G4, G6, G7, G10, G13, G15) run against `runs/v1/` bytes forever as regression fixtures. `scripts/preflight.py`
runs the ≤90 s subset; `train.sh` exits non-zero on any red. Every measured failing value is
recorded in `GATE_BASELINE.md` next to its passing value when it turns green — **a gate with no
baseline row cannot be marked green.**

**Added 2026-08-10 after the data audit:** G15 (asset provenance), G16 (train/test acoustic
match) and G17 (no unphysical silence) come from [`DATA-QC.md`](./DATA-QC.md). G16 is unusual
among the gates in that it does not encode a v1 defect — it encodes a **v2 defect**, caught in
the plan before any code was written, and it is the only gate whose failing baseline is the
roadmap's own first draft.

Two corrections to this file's first draft, found before execution: G1's "bit-identical to CPU"
demanded the impossible (cross-backend bit-identity does not exist — v1's own safe-size
measurement was 3e-6, not 0), and G2's frame-coverage arithmetic was unsatisfiable under the 8 s
crop design it was meant to protect. Both are restated correctly below. G10's first draft was
untestable against v1 — v1's manifests record neither noise nor RIR membership (verified), so a
manifest-scan gate passes vacuously; v2 manifests must enumerate membership, and the v1 failure
is demonstrated via a reconstructed aug-template membership file.

---

| ID | Gate | Asserts | Fails at (measured v1 value) |
|---|---|---|---|
| **G1** | MPS conv parity | For rows ∈ {65535, 65536, 65537, 100000, 162368} on the real conv stack at `[rows, 1, 640]`: `max abs(cpu − mps) < 1e-4`. The chunked path is **bit-identical to the unchunked MPS path at safe sizes**, and within 1e-4 of CPU. The trainer refuses `batch_size × max_chunks > 65535`. | max diff **3.9** at 65537 rows; 34,464 corrupted rows at 100,000. v1 val batches reached 162,368 rows. |
| **G2** | Supervision density | Per instrumented epoch under the fixed-crop design: **100% of examples contribute ≥ 1 supervised frame**; `supervised_cells / forward_cells ≥ 0.95` (warm-up mask, ≤ 3.2%, is excluded from the denominator); **cumulative labelled-frame coverage over any 4 consecutive epochs ≥ 0.95** (crop offsets are per-item RNG draws, so coverage accrues across epochs). | v1: 44.9% of examples contributed zero gradient; 32.7% of forward cells supervised; 27.6% of frames seen per epoch. |
| **G3** | Epoch propagation | Real DataLoader, 2 workers, persistent, spawn, two epochs: per-example augmentation fingerprints differ between epochs for ≥ 90% of ids; the epoch observed inside the worker equals the outer epoch for 100% of samples. | 0% differ; worker epoch frozen at 0 for the entire 50-epoch run. |
| **G4** | Hysteresis invariants | `apply_hysteresis` raises on `theta_off > theta_on`; the config loader validates the block at load (v1's shipped 0.15/0.35 must raise); lowering `theta_on` never removes a predicted speech frame; the sweep and the state machine use the same comparison operator. | The shipped config inverts the Schmitt trigger into a fixed 8-on/6-off oscillator carrying no probability information. |
| **G5** | Dead parameters | After one backward on a real batch: every parameter has non-zero gradient, **per kernel tap**; dead output channels ≤ 2% over a 200-example probe. Runs on random init — architecture correctness is not contingent on weights. | 20,480 params with identically-zero gradient (9.7% CRNN / 11.4% TCN); 19/128 channels dead. |
| **G6** | Streaming budget | ≥ 2,000 chunks after ≥ 200 warm-up discarded, CPU single-threaded, PyTorch **and** ONNX paths: p50 ≤ 4 ms, p95 ≤ 10 ms, p99 ≤ 16 ms, max ≤ 32 ms. No architecture enters a training run without passing on CPU. | TCN CPU p50 17.8 / p95 103 / p99 179 ms — 5.6× over budget at p99. |
| **G7** | Metric correctness | Zero-predicted-onset files contribute precision 0 (never dropped) and increment `files_with_no_pred`; label/probability lengths match for 100% of eval items; turn cost reproduces 5099 / 6491 / 2585 within ±2% on the pinned v1 probabilities; `ConstantSource` reproduces the always-speech floor F1 0.8586 exactly; the cluster bootstrap achieves 93–97% empirical coverage on a synthetic autocorrelated fixture where naive iid demonstrably under-covers; **the report schema rejects any bare-float metric and any `rtf` key**. | Silent upward precision bias; 149/563 val items off by one; iid CIs 3.2× too narrow. |
| **G8** | Determinism | Two eval runs at the same seed: byte-identical output minus timestamps. Two trainer constructions at the same seed: identical initial parameters and identical first-epoch batch order (CPU claim; MPS lacks deterministic kernels). | Nothing seeded: model init, shuffle order, crop offset. |
| **G9** | Config liveness | Access-tracking wrapper records every key read during a dry run; assert zero unread keys. | 13 dead keys in v1 (`precision`, `loss.type`, `loss.class_weighting`, `checkpoint.save_every_epochs`, `checkpoint.keep_best_metric`, `env.pytorch_enable_mps_fallback`, three `dataloader.*`, `target_duration_s`, `speech_occupancy_sanity_band`, `esc50_holdout_fold`, `gain_range_db`). |
| **G10** | Leakage | v2 manifests **enumerate noise-clip and RIR membership per record** (or a committed per-split template file). No audio path, LibriSpeech speaker (including the named 48-speaker benchmark reservation), AMI meeting **or series**, held-out ESC-50 clip, or RIR appears in both a train manifest and any calibration/test/benchmark manifest. | Demonstrated against v1 via a reconstructed aug-template membership file: one template from folds 1–4 served train, val, and sanity alike; the fold-5 holdout key was never read. |
| **G11** | Loss gradient | On the extreme counterexample (label 0, logit +40): the loss gradient w.r.t. the logit is finite and **non-zero** and matches `bce_with_logits` closed form; the NaN guard is `math.isfinite`, not a bare `assert` (which vanishes under `python -O`). | v1's clamp-then-log produced exactly zero gradient past sigmoid saturation; 3.63% of TCN test frames sat there. |
| **G12** | Schedule | `lr(step 0) > 0`; warmup is `(step+1)/warmup_steps`; the recorded LR trace contains no zero entries. | v1's first optimizer step ran at lr = 0.0 exactly. |
| **G13** | Export parity | ONNX streaming ≡ PyTorch streaming ≤ 1e-3 over ≥ 5 real files; eval metrics via the ONNX source match the PyTorch source to 4 decimals; frontend cross-parity (rfft vs conv-DFT): ≤ 1e-3 log-magnitude, ≤ 1e-3 probability, ≤ 1e-4 end-to-end AUROC on the 32-example startup set. Runs on random init. | Not a v1 failure (v1's export parity was 4.2e-4 — sound); the gate exists because the v2 frontend split makes parity *more* fragile, and the linear-domain 1.9e-4 becomes ~0.016 after the log. |
| **G14** | Memory budget | 300 real batches at configured dataloader settings: peak process-tree RSS under the v1 ceiling (~7.6 GB), soft runaway guard on the median. Carried from v1 — the sampler/dataset/crop path it guards is exactly what the fresh build replaces. | Not a v1 failure; a regression tripwire. |
| **G15** | Asset provenance | Every path referenced by any manifest appears in `asset_qc.json` with `status: ok`; no quarantined asset is reachable from any split; the QC report's own hash is recorded in `manifest_set_id`. Noise-pool records carry the measured `speech_frac` and `active_region`; RIR records carry measured RT60, DRR and direct delay. | v1 has no asset QC of any kind: 137 speech-bearing ESC-50 clips, 48 hard-clipped clips and 83 unusable RIRs were reachable from the train manifest (DATA-QC §F4, §F5, §F6). |
| **G16** | Train/test acoustic match | On ≥ 500 realised training examples (built by the actual dataset code, not the config): two-sample KS against TEN-30 on (a) speech/non-speech contrast in dB and (b) median non-speech level, **both p > 0.10**. Measured on what the sampler emits, so a policy that drifts from its YAML still fails. | v1 policy passes contrast (D=0.129, p=0.68) and marginally the level test (D=0.235, p=0.074). **The v2 policy as first specified fails both** (p=0.005 / p=1.1e-08) — the gate exists because the plan's own augmentation numbers were the thing that broke it. |
| **G17** | No unphysical silence | Zero emitted training examples contain an all-zero 512-sample frame; the gap generator always produces a floor; every AMI window overlapping a digital-dropout region is excluded by the manifest builder. | v1: gap audio coin-flips to exact zeros; 1.64% of AMI non-speech frames are exact zero, including a 149.9 s continuous dropout in ES2002d (DATA-QC §F2). |

---

## Phase-0 acceptance: the five pinned corrections, enumerated

The WP0 regression test asserts, from the pinned `.npz` probabilities, on CPU:

1. CRNN `last.pt` val AUROC 0.9489 > `best.pt` 0.9441 (± 1e-4), and the same ordering for TCN
   (0.9415 > 0.9323).
2. Val F1 @ 0.5 = 0.9211 (CRNN) / 0.9226 (TCN) ± 1e-4 — not the corrupted 0.7325 / 0.7944.
3. Val loss at epoch 49 < val loss at the "best" epoch, both architectures — no overfitting.
4. TEN F1 0.8974 / AUROC 0.9046 (CRNN) ± 1e-4 — the test-set numbers were never corrupted.
5. Shipped-vs-raw onset ΔF1 = −0.1209 with a 95% bootstrap CI excluding zero — the
   post-processing regression reproduces.

Any of the five failing to reproduce is itself a finding, and stops the cutover.
