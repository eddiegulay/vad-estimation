# v2 Correctness Gates

Every gate here must **exist and fail against tag `v1.0`** before Phase 4 begins. They turn green
only as the corresponding fixes land. A gate that has never failed has not been shown to work.

`scripts/preflight.py` runs all of them; `train.sh` exits non-zero and refuses to launch if any
fails. Target preflight runtime: ≤ 90 s.

---

| ID | Gate | Asserts | Fails at (measured v1 value) |
|---|---|---|---|
| **G1** | MPS conv parity | On the real conv stack at `[rows, 1, 640]` for rows ∈ {65535, 65536, 65537, 100000, 162368}: `max abs(cpu − mps) < 1e-4`. Plus the chunked path is bit-identical to CPU, and the trainer refuses `batch_size × max_chunks > 65535` | max diff **3.9** at 65537 rows; 34,464 corrupted rows at 100,000 |
| **G2** | Supervision density | Over one instrumented epoch on the real manifest: `frames_seen / frames_total ≥ 0.95`; `examples_with_zero_gradient == 0`; `supervised_cells / forward_cells ≥ 0.80` | **0.276 / 44.9% / 0.327** |
| **G3** | Epoch propagation | Real DataLoader, 2 workers, persistent, spawn, two epochs. Per-example augmentation fingerprint differs between epochs for ≥ 90% of ids, and the epoch observed inside the worker equals the outer epoch for 100% of samples | **0% differ**; worker epoch frozen at 0 |
| **G4** | Hysteresis invariants | `apply_hysteresis` raises on `theta_off > theta_on`; the config loader validates the post-processing block at load time; lowering `theta_on` never removes a predicted speech frame; the threshold sweep and the state machine use the same comparison operator | The shipped 0.15/0.35 config inverts the Schmitt trigger into a fixed 8-on/6-off oscillator |
| **G5** | Dead parameters | After one backward on a real batch, every named parameter has non-zero gradient — **and for every convolution, every kernel tap index individually receives non-zero gradient**. Plus dead output channels ≤ 2% over a 200-example probe | 20,480 params with identically-zero gradient (9.7% CRNN / 11.4% TCN); 19/128 channels dead (14.8%) |
| **G6** | Streaming budget | ≥ 2,000 chunks after ≥ 200 warm-up discarded. CPU single-threaded: p50 ≤ 4 ms, p95 ≤ 10 ms, p99 ≤ 16 ms, max ≤ 32 ms. Also measured on MPS and through ONNX. No architecture enters a run without passing on CPU | TCN CPU p50 17.8 / p95 103 / p99 179 ms — misses budget by 5.6× at p99 |
| **G7** | Metric correctness | A file with zero predicted onsets contributes precision 0 rather than being dropped; label and probability lengths match for 100% of eval items; the turn cost reproduces the audit's v1 values within ±2%; always-speech scores its known floor exactly; the cluster bootstrap achieves 93–97% empirical coverage where a naive iid bootstrap demonstrably does not | Silent upward precision bias; 149 of 563 val items off by one, unmasked, paired with probabilities from zero padding |
| **G8** | Determinism | Two evaluation runs at the same seed produce byte-identical output except timestamps. Two trainer constructions at the same seed produce identical initial parameters and identical first-epoch batch order | Nothing is seeded: model init, shuffle order and crop offset all unseeded |
| **G9** | Config liveness | An access-tracking config wrapper records every key read during a dry run; assert no unread keys | ≥ 12 dead keys. Dead-but-agreeing config is worse than absent — editing it silently does nothing |
| **G10** | Leakage | No audio path, LibriSpeech speaker, AMI meeting **or series**, held-out ESC-50 clip, or RIR file appears in both a training manifest and any calibration or test manifest | One augmentation template built from folds 1–4 was passed to train, val and sanity alike; the fold holdout key was never read |

---

## Two that are not gates but belong in the same commit

- Replace `assert loss == loss` with a real finite check — the assert vanishes under `python -O`,
  and it is currently the only NaN guard in the training loop.
- Replace the clamp-then-log loss with `binary_cross_entropy_with_logits`, so the zero-gradient-
  on-confidently-wrong-frames failure cannot recur.
