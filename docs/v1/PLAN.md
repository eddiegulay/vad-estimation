# VAD Models — Path to Production-Grade Performance

## Context

The 9-phase pipeline built in the previous session (label synthesis → data processing → augmentation → loading → CRNN model → training → evaluation → ONNX export) is real, tested (141 tests), and verified end-to-end on real data. But the only checkpoint that exists is a **20-optimizer-step smoke test** (0.03% of one configured full run) — reported numbers (F1 0.890, AUROC 0.867 on TEN) are pipeline-correctness proof, not a trained model's real performance.

The user now wants to target actual usable performance ("95%") and push the project to `git@github.com:eddiegulay/vad-estimation.git`. **The git setup is already done**: repo initialized, `.claude/` added to `.gitignore`, initial commit made (no co-author trailer, author = the logged-in `gh` account's git identity, Eddie Gulay), pushed to `origin/master`.

Research into what actually caps performance (grounded in this session's code audit + the vault's production-VAD teardown notes 07/08/09) surfaced concrete, addressable gaps — not just "train longer":

1. **No LR schedule / warmup** — `configs/train/default.yaml` declares `warmup_steps: 500` but neither `Trainer` nor `scripts/train.py` reference it; flat `lr=1e-3` AdamW for 50 epochs.
2. **No gradient clipping.**
3. **No best-checkpoint tracking** — config declares `keep_best_metric: val_f1`, but `train.py` only computes `val_loss` and only ever writes `last.pt`.
4. **No post-processing hangover/hysteresis** — the *original* architecture research explicitly called for "external (non-learned) hysteresis/hangover" (matching how every real production VAD ships: Silero wraps its classifier in a dual-threshold + min-speech/min-silence state machine; TEN VAD applies a 10-frame/160ms score-smoothing filter). This was never built. `evaluate.py` thresholds raw per-frame probabilities at a flat 0.5 with no smoothing at all — this is the single cheapest, highest-leverage fix available and needs **zero retraining**.
5. **Threshold never calibrated** — flat 0.5 cutoff, no sweep against val data for the actual F1-maximizing operating point.
6. **`max_train_duration_s: 10.0` truncates AMI's 20s windows** — the model never sees the second half of any AMI training window, exactly where longer pauses live.
7. **LibriSpeech label noise** — a whole utterance is labeled 100% speech with no internal-silence detection, teaching the model that natural pauses/breaths inside an utterance are "speech."
8. **Single 30-file TEN test set, single checkpoint** — not statistically robust enough to trust a single F1 number, let alone claim "95%."
9. Only one architecture exists — can't tell whether CRNN's ~210K-param, DFT-frontend-plus-1-GRU shape is the accuracy ceiling for this data budget, or whether a different inductive bias does better. Research note 08's distilled recipe explicitly says "going bigger does not help" (100-400K params is the sweet spot) — so the lever is architecture *shape*, not size.

**Reality check to carry forward, not hide**: note 09/07 confirm the production VADs that hit 95%+ (Silero AUC 0.97, TEN VAD F1 95.2 on FLEURS-VAD-102, MarbleNet AUROC 95-97) are trained on 1,000-5,000+ hours; this kit is ~40h (~19h real conversational). Every fix below is a real, defensible lever — but "95%" should be tracked as a directional target we push hard toward and report honestly against, not silently redefined as already met. Some failure modes (babble/overlapping speech, music-with-vocals) are noted in the research as **not fixable by any binary classifier** at all, ours included.

**User confirmed** (via question): build and train the second architecture (TCN/MarbleNet-style) in this same pass, not deferred.

**Standing instructions carried forward from the previous session**: no pausing mid-way for confirmation between phases — work continuously, using `ROADMAP.md` as the running progress record (updated after each phase, "you are here" style, same as before). Git commits: create them at meaningful milestones, no `Co-Authored-By`/attribution trailers, author is the logged-in `gh` account (Eddie Gulay) — already configured correctly in git config. Push to `origin/master` after milestone commits (already-authorized remote).

---

## 1. Training loop hardening (`src/vad/engine/trainer.py`, `scripts/train.py`)

- **LR schedule**: linear warmup over `warmup_steps` (500, already in config) → cosine decay to ~1e-5 over the remaining scheduled steps. Implement with `torch.optim.lr_scheduler.LambdaLR` (simple closed-form warmup+cosine lambda, no new dependency), stepped once per optimizer step (not per epoch).
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` before `optimizer.step()`. Add `grad_clip_norm: 5.0` to `configs/train/default.yaml`.
- **Raise `max_train_duration_s`** from 10.0 → 20.0s in `configs/train/default.yaml` so AMI's full 20s windows are seen untruncated during training (only the rarer 40-60s LibriSpeech-concat outliers still get cropped). Throughput scales roughly linearly with crop length (measured 0.163s/step at 10s) — expect ~0.3s/step, full 50-epoch run (~7,300 steps) still well under an hour.
- **Per-epoch val F1/AUROC**, not just val loss**: after each epoch's val-loss loop, run a lightweight full-sequence forward pass over the val set (reuse `run_evaluation`'s core logic from `src/vad/eval/evaluate.py`, factored so `train.py` can call it without needing plotting) to get real `val_f1`/`val_auroc`.
- **`best.pt` tracking**: alongside the existing `last.pt` save, track the highest `val_f1` seen and save `best.pt` whenever it improves — matches the config's already-declared (but unimplemented) `keep_best_metric: val_f1`.
- **TensorBoard scalars** (dependency already installed, currently unused): train_loss, val_loss, val_f1, val_auroc, learning_rate, written per epoch under `checkpoints/<run_name>/tb/`.

## 2. Post-processing: hysteresis + hangover (new module)

- New `src/vad/postprocess/hysteresis.py`: a pure-numpy state machine operating on a probability sequence at the model's native 32ms hop —
  - dual threshold (`theta_on`, `theta_off`, default 0.5/0.35, matching Silero's design),
  - `min_speech_frames` (onset debounce, default ~250ms ≈ 8 frames),
  - `min_silence_frames` (hangover, default ~200ms ≈ 6 frames — inside Silero's 100ms and voice-agent-typical 300-800ms range).
  - All three configurable via a new `configs/eval/default.yaml` `postprocess:` block.
- Deliberately **not** baked into the ONNX graph — matches both the original architecture research and note 08's explicit "ship the state machine separately" guidance; it's a downstream-consumer concern, same as Silero's own reference implementation.
- Wire into `src/vad/eval/evaluate.py::run_evaluation`: compute both **raw** (existing flat-0.5) and **smoothed** (hysteresis-applied) frame + event metrics, report both so the lift from post-processing alone is visible.
- **Threshold calibration**: add a small val-set sweep (reuse `frame_precision_recall_f1` over a grid of thresholds, e.g. 0.1-0.9 step 0.05) to pick the F1-maximizing `theta_on` per architecture; record the chosen value in the eval report and `ROADMAP.md` instead of hardcoding 0.5.

## 3. Label & data quality (`src/vad/labels/synthetic.py`, `src/vad/data/manifest.py`)

- **LibriSpeech internal-silence trimming**: before building `label_intervals` for a concat example, run a cheap RMS-energy-threshold pass (no new dependency — pure numpy on the already-loaded waveform) over each source utterance to detect leading/trailing/internal near-silence and mark those spans `label=0` instead of blanket `1`. Conservative threshold (e.g. relative to per-utterance RMS, minimum gap duration ~150ms to avoid chopping stop-consonant closures) — the goal is fixing obviously-wrong silence-as-speech labeling, not full re-segmentation.
- Verify ESC-50's human-non-speech-vocalization classes (laughter, coughing, breathing, sneezing, crying — the note 08/MarbleNet-flagged highest-value hard-negative category) are actually included in the training folds (1-4) currently configured, not just present in the corpus.

## 4. Full training runs (both architectures)

- Run `scripts/train.py` full 50-epoch schedule (no `--subset`) for `crnn_v1`, in background, monitored via the new TensorBoard scalars / epoch print lines.
- Build the second architecture through the *unchanged* pipeline (registry pattern already supports this — only new files needed):
  - `src/vad/models/tcn.py`: MarbleNet-style causal, depthwise-separable dilated Conv1d stack (no GRU) — reuses the same `DFTFrontend`, implements the same `forward(chunk, state) -> (prob, state')` / `forward_full(waveform)` streaming contract as `CRNN`, with a fixed dilation-derived receptive-field context buffer (analogous to `CRNN.CONTEXT_SAMPLES`) instead of a GRU hidden state. Target 100-400K params per the research recipe.
  - `configs/model/tcn_v1.yaml` following the same schema as `crnn_v1.yaml`.
  - Same Phase-5-style gate before training: streaming-vs-`forward_full` equivalence test, param-count band check.
  - Full 50-epoch run through the same `scripts/train.py` (already architecture-agnostic via `build_model`).

## 5. Evaluation rigor

- Re-run `scripts/evaluate.py` for both architectures' best checkpoints against `test_ten.jsonl`, reporting raw + hysteresis-smoothed metrics side by side.
- Add a second held-out slice for evaluation robustness: use the val split's AMI windows (real conversational audio, not synthetic) as a second scoring set distinct from TEN, so "performance" isn't a single 30-file number.
- Re-export ONNX from each architecture's best checkpoint; re-run `scripts/verify_export.py` parity check.
- Update `ROADMAP.md`'s Model/Run Comparison Table with real numbers for both architectures (raw + smoothed), the calibrated threshold used, and RTF.

## 6. CI + repo hygiene

- Add `.github/workflows/test.yml`: run `pytest -m "not slow"` (fast tests only, ~seconds) on push/PR, now that there's a real shared remote to protect with a regression gate.
- Commit at each meaningful milestone (training-loop hardening, post-processing module, label-quality fix, each architecture's trained checkpoint metadata — **not the checkpoint binaries themselves**, `checkpoints/` stays gitignored) and push to `origin/master`. No attribution trailers.

---

## Verification

- Existing test suite must stay green throughout (`pytest`, 141+ tests) — new modules (`postprocess/hysteresis.py`, `models/tcn.py`) get their own unit tests following the existing per-module test-file pattern (`tests/postprocess/test_hysteresis.py`, `tests/models/test_tcn.py`), including the same overfit-gate and streaming-equivalence gates the CRNN went through.
- Training runs are judged by: no NaN/inf, monotonic-ish loss trend, `best.pt` actually differing from `last.pt` (proves selection is doing something), and a real F1/AUROC lift over the 20-step smoke baseline (0.890/0.867) — if a full run doesn't clear the smoke baseline by a wide margin, that itself is a signal to investigate before declaring success.
- Post-processing lift is verified by comparing raw-vs-smoothed metrics on the same checkpoint — expect event-level (onset boundary) metrics to improve the most, since that's exactly what hangover/hysteresis targets.
- `ROADMAP.md` gets a new phase block (Phase 9 — Performance Hardening, or similar) with the same checklist/decisions-log/session-log structure as before, updated continuously as work completes.
