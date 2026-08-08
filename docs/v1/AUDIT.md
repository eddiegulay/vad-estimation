# vad-models — v1 audit findings (verified)

Repo: /Users/eddiegulay/Documents/Work/Misc/vad-models
These findings come from a 5-agent audit of the completed v1 run. Every number below was
measured, not estimated. Treat them as established facts; do not re-derive them.

## v1 as it stands

Two models trained 50 epochs each on a ~40h local kit (LibriSpeech, AMI, ESC-50 noise,
Aachen AIR RIRs; TEN VAD as benchmark; FLEURS Swahili as cross-lingual sanity).

- CRNN (Silero shape): 210,561 params. TEN F1 0.8974, AUROC 0.9046.
- TCN (MarbleNet shape): 179,585 params. TEN F1 0.8961, AUROC 0.8876.
- Trivial "always speech" predictor scores F1 0.8586 on TEN. So the lift is +3.9 points.
- 16 kHz, 512-sample (32 ms) chunks, one decision per chunk, zero lookahead.
- Fixed non-trainable DFT frontend (256-pt, hop 128, 129 linear magnitude bins) inside the model.
- Shared 4-layer plain Conv1d+ReLU encoder 129->128->64->64->128, strides 1,2,2,1.
  NO normalization, NO dropout, NO residuals anywhere in either model.
- Loss: hand-written masked class-weighted BCE on probabilities (sigmoid is inside the model).
- Manifests: train 9318 examples (3000 synthetic LibriSpeech concats + 6318 AMI 20s windows),
  83.8% speech. Val 563, 87.2% speech. Test TEN 30 files / 261.9 s / 8184 frames, 75.2% speech.

## BLOCKING BUGS (all verified, ordered by severity)

### B1. MPS nn.Conv1d silently corrupts rows >= 65536 (torch 2.13.0)
Independently reproduced twice. rows=65536 -> maxdiff 3e-6; rows=65537 -> row 65536 wrong,
maxdiff 3.9; rows=100000 -> 34,464 bad rows. No error raised.
The model reshapes to [batch*num_chunks, 1, 640] before the conv stack. Val batches reach
162,368 rows, so up to 60% of every val batch was scored from corrupted activations.
Training is under the limit (625-chunk crop * 64 = 40,000) so GRADIENTS ARE FINE — only the
metrics are wrong. Margin is thin: batch_size >= 105 would corrupt training too.
Consequences:
 - The 0.7325 vs 0.9212 val F1 discrepancy is fully explained. 0.9212 is the truth.
 - THERE IS NO OVERFITTING. Re-scored safely, true val loss FALLS from ep28->ep49 and AUROC
   RISES, for both models. The rising val-loss curve is an artifact.
 - best.pt is WORSE than last.pt on every threshold-free metric, for both architectures, and
   train.sh evaluates and exports best.pt. The shipped ONNX models are mis-selected.
 - ROADMAP's "TCN fits val better (0.79 vs 0.73)" compares two corruption patterns. True F1
   is 0.9211 (CRNN) vs 0.9226 (TCN) — a wash.

### B2. Batch-level crop discards ~72% of training supervision
Trainer._crop_to_max_chunks draws ONE random 625-chunk window and applies it to the whole
batch, indexed against the batch's PADDED length (set by the longest example, up to 94 s).
Examples shorter than the crop start land entirely in their own zero padding.
Measured over the real manifest: 27.6% of labelled frames survive per epoch; 44.9% of
examples contribute zero gradient; 32.7% of forward-pass cells are supervised.
"50 epochs" was ~14 epochs of data at full compute cost.

### B3. set_epoch never reaches the dataloader workers
persistent_workers=True + num_workers=4 (spawn start method) means workers hold pickled
dataset copies with epoch frozen at 0 forever. Reproduced empirically.
The run was 50 passes over ONE static augmentation realisation: 9,318 (speech,noise,SNR,RIR,gain)
pairings instead of the intended ~466,000. Frozen: noise clip choice, SNR, RIR presence and
identity, noise crop offset, gain, gap silence/noise coin flip.
This matters doubly because neither model has any other regularizer.
Recommended fix: a custom Sampler yielding (epoch, index) tuples; the sampler lives in the
main process and IS re-iterated each epoch even under persistent_workers. Rejected alternatives:
persistent_workers=False (hides the flaw, costs spawn cost), worker_init_fn (runs at worker
construction, too early), shared mp.Value (fragile), IterableDataset (loses shuffle and len()).

### B4. Shipped hysteresis is a statistically significant regression vs no post-processing
Calibration sets theta_on from a val sweep (0.15 CRNN / 0.20 TCN) but theta_off stays at the
config default 0.35, so theta_off > theta_on and the Schmitt trigger inverts: frames in
[0.15,0.35) satisfy enter AND exit simultaneously, degenerating into a fixed 8-on/6-off
oscillator that carries zero information about the probability.
Paired file-level bootstrap, shipped minus raw: onset F1 -0.1209 [-0.1966,-0.0551] p<0.001;
frame F1 -0.0117 p=0.050; turn cost +27%; median endpoint dead-air 608 ms -> 1552 ms.
IMPORTANT: the inversion is only ~10% of the damage. Fixing it alone moves macro onset recall
0.370 -> 0.402. The dominant cause is that theta_off=0.35 is far below where the models put
their silence probability: only 28/96 of TEN's interior true pauses ever produce 6 consecutive
frames below 0.35 (at theta_off=0.65 it becomes 53/96). min_speech=256 ms deletes nothing
(TEN's shortest true speech segment is 256 ms) — it is pure onset delay with no benefit.

### B5. 9.7% of the CRNN's parameters are structurally dead
The frontend emits 4 frames/chunk; conv strides 1,2,2,1 with padding=kernel//2 collapse the
time axis 4->4->2->1->1. conv3 tap 0 (4,096 params) and conv4 taps 0 and 2 (16,384 params)
only ever multiply zero padding. Verified: gradient identically 0, and in the trained
checkpoint those taps sit at exactly the uniform-init RMS while live taps moved.
20,480 / 210,561 = 9.7% of the CRNN never trained. Same ConvStack is imported by the TCN
(20,480 / 179,585 = 11.4%). Additionally 19/128 encoder output channels are permanently zero
after relu4 on the trained CRNN, so the GRU's effective input width is 109, not 128.

### B6. Zero-gradient clamp in the loss
masked_bce_loss clamps probs to [1e-7, 1-1e-7] and takes logs. The clamp has zero gradient
outside its range. fp32 sigmoid(z)==1.0 exactly for z >= 16.64; on the trained TCN 3.63% of
TEN frames hit p==1.0 exactly (max logit 51.3). A confidently-WRONG frame then produces
exactly zero gradient instead of the largest one. Measured AUROC cost today: +0.00004
(negligible), so this is a latent training bug, not a current accuracy loss.

### B7. First optimizer step runs at lr=0.0
build_lr_scheduler's lr_lambda(0) returns exactly 0.0 and LambdaLR applies it at construction.

## MEASURED IMPROVEMENT OPPORTUNITIES

### I1. log compression of the DFT magnitude — the single best accuracy change
Controlled A/B, CRNN, 500 steps, batch 32, 512 cached real train / 192 real val examples,
identical seed and init across arms, only the frontend transform varying:
  D log + BatchNorm       0.9000  (+0.0295)
  B log(mag+1e-5)         0.8975  (+0.0270)   <- one line
  E D + dropout 0.15      0.8956  (+0.0251)
  G linear + BatchNorm    0.8883  (+0.0178)
  C log + per-bin standardization 0.8866 (+0.0161)
  A linear (current)      0.8705  (baseline)
  F log + per-chunk instance norm 0.7588 (-0.1117)  <- DO NOT DO THIS
Final val loss 0.4811 -> 0.4165 (-13%); reaches 0.889 by step 200 vs baseline 0.857.
Mechanism: measured input magnitudes span 4 decades (median 0.0118, max 184.8), 264x
per-example loudness variation, 61x per-bin tilt — and frequency is the CHANNEL dim of an
unnormalized Conv1d. Gradient into conv_blocks[0] is 15x weaker with linear input
(0.0034 vs 0.0503; 1.6% vs 23% of total grad norm).
Key negative results: do NOT add per-bin affine standardization (removes spectral tilt, which
is itself a speech cue); do NOT add instance norm (destroys the absolute-level cue VAD depends
on). BatchNorm on top of log adds only ~+0.003.

### I2. Timescale mismatch is the biggest data problem — bigger than class imbalance
Measured run-length statistics from the manifests:
                        transitions/min | non-speech run median | mass in runs <0.5s | >2s
  TEN (test)                       54.2 |                0.40 s |              39.0% | 0.0%
  train concat_synthetic           11.5 |                0.88 s |               7.4% | 55.5%
  train ami_window                 10.5 |                0.99 s |               3.7% | 67.1%
87% of TEN's non-speech mass is in runs shorter than 1 s; TEN contains ZERO non-speech runs
longer than 2 s. Training negatives are long, easy and low-energy. The model is taught to be
sluggish and then scored on agility.
Fix: retarget the synthetic gap distribution from log-uniform [0.2,3.0] + 8% tail into [3.0,8.0]
to roughly [0.15,1.2] + ~5% into [1.5,4.0]; raise utterances_per_example above [2,6] so each
example carries more boundaries.

### I3. trim_internal_silence is systematically too conservative, in the worst possible place
Defaults rel_threshold_db=-35.0, min_run_frames=5. At a 32 ms hop, min_run_frames=5 makes any
pause shorter than 160 ms STRUCTURALLY UNLABELLABLE — exactly TEN's dominant negative class.
Measured over 120 real train examples:
  occupancy pre-trim  0.8624 -> post-trim 0.8213  (4.75% of speech frames flipped)
  sensitivity: (-35,5)=0.8106  (-30,3)=0.7772  (-25,2)=0.7378  (-20,2)=0.6978
Forced-aligned read speech sits near 0.75-0.80. So ~5-10 points of frames are mislabelled as
speech, concentrated on the test set's dominant event class.
RISK: an RMS gate below -25 dB clips unvoiced fricatives (/s/,/f/,/th/) and stop closures.
Do not push past -25 dB with a bare RMS gate — use a teacher VAD instead (see I4).

### I4. Teacher-VAD pseudo-labelling is available and cheap
Verified: network is up, pypi and models.silero.ai both return 200, onnxruntime 1.28.0 is
already installed, the silero-vad wheel ships its ONNX weights inside the package (no torch
hub, no torchaudio — torchaudio is NOT installed and was deliberately dropped).
Train pool is only 5,567 utterances / 10.5 h; at RTF ~0.01 this is minutes of compute.
CRITICAL: use raw frame probabilities at a fixed threshold, NOT the teacher's
get_speech_timestamps() API — its built-in min_silence_duration_ms hysteresis would re-erase
exactly the short pauses you are trying to recover.
Forced alignment is the alternative (LibriSpeech transcripts ARE present: 97/91/87/90
*.trans.txt files across the four splits) but costs a torchaudio install + ~1 GB model
download + a gap-classification heuristic, and answers a proxy question. Teacher VAD is better.
Tradeoff to flag: distillation caps you near teacher quality and compromises the "from
scratch" framing of the labels (not of the model).

### I5. 20.7% of AMI train windows contain zero negative frames
Occupancy histogram of the 6,318 AMI train windows:
  [0.00,0.20) 361 (5.7%) | [0.20,0.40) 206 | [0.40,0.60) 393 | [0.60,0.80) 753
  [0.80,0.95) 2273 (36.0%) | [0.95,1.00) 1026 | ==1.00: 1306 (20.7%), 7.26 h of pure positives
Filtering to occupancy < 0.9 retains 2,862 windows / 15.9 h at 62.9% speech (from 81.5%).
configs/data/default.yaml already declares speech_occupancy_sanity_band: [0.20, 0.80] and
NOTHING READS IT.
Prefer reweighted sampling over hard filtering (low-occupancy windows correlate with meeting
starts/ends). Also drop train_overlap from 0.5 to ~0.25: 6,318 windows * 20 s = 35.1 h of
window-time from only ~17.5 h of unique audio.

### I6. The validation set cannot do its job — four independent defects
(a) Val AMI is ONE meeting. build_ami_examples calls discover_meetings on the SEGMENTS dir
    (171 meetings) while only 33 have audio. It seeds val with EN2001a then draws 3 extras
    from the remaining 170; P(all three lack audio) = 0.53, and that is what happened — they
    are silently dropped. Worse, EN2001a is the ONLY EN-series meeting with audio; all 32
    train meetings are ES/IS/TS. Val measures cross-site generalization from a sample of one.
    Fix: restrict discovery to meetings with audio, then hold out at SERIES level (ES2002a-d
    are the same four people in the same room — a naive per-meeting split leaks speakers).
(b) Val noise is not held out. esc50_holdout_fold: [5] is declared and NEVER READ; one
    aug_template built from folds 1-4 is passed to train, val, AMI and FLEURS alike.
(c) Val prevalence is wrong AND it sets the threshold. Val is 87.2% speech, TEN is 75.2%.
    The F1-maximizing threshold is strongly prevalence-dependent, so calibration transfers a
    systematically too-permissive threshold onto TEN.
(d) FLEURS sanity set is 93.1% speech and is BUILT BUT NEVER EVALUATED — no report exists for
    it in any checkpoint directory.

### I7. Class weighting shifts the operating point away from 0.5 — this is mechanical
Minimizing the class-weighted BCE makes the model output the REWEIGHTED posterior, whose 0.5
crossing corresponds to a true posterior of w_neg/(w_pos+w_neg) = 3.083/3.680 = 0.838.
Equivalently the calibrated threshold is w_pos/(w_pos+w_neg) = 0.162. Observed swept optima:
0.15, 0.15, 0.20, 0.10 — near-exact match. Evaluating at a fixed 0.5 costs ~2.5 points of F1.
Also measured: manifest-derived weights (0.5968/3.0830) vs true post-trim weights
(0.6007/2.9836) — a <=3% error. The weighting itself is fine; the fixed-0.5 selection is not.
Focal loss and label smoothing are NOT indicated (no genuine overfitting, mild imbalance).

### I8. RTF as reported is a measurement artifact; the real latency story is inverted
eval/evaluate.py starts the timer around the FIRST forward on each new sequence length, so
every MPSGraph shape-specialization compile is billed to a 261.9 s audio denominator (first
example costs 1713 ms CRNN / 538 ms TCN).
Re-measured offline forward_full: CRNN cold 0.0135 / warm 0.0031; TCN cold 0.0047 / warm
0.00037. The TCN is 2.8x faster cold, 8.3x faster warm — the opposite of what was reported.
MPS forward_full breakdown (8.7 s audio, batch 1): frontend 1.30 ms, conv stack 0.66 ms,
GRU 25.5 ms, TCN temporal stack 0.43 ms. The GRU is ~90% of the CRNN's offline cost and is
59x more expensive than the entire TCN temporal stack.
BUT per-chunk STREAMING (which is what ships), against a 32 ms budget:
  crnn cpu: p50 0.59 ms  p95 1.85 ms  p99 3.92 ms  streaming RTF 0.026
  crnn mps: p50 1.25 ms  p95 7.80 ms  p99 14.1 ms  streaming RTF 0.074
  tcn  cpu: p50 17.8 ms  p95 103 ms   p99 179 ms   streaming RTF 0.91   <- MISSES BUDGET
  tcn  mps: p50 2.21 ms  p95 11.1 ms  p99 20.1 ms  streaming RTF 0.119
Root cause isolated to nn.Conv1d(C,C,3,groups=C): ~2 us of fixed dispatch overhead PER GROUP.
Group sweep on the same tensor: g=1 -> 0.099 ms, g=8 -> 0.109, g=32 -> 0.157, g=64 -> 0.226,
g=128 -> 0.346. A DENSE groups=1 conv doing 128x the arithmetic is 3.5x FASTER (0.097 ms).
The identical math as (window[:,:,::d] * w).sum(-1) is 0.0057 ms — 61x faster.
Rewriting the temporal stack that way: streaming step 11.33 ms -> 0.084 ms, bit-identical.
Also: torch.set_num_threads(1) alone takes the TCN step 11.69 -> 1.53 ms (7.6x); threading is
pure overhead at these tensor sizes. And CRNN streaming is FASTER on CPU than MPS.
Also: replacing the conv-DFT frontend with unfold + torch.fft.rfft().abs() is 5.5x faster
(0.105 ms -> 0.0191 ms, 42% of the CRNN's single-thread step), max diff 1.9e-4.
So the CRNN-over-TCN decision is CORRECT but every reason given for it in the ROADMAP is wrong.

### I9. The benchmark cannot resolve any of these changes
Cluster bootstrap over the 30 TEN files (frames are massively autocorrelated; a naive iid
binomial CI over 8184 frames is 3.2x TOO NARROW):
  F1:    CRNN 0.8974 [0.8748,0.9172]  TCN 0.8961 [0.8765,0.9130]
         diff +0.0013 [-0.0087,+0.0111]  p=0.78   <- pure noise
  AUROC: CRNN 0.9046 [0.8722,0.9332]  TCN 0.8876 [0.8538,0.9198]
         diff +0.0169 [-0.0004,+0.0348]  p=0.057  <- marginal
Detecting the F1 gap at 80% power needs ~3,300 files (~8 h of TEN-like audio). The AUROC gap
needs ~67 files — an achievable 2x expansion.
Required benchmark size for a target precision (per-file SD: F1 0.059, AUROC 0.085):
  F1  +-0.02 -> 34 files | +-0.01 -> 136 | +-0.005 -> 543
  AUROC +-0.02 -> 71 files | +-0.01 -> 281
The +0.027 log-compression gain from I1 is BELOW the resolution of the current test set.
Split-half calibration inside TEN: chosen theta_on has a 5th-95th percentile spread of
[0.50,0.80]; held-out onset F1 averages 0.672 vs a held-out oracle of 0.744, i.e. ~7 points of
any 15-file calibration is overfitting. 30 files cannot pin a four-parameter operating point.

### I10. Per-slice evaluation reveals the models have not learned conversational speech
Val slices:
  concat_synthetic (n=300): prior 0.828, all-speech floor 0.9059, CRNN 0.9145, TCN 0.9294
  ami_window       (n=263): prior 0.875, all-speech floor 0.9334, CRNN 0.9322, TCN 0.9110
BOTH MODELS SCORE AT OR BELOW THE TRIVIAL FLOOR ON THE AMI SLICE. The model ranking REVERSES
between slices (TCN wins synthetic by 1.5 pts, CRNN wins AMI by 2.1 pts) while pooled val F1
(0.9212 vs 0.9227) shows nothing. kind is already in every manifest record — this is a groupby.
Note also that AMI is mostly SCRIPTED role-play (only EN2001a is spontaneous, and it is in val).

### I11. Missing metrics, ranked by what a turn-taker actually needs
1. OFFSET boundary metric. event_boundary_metrics filters to label==1 STARTS only. Endpointing
   IS offset detection. Measured on TEN: CRNN raw onset F1 0.5903 vs offset F1 0.3819; TCN
   0.5320 vs 0.3367. Offsets are ~20 points worse and this has never been visible.
2. Latency in MILLISECONDS as a distribution, not a within-tolerance match rate. Endpoint
   dead-air, CRNN, 114 observable turn-ends: raw p50 608 ms / p90 4592 ms / 37 of 114 turn-ends
   NEVER RELEASED before EOF. Shipped config: p50 1552 ms / p90 5331 ms / 59 of 114 never
   released. Completely invisible in F1 0.8974. Report p50/p90/p99 plus a miss count, never a
   mean (heavily right-skewed).
3. FAR/MISS pair. At 0.5 on TEN, CRNN FAR = 0.5005, MISS = 0.0518 — HALF of all true-silence
   frames are called speech. Balanced accuracy 0.7238, MCC 0.524, silence-class F1 0.6030
   (vs the speech-class 0.8974 that gets reported). TCN FAR 0.5404, MISS 0.0437, silence-F1
   0.5773. Also AP_silence 0.749 vs AP_speech 0.966.
4. Per-chunk streaming latency p50/p95/p99 (see I8). The ONNX reports contain no RTF field at all.
5. Confidence intervals (see I9).
6. Per-corpus / per-condition / per-gap-duration slices (see I10). TEN's 96 interior pauses:
   p10 192 ms, p50 480 ms, p90 944 ms.
Two latent metric bugs: evaluate.py drops NaN precision values before macro-averaging, so a
file where the model predicts NO onsets is excluded from the precision mean while still
counting 0 in recall (systematic upward bias; 0/30 files affected today but will fire at higher
thresholds). And sweep_thresholds selects under probs > threshold while the state machine
applies p >= theta_on.

### I12. Calibration objective should be a turn-taking cost, not frame F1
Full joint grid sweep (theta_on 0.10-0.90 x theta_off <= theta_on x min_speech {32,64,128,192,
256} ms x min_silence {64,128,192,256,384} ms = 3825 configs) on val, transferred to TEN:
  objective optimized on VAL | winner              | TEN frame F1 | TEN onset F1 | TEN cost/turn
  -- (raw p>0.5)             | --                  | 0.8974       | 0.5903       | 5099
  -- (shipped)               | 0.15/0.35/256/192   | 0.8857       | 0.4694       | 6491
  frame F1                   | 0.25/0.25/32/384    | 0.8732       | 0.2959       | 10269
  onset F1 @200ms            | 0.15/0.15/128/64    | 0.8731       | 0.3799       | 10332
  FSTTM turn cost            | 0.65/0.30/256/64    | 0.8778       | 0.5408       | 5678
EVERY val-optimized config except the cost one transfers WORSE than no post-processing at all.
Frame F1 on val selects thresholds near 0.25 because val is 84.5% speech — the sweep in the
shipped JSON is monotone-decreasing from its low end, the signature of a metric being pulled
toward the majority class.
TEN-oracle optima are theta_on 0.65-0.90; val optima are 0.15-0.25 — opposite ends of the grid.
Calibrating on val and reporting on TEN is invalid as currently set up.
Every good config found puts min_speech at the grid floor (32 ms): onset debounce buys nothing
on this data and directly costs barge-in latency.
The cost objective should use note 04's published FSTTM defaults (C_cutin=5000,
c_deadair=3000/s, 200 ms grab floor), fixed by Raux's budget method, NOT tuned to the VAD.
Under it, shipped costs 6491/turn vs 2585 for the TEN-oracle config — a 2.5x reduction
available from parameters alone.

### I13. Other wrong-but-smaller things
- target_duration_s: [8.0,20.0] is declared and NEVER READ. Concat examples actually run
  4.5-94.2 s, mean 31.6 s. Combined with the 20 s crop, ~1/3 of decoded audio is thrown away
  every epoch.
- gain_range_db is a constructor default (+-3 dB), unreachable from any YAML.
- RIR pool is skewed to the wrong rooms: 214 RIRs, median RT60 1.30 s, 62% above 1.0 s.
  Composition: stairway 78 (RT60 1.92 s), aula_carolina 22 (RT60 7.35 s), lecture 24,
  meeting 20, office 12, booth 12. Measured leak into gap frames: dry -37.2 dB, meeting
  -33.1 dB, stairway -26.6 dB, aula_carolina -15.3 dB (audible speech energy inside frames
  labelled 0). select_rir_pool already supports substring pools; build_manifests passes [].
- RIR direct-path delay is not compensated: median peak at 5.9 ms, p90 15.8 ms, max 90.2 ms —
  up to ~3 frames of systematic late-shift on ~50% of training examples, against unshifted
  labels, concentrated exactly where boundary metrics are measured.
- Noise is applied with probability 1.0 (no gate, unlike rir_prob). Mean SNR ~7.5 dB. THE
  MODEL HAS NEVER SEEN CLEAN SPEECH, while TEN and FLEURS are comparatively clean.
- Uniform SNR [-5,20] is the wrong shape: 20 dB is not clean (real close-mic is 30-50 dB), and
  the -5 dB floor puts 20% of examples in a regime where ESC-50's vocal classes (laughing,
  coughing, crying_baby, breathing, snoring) DOMINATE the mixture inside frames labelled 0.
- Loss averaging bias: eval_epoch averages per-batch losses rather than weighting by valid
  frame count.
- Label/prob tail misalignment: num_frames is round(len/512) but the model produces
  floor(len/512). 149 of 563 val items have one extra label frame, unmasked, paired with a
  probability computed from zero padding. ~0.03% of frames but genuine.
- FIVE config sections are read by nobody: precision; loss.type/loss.class_weighting;
  checkpoint.save_every_epochs/keep_best_metric; env.pytorch_enable_mps_fallback;
  dataloader.persistent_workers/prefetch_factor/pin_memory. They currently agree with the
  hardcoded values, which makes them WORSE than absent — editing them silently does nothing.
- No per-epoch checkpoint history (last.pt is overwritten), so the true val curve cannot be
  reconstructed without retraining.
- No resume: best_val_f1 is not persisted, there is no --resume flag. A killed run restarts.
- NOTHING is seeded: zero hits for manual_seed / np.random.seed / use_deterministic_algorithms
  / generator= across src/ and scripts/. The configured seed reaches only the augmentation RNG.
  Unseeded: model init, DataLoader shuffle order, and the crop offset.
- assert loss == loss is the only NaN guard and vanishes under python -O.
- ESC-50 is CC BY-NC (non-commercial); Aachen AIR licence is unstated. Both research-only.
- No music corpus is available (ESC-50 has no music class). AVA-Speech is labels-only and
  needs bulk YouTube download — multi-day, deprioritize.
- Babble CAN be synthesized from LibriSpeech but must be labelled SPEECH, not non-speech —
  AMI and TEN both label overlapping speech as speech.

## SUGGESTED ORDER OF WORK (from the audit)
1. Fix B1 (MPS conv), re-score both v1 checkpoints, re-select from last.pt.
2. Fix B2 (per-example crop or length-bucketed sampler) — ~3.5x more supervision at same cost.
3. EXPAND THE BENCHMARK — nothing below +-0.03 AUROC is measurable until then.
4. I1 (log compression).
5. I2 + I3 + I5 (gap distribution, trim, AMI reweighting) — the data timescale fix.
6. B3 (sampler-carried epoch).
7. I12 (retune the operating point against a turn cost, not frame F1).
