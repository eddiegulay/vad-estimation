# Data QC — what the audio actually contains

A waveform-level audit of every asset the project uses, run 2026-08-10 before any v2 data work.
Scripts in [`../../scripts/qc/`](../../scripts/qc/); raw per-file output in `data_cache/qc/`
(gitignored — regenerate with the scripts, ~6 min total).

**Scope:** 16,957 audio files (43.5 h) + 214 impulse responses.
LibriSpeech 11,126 / 21.25 h · AMI 33 / 19.06 h · ESC-50 2,000 / 2.78 h ·
FLEURS 3,768 / 16.20 h (all splits on disk) · TEN 30 / 0.07 h.

**Integrity baseline — clean.** Zero decode errors, zero non-finite samples, zero empty files,
zero duplicate files by content hash across all corpora, and every file is 16 kHz mono as the
converter promises (FLEURS is float WAV, the rest PCM_16). Nothing below is a broken-file
problem. Everything below is a *content* problem that the existing pipeline cannot see.

| script | what it answers |
|---|---|
| `qc_scan.py` | per-file level, pathology, dynamics, bandwidth, hum, hashes |
| `qc_rir.py` | direct path, DRR, Schroeder decay times, tail contamination |
| `qc_label_energy.py` | do the labels agree with the audio |
| `qc_esc50_speech.py` | is the noise corpus speech-free (silero-vad 6.2.1, raw probs @ 0.5) |
| `qc_contrast.py` | does an augmented training example look like a test file |

---

## F1 — Labels claim speech over silence, measurably, without needing a teacher

The teacher-labelling plan (WP5) was justified on transition *timing*. Energy alone shows the
same defect on the *level* axis, and quantifies it from the audio:

| corpus | speech-labelled frames that are actually silent |
|---|---|
| LibriSpeech (v1 labels every utterance 100% speech) | 13.3% of frames below `ref − 35 dB`, duration-weighted; 25.3% below `ref − 25 dB` |
| — leading + trailing silence | 1.22 h = **5.8% of all LibriSpeech time**, labelled speech |
| — interior pauses | 1,470 utterances contain an interior silence > 0.5 s; 117 exceed 1.0 s (max 2.08 s) |
| AMI (union-of-channels annotation) | **22.4%** of speech-labelled frames below −60 dBFS; **39.5%** below −50 dBFS; 64.6% below −40 dBFS |

Good news on AMI: **no digital-silence frame is labelled speech** (`zero_in_speech = 0.0000`
across all 33 meetings), so the annotation is time-aligned. The problem is that it is
utterance-level, exactly as the design notes say — now with a number that does not depend on
trusting silero.

## F2 — AMI contains long channel dropouts, and they are digitally silent

| meeting | duration | exact-zero samples | longest continuous zero run |
|---|---|---|---|
| ES2002d | 43.7 min | 22.2% | **149.9 s** |
| ES2002b | 38.0 min | 5.4% | 36.5 s |
| EN2001a | 87.5 min | 4.1% | 12.3 s |
| ES2002c | 40.4 min | 13.6% | 3.7 s |

These are labelled non-speech, which is correct, but 1.64% of all AMI non-speech frames are
mathematically zero. A model can separate `x == 0` from speech with no acoustic reasoning at
all, and that skill transfers to nothing.

## F3 — The AMI series are two different acoustic domains

`nonspeech_below_60db`, per series: ES 0.68–0.97 · TS 0.79–0.99 · EN 0.84 · **IS 0.000**.
The IS1000/IS1001 meetings contain no near-silence at all — a continuous noise floor where the
others have near-digital silence. Series-level splitting (WP6) is right for speaker leakage but
must also be **stratified across this divide**, or the val/test series measures a different
recording setup than training saw.

## F4 — The noise corpus is not speech-free, and category names do not find the speech

Every ESC-50 clip scored with silero-vad raw frame probabilities at 0.5:

- **296 / 2,000 clips (14.8%)** contain at least one speech frame.
- 137 (6.9%) are > 2% speech; 79 (4.0%) are > 5%; 7 contain a continuous speech run ≥ 0.5 s.
- 70 clips reach p ≥ 0.9 on some frame.

v1 identified "vocal confusers" by ESC-50 category name (laughing, coughing, crying_baby,
sneezing, breathing). Of the 79 clips above 5% speech, **44 fall outside that list** — insects,
pig, drinking_sipping, snoring, cow, chainsaw, siren, toilet_flush, door_wood_creaks,
church_bells, train, hand_saw, dog, sheep. Category-name curation finds under half of the
contamination. Mixed into a silence-labelled frame, these clips teach that speech is silence.

## F5 — ESC-50 zero padding silently breaks the SNR mixer

`mix_at_snr` scales noise by its RMS over the whole segment. Many ESC-50 clips are 5 s of
mostly digital-zero padding around a short event:

- 323 clips are > 30% exact zeros; 335 have a zero run > 1 s (max 4.64 s of a 5 s clip).
- Consequence: **374 clips (18.7%) put audible noise more than 3 dB louder than requested**;
  161 (8.1%) exceed 6 dB; worst case **13.5 dB**.
- The converse also happens: a noise crop landing entirely inside the padding adds nothing, so
  the realised `noise_prob` is below the configured one.

## F6 — Impulse responses: the pool needs a measured criterion, not a room-name list

| room | n | median RT60 | median direct delay | tail contamination |
|---|---|---|---|---|
| aula_carolina | 22 | **6.62 s** | 9.2 ms | 2.6% of energy arrives > 1 s after the direct path, peak −30 dB |
| stairway | 78 | 1.00 s | 6.0 ms | clean |
| lecture | 24 | 0.88 s | **18.1 ms** | clean |
| office | 12 | 0.64 s | 6.1 ms | clean |
| meeting | 20 | 0.35 s | 6.0 ms | clean |
| booth | 12 | 0.30 s | 3.1 ms | clean |

Pool-wide: direct-path delay median 5.9 ms, **max 90.2 ms** (2.8 frames of label
misalignment — confirms the audit figure exactly). `stairway` alone is 36% of the pool.
Filtering on *measured* properties — RT60 ≤ 1.0 s, late-tail energy ≤ 0.5%, direct delay
≤ 20 ms — leaves **131 usable IRs**, against 86 from the roadmap's room-name list. The measured
criterion is the one to keep; the room names were a proxy for it.

## F7 — FLEURS has a 53 dB level spread and a split-dependent offset

Level range p01→p99 = −67.6 → −14.6 dBFS. 327 files (8.7%) are near-empty or peak below
−40 dBFS. Median RMS: train −33.0 dB, dev −43.0, **test −42.4** — the split v2 plans to re-cut
into the benchmark sits ~10 dB below the split it does not use. Any FLEURS-derived benchmark
must state its level distribution, or it measures gain sensitivity rather than VAD quality.

## F8 — Clipping, hum, bandwidth

- **Clipping:** ESC-50 519 clips affected, 48 above 1% of samples (worst 12.9%); FLEURS 328;
  AMI 8 meetings with runs up to 221 samples; **TEN itself has 5 clipped files**, so clipping is
  in-domain and must not be filtered out of the test set — only out of the noise pool.
- **Mains hum:** LibriSpeech 1,526 files (13.7%) show a 50/60 Hz peak > 10 dB above its
  neighbourhood, 299 above 20 dB. AMI shows none (max 4.2 dB). Hum lands in frontend bins 0–1,
  which the model sees directly.
- **Bandwidth:** 1,237 LibriSpeech files (11%) are effectively lowpassed below 7 kHz, 502 below
  6 kHz; FLEURS 760 / 484. Natural variation, but it correlates with source and therefore with
  speaker, which makes it a leakage vector for speaker-split benchmarks.

## F9 — The training corpora are 23 dB "cleaner" than the test set

Median per file:

| | LibriSpeech | AMI | FLEURS | **TEN** |
|---|---|---|---|---|
| dynamic range (frame p95 − p05) | 43.5 dB | 43.6 dB | 35.4 dB | **20.5 dB** |
| own noise floor (frame p05) | −62.5 dB | −77.3 dB | −66.3 dB | **−41.4 dB** |

Unaugmented, the training sources present a speech/silence contrast twice as large as the test
set's. Augmentation is therefore not a regulariser here — **it is the only mechanism aligning
the training and test acoustics**, which makes the SNR and gain distributions first-order
design parameters rather than nuisance knobs.

## F10 — The v2 augmentation policy, as specified, is an acoustic regression

`qc_contrast.py` builds synthetic training examples under each policy and compares the
speech/non-speech level gap against TEN's 30 human-labelled files (500 simulated examples per
policy, same seed):

| policy | contrast dB p10/p50/p90 | non-speech dB p10/p50/p90 | KS vs TEN (contrast) | KS vs TEN (non-speech level) | non-speech spectral KL |
|---|---|---|---|---|---|
| **TEN (target)** | 2.1 / 10.0 / 29.0 | −49.4 / −39.0 / −30.3 | — | — | — |
| v1 policy | 2.3 / 10.6 / 24.2 | −52.6 / −39.4 / −25.7 | D=0.129 **p=0.68** | D=0.235 p=0.074 | 0.260 |
| **v2 as written** (DESIGN-NOTES §7) | 3.4 / 15.6 / 59.5 | −100.0 / −51.1 / −33.0 | D=0.318 **p=0.005** | D=0.555 p=1.1e-08 | 0.295 |
| v2 + room-tone floor | 3.5 / 13.9 / 22.2 | −63.3 / −49.2 / −33.0 | D=0.269 p=0.027 | D=0.557 p=9.6e-09 | 0.366 |

v1's policy is statistically indistinguishable from TEN on contrast. The v2 policy is
significantly different, in the direction of *easier than reality*. Three causes, each
independently fixable:

1. **`noise_prob: 0.85`** leaves 15% of examples with digital-silence gaps — non-speech at
   −100 dBFS, which no microphone produces. The intent ("the model must see clean speech") is
   right; the implementation makes silence unphysical. Clean must mean *no added event*, not
   *no noise floor*.
2. **The SNR mixture is centred at 18 dB** where v1's uniform [−5, 20] had a median of 7.5 dB.
   Raising the floor from −5 dB was correct (see F4/F5); raising the *centre* by 10 dB was not
   part of that argument and was never measured.
3. **Gain [−18, +6] dB** has a −6 dB mean, pushing absolute levels below TEN's. Absolute level
   is a real feature here because the frontend is a linear magnitude spectrum with no
   per-utterance normalisation (instance norm was measured at −0.112 AUROC and excluded).

The v2 timing fixes are not in question — this is the level axis only, and the two are
independent. But **the policy must be re-derived against this measurement before WP6**, and the
target is coverage of the test distribution, not point-matching to TEN's 30 files.

## F11 — Reference checks that passed

- TEN labels: 267 segments, contiguous, coverage matches audio duration to < 1 ms, pooled
  occupancy **0.7522** (reproduces DESIGN-NOTES §2 exactly), shortest segment 53 ms — above the
  32 ms frame, so no segment is unrepresentable on the frame grid.
- No duplicate audio anywhere, by content hash.
- LibriSpeech: 146 speakers, 40/40/33/33 across dev-clean / test-clean / dev-other / test-other.
  The named 48-speaker benchmark reservation costs **6.53 h — 33% of the speakers and 31% of
  LibriSpeech time**. Affordable, but it is a real cost the roadmap did not price.

---

## The preparation and filtering the data needs

Ordered as a pipeline stage. Everything here is measurable and gated; nothing depends on
judgement calls that can drift.

### P0 — Integrity (keep, extend)
Existing conversion is sound: polyphase resample, mono downmix, int16, idempotent, duration
assertion. Add: finite-sample assert, non-empty assert, content hash per file, and a committed
`asset_qc.json` that every manifest record must reference. **New gate G15: no manifest may
reference a file absent from the QC report or marked quarantined.**

### P1 — Level conditioning
- **DC removal**: 1st-order high-pass at 20 Hz on every source at cache time. Cheap, removes
  the DC offsets and sub-audio rumble without touching the speech band.
- **Do not loudness-normalise.** Level variation is signal, not noise, for a model with no input
  normalisation — it must be *controlled by the gain augmentation*, not erased at the source.
- Record per-file peak / RMS / crest in the QC report so the augmentation policy can be derived
  from the real distribution instead of assumed.

### P2 — Noise-pool preparation (the largest single win available)
1. **Speech filter, measured not curated.** Drop every ESC-50 clip with > 2% speech frames
   (137 clips). Route clips with any speech frame at all (296) to the vocal-confuser class with
   its own SNR floor, replacing the category-name list entirely.
2. **Trim the padding.** Store each clip's active region (frames within 40 dB of its own peak)
   and compute mixing RMS over that region only. Fixes the 3–13.5 dB SNR error on 374 clips and
   makes `noise_prob` mean what it says.
3. **Drop hard-clipped clips** (48 above 1%) — clipping in the *noise* is an artifact the model
   should not learn to associate with non-speech; clipping in the *speech* stays, because TEN
   has it.
4. Union of the above: **553 of 2,000 clips (27.6%) are filtered or reclassified.** The fold-5
   holdout must be applied *after* this filter so both pools shrink proportionally.

### P3 — Impulse-response pool
Filter on measurement: RT60 ≤ 1.0 s, late-tail energy ≤ 0.5%, direct-path delay ≤ 20 ms →
**131 IRs**. Mandatory direct-path trim before convolution. Store RT60, DRR and delay per IR in
the manifest so the pool can be re-derived rather than re-curated.

### P4 — Speech-source preparation
- **De-hum** the 299 LibriSpeech files with > 20 dB mains peaks (notch at the detected f0 and
  two harmonics), or tag them and cap their share of any single training batch. Tagging is
  preferable — real deployments contain hum, and the model should be robust to it, not blind
  to it.
- Tag effective bandwidth per file. Do not resample or filter; ensure the benchmark speaker
  reservation is not accidentally the narrowband subset.
- **Quarantine the 327 near-empty FLEURS files** before any benchmark re-cut, and report the
  level distribution of whatever survives.

### P5 — AMI-specific repair
- Excise the digital-dropout regions (149.9 s + 36.5 s + 12.3 s + …) from the window generator
  rather than emitting windows that are mostly mathematical zero.
- Stratify the series split across the IS-vs-rest noise-floor divide (F3).
- Keep the union annotation as the *speech* upper bound and intersect with the teacher as
  planned — F1 now gives an energy-only cross-check on that intersection: it should remove most
  of the 22.4% of speech-labelled frames below −60 dBFS and very little else.

### P6 — Gap and augmentation synthesis (re-derive, do not assume)
- **Never emit mathematically silent gaps.** `make_gap_audio` currently coin-flips between
  zeros and −50 dBFS Gaussian white noise; both are wrong. Gaps should carry a room-tone floor
  with a realistic spectral tilt, at a level drawn to match the measured non-speech distribution.
- Re-derive `noise_prob`, the SNR mixture and the gain range by **fitting `qc_contrast.py`'s
  output to TEN's distribution**, with the acceptance criterion "KS p > 0.10 against TEN on
  contrast and on non-speech level" — a criterion v1 passes on contrast and v2-as-written fails.
- Keep the F4/F5 fixes (confuser floor, active-region SNR); they are independent of the level
  question and are the reason the SNR floor was raised in the first place.

### P7 — Report and gate
One `asset_qc.json` + `BENCHMARK.lock`-style hash, referenced from `run.json`. Gates G15
(quarantine enforcement) and G16 (train-vs-test acoustic match: KS p > 0.10 on contrast and
non-speech level, measured on the realised training distribution, not the config) join
[`GATES.md`](./GATES.md).

---

## Amendments this forces on the committed plan

Per the process rule in [`BETS.md`](./BETS.md) — an unamended falsified claim is a correctness
bug, not a documentation nit:

| where | claim | status |
|---|---|---|
| DESIGN-NOTES §7 | `noise_prob: 0.85` | **falsified as specified** — produces −100 dBFS non-speech; needs a room-tone floor (F10) |
| DESIGN-NOTES §7 | SNR mixture centred at 18 dB | **unmeasured and mismatched** — 10 dB above v1's centre; re-derive against F9/F10 |
| DESIGN-NOTES §7 | gain [−18, +6] dB | **unmeasured** — −6 dB mean shifts training below the test distribution |
| DESIGN-NOTES §7 | RIR pool = 86 by room name | **superseded** — the measured criterion gives 131 and is reproducible |
| DESIGN-NOTES §7 | vocal confusers = 5 named categories | **falsified** — finds 35 of 79 contaminated clips; use the measured filter |
| ROADMAP WP6 | AMI series split | **incomplete** — must stratify across the IS-vs-rest noise-floor divide |
| ROADMAP WP4 | 48-speaker reservation | **priced**: 6.53 h, 33% of LibriSpeech speakers |
| GATES.md | — | **G15, G16 added** (asset quarantine; train/test acoustic match) |
