# v2 Bet Register

Every claim in the v2 plan, classified by evidence status. The standing rule: **v2 makes no
decision on a class-B item until its spike has run.** Class-D items are named as bets, with
fallbacks, so nothing is a bet by accident.

Classes: **A** measured (cite) · **B** cheaply verifiable now, no training (spike specified) ·
**C** verifiable only by a training run (rung named) · **D** irreducible bet (fallback named).

Ranked by danger-if-wrong × cheapness-to-verify. Items already settled this session are marked
⏹ with their outcome.

---

## Settled this session (moved to A)

| # | claim | outcome |
|---|---|---|
| ⏹ | "The pipeline is dataloader-bound; co-training is nearly free" | **FALSE.** Model-bound everywhere; co-training ≈ 1.8× per step (DESIGN-NOTES §3). Roadmap and DECISIONS amended. |
| ⏹ | "48 held-out LibriSpeech speakers exist on disk" | **FALSE as stated.** Zero held out today; v2 names an explicit reservation, enforced by G10 (DESIGN-NOTES §5). |
| ⏹ | Paired-σ ratios 1.7× / 2.1× | Transposed; measured 1.95× / 1.69×. Conclusion strengthens (DESIGN-NOTES §1). |
| ⏹ | "A paired bootstrap on 30 TEN files detects +0.027" | **TRUE**: MDE(30, paired) = 0.0188 (DESIGN-NOTES §1). |
| ⏹ | 8-chunk warm-up mask covers the cold-start transient | **Under-sized**: transient p95 persists 20–30 chunks. Mask length now derived, not assumed (DESIGN-NOTES §4). |
| ⏹ | Ship gate 3 margin −0.005 is "non-inferiority" | **FALSE**: below TEN-30's measured resolution; it demanded a +0.014 win. Widened to −0.015. |
| ⏹ | TEN gap/occupancy targets (202/463/926 ms etc.) | Verified exactly; macro-vs-pooled convention pinned (DESIGN-NOTES §2). |
| ⏹ | 42% per-step compute at 8 s crops | Confirmed: 0.22/0.54 = 0.41 (DESIGN-NOTES §3). |
| ⏹ | "ESC-50 is a noise corpus; vocal confusers are the 5 named categories" | **FALSE.** 296/2000 clips carry speech; 44 of the 79 worst lie outside the named categories (DATA-QC §F4). Category curation replaced by a measured filter. |
| ⏹ | `mix_at_snr` delivers the requested SNR | **FALSE for 18.7% of clips** — zero padding inflates audible noise by 3–13.5 dB (DATA-QC §F5). |
| ⏹ | The v2 augmentation policy improves distribution match | **FALSE on the level axis.** v1 matches TEN (KS p=0.68), v2-as-written does not (p=0.005), in the direction of easier-than-reality (DATA-QC §F10). DESIGN-NOTES §7 banner-marked; policy re-derived before WP6. |
| ⏹ | RIR pool = 86 IRs by room name | **Superseded**: measured criterion (RT60 ≤ 1.0 s, tail ≤ 0.5%, delay ≤ 20 ms) gives 131 (DATA-QC §F6). Direct-delay max 90.2 ms confirmed. |
| ⏹ | Label noise needs the teacher to detect | **FALSE** — energy alone shows 13.3% of LibriSpeech and 22.4% of AMI speech-labelled frames are silent (DATA-QC §F1). Independent cross-check on the teacher's output. |
| ⏹ | 48-speaker reservation is free | **Priced**: 6.53 h, 33% of LibriSpeech speakers (DATA-QC §F11). |

## Open — class B (spikes; all in WP0.5, zero training runs)

| # | claim at risk | spike | decision rule |
|---|---|---|---|
| **B1** | All six teacher-protocol numbers (F1 0.9383, FAR 0.015, 45.4 / 63.5 trans/min, deletion-run shape, threshold sweep) — currently reproducible from **no artifact in this repo**: no silero code, no cached teacher output exists here | **(a)** Install pinned `silero-vad`; script teacher probs @ 0.5 with per-utterance/meeting state reset over TEN-30 and the AMI mixdowns; recompute all six. ~2 h. **Commit the script and the cached teacher outputs.** | Each reproduces within ±0.005 F1 / ±0.005 FAR / ±3 trans/min; else Phase 2's gates are recalibrated before any rebuild. |
| **B2** | +0.027 log-compression transfers to rebuilt data; R2's positive-control logic (as first drafted, non-reproduction couldn't distinguish "harness broken" from "effect didn't survive the rebuild") | **(b)** Rebuild the 500-step A/B harness (not in the repo); 3 seeds linear-vs-log on **v1 data** (validates harness against the known +0.027), then 3 seeds on a **rebuilt-data sample** (measures transfer). ~1 day. | v1-data replication within ±0.008 validates the harness; the rebuilt-data number becomes R2's pre-registered expectation. |
| **B3** | BatchNorm's +0.003 is real signal, not harness noise | **(c)** A/A run: 4 seed-replicates of the identical log arm. ~20 min compute at measured step times. | If replicate σ ≥ 0.003, R3 loses its accuracy claim and is judged on dead-channel count only. |
| **B4** | The FSTTM operating point is insensitive to the hand-tuned c_deadair | **(d)** Sweep the cost ratio over [1.0, 2.5] s-per-interruption on cached v1 probabilities, re-running the joint grid. Needs the turn-cost code (WP2). Minutes of compute. | Selected θ and dead-air p90 stable across the range → knob immaterial. Unstable → ratio becomes a reported sensitivity; fallback: ship the operating-point frontier. |
| **B5** | Target mix statistics (0.697 occupancy, ~52 trans/min, calibration slice 0.767/67, pure-positives → 0.0%) | Fall out of spike (a) + a manifest rebuild; no training. | The WP6 gates as written, pooled convention. |
| **B6** | 2.0 s hybrid deletion cap is at the knee | Sweep cap ∈ {1.0, 1.5, 2.0, 3.0, ∞} over cached teacher/AMI disagreement runs (spike (a) output). Minutes. | Choose the knee; pre-register it. |
| **B7** | "278-cluster composite MDE ≈ 0.006" transfers beyond TEN | Score both pinned v1 checkpoints on the built benchmark at WP4 exit; compute realized per-cluster σ per subset. Already the Phase 1 acceptance. | MDE ≤ 0.010 pooled or the benchmark grows before WP10. |
| **B9** | A fitted augmentation policy can pass G16 (KS p > 0.10 on both contrast and non-speech level) **while keeping** the F4/F5 fixes — the raised SNR floor and the +6 dB confuser floor both push toward *cleaner*, which is the direction that failed | Grid over `noise_prob` × SNR-mixture centre × gain range with `qc_contrast.py`, 500 examples per point. Hours of compute, no training. WP1.5. | A feasible region exists → adopt its centre. None exists → the confuser floor and the level target are in genuine conflict, and the conflict is reported and resolved by measurement rather than split-the-difference. |
| **B10** | LOSO over 9 AMI series meaningfully beats a 3-series holdout for ship gate 4 | Compute realized per-fold σ from the pinned v1 probabilities across all 9 folds at WP4 exit (no training — v1 is already trained). | Per-fold σ implies a LOSO MDE materially below the 3-series MDE → adopt. Not materially better → keep the single split and save 9× the final-config compute. |
| **B8** | 40-epoch total-supervision equivalence | Recompute against the actual rebuilt manifest counts on rebuild day. Minutes. | Adjust epochs to hold total supervised frames ≈ v1's. |

## Open — class C (training-run evidence; ladder rungs)

| # | claim | rung | note |
|---|---|---|---|
| C1 | The data rebuild itself helps (separate from the supervision fixes) | **R1a vs R1b** — added precisely so the label rebuild's own Δ is measured; the first draft confounded them | |
| C2 | Encoder reshape is "not worse" | R4 | At the resolution limit (MDE ≈ 0.006); "~0" is unfalsifiable below that — only "not worse" is claimable. |
| C3 | Future-frame aux head improves offsets | R7 | Judged on offset F1 + dead-air percentiles only. |
| C4 | Hard negatives move FAR without hurting recall | pooled FAR gate + ship gate 6 | Pre-register the FAR threshold before training; the knobs are removable on regression. |
| C5 | EMA beats raw selection | R8 (post-hoc, zero extra runs) | Fallback: ignore EMA weights. |

## Open — class D (irreducible; named fallbacks)

| # | bet | fallback |
|---|---|---|
| D1 | The teacher's label quality ceiling (~F1 0.94 on TEN) is high enough for the conversational-slice ship gate | The teacher-free AMI series is the audit slice; if conclusions hold only under teacher labels, the label strategy — not the model — is revisited. |
| D2 | Oracle-gap ≤ 0.03 and θ-spread ≤ 0.15 are the right calibration-stability thresholds | Provisional until the first nested-resample run; the in-plan fallback (grow the calibration set) already exists. |
| D3 | ~19 h of real conversational speech is enough to clear the always-speech floor on the conversational slice | None inside this plan. If R1b + the full ladder still fails ship gate 4, the next version's first line item is data acquisition, not architecture. |

---

**Process rule:** when a spike settles an item, move it to ⏹ with the measured outcome and amend
the affected roadmap/gate text in the same commit. An unamended falsified claim is treated as a
correctness bug, not a documentation nit.
