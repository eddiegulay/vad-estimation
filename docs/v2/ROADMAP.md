# VAD Models — v2 Roadmap

**v2 is a fresh build, not a patch series.** v1 survives as three things only: the frozen tag
(`v1.0` — the code archive), the committed artifacts (`runs/v1/` — the results archive), and
knowledge (the audit's measured facts, which inform every design choice below). The new code
replaces `src/vad/` and `tests/` wholesale on `master`; there are no versioned source packages
(see [`../CONVENTIONS.md`](../CONVENTIONS.md) §1 for why).

v1 record: [`../v1/`](../v1/) · corrected results: [`../v1/RESULTS.md`](../v1/RESULTS.md) ·
audit: [`../v1/AUDIT.md`](../v1/AUDIT.md) · gates: [`GATES.md`](./GATES.md) · measured design
inputs: [`DESIGN-NOTES.md`](./DESIGN-NOTES.md) · open-bet register: [`BETS.md`](./BETS.md).

Started 2026-08-08. Status: **WP0 — not started.**

---

## What v1 taught, in one paragraph

v1 shipped two models that beat a parameterless baseline by 3.9 points of F1, scored *at or
below* that baseline on the only conversational data in the kit, never released the floor on a
third of turn-ends, and reported numbers that a five-agent audit found were largely measuring
corrupted activations. Every one of the six blocking defects was invisible in the metrics being
reported. **The binding constraint on v1 was never the model or the data — it was that the
project could not tell a real improvement from noise.** v2 is organised around fixing that
first.

## The v2 thesis

> Pin what v1 actually produced. Build the measurement harness and prove it against those pinned
> bytes. Then build the new pipeline from spec, proving each seam before the next, and train
> only when the benchmark can resolve the changes it is meant to evaluate.

Four consequences that shape everything below:

1. **Pin before delete.** The v1 probability files that anchor the new harness must be generated
   by v1's own code, in a worktree at the tag, *before* `src/vad` is replaced. They cannot be
   produced by the code they exist to validate.
2. **No training run happens until the benchmark can resolve the changes.** Measured this
   session: the paired per-cluster σ on TEN is 0.0367, giving an MDE of 0.019 on 30 files and
   ~0.006 on the full composite — so the plan's resolution targets are real, but only with
   paired scoring, which v1 never used.
3. **Correctness fixes are not ablations.** They are verified by gates, not by spending training
   runs proving correct code beats broken code.
4. **Nothing ships as a claim without being reproducible from the repo.** Every number in this
   plan that was measured in a design session is either committed in
   [`DESIGN-NOTES.md`](./DESIGN-NOTES.md) with its script, or listed in [`BETS.md`](./BETS.md)
   with the spike that will settle it. Two claims from the first draft of this roadmap were
   *falsified* by measurement before any code was written — see "Corrections" below. That is the
   process working.

## Corrections to this plan's own first draft (2026-08-08)

Measured before execution started; the plan below already incorporates them:

- **"The pipeline is dataloader-bound" is false.** Measured: item production ≈ 0.19 s per
  batch-64 across 4 workers; the model step is 0.22–0.54 s. The pipeline is model-bound in every
  regime, and co-training the TCN costs ~1.8× per step, not "nearly free". Co-training is kept
  for its perfect pairing, with the real cost budgeted.
- **"48 held-out LibriSpeech speakers" did not exist.** v1 used all 146 speakers across its
  train and val pools; zero were held out. v2 names an explicit 48-speaker benchmark reservation
  excluded from every training manifest, enforced by the leakage gate.
- The paired-σ ratios in the first draft were transposed (measured: 1.95× AUROC, 1.69× F1); the
  conclusion (pairing pays) strengthens.
- The 8-chunk loss warm-up mask was under-sized: the cold-start transient's p95 persists 20–30
  chunks on the trained v1 models. The mask length is now *derived* from the measured transient,
  not assumed (WP7).
- The first draft's ship gate 3 margin (−0.005) was tighter than TEN-30's measured resolution
  (±0.013), silently demanding a large win while claiming non-inferiority. Widened to −0.015.

---

## The fresh build: what carries, what is rewritten, what is dropped

**CARRY** means: the file is copied from `v1.0` via `git checkout v1.0 -- <path>`, together with
its test file, and reviewed once against the audit — the diff at review must be import-paths
only. **REWRITE** means: new code against a written spec, with the v1 test file imported first
as the spec's executable half (each changed assertion carries a `# CHANGED-FROM-V1:` comment
naming the audit finding). **DROP** means: no successor.

| Module | Verdict | Why |
|---|---|---|
| `labels/intervals.py` | **CARRY** | Zero audit findings; involution/idempotence/round-trip invariants tested. The off-by-one bug lived in `dataset.py`, not here. |
| `labels/ten.py` | **CARRY** | Trivial parser of a frozen format. Must never change. |
| `labels/ami.py` | split | Parser CARRIED; discovery REWRITTEN (root of the one-meeting-val bug: it listed 171 annotation dirs when 33 have audio). Series-split helpers NEW. |
| `labels/synthetic.py` | split | Sampling/assembly functions CARRIED (the *parameters* were wrong, not the code). `trim_internal_silence` **DROPPED** — measured incapable (target pauses sit at −21.5 dB median, unreachable by an RMS gate without clipping fricatives). |
| `labels/teacher.py`, `hybrid.py` | **NEW** | Silero frame probabilities at fixed 0.5; AMI human-∩-teacher with 2.0 s deletion cap. |
| `data/assets.py` | **CARRY** + extend | Verified converters; extend with RIR direct-path trim and room/RT60 metadata. |
| `data/manifest.py` | **REWRITE** | Embodies four audit findings; schema changes (provenance, tags, noise/RIR membership, versioned dirs, refuse-overwrite). |
| `data/dataset.py` | **REWRITE** | Built on three defective assumptions (mutable epoch, variable-length output, `round()` frame count). The `(run_seed, epoch, index)` seeding idea survives as spec. |
| `data/collate.py` | **DROP** | Fixed 8 s crops make training collate `torch.stack`; eval scores one file at a time. The padded-tail bug class retires with it. |
| `augment/*` (all four) | **CARRY** | Pure functions, fuzz-tested, zero code defects. Every augmentation finding was a *policy* defect, and policy moves to the manifest builder. |
| `models/*` (all four) | **REWRITE** | Encoders structurally defective (9.7%/11.4% dead taps); frontend gains log compression; `forward_full` returns logits. Streaming-state design and the TCN cache rationale carry as spec, docstrings verbatim. |
| `engine/*` | **REWRITE** | Contains B2, B6, B7, the biased loss averaging, no seeding. Warmup+cosine shape and class-weight math carry as spec. |
| `eval/*` | **REWRITE** | The heart of v2 (`measure/`). Rank-sum AUROC implementation carried with its tests; everything else replaced. |
| `export/onnx_export.py` | **REWRITE from proven spec** | The contract (I/O names, static shapes, opset 18, caller-seeded state) carries verbatim — v1's export was sound. Code follows the new models; single self-contained file. |
| `postprocess/hysteresis.py` | **CARRY** + guard | The state machine is correct and tested; B4 was a calibration defect. Add the `theta_off ≤ theta_on` raise. |
| `config.py` | **CARRY** + wrap | `load_yaml`/`deep_merge` fine; add access tracking and postprocess validation. |

**The expected-divergence register.** Every seam proof against v1's pinned bytes must either
match to tolerance or match a pre-registered entry in the divergence table in
[`DESIGN-NOTES.md`](./DESIGN-NOTES.md) (floor-vs-round frame counts, NaN zero-fill vs drop,
`>=` vs `>` in the sweep, trim removed, val meeting selection). A third outcome is a
stop-the-line bug. Nothing joins the register after its seam proof has run.

**The do-not-redesign list.** Agreed before the cutover; changes require measured justification:
the interval algebra and its on-disk format; the 512/128/256 frontend constants and zero
lookahead; the streaming state contract; the hysteresis machine; plain-YAML config (no Hydra, no
Lightning, no plugin systems); the ONNX I/O contract; the registry pattern; JSONL manifests.

---

## Work packages

Each WP has an exit check; nothing starts until the previous WP's check is green. Training is
WP10 of 12.

| WP | What | Exit check | Phase |
|---|---|---|---|
| **WP0** | **Pin v1.** In a worktree at `v1.0`, score both archs × {best, last} × {test_ten, val, sanity_fleurs} on **CPU**; freeze per-file raw probabilities + labels to `.npz` under `runs/v1/`; update `SHA256SUMS`. The pin script asserts it is running at the tag and refuses elsewhere. Includes the FLEURS evaluation that v1 never ran. | `.npz` hashes committed; the five pinned corrections (enumerated in GATES.md Phase-0 acceptance) reproduce from the new bytes. | 0 |
| **WP0.5** | **De-risk spikes** (no training): (a) install pinned `silero-vad`, reproduce the six teacher-protocol numbers against TEN and AMI, commit script + cached teacher outputs; (b) rebuild the 500-step A/B harness, replicate linear-vs-log on v1 data (validates the harness against the known +0.027), then run it on a rebuilt-data sample (measures transfer — this number becomes R2's pre-registered expectation); (c) A/A seed-replicate run to measure the harness noise floor; (d) c_deadair ratio sweep on cached v1 probabilities. Full specs in [`BETS.md`](./BETS.md). | Each spike's decision rule met or the affected plan item amended before dependent WPs start. | 0/2 |
| **WP1** | **Cutover.** One commit deletes `src/vad/` + `tests/`, creates the new skeleton (`core/`, `labels/`, `data/`, `measure/`, `models/`, `engine/`, `export/`, `postprocess/`, `tests/gates/`), carries the CARRY modules with their tests, renames configs per CONVENTIONS (`default.yaml` retired; v1 YAMLs frozen read-only). | Carried suites pass unmodified except imports; carried files provably copies from `v1.0`. | 0 |
| **WP2** | **`measure/`** — metrics (both classes, FAR/MISS, bal-acc, MCC, AP×2, rank-sum AUROC), events (onset **and offset** F1, latency distributions + never-released), FSTTM cost, cluster bootstrap (paired by construction, B=10000, NaN→zero-fill), report schema (no bare floats), slices (groupby over tags), calibration, benchmark acceptance. **Imports numpy/scipy only — a torch import in `measure/` fails CI.** Scores any `ProbabilitySource`, including WP0's `.npz`. | Seam proofs S1–S4: reproduces every corrected v1 number from the pinned bytes to 1e-9 (frame metrics), reproduces the audit's event/cost/bootstrap numbers to their stated tolerances; `ConstantSource` reproduces the always-speech floor exactly. | 1 |
| **WP3** | **Gates + preflight.** All gates in GATES.md as `tests/gates/test_g*.py`; `scripts/preflight.py` (≤90 s subset); `train.sh` refuses on red. Behavioral gates run against the `v1.0` worktree once; artifact gates run against `runs/v1/` bytes forever. Every measured failing value recorded in `GATE_BASELINE.md` — **a gate with no baseline row cannot be marked green.** | Every gate demonstrably fails against v1 with the predicted value (G10 via the reconstructed v1 aug-template membership file). | 0 |
| **WP4** | **Benchmark.** TEN-matched LibriSpeech concat over the named 48-speaker reservation; FLEURS test **re-cut** to TEN statistics; AMI diagnostic from the **val+test series only** (3 series — the 9-series version in this plan's first draft leaked the six training series); silence-only set from ESC-50 fold 5; KS acceptance (distribution-matched sets only); materialise + `BENCHMARK.lock`. Kick off the 50-clip hand-labelling (human task, parallel from here). **Phase 1 exit: score v1 `last.pt` on every new set and pin those probabilities too** — ship gates 1, 2 and 6 are undefined without them. | KS p > 0.10 per matched set; pooled paired ΔAUROC MDE ≤ 0.010 *measured from pinned v1 probabilities on the new composite*; v1 baselines pinned; lock verified. | 1 |
| **WP5** | **Labels.** `teacher.py` (pinned version, weights hash folded into `manifest_set_id`), `hybrid.py`, split logic. | Teacher-vs-TEN F1 ≥ 0.93; AMI FAR ≤ 0.03; deletion-run p99 < 2 s; one AMI series teacher-free. | 2 |
| **WP6** | **Manifests v2.** Series-level AMI splits, speaker splits honouring the benchmark reservation, three disjoint aug templates (fold-5 holdout real at last), retargeted gaps ([0.15, 1.2] s, 5% long-pause into [1.5, 4.0] s), duration-driven assembly, calibration slice by occupancy band, `noise_only` hard negatives, RIR pool restriction + direct-path trim, tags + **noise/RIR membership on every record** (without which the leakage gate is unenforceable — verified against v1's manifests, which record neither). | Leakage gate green on v2 manifests, red on v1's; pooled train transitions/min ≥ 45 (pooled convention — see DESIGN-NOTES for the macro/pooled distinction); occupancy ∈ [0.66, 0.72]; calibration slice KS p > 0.10; zero-lag test (cross-correlation peak within ±1 frame) for 100% of pool RIRs; clean-speech fraction matches the configured gate ± 2 pts. | 2/3 |
| **WP7** | **Dataset + sampler.** `(epoch, index)`-keyed dataset, fixed 8 s crops from the per-item RNG, warm-up mask with **k derived from the measured cold-start transient** (rule: smallest k with median |Δp| < 0.02; re-measure on the trained v2 model), augmentation fingerprint in every item. | Supervision gate green under its corrected definition (every example ≥ 1 supervised frame; supervised cells ≥ 0.95; 4-epoch cumulative frame coverage ≥ 0.95); epoch-propagation gate green under persistent spawn workers; dataset replay of v1's manifests reproduces v1's label statistics (seam proof S7). | 4 |
| **WP8** | **Models + export — before any training.** Log-compressed frontend pair (`rfft` for training/streaming, conv-DFT for export; both epsilons frozen), valid-conv encoder + BatchNorm (param-matched to v1 ± 2%), CRNN + TCN with unfold-rewritten depthwise, chunked encode, `set_num_threads(1)`, ONNX export (sigmoid in graph, single file). | Dead-parameter gate green on random init; streaming ≡ `sigmoid(forward_full)` ≤ 1e-6; ONNX parity ≤ 1e-3 + metric match to 4 dp via the harness; frontend cross-parity ≤ 1e-3 log-mag / ≤ 1e-3 prob / ≤ 1e-4 AUROC; CPU streaming p95 ≤ 10 ms both archs (weight-independent, so testable now). | 5 |
| **WP9** | **Engine.** `bce_with_logits` loss (weight-mass denominator, warm-up mask, `isfinite` guard), schedule with `lr_lambda(0) > 0`, trainer (EMA, AUROC selection, ship-`last`, `--resume`, per-epoch weight-only history, `metrics.jsonl`, `run.json`), cross-backend startup guard, memory-budget test carried forward from v1 (the sampler/dataset path it guards is exactly what changed). | Loss unit-matches closed form including the confidently-wrong extreme (B6's counterexample as a test); lr(0) > 0; determinism gate green; overfit-8 near-zero; quarter-length dry run writes all bookkeeping and resumes mid-run. | 0/4 |
| **WP10** | **Ablation ladder — first training.** R0 (v1 arch + fixes) → R8 per the table below, 500-step paired 3-seed screens, full runs for survivors, both archs co-trained at the final config (cost: ~1.8× per step, budgeted, justified by perfect pairing). | R2 within its pre-registered band from spike (b) — **the two-stage protocol separates "harness broken" from "effect didn't transfer", which a bare positive control cannot do.** Every rung's threshold committed to `ablation-preregistration.md` before its first run (git timestamps are the check). | 6 |
| **WP11** | **Operating point.** Joint 4-param grid on cached probabilities, FSTTM cost objective (ratio fixed a priori; sensitivity from spike (d) reported), θ_off ≤ θ_on hard, zero-debounce arm, nested resampling (≥1000 outer), oracle gap, per-param spread. | Oracle gap ≤ 0.03 onset F1; θ spread ≤ 0.15 (both provisional until the first nested run — marked as such); calibrated beats raw with paired CI excluding zero. v1's pinned probabilities are the permanent failing control. | 7 |
| **WP12** | **Ship.** The eight criteria below against pinned v1 `last.pt`; promote to `runs/v2/`; tag `v2.0`. | All eight. | 8 |

### Dependency structure

```
WP0 ── WP0.5 ── WP1 ──┬── WP2 ── WP3 ── WP4 ─────────┐
                      ├── WP5 ── WP6 ──┬── WP7 ──────┤
                      └── WP9 ─────────┴── WP8 ──────┴── WP10 ── WP11 ── WP12
```

WP2/WP3 (measurement track) and WP5 (labels) run concurrently after the cutover. **WP4 gates
WP10 absolutely** — `train.sh` mechanically refuses to launch while preflight is red or
`BENCHMARK.lock` is absent, so the dependency is enforced by tooling, not discipline.

---

## The ablation ladder (WP10)

| Rung | Arm | Pre-registered expectation | Judge on |
|---|---|---|---|
| R0 | v1 architecture + correctness fixes | — | reference |
| R1a | + supervision bundle, **v1 labels** | largest single delta | AUROC |
| R1b | + rebuilt labels/data | *separately measured* — the first draft confounded data and supervision in one rung, so the label rebuild's own Δ would never have been known | AUROC |
| R2 | + log compression | the transfer number from spike (b), not the raw +0.027 | **positive control** |
| R3 | + BatchNorm | accuracy claim only if the A/A noise floor from spike (c) is below +0.003; otherwise judged on dead-channel count alone | AUROC / dead channels |
| R4 | + encoder reshape | ~0 ± 0.005 — **at the resolution limit (MDE ≈ 0.006); "not worse" is the only falsifiable claim** | params, dead params, latency |
| R5 | frequency spacing (3 arms) | ≤ ±0.005 | sample efficiency |
| R6 | temporal layer (4 arms) | wash | latency p95, params, then AUROC |
| R7 | future-frame aux head | unknown | offset F1, dead-air p50/p90 — never frame F1 |
| R8 | selection metric, EMA, operating point | — | post-hoc, zero training runs |

Bundled without ablation (no accuracy hypothesis): all correctness work; all bit-identical
implementation/latency work.

---

## Ship criteria (WP12)

Against pinned v1 `last.pt`, scored on the same sets:

1. Paired ΔAUROC on the pooled TEN-matched composite: 95% CI lower bound **> 0**. (Composite
   composition and its 72%-FLEURS caveat are stated in DESIGN-NOTES — the per-set breakdown is
   reported alongside, always.)
2. Paired Δ turn-cost-per-turn: 95% CI upper bound **< 0**.
3. TEN-30 paired ΔAUROC: CI lower bound **> −0.015** (non-inferiority at the instrument's
   measured resolution; the first draft's −0.005 secretly demanded a +0.014 win).
4. **Conversational slice frame F1 exceeds its always-speech floor with CI lower bound > 0.**
   v1 fails this. It is the single criterion that says whether v2 solved the actual problem.
5. CPU streaming p95 within budget for every shipped architecture.
6. False alarms per hour on the silence-only set: paired against v1 `last.pt` on the same set
   (pinned at WP4 exit), 95% CI of the difference entirely below zero — not a bare point
   comparison.
7. Every shipped artifact carries `manifest_set_id`, teacher version + weights hash, and
   `BENCHMARK.lock` hash in `run.json`.
8. The ship report states the data-licence constraints (ESC-50 CC BY-NC, Aachen AIR unstated →
   research-only; the materialised benchmark inherits ESC-50's non-commercial restriction and is
   therefore **not redistributable**) or documents their replacement.

---

## CI

The v1 workflow runs `pytest -v` on ubuntu, where every data-backed test silently skips — a
green check attests to almost nothing. v2:

- CI fails if any test *skipped* is one of the gates without an explicit
  `preflight --report-skips` accounting artifact attached to the run.
- The `measure/`-imports-no-torch check runs on every push (pure grep, no data needed).
- The full gate suite runs on the data-present machine via `scripts/preflight.py`; its
  `preflight_report.json` is archived beside every checkpoint. CI's job is to prove the *code*
  of the gates is importable and their fixtures load; the data machine's job is to prove they
  pass.

## What would make v2 fail

- **Shipping before WP4.** Every accuracy claim unfalsifiable, exactly as in v1.
- **Trusting the teacher.** Validated on 30 human-labelled files; capped near its own quality.
  The teacher-free AMI series exists for this reason.
- **Treating a seam-proof mismatch as noise.** Match, or match the divergence register, or stop.
- **Tuning residual distribution mismatches to zero.** TEN's zero long-silence mass is an
  artifact of its file lengths, not a property of conversation.
- **Reading frame F1 as progress.** Retired as an objective; it hid every failure that mattered.
- **Assuming ~19 h of real conversational speech is enough.** Still the binding constraint;
  nothing in this roadmap fixes it.
