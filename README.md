# vad-models

Training pipeline for lightweight, real-time Voice Activity Detection (VAD) models, built to host
multiple architectures (CRNN, TCN, FSMN, ...) through a shared, config-driven pipeline.

**Status: v1 complete and archived at tag `v1.0`. v2 in planning.**
See [`docs/v2/ROADMAP.md`](./docs/v2/ROADMAP.md) for current work.

## Where things are

| | |
|---|---|
| Current plan | [`docs/v2/ROADMAP.md`](./docs/v2/ROADMAP.md) |
| v1 results, corrected | [`docs/v1/RESULTS.md`](./docs/v1/RESULTS.md) |
| v1 archive (weights, exports, manifests, logs) | `runs/v1/` |
| Naming and versioning rules | [`docs/CONVENTIONS.md`](./docs/CONVENTIONS.md) |
| Everything else | [`docs/README.md`](./docs/README.md) |

## What v1 achieved

Two models — a 210K-parameter CRNN in the Silero shape and a 180K-parameter TCN in the MarbleNet
shape — trained 50 epochs each on a ~40 h local kit.

| | CRNN | TCN | always-speech baseline |
|---|---|---|---|
| TEN F1 @ 0.5 | 0.8974 | 0.8961 | **0.8586** |
| TEN AUROC | **0.9046** | 0.8876 | 0.500 |
| False-alarm rate | 0.5005 | 0.5404 | 1.000 |
| CPU streaming p95 | 1.85 ms | 103 ms | — |

Read that against its baseline: the F1 lift over a parameterless predictor is **+3.9 points**, and
on the only conversational slice in the kit both models score at or below the trivial floor. A
2026-08-08 audit found six defects that invalidated much of v1's reported measurement — see
[`docs/v1/ERRATA.md`](./docs/v1/ERRATA.md). v2 is organised around fixing the measurement first.

## Origin

This project follows research done in a companion Obsidian vault ("Voice Activity Research"),
which tears down production lightweight VADs (Silero, TEN VAD, MarbleNet, FSMN) and assembles a
~8 GB local training-data kit (LibriSpeech, ESC-50, AMI, Aachen AIR RIRs, a TEN-VAD benchmark set,
FLEURS Swahili). That data lives outside this repo and is referenced in place via
`configs/data/paths.yaml` — never copied in bulk.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/check_data_paths.py
```

## Running the full pipeline unattended

`./train.sh` runs the whole pipeline end to end (data verification, manifest build, full training
+ evaluation + ONNX export for both architectures) and self-caffeinates so it survives
lid-close/display-sleep — safe to kick off and walk away from:

```bash
./train.sh                  # full runs
EPOCHS=10 ./train.sh        # override epoch count for a shorter run
```

Logs to `train_run_<timestamp>.log` in the repo root. Preprocessing and manifest steps are
idempotent; each training step starts a fresh model under its run name.

Note for v2: `train.sh` currently evaluates and exports `best.pt`, which the audit found is the
*worse* checkpoint for both architectures. Phase 0 changes this.

## Project layout

- `configs/` — YAML configs for data, model, train, eval
- `src/vad/` — library code (labels, data, augment, models, engine, eval, export)
- `scripts/` — thin CLI entrypoints
- `tests/` — unit tests, mirrors `src/vad/`
- `docs/` — roadmaps, decisions, errata, conventions
- `runs/` — committed archive of completed versions' artifacts
- `data_cache/` — generated, gitignored (preprocessed assets, manifests, indexes)
- `checkpoints/` — generated, gitignored (live run output)
