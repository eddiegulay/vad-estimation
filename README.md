# vad-models

Training pipeline for lightweight, real-time Voice Activity Detection (VAD) models, built to host multiple architectures (CRNN, TCN, FSMN, ...) through a shared, config-driven pipeline.

See [`ROADMAP.md`](./ROADMAP.md) for current status and the phase-by-phase plan.

## Origin

This project follows research done in a companion Obsidian vault ("Voice Activity Research"), which tears down production lightweight VADs (Silero, TEN VAD, MarbleNet, FSMN) and assembles a ~8GB local training-data kit (LibriSpeech, ESC-50, AMI, Aachen AIR RIRs, a TEN-VAD benchmark set, FLEURS Swahili). That data lives outside this repo and is referenced in place via `configs/data/paths.yaml` — never copied in bulk.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/check_data_paths.py
```

## Running the full pipeline unattended

`./train.sh` runs the whole pipeline end to end (data verification, manifest
build, full training + evaluation + ONNX export for both `crnn_v1` and
`tcn_v1`) and self-caffeinates so it survives lid-close/display-sleep — safe
to kick off and walk away from:

```bash
./train.sh                 # full runs, per configs/train/default.yaml
EPOCHS=10 ./train.sh        # override epoch count for a shorter run
```

Logs to `train_run_<timestamp>.log` in the repo root. Preprocessing/manifest
steps are idempotent (safe to re-run); each training step always starts a
fresh model under its `_full` run name.

## Project layout

- `configs/` — YAML configs for data, model, train, eval
- `src/vad/` — library code (labels, data, augment, models, engine, eval, export)
- `scripts/` — thin CLI entrypoints
- `tests/` — unit tests, mirrors `src/vad/`
- `data_cache/` — generated, gitignored (preprocessed assets, manifests, indexes)
- `checkpoints/` — generated, gitignored (per-run checkpoints, ONNX exports, eval reports)
