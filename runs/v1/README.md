# runs/v1 — archived v1 artifacts

Frozen 2026-08-08 at tag `v1.0`. Written once, never rewritten.

## Why this directory exists

`checkpoints/` and `data_cache/` are gitignored, and **v1 training was unseeded** — model
initialisation, DataLoader shuffle order and the crop offset all draw from unseeded RNGs, and
MPS adds its own nondeterminism on top. `git checkout v1.0 && ./train.sh` therefore produces
*a* model, not *the* model.

v1 cannot be preserved as a recipe. It is preserved here as artifacts.

## Contents

| Path | What |
|---|---|
| `crnn_v1_full/`, `tcn_v1_full/` | weights, ONNX exports, eval reports, TensorBoard events, per-epoch curves |
| `smoke/` | eval reports for the 20-step smoke run and the random-init plumbing check |
| `manifests/` | the exact train/val/test manifests v1 trained on, gzipped |
| `logs/` | full training logs, gzipped |
| `env.txt`, `pip-freeze.txt` | the environment |
| `data_fingerprint.json` | sha256 of each corpus index, so a future rebuild can be verified identical |
| `SHA256SUMS` | integrity of everything above |

## Run name aliases

v1 predates the naming convention in `docs/CONVENTIONS.md`. The produced names are kept
verbatim so the archive matches what the ROADMAP and logs reference.

| as produced | v2 grammar equivalent |
|---|---|
| `crnn_v1_full` | `v1-crnn-full` |
| `tcn_v1_full` | `v1-tcn-full` |
| `crnn_v1_smoke` | `v1-crnn-smoke` |

## Read this before using anything here

- **The ONNX exports are the mis-selected checkpoint.** `train.sh` exported `best.pt`, but
  `best.pt` is worse than `last.pt` on every threshold-free metric for both architectures.
  Prefer `last.pt`. See `docs/v1/ERRATA.md` E4.
- **Every validation metric in these reports is affected by a silent MPS bug.** `nn.Conv1d` on
  MPS corrupts rows ≥ 65536, and validation batches reached 162,368 rows. Gradients were
  unaffected — only the metrics are wrong. See `docs/v1/ERRATA.md` E2, E3.
- **Optimizer state is stripped** from all four `.pt` files (61% of each file). They load for
  inference and re-scoring; they will not resume.
- Log→run mapping: `train_run_20260807_070048` is CRNN, `train_run_20260807_093305` is TCN.
  The second log resumes the pipeline and skips CRNN training.

## Code provenance

Both checkpoints record `git_commit=d366899`. `git diff d366899 v1.0 -- src scripts configs
tests pyproject.toml` is empty — only `train.sh` differs, and only in orchestration. So the
code at tag `v1.0` is provably the code that trained these models.

## Re-scoring v1 under a fixed harness

```bash
mkdir -p data_cache/manifests/v1
gunzip -c runs/v1/manifests/train.jsonl.gz        > data_cache/manifests/v1/train.jsonl
gunzip -c runs/v1/manifests/val.jsonl.gz          > data_cache/manifests/v1/val.jsonl
gunzip -c runs/v1/manifests/test_ten.jsonl.gz     > data_cache/manifests/v1/test_ten.jsonl
gunzip -c runs/v1/manifests/sanity_fleurs.jsonl.gz > data_cache/manifests/v1/sanity_fleurs.jsonl
```

Then evaluate `runs/v1/crnn_v1_full/last.pt` against those manifests. Score on CPU, or with the
MPS row-chunking fix in place — otherwise you will reproduce the corrupted numbers rather than
the corrected ones.

## What cannot be preserved

The source corpora live in an external Obsidian vault (~8 GB: LibriSpeech, AMI, ESC-50, Aachen
AIR, TEN testset, FLEURS Swahili) and are not archivable here. ESC-50 is CC BY-NC and the
Aachen AIR licence is unstated, so redistribution is not an option regardless.
`data_fingerprint.json` is the compromise: it cannot restore the kit, but it can tell you
whether the kit you have is the kit v1 used.
