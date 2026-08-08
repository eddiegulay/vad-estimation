# Session measurements (2026-08-08)

Scripts behind the numbers in `../DESIGN-NOTES.md` and the settled items in `../BETS.md`,
preserved verbatim from the verification session so the evidence is reproducible, not folklore.
They assume the repo venv (`.venv/bin/python`), `runs/v1/` artifacts, and the local data cache.

| file | backs |
|---|---|
| `m1_paired_sigma.py`, `m1_per_file.npz` | paired vs absolute per-cluster sigma on TEN; MDE arithmetic (DESIGN-NOTES §1) |
| `m2_manifest_stats.py` | TEN/train timescale stats; macro-vs-pooled convention (§2) |
| `m4_clusters.py` | what actually exists on disk per corpus; the zero-held-out-speakers finding (§5) |
| `m5_warmup.py` | cold-start transient curve; warm-up mask sizing (§4) |
| `m7_bound.py` | dataloader-vs-model bound; co-training cost (§3) |

The WP0.5 spikes (teacher protocol, A/B harness, A/A noise floor, cost-ratio sweep) will add
their scripts and cached outputs here or under `scripts/` as they run.
