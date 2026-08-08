# Conventions

Versioning, naming and provenance rules. Introduced at v2; v1 predates them and keeps its
produced names (see `runs/v1/README.md` for the alias table).

---

## 1. What "a version" is

A project version (`v1`, `v2`) is a coherent generation of the pipeline: data construction,
labels, model, training loop, and evaluation harness together. Versions are not branches and not
source packages.

- **Code lives on `master` and evolves in place.** Each completed version is an annotated tag.
  `git checkout v1.0` is the code archive.
- **Artifacts live in `runs/v<N>/`, committed.** This is the part a tag cannot provide: training
  is stochastic and the source corpora are external, so a version's *results* are not
  re-derivable from its code. They are preserved as bytes or not at all.
- **Documentation lives in `docs/v<N>/`.** Archived documents are never edited except to add a
  pointer to their errata.

Explicitly rejected: versioned source packages (`src/vad/v1/`). v2 changes every subpackage, so
it would be a permanent wholesale duplication of a known-defective library, doubling the test
surface to keep a copy of code the audit measured as costing 0.027 AUROC. `git checkout` already
delivers the only real benefit at zero maintenance cost.

Also rejected as a *primary* mechanism: config directory namespaces (`configs/v1/`). They
advertise that an old config still reproduces an old result, which stops being true the moment
the code changes — you get a third thing nobody has evaluated. Version-stamped config
*filenames* give the coexistence that is actually needed (running a new harness against old
checkpoints) without the false promise.

---

## 2. Run IDs

```
run_id ::= v<major> "-" <arch> "-" <slug>
```

`<arch>` is the model-registry key (`crnn`, `tcn`). `<slug>` is `full` for the canonical
full-schedule run, `smoke` for plumbing runs, otherwise a hyphenated experiment name.

Examples: `v2-crnn-full`, `v2-crnn-logfe`, `v2-tcn-smoke`.

Two rules that make this mechanically safe:

- **Hyphens inside a run ID, never underscores** — so a run ID is one shell token, splits cleanly,
  and globs.
- **Filenames separate role from run ID with a dot**: `<role>.<run_id>.<qualifier>.<ext>`, so a
  filename always parses.

---

## 3. Naming by axis

| Axis | Convention |
|---|---|
| Model config | `configs/model/<arch>_v<N>.yaml` canonical, `<arch>_v<N>_<slug>.yaml` for variants. **A model config that has been trained with is immutable** — append across versions, never edit in place. |
| Pipeline configs | `configs/{train,eval,data}/v<N>.yaml`. **The name `default` is retired** — a mutable file called "default" is what made v1 unreconstructible from configs alone. Useful side effect: `git diff configs/train/v1.yaml configs/train/v2.yaml` is a live, always-accurate changelog. |
| Environment config | `configs/data/paths.yaml` stays unversioned. A machine-specific absolute path is environment, not version. |
| Live checkpoints | `checkpoints/<run_id>/` — gitignored scratch, where running jobs write |
| Archived runs | `runs/v<N>/<run_id>/` — promoted deliberately, never automatically |
| Weights | `best.pt`, `last.pt`, `epoch<NN>.pt` |
| Eval reports | `eval.<run_id>.<selection>.<manifest>.<backend>.json`. v1's `eval_report_<manifest>.json` carried no selection token, so scoring `last` silently overwrote `best`. |
| ONNX | `model.<selection>.onnx`. Prefer a single self-contained file — 1.1 MB does not need external data, and the external-data sidecar's filename is recorded *inside* the protobuf, so the pair cannot be renamed independently. |
| Manifests | `data_cache/manifests/v<N>/<split>.jsonl` plus `manifest_meta.json` |

---

## 4. Manifest identity

A path prefix is a convention, and conventions get violated silently. Manifests carry an identity
stamp instead.

`manifest_meta.json` records a `manifest_set_id` derived from the builder commit, the data config
hash, and the seed — alongside per-split counts, durations and occupancies.

Three mechanisms make a mix-up detectable rather than silent:

1. The manifest builder **refuses to overwrite an existing versioned directory** without an
   explicit force flag. v1's builder overwrote in place.
2. The `manifest_set_id` is stamped into every checkpoint and every eval report.
3. Evaluation **warns loudly** when a checkpoint's manifest set does not match the manifest it is
   being scored against.

---

## 5. Run provenance

Every run writes a `run.json` capturing: run id, version, git commit and dirty flag, every config
path with its hash, manifest set id, all seeds, device, library versions, wall time, and final
metrics.

This is what makes a future audit possible without archaeology through a 1.6 MB progress-bar log.
v1 had none, which is why its `run.json` files had to be reconstructed by hand.

---

## 6. Documentation lifecycle

v1's roadmap rotted because it mixed four documents with four different lifecycles — a phase
checklist, an append-only decisions log, a session log, and a mutable results table. From v2 they
are separate files.

**Corrections are additive and one-directional.** An archived document is moved with `git mv` so
history follows it, gains an errata banner above its first line, and is otherwise byte-identical.
Every wrong claim gets an errata entry quoting the original. Nothing is silently rewritten; the
wrongness is visible from the top of the file.
