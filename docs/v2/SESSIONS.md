# v2 Session Log

One entry per working session. Dense, factual, append-only.

---

**2026-08-08 — v1 audited, archived, tagged; v2 planned.**
Five-agent audit of the completed v1 run found six blocking defects, the largest being a silent
MPS `nn.Conv1d` corruption above 65,536 rows that invalidated every validation metric (and
therefore checkpoint selection, and therefore the shipped ONNX exports). Independently reproduced.
Also: the training crop discarded ~72% of supervision, per-epoch augmentation never reached the
dataloader workers, the shipped hysteresis was a statistically significant regression against no
post-processing, 9.7% of the CRNN's parameters were structurally dead, and the reported RTF
figures were a measurement artifact that inverted the true latency ordering.

Four further agents designed v2. Key measured inputs: log-compressing the frontend is worth +0.027
AUROC in a controlled A/B; a teacher VAD moves training transitions/min from 11 to 45 where the gap
retune alone moves it to 14; AMI's human labels are 10.25 transitions/min against TEN's 57.5, so
they cannot serve as the conversational timescale target; the current benchmark cannot resolve
anything below ±0.03 AUROC, which is larger than the best available improvement.

Archived v1 to `runs/v1/` (7.3 MB: optimizer-stripped weights, exports, reports, manifests, logs,
curves, environment, data fingerprint). Manifests were the urgent part — the builder writes to a
flat unversioned directory and the next v2 build would have destroyed the record of v1's split.
Moved `ROADMAP.md` to `docs/v1/` verbatim with an errata banner; wrote twelve errata entries,
corrected results, conventions, v2 roadmap and gates. Tagged `v1.0`.

Not yet started: any v2 code. Phase 0 begins with the MPS row-limit fix and the config/manifest
versioning described in `CONVENTIONS.md`.
