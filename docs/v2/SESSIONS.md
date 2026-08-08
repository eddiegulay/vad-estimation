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

---

**2026-08-08 (later) — v2 reframed as a fresh build; plan de-risked before execution.**
Three agents: greenfield build design, a "no betting" audit of every claim in the plan, and a
coverage/consistency audit of the committed docs.

The greenfield design produced the carry/rewrite/drop verdict per module, the WP0–WP12 build
order with seam proofs at every trust boundary, and the load-bearing ordering constraint: the v1
probability pins do not exist yet and must be generated at the tag before the cutover.

The bet audit ran seven measurements and falsified two of the plan's premises ("dataloader-
bound" — actually model-bound, co-training 1.8×; "48 held-out speakers" — zero were held out),
corrected two more (paired-sigma ratios transposed; 8-chunk warm-up mask under-sized against a
20–30-chunk transient), and confirmed the core resolution arithmetic (paired MDE 0.019 on TEN-30
alone). It also found that every Phase 2 number, the +0.027 A/B, and the turn-cost figures were
reproducible from no artifact in the repo — now fixed by DESIGN-NOTES.md, BETS.md, the committed
measurement scripts, and the WP0.5 spikes.

The coverage audit found ten internal contradictions in the committed docs, the worst being: G2
arithmetically unsatisfiable under the 8 s crop design it protects; G10 untestable against v1
(v1 manifests record neither noise nor RIR membership — verified); the ship gates needing v1
baselines on the new benchmark that nothing scheduled; and the 9-series AMI benchmark leaking
all six training series. All fixed in the rewritten ROADMAP/GATES, which also add G11–G14 (loss
gradient, lr>0, export parity, memory budget), CI gate accounting, licensing in the ship
criteria, and the enumerated five pinned corrections for Phase 0.

Committed: rewritten ROADMAP.md and GATES.md, new DESIGN-NOTES.md / BETS.md / measurements/.
Next: WP0 (pin v1 at the tag), then the WP0.5 spikes.
