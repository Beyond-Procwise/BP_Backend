# Extraction Pipeline — Fix Plan

> **Status:** Reconstructed 2026-07-11 from `.remember` session notes after the
> original `FIXPLAN.md` was lost with the previous session. Fixes below are the
> ones that were **proposed but not yet implemented** — the last session ended at
> the diagnosis stage. No code changes were made.
>
> See `FINDINGS.md` for the evidence behind each item.

## Fix priority order

Fixes are ordered by impact. The first two break the root-cause chain that
produces the 35% false-failure rate; the third stops real money being lost.

---

### P0 — Break the S3 ordering race (fixes F22, F28, and the deletion chain)

**Problem:** the DB row is written ~2s before the file lands in S3, so the file
looks missing and the doc is marked `failed` (and its content hash is left NULL).

**Proposed fix:**
1. **Retry-with-backoff on file reads** — before declaring a file missing, retry
   the S3 read a few times with backoff to absorb the upload gap, instead of
   failing on the first miss.
2. **Do not create the outcome row until the file is confirmed present** (or defer
   the `failed` verdict until after the retry window closes).
3. Once the file is confirmed present, **compute the content hash** so duplicate
   detection (F28) works on late uploads.

**Verifies against:** the 6 sessions / 11 documents that each failed once then
extracted at 100% — after the fix they should succeed on the first pass and no
longer be marked `failed`.

---

### P0 — Make the `failed` verdict correctable (fixes V3 / C2F4)

**Problem:** `action_status` is write-once because of the `IS NULL` guard in
`fn_try_resolve_session`; a wrongly-marked `failed` doc is locked forever.

**Proposed fix:**
1. **Recalculate the verdict on mismatch** — when a later run produces a different
   (better) outcome than the stored one, allow the status to be **corrected**
   rather than blocked by the `IS NULL` guard.
2. Add a **status-correction path** so documents that later extract at 100% are
   flipped off `failed`.

**Note:** P0-race + P0-verdict together clear finding C2F4 (100%-extracted doc
stuck at `failed`).

---

### P1 — Stop dropping implementation fees on bid reordering (fixes F25)

**Problem:** implementation / one-off fees are dropped when bid rankings are
recomputed, changing which supplier wins. £58k–£72k in the analysed batch,
£130K+ aggregate.

**Proposed fix (to be designed):** ensure implementation fees are **carried into
the ranking calculation** rather than dropped when rows are reordered. Needs a
targeted look at the bid-ranking / reorder code path before implementation.

---

### P1 — Eliminate non-deterministic extraction (fixes F19)

**Problem:** identical input scores 0.0 then 1.0 across runs (AFS V2 PDF).

**Proposed fix (to be diagnosed):** trace the source of run-to-run variance in the
extraction/scoring path. Note: prior work established Ollama at temp 0 is **not**
bit-deterministic, so the fix likely needs a deterministic scoring/gate rather
than relying on exact model reproducibility.

---

### P2 — Supporting fixes

- **F23** — add the missing `file_path` to the JOIN so progress counts stop being
  squared.
- **F24** — guard the WebSocket path against deleted sessions so it doesn't hang.
- **F21** — revisit the 15-minute promotion cron cadence to cut the 11–14m PO/
  invoice latency.
- **F27** — tighten line-item validation so junk lines (73% on sheets 1086–1088)
  don't promote.
- **Version collision** — prevent `V(1)` collisions (the 1091 → 1086 case).
- **Re-upload hang** — fix the UPDATE-trigger path that deadlocks on re-upload.

---

---

### P1 — Fixes for the re-derived F1–F18 (2026-07-11 live audit)

Grouped by theme; see `FINDINGS.md` for the evidence behind each.

**Accuracy (highest priority — wrong data in `_trgt`):**
- **F1 hallucinated evidence — ✅ DONE 2026-07-11.** Added a format-tolerant
  grounding guard in `PipelineV3` (`src/services/extraction_v3/grounding.py`) that
  demotes genuinely-ungrounded header values to review before promotion, without
  blocking correct-but-reformatted values (dates→ISO, amounts→decimal, newline
  spans). Live-validated: catches the real ~2% (e.g. `INV615597`'s fabricated
  `2024-02-15`), zero regression.
- **F1 follow-up (offline audit + doc_pk collisions) — ✅ DONE 2026-07-11.** The
  health-check audit (`scripts/extraction_health_check.py`) now uses the tolerant
  grounding and checks all snapshots per doc_pk (live: 40→17 flags, 23 false
  positives removed). New `doc_pk_collision_audit` surfaces genuine collisions
  (one id → multiple distinct docs) as events + a `doc_pk_collisions` metric;
  caught the real `TEC-Q-2022-Q3` clash. Note: the ROOT cause of that collision
  was a mis-extracted quote_id, which the F1 grounding guard now prevents going
  forward. Tests: `tests/test_extraction_health_check.py`.
- **F2 `no_stg_row` (24%)** — find why "Extracted" docs never get a `_stg` row;
  make staging failures loud (telemetry + retry) instead of a silent drop.
- **F3 zero-header quotes (47%)** — investigate the quote parser/schema path; this
  is the single worst extraction bucket.
- **F5 line-item numbers / F15 invariants** — strengthen line-item numeric
  extraction and make closure invariants (subtotal/tax/line-sum) block or
  quarantine rather than warn-and-promote.
- **F6 non-blocking promotion** — revisit which discrepancy types should set
  `blocks_promotion = true` (only 6 of 320 currently do).

**Integrity:**
- **F7 funnel leakage** — reconcile raw↔stg↔trgt counts; fix the quote `stg>raw`
  orphan case and the 22% invoice loss.
- **F8 `Deal_Linked` + `failed`** — fix the status machine so a linked doc cannot
  also be `failed` (ties into the V3 write-once work).
- **F10 stuck `missing_required`** — triage the 6 blocked docs.

**Observability (cheap, high-leverage):**
- **F12 dead health cron — ✅ DIAGNOSED + PARTIALLY FIXED 2026-07-11.** Root cause:
  fragile monotonic-only timer (`OnBootSec`+`OnUnitActiveSec`) that never re-armed
  after a missed firing (`Persistent=` is ignored for monotonic timers). Ran the
  check manually (metrics gap 48d → 0d) and rewrote the timer to
  `OnCalendar=*:0/5` + `Persistent=true`. **Remaining:** run
  `sudo deploy/systemd/install_health_timer.sh` to install the fixed unit
  (needs password sudo).
- **F13 NULL parser/version** — populate `parser_backend` and `pipeline_version`
  in telemetry so failures are attributable.
- **F14 empty retry_log** — persist retry/stuck-reset activity.

**Hygiene:**
- **F16 doc_type labels** — normalise `invoices`→`invoice`, reject empty type.
- **F17 NULL content_hash** — same fix as F28 (backfill/repair hash so dedup sees
  all uploads).
- **F18 unresolved duplicates** — action the 5 open `duplicate_document` flags.

---

## Before implementing
- ✅ **F1–F18 re-derived** (2026-07-11) — done; see `FINDINGS.md`.
- **Test the OCR path**, which this audit never exercised.
- Per project convention: **prove each fix on the running local server against
  live `bp_sqldb`**, not just tests/mocks, and keep changes behaviour-preserving
  (accuracy must not regress).
