# Extraction Pipeline Audit — Findings

> **Status:** Reconstructed 2026-07-11 from `.remember` session notes after the
> original `FINDINGS.md` was lost when the previous session (5.5-hour live audit,
> session `afd83873`) was shut down abruptly.
>
> **Scope of this rebuild:** All 7 critical findings and the later findings
> (F19–F28, V3) survived in the memory notes and are reproduced verbatim below.
> The earlier items **F1–F18 were lost with the original document** (only the
> running count 18 → 24 → 29 survived). They have now been **re-derived on
> 2026-07-11 by a fresh live audit** — see the "F1–F18 (re-derived)" section.
> These are an *equivalent, evidence-backed* set, not guaranteed to be the exact
> originals; each is grounded in a live query against `bp_sqldb` (run stamps below).
>
> **Audit method:** live, read-only queries against `bp_sqldb` (RDS
> `procwisemvpdb01`), 231 distinct extracted documents, across the telemetry,
> discrepancy, hallucination-audit, provenance-v3, process-monitor and
> raw→stg→trgt tables. OCR path was **not** tested.

## Summary

- **29 findings total, 7 critical.**
- Most critical failures share **one root cause chain**:
  **S3 upload timing gap → DB row created before the file lands → file looks
  missing → document marked `failed`/deleted → orphaned rows → status locked
  irreversibly.**
- Highest business impact: **£130K+ of implementation fees dropped in aggregate**
  (£58k–£72k in a single analysed batch) because bid rankings get reordered.

---

## Critical findings (7)

### F22 — S3 ordering race → 35% false-failure rate  🔴
- The pipeline writes the document row **~2 seconds before** the file finishes
  uploading to S3.
- During that gap the file looks missing, so the doc is marked **`failed`**.
- **Measured impact: 35% false-failure rate** — 6 sessions / 11 documents, each of
  which **failed once then extracted at 100% on retry**. The documents were never
  actually broken.
- Interacts with V3: the false `failed` mark is then **irreversible**.

### V3 — `failed` status is write-once / irreversible  🔴
- `action_status` is guarded by an `IS NULL` check in `fn_try_resolve_session`,
  making it **write-once**.
- Once a document is (wrongly) marked `failed`, the verdict **can never be
  corrected**, even when a later run extracts it perfectly.
- Manifestation **C2F4**: a document with 100% successful extraction stays marked
  `failed` forever.

### F28 — Content-hash NULL → silent duplicate-detection failure  🔴
- The same S3 race leaves the **content hash NULL** on late uploads.
- Duplicate detection keys on that hash, so **duplicates silently slip through**
  undetected.

### F25 — £58k–£72k implementation fees dropped via bid reordering  🔴
- When bid rankings are recomputed, **implementation / one-off fees are dropped**,
  shifting the ranking.
- **£58k–£72k lost in the analysed batch (sheets 1086–1088); £130K+ in aggregate.**
- Assessed as **the single most damaging finding** — it changes which supplier
  "wins" a deal.

### F19 — Non-deterministic extraction  🔴
- The **same AFS V2 PDF, identical input, scored 0.0 on one run and 1.0 on the
  next.** Proven, reproducible non-determinism.
- Makes the confidence score untrustworthy and every downstream gate unreliable.
- (F26 was originally logged as a separate explanation for this behaviour but was
  **retracted** once F19 non-determinism was confirmed as the real cause.)

### Re-upload hang (UPDATE-trigger deadlock)  🔴
- Re-uploading a document **hangs** via the row `UPDATE` trigger path.

### S3 gap → deletion → orphaned rows → permanent status lock  🔴
- The end-to-end causal chain: S3 gap → the file is treated as missing and
  **deleted** → **orphaned rows** in `session_document_outcome` (FK orphaning,
  see F20) → session **status locked permanently** (via V3).
- This chain is the umbrella cause tying F22, V3, F28 and the deletion behaviour
  together.

---

## Other findings (non-critical, recovered)

### F20 — `session_document_outcome` FK orphaning
- Foreign-key rows in `session_document_outcome` get orphaned when the parent is
  removed, feeding the permanent-status-lock chain above.

### F21 — 11–14 minute PO/invoice promotion latency
- POs and invoices take **11–14 minutes** to promote because the promotion cron
  runs on a **15-minute** cycle.

### F23 — Missing `file_path` JOIN → squared progress counts
- A missing `file_path` in a JOIN causes **progress counts to be squared**
  (Cartesian blow-up), so the UI reports wrong document totals.

### F24 — WebSocket hangs on deleted sessions
- The WS connection **hangs** when a session has been deleted underneath it.

### F27 — 73% junk line-items
- On sheets 1086–1088, **73% of extracted line-items were junk** (garbage lines
  promoted through the pipeline).

### Version-collision risk (V(1) collision)
- Document versioning can **collide** — hypothesis proven with the **1091 → 1086
  duplicate** case; a `V(1)` collision can overwrite/confuse distinct documents.

---

## F1–F18 (re-derived 2026-07-11 from the live pipeline)

Re-derived by a fresh read-only audit of `bp_sqldb` over **231 distinct extracted
documents** (telemetry de-duplicated to the latest run per document). Every finding
carries the live number behind it. These replace the lost originals as an
equivalent, evidence-backed set. Where one corroborates an existing F19–F28, it is
noted.

### Extraction accuracy (data that reaches `_trgt` can be wrong)

**F1 — Ungrounded (hallucinated) values could promote to the final tables  🔴 — FIXED 2026-07-11**
The `bp_extraction_hallucination_audit` flags fields whose `evidence_text` is not
found in the document. Live investigation (797 recent header fields, snapshots
intact) refined the picture:
- The audit's **byte-exact** test **over-reports**: ~96–97% of "hallucinations"
  are correct values whose evidence was merely reformatted (a newline in the
  span, dates normalized to ISO, amounts to plain decimal).
- The **genuine** rate is **~2%**, and it is real: e.g. invoice `INV615597`
  extracted `invoice_date = 2024-02-15` when the document contains **no Feb-2024
  date at all** (its only date is `2026-07-01`). Such fabricated values were
  promoting at conf 0.70–0.72, above the floor.

**Fix:** a **format-tolerant grounding guard** now runs in `PipelineV3` before
promotion (`src/services/extraction_v3/grounding.py`). A committed header field is
demoted to a residual (`reason="ungrounded_value"` → NULL / manual review) only
when its value cannot be located in the document by *any* tolerant strategy
(whitespace/case-normalized substring, date-aware cross-rendering, digit
signature); deterministic `pipeline_recovery` derivations and synthetic ids are
exempt. Validated on live data: blocks the genuine ~2% (real fabrications) while
preserving every correctly-extracted-but-reformatted field — **no accuracy
regression**. Tests: `tests/extraction_v3/test_grounding.py`,
`tests/extraction_v3/test_pipeline_grounding_guard.py`.

*Follow-up — DONE 2026-07-11:* the offline audit
(`scripts/extraction_health_check.py`) now uses the same tolerant grounding and
checks **all** snapshots sharing a doc_pk, so it stops over-reporting and stops
false-flagging on collisions. Live: byte-exact flagged 40/760 (5.3%); tolerant
flags 17 (2.2%) — 23 false positives removed. A new `doc_pk_collision_audit`
surfaces genuine collisions (one doc_pk → multiple distinct documents) as
`doc_pk_collision` events + a `doc_pk_collisions` health metric; it caught the
real `TEC-Q-2022-Q3` case (the file `TEC-QTR-2022-Q3` had its quote_id
mis-extracted, dropping "TR"). Tests: `tests/test_extraction_health_check.py`.

**F2 — 24% of documents extract but never stage (`no_stg_row`)**
55 of 231 distinct docs have `completeness_status = no_stg_row`: extraction
reports success (`status = Extracted`) but **no `_stg` row is ever written**, so
the document silently drops out of the pipeline before promotion.

**F3 — 47% of quotes extract zero header fields**
Zero-header-field rate (latest run per doc): **quote 45/96 (47%)**, invoice 13/74
(18%), PO 12/52 (23%), contract 4/4, `invoices` 4/4. Zero-line-item counts track
almost identically (quote 45, invoice 14, PO 14). Quote header extraction fails
nearly half the time.

**F4 — 35% of documents carry no confidence score**
80 of 231 distinct docs have `confidence = NULL` in telemetry — no score was
recorded, so every confidence-gated promotion/HITL decision on them is blind.

**F5 — 215 line-items missing their numbers; 16 docs fail sum reconciliation**
Open discrepancies: `line_missing_numbers` ×215, `line_sum_mismatch` on 16 docs,
`tax_percent_mismatch` ×6, `line_total_mismatch` ×2, `missing_line_items` ×21.
Line-item numeric extraction is the weakest area. *Corroborates F27 (junk lines).*

**F6 — Known-bad data promotes anyway (314/320 discrepancies non-blocking)**
Of 320 open discrepancies only **6** carry `blocks_promotion = true`; the other
314 are `warning/open` and do **not** stop promotion, so documents with flagged
defects still reach `_trgt`. This is by-design (per the Line-Items-Gap decision)
but is the mechanism by which F1/F3/F5 defects land in final tables.

### Pipeline & persistence integrity

**F7 — raw→stg→trgt funnel leaks (and a negative-leak anomaly)**
Invoice funnel **18 raw → 16 stg → 14 trgt** (22% lost before final). PO
**15 → 15 → 14** (1 lost). Quote **39 raw → 41 stg → 41 trgt** — *more staged rows
than raw parents*, i.e. orphaned/duplicate `_stg` rows with no surviving raw
record. Both directions signal referential drift across tiers.

**F8 — 10 documents are `Deal_Linked` yet `action_status = failed`**
process_monitor rows 1096–1099, 1118–1119, 1146–1149 (`Tec Deal` / `deal_tec_new`)
are simultaneously **linked to a deal and marked failed** — a contradictory state.
`content_hash` is present, so this is a status-machine bug, not the S3-hash race.
*Related to V3 write-once status handling.*

**F9 — 20 documents (8.7%) end in `Extraction_Failed`; 24 have no completeness verdict**
Latest-run status: 211 `Extracted`, **20 `Extraction_Failed`**; separately 24 docs
have `completeness_status = NULL` (no verdict recorded at all).

**F10 — 6 documents stuck behind critical `missing_required` blocks**
6 `missing_required` discrepancies are `critical/open` with
`blocks_promotion = true` — permanently held out of `_trgt` until resolved.

**F11 — Non-determinism still live (confirms F19)**
6 distinct documents show a >5-point confidence spread across re-runs; worst is
PO `526689` swinging **50 → 95 over 7 runs**. Same input, different verdict.
*Directly corroborates F19; still reproducing on 2026-07-11.*

### Observability & data hygiene

**F12 — Health-metrics monitoring is dead (48 days stale)  — DIAGNOSED + PARTIALLY FIXED 2026-07-11**
`bp_extraction_health_metrics` last recorded **2026-05-24** — 48 days — while
telemetry (today) and provenance (yesterday) were current. The health/reaper
`bp-extraction-health.timer` runs the check every 5 min.

**Root cause:** the timer used only **monotonic** triggers
(`OnBootSec=30s` + `OnUnitActiveSec=5min`). That chain re-arms only after a
successful activation, and `Persistent=true` has **no effect** on monotonic
timers — so one missed/failed firing left it with `NextElapse=infinity`
("active (elapsed)", `LastTrigger=2026-05-24`) and it never recovered.

**Fixed:**
- *Immediate:* ran the check manually — fresh metrics row written
  (`days_stale` 48 → 0), the tolerant audit + new collision detector both ran.
- *Durable (needs one sudo step):* rewrote
  `deploy/systemd/bp-extraction-health.timer` to `OnCalendar=*:0/5` +
  `Persistent=true` (self-healing wall-clock schedule). Install with
  `sudo deploy/systemd/install_health_timer.sh` — the root-owned unit under
  `/etc/systemd/system` can't be edited without password sudo.

**F13 — Telemetry records no parser or pipeline version (100% NULL)**
`parser_backend` and `pipeline_version` are NULL on **all 535** telemetry rows, so
failures cannot be attributed to a parser backend or a pipeline release — root-cause
analysis is hobbled.

**F14 — `retry_log` is empty despite retries happening**
`proc.bp_extraction_retry_log` has **0 rows** even though the pipeline demonstrably
re-runs documents (F11). Retry/stuck-reset activity is not being persisted.

**F15 — 64 invariant checks failing (arithmetic/consistency)**
`invariant_failed` ×64 open, spread evenly (8 docs each) across `subtotal_closure`,
`tax_closure`, `line_sum_closure`, `currency_consistency`, `date_sanity` — totals
and taxes don't close and dates fail sanity on those documents.

### Labeling & dedup hygiene

**F16 — Inconsistent `doc_type` labels split the same type**
Telemetry carries `invoice` (136) **and** `invoices` (6), plus an empty-string
type (1). The `invoices` and `''` rows are also 100% zero-header — they are
mis-labeled/mis-routed and never extract. Splits metrics and breaks type-keyed logic.

**F17 — 14/75 process_monitor rows have NULL `content_hash` (18.7%)**
Dedup keys on `content_hash`; 14 of 75 upload rows have none, so those documents
are invisible to duplicate detection. *Same failure surface as F28 (S3-race hash),
quantified on current data.*

**F18 — 5 duplicate documents detected but left unresolved**
`duplicate_document` discrepancies ×5 are `warning/open` — flagged as duplicates
but never actioned, so duplicate rows persist alongside their originals.

---

## Reproducing the re-derivation
The audit queries are saved (read-only) in the repo at
`scripts/rederive_findings_audit_1.py` and `scripts/rederive_findings_audit_2.py`.
Re-run against `bp_sqldb` via `src.services.db.get_conn()` to refresh the numbers
above.

## Not yet tested
- **OCR extraction path** — never exercised in this audit.
