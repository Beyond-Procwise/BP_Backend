# Extraction Renovation — Design

**Date:** 2026-05-16
**Author:** Muthu (direction); Claude (drafting)
**Status:** Draft, awaiting spec review
**Scope:** Procurement extraction module only — invoice, purchase_order, quote, contract
**Approach:** Renovation in place. One pipeline. No parallel build, no waves.

## 1. Problem

The live extraction path is `extraction_v3/extraction_v4/engine.py` — a 7,935-line monolith driven by hardcoded regex blocks, with the multi-model ML stack proposed in the 2026-05-06 spec sitting dormant beside it. The judge orchestrator runs only on contracts and on a fallback Qwen-VL path. `_raw` tables capture extracted data as a single JSONB blob, making field-level inspection and human correction awkward. There is no closed feedback loop and no production accuracy measurement.

This renovation collapses the pipeline to a single flow whose shape matches the actual problem: a narrow procurement vocabulary, well suited to regex as the primary extractor, with engineered fallbacks and an AI judge that confirms accuracy rather than producing values.

The earlier 2026-05-06 redesign spec is superseded by this document for the implementation. Its useful infrastructure (provenance, invariants, judge contracts, schemas) is retained and wired in; its multi-model ML extractor stack (LayoutLMv3 fine-tune, Table Transformer, sBERT, QA-RoBERTa as primary extractors) is dropped.

## 2. Locked direction

These are the user-confirmed requirements. They are not open for re-litigation in spec review.

1. **One flow, one entry point.** `dispatch_document()` collapses to a single pipeline; no engine switch.
2. **Tiers:** L0 parse → L1 regex (primary) → L2 engineered (secondary, only when L1 misses or confidence is low) → L3 AI judge (substring grounding, invariants, coherence, discrepancy detection).
3. **`_raw` tables become flat columns.** One row per document; one column per extracted field. The single `raw_payload JSONB` column is replaced. Line items go to a separate `bp_<doctype>_line_items_raw` flat-column table.
4. **`_stg` is the validated, computed, judge-confirmed snapshot.** Promotion happens only when every required field has evidence in source, invariants pass, supplier resolves, and the judge returns coherent.
5. **Discrepancy table holds one row per failing field** with enough detail for a human to fix it.
6. **HITL fixes discrepancies in place.** When all blocking discrepancies for a `_raw` row are resolved, that row promotes to `_stg` automatically (DB trigger).
7. **Per-decision lock-ins:** `_raw` keeps one `parser_snapshot JSONB` column (the L0 parser output — full text, tokens, bboxes, table cells) for re-grounding and re-extraction; no field-level JSONB. The HITL UI/API is **not** in this build's scope — this build owns the discrepancy schema, the re-promotion trigger, and a programmatic write contract for HITL clients. Re-promotion is triggered by a DB trigger on discrepancy status change.

## 3. Architecture

```
process_monitor (status=Completed) ──notify──▶ process_monitor_watcher
                                                       │
                                                       ▼
                                            dispatch.dispatch_document()
                                                       │  (single path; no engine switch)
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  L0  Parse        per file format → ParsedDocument                       │
│                   PDF native      → PyMuPDF blocks                       │
│                   DOCX            → python-docx                          │
│                   Excel           → openpyxl                             │
│                   Image / scanned → PaddleOCR PP-Structure               │
│                   Output: text + tokens(bbox) + tables(cells) + pages    │
└──────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  L1  REGEX (PRIMARY)  PatternRegistry, YAML-driven, per doc-type/field   │
│                       each rule: label anchor + value regex + format hint│
│                       emits Candidate(field, value, page, bbox, span)    │
│                       always carries source span                         │
└──────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  L2  ENGINEERED (SECONDARY)  fires only on L1 miss or conf < threshold:  │
│                       • table_extractor   (line items in tables)         │
│                       • ner_validator     (spaCy ORG/PERSON/GPE typing)  │
│                       • address_parser    (multi-line block reassembly)  │
│                       • date_normaliser   (ISO conversion)               │
│                       • bbox_proximity    (resolve label↔value)          │
│                       emits Candidate same shape as L1                   │
└──────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  L3  AI JUDGE (FINAL)  per-doc, structured calls only — never extracts:  │
│                       • substring_grounding   (value ⊂ ParsedDoc.full_text)
│                       • invariants_runner     (11 existing invariants)   │
│                       • schema_coherence      (LLM cross-field pass)     │
│                       • discrepancy_detector  (emit per-field issues)    │
│                       Verdict per field: commit | demote | flag          │
└──────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                       persist_raw (flat columns, per-field provenance)
                                                       │
                          ┌────────────────────────────┴───────────────────┐
                          ▼                                                ▼
                  clean: promote to _stg                  any blocking discrepancy:
                  delete _raw row                         status='discrepancy'; rows
                                                          written to bp_extraction_discrepancy
                                                                            │
                                                                            ▼
                                                              HITL resolves rows
                                                              status='resolved' (DB trigger)
                                                              re-promotion → _stg
```

Every candidate carries `(value, page, bbox, source_span)` from L0 onwards. The substring grounding gate at L3 is the structural anti-hallucination guarantee — applied to every candidate regardless of which tier produced it.

## 4. Data model changes

### 4.1 `_raw` tables: flat columns

Replace the four `proc.bp_<doctype>_raw` tables. Each new shape: control columns + one column per extracted field, mirroring the `_stg` columns of that doc type, plus a `parser_snapshot JSONB` audit column. No field-level JSONB.

The exact field set for each doc type is the union of `db_column` entries in `extraction_schemas/<doctype>.yaml`. The fields are already declared there; the migration script reads the YAML and emits the DDL so they cannot drift.

Common control columns on every `_raw` table:

| Column | Type | Notes |
|---|---|---|
| `raw_id` | `BIGSERIAL PRIMARY KEY` | |
| `doc_pk_candidate` | `TEXT` | best-effort PK; nullable until resolved |
| `source_file` | `TEXT NOT NULL` | |
| `process_monitor_id` | `INT` | FK to `proc.process_monitor.id` |
| `pipeline_version` | `TEXT NOT NULL` | git short SHA at extraction time |
| `extracted_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | |
| `parser_snapshot` | `JSONB NOT NULL` | full L0 ParsedDocument (text + tokens + tables + bboxes) |
| `promotion_status` | `TEXT NOT NULL DEFAULT 'pending'` | enum: `pending | promoted | discrepancy | failed` |
| `promoted_at` | `TIMESTAMPTZ` | set when row leaves _raw |

Field columns follow the YAML. Each field column is **nullable** because incomplete extractions still land in `_raw` (they enter `discrepancy` status, not `failed`). Failures are reserved for L0 parse errors and unrecoverable infrastructure issues.

For each doc type also create `proc.bp_<doctype>_line_items_raw`:

| Column | Type | Notes |
|---|---|---|
| `line_raw_id` | `BIGSERIAL PRIMARY KEY` | |
| `raw_id` | `BIGINT NOT NULL REFERENCES proc.bp_<doctype>_raw(raw_id) ON DELETE CASCADE` | |
| `line_index` | `INT NOT NULL` | order within the document |
| `item_description` | `TEXT` | |
| `quantity` | `NUMERIC(18, 4)` | |
| `unit_price` | `NUMERIC(18, 4)` | |
| `line_amount` | `NUMERIC(18, 2)` | |
| `tax_percent` | `NUMERIC(6, 4)` | |
| `tax_amount` | `NUMERIC(18, 2)` | |
| `total_amount_incl_tax` | `NUMERIC(18, 2)` | |

Migrating existing `_raw` data: in scope. The migration script reads existing rows whose `raw_payload JSONB` is well-formed, flattens header fields into columns, splits line items into the line-items raw table, then drops `raw_payload`. Rows that fail to flatten are dropped (they would have been unusable anyway — `_raw` is a transient landing zone, not historical archive).

### 4.2 Discrepancy table extensions

Existing `proc.bp_extraction_discrepancy` is kept; columns added for HITL fix-and-promote:

| New column | Type | Notes |
|---|---|---|
| `resolved_value` | `TEXT` | the value HITL chose to commit (may equal `raw_value` if "accept as-is") |
| `resolution_action` | `TEXT` | enum: `apply_value | keep_null | dismiss` |
| `resolved_by` | `TEXT` | HITL identity supplied by the client |
| `blocks_promotion` | `BOOLEAN NOT NULL DEFAULT TRUE` | severity=critical → TRUE; warning/info → FALSE |
| `evidence_page` | `INT` | source page (nullable) |
| `evidence_bbox` | `REAL[]` | `[x0, y0, x1, y1]` (nullable) |
| `evidence_text` | `TEXT` | source substring (nullable) |

`status` enum extended to: `open | resolved | ignored | superseded`.

### 4.3 Re-promotion DB trigger

```sql
CREATE OR REPLACE FUNCTION proc.fn_extraction_discrepancy_resolved()
RETURNS TRIGGER AS $$
DECLARE
    open_blocking_count INT;
    target_raw_table   TEXT;
BEGIN
    IF NEW.status = 'resolved' AND (OLD.status IS DISTINCT FROM NEW.status) THEN
        -- count remaining blocking-open discrepancies for the same raw_id
        SELECT COUNT(*) INTO open_blocking_count
          FROM proc.bp_extraction_discrepancy
         WHERE raw_id = NEW.raw_id
           AND blocks_promotion = TRUE
           AND status = 'open';

        IF open_blocking_count = 0 THEN
            -- ask the extraction service to promote this raw_id
            PERFORM pg_notify(
                'extraction_raw_ready_for_promotion',
                json_build_object('doc_type', NEW.doc_type, 'raw_id', NEW.raw_id)::text
            );
        END IF;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_extraction_discrepancy_resolved
    ON proc.bp_extraction_discrepancy;
CREATE TRIGGER trg_extraction_discrepancy_resolved
AFTER UPDATE ON proc.bp_extraction_discrepancy
FOR EACH ROW EXECUTE FUNCTION proc.fn_extraction_discrepancy_resolved();
```

The trigger notifies on a dedicated channel rather than running the promotion in-band. The extraction service listens on `extraction_raw_ready_for_promotion` and runs the promotion path (apply `resolved_value` updates to the `_raw` row, re-run invariants, promote to `_stg`, write provenance). A trigger-side promotion would couple business logic to PL/pgSQL and bypass the judge, which we do not want.

### 4.4 What stays unchanged

- `proc.bp_extraction_provenance_v3` — keep. Atomic-with-promote write of per-field bbox + evidence_text is the audit substrate.
- `proc.bp_<doctype>_stg` and `_line_items_stg` — keep, unchanged. Renovation does not touch the curated staging shape.
- `proc.process_monitor` and its NOTIFY trigger — keep.
- `proc.bp_extraction_template`, `bp_extraction_patterns`, `bp_extraction_type_priors` — keep; renovation wires them up but does not redesign them.
- Legacy `proc.bp_extraction_provenance` and `proc.extraction_review_queue` — keep.

## 5. Module structure (target tree)

The renovation introduces one new package and refactors the engine. Everything else stays where it is.

```
src/services/extraction/                ← NEW package (target)
    __init__.py
    dispatch.py                         ← single entry point; replaces extraction_v3/dispatch.py
    parser.py                           ← L0 — calls existing parsers/* backends
    pattern_registry.py                 ← L1 — loads PatternRegistry from YAML
    pattern_extractor.py                ← L1 — runs the registry over ParsedDocument
    engineered/
        __init__.py
        table_extractor.py              ← L2
        ner_validator.py                ← L2 — wraps existing extractors/spacy_ner.py
        address_parser.py               ← L2
        date_normaliser.py              ← L2
        bbox_proximity.py               ← L2
    judge_runner.py                     ← L3 — calls existing judge/* modules
    grounding.py                        ← L3 — substring grounding gate
    invariants.py                       ← L3 — wraps existing binding/invariants_runner.py
    persistence.py                      ← writes _raw flat columns + provenance
    promotion.py                        ← runs _raw → _stg promotion + listens on notify
    types.py                            ← Candidate, ParsedDocument shared with schemas/

extraction_schemas/*.yaml               ← extended: each field gains `patterns: [...]`

scripts/migrations/
    2026-05-16-extraction-raw-flat-columns.sql      ← NEW (DDL + data flattening)
    2026-05-16-extraction-discrepancy-hitl.sql      ← NEW (discrepancy column adds + trigger)
```

### Code that goes away after cutover

- `src/services/extraction_v3/extraction_v4/engine.py` — fully decomposed; its regex bodies migrate to YAML `patterns:` and its mapping logic migrates to `pattern_extractor.py` + `persistence.py`. Deleted at the end of renovation.
- `src/services/extraction_v3/extraction_v4/adapter.py` — replaced by `extraction/types.py`. Deleted.
- `src/services/extraction_v3/extraction_v4/llm_extractor.py` — replaced by `judge_runner.py` (the LLM stops filling values; it only judges). Deleted.
- `src/services/extraction_v3/pipeline.py` (the old V3 PipelineV3 path) — single-flow renders it redundant. Deleted.
- `src/services/extraction_v3/extractors/layoutlmv3.py`, `layoutlmv3_finetuned.py`, `table_transformer.py`, `qa_roberta.py`, `sbert_anchor.py`, `vendor_template.py` — out of scope, never wired in production. Deleted.
- `src/services/extraction_v3/extractors/vlm.py` — Qwen-VL path. Out of scope for this build (LLM is judge only). Deleted after the single flow is verified on all four doc types.

### Code that is kept and wired in

- `src/services/extraction_v3/parsers/{router,docling_backend,paddleocr_backend,donut_backend,scanned_classifier}.py` — used by `extraction/parser.py`.
- `src/services/extraction_v3/extractors/spacy_ner.py` — used by `extraction/engineered/ner_validator.py` (as a type-validator only).
- `src/services/extraction_v3/judge/{orchestrator,tiebreaker,grounded_last_resort,schema_coherence,contracts}.py` — called by `extraction/judge_runner.py`.
- `src/services/extraction_v3/binding/{type_binder,invariants_runner,scale_mismatch}.py` — called by `extraction/invariants.py`.
- `src/services/extraction_v3/supplier_resolver.py` — called by `extraction/persistence.py` before promotion.
- `src/services/extraction_observer/observer.py` — wired as a systemd unit during this build; not redesigned.

## 6. End-to-end flow for one document

1. `process_monitor` row updates to `status='Completed'`. Trigger fires `pg_notify('process_monitor_ready', id)`.
2. `process_monitor_watcher` receives the notify, reads the row, and calls `extraction.dispatch.dispatch_document(process_monitor_id, file_path, doc_type)`.
3. **L0 Parse.** `parser.parse_document(file_path)` returns a `ParsedDocument` (full_text, pages, tokens with bboxes, tables, parser_backend, parser_confidence). On parse error: write a single `_raw` row with `promotion_status='failed'`, attach the error to the discrepancy table with `issue_type='parse_failed'`, return.
4. **L1 Regex.** For every field in the YAML for this doc type, `pattern_extractor.run(parsed_doc, registry, doc_type)` walks ordered patterns. Each pattern produces zero or one `Candidate(field, value, page, bbox, span, confidence)`. Default confidence is the pattern's declared prior (typically 0.65–0.85).
5. **L2 Engineered (gated).** For each field where L1 yielded no candidate or `confidence < threshold` (per-field, declared in YAML; default 0.70), the relevant secondary extractor runs:
   - missing `supplier_name` / `buyer_id` / `requested_by` with NER type check declared → `ner_validator` runs.
   - missing line items → `table_extractor` runs across `ParsedDocument.tables`.
   - missing address fields → `address_parser` runs against the address-block region.
   - any date field present but unparseable → `date_normaliser` runs.
   - L1 emitted >1 candidate for the same field → `bbox_proximity` picks the candidate closest to the field's declared label anchors before judge tiebreaker is needed.
6. **L3 Judge.**
   - **Grounding gate.** Every candidate's `evidence_span` is asserted to be a substring of `ParsedDocument.full_text`. Non-substring candidates are dropped, not "fixed."
   - **Tiebreaker.** For fields with ≥2 surviving candidates that disagree on normalized value, call `judge/tiebreaker.py`. Output validated to be an index into the candidate list or null.
   - **Grounded last resort.** For required fields with zero surviving candidates, call `judge/grounded_last_resort.py`. The judge's `value` and `evidence_text` are post-validated to satisfy `value == evidence_text` AND `evidence_text in ParsedDocument.full_text`. Failure → field stays NULL, discrepancy emitted.
   - **Type bind.** `type_binder` coerces each provisional value to its declared type (`iso_date`, `money`, `decimal`, etc.). Failure → discrepancy `type_bind_error`.
   - **Invariants.** All 11 invariants run on the assembled record. CRITICAL failure → fields involved in the failure go to discrepancy; record proceeds to `_raw` with `promotion_status='discrepancy'`. WARNING failure → demotes confidence but does not block.
   - **Schema coherence.** One LLM call on the full assembled record returns `coherent | incoherent`. Incoherent → emit a discrepancy with `issue_type='judge_incoherent'`; the record proceeds to `_raw` with `promotion_status='discrepancy'`.
7. **Persist `_raw`.** Insert one row into `proc.bp_<doctype>_raw` with flat columns. Insert N rows into `proc.bp_<doctype>_line_items_raw`. Write `parser_snapshot` JSONB. Write `proc.bp_extraction_provenance_v3` rows for every field that has a value (header + line). All four writes occur in a single transaction.
8. **Decision: promote or hold.**
   - No blocking discrepancies were emitted in step 6 → call `promotion.promote(raw_id, doc_type)` immediately. Copies _raw columns to `_stg`, deletes the `_raw` row, updates `promoted_at`, runs supplier resolution, updates `process_monitor.status='Completed'`.
   - Otherwise → set `promotion_status='discrepancy'`, leave the `_raw` row in place. HITL is now responsible.
9. **HITL fix loop.** External clients (out of scope for this build) write to `proc.bp_extraction_discrepancy`: set `resolved_value`, `resolution_action`, `resolved_by`, `status='resolved'`. The DB trigger (§4.3) fires `pg_notify('extraction_raw_ready_for_promotion', ...)` when the last blocking discrepancy on a `raw_id` resolves.
10. **Re-promotion listener.** `extraction.promotion` listens on that channel. On notify: it applies every `resolved_value` to the corresponding `_raw` column (or sets NULL for `keep_null`), re-runs invariants and grounding against the updated row, then promotes to `_stg`. If invariants now fail against the human-supplied values, new discrepancies are emitted and the row stays in `discrepancy`. The trigger fires again when those resolve. This continues until promotion succeeds or HITL marks the row `dismiss`-all (an admin action — out of scope for this build's UI).

## 7. Discrepancy → HITL contract

This is the public contract between the extraction service and any HITL client. The schema is in §4.2; the semantics:

- **Read.** Clients list open discrepancies: `SELECT * FROM proc.bp_extraction_discrepancy WHERE status = 'open' AND doc_type = ? ORDER BY created_at`.
- **Inspect.** For each discrepancy, the `_raw` row (`raw_id`) carries the parser_snapshot and the other fields' extracted values. Provenance for the same `doc_pk_candidate` is in `bp_extraction_provenance_v3`.
- **Fix.** Update the discrepancy row with `resolved_value`, `resolution_action`, `resolved_by`, `status='resolved'`. The trigger handles the rest.

Resolution actions:

| `resolution_action` | Meaning |
|---|---|
| `apply_value` | Take `resolved_value`, write it to the `_raw` row's field column, re-validate. |
| `keep_null` | Accept that the field is genuinely absent; mark NULL on the `_raw` row, re-validate. |
| `dismiss` | This discrepancy is wrong / noise; do not change the `_raw` row. (`blocks_promotion` is treated as FALSE on this row for the purposes of the trigger.) |

The HITL UI/API and access control are out of scope for this build.

## 8. Pattern registry — YAML extension

Each field in `extraction_schemas/<doctype>.yaml` is extended with a `patterns:` list ordered by descending confidence. Existing keys (`canonical_labels`, `judge`, `invariants`, `db_column`) are untouched. Example for `invoice.invoice_id`:

```yaml
  - name: invoice_id
    type: string
    required: true
    db_column: invoice_id
    canonical_labels: [...]                    # already present, kept
    confidence_threshold: 0.70                 # NEW — gates L2 fallback
    patterns:                                  # NEW
      - name: anchored_inv_no
        anchor:  "(?:Invoice\\s*(?:Number|No\\.?|#)\\b)"
        value:   "([A-Z][A-Z0-9\\-/]{3,32})"
        max_span_after_anchor_chars: 60
        prior_confidence: 0.85
      - name: bareword_inv
        anchor:  "(?:INV)"
        value:   "([A-Z]{2,4}[0-9]{4,10})"
        prior_confidence: 0.72
    judge:                                     # already present, kept
      tiebreaker: true
      grounded_last_resort: true
      ner_type_check: "none"
    invariants: []
```

`extractors:` is removed from each field — the registry is now the authoritative list, not a per-field model-name array. Schemas are still verified at startup against the runtime registry and the DB columns.

The pattern registry lives in code at `extraction/pattern_registry.py`; the YAML is its declarative source. A change to a regex is a YAML edit, not a Python edit. The 7,935-line engine.py's regex bodies migrate here as part of the renovation, deduplicated.

## 9. Out of scope

- **HITL UI / API.** This build owns the discrepancy table contract and the re-promotion trigger. The interactive interface is a separate workstream.
- **Fine-tuning corpus accrual.** Out of scope. The judge is treated as a black-box local Ollama model; no training loop in this renovation.
- **LayoutLMv3 / Table Transformer / QA-RoBERTa / sBERT / vendor_template ML.** Dropped.
- **Pattern auto-learning.** `bp_extraction_patterns` is kept structurally but the renovation does not populate it from successful extractions. Future work.
- **Email-body extraction.** Out of scope. Email attachments only.
- **Email/Excel-only pipeline changes.** Excel parsing is kept via `parser.py` (openpyxl backend) but no Excel-specific extractor branch.
- **Confidence calibration / pattern priors auto-tuning.** Priors come from YAML; tuning is by hand.
- **Active backfill of historical `_raw` JSONB rows.** The migration flattens existing rows where possible; rows that can't be flattened are dropped (they were unpromotable anyway).
- **Multi-language support.** English-only.

## 10. Success criteria

The renovation is done when **all** of these hold:

1. **Single flow.** `dispatch_document` has one execution path. No `EXTRACTION_V3_ENGINE` env switch. `extraction_v4/engine.py`, `adapter.py`, `llm_extractor.py`, `pipeline.py`, and the unused `extractors/*` ML modules listed in §5 are deleted.
2. **Flat-column `_raw`.** Every `_raw` table has flat columns matching its `_stg` table; no field-level JSONB. `parser_snapshot JSONB` is the only JSONB column.
3. **Grounding holds.** A scripted audit over the last 100 promoted rows shows every committed value's `evidence_text` is a substring of the `parser_snapshot.full_text` for that row. Zero violations.
4. **Discrepancies actionable.** Every blocking discrepancy carries a non-null `field_name`, `issue_type`, and (where applicable) `evidence_page` / `evidence_bbox` / `evidence_text`. An external client with read access to `_raw` and write access to the discrepancy table can fix a discrepancy and observe the row promote without further intervention.
5. **DB trigger live.** A test that inserts a fake discrepancy with `blocks_promotion=true`, resolves it, and asserts `pg_notify` fires on `extraction_raw_ready_for_promotion`, passes.
6. **Re-promotion loop.** An end-to-end test: a real document arrives with one critical discrepancy on `invoice_amount`; a write to the discrepancy table with `resolution_action='apply_value'` and a correct `resolved_value` causes the row to promote to `_stg` with the human-supplied value and a fresh provenance row marking model=`hitl`.
7. **Judge runs on every doc.** No doc type bypasses the judge tier. The judge call count per document does not exceed `(fields_needing_tiebreak + required_missing + 1)`.
8. **Observer wired.** `extraction_observer` runs as a systemd unit; `proc.bp_extraction_observation` accrues rows automatically; an alert rule on `obs_type IN ('no_pk', 'wrong_label_leak')` rate is configured (alerting backend is whatever the host supports; out of scope to choose one).
9. **Observable.** Every document is tagged with a `trace_id` threaded through `_raw`, `bp_extraction_provenance_v3`, and any discrepancy row it produces. Joining the three by `trace_id` reconstructs the full extraction trace.
10. **Integration regressions.** A fixed test set (the existing `tests/extraction_v3/fixtures/invoices/*` and equivalents the renovation accrues for PO/quote/contract) passes field-for-field against committed `_stg` values. The set is small (≤30 docs total in this build); growth is future work.

Items 1–7 gate the cutover. Items 8–10 are required for the build to be called complete but do not block doc-type cutover one by one.

## 11. Renovation steps (in execution order)

Sequenced so each step is independently testable and the live path stays working throughout.

1. **DDL migration.** Write `2026-05-16-extraction-raw-flat-columns.sql` and `2026-05-16-extraction-discrepancy-hitl.sql`. Apply against bp_sqldb. Test that the existing pipeline still writes to the new schema (write to flat columns; `parser_snapshot` gets the same dict that used to be `raw_payload`).
2. **Pattern registry skeleton + first three fields.** Build `pattern_registry.py` and `pattern_extractor.py`; migrate the regex for `invoice_id`, `supplier_name`, `invoice_amount` into `extraction_schemas/invoice.yaml`. Wire `pattern_extractor` into the live dispatch behind a feature flag, asserting parity with engine.py output on the existing test fixtures.
3. **L2 wrappers.** Build the five `engineered/*` modules as thin wrappers around what's already in the codebase (spaCy NER, table reading via PyMuPDF, address parser, date normaliser, bbox proximity). Each is callable in isolation and unit-tested.
4. **L3 judge wiring.** Build `judge_runner.py` calling existing `judge/*` modules. Add the universal substring grounding gate in `grounding.py`. Run on every doc, not just contracts.
5. **DB trigger.** Apply the discrepancy trigger (§4.3). Build `promotion.py` listening on the notify channel. Test the re-promotion loop end-to-end with a forged discrepancy.
6. **Full single flow.** Switch `dispatch.dispatch_document` to call the new `extraction/dispatch.py`. Run the integration set; verify §10 items 1–7. Remove `EXTRACTION_V3_ENGINE` env switch.
7. **Decommission.** Delete the modules listed under §5 "Code that goes away." Verify nothing imports them.
8. **Observer + trace IDs.** Wire `extraction_observer` as a systemd unit. Thread `trace_id` (a uuid generated at L0) through `_raw`, provenance, and discrepancy writes.
9. **Coverage extension.** Add PO, quote, and contract test fixtures equivalent to the existing invoice fixtures (≤10 per doc type for this build). Lock the integration suite into CI.

Each step ends with a runnable test. No step lands without one.

## 12. Risks and explicit mitigations

| Risk | Mitigation |
|---|---|
| Regex priors in YAML are wrong and L1 confidence overshoots, leading to incorrect commits | Confidence threshold is per-field in YAML; L2 fallback fires below it; judge tiebreaker fires on disagreement; grounding gate is universal. Wrong priors are observable in the discrepancy stream within hours. |
| Flattening existing JSONB `_raw` rows fails on edge cases | Migration is a one-time script with a dry-run mode. Rows that fail to flatten are reported, then dropped (they would never have promoted). The renovation does not depend on historical `_raw` data. |
| DB trigger fires too aggressively and floods the notify channel | The trigger fires only on transitions from non-resolved to resolved AND only when blocking count hits zero. Listener is idempotent: a notify for an already-promoted `raw_id` is a no-op. |
| Engine.py contains regex/heuristics not captured by the YAML migration | The renovation runs in parallel with engine.py behind a feature flag through step 5. Every fixture that engine.py extracts correctly but the new path doesn't is a discrepancy in the parity test, fixed by adding/adjusting a YAML pattern before cutover. |
| HITL writes invalid values that pass the trigger and re-fail invariants | The re-promotion listener re-runs invariants and grounding against the human-supplied values. Failed re-promotion emits fresh discrepancies; the row stays in `discrepancy`. No silent commits of bad human input. |
| Judge LLM unavailable mid-document | Existing circuit breaker in `judge/orchestrator.py` skips remaining judge calls after consecutive failures; affected fields fall to discrepancy with `issue_type='judge_unavailable'`. Re-extraction is possible by HITL action `dismiss` followed by manual re-trigger of the source `process_monitor` row. |
