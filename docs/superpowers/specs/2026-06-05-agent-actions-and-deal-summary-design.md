# Agent Actions Event Log + Deal Summarization API — Design

**Date:** 2026-06-05
**Status:** Approved (design), pending implementation plan
**Author:** muthu + Claude

## Problem

The extraction pipeline (L0 parse → L1 regex → L2 engineered/NER → L3 AI judge →
context layer → persist) performs many discrete actions per document, plus
validation (discrepancy capture) and, in future, cross-document reconciliation.
Today these are only partially observable via `bp_extraction_telemetry`
(one summary row per doc) and `bp_extraction_observation` (aggregated anomalies).

We need:
1. A fine-grained, append-only **`agent_actions`** event log — one row per
   action/step — capturing extraction, validation, and (future) consolidation
   actions.
2. An **AI summarization API** that, given a `deal_id`, consolidates the final
   (`_trgt`) records and the action trail into a clear-text description.

## Scope

In scope:
- `proc.agent_actions` table (migration).
- A best-effort writer `record_action()` / `bulk_record()`.
- Wiring the **existing** extraction and validation actions to the writer.
- A reconciliation **writer slot** (no reconciliation logic built now).
- Deal context gatherer + LLM summarizer + FastAPI endpoint.
- Unit tests.

Out of scope (deferred to their own specs):
- Building cross-document reconciliation logic (3-way match, tolerances,
  line-level matching, status transitions). Only the action-writing interface
  is provided.

## Background facts (verified against live `proc` schema, 2026-06-05)

- Final/authoritative tables are the `_trgt` family:
  `bp_invoice_trgt`, `bp_purchase_order_trgt`, `bp_quote_trgt`, `bp_contracts`,
  plus `_line_items_trgt` for invoice/po/quote.
- `deal_id` (with `deal_name`, `document_id`) is present across all
  `_raw`/`_stg`/`_trgt` doc tables and `process_monitor`; it is the entity key.
- `deal_document_id_map(deal_name, document_id)` links deals to documents.
- DB access: `src/services/db.py` `get_conn()` context manager, psycopg2,
  explicit `autocommit=False` / `commit` / `rollback`, schema `proc`.
- Migrations: plain SQL DDL in `deploy/sql/YYYY-MM-DD_*.sql`
  (`CREATE TABLE IF NOT EXISTS`, `bigserial` PK, `timestamptz DEFAULT now()`,
  `jsonb`). No Alembic.
- Pipeline entry: `src/services/extraction/dispatch.py:dispatch_document()`.
- Validation/persistence: `src/services/extraction/persistence.py`
  (`write_raw`, `write_discrepancies`, `write_provenance`).
- LLM: `src/services/ollama_client.py:ollama_generate()`; routing policy in
  `src/services/llm_router.py` (summarization is cloud-preferred `qwen2.5:7b`,
  extraction stays local).
- API: FastAPI, app at `src/api/main.py`, routers under `src/api/routers/`.

## 1. Data model — `proc.agent_actions`

Append-only, one row per action/step. Best-effort writes that must never break
extraction.

```sql
CREATE TABLE IF NOT EXISTS proc.agent_actions (
    action_id          bigserial PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    deal_id            text,          -- nullable: often unknown mid-extraction
    document_id        text,
    doc_pk             text,          -- e.g. invoice_id / po_id / quote_id
    doc_type           text,
    process_monitor_id integer,
    trace_id           text,
    phase              text NOT NULL, -- 'extraction' | 'validation' | 'consolidation'
    action_type        text NOT NULL, -- e.g. 'regex_extract','context_synthesize','grounding_gate','persist','discrepancy','reconcile_match'
    agent              text,          -- 'L1_regex','L3_judge','AgentNick','validator','reconciler'
    field_name         text,          -- nullable, for field-level actions
    status             text,          -- 'ok'|'warn'|'error'|'skipped'
    summary            text,          -- short human-readable line
    details            jsonb,         -- structured payload
    confidence         numeric,
    pipeline_version   text
);
CREATE INDEX IF NOT EXISTS ix_agent_actions_deal     ON proc.agent_actions (deal_id);
CREATE INDEX IF NOT EXISTS ix_agent_actions_document ON proc.agent_actions (document_id);
CREATE INDEX IF NOT EXISTS ix_agent_actions_doc_pk   ON proc.agent_actions (doc_pk);
CREATE INDEX IF NOT EXISTS ix_agent_actions_trace    ON proc.agent_actions (trace_id);
CREATE INDEX IF NOT EXISTS ix_agent_actions_created  ON proc.agent_actions (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_actions_phase    ON proc.agent_actions (phase);
```

Migration file: `deploy/sql/2026-06-05_agent_actions.sql`.

`phase` and `action_type` are free text (not enums) to avoid migrations when new
action types appear; canonical values are documented in the writer module.

## 2. Writer — `src/services/agent_actions.py`

```python
def record_action(
    *, phase, action_type,
    deal_id=None, document_id=None, doc_pk=None, doc_type=None,
    process_monitor_id=None, trace_id=None, agent=None, field_name=None,
    status="ok", summary=None, details=None, confidence=None,
    pipeline_version=None, conn=None,
) -> None: ...

def bulk_record(rows, *, conn=None) -> None: ...
```

Behavior:
- **Best-effort & isolated:** if `conn` is omitted, open a short transaction via
  `get_conn()`, insert, commit. **Catch and log all exceptions; never
  re-raise** — a logging failure must not fail a good extraction
  (Extraction-Accuracy-Priority).
- **Optional `conn`:** when provided, use the caller's cursor and do not
  commit/rollback (the caller owns the transaction) — for callers wanting the
  log write to be atomic with their work.
- `details` serialized to JSONB via `json.dumps` with a safe default.
- `pipeline_version` defaults from the same source the pipeline already uses.
- Module-level constants enumerate canonical `phase` and `action_type` values
  for callers and documentation.

## 3. Hook-up (existing actions only)

- `src/services/extraction/dispatch.py:dispatch_document()` — a handful of
  `record_action(phase="extraction", ...)` calls at phase boundaries:
  - L1/L2 candidate counts (`action_type="regex_extract"` / `"engineered_extract"`),
  - L3 judge (`"judge"`),
  - context-layer synthesize (`"context_synthesize"`),
  - grounding gate (`"grounding_gate"`, status reflects dropped candidates),
  - persist (`"persist"`, details include `raw_id`, `promotion_status`).
  Calls are best-effort and placed so they cannot alter control flow.
- `src/services/extraction/persistence.py:write_discrepancies()` — one
  `record_action(phase="validation", action_type="discrepancy", ...)` per
  discrepancy (use `bulk_record`). Uses the existing persistence `conn` so the
  validation log is atomic with the discrepancy write.
- **Consolidation:** documented usage only —
  `record_action(phase="consolidation", action_type="reconcile_match"|"reconcile_mismatch", deal_id=...)`.
  No reconciliation logic is implemented in this task.

## 4. AI summarization API

### Gatherer — `src/services/deal_summary.py:gather_deal_context(deal_id, conn=None)`
Returns a plain dict (pure data, independently testable):
```python
{
  "deal_id": str,
  "deal_name": str | None,
  "documents": {
    "invoices":  [ {<bp_invoice_trgt row>, "line_items": [...] }, ... ],
    "purchase_orders": [ ... ],
    "quotes": [ ... ],
    "contracts": [ ... ],
  },
  "actions": [ {<agent_actions row>}, ... ],        # ordered by created_at
  "discrepancies": [ {<bp_extraction_discrepancy row>}, ... ],
}
```
- Selects `_trgt` rows where `deal_id = %s` for each doc type; joins line items
  by the doc PK.
- Loads `agent_actions` for the deal's documents (by `deal_id` or `document_id`).
- Loads related rows from `bp_extraction_discrepancy` for those documents.
- Returns `None`/empty marker if the deal has no `_trgt` rows.

### Summarizer — `summarize_deal(deal_id, conn=None)`
- Calls `gather_deal_context`; if empty, signals not-found.
- Builds a **grounded** prompt: facts-only, "use only the data provided, leave
  out anything not present, do not fabricate, do not infer values"
  (No-Fabrication-NULL-When-Absent).
- `temperature=0`. **Runs on the Ollama Cloud API** (`OLLAMA_CLOUD_BASE_URL`,
  bearer `OLLAMA_CLOUD_API_KEY`) via `ollama_cloud_generate()`, NOT the local
  GPU — the local GPU stays dedicated to AgentNick extraction, and we do not
  add a second local model. Model is env-configurable via
  `PROCWISE_SUMMARY_MODEL` (default `gpt-oss:120b`). (Decision 2026-06-06: user
  chose cloud summarization + single local model AgentNick, superseding the
  earlier local-`qwen2.5:7b` plan.)
- Returns `{deal_id, summary, generated_at, sources: {...counts}}`.
  (`generated_at` stamped by the caller/endpoint, not inside a workflow.)

### Endpoint — `src/api/routers/deal_summary.py`
- `GET /deals/{deal_id}/summary` → 200 with the summarizer payload.
- 404 when the deal has no `_trgt` rows.
- Registered via `app.include_router(...)` in `src/api/main.py`.

## 5. Error handling

- Writer swallows and logs all errors (best-effort).
- Gatherer raises on DB errors (caller/endpoint maps to 500); returns empty for
  unknown deal (endpoint maps to 404).
- Summarizer: on LLM failure, return 502/503-style error from the endpoint with
  a clear message; do not fabricate a summary.

## 6. Testing

- `tests/.../test_agent_actions.py`:
  - `record_action` swallows DB errors (monkeypatch `get_conn` to raise) and
    does not propagate.
  - `record_action` with a passed `conn` uses it and does not commit.
  - happy-path insert builds expected SQL/params.
- `tests/.../test_deal_summary.py`:
  - `gather_deal_context` assembles the expected dict from fixture rows
    (fake conn / monkeypatched cursor).
  - unknown deal → empty/None.
  - prompt construction asserts no-fabrication framing and that only provided
    facts appear; `ollama_generate`/`llm_router` mocked.

## Design principles honored

- **Isolation:** writer, gatherer, summarizer, and endpoint are separate units
  with clear interfaces; each is testable without the others.
- **Extraction accuracy is #1:** logging is best-effort and cannot fail
  extraction.
- **No fabrication:** summarizer is grounded to provided facts only.
- **No over-engineering:** reconciliation logic deferred; free-text
  phase/action_type avoids churn.
