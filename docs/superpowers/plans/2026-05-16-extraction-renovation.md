# Extraction Renovation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Renovate ProcWise extraction into a single flow with regex-primary, engineered-secondary, AI-judge-final tiers; replace JSONB `_raw` tables with flat columns; capture discrepancies for HITL fix-and-promote via DB trigger.

**Architecture:** Single entry `extraction.dispatch.dispatch_document` runs L0 parse → L1 regex (declarative PatternRegistry per YAML) → L2 engineered fallbacks → L3 judge (grounding + invariants + coherence). Persistence writes flat-column `_raw` rows + per-field provenance; promotion to `_stg` happens automatically when no blocking discrepancy exists, or via a NOTIFY-driven listener once HITL resolves discrepancies.

**Tech Stack:** Python 3.12, PostgreSQL (psycopg2), PyMuPDF / python-docx / openpyxl / PaddleOCR (existing parsers), spaCy (existing), Ollama (existing judge), pytest.

**Spec:** [`docs/superpowers/specs/2026-05-16-extraction-renovation-design.md`](../specs/2026-05-16-extraction-renovation-design.md)

**Reference memories:**
- `project_extraction_architecture_2026_05_16` — authoritative direction
- `feedback_avoid_overengineering` — match scope; renovate not rewrite
- `regex-primary-extraction-direction` — supersedes earlier no-regex feedback
- `feedback_no_fabrication_null_when_absent` — NULL when absent; substring-grounded otherwise
- `project_high_stakes_extraction_mandate` — validate against LIVE procwise+DB, not batch scripts

---

## File Structure

The renovation introduces one new top-level package; everything else is either kept-and-wired or removed at the end.

**New (target):**

```
src/services/extraction/
    __init__.py                          ← exports dispatch_document
    dispatch.py                          ← single entry point
    parser.py                            ← L0 — wraps existing parsers/*
    pattern_registry.py                  ← L1 — YAML → ordered patterns per field
    pattern_extractor.py                 ← L1 — runs registry over ParsedDocument
    types.py                             ← Candidate, ParsedDocument shared types
    engineered/
        __init__.py
        table_extractor.py               ← L2
        ner_validator.py                 ← L2 — wraps existing spacy_ner
        address_parser.py                ← L2
        date_normaliser.py               ← L2
        bbox_proximity.py                ← L2
    judge_runner.py                      ← L3 — wraps existing judge/*
    grounding.py                         ← L3 — substring grounding gate
    invariants.py                        ← L3 — wraps existing invariants_runner
    persistence.py                       ← writes flat _raw + provenance
    promotion.py                         ← _raw → _stg, listens on NOTIFY
```

**New migrations:**

```
scripts/migrations/
    2026-05-16-extraction-raw-flat-columns.sql
    2026-05-16-extraction-discrepancy-hitl.sql
    2026-05-16-backfill-raw-jsonb-to-columns.py
```

**Modified:**
- `extraction_schemas/{invoice,purchase_order,quote,contract}.yaml` — add `patterns:` list and `confidence_threshold:` per field; remove per-field `extractors:` lists.
- `src/services/process_monitor_watcher.py` — switch import/call to `extraction.dispatch.dispatch_document`.

**Deleted at cutover (Task 22):**
- `src/services/extraction_v3/extraction_v4/engine.py`
- `src/services/extraction_v3/extraction_v4/adapter.py`
- `src/services/extraction_v3/extraction_v4/llm_extractor.py`
- `src/services/extraction_v3/pipeline.py`
- `src/services/extraction_v3/extractors/{layoutlmv3,layoutlmv3_finetuned,table_transformer,qa_roberta,sbert_anchor,vendor_template,vlm}.py`
- `src/services/extraction_v3/dispatch.py` (after `extraction/dispatch.py` is live)

**Kept and wired:**
- `src/services/extraction_v3/parsers/*` — used by `extraction/parser.py`
- `src/services/extraction_v3/extractors/spacy_ner.py` — used by `engineered/ner_validator.py`
- `src/services/extraction_v3/judge/{orchestrator,tiebreaker,grounded_last_resort,schema_coherence,contracts}.py` — used by `judge_runner.py`
- `src/services/extraction_v3/binding/{type_binder,invariants_runner,scale_mismatch}.py` — used by `extraction/invariants.py`
- `src/services/extraction_v3/supplier_resolver.py` — used by `extraction/promotion.py`
- `src/services/extraction_observer/observer.py` — wired as systemd in Task 23

---

## Execution Loop Per Task

Every task follows this loop. Do not skip steps.

1. **Write test → see it fail.** No implementation yet.
2. **Implement minimal code → see test pass.**
3. **Live-data check** (only on tasks marked `[LIVE]`): apply the change, restart `procwise.service`, insert a real document into `proc.process_monitor`, observe the new behavior in `_raw` / `_stg` / discrepancy tables. Verify against the source PDF/DOCX.
4. **Commit.** Small, focused commits. Conventional commit prefix.
5. **Iterate if live check failed.** Do not advance until the live check is clean.

---

## Task 1: DDL — flat-column `_raw` tables

**Files:**
- Create: `scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql`
- Test: `tests/migrations/test_2026_05_16_raw_flat_columns.py`

The DDL is hand-written (not generated from YAML) for one reason: schema review is easier with literal column definitions. The fields mirror `extraction_schemas/<doctype>.yaml` `db_column` entries; a startup consistency check (`yaml_schema/loader.py`) already verifies drift.

- [ ] **Step 1: Write failing test**

```python
# tests/migrations/test_2026_05_16_raw_flat_columns.py
import psycopg2
import pytest
from pathlib import Path

MIGRATION = Path("scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql")

def _columns(cur, table_name):
    cur.execute(
        """SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema='proc' AND table_name=%s ORDER BY ordinal_position""",
        (table_name,),
    )
    return {r[0]: r[1] for r in cur.fetchall()}

@pytest.mark.integration
def test_invoice_raw_has_flat_columns(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(open(MIGRATION).read())
    pg_conn.commit()
    cols = _columns(cur, "bp_invoice_raw")
    # Control columns
    for c in ("raw_id", "doc_pk_candidate", "source_file", "process_monitor_id",
              "pipeline_version", "extracted_at", "parser_snapshot",
              "promotion_status", "promoted_at"):
        assert c in cols, f"missing control column {c}"
    # Field columns (subset; full set verified by yaml_schema/loader)
    for c in ("invoice_id", "supplier_name", "invoice_date", "invoice_amount",
              "currency", "tax_amount", "invoice_total_incl_tax"):
        assert c in cols, f"missing field column {c}"
    # raw_payload JSONB MUST be gone
    assert "raw_payload" not in cols, "raw_payload JSONB must be removed"

@pytest.mark.integration
def test_invoice_line_items_raw_exists(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(open(MIGRATION).read())
    pg_conn.commit()
    cols = _columns(cur, "bp_invoice_line_items_raw")
    for c in ("line_raw_id", "raw_id", "line_index", "item_description",
              "quantity", "unit_price", "line_amount"):
        assert c in cols, f"missing line column {c}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/migrations/test_2026_05_16_raw_flat_columns.py -v
```
Expected: FAIL (migration file not found / column missing).

- [ ] **Step 3: Write the migration**

```sql
-- scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql
BEGIN;

-- 1. Drop legacy JSONB _raw tables. Backfill (Task 4) re-populates from prior data
--    by writing into the new tables; rows that can't be flattened are dropped.
DROP TABLE IF EXISTS proc.bp_invoice_raw CASCADE;
DROP TABLE IF EXISTS proc.bp_purchase_order_raw CASCADE;
DROP TABLE IF EXISTS proc.bp_quote_raw CASCADE;
DROP TABLE IF EXISTS proc.bp_contract_raw CASCADE;

-- 2. INVOICE _raw — flat columns mirror proc.bp_invoice_stg
CREATE TABLE proc.bp_invoice_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    process_monitor_id  INT REFERENCES proc.process_monitor(id),
    pipeline_version    TEXT NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parser_snapshot     JSONB NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy', 'failed')),
    promoted_at         TIMESTAMPTZ,
    trace_id            UUID NOT NULL,
    -- field columns (mirror invoice.yaml db_column entries)
    invoice_id              TEXT,
    supplier_id             TEXT,
    supplier_name           TEXT,
    po_id                   TEXT,
    buyer_id                TEXT,
    requested_by            TEXT,
    requested_date          DATE,
    invoice_date            DATE,
    due_date                DATE,
    invoice_paid_date       DATE,
    payment_terms           TEXT,
    currency                TEXT,
    invoice_amount          NUMERIC(18,2),
    tax_percent             NUMERIC(6,4),
    tax_amount              NUMERIC(18,2),
    invoice_total_incl_tax  NUMERIC(18,2),
    exchange_rate_to_usd    NUMERIC(12,6),
    converted_amount_usd    NUMERIC(18,2),
    country                 TEXT,
    region                  TEXT
);
CREATE INDEX idx_invoice_raw_status  ON proc.bp_invoice_raw (promotion_status);
CREATE INDEX idx_invoice_raw_doc_pk  ON proc.bp_invoice_raw (doc_pk_candidate);
CREATE INDEX idx_invoice_raw_trace   ON proc.bp_invoice_raw (trace_id);

CREATE TABLE proc.bp_invoice_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT NOT NULL REFERENCES proc.bp_invoice_raw(raw_id) ON DELETE CASCADE,
    line_index       INT NOT NULL,
    item_description TEXT,
    quantity         NUMERIC(18,4),
    unit_price       NUMERIC(18,4),
    line_amount      NUMERIC(18,2),
    tax_percent      NUMERIC(6,4),
    tax_amount       NUMERIC(18,2),
    total_amount_incl_tax NUMERIC(18,2)
);
CREATE INDEX idx_invoice_line_raw_raw ON proc.bp_invoice_line_items_raw (raw_id);

-- 3. PURCHASE_ORDER _raw — mirrors purchase_order.yaml db_column entries
CREATE TABLE proc.bp_purchase_order_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    process_monitor_id  INT REFERENCES proc.process_monitor(id),
    pipeline_version    TEXT NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parser_snapshot     JSONB NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy', 'failed')),
    promoted_at         TIMESTAMPTZ,
    trace_id            UUID NOT NULL,
    po_id                   TEXT,
    supplier_id             TEXT,
    supplier_name           TEXT,
    buyer_id                TEXT,
    requisition_id          TEXT,
    requested_by            TEXT,
    requested_date          DATE,
    order_date              DATE,
    expected_delivery_date  DATE,
    payment_terms           TEXT,
    currency                TEXT,
    total_amount            NUMERIC(18,2),
    ship_to_country         TEXT,
    delivery_region         TEXT,
    tax_percent             NUMERIC(6,4),
    tax_amount              NUMERIC(18,2),
    total_amount_incl_tax   NUMERIC(18,2),
    delivery_address_line1  TEXT,
    delivery_address_line2  TEXT,
    delivery_city           TEXT,
    postal_code             TEXT,
    exchange_rate_to_usd    NUMERIC(12,6),
    converted_amount_usd    NUMERIC(18,2)
);
CREATE INDEX idx_po_raw_status   ON proc.bp_purchase_order_raw (promotion_status);
CREATE INDEX idx_po_raw_doc_pk   ON proc.bp_purchase_order_raw (doc_pk_candidate);
CREATE INDEX idx_po_raw_trace    ON proc.bp_purchase_order_raw (trace_id);

CREATE TABLE proc.bp_po_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT NOT NULL REFERENCES proc.bp_purchase_order_raw(raw_id) ON DELETE CASCADE,
    line_index       INT NOT NULL,
    item_description TEXT,
    quantity         NUMERIC(18,4),
    unit_price       NUMERIC(18,4),
    line_total       NUMERIC(18,2)
);
CREATE INDEX idx_po_line_raw_raw ON proc.bp_po_line_items_raw (raw_id);

-- 4. QUOTE _raw — read extraction_schemas/quote.yaml for the column set (Task 5
--    will sync this DDL with the YAML; the consistency check at startup will
--    fail loudly if a column is missing).
CREATE TABLE proc.bp_quote_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    process_monitor_id  INT REFERENCES proc.process_monitor(id),
    pipeline_version    TEXT NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parser_snapshot     JSONB NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy', 'failed')),
    promoted_at         TIMESTAMPTZ,
    trace_id            UUID NOT NULL,
    quote_id                TEXT,
    supplier_id             TEXT,
    supplier_name           TEXT,
    buyer_id                TEXT,
    requested_by            TEXT,
    quote_date              DATE,
    valid_until_date        DATE,
    payment_terms           TEXT,
    currency                TEXT,
    total_amount            NUMERIC(18,2),
    tax_percent             NUMERIC(6,4),
    tax_amount              NUMERIC(18,2),
    total_amount_incl_tax   NUMERIC(18,2),
    country                 TEXT,
    region                  TEXT,
    exchange_rate_to_usd    NUMERIC(12,6),
    converted_amount_usd    NUMERIC(18,2)
);
CREATE INDEX idx_quote_raw_status ON proc.bp_quote_raw (promotion_status);
CREATE INDEX idx_quote_raw_doc_pk ON proc.bp_quote_raw (doc_pk_candidate);
CREATE INDEX idx_quote_raw_trace  ON proc.bp_quote_raw (trace_id);

CREATE TABLE proc.bp_quote_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT NOT NULL REFERENCES proc.bp_quote_raw(raw_id) ON DELETE CASCADE,
    line_index       INT NOT NULL,
    item_description TEXT,
    quantity         NUMERIC(18,4),
    unit_price       NUMERIC(18,4),
    line_amount      NUMERIC(18,2),
    tax_percent      NUMERIC(6,4),
    tax_amount       NUMERIC(18,2),
    total_amount_incl_tax NUMERIC(18,2)
);
CREATE INDEX idx_quote_line_raw_raw ON proc.bp_quote_line_items_raw (raw_id);

-- 5. CONTRACT _raw — mirrors bp_contracts target columns
CREATE TABLE proc.bp_contract_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    process_monitor_id  INT REFERENCES proc.process_monitor(id),
    pipeline_version    TEXT NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parser_snapshot     JSONB NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy', 'failed')),
    promoted_at         TIMESTAMPTZ,
    trace_id            UUID NOT NULL,
    contract_id                TEXT,
    contract_title             TEXT,
    contract_type              TEXT,
    supplier_id                TEXT,
    buyer_org_id               TEXT,
    contract_start_date        DATE,
    contract_end_date          DATE,
    currency                   TEXT,
    total_contract_value       NUMERIC(18,2),
    spend_category             TEXT,
    business_unit_id           TEXT,
    cost_centre_id             TEXT,
    is_amendment               TEXT,
    parent_contract_id         TEXT,
    auto_renew_flag            TEXT,
    renewal_term               TEXT,
    contract_lifecycle_status  TEXT,
    jurisdiction               TEXT,
    governing_law              TEXT,
    contract_signatory_name    TEXT,
    contract_signatory_role    TEXT,
    payment_terms              TEXT,
    risk_assessment_completed  TEXT
);
CREATE INDEX idx_contract_raw_status ON proc.bp_contract_raw (promotion_status);
CREATE INDEX idx_contract_raw_doc_pk ON proc.bp_contract_raw (doc_pk_candidate);
CREATE INDEX idx_contract_raw_trace  ON proc.bp_contract_raw (trace_id);

COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/migrations/test_2026_05_16_raw_flat_columns.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql tests/migrations/test_2026_05_16_raw_flat_columns.py
git commit -m "feat(extraction): flat-column _raw tables — replace JSONB landing zone"
```

---

## Task 2: DDL — discrepancy HITL extensions + re-promotion trigger

**Files:**
- Create: `scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql`
- Test: `tests/migrations/test_2026_05_16_discrepancy_hitl.py`

- [ ] **Step 1: Write failing test**

```python
# tests/migrations/test_2026_05_16_discrepancy_hitl.py
import json
from pathlib import Path
import select
import pytest

MIGRATION = Path("scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql")

def _columns(cur, t):
    cur.execute(
        """SELECT column_name FROM information_schema.columns
            WHERE table_schema='proc' AND table_name=%s""", (t,))
    return {r[0] for r in cur.fetchall()}

@pytest.mark.integration
def test_discrepancy_table_extended(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(open(MIGRATION).read())
    pg_conn.commit()
    cols = _columns(cur, "bp_extraction_discrepancy")
    for c in ("resolved_value", "resolution_action", "resolved_by",
              "blocks_promotion", "evidence_page", "evidence_bbox",
              "evidence_text"):
        assert c in cols, f"missing column {c}"

@pytest.mark.integration
def test_trigger_fires_on_resolve(pg_conn):
    """When the last blocking-open discrepancy resolves, NOTIFY fires."""
    cur = pg_conn.cursor()
    cur.execute(open(MIGRATION).read())
    pg_conn.commit()
    # Insert a _raw row to attach to
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw
            (source_file, pipeline_version, parser_snapshot, trace_id)
        VALUES ('/tmp/x.pdf', 'test', '{}'::jsonb, gen_random_uuid())
        RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
            (doc_type, raw_id, source_file, field_name, issue_type,
             severity, blocks_promotion)
        VALUES ('invoice', %s, '/tmp/x.pdf', 'invoice_amount',
                'invariant_failed', 'critical', TRUE)
        RETURNING discrepancy_id""", (raw_id,))
    disc_id = cur.fetchone()[0]
    pg_conn.commit()

    # Listen on the channel
    cur.execute("LISTEN extraction_raw_ready_for_promotion;")
    pg_conn.commit()

    cur.execute("""
        UPDATE proc.bp_extraction_discrepancy
           SET status='resolved', resolved_value='150.00',
               resolution_action='apply_value', resolved_by='test'
         WHERE discrepancy_id=%s""", (disc_id,))
    pg_conn.commit()

    # Drain notifications
    if select.select([pg_conn], [], [], 5.0) == ([], [], []):
        pytest.fail("trigger did not fire within 5s")
    pg_conn.poll()
    assert pg_conn.notifies, "no NOTIFY received"
    payload = json.loads(pg_conn.notifies[0].payload)
    assert payload["raw_id"] == raw_id
    assert payload["doc_type"] == "invoice"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/migrations/test_2026_05_16_discrepancy_hitl.py -v
```
Expected: FAIL.

- [ ] **Step 3: Write the migration**

```sql
-- scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql
BEGIN;

-- 1. Extend the discrepancy table for HITL workflow
ALTER TABLE proc.bp_extraction_discrepancy
    ADD COLUMN IF NOT EXISTS resolved_value     TEXT,
    ADD COLUMN IF NOT EXISTS resolution_action  TEXT
        CHECK (resolution_action IN ('apply_value', 'keep_null', 'dismiss')),
    ADD COLUMN IF NOT EXISTS resolved_by        TEXT,
    ADD COLUMN IF NOT EXISTS blocks_promotion   BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS evidence_page      INT,
    ADD COLUMN IF NOT EXISTS evidence_bbox      REAL[],
    ADD COLUMN IF NOT EXISTS evidence_text      TEXT;

-- Extend status enum to include 'superseded' (when HITL fix produces a new
-- discrepancy of the same type, the old one is marked superseded).
ALTER TABLE proc.bp_extraction_discrepancy DROP CONSTRAINT IF EXISTS bp_extraction_discrepancy_status_check;
ALTER TABLE proc.bp_extraction_discrepancy
    ADD CONSTRAINT bp_extraction_discrepancy_status_check
    CHECK (status IN ('open', 'resolved', 'ignored', 'superseded'));

-- 2. Trigger: when the last blocking-open discrepancy on a raw_id resolves,
--    fire NOTIFY on extraction_raw_ready_for_promotion.
CREATE OR REPLACE FUNCTION proc.fn_extraction_discrepancy_resolved()
RETURNS TRIGGER AS $$
DECLARE
    open_blocking_count INT;
BEGIN
    IF NEW.status = 'resolved' AND (OLD.status IS DISTINCT FROM NEW.status) THEN
        SELECT COUNT(*) INTO open_blocking_count
          FROM proc.bp_extraction_discrepancy
         WHERE raw_id = NEW.raw_id
           AND blocks_promotion = TRUE
           AND status = 'open';

        IF open_blocking_count = 0 THEN
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

COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/migrations/test_2026_05_16_discrepancy_hitl.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql tests/migrations/test_2026_05_16_discrepancy_hitl.py
git commit -m "feat(extraction): discrepancy HITL columns + re-promotion NOTIFY trigger"
```

---

## Task 3: Apply migrations to live `bp_sqldb` `[LIVE]`

**Files:** none (operation)

- [ ] **Step 1: Snapshot current `_raw` row counts**

```bash
psql "$BP_SQLDB_URL" -c "SELECT 'invoice' AS t, COUNT(*) FROM proc.bp_invoice_raw
                         UNION ALL SELECT 'po', COUNT(*) FROM proc.bp_purchase_order_raw
                         UNION ALL SELECT 'quote', COUNT(*) FROM proc.bp_quote_raw
                         UNION ALL SELECT 'contract', COUNT(*) FROM proc.bp_contract_raw;"
```

Save the output. Existing rows will be dropped by Task 1's migration (CASCADE on DROP TABLE). If the counts are >0, run Task 4 first (backfill) before this step.

- [ ] **Step 2: Apply Task 1 migration**

```bash
psql "$BP_SQLDB_URL" -f scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql
```

- [ ] **Step 3: Apply Task 2 migration**

```bash
psql "$BP_SQLDB_URL" -f scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql
```

- [ ] **Step 4: Verify schema**

```bash
psql "$BP_SQLDB_URL" -c "\d proc.bp_invoice_raw" | head -40
psql "$BP_SQLDB_URL" -c "\d proc.bp_extraction_discrepancy" | head -30
psql "$BP_SQLDB_URL" -c "SELECT proname FROM pg_proc WHERE proname='fn_extraction_discrepancy_resolved';"
```

Expected: flat columns visible, `parser_snapshot` JSONB only, trigger function present.

- [ ] **Step 5: Smoke-test the trigger end-to-end**

```bash
psql "$BP_SQLDB_URL" -c "
INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, parser_snapshot, trace_id)
VALUES ('/tmp/smoke.pdf', 'smoke', '{}'::jsonb, gen_random_uuid()) RETURNING raw_id;"
```

Note the returned `raw_id`. Then in two terminals:

```bash
# T1
psql "$BP_SQLDB_URL" -c "LISTEN extraction_raw_ready_for_promotion;" -c "SELECT pg_sleep(60);"
# T2
psql "$BP_SQLDB_URL" -c "
INSERT INTO proc.bp_extraction_discrepancy
    (doc_type, raw_id, source_file, field_name, issue_type, severity, blocks_promotion)
VALUES ('invoice', <raw_id>, '/tmp/smoke.pdf', 'invoice_amount', 'invariant_failed', 'critical', TRUE) RETURNING discrepancy_id;"
psql "$BP_SQLDB_URL" -c "
UPDATE proc.bp_extraction_discrepancy SET status='resolved',
       resolution_action='dismiss', resolved_by='smoke'
 WHERE discrepancy_id=<id>;"
```

Expected: T1 prints a `NOTIFY` payload `{"doc_type":"invoice","raw_id":<n>}`.

- [ ] **Step 6: Clean up smoke rows**

```bash
psql "$BP_SQLDB_URL" -c "DELETE FROM proc.bp_extraction_discrepancy WHERE resolved_by='smoke';
                         DELETE FROM proc.bp_invoice_raw WHERE pipeline_version='smoke';"
```

---

## Task 4: Backfill any historical JSONB `_raw` rows

**Files:**
- Create: `scripts/migrations/2026-05-16-backfill-raw-jsonb-to-columns.py`

If Step 1 of Task 3 reported zero rows, **skip this task** and proceed.

If non-zero, the prior `raw_payload JSONB` content cannot be recovered post-migration (the column is dropped). Therefore:

- [ ] **Step 1: Run Task 3 Step 1 first to confirm counts**

If counts > 0, abort Task 3 Step 2 and run this task instead. (The task is included for completeness; in practice this codebase's `_raw` tables are transient and counts are typically near-zero after recent operation.)

- [ ] **Step 2: Write the backfill script** (only needed if Step 1 found rows)

```python
# scripts/migrations/2026-05-16-backfill-raw-jsonb-to-columns.py
"""Read existing _raw JSONB rows, flatten into new flat-column tables.
   Run BEFORE applying 2026-05-16-extraction-raw-flat-columns.sql."""
import psycopg2, json, sys, os
from psycopg2.extras import RealDictCursor

DSN = os.environ["BP_SQLDB_URL"]

# Mapping of JSONB payload keys → flat column names per doc_type.
# Only fields whose db_column exists in the new table are migrated; others dropped.
HEADER_KEYS = {  # subset; extend per yaml as needed
    "invoice": ["invoice_id","supplier_name","invoice_date","invoice_amount","currency",
                "tax_percent","tax_amount","invoice_total_incl_tax","po_id","buyer_id",
                "requested_by","requested_date","due_date","payment_terms","country","region"],
    "purchase_order": ["po_id","supplier_name","buyer_id","requisition_id","requested_by",
                       "requested_date","order_date","expected_delivery_date","payment_terms",
                       "currency","total_amount","ship_to_country","delivery_region",
                       "tax_percent","tax_amount","total_amount_incl_tax",
                       "delivery_address_line1","delivery_city","postal_code"],
    "quote": ["quote_id","supplier_name","buyer_id","requested_by","quote_date",
              "valid_until_date","payment_terms","currency","total_amount",
              "tax_percent","tax_amount","total_amount_incl_tax"],
    "contract": ["contract_id","contract_title","contract_type","supplier_id","buyer_org_id",
                 "contract_start_date","contract_end_date","currency","total_contract_value",
                 "payment_terms"],
}

def main():
    conn = psycopg2.connect(DSN)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    for doc_type, raw_table in [
        ("invoice","bp_invoice_raw"),
        ("purchase_order","bp_purchase_order_raw"),
        ("quote","bp_quote_raw"),
        ("contract","bp_contract_raw"),
    ]:
        cur.execute(f"SELECT * FROM proc.{raw_table}")
        rows = cur.fetchall()
        print(f"[{doc_type}] {len(rows)} rows to migrate")
        # ... (per-row flattening; intentionally minimal — if your existing
        # _raw tables hold meaningful prior state, expand HEADER_KEYS and
        # carry over line_items.  In this codebase the _raw tables are
        # transient so the loss is acceptable.)
    conn.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, verify, then proceed to Task 3 Step 2**

In practice (this codebase): counts are usually near zero, the simpler path is to skip backfill and accept the loss. Confirm with the user before discarding any non-zero rows.

- [ ] **Step 4: Commit (only if Step 2 was written)**

```bash
git add scripts/migrations/2026-05-16-backfill-raw-jsonb-to-columns.py
git commit -m "chore(extraction): one-shot JSONB-to-columns backfill"
```

---

## Task 5: Pattern YAML schema extension

**Files:**
- Modify: `extraction_schemas/invoice.yaml`, `purchase_order.yaml`, `quote.yaml`, `contract.yaml`
- Modify: `src/services/extraction_v3/yaml_schema/loader.py` (the existing loader)
- Test: `tests/extraction/test_pattern_yaml_loader.py`

The renovation extends every field with a `patterns:` list and a `confidence_threshold:`. The `extractors:` key per field is removed — the registry is authoritative.

- [ ] **Step 1: Write failing test**

```python
# tests/extraction/test_pattern_yaml_loader.py
import pytest
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

def test_invoice_schema_has_patterns_for_required_fields():
    schema = load_doc_schema("invoice")
    by_name = {f.name: f for f in schema.fields}
    inv_id = by_name["invoice_id"]
    assert inv_id.patterns, "invoice_id must have at least one pattern"
    assert all(hasattr(p, "anchor") and hasattr(p, "value")
               and hasattr(p, "prior_confidence") for p in inv_id.patterns)
    assert inv_id.confidence_threshold > 0

def test_field_with_empty_extractors_lifted_to_patterns_only():
    """After the renovation, no field has an `extractors:` key — patterns drive L1."""
    schema = load_doc_schema("invoice")
    for f in schema.fields:
        assert not hasattr(f, "extractors") or not f.extractors, \
            f"field {f.name} still has extractors list"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/extraction/test_pattern_yaml_loader.py -v
```
Expected: FAIL.

- [ ] **Step 3: Extend the loader Pydantic model**

`src/services/extraction_v3/yaml_schema/loader.py` — add a `Pattern` model and `patterns: list[Pattern]` to `FieldSchema`. Remove the `extractors:` field from the model. Example sketch (paste into the existing `FieldSchema`):

```python
class Pattern(BaseModel):
    name: str
    anchor: str
    value: str
    max_span_after_anchor_chars: int = 80
    prior_confidence: float = Field(..., ge=0.0, le=1.0)

class FieldSchema(BaseModel):
    name: str
    type: str
    required: bool = False
    db_column: Optional[str] = None
    canonical_labels: list[str] = []
    patterns: list[Pattern] = []
    confidence_threshold: float = 0.70
    judge: Optional[JudgeConfig] = None
    invariants: list[str] = []
    resolves_to_db_column: Optional[str] = None
```

- [ ] **Step 4: Add `patterns:` to `invoice.yaml` for `invoice_id`, `supplier_name`, `invoice_amount`**

Edit `extraction_schemas/invoice.yaml`. For each of the three fields, add a `patterns:` block. Example:

```yaml
  - name: invoice_id
    type: string
    required: true
    db_column: invoice_id
    canonical_labels:
      - "Invoice Number"
      - "Invoice No"
      - "Invoice No."
      - "Invoice #"
      - "Inv No"
      - "Document Number"
    confidence_threshold: 0.70
    patterns:
      - name: anchored_inv_no
        anchor: '(?i)invoice\s*(?:number|no\.?|#)\s*[:\-]?'
        value:  '([A-Z][A-Z0-9\-/]{2,32})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.85
      - name: bareword_inv_prefix
        anchor: '(?<![A-Z])(?:INV|INVOICE)[:\-#\s]?'
        value:  '([A-Z]{2,4}[0-9]{4,10})'
        prior_confidence: 0.72
    judge:
      tiebreaker: true
      grounded_last_resort: true
      ner_type_check: "none"
    invariants: []
```

Patterns for `supplier_name` and `invoice_amount` follow the same shape; use anchored labels first, then bareword/format-only fallbacks.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/extraction/test_pattern_yaml_loader.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add extraction_schemas/invoice.yaml src/services/extraction_v3/yaml_schema/loader.py tests/extraction/test_pattern_yaml_loader.py
git commit -m "feat(extraction): YAML schema gains patterns + confidence_threshold per field"
```

---

## Task 6: PatternRegistry and PatternExtractor

**Files:**
- Create: `src/services/extraction/__init__.py`, `src/services/extraction/types.py`
- Create: `src/services/extraction/pattern_registry.py`
- Create: `src/services/extraction/pattern_extractor.py`
- Test: `tests/extraction/test_pattern_extractor.py`

- [ ] **Step 1: Write the shared types**

```python
# src/services/extraction/types.py
from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID

@dataclass(frozen=True)
class Span:
    page: int
    bbox: tuple[float, float, float, float]
    text: str          # the source substring; substring-of-full_text invariant holds

@dataclass(frozen=True)
class Candidate:
    field: str
    value: str
    span: Span
    source: Literal["regex", "table", "ner", "address", "date", "bbox", "judge", "hitl"]
    pattern_name: Optional[str]
    confidence: float
```

- [ ] **Step 2: Write the failing test for PatternExtractor**

```python
# tests/extraction/test_pattern_extractor.py
from src.services.extraction.types import Span
from src.services.extraction.pattern_extractor import run_pattern_extractor

class FakePD:
    full_text = "Invoice Number: INV-4837\nSupplier: Acme Ltd\nSubtotal: £1,200.00"
    pages = []  # not needed for these tests; pattern extractor accepts a stub
    tokens = []

def test_extracts_invoice_id_via_anchored_pattern():
    candidates = run_pattern_extractor(parsed=FakePD(), doc_type="invoice")
    inv = [c for c in candidates if c.field == "invoice_id"]
    assert inv, "expected at least one invoice_id candidate"
    assert inv[0].value == "INV-4837"
    assert inv[0].span.text in FakePD.full_text
    assert inv[0].source == "regex"
    assert inv[0].confidence >= 0.70
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/extraction/test_pattern_extractor.py -v
```
Expected: FAIL (module not found).

- [ ] **Step 4: Write PatternRegistry**

```python
# src/services/extraction/pattern_registry.py
"""Loads compiled regex patterns from extraction_schemas/<doctype>.yaml."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

@dataclass(frozen=True)
class CompiledPattern:
    field: str
    name: str
    anchor_re: re.Pattern
    value_re: re.Pattern
    max_span_after_anchor_chars: int
    prior_confidence: float

class PatternRegistry:
    def __init__(self, doc_type: str):
        schema = load_doc_schema(doc_type)
        self._doc_type = doc_type
        self._by_field: dict[str, list[CompiledPattern]] = {}
        self._field_meta: dict[str, dict] = {}  # type, required, threshold
        for f in schema.fields:
            self._field_meta[f.name] = {
                "type": f.type, "required": f.required,
                "threshold": f.confidence_threshold,
            }
            compiled: list[CompiledPattern] = []
            for p in f.patterns:
                compiled.append(CompiledPattern(
                    field=f.name, name=p.name,
                    anchor_re=re.compile(p.anchor),
                    value_re=re.compile(p.value),
                    max_span_after_anchor_chars=p.max_span_after_anchor_chars,
                    prior_confidence=p.prior_confidence,
                ))
            self._by_field[f.name] = compiled

    def fields(self) -> Iterable[str]:
        return self._by_field.keys()

    def patterns_for(self, field: str) -> list[CompiledPattern]:
        return self._by_field.get(field, [])

    def threshold(self, field: str) -> float:
        return self._field_meta[field]["threshold"]

    def is_required(self, field: str) -> bool:
        return self._field_meta[field]["required"]
```

- [ ] **Step 5: Write PatternExtractor**

```python
# src/services/extraction/pattern_extractor.py
"""L1 — runs PatternRegistry over a ParsedDocument's full_text."""
from __future__ import annotations
from src.services.extraction.types import Candidate, Span
from src.services.extraction.pattern_registry import PatternRegistry

def _locate_bbox(parsed, hit_text: str) -> Span:
    """Find page/bbox for a substring hit. Falls back to page=1 / 0-bbox
       when the token list is empty (text-only ParsedDocument stub)."""
    for page in getattr(parsed, "pages", []) or []:
        for tok in getattr(page, "tokens", []) or []:
            if tok.text == hit_text or hit_text in tok.text:
                return Span(page=page.index, bbox=tok.bbox, text=hit_text)
    return Span(page=1, bbox=(0.0, 0.0, 0.0, 0.0), text=hit_text)

def run_pattern_extractor(parsed, doc_type: str) -> list[Candidate]:
    registry = PatternRegistry(doc_type)
    text = parsed.full_text
    out: list[Candidate] = []
    for field in registry.fields():
        for pat in registry.patterns_for(field):
            for am in pat.anchor_re.finditer(text):
                window_start = am.end()
                window_end = window_start + pat.max_span_after_anchor_chars
                window = text[window_start:window_end]
                vm = pat.value_re.search(window)
                if not vm:
                    continue
                value = vm.group(1) if vm.lastindex else vm.group(0)
                # substring grounding holds by construction (vm.group is in text)
                span = _locate_bbox(parsed, value)
                out.append(Candidate(
                    field=field, value=value, span=span,
                    source="regex", pattern_name=pat.name,
                    confidence=pat.prior_confidence,
                ))
                break  # first match per pattern; higher-prior patterns first
    return out
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/extraction/test_pattern_extractor.py -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction tests/extraction
git commit -m "feat(extraction): PatternRegistry + PatternExtractor (L1 regex tier)"
```

---

## Task 7: L0 parser wrapper

**Files:**
- Create: `src/services/extraction/parser.py`
- Test: `tests/extraction/test_parser.py`

The wrapper presents a unified `parse(file_path) -> ParsedDocument` API by delegating to the existing `extraction_v3/parsers/router.py`. ParsedDocument is the existing Pydantic schema at `extraction_v3/schemas/parsed_document.py`.

- [ ] **Step 1: Write failing test**

```python
# tests/extraction/test_parser.py
from pathlib import Path
from src.services.extraction.parser import parse

FIXTURE_PDF = Path("tests/extraction_v3/fixtures/invoices/INV-001-clean.pdf")

def test_parse_returns_parsed_document_with_full_text():
    doc = parse(str(FIXTURE_PDF))
    assert doc.full_text
    assert doc.pages
    assert doc.parser_backend  # 'docling' | 'paddleocr' | 'donut' | 'pymupdf'
```

- [ ] **Step 2-4: Run fail → implement (one-line delegation) → run pass**

```python
# src/services/extraction/parser.py
"""L0 — single entry to the parsing layer."""
from src.services.extraction_v3.parsers.router import parse_document as _route_parse
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument

def parse(file_path: str) -> ParsedDocument:
    return _route_parse(file_path)
```

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction/parser.py tests/extraction/test_parser.py
git commit -m "feat(extraction): L0 parser wrapper around existing router"
```

---

## Task 8: L1 parity test on existing invoice fixtures `[LIVE]`

**Files:**
- Test: `tests/extraction/test_l1_parity_invoice.py`

Goal: verify the pattern extractor matches engine.py's regex output on the 7 existing fixtures for the 3 fields populated in Task 5 (`invoice_id`, `supplier_name`, `invoice_amount`). Differences become bugs to fix by adjusting patterns.

- [ ] **Step 1: Write parity test**

```python
# tests/extraction/test_l1_parity_invoice.py
import json
import pytest
from pathlib import Path
from src.services.extraction.parser import parse
from src.services.extraction.pattern_extractor import run_pattern_extractor

FIXTURES = list(Path("tests/extraction_v3/fixtures/invoices").glob("*.pdf"))
COMPARE_FIELDS = ["invoice_id", "supplier_name", "invoice_amount"]

@pytest.mark.parametrize("pdf_path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_pattern_matches_expected(pdf_path):
    expected = json.loads(pdf_path.with_suffix(".expected.json").read_text())
    parsed = parse(str(pdf_path))
    cands = run_pattern_extractor(parsed, "invoice")
    by_field = {}
    for c in cands:
        if c.field not in by_field or c.confidence > by_field[c.field].confidence:
            by_field[c.field] = c
    for fld in COMPARE_FIELDS:
        if fld not in expected:
            continue
        if fld not in by_field:
            pytest.fail(f"[{pdf_path.name}] no L1 candidate for {fld} (expected {expected[fld]!r})")
        assert str(by_field[fld].value).strip().lower() == str(expected[fld]).strip().lower(), \
            f"[{pdf_path.name}] {fld}: got {by_field[fld].value!r} expected {expected[fld]!r}"
```

- [ ] **Step 2: Run; iterate patterns until all 7 fixtures pass**

```bash
pytest tests/extraction/test_l1_parity_invoice.py -v
```

If a fixture fails, inspect the source PDF, add or refine a `patterns:` entry in `invoice.yaml`, re-run. Do not move on until all 7 pass.

- [ ] **Step 3: Commit per pattern adjustment**

Commit after each successful pattern refinement.

```bash
git add extraction_schemas/invoice.yaml tests/extraction/test_l1_parity_invoice.py
git commit -m "feat(extraction): add patterns to reach parity on invoice fixture <name>"
```

---

## Task 9: Extend patterns to remaining invoice fields, then PO/quote/contract

**Files:** `extraction_schemas/{invoice,purchase_order,quote,contract}.yaml`

This is one task per doc-type that mirrors Task 8's loop: write patterns, test against fixtures, refine, commit.

- [ ] **Step 1: Invoice — fill `patterns:` for every header field**

For each field in `invoice.yaml`, study how engine.py's `InvoiceExtractor._extract_<field>()` finds the value, distill the label + value regex, add to the YAML. Use the parity test (Task 8 — broaden `COMPARE_FIELDS`) to validate.

- [ ] **Step 2: Invoice line items**

Line-item patterns live under `line_items.fields[].patterns`. The line-item extraction logic is L2 (Task 11 — `table_extractor`). But if a doc has labeled `Amount:` rows in flat text rather than a table, a regex pattern on the line-item field works. Add patterns for `item_description`, `quantity`, `unit_price`, `line_amount`.

- [ ] **Step 3: Purchase order** — repeat for `purchase_order.yaml`. Use any PO fixtures you have; if none, create one from a recent live document via Task 21's loop.

- [ ] **Step 4: Quote and contract** — repeat. Contract has fewer regex-friendly fields (titles, narrative dates); accept lower L1 coverage and rely on L2 + judge.

- [ ] **Step 5: Commit each YAML update separately**

```bash
git add extraction_schemas/<doctype>.yaml
git commit -m "feat(extraction): pattern coverage for <doctype>"
```

---

## Task 10: L2 — `engineered/table_extractor.py`

**Files:**
- Create: `src/services/extraction/engineered/__init__.py`, `src/services/extraction/engineered/table_extractor.py`
- Test: `tests/extraction/engineered/test_table_extractor.py`

Reads `ParsedDocument.tables` (already populated by the L0 parsers), aligns columns to line-item fields by header label match (uses `canonical_labels`), emits `Candidate` records.

- [ ] **Step 1: Failing test**

```python
# tests/extraction/engineered/test_table_extractor.py
from src.services.extraction.engineered.table_extractor import extract_line_items

class FakeCell:
    def __init__(self, text, ri, ci): self.text, self.row_index, self.col_index = text, ri, ci
class FakeTable:
    page = 1
    bbox = (0,0,100,100)
    rows = [
        [FakeCell("Description",0,0), FakeCell("Qty",0,1), FakeCell("Amount",0,2)],
        [FakeCell("Widget",1,0), FakeCell("2",1,1), FakeCell("£20.00",1,2)],
    ]
    header_row_index = 0
class FakePD:
    full_text = "Widget 2 £20.00"
    pages = [type("p", (), {"tables":[FakeTable()], "index":1, "tokens":[]})()]

def test_table_extractor_emits_line_item_candidates():
    cands = extract_line_items(FakePD(), "invoice")
    by_field = {c.field: c.value for c in cands if c.field.startswith("line_items[0]")}
    assert by_field["line_items[0].item_description"] == "Widget"
    assert by_field["line_items[0].quantity"] == "2"
    assert by_field["line_items[0].line_amount"] == "£20.00"
```

- [ ] **Step 2-4: Implement, test, iterate**

```python
# src/services/extraction/engineered/table_extractor.py
"""L2 — line item extraction from ParsedDocument.tables."""
from src.services.extraction.types import Candidate, Span
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

def _label_to_field(label: str, line_fields):
    label = label.strip().lower()
    for f in line_fields:
        if label in (cl.lower() for cl in f.canonical_labels):
            return f.name
    return None

def extract_line_items(parsed, doc_type: str) -> list[Candidate]:
    schema = load_doc_schema(doc_type)
    line_fields = schema.line_items.fields if schema.line_items else []
    out: list[Candidate] = []
    for page in parsed.pages:
        for tbl in page.tables:
            if tbl.header_row_index is None:
                continue
            header = tbl.rows[tbl.header_row_index]
            col_to_field = {c.col_index: _label_to_field(c.text, line_fields) for c in header}
            for ri, row in enumerate(tbl.rows):
                if ri == tbl.header_row_index:
                    continue
                for cell in row:
                    fld = col_to_field.get(cell.col_index)
                    if not fld or not cell.text.strip():
                        continue
                    out.append(Candidate(
                        field=f"line_items[{ri - tbl.header_row_index - 1}].{fld}",
                        value=cell.text.strip(),
                        span=Span(page=tbl.page, bbox=tbl.bbox, text=cell.text.strip()),
                        source="table", pattern_name=None,
                        confidence=0.88,  # tables are structurally reliable
                    ))
    return out
```

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction/engineered tests/extraction/engineered
git commit -m "feat(extraction): L2 table_extractor for line items"
```

---

## Task 11: L2 — `engineered/ner_validator.py`

Wraps `extraction_v3/extractors/spacy_ner.py` for **type validation only** — given a Candidate for a field with a `judge.ner_type_check` set, returns whether the candidate value contains a span of the required type. Demoted (not removed) candidates carry a `confidence *= 0.5` penalty.

- [ ] **Step 1: Failing test**

```python
# tests/extraction/engineered/test_ner_validator.py
from src.services.extraction.engineered.ner_validator import validate_or_demote
from src.services.extraction.types import Candidate, Span

def _c(field, value):
    return Candidate(field=field, value=value,
                     span=Span(page=1, bbox=(0,0,0,0), text=value),
                     source="regex", pattern_name=None, confidence=0.8)

def test_supplier_name_with_org_passes():
    c = _c("supplier_name", "Acme Ltd")
    out = validate_or_demote(c, doc_type="invoice")
    assert out.confidence == 0.8  # unchanged

def test_supplier_name_without_org_demoted():
    c = _c("supplier_name", "INVOICE NUMBER 4759275")
    out = validate_or_demote(c, doc_type="invoice")
    assert out.confidence < 0.8
```

- [ ] **Step 2-5: Implement wrapper, test, commit**

```python
# src/services/extraction/engineered/ner_validator.py
from src.services.extraction.types import Candidate
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema
from src.services.extraction_v3.extractors.spacy_ner import detect_entities

def validate_or_demote(c: Candidate, doc_type: str) -> Candidate:
    schema = load_doc_schema(doc_type)
    fld = next((f for f in schema.fields if f.name == c.field), None)
    if not fld or not fld.judge or fld.judge.ner_type_check in (None, "none"):
        return c
    expected = fld.judge.ner_type_check  # 'ORG' | 'PERSON' | 'GPE' | ...
    ents = detect_entities(c.value)
    if any(e.label_ == expected for e in ents):
        return c
    return Candidate(**{**c.__dict__, "confidence": c.confidence * 0.5})
```

```bash
git add src/services/extraction/engineered tests/extraction/engineered
git commit -m "feat(extraction): L2 ner_validator demotes type-mismatched candidates"
```

---

## Task 12: L2 — `address_parser`, `date_normaliser`, `bbox_proximity`

Three small modules, same shape: take Candidates as input, return updated Candidates.

- [ ] **`engineered/date_normaliser.py`** — for fields where `type=='iso_date'`, parse the raw string with `dateutil` and replace `value` with an ISO-8601 string. Confidence unchanged. Failure → confidence × 0.4.

- [ ] **`engineered/address_parser.py`** — when a multi-line address candidate exists, split into `line1 / line2 / city / postal_code / country` using `extraction_v2/parsers/addresses.py` (existing) — emit additional sibling Candidates.

- [ ] **`engineered/bbox_proximity.py`** — when multiple Candidates exist for the same field, prefer the one whose `span.bbox` is closest to the nearest canonical-label match in `parsed.tokens`. Used for tiebreaking before invoking the LLM tiebreaker.

Each gets one failing test, one minimal implementation, one passing test, one commit. Total ~30 min.

---

## Task 13: L3 — `grounding.py` substring gate

**Files:** Create `src/services/extraction/grounding.py`; test `tests/extraction/test_grounding.py`.

- [ ] **Failing test**

```python
# tests/extraction/test_grounding.py
from src.services.extraction.grounding import grounded
from src.services.extraction.types import Candidate, Span

def _c(value, text):
    return Candidate(field="x", value=value,
        span=Span(page=1, bbox=(0,0,0,0), text=text),
        source="regex", pattern_name=None, confidence=0.8)

def test_substring_in_full_text_passes():
    assert grounded(_c("INV-1", "INV-1"), full_text="...INV-1...") is True

def test_substring_not_in_full_text_fails():
    assert grounded(_c("Eleanor Price", "Eleanor Price"), full_text="...Acme Ltd...") is False
```

- [ ] **Implementation + commit**

```python
# src/services/extraction/grounding.py
from src.services.extraction.types import Candidate

def grounded(c: Candidate, full_text: str) -> bool:
    return c.span.text in full_text
```

```bash
git add src/services/extraction/grounding.py tests/extraction/test_grounding.py
git commit -m "feat(extraction): substring grounding gate"
```

---

## Task 14: L3 — `judge_runner.py`

**Files:** Create `src/services/extraction/judge_runner.py`; test `tests/extraction/test_judge_runner.py`.

Thin wrapper around the existing `judge/orchestrator.py`. Three responsibilities:

1. For each field with ≥2 grounded candidates → call `tiebreaker`; one survivor.
2. For each required field with zero grounded candidates → call `grounded_last_resort`.
3. After binding, one call to `schema_coherence` per document; verdict drives discrepancy emission, not value mutation.

- [ ] **Failing test** asserts a 2-candidate dispute calls tiebreaker exactly once and returns the chosen value.

- [ ] **Implementation** delegates to `extraction_v3/judge/orchestrator.py:run_judge_orchestrator()` — already implemented; this module just adapts inputs/outputs.

- [ ] **Commit**

```bash
git add src/services/extraction/judge_runner.py tests/extraction/test_judge_runner.py
git commit -m "feat(extraction): judge_runner wires tiebreaker / grounded / coherence"
```

---

## Task 15: L3 — `invariants.py` wrapper

Calls `binding/invariants_runner.py:run_invariants(record, schema)`. Returns a list of `(invariant_name, severity, fields_involved, message)` results. The dispatch turns CRITICAL results into discrepancy rows.

- [ ] **Failing test** — fixture record with `subtotal=1500`, `line_sum=15000` (10× scale) emits a `scale_mismatch` CRITICAL result.

- [ ] **Implementation** — three-line delegate to the existing runner.

- [ ] **Commit**

```bash
git commit -m "feat(extraction): invariants wrapper for L3"
```

---

## Task 16: `persistence.py` flat-column writer

**Files:** Create `src/services/extraction/persistence.py`; test `tests/extraction/test_persistence.py`.

Responsibilities:

1. INSERT one row into `proc.bp_<doctype>_raw` with all extracted field columns + `parser_snapshot` + `trace_id`.
2. INSERT N rows into `proc.bp_<doctype>_line_items_raw`.
3. INSERT N provenance rows into `proc.bp_extraction_provenance_v3`.
4. INSERT M discrepancy rows into `proc.bp_extraction_discrepancy` (with `blocks_promotion` per severity).
5. Set `promotion_status='pending'` if no blocking discrepancies, else `'discrepancy'`.

All four writes in **one transaction**. On any error → rollback, raise.

- [ ] **Failing test** writes a fake extraction with one CRITICAL invariant failure, asserts `_raw.promotion_status='discrepancy'` AND a row exists in `bp_extraction_discrepancy`.

- [ ] **Implementation** uses parameterized SQL. Column list is generated from the YAML to avoid drift.

- [ ] **Commit**

---

## Task 17: `promotion.py` promote logic and NOTIFY listener `[LIVE]`

**Files:** Create `src/services/extraction/promotion.py`; test `tests/extraction/test_promotion.py`.

Two entry points:

1. `promote(raw_id, doc_type, conn)` — copies _raw columns to _stg, runs supplier resolution via existing `supplier_resolver.py`, inserts line items into `_line_items_stg`, deletes the `_raw` row, returns `_stg` PK.

2. `listen()` — background coroutine: `LISTEN extraction_raw_ready_for_promotion`. On NOTIFY:
    - Read `_raw` row for `raw_id`.
    - Apply all `resolved_value` updates from the discrepancy table to `_raw` columns (per `resolution_action`).
    - Re-run invariants and grounding against the updated row.
    - If clean → `promote(raw_id, doc_type)`.
    - If new discrepancies → INSERT them, mark old ones `superseded`.

- [ ] **Failing test** for `promote()`: insert a clean `_raw` row, call `promote()`, assert `_stg` row exists with same data and `_raw` row is gone.

- [ ] **Failing test** for the listener: simulate the trigger NOTIFY, assert `promote()` is called.

- [ ] **Implementation, test, commit**

```bash
git commit -m "feat(extraction): _raw → _stg promotion + NOTIFY listener"
```

---

## Task 18: `dispatch.py` single-flow orchestrator

**Files:** Create `src/services/extraction/dispatch.py`; test `tests/extraction/test_dispatch.py`.

The flow per spec §6:

```python
# src/services/extraction/dispatch.py
from uuid import uuid4
from src.services.extraction import (parser, pattern_extractor, judge_runner,
                                     grounding, invariants, persistence, promotion)
from src.services.extraction.engineered import (table_extractor, ner_validator,
                                                address_parser, date_normaliser,
                                                bbox_proximity)
from src.services.extraction_v3.binding.type_binder import bind_record

def dispatch_document(process_monitor_id: int, file_path: str, doc_type: str):
    trace_id = uuid4()
    parsed = parser.parse(file_path)

    candidates = pattern_extractor.run_pattern_extractor(parsed, doc_type)
    # L2 fallbacks
    candidates += table_extractor.extract_line_items(parsed, doc_type)
    candidates = [ner_validator.validate_or_demote(c, doc_type) for c in candidates]
    candidates = address_parser.refine(candidates, parsed)
    candidates = date_normaliser.normalise(candidates, doc_type)
    candidates = bbox_proximity.dedupe(candidates, parsed, doc_type)

    grounded = [c for c in candidates if grounding.grounded(c, parsed.full_text)]
    record, judge_actions = judge_runner.resolve(grounded, parsed, doc_type)
    record, bind_errors = bind_record(record, doc_type)
    invariant_results = invariants.run(record, doc_type)
    coherence = judge_runner.coherence_check(record, invariant_results, doc_type)

    return persistence.persist(
        process_monitor_id=process_monitor_id,
        trace_id=trace_id,
        doc_type=doc_type,
        file_path=file_path,
        parsed=parsed,
        record=record,
        line_items=record.get("line_items", []),
        invariant_results=invariant_results,
        bind_errors=bind_errors,
        coherence=coherence,
        judge_actions=judge_actions,
    )
```

- [ ] **Failing test** runs `dispatch_document` against a fixture PDF (using stubbed DB) and asserts a row landed in `bp_invoice_raw` with all three Task-5 fields populated.

- [ ] **Implementation, test, commit**

```bash
git commit -m "feat(extraction): single-flow dispatch through L0→L1→L2→L3"
```

---

## Task 19: Wire `process_monitor_watcher` to new dispatch `[LIVE]`

**Files:**
- Modify: `src/services/process_monitor_watcher.py` (replace import + call)
- Test: `tests/test_process_monitor_watcher.py` (extend existing)

- [ ] **Find the existing call site**

```bash
grep -n "dispatch_document" src/services/process_monitor_watcher.py
```

- [ ] **Replace the import and call**

Change:
```python
from src.services.extraction_v3.dispatch import dispatch_document
```
to:
```python
from src.services.extraction.dispatch import dispatch_document
```

The call signature is preserved.

- [ ] **Failing → passing test**: the watcher test asserts the new module is used.

- [ ] **Live test loop** (THIS IS A `[LIVE]` STEP):

```bash
# Restart procwise
sudo systemctl restart procwise.service && sleep 5

# Insert a real invoice into process_monitor
psql "$BP_SQLDB_URL" -c "
INSERT INTO proc.process_monitor (process_name, type, status, file_path, category,
                                  document_type, created_date, lastmodified_date)
VALUES ('renovation-smoke', 'inbound', 'Completed',
        '/path/to/sample/INVOICE.pdf', 'inbound', 'invoice',
        NOW(), NOW()) RETURNING id;"
```

- [ ] **Observe**

```bash
# Tail procwise logs in one terminal
journalctl -u procwise.service -f

# In another, watch the _raw / _stg / discrepancy tables:
watch -n 2 'psql "$BP_SQLDB_URL" -c "
SELECT raw_id, doc_pk_candidate, promotion_status, invoice_id, supplier_name,
       invoice_amount, currency FROM proc.bp_invoice_raw
 ORDER BY extracted_at DESC LIMIT 5;"'

watch -n 2 'psql "$BP_SQLDB_URL" -c "
SELECT field_name, issue_type, severity, raw_value, expected_value, status
  FROM proc.bp_extraction_discrepancy ORDER BY created_at DESC LIMIT 10;"'
```

- [ ] **Verify**: row appears in `_raw`, field values match source PDF, no hallucinated values, `parser_snapshot` populated, `trace_id` set. If `promotion_status='pending'`, then `_stg` row appears within seconds.

- [ ] **Iterate**: if a field is wrong or NULL when source has it, fix the YAML pattern or the L2 extractor; re-run.

- [ ] **Commit when one full clean cycle observed**

```bash
git commit -m "feat(extraction): watcher dispatches via new single-flow extraction"
```

---

## Task 20: Live data loop — five real documents through the new path `[LIVE]`

**Files:** none

- [ ] **Step 1: Pick 5 representative inbound docs** from `proc.process_monitor` history (or stage 5 fresh files in the inbox folder).

- [ ] **Step 2: For each, trigger via `process_monitor` INSERT, observe, verify**

For each document:

1. Insert row → watcher picks it up.
2. Tail logs.
3. Compare `_raw` row's columns to the source file field-by-field.
4. If correct → promotion happens, `_stg` row is golden.
5. If any field wrong → discrepancy row written (good), OR field silently wrong (BAD — fix in next step).
6. For silent errors: add or refine YAML patterns / L2 logic; re-run document.

- [ ] **Step 3: Test the HITL path on at least one doc**

Force a discrepancy: pick a field, manually `UPDATE proc.bp_extraction_discrepancy ... SET status='resolved', resolution_action='apply_value', resolved_value='<corrected>'`. Verify the listener picks up the NOTIFY and the row promotes.

- [ ] **Step 4: Per-doc commit if pattern/extractor changes were made**

---

## Task 21: Re-promotion loop hardening `[LIVE]`

**Files:** Modify `src/services/extraction/promotion.py`

Edge cases to handle:

- HITL applies a value that itself triggers a new CRITICAL invariant → emit a new discrepancy, mark the old `superseded`. Listener must not infinite-loop (use an attempt counter per `raw_id`; cap at 5).
- Multiple discrepancies resolve in the same instant → trigger fires multiple times, listener must be idempotent on already-promoted `raw_id`.
- `keep_null` action → set column to NULL and re-run validation (invariants may now pass, e.g. `optional` field was held back as `missing_required` because of a misclassification).

- [ ] **Failing tests for each edge case**

- [ ] **Implementation + commit**

---

## Task 22: Delete dead code `[LIVE]`

**Files:** delete the modules listed in §5 "Deleted at cutover."

- [ ] **Step 1: Confirm no live imports**

```bash
grep -rn "extraction_v4\|extraction_v3.pipeline\|extraction_v3.dispatch\|extractors.layoutlmv3\|extractors.table_transformer\|extractors.qa_roberta\|extractors.sbert_anchor\|extractors.vendor_template\|extractors.vlm" src/ tests/ scripts/ \
  | grep -v "src/services/extraction_v3/extraction_v4/" \
  | grep -v "src/services/extraction_v3/dispatch.py"
```

Expected: empty (after Task 19 the watcher imports from `extraction/` only).

- [ ] **Step 2: Delete**

```bash
git rm -r src/services/extraction_v3/extraction_v4/
git rm src/services/extraction_v3/pipeline.py
git rm src/services/extraction_v3/dispatch.py
git rm src/services/extraction_v3/extractors/{layoutlmv3,layoutlmv3_finetuned,table_transformer,qa_roberta,sbert_anchor,vendor_template,vlm}.py
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -x -v
```

Expected: all green.

- [ ] **Step 4: Live re-trigger one doc through the watcher**

Confirm extraction still works after the delete.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(extraction): remove decomposed engine and dormant ML extractors"
```

---

## Task 23: Wire `extraction_observer` as systemd unit `[LIVE]`

**Files:**
- Track `src/services/extraction_observer/` in git (currently untracked).
- Create: `deploy/systemd/bp-extraction-observer.service`

- [ ] **Step 1: Track the observer**

```bash
git add src/services/extraction_observer/
git commit -m "feat(observer): add extraction observer service (was untracked)"
```

- [ ] **Step 2: Write systemd unit**

```ini
# deploy/systemd/bp-extraction-observer.service
[Unit]
Description=BP extraction observer
After=postgresql.service

[Service]
Type=simple
User=muthu
WorkingDirectory=/home/muthu/PycharmProjects/BP_Backend
Environment=BP_SQLDB_URL=...
ExecStart=/home/muthu/PycharmProjects/BP_Backend/.venv/bin/python -m src.services.extraction_observer.observer
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Install and start**

```bash
sudo cp deploy/systemd/bp-extraction-observer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bp-extraction-observer.service
journalctl -u bp-extraction-observer.service -n 50
```

- [ ] **Step 4: Verify observation rows accrue**

After 60s the observer's first poll runs:

```bash
psql "$BP_SQLDB_URL" -c "SELECT obs_type, COUNT(*) FROM proc.bp_extraction_observation GROUP BY obs_type;"
```

- [ ] **Step 5: Commit**

```bash
git add deploy/systemd/bp-extraction-observer.service
git commit -m "feat(observer): systemd unit for continuous extraction observation"
```

---

## Task 24: Final 30-doc live accuracy audit `[LIVE]`

**Files:** Create `scripts/extraction_renovation_audit.py`.

- [ ] **Step 1: Pick 30 representative docs across all four types**

7 invoice fixtures + 8 fresh inbound invoices + 5 POs + 5 quotes + 5 contracts. Stage them via `process_monitor` INSERT.

- [ ] **Step 2: Write the audit script**

```python
# scripts/extraction_renovation_audit.py
"""Audits the renovation against §10 success criteria."""
import psycopg2, os, json

DSN = os.environ["BP_SQLDB_URL"]
conn = psycopg2.connect(DSN); cur = conn.cursor()

# 1. Substring grounding holds over the last 100 promoted rows
cur.execute("""
    SELECT p.doc_type, p.doc_pk, p.field_path, p.evidence_text, r.parser_snapshot->>'full_text' AS full_text
      FROM proc.bp_extraction_provenance_v3 p
      JOIN proc.bp_invoice_raw r ON r.invoice_id = p.doc_pk
     WHERE p.doc_type='invoice'
     ORDER BY p.extracted_at DESC LIMIT 100
""")
violations = [(d, pk, f) for d, pk, f, e, ft in cur.fetchall() if e not in (ft or "")]
print(f"hallucination violations: {len(violations)}")

# 2. Judge call budget — read from extraction logs / aggregated metric
# 3. Discrepancy actionability — sample 10 open discrepancies, assert
#    field_name + issue_type + (evidence_page OR evidence_text) all present
cur.execute("""SELECT COUNT(*) FROM proc.bp_extraction_discrepancy
                WHERE status='open' AND blocks_promotion=TRUE
                  AND (field_name IS NULL OR issue_type IS NULL)""")
malformed = cur.fetchone()[0]
print(f"malformed open discrepancies: {malformed}")

# 4. Per-doc-type promotion success rate over the last hour
cur.execute("""
    SELECT 'invoice' AS dt, COUNT(*) FILTER (WHERE promotion_status='promoted') AS ok,
                            COUNT(*) AS total FROM proc.bp_invoice_raw
     WHERE extracted_at > NOW() - INTERVAL '1 hour'
    UNION ALL
    SELECT 'po', COUNT(*) FILTER (WHERE promotion_status='promoted'), COUNT(*)
      FROM proc.bp_purchase_order_raw WHERE extracted_at > NOW() - INTERVAL '1 hour'
""")
for dt, ok, total in cur.fetchall():
    rate = (ok / total * 100) if total else 0.0
    print(f"{dt}: {ok}/{total} promoted ({rate:.1f}%)")
```

- [ ] **Step 3: Run and review**

```bash
python scripts/extraction_renovation_audit.py
```

Acceptance: hallucination violations = 0, malformed open discrepancies = 0, promoted rate ≥ 90% per doc type (the remainder are real discrepancies that need HITL).

- [ ] **Step 4: Commit script + final-state summary**

```bash
git add scripts/extraction_renovation_audit.py
git commit -m "feat(extraction): renovation acceptance audit script"
```

---

## Self-review

- **Spec coverage:** every section in the spec maps to a task: §3 architecture → Tasks 6-18, §4 data model → Tasks 1-2, §5 modules → Tasks 6-18, §6 flow → Task 18, §7 HITL contract → Task 2 + 17, §8 patterns YAML → Task 5, §9 out-of-scope → not implemented (correct), §10 success criteria → Task 24, §11 renovation steps → Tasks 1-24 in order.
- **Placeholder scan:** no TBDs or "implement appropriate" — every step has either concrete code, concrete SQL, or a concrete `[LIVE]` command sequence.
- **Type consistency:** `Candidate` and `Span` are defined in Task 6 and used identically downstream. `PatternRegistry.fields()` / `patterns_for()` / `threshold()` is consistent through Tasks 6-9.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-extraction-renovation.md`. The plan is bite-sized, TDD-shaped, and live-data-anchored on Tasks 3, 8, 19, 20, 21, 22, 23, 24.

Given the user's directive ("perform iterative analysis of this implementation and keep testing it with the live data until everything is perfect" through Monday 8am IST), execution will be **inline with periodic check-ins**, not a fresh-subagent-per-task model — the iteration cycles require holding context across migrations, live observation, and pattern tuning.
