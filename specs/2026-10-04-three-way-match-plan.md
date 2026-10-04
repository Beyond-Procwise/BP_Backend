# Three-Way Match Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture goods receipts as a first-class document, then prove per purchase-order line that what was billed does not exceed what was received — and say plainly which lines could not be checked.

**Architecture:** A goods receipt is a new entry in the data-driven document vocabulary (`proc.bp_document_type`) routed at a fifth physical pipeline (`goods_receipt`), stored in the house `raw → _stg → _trgt` shape. The match reuses `two_way_match._assign_lines` a second time so receipt lines and invoice lines both assign to the same PO line through the resolution layer; the PO line is the spine and the three quantities are compared per line, over the whole document set. Findings go to the existing discrepancy queue. Nothing blocks.

**Tech Stack:** Python 3, psycopg2, PostgreSQL (`proc` schema), pytest. No new dependencies.

**Spec:** `specs/2026-10-04-three-way-match-design.md` — **DRAFT, not yet approved.** §13 names the gate.

**Predecessor:** `5d1bfc4` renamed `three_way_match.py` → `two_way_match.py`. `specs/2026-10-01-document-relationship-layer-rulings.md` governs classification behaviour; read it before changing any.

---

## STOP — Phase 0 gates this plan

**Do not start Task 2 until someone confirms that customers actually receive goods receipts.**

The corpus holds **zero** receipt-like documents — 0 of 184, under any spelling (`delivery`, `despatch`, `goods received`, `grn`, `packing`, `proof of delivery`). If a customer's ERP books receipts without producing a document, or their spend is all services, this control cannot be delivered to them at any price and every task below is waste.

Task 1 is safe to run regardless: it measures the current classifier so criterion 5 stays provable. Everything after it assumes the gate passed and that **10–20 real goods receipts** are in hand to test against.

---

## Global Constraints

- **Every guard must be proven to fail.** Break the behaviour on purpose, watch the test go red, restore, watch it go green. A test whose red state was never observed is not evidence. Fourteen guards in an earlier plan were found green while checking nothing.
- **Migrations are additive, idempotent and reversible.** Every `deploy/sql/<name>.sql` gets a `deploy/sql/<name>_rollback.sql` sibling. `CREATE TABLE IF NOT EXISTS`, `ON CONFLICT DO NOTHING` on seeds, so a re-run never overwrites a row a human has edited.
- **Both databases.** Every migration applies to `bp_testdb` (the configured `.env` database) **and** `bp_sqldb`. Task 11 owns that; no task is done until Task 11 covers it.
- **NULL when absent — this plan's central rule.** A goods receipt carries no prices, and a PO line with no receipt is *not assessed*, never *failed*. See Task 4 Step 1 and Task 8 Step 5; both are guard tests, not comments.
- **Never modify source data.** `_trgt` rows for existing document types are read, never written, by anything in this plan.
- **New tables take the `bp_` prefix; indexes are `ix_bp_<table>_<cols>`.**
- **`get_conn()` is autocommit.** `rollback()` is a no-op and `FOR UPDATE` locks end with the statement. Set `conn.autocommit = False` explicitly for a multi-statement transaction, as `persistence.write_raw` does.
- **Findings carry a normalised `source_file`.** The open-row key is `(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name)` and `persistence.normalise_source_file()` must be applied on write. Never reduce it to a basename. Two clause sites, four normalise sites.
- **Alias order is part of the data.** `tests/services/concepts/test_concept_table.py::test_document_type_rows_equal_the_seed_column_for_column` compares `aliases` as an **ordered list** against `seed.py`. The migration and the seed must list them identically, new ones last.
- **Test invocation:**
  ```bash
  set -a && . ./.env && set +a
  CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:1 ./venv/bin/python -m pytest <path> -v
  ```
  Add `PROCWISE_TEST_LIVE_DB=1` for tests that read the database. Use `./venv/bin/python` (the test venv), not `.venv`.
- **Committing: this checkout and its index are shared with another session.** Never `git add -A`, never a bare `git commit`. Use a private index, re-reading HEAD inside the same step:
  ```bash
  HEAD_SHA=$(git rev-parse HEAD)
  export GIT_INDEX_FILE=/tmp/claude-1001/<session>/scratchpad/idx; rm -f "$GIT_INDEX_FILE"
  git read-tree "$HEAD_SHA" && git add -- <my paths>
  TREE=$(git write-tree)
  COMMIT=$(git commit-tree "$TREE" -p "$HEAD_SHA" -F msg.txt)
  git update-ref refs/heads/Development "$COMMIT" "$HEAD_SHA"
  unset GIT_INDEX_FILE && git reset -q HEAD -- <my paths>
  ```
  **Before any push: `git diff HEAD --stat -- <every path you committed>` must be empty.** A peer who staged a path before your commit and commits after it writes their stale blob over you, and the working tree still looks right. This has happened twice.
- **If a file you must edit is already dirty from another session, stage HUNKS, not the path.** Derive any synthesised blob from `"$HEAD_SHA"`, never from symbolic `HEAD` captured in an earlier step.
- **Work stays on `Development`.** Never push to `main`.
- **Implementers must not be Haiku** — it has twice swept another session's staged work into commits on this shared index.

---

## Review Focus

Five things the spec implies, that no task's happy path exercises, most likely to bite first.

1. **A receipt whose unit differs from its PO line's** — `box` against `each` is the most likely real failure, and the corpus cannot size it. Must report `UNVERIFIABLE_UOM`, never a silent pass and never a false over-billing finding. *Pinned in Task 7 Step 7.*
2. **A PO line with no receipt at all** must read as not assessed, not failed — otherwise missing paperwork manufactures a failure rate. *Pinned in Task 8 Step 5.*
3. **Two invoices that each pass alone and together exceed what was received.** The value check already learned this; the quantity check must not relearn it. *Pinned in Task 8 Step 3.*
4. **A delivery note that restates PO prices** must not populate any price field on the receipt row. The extractor will try. *Pinned in Task 4 Step 1.*
5. **Goods delivered and refused.** `quantity_rejected` must not be silently summed into `quantity_received`, or a rejected delivery reads as received. *Pinned in Task 8 Step 6.*

---

## File Structure

**Created**

| Path | Responsibility |
|---|---|
| `deploy/sql/2026-10-04_goods_receipt_tables.sql` (+`_rollback`) | The six new tables |
| `deploy/sql/2026-10-04_goods_receipt_doctype.sql` (+`_rollback`) | The vocabulary row |
| `deploy/sql/2026-10-04_uom_receipt_basis.sql` (+`_rollback`) | `receipt_basis` column and seeding |
| `deploy/sql/2026-10-04_deal_overview_three_way.sql` (+`_rollback`) | The two view columns |
| `src/services/extraction/three_way_match.py` | The match. New file; the name is free again and now means it. |
| `tests/extraction/test_three_way_match.py` | Its tests |
| `tests/extraction/test_goods_receipt_extraction.py` | Capture and the no-price guard |

**Modified**

| Path | Change |
|---|---|
| `src/services/concepts/seed.py` | `doctype.goods_receipt` concept + DocumentType |
| `src/services/concepts/validate.py:171` | `_PIPELINES` gains `goods_receipt` |
| `src/services/extraction/dispatch.py` | Route the new pipeline; call the match |
| `src/services/extraction/two_way_match.py` | Export `_assign_lines` for reuse (no logic change) |
| `src/services/rga/builders/board_paper.py`, `exec_procurement_summary.py`, `analysis_findings.py`, `opportunity_miner_agent.py` | Read the new columns |
| `beyond_procwise_ui/src/modules/SpendIQ/engine.js` | Read the new columns |

---

## Task 1: Baseline the classifier before touching the vocabulary

Criterion 5 of the design — *"No document that classifies correctly today classifies differently afterwards"* — is only provable against a recorded before-state. Adding eleven aliases to the vocabulary is exactly the change that can steal a document from another type.

**Files:**
- Create: `tests/extraction/test_goods_receipt_classification_baseline.py`
- Create: `specs/2026-10-04-classification-baseline.json` (the recorded answer)

**Interfaces:**
- Produces: `baseline.json`, a mapping `{source_file: concept_code}` for every document in `proc.process_monitor`, which Task 2 asserts against.

- [ ] **Step 1: Write the script that records the baseline**

```python
# tests/extraction/test_goods_receipt_classification_baseline.py
"""The classification answer for every corpus document, before the vocabulary
gains a goods receipt. Task 2 adds eleven aliases -- 'delivery note', 'advice
note', 'packing list' among them -- and any of those could pull a document away
from the type it reads as today. This file is the only way to know."""
import json, os, pathlib, pytest

BASELINE = pathlib.Path(__file__).parents[2] / "specs" / "2026-10-04-classification-baseline.json"

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="reads the live corpus"
)

def _classify_corpus() -> dict:
    from src.services.extraction.type_resolver import resolve_document_type
    from src.services.db import get_conn
    out = {}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT file_path FROM proc.process_monitor WHERE file_path IS NOT NULL")
        for (path,) in cur.fetchall():
            out[path] = getattr(resolve_document_type(path), "concept_code", None)
    return out

def test_classification_is_unchanged_from_the_baseline():
    assert BASELINE.exists(), "run scripts/record_classification_baseline.py first"
    assert _classify_corpus() == json.loads(BASELINE.read_text())
```

- [ ] **Step 2: Record the baseline**

```bash
set -a && . ./.env && set +a
PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python - <<'PY'
import json, pathlib, sys; sys.path[:0]=["src","."]
from tests.extraction.test_goods_receipt_classification_baseline import _classify_corpus
p = pathlib.Path("specs/2026-10-04-classification-baseline.json")
p.write_text(json.dumps(_classify_corpus(), indent=1, sort_keys=True))
print("recorded", len(json.loads(p.read_text())), "documents")
PY
```
Expected: a count equal to the `process_monitor` row count with a non-null `file_path`.

- [ ] **Step 3: Run the test to verify it passes against its own baseline**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/extraction/test_goods_receipt_classification_baseline.py -v`
Expected: PASS.

- [ ] **Step 4: Prove the guard fails**

Temporarily append `"delivery note"` to `doctype.invoice`'s aliases in `src/services/concepts/seed.py`, re-run, and confirm the test goes **red** naming the documents that moved. Revert.
Expected: FAIL. If it still passes, the test is not reading the resolver and must be fixed before Task 2.

- [ ] **Step 5: Commit**

```bash
# private index, per Global Constraints
git add -- tests/extraction/test_goods_receipt_classification_baseline.py \
            specs/2026-10-04-classification-baseline.json
# subject: test(extraction): record the classification baseline before the receipt vocabulary
```

---

## Task 2: The storage, before the vocabulary can name it

**Order matters and is not negotiable.** `src/services/concepts/validate.py:171` holds
`_PIPELINES = frozenset({"invoice", "purchase_order", "quote", "contract"})` with
`check_pipeline_targets_exist` rejecting any `pipeline_doc_type` outside it, because *"a fifth value would route a document at _raw/_stg/_trgt tables that are not there."* The tables must exist first.

**Files:**
- Create: `deploy/sql/2026-10-04_goods_receipt_tables.sql`, `deploy/sql/2026-10-04_goods_receipt_tables_rollback.sql`
- Create: `tests/sql/test_goods_receipt_tables.py`

**Interfaces:**
- Produces: `proc.bp_goods_receipt_{raw,stg,trgt}` and `proc.bp_goods_receipt_line_items_{raw,stg,trgt}`, consumed by Tasks 4, 5, 7, 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/sql/test_goods_receipt_tables.py
"""The six tables, and the one column that must NOT exist on any of them."""
import os, pytest
from src.services.db import get_conn

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")

TIERS = ("raw", "stg", "trgt")

@pytest.mark.parametrize("tier", TIERS)
def test_header_and_line_tables_exist(tier):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT table_name FROM information_schema.tables
                        WHERE table_schema='proc' AND table_name = ANY(%s)""",
                    ([f"bp_goods_receipt_{tier}", f"bp_goods_receipt_line_items_{tier}"],))
        assert len({r[0] for r in cur.fetchall()}) == 2

@pytest.mark.parametrize("tier", TIERS)
def test_no_price_column_anywhere_on_a_goods_receipt(tier):
    """A goods receipt has no prices. A column is an invitation; there is none."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT table_name, column_name FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name LIKE %s
                          AND (column_name ~* '(price|amount|total|value|cost|currency|tax)')""",
                    (f"bp_goods_receipt%{tier}",))
        assert cur.fetchall() == []

def test_trgt_is_keyed_by_deal_id():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name='bp_goods_receipt_trgt'
                          AND column_name='deal_id'""")
        assert cur.fetchone() is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sql/test_goods_receipt_tables.py -v`
Expected: FAIL, `assert 0 == 2`.

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-10-04_goods_receipt_tables.sql
-- The fifth physical pipeline. Mirrors the invoice family's shape so dispatch,
-- promotion and deal assignment need no special case.
--
-- There is deliberately NO price, amount, total, value, currency or tax column
-- on any of these six tables. A goods receipt proves DELIVERY; it carries
-- quantities and nothing else. A price column here would be filled by the
-- extractor from the PO the note references, and a fabricated price on the
-- document that proves delivery is the worst place in the product for one.
-- tests/sql/test_goods_receipt_tables.py asserts the absence.

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_raw (
    raw_id            bigserial PRIMARY KEY,
    grn_id            text,
    po_id             text,
    supplier_id       text,
    supplier_name     text,
    receipt_date      date,
    delivery_note_ref text,
    carrier_ref       text,
    received_by       text,
    source_file       text,
    content_hash      text,
    extraction_conf   numeric,
    pipeline_version  text,
    recorded_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_po_id      ON proc.bp_goods_receipt_raw (po_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_grn_id     ON proc.bp_goods_receipt_raw (grn_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_source_file ON proc.bp_goods_receipt_raw (source_file);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_raw (
    raw_line_id       bigserial PRIMARY KEY,
    raw_id            bigint REFERENCES proc.bp_goods_receipt_raw(raw_id),
    line_no           integer,
    description       text,
    quantity_received numeric,
    quantity_rejected numeric,
    unit_of_measure   text,
    po_line_ref       text,
    recorded_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_raw_raw_id
    ON proc.bp_goods_receipt_line_items_raw (raw_id);

-- _stg: same columns plus the reconciliation the other pipelines carry.
CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_stg (
    LIKE proc.bp_goods_receipt_raw INCLUDING DEFAULTS INCLUDING INDEXES
);
ALTER TABLE proc.bp_goods_receipt_stg ADD COLUMN IF NOT EXISTS link_score numeric;
ALTER TABLE proc.bp_goods_receipt_stg ADD COLUMN IF NOT EXISTS link_conf  numeric;

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_stg (
    LIKE proc.bp_goods_receipt_line_items_raw INCLUDING DEFAULTS INCLUDING INDEXES
);

-- _trgt: the final record, keyed by deal_id like every other _trgt table.
CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_trgt (
    LIKE proc.bp_goods_receipt_stg INCLUDING DEFAULTS INCLUDING INDEXES
);
ALTER TABLE proc.bp_goods_receipt_trgt ADD COLUMN IF NOT EXISTS deal_id text;
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_trgt_deal_id
    ON proc.bp_goods_receipt_trgt (deal_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_trgt (
    LIKE proc.bp_goods_receipt_line_items_stg INCLUDING DEFAULTS INCLUDING INDEXES
);
ALTER TABLE proc.bp_goods_receipt_line_items_trgt ADD COLUMN IF NOT EXISTS deal_id text;
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_trgt_deal_id
    ON proc.bp_goods_receipt_line_items_trgt (deal_id);
```

```sql
-- deploy/sql/2026-10-04_goods_receipt_tables_rollback.sql
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_raw;
DROP TABLE IF EXISTS proc.bp_goods_receipt_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_raw;
```

- [ ] **Step 4: Apply to bp_testdb and run the tests**

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-04_goods_receipt_tables.sql
PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sql/test_goods_receipt_tables.py -v
```
Expected: PASS, 7 tests.

- [ ] **Step 5: Prove idempotency and the rollback**

Re-run the migration (expect no error), then apply the rollback, re-run the tests (expect FAIL), re-apply the migration (expect PASS).

- [ ] **Step 6: Commit** — `feat(extraction): the goods receipt gets its tables`

---

## Task 3: The vocabulary names it

**Files:**
- Modify: `src/services/concepts/validate.py:171`
- Modify: `src/services/concepts/seed.py` (two places: `_DOCUMENT_TYPE_CONCEPTS`, `DOCUMENT_TYPES`)
- Create: `deploy/sql/2026-10-04_goods_receipt_doctype.sql` (+ rollback)

**Interfaces:**
- Consumes: the six tables from Task 2.
- Produces: `doctype.goods_receipt` resolvable by `type_resolver`, with `pipeline_doc_type="goods_receipt"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/extraction/test_goods_receipt_extraction.py
import pytest
from src.services.concepts.seed import DOCUMENT_TYPES

def test_the_vocabulary_knows_a_goods_receipt():
    dt = DOCUMENT_TYPES["doctype.goods_receipt"]
    assert dt.pipeline_doc_type == "goods_receipt"
    assert dt.default_parent_type == "doctype.order"
    assert dt.role == "role.transaction"

@pytest.mark.parametrize("alias", ["goods receipt", "grn", "delivery note",
                                   "despatch note", "proof of delivery", "packing slip"])
def test_the_names_a_supplier_actually_uses_resolve(alias):
    assert alias in DOCUMENT_TYPES["doctype.goods_receipt"].aliases
```

- [ ] **Step 2: Run it to verify it fails**

Expected: `KeyError: 'doctype.goods_receipt'`.

- [ ] **Step 3: Open the pipeline set**

```python
# src/services/concepts/validate.py:171
#: The physical table families the extraction pipeline actually has.
#: goods_receipt joined 2026-10-04; its six tables land in
#: deploy/sql/2026-10-04_goods_receipt_tables.sql, which MUST be applied first --
#: this frozenset is the only thing standing between a vocabulary row and a
#: SQL error mid-extraction.
_PIPELINES = frozenset({"invoice", "purchase_order", "quote", "contract", "goods_receipt"})
```

- [ ] **Step 4: Add the concept and the document type**

```python
# src/services/concepts/seed.py -- in _DOCUMENT_TYPE_CONCEPTS, after doctype.invoice
("doctype.goods_receipt", "Records what was physically delivered and accepted.", ()),
```

```python
# src/services/concepts/seed.py -- in DOCUMENT_TYPES, after doctype.invoice.
# Alias order is part of the data: the migration below lists them identically,
# and test_document_type_rows_equal_the_seed_column_for_column compares them as
# an ORDERED list.
DocumentType(
    "doctype.goods_receipt", "role.transaction", "doctype.order", "exec.unilateral",
    ("goods receipt", "goods received note", "grn", "delivery note", "despatch note",
     "dispatch note", "advice note", "packing list", "packing slip",
     "proof of delivery", "pod"),
    ({"field": "grn_id", "pattern": None, "parent_type": None},
     {"field": "po_id", "pattern": None, "parent_type": "doctype.order"}),
    ("quantities with no prices", "signed for on receipt",
     "a carrier, vehicle or consignment reference"),
    "goods_receipt",
),
```

- [ ] **Step 5: Write the migration, aliases in the same order**

```sql
-- deploy/sql/2026-10-04_goods_receipt_doctype.sql
INSERT INTO proc.bp_document_type
  (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
   structural_signals, pipeline_doc_type, tenant_id, status, source, observed_count,
   requires_parent_evidence, parent_evidence_phrases, recorded_at, valid_from)
VALUES
  ('doctype.goods_receipt', 'role.transaction', 'doctype.order', 'exec.unilateral',
   ARRAY['goods receipt','goods received note','grn','delivery note','despatch note',
         'dispatch note','advice note','packing list','packing slip',
         'proof of delivery','pod'],
   '[{"field":"grn_id","pattern":null,"parent_type":null},
     {"field":"po_id","pattern":null,"parent_type":"doctype.order"}]'::jsonb,
   ARRAY['quantities with no prices','signed for on receipt',
         'a carrier, vehicle or consignment reference'],
   'goods_receipt', 'default', 'active', 'seed', 0,
   false, ARRAY[]::text[], now(), now())
ON CONFLICT (concept_code, tenant_id) DO NOTHING;
```

```sql
-- deploy/sql/2026-10-04_goods_receipt_doctype_rollback.sql
DELETE FROM proc.bp_document_type
 WHERE concept_code = 'doctype.goods_receipt' AND tenant_id = 'default';
```

- [ ] **Step 6: Apply, then run BOTH the new tests and Task 1's baseline**

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-04_goods_receipt_doctype.sql
PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
  tests/extraction/test_goods_receipt_extraction.py \
  tests/extraction/test_goods_receipt_classification_baseline.py \
  tests/services/concepts/ -v
```
Expected: all PASS. **If the baseline test fails, an alias has stolen a document** — read which, and either narrow the alias or record the move as a deliberate, explained change. Do not update the baseline to make it green.

- [ ] **Step 7: Commit** — `feat(extraction): the vocabulary learns the goods receipt`

---

## Task 4: Extraction, with the no-price guard

**Files:**
- Create: `extraction_schemas/goods_receipt.yaml` — the top-level schema directory `yaml_schema/loader.py:11` resolves as `SCHEMAS_DIR`, alongside `invoice.yaml`, `purchase_order.yaml`, `quote.yaml`, `contract.yaml`. `load_doc_schema(doc_type)` reads `SCHEMAS_DIR/{doc_type}.yaml`, so the file name must equal the `pipeline_doc_type`.
- Modify: `src/services/extraction/dispatch.py` — route `pipeline_doc_type == "goods_receipt"`
- Test: `tests/extraction/test_goods_receipt_extraction.py`

**Interfaces:**
- Produces: `extract_goods_receipt(document) -> dict` with keys `grn_id, po_id, supplier_name, receipt_date, lines[]`; each line `{line_no, description, quantity_received, quantity_rejected, unit_of_measure, po_line_ref}`.

- [ ] **Step 1: Write the guard test FIRST — it is Review Focus #4**

```python
def test_a_delivery_note_restating_po_prices_populates_no_price_field():
    """Real delivery notes often restate the order's prices for the driver's
    paperwork. The extractor will find them. The receipt record must not keep
    them: this document proves delivery, and a price on it would be read as
    corroboration of a value it never witnessed."""
    text = ("DELIVERY NOTE DN-5521   Against PO 4500018832\n"
            "10 x Widget A   unit price GBP 12.50   line total GBP 125.00\n"
            "Received by: J. Okafor")
    record = extract_goods_receipt(_doc(text))
    flat = json.dumps(record).lower()
    for forbidden in ("12.50", "125.00", "unit_price", "line_total", "currency"):
        assert forbidden not in flat, f"{forbidden} reached the receipt record"
    assert record["lines"][0]["quantity_received"] == 10
```

- [ ] **Step 2: Run it to verify it fails**

Expected: `NameError: extract_goods_receipt is not defined`.

- [ ] **Step 3: Write the schema with no price fields**

```yaml
# extraction_schemas/goods_receipt.yaml
# Name MUST equal the pipeline_doc_type: loader.load_doc_schema() reads
# SCHEMAS_DIR/{doc_type}.yaml, and SCHEMAS_DIR is the repo-root extraction_schemas/.
# No price, amount, total, currency or tax field exists here, deliberately.
# See deploy/sql/2026-10-04_goods_receipt_tables.sql for the reasoning; the
# database enforces the same absence.
fields:
  grn_id:            {type: string, patterns: ["GRN[- ]?(\\d{3,10})", "Delivery Note\\s*(?:No\\.?|#)?\\s*([A-Z0-9-]{3,20})"]}
  po_id:             {type: string, patterns: ["(?:against|ref(?:erence)?|for)\\s+PO\\s*([A-Z0-9-]{4,15})", "P\\.?O\\.?\\s*(?:No\\.?|#)?\\s*(\\d{4,10})"]}
  supplier_name:     {type: string}
  receipt_date:      {type: date}
  delivery_note_ref: {type: string}
  carrier_ref:       {type: string}
  received_by:       {type: string}
lines:
  quantity_received: {type: number, required: true}
  quantity_rejected: {type: number, default: 0}
  unit_of_measure:   {type: string}
  description:       {type: string}
  po_line_ref:       {type: string}
```

- [ ] **Step 4: Wire the pipeline in `dispatch.py`**

Beside the existing `pipeline_doc_type` branches, add the `goods_receipt` arm writing to `proc.bp_goods_receipt_raw` and `..._line_items_raw` through `persistence.write_raw`. No new persistence code: the tables mirror the invoice family precisely so the existing writer applies.

- [ ] **Step 5: Run the tests**

Expected: PASS, including Step 1's guard.

- [ ] **Step 6: Prove the guard fails**

Add `unit_price: {type: number}` to the schema's `lines`, re-run Step 1's test, confirm **red**, remove it, confirm green.
Expected: FAIL then PASS. A guard that cannot go red is not protecting the rule.

- [ ] **Step 7: Commit** — `feat(extraction): read a goods receipt, and no price on it`

---

## Task 5: Link the receipt to its purchase order

**Files:**
- Modify: `src/services/extraction/dispatch.py` (promotion arm)
- Test: `tests/extraction/test_goods_receipt_linking.py`

**Interfaces:**
- Consumes: Task 4's extracted record.
- Produces: `bp_goods_receipt_trgt` rows carrying `po_id` and `deal_id`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_receipt_citing_a_po_lands_on_that_po_s_deal():
    """The receipt's whole value is that it attaches to the order. A receipt
    that promotes without a deal_id is a document nobody will ever find."""
    _seed_po("4500018832", deal_id="DEALV2-000123")
    promote_goods_receipt(_receipt(grn_id="GRN-5521", po_id="4500018832"))
    row = _fetch_trgt("GRN-5521")
    assert row["po_id"] == "4500018832"
    assert row["deal_id"] == "DEALV2-000123"

def test_a_receipt_whose_po_does_not_exist_stays_in_stg():
    promote_goods_receipt(_receipt(grn_id="GRN-9999", po_id="NOSUCHPO"))
    assert _fetch_trgt("GRN-9999") is None
    assert _fetch_stg("GRN-9999") is not None
```

- [ ] **Step 2: Run to verify both fail.** Expected: FAIL.
- [ ] **Step 3: Implement**, reusing `linking_engine._pick_po` exactly as `two_way_match` does — same normalisation (`_norm_po`), so a receipt and an invoice citing the same PO in different formats reach the same order.
- [ ] **Step 4: Run to verify both pass.**
- [ ] **Step 5: Commit** — `feat(extraction): a goods receipt finds its purchase order`

---

## Task 6: `receipt_basis` — which lines can be received at all

39.5% of PO lines carry a time-based unit. `bp_uom_canonical.dimension` is close but wrong for this: `licence`, `seat` and `module` sit under `count` while being no more deliverable than an hour.

**Files:**
- Create: `deploy/sql/2026-10-04_uom_receipt_basis.sql` (+ rollback)
- Test: `tests/sql/test_uom_receipt_basis.py`

**Interfaces:**
- Produces: `proc.bp_uom_canonical.receipt_basis ∈ {goods_receipt, service_entry, none}`, read by Task 8.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("uom,expected", [
    ("each", "goods_receipt"), ("box", "goods_receipt"), ("tonne", "goods_receipt"),
    ("metre", "goods_receipt"), ("case", "goods_receipt"), ("pack", "goods_receipt"),
    ("hour", "service_entry"), ("day", "service_entry"), ("month", "service_entry"),
    ("licence", "service_entry"),   # counted, but nobody takes delivery of one
    ("seat", "service_entry"), ("module", "service_entry"),
])
def test_every_unit_declares_whether_it_can_be_received(uom, expected):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT receipt_basis FROM proc.bp_uom_canonical WHERE uom_code=%s", (uom,))
        assert cur.fetchone()[0] == expected

def test_no_unit_is_left_unclassified():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT uom_code FROM proc.bp_uom_canonical WHERE receipt_basis IS NULL")
        assert cur.fetchall() == [], "an unclassified unit silently skips the match"
```

- [ ] **Step 2: Run to verify it fails.** Expected: `UndefinedColumn: receipt_basis`.

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-10-04_uom_receipt_basis.sql
-- dimension describes the PHYSICAL MEASURE; receipt_basis describes whether a
-- human can take delivery of it. They disagree on licence, seat and module --
-- all dimension 'count', none of them deliverable -- which is why this is a
-- separate column and not a view over dimension.
ALTER TABLE proc.bp_uom_canonical ADD COLUMN IF NOT EXISTS receipt_basis text;

UPDATE proc.bp_uom_canonical SET receipt_basis = 'goods_receipt'
 WHERE dimension IN ('count','mass','length','volume') AND receipt_basis IS NULL;
UPDATE proc.bp_uom_canonical SET receipt_basis = 'service_entry'
 WHERE dimension = 'time' AND receipt_basis IS NULL;
-- the intangible counts, corrected by hand
UPDATE proc.bp_uom_canonical SET receipt_basis = 'service_entry'
 WHERE uom_code IN ('licence','license','seat','module','subscription');
-- the 18 rows whose dimension is NULL are extraction noise that reached a
-- reference table ('30 days from invoice', 'implementation (one-off, fixed) —
-- £58,000.00'). They are not units; they are retired, not classified.
UPDATE proc.bp_uom_canonical SET receipt_basis = 'none', status = 'retired'
 WHERE dimension IS NULL;

ALTER TABLE proc.bp_uom_canonical
  ADD CONSTRAINT ck_bp_uom_canonical_receipt_basis
  CHECK (receipt_basis IN ('goods_receipt','service_entry','none')) NOT VALID;
```

```sql
-- deploy/sql/2026-10-04_uom_receipt_basis_rollback.sql
ALTER TABLE proc.bp_uom_canonical DROP CONSTRAINT IF EXISTS ck_bp_uom_canonical_receipt_basis;
ALTER TABLE proc.bp_uom_canonical DROP COLUMN IF EXISTS receipt_basis;
```

- [ ] **Step 4: Apply and run.** Expected: PASS, 13 tests.
- [ ] **Step 5: Commit** — `feat(extraction): units declare whether they can be received`

---

## Task 7: Receipt lines assign to PO lines

**Files:**
- Modify: `src/services/extraction/two_way_match.py` — no logic change; `_assign_lines` becomes importable under a stable name
- Create: `src/services/extraction/three_way_match.py`
- Test: `tests/extraction/test_three_way_match.py`

**Interfaces:**
- Consumes: `two_way_match._assign_lines(line_items, po_lines, ...) -> dict[int, dict]`
- Produces: `assign_receipt_lines(receipt_lines, po_lines) -> dict[int, dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction/test_three_way_match.py
from src.services.extraction import three_way_match as twm3

def test_receipt_lines_assign_to_the_po_lines_they_describe():
    po = [{"line_no": 1, "description": "FORD Focus 1.9TDI, 100 HP", "quantity": 10,
           "unit_of_measure": "each"},
          {"line_no": 2, "description": "Floor mats, set", "quantity": 10,
           "unit_of_measure": "set"}]
    rec = [{"line_no": 1, "description": "FORD Focus 1.9TDI", "quantity_received": 6,
            "unit_of_measure": "each"}]
    assigned = twm3.assign_receipt_lines(rec, po)
    assert assigned[0]["line_no"] == 1
```

- [ ] **Step 2: Run to verify it fails.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement by delegation, not duplication**

```python
# src/services/extraction/three_way_match.py
"""The real three-way match: PO + goods receipt + invoice, on quantity.

two_way_match compares an invoice to its order by value. This compares three
documents by quantity, which is the only comparison that can prove delivery.
The PO line is the spine: invoice lines and receipt lines both assign to it,
and the sums are compared per line over the whole document set -- because two
invoices that each pass alone can together bill more than was received, which
is the lesson two_way_match._check_po_consumed_as_a_set already encodes.
"""
from src.services.extraction.two_way_match import _assign_lines

RECEIPT_LINE_PROFILE = "receipt_line_po_line"

def assign_receipt_lines(receipt_lines, po_lines):
    """Receipt lines onto PO lines, by the same machinery the invoice uses.

    Delegation, not a parallel implementation: a second line-matcher would
    drift from the first and the two sides of the comparison would stop
    agreeing about which PO line they mean.
    """
    normalised = [{**l, "quantity": l.get("quantity_received")} for l in receipt_lines]
    return _assign_lines(normalised, po_lines, profile=RECEIPT_LINE_PROFILE)
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Write the Review Focus #1 test — mismatched units**

```python
def test_a_receipt_in_a_different_unit_is_unverifiable_not_a_finding():
    """box against each is the likeliest real failure and the corpus cannot
    size it. Refusing is correct; guessing a conversion is not, and reporting a
    shortfall that is really a unit difference would be worse than silence."""
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "box"}]
    rec = [{"line_no": 1, "description": "Widget A", "quantity_received": 100,
            "unit_of_measure": "each"}]
    inv = [{"line_no": 1, "description": "Widget A", "quantity": 100,
            "unit_of_measure": "each"}]
    result = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert result.findings == []
    assert result.unverifiable == [{"po_line": 1, "reason": "UNVERIFIABLE_UOM"}]
```

- [ ] **Step 6: Run — expect FAIL (`check` not defined). Task 8 implements it.** Leave this test red and say so in the commit; Task 8's Step 4 turns it green.
- [ ] **Step 7: Commit** — `feat(extraction): receipt lines assign to purchase-order lines`

---

## Task 8: The match, and the three findings

**Files:**
- Modify: `src/services/extraction/three_way_match.py`
- Modify: `src/services/extraction/dispatch.py` (call it where `check_against_po` is called)
- Create: `deploy/sql/2026-10-04_receipt_tolerances_policy.sql` (+ rollback)
- Test: `tests/extraction/test_three_way_match.py`

**Interfaces:**
- Produces: `check(po_lines, receipt_lines, invoice_lines) -> MatchResult(findings, unverifiable, assessed)`

- [ ] **Step 1: Write the core failing test**

```python
def test_billed_more_than_received_raises_the_finding_this_exists_for():
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    rec = [{"line_no": 1, "description": "Widget A", "quantity_received": 6,
            "unit_of_measure": "each"}]
    inv = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["BILLED_NOT_RECEIVED"]
    assert f[0]["ordered"] == 10 and f[0]["received"] == 6 and f[0]["billed"] == 10

def test_billed_less_than_received_is_silent():
    """Partial invoicing is normal procurement. Crying wolf on it is how a
    check gets ignored -- the same reasoning two_way_match applies to value."""
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    rec = [{"line_no": 1, "description": "Widget A", "quantity_received": 10,
            "unit_of_measure": "each"}]
    inv = [{"line_no": 1, "description": "Widget A", "quantity": 4, "unit_of_measure": "each"}]
    assert twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings == []
```

- [ ] **Step 2: Run to verify both fail.** Expected: FAIL.

- [ ] **Step 3: Write the Review Focus #3 test — the set-level case**

```python
def test_two_invoices_each_fitting_alone_raise_once_together():
    """6 received; two invoices of 4 each. Either alone is under. Together they
    bill 8. Checking one document at a time cannot see this, which is exactly
    the hole the value check had to close."""
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    rec = [{"line_no": 1, "description": "Widget A", "quantity_received": 6,
            "unit_of_measure": "each"}]
    inv = [{"line_no": 1, "description": "Widget A", "quantity": 4, "unit_of_measure": "each",
            "invoice_id": "INV-1"},
           {"line_no": 1, "description": "Widget A", "quantity": 4, "unit_of_measure": "each",
            "invoice_id": "INV-2"}]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert len(f) == 1 and f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["billed"] == 8
```

- [ ] **Step 4: Implement the aggregation**

```python
def check(po_lines, receipt_lines, invoice_lines, limits=None):
    """Compare ordered / received / billed per PO line, over the whole set."""
    # governed_limits.limit(policy_identifier, rule) -- raises when the rule is
    # absent, which is the point. A None return means the policy states null
    # ("no limit"), which is NOT the same as the rule being missing.
    limits = limits or {
        k: governed_limits.limit("receipt_tolerances", k)
        for k in ("over_delivery_pct", "billed_over_received_qty")
    }
    rec_by_po = assign_receipt_lines(receipt_lines, po_lines)
    inv_by_po = _assign_lines(invoice_lines, po_lines, profile=INVOICE_LINE_PROFILE)

    findings, unverifiable, assessed = [], [], []
    for idx, po in enumerate(po_lines):
        basis = _receipt_basis(po.get("unit_of_measure"))
        if basis != "goods_receipt":
            unverifiable.append({"po_line": po["line_no"], "reason": "UNVERIFIABLE_BY_RECEIPT"})
            continue
        recs = [r for i, r in rec_by_po.items() if r is po]
        invs = [v for i, v in inv_by_po.items() if v is po]
        if not _units_comparable(po, recs, invs):
            unverifiable.append({"po_line": po["line_no"], "reason": "UNVERIFIABLE_UOM"})
            continue

        ordered  = _f(po.get("quantity")) or 0.0
        received = sum((_f(r.get("quantity_received")) or 0.0)
                       - (_f(r.get("quantity_rejected")) or 0.0) for r in recs)
        billed   = sum(_f(v.get("quantity")) or 0.0 for v in invs)
        assessed.append(po["line_no"])

        if billed > 0 and not recs:
            findings.append(_finding("NOTHING_RECEIVED", po, ordered, received, billed))
        elif billed > received + limits["billed_over_received_qty"]:
            findings.append(_finding("BILLED_NOT_RECEIVED", po, ordered, received, billed))
        if received > ordered * (1 + limits["over_delivery_pct"]):
            findings.append(_finding("OVER_DELIVERED", po, ordered, received, billed))
    return MatchResult(findings, unverifiable, assessed)
```

Run Steps 1, 3 and Task 7 Step 5. Expected: all PASS.

- [ ] **Step 5: Write the Review Focus #2 test — absence is not failure**

```python
def test_a_po_line_with_no_receipt_is_not_assessed_not_failed():
    """Missing paperwork must not manufacture a failure rate. A line nobody
    sent a GRN for has not failed the match; it has not been checked."""
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    r = twm3.check(po_lines=po, receipt_lines=[], invoice_lines=[])
    assert r.findings == [] and r.assessed == []
```

- [ ] **Step 6: Write the Review Focus #5 test — rejected goods**

```python
def test_goods_delivered_and_refused_were_not_received():
    po  = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    rec = [{"line_no": 1, "description": "Widget A", "quantity_received": 10,
            "quantity_rejected": 4, "unit_of_measure": "each"}]
    inv = [{"line_no": 1, "description": "Widget A", "quantity": 10, "unit_of_measure": "each"}]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["received"] == 6
```

- [ ] **Step 7: Tolerances into policy**

```sql
-- deploy/sql/2026-10-04_receipt_tolerances_policy.sql
INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details,
                            policy_linked_agents, version, policy_status)
VALUES ('ReceiptTolerancePolicy', 'extraction',
        'How far delivery may differ from order and invoice before it is a finding',
        '{"policy_identifier":"receipt_tolerances",
          "rules":{"over_delivery_pct":0.00,
                   "billed_over_received_qty":0,
                   "uom_conversion_required":true}}'::jsonb,
        'data_extraction', 1, 1)
ON CONFLICT DO NOTHING;
```
A missing limit **raises**, per `project_governed_limits`. Do not add a code-side default.

- [ ] **Step 8: Prove the guard fails**

Change `billed > received + tol` to `billed > received + 1e9`, run Steps 1/3/6, confirm all **red**, restore, confirm green.

- [ ] **Step 9: Run the whole extraction suite** — `pytest tests/extraction/ -q`. Expected: no new failures against the six already failing at HEAD.
- [ ] **Step 10: Commit** — `feat(extraction): a real three-way match, on quantity`

---

## Task 9: The view columns

**Files:**
- Create: `deploy/sql/2026-10-04_deal_overview_three_way.sql` (+ rollback)
- Test: `tests/sql/test_deal_overview_three_way.py`

- [ ] **Step 1: Write the failing test**

```python
def test_value_reconciled_reproduces_todays_three_way_match_exactly():
    """The rename must not change a single deal's answer."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview
                        WHERE value_reconciled IS DISTINCT FROM three_way_match""")
        assert cur.fetchone()[0] == 0

def test_three_way_matched_is_null_where_no_receipt_exists():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview d
                        WHERE d.three_way_matched IS NOT NULL
                          AND NOT EXISTS (SELECT 1 FROM proc.bp_goods_receipt_trgt g
                                           WHERE g.deal_id = d.deal_id)""")
        assert cur.fetchone()[0] == 0, "a deal with no receipt was given a verdict"
```

- [ ] **Step 2: Run to verify both fail.** Expected: `UndefinedColumn`.
- [ ] **Step 3: Write the migration** — `CREATE OR REPLACE VIEW` adding `value_reconciled` (today's expression verbatim) and `three_way_matched` (NULL when the deal has no receipt row). **`three_way_match` stays**, untouched, until Task 10 has moved every reader.
- [ ] **Step 4: Apply and run.** Expected: PASS.
- [ ] **Step 5: Commit** — `feat(reporting): the deal overview separates reconciliation from the three-way match`

---

## Task 10: Move the readers, then drop the old column

**Files:**
- Modify: `src/services/rga/builders/board_paper.py:43`, `exec_procurement_summary.py:55-56`, `src/services/analysis_findings.py:37`, `src/agents/opportunity_miner_agent.py:5712`
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/engine.js`
- Modify: `tests/sql/test_bp_deal_overview_reconciliation_sql.py`
- Create: `deploy/sql/2026-10-04_deal_overview_drop_three_way_match.sql` (+ rollback)

- [ ] **Step 1: Write the failing test**

```python
def test_no_source_file_reads_the_old_column():
    import subprocess
    hits = subprocess.run(["grep","-rn","three_way_match","src/","tests/sql/"],
                          capture_output=True, text=True).stdout
    allowed = ("two_way_match.py",)  # the pinned profile_registry_version only
    offenders = [l for l in hits.splitlines() if not any(a in l for a in allowed)]
    assert offenders == [], "\n".join(offenders)
```

- [ ] **Step 2: Run to verify it fails**, listing the five readers.
- [ ] **Step 3: Move each reader to `value_reconciled`**, and add the board paper's second line reading `three_way_matched`, rendering **"not assessed"** when NULL.
- [ ] **Step 4: Update the UI** — `engine.js`, same two columns. The UI repo is shared and dirty; check `git status` and stage filtered hunks.
- [ ] **Step 5: Run.** Expected: PASS, plus `pytest tests/services/rga/ -q` green.
- [ ] **Step 6: Drop `three_way_match` from the view**, apply, re-run.
- [ ] **Step 7: Commit** — two commits, one per repo.

---

## Task 11: Deploy to bp_sqldb, and prove it on the running server

`bp_sqldb` has run migrations behind before — the governance tables were eight behind as recently as 2026-09-15, and a missing index there cost 65 days of findings. **Deployment to both databases is a prerequisite, not a follow-up.**

- [ ] **Step 1: Apply all five migrations to `bp_sqldb`,** in order: tables → doctype → uom → view → drop.
- [ ] **Step 2: Assert both databases agree**

```python
def test_both_databases_carry_the_same_goods_receipt_schema():
    for dsn in (TESTDB, SQLDB):
        cols = _columns(dsn, "bp_goods_receipt_trgt")
        assert "deal_id" in cols and not any(_is_price(c) for c in cols)
```

- [ ] **Step 3: Restart procwise and check the route count**

```bash
# Never `pkill -f uvicorn` -- it kills other sessions' servers.
sudo systemctl restart procwise && sleep 20
curl -s localhost:8000/openapi.json | ./venv/bin/python -c "import json,sys;print(len(json.load(sys.stdin)['paths']),'routes')"
```
Expected: no fewer routes than before.

- [ ] **Step 4: Upload a real goods receipt from phase 0** through the running server, against the live database. Confirm: it types as `doctype.goods_receipt`, extracts with no price field populated, links to its PO, reaches `_trgt` with a `deal_id`, and that a deliberately over-billed invoice against it raises `BILLED_NOT_RECEIVED` in the Action Centre.
- [ ] **Step 5: Confirm the board paper** shows **"not assessed"** for a deal with no receipt, and a verdict for the phase-0 deal.
- [ ] **Step 6: Commit** — `docs(extraction): three-way match deployment and live verification record`

---

## Plan Self-Review

**Spec coverage.** §4 → Tasks 2, 3, 4. §5 → Tasks 7, 8. §6 → Task 6, and Task 8 Step 4's `UNVERIFIABLE_BY_RECEIPT` branch. §7 → Task 8 Step 7. §8 → Task 8 (findings carry the PO line, three quantities and both `source_file`s). §9 → Tasks 9, 10. §10 → the task order. §11 → each task's guard-fails step and Task 11 Step 4. §12 out-of-scope items appear in no task, correctly. §13's gate is the STOP block.

**Placeholders.** None. Every code step carries the code; every test step carries the assertion.

**Type consistency.** `assign_receipt_lines` (Task 7) is called by `check` (Task 8) under that name. `MatchResult(findings, unverifiable, assessed)` is used identically in Tasks 7 and 8. `receipt_basis` values match between Task 6's migration, its test and Task 8's `_receipt_basis`.

**Review Focus coverage.** #1 Task 7 Step 5 · #2 Task 8 Step 5 · #3 Task 8 Step 3 · #4 Task 4 Step 1 · #5 Task 8 Step 6. All five pinned.

**Known gap, stated rather than hidden.** Tasks 7 and 8 are tested entirely on fixtures until phase 0 supplies real receipts. Task 11 Step 4 is the only step that exercises the feature on a real document, and it cannot run without the gate. If phase 0 fails, Tasks 1 and 6 still have standalone value (a measured classification baseline; a correctly classified unit table) and Tasks 2–5, 7–11 should not be built.
