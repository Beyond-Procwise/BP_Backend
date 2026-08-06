# Phase 1a — Extraction Schema Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extraction pipeline capture the commercial fields it currently never looks for — unit of measure on invoice and PO lines, the contract link on every transaction, and the contract's commercial terms — so that Phase 1b has real data to carry across the seam instead of a richly-typed record full of NULLs.

**Architecture:** Every field addition is two artefacts in a fixed order — a forward+rollback migration that creates the column on `_raw`, `_stg` and `_trgt`, then a field entry in `extraction_schemas/<doc>.yaml` binding to it. The order is not stylistic: `load_doc_schema` validates every declared `db_column` against the live database on every load and raises `SchemaDriftError` if it is missing, so a YAML-first commit breaks all extraction the moment the process restarts. No Python changes are needed — `promote()` intersects the built record against the physical table columns, and `table_extractor` maps table headers to fields purely from each field's `canonical_labels`, so both pick up new fields automatically.

**Tech Stack:** Python 3.12, Pydantic v2 (`src/services/extraction_v3/yaml_schema/loader.py`), psycopg2, PostgreSQL (`proc` schema on `procwisemvpdb01…rds.amazonaws.com`), pytest, YAML schemas in `extraction_schemas/`.

## Global Constraints

- **Migration before YAML, always.** `loader.py:110–128` `_verify_db_consistency_with_conn` raises `SchemaDriftError` for any `db_column` not present on the declared table. Landing YAML first breaks every extraction run.
- **Every migration ships forward and reversible**, as `deploy/sql/2026-08-06_<name>.sql` and `deploy/sql/2026-08-06_<name>_rollback.sql`, following the existing convention (`BEGIN; … COMMIT;`, additive, idempotent via `IF NOT EXISTS`).
- **Every migration is applied to BOTH `bp_sqldb` and `bp_testdb`.** `.env` points at `bp_testdb`; `bp_sqldb` is the other live database and holds the 64,118-row provenance corpus. A schema present in one and absent from the other will raise `SchemaDriftError` on whichever host loads the other.
- **Every new field is `required: false`.** A `required: true` field produces a `missing_required` discrepancy that blocks promotion (`persistence.py` `Discrepancy`, `blocks_promotion=True`) on every existing document.
- **`type:` is a closed set** — `string | iso_date | money | decimal | address | postcode | currency` (`loader.py:38`). There is no integer type; counts and percentages use `decimal`.
- **A `decimal` field's regex must capture the bare number in group 1.** `bind_typed` (`type_binder.py:58–68`) calls `parse_amount`, which returns `None` for `"3%"` and `"36 months"`, then falls back to `float(raw)` which raises — producing a `type_bind_error` and discarding the value. Verified: `parse_amount('3%') is None`, `parse_amount('3') == 3.00`.
- **No arithmetic in the extractor.** A term stated in years is not converted to months; an escalator stated as a multiplier is not converted to a percentage. Capture what the document says or capture nothing. Derived values are Phase 1b's job.
- **Grounding holds by construction.** `pattern_extractor._emit_pattern_hits` takes `vm.group(1) if vm.lastindex else vm.group(0)` and uses that literal text as the `Span.text`, so a capture group narrower than the match is still a substring of `full_text`.
- **No new tables in this phase**, therefore the "`tenant_id` on new tables only" decision has no effect here. It first applies in Phase 1b, to `commercial_fact` and `constraint`.
- **No GPSS codes in this phase.** B3 is unresolved; Phase 1a binds to the physical column names that already exist (`cost_centre_id`, `parent_contract_id`, `is_amendment`). Phase 1b adds `gpss_code` once the dictionary question is settled.
- **Run tests with the environment loaded:** `set -a && . ./.env && set +a && ./venv/bin/python -m pytest …`. Tests that call `load_doc_schema` open a real DB connection.
- **Commit style:** no `Co-Authored-By` lines. Work stays on `Development`.

---

## The regexes in this plan are pre-validated

Every pattern in Tasks 5 and 6 was run against the Task 5 fixture text before this plan was written. All eleven fields extract their expected value, `term_months` correctly captures nothing from a term stated in years, and the `contract_id` patterns correctly capture nothing from the boilerplate sentence "This contract is subject to our standard terms and conditions."

So these tests should pass on the first run. If one fails, the cause is a difference between this plan's assumptions and the code — a changed `pattern_extractor` capture rule, a schema that failed to load, YAML indentation — **not** normal regex iteration. Investigate before editing the pattern.

The Task 3 transaction-level `contract_id` patterns are the exception: they were validated against the inline strings in that task's test, not against a real PO or invoice. Expect iteration there.

---

## Known limitation, stated up front

`proc.bp_contracts` has **0 rows** in both databases and there are **no contract fixture documents** anywhere in the repo (`tests/extraction_v3/fixtures/` contains invoices only). Tasks 5 and 6 therefore validate contract patterns against **synthetic plain-text fixtures**, not real contract PDFs. This is honest and still worth doing — `run_pattern_extractor` operates on `ParsedDocument.full_text`, so a text fixture exercises the identical code path, and the columns and schema are then ready the moment a real corpus arrives. It is **not** evidence that the patterns work on real-world contract prose. That validation is blocked on B1 and must be re-run when a contract corpus lands.

---

## File Structure

| File | Responsibility |
|---|---|
| `deploy/sql/2026-08-06_transaction_contract_link.sql` (+ `_rollback`) | `contract_id` on quote and invoice `_raw`/`_stg`/`_trgt` |
| `deploy/sql/2026-08-06_contract_commercial_terms.sql` (+ `_rollback`) | escalator / term / billing / amendment / version columns on `bp_contract_raw` and `bp_contracts` |
| `extraction_schemas/invoice.yaml` | add `unit_of_measure` line field, `contract_id` header field |
| `extraction_schemas/purchase_order.yaml` | add `unit_of_measure` line field, `contract_id` header field |
| `extraction_schemas/quote.yaml` | add `contract_id` header field |
| `extraction_schemas/contract.yaml` | add 3 existing-column fields + 7 new-column fields, and L1 patterns (this schema has none today) |
| `tests/extraction/test_schema_db_consistency.py` (new) | the drift guard — every declared column exists, and a bogus one is rejected |
| `tests/extraction/test_line_uom_schema.py` (new) | UoM field present and header-mappable on invoice + PO |
| `tests/extraction/test_contract_link_schema.py` (new) | `contract_id` present on all three transaction schemas, patterns fire |
| `tests/extraction/fixtures/contracts/*.txt` (new) | synthetic contract prose fixtures |
| `tests/extraction/test_contract_l1_parity.py` (new) | L1 patterns extract expected values from the text fixtures |
| `scripts/field_coverage.py` (new) | field-coverage report from `bp_extraction_provenance_v3` — the phase's acceptance instrument |
| `docs/remediation/01a_extraction_schema_extension.md` (new) | decision record |

---

## Task 1: Schema/DB drift guard

The guard that protects every later task. Without it, a mistyped `db_column` in Task 2 fails at runtime in production rather than in CI.

**Files:**
- Create: `tests/extraction/test_schema_db_consistency.py`

**Interfaces:**
- Consumes: `src.services.extraction_v3.yaml_schema.loader.load_all_schemas`, `load_doc_schema_path`, `SchemaDriftError`
- Produces: nothing consumed by later tasks; it is a standing guard.

- [ ] **Step 1: Write the failing test**

```python
"""Guard: every db_column declared in extraction_schemas/*.yaml exists in the DB.

This is the reason migrations must land before YAML. loader._verify_db_consistency
raises SchemaDriftError on a missing column, and it runs on EVERY schema load —
so a YAML-first commit takes extraction down at the next process restart, not at
the next test run. This test moves that failure into CI.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction_v3.yaml_schema.loader import (  # noqa: E402
    SchemaDriftError,
    load_all_schemas,
    load_doc_schema_path,
)

EXPECTED_DOC_TYPES = {"contract", "invoice", "purchase_order", "quote"}


def test_every_schema_loads_against_the_live_database():
    schemas = load_all_schemas()
    assert EXPECTED_DOC_TYPES.issubset(set(schemas)), (
        f"missing schemas: {EXPECTED_DOC_TYPES - set(schemas)}"
    )
    for name, schema in schemas.items():
        assert schema.fields, f"{name}.yaml declared no fields"


def test_guard_rejects_a_column_that_does_not_exist(tmp_path):
    """Prove the guard actually fails. A guard that has never been seen red is
    not a guard."""
    bogus = tmp_path / "quote.yaml"
    bogus.write_text(textwrap.dedent("""
        doc_type: quote
        db_table: proc.bp_quote_stg
        db_lines_table: null
        fields:
          - name: definitely_not_a_column
            type: string
            required: false
            db_column: definitely_not_a_column
            canonical_labels: ["Nope"]
    """).strip())

    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bogus)
    assert "definitely_not_a_column" in str(exc.value)


def test_guard_rejects_a_missing_lines_table_column(tmp_path):
    bogus = tmp_path / "quote.yaml"
    bogus.write_text(textwrap.dedent("""
        doc_type: quote
        db_table: proc.bp_quote_stg
        db_lines_table: proc.bp_quote_line_items_stg
        fields:
          - name: quote_id
            type: string
            required: true
            db_column: quote_id
            canonical_labels: ["Quote No"]
        line_items:
          primary_extractor: qwen_vlm
          fields:
            - name: not_a_line_column
              type: string
              required: false
              db_column: not_a_line_column
              canonical_labels: ["Nope"]
    """).strip())

    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bogus)
    assert "not_a_line_column" in str(exc.value)
```

- [ ] **Step 2: Run the test to verify the guard fires**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_schema_db_consistency.py -v
```

Expected: all three PASS on current `main`. The two `pytest.raises` tests are the proof the guard works; if either fails, `_verify_db_consistency` is not doing what this plan assumes and **you must stop and report that** rather than proceeding — every later task depends on it.

- [ ] **Step 3: Commit**

```bash
git add tests/extraction/test_schema_db_consistency.py
git commit -m "test(extraction): guard that every declared db_column exists in the database"
```

---

## Task 2: Unit of measure on invoice and PO line items

The columns already exist on all three layers (`bp_invoice_line_items_raw/stg/trgt.unit_of_measure`, `bp_po_line_items_raw/stg/trgt.unit_of_measure`) — verified live. **No migration.** `quote.yaml` already declares this field; invoice and PO do not, so the extractor never looks for it and `table_extractor` cannot map a "UOM" column header to anything.

**Files:**
- Modify: `extraction_schemas/invoice.yaml` (append to `line_items.fields`)
- Modify: `extraction_schemas/purchase_order.yaml` (append to `line_items.fields`)
- Create: `tests/extraction/test_line_uom_schema.py`

**Interfaces:**
- Consumes: `PatternRegistry`, `load_doc_schema`, `table_extractor._header_to_field`
- Produces: line-item field `unit_of_measure` on doc types `invoice` and `purchase_order`, `db_column="unit_of_measure"`, `type="string"`.

- [ ] **Step 1: Write the failing test**

```python
"""unit_of_measure must be declared on invoice and PO line items, and a table
header cell reading 'UOM' or 'Unit' must map to it — without stealing the
'Unit Price' column, which _header_to_field resolves by longest-label-wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.engineered.table_extractor import _header_to_field  # noqa: E402
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema  # noqa: E402

DOC_TYPES = ["invoice", "purchase_order", "quote"]


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_line_items_declare_unit_of_measure(doc_type):
    schema = load_doc_schema(doc_type)
    by_name = {f.name: f for f in schema.line_items.fields}
    assert "unit_of_measure" in by_name, f"{doc_type} line items must declare unit_of_measure"
    f = by_name["unit_of_measure"]
    assert f.db_column == "unit_of_measure"
    assert f.type == "string"
    assert f.required is False, "a new field must never block promotion"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("header", ["UOM", "Unit", "Unit of Measure", "U/M", "Measure"])
def test_uom_headers_map_to_unit_of_measure(doc_type, header):
    schema = load_doc_schema(doc_type)
    assert _header_to_field(header, schema.line_items.fields) == "unit_of_measure"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("header", ["Unit Price", "Unit Cost", "Price per Unit"])
def test_uom_does_not_steal_the_unit_price_column(doc_type, header):
    schema = load_doc_schema(doc_type)
    assert _header_to_field(header, schema.line_items.fields) == "unit_price"
```

- [ ] **Step 2: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_line_uom_schema.py -v
```

Expected: the `invoice` and `purchase_order` cases FAIL with `assert 'unit_of_measure' in by_name`. The `quote` cases PASS (it already has the field) — that is the control proving the test is wired correctly.

- [ ] **Step 3: Add the field to both schemas**

Append this block to `line_items.fields` in **both** `extraction_schemas/invoice.yaml` and `extraction_schemas/purchase_order.yaml`. It is copied verbatim from the `unit_of_measure` entry already in `quote.yaml`, so all three doc types share one definition and one label set.

```yaml
    - name: unit_of_measure
      type: string
      required: false
      db_column: unit_of_measure
      canonical_labels:
        - UOM
        - Unit
        - Unit of Measure
        - U/M
        - Measure
      extractors:
        - qwen_vlm
      judge:
        tiebreaker: true
        grounded_last_resort: true
        ner_type_check: "none"
      invariants: []
```

Indentation: `line_items.fields` entries are indented four spaces for the `- name:` key in these files. Match the surrounding entries exactly — YAML is whitespace-significant and the loader will raise a Pydantic validation error, not a helpful one, if the nesting is wrong.

- [ ] **Step 4: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_line_uom_schema.py tests/extraction/test_schema_db_consistency.py -v
```

Expected: all PASS.

- [ ] **Step 5: Check nothing else regressed**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/ -v
```

Expected: no new failures versus the pre-change baseline. Capture the baseline first if you have not:
`git stash list` is not an option here — another session shares this checkout. If you need a clean baseline, use `git worktree add`.

- [ ] **Step 6: Commit**

```bash
git add extraction_schemas/invoice.yaml extraction_schemas/purchase_order.yaml tests/extraction/test_line_uom_schema.py
git commit -m "feat(extraction): capture unit_of_measure on invoice and purchase-order line items"
```

---

## Task 3: The contract link on every transaction

`bp_purchase_order_raw/stg/trgt.contract_id` already exists (all `text`) and is populated on **0 of 5,041 rows** because no schema declares it. Quote and invoice have no such column at any layer.

**Files:**
- Create: `deploy/sql/2026-08-06_transaction_contract_link.sql`
- Create: `deploy/sql/2026-08-06_transaction_contract_link_rollback.sql`
- Modify: `extraction_schemas/purchase_order.yaml`, `extraction_schemas/quote.yaml`, `extraction_schemas/invoice.yaml` (header `fields`)
- Create: `tests/extraction/test_contract_link_schema.py`

**Interfaces:**
- Consumes: `load_doc_schema`, `PatternRegistry`, `run_pattern_extractor`
- Produces: header field `contract_id` on doc types `invoice`, `purchase_order`, `quote`, `db_column="contract_id"`, `type="string"`, with the two L1 patterns defined in Step 7: `anchored_contract_no` and `under_agreement`. (Corrected 2026-08-06: an earlier draft of this line also named a third pattern `contract_ref_label`, which Step 7 never defined. Step 7's two patterns cover all five test inputs — `anchored_contract_no` handles the four labelled forms including `Contract Reference:`, `under_agreement` handles the prose form. Step 7 governs.)

- [ ] **Step 1: Write the forward migration**

Create `deploy/sql/2026-08-06_transaction_contract_link.sql`:

```sql
-- 2026-08-06 Phase 1a: the contract link on quote and invoice documents.
--
-- purchase_order already has contract_id on _raw/_stg/_trgt (all text, all NULL,
-- because no extraction schema declared it). Quote and invoice have no such
-- column at any layer, so a quote that cites its governing agreement has
-- nowhere to record it.
--
-- Additive and idempotent. Type is `text` to match bp_purchase_order_*.contract_id
-- and bp_contracts.contract_id, so a future join needs no cast.
BEGIN;

ALTER TABLE proc.bp_quote_raw    ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_quote_stg    ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_quote_trgt   ADD COLUMN IF NOT EXISTS contract_id TEXT;

ALTER TABLE proc.bp_invoice_raw  ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_invoice_stg  ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_invoice_trgt ADD COLUMN IF NOT EXISTS contract_id TEXT;

-- Indexed because every Phase 3.2 baseline-integrity check joins a transaction
-- to its governing contract on this column.
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_contract_id
    ON proc.bp_quote_trgt (contract_id);
CREATE INDEX IF NOT EXISTS ix_bp_invoice_trgt_contract_id
    ON proc.bp_invoice_trgt (contract_id);
CREATE INDEX IF NOT EXISTS ix_bp_purchase_order_trgt_contract_id
    ON proc.bp_purchase_order_trgt (contract_id);

COMMIT;
```

- [ ] **Step 2: Write the rollback migration**

Create `deploy/sql/2026-08-06_transaction_contract_link_rollback.sql`:

```sql
-- Rollback of 2026-08-06_transaction_contract_link.sql.
--
-- Drops the columns this migration added. It does NOT drop
-- ix_bp_purchase_order_trgt_contract_id's column — that column predates this
-- migration — only the index the migration created on it.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_quote_trgt_contract_id;
DROP INDEX IF EXISTS proc.ix_bp_invoice_trgt_contract_id;
DROP INDEX IF EXISTS proc.ix_bp_purchase_order_trgt_contract_id;

ALTER TABLE proc.bp_quote_raw    DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_quote_stg    DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_quote_trgt   DROP COLUMN IF EXISTS contract_id;

ALTER TABLE proc.bp_invoice_raw  DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_invoice_stg  DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_invoice_trgt DROP COLUMN IF EXISTS contract_id;

COMMIT;
```

- [ ] **Step 3: Apply the migration to both databases**

```bash
set -a && . ./.env && set +a
for DB in bp_sqldb bp_testdb; do
  echo "=== $DB ==="
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-08-06_transaction_contract_link.sql
done
```

Expected: `COMMIT` on both, no errors. If `psql` is unavailable, apply with psycopg2 reading the same file — do not hand-retype the SQL.

- [ ] **Step 4: Verify the columns landed on both databases**

```bash
set -a && . ./.env && set +a && ./venv/bin/python - <<'EOF'
import os, psycopg2
for db in ("bp_sqldb", "bp_testdb"):
    c = psycopg2.connect(host=os.environ["DB_HOST"], dbname=db, user=os.environ["DB_USER"],
                         password=os.environ["DB_PASSWORD"], port=os.environ["DB_PORT"])
    cur = c.cursor()
    cur.execute("""select table_name from information_schema.columns
                   where table_schema='proc' and column_name='contract_id'
                   and table_name like 'bp_%' order by table_name""")
    print(db, [r[0] for r in cur.fetchall()])
    c.close()
EOF
```

Expected on both: `bp_contract_raw, bp_contracts, bp_invoice_raw, bp_invoice_stg, bp_invoice_trgt, bp_purchase_order_raw, bp_purchase_order_stg, bp_purchase_order_trgt, bp_quote_raw, bp_quote_stg, bp_quote_trgt`.

- [ ] **Step 5: Write the failing test**

```python
"""contract_id must be declared on all three transaction schemas, and its L1
patterns must fire on the labelled forms that appear on real POs and invoices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.pattern_registry import PatternRegistry, clear_cache  # noqa: E402
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema  # noqa: E402

DOC_TYPES = ["invoice", "purchase_order", "quote"]


def setup_function():
    clear_cache()


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_schema_declares_contract_id(doc_type):
    schema = load_doc_schema(doc_type)
    by_name = {f.name: f for f in schema.fields}
    assert "contract_id" in by_name, f"{doc_type} must declare contract_id"
    f = by_name["contract_id"]
    assert f.db_column == "contract_id"
    assert f.type == "string"
    assert f.required is False
    assert f.patterns, "contract_id needs L1 patterns; the VLM alone has never populated it"


def _hits(doc_type: str, field: str, text: str) -> list[str]:
    """Every value the field's L1 patterns extract from `text`."""
    reg = PatternRegistry(doc_type)
    out: list[str] = []
    for cp in reg.patterns_for(field):  # sorted by prior_confidence desc
        for m in cp.anchor_re.finditer(text):
            window = text[m.end():m.end() + cp.max_span_after_anchor_chars]
            vm = cp.value_re.search(window)
            if vm:
                out.append(vm.group(1) if vm.lastindex else vm.group(0))
    return out


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("text,expected", [
    ("Contract No: MSA-2024-0087", "MSA-2024-0087"),
    ("Contract Number: CTR/2025/119", "CTR/2025/119"),
    ("Agreement No. AGR-4471", "AGR-4471"),
    ("Contract Reference: FRM-2023-88", "FRM-2023-88"),
    ("Issued under Master Agreement MSA-9921", "MSA-9921"),
])
def test_contract_id_patterns_extract_the_identifier(doc_type, text, expected):
    got = _hits(doc_type, "contract_id", text)
    assert expected in got, f"{doc_type}: no pattern extracted {expected!r} from {text!r}"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_contract_id_patterns_do_not_fire_on_boilerplate(doc_type):
    """A false contract link is worse than none — it would attach a transaction
    to an agreement that does not govern it."""
    noise = "This contract is subject to our standard terms and conditions."
    got = _hits(doc_type, "contract_id", noise)
    assert not got, f"{doc_type}: patterns fired on boilerplate: {got}"
```

`PatternRegistry.patterns_for(field)` is a public accessor (`pattern_registry.py:105`) returning a copy of the compiled patterns already ordered by `prior_confidence` descending. Use it; do not reach into `_by_field`.

- [ ] **Step 6: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_link_schema.py -v
```

Expected: FAIL with `assert 'contract_id' in by_name` for all three doc types.

- [ ] **Step 7: Add the field to all three schemas**

Append to the top-level `fields:` list of `extraction_schemas/purchase_order.yaml`, `extraction_schemas/quote.yaml` and `extraction_schemas/invoice.yaml` (two-space indent for `- name:` at the top level in these files — match the surrounding entries):

```yaml
  - name: contract_id
    type: string
    required: false
    db_column: contract_id
    canonical_labels:
      - "Contract No"
      - "Contract Number"
      - "Contract Reference"
      - "Agreement No"
      - "Agreement Number"
      - "Master Agreement"
      - "MSA No"
      - "Framework Agreement"
    patterns:
      - name: anchored_contract_no
        anchor: '(?i)\b(?:contract|agreement|msa|framework(?:\s+agreement)?)\s*(?:number|no\.?|reference|ref\.?|#)\s*[:\-]?\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.90
      - name: under_agreement
        anchor: '(?i)\b(?:issued\s+)?under\s+(?:master\s+)?(?:agreement|contract)\s*[:\-]?\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.82
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []
```

Two deliberate choices, both erring toward silence:

- `grounded_last_resort: false` — the judge must not invent a contract reference when the regexes find none. A wrong contract link silently attaches a transaction to an agreement that does not govern it, and every Phase 3.2 check would then compare against the wrong baseline. Absent is correct here; guessed is not.
- `confidence_threshold: 0.75` (above the 0.70 default) — same reasoning.

- [ ] **Step 8: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_link_schema.py tests/extraction/test_schema_db_consistency.py -v
```

Expected: all PASS. If `test_contract_id_patterns_extract_the_identifier` fails for a specific input, refine the regex in the YAML and re-run — that is the intended iteration loop, matching the note in `tests/extraction/test_l1_parity_invoice.py`. Do not weaken the boilerplate test to make an extraction case pass.

- [ ] **Step 9: Run the full extraction suite**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/ tests/extraction_v3/ -v
```

Expected: no new failures versus baseline.

- [ ] **Step 10: Commit**

```bash
git add deploy/sql/2026-08-06_transaction_contract_link.sql \
        deploy/sql/2026-08-06_transaction_contract_link_rollback.sql \
        extraction_schemas/invoice.yaml extraction_schemas/purchase_order.yaml \
        extraction_schemas/quote.yaml tests/extraction/test_contract_link_schema.py
git commit -m "feat(extraction): capture the governing contract reference on quotes, POs and invoices"
```

---

## Task 4: Contract commercial-term columns (migration only)

`bp_contract_raw` and `bp_contracts` already carry `cost_centre_id`, `parent_contract_id`, `is_amendment`, `business_unit_id`, `renewal_term`, `auto_renew_flag` — all `text`, all NULL. They carry **nothing** for escalation, term length, billing cadence, amendment reference or document version.

This task is the migration alone, so a reviewer can approve the column shape before any extraction logic depends on it.

**Files:**
- Create: `deploy/sql/2026-08-06_contract_commercial_terms.sql`
- Create: `deploy/sql/2026-08-06_contract_commercial_terms_rollback.sql`

**Interfaces:**
- Produces: columns `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, `amendment_ref`, `document_version` on `proc.bp_contract_raw` and `proc.bp_contracts`. Consumed by Task 6's YAML.

- [ ] **Step 1: Write the forward migration**

```sql
-- 2026-08-06 Phase 1a: commercial terms on the contract record.
--
-- Phase 0 established that escalator, term length, billing cadence, amendment
-- reference and document version exist nowhere in this codebase — not in a
-- schema, not in a column, not in a grep. Every Phase 3.2 check (escalator
-- conformance, rate drift against the governing amendment, co-termination) and
-- the Phase 1b CommercialFact Term group need them.
--
-- Columns are added to BOTH bp_contract_raw and bp_contracts because contract
-- promotion goes raw -> bp_contracts directly with no _stg layer
-- (promotion.py:29 maps "contract" to that pair).
--
-- Numeric precision follows the existing bp_contracts convention
-- (total_contract_value is numeric(18,2)). Percentages get (9,4) so a rate of
-- 3.125% survives without rounding.
--
-- NOT converted, NOT derived: term_months records a term the document STATES in
-- months. A term stated in years is left NULL here — converting it would be
-- arithmetic performed by the extractor, and the derived value belongs to
-- Phase 1b where it can carry its own basis and provenance.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS term_months        NUMERIC(9,2),
    ADD COLUMN IF NOT EXISTS billing_frequency  TEXT,
    ADD COLUMN IF NOT EXISTS escalator_pct      NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS escalator_basis    TEXT,
    ADD COLUMN IF NOT EXISTS escalator_cap_pct  NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS amendment_ref      TEXT,
    ADD COLUMN IF NOT EXISTS document_version   TEXT;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS term_months        NUMERIC(9,2),
    ADD COLUMN IF NOT EXISTS billing_frequency  TEXT,
    ADD COLUMN IF NOT EXISTS escalator_pct      NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS escalator_basis    TEXT,
    ADD COLUMN IF NOT EXISTS escalator_cap_pct  NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS amendment_ref      TEXT,
    ADD COLUMN IF NOT EXISTS document_version   TEXT;

-- The amendment chain is walked parent-first by every integrity check.
CREATE INDEX IF NOT EXISTS ix_bp_contracts_parent_contract_id
    ON proc.bp_contracts (parent_contract_id);

COMMIT;
```

- [ ] **Step 2: Write the rollback migration**

```sql
-- Rollback of 2026-08-06_contract_commercial_terms.sql.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_contracts_parent_contract_id;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS term_months,
    DROP COLUMN IF EXISTS billing_frequency,
    DROP COLUMN IF EXISTS escalator_pct,
    DROP COLUMN IF EXISTS escalator_basis,
    DROP COLUMN IF EXISTS escalator_cap_pct,
    DROP COLUMN IF EXISTS amendment_ref,
    DROP COLUMN IF EXISTS document_version;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS term_months,
    DROP COLUMN IF EXISTS billing_frequency,
    DROP COLUMN IF EXISTS escalator_pct,
    DROP COLUMN IF EXISTS escalator_basis,
    DROP COLUMN IF EXISTS escalator_cap_pct,
    DROP COLUMN IF EXISTS amendment_ref,
    DROP COLUMN IF EXISTS document_version;

COMMIT;
```

- [ ] **Step 3: Apply to both databases**

```bash
set -a && . ./.env && set +a
for DB in bp_sqldb bp_testdb; do
  echo "=== $DB ==="
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-08-06_contract_commercial_terms.sql
done
```

- [ ] **Step 4: Verify the rollback actually reverses it**

A rollback that has never been run is not a rollback. Prove it on `bp_testdb` only, then re-apply.

```bash
set -a && . ./.env && set +a
PSQL="PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d bp_testdb -v ON_ERROR_STOP=1"
eval $PSQL -f deploy/sql/2026-08-06_contract_commercial_terms_rollback.sql
eval $PSQL -c "\"select count(*) from information_schema.columns where table_schema='proc' and table_name='bp_contracts' and column_name in ('term_months','escalator_pct','escalator_basis','escalator_cap_pct','billing_frequency','amendment_ref','document_version')\""
# expect 0
eval $PSQL -f deploy/sql/2026-08-06_contract_commercial_terms.sql
eval $PSQL -c "\"select count(*) from information_schema.columns where table_schema='proc' and table_name='bp_contracts' and column_name in ('term_months','escalator_pct','escalator_basis','escalator_cap_pct','billing_frequency','amendment_ref','document_version')\""
# expect 7
```

Expected: `0` then `7`. If the rollback leaves anything behind, fix it before committing.

- [ ] **Step 5: Commit**

```bash
git add deploy/sql/2026-08-06_contract_commercial_terms.sql \
        deploy/sql/2026-08-06_contract_commercial_terms_rollback.sql
git commit -m "feat(db): commercial-term columns on the contract record (escalator, term, billing, amendment)"
```

---

## Task 5: Contract allocation and amendment fields (existing columns)

`contract.yaml` currently declares 17 fields and **zero patterns** — it is the only schema still driven purely by `extractors: [qwen_vlm]`. This task adds the three fields whose columns already exist, and introduces L1 patterns to contract extraction for the first time.

**Files:**
- Modify: `extraction_schemas/contract.yaml`
- Create: `tests/extraction/fixtures/contracts/msa_with_amendment.txt`
- Create: `tests/extraction/fixtures/contracts/msa_with_amendment.expected.json`
- Create: `tests/extraction/test_contract_l1_parity.py`

**Interfaces:**
- Consumes: `run_pattern_extractor`, `PatternRegistry`, migration from Task 4 (not strictly — these three columns predate it, but Task 4 must land first so the schema loads once Task 6 adds the rest)
- Produces: header fields `cost_centre_id`, `parent_contract_id`, `is_amendment` on doc type `contract`; fixture loader helper `_load_fixture(name)` reused by Task 6.

- [ ] **Step 1: Write the synthetic fixture**

Create `tests/extraction/fixtures/contracts/msa_with_amendment.txt`. This is deliberately plain prose, not a rendered PDF — `run_pattern_extractor` works on `full_text`, so the text file exercises the identical code path.

```text
AMENDMENT No. AMD-2025-004

to the MASTER SERVICES AGREEMENT

Contract Number: MSA-2024-0087
Parent Contract: MSA-2024-0001
Document Version: v3.2

Supplier: Northwind Technology Services Limited
Buyer: Assurity Group Holdings plc

Cost Centre: CC-4471
Business Unit: Group Technology

Commencement Date: 01 April 2025
Expiry Date: 31 March 2028
Initial Term: 36 months

Charges shall be invoiced quarterly in advance.

Annual price escalation: 3.5% per annum, indexed to CPI.
Any increase under this clause shall be capped at 5% in any contract year.

Payment Terms: Net 45 days
Governing Law: England and Wales
```

- [ ] **Step 2: Write the expected values**

Create `tests/extraction/fixtures/contracts/msa_with_amendment.expected.json`:

```json
{
  "contract_id": "MSA-2024-0087",
  "parent_contract_id": "MSA-2024-0001",
  "cost_centre_id": "CC-4471",
  "is_amendment": "AMENDMENT",
  "amendment_ref": "AMD-2025-004",
  "document_version": "v3.2",
  "term_months": "36",
  "billing_frequency": "quarterly",
  "escalator_pct": "3.5",
  "escalator_basis": "CPI",
  "escalator_cap_pct": "5"
}
```

Task 5 asserts the first four keys; Task 6 asserts the rest. One fixture, two tasks, so the same document proves both.

- [ ] **Step 3: Write the failing test**

Create `tests/extraction/test_contract_l1_parity.py`:

```python
"""L1 parity for contract.yaml against synthetic prose fixtures.

NOT REAL-WORLD VALIDATION. proc.bp_contracts has 0 rows in both databases and
the repo contains no contract PDFs, so these fixtures are hand-written prose in
the shapes contracts normally use. They prove the regexes do what they claim on
text; they do not prove the regexes survive real contract layout. Re-run this
against a real corpus when blocker B1 is resolved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.pattern_registry import PatternRegistry, clear_cache  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "contracts"


def setup_function():
    clear_cache()


def _load_fixture(name: str) -> tuple[str, dict]:
    text = (FIXTURES / f"{name}.txt").read_text()
    expected = json.loads((FIXTURES / f"{name}.expected.json").read_text())
    return text, expected


def _best_hit(text: str, field: str) -> str | None:
    """Highest-prior pattern that matches, mirroring how L1 orders candidates.

    patterns_for() returns them already sorted by prior_confidence descending
    (PatternRegistry._compile), so the first match is the highest-prior one.
    """
    reg = PatternRegistry("contract")
    for cp in reg.patterns_for(field):
        for m in cp.anchor_re.finditer(text):
            window = text[m.end():m.end() + cp.max_span_after_anchor_chars]
            vm = cp.value_re.search(window)
            if vm:
                return vm.group(1) if vm.lastindex else vm.group(0)
    return None


ALLOCATION_FIELDS = ["contract_id", "parent_contract_id", "cost_centre_id", "is_amendment"]


@pytest.mark.parametrize("field", ALLOCATION_FIELDS)
def test_allocation_fields_extract_from_fixture(field):
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, field) == expected[field]


def test_parent_contract_is_not_confused_with_contract_id():
    """Both are MSA-prefixed identifiers on adjacent lines. Getting these the
    wrong way round would invert every amendment chain."""
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, "contract_id") != _best_hit(text, "parent_contract_id")
```

- [ ] **Step 4: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_l1_parity.py -v
```

Expected: FAIL — `_best_hit` returns `None` for every field, because `contract.yaml` has no patterns at all.

- [ ] **Step 5: Add patterns to the existing `contract_id` field**

In `extraction_schemas/contract.yaml`, add a `patterns:` block to the existing `contract_id` entry, between `canonical_labels:` and `extractors:`:

```yaml
    patterns:
      - name: anchored_contract_no
        anchor: '(?i)\bcontract\s*(?:number|no\.?|reference|ref\.?|#)\s*[:\-]\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.92
      - name: anchored_agreement_no
        anchor: '(?i)\bagreement\s*(?:number|no\.?|reference|ref\.?|#)\s*[:\-]\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.88
```

Note the anchor requires the word `contract` or `agreement` immediately before the number word — `Parent Contract:` does not match, because `parent` sits between nothing and `contract` but the anchor demands `contract` be followed by `number|no|reference|ref|#`. `Parent Contract: MSA-…` has no such word, so it cannot be captured here.

- [ ] **Step 6: Add the three new fields**

Append to the top-level `fields:` list of `extraction_schemas/contract.yaml`:

```yaml
  - name: parent_contract_id
    type: string
    required: false
    db_column: parent_contract_id
    canonical_labels:
      - "Parent Contract"
      - "Parent Agreement"
      - "Master Agreement"
      - "Amends Contract"
      - "Supplements Agreement"
    patterns:
      - name: anchored_parent_contract
        anchor: '(?i)\b(?:parent|master|principal)\s+(?:contract|agreement)\s*(?:number|no\.?|reference|ref\.?|#)?\s*[:\-]\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.90
      - name: amends_contract
        anchor: '(?i)\b(?:amends|amendment\s+to|supplements|varies)\s+(?:the\s+)?(?:contract|agreement)\s*(?:number|no\.?|#)?\s*[:\-]?\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{4,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.84
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: cost_centre_id
    type: string
    required: false
    db_column: cost_centre_id
    canonical_labels:
      - "Cost Centre"
      - "Cost Center"
      - "Cost Centre Code"
      - "Cost Ctr"
      - "Charge Code"
    patterns:
      - name: anchored_cost_centre
        anchor: '(?i)\bcost\s*(?:centre|center|ctr)\s*(?:code|id|no\.?|#)?\s*[:\-]\s*'
        value: '([A-Z0-9][A-Z0-9\-\/\.]{1,29})'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.90
      - name: anchored_charge_code
        anchor: '(?i)\bcharge\s*code\s*[:\-]\s*'
        value: '([A-Z0-9][A-Z0-9\-\/\.]{1,29})'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.82
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: is_amendment
    type: string
    required: false
    db_column: is_amendment
    canonical_labels:
      - "Amendment"
      - "Addendum"
      - "Variation"
      - "Change Order"
    patterns:
      # Heading mode (empty anchor): the value regex runs against the whole
      # document. The lookahead requires an identifier to follow, so prose like
      # "no amendment to this agreement shall be effective unless..." cannot
      # match — only a document that titles itself an amendment.
      - name: amendment_titled
        anchor: ''
        value: '\b(AMENDMENT|Amendment|ADDENDUM|Addendum|VARIATION|Variation)\b(?=\s*(?:No\.?|Number|#|\d))'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.88
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []
```

`is_amendment` captures the literal marker word the document uses, not a normalised `Y`/`N`. Normalising would be an interpretation performed inside the extractor, and the column is `text`, so the literal is both truthful and storable. Mapping it to a boolean is Phase 1b's job, where the mapping can carry its own provenance.

- [ ] **Step 7: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_l1_parity.py tests/extraction/test_schema_db_consistency.py -v
```

Expected: the four `ALLOCATION_FIELDS` cases and the confusion test PASS. Iterate on the YAML regexes until they do; do not relax the expected values.

- [ ] **Step 8: Commit**

```bash
git add extraction_schemas/contract.yaml tests/extraction/test_contract_l1_parity.py \
        tests/extraction/fixtures/contracts/
git commit -m "feat(extraction): capture cost centre, parent contract and amendment marker on contracts"
```

---

## Task 6: Contract commercial terms (new columns)

Depends on Task 4's migration being applied to both databases, or the schema will not load.

**Files:**
- Modify: `extraction_schemas/contract.yaml`
- Modify: `tests/extraction/test_contract_l1_parity.py`

**Interfaces:**
- Consumes: Task 4's columns, Task 5's `_load_fixture` and `_best_hit` helpers
- Produces: header fields `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, `amendment_ref`, `document_version` on doc type `contract`.

- [ ] **Step 1: Write the failing test**

Append to `tests/extraction/test_contract_l1_parity.py`:

```python
TERM_FIELDS = [
    "amendment_ref",
    "document_version",
    "term_months",
    "billing_frequency",
    "escalator_pct",
    "escalator_basis",
    "escalator_cap_pct",
]


@pytest.mark.parametrize("field", TERM_FIELDS)
def test_commercial_term_fields_extract_from_fixture(field):
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, field) == expected[field]


@pytest.mark.parametrize("field", ["term_months", "escalator_pct", "escalator_cap_pct"])
def test_decimal_fields_capture_a_bindable_number(field):
    """parse_amount returns None for '3.5%' and '36 months', and the decimal
    fallback float() then raises — the value would be discarded as a
    type_bind_error. The capture group must yield the bare number."""
    from src.services.extraction_v2.parsers.amounts import parse_amount

    text, _ = _load_fixture("msa_with_amendment")
    hit = _best_hit(text, field)
    assert hit is not None
    assert parse_amount(hit) is not None, f"{field} captured {hit!r}, which will not bind to decimal"


def test_escalator_and_cap_are_not_the_same_number():
    """3.5% escalation capped at 5% — reading the cap as the rate would
    understate every uplift check by 43%."""
    text, _ = _load_fixture("msa_with_amendment")
    assert _best_hit(text, "escalator_pct") != _best_hit(text, "escalator_cap_pct")


def test_term_stated_in_years_is_not_silently_converted():
    """A three-year term must yield NULL, not 36. Conversion is arithmetic, and
    the extractor does not do arithmetic."""
    text = "Initial Term: three (3) years from the Commencement Date."
    assert _best_hit(text, "term_months") is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_l1_parity.py -v
```

Expected: the `TERM_FIELDS` cases FAIL with `None != …`.

- [ ] **Step 3: Add the fields**

Append to the top-level `fields:` list of `extraction_schemas/contract.yaml`:

```yaml
  - name: amendment_ref
    type: string
    required: false
    db_column: amendment_ref
    canonical_labels:
      - "Amendment No"
      - "Amendment Number"
      - "Variation No"
      - "Addendum No"
      - "Change Order No"
    patterns:
      - name: anchored_amendment_no
        anchor: '(?i)\b(?:amendment|addendum|variation|change\s+order)\s*(?:number|no\.?|#)\s*[:\-]?\s*'
        value: '([A-Z][A-Z0-9\-\/\.]{2,30}|\d{1,12}[A-Z0-9\-\/\.]{0,20})'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.90
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: document_version
    type: string
    required: false
    db_column: document_version
    canonical_labels:
      - "Version"
      - "Document Version"
      - "Revision"
      - "Rev"
    patterns:
      - name: anchored_version
        anchor: '(?i)\b(?:document\s+)?(?:version|revision|rev\.?)\s*[:\-]?\s*'
        value: '(v?\d+(?:\.\d+){0,3})'
        max_span_after_anchor_chars: 24
        prior_confidence: 0.86
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: term_months
    type: decimal
    required: false
    db_column: term_months
    canonical_labels:
      - "Initial Term"
      - "Term"
      - "Contract Term"
      - "Minimum Term"
    patterns:
      # Captures ONLY a term the document states in months. A term stated in
      # years is deliberately left NULL — converting it is arithmetic, and the
      # derived value belongs in Phase 1b with its own basis and provenance.
      - name: term_in_months
        anchor: '(?i)\b(?:initial\s+term|minimum\s+term|contract\s+term|term)\s*(?:of|is|:|\-)\s*'
        value: '(\d{1,3})\s*(?:months|month|mths|mos\.?)\b'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.90
      - name: period_of_months
        anchor: '(?i)\bfor\s+a\s+period\s+of\s+'
        value: '(\d{1,3})\s*(?:months|month|mths|mos\.?)\b'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.84
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: billing_frequency
    type: string
    required: false
    db_column: billing_frequency
    canonical_labels:
      - "Billing Frequency"
      - "Invoicing Frequency"
      - "Payment Frequency"
      - "Billed"
    patterns:
      - name: anchored_billing_frequency
        anchor: '(?i)\b(?:billing|invoicing|payment)\s+frequency\s*[:\-]\s*'
        value: '\b(monthly|quarterly|annually|annual|yearly|semi-annually|bi-annually|weekly)\b'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.92
      - name: invoiced_cadence
        anchor: '(?i)\b(?:shall\s+be\s+)?(?:invoiced|billed|payable)\s+'
        value: '\b(monthly|quarterly|annually|annual|yearly|semi-annually|bi-annually|weekly)\b'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.84
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: escalator_pct
    type: decimal
    required: false
    db_column: escalator_pct
    canonical_labels:
      - "Annual Increase"
      - "Price Escalation"
      - "Escalation"
      - "Uplift"
      - "Indexation"
    patterns:
      # group(1) is the bare number. parse_amount('3.5%') is None and the
      # decimal fallback float('3.5%') raises, so capturing the '%' would
      # discard the value as a type_bind_error.
      - name: anchored_escalation
        anchor: '(?i)\b(?:annual\s+)?(?:price\s+)?(?:escalation|escalator|uplift|increase|indexation)\s*(?:of|is|:|\-|shall\s+be)?\s*'
        value: '(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|per\s*cent)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.90
      - name: shall_increase_by
        anchor: '(?i)\b(?:charges|prices|fees|rates)\s+shall\s+(?:be\s+)?increase[d]?\s+by\s+'
        value: '(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|per\s*cent)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.86
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: escalator_cap_pct
    type: decimal
    required: false
    db_column: escalator_cap_pct
    canonical_labels:
      - "Cap"
      - "Capped At"
      - "Maximum Increase"
      - "Not to Exceed"
    patterns:
      - name: capped_at
        anchor: '(?i)\b(?:capped\s+at|cap\s+of|subject\s+to\s+a\s+cap\s+of|shall\s+not\s+exceed|not\s+to\s+exceed|maximum\s+(?:of\s+)?increase\s+of)\s*'
        value: '(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|per\s*cent)'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.90
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  - name: escalator_basis
    type: string
    required: false
    db_column: escalator_basis
    canonical_labels:
      - "Index"
      - "Indexed To"
      - "Index Basis"
      - "CPI"
      - "RPI"
    patterns:
      - name: indexed_to
        anchor: '(?i)\b(?:indexed\s+to|linked\s+to|by\s+reference\s+to|index\s*[:\-])\s*(?:the\s+)?'
        value: '\b(CPIH|CPI-U|CPI|RPIX|RPI|HICP)\b'
        max_span_after_anchor_chars: 40
        prior_confidence: 0.92
      - name: fixed_basis
        anchor: '(?i)\b(?:fixed|flat)\s+(?:annual\s+)?(?:increase|escalation|uplift)\b'
        value: '\b(fixed|flat)\b'
        max_span_after_anchor_chars: 30
        prior_confidence: 0.80
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []
```

- [ ] **Step 4: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/test_contract_l1_parity.py -v
```

Expected: all PASS. If `test_escalator_and_cap_are_not_the_same_number` fails, the escalation anchor's window is reaching into the cap sentence — shorten `max_span_after_anchor_chars`. Do not delete the test.

- [ ] **Step 5: Run the full extraction suite**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/ tests/extraction_v3/ -v
```

Expected: no new failures versus baseline.

- [ ] **Step 6: Commit**

```bash
git add extraction_schemas/contract.yaml tests/extraction/test_contract_l1_parity.py
git commit -m "feat(extraction): capture escalator, term, billing frequency and version on contracts"
```

---

## Task 7: Field-coverage harness

Phase 0 found that `bp_testdb` column fill rates suggest UoM is captured 99.9% of the time, while the real extraction record shows 12 captures in 64,118. That trap will recur at every later phase unless there is one command that measures the right thing. This script is Phase 1a's acceptance instrument and the baseline Phase 1b is measured against.

**Files:**
- Create: `scripts/field_coverage.py`
- Create: `tests/services/test_field_coverage.py`

**Interfaces:**
- Consumes: `src.services.db.get_conn`
- Produces: `field_coverage(conn, doc_type=None) -> list[dict]` with keys `doc_type`, `field_path`, `documents_with_field`, `documents_total`, `coverage_pct`.

- [ ] **Step 1: Write the failing test**

```python
"""Coverage must be measured from the extraction record, never from column NULL
counts — seeded data makes a never-extracted column look fully populated."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.field_coverage import field_coverage  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.description = [("doc_type",), ("field_path",),
                            ("documents_with_field",), ("documents_total",)]
        self.executed = None

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._cur = _FakeCursor(rows)

    def cursor(self):
        return self._cur


def test_coverage_pct_is_computed_not_read():
    conn = _FakeConn([("quote", "line_items[].unit_of_measure", 3, 120)])
    out = field_coverage(conn)
    assert out[0]["coverage_pct"] == 2.5


def test_zero_total_does_not_divide_by_zero():
    conn = _FakeConn([("contract", "contract_id", 0, 0)])
    out = field_coverage(conn)
    assert out[0]["coverage_pct"] is None


def test_query_reads_the_provenance_table_not_the_trgt_tables():
    conn = _FakeConn([])
    field_coverage(conn)
    sql = conn.cursor().executed[0].lower()
    assert "bp_extraction_provenance_v3" in sql
    assert "_trgt" not in sql, "coverage must not be inferred from seeded target columns"
```

- [ ] **Step 2: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/services/test_field_coverage.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.field_coverage'`.

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python
"""Per-field extraction coverage, read from the extraction record.

Phase 0 finding: unit_of_measure is populated on 99.9% of bp_testdb line rows
and was extracted 12 times in 64,118 provenance records. The column fill rate
is seeded data; the provenance record is what extraction actually produced.
Measure here, never from _trgt NULL counts.

Usage:
    set -a && . ./.env && set +a && ./venv/bin/python scripts/field_coverage.py
    ./venv/bin/python scripts/field_coverage.py --doc-type contract
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.db import get_conn  # noqa: E402

# Collapses line_items[0].x / line_items[7].x into one row.
_SQL = """
WITH totals AS (
    SELECT doc_type, count(DISTINCT doc_pk) AS documents_total
    FROM proc.bp_extraction_provenance_v3
    GROUP BY doc_type
),
per_field AS (
    SELECT doc_type,
           regexp_replace(field_path, '\\[[0-9]+\\]', '[]', 'g') AS field_path,
           count(DISTINCT doc_pk) AS documents_with_field
    FROM proc.bp_extraction_provenance_v3
    GROUP BY 1, 2
)
SELECT p.doc_type, p.field_path, p.documents_with_field, t.documents_total
FROM per_field p
JOIN totals t USING (doc_type)
WHERE (%(doc_type)s::text IS NULL OR p.doc_type = %(doc_type)s)
ORDER BY p.doc_type, p.documents_with_field DESC, p.field_path
"""


def field_coverage(conn: Any, doc_type: Optional[str] = None) -> list[dict]:
    """Coverage per (doc_type, field_path). coverage_pct is None when the
    doc_type has no documents — never 0, which would read as 'extracted nothing'
    rather than 'nothing to extract from'."""
    cur = conn.cursor()
    cur.execute(_SQL, {"doc_type": doc_type})
    cols = [d[0] for d in cur.description]
    out: list[dict] = []
    for row in cur.fetchall():
        rec = dict(zip(cols, row))
        total = rec.get("documents_total") or 0
        rec["coverage_pct"] = (
            round(100.0 * rec["documents_with_field"] / total, 1) if total else None
        )
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-type", default=None)
    args = ap.parse_args()

    with get_conn() as conn:
        rows = field_coverage(conn, args.doc_type)

    if not rows:
        print("no provenance records found")
        return 0

    current = None
    for r in rows:
        if r["doc_type"] != current:
            current = r["doc_type"]
            print(f"\n=== {current} ({r['documents_total']} documents) ===")
        pct = "n/a" if r["coverage_pct"] is None else f"{r['coverage_pct']:5.1f}%"
        print(f"  {pct}  {r['documents_with_field']:6d}  {r['field_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/services/test_field_coverage.py -v
```

Expected: all three PASS.

- [ ] **Step 5: Capture the pre-reprocessing baseline**

```bash
set -a && . ./.env && set +a && ./venv/bin/python scripts/field_coverage.py > /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/47339042-d85e-48c7-a214-fccc8f5fca68/scratchpad/coverage_before.txt
cat /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/47339042-d85e-48c7-a214-fccc8f5fca68/scratchpad/coverage_before.txt
```

Expected on `bp_testdb`: only `currency` appears, at low counts. This is the "before" figure for the decision record. **This number will not improve until documents are re-extracted** — Phase 1a changes what the pipeline *looks for*, not what it has already found. Say that plainly rather than implying the schema change alone lifted coverage.

- [ ] **Step 6: Commit**

```bash
git add scripts/field_coverage.py tests/services/test_field_coverage.py
git commit -m "feat(scripts): per-field extraction coverage measured from the provenance record"
```

---

## Task 8: Decision record

**Files:**
- Create: `docs/remediation/01a_extraction_schema_extension.md`

- [ ] **Step 1: Write the record**

It must cover, in this order:

1. **What changed** — the two migrations and the four schemas, with the field list per doc type.
2. **Why migration-before-YAML is a hard ordering constraint**, citing `loader.py:110–128`.
3. **Why no Python changed** — `promote()` intersects against physical columns; `table_extractor._header_to_field` resolves from `canonical_labels`. Both pick new fields up automatically.
4. **Three deliberate refusals**, each with its reason: `grounded_last_resort: false` on every new field (a guessed contract link corrupts every downstream baseline comparison); no years→months conversion on `term_months`; `is_amendment` stores the literal marker word rather than a normalised boolean.
5. **The decimal capture-group rule** — `parse_amount('3%') is None`, so a `%` inside the capture group discards the value as a `type_bind_error`.
6. **Coverage before, and the honest caveat** — the numbers from Task 7 Step 5, with the statement that they cannot move until documents are re-extracted.
7. **The contract-validation limitation** — synthetic text fixtures only; no contract corpus exists; blocked on B1; re-run `tests/extraction/test_contract_l1_parity.py` against real documents when it is resolved.
8. **What Phase 1b now depends on** — `contract_id` on all three transaction types, `unit_of_measure` on all three line tables, and seven commercial-term columns on `bp_contracts`, all reachable without parsing free text.
9. **Open blockers carried forward** — B1 (no contract corpus), B3 (no GPSS dictionary). B2 is resolved: `tenant_id` on new tables only, first applied in Phase 1b.

- [ ] **Step 2: Commit**

```bash
git add docs/remediation/01a_extraction_schema_extension.md
git commit -m "docs(remediation): Phase 1a decision record"
```

---

## Acceptance

Phase 1a is done when all of the following hold:

- [ ] `set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/extraction/ tests/extraction_v3/ tests/services/test_field_coverage.py -v` passes with no new failures versus the pre-phase baseline.
- [ ] `load_all_schemas()` succeeds against **both** `bp_sqldb` and `bp_testdb`.
- [ ] Both rollback scripts have been run and re-applied on `bp_testdb`, with the column counts verified before and after.
- [ ] A query over `proc.bp_quote_trgt`, `bp_invoice_trgt` and `bp_purchase_order_trgt` can select `contract_id` without error on both databases.
- [ ] `scripts/field_coverage.py` runs and its output is recorded in the decision record.
- [ ] The decision record states plainly that contract patterns are validated against synthetic fixtures only.

**Explicitly NOT in scope:** re-extracting the existing corpus (a separate operational decision — reprocessing ~88k documents), any change to `Finding` or `bp_opportunity`, any `CommercialFact` type, and any UoM normalisation. Those are Phase 1b.
