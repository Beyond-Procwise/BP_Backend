# Discrepancy Triage (Procure-to-Pay) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A rules-only engine that compares each deal's quote, PO and invoices, turns differences into a short list of scored findings in the SpendIQ Action Centre (`proc.bp_detection_finding`), keeps every comparison in an audit table, and is proven by a full backfill of `bp_testdb` (5,041 deals).

**Architecture:** New package `src/services/triage/`. Pure stages (normalise → link → checks → score → group → text → verdict) over dataclasses; only `loader.py` reads and only `writer.py` writes. Tolerances come from one new governed-limit policy row read through `resolve_tolerance`. Entry points: a backfill script, a rollback script, a scheduler job, and two FastAPI routes.

**Tech Stack:** Python 3, psycopg2 (`execute_values`), FastAPI, pytest, PostgreSQL (`proc` schema), stdlib `decimal` / `difflib`.

**Spec:** `docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md` (read it first; section numbers below are §N of that file).

## Global Constraints

- Money is `decimal.Decimal` end to end; `float` only for scores and confidences.
- No LLM, GPU, network or clock call anywhere in the engine (the run report's timer is the one exception).
- New tables use the `bp_` prefix; indexes are `ix_bp_<table>_<col>`.
- Every tolerance/threshold is read from governed-limit policy `triage_tolerances` via `tolerance.resolve_tolerance` / `TriageConfig`; a missing or null value raises `LimitUnavailable` and a run aborts **before writing anything**.
- Never modify `_trgt`, `_stg`, `_raw` or any source row. Never modify `src/services/reconciliation.py`, `src/agents/discrepancy_detection_agent.py` or `src/services/duplicate_invoice_detector.py`.
- Only S1 and S2 findings are written to `proc.bp_detection_finding` (`critical` / `warning`); S3 and S0 live only in `proc.bp_triage_result`.
- The engine moves a `bp_detection_finding` row only from `status='open'`, and moves `status` and `lifecycle_status` together (`superseded` + `resolved`).
- UI and gateway repos are read-only.
- Imports inside the package: `from src.services...` (as `reconciliation.py` does). Routers import auth as `from api.auth import require_user`.
- Tests: `set -a; . ./.env; set +a` first; run with `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest ... -q -p no:randomly`. Tests that need the real database carry `PROCWISE_TEST_LIVE_DB=1` (under pytest the DB is a fake by default).
- Another session shares this checkout **and its index**: commit with `git add <paths> && git commit -o <paths> -m "..."`. Never `git stash`. Stay on `Development`; never push.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Credit notes** (negative `invoice_amount`, 191 in the corpus) — expected: they reduce a PO's running total and invoiced quantity, and never raise price or arithmetic conflicts on their own. Tests: Task 6 (`test_credit_note_*`), Task 7 (`test_credit_note_brings_running_total_back_within_po`).
2. **A non-GBP invoice whose currency has no FX rate** — expected: the finding is still raised, capped at S2 unless an always-S1 rule applies, and its text says "no FX rate". Tests: Task 8 (`test_missing_fx_caps_at_s2`, `test_override_beats_fx_cap`), Task 10 (`test_text_without_fx_rate`).
3. **A person resolves a finding while the scheduler re-runs the deal** — expected: the person's decision stands; the engine never reopens or overwrites it unless severity rises. Tests: Task 11 (`test_resolved_finding_is_not_reopened`, `test_severity_rise_opens_a_new_finding`).
4. **A deal with only a quote, or a PO that nobody has invoiced** — expected: verdict `Incomplete`, no crash, no findings. Tests: Task 10 (`test_quote_only_deal_is_incomplete`, `test_uninvoiced_po_is_incomplete`).
5. **Lines with missing quantity, price or amount, and a PO carrying the same item on two lines** (20 such POs in the corpus) — expected: skipped or linked to the price-matching line, never an exception. Tests: Task 5 (`test_repeated_item_prefers_same_price_line`), Task 6 (`test_line_without_price_or_quantity_is_skipped_not_crashed`).

## File Map

| File | Responsibility |
|---|---|
| `deploy/sql/2026-09-24_bp_triage.sql` / `_rollback.sql` | Three tables + `triage_tolerances` policy row |
| `tests/conftest.py` | Seed copy of `triage_tolerances` (modify) |
| `src/services/triage/__init__.py` | Package marker + one-line purpose |
| `src/services/triage/model.py` | Dataclasses, enums, fingerprint, `money`, `pct_change` |
| `src/services/triage/tolerance.py` | `TriageConfig`, `load_config`, `Tolerance`, `resolve_tolerance` |
| `src/services/triage/normalise.py` | Decimal/confidence/text/terms helpers |
| `src/services/triage/loader.py` | Batched SQL → `DocumentSet`s, FX attached |
| `src/services/triage/link.py` | Invoice→PO, line→line, roll-ups, PO→quote |
| `src/services/triage/checks.py` | The fifteen checks, `run_checks` |
| `src/services/triage/score.py` | Materiality, bands, overrides |
| `src/services/triage/group.py` | Results → findings |
| `src/services/triage/text.py` | Finding headline and text |
| `src/services/triage/verdict.py` | Deal verdict |
| `src/services/triage/writer.py` | Runs, audit rows, finding upsert/supersede/reopen, rollback |
| `src/services/triage/report.py` | `RunReport` |
| `src/services/triage/engine.py` | `triage_set`, `run_triage`, `triage_deal_view`, `run_changed` |
| `src/api/routers/triage.py` | `GET /triage/deals/{id}`, `POST /triage/deals/{id}/run` |
| `src/api/main.py` | Register router (modify) |
| `src/services/backend_scheduler.py` | Register triage job (modify) |
| `scripts/triage_backfill.py`, `scripts/triage_rollback.py` | Operator scripts |
| `tests/triage/*` | Unit, live-DB and recall tests |
| `tests/migrations/test_2026_09_24_bp_triage.py` | Migration test |

---

### Task 1: Tables, policy row and test seed

**Files:**
- Create: `deploy/sql/2026-09-24_bp_triage.sql`
- Create: `deploy/sql/2026-09-24_bp_triage_rollback.sql`
- Modify: `tests/conftest.py` (add one key to `GOVERNED_LIMIT_SEED`)
- Test: `tests/migrations/test_2026_09_24_bp_triage.py`

**Interfaces:**
- Produces: tables `proc.bp_triage_run`, `proc.bp_triage_result`, `proc.bp_triage_finding`; policy `triage_tolerances` with the 22 keys below; `GOVERNED_LIMIT_SEED["triage_tolerances"]`.

- [ ] **Step 1: Write the failing migration test**

`tests/migrations/test_2026_09_24_bp_triage.py`:

```python
"""Integration test for 2026-09-24_bp_triage.sql (runs against the .env database)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402
from tests.conftest import GOVERNED_LIMIT_SEED  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-09-24_bp_triage.sql")


def _conn():
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name,
                         user=s.db_user, password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


@pytest.fixture(scope="module")
def applied():
    conn = _conn()
    conn.cursor().execute(MIGRATION.read_text())
    conn.cursor().execute(MIGRATION.read_text())   # idempotent: a second apply is harmless
    yield conn
    conn.close()


def test_tables_exist(applied):
    cur = applied.cursor()
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='proc' AND table_name LIKE 'bp_triage_%'""")
    assert {r[0] for r in cur.fetchall()} >= {
        "bp_triage_run", "bp_triage_result", "bp_triage_finding"}


def test_policy_row_matches_the_test_seed(applied):
    cur = applied.cursor()
    cur.execute("""SELECT policy_details->'rules' FROM proc.bp_policy
                   WHERE policy_type='limit'
                     AND policy_details->>'policy_identifier'='triage_tolerances'""")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == GOVERNED_LIMIT_SEED["triage_tolerances"]
```

- [ ] **Step 2: Run it to see it fail**

Run: `set -a; . ./.env; set +a; CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/migrations/test_2026_09_24_bp_triage.py -q -p no:randomly`
Expected: FAIL/ERROR — the SQL file does not exist.

- [ ] **Step 3: Write the migration**

`deploy/sql/2026-09-24_bp_triage.sql`:

```sql
-- Discrepancy triage, procure-to-pay slice.
-- Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md (§5.3, §8.2).
--
-- Findings that need action go to the EXISTING proc.bp_detection_finding (the Action
-- Centre reads it). These three tables hold everything else: one row per run, one row
-- per comparison (including matches -- "suppress from view, never from record"), and a
-- map from each finding's stable fingerprint to its bp_detection_finding row, kept here
-- so no column is added to a table the gateway maps field by field.
--
-- The tolerance row: every value that decides whether a difference is a discrepancy.
-- Starting values are the triage spec's §14 example; tune them after the first backfill.
-- Additive and idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_triage_run (
    run_id              uuid PRIMARY KEY,
    mode                varchar NOT NULL CHECK (mode IN ('backfill', 'scheduled', 'single')),
    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    rolled_back_at      timestamptz,
    config_fingerprint  varchar NOT NULL,
    config_values       jsonb NOT NULL,
    deal_count          integer,
    failed_deals        jsonb,
    report              jsonb
);

CREATE TABLE IF NOT EXISTS proc.bp_triage_result (
    result_id     bigserial PRIMARY KEY,
    run_id        uuid NOT NULL REFERENCES proc.bp_triage_run (run_id) ON DELETE CASCADE,
    deal_id       varchar NOT NULL,
    rule_id       varchar NOT NULL,
    claim_doc     varchar,
    claim_line    varchar,
    auth_doc      varchar,
    auth_line     varchar,
    field_name    varchar,
    claim_value   varchar,
    auth_value    varchar,
    outcome       varchar NOT NULL,
    severity      varchar NOT NULL,
    exposure_gbp  numeric,
    score         numeric,
    score_inputs  jsonb,
    tolerance     jsonb,
    fingerprint   varchar NOT NULL,
    finding_id    bigint
);
CREATE INDEX IF NOT EXISTS ix_bp_triage_result_run  ON proc.bp_triage_result (run_id);
CREATE INDEX IF NOT EXISTS ix_bp_triage_result_deal ON proc.bp_triage_result (deal_id);

CREATE TABLE IF NOT EXISTS proc.bp_triage_finding (
    fingerprint    varchar PRIMARY KEY,
    finding_id     bigint NOT NULL,
    deal_id        varchar NOT NULL,
    first_run_id   uuid NOT NULL,
    last_run_id    uuid NOT NULL,
    last_severity  varchar NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_deal      ON proc.bp_triage_finding (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_first_run ON proc.bp_triage_finding (first_run_id);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'TriageTolerancePolicy', 'limit',
       'What counts as a discrepancy between a deal''s quote, PO and invoices, and how '
       'serious it is. Widen these and findings stop reaching the Action Centre.',
       jsonb_build_object('policy_identifier', 'triage_tolerances', 'rules', jsonb_build_object(
           'unit_price_over_pct',        1.0,
           'unit_price_over_abs',        5.0,
           'unit_price_combine',         'min',
           'unit_price_under_pct',       5.0,
           'quantity_over_pct',          5.0,
           'rounding_per_line',          0.01,
           'cumulative_total_pct',       0.5,
           'cumulative_total_abs',       50.0,
           'cumulative_total_combine',   'min',
           'allowed_tax_rates',          jsonb_build_array(0, 5, 20),
           'min_link_confidence',        0.8,
           'unlinked_below',             0.5,
           'min_extraction_confidence',  0.7,
           'description_min_similarity', 0.4,
           'materiality_pct_of_total',   0.5,
           'materiality_floor',          25,
           'materiality_ceiling',        5000,
           'band_s1',                    70,
           'band_s2',                    40,
           'uplift_min_lines',           3,
           'uplift_same_pct_within',     0.1,
           'batch_size',                 200
       )),
       '', 1, 1, now(), 'triage', now(), 'triage'
 WHERE NOT EXISTS (
       SELECT 1 FROM proc.bp_policy
        WHERE policy_type = 'limit'
          AND policy_details->>'policy_identifier' = 'triage_tolerances');

COMMIT;
```

`deploy/sql/2026-09-24_bp_triage_rollback.sql`:

```sql
-- Reverses 2026-09-24_bp_triage.sql. Removes the Action Centre findings triage created
-- that nobody has touched; findings a person acted on stay (their map rows go, so they
-- become ordinary findings).
BEGIN;
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL;
DROP TABLE IF EXISTS proc.bp_triage_result;
DROP TABLE IF EXISTS proc.bp_triage_finding;
DROP TABLE IF EXISTS proc.bp_triage_run;
DELETE FROM proc.bp_policy
 WHERE policy_type = 'limit'
   AND policy_details->>'policy_identifier' = 'triage_tolerances';
COMMIT;
```

- [ ] **Step 4: Add the seed copy**

In `tests/conftest.py`, inside `GOVERNED_LIMIT_SEED`, after the `"reseller_catalog": {...},` entry add:

```python
    "triage_tolerances": {
        "unit_price_over_pct": 1.0, "unit_price_over_abs": 5.0,
        "unit_price_combine": "min", "unit_price_under_pct": 5.0,
        "quantity_over_pct": 5.0, "rounding_per_line": 0.01,
        "cumulative_total_pct": 0.5, "cumulative_total_abs": 50.0,
        "cumulative_total_combine": "min", "allowed_tax_rates": [0, 5, 20],
        "min_link_confidence": 0.8, "unlinked_below": 0.5,
        "min_extraction_confidence": 0.7, "description_min_similarity": 0.4,
        "materiality_pct_of_total": 0.5, "materiality_floor": 25,
        "materiality_ceiling": 5000, "band_s1": 70, "band_s2": 40,
        "uplift_min_lines": 3, "uplift_same_pct_within": 0.1, "batch_size": 200},
```

- [ ] **Step 5: Apply to both databases**

```bash
set -a; . ./.env; set +a
for db in bp_testdb bp_sqldb; do
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$db" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-24_bp_triage.sql
done
```
Expected: `BEGIN … CREATE TABLE … INSERT 0 1 … COMMIT` for each (a re-run prints `INSERT 0 0`).

- [ ] **Step 6: Run the migration test and the seed-drift test**

Run: `set -a; . ./.env; set +a; CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/migrations/test_2026_09_24_bp_triage.py tests/governance/test_governed_limits.py -q -p no:randomly`
Expected: all pass (the governed-limits suite contains the seed-vs-live comparison; it now includes `triage_tolerances`).

- [ ] **Step 7: Commit**

```bash
git add deploy/sql/2026-09-24_bp_triage.sql deploy/sql/2026-09-24_bp_triage_rollback.sql tests/conftest.py tests/migrations/test_2026_09_24_bp_triage.py
git commit -o deploy/sql/2026-09-24_bp_triage.sql deploy/sql/2026-09-24_bp_triage_rollback.sql tests/conftest.py tests/migrations/test_2026_09_24_bp_triage.py -m "feat(triage): tables and tolerance policy for discrepancy triage

Three tables (run log, per-comparison audit, finding fingerprint map) and one
governed-limit row, triage_tolerances, holding every value that decides whether
a difference is a discrepancy. Applied to bp_testdb and bp_sqldb.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Model and tolerances

**Files:**
- Create: `src/services/triage/__init__.py`, `src/services/triage/model.py`, `src/services/triage/tolerance.py`
- Create: `tests/triage/__init__.py` (empty), `tests/triage/helpers.py`
- Test: `tests/triage/test_model.py`, `tests/triage/test_tolerance.py`

**Interfaces:**
- Produces (model): `Outcome`, `SCORED`, `NOTE`, `Severity` (IntEnum, S0=0 < S3=1 < S2=2 < S1=3), `ACTION_CENTRE_SEVERITY`, `CRITICALITY`, `CATEGORY`, `Line`, `Doc` (with `.is_credit_note`, `.po_ref`), `DuplicateFlag`, `DocumentSet`, `LineLink`, `Links`, `Result` (with `.cause_key`, `.fingerprint`, `.exposure_gbp`), `Finding` (with `.lead`, `.severity`, `.exposure`, `.exposure_gbp`, `.fingerprint`, `.category`, `.confidence`), `Verdict` (with `.summary`), `fingerprint(deal_id, rule_id, cause_key) -> str`, `money(amount, currency) -> str`, `pct_change(result) -> Optional[Decimal]`.
- Produces (tolerance): `POLICY`, `KEYS`, `TriageConfig` (`cfg["key"]`, `.values`, `.fingerprint`), `load_config(read=None) -> TriageConfig`, `Tolerance(pct, abs_gbp, combine, source)` with `.allowance(base, fx_to_gbp) -> Decimal` and `.as_dict()`, `resolve_tolerance(check, cfg, ctx=None) -> Tolerance` for checks `unit_price_over`, `unit_price_under`, `quantity_over`, `cumulative_total`.
- Produces (tests/triage/helpers.py): `make_cfg(**overrides)`, `line(...)`, `po(...)`, `inv(...)`, `quote(...)`, `deal(*docs, duplicates=(), deal_id="DEAL-1")`.

- [ ] **Step 1: Write the test helpers**

`tests/triage/helpers.py`:

```python
"""Hand-built deals for triage unit tests. No database, ever."""
from __future__ import annotations

from datetime import date
from decimal import Decimal as D
from typing import Optional

from src.services.triage.model import Doc, DocumentSet, Line
from src.services.triage.tolerance import load_config
from tests.conftest import GOVERNED_LIMIT_SEED


def make_cfg(**overrides):
    rules = dict(GOVERNED_LIMIT_SEED["triage_tolerances"], **overrides)
    return load_config(read=lambda policy, key, cast: cast(rules[key]))


def line(ref, item: Optional[str] = "ITEM-1", qty: Optional[str] = "10",
         price: Optional[str] = "12.00", amount: Optional[str] = None,
         desc: str = "Widget", po_id: Optional[str] = None) -> Line:
    q = D(qty) if qty is not None else None
    p = D(price) if price is not None else None
    if amount is not None:
        amt = D(amount)
    else:
        amt = q * p if q is not None and p is not None else None
    return Line(line_ref=str(ref), item_id=item, description=desc, quantity=q,
                uom="each", unit_price=p, line_amount=amt, po_id=po_id)


def _net(lines, net):
    if net is not None:
        return D(net)
    return sum((l.line_amount or D("0")) for l in lines)


def po(po_id="PO-1", lines=None, net=None, currency="GBP", supplier="SUP-1",
       order_date=date(2026, 1, 10), terms="Net 30", quote_ref=None, fx="1") -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, net)
    tax = (n * D("0.2")).quantize(D("0.01"))
    return Doc(doc_id=po_id, doc_type="purchase_order", supplier_id=supplier,
               currency=currency, doc_date=order_date, net=n, tax=tax, gross=n + tax,
               payment_terms=terms, quote_ref=quote_ref,
               fx_to_gbp=D(fx) if fx else None, lines=lines)


def inv(inv_id="INV-1", po_id="PO-1", lines=None, net=None, tax=None, gross=None,
        currency="GBP", supplier="SUP-1", inv_date=date(2026, 2, 1), terms="30 days",
        fx="1", confidence=None) -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, net)
    t = D(tax) if tax is not None else (n * D("0.2")).quantize(D("0.01"))
    g = D(gross) if gross is not None else n + t
    return Doc(doc_id=inv_id, doc_type="invoice", supplier_id=supplier, currency=currency,
               doc_date=inv_date, net=n, tax=t, gross=g, payment_terms=terms, po_id=po_id,
               confidence=confidence, fx_to_gbp=D(fx) if fx else None, lines=lines)


def quote(quote_id="Q-1", lines=None, currency="GBP", supplier="SUP-1", fx="1") -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, None)
    return Doc(doc_id=quote_id, doc_type="quote", supplier_id=supplier, currency=currency,
               doc_date=date(2026, 1, 1), net=n, fx_to_gbp=D(fx) if fx else None, lines=lines)


def deal(*docs, duplicates=(), deal_id="DEAL-1") -> DocumentSet:
    ds = DocumentSet(deal_id)
    for d in docs:
        {"quote": ds.quotes, "purchase_order": ds.pos, "invoice": ds.invoices}[d.doc_type].append(d)
    ds.duplicates = list(duplicates)
    return ds
```

- [ ] **Step 2: Write the failing tests**

`tests/triage/test_model.py`:

```python
from decimal import Decimal as D

from src.services.triage.model import (
    Outcome, Result, Severity, fingerprint, money, pct_change)
from tests.triage.helpers import inv, line


def test_severity_orders_worst_highest():
    assert max(Severity.S3, Severity.S1, Severity.S2) == Severity.S1
    assert min(Severity.S1, Severity.S2) == Severity.S2


def test_fingerprint_is_stable_and_distinguishes_causes():
    a = fingerprint("D1", "unit_price", "INV-1|3")
    assert a == fingerprint("D1", "unit_price", "INV-1|3")
    assert a != fingerprint("D1", "unit_price", "INV-1|4")


def test_result_exposure_gbp_uses_fx_and_magnitude():
    r = Result("D1", "unit_price", "money", Outcome.CONFLICT, "INV-1", "unit_price",
               exposure=D("-100"), fx_to_gbp=D("0.85"))
    assert r.exposure_gbp == D("85.00")
    assert Result("D1", "x", "money", Outcome.CONFLICT, "INV-1", "x",
                  exposure=D("1")).exposure_gbp is None


def test_money_formats_gbp_and_others():
    assert money(D("1234.5"), "GBP") == "£1,234.50"
    assert money(D("1234.5"), "EUR") == "1,234.50 EUR"
    assert money(None, "GBP") == "n/a"


def test_pct_change():
    r = Result("D1", "unit_price", "money", Outcome.CONFLICT, "INV-1", "unit_price",
               auth_value="100", delta=D("3.5"))
    assert pct_change(r) == D("3.5")


def test_invoice_po_ref_falls_back_to_lines_and_credit_note_flag():
    doc = inv(po_id=None, lines=[line(1, po_id="PO-9")])
    assert doc.po_ref == "PO-9"
    assert inv(net="-10").is_credit_note
```

`tests/triage/test_tolerance.py`:

```python
from decimal import Decimal as D

import pytest

from src.services.governed_limits import LimitUnavailable
from src.services.triage.tolerance import load_config, resolve_tolerance
from tests.conftest import GOVERNED_LIMIT_SEED
from tests.triage.helpers import make_cfg

SEED = GOVERNED_LIMIT_SEED["triage_tolerances"]


def test_missing_key_refuses():
    rules = dict(SEED)
    rules.pop("band_s1")

    def read(policy, key, cast):
        if key not in rules:
            raise LimitUnavailable(key)
        return cast(rules[key])

    with pytest.raises(LimitUnavailable):
        load_config(read=read)


def test_null_value_refuses():
    rules = dict(SEED, band_s1=None)
    with pytest.raises(LimitUnavailable):
        load_config(read=lambda p, k, cast: None if rules[k] is None else cast(rules[k]))


def test_unit_price_over_takes_the_stricter_of_pct_and_abs():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    assert tol.allowance(D("12.00"), D("1")) == D("0.12")   # 1% < £5
    assert tol.allowance(D("2300"), D("1")) == D("5")       # £5 < 1%


def test_absolute_part_converts_to_document_currency():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    # 1 unit of document currency = £0.50, so £5 = 10 units; 1% of 2,300 = 23 -> min is 10
    assert tol.allowance(D("2300"), D("0.5")) == D("10")


def test_without_fx_the_percentage_applies():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    assert tol.allowance(D("2300"), None) == D("23")


def test_under_and_quantity_are_percentage_only():
    cfg = make_cfg()
    assert resolve_tolerance("unit_price_under", cfg).allowance(D("12"), D("1")) == D("0.6")
    assert resolve_tolerance("quantity_over", cfg).allowance(D("10"), None) == D("0.5")


def test_cumulative_total():
    tol = resolve_tolerance("cumulative_total", make_cfg())
    assert tol.allowance(D("120"), D("1")) == D("0.6")
    assert tol.allowance(D("100000"), D("1")) == D("50")
    assert tol.as_dict()["source"].startswith("bp_policy:triage_tolerances")


def test_unknown_check_raises():
    with pytest.raises(KeyError):
        resolve_tolerance("nonsense", make_cfg())


def test_fingerprint_tracks_values():
    assert make_cfg().fingerprint == make_cfg().fingerprint
    assert make_cfg().fingerprint != make_cfg(band_s1=75).fingerprint
```

- [ ] **Step 3: Run to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_model.py tests/triage/test_tolerance.py -q -p no:randomly`
Expected: collection ERROR — `No module named 'src.services.triage'`.

- [ ] **Step 4: Write the package, model and tolerances**

`src/services/triage/__init__.py`:

```python
"""Discrepancy triage for procure-to-pay deals.

Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md
"""
```

`src/services/triage/model.py`:

```python
"""Data shapes for discrepancy triage.

Every stage after the loader is a pure function over these dataclasses, which is
what lets each one be tested on a hand-built deal and keeps a 5,000-deal run fast.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Optional

ZERO = Decimal("0")


class Outcome(str, Enum):
    """What kind of difference one comparison found (triage spec §5)."""
    MATCH = "MATCH"
    WITHIN_TOL = "WITHIN_TOL"
    EXPLAINED = "EXPLAINED"
    ABSENT_SUBORDINATE = "ABSENT_SUBORDINATE"
    ABSENT_AUTHORITATIVE = "ABSENT_AUTHORITATIVE"
    CONFLICT = "CONFLICT"
    UNVERIFIABLE = "UNVERIFIABLE"


#: Only these go on to scoring; the rest are matches or notes.
SCORED = frozenset({Outcome.CONFLICT, Outcome.ABSENT_AUTHORITATIVE, Outcome.UNVERIFIABLE})
NOTE = frozenset({Outcome.EXPLAINED, Outcome.ABSENT_SUBORDINATE})


class Severity(IntEnum):
    """Higher is worse, so max() picks the most severe and min() caps."""
    S0 = 0
    S3 = 1
    S2 = 2
    S1 = 3


#: The Action Centre's own words. S3 and S0 never reach it.
ACTION_CENTRE_SEVERITY = {Severity.S1: "critical", Severity.S2: "warning"}

CRITICALITY = {"party": 1.0, "currency": 1.0, "money": 0.8, "quantity": 0.8,
               "date": 0.6, "reference": 0.6, "terms": 0.6, "description": 0.2}

CATEGORY = {
    "unit_price": "price", "uniform_uplift": "price", "quantity": "quantity",
    "cumulative_total": "overbilling", "duplicate": "duplicate", "tax_rate": "tax",
    "currency": "currency", "supplier": "supplier", "invoice_date": "date",
    "payment_terms": "terms", "description": "description", "unlinked_line": "linking",
    "rollup": "linking", "bad_po_ref": "linking", "no_po": "linking",
    "line_arithmetic": "arithmetic", "invoice_totals": "arithmetic",
}


def fingerprint(deal_id: str, rule_id: str, cause_key: str) -> str:
    return hashlib.sha1(f"{deal_id}|{rule_id}|{cause_key}".encode()).hexdigest()


def money(amount, currency: Optional[str]) -> str:
    if amount is None:
        return "n/a"
    text = f"{Decimal(amount).quantize(Decimal('0.01')):,}"
    if (currency or "").upper() == "GBP":
        return f"£{text}"
    return f"{text} {currency or ''}".strip()


@dataclass(frozen=True)
class Line:
    line_ref: str
    item_id: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    uom: Optional[str] = None
    unit_price: Optional[Decimal] = None
    line_amount: Optional[Decimal] = None
    po_id: Optional[str] = None
    delivery_date: Optional[date] = None


@dataclass
class Doc:
    doc_id: str
    doc_type: str                         # quote | purchase_order | invoice
    supplier_id: Optional[str] = None
    currency: Optional[str] = None
    doc_date: Optional[date] = None
    net: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    gross: Optional[Decimal] = None
    payment_terms: Optional[str] = None
    po_id: Optional[str] = None           # invoice -> PO number on its header
    quote_ref: Optional[str] = None       # PO -> quote
    confidence: Optional[float] = None    # 0-1; None = not reported
    fx_to_gbp: Optional[Decimal] = None
    fx_rate_date: Optional[datetime] = None
    lines: list[Line] = field(default_factory=list)

    @property
    def is_credit_note(self) -> bool:
        return self.net is not None and self.net < 0

    @property
    def po_ref(self) -> Optional[str]:
        """The PO this invoice names: the header's, else the first line's."""
        return self.po_id or next((l.po_id for l in self.lines if l.po_id), None)


@dataclass(frozen=True)
class DuplicateFlag:
    invoice_id: str
    earlier_invoice_id: Optional[str]
    amount: Optional[Decimal]


@dataclass
class DocumentSet:
    deal_id: str
    quotes: list[Doc] = field(default_factory=list)
    pos: list[Doc] = field(default_factory=list)
    invoices: list[Doc] = field(default_factory=list)
    duplicates: list[DuplicateFlag] = field(default_factory=list)


@dataclass
class LineLink:
    invoice: Doc
    inv_line: Line
    po: Doc
    po_line: Optional[Line]
    confidence: float
    rollup: bool = False


@dataclass
class Links:
    invoice_po: dict = field(default_factory=dict)   # invoice id -> PO Doc, or None
    bad_refs: set = field(default_factory=set)       # invoice ids naming a PO that is not there
    no_ref: set = field(default_factory=set)         # invoice ids naming no PO at all
    line_links: list = field(default_factory=list)   # list[LineLink]
    po_quote: dict = field(default_factory=dict)     # PO id -> quote Doc, or None


@dataclass
class Result:
    deal_id: str
    rule_id: str
    field_class: str
    outcome: Outcome
    claim_doc: Optional[str]
    field_name: str
    claim_line: Optional[str] = None
    auth_doc: Optional[str] = None
    auth_line: Optional[str] = None
    po_id: Optional[str] = None
    claim_value: Optional[str] = None
    auth_value: Optional[str] = None
    delta: Optional[Decimal] = None        # claim - authoritative, signed
    exposure: Decimal = ZERO               # money at risk, in `currency`
    currency: Optional[str] = None
    fx_to_gbp: Optional[Decimal] = None
    fx_rate_date: Optional[datetime] = None
    basis_total: Optional[Decimal] = None  # claim document total, for the materiality threshold
    confidence: float = 1.0
    tolerance: dict = field(default_factory=dict)
    note: str = ""
    severity: Optional[Severity] = None
    score: Optional[float] = None
    score_inputs: dict = field(default_factory=dict)

    @property
    def cause_key(self) -> str:
        return f"{self.claim_doc or ''}|{self.claim_line or ''}"

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.deal_id, self.rule_id, self.cause_key)

    @property
    def exposure_gbp(self) -> Optional[Decimal]:
        if self.fx_to_gbp is None:
            return None
        return (abs(self.exposure) * self.fx_to_gbp).quantize(Decimal("0.01"))


def pct_change(r: Result) -> Optional[Decimal]:
    """The claim's difference as a percentage of the authoritative value."""
    try:
        auth = Decimal(str(r.auth_value))
    except Exception:
        return None
    if auth == 0 or r.delta is None:
        return None
    return r.delta / auth * 100


@dataclass
class Finding:
    """One cause (or one group of same-cause results) and its knock-on effects."""
    deal_id: str
    rule_id: str
    causes: list[Result]
    cause_key: str
    effects: list[Result] = field(default_factory=list)
    headline: str = ""
    text: str = ""

    @property
    def lead(self) -> Result:
        return self.causes[0]

    @property
    def severity(self) -> Severity:
        # Effects count too: an always-S1 effect (over-billing) must never be softened
        # by being attached to a milder cause.
        return max(r.severity for r in (*self.causes, *self.effects) if r.severity is not None)

    @property
    def exposure(self) -> Decimal:
        return sum((abs(r.exposure) for r in self.causes), ZERO)

    @property
    def exposure_gbp(self) -> Optional[Decimal]:
        values = [r.exposure_gbp for r in self.causes]
        return None if any(v is None for v in values) else sum(values, ZERO)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.deal_id, self.rule_id, self.cause_key)

    @property
    def category(self) -> str:
        return CATEGORY.get(self.rule_id, "other")

    @property
    def confidence(self) -> float:
        return min(r.confidence for r in self.causes)


@dataclass
class Verdict:
    deal_id: str
    verdict: str
    s1: int
    s2: int
    notes: int
    exposure_gbp: Decimal
    incomplete: bool

    @property
    def summary(self) -> str:
        parts = [self.verdict]
        action = self.s1 + self.s2
        if action:
            parts.append(f"{action} finding{'s' if action != 1 else ''} "
                         f"need{'s' if action == 1 else ''} action")
        if self.notes:
            parts.append(f"{self.notes} note{'s' if self.notes != 1 else ''}")
        if self.exposure_gbp:
            parts.append(f"exposure {money(self.exposure_gbp, 'GBP')}")
        return " · ".join(parts)
```

`src/services/triage/tolerance.py`:

```python
"""Tolerances for triage: the ONLY place one is decided (spec §5.3).

Every value comes from governed-limit policy `triage_tolerances`. A missing or null
value refuses (LimitUnavailable) rather than falling back to a number in code — a
guard that cannot tell "policy missing" from "policy says 1%" is not a guard.

`resolve_tolerance(check, cfg, ctx)` takes a context it does not yet use. That is the
seam for the triage spec's leniency hierarchy (§7.3: customer, category, counterparty,
document, field), which arrives when the data has those dimensions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional

from src.services import governed_limits
from src.services.governed_limits import LimitUnavailable

POLICY = "triage_tolerances"


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def _rates(value: Any) -> tuple:
    return tuple(Decimal(str(v)) for v in value)


KEYS: dict[str, Callable[[Any], Any]] = {
    "unit_price_over_pct": _dec, "unit_price_over_abs": _dec, "unit_price_combine": str,
    "unit_price_under_pct": _dec, "quantity_over_pct": _dec, "rounding_per_line": _dec,
    "cumulative_total_pct": _dec, "cumulative_total_abs": _dec,
    "cumulative_total_combine": str, "allowed_tax_rates": _rates,
    "min_link_confidence": float, "unlinked_below": float,
    "min_extraction_confidence": float, "description_min_similarity": float,
    "materiality_pct_of_total": _dec, "materiality_floor": _dec,
    "materiality_ceiling": _dec, "band_s1": float, "band_s2": float,
    "uplift_min_lines": int, "uplift_same_pct_within": _dec, "batch_size": int,
}

_COMBINES = ("min", "max", "pct_only", "abs_only")


@dataclass(frozen=True)
class TriageConfig:
    values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({k: str(v) for k, v in sorted(self.values.items())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(read: Optional[Callable[..., Any]] = None) -> TriageConfig:
    read = read or governed_limits.limit
    values = {}
    for key, cast in KEYS.items():
        value = read(POLICY, key, cast=cast)
        if value is None:
            raise LimitUnavailable(f"{POLICY} states null for {key!r}; triage needs a value")
        values[key] = value
    return TriageConfig(values)


@dataclass(frozen=True)
class Tolerance:
    pct: Decimal
    abs_gbp: Optional[Decimal]
    combine: str
    source: str

    def allowance(self, base: Decimal, fx_to_gbp: Optional[Decimal]) -> Decimal:
        """How far a value may differ from `base` (document currency) and still pass."""
        if self.combine not in _COMBINES:
            raise ValueError(f"unknown combine {self.combine!r}")
        pct_part = abs(base) * self.pct / Decimal(100)
        abs_part = (self.abs_gbp / fx_to_gbp
                    if self.abs_gbp is not None and fx_to_gbp else None)
        if self.combine == "pct_only" or abs_part is None:
            return pct_part
        if self.combine == "abs_only":
            return abs_part
        return min(pct_part, abs_part) if self.combine == "min" else max(pct_part, abs_part)

    def as_dict(self) -> dict:
        return {"pct": str(self.pct),
                "abs_gbp": None if self.abs_gbp is None else str(self.abs_gbp),
                "combine": self.combine, "source": self.source}


def resolve_tolerance(check: str, cfg: TriageConfig, ctx: Optional[dict] = None) -> Tolerance:
    src = f"bp_policy:{POLICY}"
    if check == "unit_price_over":
        return Tolerance(cfg["unit_price_over_pct"], cfg["unit_price_over_abs"],
                         cfg["unit_price_combine"], f"{src}.unit_price_over_*")
    if check == "unit_price_under":
        return Tolerance(cfg["unit_price_under_pct"], None, "pct_only",
                         f"{src}.unit_price_under_pct")
    if check == "quantity_over":
        return Tolerance(cfg["quantity_over_pct"], None, "pct_only",
                         f"{src}.quantity_over_pct")
    if check == "cumulative_total":
        return Tolerance(cfg["cumulative_total_pct"], cfg["cumulative_total_abs"],
                         cfg["cumulative_total_combine"], f"{src}.cumulative_total_*")
    raise KeyError(f"no tolerance defined for {check!r}")
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_model.py tests/triage/test_tolerance.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
P="src/services/triage/__init__.py src/services/triage/model.py src/services/triage/tolerance.py tests/triage/__init__.py tests/triage/helpers.py tests/triage/test_model.py tests/triage/test_tolerance.py"
git add $P && git commit -o $P -m "feat(triage): data model and governed tolerances

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Normalisation helpers

**Files:**
- Create: `src/services/triage/normalise.py`
- Test: `tests/triage/test_normalise.py`

**Interfaces:**
- Produces: `to_decimal(v) -> Optional[Decimal]`, `to_confidence(v) -> Optional[float]` (0–100 scale becomes 0–1), `norm_text(v) -> str`, `similarity(a, b) -> float` (0–1), `terms_days(v) -> Optional[int]`.

- [ ] **Step 1: Write the failing test**

`tests/triage/test_normalise.py`:

```python
from decimal import Decimal as D

from src.services.triage.normalise import (
    norm_text, similarity, terms_days, to_confidence, to_decimal)


def test_to_decimal():
    assert to_decimal("1,234.50") == D("1234.50")
    assert to_decimal(12) == D("12")
    assert to_decimal("") is None
    assert to_decimal(None) is None
    assert to_decimal("n/a") is None


def test_to_confidence_reads_both_scales():
    assert to_confidence(D("88.89")) == 0.8889
    assert to_confidence(0.5) == 0.5
    assert to_confidence(None) is None


def test_norm_text():
    assert norm_text("  Steel-Bolts, M8 ") == "steel bolts m8"
    assert norm_text(None) == ""


def test_similarity():
    assert similarity("Widget large", "widget  LARGE") == 1.0
    assert similarity("Steel bolts M8", "Office chair") < 0.4
    assert similarity("", "x") == 0.0


def test_terms_days():
    assert terms_days("30 days — due 30 Jul 2025") == 30
    assert terms_days("Net 30") == 30
    assert terms_days("60 days") == 60
    assert terms_days("on receipt") is None
    assert terms_days(None) is None
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_normalise.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 3: Implement**

`src/services/triage/normalise.py`:

```python
"""Normalisation helpers (spec §4, step 1). Pure."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Optional

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DAYS = re.compile(r"(\d+)\s*days?\b")
_NET = re.compile(r"\bnet\s*(\d+)\b")


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def to_confidence(value: Any) -> Optional[float]:
    """Extraction confidence as 0-1. The _trgt tables store 0-100."""
    d = to_decimal(value)
    if d is None:
        return None
    f = float(d)
    return round(f / 100.0, 6) if f > 1.0 else f


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(_NON_ALNUM.sub(" ", str(value).lower()).split())


def similarity(a: Any, b: Any) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def terms_days(value: Any) -> Optional[int]:
    """'30 days — due 30 Jul 2025' and 'Net 30' are both 30."""
    text = norm_text(value)
    m = _DAYS.search(text) or _NET.search(text)
    return int(m.group(1)) if m else None
```

- [ ] **Step 4: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_normalise.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/normalise.py tests/triage/test_normalise.py"
git add $P && git commit -o $P -m "feat(triage): normalisation helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Batched loader

**Files:**
- Create: `src/services/triage/loader.py`
- Test: `tests/triage/test_loader_live.py`

**Interfaces:**
- Consumes: `model.Doc/Line/DocumentSet/DuplicateFlag`, `normalise.to_decimal/to_confidence`, `src.services.facts.fx.resolve_fx(cur, from_ccy, to_ccy) -> FxResult(rate, rate_date, source, reason_codes)`.
- Produces: `ALL_DEALS_SQL`, `list_deal_ids(cur) -> list[str]`, `load_deal_sets(cur, deal_ids) -> dict[str, DocumentSet]` (deals with no documents are absent from the result).

- [ ] **Step 1: Write the failing live test**

`tests/triage/test_loader_live.py`:

```python
"""Loader against the real database. Run with PROCWISE_TEST_LIVE_DB=1."""
import os

import pytest

from src.services.db import get_conn
from src.services.triage.loader import list_deal_ids, load_deal_sets

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")

KNOWN = "DEALV2-005049"   # Kestrel: one INR PO, three invoices


class CountingCursor:
    def __init__(self, cur):
        self.cur, self.n = cur, 0

    def execute(self, *args):
        self.n += 1
        return self.cur.execute(*args)

    def fetchall(self):
        return self.cur.fetchall()

    def fetchone(self):
        return self.cur.fetchone()


def test_known_deal_loads_with_lines_and_fx():
    with get_conn() as conn:
        sets = load_deal_sets(conn.cursor(), [KNOWN])
    ds = sets[KNOWN]
    assert len(ds.pos) >= 1 and ds.pos[0].lines
    assert len(ds.invoices) >= 3
    assert all(i.lines for i in ds.invoices)
    assert ds.invoices[0].currency == "INR"
    assert ds.invoices[0].fx_to_gbp is not None


def test_unknown_deal_is_absent():
    with get_conn() as conn:
        assert load_deal_sets(conn.cursor(), ["NO-SUCH-DEAL"]) == {}


def test_query_count_does_not_grow_with_batch_size():
    with get_conn() as conn:
        ids = list_deal_ids(conn.cursor())[:50]
        cur = CountingCursor(conn.cursor())
        sets = load_deal_sets(cur, ids)
    assert len(sets) == 50
    # 7 table queries + at most 4 FX lookups per distinct currency (5 in the corpus)
    assert cur.n <= 7 + 4 * 5
```

- [ ] **Step 2: Run to see it fail**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_loader_live.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 3: Implement**

`src/services/triage/loader.py`:

```python
"""Batch loading of deal document sets from the _trgt tables (spec §4).

A fixed number of queries per batch whatever its size — deal assignment showed that a
query per document does not survive 5,000 deals. Read-only.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from src.services.facts.fx import resolve_fx

from .model import Doc, DocumentSet, DuplicateFlag, Line
from .normalise import to_confidence, to_decimal

BASE_CURRENCY = "GBP"

ALL_DEALS_SQL = """
SELECT deal_id FROM (
    SELECT deal_id FROM proc.bp_purchase_order_trgt
    UNION SELECT deal_id FROM proc.bp_invoice_trgt
    UNION SELECT deal_id FROM proc.bp_quote_trgt
) d WHERE deal_id IS NOT NULL ORDER BY deal_id
"""

_INVOICES = """
SELECT invoice_id, deal_id, po_id, supplier_id, currency, invoice_date,
       invoice_amount, tax_amount, invoice_total_incl_tax, payment_terms, confidence_score
  FROM proc.bp_invoice_trgt WHERE deal_id = ANY(%s)
"""
_INVOICE_LINES = """
SELECT invoice_id, COALESCE(line_no::text, invoice_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_amount, po_id, delivery_date
  FROM proc.bp_invoice_line_items_trgt WHERE invoice_id = ANY(%s)
 ORDER BY invoice_id, line_no
"""
_POS = """
SELECT po_id, deal_id, supplier_id, currency, order_date, total_amount, tax_amount,
       total_amount_incl_tax, payment_terms, quote_reference, confidence_score
  FROM proc.bp_purchase_order_trgt WHERE deal_id = ANY(%s) OR po_id = ANY(%s)
"""
_PO_LINES = """
SELECT po_id, COALESCE(line_number::text, po_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_total
  FROM proc.bp_po_line_items_trgt WHERE po_id = ANY(%s)
 ORDER BY po_id, line_number
"""
_QUOTES = """
SELECT quote_id, deal_id, supplier_id, currency, quote_date, total_amount, tax_amount,
       total_amount_incl_tax, confidence_score
  FROM proc.bp_quote_trgt WHERE deal_id = ANY(%s) OR quote_id = ANY(%s)
"""
_QUOTE_LINES = """
SELECT quote_id, COALESCE(line_number::text, quote_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_total
  FROM proc.bp_quote_line_items_trgt WHERE quote_id = ANY(%s)
 ORDER BY quote_id, line_number
"""
_DUPLICATES = """
SELECT doc_pk_candidate, raw_value, notes FROM proc.bp_extraction_discrepancy
 WHERE issue_type = 'duplicate_invoice' AND status = 'open' AND doc_pk_candidate = ANY(%s)
"""
_EARLIER = re.compile(r"possible duplicate of (\S+)")


def _s(value) -> Optional[str]:
    return None if value is None else str(value)


def list_deal_ids(cur) -> list[str]:
    cur.execute(ALL_DEALS_SQL)
    return [r[0] for r in cur.fetchall()]


def _line(row) -> Line:
    _doc, ref, item, desc, qty, uom, price, amount, *rest = row
    return Line(line_ref=str(ref), item_id=_s(item), description=desc,
                quantity=to_decimal(qty), uom=uom, unit_price=to_decimal(price),
                line_amount=to_decimal(amount),
                po_id=_s(rest[0]) if rest else None,
                delivery_date=rest[1] if len(rest) > 1 else None)


def _lines_by_doc(cur, sql: str, ids) -> dict[str, list[Line]]:
    out: dict[str, list[Line]] = {}
    if not ids:
        return out
    cur.execute(sql, (list(ids),))
    for row in cur.fetchall():
        out.setdefault(str(row[0]), []).append(_line(row))
    return out


def load_deal_sets(cur, deal_ids: Iterable[str]) -> dict[str, DocumentSet]:
    wanted = list(dict.fromkeys(str(d) for d in deal_ids if d))
    if not wanted:
        return {}
    sets = {d: DocumentSet(d) for d in wanted}

    # Invoices and their lines.
    cur.execute(_INVOICES, (wanted,))
    invoices: list[tuple[str, Doc]] = []
    for (inv_id, deal, po_id, sup, ccy, when, net, tax, gross, terms, conf) in cur.fetchall():
        invoices.append((str(deal), Doc(
            doc_id=str(inv_id), doc_type="invoice", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            payment_terms=terms, po_id=_s(po_id), confidence=to_confidence(conf))))
    inv_lines = _lines_by_doc(cur, _INVOICE_LINES, [d.doc_id for _, d in invoices])
    for deal, doc in invoices:
        doc.lines = inv_lines.get(doc.doc_id, [])
        sets[deal].invoices.append(doc)

    # POs in these deals, plus any PO an invoice here names.
    po_refs = sorted({d.po_ref for _, d in invoices if d.po_ref})
    cur.execute(_POS, (wanted, po_refs))
    pos: dict[str, Doc] = {}
    po_deal: dict[str, Optional[str]] = {}
    for (po_id, deal, sup, ccy, when, net, tax, gross, terms, qref, conf) in cur.fetchall():
        pos[str(po_id)] = Doc(
            doc_id=str(po_id), doc_type="purchase_order", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            payment_terms=terms, quote_ref=_s(qref), confidence=to_confidence(conf))
        po_deal[str(po_id)] = _s(deal)
    po_lines = _lines_by_doc(cur, _PO_LINES, list(pos))
    po_owners: dict[str, set] = {}
    for po_id, doc in pos.items():
        doc.lines = po_lines.get(po_id, [])
        owners = {po_deal[po_id]} | {deal for deal, i in invoices if i.po_ref == po_id}
        po_owners[po_id] = owners
        for owner in owners:
            if owner in sets:
                sets[owner].pos.append(doc)

    # Quotes in these deals, plus any quote a PO here references.
    quote_refs = sorted({d.quote_ref for d in pos.values() if d.quote_ref})
    cur.execute(_QUOTES, (wanted, quote_refs))
    quotes: dict[str, Doc] = {}
    quote_deal: dict[str, Optional[str]] = {}
    for (qid, deal, sup, ccy, when, net, tax, gross, conf) in cur.fetchall():
        quotes[str(qid)] = Doc(
            doc_id=str(qid), doc_type="quote", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            confidence=to_confidence(conf))
        quote_deal[str(qid)] = _s(deal)
    quote_lines = _lines_by_doc(cur, _QUOTE_LINES, list(quotes))
    for qid, doc in quotes.items():
        doc.lines = quote_lines.get(qid, [])
        owners = {quote_deal[qid]}
        for po_id, p in pos.items():
            if p.quote_ref == qid:
                owners |= po_owners[po_id]
        for owner in owners:
            if owner in sets:
                sets[owner].quotes.append(doc)

    # Open duplicate findings from the duplicate-invoice detector (read, never re-detected).
    inv_deal = {d.doc_id: deal for deal, d in invoices}
    if inv_deal:
        cur.execute(_DUPLICATES, (list(inv_deal),))
        for pk, raw, notes in cur.fetchall():
            m = _EARLIER.search(notes or "")
            sets[inv_deal[str(pk)]].duplicates.append(
                DuplicateFlag(str(pk), m.group(1) if m else None, to_decimal(raw)))

    # FX to GBP, once per currency per batch.
    rates = {}
    for s in sets.values():
        for doc in (*s.quotes, *s.pos, *s.invoices):
            ccy = (doc.currency or "").strip().upper()
            if not ccy:
                continue
            if ccy not in rates:
                rates[ccy] = resolve_fx(cur, ccy, BASE_CURRENCY)
            doc.fx_to_gbp, doc.fx_rate_date = rates[ccy].rate, rates[ccy].rate_date

    return {d: s for d, s in sets.items() if s.quotes or s.pos or s.invoices}
```

- [ ] **Step 4: Run the live test**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_loader_live.py -q -p no:randomly`
Expected: 3 passed (not skipped — if it says skipped, the flag was not set).

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/loader.py tests/triage/test_loader_live.py"
git add $P && git commit -o $P -m "feat(triage): batched loader for deal document sets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Linking

**Files:**
- Create: `src/services/triage/link.py`
- Test: `tests/triage/test_link.py`

**Interfaces:**
- Consumes: `model.Links/LineLink/DocumentSet/Doc/Line`, `normalise.similarity`, `TriageConfig` keys `unlinked_below`, `rounding_per_line`.
- Produces: `link(ds, cfg) -> Links`.

- [ ] **Step 1: Write the failing test**

`tests/triage/test_link.py`:

```python
from decimal import Decimal as D

from src.services.triage.link import link
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def test_item_code_match_links_with_full_confidence():
    ds = deal(po(), inv())
    links = link(ds, CFG)
    (lk,) = links.line_links
    assert lk.po_line.line_ref == "1" and lk.confidence == 1.0
    assert links.invoice_po["INV-1"].doc_id == "PO-1"


def test_repeated_item_prefers_same_price_line():
    ds = deal(po(lines=[line(1, price="10.00"), line(2, price="12.00")]),
              inv(lines=[line(1, price="12.00")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line.line_ref == "2"


def test_description_fallback_links_below_full_confidence():
    ds = deal(po(lines=[line(1, item=None, desc="Widget large")]),
              inv(lines=[line(1, item=None, desc="Widgets, large")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line is not None and 0.5 <= lk.confidence < 1.0


def test_dissimilar_line_is_unlinked():
    ds = deal(po(), inv(lines=[line(1, item="FRT", desc="Expedited freight")]))
    (lk,) = link(ds, CFG).line_links
    assert lk.po_line is None


def test_po_number_on_lines_is_used_when_header_is_blank():
    ds = deal(po(), inv(po_id=None, lines=[line(1, po_id="PO-1")]))
    assert link(ds, CFG).invoice_po["INV-1"].doc_id == "PO-1"


def test_bad_and_missing_po_references():
    ds = deal(po(), inv("INV-1", po_id="PO-404"), inv("INV-2", po_id=None))
    links = link(ds, CFG)
    assert links.bad_refs == {"INV-1"} and links.no_ref == {"INV-2"}
    assert links.line_links == []


def test_itemised_lines_roll_up_into_one_po_line():
    ds = deal(po(lines=[line(1), line(2, item=None, desc="Installation", qty="1", price="5000.00")]),
              inv(lines=[line(1)] + [line(n, item=None, desc=f"Day {n}", qty="1", price="1250.00")
                                     for n in (2, 3, 4, 5)]))
    links = link(ds, CFG)
    rolled = [lk for lk in links.line_links if lk.rollup]
    assert len(rolled) == 4 and {lk.po_line.line_ref for lk in rolled} == {"2"}


def test_po_to_quote():
    ds = deal(quote(), po(quote_ref="Q-1"), inv())
    assert link(ds, CFG).po_quote["PO-1"].doc_id == "Q-1"
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_link.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 3: Implement**

`src/services/triage/link.py`:

```python
"""Which invoice belongs to which PO, and which invoice line to which PO line (spec §5.1).

Also the group match of §6: an invoice's unlinked lines that together equal one PO line
nothing else linked to are a roll-up (EXPLAINED), not N unlinked lines. Pure.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from .model import Doc, DocumentSet, Line, LineLink, Links
from .normalise import similarity


def _gap(a: Line, b: Line) -> Decimal:
    if a.line_amount is None or b.line_amount is None:
        return Decimal("Infinity")
    return abs(a.line_amount - b.line_amount)


def _link_line(inv: Doc, line: Line, po: Doc, cfg) -> LineLink:
    if line.item_id:
        same = [pl for pl in po.lines if pl.item_id == line.item_id]
        if same:
            exact = [pl for pl in same if pl.unit_price == line.unit_price]
            return LineLink(inv, line, po, (exact or same)[0], 1.0)
    best: Optional[Line] = None
    best_key = None
    for pl in po.lines:
        key = (similarity(line.description, pl.description), -_gap(line, pl))
        if best_key is None or key > best_key:
            best, best_key = pl, key
    score = best_key[0] if best_key else 0.0
    return LineLink(inv, line, po, best if score >= cfg["unlinked_below"] else None,
                    round(score, 4))


def _rollup(links: list[LineLink], po: Doc, cfg) -> list[LineLink]:
    unlinked = [lk for lk in links if lk.po_line is None]
    if len(unlinked) < 2:
        return links
    total = sum((lk.inv_line.line_amount or Decimal("0")) for lk in unlinked)
    used = {id(lk.po_line) for lk in links if lk.po_line is not None}
    tol = cfg["rounding_per_line"] * len(unlinked)
    for pl in po.lines:
        if id(pl) in used or pl.line_amount is None:
            continue
        if abs(pl.line_amount - total) <= tol:
            for lk in unlinked:
                lk.po_line, lk.confidence, lk.rollup = pl, 1.0, True
            break
    return links


def link(ds: DocumentSet, cfg) -> Links:
    pos = {p.doc_id: p for p in ds.pos}
    quotes = {q.doc_id: q for q in ds.quotes}
    out = Links()
    for inv in ds.invoices:
        ref = inv.po_ref
        po = pos.get(ref) if ref else None
        out.invoice_po[inv.doc_id] = po
        if ref is None:
            out.no_ref.add(inv.doc_id)
            continue
        if po is None:
            out.bad_refs.add(inv.doc_id)
            continue
        out.line_links.extend(_rollup([_link_line(inv, l, po, cfg) for l in inv.lines], po, cfg))
    out.po_quote = {p.doc_id: (quotes.get(p.quote_ref) if p.quote_ref else None) for p in ds.pos}
    return out
```

- [ ] **Step 4: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_link.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/link.py tests/triage/test_link.py"
git add $P && git commit -o $P -m "feat(triage): link invoices and lines to POs, with roll-ups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Line-level checks

**Files:**
- Create: `src/services/triage/checks.py`
- Test: `tests/triage/test_checks_lines.py`

**Interfaces:**
- Consumes: `link.link`, `model.*`, `normalise.similarity/norm_text/terms_days`, `tolerance.resolve_tolerance`.
- Produces: `check_unit_price`, `check_quantity`, `check_line_arithmetic`, `check_description`, `check_unlinked_lines` — each `(ds, links, cfg) -> list[Result]`; `run_checks(ds, links, cfg) -> list[Result]` (extended in Task 7).

- [ ] **Step 1: Write the failing test**

`tests/triage/test_checks_lines.py`:

```python
from datetime import date
from decimal import Decimal as D

from src.services.triage.checks import (
    check_description, check_line_arithmetic, check_quantity, check_unit_price,
    check_unlinked_lines)
from src.services.triage.link import link
from src.services.triage.model import Outcome
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def _run(check, ds):
    return check(ds, link(ds, CFG), CFG)


def _one(results):
    assert len(results) == 1, results
    return results[0]


# --- unit price -------------------------------------------------------------

def test_price_above_po_beyond_tolerance_is_conflict_with_exposure():
    ds = deal(po(lines=[line(1, qty="300", price="12.00")]),
              inv(lines=[line(1, qty="300", price="13.50")]))
    r = _one(_run(check_unit_price, ds))
    assert r.outcome == Outcome.CONFLICT
    assert r.exposure == D("450")
    assert r.delta == D("1.50")
    assert (r.claim_doc, r.auth_doc, r.po_id, r.claim_line) == ("INV-1", "PO-1", "PO-1", "1")


def test_price_within_one_percent_is_within_tolerance():
    ds = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="12.10")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.WITHIN_TOL


def test_price_just_beyond_one_percent_conflicts():
    ds = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="12.13")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.CONFLICT


def test_underpricing_has_a_looser_tolerance():
    ok = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="11.50")]))
    bad = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="11.00")]))
    assert _one(_run(check_unit_price, ok)).outcome == Outcome.WITHIN_TOL
    assert _one(_run(check_unit_price, bad)).outcome == Outcome.CONFLICT


def test_price_falls_back_to_quote_when_po_line_has_none():
    ds = deal(quote(lines=[line(1, price="12.00")]),
              po(quote_ref="Q-1", lines=[line(1, price=None, amount="120")]),
              inv(lines=[line(1, price="13.50")]))
    r = _one(_run(check_unit_price, ds))
    assert r.auth_doc == "Q-1" and r.outcome == Outcome.CONFLICT


def test_low_extraction_confidence_makes_conflict_unverifiable():
    ds = deal(po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="13.50")], confidence=0.5))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.UNVERIFIABLE


def test_credit_note_lines_are_not_price_checked():
    ds = deal(po(), inv("CN-1", lines=[line(1, price="99", amount="-990")], net="-990"))
    assert _run(check_unit_price, ds) == []


def test_line_without_price_or_quantity_is_skipped_not_crashed():
    ds = deal(po(), inv(lines=[line(1, qty=None, price=None, amount="120")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.ABSENT_SUBORDINATE
    assert _run(check_quantity, ds) == []
    assert _run(check_line_arithmetic, ds) == []


# --- quantity ---------------------------------------------------------------

def test_partial_invoicing_is_explained():
    ds = deal(po(lines=[line(1, qty="10")]), inv(lines=[line(1, qty="6")]))
    r = _one(_run(check_quantity, ds))
    assert r.outcome == Outcome.EXPLAINED and "partially invoiced" in r.note


def test_cumulative_quantity_over_po_conflicts_on_the_later_invoice():
    ds = deal(po(lines=[line(1, qty="10")]),
              inv("INV-1", lines=[line(1, qty="6")], inv_date=date(2026, 2, 1)),
              inv("INV-2", lines=[line(1, qty="6")], inv_date=date(2026, 3, 1)))
    r = _one(_run(check_quantity, ds))
    assert r.outcome == Outcome.CONFLICT
    assert r.claim_doc == "INV-2" and r.exposure == D("24")
    assert "INV-1" in r.note and "INV-2" in r.note


def test_small_over_delivery_is_within_allowance():
    ds = deal(po(lines=[line(1, qty="10")]), inv(lines=[line(1, qty="10.4")]))
    assert _one(_run(check_quantity, ds)).outcome == Outcome.WITHIN_TOL


def test_credit_note_quantity_is_subtracted():
    ds = deal(po(lines=[line(1, qty="10")]),
              inv("INV-1", lines=[line(1, qty="10")]),
              inv("INV-2", lines=[line(1, qty="10")]),
              inv("CN-1", lines=[line(1, qty="10", amount="-120")], net="-120"))
    assert _one(_run(check_quantity, ds)).outcome == Outcome.EXPLAINED


# --- line arithmetic ---------------------------------------------------------

def test_line_amount_not_equal_to_qty_times_price_conflicts():
    ds = deal(po(), inv(lines=[line(1, amount="125.00")]))
    r = _one(_run(check_line_arithmetic, ds))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("5.00")


def test_rounding_difference_is_within_tolerance():
    ds = deal(po(), inv(lines=[line(1, amount="120.01")]))
    assert _one(_run(check_line_arithmetic, ds)).outcome == Outcome.WITHIN_TOL


def test_credit_note_line_arithmetic_uses_magnitudes():
    ds = deal(po(), inv("CN-1", lines=[line(1, amount="-120")], net="-120"))
    assert _one(_run(check_line_arithmetic, ds)).outcome == Outcome.MATCH


# --- description and unlinked lines ------------------------------------------

def test_description_conflict_when_item_codes_match_but_text_differs():
    ds = deal(po(lines=[line(1, desc="Steel bolts M8")]), inv(lines=[line(1, desc="Office chair")]))
    r = _one(_run(check_description, ds))
    assert r.outcome == Outcome.CONFLICT and r.field_class == "description"


def test_invoice_line_with_no_po_line_is_absent_authoritative():
    ds = deal(po(), inv(lines=[line(1), line(2, item="FRT", desc="Expedited freight",
                                           qty="1", price="120.00")]))
    r = _one(_run(check_unlinked_lines, ds))
    assert r.outcome == Outcome.ABSENT_AUTHORITATIVE
    assert r.exposure == D("120.00") and r.claim_line == "2"


def test_rolled_up_lines_are_explained_notes():
    ds = deal(po(lines=[line(1), line(2, item=None, desc="Installation", qty="1", price="5000.00")]),
              inv(lines=[line(1)] + [line(n, item=None, desc=f"Day {n}", qty="1", price="1250.00")
                                     for n in (2, 3, 4, 5)]))
    rs = _run(check_unlinked_lines, ds)
    assert [r.outcome for r in rs] == [Outcome.EXPLAINED] * 4
    assert {r.rule_id for r in rs} == {"rollup"}
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_checks_lines.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 3: Implement the line checks**

`src/services/triage/checks.py`:

```python
"""The procure-to-pay checks (spec §5.2).

Each takes (ds, links, cfg) and returns Results, one outcome per compared pair. None
reads the database, the clock or the network.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

from .model import Doc, DocumentSet, Links, Outcome, Result
from .normalise import norm_text, similarity, terms_days
from .tolerance import resolve_tolerance

ZERO = Decimal("0")


def _s(value) -> Optional[str]:
    return None if value is None else str(value)


def _doc_conf(*docs: Optional[Doc]) -> float:
    c = 1.0
    for d in docs:
        if d is not None and d.confidence is not None:
            c *= d.confidence
    return c


def _fail(cfg, link_conf: float, *docs: Optional[Doc]) -> Outcome:
    """CONFLICT, unless a misread number or a weak link could explain it."""
    low_doc = any(d is not None and d.confidence is not None
                  and d.confidence < cfg["min_extraction_confidence"] for d in docs)
    if low_doc or link_conf < cfg["min_link_confidence"]:
        return Outcome.UNVERIFIABLE
    return Outcome.CONFLICT


def _r(ds: DocumentSet, rule: str, cls: str, outcome: Outcome, claim: Doc, field: str,
       **kw) -> Result:
    kw.setdefault("currency", claim.currency)
    kw.setdefault("fx_to_gbp", claim.fx_to_gbp)
    kw.setdefault("fx_rate_date", claim.fx_rate_date)
    kw.setdefault("basis_total", claim.gross if claim.gross is not None else claim.net)
    return Result(deal_id=ds.deal_id, rule_id=rule, field_class=cls, outcome=outcome,
                  claim_doc=claim.doc_id, field_name=field, **kw)


# --- 1. unit price ---------------------------------------------------------------

def check_unit_price(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.invoice.is_credit_note:
            continue
        claim = lk.inv_line.unit_price
        auth, auth_doc = lk.po_line.unit_price, lk.po.doc_id
        if auth is None:
            q = links.po_quote.get(lk.po.doc_id)
            ql = next((l for l in q.lines if l.item_id and l.item_id == lk.po_line.item_id),
                      None) if q else None
            if ql is not None and ql.unit_price is not None:
                auth, auth_doc = ql.unit_price, q.doc_id
        if claim is None and auth is None:
            continue
        common = dict(claim_line=lk.inv_line.line_ref, auth_doc=auth_doc,
                      auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id,
                      claim_value=_s(claim), auth_value=_s(auth),
                      confidence=_doc_conf(lk.invoice, lk.po) * lk.confidence)
        if claim is None:
            out.append(_r(ds, "unit_price", "money", Outcome.ABSENT_SUBORDINATE,
                          lk.invoice, "unit_price", **common))
            continue
        if auth is None:
            out.append(_r(ds, "unit_price", "money", Outcome.ABSENT_AUTHORITATIVE,
                          lk.invoice, "unit_price", **common))
            continue
        diff = claim - auth
        if diff == 0:
            out.append(_r(ds, "unit_price", "money", Outcome.MATCH, lk.invoice,
                          "unit_price", delta=diff, **common))
            continue
        tol = resolve_tolerance("unit_price_over" if diff > 0 else "unit_price_under", cfg)
        allow = tol.allowance(auth, lk.invoice.fx_to_gbp)
        outcome = (Outcome.WITHIN_TOL if abs(diff) <= allow
                   else _fail(cfg, lk.confidence, lk.invoice, lk.po))
        qty = lk.inv_line.quantity or ZERO
        out.append(_r(ds, "unit_price", "money", outcome, lk.invoice, "unit_price",
                      delta=diff, exposure=abs(diff * qty),
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 2. quantity (cumulative across invoices) -------------------------------------

def check_quantity(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    by_po_line = defaultdict(list)
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.inv_line.quantity is None:
            continue
        by_po_line[(lk.po.doc_id, lk.po_line.line_ref)].append(lk)
    out = []
    for (po_id, ref), lks in by_po_line.items():
        po_line, po = lks[0].po_line, lks[0].po
        if po_line.quantity is None:
            continue
        cum = sum(((-abs(l.inv_line.quantity)) if l.invoice.is_credit_note
                   else l.inv_line.quantity) for l in lks)
        last = max(lks, key=lambda l: (l.invoice.doc_date or date.min, l.invoice.doc_id,
                                       l.inv_line.line_ref))
        over = cum - po_line.quantity
        link_conf = min(l.confidence for l in lks)
        invoices = ", ".join(sorted({l.invoice.doc_id for l in lks}))
        common = dict(claim_line=last.inv_line.line_ref, auth_doc=po_id, auth_line=ref,
                      po_id=po_id, claim_value=_s(cum), auth_value=_s(po_line.quantity),
                      delta=over, confidence=link_conf * _doc_conf(last.invoice, po))
        if over <= 0:
            if over == 0 and len(lks) == 1:
                outcome, note = Outcome.MATCH, ""
            elif over < 0:
                outcome, note = Outcome.EXPLAINED, f"partially invoiced: {cum} of {po_line.quantity}"
            else:
                outcome, note = Outcome.EXPLAINED, f"invoiced across {len(lks)} lines ({invoices})"
            out.append(_r(ds, "quantity", "quantity", outcome, last.invoice, "quantity",
                          note=note, **common))
            continue
        tol = resolve_tolerance("quantity_over", cfg)
        allow = tol.allowance(po_line.quantity, None)
        outcome = (Outcome.WITHIN_TOL if over <= allow
                   else _fail(cfg, link_conf, last.invoice, po))
        price = po_line.unit_price or last.inv_line.unit_price or ZERO
        out.append(_r(ds, "quantity", "quantity", outcome, last.invoice, "quantity",
                      exposure=abs(over * price), note=f"invoiced on {invoices}",
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 3. line arithmetic -----------------------------------------------------------

def check_line_arithmetic(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    tol = cfg["rounding_per_line"]
    out = []
    for inv in ds.invoices:
        for l in inv.lines:
            if l.quantity is None or l.unit_price is None or l.line_amount is None:
                continue
            expected = abs(l.quantity * l.unit_price)
            diff = abs(l.line_amount) - expected
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= tol else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "line_arithmetic", "money", outcome, inv, "line_amount",
                          claim_line=l.line_ref, po_id=inv.po_ref,
                          claim_value=_s(l.line_amount), auth_value=_s(expected),
                          delta=diff, exposure=abs(diff), confidence=_doc_conf(inv),
                          tolerance={"rounding": str(tol)}))
    return out


# --- 11. description ----------------------------------------------------------------

def check_description(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.confidence < 1.0 or lk.invoice.is_credit_note:
            continue
        a, b = lk.inv_line.description, lk.po_line.description
        if not a or not b:
            continue
        sim = similarity(a, b)
        outcome = Outcome.MATCH if sim >= cfg["description_min_similarity"] else Outcome.CONFLICT
        out.append(_r(ds, "description", "description", outcome, lk.invoice, "description",
                      claim_line=lk.inv_line.line_ref, auth_doc=lk.po.doc_id,
                      auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id, claim_value=a,
                      auth_value=b, note=f"similarity {sim:.2f}",
                      confidence=_doc_conf(lk.invoice, lk.po)))
    return out


# --- 12. unlinked lines, and roll-ups -------------------------------------------------

def check_unlinked_lines(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.rollup:
            out.append(_r(ds, "rollup", "reference", Outcome.EXPLAINED, lk.invoice, "line",
                          claim_line=lk.inv_line.line_ref, auth_doc=lk.po.doc_id,
                          auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description,
                          note=f"itemises PO line {lk.po_line.line_ref}"))
        elif lk.po_line is None:
            out.append(_r(ds, "unlinked_line", "money", Outcome.ABSENT_AUTHORITATIVE,
                          lk.invoice, "line", claim_line=lk.inv_line.line_ref,
                          auth_doc=lk.po.doc_id, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description or lk.inv_line.item_id,
                          exposure=abs(lk.inv_line.line_amount or ZERO),
                          confidence=_doc_conf(lk.invoice)))
    return out


LINE_CHECKS = (check_unit_price, check_quantity, check_line_arithmetic,
               check_description, check_unlinked_lines)


def run_checks(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    return [r for check in LINE_CHECKS for r in check(ds, links, cfg)]
```

- [ ] **Step 4: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_checks_lines.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/checks.py tests/triage/test_checks_lines.py"
git add $P && git commit -o $P -m "feat(triage): line-level checks (price, quantity, arithmetic, description, unlinked)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Document-level checks

**Files:**
- Modify: `src/services/triage/checks.py` (append checks, replace `run_checks`)
- Test: `tests/triage/test_checks_documents.py`

**Interfaces:**
- Produces: `check_invoice_totals`, `check_cumulative_total`, `check_tax_rate`, `check_currency`, `check_supplier`, `check_invoice_date`, `check_payment_terms`, `check_duplicates`, `check_po_links`, `DOC_CHECKS`, and the final `run_checks` (all fifteen checks).

- [ ] **Step 1: Write the failing test**

`tests/triage/test_checks_documents.py`:

```python
from datetime import date
from decimal import Decimal as D

from src.services.triage.checks import (
    check_cumulative_total, check_currency, check_duplicates, check_invoice_date,
    check_invoice_totals, check_payment_terms, check_po_links, check_supplier,
    check_tax_rate, run_checks)
from src.services.triage.link import link
from src.services.triage.model import DuplicateFlag, Outcome
from tests.triage.helpers import deal, inv, line, make_cfg, po

CFG = make_cfg()


def _run(check, ds):
    return check(ds, link(ds, CFG), CFG)


def _one(results):
    assert len(results) == 1, results
    return results[0]


def test_invoice_lines_not_summing_to_net_conflict():
    rs = _run(check_invoice_totals, deal(po(), inv(net="130")))
    net = next(r for r in rs if r.field_name == "net")
    assert net.outcome == Outcome.CONFLICT and net.exposure == D("10")


def test_gross_not_equal_net_plus_tax_conflicts():
    rs = _run(check_invoice_totals, deal(po(), inv(gross="150")))
    gross = next(r for r in rs if r.field_name == "gross")
    assert gross.outcome == Outcome.CONFLICT and gross.exposure == D("6")


def test_invoices_beyond_po_total_conflict_on_the_po():
    ds = deal(po(), inv("INV-1"), inv("INV-2"))
    r = _one(_run(check_cumulative_total, ds))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("120")
    assert r.claim_doc == "PO-1" and r.po_id == "PO-1"
    assert D(r.tolerance["allowance"]) == D("0.6")


def test_credit_note_brings_running_total_back_within_po():
    ds = deal(po(), inv("INV-1"), inv("INV-2"), inv("CN-1", net="-120"))
    assert _one(_run(check_cumulative_total, ds)).outcome == Outcome.MATCH


def test_partly_invoiced_po_is_explained():
    ds = deal(po(net="500", lines=[line(1, qty="50")]), inv())
    assert _one(_run(check_cumulative_total, ds)).outcome == Outcome.EXPLAINED


def test_tax_at_an_allowed_rate_matches_and_off_rate_conflicts():
    assert _one(_run(check_tax_rate, deal(po(), inv()))).outcome == Outcome.MATCH
    assert _one(_run(check_tax_rate, deal(po(), inv(tax="6")))).outcome == Outcome.MATCH
    r = _one(_run(check_tax_rate, deal(po(), inv(tax="25"))))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("1.00")


def test_currency_mismatch_conflicts():
    r = _one(_run(check_currency, deal(po(), inv(currency="EUR"))))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("120")


def test_supplier_mismatch_conflicts():
    r = _one(_run(check_supplier, deal(po(), inv(supplier="SUP-2"))))
    assert r.outcome == Outcome.CONFLICT and r.field_class == "party"


def test_invoice_before_order_conflicts():
    r = _one(_run(check_invoice_date, deal(po(), inv(inv_date=date(2026, 1, 1)))))
    assert r.outcome == Outcome.CONFLICT


def test_payment_terms():
    assert _one(_run(check_payment_terms, deal(po(), inv()))).outcome == Outcome.MATCH
    assert _one(_run(check_payment_terms, deal(po(), inv(terms="60 days")))).outcome == Outcome.CONFLICT
    assert _one(_run(check_payment_terms, deal(po(), inv(terms=None)))).outcome == Outcome.ABSENT_SUBORDINATE
    assert _one(_run(check_payment_terms, deal(po(), inv(terms="on receipt")))).outcome == Outcome.UNVERIFIABLE


def test_flagged_duplicate_conflicts_with_invoice_net_as_exposure():
    ds = deal(po(), inv("INV-1"), inv("INV-2"),
              duplicates=[DuplicateFlag("INV-2", "INV-1", D("144"))])
    r = _one(_run(check_duplicates, ds))
    assert (r.claim_doc, r.auth_doc, r.po_id) == ("INV-2", "INV-1", "PO-1")
    assert r.exposure == D("120")


def test_bad_and_missing_po_references():
    ds = deal(po(), inv("INV-1", po_id="PO-404"), inv("INV-2", po_id=None))
    rs = {r.rule_id: r for r in _run(check_po_links, ds)}
    assert rs["bad_po_ref"].claim_doc == "INV-1" and rs["bad_po_ref"].claim_value == "PO-404"
    assert rs["no_po"].claim_doc == "INV-2"
    assert {r.outcome for r in rs.values()} == {Outcome.ABSENT_AUTHORITATIVE}


def test_clean_deal_produces_only_matches():
    rs = run_checks(deal(po(), inv()), link(deal(po(), inv()), CFG), CFG)
    assert rs and {r.outcome for r in rs} == {Outcome.MATCH}
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_checks_documents.py -q -p no:randomly`
Expected: ImportError — the new check names do not exist.

- [ ] **Step 3: Append the document checks**

In `src/services/triage/checks.py`, **replace** the block from `LINE_CHECKS = (` to the end of the file with:

```python
# --- 4. invoice totals ------------------------------------------------------------------

def check_invoice_totals(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    tol = cfg["rounding_per_line"]
    out = []
    for inv in ds.invoices:
        amounts = [l.line_amount for l in inv.lines]
        if inv.lines and inv.net is not None and all(a is not None for a in amounts):
            total = sum(amounts, ZERO)
            diff = abs(inv.net) - abs(total)
            allow = tol * max(1, len(inv.lines))
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= allow else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "invoice_totals", "money", outcome, inv, "net",
                          po_id=inv.po_ref, claim_value=_s(inv.net), auth_value=_s(total),
                          delta=diff, exposure=abs(diff), confidence=_doc_conf(inv),
                          tolerance={"rounding": str(allow)}))
        if inv.net is not None and inv.tax is not None and inv.gross is not None:
            expected = inv.net + inv.tax
            diff = inv.gross - expected
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= tol else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "invoice_totals", "money", outcome, inv, "gross",
                          po_id=inv.po_ref, claim_value=_s(inv.gross),
                          auth_value=_s(expected), delta=diff, exposure=abs(diff),
                          confidence=_doc_conf(inv), tolerance={"rounding": str(tol)}))
    return out


# --- 5. running total against the PO ---------------------------------------------------

def check_cumulative_total(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    by_po = defaultdict(list)
    for inv in ds.invoices:
        p = links.invoice_po.get(inv.doc_id)
        if p is not None:
            by_po[p.doc_id].append(inv)
    out = []
    for p in ds.pos:
        invs = by_po.get(p.doc_id)
        if not invs or p.net is None:
            continue
        total = sum((i.net for i in invs if i.net is not None), ZERO)
        over = total - p.net
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=_s(total),
                      auth_value=_s(p.net), delta=over,
                      note="invoiced by " + ", ".join(sorted(i.doc_id for i in invs)),
                      confidence=_doc_conf(p, *invs))
        if over <= 0:
            outcome = Outcome.MATCH if over == 0 else Outcome.EXPLAINED
            out.append(_r(ds, "cumulative_total", "money", outcome, p, "net", **common))
            continue
        tol = resolve_tolerance("cumulative_total", cfg)
        allow = tol.allowance(p.net, p.fx_to_gbp)
        outcome = Outcome.WITHIN_TOL if over <= allow else _fail(cfg, 1.0, p, *invs)
        out.append(_r(ds, "cumulative_total", "money", outcome, p, "net", exposure=over,
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 6. tax rate ----------------------------------------------------------------------

def check_tax_rate(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    rates = cfg["allowed_tax_rates"]
    out = []
    for inv in ds.invoices:
        if inv.net is None or inv.net == 0:
            continue
        if inv.tax is None:
            out.append(_r(ds, "tax_rate", "money", Outcome.UNVERIFIABLE, inv, "tax",
                          po_id=inv.po_ref, note="no tax amount", confidence=_doc_conf(inv)))
            continue
        implied = inv.tax / inv.net * 100
        nearest = min(rates, key=lambda r: abs(r - implied))
        expected = (inv.net * nearest / 100).quantize(Decimal("0.01"))
        diff = inv.tax - expected
        allow = cfg["rounding_per_line"] * max(1, len(inv.lines))
        outcome = (Outcome.MATCH if diff == 0 else
                   Outcome.WITHIN_TOL if abs(diff) <= allow else _fail(cfg, 1.0, inv))
        out.append(_r(ds, "tax_rate", "money", outcome, inv, "tax", po_id=inv.po_ref,
                      claim_value=f"{implied:.2f}%", auth_value=f"{nearest}%", delta=diff,
                      exposure=abs(diff), confidence=_doc_conf(inv),
                      tolerance={"allowed_rates": [str(r) for r in rates],
                                 "rounding": str(allow)}))
    return out


# --- 7-9, 13. header fields compared with the PO -----------------------------------------

def _against_po(ds: DocumentSet, links: Links):
    for inv in ds.invoices:
        p = links.invoice_po.get(inv.doc_id)
        if p is not None:
            yield inv, p


def check_currency(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        a, b = (inv.currency or "").strip().upper(), (p.currency or "").strip().upper()
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=a or None,
                      auth_value=b or None, confidence=_doc_conf(inv, p))
        if not a or not b:
            outcome, exposure = Outcome.UNVERIFIABLE, ZERO
        elif a != b:
            outcome, exposure = Outcome.CONFLICT, abs(inv.net or ZERO)
        else:
            outcome, exposure = Outcome.MATCH, ZERO
        out.append(_r(ds, "currency", "currency", outcome, inv, "currency",
                      exposure=exposure, **common))
    return out


def check_supplier(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        a, b = inv.supplier_id, p.supplier_id
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=a, auth_value=b,
                      confidence=_doc_conf(inv, p))
        if not a or not b:
            outcome, exposure = Outcome.UNVERIFIABLE, ZERO
        elif str(a) != str(b):
            outcome, exposure = Outcome.CONFLICT, abs(inv.net or ZERO)
        else:
            outcome, exposure = Outcome.MATCH, ZERO
        out.append(_r(ds, "supplier", "party", outcome, inv, "supplier_id",
                      exposure=exposure, **common))
    return out


def check_invoice_date(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        if inv.doc_date is None or p.doc_date is None:
            continue
        outcome = Outcome.CONFLICT if inv.doc_date < p.doc_date else Outcome.MATCH
        out.append(_r(ds, "invoice_date", "date", outcome, inv, "invoice_date",
                      auth_doc=p.doc_id, po_id=p.doc_id, claim_value=_s(inv.doc_date),
                      auth_value=_s(p.doc_date), confidence=_doc_conf(inv, p)))
    return out


def check_payment_terms(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        if not p.payment_terms:
            continue
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=inv.payment_terms,
                      auth_value=p.payment_terms, confidence=_doc_conf(inv, p))
        if not inv.payment_terms:
            outcome = Outcome.ABSENT_SUBORDINATE
        else:
            da, db = terms_days(inv.payment_terms), terms_days(p.payment_terms)
            if da is not None and db is not None:
                outcome = Outcome.MATCH if da == db else Outcome.CONFLICT
            elif norm_text(inv.payment_terms) == norm_text(p.payment_terms):
                outcome = Outcome.MATCH
            else:
                outcome = Outcome.UNVERIFIABLE
        out.append(_r(ds, "payment_terms", "terms", outcome, inv, "payment_terms", **common))
    return out


# --- 10. duplicates (read from the duplicate detector) -----------------------------------

def check_duplicates(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    invs = {i.doc_id: i for i in ds.invoices}
    out = []
    for flag in ds.duplicates:
        inv = invs.get(flag.invoice_id)
        if inv is None:
            continue
        p = links.invoice_po.get(inv.doc_id)
        exposure = abs(inv.net) if inv.net is not None else abs(flag.amount or ZERO)
        out.append(_r(ds, "duplicate", "money", Outcome.CONFLICT, inv, "invoice_id",
                      auth_doc=flag.earlier_invoice_id, po_id=p.doc_id if p else None,
                      claim_value=inv.doc_id, auth_value=flag.earlier_invoice_id,
                      exposure=exposure, confidence=_doc_conf(inv),
                      note="flagged by the duplicate-invoice detector"))
    return out


# --- 14-15. invoices that do not reach a PO -----------------------------------------------

def check_po_links(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv in ds.invoices:
        if inv.doc_id in links.bad_refs:
            out.append(_r(ds, "bad_po_ref", "reference", Outcome.ABSENT_AUTHORITATIVE, inv,
                          "po_id", claim_value=inv.po_ref, exposure=abs(inv.net or ZERO),
                          confidence=_doc_conf(inv)))
        elif inv.doc_id in links.no_ref:
            out.append(_r(ds, "no_po", "reference", Outcome.ABSENT_AUTHORITATIVE, inv,
                          "po_id", exposure=abs(inv.net or ZERO), confidence=_doc_conf(inv)))
    return out


LINE_CHECKS = (check_unit_price, check_quantity, check_line_arithmetic,
               check_description, check_unlinked_lines)
DOC_CHECKS = (check_invoice_totals, check_cumulative_total, check_tax_rate, check_currency,
              check_supplier, check_invoice_date, check_payment_terms, check_duplicates,
              check_po_links)


def run_checks(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    return [r for check in (*LINE_CHECKS, *DOC_CHECKS) for r in check(ds, links, cfg)]
```

- [ ] **Step 4: Run both check suites**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_checks_lines.py tests/triage/test_checks_documents.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/checks.py tests/triage/test_checks_documents.py"
git add $P && git commit -o $P -m "feat(triage): document-level checks and the full check list

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Scoring and overrides

**Files:**
- Create: `src/services/triage/score.py`
- Test: `tests/triage/test_score.py`

**Interfaces:**
- Produces: `ALWAYS_S1`, `MIN_S2`, `MAX_S3`, `threshold_gbp(r, cfg) -> Decimal`, `impact(exposure, threshold) -> float`, `score_result(r, cfg) -> Result` (sets `severity`, `score`, `score_inputs`), `apply_overrides(r) -> Result`.

- [ ] **Step 1: Write the failing test**

`tests/triage/test_score.py`:

```python
from decimal import Decimal as D

from src.services.triage.model import Outcome, Result, Severity
from src.services.triage.score import score_result, threshold_gbp
from tests.triage.helpers import make_cfg

CFG = make_cfg()


def res(rule="unit_price", cls="money", outcome=Outcome.CONFLICT, exposure="450",
        basis="9000", fx="1", conf=1.0):
    return Result(deal_id="D", rule_id=rule, field_class=cls, outcome=outcome,
                  claim_doc="INV-1", field_name="x", exposure=D(exposure),
                  basis_total=D(basis) if basis else None, fx_to_gbp=D(fx) if fx else None,
                  confidence=conf, currency="GBP")


def test_threshold_floor_and_ceiling():
    assert threshold_gbp(res(basis="9000"), CFG) == D("45")
    assert threshold_gbp(res(basis="1000"), CFG) == D("25")
    assert threshold_gbp(res(basis="10000000"), CFG) == D("5000")


def test_worked_examples_from_the_spec():
    assert score_result(res(exposure="450"), CFG).severity == Severity.S1
    s2 = score_result(res(exposure="45"), CFG)
    assert s2.severity == Severity.S2 and s2.score == 40.0
    s3 = score_result(res(exposure="10"), CFG)
    assert s3.severity == Severity.S3 and round(s3.score, 1) == 13.9


def test_matches_and_notes():
    assert score_result(res(outcome=Outcome.MATCH), CFG).severity == Severity.S0
    assert score_result(res(outcome=Outcome.WITHIN_TOL), CFG).severity == Severity.S0
    assert score_result(res(outcome=Outcome.EXPLAINED), CFG).severity == Severity.S3


def test_unverifiable_is_capped_at_s2():
    assert score_result(res(outcome=Outcome.UNVERIFIABLE, exposure="100000"), CFG).severity == Severity.S2


def test_unverifiable_money_is_at_least_s2():
    assert score_result(res(outcome=Outcome.UNVERIFIABLE, exposure="0"), CFG).severity == Severity.S2


def test_missing_fx_caps_at_s2():
    r = score_result(res(exposure="100000", fx=None), CFG)
    assert r.severity == Severity.S2 and r.score_inputs["fx_missing"] is True


def test_always_s1_rules_ignore_the_amount():
    for rule in ("currency", "cumulative_total", "duplicate"):
        assert score_result(res(rule=rule, exposure="0.01"), CFG).severity == Severity.S1


def test_override_beats_fx_cap():
    assert score_result(res(rule="currency", exposure="100000", fx=None), CFG).severity == Severity.S1


def test_min_s2_rules():
    assert score_result(res(rule="payment_terms", cls="terms", exposure="0"), CFG).severity == Severity.S2
    assert score_result(res(rule="invoice_date", cls="date", exposure="0"), CFG).severity == Severity.S2


def test_max_s3_rules():
    assert score_result(res(rule="description", cls="description", exposure="1000000"), CFG).severity == Severity.S3
    assert score_result(res(rule="no_po", cls="reference", outcome=Outcome.ABSENT_AUTHORITATIVE,
                            exposure="1000000"), CFG).severity == Severity.S3


def test_low_confidence_lowers_the_score():
    assert score_result(res(exposure="450", conf=0.5), CFG).score == 40.0
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_score.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 3: Implement**

`src/services/triage/score.py`:

```python
"""Materiality and severity (spec §7). Pure.

Overrides run LAST so nothing before them — and no rule added later — can soften
them. That ordering is the guard; tests break it on purpose (Task 15).
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

from .model import CRITICALITY, NOTE, SCORED, Outcome, Result, Severity

ALWAYS_S1 = frozenset({"cumulative_total", "currency", "duplicate"})
MIN_S2 = frozenset({"payment_terms", "invoice_date"})
MAX_S3 = frozenset({"description", "no_po"})
_MONEY_OR_PARTY = frozenset({"money", "party"})


def threshold_gbp(r: Result, cfg) -> Decimal:
    floor, ceiling = cfg["materiality_floor"], cfg["materiality_ceiling"]
    if r.basis_total is None or r.fx_to_gbp is None:
        return floor
    pct = abs(r.basis_total) * r.fx_to_gbp * cfg["materiality_pct_of_total"] / Decimal(100)
    return min(max(pct, floor), ceiling)


def impact(exposure: Optional[Decimal], threshold: Optional[Decimal]) -> float:
    if exposure is None or exposure <= 0 or threshold is None or threshold <= 0:
        return 0.0
    return max(0.0, min(100.0, 50.0 * (1.0 + math.log10(float(exposure / threshold)))))


def _band(score: float, cfg) -> Severity:
    if score >= cfg["band_s1"]:
        return Severity.S1
    if score >= cfg["band_s2"]:
        return Severity.S2
    return Severity.S3


def apply_overrides(r: Result) -> Result:
    if r.outcome not in SCORED:
        return r
    applied = []
    if r.rule_id in ALWAYS_S1 and r.outcome == Outcome.CONFLICT:
        r.severity = Severity.S1
        applied.append("always_s1")
    if r.rule_id in MIN_S2 and r.outcome == Outcome.CONFLICT:
        r.severity = max(r.severity, Severity.S2)
        applied.append("min_s2")
    if r.outcome == Outcome.UNVERIFIABLE and r.field_class in _MONEY_OR_PARTY:
        r.severity = max(r.severity, Severity.S2)
        applied.append("min_s2_unverifiable")
    if r.rule_id in MAX_S3:
        r.severity = min(r.severity, Severity.S3)
        applied.append("max_s3")
    if applied:
        r.score_inputs["overrides"] = applied
    return r


def score_result(r: Result, cfg) -> Result:
    if r.outcome in (Outcome.MATCH, Outcome.WITHIN_TOL):
        r.severity, r.score = Severity.S0, 0.0
        return r
    if r.outcome in NOTE:
        r.severity, r.score = Severity.S3, 0.0
        return r
    fx_missing = r.fx_to_gbp is None and r.exposure != 0
    if fx_missing:
        # Score in document currency against the percentage part of the threshold,
        # then cap: without a rate we cannot know the amount is large in GBP.
        thr_doc = (abs(r.basis_total) * cfg["materiality_pct_of_total"] / Decimal(100)
                   if r.basis_total else None)
        imp = impact(abs(r.exposure), thr_doc) if thr_doc else 50.0
        thr = None
    else:
        thr = threshold_gbp(r, cfg)
        imp = impact(r.exposure_gbp, thr)
    crit = CRITICALITY[r.field_class]
    score = max(0.0, min(100.0, imp * crit * r.confidence))
    sev = _band(score, cfg)
    if r.outcome == Outcome.UNVERIFIABLE or fx_missing:
        sev = min(sev, Severity.S2)
    r.score, r.severity = round(score, 2), sev
    r.score_inputs = {"exposure_gbp": None if r.exposure_gbp is None else str(r.exposure_gbp),
                      "threshold_gbp": None if thr is None else str(thr),
                      "impact": round(imp, 2), "criticality": crit,
                      "confidence": round(r.confidence, 4), "fx_missing": fx_missing,
                      "fx_rate_date": None if r.fx_rate_date is None else str(r.fx_rate_date)}
    return apply_overrides(r)
```

- [ ] **Step 4: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_score.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="src/services/triage/score.py tests/triage/test_score.py"
git add $P && git commit -o $P -m "feat(triage): materiality scoring, bands and overrides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Grouping into findings

**Files:**
- Create: `src/services/triage/group.py`
- Modify: `tests/triage/helpers.py` (add `scored(ds)`)
- Test: `tests/triage/test_group.py`

**Interfaces:**
- Consumes: `score.score_result`, `model.Finding/pct_change/SCORED/Severity`.
- Produces: `group(results, cfg) -> list[Finding]`; test helper `scored(ds, cfg=None) -> tuple[Links, list[Result]]`.

- [ ] **Step 1: Add the test helper**

Append to `tests/triage/helpers.py`:

```python
def scored(ds, cfg=None):
    """link -> checks -> score, as the engine does it."""
    from src.services.triage.checks import run_checks
    from src.services.triage.link import link
    from src.services.triage.score import score_result
    cfg = cfg or make_cfg()
    links = link(ds, cfg)
    results = run_checks(ds, links, cfg)
    for r in results:
        score_result(r, cfg)
    return links, results
```

- [ ] **Step 2: Write the failing test**

`tests/triage/test_group.py`:

```python
from datetime import date
from decimal import Decimal as D

from src.services.triage.group import group
from src.services.triage.model import DuplicateFlag, Severity
from tests.triage.helpers import deal, inv, line, make_cfg, po, scored

CFG = make_cfg()


def _findings(ds):
    _links, results = scored(ds, CFG)
    return group(results, CFG)


def test_clean_deal_has_no_findings():
    assert _findings(deal(po(), inv())) == []


def test_price_error_is_one_finding_with_the_po_total_as_its_effect():
    ds = deal(po(lines=[line(1, qty="300", price="12.00")]),
              inv(lines=[line(1, qty="300", price="13.50")]))
    (f,) = _findings(ds)
    assert f.rule_id == "unit_price"
    assert [e.rule_id for e in f.effects] == ["cumulative_total"]
    assert f.severity == Severity.S1 and f.exposure == D("450")


def test_flagged_duplicate_absorbs_quantity_and_overbilling():
    ds = deal(po(), inv("INV-1", inv_date=date(2026, 2, 1)), inv("INV-2", inv_date=date(2026, 2, 2)),
              duplicates=[DuplicateFlag("INV-2", "INV-1", D("144"))])
    (f,) = _findings(ds)
    assert f.rule_id == "duplicate"
    assert sorted(e.rule_id for e in f.effects) == ["cumulative_total", "quantity"]
    assert f.severity == Severity.S1


def test_unflagged_rebill_is_one_quantity_finding():
    ds = deal(po(lines=[line(1), line(2, item="ITEM-2")]),
              inv("INV-1", lines=[line(1), line(2, item="ITEM-2")], inv_date=date(2026, 2, 1)),
              inv("INV-2", lines=[line(1), line(2, item="ITEM-2")], inv_date=date(2026, 2, 2)))
    (f,) = _findings(ds)
    assert f.rule_id == "quantity" and len(f.causes) == 2
    assert [e.rule_id for e in f.effects] == ["cumulative_total"]
    assert f.severity == Severity.S1          # the always-S1 effect is never softened


def test_uniform_uplift_groups_lines_with_the_same_percentage():
    items = "ABCD"
    ds = deal(po(lines=[line(i, item=items[i], qty="1", price="100.00") for i in range(4)]),
              inv(lines=[line(i, item=items[i], qty="1", price="103.50") for i in range(4)]))
    (f,) = _findings(ds)
    assert f.rule_id == "uniform_uplift" and len(f.causes) == 4


def test_mixed_percentages_below_the_group_size_stay_separate():
    prices = ["103.50", "103.50", "110.00"]
    ds = deal(po(lines=[line(i, item=str(i), qty="1", price="100.00") for i in range(3)]),
              inv(lines=[line(i, item=str(i), qty="1", price=p) for i, p in enumerate(prices)]))
    fs = _findings(ds)
    assert sorted(f.rule_id for f in fs) == ["unit_price"] * 3


def test_unexplained_part_of_an_overage_is_its_own_finding():
    ds = deal(po(), inv(lines=[line(1, price="12.50")], net="200"))
    fs = {f.rule_id: f for f in _findings(ds)}
    assert set(fs) == {"unit_price", "cumulative_total", "invoice_totals"}
    assert fs["cumulative_total"].exposure == D("75")
```

- [ ] **Step 3: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_group.py -q -p no:randomly`
Expected: ERROR — module not found.

- [ ] **Step 4: Implement**

`src/services/triage/group.py`:

```python
"""One cause, one finding (spec §6). Pure.

Order matters: duplicates first (they can absorb quantity and over-billing), then
quantity groups and price groups, then the PO running total, which becomes an effect
of whatever explains it and is its own finding only for the part nothing explains.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .model import SCORED, Finding, Result, Severity, pct_change
from .score import score_result

_EXPLAINS_OVERAGE = ("duplicate", "quantity", "unit_price", "uniform_uplift")
_GROUPED = frozenset({"duplicate", "quantity", "unit_price", "cumulative_total"})


def _uplift_clusters(rs: list[Result], within: Decimal) -> list[list[Result]]:
    keyed = sorted(((pct_change(r), r) for r in rs if pct_change(r) is not None),
                   key=lambda t: t[0])
    clusters: list[list[Result]] = []
    current: list[Result] = []
    start = None
    for p, r in keyed:
        if current and p - start <= within:
            current.append(r)
        else:
            if current:
                clusters.append(current)
            current, start = [r], p
    if current:
        clusters.append(current)
    clusters.extend([r] for r in rs if pct_change(r) is None)
    return clusters


def group(results: list[Result], cfg) -> list[Finding]:
    live = [r for r in results if r.outcome in SCORED
            and r.severity is not None and r.severity > Severity.S0]
    findings: list[Finding] = []

    dup_by_invoice: dict[str, Finding] = {}
    for r in live:
        if r.rule_id == "duplicate":
            f = Finding(r.deal_id, "duplicate", [r], r.cause_key)
            findings.append(f)
            dup_by_invoice[r.claim_doc] = f

    by_invoice_po = defaultdict(list)
    for r in live:
        if r.rule_id == "quantity":
            by_invoice_po[(r.claim_doc, r.po_id)].append(r)
    for (inv_id, po_id), rs in by_invoice_po.items():
        if inv_id in dup_by_invoice:
            dup_by_invoice[inv_id].effects.extend(rs)
        else:
            findings.append(Finding(rs[0].deal_id, "quantity", rs, f"{inv_id}|{po_id}"))

    by_invoice = defaultdict(list)
    for r in live:
        if r.rule_id == "unit_price":
            by_invoice[r.claim_doc].append(r)
    for inv_id, rs in by_invoice.items():
        for cluster in _uplift_clusters(rs, cfg["uplift_same_pct_within"]):
            if len(cluster) >= cfg["uplift_min_lines"]:
                findings.append(Finding(cluster[0].deal_id, "uniform_uplift", cluster,
                                        f"{inv_id}|uplift"))
            else:
                findings.extend(Finding(r.deal_id, "unit_price", [r], r.cause_key)
                                for r in cluster)

    for r in live:
        if r.rule_id != "cumulative_total":
            continue
        related = [f for f in findings if f.rule_id in _EXPLAINS_OVERAGE
                   and any(c.po_id == r.po_id for c in f.causes)]
        explained = sum((f.exposure for f in related), Decimal("0"))
        allowance = Decimal(str(r.tolerance.get("allowance", "0")))
        if related and abs(r.exposure) <= explained + allowance:
            max(related, key=lambda f: (f.severity, f.exposure)).effects.append(r)
            continue
        if related:
            r.exposure = abs(r.exposure) - explained
            r.note = f"{r.note}; {explained} of the overage is explained by findings on its lines"
            score_result(r, cfg)
        findings.append(Finding(r.deal_id, "cumulative_total", [r], r.cause_key))

    findings.extend(Finding(r.deal_id, r.rule_id, [r], r.cause_key)
                    for r in live if r.rule_id not in _GROUPED)
    return findings
```

- [ ] **Step 5: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_group.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
P="src/services/triage/group.py tests/triage/helpers.py tests/triage/test_group.py"
git add $P && git commit -o $P -m "feat(triage): group results by cause into findings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Finding text and deal verdict

**Files:**
- Create: `src/services/triage/text.py`, `src/services/triage/verdict.py`
- Modify: `tests/triage/helpers.py` (add `pipeline(ds)`)
- Test: `tests/triage/test_text.py`, `tests/triage/test_verdict.py`

**Interfaces:**
- Produces: `describe(f) -> Finding` (sets `headline`, `text`), `verdict(ds, links, findings, results) -> Verdict`; test helper `pipeline(ds, cfg=None) -> SimpleNamespace(deal_id, results, findings, verdict)`.

- [ ] **Step 1: Add the helper**

Append to `tests/triage/helpers.py`:

```python
def pipeline(ds, cfg=None):
    """The whole pure pipeline for one deal, shaped like engine.DealOutput."""
    from types import SimpleNamespace
    from src.services.triage.group import group
    from src.services.triage.text import describe
    from src.services.triage.verdict import verdict
    cfg = cfg or make_cfg()
    links, results = scored(ds, cfg)
    findings = [describe(f) for f in group(results, cfg)]
    return SimpleNamespace(deal_id=ds.deal_id, results=results, findings=findings,
                           verdict=verdict(ds, links, findings, results))
```

- [ ] **Step 2: Write the failing tests**

`tests/triage/test_text.py`:

```python
from decimal import Decimal as D

from tests.triage.helpers import deal, inv, line, pipeline, po


def test_price_finding_text_follows_the_writing_rules():
    out = pipeline(deal(po(lines=[line(3, qty="300", price="12.00")]),
                        inv(lines=[line(3, qty="300", price="13.50")])))
    (f,) = out.findings
    assert f.headline == "Unit price above PO on line 3"
    assert "£450.00" in f.text
    assert "PO-1 12.00" in f.text and "INV-1 13.50" in f.text
    assert "Also changes: PO running total" in f.text


def test_non_gbp_exposure_shows_both_currencies():
    out = pipeline(deal(po(currency="EUR", fx="0.5", lines=[line(1, qty="300", price="12.00")]),
                        inv(currency="EUR", fx="0.5", lines=[line(1, qty="300", price="13.50")])))
    (f,) = out.findings
    assert "£225.00 (450.00 EUR)" in f.text


def test_text_without_fx_rate():
    out = pipeline(deal(po(currency="XXX", fx=None, lines=[line(1, qty="300", price="12.00")]),
                        inv(currency="XXX", fx=None, lines=[line(1, qty="300", price="13.50")])))
    assert any("(no FX rate)" in f.text for f in out.findings)


def test_uplift_headline():
    items = "ABCD"
    out = pipeline(deal(po(lines=[line(i, item=items[i], qty="1", price="100.00") for i in range(4)]),
                        inv(lines=[line(i, item=items[i], qty="1", price="103.50") for i in range(4)])))
    (f,) = out.findings
    assert f.headline == "Prices 3.5% above PO on 4 lines"
```

`tests/triage/test_verdict.py`:

```python
from datetime import date

from tests.triage.helpers import deal, inv, line, pipeline, po, quote


def test_clean_deal_is_matched():
    v = pipeline(deal(po(), inv())).verdict
    assert v.verdict == "Matched" and v.summary == "Matched"


def test_partial_invoicing_is_matched_with_notes():
    v = pipeline(deal(po(lines=[line(1, qty="20")]), inv(lines=[line(1, qty="10")]))).verdict
    assert v.verdict == "Matched with notes" and v.notes >= 1


def test_price_error_is_blocked_with_a_summary():
    v = pipeline(deal(po(lines=[line(1, qty="300", price="12.00")]),
                      inv(lines=[line(1, qty="300", price="13.50")]))).verdict
    assert v.verdict == "Blocked" and v.s1 == 1
    assert v.summary.startswith("Blocked · 1 finding needs action")
    assert "exposure £450.00" in v.summary


def test_payment_terms_difference_needs_review():
    v = pipeline(deal(po(), inv(terms="60 days"))).verdict
    assert v.verdict == "Needs review" and v.s2 == 1


def test_quote_only_deal_is_incomplete():
    v = pipeline(deal(quote())).verdict
    assert v.verdict == "Incomplete" and v.s1 == v.s2 == 0


def test_uninvoiced_po_is_incomplete():
    assert pipeline(deal(po())).verdict.verdict == "Incomplete"


def test_invoice_without_po_is_incomplete():
    assert pipeline(deal(po(), inv(), inv("INV-2", po_id=None))).verdict.verdict == "Incomplete"
```

- [ ] **Step 3: Run to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_text.py tests/triage/test_verdict.py -q -p no:randomly`
Expected: ERROR — modules not found.

- [ ] **Step 4: Implement**

`src/services/triage/text.py`:

```python
"""Plain-language finding text (spec §8.1, triage spec §10 "Writing findings").

Field and direction first; both documents' values with their ids; exposure in GBP and
the document currency where it differs; knock-on effects on one line.
"""
from __future__ import annotations

from decimal import Decimal

from .model import Finding, Outcome, money, pct_change

_EFFECT = {"cumulative_total": "PO running total", "quantity": "quantity invoiced",
           "unit_price": "unit price"}


def _avg_pct(f: Finding) -> Decimal:
    ps = [p for p in (pct_change(c) for c in f.causes) if p is not None]
    return sum(ps, Decimal("0")) / len(ps) if ps else Decimal("0")


def _headline(f: Finding) -> str:
    r = f.lead
    n = len(f.causes)
    up = r.delta is not None and r.delta > 0
    heads = {
        "unit_price": lambda: f"Unit price {'above' if up else 'below'} PO on line {r.claim_line}",
        "uniform_uplift": lambda: (f"Prices {abs(_avg_pct(f)):.1f}% "
                                   f"{'above' if up else 'below'} PO on {n} lines"),
        "quantity": lambda: f"Quantity above PO on {n} line{'s' if n != 1 else ''} of {r.claim_doc}",
        "cumulative_total": lambda: f"Invoices exceed PO {r.claim_doc} total",
        "duplicate": lambda: f"Possible duplicate of {r.auth_doc or 'an earlier invoice'}",
        "tax_rate": lambda: f"Tax rate {r.claim_value} is not an allowed rate",
        "currency": lambda: f"Invoice currency {r.claim_value} differs from PO ({r.auth_value})",
        "supplier": lambda: "Invoice supplier differs from PO",
        "invoice_date": lambda: "Invoice dated before the PO",
        "payment_terms": lambda: "Payment terms differ from PO",
        "description": lambda: f"Line {r.claim_line} description differs from PO",
        "unlinked_line": lambda: f"Invoice line with no PO line: “{r.claim_value}”",
        "bad_po_ref": lambda: f"Invoice names PO {r.claim_value}, which does not exist",
        "no_po": lambda: "Invoice has no PO",
        "line_arithmetic": lambda: f"Line {r.claim_line} amount is not quantity × price",
        "invoice_totals": lambda: f"Invoice {r.field_name} does not add up",
    }
    head = heads.get(f.rule_id, lambda: f.rule_id.replace("_", " "))()
    if any(c.outcome == Outcome.UNVERIFIABLE for c in f.causes):
        head = f"Could not verify: {head[0].lower()}{head[1:]}"
    return head


def describe(f: Finding) -> Finding:
    r = f.lead
    head = _headline(f)
    if f.exposure_gbp is not None:
        exposure = money(f.exposure_gbp, "GBP")
        if (r.currency or "GBP").upper() != "GBP":
            exposure += f" ({money(f.exposure, r.currency)})"
    else:
        exposure = f"{money(f.exposure, r.currency)} (no FX rate)"
    values = (f"{r.auth_doc or 'expected'} {r.auth_value if r.auth_value is not None else '-'}"
              f" · {r.claim_doc} {r.claim_value if r.claim_value is not None else '-'}")
    effects = ""
    if f.effects:
        effects = "Also changes: " + "; ".join(
            f"{_EFFECT.get(e.rule_id, e.rule_id.replace('_', ' '))} "
            f"({money(abs(e.exposure), e.currency)})" for e in f.effects)
    f.headline = head
    f.text = " · ".join(p for p in (head, exposure, values, r.note, effects) if p)
    return f
```

`src/services/triage/verdict.py`:

```python
"""The deal's verdict (spec §9.2). Pure.

Precedence: Blocked -> Needs review -> Incomplete -> Matched with notes -> Matched.
"""
from __future__ import annotations

from decimal import Decimal

from .model import NOTE, DocumentSet, Finding, Links, Result, Severity, Verdict


def verdict(ds: DocumentSet, links: Links, findings: list[Finding],
            results: list[Result]) -> Verdict:
    s1 = sum(1 for f in findings if f.severity == Severity.S1)
    s2 = sum(1 for f in findings if f.severity == Severity.S2)
    notes = (sum(1 for r in results if r.outcome in NOTE)
             + sum(1 for f in findings if f.severity == Severity.S3))
    invoiced = {p.doc_id for p in links.invoice_po.values() if p is not None}
    incomplete = (not ds.pos or not ds.invoices or bool(links.no_ref) or bool(links.bad_refs)
                  or any(p.doc_id not in invoiced for p in ds.pos))
    exposure = sum(((f.exposure_gbp or Decimal("0")) for f in findings
                    if f.severity >= Severity.S2), Decimal("0"))
    label = ("Blocked" if s1 else "Needs review" if s2 else "Incomplete" if incomplete
             else "Matched with notes" if notes else "Matched")
    return Verdict(ds.deal_id, label, s1, s2, notes, exposure, incomplete)
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/ -q -p no:randomly --ignore=tests/triage/test_loader_live.py`
Expected: PASS (every triage unit test so far).

- [ ] **Step 6: Commit**

```bash
P="src/services/triage/text.py src/services/triage/verdict.py tests/triage/helpers.py tests/triage/test_text.py tests/triage/test_verdict.py"
git add $P && git commit -o $P -m "feat(triage): finding text and deal verdict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Writer (runs, audit, findings, rollback)

**Files:**
- Create: `src/services/triage/writer.py`
- Test: `tests/triage/test_writer_live.py`

**Interfaces:**
- Consumes: outputs shaped `deal_id, results, findings` (engine.DealOutput or `helpers.pipeline`); `TriageConfig`; a report object with `.to_dict()`.
- Produces: `start_run(conn, mode, cfg) -> str`, `write_batch(conn, run_id, outputs) -> dict` (keys `inserted, updated, unchanged, reopened, superseded, audit_rows`), `finish_run(conn, run_id, report)`, `finding_ids(cur, fingerprints) -> dict[str, int]`, `rollback_run(conn, run_id) -> dict` (keys `findings_removed, findings_kept, audit_rows_removed`).

- [ ] **Step 1: Write the failing live test**

`tests/triage/test_writer_live.py`:

```python
"""Writer against the real database. Run with PROCWISE_TEST_LIVE_DB=1.

Uses synthetic deal ids (TRIAGE-TEST-*) and removes every row it made.
"""
import os
import uuid
from types import SimpleNamespace

import pytest

from src.services.db import get_conn
from src.services.triage import writer
from tests.triage.helpers import deal, inv, line, make_cfg, pipeline, po

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")
CFG = make_cfg()


@pytest.fixture
def ctx():
    deal_id = f"TRIAGE-TEST-{uuid.uuid4().hex[:8]}"
    runs = []
    with get_conn() as conn:
        yield SimpleNamespace(conn=conn, deal_id=deal_id, runs=runs)
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_detection_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_result WHERE deal_id = %s", (deal_id,))
        for run_id in runs:
            cur.execute("DELETE FROM proc.bp_triage_run WHERE run_id = %s", (run_id,))


def _write(ctx, ds):
    run_id = writer.start_run(ctx.conn, "single", CFG)
    ctx.runs.append(run_id)
    counts = writer.write_batch(ctx.conn, run_id, [pipeline(ds, CFG)])
    return run_id, counts


def _findings(ctx):
    cur = ctx.conn.cursor()
    cur.execute("""SELECT finding_id, rule_id, severity, status, lifecycle_status,
                          blocks_promotion, pipeline_record_id
                     FROM proc.bp_detection_finding WHERE deal_id = %s ORDER BY finding_id""",
                (ctx.deal_id,))
    return cur.fetchall()


def _currency_deal(ctx, currency="EUR"):
    return deal(po(), inv(currency=currency), deal_id=ctx.deal_id)


def test_first_write_inserts_s1_and_audits_everything(ctx):
    ds = _currency_deal(ctx)
    run_id, counts = _write(ctx, ds)
    rows = _findings(ctx)
    assert counts["inserted"] == 1 and len(rows) == 1
    _fid, rule, sev, status, life, blocks, record = rows[0]
    assert (rule, sev, status, life, blocks, record) == (
        "currency", "critical", "open", "open", True, ctx.deal_id)
    cur = ctx.conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_triage_result WHERE run_id = %s", (run_id,))
    assert cur.fetchone()[0] == counts["audit_rows"] == len(pipeline(ds, CFG).results)


def test_rerun_is_idempotent(ctx):
    _write(ctx, _currency_deal(ctx))
    first = _findings(ctx)
    _run_id, counts = _write(ctx, _currency_deal(ctx))
    assert counts["inserted"] == 0 and counts["updated"] == 1
    assert _findings(ctx) == first


def test_fixed_problem_is_superseded(ctx):
    _write(ctx, _currency_deal(ctx))
    _run_id, counts = _write(ctx, _currency_deal(ctx, currency="GBP"))
    assert counts["superseded"] == 1
    (_fid, _rule, _sev, status, life, _b, _r), = _findings(ctx)
    assert (status, life) == ("superseded", "resolved")


def test_resolved_finding_is_not_reopened(ctx):
    _write(ctx, _currency_deal(ctx))
    ctx.conn.cursor().execute(
        """UPDATE proc.bp_detection_finding SET status='resolved', lifecycle_status='resolved',
                  resolved_by='tester' WHERE deal_id = %s""", (ctx.deal_id,))
    _run_id, counts = _write(ctx, _currency_deal(ctx))
    rows = _findings(ctx)
    assert counts["unchanged"] == 1 and len(rows) == 1 and rows[0][3] == "resolved"


def test_severity_rise_opens_a_new_finding(ctx):
    def price_deal(price):
        return deal(po(lines=[line(1, qty="600", price="30.00")]),
                    inv(lines=[line(1, qty="300", price=price)]), deal_id=ctx.deal_id)

    _write(ctx, price_deal("30.50"))                   # exposure £150 -> S2
    (fid, rule, sev, *_rest), = _findings(ctx)
    assert (rule, sev) == ("unit_price", "warning")
    ctx.conn.cursor().execute(
        "UPDATE proc.bp_detection_finding SET status='ignored', lifecycle_status='accepted_risk' "
        "WHERE finding_id = %s", (fid,))
    _run_id, counts = _write(ctx, price_deal("32.00"))  # exposure £600 -> S1
    rows = _findings(ctx)
    assert counts["reopened"] == 1 and len(rows) == 2
    assert rows[0][3] == "ignored" and rows[1][2] == "critical" and rows[1][3] == "open"


def test_rollback_removes_only_untouched_findings(ctx):
    ds = deal(po(), inv("INV-1", currency="EUR"), inv("INV-2", currency="USD", po_id="PO-1"),
              deal_id=ctx.deal_id)
    run_id, _counts = _write(ctx, ds)
    fids = [r[0] for r in _findings(ctx) if r[1] == "currency"]
    ctx.conn.cursor().execute(
        "UPDATE proc.bp_detection_finding SET owner='buyer@example.com' WHERE finding_id = %s",
        (fids[0],))
    result = writer.rollback_run(ctx.conn, run_id)
    remaining = [r[0] for r in _findings(ctx)]
    assert fids[0] in remaining and fids[1] not in remaining
    assert result["findings_kept"] >= 1 and result["audit_rows_removed"] > 0
```

- [ ] **Step 2: Run to see it fail**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_writer_live.py -q -p no:randomly`
Expected: ERROR — `cannot import name 'writer'`.

- [ ] **Step 3: Implement**

`src/services/triage/writer.py`:

```python
"""Persistence for triage (spec §8). The only module that writes.

One real transaction per batch: get_conn() hands out AUTOCOMMIT connections, where
rollback() is a no-op, so without switching autocommit off a crash mid-batch would
leave half a batch of findings behind.

A bp_detection_finding row is only ever moved from status 'open', and status and
lifecycle_status move together — the gateway's stage gate reads lifecycle_status.
"""
from __future__ import annotations

import json
import uuid
from typing import Iterable

from psycopg2.extras import execute_values

from .model import ACTION_CENTRE_SEVERITY, Finding, Result, Severity, money

_EXISTING = """
SELECT m.fingerprint, m.finding_id, m.last_severity, f.status
  FROM proc.bp_triage_finding m
  JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
 WHERE m.deal_id = ANY(%s)
"""
_INSERT_FINDING = """
INSERT INTO proc.bp_detection_finding
    (engine_run_id, rule_id, category, severity, doc_type, doc_pk, deal_id,
     pipeline_record_id, field_name, observed_value, expected_value, delta,
     blocks_promotion, confidence, notes, status, lifecycle_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', 'open')
RETURNING finding_id
"""
_UPDATE_FINDING = """
UPDATE proc.bp_detection_finding
   SET engine_run_id = %s, severity = %s, observed_value = %s, expected_value = %s,
       delta = %s, blocks_promotion = %s, confidence = %s, notes = %s
 WHERE finding_id = %s AND status = 'open'
"""
_SUPERSEDE = """
UPDATE proc.bp_detection_finding
   SET status = 'superseded', lifecycle_status = 'resolved'
 WHERE finding_id = %s AND status = 'open'
"""
_UPSERT_MAP = """
INSERT INTO proc.bp_triage_finding
    (fingerprint, finding_id, deal_id, first_run_id, last_run_id, last_severity)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (fingerprint) DO UPDATE
   SET finding_id = EXCLUDED.finding_id, first_run_id = EXCLUDED.first_run_id,
       last_run_id = EXCLUDED.last_run_id, last_severity = EXCLUDED.last_severity
"""
_TOUCH_MAP = """
UPDATE proc.bp_triage_finding SET last_run_id = %s, last_severity = %s WHERE fingerprint = %s
"""
_TOUCH_MAP_RUN_ONLY = "UPDATE proc.bp_triage_finding SET last_run_id = %s WHERE fingerprint = %s"
_AUDIT = """
INSERT INTO proc.bp_triage_result
    (run_id, deal_id, rule_id, claim_doc, claim_line, auth_doc, auth_line, field_name,
     claim_value, auth_value, outcome, severity, exposure_gbp, score, score_inputs,
     tolerance, fingerprint, finding_id)
VALUES %s
"""
_ROLLBACK_FINDINGS = """
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id AND m.first_run_id = %s
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL
RETURNING f.finding_id
"""


def start_run(conn, mode: str, cfg) -> str:
    run_id = str(uuid.uuid4())
    conn.cursor().execute(
        """INSERT INTO proc.bp_triage_run (run_id, mode, config_fingerprint, config_values)
           VALUES (%s, %s, %s, %s)""",
        (run_id, mode, cfg.fingerprint, json.dumps(dict(cfg.values), default=str)))
    return run_id


def finish_run(conn, run_id: str, report) -> None:
    data = report.to_dict()
    conn.cursor().execute(
        """UPDATE proc.bp_triage_run
              SET finished_at = now(), deal_count = %s, failed_deals = %s, report = %s
            WHERE run_id = %s""",
        (data["deals_done"], json.dumps(data["failed"]), json.dumps(data, default=str), run_id))


def finding_ids(cur, fingerprints: Iterable[str]) -> dict[str, int]:
    fps = list(fingerprints)
    if not fps:
        return {}
    cur.execute("SELECT fingerprint, finding_id FROM proc.bp_triage_finding "
                "WHERE fingerprint = ANY(%s)", (fps,))
    return {fp: fid for fp, fid in cur.fetchall()}


def _finding_values(run_id: str, f: Finding) -> tuple:
    r = f.lead
    if f.exposure_gbp is not None:
        delta = money(f.exposure_gbp, "GBP")
        if (r.currency or "GBP").upper() != "GBP":
            delta += f" ({money(f.exposure, r.currency)})"
    else:
        delta = f"{money(f.exposure, r.currency)} (no FX rate)"
    return (run_id, f.rule_id, f.category, ACTION_CENTRE_SEVERITY[f.severity],
            "purchase_order" if f.rule_id == "cumulative_total" else "invoice",
            r.claim_doc, f.deal_id, f.deal_id, r.field_name,
            f"{r.claim_doc}: {r.claim_value}", f"{r.auth_doc}: {r.auth_value}", delta,
            f.severity == Severity.S1, round(f.confidence, 4), f.text)


def _insert(cur, run_id: str, f: Finding) -> int:
    cur.execute(_INSERT_FINDING, _finding_values(run_id, f))
    return cur.fetchone()[0]


def _audit_row(run_id: str, r: Result, fid) -> tuple:
    return (run_id, r.deal_id, r.rule_id, r.claim_doc, r.claim_line, r.auth_doc, r.auth_line,
            r.field_name, r.claim_value, r.auth_value, r.outcome.value,
            (r.severity or Severity.S0).name, r.exposure_gbp, r.score,
            json.dumps(r.score_inputs, default=str), json.dumps(r.tolerance, default=str),
            r.fingerprint, fid)


def write_batch(conn, run_id: str, outputs) -> dict:
    counts = dict(inserted=0, updated=0, unchanged=0, reopened=0, superseded=0, audit_rows=0)
    outputs = list(outputs)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(_EXISTING, ([o.deal_id for o in outputs],))
        existing = {fp: (fid, sev, status) for fp, fid, sev, status in cur.fetchall()}
        seen: set[str] = set()
        fid_of: dict[int, int] = {}
        for o in outputs:
            for f in o.findings:
                if f.severity < Severity.S2:
                    continue
                fp = f.fingerprint
                seen.add(fp)
                prior = existing.get(fp)
                if prior is None:
                    fid = _insert(cur, run_id, f)
                    cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id, f.severity.name))
                    counts["inserted"] += 1
                else:
                    old_fid, old_sev, status = prior
                    if status == "open":
                        v = _finding_values(run_id, f)
                        cur.execute(_UPDATE_FINDING, (v[0], v[3], v[9], v[10], v[11], v[12],
                                                      v[13], v[14], old_fid))
                        cur.execute(_TOUCH_MAP, (run_id, f.severity.name, fp))
                        fid = old_fid
                        counts["updated"] += 1
                    elif f.severity > Severity[old_sev]:
                        f.text = (f"Reopened: finding {old_fid} was {status}; this is now "
                                  f"{f.severity.name}. {f.text}")
                        fid = _insert(cur, run_id, f)
                        cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id,
                                                  f.severity.name))
                        counts["reopened"] += 1
                    else:
                        fid = old_fid
                        cur.execute(_TOUCH_MAP_RUN_ONLY, (run_id, fp))
                        counts["unchanged"] += 1
                for r in (*f.causes, *f.effects):
                    fid_of[id(r)] = fid
        for fp, (fid, _sev, status) in existing.items():
            if fp not in seen and status == "open":
                cur.execute(_SUPERSEDE, (fid,))
                counts["superseded"] += cur.rowcount
        rows = [_audit_row(run_id, r, fid_of.get(id(r))) for o in outputs for r in o.results]
        if rows:
            execute_values(cur, _AUDIT, rows, page_size=1000)
        counts["audit_rows"] = len(rows)
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def rollback_run(conn, run_id: str) -> dict:
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(_ROLLBACK_FINDINGS, (run_id,))
        removed = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_triage_finding WHERE finding_id = ANY(%s)", (removed,))
        cur.execute("SELECT count(*) FROM proc.bp_triage_finding WHERE first_run_id = %s",
                    (run_id,))
        kept = cur.fetchone()[0]
        cur.execute("DELETE FROM proc.bp_triage_result WHERE run_id = %s", (run_id,))
        audit = cur.rowcount
        cur.execute("UPDATE proc.bp_triage_run SET rolled_back_at = now() WHERE run_id = %s",
                    (run_id,))
        conn.commit()
        return {"findings_removed": len(removed), "findings_kept": kept,
                "audit_rows_removed": audit}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
```

- [ ] **Step 4: Run the live test**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_writer_live.py -q -p no:randomly`
Expected: 6 passed (not skipped).

- [ ] **Step 5: Confirm no test rows were left behind**

Run: `set -a; . ./.env; set +a; PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -Atc "SELECT count(*) FROM proc.bp_detection_finding WHERE deal_id LIKE 'TRIAGE-TEST-%'"`
Expected: `0`.

- [ ] **Step 6: Commit**

```bash
P="src/services/triage/writer.py tests/triage/test_writer_live.py"
git add $P && git commit -o $P -m "feat(triage): writer -- audit rows, finding upsert/supersede/reopen, rollback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Engine and run report

**Files:**
- Create: `src/services/triage/report.py`, `src/services/triage/engine.py`
- Test: `tests/triage/test_engine.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `report.RunReport` (`add(out)`, `fail(deal_id, why)`, `finish()`, `noise_ratio`, `deals_per_second`, `to_dict()`, `render()`, fields `run_id`, `deals_done`, `deals_requested`, `failed`, `write_counts`, `known_gaps`); `engine.DealOutput`, `engine.triage_set(ds, cfg) -> DealOutput`, `engine.run_triage(deal_ids, mode, *, dry_run=False, cfg=None, connect=get_conn, on_batch=None, known_gaps=()) -> RunReport`, `engine.view_dict(out, finding_ids=None) -> dict`, `engine.triage_deal_view(deal_id, *, cfg=None, connect=get_conn) -> Optional[dict]`, `engine.run_changed(*, cfg=None, connect=get_conn) -> Optional[RunReport]`.

- [ ] **Step 1: Write the failing test**

`tests/triage/test_engine.py`:

```python
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.services.governed_limits import LimitUnavailable
from src.services.triage import engine
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def test_triage_set_on_a_clean_deal():
    out = engine.triage_set(deal(po(), inv()), CFG)
    assert out.findings == [] and out.verdict.verdict == "Matched"


def test_triage_set_on_a_quote_only_deal_does_not_crash():
    assert engine.triage_set(deal(quote()), CFG).verdict.verdict == "Incomplete"


@pytest.fixture
def fakes(monkeypatch):
    sets = {
        "D-OK": deal(po(), inv(), deal_id="D-OK"),
        "D-BAD": deal(po(), inv(currency="EUR"), deal_id="D-BAD"),
        "D-BOOM": deal(po(), inv(), deal_id="D-BOOM"),
    }
    calls = []

    @contextmanager
    def connect():
        yield SimpleNamespace(cursor=lambda: None)

    monkeypatch.setattr(engine.loader, "load_deal_sets",
                        lambda cur, ids: {d: sets[d] for d in ids if d in sets})
    monkeypatch.setattr(engine.writer, "start_run", lambda conn, mode, cfg: calls.append("start") or "RUN-1")
    monkeypatch.setattr(engine.writer, "write_batch",
                        lambda conn, run_id, outs: calls.append([o.deal_id for o in outs]) or {"inserted": 1})
    monkeypatch.setattr(engine.writer, "finish_run", lambda conn, run_id, report: calls.append("finish"))
    real = engine.triage_set

    def maybe_boom(ds, cfg):
        if ds.deal_id == "D-BOOM":
            raise ValueError("boom")
        return real(ds, cfg)

    monkeypatch.setattr(engine, "triage_set", maybe_boom)
    return SimpleNamespace(connect=connect, calls=calls)


def test_one_failing_deal_does_not_stop_the_run(fakes):
    report = engine.run_triage(["D-OK", "D-BOOM", "D-BAD", "D-NONE"], "backfill",
                               cfg=CFG, connect=fakes.connect)
    assert report.deals_done == 2 and list(report.failed) == ["D-BOOM"]
    assert report.deals_without_documents == 1
    assert fakes.calls == ["start", ["D-OK", "D-BAD"], "finish"]
    assert report.verdicts["Blocked"] == 1 and report.shown == 1
    assert 0 < report.noise_ratio <= 1


def test_dry_run_writes_nothing(fakes):
    report = engine.run_triage(["D-OK", "D-BAD"], "backfill", dry_run=True, cfg=CFG,
                               connect=fakes.connect)
    assert fakes.calls == [] and report.deals_done == 2


def test_missing_policy_aborts_before_any_write(fakes, monkeypatch):
    def refuse():
        raise LimitUnavailable("no triage_tolerances")
    monkeypatch.setattr(engine, "load_config", refuse)
    with pytest.raises(LimitUnavailable):
        engine.run_triage(["D-OK"], "backfill", connect=fakes.connect)
    assert fakes.calls == []


def test_report_renders_the_scale_measures(fakes):
    report = engine.run_triage(["D-OK", "D-BAD"], "backfill", cfg=CFG, connect=fakes.connect,
                               known_gaps=["Invoices with no deal_id are not triaged: 7"])
    text = report.render()
    for needle in ("Noise ratio", "deals/s", "Blocked", "Payee bank details",
                   "Invoices with no deal_id are not triaged: 7"):
        assert needle in text
    assert report.to_dict()["verdicts"]["Blocked"] == 1


def test_view_dict_lists_only_s1_and_s2():
    out = engine.triage_set(deal(po(), inv(currency="EUR", terms="60 days")), CFG)
    view = engine.view_dict(out, {})
    assert view["verdict"] == "Blocked"
    assert [f["severity"] for f in view["findings"]] == ["S1", "S2"]
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_engine.py -q -p no:randomly`
Expected: ERROR — `cannot import name 'engine'`.

- [ ] **Step 3: Implement the report**

`src/services/triage/report.py`:

```python
"""The run report: the triage spec's §15 measures for one run (spec §1, criterion 4)."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .model import Outcome, Severity, money

BANK_GAP = "Payee bank details are not checked: invoices carry no bank data."


@dataclass
class RunReport:
    mode: str
    dry_run: bool = False
    deals_requested: int = 0
    run_id: Optional[str] = None
    deals_done: int = 0
    deals_without_documents: int = 0
    failed: dict = field(default_factory=dict)
    raw_differences: int = 0
    shown: int = 0
    outcomes: Counter = field(default_factory=Counter)
    verdicts: Counter = field(default_factory=Counter)
    findings_by_rule: Counter = field(default_factory=Counter)
    findings_by_severity: Counter = field(default_factory=Counter)
    exposure_by_severity: dict = field(default_factory=dict)
    write_counts: Counter = field(default_factory=Counter)
    known_gaps: list = field(default_factory=lambda: [BANK_GAP])
    elapsed_s: float = 0.0
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def fail(self, deal_id: str, why: str) -> None:
        self.failed[deal_id] = why

    def add(self, out) -> None:
        self.deals_done += 1
        for r in out.results:
            self.outcomes[r.outcome.value] += 1
            if r.outcome != Outcome.MATCH:
                self.raw_differences += 1
        for f in out.findings:
            sev = f.severity.name
            self.findings_by_rule[f.rule_id] += 1
            self.findings_by_severity[sev] += 1
            self.exposure_by_severity[sev] = (self.exposure_by_severity.get(sev, Decimal("0"))
                                              + (f.exposure_gbp or Decimal("0")))
            if f.severity >= Severity.S2:
                self.shown += 1
        self.verdicts[out.verdict.verdict] += 1

    def finish(self) -> None:
        self.elapsed_s = round(time.monotonic() - self._t0, 2)

    @property
    def noise_ratio(self) -> float:
        return self.shown / self.raw_differences if self.raw_differences else 0.0

    @property
    def deals_per_second(self) -> float:
        return self.deals_done / self.elapsed_s if self.elapsed_s else 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode, "dry_run": self.dry_run, "run_id": self.run_id,
            "deals_requested": self.deals_requested, "deals_done": self.deals_done,
            "deals_without_documents": self.deals_without_documents,
            "failed": dict(self.failed), "elapsed_s": self.elapsed_s,
            "deals_per_second": round(self.deals_per_second, 2),
            "raw_differences": self.raw_differences, "shown": self.shown,
            "noise_ratio": round(self.noise_ratio, 4),
            "outcomes": dict(self.outcomes), "verdicts": dict(self.verdicts),
            "findings_by_rule": dict(self.findings_by_rule),
            "findings_by_severity": dict(self.findings_by_severity),
            "exposure_by_severity": {k: str(v) for k, v in self.exposure_by_severity.items()},
            "write_counts": dict(self.write_counts), "known_gaps": list(self.known_gaps),
        }

    def render(self) -> str:
        def counts(c):
            return " · ".join(f"{k} {v:,}" for k, v in c.most_common()) or "none"

        lines = [
            f"Discrepancy triage — {self.mode}{' (dry run)' if self.dry_run else ''} "
            f"· run {self.run_id or '-'}",
            f"Deals: {self.deals_done:,} checked of {self.deals_requested:,} · "
            f"{self.deals_without_documents:,} without documents · {len(self.failed):,} failed",
            f"Time: {self.elapsed_s:,.1f}s · {self.deals_per_second:,.1f} deals/s",
            f"Noise ratio: {self.shown:,} findings shown / {self.raw_differences:,} raw "
            f"differences = {self.noise_ratio:.2%}",
            f"Verdicts: {counts(self.verdicts)}",
            "Exposure by severity: " + (" · ".join(
                f"{s} {money(self.exposure_by_severity.get(s), 'GBP')}"
                for s in ("S1", "S2", "S3") if s in self.exposure_by_severity) or "none"),
            f"Findings by rule: {counts(self.findings_by_rule)}",
            f"Outcomes: {counts(self.outcomes)}",
            f"Writes: {counts(self.write_counts)}",
            "Known gaps:",
            *[f"  - {g}" for g in self.known_gaps],
        ]
        if self.failed:
            lines.append("Failed deals (first 20):")
            lines += [f"  - {d}: {why}" for d, why in list(self.failed.items())[:20]]
        return "\n".join(lines)
```

- [ ] **Step 4: Implement the engine**

`src/services/triage/engine.py`:

```python
"""Discrepancy triage engine.

Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md

triage_set is the whole pure pipeline for one deal. run_triage batches it over many
deals: loads each batch in a fixed number of queries, isolates a failing deal, and
writes each batch in one transaction. load_config() runs first, so a missing
tolerance aborts the run before anything is written.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Optional

from src.services.db import get_conn

from . import loader, writer
from .checks import run_checks
from .group import group
from .link import link
from .model import DocumentSet, Finding, Result, Severity, Verdict
from .report import RunReport
from .score import score_result
from .text import describe
from .tolerance import TriageConfig, load_config
from .verdict import verdict

log = logging.getLogger(__name__)

_BASELINE_SQL = """
SELECT max(started_at) FROM proc.bp_triage_run
 WHERE mode IN ('backfill', 'scheduled') AND finished_at IS NOT NULL AND rolled_back_at IS NULL
"""
_CHANGED_SQL = """
SELECT DISTINCT deal_id FROM (
    SELECT deal_id, greatest(created_date, last_modified_date) AS ts FROM proc.bp_invoice_trgt
    UNION ALL
    SELECT deal_id, greatest(created_date, last_modified_date) FROM proc.bp_purchase_order_trgt
    UNION ALL
    SELECT deal_id, greatest(created_date, last_modified_date) FROM proc.bp_quote_trgt
) d
 WHERE deal_id IS NOT NULL AND ts > (%s::timestamptz AT TIME ZONE 'UTC')
"""


@dataclass
class DealOutput:
    deal_id: str
    results: list[Result]
    findings: list[Finding]
    verdict: Verdict


def triage_set(ds: DocumentSet, cfg: TriageConfig) -> DealOutput:
    links = link(ds, cfg)
    results = run_checks(ds, links, cfg)
    for r in results:
        score_result(r, cfg)
    findings = [describe(f) for f in group(results, cfg)]
    return DealOutput(ds.deal_id, results, findings, verdict(ds, links, findings, results))


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_triage(deal_ids: Iterable[str], mode: str, *, dry_run: bool = False,
               cfg: Optional[TriageConfig] = None, connect: Callable = get_conn,
               on_batch: Optional[Callable[[RunReport], None]] = None,
               known_gaps: Iterable[str] = ()) -> RunReport:
    cfg = cfg or load_config()
    ids = list(dict.fromkeys(deal_ids))
    report = RunReport(mode=mode, dry_run=dry_run, deals_requested=len(ids))
    report.known_gaps.extend(known_gaps)
    with connect() as conn:
        if not dry_run:
            report.run_id = writer.start_run(conn, mode, cfg)
        for batch in _chunks(ids, int(cfg["batch_size"])):
            try:
                sets = loader.load_deal_sets(conn.cursor(), batch)
            except Exception as exc:  # noqa: BLE001 - a batch that cannot load is reported
                log.exception("triage: loading batch failed")
                for d in batch:
                    report.fail(d, f"load failed: {exc}")
                continue
            outputs = []
            for d in batch:
                ds = sets.get(d)
                if ds is None:
                    report.deals_without_documents += 1
                    continue
                try:
                    outputs.append(triage_set(ds, cfg))
                except Exception as exc:  # noqa: BLE001 - one bad deal must not stop the run
                    log.exception("triage: deal %s failed", d)
                    report.fail(d, repr(exc))
            if not dry_run and outputs:
                try:
                    report.write_counts.update(writer.write_batch(conn, report.run_id, outputs))
                except Exception as exc:  # noqa: BLE001 - the batch rolled back; say so
                    log.exception("triage: writing batch failed")
                    for o in outputs:
                        report.fail(o.deal_id, f"write failed: {exc}")
                    continue
            for o in outputs:
                report.add(o)
            if on_batch:
                on_batch(report)
        report.finish()
        if not dry_run:
            writer.finish_run(conn, report.run_id, report)
    return report


def view_dict(out: DealOutput, finding_ids: Optional[dict] = None) -> dict:
    ids = finding_ids or {}
    v = out.verdict
    shown = sorted((f for f in out.findings if f.severity >= Severity.S2),
                   key=lambda f: (-int(f.severity), -(f.exposure_gbp or Decimal("0"))))
    return {
        "deal_id": v.deal_id, "verdict": v.verdict, "summary": v.summary,
        "counts": {"s1": v.s1, "s2": v.s2, "notes": v.notes},
        "exposure_gbp": str(v.exposure_gbp),
        "findings": [{"rule_id": f.rule_id, "category": f.category,
                      "severity": f.severity.name, "headline": f.headline, "text": f.text,
                      "exposure_gbp": None if f.exposure_gbp is None else str(f.exposure_gbp),
                      "finding_id": ids.get(f.fingerprint)} for f in shown],
    }


def triage_deal_view(deal_id: str, *, cfg: Optional[TriageConfig] = None,
                     connect: Callable = get_conn) -> Optional[dict]:
    """The deal's verdict now (a dry run), with Action Centre ids where they exist."""
    cfg = cfg or load_config()
    with connect() as conn:
        cur = conn.cursor()
        ds = loader.load_deal_sets(cur, [deal_id]).get(deal_id)
        if ds is None:
            return None
        out = triage_set(ds, cfg)
        ids = writer.finding_ids(cur, [f.fingerprint for f in out.findings
                                       if f.severity >= Severity.S2])
    return view_dict(out, ids)


def run_changed(*, cfg: Optional[TriageConfig] = None,
                connect: Callable = get_conn) -> Optional[RunReport]:
    """Re-triage deals whose final documents changed since the last completed run.

    Does nothing until a backfill has completed: the first pass over the whole corpus
    is a deliberate, reported act, not a side effect of the scheduler starting.
    """
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(_BASELINE_SQL)
        baseline = cur.fetchone()[0]
        if baseline is None:
            log.info("triage: no completed backfill yet; nothing scheduled")
            return None
        cur.execute(_CHANGED_SQL, (baseline,))
        ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return None
    return run_triage(ids, "scheduled", cfg=cfg, connect=connect)
```

- [ ] **Step 5: Run the engine test and every triage unit test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/ -q -p no:randomly`
Expected: PASS; the two `*_live.py` files report as skipped (no flag) — that is expected here only.

- [ ] **Step 6: Commit**

```bash
P="src/services/triage/report.py src/services/triage/engine.py tests/triage/test_engine.py"
git add $P && git commit -o $P -m "feat(triage): engine and run report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: API routes and scheduler job

**Files:**
- Create: `src/api/routers/triage.py`
- Modify: `src/api/main.py` (import + register), `src/services/backend_scheduler.py` (job)
- Test: `tests/triage/test_router.py`, `tests/triage/test_scheduler_job.py`

**Interfaces:**
- Consumes: `engine.triage_deal_view`, `engine.run_triage`, `engine.run_changed`, `api.auth.require_user`.
- Produces: `GET /triage/deals/{deal_id}` → view dict (404 unknown deal, 503 policy missing); `POST /triage/deals/{deal_id}/run` → `{"run_id", "writes", ...view}`; scheduler job `discrepancy-triage`.

- [ ] **Step 1: Write the failing tests**

`tests/triage/test_router.py`:

```python
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import require_user
from api.routers import triage as triage_router
from src.services.governed_limits import LimitUnavailable

VIEW = {"deal_id": "D1", "verdict": "Blocked", "summary": "Blocked · 1 finding needs action",
        "counts": {"s1": 1, "s2": 0, "notes": 0}, "exposure_gbp": "450.00", "findings": []}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(triage_router.router)
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


def test_get_returns_the_view(client, monkeypatch):
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: VIEW)
    r = client.get("/triage/deals/D1")
    assert r.status_code == 200 and r.json()["verdict"] == "Blocked"


def test_get_unknown_deal_is_404(client, monkeypatch):
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: None)
    assert client.get("/triage/deals/NOPE").status_code == 404


def test_missing_policy_is_503(client, monkeypatch):
    def refuse(d):
        raise LimitUnavailable("no triage_tolerances")
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", refuse)
    assert client.get("/triage/deals/D1").status_code == 503


def test_post_runs_and_returns_writes(client, monkeypatch):
    report = SimpleNamespace(run_id="RUN-1", failed={}, deals_done=1,
                             write_counts={"inserted": 1})
    monkeypatch.setattr(triage_router.engine, "run_triage", lambda ids, mode: report)
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: VIEW)
    r = client.post("/triage/deals/D1/run")
    assert r.status_code == 200
    assert r.json()["run_id"] == "RUN-1" and r.json()["writes"] == {"inserted": 1}
```

`tests/triage/test_scheduler_job.py`:

```python
from datetime import timedelta


def test_triage_job_registers_once():
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    calls = []

    def register_job(name, runner, interval, initial_delay=None, **kw):
        calls.append((name, interval))
        s._jobs[name] = runner

    s.register_job = register_job
    s._register_triage_job()
    s._register_triage_job()
    assert calls == [("discrepancy-triage", timedelta(minutes=15))]


def test_triage_job_never_raises(monkeypatch):
    from src.services import backend_scheduler
    from src.services.triage import engine

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(engine, "run_changed", boom)
    s = backend_scheduler.BackendScheduler.__new__(backend_scheduler.BackendScheduler)
    s._run_triage_job()   # logs, does not raise
```

- [ ] **Step 2: Run to see them fail**

Run: `set -a; . ./.env; set +a; CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_router.py tests/triage/test_scheduler_job.py -q -p no:randomly`
Expected: ImportError for `api.routers.triage`; AttributeError `_register_triage_job`.

- [ ] **Step 3: Implement the router**

`src/api/routers/triage.py`:

```python
"""Discrepancy triage for one deal.

GET  /triage/deals/{deal_id}      the deal's verdict now, with its S1/S2 findings
POST /triage/deals/{deal_id}/run  re-check the deal and update the Action Centre

Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md §9.1
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_user
from src.services.governed_limits import LimitUnavailable
from src.services.triage import engine

router = APIRouter(prefix="/triage", tags=["Triage"])


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"triage tolerances unavailable: {exc}")


@router.get("/deals/{deal_id}", summary="A deal's triage verdict and the findings that need action")
def get_deal_triage(deal_id: str) -> dict[str, Any]:
    try:
        view = engine.triage_deal_view(deal_id)
    except LimitUnavailable as exc:
        raise _unavailable(exc) from exc
    if view is None:
        raise HTTPException(status_code=404, detail=f"no documents for deal {deal_id}")
    return view


@router.post("/deals/{deal_id}/run", summary="Re-triage one deal and update the Action Centre")
def run_deal_triage(deal_id: str, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        report = engine.run_triage([deal_id], "single")
    except LimitUnavailable as exc:
        raise _unavailable(exc) from exc
    if report.failed:
        raise HTTPException(status_code=500, detail=report.failed.get(deal_id, "triage failed"))
    if report.deals_done == 0:
        raise HTTPException(status_code=404, detail=f"no documents for deal {deal_id}")
    return {"run_id": report.run_id, "writes": dict(report.write_counts),
            **(engine.triage_deal_view(deal_id) or {})}
```

- [ ] **Step 4: Register the router**

In `src/api/main.py`, directly after the line `from api.routers import reports as reports_router` add:

```python
from api.routers import triage as triage_router
```

and in the `_AUTHENTICATED_ROUTERS` list, directly after `    reports_router.router,` add:

```python
    triage_router.router,
```

- [ ] **Step 5: Add the scheduler job**

In `src/services/backend_scheduler.py`, inside `_register_default_jobs`, after `self._register_style_feedback_job()` add `self._register_triage_job()`. Then add these members to `BackendScheduler` directly after `_register_default_jobs`:

```python
    TRIAGE_JOB_NAME = "discrepancy-triage"

    def _register_triage_job(self) -> None:
        """Re-triage deals whose final documents changed since the last triage run.

        Does nothing until a backfill has completed (spec §9.1): the first pass over the
        whole corpus is a deliberate, reported act, not a side effect of startup.

        Interval via TRIAGE_INTERVAL_MINUTES (default 15).
        """
        import os
        if self.TRIAGE_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("TRIAGE_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.TRIAGE_JOB_NAME,
            self._run_triage_job,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _run_triage_job(self) -> None:
        try:
            from src.services.triage.engine import run_changed

            report = run_changed()
            if report is not None:
                logger.info("Discrepancy triage: %s", report.render().splitlines()[1])
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Discrepancy triage job failed")
```

- [ ] **Step 6: Run the tests**

Run: `set -a; . ./.env; set +a; CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_router.py tests/triage/test_scheduler_job.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 7: Check the app still imports with the router registered**

Run: `set -a; . ./.env; set +a; CUDA_VISIBLE_DEVICES="" PYTHONPATH=src timeout 300 ./venv/bin/python -c "import api.main as m; print(sorted(r.path for r in m.app.routes if r.path.startswith('/triage')))"`
Expected: `['/triage/deals/{deal_id}', '/triage/deals/{deal_id}/run']`

- [ ] **Step 8: Commit**

```bash
P="src/api/routers/triage.py src/api/main.py src/services/backend_scheduler.py tests/triage/test_router.py tests/triage/test_scheduler_job.py"
git add $P && git commit -o $P -m "feat(triage): deal verdict routes and the scheduled re-triage job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Backfill and rollback scripts

**Files:**
- Create: `scripts/triage_backfill.py`, `scripts/triage_rollback.py`
- Test: `tests/triage/test_scripts.py`

**Interfaces:**
- Consumes: `loader.list_deal_ids`, `engine.run_triage`, `writer.rollback_run`, `src.services.db.get_conn`.
- Produces: `scripts/triage_backfill.py:main(argv) -> int` (0 ok, 1 if any deal failed); `scripts/triage_rollback.py:main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/triage/test_scripts.py`:

```python
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cur:
    def execute(self, *a):
        pass

    def fetchone(self):
        return (7,)


@contextmanager
def _conn():
    yield SimpleNamespace(cursor=lambda: _Cur())


def test_backfill_dry_run_passes_ids_and_the_no_deal_gap(monkeypatch, capsys):
    mod = _load("triage_backfill")
    seen = {}

    def run_triage(ids, mode, **kw):
        seen.update(ids=ids, mode=mode, **kw)
        return SimpleNamespace(failed={}, render=lambda: "REPORT", to_dict=lambda: {})

    monkeypatch.setattr(mod, "get_conn", _conn)
    monkeypatch.setattr(mod, "run_triage", run_triage)
    assert mod.main(["--deals", "A, B", "--dry-run"]) == 0
    assert seen["ids"] == ["A", "B"] and seen["mode"] == "backfill" and seen["dry_run"] is True
    assert seen["known_gaps"] == ["Invoices with no deal_id are not triaged: 7"]
    assert "REPORT" in capsys.readouterr().out


def test_rollback_prints_what_it_did(monkeypatch, capsys):
    mod = _load("triage_rollback")
    monkeypatch.setattr(mod, "get_conn", _conn)
    monkeypatch.setattr(mod.writer, "rollback_run", lambda conn, run_id: {
        "findings_removed": 3, "findings_kept": 1, "audit_rows_removed": 40})
    assert mod.main(["--run-id", "RUN-1"]) == 0
    out = capsys.readouterr().out
    assert "removed 3" in out and "kept 1" in out and "40 audit rows" in out
```

- [ ] **Step 2: Run to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_scripts.py -q -p no:randomly`
Expected: FAIL — script files missing.

- [ ] **Step 3: Implement the scripts**

`scripts/triage_backfill.py`:

```python
#!/usr/bin/env python
"""Run discrepancy triage over many deals and print the scale report (spec §9.1).

    set -a; . ./.env; set +a
    ./.venv/bin/python scripts/triage_backfill.py --all
    ./.venv/bin/python scripts/triage_backfill.py --all --dry-run --limit 200
    ./.venv/bin/python scripts/triage_backfill.py --deals DEALV2-005049,DEALV2-000001

--dry-run computes everything and writes nothing. Exit code 1 if any deal failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from src.services.db import get_conn  # noqa: E402
from src.services.triage import loader  # noqa: E402
from src.services.triage.engine import run_triage  # noqa: E402

_NO_DEAL_SQL = "SELECT count(*) FROM proc.bp_invoice_trgt WHERE deal_id IS NULL"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every deal with documents")
    which.add_argument("--deals", help="comma-separated deal ids")
    p.add_argument("--dry-run", action="store_true", help="compute and report; write nothing")
    p.add_argument("--limit", type=int, help="only the first N deals")
    p.add_argument("--report-json", help="also write the report as JSON to this path")
    a = p.parse_args(argv)

    with get_conn() as conn:
        cur = conn.cursor()
        ids = (loader.list_deal_ids(cur) if a.all
               else [d.strip() for d in a.deals.split(",") if d.strip()])
        cur.execute(_NO_DEAL_SQL)
        no_deal = cur.fetchone()[0]
    if a.limit:
        ids = ids[:a.limit]

    def progress(r):
        print(f"  ... {r.deals_done:,}/{r.deals_requested:,} deals, {len(r.failed)} failed",
              flush=True)

    report = run_triage(ids, "backfill", dry_run=a.dry_run, on_batch=progress,
                        known_gaps=[f"Invoices with no deal_id are not triaged: {no_deal:,}"])
    print(report.render())
    if a.report_json:
        with open(a.report_json, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/triage_rollback.py`:

```python
#!/usr/bin/env python
"""Undo one triage run (spec §9.1).

    set -a; . ./.env; set +a
    ./.venv/bin/python scripts/triage_rollback.py --run-id <uuid>

Removes the Action Centre findings that run created which nobody has touched (still
open, no owner, no due date, no resolver) and the run's audit rows. Findings a person
has acted on are kept.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from src.services.db import get_conn  # noqa: E402
from src.services.triage import writer  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    with get_conn() as conn:
        result = writer.rollback_run(conn, a.run_id)
    print(f"Run {a.run_id}: removed {result['findings_removed']} findings nobody had touched, "
          f"kept {result['findings_kept']} that a person had acted on, "
          f"removed {result['audit_rows_removed']} audit rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_scripts.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
P="scripts/triage_backfill.py scripts/triage_rollback.py tests/triage/test_scripts.py"
git add $P && git commit -o $P -m "feat(triage): backfill and rollback scripts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Planted-error recall, and proving the guards fail

**Files:**
- Test: `tests/triage/test_recall_live.py`

**Interfaces:**
- Consumes: `loader.list_deal_ids`, `loader.load_deal_sets`, `engine.triage_set`, `tolerance.load_config`.

- [ ] **Step 1: Write the recall test**

`tests/triage/test_recall_live.py`:

```python
"""Plant known problems in real deals, in memory only, and measure how many are caught.

Source data is never modified: each plant works on a deep copy of a loaded deal.
Run with PROCWISE_TEST_LIVE_DB=1. Prints a recall table (run with -s to see it).
"""
import copy
import dataclasses
import os
from decimal import Decimal as D

import pytest

from src.services.db import get_conn
from src.services.triage.engine import triage_set
from src.services.triage.link import link
from src.services.triage.loader import list_deal_ids, load_deal_sets
from src.services.triage.model import Line
from tests.triage.helpers import make_cfg

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")
CFG = make_cfg()


@pytest.fixture(scope="module")
def sample():
    with get_conn() as conn:
        cur = conn.cursor()
        ids = list_deal_ids(cur)[:200]
        return list(load_deal_sets(cur, ids).values())


def _target(ds):
    """First exactly-linked, non-credit invoice line with a priced PO line."""
    for lk in link(ds, CFG).line_links:
        if (lk.po_line is not None and lk.confidence == 1.0 and not lk.rollup
                and not lk.invoice.is_credit_note and lk.po_line.unit_price
                and lk.inv_line.unit_price and lk.inv_line.quantity and lk.po_line.quantity):
            return lk
    return None


def _all_results(out):
    return [r for f in out.findings for r in (*f.causes, *f.effects)]


def _plant_price(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, unit_price=old.unit_price * D("1.10"))
    return lambda out: any(r.rule_id == "unit_price" and r.claim_doc == inv.doc_id
                           and r.claim_line == old.line_ref for r in _all_results(out))


def _plant_quantity(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    idx = next(n for n, l in enumerate(inv.lines) if l.line_ref == lk.inv_line.line_ref)
    old = inv.lines[idx]
    inv.lines[idx] = dataclasses.replace(old, quantity=old.quantity + lk.po_line.quantity)
    return lambda out: any(r.rule_id == "quantity" and r.po_id == lk.po.doc_id
                           and r.auth_line == lk.po_line.line_ref for r in _all_results(out))


def _plant_rebill(ds, lk):
    src = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    billed = sum((i.net or D("0")) for i in ds.invoices if i.po_ref == lk.po.doc_id)
    if lk.po.net is None or src.net is None or billed + src.net <= lk.po.net * D("1.01"):
        return None   # a re-bill here would not exceed the PO, so there is nothing to catch
    copy_ = copy.deepcopy(src)
    copy_.doc_id = f"{src.doc_id}-PLANTED"
    ds.invoices.append(copy_)
    return lambda out: any(r.rule_id in ("quantity", "cumulative_total")
                           and r.po_id == lk.po.doc_id for r in _all_results(out))


def _plant_currency(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.currency = "XXX"
    return lambda out: any(r.rule_id == "currency" and r.claim_doc == inv.doc_id
                           for r in _all_results(out))


def _plant_unlinked(ds, lk):
    inv = next(i for i in ds.invoices if i.doc_id == lk.invoice.doc_id)
    inv.lines.append(Line(line_ref="PLANTED", item_id="PLANTED-ITEM",
                          description="zzqx planted qqzz", quantity=D("1"),
                          unit_price=D("100"), line_amount=D("100")))
    return lambda out: any(r.rule_id == "unlinked_line" and r.claim_line == "PLANTED"
                           for r in _all_results(out))


PLANTS = {"price +10%": _plant_price, "quantity doubled": _plant_quantity,
          "invoice re-billed": _plant_rebill, "currency swapped": _plant_currency,
          "line with no PO line": _plant_unlinked}


def test_planted_problems_are_caught(sample):
    table = {}
    for name, plant in PLANTS.items():
        caught = tried = 0
        for original in sample:
            ds = copy.deepcopy(original)
            lk = _target(ds)
            if lk is None:
                continue
            detected = plant(ds, lk)
            if detected is None:
                continue
            tried += 1
            caught += bool(detected(triage_set(ds, CFG)))
        table[name] = (caught, tried)
    print("\nPlanted-error recall:")
    for name, (caught, tried) in table.items():
        print(f"  {name:22s} {caught}/{tried} = {caught / tried:.1%}")
    for name, (caught, tried) in table.items():
        assert tried >= 50, f"{name}: too few eligible deals ({tried})"
        assert caught / tried >= 0.95, f"{name}: recall {caught}/{tried}"
```

- [ ] **Step 2: Run it**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_recall_live.py -q -s -p no:randomly`
Expected: PASS, with a recall table printed. If a plant falls below 95%, do **not** lower the bar: print the missed deals (add a temporary `print(ds.deal_id)` on a miss), diagnose with superpowers:systematic-debugging, fix the engine, re-run.

- [ ] **Step 3: Prove the always-S1 guard fails when broken**

Edit `src/services/triage/score.py`: change `ALWAYS_S1 = frozenset({"cumulative_total", "currency", "duplicate"})` to `ALWAYS_S1 = frozenset()`.
Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_score.py tests/triage/test_group.py -q -p no:randomly`
Expected: FAIL — at least `test_always_s1_rules_ignore_the_amount` and `test_override_beats_fx_cap`.
Restore: `git checkout -- src/services/triage/score.py` and re-run → PASS.

- [ ] **Step 4: Prove the max-S3 guard fails when broken**

Edit `score.py`: `MAX_S3 = frozenset()`. Run the same command. Expected: FAIL — `test_max_s3_rules`. Restore with `git checkout -- src/services/triage/score.py` → PASS.

- [ ] **Step 5: Prove effects cannot soften an always-S1 finding**

Edit `src/services/triage/model.py`, in `Finding.severity` change `(*self.causes, *self.effects)` to `self.causes`. Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_group.py -q -p no:randomly`.
Expected: FAIL — `test_unflagged_rebill_is_one_quantity_finding` (severity drops below S1). Restore with `git checkout -- src/services/triage/model.py` → PASS.

- [ ] **Step 6: Prove the writer never touches a person's decision**

Edit `src/services/triage/writer.py`: in `_UPDATE_FINDING` delete ` AND status = 'open'`, and in `write_batch` change `if status == "open":` to `if True:`. Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/test_writer_live.py -q -p no:randomly`.
Expected: FAIL — `test_resolved_finding_is_not_reopened`. Restore with `git checkout -- src/services/triage/writer.py` → PASS.

- [ ] **Step 7: Commit**

```bash
P="tests/triage/test_recall_live.py"
git add $P && git commit -o $P -m "test(triage): planted-error recall on 200 real deals

Proved by breaking them: emptying ALWAYS_S1 or MAX_S3, letting effects out of a
finding's severity, or letting the writer update a closed finding each turns its
test red; restored, all pass.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Live proof at scale

No new code. Every step runs against the running local stack and `bp_testdb` (the `.env` database). Record each step's output; the final report to the user quotes them.

- [ ] **Step 1: Full test sweep**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/triage/ tests/migrations/test_2026_09_24_bp_triage.py tests/governance/test_governed_limits.py -q -p no:randomly`
Expected: all pass, **0 skipped**.

- [ ] **Step 2: Restart the API on the committed code**

Run: `sudo -n systemctl restart procwise && sleep 20 && systemctl show procwise -p ExecMainStartTimestamp && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs`
Expected: a start timestamp from just now, and `200`.

- [ ] **Step 3: Dry run on 200 deals (speed check before the real thing)**

Run: `set -a; . ./.env; set +a; time ./.venv/bin/python scripts/triage_backfill.py --all --dry-run --limit 200`
Expected: report printed, 0 failed. Multiply the elapsed time by 5041/200; if the projection exceeds 15 minutes, stop and profile (`python -m cProfile -s cumtime`) before continuing.

- [ ] **Step 4: Full backfill**

Run: `set -a; . ./.env; set +a; time ./.venv/bin/python scripts/triage_backfill.py --all --report-json /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/719b52d2-29eb-4652-a23d-1d1d07af55c3/scratchpad/triage_backfill.json`
Expected: finishes in < 15 minutes, 0 failed deals, and a noise ratio is printed. Note the run id from the first line.

- [ ] **Step 5: Re-run to prove idempotence**

Run the Step 4 command again (new run id).
Expected: `Writes:` shows `inserted 0` and `superseded 0`; `updated` equals the first run's `inserted`.

- [ ] **Step 6: Database cross-check**

```bash
set -a; . ./.env; set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT severity, category, count(*) FROM proc.bp_detection_finding
 WHERE finding_id IN (SELECT finding_id FROM proc.bp_triage_finding) AND status='open'
 GROUP BY 1,2 ORDER BY 1,3 DESC;"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT count(*) AS audit_rows FROM proc.bp_triage_result
 WHERE run_id = (SELECT run_id FROM proc.bp_triage_run ORDER BY started_at DESC LIMIT 1);"
```
Expected: counts agree with the report's S1/S2 findings and its outcome total.

- [ ] **Step 7: Hand-check 10 findings against the raw rows**

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT f.finding_id, f.deal_id, f.rule_id, f.severity, f.doc_pk, f.observed_value,
       f.expected_value, f.delta
  FROM proc.bp_detection_finding f JOIN proc.bp_triage_finding m USING (finding_id)
 WHERE f.status='open' ORDER BY random() LIMIT 10;"
```
For each row, query the underlying `proc.bp_invoice_trgt` / `bp_invoice_line_items_trgt` / `bp_purchase_order_trgt` / `bp_po_line_items_trgt` rows for its `doc_pk` and deal, and confirm the observed and expected values and the exposure by hand. Write the ten verdicts (correct / wrong + why) into `scratchpad/triage_handcheck.md`. Any "wrong" → superpowers:systematic-debugging, fix, re-run from Step 1.

- [ ] **Step 8: The verdict endpoint on the running API**

Run: `curl -s http://localhost:8000/triage/deals/DEALV2-005049 | python3 -m json.tool | head -40`
Expected: `"verdict": "Blocked"`, the over-billing visible, and each finding carrying a non-null `finding_id`. (If the router-level auth returns 401, repeat with the bearer token the local UI uses, or with auth explicitly off as configured in `.env`; record which.)

- [ ] **Step 9: The Action Centre's own endpoint on the gateway**

First confirm the gateway reads the same database: `grep -E "DB_NAME|DATABASE" /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api/.env* 2>/dev/null`. It must name `bp_testdb`; if it names another database, stop and report that the Action Centre shows a different database from the one backfilled.

Start it (in the background) with the local auth bypass:
```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api
AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development AUTH_BYPASS_SUB=triage-check \
  AUTH_BYPASS_EMAIL=triage-check@local node --experimental-global-webcrypto \
  --max-old-space-size=16384 ./dist/main.js
```
Then: `curl -s http://localhost:3001/discrepancies/metrics | python3 -m json.tool` and `curl -s "http://localhost:3001/discrepancies?severity=critical" | python3 -m json.tool | head -40`
Expected: `total_open` and `critical_open` include the triage findings; the list shows triage rows (`rule_id` such as `quantity`, `duplicate`, `unit_price`) with their plain-language `notes`.

- [ ] **Step 10: Report**

Write the outcome for the user from the recorded outputs: runtime and deals/second; noise ratio; deals per verdict; exposure by severity; findings per rule; the recall table (Task 15); the ten hand-checks; the gateway evidence; the idempotence result; and the run id to pass to `scripts/triage_rollback.py` if they want the backfill removed. State plainly anything that did not meet a success criterion.
```
