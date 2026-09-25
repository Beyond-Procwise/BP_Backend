# Value Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record money avoided, claimed, recovered and realised in an append-only ledger, written from the Action Centre and the Value Found drawer, and read by every surface that states recovered value.

**Architecture:**
- **Table:** a new append-only table, `proc.bp_value_outcome`, guarded by a database trigger.
- **Writes:** a backend service, `src/services/value_ledger.py`, writes to it. Each write happens in one explicit transaction together with the finding's or opportunity's status change and an audit row.
- **Reads:** `value_summary_service`, the opportunities dashboard, the executive-summary report and the weekly digest read it.
- **UI:** one shared React panel (`ValueOutcomePanel`) records outcomes from both the SpendIQ Action Centre and Procurement Home. The Value Found drawer gains a "Being claimed" section and a "Mark realised" action.
- **Gateway:** unchanged.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, PostgreSQL (bp_testdb / bp_sqldb), pytest; React 19 (JS), axios, vitest.

**Spec:** `docs/superpowers/specs/2026-09-25-value-ledger-design.md` (approved 2026-09-25, commit `800e6ba`)

## Global Constraints

**Money and data**
- Two-step recovery: resolving a finding records `claimed`; `recovered` only follows a later "credit received". Only confirmed money (`avoided`, `recovered`, `realised_saving`) counts in the headline.
- Amounts are stored in the document currency. GBP is converted at record time with `amount / rates[ccy] * rates["GBP"]`, and the rate and date are stored with it. With no rate, `amount_gbp` is NULL and the row counts in no GBP total. Never invent a figure.
- `tenant_id` is the constant `'default'`. No RLS in this work.
- `proc.bp_value_outcome` is append-only: a BEFORE UPDATE/DELETE row trigger plus a TRUNCATE statement trigger, per `deploy/sql/2026-09-16_bp_agent_actions_immutable.sql`. Corrections are new rows with `supersedes_id`.
- `evidence_ref` is required for `recovered`. `amount > 0` for every type except `claim_dropped` (where it is NULL).
- The legacy columns `resolution_outcome`, `recovered_amount` and `realised_savings_gbp` are neither written by new code nor read. The gateway is not modified.

**Backend conventions**
- Every write route takes `principal=Depends(require_user)`. The actor comes from the token (`principal.subject`), never from the body.
- `get_conn()` is AUTOCOMMIT. Every write sets `conn.autocommit = False` and commits or rolls back explicitly, exactly as `opportunity_store.set_stage` does. Service functions accept `conn=` and, when it is given, the caller owns the transaction.
- Every write records `record_action_or_fail(phase="value", action_type="value.outcome_recorded", conn=conn, ...)`. If the audit row cannot be written, the write is refused.
- Refused lifecycle moves surface as `IllegalTransition` via `src.services.lifecycle.refusal`, mapped to HTTP 409.
- New tables use the `bp_` prefix and indexes are named `ix_bp_<table>_<cols>`.

**UI conventions**
- All new UI strings go through `tOr('key', 'English fallback', vars)` from `src/lib/i18n`.
- The UI checkout is shared with other sessions that have uncommitted edits (`App.jsx`, `index.css`, i18n files). Stage only this plan's hunks (`git add -p`). Never `git add -A`, never `git commit -a`, never `git commit -o` on a file another session has dirty.

**Commits and test environment**
- BP_Backend commits use `git commit -o <paths>`, because the index is shared with another session. End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work stays on `Development`; no push without the user's say-so.
- Test environment for every pytest command in this plan:
  ```bash
  cd /home/muthu/PycharmProjects/BP_Backend; set -a; . ./.env >/dev/null 2>&1; set +a
  export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9 OLLAMA_CLOUD_BASE_URL=http://127.0.0.1:9 OLLAMA_CLOUD_API_KEY=
  ```
  Live-DB tests add `PROCWISE_TEST_LIVE_DB=1` (DB_NAME in `.env` is `bp_testdb`). Live tests insert probe rows and **roll back**, the pattern of `tests/guardrails/test_lifecycle_transitions.py`.
  > This replaces the spec §8 cleanup-by-trigger-disable. Rollback is the repo's established pattern, needs no DDL in a test, and leaves nothing behind. Recorded here as a deliberate deviation.

## Review Focus

1. **Double-clicks and races.** Two people (or two clicks) record an outcome on the same open finding. The second must get 409 `finding_already_moved`, and the ledger must hold exactly one row. Test: Task 3 `test_second_outcome_on_same_finding_is_refused`.
2. **Recovered differs from claimed.** A claim of £500 is settled as recovered £320 (partial credit). The headline must show £320 recovered, and the finding must drop out of "Being claimed". Test: Task 4 `test_partial_recovery_counts_the_recovered_figure`.
3. **Foreign currency with no FX rate.** An outcome in NZD while rates are unavailable is saved with `amount_gbp` NULL. It is excluded from GBP totals, not shown as £0, and the drawer shows the native amount. Tests: Task 2 `test_unconvertible_currency_keeps_native_amount_and_null_gbp`, Task 9 `claimLabel` test.
4. **"In play" after money is settled.** Once a finding is avoided, recovered or dropped, the Home hero's "in play" figure must fall. A person would expect settled money to stop being "in play". Test: Task 4 `test_in_play_excludes_settled_findings`.
5. **Double-counting a PO and its invoices.** An "invoices exceed PO total" finding on PO X and line findings on invoices against PO X (340 such rows in bp_testdb on 2026-09-25) must count once. Test: Task 4 `test_line_findings_under_an_overbilled_po_are_superseded`.

---

## File Structure

**BP_Backend**

| path | role |
|---|---|
| `deploy/sql/2026-09-26_bp_value_outcome.sql` (+ `_rollback.sql`) | the table, indexes and append-only guard |
| `src/services/value_ledger.py` | pure validation, GBP conversion and state derivation, plus the SQL write and read functions |
| `src/api/routers/value_ledger.py` | `/value/...` routes |
| `src/api/main.py` | include the new router |
| `src/services/value_summary_service.py` | reads the ledger; triage money types; PO-over-line supersede; new summary fields |
| `src/services/opportunity_dashboard.py` | realised KPIs from the ledger |
| `src/services/rga/builders/exec_procurement_summary.py` | "Saved (GBP)" fact from the ledger |
| `src/services/value_digest.py` | "recovered this week" keyed on the ledger date |

Tests: `tests/migrations/test_2026_09_26_bp_value_outcome.py`, `tests/services/test_value_ledger.py`, `tests/services/test_value_ledger_live.py`, `tests/api/test_value_ledger_router.py`, `tests/services/test_value_summary_service.py` (extend), `tests/services/test_value_summary_ledger_live.py`, `tests/services/test_opportunity_dashboard.py` (extend), `tests/services/test_value_digest.py` (fix + extend), `tests/services/rga/test_exec_summary_ledger.py`.

**beyond_procwise_ui**

| path | role |
|---|---|
| `src/lib/valueOutcome.js` (+ `.test.js`) | pure: money issue types, payload builders, validation, labels |
| `src/components/value/ValueOutcomePanel.jsx` | the shared "What happened to this money?" panel |
| `src/modules/SpendIQ/data/useSpendData.js` | money findings get a "Record outcome" action |
| `src/modules/SpendIQ/engine.js` | `record_outcome` opens the panel through a bridge |
| `src/modules/SpendIQ/index.jsx` | mounts the panel and exposes `window.__SPENDIQ_OPEN_OUTCOME__` |
| `src/modules/ProcurementHome/index.jsx` | the Home deck opens the panel for `record_outcome` |
| `src/lib/valueFound.js`, `src/modules/ProcurementHome/ValueFoundDrawer.jsx`, `src/modules/ProcurementHome/heroFigure.js` | "Saved" headline, "Being claimed" section, Credit received / Claim dropped / Mark realised, and "in play" from `in_play_gbp` |

---

### Task 1: The table and its append-only guard

**Files:**
- Create: `deploy/sql/2026-09-26_bp_value_outcome.sql`
- Create: `deploy/sql/2026-09-26_bp_value_outcome_rollback.sql`
- Test: `tests/migrations/test_2026_09_26_bp_value_outcome.py`

**Interfaces:**
- Produces: table `proc.bp_value_outcome` with the columns below. Everything later depends on these exact names.

- [ ] **Step 1: Write the failing live test**

```python
"""proc.bp_value_outcome exists, has the spec's shape, and refuses every edit.

Nothing real is touched: each test inserts a probe row and rolls back.
Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md §3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _insert(cur, **over):
    row = dict(source_type="finding", source_id="probe-1", outcome_type="claimed",
               amount=100, currency="GBP", amount_gbp=100, recorded_by="pytest-value-ledger",
               valid_from="2026-09-25")
    row.update(over)
    cols = ", ".join(row)
    cur.execute(f"INSERT INTO proc.bp_value_outcome ({cols}) VALUES "
                f"({', '.join(['%s'] * len(row))}) RETURNING outcome_id", tuple(row.values()))
    return cur.fetchone()[0]


def test_columns_match_the_spec(conn):
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='proc' AND table_name='bp_value_outcome'")
    cols = {r[0] for r in cur.fetchall()}
    assert cols == {"outcome_id", "tenant_id", "source_type", "source_id", "outcome_type",
                    "amount", "currency", "amount_gbp", "fx_rate", "fx_as_of", "evidence_ref",
                    "note", "supersedes_id", "recorded_by", "valid_from", "recorded_at"}


def test_tenant_defaults_to_the_constant(conn):
    cur = conn.cursor()
    oid = _insert(cur)
    cur.execute("SELECT tenant_id FROM proc.bp_value_outcome WHERE outcome_id=%s", (oid,))
    assert cur.fetchone()[0] == "default"


@pytest.mark.parametrize("stmt", [
    "UPDATE proc.bp_value_outcome SET amount = 1 WHERE outcome_id = %s",
    "DELETE FROM proc.bp_value_outcome WHERE outcome_id = %s",
])
def test_rows_cannot_be_edited_or_deleted(conn, stmt):
    import psycopg2
    cur = conn.cursor()
    oid = _insert(cur)
    with pytest.raises(psycopg2.Error, match="append-only"):
        cur.execute(stmt, (oid,))


def test_truncate_is_refused(conn):
    import psycopg2
    with pytest.raises(psycopg2.Error, match="append-only"):
        conn.cursor().execute("TRUNCATE proc.bp_value_outcome")


def test_recovered_needs_evidence(conn):
    import psycopg2
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(conn.cursor(), outcome_type="recovered", evidence_ref=None)


def test_amount_must_be_positive_except_claim_dropped(conn):
    import psycopg2
    cur = conn.cursor()
    _insert(cur, outcome_type="claim_dropped", amount=None, currency=None, amount_gbp=None)
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(cur, amount=0)


def test_unknown_outcome_type_is_refused(conn):
    import psycopg2
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(conn.cursor(), outcome_type="savings")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/migrations/test_2026_09_26_bp_value_outcome.py -q -p no:randomly`
Expected: FAIL with `relation "proc.bp_value_outcome" does not exist`.

- [ ] **Step 3: Write the migration**

`deploy/sql/2026-09-26_bp_value_outcome.sql`:

```sql
BEGIN;
-- deploy/sql/2026-09-26_bp_value_outcome.sql
--
-- The value ledger: money a finding or opportunity actually produced, recorded by a
-- person, never overwritten. Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md
--
-- Replaces the overwrite-in-place columns bp_extraction_discrepancy.resolution_outcome /
-- recovered_amount and bp_opportunity.realised_savings_gbp as the place recovered value
-- is read from. Those columns are left in place and no longer read.
--
-- Append-only by TRIGGER, not REVOKE: the application role owns this table and an owner
-- may re-grant itself anything (see 2026-09-16_bp_agent_actions_immutable.sql). A mistake
-- is corrected by a new row whose supersedes_id names the row it replaces.

CREATE TABLE IF NOT EXISTS proc.bp_value_outcome (
    outcome_id     bigserial PRIMARY KEY,
    -- B2 decision: tenant_id on every new table, defaulted to one constant.
    tenant_id      text NOT NULL DEFAULT 'default',
    source_type    text NOT NULL CHECK (source_type IN ('finding', 'opportunity')),
    -- discrepancy_id (bigint) or opportunity_id (varchar), both stored as text.
    source_id      text NOT NULL,
    outcome_type   text NOT NULL CHECK (outcome_type IN (
                       'avoided', 'claimed', 'recovered', 'claim_dropped',
                       'realised_saving', 'terms_improved', 'cycle_time')),
    amount         numeric(18,2),
    currency       char(3),
    amount_gbp     numeric(18,2),
    fx_rate        numeric,
    fx_as_of       timestamptz,
    evidence_ref   text,
    note           text,
    supersedes_id  bigint REFERENCES proc.bp_value_outcome (outcome_id),
    recorded_by    text NOT NULL,
    valid_from     date NOT NULL DEFAULT current_date,
    recorded_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT bp_value_outcome_amount_ck CHECK (
        CASE WHEN outcome_type = 'claim_dropped' THEN amount IS NULL
             ELSE amount IS NOT NULL AND amount > 0 END),
    CONSTRAINT bp_value_outcome_currency_ck CHECK (
        outcome_type IN ('claim_dropped', 'cycle_time') OR currency IS NOT NULL),
    CONSTRAINT bp_value_outcome_evidence_ck CHECK (
        outcome_type <> 'recovered' OR coalesce(btrim(evidence_ref), '') <> '')
);

CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_source
    ON proc.bp_value_outcome (source_type, source_id);
CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_type_valid
    ON proc.bp_value_outcome (outcome_type, valid_from);
CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_supersedes
    ON proc.bp_value_outcome (supersedes_id);

CREATE OR REPLACE FUNCTION proc.bp_value_outcome_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_value_outcome is append-only: % refused. Record a correcting row (supersedes_id) instead.', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS tr_bp_value_outcome_immutable ON proc.bp_value_outcome;
CREATE TRIGGER tr_bp_value_outcome_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_value_outcome
    FOR EACH ROW EXECUTE FUNCTION proc.bp_value_outcome_immutable();

DROP TRIGGER IF EXISTS tr_bp_value_outcome_no_truncate ON proc.bp_value_outcome;
CREATE TRIGGER tr_bp_value_outcome_no_truncate
    BEFORE TRUNCATE ON proc.bp_value_outcome
    FOR EACH STATEMENT EXECUTE FUNCTION proc.bp_value_outcome_immutable();

COMMIT;
```

`deploy/sql/2026-09-26_bp_value_outcome_rollback.sql`:

```sql
BEGIN;
-- Removes the ledger entirely. Only safe before any real outcome has been recorded:
-- dropping the table discards the value history it exists to keep.
DROP TABLE IF EXISTS proc.bp_value_outcome;
DROP FUNCTION IF EXISTS proc.bp_value_outcome_immutable();
COMMIT;
```

- [ ] **Step 4: Apply to bp_testdb and run the test**

Run: `PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-26_bp_value_outcome.sql`
Expected: `COMMIT`.

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/migrations/test_2026_09_26_bp_value_outcome.py -q -p no:randomly`
Expected: 8 passed.

- [ ] **Step 5: Prove the guard fails when broken** (memory rule: break it on purpose and watch it go red)

In a psql session run `BEGIN; ALTER TABLE proc.bp_value_outcome DISABLE TRIGGER tr_bp_value_outcome_immutable;` and, in that same transaction, run the edit test's UPDATE on a probe row. It must succeed, which shows the test was only green because of the trigger. Then run `ROLLBACK;`. Record the observed output in the task report.

- [ ] **Step 6: Commit**

```bash
git add -f deploy/sql/2026-09-26_bp_value_outcome.sql deploy/sql/2026-09-26_bp_value_outcome_rollback.sql tests/migrations/test_2026_09_26_bp_value_outcome.py
git commit -o deploy/sql/2026-09-26_bp_value_outcome.sql deploy/sql/2026-09-26_bp_value_outcome_rollback.sql tests/migrations/test_2026_09_26_bp_value_outcome.py \
  -m "feat(value-ledger): proc.bp_value_outcome, append-only by trigger

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Ledger rules as pure functions

**Files:**
- Create: `src/services/value_ledger.py` (pure part only)
- Test: `tests/services/test_value_ledger.py`

**Interfaces:**
- Consumes: `value_summary_service._to_gbp(amount, currency, rates) -> (gbp|None, info|None)` and `value_summary_service._get_rates() -> dict|None` (existing).
- Produces:
  - `OUTCOME_TYPES: frozenset[str]`
  - `FINDING_OPEN_OUTCOMES = ("avoided", "claimed", "accepted")`
  - `SETTLE_OUTCOMES = ("recovered", "claim_dropped")`
  - `SETTLED_STATES = frozenset({"avoided", "recovered", "claim_dropped", "realised_saving"})`
  - `class LedgerError(ValueError)` with attribute `code: str` (e.g. `"invalid_amount"`, `"evidence_required"`, `"no_open_claim"`, `"finding_already_moved"`, `"not_a_money_finding"`)
  - `validate_outcome(outcome_type: str, amount, currency: str|None, evidence_ref: str|None) -> tuple[Decimal|None, str|None]`: returns the cleaned `(amount, currency)` or raises `LedgerError`
  - `convert_to_gbp(amount: Decimal, currency: str, rates: dict|None) -> dict` with keys `amount_gbp`, `fx_rate`, `fx_as_of`
  - `current_state(rows: list[dict]) -> dict|None`: the latest non-superseded row for one source

- [ ] **Step 1: Write the failing tests**

```python
"""Pure ledger rules. Spec §3, §7."""
from decimal import Decimal

import pytest

from src.services import value_ledger as vl


def test_amount_is_required_and_positive_for_money_outcomes():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("claimed", None, "GBP", None)
    assert e.value.code == "invalid_amount"
    with pytest.raises(vl.LedgerError):
        vl.validate_outcome("claimed", "0", "GBP", None)
    with pytest.raises(vl.LedgerError):
        vl.validate_outcome("claimed", "abc", "GBP", None)


def test_amount_is_rounded_to_pence_and_currency_upper_cased():
    assert vl.validate_outcome("avoided", "120.456", "gbp", None) == (Decimal("120.46"), "GBP")


def test_recovered_needs_an_evidence_reference():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("recovered", "50", "GBP", "   ")
    assert e.value.code == "evidence_required"
    assert vl.validate_outcome("recovered", "50", "GBP", "CN-123") == (Decimal("50.00"), "GBP")


def test_claim_dropped_carries_no_amount():
    assert vl.validate_outcome("claim_dropped", "99", "GBP", None) == (None, None)


def test_currency_must_be_three_letters():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("claimed", "10", "POUNDS", None)
    assert e.value.code == "invalid_currency"


def test_gbp_passes_through_with_no_rate():
    out = vl.convert_to_gbp(Decimal("10.00"), "GBP", None)
    assert out == {"amount_gbp": Decimal("10.00"), "fx_rate": None, "fx_as_of": None}


def test_foreign_amount_is_converted_and_the_rate_kept():
    rates = {"USD": 1.0, "GBP": 0.8, "_fetched_at": "2026-09-25T09:00:00+00:00"}
    out = vl.convert_to_gbp(Decimal("100.00"), "USD", rates)
    assert out["amount_gbp"] == Decimal("80.00")
    assert out["fx_rate"] == Decimal("0.8")
    assert out["fx_as_of"] == "2026-09-25T09:00:00+00:00"


def test_unconvertible_currency_keeps_native_amount_and_null_gbp():
    out = vl.convert_to_gbp(Decimal("100.00"), "NZD", {"USD": 1.0, "GBP": 0.8})
    assert out == {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}


def test_current_state_is_the_latest_row_nobody_superseded():
    rows = [
        {"outcome_id": 1, "outcome_type": "claimed", "supersedes_id": None, "recorded_at": 1},
        {"outcome_id": 2, "outcome_type": "recovered", "supersedes_id": None, "recorded_at": 2},
        {"outcome_id": 3, "outcome_type": "recovered", "supersedes_id": 2, "recorded_at": 3},
    ]
    assert vl.current_state(rows)["outcome_id"] == 3
    assert vl.current_state([]) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./venv/bin/python -m pytest tests/services/test_value_ledger.py -q -p no:randomly`
Expected: FAIL with `ImportError: cannot import name 'value_ledger'`.

- [ ] **Step 3: Implement the pure layer**

`src/services/value_ledger.py`:

```python
"""The value ledger: money a finding or opportunity actually produced.

Pure rules first (validated, converted, derived) so they unit-test on plain values; the
SQL layer below them is thin. Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md
"""
from __future__ import annotations

import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

log = logging.getLogger(__name__)

OUTCOME_TYPES = frozenset({"avoided", "claimed", "recovered", "claim_dropped",
                           "realised_saving", "terms_improved", "cycle_time"})
# What a person may say when closing an open money finding. `accepted` closes it and
# records no row: no money moved.
FINDING_OPEN_OUTCOMES = ("avoided", "claimed", "accepted")
SETTLE_OUTCOMES = ("recovered", "claim_dropped")
SETTLED_STATES = frozenset({"avoided", "recovered", "claim_dropped", "realised_saving"})

_CCY = re.compile(r"^[A-Z]{3}$")
_PENNY = Decimal("0.01")


class LedgerError(ValueError):
    """A request the ledger refuses. ``code`` is what the route returns to the UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_outcome(outcome_type: str, amount: Any, currency: Optional[str],
                     evidence_ref: Optional[str]) -> tuple[Optional[Decimal], Optional[str]]:
    if outcome_type not in OUTCOME_TYPES:
        raise LedgerError("invalid_outcome", f"unknown outcome {outcome_type!r}")
    if outcome_type == "claim_dropped":
        return None, None
    try:
        value = Decimal(str(amount).strip())
    except (InvalidOperation, AttributeError):
        raise LedgerError("invalid_amount", f"amount {amount!r} is not a number")
    if not value.is_finite() or value <= 0:
        raise LedgerError("invalid_amount", "amount must be greater than zero")
    value = value.quantize(_PENNY, rounding=ROUND_HALF_UP)
    ccy = str(currency or "").strip().upper()
    if outcome_type != "cycle_time" and not _CCY.match(ccy):
        raise LedgerError("invalid_currency", f"currency {currency!r} is not a 3-letter code")
    if outcome_type == "recovered" and not str(evidence_ref or "").strip():
        raise LedgerError("evidence_required",
                          "a recovered amount needs its credit note or document reference")
    return value, (ccy or None)


def convert_to_gbp(amount: Decimal, currency: str, rates: Optional[dict]) -> dict:
    """GBP at record time, with the rate that produced it. No rate -> None, never a guess."""
    if currency == "GBP":
        return {"amount_gbp": amount, "fx_rate": None, "fx_as_of": None}
    if not rates or currency not in rates or "GBP" not in rates:
        return {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}
    rate = Decimal(str(rates["GBP"])) / Decimal(str(rates[currency]))
    return {"amount_gbp": (amount * rate).quantize(_PENNY, rounding=ROUND_HALF_UP),
            "fx_rate": rate, "fx_as_of": rates.get("_fetched_at")}


def current_state(rows: list[dict]) -> Optional[dict]:
    """The latest row for one source that no other row supersedes."""
    replaced = {r.get("supersedes_id") for r in rows if r.get("supersedes_id")}
    live = [r for r in rows if r["outcome_id"] not in replaced]
    if not live:
        return None
    return max(live, key=lambda r: (r["recorded_at"], r["outcome_id"]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/services/test_value_ledger.py -q -p no:randomly`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/value_ledger.py tests/services/test_value_ledger.py
git commit -o src/services/value_ledger.py tests/services/test_value_ledger.py \
  -m "feat(value-ledger): validation, GBP at record time, current-state rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Writing outcomes (SQL layer, one transaction each)

**Files:**
- Modify: `src/services/value_ledger.py` (append the SQL layer)
- Test: `tests/services/test_value_ledger_live.py`

**Interfaces:**
- Consumes:
  - Task 2's pure functions.
  - `src.services.agent_actions.record_action_or_fail(*, phase, action_type, conn=None, **fields)` (existing; fields include `doc_type`, `doc_pk`, `agent`, `status`, `summary`, `details`).
  - `src.services.lifecycle.refusal(exc)` and `IllegalTransition` (existing).
  - `src.services.opportunity_store.set_stage(opportunity_id, stage, realised_savings=None, conn=None)` (existing).
  - `value_summary_service.DISCREPANCY_VALUE_TYPES` (extended in Task 4; this task reads the constant as it exists at call time).
  - `value_summary_service.discrepancy_delta(row)` (existing).
- Produces (each takes `conn=None`; when `conn` is given the caller owns the transaction):
  - `record_finding_outcome(discrepancy_id: int, outcome: str, amount, currency, *, actor: str, valid_from: str|None=None, note: str|None=None, conn=None) -> dict` returns `{"outcome_id": int|None, "state": str, "amount_gbp": Decimal|None}`. `outcome_id` is None for `accepted`.
  - `settle_claim(discrepancy_id: int, outcome: str, amount=None, currency=None, *, actor: str, evidence_ref: str|None=None, valid_from=None, note=None, conn=None) -> dict`, same return shape.
  - `realise_opportunity(opportunity_id: str, amount, currency, *, actor: str, valid_from=None, evidence_ref=None, note=None, conn=None) -> dict`, same shape.
  - `correct_outcome(outcome_id: int, amount, currency, *, actor: str, note: str, evidence_ref=None, conn=None) -> dict`, same shape.
  - `finding_outcomes(discrepancy_id: int, conn=None) -> dict` returns `{"state": str|None, "history": list[dict], "prefill": {"amount": str|None, "currency": str|None, "is_money": bool}}`.

- [ ] **Step 1: Write the failing live tests**

```python
"""Ledger writes against the real database, each inside a transaction that is rolled back.
Spec §4, §7."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]

ACTOR = "pytest-value-ledger"


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _finding(cur, status="open", issue_type="duplicate_invoice", raw="500.00") -> int:
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice', 'probe', %s, %s, %s, %s, %s, 'warning', %s, false) "
        "RETURNING discrepancy_id",
        (f"PROBE-{uuid.uuid4().hex[:8]}", f"probe_{uuid.uuid4().hex[:8]}", raw, "+" + raw,
         issue_type, status))
    return cur.fetchone()[0]


def _rows(cur, did):
    cur.execute("SELECT outcome_type, amount, recorded_by FROM proc.bp_value_outcome "
                "WHERE source_type='finding' AND source_id=%s ORDER BY outcome_id", (str(did),))
    return cur.fetchall()


def test_claim_closes_the_finding_and_writes_one_row(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    out = vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    assert out["state"] == "claimed"
    cur.execute("SELECT status, resolved_by FROM proc.bp_extraction_discrepancy "
                "WHERE discrepancy_id=%s", (did,))
    assert cur.fetchone() == ("resolved", ACTOR)
    assert [r[0] for r in _rows(cur, did)] == ["claimed"]


def test_accepting_the_charge_closes_it_and_records_no_money(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    out = vl.record_finding_outcome(did, "accepted", None, None, actor=ACTOR, conn=conn)
    assert out == {"outcome_id": None, "state": "accepted", "amount_gbp": None}
    assert _rows(cur, did) == []


def test_second_outcome_on_same_finding_is_refused(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    vl.record_finding_outcome(did, "avoided", "500", "GBP", actor=ACTOR, conn=conn)
    with pytest.raises(vl.LedgerError) as e:
        vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "finding_already_moved"
    assert len(_rows(cur, did)) == 1


def test_a_non_money_finding_cannot_record_money(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, issue_type="line_missing_amount")
    with pytest.raises(vl.LedgerError) as e:
        vl.record_finding_outcome(did, "claimed", "10", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "not_a_money_finding"


def test_settle_needs_an_open_claim(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    with pytest.raises(vl.LedgerError) as e:
        vl.settle_claim(did, "recovered", "500", "GBP", actor=ACTOR, evidence_ref="CN-1", conn=conn)
    assert e.value.code == "no_open_claim"


def test_claim_then_partial_credit(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    out = vl.settle_claim(did, "recovered", "320", "GBP", actor=ACTOR,
                          evidence_ref="CN-77", conn=conn)
    assert out["state"] == "recovered"
    assert [(r[0], str(r[1])) for r in _rows(cur, did)] == [("claimed", "500.00"),
                                                            ("recovered", "320.00")]
    with pytest.raises(vl.LedgerError):          # settled: no second settle
        vl.settle_claim(did, "claim_dropped", actor=ACTOR, conn=conn)


def test_a_failed_audit_write_leaves_the_finding_open(conn, monkeypatch):
    from src.services import value_ledger as vl
    from src.services.agent_actions import AuditWriteError
    cur = conn.cursor()
    did = _finding(cur)
    cur.execute("SAVEPOINT before_write")

    def _boom(**kw):
        raise AuditWriteError("audit down")
    monkeypatch.setattr(vl, "record_action_or_fail", _boom)
    with pytest.raises(AuditWriteError):
        vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    cur.execute("ROLLBACK TO SAVEPOINT before_write")
    cur.execute("SELECT status FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (did,))
    assert cur.fetchone()[0] == "open"
    assert _rows(cur, did) == []


def test_correction_supersedes_and_becomes_the_state(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    first = vl.record_finding_outcome(did, "avoided", "500", "GBP", actor=ACTOR, conn=conn)
    fixed = vl.correct_outcome(first["outcome_id"], "450", "GBP", actor=ACTOR,
                               note="invoice was 450 net", conn=conn)
    hist = vl.finding_outcomes(did, conn=conn)
    assert hist["state"] == "avoided"
    assert hist["history"][-1]["outcome_id"] == fixed["outcome_id"]
    assert str(hist["history"][-1]["amount"]) == "450.00"


def test_prefill_offers_the_findings_own_figure(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, raw="1234.50")
    pre = vl.finding_outcomes(did, conn=conn)["prefill"]
    assert pre["is_money"] is True
    assert pre["amount"] == "1234.50"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_value_ledger_live.py -q -p no:randomly`
Expected: FAIL with `AttributeError: module 'src.services.value_ledger' has no attribute 'record_finding_outcome'`.

- [ ] **Step 3: Implement the SQL layer** (append to `src/services/value_ledger.py`)

```python
# --------------------------------------------------------------------------
# SQL layer. Every writer takes ``conn``: given one, the caller owns the transaction;
# without one, a private connection is opened with autocommit OFF (get_conn() is
# AUTOCOMMIT, on which rollback is a no-op) and committed or rolled back here.
# --------------------------------------------------------------------------
from src.services.agent_actions import record_action_or_fail  # noqa: E402
from src.services.db import get_conn  # noqa: E402
from src.services.lifecycle import IllegalTransition, refusal  # noqa: E402

_INSERT = """
INSERT INTO proc.bp_value_outcome
    (source_type, source_id, outcome_type, amount, currency, amount_gbp, fx_rate, fx_as_of,
     evidence_ref, note, supersedes_id, recorded_by, valid_from)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, coalesce(%s::date, current_date))
RETURNING outcome_id
"""
_HISTORY = """
SELECT outcome_id, outcome_type, amount, currency, amount_gbp, evidence_ref, note,
       supersedes_id, recorded_by, valid_from, recorded_at
  FROM proc.bp_value_outcome
 WHERE source_type = %s AND source_id = %s
 ORDER BY recorded_at, outcome_id
"""


def _in_tx(conn, fn):
    if conn is not None:
        return fn(conn)
    with get_conn() as own:
        own.autocommit = False
        try:
            result = fn(own)
            own.commit()
            return result
        except Exception:
            own.rollback()
            raise


def _rows(cur, sql, params) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _rates() -> Optional[dict]:
    from src.services import value_summary_service
    return value_summary_service._get_rates()


def _write(cur, conn, *, source_type, source_id, outcome_type, amount, currency, actor,
           evidence_ref=None, note=None, valid_from=None, supersedes_id=None) -> dict:
    fx = (convert_to_gbp(amount, currency, _rates()) if amount is not None and currency
          else {"amount_gbp": None, "fx_rate": None, "fx_as_of": None})
    cur.execute(_INSERT, (source_type, str(source_id), outcome_type, amount, currency,
                          fx["amount_gbp"], fx["fx_rate"], fx["fx_as_of"],
                          (evidence_ref or None), (note or None), supersedes_id, actor,
                          valid_from))
    outcome_id = cur.fetchone()[0]
    record_action_or_fail(
        phase="value", action_type="value.outcome_recorded", conn=conn,
        doc_type=source_type, doc_pk=str(source_id), agent=actor, status=outcome_type,
        summary=f"{outcome_type} {amount or ''} {currency or ''}".strip(),
        details={"outcome_id": outcome_id, "amount_gbp": fx["amount_gbp"],
                 "supersedes_id": supersedes_id})
    return {"outcome_id": outcome_id, "state": outcome_type, "amount_gbp": fx["amount_gbp"]}


def _lock_finding(cur, discrepancy_id: int) -> dict:
    rows = _rows(cur, "SELECT discrepancy_id, status, issue_type FROM "
                      "proc.bp_extraction_discrepancy WHERE discrepancy_id = %s FOR UPDATE",
                 (int(discrepancy_id),))
    if not rows:
        raise LedgerError("not_found", f"finding {discrepancy_id} does not exist")
    return rows[0]


def _money_types() -> tuple:
    from src.services import value_summary_service
    return value_summary_service.DISCREPANCY_VALUE_TYPES


def record_finding_outcome(discrepancy_id: int, outcome: str, amount, currency, *,
                           actor: str, valid_from: Optional[str] = None,
                           note: Optional[str] = None, conn=None) -> dict:
    if outcome not in FINDING_OPEN_OUTCOMES:
        raise LedgerError("invalid_outcome", f"{outcome!r} cannot close a finding")

    def run(c):
        cur = c.cursor()
        row = _lock_finding(cur, discrepancy_id)
        if row["status"] != "open":
            raise LedgerError("finding_already_moved",
                              f"finding {discrepancy_id} is already {row['status']}")
        if outcome != "accepted" and row["issue_type"] not in _money_types():
            raise LedgerError("not_a_money_finding",
                              f"a {row['issue_type']} finding carries no recoverable amount")
        clean_amount, clean_ccy = (None, None) if outcome == "accepted" else \
            validate_outcome(outcome, amount, currency, None)
        try:
            cur.execute("UPDATE proc.bp_extraction_discrepancy SET status = 'resolved', "
                        "resolved_by = %s, resolved_at = now() WHERE discrepancy_id = %s",
                        (actor, int(discrepancy_id)))
        except Exception as exc:
            reason = refusal(exc)
            if reason:
                raise LedgerError("finding_already_moved", reason) from exc
            raise
        if outcome == "accepted":
            record_action_or_fail(phase="value", action_type="value.outcome_recorded",
                                  conn=c, doc_type="finding", doc_pk=str(discrepancy_id),
                                  agent=actor, status="accepted",
                                  summary="charge accepted; no money moved")
            return {"outcome_id": None, "state": "accepted", "amount_gbp": None}
        return _write(cur, c, source_type="finding", source_id=discrepancy_id,
                      outcome_type=outcome, amount=clean_amount, currency=clean_ccy,
                      actor=actor, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def settle_claim(discrepancy_id: int, outcome: str, amount=None, currency=None, *,
                 actor: str, evidence_ref: Optional[str] = None,
                 valid_from: Optional[str] = None, note: Optional[str] = None,
                 conn=None) -> dict:
    if outcome not in SETTLE_OUTCOMES:
        raise LedgerError("invalid_outcome", f"{outcome!r} cannot settle a claim")

    def run(c):
        cur = c.cursor()
        _lock_finding(cur, discrepancy_id)          # serialises settles on this finding
        state = current_state(_rows(cur, _HISTORY, ("finding", str(discrepancy_id))))
        if not state or state["outcome_type"] != "claimed":
            raise LedgerError("no_open_claim", f"finding {discrepancy_id} has no open claim")
        clean_amount, clean_ccy = validate_outcome(outcome, amount, currency, evidence_ref)
        return _write(cur, c, source_type="finding", source_id=discrepancy_id,
                      outcome_type=outcome, amount=clean_amount, currency=clean_ccy,
                      actor=actor, evidence_ref=evidence_ref, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def realise_opportunity(opportunity_id: str, amount, currency, *, actor: str,
                        valid_from: Optional[str] = None, evidence_ref: Optional[str] = None,
                        note: Optional[str] = None, conn=None) -> dict:
    from src.services.opportunity_store import set_stage
    clean_amount, clean_ccy = validate_outcome("realised_saving", amount, currency, None)

    def run(c):
        cur = c.cursor()
        cur.execute("SELECT stage FROM proc.bp_opportunity WHERE opportunity_id = %s FOR UPDATE",
                    (str(opportunity_id),))
        found = cur.fetchone()
        if not found:
            raise LedgerError("not_found", f"opportunity {opportunity_id} does not exist")
        try:
            set_stage(str(opportunity_id), "realised", conn=c)
        except IllegalTransition as exc:
            raise LedgerError("finding_already_moved", str(exc)) from exc
        return _write(cur, c, source_type="opportunity", source_id=opportunity_id,
                      outcome_type="realised_saving", amount=clean_amount, currency=clean_ccy,
                      actor=actor, evidence_ref=evidence_ref, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def correct_outcome(outcome_id: int, amount, currency, *, actor: str, note: str,
                    evidence_ref: Optional[str] = None, conn=None) -> dict:
    if not str(note or "").strip():
        raise LedgerError("note_required", "say why the figure is being corrected")

    def run(c):
        cur = c.cursor()
        rows = _rows(cur, "SELECT * FROM proc.bp_value_outcome WHERE outcome_id = %s",
                     (int(outcome_id),))
        if not rows:
            raise LedgerError("not_found", f"outcome {outcome_id} does not exist")
        old = rows[0]
        cur.execute("SELECT 1 FROM proc.bp_value_outcome WHERE supersedes_id = %s",
                    (int(outcome_id),))
        if cur.fetchone():
            raise LedgerError("already_corrected", f"outcome {outcome_id} was already corrected")
        ev = evidence_ref if evidence_ref is not None else old["evidence_ref"]
        clean_amount, clean_ccy = validate_outcome(old["outcome_type"], amount, currency, ev)
        return _write(cur, c, source_type=old["source_type"], source_id=old["source_id"],
                      outcome_type=old["outcome_type"], amount=clean_amount,
                      currency=clean_ccy, actor=actor, evidence_ref=ev, note=note,
                      valid_from=old["valid_from"].isoformat(), supersedes_id=int(outcome_id))

    return _in_tx(conn, run)


def finding_outcomes(discrepancy_id: int, conn=None) -> dict:
    from src.services import value_summary_service as vss

    def run(c):
        cur = c.cursor()
        history = _rows(cur, _HISTORY, ("finding", str(discrepancy_id)))
        state = current_state(history)
        rows = _rows(cur, """
            SELECT e.issue_type, e.raw_value, e.expected_value, e.computed_value, i.currency,
                   (SELECT r.exposure_gbp FROM proc.bp_triage_finding m
                      JOIN proc.bp_triage_result r ON r.finding_id = m.finding_id
                      JOIN proc.bp_triage_run u ON u.run_id = r.run_id
                     WHERE m.mirror_id = e.discrepancy_id
                     ORDER BY u.started_at DESC LIMIT 1) AS exposure_gbp
              FROM proc.bp_extraction_discrepancy e
              LEFT JOIN proc.bp_invoice_trgt i
                     ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
             WHERE e.discrepancy_id = %s""", (int(discrepancy_id),))
        prefill = {"amount": None, "currency": None, "is_money": False}
        if rows:
            r = rows[0]
            prefill["is_money"] = r["issue_type"] in vss.DISCREPANCY_VALUE_TYPES
            if r.get("exposure_gbp") is not None:
                prefill.update(amount=f"{Decimal(r['exposure_gbp']):.2f}", currency="GBP")
            else:
                delta = vss.discrepancy_delta(r)
                if delta is not None:
                    prefill.update(amount=f"{delta:.2f}", currency=r.get("currency"))
        return {"state": state["outcome_type"] if state else None,
                "history": history, "prefill": prefill}

    return _in_tx(conn, run)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_value_ledger_live.py tests/services/test_value_ledger.py -q -p no:randomly`
Expected: 18 passed.

- [ ] **Step 5: Confirm nothing was left behind**

Run: `PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -At -c "select count(*) from proc.bp_value_outcome where recorded_by='pytest-value-ledger'"`
Expected: `0`.

- [ ] **Step 6: Commit**

```bash
git add src/services/value_ledger.py tests/services/test_value_ledger_live.py
git commit -o src/services/value_ledger.py tests/services/test_value_ledger_live.py \
  -m "feat(value-ledger): claim, settle, realise and correct in one transaction each

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Value Found reads the ledger and counts triage money

**Files:**
- Modify: `src/services/value_summary_service.py` (constants at `:31-35`, `classify_discrepancy` `:65-104`, `summarise` `:183-205`, loaders `:212-257`, `build_value_summary` `:317-375`)
- Test: `tests/services/test_value_summary_service.py` (extend); `tests/services/test_value_summary_ledger_live.py` (create)

**Interfaces:**
- Consumes: `value_ledger.current_state`, `value_ledger.SETTLED_STATES`.
- Produces:
  - `DISCREPANCY_VALUE_TYPES` now also includes `quantity_invoiced_above_po`, `invoices_exceed_po_total` and `unit_price_differs_from_po`.
  - `TRIAGE_VALUE_TYPES` (those three).
  - `apply_ledger(findings: list[dict], ledger_rows: list[dict]) -> list[dict]` sets per-finding `ledger_state`, `claimed_gbp`, `recovered_gbp`, `avoided_gbp`, `realised_gbp`, `settled_at`, `claimed_at` and `claim` (`{"amount", "currency", "amount_gbp"}` for a claimed finding, else None).
  - `ledger_totals(ledger_rows: list[dict]) -> dict` returns `avoided_gbp`, `recovered_gbp`, `realised_gbp`, `saved_gbp`, `claimed_open_gbp` and `by_month` (a list of `{"month": "YYYY-MM", "avoided_gbp", "recovered_gbp", "realised_gbp"}`).
  - `supersede_lines_under_overbilled_po(findings) -> list[dict]`.
  - `build_value_summary()` response gains `avoided_gbp`, `realised_gbp`, `saved_gbp`, `claimed_open_gbp`, `in_play_gbp`, `by_month`. `recovered_gbp` now means ledger-recovered.

- [ ] **Step 1: Write the failing unit tests** (append to `tests/services/test_value_summary_service.py`)

```python
from datetime import date, datetime, timezone

from src.services import value_summary_service as vss


def _led(source_id, outcome_type, amount_gbp, *, oid, supersedes=None, source_type="finding",
         valid_from=date(2026, 9, 10), recorded_at=None, amount=None, currency="GBP"):
    return {"outcome_id": oid, "source_type": source_type, "source_id": str(source_id),
            "outcome_type": outcome_type, "amount": amount if amount is not None else amount_gbp,
            "currency": currency, "amount_gbp": amount_gbp, "supersedes_id": supersedes,
            "valid_from": valid_from,
            "recorded_at": recorded_at or datetime(2026, 9, 10, oid, tzinfo=timezone.utc)}


def _disc(did, amount, status="resolved", issue_type="duplicate_invoice", deal="D1", doc=None):
    return {"id": f"disc:{did}", "source": "discrepancy", "tier": "verified",
            "amount_gbp": amount, "status": status, "deal_id": deal,
            "doc_pk": doc or f"INV{did}", "issue_type": issue_type, "superseded_by": None,
            "recovered_gbp": None}


def test_partial_recovery_counts_the_recovered_figure():
    rows = [_led(7, "claimed", 500, oid=1), _led(7, "recovered", 320, oid=2)]
    t = vss.ledger_totals(rows)
    assert t["recovered_gbp"] == 320.0 and t["claimed_open_gbp"] == 0.0
    f = vss.apply_ledger([_disc(7, 500)], rows)[0]
    assert f["ledger_state"] == "recovered" and f["claim"] is None


def test_open_claim_is_in_progress_not_saved():
    t = vss.ledger_totals([_led(8, "claimed", 200, oid=1)])
    assert t["claimed_open_gbp"] == 200.0 and t["saved_gbp"] == 0.0


def test_saved_is_avoided_plus_recovered_plus_realised_and_split_by_month():
    rows = [_led(1, "avoided", 100, oid=1, valid_from=date(2026, 8, 3)),
            _led(2, "claimed", 50, oid=2), _led(2, "recovered", 50, oid=3),
            _led("OPP-1", "realised_saving", 25, oid=4, source_type="opportunity")]
    t = vss.ledger_totals(rows)
    assert (t["avoided_gbp"], t["recovered_gbp"], t["realised_gbp"], t["saved_gbp"]) == \
        (100.0, 50.0, 25.0, 175.0)
    assert t["by_month"] == [
        {"month": "2026-08", "avoided_gbp": 100.0, "recovered_gbp": 0.0, "realised_gbp": 0.0},
        {"month": "2026-09", "avoided_gbp": 0.0, "recovered_gbp": 50.0, "realised_gbp": 25.0}]


def test_a_correction_replaces_the_figure_it_supersedes():
    rows = [_led(1, "avoided", 100, oid=1), _led(1, "avoided", 90, oid=2, supersedes=1)]
    assert vss.ledger_totals(rows)["avoided_gbp"] == 90.0


def test_unconverted_outcome_counts_in_no_gbp_total():
    rows = [_led(1, "avoided", None, oid=1, amount=100, currency="NZD")]
    assert vss.ledger_totals(rows)["avoided_gbp"] == 0.0


def test_in_play_excludes_settled_findings():
    findings = vss.apply_ledger(
        [_disc(1, 100, status="open"), _disc(2, 200), _disc(3, 300)],
        [_led(2, "claimed", 200, oid=1), _led(3, "avoided", 300, oid=2)])
    assert vss.in_play_gbp(findings) == 300.0   # open 100 + claimed 200; avoided 300 is settled


def test_line_findings_under_an_overbilled_po_are_superseded():
    po = _disc(10, 20000, status="open", issue_type="invoices_exceed_po_total", doc="PO-1")
    po["po_id"] = "PO-1"
    line = _disc(11, 700, status="open", issue_type="quantity_invoiced_above_po", doc="INV-9")
    line["po_id"] = "PO-1"
    other = _disc(12, 50, status="open", issue_type="unit_price_differs_from_po", doc="INV-8")
    other["po_id"] = "PO-2"
    out = vss.supersede_lines_under_overbilled_po([po, line, other])
    assert line["superseded_by"] == "disc:10"
    assert other["superseded_by"] is None and po["superseded_by"] is None
    assert len(out) == 3                      # the drawer explains, never omits


def test_triage_row_takes_its_amount_from_exposure():
    row = {"discrepancy_id": 5, "issue_type": "quantity_invoiced_above_po", "status": "open",
           "raw_value": "340.17", "expected_value": "113.39", "computed_value": None,
           "exposure_gbp": 226.78, "currency": "USD", "doc_type": "invoice",
           "doc_pk_candidate": "INV5", "created_at": None, "resolved_at": None,
           "query_sent_at": None}
    f = vss.classify_discrepancy(row)
    assert f["amount_gbp"] == 226.78
    assert f["currency"] == "GBP"            # exposure is already sterling: no second FX
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./venv/bin/python -m pytest tests/services/test_value_summary_service.py -q -p no:randomly`
Expected: FAIL with `AttributeError: module ... has no attribute 'ledger_totals'` (and the triage test fails on the amount).

- [ ] **Step 3: Implement**

(a) Constants at `:31`:

```python
TRIAGE_VALUE_TYPES = ("quantity_invoiced_above_po", "invoices_exceed_po_total",
                      "unit_price_differs_from_po")
DISCREPANCY_VALUE_TYPES = ("amount_over_po", "line_amount_over_po", "duplicate_invoice",
                           *TRIAGE_VALUE_TYPES)
_PO_LEVEL_TYPES = ("invoices_exceed_po_total",)
```

(b) In `classify_discrepancy`:
- Replace the delta and recovered block. A triage mirror's figure is its exposure (GBP); the mirror leaves `computed_value` NULL on purpose (`triage/writer.py`), and its raw/expected values are not a currency basis.
- Recovered now comes only from the ledger (`apply_ledger`), so the legacy-column fallback is removed.

```python
    if row.get("issue_type") in TRIAGE_VALUE_TYPES:
        exposure = parse_amount(row.get("exposure_gbp"))
        if exposure is None or exposure <= 0:
            return None
        delta, currency = round(exposure, 2), "GBP"
    else:
        delta = discrepancy_delta(row)
        if delta is None:
            return None
        currency = row.get("currency") or None
```

Then:
- In the returned dict, set `"recovered_gbp": None`, `"currency": currency`, `"issue_type": row.get("issue_type")` and `"po_id": row.get("po_id")`.
- Delete the old `recovered = ...` lines (`:73-77`).

(c) `_DISCREPANCY_SQL` (`:226`): add the triage exposure and the PO. The PO-level rows (`doc_type='purchase_order'`) join the PO for deal, currency and supplier.

```python
_DISCREPANCY_SQL = """
SELECT e.discrepancy_id, e.doc_type, e.doc_pk_candidate, e.field_name, e.raw_value,
       e.expected_value, e.computed_value, e.issue_type, e.status, e.notes, e.created_at,
       e.query_sent_at, e.resolved_at,
       coalesce(i.deal_id, p.deal_id) AS deal_id,
       coalesce(s.supplier_name, ps.supplier_name) AS supplier_name,
       coalesce(i.currency, p.currency) AS currency,
       CASE WHEN e.doc_type = 'purchase_order' THEN e.doc_pk_candidate ELSE i.po_id END AS po_id,
       tx.exposure_gbp
  FROM proc.bp_extraction_discrepancy e
  LEFT JOIN proc.bp_invoice_trgt i
         ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
  LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
  LEFT JOIN proc.bp_purchase_order_trgt p
         ON e.doc_type = 'purchase_order' AND p.po_id = e.doc_pk_candidate
  LEFT JOIN proc.bp_supplier ps ON ps.supplier_id = p.supplier_id
  LEFT JOIN LATERAL (
        SELECT r.exposure_gbp
          FROM proc.bp_triage_finding m
          JOIN proc.bp_triage_result r ON r.finding_id = m.finding_id
          JOIN proc.bp_triage_run u ON u.run_id = r.run_id
         WHERE m.mirror_id = e.discrepancy_id
         ORDER BY u.started_at DESC
         LIMIT 1) tx ON true
 WHERE e.issue_type IN %s
"""

_LEDGER_SQL = """
SELECT outcome_id, source_type, source_id, outcome_type, amount, currency, amount_gbp,
       supersedes_id, valid_from, recorded_at
  FROM proc.bp_value_outcome
"""


def _load_ledger(cur) -> list[dict]:
    return _rows(cur, _LEDGER_SQL)
```

(d) New pure functions (place after `dedupe`):

```python
from src.services.value_ledger import SETTLED_STATES, current_state  # top of file

_KEY = {"finding": "disc", "opportunity": "opp"}


def _by_source(ledger_rows: list[dict]) -> dict:
    grouped: dict[str, list] = {}
    for r in ledger_rows:
        grouped.setdefault(f"{_KEY[r['source_type']]}:{r['source_id']}", []).append(r)
    return grouped


def _gbp(v) -> float:
    return round(float(v), 2) if v is not None else 0.0


def ledger_totals(ledger_rows: list[dict]) -> dict:
    """Saved money from each source's CURRENT state (a correction replaces what it
    supersedes). An unconvertible row (amount_gbp NULL) counts in no GBP total."""
    totals = {"avoided_gbp": 0.0, "recovered_gbp": 0.0, "realised_gbp": 0.0,
              "claimed_open_gbp": 0.0}
    months: dict[str, dict] = {}
    field = {"avoided": "avoided_gbp", "recovered": "recovered_gbp",
             "realised_saving": "realised_gbp"}
    for rows in _by_source(ledger_rows).values():
        state = current_state(rows)
        if state is None:
            continue
        kind = state["outcome_type"]
        if kind == "claimed":
            totals["claimed_open_gbp"] += _gbp(state["amount_gbp"])
        elif kind in field:
            totals[field[kind]] += _gbp(state["amount_gbp"])
            m = months.setdefault(state["valid_from"].strftime("%Y-%m"),
                                  {"avoided_gbp": 0.0, "recovered_gbp": 0.0, "realised_gbp": 0.0})
            m[field[kind]] = round(m[field[kind]] + _gbp(state["amount_gbp"]), 2)
    totals = {k: round(v, 2) for k, v in totals.items()}
    totals["saved_gbp"] = round(totals["avoided_gbp"] + totals["recovered_gbp"]
                                + totals["realised_gbp"], 2)
    totals["by_month"] = [{"month": k, **v} for k, v in sorted(months.items())]
    return totals


def apply_ledger(findings: list[dict], ledger_rows: list[dict]) -> list[dict]:
    grouped = _by_source(ledger_rows)
    for f in findings:
        rows = grouped.get(f["id"], [])
        state = current_state(rows)
        kind = state["outcome_type"] if state else None
        f["ledger_state"] = kind
        f["claim"] = ({"amount": str(state["amount"]), "currency": state["currency"],
                       "amount_gbp": _gbp(state["amount_gbp"]) if state["amount_gbp"] is not None else None}
                      if kind == "claimed" else None)
        claimed = [r for r in rows if r["outcome_type"] == "claimed"]
        f["claimed_at"] = claimed[0]["recorded_at"].isoformat() if claimed else None
        f["recovered_gbp"] = _gbp(state["amount_gbp"]) if kind == "recovered" and state["amount_gbp"] is not None else None
        f["avoided_gbp"] = _gbp(state["amount_gbp"]) if kind == "avoided" and state["amount_gbp"] is not None else None
        f["realised_gbp"] = _gbp(state["amount_gbp"]) if kind == "realised_saving" and state["amount_gbp"] is not None else None
        f["settled_at"] = (state["valid_from"].isoformat()
                           if kind in SETTLED_STATES else None)
    return findings


def in_play_gbp(findings: list[dict]) -> float:
    """Money still to act on: open findings, live opportunities, and claims not yet settled."""
    total = 0.0
    for f in findings:
        if f.get("superseded_by") or f.get("amount_gbp") is None:
            continue
        if f.get("ledger_state") in SETTLED_STATES:
            continue
        live = (f.get("ledger_state") == "claimed"
                or (f["source"] == "discrepancy" and f.get("status") == "open")
                or (f["source"] == "opportunity" and f.get("status") in ("identified", "negotiation", "agreed")))
        if live:
            total += f["amount_gbp"]
    return round(total, 2)


def supersede_lines_under_overbilled_po(findings: list[dict]) -> list[dict]:
    """A PO whose invoices exceed its total already counts the overbilled money once; the
    line findings on invoices against that PO are the same money, seen line by line."""
    po_level = {f["po_id"]: f["id"] for f in findings
                if f.get("issue_type") in _PO_LEVEL_TYPES and f.get("po_id")
                and f.get("superseded_by") is None}
    for f in findings:
        if (f.get("issue_type") in TRIAGE_VALUE_TYPES and f.get("issue_type") not in _PO_LEVEL_TYPES
                and f.get("po_id") in po_level and f.get("superseded_by") is None):
            f["superseded_by"] = po_level[f["po_id"]]
    return findings
```

(e) `classify_opportunity`: set `"recovered_gbp": None` and delete the `recovered = ...` line (`:127`). Change the guard to `if amount <= 0: return None`.

(f) `summarise`: remove the `"recovered_gbp"` line (the ledger now owns it).

(g) `build_value_summary`:
- Inside the `with` block, after the opportunity source, add:

```python
        try:
            ledger_rows = _load_ledger(cur)
            sources["ledger"] = "ok"
        except Exception:
            log.exception("value_summary_service: ledger source failed")
            ledger_rows, sources["ledger"] = [], "unavailable"
```

- After the loaders, replace `findings = dedupe(findings); summary = summarise(findings)` with:

```python
    findings = supersede_lines_under_overbilled_po(dedupe(findings))
    findings = apply_ledger(findings, ledger_rows)
    summary = summarise(findings)
    summary.update(ledger_totals(ledger_rows))
    summary["in_play_gbp"] = in_play_gbp(findings)
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/services/test_value_summary_service.py tests/api/test_value_summary_router.py -q -p no:randomly`
Expected: all pass. Existing tests that asserted `recovered_gbp` from the legacy columns must be updated to feed ledger rows instead. Make those edits in this step and name each changed test in the commit message.

- [ ] **Step 5: Write and run a live read test**

`tests/services/test_value_summary_ledger_live.py`:

```python
"""build_value_summary reads a real ledger row, inside a rolled-back transaction."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def test_a_recovered_claim_reaches_the_summary(conn):
    from src.services import value_ledger as vl, value_summary_service as vss
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice','probe',%s,%s,'900.00','+900.00','duplicate_invoice','warning','open',false) "
        "RETURNING discrepancy_id", (f"PROBE-{uuid.uuid4().hex[:8]}", f"p_{uuid.uuid4().hex[:8]}"))
    did = cur.fetchone()[0]
    before = vss.build_value_summary(conn=conn)
    vl.record_finding_outcome(did, "claimed", "900", "GBP", actor="pytest-value-ledger", conn=conn)
    vl.settle_claim(did, "recovered", "900", "GBP", actor="pytest-value-ledger",
                    evidence_ref="CN-PROBE", conn=conn)
    after = vss.build_value_summary(conn=conn)
    assert round(after["recovered_gbp"] - before["recovered_gbp"], 2) == 900.0
    assert after["sources"]["ledger"] == "ok"
```

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_value_summary_ledger_live.py -q -p no:randomly`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/services/test_value_summary_ledger_live.py
git commit -o src/services/value_summary_service.py tests/services/test_value_summary_service.py tests/services/test_value_summary_ledger_live.py \
  -m "feat(value-found): recovered value from the ledger; triage money counted once

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The `/value` routes

**Files:**
- Create: `src/api/routers/value_ledger.py`
- Modify: `src/api/main.py` (import beside `:70`; add to the router list beside `:588`)
- Test: `tests/api/test_value_ledger_router.py`

**Interfaces:**
- Consumes: the Task 3 service functions and `LedgerError.code`.
- Produces HTTP routes:
  - `POST /value/findings/{id}/outcome`
  - `POST /value/findings/{id}/settle`
  - `POST /value/opportunities/{id}/realise`
  - `POST /value/outcomes/{id}/correct`
  - `GET /value/findings/{id}/outcomes`

  Error bodies are `{"detail": {"error": <code>, "message": <str>}}`.

- [ ] **Step 1: Write the failing router tests**

```python
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from src.api.main import app
    from src.api.routers import value_ledger as router_mod
    app.dependency_overrides[router_mod.require_user] = lambda: SimpleNamespace(subject="buyer-1")
    yield TestClient(app), router_mod
    app.dependency_overrides.clear()


def test_outcome_requires_a_user():
    from src.api.main import app
    r = TestClient(app).post("/value/findings/1/outcome", json={"outcome": "claimed"})
    assert r.status_code in (401, 403, 503)


def test_outcome_passes_the_token_actor_not_the_body(client, monkeypatch):
    c, mod = client
    seen = {}
    monkeypatch.setattr(mod.value_ledger, "record_finding_outcome",
                        lambda did, outcome, amount, currency, **kw: seen.update(kw, did=did) or
                        {"outcome_id": 9, "state": outcome, "amount_gbp": 10})
    r = c.post("/value/findings/42/outcome",
               json={"outcome": "claimed", "amount": "10", "currency": "GBP", "actor": "forged"})
    assert r.status_code == 200 and r.json()["state"] == "claimed"
    assert seen["actor"] == "buyer-1" and seen["did"] == 42


@pytest.mark.parametrize("code,status", [("finding_already_moved", 409), ("no_open_claim", 409),
                                         ("already_corrected", 409), ("not_found", 404),
                                         ("evidence_required", 422), ("invalid_amount", 422),
                                         ("not_a_money_finding", 422)])
def test_ledger_errors_map_to_http(client, monkeypatch, code, status):
    c, mod = client

    def _raise(*a, **kw):
        raise mod.value_ledger.LedgerError(code, "nope")
    monkeypatch.setattr(mod.value_ledger, "settle_claim", _raise)
    r = c.post("/value/findings/1/settle", json={"outcome": "recovered"})
    assert r.status_code == status
    assert r.json()["detail"]["error"] == code


def test_history_is_readable(client, monkeypatch):
    c, mod = client
    monkeypatch.setattr(mod.value_ledger, "finding_outcomes", lambda did: {
        "state": None, "history": [], "prefill": {"amount": "5.00", "currency": "GBP", "is_money": True}})
    r = c.get("/value/findings/3/outcomes")
    assert r.status_code == 200 and r.json()["prefill"]["amount"] == "5.00"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./venv/bin/python -m pytest tests/api/test_value_ledger_router.py -q -p no:randomly`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.routers.value_ledger'`.

- [ ] **Step 3: Implement the router**

```python
"""/value — recording the money a finding or opportunity actually produced.
Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md §4"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.auth import require_user
from src.services import value_ledger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/value", tags=["Value ledger"])

_STATUS = {"finding_already_moved": 409, "no_open_claim": 409, "already_corrected": 409,
           "not_found": 404}


def _actor(principal: Any) -> str:
    """Who recorded it — from the token, never from the body."""
    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        raise HTTPException(status_code=401, detail="no authenticated subject")
    return subject


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except value_ledger.LedgerError as exc:
        raise HTTPException(status_code=_STATUS.get(exc.code, 422),
                            detail={"error": exc.code, "message": str(exc)})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("value ledger write failed")
        raise HTTPException(status_code=500, detail={"error": "write_failed", "message": str(exc)})


class OutcomeBody(BaseModel):
    outcome: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    valid_from: Optional[str] = None
    note: Optional[str] = None


class SettleBody(BaseModel):
    outcome: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    evidence_ref: Optional[str] = None
    valid_from: Optional[str] = None
    note: Optional[str] = None


class RealiseBody(BaseModel):
    amount: str
    currency: str
    valid_from: Optional[str] = None
    evidence_ref: Optional[str] = None
    note: Optional[str] = None


class CorrectBody(BaseModel):
    amount: str
    currency: str
    note: str
    evidence_ref: Optional[str] = None


@router.post("/findings/{discrepancy_id}/outcome", summary="Close a money finding with what happened to it")
def post_outcome(discrepancy_id: int, body: OutcomeBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.record_finding_outcome, discrepancy_id, body.outcome, body.amount,
                 body.currency, actor=_actor(principal), valid_from=body.valid_from, note=body.note)


@router.post("/findings/{discrepancy_id}/settle", summary="Credit received, or the claim dropped")
def post_settle(discrepancy_id: int, body: SettleBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.settle_claim, discrepancy_id, body.outcome, body.amount,
                 body.currency, actor=_actor(principal), evidence_ref=body.evidence_ref,
                 valid_from=body.valid_from, note=body.note)


@router.post("/opportunities/{opportunity_id}/realise", summary="Mark an opportunity's saving realised")
def post_realise(opportunity_id: str, body: RealiseBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.realise_opportunity, opportunity_id, body.amount, body.currency,
                 actor=_actor(principal), valid_from=body.valid_from,
                 evidence_ref=body.evidence_ref, note=body.note)


@router.post("/outcomes/{outcome_id}/correct", summary="Correct a recorded figure (a new row)")
def post_correct(outcome_id: int, body: CorrectBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.correct_outcome, outcome_id, body.amount, body.currency,
                 actor=_actor(principal), note=body.note, evidence_ref=body.evidence_ref)


@router.get("/findings/{discrepancy_id}/outcomes", summary="A finding's outcome history and suggested figure")
def get_outcomes(discrepancy_id: int) -> dict:
    return _call(value_ledger.finding_outcomes, discrepancy_id)
```

In `src/api/main.py`:
- Next to `from api.routers import value_summary as value_summary_router` (`:70`), add `from api.routers import value_ledger as value_ledger_router`.
- In the router list next to `value_summary_router.router,` (`:588`), add `value_ledger_router.router,`.
- Read the comment block at `:534` first. If that list is the authenticated-routers list, the new router belongs in it.

**Note on import paths:** `main.py` imports `api.routers...` while the tests import `src.api.routers...`. If `dependency_overrides[router_mod.require_user]` does not take effect, override `src.api.auth.require_user` and `api.auth.require_user` both. `tests/api/test_decisions_email_endpoints.py:97` shows the working form.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/api/test_value_ledger_router.py -q -p no:randomly`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/value_ledger.py tests/api/test_value_ledger_router.py
git commit -o src/api/routers/value_ledger.py src/api/main.py tests/api/test_value_ledger_router.py \
  -m "feat(value-ledger): authenticated /value routes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Dashboard, executive report and digest read the ledger; stale digest test fixed

**Files:**
- Modify: `src/services/opportunity_dashboard.py:60-120`
- Modify: `src/services/rga/builders/exec_procurement_summary.py:71-78,203-233`
- Modify: `src/services/value_digest.py:94-107`
- Test: `tests/services/test_opportunity_dashboard.py`, `tests/services/test_value_digest.py`, `tests/services/rga/test_exec_summary_ledger.py`

**Interfaces:**
- Consumes: `proc.bp_value_outcome` (Task 1). Findings' `settled_at` and `recovered_gbp` come from `build_value_summary` (Task 4).
- Produces:
  - The dashboard's `realised` and `realisedChange` come from `realised_saving` ledger rows by `valid_from`.
  - The exec summary gains the fact `"Saved (GBP)"` with derivation `exec_summary.value_saved`. `"Realised savings (GBP)"` keeps its derivation but reads the ledger.
  - The digest's "recovered this week" uses `settled_at`.

- [ ] **Step 1: Fix the stale digest test and add the ledger-date test** (in `tests/services/test_value_digest.py`)

Replace the body of `test_sends_to_every_configured_recipient` (`:155-163`) with:

```python
def test_sends_to_every_configured_recipient(monkeypatch):
    # Since ee01b12 the digest sends only as a named identity, to in-domain recipients,
    # through guardrail.authorize. The test sets all three.
    from types import SimpleNamespace
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com, finance@example.com")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@example.com")
    monkeypatch.setenv("SES_DEFAULT_SENDER", "noreply@example.com")
    monkeypatch.setattr(vd.guardrail, "authorize",
                        lambda *a, **kw: SimpleNamespace(allowed=True, reason=""))
    monkeypatch.setattr(vd, "_load_summary", lambda: _summary([_finding()]))
    sent = {}
    monkeypatch.setattr(vd, "_send_email", lambda **kw: sent.update(kw) or True)
    assert vd.run_weekly_digest() == 1
    assert sent["to"] == ["ap@example.com", "finance@example.com"]
    assert "Value found this week" in sent["subject"]


def test_recovered_this_week_uses_the_ledger_date():
    from datetime import datetime, timezone
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    old_find = _finding(found_at=_iso(40), age_days=40)
    old_find.update(recovered_gbp=320.0, settled_at="2026-09-23", resolved_at=_iso(40))
    digest = vd.compose_digest(_summary([old_find]), now)
    assert digest is not None and "£320" in digest["body"]
```

- [ ] **Step 2: Run the digest tests to see the new one fail**

Run: `./venv/bin/python -m pytest tests/services/test_value_digest.py -q -p no:randomly`
Expected: `test_sends_to_every_configured_recipient` PASSES (it was only stale). `test_recovered_this_week_uses_the_ledger_date` FAILS, because the digest keys on `resolved_at`. If `compose_digest`'s return shape is not `{"body": ...}`, read it at `value_digest.py:90-135` and assert on the field it really returns.

- [ ] **Step 3: Change the digest**

At `value_digest.py:95-96`:

```python
    recovered_this_week = [f for f in live
                           if f.get("recovered_gbp")
                           and _within(f.get("settled_at") or f.get("resolved_at"), since)]
```

`settled_at` is a date string (`YYYY-MM-DD`). Check that `_within` parses it. If it parses only full ISO timestamps, extend it to accept a date by treating it as midnight UTC, and add that case to the test.

- [ ] **Step 4: Write the failing dashboard and exec-summary tests**

In `tests/services/test_opportunity_dashboard.py`, add a test that monkeypatches the dashboard's SQL fetch helper (read `opportunity_dashboard.py:55-100` for its name) to return ledger-shaped aggregates. It should assert that `realised` renders the ledger sum. Concretely:

```python
def test_realised_comes_from_the_ledger():
    # The realised figures must be read from recorded outcomes, not the legacy column.
    # The live proof is the Task 10 demo; this pins the source.
    from src.services import opportunity_dashboard as od
    src = open(od.__file__).read()
    assert "bp_value_outcome" in src
    assert "sum(realised_savings_gbp)" not in src
```

(A source-level assertion is acceptable here. The live proof is the Task 10 demo. Also add a fake-DB test if the module already has a query seam: check `tests/services/test_opportunity_dashboard.py` for how it currently injects rows and follow that pattern with ledger rows.)

`tests/services/rga/test_exec_summary_ledger.py`:

```python
from datetime import date


def test_saved_fact_reads_the_ledger(monkeypatch):
    from src.services.rga.builders import exec_procurement_summary as ex
    src = open(ex.__file__).read()
    assert "proc.bp_value_outcome" in src
    assert "exec_summary.value_saved" in src
```

Then read `tests/services/rga/test_exec_summary_live.py` for how the builder is invoked. Add one live test there, gated on `PROCWISE_TEST_LIVE_DB`, that:
- inserts an `avoided` ledger row inside a rolled-back transaction dated in the period;
- asserts the pack contains a `Saved (GBP)` fact whose value includes it.

If the builder cannot take an injected connection, record that in the task report and rely on the Task 10 demo instead. Do not add a connection seam just for the test.

- [ ] **Step 5: Implement the dashboard and exec-summary reads**

In `opportunity_dashboard.py`, replace each `sum(realised_savings_gbp) filter (where stage='realised' ...)` (`:68,80,81,112`) with a scalar subquery on the ledger's current `realised_saving` rows:

```python
_REALISED = """(SELECT coalesce(sum(o.amount_gbp), 0) FROM proc.bp_value_outcome o
  WHERE o.source_type = 'opportunity' AND o.outcome_type = 'realised_saving'
    AND NOT EXISTS (SELECT 1 FROM proc.bp_value_outcome s WHERE s.supersedes_id = o.outcome_id)
    {when})"""
```

- `:68` and `:112` use `_REALISED.format(when="")`.
- `cur_r` uses `when="AND date_trunc('month', o.valid_from) = date_trunc('month', now())"`.
- `prev_r` uses `when="AND date_trunc('month', o.valid_from) = date_trunc('month', now() - interval '1 month')"`.

In `exec_procurement_summary.py`:
- Replace `_OPPORTUNITIES` `realised_gbp` and `realised_n` with ledger subqueries over `valid_from BETWEEN %s AND %s`. Add a `_SAVED` query:

```python
_SAVED = """
SELECT o.outcome_type, coalesce(sum(o.amount_gbp), 0)::numeric, count(*)::int
  FROM proc.bp_value_outcome o
 WHERE o.outcome_type IN ('avoided', 'recovered', 'realised_saving')
   AND o.valid_from BETWEEN %s AND %s
   AND NOT EXISTS (SELECT 1 FROM proc.bp_value_outcome s WHERE s.supersedes_id = o.outcome_id)
 GROUP BY o.outcome_type
"""
```

- After the realised fact, add:

```python
    saved = {t: (v, n) for t, v, n in _fetch(_SAVED, (start, end))}
    saved_n = sum(n for _, n in saved.values())
    if saved_n:
        total = sum(v for v, _ in saved.values())
        fb.add(label="Saved (GBP)", value=Decimal(total),
               derivation="exec_summary.value_saved",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY,
               currency="GBP")
    else:
        fb.unmeasured(label="Saved (GBP)", derivation="exec_summary.value_saved",
                      reason="no money was recorded as stopped, recovered or realised "
                             "in this period")
```

- The realised `else` reason text (`:229-233`) currently says the corpus has 0 realised opportunities. Change it to: `"no opportunity saving was recorded as realised in this period"`.
- Check whether `fb.add` accepts a provenance or source argument naming `proc.bp_value_outcome`. If it does, pass it for both facts.

- [ ] **Step 6: Run the tests**

Run: `./venv/bin/python -m pytest tests/services/test_value_digest.py tests/services/test_value_digest_gate.py tests/services/test_opportunity_dashboard.py tests/services/rga -q -p no:randomly`
Expected: all pass, with no new failures against the 2026-09-25 baseline for these files.

- [ ] **Step 7: Commit**

```bash
git add tests/services/rga/test_exec_summary_ledger.py
git commit -o src/services/opportunity_dashboard.py src/services/rga/builders/exec_procurement_summary.py src/services/value_digest.py tests/services/test_value_digest.py tests/services/test_opportunity_dashboard.py tests/services/rga/test_exec_summary_ledger.py \
  -m "feat(value-ledger): dashboard, exec summary and digest read recorded outcomes

Also fixes test_sends_to_every_configured_recipient, stale since ee01b12.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: UI rules for recording outcomes (pure)

**Files (beyond_procwise_ui):**
- Create: `src/lib/valueOutcome.js`
- Test: `src/lib/valueOutcome.test.js`

**Interfaces:**
- Produces:
  - `MONEY_ISSUE_TYPES: Set<string>`, the same six types as backend `DISCREPANCY_VALUE_TYPES`.
  - `isMoneyFinding(issueType) -> boolean`
  - `OUTCOME_CHOICES`: `[{value:'avoided'}, {value:'claimed'}, {value:'accepted'}]`, each with `labelKey` and `fallback`.
  - `outcomePath(id)`, `settlePath(id)`, `realisePath(id)`, `historyPath(id)`
  - `buildOutcomeBody({outcome, amount, currency, validFrom, note}) -> object`
  - `buildSettleBody({outcome, amount, currency, evidenceRef, validFrom, note}) -> object`
  - `validateOutcomeForm({outcome, amount, currency, evidenceRef, noteRequired, note}) -> {ok:boolean, errors:{[field]:string}}`
  - `ledgerErrorMessage(err) -> string`
  - `claimLabel(finding) -> string`

- [ ] **Step 1: Write the failing tests**

```js
import { describe, it, expect } from 'vitest';
import {
  MONEY_ISSUE_TYPES, isMoneyFinding, buildOutcomeBody, buildSettleBody,
  validateOutcomeForm, ledgerErrorMessage, claimLabel, outcomePath, settlePath,
} from './valueOutcome';

describe('valueOutcome', () => {
  it('knows the same money issue types as the backend', () => {
    expect([...MONEY_ISSUE_TYPES].sort()).toEqual([
      'amount_over_po', 'duplicate_invoice', 'invoices_exceed_po_total', 'line_amount_over_po',
      'quantity_invoiced_above_po', 'unit_price_differs_from_po']);
    expect(isMoneyFinding('line_missing_amount')).toBe(false);
  });

  it('accepting the charge sends no amount', () => {
    expect(buildOutcomeBody({ outcome: 'accepted', amount: '10', currency: 'GBP' }))
      .toEqual({ outcome: 'accepted' });
  });

  it('a claim sends the amount as a string and the currency upper-cased', () => {
    expect(buildOutcomeBody({ outcome: 'claimed', amount: 1234.5, currency: 'gbp', validFrom: '2026-09-25', note: '' }))
      .toEqual({ outcome: 'claimed', amount: '1234.5', currency: 'GBP', valid_from: '2026-09-25' });
  });

  it('a dropped claim sends no money fields', () => {
    expect(buildSettleBody({ outcome: 'claim_dropped', amount: '5', note: 'supplier refused' }))
      .toEqual({ outcome: 'claim_dropped', note: 'supplier refused' });
  });

  it('recovered needs a credit note reference', () => {
    const r = validateOutcomeForm({ outcome: 'recovered', amount: '10', currency: 'GBP', evidenceRef: ' ' });
    expect(r.ok).toBe(false);
    expect(r.errors.evidenceRef).toBeTruthy();
  });

  it('amount must be a positive number', () => {
    expect(validateOutcomeForm({ outcome: 'claimed', amount: '0', currency: 'GBP' }).errors.amount).toBeTruthy();
    expect(validateOutcomeForm({ outcome: 'claimed', amount: 'ten', currency: 'GBP' }).errors.amount).toBeTruthy();
  });

  it('a note becomes required when the engine disagreed', () => {
    const r = validateOutcomeForm({ outcome: 'claimed', amount: '5', currency: 'GBP', noteRequired: true, note: '' });
    expect(r.errors.note).toBeTruthy();
  });

  it('turns a 409 into plain words', () => {
    const err = { response: { status: 409, data: { detail: { error: 'finding_already_moved' } } } };
    expect(ledgerErrorMessage(err)).toMatch(/already/i);
  });

  it('shows the native amount when a claim could not be converted', () => {
    expect(claimLabel({ claim: { amount: '100.00', currency: 'NZD', amount_gbp: null } }))
      .toMatch(/NZ\$100\.00/);
  });

  it('builds the backend paths', () => {
    expect(outcomePath(42)).toBe('/value/findings/42/outcome');
    expect(settlePath('disc:42')).toBe('/value/findings/42/settle');
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/muthu/PycharmProjects/beyond_procwise_ui && npx vitest run src/lib/valueOutcome.test.js`
Expected: FAIL with "Failed to resolve import './valueOutcome'".

- [ ] **Step 3: Implement**

```js
// Recording what happened to the money a finding found. Presentation-free: the panel and
// the Value Found drawer both import these rules, so the two cannot drift.
// Spec: BP_Backend docs/superpowers/specs/2026-09-25-value-ledger-design.md §6

import { formatCompactCurrency } from './format/currency';
import { tOr } from './i18n';
import { formatForeignAmount } from './valueFound';

// Mirrors BP_Backend value_summary_service.DISCREPANCY_VALUE_TYPES.
export const MONEY_ISSUE_TYPES = new Set([
  'amount_over_po', 'line_amount_over_po', 'duplicate_invoice',
  'quantity_invoiced_above_po', 'invoices_exceed_po_total', 'unit_price_differs_from_po',
]);
export const isMoneyFinding = (issueType) => MONEY_ISSUE_TYPES.has(String(issueType || ''));

export const OUTCOME_CHOICES = [
  { value: 'avoided', labelKey: 'vo.avoided', fallback: 'Stopped before payment' },
  { value: 'claimed', labelKey: 'vo.claimed', fallback: 'Claiming it back from the supplier' },
  { value: 'accepted', labelKey: 'vo.accepted', fallback: 'Accept the charge' },
];

const idOf = (id) => String(id).replace(/^disc:/, '');
export const outcomePath = (id) => `/value/findings/${idOf(id)}/outcome`;
export const settlePath = (id) => `/value/findings/${idOf(id)}/settle`;
export const historyPath = (id) => `/value/findings/${idOf(id)}/outcomes`;
export const realisePath = (id) => `/value/opportunities/${encodeURIComponent(String(id).replace(/^opp:/, ''))}/realise`;

const clean = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== ''));

export function buildOutcomeBody({ outcome, amount, currency, validFrom, note }) {
  if (outcome === 'accepted') return clean({ outcome, note });
  return clean({ outcome, amount: amount == null ? undefined : String(amount),
    currency: currency ? String(currency).toUpperCase() : undefined, valid_from: validFrom, note });
}

export function buildSettleBody({ outcome, amount, currency, evidenceRef, validFrom, note }) {
  if (outcome === 'claim_dropped') return clean({ outcome, note });
  return clean({ outcome, amount: amount == null ? undefined : String(amount),
    currency: currency ? String(currency).toUpperCase() : undefined,
    evidence_ref: evidenceRef ? String(evidenceRef).trim() : undefined, valid_from: validFrom, note });
}

export function validateOutcomeForm({ outcome, amount, currency, evidenceRef, noteRequired, note }) {
  const errors = {};
  const moneyless = outcome === 'accepted' || outcome === 'claim_dropped';
  if (!moneyless) {
    const n = Number(String(amount ?? '').trim());
    if (!Number.isFinite(n) || n <= 0) errors.amount = tOr('vo.errAmount', 'Enter an amount greater than zero');
    if (!/^[A-Za-z]{3}$/.test(String(currency || ''))) errors.currency = tOr('vo.errCurrency', 'Use a 3-letter currency code, such as GBP');
  }
  if (outcome === 'recovered' && !String(evidenceRef || '').trim()) {
    errors.evidenceRef = tOr('vo.errEvidence', 'Add the credit note or document reference');
  }
  if (noteRequired && !String(note || '').trim()) {
    errors.note = tOr('vo.errNote', 'The evidence points the other way. Say why you are going ahead.');
  }
  return { ok: Object.keys(errors).length === 0, errors };
}

const MESSAGES = {
  finding_already_moved: ['vo.errMoved', 'Someone has already settled this finding. The list has been refreshed.'],
  no_open_claim: ['vo.errNoClaim', 'This finding has no open claim to settle.'],
  evidence_required: ['vo.errEvidence', 'Add the credit note or document reference'],
  not_a_money_finding: ['vo.errNotMoney', 'This finding has no amount that can be recovered.'],
};

export function ledgerErrorMessage(err) {
  const code = err?.response?.data?.detail?.error;
  const [key, fallback] = MESSAGES[code] || ['vo.errSave', 'That could not be saved. Please try again.'];
  return tOr(key, fallback);
}

export function claimLabel(finding) {
  const c = finding?.claim;
  if (!c) return '';
  if (c.amount_gbp != null) return formatCompactCurrency(c.amount_gbp, { currency: 'GBP' });
  return `${formatForeignAmount(c.amount, c.currency)} (${tOr('vo.noRate', 'no GBP rate')})`;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/lib/valueOutcome.test.js`
Expected: 10 passed.

- [ ] **Step 5: Commit** (UI repo; stage only these files)

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/lib/valueOutcome.js src/lib/valueOutcome.test.js
git commit -m "feat(value-ledger): rules for recording what happened to found money

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The "What happened to this money?" panel, wired into the Action Centre and Home

**Files (beyond_procwise_ui):**
- Create: `src/components/value/ValueOutcomePanel.jsx`
- Create: `src/components/value/ValueOutcomePanel.test.jsx`
- Modify: `src/modules/SpendIQ/data/useSpendData.js` (`findingToAction`, around `:1822-1838`)
- Modify: `src/modules/SpendIQ/engine.js` (`ACTION_INTENT` `:3533`, `siqAction` `:3552`, `resolveFinding` `:3307`)
- Modify: `src/modules/SpendIQ/index.jsx` (bridge block near `:410-420` and the render tree)
- Modify: `src/modules/ProcurementHome/index.jsx` (`decideCard` `:2083-2105` and render tree)

**Interfaces:**
- Consumes: Task 7 helpers; backend `GET /value/findings/{id}/outcomes`, `POST /value/findings/{id}/outcome`, `POST /decisions/finding/{id}`.
- Produces:
  - `<ValueOutcomePanel findingId onClose onDone />`
  - `window.__SPENDIQ_OPEN_OUTCOME__(findingId)`
  - Queue items for money findings carry `dec[0] = 'Record outcome'` and `acts[0] = 'record_outcome'`.

- [ ] **Step 1: Write the failing component test**

```jsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import ValueOutcomePanel from './ValueOutcomePanel';

vi.mock('axios');

describe('ValueOutcomePanel', () => {
  beforeEach(() => {
    axios.get.mockResolvedValue({ data: { state: null, history: [],
      prefill: { amount: '226.78', currency: 'GBP', is_money: true } } });
    axios.post.mockReset();
  });

  it('pre-fills the finding\'s own figure', async () => {
    render(<ValueOutcomePanel findingId={42} onClose={() => {}} onDone={() => {}} />);
    expect(await screen.findByDisplayValue('226.78')).toBeTruthy();
  });

  it('checks with the decision engine, then records a claim', async () => {
    axios.post
      .mockResolvedValueOnce({ data: { decision: 'approve', rationale: 'ok' } })
      .mockResolvedValueOnce({ data: { outcome_id: 1, state: 'claimed' } });
    const onDone = vi.fn();
    render(<ValueOutcomePanel findingId={42} onClose={() => {}} onDone={onDone} />);
    fireEvent.click(await screen.findByLabelText(/Claiming it back/));
    fireEvent.click(screen.getByRole('button', { name: /Save/ }));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(axios.post.mock.calls[0][0]).toMatch(/\/decisions\/finding\/42$/);
    expect(axios.post.mock.calls[1][0]).toMatch(/\/value\/findings\/42\/outcome$/);
    expect(axios.post.mock.calls[1][1]).toMatchObject({ outcome: 'claimed', amount: '226.78', currency: 'GBP' });
  });

  it('asks for a reason when the engine disagrees, and sends it as the note', async () => {
    axios.post.mockResolvedValueOnce({ data: { decision: 'escalate', rationale: 'PO was amended',
      conflicts_with_request: { requested: 'approve' } } });
    render(<ValueOutcomePanel findingId={42} onClose={() => {}} onDone={() => {}} />);
    fireEvent.click(await screen.findByLabelText(/Claiming it back/));
    fireEvent.click(screen.getByRole('button', { name: /Save/ }));
    expect(await screen.findByText(/PO was amended/)).toBeTruthy();
    expect(axios.post).toHaveBeenCalledTimes(1);          // nothing written yet
  });
});
```

(Check `package.json` devDependencies for `@testing-library/react`. If it is absent, do not install it. Rewrite these three tests against exported pure handlers, i.e. export `submitOutcome({findingId, form, post})` from the component file and test that. Record which form was used.)

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/components/value/ValueOutcomePanel.test.jsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the panel**

```jsx
// "What happened to this money?" — one panel, mounted by both the SpendIQ Action Centre and
// Procurement Home, so recording an outcome looks and behaves the same wherever it starts.
// The decision engine is consulted first, as it is for every close: when it disagrees the
// buyer is not blocked, only asked to say why (the reason is stored as the ledger note).
import { useEffect, useState } from 'react';
import axios from 'axios';
import { tOr } from '../../lib/i18n';
import { AI_API } from '../../lib/api';
import {
  OUTCOME_CHOICES, buildOutcomeBody, historyPath, ledgerErrorMessage, outcomePath,
  validateOutcomeForm,
} from '../../lib/valueOutcome';

const INTENT = { avoided: 'approve', claimed: 'approve', accepted: 'dismiss' };
const today = () => new Date().toISOString().slice(0, 10);

export async function submitOutcome({ findingId, form, post, conflictAcknowledged }) {
  if (!conflictAcknowledged) {
    const d = (await post(`/decisions/finding/${findingId}`,
      { requested: INTENT[form.outcome], user_id: 'ui' })).data;
    if (d.conflicts_with_request) return { conflict: d };
  }
  const r = (await post(outcomePath(findingId), buildOutcomeBody(form))).data;
  return { saved: r };
}

export default function ValueOutcomePanel({ findingId, onClose, onDone }) {
  const [form, setForm] = useState({ outcome: 'claimed', amount: '', currency: 'GBP', validFrom: today(), note: '' });
  const [conflict, setConflict] = useState(null);
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState('');

  useEffect(() => {
    let live = true;
    axios.get(`${AI_API}${historyPath(findingId)}`).then(({ data }) => {
      if (!live) return;
      const p = data.prefill || {};
      setForm((f) => ({ ...f, amount: p.amount ?? '', currency: p.currency || 'GBP' }));
    }).catch(() => {});
    return () => { live = false; };
  }, [findingId]);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const post = (path, body) => axios.post(`${AI_API}${path}`, body);

  const save = async () => {
    const check = validateOutcomeForm({ ...form, noteRequired: !!conflict });
    setErrors(check.errors);
    if (!check.ok) return;
    setBusy(true); setFailure('');
    try {
      const r = await submitOutcome({ findingId, form, post, conflictAcknowledged: !!conflict });
      if (r.conflict) { setConflict(r.conflict); return; }
      onDone(r.saved);
    } catch (err) {
      setFailure(ledgerErrorMessage(err));
      if (err?.response?.status === 409) onDone(null);
    } finally { setBusy(false); }
  };

  const moneyless = form.outcome === 'accepted';
  return (
    <div className="vo-panel" role="dialog" aria-modal="true" aria-labelledby="vo-title">
      <h2 id="vo-title">{tOr('vo.title', 'What happened to this money?')}</h2>
      <fieldset>
        {OUTCOME_CHOICES.map((c) => (
          <label key={c.value}>
            <input type="radio" name="vo-outcome" id={`vo-${c.value}`} value={c.value}
              checked={form.outcome === c.value} onChange={set('outcome')} />
            {tOr(c.labelKey, c.fallback)}
          </label>
        ))}
      </fieldset>
      {!moneyless && (
        <div className="vo-row">
          <label htmlFor="vo-amount">{tOr('vo.amount', 'Amount')}</label>
          <input id="vo-amount" inputMode="decimal" value={form.amount} onChange={set('amount')} />
          <label htmlFor="vo-currency">{tOr('vo.currency', 'Currency')}</label>
          <input id="vo-currency" maxLength={3} value={form.currency} onChange={set('currency')} />
          <label htmlFor="vo-date">{tOr('vo.date', 'Date')}</label>
          <input id="vo-date" type="date" value={form.validFrom} onChange={set('validFrom')} />
          {errors.amount && <p className="vo-err">{errors.amount}</p>}
          {errors.currency && <p className="vo-err">{errors.currency}</p>}
        </div>
      )}
      {conflict && (
        <p className="vo-conflict">
          {tOr('vo.conflict', 'The evidence suggests {d}: {why}', { d: conflict.decision, why: conflict.rationale })}
        </p>
      )}
      <label htmlFor="vo-note">{conflict ? tOr('vo.noteWhy', 'Why are you going ahead?') : tOr('vo.note', 'Note (optional)')}</label>
      <textarea id="vo-note" value={form.note} onChange={set('note')} />
      {errors.note && <p className="vo-err">{errors.note}</p>}
      {failure && <p className="vo-err" role="alert">{failure}</p>}
      <div className="vo-actions">
        <button type="button" onClick={onClose} disabled={busy}>{tOr('vo.cancel', 'Cancel')}</button>
        <button type="button" onClick={save} disabled={busy}>{tOr('vo.save', 'Save')}</button>
      </div>
    </div>
  );
}
```

Before using the import `AI_API` from `'../../lib/api'`, check it: `grep -rn "export const AI_API" src/lib`. Import it from wherever `useHomeData.js` gets it. Styles: add `.vo-panel` rules beside the drawer's existing styles (find the stylesheet the drawer uses with `grep -rn "vf-" src --include=*.css | head`). Use that sheet's existing tokens only; no new colours.

- [ ] **Step 4: Route money findings to the panel**

`useSpendData.js`, in `findingToAction` (`:1827-1838`), before `const dec = ...`:

```js
  // A finding that found money is closed by saying what happened to that money. Its first
  // action opens the outcome panel; the second is unchanged.
  const money = isMoneyFinding(f.rule_id);
```

Then change the triage and resolve branches:

```js
  const dec = money ? ['Record outcome', 'Flag for review']
    : triage ? ['Flag for review', 'Accept as risk']
    : type === 'resolve' ? [applyLabel, 'Dismiss'] ...
  const acts = money ? ['record_outcome', 'flag']
    : triage ? ['flag', 'dismiss'] ...
```

Import `isMoneyFinding` from `'../../../lib/valueOutcome'`. Check the relative depth against `useSpendData.js`'s other `lib` imports.

`engine.js`:
- Add `'Record outcome':'record_outcome',` to `ACTION_INTENT` (`:3533`).
- At the top of `siqAction`, after the sample-row check (`:3556`):

```js
  if(intent==='record_outcome'){
    if(window.__SPENDIQ_OPEN_OUTCOME__) window.__SPENDIQ_OPEN_OUTCOME__(findingId);
    else toast('Outcome recording unavailable');
    return;
  }
```

- At the top of `resolveFinding` (`:3307`):

```js
  if(action==='record_outcome'){ if(window.__SPENDIQ_OPEN_OUTCOME__) window.__SPENDIQ_OPEN_OUTCOME__(id); return; }
```

`SpendIQ/index.jsx`:
- In the component that installs the bridges, add state `const [outcomeFor, setOutcomeFor] = useState(null);`.
- Next to the other bridges: `window.__SPENDIQ_OPEN_OUTCOME__ = (id) => setOutcomeFor(id);`.
- Remove it in the same effect's cleanup, if the bridges have one.
- In the JSX, render:

```jsx
{outcomeFor != null && (
  <ValueOutcomePanel findingId={outcomeFor} onClose={() => setOutcomeFor(null)}
    onDone={() => { setOutcomeFor(null); if (window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__(); }} />
)}
```

`ProcurementHome/index.jsx`:
- In `decideCard`, before `setDeciding(true)`:

```js
    if (intent === 'record_outcome') { setOutcomeFor(cardCur._id); return; }
```

- Add the same `outcomeFor` state and a panel whose `onDone` does what a closed card does today: `setDeckDoneIds`, `useActionsStore.getState().dropClosed(id)`, then advance to the next card. Factor the existing post-close lines into a small `afterClose(id)` function used by both paths, so they stay one behaviour.

- [ ] **Step 5: Run the UI tests**

Run: `npx vitest run src/components/value src/lib/valueOutcome.test.js src/modules/SpendIQ src/modules/ProcurementHome`
Expected: the new tests pass and the existing ones stay green. Record the before and after counts. Run the same command before Step 4 to get the baseline.

- [ ] **Step 6: Commit** (stage hunks only; `useSpendData.js`, `engine.js` and both `index.jsx` files may carry other sessions' edits)

```bash
git status --short src/modules/SpendIQ src/modules/ProcurementHome
git add src/components/value/ValueOutcomePanel.jsx src/components/value/ValueOutcomePanel.test.jsx
git add -p src/modules/SpendIQ/data/useSpendData.js src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx src/modules/ProcurementHome/index.jsx
git diff --cached --stat
git commit -m "feat(value-ledger): money findings close by saying what happened to the money

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Accept only this task's hunks in `git add -p`. Then verify the staged tree builds in a clean worktree:
- `git worktree add /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/55b438ba-4cd7-4e9f-a05c-55689aee1472/scratchpad/ui-verify HEAD`
- `cd` into it, `npx vitest run src/components/value src/lib/valueOutcome.test.js`
- `git worktree remove` it

---

### Task 9: Value Found drawer — "Saved" headline, "Being claimed", credit received, mark realised

**Files (beyond_procwise_ui):**
- Modify: `src/lib/valueFound.js`: add `savedHeadline(summary)`, `claimsOf(summary)`, `isRealisable(finding)`
- Modify: `src/modules/ProcurementHome/ValueFoundDrawer.jsx`: the headline, a "Being claimed" section, a `SettleModal`, and "Mark realised" on opportunity rows
- Modify: `src/modules/ProcurementHome/heroFigure.js`: "in play" reads `summary.in_play_gbp`
- Test: `src/modules/SpendIQ/valueSummary.contract.test.js` (extend)

**Interfaces:**
- Consumes: the Task 4 response fields (`saved_gbp`, `avoided_gbp`, `recovered_gbp`, `realised_gbp`, `claimed_open_gbp`, `in_play_gbp`, and per finding `ledger_state`, `claim`, `claimed_at`); Task 7 helpers.
- Produces:
  - `savedHeadline(summary) -> {value:string|null, parts:string[]}`
  - `claimsOf(summary) -> finding[]`, where `ledger_state === 'claimed'`, oldest claim first
  - `isRealisable(finding) -> boolean`, true for an opportunity with status `identified`, `negotiation` or `agreed`

- [ ] **Step 1: Write the failing contract tests** (append to `valueSummary.contract.test.js`)

```js
import { savedHeadline, claimsOf, isRealisable } from '../../lib/valueFound';

describe('value ledger fields', () => {
  const summary = {
    saved_gbp: 175, avoided_gbp: 100, recovered_gbp: 50, realised_gbp: 25, claimed_open_gbp: 200,
    in_play_gbp: 300, sources: {},
    findings: [
      { id: 'disc:1', ledger_state: 'claimed', claimed_at: '2026-09-20T10:00:00Z', claim: { amount: '200.00', currency: 'GBP', amount_gbp: 200 } },
      { id: 'disc:2', ledger_state: 'claimed', claimed_at: '2026-09-01T10:00:00Z', claim: { amount: '5.00', currency: 'GBP', amount_gbp: 5 } },
      { id: 'disc:3', ledger_state: 'recovered' },
      { id: 'opp:9', source: 'opportunity', status: 'agreed' },
    ],
  };

  it('states saved money and how it was saved, naming only non-zero parts', () => {
    const h = savedHeadline(summary);
    expect(h.value).toMatch(/175/);
    expect(h.parts.length).toBe(3);
    expect(savedHeadline({ ...summary, realised_gbp: 0 }).parts.length).toBe(2);
  });

  it('says nothing rather than £0 when nothing is saved', () => {
    expect(savedHeadline({ saved_gbp: 0 }).value).toBeNull();
  });

  it('lists open claims oldest first', () => {
    expect(claimsOf(summary).map((f) => f.id)).toEqual(['disc:2', 'disc:1']);
  });

  it('only a live opportunity can be marked realised', () => {
    expect(isRealisable({ source: 'opportunity', status: 'agreed' })).toBe(true);
    expect(isRealisable({ source: 'opportunity', status: 'realised' })).toBe(false);
    expect(isRealisable({ source: 'discrepancy', status: 'open' })).toBe(false);
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/modules/SpendIQ/valueSummary.contract.test.js`
Expected: FAIL (`savedHeadline` is not exported).

- [ ] **Step 3: Implement the helpers** (append to `src/lib/valueFound.js`)

```js
/** "Saved £175" with its parts. Null when nothing is saved: never a £0 headline. */
export function savedHeadline(summary) {
  const saved = Number(summary?.saved_gbp || 0);
  if (!(saved > 0)) return { value: null, parts: [] };
  const parts = [
    [summary.avoided_gbp, 'vf.partAvoided', '{v} stopped before payment'],
    [summary.recovered_gbp, 'vf.partRecovered', '{v} recovered'],
    [summary.realised_gbp, 'vf.partRealised', '{v} realised savings'],
  ].filter(([v]) => Number(v) > 0).map(([v, k, f]) => tOr(k, f, { v: money(v) }));
  return { value: money(saved), parts };
}

export function claimsOf(summary) {
  return (summary?.findings || [])
    .filter((f) => f.ledger_state === 'claimed')
    .sort((a, b) => String(a.claimed_at || '').localeCompare(String(b.claimed_at || '')));
}

export function isRealisable(finding) {
  return finding?.source === 'opportunity' && ['identified', 'negotiation', 'agreed'].includes(finding.status);
}
```

- [ ] **Step 4: Update the drawer and the hero**

`ValueFoundDrawer.jsx`:
- **Headline.** Render `savedHeadline(summary)` above the supplier groups: a heading "Saved {value}" and its parts joined by the drawer's existing separator style. Leave it out when `value` is null.
- **Being claimed.** When `claimsOf(summary)` is non-empty, render a section headed `tOr('vf.beingClaimed', 'Being claimed')` with a line "`{claimed_open}` in progress". Each row shows the supplier, `claimLabel(f)`, days since `claimed_at`, and two buttons:
  - `tOr('vf.creditReceived', 'Credit received')` opens a `SettleModal` with outcome `recovered`. Fields: amount pre-filled from `f.claim.amount`, currency, credit note reference (required), date.
  - `tOr('vf.claimDropped', 'Claim dropped')` opens the same modal with outcome `claim_dropped` and an optional note only.
- **SettleModal.** Validates with `validateOutcomeForm` and posts `buildSettleBody(...)` to `${AI_API}${settlePath(f.id)}`. On success, call the drawer's existing refetch (`onNavigate` is for links; find how the drawer's parent refetches `valueSummary`, e.g. a `queryClient.invalidateQueries(['valueSummary'])` in `useHomeData.js`, and pass an `onChanged` prop down). On error, show `ledgerErrorMessage(err)` in the modal.
- **Mark realised.** In `FindingRow`, when `isRealisable(finding)`, show `tOr('vf.markRealised', 'Mark realised')`. It opens the same modal in a `realise` mode: amount, currency (default GBP), date, optional reference. It posts `{amount, currency, valid_from, evidence_ref}` to `${AI_API}${realisePath(finding.id)}`.
- **Existing row label.** The `recovered_gbp` label (`:99-101`) keeps working: `recovered_gbp` is now ledger-derived.

`heroFigure.js`: where "in play" is computed as verified + potential, use `summary.in_play_gbp` when it is present (a number), falling back to the existing sum when it is absent (an older backend). Update the header comment's "WHAT IN PLAY IS" paragraph to say that settled money (stopped, recovered, dropped) is no longer in play.

- [ ] **Step 5: Run the UI tests**

Run: `npx vitest run src/modules/SpendIQ/valueSummary.contract.test.js src/modules/ProcurementHome src/lib`
Expected: new and existing tests pass, with no new failures against the baseline taken before this task.

- [ ] **Step 6: Commit** (hunks only)

```bash
git add -p src/lib/valueFound.js src/modules/ProcurementHome/ValueFoundDrawer.jsx src/modules/ProcurementHome/heroFigure.js src/modules/SpendIQ/valueSummary.contract.test.js
git diff --cached --stat
git commit -m "feat(value-found): saved headline, being-claimed list, credit received, mark realised

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Apply to bp_sqldb, restart, prove it on the local server, compare the suite

**Files:** none new (a demo report goes to the scratchpad).

- [ ] **Step 1: Apply the migration to bp_sqldb**

Run: `PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d bp_sqldb -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-26_bp_value_outcome.sql`
Expected: `COMMIT`. Then confirm with `\d proc.bp_value_outcome` on bp_sqldb: both triggers are listed.

- [ ] **Step 2: Restart the backend**

Restart with `sudo systemctl restart procwise.service`. **Never `pkill -f uvicorn`** (memory: local stack run). Wait until `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/openapi.json` returns 200, then confirm that `/value/findings/{discrepancy_id}/outcome` is present in the paths.

- [ ] **Step 3: Walk the success path on the local server** (spec §2)

Pick one real open `duplicate_invoice` finding from bp_testdb (a read-only query). Then in the UI (Chrome, http://localhost:3000/spendiq):
1. Action Centre → the finding → **Record outcome** → "Claiming it back", with the pre-filled amount → Save.
2. Home → Value Found drawer: the finding appears under **Being claimed**.
3. **Credit received**: amount, reference `CN-DEMO-1` → Save.
4. Confirm that the drawer headline, `GET /opportunities/dashboard` and a freshly generated executive summary (`POST /reports/generate`, report type `exec_procurement_summary`) all show the same recovered £X.
5. Confirm that bp_testdb holds two rows (claimed, recovered) for that finding, and that `UPDATE` on either is refused.

Take screenshots. Note that Chrome MCP tabs are often hidden, which stops `requestAnimationFrame` (memory `reference_chrome_mcp_hidden_tab_raf`), so bring the tab to the front before screenshots.

This demo writes real ledger rows to bp_testdb, the seeded test corpus. Afterwards, leave them and say so in the report; they are an honest record of the demo. Do **not** try to delete them (the table refuses, by design).

- [ ] **Step 4: Full suite, compared to the baseline**

Run the full suite with the Global Constraints environment, the seven `--ignore`s and `--continue-on-collection-errors` (the command recorded in `docs/architecture/BUILD_STATUS_1_TO_5.md` §2.4).
Expected: no test that passed on 2026-09-25 (13,260 passed / 318 failed / 293 errors) now fails. List every new failure, if any, and fix it before claiming done.

- [ ] **Step 5: Report**

Write a short demo report at `/tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/55b438ba-4cd7-4e9f-a05c-55689aee1472/scratchpad/value-ledger-demo.md`: finding id, the figures on each surface, screenshots, ledger rows, and suite counts. Commits stay local on `Development` and the UI's `spendiq-ui`. Ask the user before pushing either.

---

## Self-review notes

- **Spec coverage:**
  - §3 table → Task 1; the §3 derived state → Task 2.
  - §4 routes, transactions, audit and currency → Tasks 3 and 5.
  - §5 readers and triage → Tasks 4 and 6.
  - §6 screens → Tasks 7, 8 and 9; §7 errors → Tasks 3, 5 and 7.
  - §8 testing → every task plus Task 10; §9 out of scope is respected (the gateway is untouched; no RLS).
- **Deliberate deviations from the spec, stated in the tasks:**
  - Live tests clean up by rollback, not trigger-disable (Global Constraints).
  - `saved_gbp` includes realised opportunity savings, shown as a third headline part.
  - `in_play_gbp` is a new field so the Home hero stops counting settled money (Review Focus 4).
  - "Mark realised" lives on the Value Found drawer's opportunity rows.
  - Triage money findings get a new "Record outcome" action, because today they can only be flagged or accepted as risk.
