# Opportunity Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an `OpportunityCriticAgent` that decides whether each detected opportunity would survive a negotiator's scrutiny, recording its verdict and a queryable gap register without suppressing anything on day one.

**Architecture:** The critic is a registered agent, not a library. Its system prompt is a `bp_prompt` row, its seven tests' thresholds are `bp_policy` rows, and its arithmetic comes from registered formulas with golden vectors. An `OpportunityEvidenceAgent` subagent assembles evidence and never judges it. The registry calculates, the agent judges, and code refuses malformed output rather than repairing it. Shadow mode mirrors the P0 pattern shipped 2026-09-09.

**Tech Stack:** Python 3.12, PostgreSQL (`proc` schema on `bp_testdb`), pytest, pandas, Ollama/AgentNick via `BaseAgent.reason()`.

**Spec:** `docs/superpowers/specs/2026-09-09-opportunity-critic-design.md`

## Global Constraints

- **Branch:** work on `Development`. Never push to `main`. Another session shares this checkout — never `git stash`; use `git worktree` if you need isolation.
- **Tests:** run with `./venv/bin/python -m pytest`, targeted, never the whole suite. ~250 pre-existing `extraction_v3` collection errors are unrelated and will bury your signal.
- **Table naming:** all new tables `bp_`-prefixed; indexes `ix_bp_<table>_<column>`.
- **Every guard must be seen to fail.** For each guard, break it deliberately, capture the red output in the commit message or task notes, restore it, capture the green. `feedback_prove_the_guard_fails` records three guards that shipped green while checking nothing.
- **No fabrication.** Absent data stays absent. `UNASSESSED` is never `0.0`, never `None`, never a default.
- **Docs are gitignored:** commit spec/plan files with `git add -f`.
- **Confidence vocabulary:** reuse `Confidence.ASSERTED | CORROBORATED | UNASSESSED` from `src/services/analytics/models.py:70`. Do not define a new enum.
- **Migrations:** additive, idempotent, reversible. Every migration ships with a `_rollback.sql` sibling.

---

### Task 1: The critique and gap tables

**Files:**
- Create: `deploy/sql/2026-09-09_opportunity_critique.sql`
- Create: `deploy/sql/2026-09-09_opportunity_critique_rollback.sql`
- Test: `tests/sql/test_opportunity_critique_ddl.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `proc.bp_opportunity_critique` and `proc.bp_opportunity_gap`, both keyed on `opportunity_ref_id` (TEXT).

- [ ] **Step 1: Write the failing test**

Create `tests/sql/test_opportunity_critique_ddl.py`. Asserted against migration text, not a live DB, so it runs anywhere — same approach as `tests/sql/test_bp_analysis_ddl.py`.

```python
"""The critique DDL says what the spec says it says.

Asserted against the migration text rather than a live database so the test
runs anywhere. Applying it against bp_testdb is verified in Task 9.
"""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-09-09_opportunity_critique.sql").read_text()
ROLLBACK = (Path(__file__).resolve().parents[2]
            / "deploy" / "sql" / "2026-09-09_opportunity_critique_rollback.sql").read_text()


def test_both_tables_created_idempotently():
    for table in ("bp_opportunity_critique", "bp_opportunity_gap"):
        assert f"CREATE TABLE IF NOT EXISTS proc.{table}" in SQL


def test_keyed_on_ref_id_not_the_per_run_counter():
    # opportunity_id is a per-run counter that changes for the same finding
    # between mining runs; keying on it orphans every verdict. See
    # src/services/opportunity_store.py:24.
    assert "opportunity_ref_id TEXT        NOT NULL" in SQL
    assert "opportunity_id" not in SQL


def test_verdict_is_constrained_to_the_five_values():
    assert (
        "CHECK (verdict IN ('VALID', 'VALID_REFRAMED', 'INVALID', "
        "'UNASSESSED', 'DUPLICATE'))"
    ) in SQL


def test_confidence_uses_the_existing_ladder():
    assert "CHECK (confidence IN ('ASSERTED', 'CORROBORATED', 'UNASSESSED'))" in SQL


def test_gap_type_is_constrained_to_the_seven_types():
    for gap_type in ("MISSING_EVIDENCE", "STALE_EVIDENCE", "UNVERIFIED_ASSERTION",
                     "NORMALISATION_NEEDED", "NO_LEVER", "NO_THRESHOLD",
                     "DETECTOR_LOGIC"):
        assert gap_type in SQL


def test_effort_is_constrained():
    assert "CHECK (effort IN ('LOW', 'MEDIUM', 'HIGH'))" in SQL


def test_versions_that_produced_the_verdict_are_recorded():
    # Without these, tuning a threshold leaves stale verdicts on the page
    # presenting themselves as current. Spec section 9.1.
    for col in ("prompt_version", "policy_versions", "formula_versions"):
        assert col in SQL


def test_every_test_result_is_stored_including_passes():
    assert "tests              JSONB" in SQL


def test_shadow_columns_present():
    assert "would_have_suppressed" in SQL
    assert "shadowed" in SQL


def test_gaps_cascade_so_a_critique_deletes_cleanly():
    assert "ON DELETE CASCADE" in SQL


def test_rollback_drops_both_tables():
    assert "DROP TABLE IF EXISTS proc.bp_opportunity_gap" in ROLLBACK
    assert "DROP TABLE IF EXISTS proc.bp_opportunity_critique" in ROLLBACK
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/sql/test_opportunity_critique_ddl.py -v
```

Expected: every test ERRORs with `FileNotFoundError` — the migration does not exist.

- [ ] **Step 3: Write the migration**

Create `deploy/sql/2026-09-09_opportunity_critique.sql`:

```sql
-- The Opportunity Critic's verdict, and the gap register that justifies it.
--
-- Keyed on opportunity_ref_id -- the content-derived identity -- and NOT on
-- opportunity_id, which is a per-run counter the miner assigns while walking
-- candidates. It changes for the SAME finding between runs. Keying on it means
-- every verdict orphans on the next mining pass; opportunity_store.py:24
-- records the cost of learning that once already.
--
-- Gaps are a table, not JSONB on the critique, because the gap register is the
-- part that converts "we could not decide" into work. One gap -- supplier
-- identity -- blocks hundreds of findings, and you learn that from
-- GROUP BY what_is_missing, owner_hint, not by opening 308 JSON blobs.
--
-- Additive, idempotent, reversible.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_opportunity_critique (
    critique_id        BIGSERIAL PRIMARY KEY,
    opportunity_ref_id TEXT        NOT NULL,
    detector_type      TEXT,
    critiqued_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    verdict            TEXT        NOT NULL,
    confidence         TEXT,

    original_claim     TEXT,
    critic_claim       TEXT,
    negotiator_note    TEXT,

    detector_proposed  NUMERIC,
    critic_addressable NUMERIC,
    currency           TEXT,
    value_basis        TEXT,
    haircuts           JSONB,
    lever              JSONB,
    duplicate_of       TEXT,

    -- Every test, including the ones that passed. "anchor_validity PASS" is the
    -- sentence that defends a finding in the room.
    tests              JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Shadow mode, in the shape services/guardrail.py established.
    would_have_suppressed BOOLEAN  NOT NULL DEFAULT false,
    shadowed              BOOLEAN  NOT NULL DEFAULT false,

    -- What produced this verdict. Without these, tuning a threshold silently
    -- leaves stale verdicts on the page presenting themselves as current.
    prompt_version     INTEGER,
    policy_versions    JSONB,
    formula_versions   JSONB,

    -- Pointer to the agent trace in proc.routing.process_details.
    run_id             TEXT,

    CONSTRAINT ck_bp_opportunity_critique_verdict CHECK (verdict IN
        ('VALID', 'VALID_REFRAMED', 'INVALID', 'UNASSESSED', 'DUPLICATE')),
    CONSTRAINT ck_bp_opportunity_critique_confidence CHECK (
        confidence IS NULL OR
        confidence IN ('ASSERTED', 'CORROBORATED', 'UNASSESSED'))
);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_ref
    ON proc.bp_opportunity_critique (opportunity_ref_id, critiqued_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_critiqued
    ON proc.bp_opportunity_critique (critiqued_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_verdict
    ON proc.bp_opportunity_critique (verdict);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_suppressed
    ON proc.bp_opportunity_critique (would_have_suppressed)
    WHERE would_have_suppressed;

CREATE TABLE IF NOT EXISTS proc.bp_opportunity_gap (
    gap_row_id         BIGSERIAL PRIMARY KEY,
    critique_id        BIGINT      NOT NULL
        REFERENCES proc.bp_opportunity_critique (critique_id) ON DELETE CASCADE,
    opportunity_ref_id TEXT        NOT NULL,
    gap_id             TEXT        NOT NULL,
    test               TEXT,
    gap_type           TEXT        NOT NULL,
    what_is_missing    TEXT        NOT NULL,
    why_it_matters     TEXT,
    blocking           BOOLEAN     NOT NULL DEFAULT false,
    resolves_to        TEXT,
    likely_source      TEXT,
    owner_hint         TEXT,
    effort             TEXT,
    ordinal            INTEGER     NOT NULL DEFAULT 0,

    CONSTRAINT ck_bp_opportunity_gap_type CHECK (gap_type IN (
        'MISSING_EVIDENCE', 'STALE_EVIDENCE', 'UNVERIFIED_ASSERTION',
        'NORMALISATION_NEEDED', 'NO_LEVER', 'NO_THRESHOLD', 'DETECTOR_LOGIC')),
    CONSTRAINT ck_bp_opportunity_gap_effort CHECK (
        effort IS NULL OR effort IN ('LOW', 'MEDIUM', 'HIGH'))
);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_critique
    ON proc.bp_opportunity_gap (critique_id);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_ref
    ON proc.bp_opportunity_gap (opportunity_ref_id);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_triage
    ON proc.bp_opportunity_gap (blocking, effort);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_owner
    ON proc.bp_opportunity_gap (owner_hint);

COMMIT;
```

Create `deploy/sql/2026-09-09_opportunity_critique_rollback.sql`:

```sql
-- Reverses 2026-09-09_opportunity_critique.sql. Gaps first: they reference
-- critiques.
BEGIN;
DROP TABLE IF EXISTS proc.bp_opportunity_gap;
DROP TABLE IF EXISTS proc.bp_opportunity_critique;
COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/sql/test_opportunity_critique_ddl.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Prove the constraint guard fails**

Temporarily delete the line `'UNASSESSED', 'DUPLICATE'))` from the verdict CHECK in the migration, re-run, and confirm `test_verdict_is_constrained_to_the_five_values` goes RED. Restore the line and confirm green. Record the red output in the commit message.

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-09-09_opportunity_critique.sql \
        deploy/sql/2026-09-09_opportunity_critique_rollback.sql \
        tests/sql/test_opportunity_critique_ddl.py
git commit -m "feat(critic): the critique record and its gap register

Keyed on opportunity_ref_id, not the per-run opportunity_id counter.
Gaps are their own table so one gap blocking hundreds of findings is
one GROUP BY away instead of 308 JSON blobs."
```

---

### Task 2: Anchor and inflation formulas

**Files:**
- Create: `src/services/formulas/definitions/critic.py`
- Modify: `src/services/formulas/definitions/__init__.py`
- Test: `tests/services/formulas/test_critic_anchor.py`

**Interfaces:**
- Consumes: `Term`, `Output`, `GoldenVector`, `formula` from the registry; `RATIO`, `PERCENT`, `DAYS`, `MONEY`, `BOOLEAN`, `LABEL` from `..contract`.
- Produces: registered formulas `critic.annualised_rate(anchor_value, current_value, years) -> float`, `critic.excess_over_index(annualised_pct, index_pct, band_pp) -> float`, `critic.anchor_age_days(anchor_date, current_date) -> int`, `critic.unit_basis_match(anchor_basis, current_basis) -> bool`, `critic.fabricated_anchor(unit_price, line_value, quantity) -> bool`.

> **Read first:** `src/services/formulas/definitions/opportunity.py:26-50` is the house style for a registered formula. `registry.formula()` raises at import time if `golden` is empty and re-runs every vector on import — so a module whose maths drifts cannot be imported at all. That is the guard; you do not write it.

- [ ] **Step 1: Write the failing test**

Create `tests/services/formulas/test_critic_anchor.py`:

```python
"""Anchor and inflation maths, pinned.

The canonical false positive from the spec's Appendix A is the acceptance
vector: two contracted 4% uplifts against ~3.8% CPI must read as inflation,
not opportunity, forever.
"""
import pytest

from src.services.formulas import ensure_registered, evaluate
from src.services.formulas.definitions import critic  # noqa: F401


@pytest.fixture(autouse=True)
def _registered():
    ensure_registered()


def test_annualised_rate_of_two_four_percent_uplifts():
    # 0.041 -> 0.0447 over 4 years is ~2.18% a year.
    out = evaluate("critic.annualised_rate",
                   {"anchor_value": 0.041, "current_value": 0.0447, "years": 4.0})
    assert out.value == pytest.approx(2.18, abs=0.02)


def test_annualised_rate_refuses_a_zero_anchor():
    # A zero anchor is not a 0% rise, it is an unusable comparator.
    from src.services.formulas.unassessed import UNASSESSED
    out = evaluate("critic.annualised_rate",
                   {"anchor_value": 0.0, "current_value": 0.0447, "years": 4.0})
    assert out.value is UNASSESSED


def test_excess_over_index_is_zero_inside_the_band():
    # 4.0% against a 3.8% index, 2pp band -> inflation, not opportunity.
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 4.0, "index_pct": 3.8, "band_pp": 2.0})
    assert out.value == 0.0


def test_excess_over_index_reports_only_the_excess():
    # 9.0% against 3.8% with a 2pp band -> the opportunity is 3.2pp, not 9.
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 9.0, "index_pct": 3.8, "band_pp": 2.0})
    assert out.value == pytest.approx(3.2, abs=0.001)


def test_excess_over_index_is_unassessed_without_an_index():
    from src.services.formulas.unassessed import UNASSESSED
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 9.0, "index_pct": None, "band_pp": 2.0})
    assert out.value is UNASSESSED


def test_unit_basis_mismatch_is_detected():
    out = evaluate("critic.unit_basis_match",
                   {"anchor_basis": "per_page", "current_basis": "per_month"})
    assert out.value is False


def test_unit_basis_match_is_case_and_space_tolerant():
    out = evaluate("critic.unit_basis_match",
                   {"anchor_basis": "Per Page", "current_basis": "per_page"})
    assert out.value is True


def test_fabricated_anchor_when_unit_price_equals_line_value_at_qty_one():
    # opportunity_miner_agent.py:5545 defaults a missing quantity to 1.0 and a
    # missing unit price to the whole line value. .min() then selects for
    # whichever row is most corrupted downward.
    out = evaluate("critic.fabricated_anchor",
                   {"unit_price": 4540.26, "line_value": 4540.26, "quantity": 1.0})
    assert out.value is True


def test_a_genuine_single_unit_line_is_not_fabricated():
    # Real qty-1 lines exist. The signal is the coincidence, so a line whose
    # value does not equal its unit price is clean even at quantity 1.
    out = evaluate("critic.fabricated_anchor",
                   {"unit_price": 100.0, "line_value": 250.0, "quantity": 1.0})
    assert out.value is False


def test_anchor_age_in_days():
    from datetime import date
    out = evaluate("critic.anchor_age_days",
                   {"anchor_date": date(2022, 3, 1), "current_date": date(2026, 3, 1)})
    assert out.value == 1461
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/services/formulas/test_critic_anchor.py -v
```

Expected: collection ERROR — `ModuleNotFoundError: src.services.formulas.definitions.critic`.

- [ ] **Step 3: Write the implementation**

Create `src/services/formulas/definitions/critic.py`:

```python
"""The Opportunity Critic's arithmetic, registered.

The critic agent judges; it never calculates. Every number in a verdict comes
from here, with pinned behaviour, so that a verdict can be defended by pointing
at a formula version rather than at a model's mood.

The canonical false positive from the spec is a golden vector, not a comment:
two contracted 4% uplifts against ~3.8% CPI read as inflation. If someone later
widens the band and that vector stops reproducing, this module does not import.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from ..contract import BOOLEAN, COUNT, DAYS, DATE, MONEY, PERCENT, RATIO, TEXT, Output, Term
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED

_OWNER = "opportunity_critic"
_FROM = date(2026, 9, 9)


def _num(value: Any) -> Optional[float]:
    """Coerce to float, or None. Never raises, never guesses."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


@formula(
    "critic.annualised_rate",
    version="1.0.0",
    owner=_OWNER,
    purpose="Compound annual rate of change between an anchor price and the current one",
    effective_from=_FROM,
    inputs=[
        Term("anchor_value", MONEY, "the comparator price"),
        Term("current_value", MONEY, "the price today"),
        Term("years", COUNT, "elapsed years between them", minimum=0.0),
    ],
    output=Output("float", PERCENT, "percent per year, or UNASSESSED"),
    notes=(
        "Returns UNASSESSED rather than a number when the anchor is zero or the "
        "span is zero. A zero anchor is not a 0% rise; it is an unusable "
        "comparator, and returning 0.0 would present that as 'no change'."
    ),
    golden=[
        GoldenVector(
            inputs={"anchor_value": 0.041, "current_value": 0.0447, "years": 4.0},
            expected=2.1817, tolerance=0.001,
            note="THE CANONICAL FALSE POSITIVE: managed print, two 4% uplifts "
                 "2022-2026. Must stay inside a 3.8% index +/- 2pp band.",
        ),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 100.0, "years": 1.0},
                     expected=0.0),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 110.0, "years": 1.0},
                     expected=10.0, tolerance=0.001),
        GoldenVector(inputs={"anchor_value": 0.0, "current_value": 50.0, "years": 4.0},
                     expected=UNASSESSED,
                     note="fail-closed: a zero anchor cannot produce a rate"),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 110.0, "years": 0.0},
                     expected=UNASSESSED, note="fail-closed: no elapsed time"),
    ],
)
def annualised_rate(anchor_value=None, current_value=None, years=None):
    anchor, current, span = _num(anchor_value), _num(current_value), _num(years)
    if not anchor or current is None or not span or span <= 0:
        return UNASSESSED
    if anchor <= 0 or current <= 0:
        return UNASSESSED
    return ((current / anchor) ** (1.0 / span) - 1.0) * 100.0


@formula(
    "critic.excess_over_index",
    version="1.0.0",
    owner=_OWNER,
    purpose="The part of a price rise that is above index -- the only part that is an opportunity",
    effective_from=_FROM,
    inputs=[
        Term("annualised_pct", PERCENT, "measured annual rise"),
        Term("index_pct", PERCENT, "the applicable index for category and jurisdiction",
             required=False),
        Term("band_pp", PERCENT, "tolerance in percentage points", minimum=0.0),
    ],
    output=Output("float", PERCENT, "excess percentage points, 0.0 if inside band, "
                                    "or UNASSESSED with no index"),
    notes=(
        "UNASSESSED with no index is the whole point. The prompt forbids guessing "
        "a rate, and there is no index source in this deployment -- so this "
        "formula returns UNASSESSED for every candidate until one is supplied, "
        "and the gap register says so out loud."
    ),
    golden=[
        GoldenVector(inputs={"annualised_pct": 4.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=0.0,
                     note="PINS THE CANONICAL FALSE POSITIVE: inside band is inflation"),
        GoldenVector(inputs={"annualised_pct": 9.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=3.2, tolerance=0.001,
                     note="the opportunity is the excess, never the whole delta"),
        GoldenVector(inputs={"annualised_pct": 2.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=0.0, note="below index is not a negative opportunity"),
        GoldenVector(inputs={"annualised_pct": 9.0, "index_pct": None, "band_pp": 2.0},
                     expected=UNASSESSED,
                     note="fail-closed: no index, no verdict. Never guess a rate."),
    ],
)
def excess_over_index(annualised_pct=None, index_pct=None, band_pp=None):
    measured, index, band = _num(annualised_pct), _num(index_pct), _num(band_pp)
    if measured is None or index is None:
        return UNASSESSED
    ceiling = index + (band or 0.0)
    return max(0.0, measured - ceiling) if measured > ceiling else 0.0


@formula(
    "critic.anchor_age_days",
    version="1.0.0",
    owner=_OWNER,
    purpose="How old the comparator is, so staleness can be tested against a governed rule",
    effective_from=_FROM,
    inputs=[
        Term("anchor_date", DATE, "when the anchor price was observed", required=False),
        Term("current_date", DATE, "when the current price was observed", required=False),
    ],
    output=Output("int", DAYS, "days between, or UNASSESSED if either is undated"),
    notes=(
        "An undated anchor is UNASSESSED, not age zero. The live detector emits "
        "no anchor date at all (spec 6.1), so this returning UNASSESSED is the "
        "normal case until the evidence subagent dates it via invoices."
    ),
    golden=[
        GoldenVector(inputs={"anchor_date": date(2022, 3, 1),
                             "current_date": date(2026, 3, 1)}, expected=1461),
        GoldenVector(inputs={"anchor_date": date(2026, 3, 1),
                             "current_date": date(2026, 3, 1)}, expected=0),
        GoldenVector(inputs={"anchor_date": None, "current_date": date(2026, 3, 1)},
                     expected=UNASSESSED,
                     note="fail-closed: an undated anchor has unknown age, not zero age"),
    ],
)
def anchor_age_days(anchor_date=None, current_date=None):
    if not isinstance(anchor_date, date) or not isinstance(current_date, date):
        return UNASSESSED
    return (current_date - anchor_date).days


@formula(
    "critic.unit_basis_match",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether anchor and current are priced on the same basis -- mismatch invalidates outright",
    effective_from=_FROM,
    inputs=[
        Term("anchor_basis", TEXT, "per page / per seat / per month ...", required=False),
        Term("current_basis", TEXT, "the same, for the current price", required=False),
    ],
    output=Output("bool", BOOLEAN, "True if comparable, False if not, UNASSESSED if unknown"),
    notes=(
        "Compared after case-folding and separator-stripping only. No synonym "
        "table: deciding that 'per user' means 'per seat' is a judgement, and "
        "judgement belongs to the agent, not to a formula."
    ),
    golden=[
        GoldenVector(inputs={"anchor_basis": "per_page", "current_basis": "per_page"},
                     expected=True),
        GoldenVector(inputs={"anchor_basis": "Per Page", "current_basis": "per_page"},
                     expected=True, note="case and separators are not a difference"),
        GoldenVector(inputs={"anchor_basis": "per_page", "current_basis": "per_month"},
                     expected=False, note="unit mismatch invalidates outright"),
        GoldenVector(inputs={"anchor_basis": None, "current_basis": "per_page"},
                     expected=UNASSESSED,
                     note="fail-closed: an unstated basis is unknown, not matching"),
    ],
)
def unit_basis_match(anchor_basis=None, current_basis=None):
    def _norm(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        for ch in (" ", "-", "_", "/"):
            text = text.replace(ch, "")
        return text or None

    left, right = _norm(anchor_basis), _norm(current_basis)
    if left is None or right is None:
        return UNASSESSED
    return left == right


@formula(
    "critic.fabricated_anchor",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether an anchor price is an artefact of the miner's missing-data fallbacks",
    effective_from=_FROM,
    inputs=[
        Term("unit_price", MONEY, "the anchor's unit price", required=False),
        Term("line_value", MONEY, "the anchor line's total value", required=False),
        Term("quantity", COUNT, "the anchor line's quantity", required=False),
    ],
    output=Output("bool", BOOLEAN, "True if the anchor is an artefact, not a price"),
    notes=(
        "THIS RULE IS NOT IN THE ORIGINAL PROMPT. It exists because of what the "
        "live detector actually does (spec 6.1): at "
        "opportunity_miner_agent.py:5545 a missing quantity becomes 1.0 and a "
        "missing unit price becomes the whole line value, and the anchor is then "
        "a .min() over that column -- the aggregation that selects for whichever "
        "row is most corrupted downward. A unit price identical to its line "
        "value at quantity exactly 1.0 is the fingerprint. It INVALIDATES rather "
        "than downgrades: a fabricated floor is not a weak comparator, it is not "
        "a price."
    ),
    golden=[
        GoldenVector(inputs={"unit_price": 4540.26, "line_value": 4540.26, "quantity": 1.0},
                     expected=True, note="the fallback fingerprint"),
        GoldenVector(inputs={"unit_price": 100.0, "line_value": 250.0, "quantity": 1.0},
                     expected=False,
                     note="genuine qty-1 lines exist; the coincidence is the signal"),
        GoldenVector(inputs={"unit_price": 50.0, "line_value": 500.0, "quantity": 10.0},
                     expected=False),
        GoldenVector(inputs={"unit_price": None, "line_value": 500.0, "quantity": 10.0},
                     expected=UNASSESSED,
                     note="fail-closed: cannot clear an anchor you cannot see"),
    ],
)
def fabricated_anchor(unit_price=None, line_value=None, quantity=None):
    price, value, qty = _num(unit_price), _num(line_value), _num(quantity)
    if price is None or value is None or qty is None:
        return UNASSESSED
    return abs(qty - 1.0) < 1e-9 and abs(price - value) < 1e-9
```

Then register the module. Modify `src/services/formulas/definitions/__init__.py` — add `critic` to the import tuple and to `__all__`, keeping alphabetical order:

```python
from . import (  # noqa: F401
    benchmarking,
    clustering,
    critic,
    deal,
    duplicates,
    extraction,
    linking,
    negotiation,
    opportunity,
    price_outlier,
    ranking,
    risk,
    rivalry,
)

__all__ = [
    "benchmarking", "clustering", "critic", "deal", "duplicates", "extraction",
    "linking", "negotiation", "opportunity", "price_outlier", "ranking", "risk",
    "rivalry",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/services/formulas/test_critic_anchor.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Prove the golden-vector guard fails**

Change the canonical false positive's `expected` from `2.1817` to `9.0` in `critic.annualised_rate`, then run:

```bash
./venv/bin/python -c "from src.services.formulas import ensure_registered; ensure_registered()"
```

Expected: `GoldenVectorFailure` at **import time** — the module refuses to load. This is the guard that stops someone widening the band and silently resurrecting false positives. Restore `2.1817`, re-run, confirm it imports clean. Record both outputs.

- [ ] **Step 6: Commit**

```bash
git add src/services/formulas/definitions/critic.py \
        src/services/formulas/definitions/__init__.py \
        tests/services/formulas/test_critic_anchor.py
git commit -m "feat(critic): anchor and inflation formulas, canonical false positive pinned

critic.fabricated_anchor is not in the original prompt. It exists because
the live anchor is a .min() over a column where a missing quantity becomes
1.0 and a missing unit price becomes the whole line value."
```

---

### Task 3: Scope, value, and confidence formulas

**Files:**
- Modify: `src/services/formulas/definitions/critic.py` (append)
- Test: `tests/services/formulas/test_critic_value.py`

**Interfaces:**
- Consumes: everything from Task 2's module header.
- Produces: `critic.normalise_unit_rate(total_value, quantity) -> float`, `critic.volume_delta(anchor_qty, current_qty) -> float`, `critic.addressable_value(detector_proposed, haircuts) -> float`, `critic.relative_gap(gap_value, base_value) -> float`, `critic.friction_haircut(gross_value, friction_pct) -> float`, `critic.min_confidence(confidences) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/formulas/test_critic_value.py`:

```python
"""Value, scope and confidence maths for the critic.

The rule that matters most here: a claim inherits the WEAKEST confidence of
the evidence it depends on, and nothing can upgrade evidence.
"""
import pytest

from src.services.formulas import ensure_registered, evaluate
from src.services.formulas.definitions import critic  # noqa: F401
from src.services.formulas.unassessed import UNASSESSED


@pytest.fixture(autouse=True)
def _registered():
    ensure_registered()


def test_normalise_unit_rate():
    out = evaluate("critic.normalise_unit_rate",
                   {"total_value": 500.0, "quantity": 10.0})
    assert out.value == 50.0


def test_normalise_unit_rate_refuses_zero_quantity():
    out = evaluate("critic.normalise_unit_rate",
                   {"total_value": 500.0, "quantity": 0.0})
    assert out.value is UNASSESSED


def test_volume_delta_is_a_fraction_of_the_anchor():
    out = evaluate("critic.volume_delta",
                   {"anchor_qty": 100.0, "current_qty": 130.0})
    assert out.value == pytest.approx(0.30, abs=0.001)


def test_addressable_value_subtracts_every_haircut():
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 48000.0,
                    "haircuts": [{"reason": "inflation", "amount": 30000.0},
                                 {"reason": "friction", "amount": 8000.0}]})
    assert out.value == 10000.0


def test_addressable_value_never_goes_negative():
    # A finding haircut below zero is worth nothing, not worth minus something.
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 1000.0,
                    "haircuts": [{"reason": "inflation", "amount": 5000.0}]})
    assert out.value == 0.0


def test_addressable_value_can_never_exceed_what_the_detector_proposed():
    # A negative haircut would raise the value. The prompt forbids VALID with a
    # value above the detector's, so the formula cannot produce one.
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 1000.0,
                    "haircuts": [{"reason": "friction", "amount": -5000.0}]})
    assert out.value == 1000.0


def test_relative_gap_reports_proportion():
    out = evaluate("critic.relative_gap", {"gap_value": 1200.0, "base_value": 3000.0})
    assert out.value == pytest.approx(0.40, abs=0.001)


def test_relative_gap_on_a_negligible_base_is_unassessed():
    # Live, one supplier's +345,261% was GBP 26.72 the year before.
    out = evaluate("critic.relative_gap", {"gap_value": 1200.0, "base_value": 0.0})
    assert out.value is UNASSESSED


def test_friction_haircut():
    out = evaluate("critic.friction_haircut",
                   {"gross_value": 10000.0, "friction_pct": 20.0})
    assert out.value == 2000.0


def test_min_confidence_takes_the_weakest():
    out = evaluate("critic.min_confidence",
                   {"confidences": ["CORROBORATED", "ASSERTED"]})
    assert out.value == "ASSERTED"


def test_any_unassessed_input_forces_unassessed():
    out = evaluate("critic.min_confidence",
                   {"confidences": ["CORROBORATED", "UNASSESSED", "CORROBORATED"]})
    assert out.value == "UNASSESSED"


def test_no_evidence_at_all_is_unassessed_not_corroborated():
    out = evaluate("critic.min_confidence", {"confidences": []})
    assert out.value == "UNASSESSED"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/services/formulas/test_critic_value.py -v
```

Expected: FAIL — `FormulaError: unknown formula 'critic.normalise_unit_rate'`.

- [ ] **Step 3: Write the implementation**

Append to `src/services/formulas/definitions/critic.py`:

```python
@formula(
    "critic.normalise_unit_rate",
    version="1.0.0",
    owner=_OWNER,
    purpose="Reduce a line to a per-unit rate so anchor and current can be compared like for like",
    effective_from=_FROM,
    inputs=[
        Term("total_value", MONEY, "the line total"),
        Term("quantity", COUNT, "units on the line"),
    ],
    output=Output("float", MONEY, "value per unit, or UNASSESSED"),
    notes="Zero or absent quantity is UNASSESSED. Dividing by a defaulted 1.0 is "
          "how the miner produced fabricated anchors in the first place.",
    golden=[
        GoldenVector(inputs={"total_value": 500.0, "quantity": 10.0}, expected=50.0),
        GoldenVector(inputs={"total_value": 500.0, "quantity": 0.0}, expected=UNASSESSED,
                     note="fail-closed: never divide by a defaulted quantity"),
        GoldenVector(inputs={"total_value": 500.0, "quantity": None}, expected=UNASSESSED),
    ],
)
def normalise_unit_rate(total_value=None, quantity=None):
    value, qty = _num(total_value), _num(quantity)
    if value is None or not qty or qty <= 0:
        return UNASSESSED
    return value / qty


@formula(
    "critic.volume_delta",
    version="1.0.0",
    owner=_OWNER,
    purpose="Proportional change in volume between anchor and current, to test like-for-like",
    effective_from=_FROM,
    inputs=[
        Term("anchor_qty", COUNT, "volume at the anchor", required=False),
        Term("current_qty", COUNT, "volume now", required=False),
    ],
    output=Output("float", RATIO, "signed fraction of the anchor volume, or UNASSESSED"),
    golden=[
        GoldenVector(inputs={"anchor_qty": 100.0, "current_qty": 130.0}, expected=0.30,
                     tolerance=0.001),
        GoldenVector(inputs={"anchor_qty": 100.0, "current_qty": 100.0}, expected=0.0),
        GoldenVector(inputs={"anchor_qty": 100.0, "current_qty": 70.0}, expected=-0.30,
                     tolerance=0.001),
        GoldenVector(inputs={"anchor_qty": 0.0, "current_qty": 70.0}, expected=UNASSESSED,
                     note="fail-closed: no anchor volume, no proportion"),
    ],
)
def volume_delta(anchor_qty=None, current_qty=None):
    anchor, current = _num(anchor_qty), _num(current_qty)
    if not anchor or anchor <= 0 or current is None:
        return UNASSESSED
    return (current - anchor) / anchor


@formula(
    "critic.addressable_value",
    version="1.0.0",
    owner=_OWNER,
    purpose="What is left of the detector's number after every haircut the tests justified",
    effective_from=_FROM,
    inputs=[
        Term("detector_proposed", MONEY, "the detector's claimed impact"),
        Term("haircuts", TEXT, "list of {reason, amount} deductions", required=False),
    ],
    output=Output("float", MONEY, "addressable value, floored at 0 and capped at the "
                                  "detector's own figure"),
    notes=(
        "Clamped at BOTH ends, and the upper clamp is the load-bearing one. The "
        "prompt forbids ever emitting VALID with a value above the detector's, "
        "so a negative haircut must not be able to raise it. Enforcing that here "
        "as well as in the invariant guard means a bad policy row cannot inflate "
        "a number even if the guard is bypassed."
    ),
    golden=[
        GoldenVector(inputs={"detector_proposed": 48000.0,
                             "haircuts": [{"reason": "inflation", "amount": 30000.0},
                                          {"reason": "friction", "amount": 8000.0}]},
                     expected=10000.0),
        GoldenVector(inputs={"detector_proposed": 1000.0,
                             "haircuts": [{"reason": "inflation", "amount": 5000.0}]},
                     expected=0.0, note="worth nothing, not worth minus something"),
        GoldenVector(inputs={"detector_proposed": 1000.0,
                             "haircuts": [{"reason": "friction", "amount": -5000.0}]},
                     expected=1000.0,
                     note="PINS THE INVARIANT: a haircut can never raise the value"),
        GoldenVector(inputs={"detector_proposed": 1000.0, "haircuts": None},
                     expected=1000.0),
    ],
)
def addressable_value(detector_proposed=None, haircuts=None):
    proposed = _num(detector_proposed)
    if proposed is None:
        return UNASSESSED
    total = 0.0
    for cut in haircuts or []:
        amount = _num(cut.get("amount") if isinstance(cut, dict) else cut)
        if amount:
            total += amount
    return max(0.0, min(proposed, proposed - total))


@formula(
    "critic.relative_gap",
    version="1.0.0",
    owner=_OWNER,
    purpose="The gap as a proportion of its base -- 40% of GBP 3k is not 4% of GBP 3m",
    effective_from=_FROM,
    inputs=[
        Term("gap_value", MONEY, "the gap"),
        Term("base_value", MONEY, "what it is a gap against"),
    ],
    output=Output("float", RATIO, "fraction of base, or UNASSESSED on a negligible base"),
    notes="A percentage off a base too small to mean anything is noise. Live, one "
          "supplier's +345,261% was GBP 26.72 the year before.",
    golden=[
        GoldenVector(inputs={"gap_value": 1200.0, "base_value": 3000.0}, expected=0.40,
                     tolerance=0.001),
        GoldenVector(inputs={"gap_value": 120000.0, "base_value": 3000000.0},
                     expected=0.04, tolerance=0.001),
        GoldenVector(inputs={"gap_value": 1200.0, "base_value": 0.0}, expected=UNASSESSED,
                     note="fail-closed: negligible base"),
    ],
)
def relative_gap(gap_value=None, base_value=None):
    gap, base = _num(gap_value), _num(base_value)
    if gap is None or not base or base <= 0:
        return UNASSESSED
    return gap / base


@formula(
    "critic.friction_haircut",
    version="1.0.0",
    owner=_OWNER,
    purpose="Deduction for the real cost of switching or renegotiating",
    effective_from=_FROM,
    inputs=[
        Term("gross_value", MONEY, "value before friction"),
        Term("friction_pct", PERCENT, "governed friction band for this situation",
             required=False, minimum=0.0, maximum=100.0),
    ],
    output=Output("float", MONEY, "the amount to deduct, or UNASSESSED without a band"),
    notes="No default band. An invented friction percentage is an invented number, "
          "and the governed policy row is the only source.",
    golden=[
        GoldenVector(inputs={"gross_value": 10000.0, "friction_pct": 20.0},
                     expected=2000.0),
        GoldenVector(inputs={"gross_value": 10000.0, "friction_pct": 0.0}, expected=0.0),
        GoldenVector(inputs={"gross_value": 10000.0, "friction_pct": None},
                     expected=UNASSESSED,
                     note="fail-closed: no governed band, no haircut invented"),
    ],
)
def friction_haircut(gross_value=None, friction_pct=None):
    gross, pct = _num(gross_value), _num(friction_pct)
    if gross is None or pct is None:
        return UNASSESSED
    return gross * (pct / 100.0)


@formula(
    "critic.min_confidence",
    version="1.0.0",
    owner=_OWNER,
    purpose="A claim inherits the weakest confidence of the evidence it rests on",
    effective_from=_FROM,
    inputs=[Term("confidences", TEXT, "the confidence tag of every load-bearing fact",
                 required=False)],
    output=Output("str", TEXT, "ASSERTED | CORROBORATED | UNASSESSED"),
    notes=(
        "Nothing upgrades evidence. An empty list is UNASSESSED, not CORROBORATED: "
        "'we checked nothing' and 'we checked and it was fine' must not return the "
        "same answer. Ladder matches services/analytics/models.py:70."
    ),
    golden=[
        GoldenVector(inputs={"confidences": ["CORROBORATED", "CORROBORATED"]},
                     expected="CORROBORATED"),
        GoldenVector(inputs={"confidences": ["CORROBORATED", "ASSERTED"]},
                     expected="ASSERTED", note="weakest wins"),
        GoldenVector(inputs={"confidences": ["CORROBORATED", "UNASSESSED", "ASSERTED"]},
                     expected="UNASSESSED",
                     note="any unassessed load-bearing fact forces the verdict"),
        GoldenVector(inputs={"confidences": []}, expected="UNASSESSED",
                     note="fail-closed: no evidence is not good evidence"),
        GoldenVector(inputs={"confidences": None}, expected="UNASSESSED"),
    ],
)
def min_confidence(confidences=None):
    # Weakest first. Anything unrecognised is treated as UNASSESSED.
    ladder = {"UNASSESSED": 0, "ASSERTED": 1, "CORROBORATED": 2}
    names = ["UNASSESSED", "ASSERTED", "CORROBORATED"]
    if not confidences:
        return "UNASSESSED"
    worst = 2
    for item in confidences:
        worst = min(worst, ladder.get(str(item).strip().upper(), 0))
    return names[worst]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/services/formulas/test_critic_value.py \
                            tests/services/formulas/test_critic_anchor.py -v
```

Expected: 22 passed.

- [ ] **Step 5: Prove the upper-clamp guard fails**

In `addressable_value`, change `max(0.0, min(proposed, proposed - total))` to `max(0.0, proposed - total)`. Run the import check:

```bash
./venv/bin/python -c "from src.services.formulas import ensure_registered; ensure_registered()"
```

Expected: `GoldenVectorFailure` on the negative-haircut vector — the module refuses to import. Restore the clamp and confirm clean. Record both.

- [ ] **Step 6: Commit**

```bash
git add src/services/formulas/definitions/critic.py \
        tests/services/formulas/test_critic_value.py
git commit -m "feat(critic): value, scope and confidence formulas

addressable_value is clamped at both ends; the upper clamp pins the
invariant that a haircut can never raise a finding's value. min_confidence
returns UNASSESSED for an empty list: 'we checked nothing' and 'we checked
and it was fine' must not return the same answer."
```

---

### Task 4: Governed inputs — the policy rows and the system prompt

**Files:**
- Create: `deploy/sql/2026-09-09_critic_governance.sql`
- Create: `deploy/sql/2026-09-09_critic_governance_rollback.sql`
- Create: `src/services/opportunity_critic/__init__.py`
- Create: `src/services/opportunity_critic/governed.py`
- Test: `tests/services/test_critic_governed.py`
- Test: `tests/sql/test_critic_governance_ddl.py`

**Interfaces:**
- Consumes: `proc.bp_policy`, `proc.bp_prompt`.
- Produces: `load_thresholds(policy_engine=None) -> Thresholds`, `load_system_prompt(prompt_engine=None) -> tuple[str, int]`, and the frozen dataclass `Thresholds(index_band_pp, materiality_floor_gbp, relative_gap_floor, anchor_stale_days, friction_bands, source)`.

> **Why a module here when the architecture principle says the agent decides:** this module *fetches* governed inputs and hands them over. It contains no thresholds and makes no decision. If a threshold cannot be resolved it returns `None` for that field and the agent must degrade to UNASSESSED — the `approvals_agent` pattern, where an unresolvable governed threshold escalates rather than falling back to a constant.

- [ ] **Step 1: Write the failing tests**

Create `tests/sql/test_critic_governance_ddl.py`:

```python
"""The governance seed carries the critic's rules and prompt, and ships shadow empty."""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-09-09_critic_governance.sql").read_text()


def test_system_prompt_is_seeded_as_a_governed_row():
    assert "opportunity_critic_system" in SQL
    assert "proc.bp_prompt" in SQL


def test_every_threshold_is_a_policy_row_not_a_constant():
    for key in ("index_band_pp", "materiality_floor_gbp", "relative_gap_floor",
                "anchor_stale_days", "friction_bands"):
        assert key in SQL


def test_shadow_enrolment_ships_empty():
    # Nothing is suppressed on day one. Enrolling is a governed edit.
    assert '"shadow_detectors": []' in SQL


def test_seeds_are_guarded_so_reruns_are_no_ops():
    assert SQL.count("WHERE NOT EXISTS") >= 2
```

Create `tests/services/test_critic_governed.py`:

```python
"""Governed inputs are fetched, never defaulted.

The approvals_agent lesson: an unresolvable governed threshold must escalate,
never fall back to a constant. A made-up materiality floor silently discards
real findings.
"""
from src.services.opportunity_critic.governed import Thresholds, load_thresholds


class _FakeEngine:
    def __init__(self, payload):
        self._payload = payload

    def get_policy(self, slug):
        return self._payload


def test_thresholds_come_from_the_governed_policy():
    engine = _FakeEngine({
        "policy_id": 42,
        "version": 3,
        "policy_details": {"rules": {
            "index_band_pp": 2.0,
            "materiality_floor_gbp": 5000.0,
            "relative_gap_floor": 0.05,
            "anchor_stale_days": 365,
            "friction_bands": {"default": 15.0, "sole_source": 40.0},
        }},
    })
    out = load_thresholds(engine)
    assert out.index_band_pp == 2.0
    assert out.materiality_floor_gbp == 5000.0
    assert out.friction_bands["sole_source"] == 40.0
    assert out.source == {"policy_id": 42, "version": 3}


def test_a_missing_policy_yields_none_not_a_default():
    out = load_thresholds(_FakeEngine(None))
    assert out.index_band_pp is None
    assert out.materiality_floor_gbp is None
    assert out.friction_bands == {}
    assert out.source is None


def test_a_partial_policy_leaves_the_missing_field_none():
    engine = _FakeEngine({"policy_id": 1, "version": 1,
                          "policy_details": {"rules": {"index_band_pp": 2.0}}})
    out = load_thresholds(engine)
    assert out.index_band_pp == 2.0
    assert out.materiality_floor_gbp is None


def test_thresholds_are_frozen_so_a_caller_cannot_edit_a_governed_rule():
    import dataclasses
    import pytest
    out = load_thresholds(_FakeEngine(None))
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.index_band_pp = 99.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/sql/test_critic_governance_ddl.py \
                            tests/services/test_critic_governed.py -v
```

Expected: `FileNotFoundError` for the DDL test; `ModuleNotFoundError: src.services.opportunity_critic` for the other.

- [ ] **Step 3: Write the migration**

Create `deploy/sql/2026-09-09_critic_governance.sql`. Replace `<APPENDIX A>` with the full prompt text from Appendix A of the spec, with single quotes doubled for SQL:

```sql
-- The Opportunity Critic's governed inputs: its system prompt, its thresholds,
-- and its shadow enrolment list.
--
-- Every number the critic applies lives here rather than in Python. The prompt
-- states them in prose; these rows are authoritative and the prose is
-- documentation. A customer tuning a materiality floor must not need a deploy.
--
-- shadow_detectors ships EMPTY, the same way EmailReplyAutonomyPolicy's
-- auto_reply_intents and guardrail's shadow_actions do. Nothing is suppressed
-- on day one, and enrolling a detector is a governed edit. Every entry MUST
-- carry an "until".
--
-- Idempotent: both seeds are guarded on name.
BEGIN;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'opportunity_critic_system', 'critique', 'opportunity_critic',
       jsonb_build_object('prompt_template', $CRITIC$<APPENDIX A>$CRITIC$)
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'opportunity_critic_system');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details,
                            policy_linked_agents, policy_status, version)
SELECT 'opportunity_critic_thresholds', 'critique',
       'Thresholds the Opportunity Critic applies when testing a candidate.',
       '{
          "rules": {
            "index_band_pp": 2.0,
            "materiality_floor_gbp": null,
            "relative_gap_floor": 0.05,
            "anchor_stale_days": 365,
            "friction_bands": {"default": 15.0, "sole_source": 40.0, "commodity": 5.0},
            "shadow_detectors": []
          }
        }'::jsonb,
       'opportunity_critic', 1, 1
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'opportunity_critic_thresholds');

COMMIT;
```

> **`materiality_floor_gbp` ships `null` deliberately.** The prompt says that with no threshold supplied, materiality is reported and marked UNASSESSED. Seeding an invented floor would silently discard findings against a number nobody chose.

Create `deploy/sql/2026-09-09_critic_governance_rollback.sql`:

```sql
BEGIN;
DELETE FROM proc.bp_policy WHERE policy_name = 'opportunity_critic_thresholds';
DELETE FROM proc.bp_prompt WHERE prompt_name = 'opportunity_critic_system';
COMMIT;
```

- [ ] **Step 4: Write the loader**

Create `src/services/opportunity_critic/__init__.py`:

```python
"""Support for the Opportunity Critic agent.

Nothing here decides anything. These modules fetch the agent's governed inputs
and persist its outputs; the judgement is the agent's, the arithmetic is the
formula registry's.
"""
```

Create `src/services/opportunity_critic/governed.py`:

```python
"""Fetch the critic's governed inputs. Never default them.

`approvals_agent` records what happens when a governed threshold falls back to
a constant: the agent returned SUCCESS while every threshold silently became a
hardcoded 1000. An unresolvable threshold here becomes None, and the agent
degrades that test to UNASSESSED rather than inventing a number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

POLICY_SLUG = "opportunity_critic_thresholds"
PROMPT_SLUG = "opportunity_critic_system"


@dataclass(frozen=True)
class Thresholds:
    """The governed numbers, exactly as resolved. None means unresolved."""

    index_band_pp: Optional[float] = None
    materiality_floor_gbp: Optional[float] = None
    relative_gap_floor: Optional[float] = None
    anchor_stale_days: Optional[int] = None
    friction_bands: Dict[str, float] = field(default_factory=dict)
    shadow_detectors: Tuple[Dict[str, Any], ...] = ()
    source: Optional[Dict[str, Any]] = None


def _rules(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    details = policy.get("policy_details") or {}
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules") or {}
    return rules if isinstance(rules, dict) else {}


def _opt_float(rules: Dict[str, Any], key: str) -> Optional[float]:
    value = rules.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("critic threshold %s is not numeric: %r", key, value)
        return None


def load_thresholds(policy_engine: Any = None) -> Thresholds:
    """Resolve the governed thresholds. Missing fields stay None."""
    policy = None
    if policy_engine is not None:
        try:
            policy = policy_engine.get_policy(POLICY_SLUG)
        except Exception as exc:  # noqa: BLE001 - an unreadable policy is not a crash
            logger.error("critic policy %s unreadable: %s", POLICY_SLUG, exc)
            policy = None

    rules = _rules(policy)
    bands = rules.get("friction_bands") or {}
    shadow = rules.get("shadow_detectors") or []
    stale = rules.get("anchor_stale_days")

    return Thresholds(
        index_band_pp=_opt_float(rules, "index_band_pp"),
        materiality_floor_gbp=_opt_float(rules, "materiality_floor_gbp"),
        relative_gap_floor=_opt_float(rules, "relative_gap_floor"),
        anchor_stale_days=int(stale) if stale is not None else None,
        friction_bands={str(k): float(v) for k, v in bands.items()
                        if isinstance(v, (int, float))},
        shadow_detectors=tuple(s for s in shadow if isinstance(s, dict)),
        source=({"policy_id": policy.get("policy_id"), "version": policy.get("version")}
                if isinstance(policy, dict) else None),
    )


def load_system_prompt(prompt_engine: Any = None) -> Tuple[Optional[str], Optional[int]]:
    """Return ``(prompt_text, version)`` from the governed prompt row.

    ``(None, None)`` when it cannot be resolved. The agent must refuse to run
    rather than fall back to a prompt baked into code: the DB prompt is the
    system of record and a code default that silently wins is how governance
    stops being governance.
    """
    if prompt_engine is None:
        return None, None
    try:
        row = prompt_engine.get_prompt(PROMPT_SLUG)
    except Exception as exc:  # noqa: BLE001
        logger.error("critic prompt %s unreadable: %s", PROMPT_SLUG, exc)
        return None, None
    if not isinstance(row, dict):
        return None, None
    desc = row.get("prompts_desc") or {}
    text = desc.get("prompt_template") if isinstance(desc, dict) else None
    return (text or None), row.get("version")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/sql/test_critic_governance_ddl.py \
                            tests/services/test_critic_governed.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Prove the no-default guard fails**

In `load_thresholds`, change `materiality_floor_gbp=_opt_float(rules, "materiality_floor_gbp")` to `materiality_floor_gbp=_opt_float(rules, "materiality_floor_gbp") or 5000.0`. Re-run — `test_a_missing_policy_yields_none_not_a_default` must go RED. Restore and confirm green. Record both.

- [ ] **Step 7: Commit**

```bash
git add deploy/sql/2026-09-09_critic_governance.sql \
        deploy/sql/2026-09-09_critic_governance_rollback.sql \
        src/services/opportunity_critic/ \
        tests/services/test_critic_governed.py \
        tests/sql/test_critic_governance_ddl.py
git commit -m "feat(critic): governed thresholds and system prompt

materiality_floor_gbp ships null on purpose: the prompt says an absent
threshold is reported UNASSESSED, and an invented floor silently discards
findings against a number nobody chose. shadow_detectors ships empty."
```

---

### Task 5: The evidence subagent

**Files:**
- Create: `src/agents/opportunity_evidence_agent.py`
- Create: `src/services/opportunity_critic/assemble.py`
- Modify: `agent_definitions.json`
- Test: `tests/agents/test_opportunity_evidence_agent.py`

**Interfaces:**
- Consumes: `Thresholds` from Task 4; `proc.bp_opportunity`, `proc.bp_contract_master`, `proc.bp_invoice_trgt`.
- Produces: `assemble_candidate(finding: dict, conn) -> dict` returning the critic's input envelope with keys `finding_id`, `detector_family`, `claim`, `anchor`, `current`, `delta`, `evidence`, `contract_context`, `category_context`; and `OpportunityEvidenceAgent(BaseAgent)` with slug `opportunity_evidence`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_opportunity_evidence_agent.py`:

```python
"""The evidence subagent assembles; it never judges and never infers.

Live ground truth this encodes (spec section 6.2): opportunity suppliers are
name-derived slugs (SUP-MeridianSystems12), contract suppliers are coded ids
(S9251), and zero of 308 opportunities resolve to a contract row. The correct
behaviour is to say so, not to fuzzy-match across the two key spaces.
"""
from src.services.opportunity_critic.assemble import assemble_candidate


class _FakeCursor:
    def __init__(self, results):
        self._results = results
        self._last = None

    def execute(self, sql, params=None):
        for key, rows in self._results.items():
            if key in sql:
                self._last = rows
                return
        self._last = []

    def fetchall(self):
        return self._last or []

    def fetchone(self):
        return (self._last or [None])[0]


class _FakeConn:
    def __init__(self, results):
        self._cur = _FakeCursor(results)

    def cursor(self):
        return self._cur


_FINDING = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "supplier_id": "SUP-MeridianSystems12",
    "financial_impact_gbp": 18330.80,
    "facts_state": "RESOLVED",
    "source_records": ["PO000967", "INV000967-1"],
    "calculation_details": {
        "actual_price": 4998.53, "benchmark_price": 4540.26,
        "quantity": 40.0, "variance_pct": 0.1009,
    },
}


def test_the_anchor_carries_the_benchmark_price():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["anchor"]["value"] == 4540.26


def test_the_anchor_is_labelled_as_a_cheapest_ever_comparator():
    # Not a prior price and not a market benchmark: it is a .min() over
    # avg_price. The critic cannot test anchor validity without knowing that.
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["anchor"]["kind"] == "cheapest_observed"


def test_an_unresolvable_supplier_produces_absent_contract_context():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["contract_context"] == {}


def test_an_unresolvable_supplier_is_reported_not_guessed():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["evidence"]["contract_resolution"] == "SUPPLIER_NOT_IN_CONTRACT_MASTER"


def test_contract_context_is_populated_when_the_supplier_does_resolve():
    conn = _FakeConn({"bp_contract_master": [
        ("C-1", "2022-03-01", "2028-03-01", "GBP", "United Kingdom", "NET30", None),
    ]})
    out = assemble_candidate(_FINDING, conn)
    assert out["contract_context"]["contract_id"] == "C-1"
    assert out["contract_context"]["currency"] == "GBP"


def test_indeterminate_facts_state_tags_evidence_unassessed():
    finding = dict(_FINDING, facts_state="INDETERMINATE")
    out = assemble_candidate(finding, _FakeConn({}))
    assert out["evidence"]["current_confidence"] == "UNASSESSED"


def test_resolved_facts_state_tags_evidence_asserted_not_corroborated():
    # RESOLVED means numbers were parsed out of a document, not that a second
    # source agreed with them. Nothing here can upgrade evidence.
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["evidence"]["current_confidence"] == "ASSERTED"


def test_no_index_is_reported_as_absent_rather_than_omitted():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["category_context"]["index_pct"] is None
    assert out["category_context"]["index_source"] == "NONE_AVAILABLE"


def test_the_assembler_never_returns_a_verdict_field():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    for judged in ("verdict", "tests", "critic_claim", "value"):
        assert judged not in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/agents/test_opportunity_evidence_agent.py -v
```

Expected: `ModuleNotFoundError: src.services.opportunity_critic.assemble`.

- [ ] **Step 3: Write the assembler**

Create `src/services/opportunity_critic/assemble.py`:

```python
"""Build the critic's input envelope from a finding and the corpus.

Assembly, never judgement. What cannot be found is reported absent; nothing is
inferred, and nothing is fuzzy-matched across key spaces.

Two facts about this deployment shape the code (spec section 6):

  * Opportunity suppliers are name-derived slugs (SUP-MeridianSystems12);
    contract suppliers are coded ids (S9251). Zero of 308 opportunities resolve.
    The honest output is SUPPLIER_NOT_IN_CONTRACT_MASTER, not a guess.
  * There is no market index of any kind. index_pct is None and index_source is
    NONE_AVAILABLE -- stated, so the critic can raise it as a blocking gap
    rather than silently skipping the inflation test.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FAMILY_BY_DETECTOR = {
    "Duplicate Invoice Recovery": "integrity",
    "Invoice Overbilling": "integrity",
    "Price Benchmark Variance": "position",
}

_CONTRACT_SQL = """
    SELECT contract_id, contract_start_date, contract_end_date, currency,
           jurisdiction, payment_terms, parent_contract_id
      FROM proc.bp_contract_master
     WHERE supplier_id = %s
     ORDER BY contract_start_date DESC NULLS LAST
     LIMIT 1
"""


def _confidence_for(facts_state: Optional[str]) -> str:
    """Map the finding's facts_state onto the evidence ladder.

    RESOLVED is ASSERTED, never CORROBORATED: it means numbers were parsed out
    of a document, not that a second source agreed with them. Upgrading here
    would let a parse masquerade as corroboration.
    """
    if str(facts_state or "").upper() == "RESOLVED":
        return "ASSERTED"
    return "UNASSESSED"


def _contract_context(supplier_id: Optional[str], conn) -> tuple[Dict[str, Any], str]:
    if not supplier_id or conn is None:
        return {}, "SUPPLIER_NOT_IN_CONTRACT_MASTER"
    try:
        cur = conn.cursor()
        cur.execute(_CONTRACT_SQL, (supplier_id,))
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - an unreadable corpus is a gap, not a crash
        logger.error("contract lookup failed for %s: %s", supplier_id, exc)
        return {}, "CONTRACT_LOOKUP_FAILED"

    if not row:
        return {}, "SUPPLIER_NOT_IN_CONTRACT_MASTER"

    return (
        {
            "contract_id": row[0],
            "start_date": row[1],
            "end_date": row[2],
            "currency": row[3],
            "jurisdiction": row[4],
            "payment_terms": row[5],
            "parent_contract_id": row[6],
            # Not columns anywhere in bp_contract_master. Stated as absent so
            # the critic raises them as gaps rather than treating silence as
            # "no escalation clause exists".
            "escalation_clause": None,
            "break_clause": None,
            "benchmarking_clause": None,
        },
        "RESOLVED",
    )


def assemble_candidate(finding: Dict[str, Any], conn) -> Dict[str, Any]:
    """Return the critic's input envelope for one finding."""
    calc = finding.get("calculation_details") or {}
    detector = finding.get("detector_type")
    supplier_id = finding.get("supplier_id")

    contract_context, resolution = _contract_context(supplier_id, conn)

    anchor_value = calc.get("benchmark_price")
    current_value = calc.get("actual_price")
    quantity = calc.get("quantity")

    return {
        "finding_id": finding.get("opportunity_ref_id"),
        "detector_id": detector,
        "detector_family": _FAMILY_BY_DETECTOR.get(detector, "unknown"),
        "claim": finding.get("claim"),
        "anchor": {
            "value": anchor_value,
            # The miner's anchor is grouped.groupby("item_id")["avg_price"].min()
            # -- the cheapest average any supplier ever charged, undated and
            # unnormalised. Naming it lets the critic test it properly.
            "kind": "cheapest_observed" if detector == "Price Benchmark Variance" else None,
            "date": None,
            "basis": None,
            "quantity": None,
            "line_value": None,
            "unit_price": anchor_value,
        },
        "current": {
            "value": current_value,
            "date": None,
            "basis": None,
            "quantity": quantity,
        },
        "delta": {
            "gap": (current_value - anchor_value)
            if isinstance(anchor_value, (int, float))
            and isinstance(current_value, (int, float))
            else None,
            "attributed_value": finding.get("financial_impact_gbp"),
            "currency": finding.get("currency"),
        },
        "evidence": {
            "source_records": list(finding.get("source_records") or []),
            "current_confidence": _confidence_for(finding.get("facts_state")),
            "anchor_confidence": "UNASSESSED",
            "contract_resolution": resolution,
            "facts_state": finding.get("facts_state"),
        },
        "contract_context": contract_context,
        "category_context": {
            "category_id": finding.get("category_id"),
            "index_pct": None,
            "index_source": "NONE_AVAILABLE",
        },
    }
```

- [ ] **Step 4: Write the subagent**

Create `src/agents/opportunity_evidence_agent.py`:

```python
"""Assemble the evidence one opportunity rests on. Judge nothing.

A subagent the Opportunity Critic calls as a tool. The split is deliberate: an
agent that both gathers and judges can quietly gather what supports the verdict
it already reached. This one has no verdict vocabulary at all.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.opportunity_critic.assemble import assemble_candidate

logger = logging.getLogger(__name__)


class OpportunityEvidenceAgent(BaseAgent):
    """Build the critic's input envelope for one finding."""

    AGENTIC_PLAN_STEPS = (
        "Read the finding and the source records it names.",
        "Resolve the anchor, and date it from the invoices behind it where possible.",
        "Resolve contract context by supplier, and report plainly when it cannot be resolved.",
        "Tag every fact with the confidence its provenance earns; upgrade nothing.",
    )

    def run(self, context: AgentContext) -> AgentOutput:
        finding = (context.input_data or {}).get("finding") or {}
        if not finding.get("opportunity_ref_id"):
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error="no finding supplied: evidence assembly needs opportunity_ref_id",
            )

        conn = None
        try:
            from src.services.db import get_conn

            with get_conn() as conn:
                envelope = assemble_candidate(finding, conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("evidence assembly failed for %s: %s",
                         finding.get("opportunity_ref_id"), exc)
            envelope = assemble_candidate(finding, None)

        return AgentOutput(status=AgentStatus.SUCCESS, data={"candidate": envelope})
```

Register it in `agent_definitions.json` — append to the `agents` array, using the next free `agentId` (the file currently holds 15 agents; check the maximum `agentId` in the file and add one):

```json
{
  "agentId": 16,
  "agentType": "OpportunityEvidenceAgent",
  "slug": "opportunity_evidence",
  "class_path": "agents.opportunity_evidence_agent.OpportunityEvidenceAgent",
  "description": "Assembles the evidence behind one opportunity: anchor, contract context and confidence tags. Judges nothing.",
  "role": "researcher",
  "capabilities": ["evidence_assembly"],
  "required_inputs": ["finding"],
  "output_fields": ["candidate"],
  "dependencies": [],
  "inputs": {"required": ["finding"], "optional": []},
  "outputs": ["candidate"],
  "version": "1.0.0"
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/agents/test_opportunity_evidence_agent.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Prove the no-inference guard fails**

In `_confidence_for`, change the `RESOLVED` branch to return `"CORROBORATED"`. Re-run — `test_resolved_facts_state_tags_evidence_asserted_not_corroborated` must go RED. Restore and confirm green. Record both.

- [ ] **Step 7: Commit**

```bash
git add src/agents/opportunity_evidence_agent.py \
        src/services/opportunity_critic/assemble.py \
        agent_definitions.json \
        tests/agents/test_opportunity_evidence_agent.py
git commit -m "feat(critic): the evidence subagent

Assembles, never judges. An unresolvable supplier reports
SUPPLIER_NOT_IN_CONTRACT_MASTER rather than fuzzy-matching across two key
spaces; a missing index reports NONE_AVAILABLE rather than being omitted."
```

---

### Task 6: The critique store

**Files:**
- Create: `src/services/opportunity_critic/store.py`
- Test: `tests/services/test_critic_store.py`

**Interfaces:**
- Consumes: the tables from Task 1.
- Produces: `record_critique(critique: dict, *, shadowed: bool) -> Optional[int]` returning the `critique_id`, or `None` when nothing was written.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_critic_store.py`:

```python
"""Persistence for critiques and their gaps.

The load-bearing behaviour is the return value: record_critique returns None
when it wrote nothing, and the caller must not suppress a finding on a None.
"""
from unittest.mock import MagicMock, patch

from src.services.opportunity_critic.store import record_critique

_CRITIQUE = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "verdict": "INVALID",
    "confidence": "UNASSESSED",
    "original_claim": "Paying 10% above benchmark",
    "critic_claim": None,
    "negotiator_note": "The benchmark is the cheapest price ever recorded.",
    "value": {"detector_proposed": 18330.80, "critic_addressable": None,
              "currency": "GBP", "basis": "annualised", "haircuts_applied": []},
    "lever": {"exists": False, "type": "none"},
    "tests": [{"test": "anchor_validity", "result": "INVALIDATE", "reason": "..."},
              {"test": "materiality", "result": "PASS", "reason": "..."}],
    "gaps": [
        {"gap_id": "G1", "test": "anchor_validity", "type": "DETECTOR_LOGIC",
         "what_is_missing": "Anchor selection uses .min() over avg_price",
         "why_it_matters": "Every variance from it is measured off a floor",
         "blocking": False, "resolves_to": "prevents the class of error",
         "likely_source": "detector fix", "owner_hint": "engineering",
         "effort": "LOW"},
    ],
    "prompt_version": 1,
    "policy_versions": {"opportunity_critic_thresholds": 1},
    "formula_versions": {"critic.annualised_rate": "1.0.0+abc123"},
    "run_id": "wf-9",
}


def _fake_conn():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (77,)
    conn.cursor.return_value = cur
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    return conn, cur


def test_returns_the_new_critique_id():
    conn, _ = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        assert record_critique(_CRITIQUE, shadowed=False) == 77


def test_every_test_result_is_written_including_passes():
    # "anchor_validity PASS" is the sentence that defends a finding in the
    # room, so a critique that stored only the tests that fired would be
    # useless to a negotiator.
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    written = [c for c in cur.execute.call_args_list
               if "bp_opportunity_critique" in c.args[0]][0]
    tests_blob = next(p for p in written.args[1]
                      if isinstance(p, str) and "anchor_validity" in p)
    assert "materiality" in tests_blob, "a PASS result was dropped"
    assert '"result": "PASS"' in tests_blob


def test_gaps_are_written_as_rows():
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    assert any("bp_opportunity_gap" in c.args[0] for c in cur.execute.call_args_list)


def test_a_failed_write_returns_none_so_nothing_gets_suppressed():
    conn, cur = _fake_conn()
    cur.execute.side_effect = RuntimeError("connection lost")
    with patch("src.services.db.get_conn", return_value=conn):
        assert record_critique(_CRITIQUE, shadowed=False) is None


def test_a_failed_write_never_raises():
    # Bookkeeping must not break the pipeline, exactly as
    # policy_observation.record does not.
    conn, cur = _fake_conn()
    cur.execute.side_effect = RuntimeError("connection lost")
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=True)  # must not raise


def test_versions_that_produced_the_verdict_are_persisted():
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    written = [c for c in cur.execute.call_args_list
               if "bp_opportunity_critique" in c.args[0]][0]
    assert any("critic.annualised_rate" in str(p) for p in written.args[1])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/services/test_critic_store.py -v
```

Expected: `ModuleNotFoundError: src.services.opportunity_critic.store`.

- [ ] **Step 3: Write the implementation**

Create `src/services/opportunity_critic/store.py`:

```python
"""Persist a critique and its gap register.

The return value is load-bearing. ``record_critique`` returns the new
``critique_id`` when it wrote, and ``None`` when it did not -- and the caller
must not suppress a finding on a ``None``. Suppressing without recording buys
neither the safety nor the measurement, which is precisely the reasoning at
services/guardrail.py:467, inverted.

Never raises. Bookkeeping must not break the pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_INSERT_CRITIQUE = """
    INSERT INTO proc.bp_opportunity_critique (
        opportunity_ref_id, detector_type, verdict, confidence,
        original_claim, critic_claim, negotiator_note,
        detector_proposed, critic_addressable, currency, value_basis,
        haircuts, lever, duplicate_of, tests,
        would_have_suppressed, shadowed,
        prompt_version, policy_versions, formula_versions, run_id
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s
    ) RETURNING critique_id
"""

_INSERT_GAP = """
    INSERT INTO proc.bp_opportunity_gap (
        critique_id, opportunity_ref_id, gap_id, test, gap_type,
        what_is_missing, why_it_matters, blocking, resolves_to,
        likely_source, owner_hint, effort, ordinal
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

#: Verdicts that would remove a finding from the page once enforcement is on.
SUPPRESSING_VERDICTS = ("INVALID", "DUPLICATE")


def record_critique(critique: Dict[str, Any], *, shadowed: bool) -> Optional[int]:
    """Write one critique and its gaps. Return the id, or None if nothing was written."""
    value = critique.get("value") or {}
    verdict = str(critique.get("verdict") or "")
    would_suppress = verdict in SUPPRESSING_VERDICTS

    params = (
        critique.get("opportunity_ref_id"),
        critique.get("detector_type"),
        verdict,
        critique.get("confidence"),
        critique.get("original_claim"),
        critique.get("critic_claim"),
        critique.get("negotiator_note"),
        value.get("detector_proposed"),
        value.get("critic_addressable"),
        value.get("currency"),
        value.get("basis"),
        json.dumps(value.get("haircuts_applied") or [], default=str),
        json.dumps(critique.get("lever") or {}, default=str),
        critique.get("duplicate_of"),
        json.dumps(critique.get("tests") or [], default=str),
        would_suppress,
        bool(shadowed),
        critique.get("prompt_version"),
        json.dumps(critique.get("policy_versions") or {}, default=str),
        json.dumps(critique.get("formula_versions") or {}, default=str),
        critique.get("run_id"),
    )

    try:
        from src.services.db import get_conn

        with get_conn() as conn:
            conn.autocommit = False
            cur = conn.cursor()
            try:
                cur.execute(_INSERT_CRITIQUE, params)
                critique_id = cur.fetchone()[0]
                for ordinal, gap in enumerate(critique.get("gaps") or []):
                    cur.execute(_INSERT_GAP, (
                        critique_id,
                        critique.get("opportunity_ref_id"),
                        gap.get("gap_id"),
                        gap.get("test"),
                        gap.get("type"),
                        gap.get("what_is_missing"),
                        gap.get("why_it_matters"),
                        bool(gap.get("blocking")),
                        gap.get("resolves_to"),
                        gap.get("likely_source"),
                        gap.get("owner_hint"),
                        gap.get("effort"),
                        ordinal,
                    ))
                conn.commit()
                return critique_id
            except Exception:
                conn.rollback()
                raise
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break the pipeline
        logger.error("record_critique(%s) failed: %s",
                     critique.get("opportunity_ref_id"), exc)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/services/test_critic_store.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Prove the no-record guard fails**

Change the `except` block's `return None` to `return -1`. Re-run — `test_a_failed_write_returns_none_so_nothing_gets_suppressed` must go RED. Restore and confirm green. Record both.

- [ ] **Step 6: Commit**

```bash
git add src/services/opportunity_critic/store.py tests/services/test_critic_store.py
git commit -m "feat(critic): persist critiques and their gap register

Returns None when nothing was written, so a caller cannot suppress a
finding it failed to record. Never raises: bookkeeping must not break the
pipeline."
```

---

### Task 7: The critic agent and its invariant guard

**Files:**
- Create: `src/agents/opportunity_critic_agent.py`
- Create: `src/services/opportunity_critic/invariants.py`
- Modify: `agent_definitions.json`
- Test: `tests/services/test_critic_invariants.py`
- Test: `tests/agents/test_opportunity_critic_agent.py`

**Interfaces:**
- Consumes: `load_thresholds`, `load_system_prompt` (Task 4); `assemble_candidate` (Task 5); `record_critique` (Task 6); the formulas (Tasks 2–3).
- Produces: `check_invariants(critique: dict) -> list[str]` returning violation messages (empty when clean); `OpportunityCriticAgent(BaseAgent)` with slug `opportunity_critic`.

- [ ] **Step 1: Write the failing invariant test**

Create `tests/services/test_critic_invariants.py`:

```python
"""The three rules a critique may never break.

Code refuses malformed output rather than repairing it: a critique quietly
corrected is a critique nobody knows was wrong.
"""
from src.services.opportunity_critic.invariants import check_invariants

_CLEAN = {
    "verdict": "VALID",
    "value": {"detector_proposed": 1000.0, "critic_addressable": 800.0},
    "gaps": [{"gap_id": "G1", "blocking": False}],
}


def test_a_clean_critique_has_no_violations():
    assert check_invariants(_CLEAN) == []


def test_valid_may_never_carry_a_value_above_the_detectors():
    bad = {"verdict": "VALID",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 1500.0},
           "gaps": []}
    violations = check_invariants(bad)
    assert any("above the detector" in v for v in violations)


def test_valid_at_exactly_the_detectors_value_is_allowed():
    ok = {"verdict": "VALID",
          "value": {"detector_proposed": 1000.0, "critic_addressable": 1000.0},
          "gaps": []}
    assert check_invariants(ok) == []


def test_valid_may_never_carry_a_blocking_gap():
    bad = {"verdict": "VALID",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 800.0},
           "gaps": [{"gap_id": "G1", "blocking": True}]}
    violations = check_invariants(bad)
    assert any("blocking gap" in v for v in violations)


def test_unassessed_must_carry_at_least_one_blocking_gap():
    bad = {"verdict": "UNASSESSED",
           "value": {"detector_proposed": 1000.0, "critic_addressable": None},
           "gaps": [{"gap_id": "G1", "blocking": False}]}
    violations = check_invariants(bad)
    assert any("at least one blocking gap" in v for v in violations)


def test_valid_reframed_is_held_to_the_value_ceiling_too():
    bad = {"verdict": "VALID_REFRAMED",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 1500.0},
           "gaps": []}
    assert any("above the detector" in v for v in check_invariants(bad))


def test_an_unknown_verdict_is_a_violation():
    assert check_invariants({"verdict": "PROBABLY_FINE", "value": {}, "gaps": []})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/services/test_critic_invariants.py -v
```

Expected: `ModuleNotFoundError: src.services.opportunity_critic.invariants`.

- [ ] **Step 3: Write the invariant guard**

Create `src/services/opportunity_critic/invariants.py`:

```python
"""The three rules a critique may never break.

These REFUSE. They do not repair. A critique quietly corrected is a critique
nobody knows was wrong, and the whole value of the critic is that its reasoning
can be inspected.
"""
from __future__ import annotations

from typing import Any, Dict, List

VERDICTS = ("VALID", "VALID_REFRAMED", "INVALID", "UNASSESSED", "DUPLICATE")

#: Verdicts that assert a finding is actionable, and so carry the value ceiling
#: and the no-blocking-gap rule.
_AFFIRMATIVE = ("VALID", "VALID_REFRAMED")


def check_invariants(critique: Dict[str, Any]) -> List[str]:
    """Return a list of violation messages. Empty means the critique is well-formed."""
    violations: List[str] = []

    verdict = str(critique.get("verdict") or "")
    if verdict not in VERDICTS:
        violations.append(f"unknown verdict {verdict!r}")

    value = critique.get("value") or {}
    proposed = value.get("detector_proposed")
    addressable = value.get("critic_addressable")
    gaps = critique.get("gaps") or []
    blocking = [g for g in gaps if g.get("blocking")]

    if verdict in _AFFIRMATIVE:
        if (isinstance(proposed, (int, float))
                and isinstance(addressable, (int, float))
                and addressable > proposed):
            violations.append(
                f"{verdict} carries {addressable} which is above the detector's "
                f"{proposed}; a critic may reduce a value, never raise it"
            )
        if blocking:
            violations.append(
                f"{verdict} carries a blocking gap ({blocking[0].get('gap_id')}); "
                "a verdict cannot be affirmative while something blocks it"
            )

    if verdict == "UNASSESSED" and not blocking:
        violations.append(
            "UNASSESSED carries no blocking gap; if nothing blocks the decision "
            "then it was decidable, and 'we could not say' needs a reason"
        )

    return violations
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/services/test_critic_invariants.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Prove each of the three guards fails**

Do this three times, recording the red output each time:

1. Delete the `addressable > proposed` block → `test_valid_may_never_carry_a_value_above_the_detectors` goes RED.
2. Delete the `if blocking:` block → `test_valid_may_never_carry_a_blocking_gap` goes RED.
3. Delete the `verdict == "UNASSESSED"` block → `test_unassessed_must_carry_at_least_one_blocking_gap` goes RED.

Restore after each and confirm 7 passed.

- [ ] **Step 6: Write the failing agent test**

Create `tests/agents/test_opportunity_critic_agent.py`:

```python
"""The critic agent: refuses without its governed prompt, refuses malformed output."""
import json
from unittest.mock import MagicMock, patch

from agents.base_agent import AgentContext, AgentStatus
from src.agents.opportunity_critic_agent import OpportunityCriticAgent

_FINDING = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "supplier_id": "SUP-MeridianSystems12",
    "financial_impact_gbp": 18330.80,
    "facts_state": "RESOLVED",
    "source_records": ["PO000967"],
    "calculation_details": {"actual_price": 4998.53, "benchmark_price": 4540.26,
                            "quantity": 40.0},
}

_WELL_FORMED = {
    "verdict": "INVALID",
    "confidence": "UNASSESSED",
    "critic_claim": None,
    "negotiator_note": "The benchmark is the cheapest price ever recorded, not a baseline.",
    "tests": [{"test": "anchor_validity", "result": "INVALIDATE", "reason": "fabricated"}],
    "value": {"detector_proposed": 18330.80, "critic_addressable": None,
              "currency": "GBP", "basis": "annualised", "haircuts_applied": []},
    "lever": {"exists": False, "type": "none"},
    "gaps": [{"gap_id": "G1", "test": "anchor_validity", "type": "DETECTOR_LOGIC",
              "what_is_missing": "anchor selection", "why_it_matters": "class of error",
              "blocking": False, "resolves_to": "prevents recurrence",
              "likely_source": "detector fix", "owner_hint": "engineering",
              "effort": "LOW"}],
}


def _context():
    return AgentContext(workflow_id="wf-1", agent_id="opportunity_critic",
                        user_id="tester", input_data={"finding": _FINDING})


def _agent(reason_result):
    nick = MagicMock()
    agent = OpportunityCriticAgent(nick)
    agent.reason = MagicMock(return_value={"answer": json.dumps(reason_result)})
    return agent


def test_refuses_to_run_without_its_governed_prompt():
    # The DB prompt is the system of record. A code default that silently wins
    # is how governance stops being governance.
    agent = _agent(_WELL_FORMED)
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=(None, None)):
        out = agent.run(_context())
    assert out.status == AgentStatus.FAILED
    assert "governed prompt" in (out.error or "")


def test_a_well_formed_critique_is_recorded():
    agent = _agent(_WELL_FORMED)
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         patch("src.agents.opportunity_critic_agent.record_critique",
               return_value=99) as rec:
        out = agent.run(_context())
    assert out.status == AgentStatus.SUCCESS
    assert out.data["critique_id"] == 99
    assert rec.called


def test_a_critique_breaking_an_invariant_is_refused_not_repaired():
    broken = dict(_WELL_FORMED, verdict="VALID")
    broken["value"] = dict(broken["value"], critic_addressable=99999.0)
    agent = _agent(broken)
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         patch("src.agents.opportunity_critic_agent.record_critique") as rec:
        out = agent.run(_context())
    assert out.status == AgentStatus.FAILED
    assert "above the detector" in (out.error or "")
    assert not rec.called, "a critique that breaks an invariant must not be persisted"


def test_unparseable_model_output_fails_rather_than_guessing():
    agent = _agent(_WELL_FORMED)
    agent.reason = MagicMock(return_value={"answer": "I think this one is fine, really"})
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         patch("src.agents.opportunity_critic_agent.record_critique") as rec:
        out = agent.run(_context())
    assert out.status == AgentStatus.FAILED
    assert not rec.called
```

- [ ] **Step 7: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/agents/test_opportunity_critic_agent.py -v
```

Expected: `ModuleNotFoundError: src.agents.opportunity_critic_agent`.

- [ ] **Step 8: Write the agent**

Create `src/agents/opportunity_critic_agent.py`:

```python
"""Decide whether a detected opportunity would survive a negotiator's scrutiny.

This agent holds no thresholds, no arithmetic and no prose of its own. Its
prompt is a bp_prompt row, its rules are bp_policy rows, and every number in
its verdict comes from a registered formula. What it contributes is judgement:
composing test results into a verdict a negotiator could act on, or declining
to.

Three things it will not do:

  * Run without its governed prompt. A code default that silently wins is how
    governance stops being governance.
  * Persist a critique that breaks an invariant. Code refuses; it does not
    repair. A critique quietly corrected is one nobody knows was wrong.
  * Guess at unparseable model output. "I could not read the answer" and "the
    finding is fine" are different sentences.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.opportunity_critic import governed
from src.services.opportunity_critic.assemble import assemble_candidate
from src.services.opportunity_critic.invariants import check_invariants
from src.services.opportunity_critic.store import record_critique

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpportunityCriticAgent(BaseAgent):
    """Critique one detected opportunity."""

    AGENTIC_PLAN_STEPS = (
        "Assemble the candidate's evidence through the evidence subagent.",
        "Resolve the governed thresholds and the governed system prompt.",
        "Run the seven tests, taking every number from the formula registry.",
        "Compose a verdict, and a gap register naming what would decide it.",
        "Refuse to persist anything that breaks the critic's invariants.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.policy_engine = getattr(agent_nick, "policy_engine", None)
        self.prompt_engine = getattr(agent_nick, "prompt_engine", None)

    def _parse(self, answer: Any) -> Optional[Dict[str, Any]]:
        """Pull the JSON critique out of the model's answer, or None."""
        if isinstance(answer, dict):
            return answer
        text = str(answer or "")
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def run(self, context: AgentContext) -> AgentOutput:
        finding = (context.input_data or {}).get("finding") or {}
        ref_id = finding.get("opportunity_ref_id")
        if not ref_id:
            return AgentOutput(status=AgentStatus.FAILED, data={},
                               error="no finding supplied: need opportunity_ref_id")

        prompt_text, prompt_version = governed.load_system_prompt(self.prompt_engine)
        if not prompt_text:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error=("no governed prompt: bp_prompt row 'opportunity_critic_system' "
                       "could not be resolved, and this agent has no code default"),
            )

        thresholds = governed.load_thresholds(self.policy_engine)

        conn = None
        try:
            from src.services.db import get_conn

            with get_conn() as conn:
                candidate = assemble_candidate(finding, conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("evidence assembly failed for %s: %s", ref_id, exc)
            candidate = assemble_candidate(finding, None)

        task = json.dumps({
            "candidate": candidate,
            "thresholds": {
                "index_band_pp": thresholds.index_band_pp,
                "materiality_floor_gbp": thresholds.materiality_floor_gbp,
                "relative_gap_floor": thresholds.relative_gap_floor,
                "anchor_stale_days": thresholds.anchor_stale_days,
                "friction_bands": thresholds.friction_bands,
            },
        }, default=str)

        result = self.reason(task, extra_system=prompt_text)
        critique = self._parse(
            result.get("answer") if isinstance(result, dict) else result)
        if critique is None:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error="critique could not be parsed as JSON; refusing to guess a verdict",
            )

        critique.setdefault("opportunity_ref_id", ref_id)
        critique.setdefault("detector_type", finding.get("detector_type"))
        critique.setdefault("original_claim", finding.get("claim"))
        critique["prompt_version"] = prompt_version
        critique["policy_versions"] = (
            {governed.POLICY_SLUG: (thresholds.source or {}).get("version")}
            if thresholds.source else {}
        )
        critique["formula_versions"] = self._formula_versions()
        critique["run_id"] = context.workflow_id

        violations = check_invariants(critique)
        if violations:
            logger.error("critique for %s broke invariants: %s", ref_id, violations)
            return AgentOutput(status=AgentStatus.FAILED, data={"violations": violations},
                               error="; ".join(violations))

        critique_id = record_critique(critique, shadowed=False)
        if critique_id is None:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error="critique could not be recorded; nothing is suppressed on a failed write",
            )

        return AgentOutput(status=AgentStatus.SUCCESS,
                           data={"critique_id": critique_id,
                                 "verdict": critique.get("verdict"),
                                 "critique": critique})

    @staticmethod
    def _formula_versions() -> Dict[str, str]:
        """Qualified versions of every critic formula, so a verdict can be dated."""
        try:
            from src.services.formulas import ensure_registered
            from src.services.formulas.registry import REGISTRY

            ensure_registered()
            return {name: spec.qualified_version
                    for name, spec in REGISTRY.items() if name.startswith("critic.")}
        except Exception as exc:  # noqa: BLE001
            logger.error("could not read formula versions: %s", exc)
            return {}
```

Register in `agent_definitions.json`:

```json
{
  "agentId": 17,
  "agentType": "OpportunityCriticAgent",
  "slug": "opportunity_critic",
  "class_path": "agents.opportunity_critic_agent.OpportunityCriticAgent",
  "description": "Decides whether a detected opportunity would survive a negotiator's scrutiny, and records what would decide it when it cannot.",
  "role": "validator",
  "capabilities": ["opportunity_critique"],
  "required_inputs": ["finding"],
  "output_fields": ["verdict", "critique_id"],
  "dependencies": ["policy_engine", "prompt_engine"],
  "inputs": {"required": ["finding"], "optional": ["policy_context"]},
  "outputs": ["verdict", "critique_id"],
  "version": "1.0.0",
  "elicit": [
    {"any_of": ["finding", "opportunity_ref_id"], "type": "text",
     "prompt": "Which opportunity should I critique? Give its ref id."}
  ]
}
```

- [ ] **Step 9: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/agents/test_opportunity_critic_agent.py \
                            tests/services/test_critic_invariants.py -v
```

Expected: 11 passed.

- [ ] **Step 10: Prove the refuse-don't-repair guard fails**

In `run`, change the `if violations:` block to log and continue rather than return. Re-run — `test_a_critique_breaking_an_invariant_is_refused_not_repaired` must go RED on the `not rec.called` assertion. Restore and confirm green. Record both.

- [ ] **Step 11: Commit**

```bash
git add src/agents/opportunity_critic_agent.py \
        src/services/opportunity_critic/invariants.py \
        agent_definitions.json \
        tests/agents/test_opportunity_critic_agent.py \
        tests/services/test_critic_invariants.py
git commit -m "feat(critic): the Opportunity Critic agent and its invariant guard

Refuses to run without its governed prompt, refuses to persist a critique
that breaks an invariant, and refuses to guess when it cannot parse the
model's answer. Code refuses; it does not repair."
```

---

### Task 8: Shadow mode

**Files:**
- Create: `src/services/opportunity_critic/shadow.py`
- Modify: `src/agents/opportunity_critic_agent.py`
- Modify: `src/api/main.py` (the `/health` payload)
- Test: `tests/services/test_critic_shadow.py`

**Interfaces:**
- Consumes: `Thresholds.shadow_detectors` (Task 4); `record_critique` (Task 6).
- Produces: `NEVER_SUPPRESS`, `is_shadowed(detector_type, thresholds) -> bool`, `may_suppress(finding, thresholds) -> tuple[bool, str]`, `shadow_status(thresholds) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_critic_shadow.py`:

```python
"""Shadow mode, in the shape services/guardrail.py established.

Per detector, never globally; every enrolment carries an expiry; a finding a
human has already advanced can never be suppressed whatever the config says.
"""
from datetime import datetime, timedelta, timezone

from src.services.opportunity_critic.governed import Thresholds
from src.services.opportunity_critic.shadow import (
    NEVER_SUPPRESS_STAGES, is_shadowed, may_suppress, shadow_status,
)

_FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
_PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _thresholds(detectors):
    return Thresholds(shadow_detectors=tuple(detectors))


def test_nothing_is_shadowed_by_default():
    assert is_shadowed("Price Benchmark Variance", _thresholds([])) is False


def test_an_enrolled_detector_is_shadowed():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    assert is_shadowed("Price Benchmark Variance", t) is True


def test_an_expired_enrolment_is_not_honoured():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _PAST}])
    assert is_shadowed("Price Benchmark Variance", t) is False


def test_an_enrolment_without_an_expiry_is_not_honoured():
    # Shadow mode must not become permanent by nobody getting round to it.
    t = _thresholds([{"detector": "Price Benchmark Variance"}])
    assert is_shadowed("Price Benchmark Variance", t) is False


def test_enrolment_does_not_leak_to_other_detectors():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    assert is_shadowed("Duplicate Invoice Recovery", t) is False


def test_a_finding_a_human_has_advanced_can_never_be_suppressed():
    # Someone is already acting on it; a model changing its mind must not pull
    # it out from under them. This is code, not config.
    for stage in NEVER_SUPPRESS_STAGES:
        allowed, reason = may_suppress({"stage": stage}, _thresholds([]))
        assert allowed is False
        assert stage in reason


def test_an_identified_finding_may_be_suppressed():
    allowed, _ = may_suppress({"stage": "identified"}, _thresholds([]))
    assert allowed is True


def test_a_shadowed_detector_may_not_suppress():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    allowed, reason = may_suppress(
        {"stage": "identified", "detector_type": "Price Benchmark Variance"}, t)
    assert allowed is False
    assert "shadow" in reason.lower()


def test_status_reports_the_enrolment_and_its_expiry():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    status = shadow_status(t)
    assert status["enrolled"][0]["detector"] == "Price Benchmark Variance"
    assert status["enrolled"][0]["until"] == _FUTURE
    assert "never_suppressed_stages" in status
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/services/test_critic_shadow.py -v
```

Expected: `ModuleNotFoundError: src.services.opportunity_critic.shadow`.

- [ ] **Step 3: Write the implementation**

Create `src/services/opportunity_critic/shadow.py`:

```python
"""Shadow mode for the critic, in the dialect services/guardrail.py established.

Three rules carried over verbatim, because each was paid for:

  * Enrolment is PER DETECTOR, never a global boolean. A global switch is
    fail-open, which is the defect P0 existed to fix.
  * Every enrolment carries an "until". An enrolment without one is not
    honoured, so shadow mode cannot become the permanent state by nobody
    getting round to it.
  * Some things can never be suppressed, and that list lives in code rather
    than config -- a list in a row can be edited, and the point of these is
    that they cannot.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: A finding past 'identified' has a human acting on it. A model changing its
#: mind must not pull it out from under them.
NEVER_SUPPRESS_STAGES = ("negotiation", "agreed", "realised")


def _expiry(entry: Dict[str, Any]) -> Optional[datetime]:
    raw = entry.get("until")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        logger.error("shadow enrolment for %r has an unreadable expiry %r",
                     entry.get("detector"), raw)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_shadowed(detector_type: Optional[str], thresholds) -> bool:
    """Whether this detector's verdicts are observed but not acted on."""
    if not detector_type:
        return False
    now = datetime.now(timezone.utc)
    for entry in getattr(thresholds, "shadow_detectors", ()) or ():
        if str(entry.get("detector") or "") != str(detector_type):
            continue
        expiry = _expiry(entry)
        if expiry and expiry > now:
            return True
    return False


def may_suppress(finding: Dict[str, Any], thresholds) -> Tuple[bool, str]:
    """Whether an INVALID verdict on this finding may remove it from the page."""
    stage = str(finding.get("stage") or "identified").lower()
    if stage in NEVER_SUPPRESS_STAGES:
        return False, (
            f"finding is at stage {stage}: a human is already acting on it and it "
            "can never be suppressed by the critic"
        )
    if is_shadowed(finding.get("detector_type"), thresholds):
        return False, (
            f"detector {finding.get('detector_type')!r} is in shadow mode: the "
            "verdict is recorded, the finding stands"
        )
    return True, "enforcing"


def shadow_status(thresholds) -> Dict[str, Any]:
    """What is enrolled and until when. Surfaced in /health.

    A control that is off must be visible, not something you discover by
    reading code.
    """
    now = datetime.now(timezone.utc)
    enrolled = []
    for entry in getattr(thresholds, "shadow_detectors", ()) or ():
        expiry = _expiry(entry)
        enrolled.append({
            "detector": entry.get("detector"),
            "until": entry.get("until"),
            "active": bool(expiry and expiry > now),
        })
    return {
        "enrolled": enrolled,
        "never_suppressed_stages": list(NEVER_SUPPRESS_STAGES),
    }
```

- [ ] **Step 4: Wire shadow mode into the agent**

In `src/agents/opportunity_critic_agent.py`, add the import:

```python
from src.services.opportunity_critic.shadow import may_suppress
```

Then replace the `record_critique(critique, shadowed=False)` call with:

```python
        allowed, shadow_reason = may_suppress(finding, thresholds)
        critique_id = record_critique(critique, shadowed=not allowed)
        if critique_id is None:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error="critique could not be recorded; nothing is suppressed on a failed write",
            )

        return AgentOutput(status=AgentStatus.SUCCESS,
                           data={"critique_id": critique_id,
                                 "verdict": critique.get("verdict"),
                                 "suppression": shadow_reason,
                                 "critique": critique})
```

- [ ] **Step 5: Surface it in `/health`**

In `src/api/main.py`, find the `/health` handler and add the critic's shadow status beside the existing `ask_auth` and `shadow_status` entries:

```python
    try:
        from src.services.opportunity_critic.governed import load_thresholds
        from src.services.opportunity_critic.shadow import shadow_status as _critic_shadow

        payload["critic_shadow"] = _critic_shadow(load_thresholds(policy_engine))
    except Exception as exc:  # noqa: BLE001 - health must not fail on a sub-report
        payload["critic_shadow"] = {"error": str(exc)}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/services/test_critic_shadow.py \
                            tests/agents/test_opportunity_critic_agent.py -v
```

Expected: 13 passed.

- [ ] **Step 7: Prove the never-suppress and expiry guards fail**

Twice, recording red output each time:

1. In `may_suppress`, delete the `NEVER_SUPPRESS_STAGES` block → `test_a_finding_a_human_has_advanced_can_never_be_suppressed` goes RED.
2. In `is_shadowed`, change `if expiry and expiry > now:` to `if True:` → both `test_an_expired_enrolment_is_not_honoured` and `test_an_enrolment_without_an_expiry_is_not_honoured` go RED.

Restore after each.

- [ ] **Step 8: Commit**

```bash
git add src/services/opportunity_critic/shadow.py \
        src/agents/opportunity_critic_agent.py \
        src/api/main.py \
        tests/services/test_critic_shadow.py
git commit -m "feat(critic): shadow mode, per detector, with an expiry that is honoured

Mirrors guardrail's dialect: never a global boolean, every enrolment carries
an until, and the never-suppress list lives in code because a list in a row
can be edited."
```

---

### Task 9: The report, the workflow node, and live verification

**Files:**
- Create: `scripts/critic_report.py`
- Modify: `src/orchestration/workflow_definitions.py`
- Test: `tests/orchestration/test_critic_workflow_node.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable report; an `opportunity_critic` node in `build_opportunity_workflow()`.

- [ ] **Step 1: Write the failing workflow test**

Create `tests/orchestration/test_critic_workflow_node.py`:

```python
"""The critic runs after mining, and shadow keeps that safe."""
from src.orchestration.workflow_definitions import build_opportunity_workflow


def test_the_critic_is_a_node_in_the_opportunity_workflow():
    graph = build_opportunity_workflow()
    slugs = [node.agent_type for node in graph.nodes.values()]
    assert "opportunity_critic" in slugs


def test_the_critic_runs_after_mining_not_before():
    graph = build_opportunity_workflow()
    critic = next(n for n in graph.nodes.values()
                  if n.agent_type == "opportunity_critic")
    assert "opportunity_mining" in (critic.depends_on or [])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/orchestration/test_critic_workflow_node.py -v
```

Expected: FAIL — `opportunity_critic` is not in the graph.

- [ ] **Step 3: Add the node**

In `src/orchestration/workflow_definitions.py`, inside `build_opportunity_workflow()`, add a node after the `opportunity_mining` node. Match the surrounding node-construction style exactly — read `workflow_definitions.py:323-350` first and mirror the argument names used there:

```python
    graph.add_node(
        name="opportunity_critique",
        agent_type="opportunity_critic",
        depends_on=["opportunity_mining"],
        description=(
            "Decide whether each mined opportunity would survive a negotiator's "
            "scrutiny. Records a verdict and a gap register; suppresses nothing "
            "while its detector is in shadow mode."
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./venv/bin/python -m pytest tests/orchestration/test_critic_workflow_node.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Write the report script**

Create `scripts/critic_report.py`:

```python
#!/usr/bin/env python
"""What the critic would have done, for a date range.

The point of shadow mode is that this script can be run before anything is
suppressed. It answers: how many were critiqued, how many would have come off
the page, which test killed them, and what the gap register says to fix first.

Usage:
    ./venv/bin/python scripts/critic_report.py --since 2026-09-01
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env"))

from src.services.db import get_conn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="1970-01-01")
    args = parser.parse_args()

    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT verdict, count(*), coalesce(sum(detector_proposed), 0)
              FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s
             GROUP BY verdict ORDER BY 2 DESC
        """, (args.since,))
        rows = cur.fetchall()
        print(f"\nCRITIQUES since {args.since}")
        print(f"{'verdict':<18} {'count':>7} {'detector value':>18}")
        for verdict, count, value in rows:
            print(f"{verdict:<18} {count:>7} {value:>18,.2f}")
        print(f"{'TOTAL':<18} {sum(r[1] for r in rows):>7} "
              f"{sum(r[2] for r in rows):>18,.2f}")

        cur.execute("""
            SELECT detector_type, count(*),
                   coalesce(sum(detector_proposed) FILTER (WHERE would_have_suppressed), 0)
              FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s AND would_have_suppressed
             GROUP BY detector_type ORDER BY 2 DESC
        """, (args.since,))
        print("\nWOULD HAVE BEEN SUPPRESSED, by detector")
        for detector, count, value in cur.fetchall():
            print(f"  {str(detector):<32} {count:>6}  {value:>16,.2f}")

        cur.execute("""
            SELECT t->>'test' AS test, count(*)
              FROM proc.bp_opportunity_critique c,
                   LATERAL jsonb_array_elements(c.tests) AS t
             WHERE c.critiqued_at >= %s
               AND t->>'result' = 'INVALIDATE'
             GROUP BY 1 ORDER BY 2 DESC
        """, (args.since,))
        print("\nWHICH TEST KILLED IT")
        for test, count in cur.fetchall():
            print(f"  {str(test):<28} {count:>6}")

        cur.execute("""
            SELECT g.what_is_missing, g.owner_hint, g.effort, count(*)
              FROM proc.bp_opportunity_gap g
              JOIN proc.bp_opportunity_critique c USING (critique_id)
             WHERE c.critiqued_at >= %s AND g.blocking
             GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 10
        """, (args.since,))
        print("\nBLOCKING GAPS, most findings first")
        for missing, owner, effort, count in cur.fetchall():
            print(f"  {count:>5} x [{effort or '?':^6}] {str(owner):<18} {missing}")

        cur.execute("""
            SELECT count(*) FROM proc.bp_opportunity_critique
             WHERE critiqued_at >= %s AND shadowed
        """, (args.since,))
        print(f"\nshadowed (recorded, nothing suppressed): {cur.fetchone()[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Apply the migrations to bp_testdb**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-09_opportunity_critique.sql
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-09_critic_governance.sql
```

Expected: `COMMIT` from each, no errors. Re-run both to confirm idempotence — the second run must also succeed and change nothing.

- [ ] **Step 7: Run the critic over the live corpus in shadow**

Enrol all three detectors in shadow first, so nothing can be suppressed during the first live run:

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,shadow_detectors}',
       '[{\"detector\": \"Duplicate Invoice Recovery\", \"until\": \"2026-10-09T00:00:00Z\"},
         {\"detector\": \"Invoice Overbilling\",        \"until\": \"2026-10-09T00:00:00Z\"},
         {\"detector\": \"Price Benchmark Variance\",   \"until\": \"2026-10-09T00:00:00Z\"}]'::jsonb)
 WHERE policy_name = 'opportunity_critic_thresholds';"
```

Then run the critic over the 308 live findings and produce the report:

```bash
./venv/bin/python scripts/critic_report.py --since 2026-09-01
```

- [ ] **Step 8: Check the result against the prediction, honestly**

Spec section 8.1 predicts: 300 duplicates critiqued and mostly survivable; 6 Invoice Overbilling forced to UNASSESSED by `facts_state = INDETERMINATE`; 2 Price Benchmark Variance INVALID on the anchor; one gap — supplier identity — blocking almost everything else.

**Report what the run actually said, not the prediction.** If it differs materially, that is the finding; write it into the task notes and raise it rather than adjusting the story to fit. A GPU tool-loop node takes roughly 2–4 minutes per finding, so run a bounded sample first (say 20 findings) before committing to all 308.

- [ ] **Step 9: Commit**

```bash
git add scripts/critic_report.py \
        src/orchestration/workflow_definitions.py \
        tests/orchestration/test_critic_workflow_node.py
git commit -m "feat(critic): shadow report, and the critic as a workflow node

Runs after opportunity_mining. All three detectors enrolled in shadow until
2026-10-09, so the first live pass measures rather than suppresses."
```

---

## Self-Review Notes

**Spec coverage.** Every numbered spec section maps to a task: §7.1 critic agent → Task 7; §7.2 evidence subagent → Task 5; §7.3 division of authority → Tasks 2/3 (registry calculates), 7 (agent judges, code refuses); §8 seven tests → Tasks 2, 3 (calculations) and 4 (rules); §8.1 expected distribution → Task 9 Step 8; §9.1/§9.2 tables → Tasks 1, 6; §10 shadow mode → Task 8; §11 stage 1 → Task 8 (stages 2 and 3 are deliberately out of this plan, see below); §12 testing → the prove-the-guard-fails step in every task; §13 acceptance criteria 1–8 → Tasks 5/7 (1), 4 (2), 2/3 (3), 7 (4), 1/6 (5), 8 (6), 9 (7), 8 (8).

**Deliberately not in this plan.** Spec §11 stages 2 (advisory display) and 3 (enforcing) are follow-on work. Stage 2 changes `opportunity_dashboard.py` and the UI, and the spec recommends holding at stage 2 for a period anyway — so it should be its own plan, written once the first shadow report says what there is to display. This plan ends with verdicts recorded and nothing suppressed, which is a complete, testable deliverable on its own.

**Open questions carried from spec §14** — none block this plan. `critic.excess_over_index` returns UNASSESSED without an index by design, and the gap register is what quantifies the cost of leaving that open.
