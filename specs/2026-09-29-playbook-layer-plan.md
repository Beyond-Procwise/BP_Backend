# Playbook Layer (P4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the system a way to say "a human expert already wrote the strategy for a finding like this one", and put that strategy in front of a person as a proposal rather than running it.

**Architecture:** Two additive tables (`proc.bp_playbook`, `proc.bp_playbook_proposal`). A playbook points at a graph already saved in `proc.bp_agent_workflow`; nothing new executes anything. A scheduler sweep normalises open findings from two separate stores into one `Finding` shape, a pure deterministic selector picks the single most-specific active playbook (and refuses on a tie), and a proposer writes one idempotent proposal row plus an audit event. A person approves a proposal through an endpoint, which reuses the **existing** workflow run path unchanged.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, PostgreSQL (schema `proc`), pytest. No new dependencies.

**Spec:** `specs/2026-09-27-playbook-layer-design.md` — read it before Task 1. This plan implements it and does not re-argue it.

## Global Constraints

- **Shared checkout.** Another session's work sits in this repo's index and working tree. **Never** `git add -A`, `git add .`, a bare `git commit`, or `git stash`. Every commit in this plan is `git commit -o <explicit paths> -m "..."`. Run `git status` before each commit and confirm the paths you are about to name are yours.
- **Commit trailer.** Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Branch.** Work stays on `Development`. Never push to `main`.
- **Python interpreter.** `./venv/bin/python` for tests. `./.venv` is the runtime venv and is a different package set — do not use it for pytest.
- **Test environment.** Load `.env` first and isolate from the GPU and the real Ollama:
  ```sh
  set -a; . ./.env; set +a
  export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 \
         OLLAMA_HOST=http://127.0.0.1:9 OLLAMA_CLOUD_BASE_URL=http://127.0.0.1:9 \
         OLLAMA_CLOUD_API_KEY=
  ```
- **Never run two pytest suites concurrently** on this box. They contend on the live DB and both go spuriously red.
- **Fake DB by default.** Under pytest `get_conn()` returns an in-memory fake unless `PROCWISE_TEST_LIVE_DB=1`. Unit tests in this plan inject rows and never touch a connection. Live tests carry `pytest.mark.skipif` on that flag, exactly as `tests/migrations/test_2026_09_27_bp_rule.py` does.
- **Naming.** Tables `bp_*`, indexes `ix_bp_*` / `ux_bp_*`. Action names are `domain.verb`, lower case, no plurals.
- **Migrations live in `deploy/sql/`** as a pair: `2026-09-29_bp_playbook.sql` and `2026-09-29_bp_playbook_rollback.sql`. Additive and idempotent, safe to re-run. Applied to **both** `bp_testdb` and `bp_sqldb`.
- **Databases.** `.env` points at `bp_testdb`. `bp_sqldb` is the other target and must be migrated too, by name, in the same session.
- **Nothing auto-executes.** No code path in this plan may start a workflow without a human having called an approve endpoint. If a task tempts you toward auto-run, stop and ask.
- **Do not touch** `proc.bp_policy`'s three `supplier_ranking` rows, and do not attempt to unify `bp_detection_finding.rule_id` with `proc.bp_rule.detector_slug`. Both are explicitly out of scope.

## Review Focus

These are the input classes the spec implies but does not give a task of its own. Each has a test pinned to the task that owns the code; the line here says which.

1. **A `trigger_match` value whose JSON type differs from the column's.** `{"blocks_promotion": "true"}` (string) against a boolean column, or `{"severity": "Critical"}` against `'critical'`. A playbook that silently never fires is the exact failure this layer exists to avoid. Comparison must be canonical (case-folded string form, booleans as `true`/`false`), and Task 4 tests it.
2. **A NULL finding attribute against a present match key.** `doc_type IS NULL` must not match `{"doc_type": "invoice"}`, and a playbook must not be allowed to store a `null` match value at all (it would silently mean "match nothing"). Task 2 refuses the write; Task 4 tests the non-match.
3. **A playbook that leaves `active` after it has already proposed.** Retiring or editing a playbook must stop it proposing on the next sweep, while its existing proposals stay decidable. Task 3 tests the store excludes it; Task 8 tests an existing proposal is still decidable.
4. **The same numeric id in both finding stores.** `bp_detection_finding.finding_id = 123` and `bp_opportunity.opportunity_id = '123'` both land in one TEXT column. The unique index keys on `(playbook_id, finding_source, finding_id)`, so both must be able to coexist. Task 5 tests it.
5. **A playbook edited between sweeps.** A version bump must not re-propose for a finding already proposed — the unique index deliberately ignores version. Task 5 tests that the second sweep after an edit adds no row.

---

## File Structure

**Created**

| File | Responsibility |
|---|---|
| `deploy/sql/2026-09-29_bp_playbook.sql` | Both tables and their indexes. Idempotent. |
| `deploy/sql/2026-09-29_bp_playbook_rollback.sql` | Drops both tables and the two policy rows. |
| `src/services/playbooks/__init__.py` | Package marker, re-exports the public names. |
| `src/services/playbooks/finding_source.py` | `Finding`, the per-source match-field allow-lists, row → `Finding` normalisers, `validate_trigger_match`. |
| `src/services/playbooks/store.py` | `Playbook`, `PlaybookStore`, `PlaybookStoreUnavailable`. Loads active playbooks. |
| `src/services/playbooks/selector.py` | Pure `select()` / `tied_candidates()`. No I/O. |
| `src/services/playbooks/proposer.py` | `propose()` — one idempotent row plus the audit event. |
| `src/services/playbooks/sweep.py` | `sweep()` — batch over open findings, select, propose. Returns counts. |
| `src/repositories/playbook_repo.py` | Playbook CRUD under the lifecycle, and proposal reads/decisions for the router. |
| `src/api/routers/playbooks.py` | Playbook CRUD endpoints and the three proposal endpoints. |
| `tests/services/playbooks/test_finding_source.py` | Normalisation and match-key validation. |
| `tests/services/playbooks/test_store.py` | Active-only loading; unreadable raises; empty does not. |
| `tests/services/playbooks/test_selector.py` | Exact match, most-specific wins, tie proposes nothing. |
| `tests/services/playbooks/test_proposer.py` | Idempotency, cross-source ids, version bump. |
| `tests/services/playbooks/test_sweep.py` | Counts, batching, ambiguity recorded. |
| `tests/api/test_playbooks_router.py` | Lifecycle, self-approval bar, proposal decisions. |
| `tests/migrations/test_2026_09_29_bp_playbook.py` | Live: tables, indexes, constraint, idempotency. Both DBs. |
| `tests/services/playbooks/test_guard_proofs.py` | Each guard broken on purpose and watched fail. |

**Modified**

| File | Change |
|---|---|
| `src/services/actions.py` | Two names in `ACTIONS`, both class `configure`. |
| `src/api/routers/agent_workflows.py` | Extract the run path into a module-level `start_run()` both routers call. |
| `src/api/main.py` | Register the playbooks router. |
| `src/services/backend_scheduler.py` | Register the sweep job. |

---

## Task 1: The two tables

**Files:**
- Create: `deploy/sql/2026-09-29_bp_playbook.sql`
- Create: `deploy/sql/2026-09-29_bp_playbook_rollback.sql`
- Test: `tests/migrations/test_2026_09_29_bp_playbook.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `proc.bp_playbook` and `proc.bp_playbook_proposal` in both databases, with the column names every later task depends on.

- [ ] **Step 1: Write the migration**

Create `deploy/sql/2026-09-29_bp_playbook.sql`:

```sql
-- 2026-09-29  Playbook layer (conformance P4).
--
-- A playbook is human-authored expert strategy: when a finding like this
-- appears, run the graph an expert already drew. It is PROPOSED, never run.
-- The selector queues it; a person approves; only then does it execute.
--
-- Nothing is written to proc.bp_agent_workflow -- a playbook references it.
--
-- Additive and idempotent. Safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_playbook (
    playbook_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_name     TEXT        NOT NULL,
    description       TEXT,
    -- Which finding store this playbook watches.
    trigger_source    TEXT        NOT NULL
        CHECK (trigger_source IN ('detection_finding', 'opportunity')),
    -- Equality match against that store's own columns. Keys are validated at
    -- write time by services.playbooks.finding_source.validate_trigger_match;
    -- an unknown key is refused rather than stored, because a key that matches
    -- nothing is a playbook that silently never fires.
    --   detection_finding: rule_id, category, severity, doc_type, blocks_promotion
    --   opportunity:       detector_type, supplier_id, category_id
    trigger_match     JSONB       NOT NULL DEFAULT '{}',
    -- The expert's strategy: a graph already drawn and saved.
    agent_workflow_id BIGINT      NOT NULL REFERENCES proc.bp_agent_workflow (workflow_id),
    -- Extra static inputs merged into the run payload on execution.
    params            JSONB       NOT NULL DEFAULT '{}',
    playbook_status   TEXT        NOT NULL DEFAULT 'draft'
        CHECK (playbook_status IN ('draft','pending_approval','active','retired')),
    version           INTEGER     NOT NULL DEFAULT 1,
    authored_by       TEXT        NOT NULL,
    approved_by       TEXT,
    approved_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by  TEXT        NOT NULL DEFAULT 'system',
    -- Active means approved. The two cannot drift apart.
    CONSTRAINT ck_bp_playbook_active_is_approved
        CHECK (playbook_status <> 'active' OR approved_by IS NOT NULL)
);

COMMENT ON TABLE proc.bp_playbook IS
    'Human-authored strategy. Selected for a finding and PROPOSED to a person; never auto-run.';
COMMENT ON COLUMN proc.bp_playbook.trigger_match IS
    'Equality match on the trigger_source store''s own columns. Empty means catch-all, and loses to any more specific playbook.';

CREATE INDEX IF NOT EXISTS ix_bp_playbook_status
    ON proc.bp_playbook (playbook_status);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_source
    ON proc.bp_playbook (trigger_source, playbook_status);

CREATE TABLE IF NOT EXISTS proc.bp_playbook_proposal (
    proposal_id     BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_id     BIGINT      NOT NULL REFERENCES proc.bp_playbook (playbook_id),
    finding_source  TEXT        NOT NULL
        CHECK (finding_source IN ('detection_finding', 'opportunity')),
    -- TEXT because the two stores disagree on type: bp_detection_finding
    -- .finding_id is BIGINT (stored here as ::text) and bp_opportunity
    -- .opportunity_id is VARCHAR. Deliberately no foreign key -- one column
    -- cannot reference two tables, and finding_source says which.
    finding_id      TEXT        NOT NULL,
    deal_id         TEXT,
    proposal_status TEXT        NOT NULL DEFAULT 'proposed'
        CHECK (proposal_status IN ('proposed','approved','rejected','executed','superseded')),
    -- Which match keys fired, and what the finding's values were. A proposal
    -- must be re-derivable from source, like a decision.
    evidence        JSONB       NOT NULL DEFAULT '{}',
    run_id          TEXT        REFERENCES proc.bp_workflow_run (run_id),
    proposed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by      TEXT,
    decided_at      TIMESTAMPTZ,
    decision_reason TEXT
);

COMMENT ON TABLE proc.bp_playbook_proposal IS
    'One playbook recommended for one finding, awaiting a person. Executed only after approval.';

-- Idempotency. A sweep that runs twice proposes once. Deliberately keyed on
-- playbook_id and NOT on version: editing a playbook must not re-raise a
-- proposal for a finding already proposed.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_playbook_proposal_finding
    ON proc.bp_playbook_proposal (playbook_id, finding_source, finding_id);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_proposal_status
    ON proc.bp_playbook_proposal (proposal_status, proposed_at DESC);

COMMIT;
```

- [ ] **Step 2: Write the rollback**

Create `deploy/sql/2026-09-29_bp_playbook_rollback.sql`:

```sql
-- Rollback for 2026-09-29_bp_playbook.sql.
--
-- Run this only alongside reverting the code: the sweep and the playbooks
-- router read these tables and will fail loudly without them.
--
-- Proposals are dropped with the playbooks. They are advisory records of what
-- the system recommended, not a financial ledger; an executed proposal's run
-- survives independently in proc.bp_workflow_run.

BEGIN;

DROP TABLE IF EXISTS proc.bp_playbook_proposal;
DROP TABLE IF EXISTS proc.bp_playbook;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'playbook_authority'
   AND created_by = 'bp_playbook_migration_2026_09_29';

COMMIT;
```

- [ ] **Step 3: Write the failing live test**

Create `tests/migrations/test_2026_09_29_bp_playbook.py`:

```python
"""proc.bp_playbook and proc.bp_playbook_proposal, in both live databases.

Needs PROCWISE_TEST_LIVE_DB=1; without it pytest uses a fake DB that cannot
answer any of these questions.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)

DATABASES = ("bp_testdb", "bp_sqldb")


def _connect(dbname, readonly=True):
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=dbname,
        connect_timeout=10,
    )
    if readonly:
        conn.set_session(readonly=True)
    return conn


@pytest.mark.parametrize("dbname", DATABASES)
def test_both_tables_exist(dbname):
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'proc' "
            "AND table_name IN ('bp_playbook', 'bp_playbook_proposal')"
        )
        found = {r[0] for r in cur.fetchall()}
    assert found == {"bp_playbook", "bp_playbook_proposal"}


@pytest.mark.parametrize("dbname", DATABASES)
def test_indexes_exist(dbname):
    """The unique proposal index especially -- without it a sweep duplicates."""
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'proc' "
            "AND tablename IN ('bp_playbook', 'bp_playbook_proposal')"
        )
        found = {r[0] for r in cur.fetchall()}
    for name in (
        "ix_bp_playbook_status",
        "ix_bp_playbook_source",
        "ux_bp_playbook_proposal_finding",
        "ix_bp_playbook_proposal_status",
    ):
        assert name in found, f"{name} missing in {dbname}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_active_requires_an_approver(dbname):
    """An active playbook with no approved_by must be rejected by the database,
    not merely discouraged by the endpoint."""
    with _connect(dbname, readonly=False) as conn:
        cur = conn.cursor()
        cur.execute("SELECT workflow_id FROM proc.bp_agent_workflow LIMIT 1")
        row = cur.fetchone()
        assert row, "no workflow to reference"
        with pytest.raises(Exception) as exc:
            cur.execute(
                "INSERT INTO proc.bp_playbook "
                "(playbook_name, trigger_source, agent_workflow_id, "
                " playbook_status, authored_by) "
                "VALUES ('ck probe', 'detection_finding', %s, 'active', 'tester')",
                (row[0],),
            )
        assert "ck_bp_playbook_active_is_approved" in str(exc.value)
        conn.rollback()


@pytest.mark.parametrize("dbname", DATABASES)
def test_finding_id_is_text_so_both_stores_fit(dbname):
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'proc' AND table_name = 'bp_playbook_proposal' "
            "AND column_name = 'finding_id'"
        )
        assert cur.fetchone()[0] == "text"
```

- [ ] **Step 4: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
  tests/migrations/test_2026_09_29_bp_playbook.py -v -p no:randomly
```
Expected: FAIL — `found == set()`, the tables do not exist yet.

- [ ] **Step 5: Apply to both databases**

```sh
set -a; . ./.env; set +a
for DB in bp_testdb bp_sqldb; do
  echo "=== $DB ==="
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" \
    -U "$DB_USER" -d "$DB" -v ON_ERROR_STOP=1 \
    -f deploy/sql/2026-09-29_bp_playbook.sql
done
```

- [ ] **Step 6: Run it twice — pass, and idempotent**

```sh
PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
  tests/migrations/test_2026_09_29_bp_playbook.py -v -p no:randomly
```
Expected: PASS, 8 tests (4 × 2 databases).

Then re-apply the migration to both databases with the same loop from Step 5 and re-run the tests. Expected: no error from psql, and PASS again. A migration that is not re-runnable is not idempotent, whatever the file says.

- [ ] **Step 7: Commit**

```bash
git status --short deploy/sql tests/migrations
git commit -o deploy/sql/2026-09-29_bp_playbook.sql \
             deploy/sql/2026-09-29_bp_playbook_rollback.sql \
             tests/migrations/test_2026_09_29_bp_playbook.py \
  -m "feat(playbook): the two tables a proposed strategy needs

A playbook points at a graph an expert already drew and is proposed for a
finding, never run by the system. proc.bp_playbook carries the authorship
lifecycle and a database-level check that active implies approved;
proc.bp_playbook_proposal carries the queue, keyed uniquely on
(playbook_id, finding_source, finding_id) so a sweep that runs twice
proposes once.

Applied to bp_testdb and bp_sqldb, with a rollback.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: One shape for two finding vocabularies

`bp_detection_finding.rule_id` holds triage check codes; `bp_opportunity.detector_type`
holds opportunity detector names. They are separate namespaces and this layer never
crosses them. This module keeps them apart while giving the selector one shape.

**Files:**
- Create: `src/services/playbooks/__init__.py`
- Create: `src/services/playbooks/finding_source.py`
- Test: `tests/services/playbooks/__init__.py` (empty), `tests/services/playbooks/test_finding_source.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Finding(source: str, finding_id: str, deal_id: Optional[str], attrs: Dict[str, Any])` — frozen dataclass
  - `DETECTION_FINDING = "detection_finding"`, `OPPORTUNITY = "opportunity"`
  - `MATCH_FIELDS: Dict[str, Tuple[str, ...]]`
  - `normalise(source: str, row: Mapping[str, Any]) -> Finding`
  - `validate_trigger_match(source: str, match: Mapping[str, Any]) -> Dict[str, Any]` — raises `ValueError`
  - `canonical(value: Any) -> Optional[str]`
  - `OPEN_SQL: Dict[str, str]` — the query that yields open rows per source

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/__init__.py` as an empty file, then
`tests/services/playbooks/test_finding_source.py`:

```python
"""Two stores, two vocabularies, one shape -- and they never cross."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.finding_source import (  # noqa: E402
    DETECTION_FINDING,
    OPPORTUNITY,
    MATCH_FIELDS,
    canonical,
    normalise,
    validate_trigger_match,
)


def test_detection_finding_row_normalises():
    finding = normalise(
        DETECTION_FINDING,
        {
            "finding_id": 4211,
            "rule_id": "cumulative_total",
            "category": "overbilling",
            "severity": "critical",
            "doc_type": "invoice",
            "blocks_promotion": True,
            "deal_id": "D-900",
            "notes": "ignored -- not a match field",
        },
    )
    assert finding.source == DETECTION_FINDING
    assert finding.finding_id == "4211"          # BIGINT arrives as int, stored as text
    assert finding.deal_id == "D-900"
    assert finding.attrs == {
        "rule_id": "cumulative_total",
        "category": "overbilling",
        "severity": "critical",
        "doc_type": "invoice",
        "blocks_promotion": True,
    }


def test_opportunity_row_normalises():
    finding = normalise(
        OPPORTUNITY,
        {
            "opportunity_id": "OPP-17",
            "detector_type": "Duplicate Invoice Recovery",
            "supplier_id": "SUP-3",
            "category_id": None,
            "deal_id": "D-900",
            "financial_impact_gbp": 1200,
        },
    )
    assert finding.finding_id == "OPP-17"
    assert finding.attrs == {
        "detector_type": "Duplicate Invoice Recovery",
        "supplier_id": "SUP-3",
        "category_id": None,
    }


def test_the_two_vocabularies_do_not_cross():
    """rule_id is not a field an opportunity playbook may match on, and
    detector_type is not one a detection-finding playbook may match on."""
    assert "rule_id" not in MATCH_FIELDS[OPPORTUNITY]
    assert "detector_type" not in MATCH_FIELDS[DETECTION_FINDING]
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(OPPORTUNITY, {"rule_id": "quantity"})
    assert "rule_id" in str(exc.value)
    assert "detector_type" in str(exc.value)     # names what IS allowed


def test_unknown_source_is_refused():
    with pytest.raises(ValueError):
        normalise("invoices", {"finding_id": 1})
    with pytest.raises(ValueError):
        validate_trigger_match("invoices", {})


def test_unknown_match_key_is_refused_not_stored():
    """A key that matches no column is a playbook that silently never fires."""
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(DETECTION_FINDING, {"severity": "critical", "sevrity": "high"})
    assert "sevrity" in str(exc.value)


def test_null_match_value_is_refused():
    """{"doc_type": null} would mean "match a finding whose doc_type is unset",
    which is never what an author means and always matches almost nothing."""
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(DETECTION_FINDING, {"doc_type": None})
    assert "doc_type" in str(exc.value)


def test_empty_match_is_a_legitimate_catch_all():
    assert validate_trigger_match(DETECTION_FINDING, {}) == {}


def test_validate_returns_the_match_unchanged():
    match = {"rule_id": "quantity", "severity": "critical"}
    assert validate_trigger_match(DETECTION_FINDING, match) == match


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Critical", "critical"),
        ("critical", "critical"),
        (True, "true"),
        ("true", "true"),
        (False, "false"),
        (123, "123"),
        ("  padded  ", "padded"),
        (None, None),
    ],
)
def test_canonical_folds_both_sides_to_one_form(value, expected):
    """A boolean column compared against a JSON string, or a capitalised
    severity, must not silently fail to match."""
    assert canonical(value) == expected
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_finding_source.py -v -p no:randomly
```
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.services.playbooks'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/playbooks/__init__.py`:

```python
"""The playbook layer: human-authored strategy, proposed and never auto-run.

A playbook says *when a finding like this appears, the strategy an expert wrote
for it is that one*. Selecting a playbook queues a proposal; a person approves
it; only then does the existing workflow run path execute anything. Nothing in
this package starts a workflow.
"""

from .finding_source import (
    DETECTION_FINDING,
    OPPORTUNITY,
    Finding,
    MATCH_FIELDS,
    normalise,
    validate_trigger_match,
)

__all__ = [
    "DETECTION_FINDING",
    "OPPORTUNITY",
    "Finding",
    "MATCH_FIELDS",
    "normalise",
    "validate_trigger_match",
]
```

Create `src/services/playbooks/finding_source.py`:

```python
"""Normalise a row from either finding store into one shape.

Two stores, two vocabularies. ``proc.bp_detection_finding.rule_id`` holds triage
check codes (``cumulative_total``, ``line_arithmetic``); ``proc.bp_opportunity
.detector_type`` holds opportunity detector names (``Duplicate Invoice
Recovery``). They are separate namespaces and unifying them is its own piece of
work. This module keeps them apart -- a playbook matches within one source and
never across -- while giving the selector a single ``Finding`` to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

DETECTION_FINDING = "detection_finding"
OPPORTUNITY = "opportunity"

SOURCES: Tuple[str, ...] = (DETECTION_FINDING, OPPORTUNITY)

#: The columns a playbook of each source may match on. Closed, for the same
#: reason services.actions.ACTIONS is closed: a key that matches no column is a
#: playbook that silently never fires, and that failure looks exactly like the
#: playbook working and nothing having matched yet.
MATCH_FIELDS: Dict[str, Tuple[str, ...]] = {
    DETECTION_FINDING: ("rule_id", "category", "severity", "doc_type", "blocks_promotion"),
    OPPORTUNITY: ("detector_type", "supplier_id", "category_id"),
}

#: The primary key column per source. The two disagree on type: finding_id is
#: BIGINT, opportunity_id is VARCHAR. Both become text in a Finding.
_ID_FIELD: Dict[str, str] = {
    DETECTION_FINDING: "finding_id",
    OPPORTUNITY: "opportunity_id",
}

#: Open work per source. A detection finding is open by status; an opportunity
#: is open until it is retired.
OPEN_SQL: Dict[str, str] = {
    DETECTION_FINDING: (
        "SELECT finding_id, rule_id, category, severity, doc_type, "
        "       blocks_promotion, deal_id "
        "  FROM proc.bp_detection_finding "
        " WHERE status = 'open' "
        "   AND finding_id > %s "
        " ORDER BY finding_id "
        " LIMIT %s"
    ),
    OPPORTUNITY: (
        "SELECT opportunity_id, detector_type, supplier_id, category_id, deal_id "
        "  FROM proc.bp_opportunity "
        " WHERE retired_at IS NULL "
        "   AND opportunity_id > %s "
        " ORDER BY opportunity_id "
        " LIMIT %s"
    ),
}


@dataclass(frozen=True)
class Finding:
    """One open finding, in the only shape the selector sees."""

    source: str
    finding_id: str
    deal_id: Optional[str]
    attrs: Dict[str, Any]


def _require_source(source: str) -> str:
    if source not in MATCH_FIELDS:
        raise ValueError(
            f"{source!r} is not a finding source. Known sources: "
            f"{', '.join(SOURCES)}."
        )
    return source


def canonical(value: Any) -> Optional[str]:
    """One comparable form for a value from either side of a match.

    A playbook's trigger_match arrives as JSON, where a boolean column's value
    may have been authored as ``true`` or as ``"true"``, and a severity may have
    been typed ``Critical``. The database gives us ``True`` and ``'critical'``.
    Comparing those raw produces a playbook that never fires and no error to
    say so, so both sides are folded here before they are compared.

    ``None`` stays ``None``: an absent value is not the string "none", and must
    not match anything.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def normalise(source: str, row: Mapping[str, Any]) -> Finding:
    """A database row from ``source`` as a :class:`Finding`."""

    _require_source(source)
    id_field = _ID_FIELD[source]
    raw_id = row.get(id_field)
    if raw_id is None:
        raise ValueError(f"{source} row has no {id_field}")
    return Finding(
        source=source,
        finding_id=str(raw_id),
        deal_id=(str(row["deal_id"]) if row.get("deal_id") is not None else None),
        attrs={field: row.get(field) for field in MATCH_FIELDS[source]},
    )


def validate_trigger_match(source: str, match: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``match`` if every key is a real match field for ``source``.

    Raises ``ValueError`` otherwise, rather than storing it. The detection
    registry this codebase removed bound configuration to detectors by
    accumulating aliases until something matched; four of five bindings were
    silently wrong for months. An unknown key here is the same failure in a new
    table, so it is refused at the door.
    """

    _require_source(source)
    allowed = MATCH_FIELDS[source]
    unknown = [key for key in match if key not in allowed]
    if unknown:
        raise ValueError(
            f"{', '.join(sorted(unknown))} "
            f"{'is not a match field' if len(unknown) == 1 else 'are not match fields'} "
            f"for {source}. Allowed: {', '.join(allowed)}."
        )
    null_keys = [key for key, value in match.items() if value is None]
    if null_keys:
        raise ValueError(
            f"{', '.join(sorted(null_keys))} may not be null. A null match value "
            "would mean 'fires only when this column is unset', which is never "
            "what an author means. Omit the key to ignore the column."
        )
    return dict(match)
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_finding_source.py -v -p no:randomly
```
Expected: PASS, 17 tests.

- [ ] **Step 5: Commit**

```bash
git status --short src/services/playbooks tests/services/playbooks
git commit -o src/services/playbooks/__init__.py \
             src/services/playbooks/finding_source.py \
             tests/services/playbooks/__init__.py \
             tests/services/playbooks/test_finding_source.py \
  -m "feat(playbook): one shape for two finding vocabularies, kept apart

bp_detection_finding.rule_id and bp_opportunity.detector_type are separate
namespaces. normalise() gives the selector one Finding to reason about
without merging them, and validate_trigger_match() refuses a key that
belongs to the other store -- or to neither.

A null match value is refused too: it would mean 'fires only when this
column is unset', which is never what an author means and always matches
almost nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: The playbook store

Mirrors `src/engines/rule_book.py` — same constructor shape, same caching, same
`reload()`. It differs in one deliberate way, and the difference is the whole
point: `RuleBook` treats an empty rule set as an outage, and `PlaybookStore`
does not. Zero playbooks is the honest state on the day this ships.

**Files:**
- Create: `src/services/playbooks/store.py`
- Test: `tests/services/playbooks/test_store.py`

**Interfaces:**
- Consumes: `finding_source.MATCH_FIELDS`, `finding_source.SOURCES`.
- Produces:
  - `Playbook(playbook_id: int, playbook_name: str, trigger_source: str, trigger_match: Dict[str, Any], agent_workflow_id: int, params: Dict[str, Any], version: int)` — dataclass
  - `PlaybookStoreUnavailable(RuntimeError)`
  - `PlaybookStore(agent_nick=None, connection_factory=None, playbook_rows=None)` with `.active_playbooks() -> List[Playbook]`, `.for_source(source) -> List[Playbook]`, `.reload() -> None`
  - `load_playbook_store(agent_nick=None, playbook_rows=None) -> Optional[PlaybookStore]`

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/test_store.py`:

```python
"""The store loads active playbooks and nothing else.

Rows are injected; no connection is opened. An unreadable store raises, an
empty one does not -- see the docstring on PlaybookStore for why those differ.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.store import (  # noqa: E402
    Playbook,
    PlaybookStore,
    PlaybookStoreUnavailable,
    load_playbook_store,
)


def row(**over):
    base = {
        "playbook_id": 1,
        "playbook_name": "Recover the duplicate",
        "trigger_source": "detection_finding",
        "trigger_match": {"rule_id": "duplicate"},
        "agent_workflow_id": 958,
        "params": {},
        "playbook_status": "active",
        "version": 1,
    }
    base.update(over)
    return base


def test_loads_an_active_playbook():
    store = PlaybookStore(playbook_rows=[row()])
    loaded = store.active_playbooks()
    assert len(loaded) == 1
    assert loaded[0] == Playbook(
        playbook_id=1,
        playbook_name="Recover the duplicate",
        trigger_source="detection_finding",
        trigger_match={"rule_id": "duplicate"},
        agent_workflow_id=958,
        params={},
        version=1,
    )


@pytest.mark.parametrize("status", ["draft", "pending_approval", "retired"])
def test_only_active_playbooks_load(status):
    """A retired or unapproved playbook must not be able to propose anything."""
    store = PlaybookStore(playbook_rows=[row(playbook_status=status)])
    assert store.active_playbooks() == []


def test_empty_store_does_not_raise():
    """Deliberately unlike RuleBook. An empty rule book hides work that should
    have happened; an empty playbook table simply means nobody has authored a
    strategy yet, and failing closed would make the service unbootable until
    somebody did."""
    store = PlaybookStore(playbook_rows=[])
    assert store.active_playbooks() == []


def test_unreadable_store_raises():
    def explode():
        raise RuntimeError("connection refused")

    with pytest.raises(PlaybookStoreUnavailable) as exc:
        PlaybookStore(connection_factory=explode)
    assert "connection refused" in str(exc.value)


def test_for_source_never_crosses_the_two_stores():
    store = PlaybookStore(
        playbook_rows=[
            row(playbook_id=1, trigger_source="detection_finding",
                trigger_match={"rule_id": "duplicate"}),
            row(playbook_id=2, trigger_source="opportunity",
                trigger_match={"detector_type": "Invoice Overbilling"}),
        ]
    )
    assert [p.playbook_id for p in store.for_source("detection_finding")] == [1]
    assert [p.playbook_id for p in store.for_source("opportunity")] == [2]
    assert store.for_source("invoices") == []


def test_a_row_with_an_unknown_source_is_skipped_not_crashed(caplog):
    """The CHECK constraint makes this unreachable through the endpoint, but a
    hand-edited row must not take the sweep down."""
    store = PlaybookStore(playbook_rows=[row(trigger_source="invoices")])
    assert store.active_playbooks() == []


def test_a_row_with_an_unknown_match_key_is_skipped():
    """It could only get there by hand: the endpoint validates. Loading it
    would give a playbook that never fires and no sign of why."""
    store = PlaybookStore(playbook_rows=[row(trigger_match={"sevrity": "high"})])
    assert store.active_playbooks() == []


def test_jsonb_arriving_as_text_is_parsed():
    """psycopg2 gives a dict; other drivers and some fixtures give a string."""
    store = PlaybookStore(playbook_rows=[row(trigger_match='{"rule_id": "duplicate"}')])
    assert store.active_playbooks()[0].trigger_match == {"rule_id": "duplicate"}


def test_reload_picks_up_a_change():
    rows = [row()]
    store = PlaybookStore(connection_factory=None, playbook_rows=rows)
    assert len(store.active_playbooks()) == 1
    # No connection factory: reload finds nothing and empties the store rather
    # than keeping a stale cache.
    store.reload()
    assert store.active_playbooks() == []


def test_a_playbook_whose_workflow_is_gone_is_skipped():
    """The approve endpoint refuses an inactive workflow, but a workflow can be
    deleted AFTER its playbook was approved. Selecting it would queue a
    proposal that can only fail at the moment somebody accepts it."""
    assert PlaybookStore(
        playbook_rows=[row(workflow_is_active=False)]
    ).active_playbooks() == []
    assert len(PlaybookStore(
        playbook_rows=[row(workflow_is_active=True)]
    ).active_playbooks()) == 1


def test_loader_carries_failure_rather_than_raising():
    """Follows load_rule_book: the blast radius of a playbook outage stops at
    playbooks, so the API still boots."""
    def explode():
        raise RuntimeError("connection refused")

    nick = type("N", (), {"get_db_connection": staticmethod(explode)})()
    assert load_playbook_store(agent_nick=nick) is None
    # And a healthy one still loads.
    assert load_playbook_store(playbook_rows=[row()]) is not None
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_store.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'src.services.playbooks.store'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/playbooks/store.py`:

```python
"""Active playbooks, loaded from ``proc.bp_playbook``.

Mirrors ``engines.rule_book.RuleBook``: load once, cache in memory, reload on
demand, take rows by injection so the selector can be tested without a database.

IT DIFFERS FROM THE RULE BOOK IN ONE PLACE, ON PURPOSE.

``RuleBook`` raises when it holds zero rules, because a sweep that runs no
detectors reports no findings and looks exactly like a clean scan. Zero
*playbooks* is not that. It is the honest state on the day this ships and for
as long as nobody has authored a strategy; failing closed there would make the
service unbootable until an expert wrote one. An empty rule book hides work
that should have happened -- an empty playbook table simply means no strategy
is on file. The sweep logs the count it proposed on every run, so zero stays
visible rather than becoming silence.

An unreadable store still raises. That is an outage, and it is not the same
thing as being empty.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .finding_source import MATCH_FIELDS

logger = logging.getLogger(__name__)

_SELECT = """
    SELECT p.playbook_id, p.playbook_name, p.trigger_source, p.trigger_match,
           p.agent_workflow_id, p.params, p.playbook_status, p.version,
           COALESCE(w.is_active, FALSE) AS workflow_is_active
      FROM proc.bp_playbook p
      LEFT JOIN proc.bp_agent_workflow w ON w.workflow_id = p.agent_workflow_id
     WHERE p.playbook_status = 'active'
     ORDER BY p.playbook_id
"""


class PlaybookStoreUnavailable(RuntimeError):
    """``proc.bp_playbook`` could not be read.

    Raised for an outage, never for an empty table -- see the module docstring.
    """


@dataclass
class Playbook:
    """One playbook as the database holds it."""

    playbook_id: int
    playbook_name: str
    trigger_source: str
    trigger_match: Dict[str, Any] = field(default_factory=dict)
    agent_workflow_id: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


class PlaybookStore:
    """Load and cache active playbooks from ``proc.bp_playbook``."""

    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        playbook_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None
        self._playbooks: List[Playbook] = []
        self._load(playbook_rows)

    # -- loading ---------------------------------------------------------

    def _load(self, playbook_rows: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        rows = list(playbook_rows) if playbook_rows is not None else self._fetch_rows()
        loaded = [self._to_playbook(row) for row in rows]
        self._playbooks = [pb for pb in loaded if pb is not None]

    def _fetch_rows(self) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                if conn is None:
                    return []
                cursor = conn.cursor()
                try:
                    cursor.execute(_SELECT)
                    columns = [c[0] for c in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                finally:
                    cursor.close()
        except PlaybookStoreUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage
            raise PlaybookStoreUnavailable(
                f"could not read proc.bp_playbook: {exc}"
            ) from exc

    @contextmanager
    def _connect(self):
        factory = self._connection_factory
        if factory is None:
            yield None
            return
        resolved = factory() if callable(factory) else factory
        if resolved is None:
            yield None
            return
        if hasattr(resolved, "__enter__"):
            with resolved as conn:
                yield conn
        else:
            yield resolved

    # -- coercion --------------------------------------------------------

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        """JSONB arrives as a dict from psycopg2, as text from other drivers."""
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (str, bytes)):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    def _to_playbook(self, row: Dict[str, Any]) -> Optional[Playbook]:
        pid = row.get("playbook_id")
        status = str(row.get("playbook_status") or "").strip()
        if status and status != "active":
            return None
        source = str(row.get("trigger_source") or "").strip()
        if source not in MATCH_FIELDS:
            logger.error(
                "skipping proc.bp_playbook row %s: trigger_source %r is not a "
                "finding source", pid, source,
            )
            return None
        if "workflow_is_active" in row and not row["workflow_is_active"]:
            # The approve endpoint refuses a playbook whose workflow is
            # inactive, but a workflow can be deleted AFTER its playbook was
            # approved. Selecting it would queue a proposal that can only fail
            # at the moment somebody accepts it.
            logger.error(
                "skipping proc.bp_playbook row %s (%s): agent_workflow_id %s is "
                "missing or inactive, so it can propose nothing that could run",
                pid, row.get("playbook_name"), row.get("agent_workflow_id"),
            )
            return None
        match = self._as_mapping(row.get("trigger_match"))
        unknown = [key for key in match if key not in MATCH_FIELDS[source]]
        if unknown:
            # The endpoint validates, so this row was hand-written. Loading it
            # would give a playbook that never fires and nothing to say why.
            logger.error(
                "skipping proc.bp_playbook row %s (%s): trigger_match keys %s "
                "are not match fields for %s",
                pid, row.get("playbook_name"), sorted(unknown), source,
            )
            return None
        return Playbook(
            playbook_id=int(pid or 0),
            playbook_name=str(row.get("playbook_name") or f"playbook {pid}"),
            trigger_source=source,
            trigger_match=match,
            agent_workflow_id=int(row.get("agent_workflow_id") or 0),
            params=self._as_mapping(row.get("params")),
            version=int(row.get("version") or 1),
        )

    # -- reading ---------------------------------------------------------

    def active_playbooks(self) -> List[Playbook]:
        return list(self._playbooks)

    def for_source(self, source: str) -> List[Playbook]:
        return [pb for pb in self._playbooks if pb.trigger_source == source]

    def reload(self) -> None:
        self._load()


def load_playbook_store(
    agent_nick: Optional[Any] = None,
    playbook_rows: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[PlaybookStore]:
    """Build the store at startup, carrying failure rather than raising.

    Follows ``engines.rule_book.load_rule_book``: the blast radius of "the
    playbook table cannot be read" is playbooks. Raising from here would take
    the whole API down with it.
    """

    try:
        return PlaybookStore(agent_nick=agent_nick, playbook_rows=playbook_rows)
    except PlaybookStoreUnavailable:
        logger.exception(
            "playbook store unavailable -- no playbook will be proposed until "
            "it loads. Everything else still starts."
        )
        return None
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_store.py -v -p no:randomly
```
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git status --short src/services/playbooks tests/services/playbooks
git commit -o src/services/playbooks/store.py \
             tests/services/playbooks/test_store.py \
  -m "feat(playbook): the store, and the one place it differs from the rule book

Active playbooks only, so a draft, a pending approval or a retired strategy
can never propose anything.

RuleBook raises on an empty rule set because a sweep with no detectors
reports no findings and looks like a clean scan. An empty playbook table is
not that -- it is the honest state until an expert authors a strategy, and
failing closed there would make the service unbootable until one did. An
unreadable store still raises; that is an outage and a different thing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: The selector

Pure. No I/O. This is the task the spec argues hardest about: **a tie proposes
nothing**. The detection registry this codebase just removed resolved its own
ambiguity quietly and was wrong for months. Do not add a tiebreak.

**Files:**
- Create: `src/services/playbooks/selector.py`
- Test: `tests/services/playbooks/test_selector.py`

**Interfaces:**
- Consumes: `store.Playbook`, `finding_source.Finding`, `finding_source.canonical`.
- Produces:
  - `Selection(playbook: Playbook, evidence: Dict[str, Any])` — frozen dataclass; `evidence` is `{"matched": {key: canonical_value}, "key_count": int, "playbook_version": int}`
  - `select(finding: Finding, playbooks: Sequence[Playbook]) -> Optional[Selection]`
  - `tied_candidates(finding: Finding, playbooks: Sequence[Playbook]) -> List[Playbook]` — empty unless the match was ambiguous

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/test_selector.py`:

```python
"""Selection is deterministic, and it refuses rather than guesses."""

import logging
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.finding_source import Finding  # noqa: E402
from src.services.playbooks.selector import select, tied_candidates  # noqa: E402
from src.services.playbooks.store import Playbook  # noqa: E402


def pb(pid, match, source="detection_finding", name=None, version=1):
    return Playbook(
        playbook_id=pid,
        playbook_name=name or f"playbook {pid}",
        trigger_source=source,
        trigger_match=match,
        agent_workflow_id=958,
        params={},
        version=version,
    )


def finding(**attrs):
    return Finding(
        source=attrs.pop("source", "detection_finding"),
        finding_id=attrs.pop("finding_id", "1"),
        deal_id=attrs.pop("deal_id", "D-1"),
        attrs=attrs,
    )


def test_exact_match_selects():
    chosen = select(
        finding(rule_id="duplicate", severity="critical"),
        [pb(7, {"rule_id": "duplicate"})],
    )
    assert chosen is not None
    assert chosen.playbook.playbook_id == 7
    assert chosen.evidence["matched"] == {"rule_id": "duplicate"}
    assert chosen.evidence["key_count"] == 1
    assert chosen.evidence["playbook_version"] == 1


def test_a_non_matching_value_does_not_select():
    assert select(
        finding(rule_id="quantity"),
        [pb(7, {"rule_id": "duplicate"})],
    ) is None


def test_most_specific_wins():
    chosen = select(
        finding(rule_id="duplicate", severity="critical", category="duplicate"),
        [
            pb(1, {"rule_id": "duplicate"}),
            pb(2, {"rule_id": "duplicate", "severity": "critical"}),
        ],
    )
    assert chosen.playbook.playbook_id == 2
    assert chosen.evidence["key_count"] == 2


def test_a_tie_proposes_nothing_and_names_both(caplog):
    """A configuration error that resolves itself quietly is how four of five
    detector bindings were wrong for months. A tie is visible or it is nothing."""
    two = [
        pb(1, {"rule_id": "duplicate"}, name="Recover it"),
        pb(2, {"severity": "critical"}, name="Escalate it"),
    ]
    f = finding(rule_id="duplicate", severity="critical")
    with caplog.at_level(logging.ERROR):
        assert select(f, two) is None
    logged = caplog.text
    assert "Recover it" in logged and "Escalate it" in logged
    assert "1" in logged and "2" in logged
    assert [p.playbook_id for p in tied_candidates(f, two)] == [1, 2]


def test_tied_candidates_is_empty_when_there_is_a_winner():
    f = finding(rule_id="duplicate", severity="critical")
    one = [pb(1, {"rule_id": "duplicate"}), pb(2, {"rule_id": "duplicate", "severity": "critical"})]
    assert select(f, one) is not None
    assert tied_candidates(f, one) == []


def test_tied_candidates_is_empty_when_nothing_matched():
    assert tied_candidates(finding(rule_id="quantity"), [pb(1, {"rule_id": "duplicate"})]) == []


def test_empty_trigger_match_is_a_catch_all():
    chosen = select(finding(rule_id="anything"), [pb(9, {})])
    assert chosen.playbook.playbook_id == 9
    assert chosen.evidence["matched"] == {}
    assert chosen.evidence["key_count"] == 0


def test_a_catch_all_loses_to_a_specific_playbook():
    chosen = select(
        finding(rule_id="duplicate"),
        [pb(9, {}), pb(3, {"rule_id": "duplicate"})],
    )
    assert chosen.playbook.playbook_id == 3


def test_two_catch_alls_for_one_source_are_a_tie():
    assert select(finding(rule_id="duplicate"), [pb(9, {}), pb(10, {})]) is None


def test_the_wrong_source_never_matches():
    """An opportunity playbook must not be reachable from a detection finding
    even if a key name happened to coincide."""
    assert select(
        finding(source="detection_finding", rule_id="duplicate"),
        [pb(5, {}, source="opportunity")],
    ) is None


def test_matching_is_case_and_type_insensitive_on_both_sides():
    """A boolean column authored in JSON as a string, and a capitalised
    severity, must still match -- otherwise the playbook silently never fires."""
    chosen = select(
        finding(severity="critical", blocks_promotion=True),
        [pb(4, {"severity": "Critical", "blocks_promotion": "true"})],
    )
    assert chosen is not None and chosen.playbook.playbook_id == 4


def test_a_null_finding_attribute_does_not_match_a_present_key():
    """doc_type IS NULL is not doc_type = 'invoice', and must never be treated
    as a wildcard."""
    assert select(
        finding(rule_id="duplicate", doc_type=None),
        [pb(6, {"rule_id": "duplicate", "doc_type": "invoice"})],
    ) is None


def test_a_missing_finding_attribute_does_not_match():
    assert select(
        finding(rule_id="duplicate"),
        [pb(6, {"rule_id": "duplicate", "doc_type": "invoice"})],
    ) is None


def test_evidence_records_the_findings_values_not_the_playbooks():
    """A proposal must be re-derivable from source, so the evidence has to say
    what the finding held, canonically, at the moment it matched."""
    chosen = select(
        finding(severity="CRITICAL", blocks_promotion=True),
        [pb(4, {"severity": "critical", "blocks_promotion": True})],
    )
    assert chosen.evidence["matched"] == {"severity": "critical", "blocks_promotion": "true"}


def test_no_playbooks_selects_nothing():
    assert select(finding(rule_id="duplicate"), []) is None
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_selector.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'src.services.playbooks.selector'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/playbooks/selector.py`:

```python
"""Which playbook governs a finding. Pure, deterministic, and it refuses.

    1. Take the active playbooks whose trigger_source is the finding's source.
    2. Keep those whose every trigger_match key equals the finding's value for
       that key. Equality only -- no fuzzy matching, no aliases, no substrings.
    3. The winner is the one with the MOST match keys.
    4. If two or more tie on key count, propose nothing, and say so at ERROR
       naming both by id and name.

STEP 4 IS THE LOAD-BEARING ONE.

The detection registry this codebase removed bound policies to detectors by
accumulating aliases and letting whichever matched last win. Four of five
bindings were silently wrong for months, and price variance spent them
reporting itself as maverick spend. An ambiguous playbook match is a
configuration error, and a configuration error that resolves itself quietly is
that same failure in a new table. A tie is visible or it is nothing.

Do not add a tiebreak here. Not lowest id, not most recently approved, not
highest version. Any of them would make the ambiguity disappear from the logs
while leaving it in the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .finding_source import Finding, canonical
from .store import Playbook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    """The one playbook that governs a finding, and why it matched."""

    playbook: Playbook
    evidence: Dict[str, Any]


def _matches(finding: Finding, playbook: Playbook) -> bool:
    """Every trigger_match key equals the finding's value for that key.

    Both sides go through ``canonical`` first, so a boolean column authored in
    JSON as ``"true"`` and a severity authored as ``Critical`` still match. A
    finding value that is absent or NULL matches nothing: it is not a wildcard.
    """

    for key, wanted in playbook.trigger_match.items():
        have = finding.attrs.get(key)
        if have is None:
            return False
        if canonical(have) != canonical(wanted):
            return False
    return True


def _best(finding: Finding, playbooks: Sequence[Playbook]) -> List[Playbook]:
    """The matching playbooks tied on the most match keys. Empty if none match."""

    matching = [
        pb for pb in playbooks
        if pb.trigger_source == finding.source and _matches(finding, pb)
    ]
    if not matching:
        return []
    most = max(len(pb.trigger_match) for pb in matching)
    return [pb for pb in matching if len(pb.trigger_match) == most]


def _evidence(finding: Finding, playbook: Playbook) -> Dict[str, Any]:
    """What the finding held, canonically, at the moment it matched."""

    return {
        "matched": {
            key: canonical(finding.attrs.get(key))
            for key in playbook.trigger_match
        },
        "key_count": len(playbook.trigger_match),
        "playbook_version": playbook.version,
    }


def select(finding: Finding, playbooks: Sequence[Playbook]) -> Optional[Selection]:
    """The one playbook that governs ``finding``, or ``None``.

    ``None`` covers two different situations -- nothing matched, and more than
    one thing matched equally well. Call :func:`tied_candidates` to tell them
    apart; the sweep does, and audits the second.
    """

    best = _best(finding, playbooks)
    if not best:
        return None
    if len(best) > 1:
        logger.error(
            "ambiguous playbook match for %s finding %s: %s tie on %d match "
            "key(s), so nothing is proposed. Make one of them more specific or "
            "retire one.",
            finding.source,
            finding.finding_id,
            ", ".join(f"{pb.playbook_name!r} (id {pb.playbook_id})" for pb in best),
            len(best[0].trigger_match),
        )
        return None
    winner = best[0]
    return Selection(playbook=winner, evidence=_evidence(finding, winner))


def tied_candidates(finding: Finding, playbooks: Sequence[Playbook]) -> List[Playbook]:
    """The playbooks that tied, or ``[]`` when the match was unambiguous.

    Kept separate from :func:`select` so that both stay pure and the caller,
    not this module, decides what an ambiguity is worth recording.
    """

    best = _best(finding, playbooks)
    return best if len(best) > 1 else []
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_selector.py -v -p no:randomly
```
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git status --short src/services/playbooks tests/services/playbooks
git commit -o src/services/playbooks/selector.py \
             tests/services/playbooks/test_selector.py \
  -m "feat(playbook): selection is deterministic, and a tie proposes nothing

Equality only, most-specific wins, and where two playbooks tie on key count
the selector logs both by name and id and returns nothing.

The alternative was a tiebreak, and this codebase has just finished paying
for one: the detection registry accumulated aliases and let whichever
matched last win, so four of five bindings were silently wrong for months.
A configuration error that resolves itself quietly is the same failure in a
new table.

A NULL finding value is not a wildcard, and both sides of a comparison are
folded to one canonical form first so a boolean authored in JSON as a
string still matches.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: The proposer

One row, or none. The unique index does the work; `ON CONFLICT DO NOTHING`
makes a second sweep silent rather than an error.

**Files:**
- Create: `src/services/playbooks/proposer.py`
- Test: `tests/services/playbooks/test_proposer.py`

**Interfaces:**
- Consumes: `finding_source.Finding`, `selector.Selection`, `src.services.db.get_conn`, `src.services.agent_actions.record_action`.
- Produces:
  - `PHASE = "playbook"`
  - `propose(finding: Finding, selection: Selection, *, conn=None) -> Optional[int]` — the new `proposal_id`, or `None` when one already existed
  - `record_ambiguous(finding: Finding, tied: Sequence[Playbook]) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/test_proposer.py`:

```python
"""One finding, one proposal -- however many times the sweep runs.

The insert is exercised against a fake cursor that records statements, so the
SQL's shape is pinned without a database. The live behaviour of the unique
index is proved in Task 10's guard proofs, by dropping it.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks import proposer  # noqa: E402
from src.services.playbooks.finding_source import Finding  # noqa: E402
from src.services.playbooks.selector import Selection  # noqa: E402
from src.services.playbooks.store import Playbook  # noqa: E402


class FakeCursor:
    """Answers the INSERT ... RETURNING, and remembers what it was asked."""

    def __init__(self, returns):
        self._returns = list(returns)
        self.statements = []
        self.params = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)

    def fetchone(self):
        return self._returns.pop(0) if self._returns else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def no_audit(monkeypatch):
    """The audit writer is best-effort and tested in its own suite."""
    recorded = []
    monkeypatch.setattr(
        proposer.agent_actions, "record_action",
        lambda **kw: recorded.append(kw),
    )
    return recorded


def playbook(pid=7, source="detection_finding", version=1):
    return Playbook(
        playbook_id=pid, playbook_name="Recover the duplicate",
        trigger_source=source, trigger_match={"rule_id": "duplicate"},
        agent_workflow_id=958, params={}, version=version,
    )


def selection(pb=None):
    pb = pb or playbook()
    return Selection(
        playbook=pb,
        evidence={"matched": {"rule_id": "duplicate"}, "key_count": 1,
                  "playbook_version": pb.version},
    )


def finding(source="detection_finding", fid="4211", deal="D-900"):
    return Finding(source=source, finding_id=fid, deal_id=deal,
                   attrs={"rule_id": "duplicate"})


def test_a_new_finding_gets_one_proposal():
    cur = FakeCursor([(55,)])
    pid = proposer.propose(finding(), selection(), conn=FakeConn(cur))
    assert pid == 55
    assert "INSERT INTO proc.bp_playbook_proposal" in cur.statements[0]
    assert "ON CONFLICT DO NOTHING" in cur.statements[0]
    assert "RETURNING proposal_id" in cur.statements[0]


def test_the_same_finding_twice_proposes_once():
    """ON CONFLICT DO NOTHING returns no row on the second attempt."""
    cur = FakeCursor([(55,), None])
    conn = FakeConn(cur)
    assert proposer.propose(finding(), selection(), conn=conn) == 55
    assert proposer.propose(finding(), selection(), conn=conn) is None


def test_a_version_bump_does_not_re_propose():
    """The unique index is keyed on playbook_id, not version, on purpose: an
    edited playbook must not raise a second proposal for a finding already
    queued."""
    cur = FakeCursor([(55,), None])
    conn = FakeConn(cur)
    proposer.propose(finding(), selection(playbook(version=1)), conn=conn)
    assert proposer.propose(finding(), selection(playbook(version=2)), conn=conn) is None


def test_the_same_id_in_both_stores_is_two_different_findings():
    """bp_detection_finding.finding_id 123 and bp_opportunity.opportunity_id
    '123' both land in one TEXT column; finding_source keeps them apart."""
    cur = FakeCursor([(1,), (2,)])
    conn = FakeConn(cur)
    a = proposer.propose(finding(source="detection_finding", fid="123"), selection(), conn=conn)
    b = proposer.propose(
        finding(source="opportunity", fid="123"),
        selection(playbook(pid=8, source="opportunity")),
        conn=conn,
    )
    assert (a, b) == (1, 2)
    assert cur.params[0][1] == "detection_finding"
    assert cur.params[1][1] == "opportunity"


def test_the_row_carries_playbook_finding_deal_and_evidence():
    cur = FakeCursor([(55,)])
    proposer.propose(finding(), selection(), conn=FakeConn(cur))
    playbook_id, source, finding_id, deal_id, evidence = cur.params[0]
    assert playbook_id == 7
    assert source == "detection_finding"
    assert finding_id == "4211"
    assert deal_id == "D-900"
    assert '"rule_id": "duplicate"' in evidence


def test_a_proposal_is_audited(no_audit):
    proposer.propose(finding(), selection(), conn=FakeConn(FakeCursor([(55,)])))
    assert len(no_audit) == 1
    event = no_audit[0]
    assert event["phase"] == "playbook"
    assert event["action_type"] == "playbook.propose"
    assert event["deal_id"] == "D-900"
    assert "Recover the duplicate" in event["summary"]


def test_a_duplicate_is_not_audited_as_a_new_proposal(no_audit):
    """A sweep that re-reads 4,838 open findings every fifteen minutes would
    otherwise write 4,838 audit rows an hour saying nothing happened."""
    cur = FakeCursor([None])
    assert proposer.propose(finding(), selection(), conn=FakeConn(cur)) is None
    assert no_audit == []


def test_ambiguity_is_recorded_as_its_own_event(no_audit):
    proposer.record_ambiguous(finding(), [playbook(1), playbook(2)])
    assert len(no_audit) == 1
    event = no_audit[0]
    assert event["action_type"] == "playbook.ambiguous"
    assert event["status"] == "skipped"
    assert "1" in event["summary"] and "2" in event["summary"]
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_proposer.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'src.services.playbooks.proposer'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/playbooks/proposer.py`:

```python
"""Write one proposal for one finding, and say so in the audit log.

A proposal is a recommendation awaiting a person. Writing one starts nothing:
the run_id stays NULL until somebody approves it through the endpoint.

Idempotency is the unique index ux_bp_playbook_proposal_finding, not a SELECT
first -- two sweeps overlapping would both find nothing and both insert. The
index is keyed on (playbook_id, finding_source, finding_id) and deliberately
NOT on version, so editing a playbook does not re-raise a proposal for a
finding already queued.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from src.services import agent_actions
from src.services.db import get_conn

from .finding_source import Finding
from .selector import Selection
from .store import Playbook

logger = logging.getLogger(__name__)

PHASE = "playbook"

_INSERT = """
    INSERT INTO proc.bp_playbook_proposal
        (playbook_id, finding_source, finding_id, deal_id, evidence)
    VALUES (%s, %s, %s, %s, %s::jsonb)
    ON CONFLICT DO NOTHING
    RETURNING proposal_id
"""


def propose(
    finding: Finding,
    selection: Selection,
    *,
    conn: Optional[Any] = None,
) -> Optional[int]:
    """Queue ``selection`` for ``finding``. Returns the new proposal_id.

    Returns ``None`` when this finding already has a proposal from this
    playbook, which is the ordinary case on every sweep after the first.
    """

    params = (
        selection.playbook.playbook_id,
        finding.source,
        finding.finding_id,
        finding.deal_id,
        json.dumps(selection.evidence, default=str),
    )
    if conn is not None:
        proposal_id = _insert(conn, params)
    else:
        with get_conn() as own:
            proposal_id = _insert(own, params)

    if proposal_id is None:
        # Already queued. Auditing it again would write one row per open
        # finding per sweep, saying nothing happened.
        return None

    agent_actions.record_action(
        phase=PHASE,
        action_type="playbook.propose",
        agent="PlaybookProposer",
        deal_id=finding.deal_id,
        status="proposed",
        summary=(
            f"{selection.playbook.playbook_name!r} proposed for "
            f"{finding.source} {finding.finding_id}"
        ),
        details={
            "proposal_id": proposal_id,
            "playbook_id": selection.playbook.playbook_id,
            "agent_workflow_id": selection.playbook.agent_workflow_id,
            "finding_source": finding.source,
            "finding_id": finding.finding_id,
            "evidence": selection.evidence,
        },
    )
    return proposal_id


def _insert(conn: Any, params: tuple) -> Optional[int]:
    cursor = conn.cursor()
    try:
        cursor.execute(_INSERT, params)
        row = cursor.fetchone()
        return int(row[0]) if row else None
    finally:
        cursor.close()


def record_ambiguous(finding: Finding, tied: Sequence[Playbook]) -> None:
    """Record that two or more playbooks tied, so nothing was proposed.

    The selector has already said so at ERROR. This puts it in the same event
    log as the proposals, because "no proposal appeared for this finding" is a
    question someone will ask of the audit trail, not of the log files.
    """

    names = ", ".join(f"{pb.playbook_name!r} (id {pb.playbook_id})" for pb in tied)
    agent_actions.record_action(
        phase=PHASE,
        action_type="playbook.ambiguous",
        agent="PlaybookProposer",
        deal_id=finding.deal_id,
        status="skipped",
        summary=(
            f"{len(tied)} playbooks tie for {finding.source} "
            f"{finding.finding_id}: {names}. Nothing proposed."
        ),
        details={
            "finding_source": finding.source,
            "finding_id": finding.finding_id,
            "playbook_ids": [pb.playbook_id for pb in tied],
            "playbook_names": [pb.playbook_name for pb in tied],
        },
    )
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_proposer.py -v -p no:randomly
```
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git status --short src/services/playbooks tests/services/playbooks
git commit -o src/services/playbooks/proposer.py \
             tests/services/playbooks/test_proposer.py \
  -m "feat(playbook): one finding, one proposal, however often the sweep runs

Idempotency is the unique index and ON CONFLICT DO NOTHING, not a SELECT
first -- two overlapping sweeps would both find nothing and both insert.
The index ignores version on purpose: editing a playbook must not re-raise
a proposal for a finding already queued.

A duplicate is not audited. A sweep re-reading 4,838 open findings every
fifteen minutes would otherwise write thousands of rows an hour saying
nothing happened. An ambiguous match is audited, because 'why did no
proposal appear for this finding' is a question asked of the trail.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Two action names and the authority that answers them

The class vocabulary is `RoleDefinitionPolicy`'s — `read, compute, write,
communicate, share, transact, configure, delegate, approve_email`. There is no
generic `approve` class, so both new names follow the `policy.write` /
`prompt.write` precedent and are `configure`.

Approving a *proposal* is a different act: it causes a workflow to run, and it
gates on the **existing** `workflow.run`. No new action for it.

**Files:**
- Modify: `src/services/actions.py` (the `configure` block, around line 82)
- Create: `deploy/sql/2026-09-29_bp_playbook_policy.sql`
- Test: `tests/services/playbooks/test_actions_vocabulary.py`

**Interfaces:**
- Consumes: `services.actions.ACTIONS`, `services.actions.action_class`.
- Produces: `playbook.write` and `playbook.approve` as known actions, class `configure`; one `bp_policy` authority row in both databases.

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/test_actions_vocabulary.py`:

```python
"""The two names the playbook layer gates on, and the class they belong to."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services import actions  # noqa: E402


@pytest.mark.parametrize("name", ["playbook.write", "playbook.approve"])
def test_the_name_is_known(name):
    assert actions.is_known(name)


@pytest.mark.parametrize("name", ["playbook.write", "playbook.approve"])
def test_configuring_a_playbook_is_a_configure(name):
    """There is no generic 'approve' class in RoleDefinitionPolicy, so these
    follow policy.write and prompt.write."""
    assert actions.action_class(name) == "configure"


def test_approving_a_proposal_introduces_no_new_action():
    """It causes a workflow to run and gates on workflow.run, which already
    exists and is already policied. A second name for the same act would be a
    second thing to keep in step."""
    assert actions.action_class("workflow.run") == "delegate"
    assert not actions.is_known("proposal.approve")
    assert not actions.is_known("playbook.run")


def test_the_names_follow_the_domain_verb_shape():
    for name in ("playbook.write", "playbook.approve"):
        domain, _, verb = name.partition(".")
        assert domain == "playbook"
        assert verb and verb.islower() and not verb.endswith("s")
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_actions_vocabulary.py -v -p no:randomly
```
Expected: FAIL — `assert actions.is_known("playbook.write")` is False.

- [ ] **Step 3: Add the two names**

In `src/services/actions.py`, in the `--- configuring ---` block, after
`"prompt.write": "configure",`:

```python
    "prompt.write": "configure",
    # Authoring a strategy and approving one are configuration: they change
    # what the system will recommend, for every finding that matches, until
    # someone changes it back. Approving a PROPOSAL is a different act -- it
    # runs a workflow -- and gates on workflow.run, which already exists.
    "playbook.write": "configure",
    "playbook.approve": "configure",
    "model.train": "configure",
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_actions_vocabulary.py -v -p no:randomly
```
Expected: PASS, 8 tests.

- [ ] **Step 5: Write the authority policy**

An action with no policy row defers forever and refuses every caller. Create
`deploy/sql/2026-09-29_bp_playbook_policy.sql`:

```sql
-- 2026-09-29  Who may author and approve a playbook.
--
-- Without a row, guardrail.authorize finds no policy for these names, the
-- decision never resolves, and the endpoint refuses everyone -- which looks
-- like a permissions bug rather than a missing row. Mirrors
-- GovernanceAuthorityPolicy (policy.write / prompt.write), Admin-only.
--
-- Deliberately NOT enrolled in ShadowModePolicy. The thirteen actions there
-- were enrolled to avoid breaking callers that predated the gate; these two
-- have no callers yet, so they are enforced from the first request.
--
-- Idempotent. Safe to re-run.

BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details, policy_status,
     created_by, last_modified_by)
SELECT
    'PlaybookAuthorityPolicy',
    'authority',
    'Who may author a playbook and who may approve one. Approving a proposal is workflow.run, not this.',
    '{"policy_identifier": "playbook_authority",
      "applies_to": ["playbook.write", "playbook.approve"],
      "required_role": "Admin",
      "rules": {"effect": "allow"}}'::jsonb,
    1,
    'bp_playbook_migration_2026_09_29',
    'bp_playbook_migration_2026_09_29'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
     WHERE policy_details->>'policy_identifier' = 'playbook_authority'
);

COMMIT;
```

- [ ] **Step 6: Apply it to both databases and confirm the rows**

```sh
set -a; . ./.env; set +a
for DB in bp_testdb bp_sqldb; do
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" \
    -d "$DB" -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-29_bp_playbook_policy.sql
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" \
    -d "$DB" -tAc "SELECT policy_id, policy_status, policy_details->>'applies_to'
                     FROM proc.bp_policy
                    WHERE policy_details->>'policy_identifier' = 'playbook_authority'"
done
```
Expected: exactly one row per database, `policy_status = 1`, applies_to listing both names. Re-run the whole loop: still exactly one row per database.

- [ ] **Step 7: Do NOT add a second audit writer**

The spec's event list names `playbook.authored`, `playbook.approved` and
`playbook.retired`. Those rows already get written: `gate()` audits every
attempt through `record_action_or_fail` with `action_type` set to the action
name, so authoring and approving land in `proc.bp_agent_actions` as
`playbook.write` and `playbook.approve` the moment the endpoints exist. Adding
a second writer would put the same act in the log twice under two names, and
this codebase has already paid for three near-identical audit blocks that
agreed until one drifted. The names in the spec are the information, not the
strings.

What `gate()` does **not** cover is a proposal being rejected (ungated) or
superseded (after the gate). Task 8 writes those.

- [ ] **Step 8: Commit**

```bash
git status --short src/services/actions.py deploy/sql tests/services/playbooks
git commit -o src/services/actions.py \
             deploy/sql/2026-09-29_bp_playbook_policy.sql \
             tests/services/playbooks/test_actions_vocabulary.py \
  -m "feat(playbook): two action names, and the row that answers them

playbook.write and playbook.approve, both class configure -- there is no
generic approve class in RoleDefinitionPolicy, so they follow policy.write.
Approving a PROPOSAL introduces no new name: it runs a workflow and gates
on workflow.run, which already exists and is already policied.

The authority row ships with them. An action with no policy row never
resolves and refuses everyone, which reads as a permissions bug rather than
a missing row. Neither name is enrolled in shadow mode: nothing calls them
yet, so they are enforced from the first request.

Applied to bp_testdb and bp_sqldb.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: The repository and the authorship lifecycle

`draft → pending_approval → active → retired`. The rules that make an approval
mean something live here, in one place, so the router cannot express a
different opinion: an edit to an `active` playbook returns it to
`pending_approval` and bumps `version`, and nobody approves their own work.

**Files:**
- Create: `src/repositories/playbook_repo.py`
- Test: `tests/services/playbooks/test_lifecycle.py`

**Interfaces:**
- Consumes: `services.db.get_conn`, `services.playbooks.finding_source.validate_trigger_match`.
- Produces:
  - `LifecycleError(ValueError)`
  - `next_status_for_edit(current: str) -> Tuple[str, bool]` — `(status, bump_version)`; pure
  - `check_approval(authored_by: str, approver: str, current_status: str, workflow_is_active: bool) -> None` — raises `LifecycleError`; pure
  - `create(*, name, trigger_source, trigger_match, agent_workflow_id, params, description, authored_by) -> int`
  - `get(playbook_id) -> Optional[Dict[str, Any]]`
  - `list_playbooks(status: Optional[str] = None) -> List[Dict[str, Any]]`
  - `update(playbook_id, *, name, trigger_source, trigger_match, agent_workflow_id, params, description, modified_by) -> Dict[str, Any]`
  - `submit(playbook_id, *, modified_by) -> None`
  - `approve(playbook_id, *, approver) -> None`
  - `retire(playbook_id, *, modified_by) -> None`

- [ ] **Step 1: Write the failing test**

The lifecycle rules are pure functions so they can be tested without a
database; the SQL around them is proved live in Task 10. Create
`tests/services/playbooks/test_lifecycle.py`:

```python
"""The rules that make an approval mean something."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.repositories.playbook_repo import (  # noqa: E402
    LifecycleError,
    check_approval,
    next_status_for_edit,
)


def test_editing_an_active_playbook_returns_it_for_approval_and_bumps_version():
    """An approved strategy cannot be changed underneath its approval."""
    assert next_status_for_edit("active") == ("pending_approval", True)


def test_editing_a_draft_leaves_it_a_draft():
    assert next_status_for_edit("draft") == ("draft", False)


def test_editing_something_awaiting_approval_leaves_it_awaiting_approval():
    assert next_status_for_edit("pending_approval") == ("pending_approval", False)


def test_a_retired_playbook_cannot_be_edited():
    with pytest.raises(LifecycleError) as exc:
        next_status_for_edit("retired")
    assert "retired" in str(exc.value)


def test_approval_moves_a_pending_playbook():
    check_approval(authored_by="ana", approver="bo",
                   current_status="pending_approval", workflow_is_active=True)


def test_nobody_approves_their_own_playbook():
    with pytest.raises(LifecycleError) as exc:
        check_approval(authored_by="ana", approver="ana",
                       current_status="pending_approval", workflow_is_active=True)
    assert "own" in str(exc.value).lower()


def test_self_approval_is_barred_whatever_the_case_or_spacing():
    with pytest.raises(LifecycleError):
        check_approval(authored_by="Ana@x.com", approver=" ana@X.com ",
                       current_status="pending_approval", workflow_is_active=True)


def test_an_anonymous_approver_is_refused():
    """Without a subject the self-approval bar cannot be applied at all, so an
    unattributable approval is worse than no approval."""
    for approver in (None, "", "   "):
        with pytest.raises(LifecycleError):
            check_approval(authored_by="ana", approver=approver,
                           current_status="pending_approval", workflow_is_active=True)


def test_only_a_pending_playbook_can_be_approved():
    for status in ("draft", "active", "retired"):
        with pytest.raises(LifecycleError) as exc:
            check_approval(authored_by="ana", approver="bo",
                           current_status=status, workflow_is_active=True)
        assert status in str(exc.value)


def test_a_playbook_pointing_at_an_inactive_workflow_cannot_be_approved():
    """Approving it would create a strategy that can only ever fail at the
    moment somebody accepts its proposal."""
    with pytest.raises(LifecycleError) as exc:
        check_approval(authored_by="ana", approver="bo",
                       current_status="pending_approval", workflow_is_active=False)
    assert "workflow" in str(exc.value).lower()
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_lifecycle.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'src.repositories.playbook_repo'`.

- [ ] **Step 3: Write the repository**

Create `src/repositories/playbook_repo.py`:

```python
# src/repositories/playbook_repo.py
"""Playbooks, under their authorship lifecycle.

    draft -> pending_approval -> active -> retired

Only ``active`` playbooks are loaded by the store, and therefore only an
approved strategy can propose anything.

The two rules that make an approval mean something are pure functions at the
top of this module, so the router cannot hold a different opinion about them:

  * Editing an ``active`` playbook returns it to ``pending_approval`` and bumps
    ``version``. An approved strategy cannot be changed underneath its approval.
  * Nobody approves their own work, and an unattributable approval is refused
    outright -- without a subject the bar cannot be applied at all.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.db import get_conn
from services.playbooks.finding_source import validate_trigger_match

logger = logging.getLogger(__name__)


class LifecycleError(ValueError):
    """A transition the lifecycle does not allow. The router turns this into a 400."""


_EDITABLE = {
    "draft": ("draft", False),
    "pending_approval": ("pending_approval", False),
    # An approved strategy cannot be changed underneath its approval.
    "active": ("pending_approval", True),
}


def next_status_for_edit(current: str) -> Tuple[str, bool]:
    """``(status after an edit, whether to bump version)``."""

    try:
        return _EDITABLE[str(current)]
    except KeyError:
        raise LifecycleError(
            f"a {current} playbook cannot be edited. Copy it into a new draft "
            "instead -- a retired strategy is kept so the proposals it raised "
            "stay legible."
        ) from None


def check_approval(
    *,
    authored_by: str,
    approver: Optional[str],
    current_status: str,
    workflow_is_active: bool,
) -> None:
    """Raise unless ``approver`` may move this playbook to ``active``."""

    subject = (approver or "").strip()
    if not subject:
        raise LifecycleError(
            "an approval must name who gave it: without a subject the "
            "self-approval bar cannot be applied at all."
        )
    if current_status != "pending_approval":
        raise LifecycleError(
            f"only a pending_approval playbook can be approved; this one is "
            f"{current_status}."
        )
    if subject.casefold() == (authored_by or "").strip().casefold():
        raise LifecycleError(
            "a playbook cannot be approved by its own author. Ask someone else "
            "to review it."
        )
    if not workflow_is_active:
        raise LifecycleError(
            "the workflow this playbook points at is missing or inactive. "
            "Approving it would create a strategy that can only fail at the "
            "moment somebody accepts its proposal."
        )


_COLUMNS = (
    "playbook_id, playbook_name, description, trigger_source, trigger_match, "
    "agent_workflow_id, params, playbook_status, version, authored_by, "
    "approved_by, approved_at, created_at, last_modified_at, last_modified_by"
)


def _row(r) -> Dict[str, Any]:
    def _obj(value):
        return json.loads(value) if isinstance(value, str) else (value or {})

    return {
        "playbook_id": r[0], "playbook_name": r[1], "description": r[2],
        "trigger_source": r[3], "trigger_match": _obj(r[4]),
        "agent_workflow_id": r[5], "params": _obj(r[6]),
        "playbook_status": r[7], "version": r[8], "authored_by": r[9],
        "approved_by": r[10],
        "approved_at": r[11].isoformat() if r[11] else None,
        "created_at": r[12].isoformat() if r[12] else None,
        "last_modified_at": r[13].isoformat() if r[13] else None,
        "last_modified_by": r[14],
    }


def get(playbook_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_playbook WHERE playbook_id = %s",
                (playbook_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return _row(row) if row else None


def list_playbooks(status: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = f"SELECT {_COLUMNS} FROM proc.bp_playbook"
    params: tuple = ()
    if status:
        sql += " WHERE playbook_status = %s"
        params = (status,)
    sql += " ORDER BY playbook_id DESC"
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            cur.close()
    return [_row(r) for r in rows]


def create(
    *,
    name: str,
    trigger_source: str,
    trigger_match: Dict[str, Any],
    agent_workflow_id: int,
    params: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    authored_by: str,
) -> int:
    """Insert a draft. Raises ``ValueError`` on an unknown match key."""

    match = validate_trigger_match(trigger_source, trigger_match or {})
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO proc.bp_playbook "
                "(playbook_name, description, trigger_source, trigger_match, "
                " agent_workflow_id, params, playbook_status, authored_by, "
                " last_modified_by) "
                "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, 'draft', %s, %s) "
                "RETURNING playbook_id",
                (name, description, trigger_source, json.dumps(match),
                 agent_workflow_id, json.dumps(params or {}),
                 authored_by, authored_by),
            )
            return int(cur.fetchone()[0])
        finally:
            cur.close()


def update(
    playbook_id: int,
    *,
    name: str,
    trigger_source: str,
    trigger_match: Dict[str, Any],
    agent_workflow_id: int,
    params: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    modified_by: str,
) -> Dict[str, Any]:
    """Edit a playbook. An active one returns to pending_approval, version + 1."""

    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    status, bump = next_status_for_edit(existing["playbook_status"])
    match = validate_trigger_match(trigger_source, trigger_match or {})
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET "
                "  playbook_name = %s, description = %s, trigger_source = %s, "
                "  trigger_match = %s::jsonb, agent_workflow_id = %s, "
                "  params = %s::jsonb, playbook_status = %s, "
                "  version = version + %s, "
                # An edit unapproves. Leaving approved_by set would leave a row
                # that says somebody signed off on text they never saw.
                "  approved_by = CASE WHEN %s THEN NULL ELSE approved_by END, "
                "  approved_at = CASE WHEN %s THEN NULL ELSE approved_at END, "
                "  last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (name, description, trigger_source, json.dumps(match),
                 agent_workflow_id, json.dumps(params or {}), status,
                 1 if bump else 0, bump, bump, modified_by, playbook_id),
            )
        finally:
            cur.close()
    return get(playbook_id)


def submit(playbook_id: int, *, modified_by: str) -> None:
    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    if existing["playbook_status"] not in ("draft", "pending_approval"):
        raise LifecycleError(
            f"a {existing['playbook_status']} playbook cannot be submitted for approval"
        )
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'pending_approval', "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (modified_by, playbook_id),
            )
        finally:
            cur.close()


def workflow_is_active(agent_workflow_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT is_active FROM proc.bp_agent_workflow WHERE workflow_id = %s",
                (agent_workflow_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return bool(row and row[0])


def approve(playbook_id: int, *, approver: str) -> Dict[str, Any]:
    existing = get(playbook_id)
    if existing is None:
        raise LifecycleError(f"no playbook {playbook_id}")
    check_approval(
        authored_by=existing["authored_by"],
        approver=approver,
        current_status=existing["playbook_status"],
        workflow_is_active=workflow_is_active(existing["agent_workflow_id"]),
    )
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'active', "
                "approved_by = %s, approved_at = now(), "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (approver, approver, playbook_id),
            )
        finally:
            cur.close()
    return get(playbook_id)


def retire(playbook_id: int, *, modified_by: str) -> None:
    """Stop a playbook proposing. Its existing proposals stay decidable."""

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook SET playbook_status = 'retired', "
                "last_modified_at = now(), last_modified_by = %s "
                "WHERE playbook_id = %s",
                (modified_by, playbook_id),
            )
        finally:
            cur.close()
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_lifecycle.py -v -p no:randomly
```
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git status --short src/repositories/playbook_repo.py tests/services/playbooks
git commit -o src/repositories/playbook_repo.py \
             tests/services/playbooks/test_lifecycle.py \
  -m "feat(playbook): the lifecycle that makes an approval mean something

draft -> pending_approval -> active -> retired, with only active playbooks
loadable by the store.

Editing an active playbook returns it to pending_approval, bumps version
and clears approved_by: an approved strategy cannot be changed underneath
its approval, and leaving the approver set would leave a row saying
somebody signed off on text they never saw.

Nobody approves their own work, and an approval that names nobody is
refused outright -- without a subject the bar cannot be applied at all.
Both rules are pure functions here rather than checks in the router, so
there is one place that holds the opinion.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: The endpoints, and one run path

Approving a proposal must execute through the **existing** workflow run path,
not a copy of it. The run endpoint's body is currently inline in
`run_workflow`; Step 1 lifts it into a module-level `start_run()` that both
callers use, so there stays exactly one place a workflow is claimed and
executed.

**Files:**
- Modify: `src/api/routers/agent_workflows.py:140-181` (extract `start_run`)
- Create: `src/api/routers/playbooks.py`
- Modify: `src/api/main.py` (import near line 55, registration near line 571)
- Test: `tests/api/test_playbooks_router.py`

**Interfaces:**
- Consumes: `playbook_repo.*`, `api.endpoint_gate.require as gate`, `api.auth.require_user`, `finding_source.validate_trigger_match`.
- Produces:
  - `agent_workflows.start_run(request, workflow_id, payload, principal) -> Dict[str, Any]` — the single run path
  - router at prefix `/playbooks` with `GET ""`, `POST ""`, `GET /{id}`, `PUT /{id}`, `POST /{id}/submit`, `POST /{id}/approve`, `POST /{id}/retire`, `GET /proposals`, `POST /proposals/{id}/approve`, `POST /proposals/{id}/reject`

- [ ] **Step 1: Extract the run path**

In `src/api/routers/agent_workflows.py`, replace the body of `run_workflow`
below its `gate(...)` call with a call to a new module-level function, and put
the lifted body in that function verbatim:

```python
def start_run(
    request: Request,
    workflow_id: int,
    payload: Dict[str, Any],
    principal: Any,
) -> Dict[str, Any]:
    """Start a saved workflow, or stop and ask. THE run path.

    Lifted out of ``run_workflow`` so the playbook router can accept a proposal
    without copying it. A second copy would be a second place that claims a
    run, and claiming is what stops a replay from sending the same email twice.
    """

    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")

    run_id = f"awf-{workflow_id}-{uuid.uuid4().hex[:8]}"
    answers = reqrepo.answers_for(run_id)          # empty on a fresh run
    started_by = getattr(principal, "subject", None) or None

    missing = pending_requests(wf["graph"], payload, answers)
    if missing:
        reqrepo.create_run(run_id, agent_workflow_id=workflow_id, payload=payload,
                           status="awaiting_input", initiated_by=started_by)
        reqrepo.raise_requests(run_id, missing, agent_workflow_id=workflow_id)
        return {
            "run_id": run_id, "status": "awaiting_input",
            "pending": reqrepo.open_requests(run_id),
            "nodes": _describe_nodes(wf["graph"]),
        }

    reqrepo.create_run(run_id, agent_workflow_id=workflow_id, payload=payload,
                       status="pending", initiated_by=started_by)
    return _claim_and_execute(request, run_id, wf, {**payload, **answers}, started_by)


@router.post("/{workflow_id}/run")
def run_workflow(
    workflow_id: int, body: RunBody, request: Request,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    gate("workflow.run", principal, agent="AgentWorkflowsRouter",
         context={"workflow_id": workflow_id})
    return start_run(request, workflow_id, body.payload, principal)
```

Keep every comment from the original body on the lines it belonged to — they
explain the claim, and losing them loses the reason.

- [ ] **Step 2: Confirm the existing workflow tests still pass**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/test_agent_workflows_router.py -v -p no:randomly
```
Expected: PASS at whatever count it passed at before the edit. A pure extraction
that changes a count has changed behaviour — stop and find out why.

- [ ] **Step 3: Write the failing router test**

Create `tests/api/test_playbooks_router.py`:

```python
"""The endpoints: lifecycle, the self-approval bar, and proposal decisions.

The repository is stubbed. What is being tested here is the router's contract
-- which gate it calls, which status code a refusal gets, and that approving a
proposal goes through the one run path rather than a copy of it.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from api.routers import playbooks as mod  # noqa: E402


class Principal:
    def __init__(self, subject="bo"):
        self.subject = subject


@pytest.fixture
def client(monkeypatch):
    gated = []
    monkeypatch.setattr(mod, "gate", lambda action, principal, **kw: gated.append(action))
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[mod.require_user] = lambda: Principal()
    c = TestClient(app)
    c.gated = gated
    return c


def test_creating_a_playbook_gates_on_playbook_write(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "create", lambda **kw: 12)
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_id": 12, "playbook_status": "draft"})
    r = client.post("/playbooks", json={
        "playbook_name": "Recover the duplicate",
        "trigger_source": "detection_finding",
        "trigger_match": {"rule_id": "duplicate"},
        "agent_workflow_id": 958,
    })
    assert r.status_code == 200, r.text
    assert r.json()["playbook_id"] == 12
    assert client.gated == ["playbook.write"]


def test_an_unknown_match_key_is_a_400_naming_the_key(client, monkeypatch):
    """Not a 500, and not a stored row. The author gets told which key."""
    def boom(**kw):
        raise ValueError("sevrity is not a match field for detection_finding")
    monkeypatch.setattr(mod.repo, "create", boom)
    r = client.post("/playbooks", json={
        "playbook_name": "Typo",
        "trigger_source": "detection_finding",
        "trigger_match": {"sevrity": "high"},
        "agent_workflow_id": 958,
    })
    assert r.status_code == 400
    assert "sevrity" in r.json()["detail"]


def test_approving_a_playbook_gates_on_playbook_approve(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "approve",
                        lambda pid, approver: {"playbook_id": pid, "playbook_status": "active",
                                               "approved_by": approver})
    r = client.post("/playbooks/12/approve")
    assert r.status_code == 200
    assert r.json()["playbook_status"] == "active"
    assert client.gated == ["playbook.approve"]


def test_self_approval_comes_back_as_a_403(client, monkeypatch):
    """A lifecycle refusal is not a malformed request; it is a refusal."""
    def boom(pid, approver):
        raise mod.LifecycleError("a playbook cannot be approved by its own author")
    monkeypatch.setattr(mod.repo, "approve", boom)
    r = client.post("/playbooks/12/approve")
    assert r.status_code == 403
    assert "own author" in r.json()["detail"]


def test_editing_a_retired_playbook_is_a_400(client, monkeypatch):
    def boom(pid, **kw):
        raise mod.LifecycleError("a retired playbook cannot be edited")
    monkeypatch.setattr(mod.repo, "update", boom)
    r = client.put("/playbooks/12", json={
        "playbook_name": "x", "trigger_source": "detection_finding",
        "trigger_match": {}, "agent_workflow_id": 958,
    })
    assert r.status_code == 400


def test_listing_proposals_needs_no_gate(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "list_proposals",
                        lambda status=None, limit=100: [{"proposal_id": 1}])
    r = client.get("/playbooks/proposals?status=proposed")
    assert r.status_code == 200
    assert r.json()["proposals"] == [{"proposal_id": 1}]
    assert client.gated == []


def test_approving_a_proposal_gates_on_workflow_run_and_uses_the_one_run_path(
    client, monkeypatch
):
    """No new action name, and no second copy of the claim-and-execute path."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "playbook_id": 12, "proposal_status": "proposed",
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {"tier": "gold"},
    })
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    started = {}
    def fake_start_run(request, workflow_id, payload, principal):
        started.update(workflow_id=workflow_id, payload=payload)
        return {"run_id": "awf-958-abc", "status": "completed"}
    monkeypatch.setattr(mod, "start_run", fake_start_run)
    monkeypatch.setattr(mod.repo, "mark_proposal_executed", lambda pid, run_id, by: None)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == "awf-958-abc"
    assert client.gated == ["workflow.run"]
    assert started["workflow_id"] == 958
    # The playbook's static params and the finding it was raised for both reach
    # the graph -- a strategy that does not know which finding it is answering
    # is not a strategy.
    assert started["payload"]["tier"] == "gold"
    assert started["payload"]["finding_id"] == "4211"
    assert started["payload"]["deal_id"] == "D-900"


def test_a_proposal_whose_finding_is_resolved_is_superseded_not_executed(
    client, monkeypatch
):
    """A stale queue must not act on closed work."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "playbook_id": 12, "proposal_status": "proposed",
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: False)
    superseded = {}
    monkeypatch.setattr(mod.repo, "mark_proposal_superseded",
                        lambda pid, by: superseded.update(pid=pid, by=by))
    def never(*a, **kw):
        raise AssertionError("a superseded proposal must not start a run")
    monkeypatch.setattr(mod, "start_run", never)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200
    assert r.json()["proposal_status"] == "superseded"
    assert superseded["pid"] == 5


def test_a_retired_playbooks_existing_proposal_is_still_decidable(client, monkeypatch):
    """Retiring stops a playbook proposing. It must not strand the proposals it
    already raised -- somebody still has to say yes or no to those."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected", lambda pid, by, reason: None)
    monkeypatch.setattr(mod.agent_actions, "record_action", lambda **kw: None)
    r = client.post("/playbooks/proposals/5/reject", json={"reason": "strategy retired"})
    assert r.status_code == 200


def test_a_proposal_already_decided_cannot_be_decided_again(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "executed", "playbook_id": 12,
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 409
    assert "executed" in r.json()["detail"]


def test_rejecting_a_proposal_records_who_and_why_and_runs_nothing(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    rejected = {}
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected",
                        lambda pid, by, reason: rejected.update(pid=pid, by=by, reason=reason))
    def never(*a, **kw):
        raise AssertionError("rejecting must not start a run")
    monkeypatch.setattr(mod, "start_run", never)

    r = client.post("/playbooks/proposals/5/reject", json={"reason": "already credited"})
    assert r.status_code == 200
    assert rejected == {"pid": 5, "by": "bo", "reason": "already credited"}


def test_a_rejection_is_recorded_in_the_event_log(client, monkeypatch):
    """Rejecting is ungated, so without this the most interesting outcome -- a
    person looked at the recommendation and said no -- leaves no trail."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected", lambda pid, by, reason: None)
    events = []
    monkeypatch.setattr(mod.agent_actions, "record_action", lambda **kw: events.append(kw))

    client.post("/playbooks/proposals/5/reject", json={"reason": "already credited"})
    assert [e["action_type"] for e in events] == ["proposal.rejected"]
    assert events[0]["details"]["reason"] == "already credited"
    assert events[0]["details"]["decided_by"] == "bo"


def test_a_rejection_must_say_why(client, monkeypatch):
    """A rejected recommendation with no reason teaches nobody anything."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    r = client.post("/playbooks/proposals/5/reject", json={"reason": "   "})
    assert r.status_code == 400
```

- [ ] **Step 4: Run it and watch it fail**

```sh
./venv/bin/python -m pytest tests/api/test_playbooks_router.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'api.routers.playbooks'`.

- [ ] **Step 5: Write the router**

Create `src/api/routers/playbooks.py`:

```python
"""Playbooks and the proposals they raise.

Two different approvals live here and they are not the same act:

  * Approving a PLAYBOOK makes a strategy active. It changes what the system
    will recommend for every matching finding from now on, so it is
    configuration and gates on ``playbook.approve``.
  * Approving a PROPOSAL runs a workflow, once, for one finding. It gates on
    the existing ``workflow.run`` and executes through
    ``agent_workflows.start_run`` -- the one run path -- rather than a copy.

Nothing here runs a workflow without a person having called an approve
endpoint. There is deliberately no autorun flag to add later without a
conversation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.routers.agent_workflows import start_run
from repositories import playbook_repo as repo
from repositories.playbook_repo import LifecycleError
from services import agent_actions
from services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playbooks", tags=["Playbooks"])


class PlaybookBody(BaseModel):
    playbook_name: str
    trigger_source: str
    trigger_match: Dict[str, Any] = Field(default_factory=dict)
    agent_workflow_id: int
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class RejectBody(BaseModel):
    reason: str = ""


def _subject(principal: Any) -> str:
    return getattr(principal, "subject", None) or ""


_OPEN_SQL = {
    "detection_finding": (
        "SELECT 1 FROM proc.bp_detection_finding "
        " WHERE finding_id = %s::bigint AND status = 'open'"
    ),
    "opportunity": (
        "SELECT 1 FROM proc.bp_opportunity "
        " WHERE opportunity_id = %s AND retired_at IS NULL"
    ),
}


def finding_is_open(source: str, finding_id: str) -> bool:
    """Is the finding this proposal was raised for still open?

    Asked at approval time, not at proposal time. A queue that sat for a week
    must not act on work somebody has since closed.
    """

    sql = _OPEN_SQL.get(source)
    if sql is None:
        return False
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, (finding_id,))
            return cur.fetchone() is not None
        finally:
            cur.close()


# -- playbooks -----------------------------------------------------------

@router.get("")
def list_playbooks(status: Optional[str] = None) -> Dict[str, Any]:
    return {"playbooks": repo.list_playbooks(status)}


@router.post("")
def create_playbook(body: PlaybookBody, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter")
    try:
        playbook_id = repo.create(
            name=body.playbook_name,
            trigger_source=body.trigger_source,
            trigger_match=body.trigger_match,
            agent_workflow_id=body.agent_workflow_id,
            params=body.params,
            description=body.description,
            authored_by=_subject(principal),
        )
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        # An unknown or null match key. The author is told which one, because a
        # playbook that never fires is indistinguishable from one that has not
        # matched yet.
        raise HTTPException(status_code=400, detail=str(exc))
    return {"playbook_id": playbook_id, "playbook": repo.get(playbook_id)}


@router.get("/proposals")
def list_proposals(status: Optional[str] = "proposed", limit: int = 100) -> Dict[str, Any]:
    return {"proposals": repo.list_proposals(status=status, limit=limit)}


@router.get("/{playbook_id}")
def get_playbook(playbook_id: int) -> Dict[str, Any]:
    found = repo.get(playbook_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such playbook")
    return found


@router.put("/{playbook_id}")
def update_playbook(
    playbook_id: int, body: PlaybookBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        return repo.update(
            playbook_id,
            name=body.playbook_name,
            trigger_source=body.trigger_source,
            trigger_match=body.trigger_match,
            agent_workflow_id=body.agent_workflow_id,
            params=body.params,
            description=body.description,
            modified_by=_subject(principal),
        )
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{playbook_id}/submit")
def submit_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        repo.submit(playbook_id, modified_by=_subject(principal))
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return repo.get(playbook_id)


@router.post("/{playbook_id}/approve")
def approve_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.approve", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        return repo.approve(playbook_id, approver=_subject(principal))
    except LifecycleError as exc:
        # Self-approval and "this workflow is inactive" are refusals, not
        # malformed requests, so they are 403 rather than 400.
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/{playbook_id}/retire")
def retire_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id, "retiring": True})
    repo.retire(playbook_id, modified_by=_subject(principal))
    return repo.get(playbook_id)


# -- proposals -----------------------------------------------------------

def _audit(
    action_type: str,
    proposal: Dict[str, Any],
    subject: str,
    *,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record what was decided about a proposal.

    gate() already audits playbook.write, playbook.approve and workflow.run.
    It does not cover these: rejecting is ungated (refusing to act is not an
    act) and superseding happens after the gate has passed. Without this, the
    two most interesting outcomes -- a person said no, and the work had already
    been closed -- would leave no trail at all.
    """

    agent_actions.record_action(
        phase="playbook",
        action_type=action_type,
        agent="PlaybooksRouter",
        deal_id=proposal.get("deal_id"),
        status=status,
        summary=(
            f"proposal {proposal['proposal_id']} ({proposal.get('playbook_name')}) "
            f"for {proposal['finding_source']} {proposal['finding_id']}: {status}"
        ),
        details={
            "proposal_id": proposal["proposal_id"],
            "playbook_id": proposal["playbook_id"],
            "finding_source": proposal["finding_source"],
            "finding_id": proposal["finding_id"],
            "decided_by": subject,
            **(extra or {}),
        },
    )


def _decidable(proposal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not proposal:
        raise HTTPException(status_code=404, detail="No such proposal")
    status = proposal.get("proposal_status")
    if status != "proposed":
        raise HTTPException(
            status_code=409,
            detail=f"proposal {proposal['proposal_id']} is already {status}",
        )
    return proposal


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal(
    proposal_id: int, request: Request, principal=Depends(require_user)
) -> Dict[str, Any]:
    """Accept the recommendation and run the strategy. Once."""

    gate("workflow.run", principal, agent="PlaybooksRouter",
         context={"proposal_id": proposal_id})
    proposal = _decidable(repo.get_proposal(proposal_id))
    subject = _subject(principal)

    if not finding_is_open(proposal["finding_source"], proposal["finding_id"]):
        repo.mark_proposal_superseded(proposal_id, subject)
        _audit("proposal.superseded", proposal, subject, status="superseded")
        return {"proposal_id": proposal_id, "proposal_status": "superseded",
                "detail": "the finding this was raised for is no longer open"}

    payload = dict(proposal.get("params") or {})
    # The strategy is told which finding it is answering. A graph that does not
    # know that can only act on the whole corpus.
    payload.update({
        "finding_source": proposal["finding_source"],
        "finding_id": proposal["finding_id"],
        "deal_id": proposal.get("deal_id"),
        "playbook_id": proposal["playbook_id"],
        "proposal_id": proposal_id,
    })
    result = start_run(request, proposal["agent_workflow_id"], payload, principal)
    repo.mark_proposal_executed(proposal_id, result.get("run_id"), subject)
    _audit("proposal.executed", proposal, subject, status="executed",
           extra={"run_id": result.get("run_id")})
    return {"proposal_id": proposal_id, "proposal_status": "executed", **result}


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal(
    proposal_id: int, body: RejectBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    """Decline the recommendation. Runs nothing, and needs no gate: refusing to
    act is not an act. The reason is required, because a rejected
    recommendation with no reason teaches nobody anything."""

    proposal = _decidable(repo.get_proposal(proposal_id))
    reason = (body.reason or "").strip()
    if not reason:
        raise HTTPException(
            status_code=400,
            detail="a rejection must say why -- it is the only signal an author gets",
        )
    repo.mark_proposal_rejected(proposal_id, _subject(principal), reason)
    _audit("proposal.rejected", proposal, _subject(principal),
           status="rejected", extra={"reason": reason})
    return {"proposal_id": proposal_id, "proposal_status": "rejected"}
```

- [ ] **Step 6: Add the proposal functions to the repository**

Append to `src/repositories/playbook_repo.py`:

```python
# -- proposals -----------------------------------------------------------

_PROPOSAL_COLUMNS = (
    "p.proposal_id, p.playbook_id, p.finding_source, p.finding_id, p.deal_id, "
    "p.proposal_status, p.evidence, p.run_id, p.proposed_at, p.decided_by, "
    "p.decided_at, p.decision_reason, b.playbook_name, b.agent_workflow_id, b.params"
)


def _proposal_row(r) -> Dict[str, Any]:
    def _obj(value):
        return json.loads(value) if isinstance(value, str) else (value or {})

    return {
        "proposal_id": r[0], "playbook_id": r[1], "finding_source": r[2],
        "finding_id": r[3], "deal_id": r[4], "proposal_status": r[5],
        "evidence": _obj(r[6]), "run_id": r[7],
        "proposed_at": r[8].isoformat() if r[8] else None,
        "decided_by": r[9],
        "decided_at": r[10].isoformat() if r[10] else None,
        "decision_reason": r[11], "playbook_name": r[12],
        "agent_workflow_id": r[13], "params": _obj(r[14]),
    }


def get_proposal(proposal_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM proc.bp_playbook_proposal p "
                "JOIN proc.bp_playbook b ON b.playbook_id = p.playbook_id "
                "WHERE p.proposal_id = %s",
                (proposal_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    return _proposal_row(row) if row else None


def list_proposals(status: Optional[str] = "proposed", limit: int = 100) -> List[Dict[str, Any]]:
    sql = (
        f"SELECT {_PROPOSAL_COLUMNS} FROM proc.bp_playbook_proposal p "
        "JOIN proc.bp_playbook b ON b.playbook_id = p.playbook_id"
    )
    params: tuple = ()
    if status:
        sql += " WHERE p.proposal_status = %s"
        params = (status,)
    sql += " ORDER BY p.proposed_at DESC LIMIT %s"
    params = params + (max(1, min(int(limit), 1000)),)
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            cur.close()
    return [_proposal_row(r) for r in rows]


def _decide(proposal_id: int, status: str, by: str,
            reason: Optional[str] = None, run_id: Optional[str] = None) -> None:
    """Record a decision, once.

    The WHERE clause pins proposal_status = 'proposed' so two people clicking
    approve at the same moment cannot both write a decision -- only the first
    UPDATE matches.
    """

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE proc.bp_playbook_proposal "
                "   SET proposal_status = %s, decided_by = %s, decided_at = now(), "
                "       decision_reason = COALESCE(%s, decision_reason), "
                "       run_id = COALESCE(%s, run_id) "
                " WHERE proposal_id = %s AND proposal_status = 'proposed'",
                (status, by, reason, run_id, proposal_id),
            )
        finally:
            cur.close()


def mark_proposal_executed(proposal_id: int, run_id: Optional[str], by: str) -> None:
    _decide(proposal_id, "executed", by, run_id=run_id)


def mark_proposal_rejected(proposal_id: int, by: str, reason: str) -> None:
    _decide(proposal_id, "rejected", by, reason=reason)


def mark_proposal_superseded(proposal_id: int, by: str) -> None:
    _decide(proposal_id, "superseded", by,
            reason="the finding this was raised for is no longer open")
```

- [ ] **Step 7: Register the router**

In `src/api/main.py`, beside the other router imports (near line 55):

```python
from api.routers import playbooks as playbooks_router
```

and in the registration list (near line 571), after `agent_workflows_router.router,`:

```python
    agent_workflows_router.router,
    playbooks_router.router,
```

- [ ] **Step 8: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/api/test_playbooks_router.py \
                           tests/test_agent_workflows_router.py -v -p no:randomly
```
Expected: PASS, 13 new tests plus the workflow router's existing count unchanged.

- [ ] **Step 9: Commit**

```bash
git status --short src/api tests/api src/repositories/playbook_repo.py
git commit -o src/api/routers/agent_workflows.py \
             src/api/routers/playbooks.py \
             src/api/main.py \
             src/repositories/playbook_repo.py \
             tests/api/test_playbooks_router.py \
  -m "feat(playbook): the endpoints, and still only one run path

Approving a playbook and approving a proposal are different acts. The first
makes a strategy active for every future matching finding -- configuration,
gated on playbook.approve. The second runs one workflow for one finding,
gated on the existing workflow.run.

run_workflow's body is lifted into start_run() and both callers use it. A
second copy would be a second place that claims a run, and the claim is
what stops a replay from sending the same email twice.

A proposal whose finding has since closed is marked superseded rather than
executed: a queue that sat for a week must not act on closed work. A
rejection must say why -- it is the only signal the author gets.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: The sweep, and the job that runs it

Batched by key, not by `OFFSET` — deal assignment already proved that a scan
which re-reads from the start does not survive the corpus. There are 4,838 open
detection findings and 308 opportunities today.

Registered on `BackendScheduler` following `_register_style_staging_sweep_job` /
`_run_style_staging_sweep`. It ships proposing into an empty playbook table, so
it is a no-op until an expert authors something.

**Files:**
- Create: `src/services/playbooks/sweep.py`
- Modify: `src/services/backend_scheduler.py` (`_register_default_jobs`, near line 426)
- Test: `tests/services/playbooks/test_sweep.py`

**Interfaces:**
- Consumes: `store.PlaybookStore`, `selector.select`, `selector.tied_candidates`, `proposer.propose`, `proposer.record_ambiguous`, `finding_source.OPEN_SQL`, `finding_source.normalise`.
- Produces:
  - `SweepReport(scanned: int, proposed: int, already_queued: int, ambiguous: int, unmatched: int)` with `.render() -> str`
  - `sweep(*, store=None, batch_size=500, conn=None) -> SweepReport`

- [ ] **Step 1: Write the failing test**

Create `tests/services/playbooks/test_sweep.py`:

```python
"""The sweep: what it counts, and what it refuses to do."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks import sweep as mod  # noqa: E402
from src.services.playbooks.store import Playbook, PlaybookStore  # noqa: E402


class FakeCursor:
    """Yields each source's rows once, then an empty page to end the loop."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.description = None
        self._rows = []

    def execute(self, sql, params=None):
        page = self._pages.pop(0) if self._pages else []
        self.description = [(c,) for c in page[0]] if page else [("finding_id",)]
        self._rows = page[1] if page else []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


DF_COLS = ("finding_id", "rule_id", "category", "severity", "doc_type",
           "blocks_promotion", "deal_id")
OPP_COLS = ("opportunity_id", "detector_type", "supplier_id", "category_id", "deal_id")


def pages(detection_rows=(), opportunity_rows=()):
    """One full page then an empty one, per source, in sweep order."""
    return [
        (DF_COLS, list(detection_rows)), (DF_COLS, []),
        (OPP_COLS, list(opportunity_rows)), (OPP_COLS, []),
    ]


def store_with(*playbooks):
    s = PlaybookStore(playbook_rows=[])
    s._playbooks = list(playbooks)
    return s


def pb(pid, match, source="detection_finding", name=None):
    return Playbook(playbook_id=pid, playbook_name=name or f"pb{pid}",
                    trigger_source=source, trigger_match=match,
                    agent_workflow_id=958, params={}, version=1)


@pytest.fixture(autouse=True)
def captured(monkeypatch):
    calls = {"proposed": [], "ambiguous": []}
    monkeypatch.setattr(mod.proposer, "propose",
                        lambda f, s, conn=None: calls["proposed"].append((f.finding_id, s.playbook.playbook_id)) or len(calls["proposed"]))
    monkeypatch.setattr(mod.proposer, "record_ambiguous",
                        lambda f, tied: calls["ambiguous"].append(f.finding_id))
    return calls


def test_a_matching_finding_is_proposed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert captured["proposed"] == [("4211", 7)]
    assert (report.scanned, report.proposed, report.unmatched) == (1, 1, 0)


def test_an_unmatched_finding_is_counted_not_proposed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "quantity", "quantity", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert captured["proposed"] == []
    assert (report.scanned, report.proposed, report.unmatched) == (1, 0, 1)


def test_an_ambiguous_match_proposes_nothing_and_is_recorded(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(
        store=store_with(pb(1, {"rule_id": "duplicate"}), pb(2, {"severity": "critical"})),
        conn=conn,
    )
    assert captured["proposed"] == []
    assert captured["ambiguous"] == ["4211"]
    assert report.ambiguous == 1


def test_both_sources_are_swept_and_never_crossed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")],
        opportunity_rows=[("OPP-1", "Invoice Overbilling", "SUP-3", None, "D-901")],
    )))
    report = mod.sweep(
        store=store_with(
            pb(1, {"rule_id": "duplicate"}),
            pb(2, {"detector_type": "Invoice Overbilling"}, source="opportunity"),
        ),
        conn=conn,
    )
    assert sorted(captured["proposed"]) == [("4211", 1), ("OPP-1", 2)]
    assert report.scanned == 2


def test_an_already_queued_finding_counts_separately(monkeypatch):
    """propose() returning None is the ordinary case on every sweep after the
    first, and must not be reported as work done."""
    monkeypatch.setattr(mod.proposer, "propose", lambda f, s, conn=None: None)
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert (report.proposed, report.already_queued) == (0, 1)


def test_an_empty_playbook_table_sweeps_to_a_clean_zero(captured):
    """Shipping day. Not an error, and the count still gets logged."""
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(), conn=conn)
    assert (report.scanned, report.proposed) == (1, 0)
    assert "0 proposed" in report.render()


def test_the_report_renders_every_count():
    text = mod.SweepReport(scanned=10, proposed=2, already_queued=7,
                           ambiguous=1, unmatched=0).render()
    for fragment in ("10 scanned", "2 proposed", "7 already queued",
                     "1 ambiguous", "0 unmatched"):
        assert fragment in text
```

- [ ] **Step 2: Run it and watch it fail**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_sweep.py -v -p no:randomly
```
Expected: FAIL at collection — `No module named 'src.services.playbooks.sweep'`.

- [ ] **Step 3: Write the sweep**

Create `src/services/playbooks/sweep.py`:

```python
"""Walk the open findings, select, propose. Nothing else.

Batched on the key, not on OFFSET. Deal assignment already proved that a scan
re-reading from the start does not survive this corpus, and there are 4,838
open detection findings today.

The count is logged on EVERY run, including zero. A sweep that logged only
when it found something would be indistinguishable from one that had silently
stopped running -- and an empty playbook table, which is the honest state until
an expert authors a strategy, produces exactly that zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.services.db import get_conn

from . import proposer
from .finding_source import OPEN_SQL, SOURCES, normalise
from .selector import select, tied_candidates
from .store import PlaybookStore, load_playbook_store

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


@dataclass
class SweepReport:
    scanned: int = 0
    proposed: int = 0
    already_queued: int = 0
    ambiguous: int = 0
    unmatched: int = 0

    def render(self) -> str:
        return (
            f"playbook sweep: {self.scanned} scanned, {self.proposed} proposed, "
            f"{self.already_queued} already queued, {self.ambiguous} ambiguous, "
            f"{self.unmatched} unmatched"
        )


def _pages(conn: Any, source: str, batch_size: int) -> Iterator[list]:
    """Rows of ``source``, keyset-paginated. Yields dicts, a page at a time."""

    # Start below every key the source holds. finding_id is a BIGINT and its
    # identity sequence starts at 1; opportunity_id is a VARCHAR and the empty
    # string sorts below every non-empty one.
    cursor_key: Any = 0 if source == "detection_finding" else ""
    while True:
        cur = conn.cursor()
        try:
            cur.execute(OPEN_SQL[source], (cursor_key, batch_size))
            columns = [c[0] for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
        finally:
            cur.close()
        if not rows:
            return
        yield rows
        last = rows[-1]
        cursor_key = last["finding_id" if source == "detection_finding" else "opportunity_id"]


def sweep(
    *,
    store: Optional[PlaybookStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    conn: Optional[Any] = None,
) -> SweepReport:
    """Propose a playbook for every open finding that one governs.

    Proposes only. Nothing here starts a workflow; the proposal waits for a
    person to accept it through the endpoint.
    """

    active = store if store is not None else load_playbook_store()
    if active is None:
        logger.error("playbook sweep skipped: the store could not be read")
        return SweepReport()

    report = SweepReport()
    if conn is not None:
        _run(conn, active, batch_size, report)
    else:
        with get_conn() as own:
            _run(own, active, batch_size, report)

    logger.info("%s", report.render())
    return report


def _run(conn: Any, store: PlaybookStore, batch_size: int, report: SweepReport) -> None:
    for source in SOURCES:
        playbooks = store.for_source(source)
        for page in _pages(conn, source, batch_size):
            for row in page:
                report.scanned += 1
                try:
                    finding = normalise(source, row)
                except ValueError:
                    logger.exception("skipping unreadable %s row", source)
                    continue
                if not playbooks:
                    report.unmatched += 1
                    continue
                selection = select(finding, playbooks)
                if selection is None:
                    tied = tied_candidates(finding, playbooks)
                    if tied:
                        proposer.record_ambiguous(finding, tied)
                        report.ambiguous += 1
                    else:
                        report.unmatched += 1
                    continue
                if proposer.propose(finding, selection, conn=conn) is None:
                    report.already_queued += 1
                else:
                    report.proposed += 1
```

- [ ] **Step 4: Run it and watch it pass**

```sh
./venv/bin/python -m pytest tests/services/playbooks/test_sweep.py -v -p no:randomly
```
Expected: PASS, 7 tests.

- [ ] **Step 5: Register the job**

In `src/services/backend_scheduler.py`, add to `_register_default_jobs()` after
`self._register_triage_job()`:

```python
        self._register_playbook_sweep_job()
```

and add the pair beside the other sweeps:

```python
    PLAYBOOK_SWEEP_JOB_NAME = "playbook-sweep"

    def _register_playbook_sweep_job(self) -> None:
        """Propose a playbook for every open finding that one governs.

        Proposes only — a proposal waits for a person, and nothing in this job
        can start a workflow. It ships against an empty proc.bp_playbook, so it
        is a deliberate no-op until an expert authors a strategy.

        Interval via PLAYBOOK_SWEEP_INTERVAL_MINUTES (default 30).
        """
        import os
        if self.PLAYBOOK_SWEEP_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("PLAYBOOK_SWEEP_INTERVAL_MINUTES", "30"))
        except ValueError:
            minutes = 30
        self.register_job(
            self.PLAYBOOK_SWEEP_JOB_NAME,
            self._run_playbook_sweep,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _run_playbook_sweep(self) -> None:
        """Run one playbook sweep, logging the count whatever it is."""
        try:
            from services.playbooks.sweep import sweep

            sweep()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Playbook sweep failed")
```

- [ ] **Step 6: Confirm the job registers**

`tests/test_backend_scheduler.py` is one of the known-broken collectors, so
check the registration directly:

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from services.backend_scheduler import BackendScheduler
names = [n for n in dir(BackendScheduler) if 'playbook' in n.lower()]
print('methods:', names)
assert '_register_playbook_sweep_job' in names and '_run_playbook_sweep' in names
print('job name:', BackendScheduler.PLAYBOOK_SWEEP_JOB_NAME)
"
```
Expected: both methods listed, job name `playbook-sweep`.

- [ ] **Step 7: Commit**

```bash
git status --short src/services/playbooks src/services/backend_scheduler.py tests/services/playbooks
git commit -o src/services/playbooks/sweep.py \
             src/services/backend_scheduler.py \
             tests/services/playbooks/test_sweep.py \
  -m "feat(playbook): the sweep, proposing into an empty table on purpose

Keyset pagination, not OFFSET -- deal assignment already proved a scan that
re-reads from the start does not survive 4,838 open findings.

The count is logged on every run including zero. A sweep that logged only
on a hit would be indistinguishable from one that had silently stopped, and
zero is exactly what an empty playbook table produces until somebody
authors a strategy. It ships as a deliberate no-op.

Nothing here starts a workflow. The proposal waits for a person.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: Break every guard on purpose, then prove it live

Three guards have shipped green in this codebase while checking nothing. These
will not be the next three. **A guard you have not watched go red is a guard you
have not tested** — do not skip a proof because the code "obviously" works.

This task has no `git add` of its own until the end: the deliverable is the
evidence, and the evidence goes in the commit message.

**Files:**
- Create: `tests/services/playbooks/test_guard_proofs.py`
- Test: itself, plus a live run against the local server

**Interfaces:**
- Consumes: everything above.
- Produces: the acceptance evidence.

- [ ] **Step 1: Write the guard proofs that can be proved in-process**

Create `tests/services/playbooks/test_guard_proofs.py`:

```python
"""Each guard, broken on purpose, watched fail.

A test that only exercises the happy path proves the code runs, not that the
guard guards. These invert each one.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.repositories.playbook_repo import LifecycleError, check_approval  # noqa: E402
from src.services.playbooks.finding_source import Finding, validate_trigger_match  # noqa: E402
from src.services.playbooks.selector import select  # noqa: E402
from src.services.playbooks.store import Playbook, PlaybookStore  # noqa: E402


def pb(pid, match, source="detection_finding"):
    return Playbook(playbook_id=pid, playbook_name=f"pb{pid}", trigger_source=source,
                    trigger_match=match, agent_workflow_id=958, params={}, version=1)


def f(**attrs):
    return Finding(source="detection_finding", finding_id="1", deal_id="D-1", attrs=attrs)


def test_guard_the_tie_rule_actually_refuses():
    """Remove the tie and one is chosen; restore it and nothing is. If the
    first half of this passes and the second does too, the tie rule is doing
    the work rather than the test asserting a foregone conclusion."""
    one = [pb(1, {"rule_id": "duplicate"})]
    two = [pb(1, {"rule_id": "duplicate"}), pb(2, {"severity": "critical"})]
    finding = f(rule_id="duplicate", severity="critical")
    assert select(finding, one) is not None      # the same finding DOES match
    assert select(finding, two) is None          # and the tie is what stops it


def test_guard_the_self_approval_bar_is_the_thing_refusing():
    """Change only the approver and the same call succeeds."""
    with pytest.raises(LifecycleError):
        check_approval(authored_by="ana", approver="ana",
                       current_status="pending_approval", workflow_is_active=True)
    check_approval(authored_by="ana", approver="bo",
                   current_status="pending_approval", workflow_is_active=True)


def test_guard_the_match_key_allow_list_is_the_thing_refusing():
    validate_trigger_match("detection_finding", {"severity": "critical"})
    with pytest.raises(ValueError):
        validate_trigger_match("detection_finding", {"sevrity": "critical"})


def test_guard_active_only_is_the_thing_filtering():
    row = {
        "playbook_id": 1, "playbook_name": "x", "trigger_source": "detection_finding",
        "trigger_match": {}, "agent_workflow_id": 958, "params": {},
        "playbook_status": "active", "version": 1,
    }
    assert len(PlaybookStore(playbook_rows=[dict(row)]).active_playbooks()) == 1
    assert PlaybookStore(
        playbook_rows=[dict(row, playbook_status="retired")]
    ).active_playbooks() == []


def test_guard_a_null_attribute_is_not_a_wildcard():
    """If this ever starts passing with doc_type=None, the comparison has
    started treating absence as a match and every specific playbook has
    quietly become a catch-all."""
    specific = [pb(1, {"rule_id": "duplicate", "doc_type": "invoice"})]
    assert select(f(rule_id="duplicate", doc_type="invoice"), specific) is not None
    assert select(f(rule_id="duplicate", doc_type=None), specific) is None
```

- [ ] **Step 2: Run them**

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9
./venv/bin/python -m pytest tests/services/playbooks/test_guard_proofs.py -v -p no:randomly
```
Expected: PASS, 5 tests.

- [ ] **Step 3: Break the unique index on purpose, in bp_testdb, and watch duplicates appear**

This one cannot be proved in-process — the index is the guard. Record the real
output of every command below; it goes in the commit message.

Steps 3 to 5 share one shell. Set it up once and keep it:

```sh
set -a; . ./.env; set +a
psql_t() { PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" \
             -U "$DB_USER" -d bp_testdb -tAc "$1"; }

# A throwaway active playbook on the smallest rule in the corpus (bad_po_ref,
# 3 open findings), approved by somebody other than its author, pointing at an
# ACTIVE workflow so the store will load it.
PB=$(psql_t "INSERT INTO proc.bp_playbook
          (playbook_name, trigger_source, trigger_match, agent_workflow_id,
           playbook_status, authored_by, approved_by, approved_at)
        VALUES ('__guard_proof', 'detection_finding', '{\"rule_id\":\"bad_po_ref\"}',
                958, 'active', 'proof_author', 'proof_approver', now())
        RETURNING playbook_id")
echo "playbook $PB"

# Sweep twice with the index in place.
./venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from services.playbooks.sweep import sweep
print(sweep().render()); print(sweep().render())
"
psql_t "SELECT count(*) FROM proc.bp_playbook_proposal WHERE playbook_id = $PB"
# EXPECT: the second sweep reports 0 proposed / N already queued, and the count
# equals the number of open bad_po_ref findings (3 today), not twice it.

# Now drop the guard and sweep again.
psql_t "DROP INDEX proc.ux_bp_playbook_proposal_finding"
./venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from services.playbooks.sweep import sweep
print(sweep().render())
"
psql_t "SELECT count(*) FROM proc.bp_playbook_proposal WHERE playbook_id = $PB"
# EXPECT (THE GUARD GOING RED): the count has doubled. Duplicates appeared the
# moment the index went. If it did NOT double, idempotency is coming from
# somewhere other than the index and you do not yet know where.
```

- [ ] **Step 4: Put it back and clean up**

```sh
psql_t "DELETE FROM proc.bp_playbook_proposal WHERE playbook_id = $PB"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" \
  -d bp_testdb -v ON_ERROR_STOP=1 -f deploy/sql/2026-09-29_bp_playbook.sql
psql_t "SELECT indexname FROM pg_indexes
         WHERE schemaname='proc' AND indexname='ux_bp_playbook_proposal_finding'"
# EXPECT: the index is back (the migration is idempotent and re-creates it).
psql_t "DELETE FROM proc.bp_playbook WHERE playbook_name = '__guard_proof'"
```

- [ ] **Step 5: Break the database constraint on purpose**

```sh
psql_t "INSERT INTO proc.bp_playbook
          (playbook_name, trigger_source, agent_workflow_id, playbook_status, authored_by)
        VALUES ('__ck_proof', 'detection_finding', 958, 'active', 'proof')"
# EXPECT (THE GUARD GOING RED):
#   ERROR:  new row for relation "bp_playbook" violates check constraint
#           "ck_bp_playbook_active_is_approved"
```

- [ ] **Step 6: The acceptance proof — one real playbook, end to end, on the local server**

The endpoints must be exercised against the running server, not a `TestClient`.
Author a strategy, get it approved by someone else, sweep, and inspect by hand
the proposal it raises. Do **not** approve the proposal unless the workflow it
points at is safe to run — check what graph 958 does first.

```sh
# The server, if it is not already up. Never pkill -f uvicorn: other sessions
# share this box.
systemctl --user status procwise 2>/dev/null | head -3 || \
  curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
```

Then, with a real Admin token in `$TOKEN`:

```sh
API=http://127.0.0.1:8000

# 1. Author it (as the author's token).
curl -s -X POST $API/playbooks -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{
    "playbook_name": "Recover the duplicate invoice",
    "description": "When a duplicate invoice is found, run the recovery graph.",
    "trigger_source": "detection_finding",
    "trigger_match": {"rule_id": "duplicate", "severity": "critical"},
    "agent_workflow_id": 958
  }' | tee /tmp/pb.json

# 2. Submit it.
curl -s -X POST $API/playbooks/<id>/submit -H "Authorization: Bearer $TOKEN"

# 3. Approve it AS SOMEBODY ELSE. First confirm the bar bites: approving with
#    the author's own token must come back 403.
curl -s -o /dev/null -w 'self-approval: %{http_code}\n' \
  -X POST $API/playbooks/<id>/approve -H "Authorization: Bearer $TOKEN"
curl -s -X POST $API/playbooks/<id>/approve -H "Authorization: Bearer $OTHER_TOKEN"

# 4. Sweep, and read the proposals.
curl -s "$API/playbooks/proposals?status=proposed&limit=5" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Record, in the commit message:
- the 403 on self-approval and the 200 on approval by another subject;
- the sweep's rendered counts;
- **one proposal read by hand**: its `evidence` must name the keys that fired
  and the finding's own values, and you must be able to open that finding in
  `proc.bp_detection_finding` and see those same values. A proposal that is not
  re-derivable from source is not evidence.
- a confirmation that `run_id IS NULL` on every proposal — nothing ran.

```sh
psql_t "SELECT proposal_status, count(*), count(run_id) AS with_a_run
          FROM proc.bp_playbook_proposal GROUP BY 1"
# EXPECT: every row proposed, with_a_run = 0. If anything has a run_id, a
# workflow started without a person and that is the one thing this layer
# promised not to do. Stop and find out why.
```

- [ ] **Step 7: Full suite, against the recorded baseline**

Baseline is **635 failures at pristine HEAD**. Run once, not concurrently with
any other suite on this box.

```sh
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 \
       OLLAMA_HOST=http://127.0.0.1:9 OLLAMA_CLOUD_BASE_URL=http://127.0.0.1:9 \
       OLLAMA_CLOUD_API_KEY=
./venv/bin/python -m pytest tests/ -q -p no:randomly \
  --ignore=tests/extraction_v2/test_field_recovery.py \
  --ignore=tests/extraction_v2/test_line_recovery.py \
  --ignore=tests/extraction_v2/test_pdf_table_recovery.py \
  --ignore=tests/extraction_v2/test_recovery_integration.py \
  --ignore=tests/extraction_v2/test_template_service.py \
  --ignore=tests/test_backend_scheduler.py \
  --ignore=tests/test_summary_agent.py 2>&1 | tail -20
```

Expected: at or below 635 failures. Any **new** failure is this work's and must
be fixed or named. Another session's uncommitted files can add failures that are
not yours — check `git status` before attributing one.

- [ ] **Step 8: Commit the proofs**

```bash
git status --short tests/services/playbooks
git commit -o tests/services/playbooks/test_guard_proofs.py \
  -m "test(playbook): every guard broken on purpose and watched go red

Three guards have shipped green in this codebase while checking nothing.
These were inverted before they were believed.

In process: remove the tie and the same finding IS selected, restore it and
nothing is; change only the approver and the same approval succeeds; a
match key one letter wrong is refused while the correct one passes; a
retired playbook disappears from a store that loads the active one; a NULL
doc_type stops matching a playbook that a set doc_type matches.

Live, against bp_testdb: the unique index dropped, and the proposal count
doubled on the next sweep -- so idempotency is the index and not an
accident. The index restored by re-running the migration. An active
playbook with a null approved_by rejected by
ck_bp_playbook_active_is_approved.

Acceptance on the local server: <counts>. Self-approval 403, approval by
another subject 200, one proposal read by hand and its evidence matched
back to the finding row it names. Every proposal has run_id NULL -- nothing
ran.

Full suite: <n> failing against 635 at pristine HEAD.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Push**

```bash
git log --oneline origin/Development..HEAD
git push
```

Expected: the two pre-existing unpushed commits (`c5341dc` playbook spec,
`15a7794` langextract) go up alongside this work. Confirm with Nick before
pushing if the spec commit's review is still open.

---

## Done when

- Both tables exist in `bp_testdb` and `bp_sqldb`, with all four indexes.
- `playbook.write` and `playbook.approve` are known actions with one authority row each, in both databases.
- An expert can author a strategy, someone else can approve it, and the sweep raises a proposal naming the evidence for the match.
- No proposal has ever appeared twice for the same finding.
- No workflow has ever started because the system chose to — every executed proposal names the person who accepted it.
