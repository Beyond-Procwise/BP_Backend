# Analysis Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an analysis a first-class, date-stamped, versioned record that lives in the Analyse area and on every deal it produced — instead of being indistinguishable from a draft deal.

**Architecture:** Three new `proc.bp_analysis*` tables. A new `src/services/analysis_store.py` owns all writes; the existing `SessionNotifyListener` calls it to freeze an analysis at the exact moment an upload session resolves, and the existing backend scheduler calls it to sweep up anything the listener missed. A new `src/api/routers/analysis.py` serves five read endpoints that `spendiq-ui` calls directly on BP_Backend (`VITE_AI_API_URL`), so the Node gateway needs no change.

**Tech Stack:** Python 3.12 / FastAPI / psycopg2 / PostgreSQL (`bp_sqldb`, schema `proc`); pytest. UI is React + a classic `engine.js` script, tested with vitest.

**Spec:** `docs/superpowers/specs/2026-08-01-analysis-events-design.md`

## Global Constraints

- **No fabrication.** Anything not derivable stays `NULL`. A failed fetch renders as "couldn't load", never as "none" or `0`. Historical findings that were never captured stay `NULL` and say so.
- **Table naming:** all new tables `bp_` prefixed in schema `proc`; indexes `ix_bp_<table>_<cols>`.
- **Migrations** live in `deploy/sql/YYYY-MM-DD_<name>.sql`, must be idempotent (`IF NOT EXISTS`, `CREATE OR REPLACE`), and are run against `bp_sqldb`.
- **Never break the broadcast.** Anything added inside `SessionNotifyListener` must fail open — an exception must be logged and swallowed so the client is still told the session resolved. This contract is documented at `src/services/session_notify_listener.py:187-192`.
- **Running the test suite:** `./venv/bin/python -m pytest` with `.env` loaded. Roughly 250 pre-existing collection errors in `tests/extraction_v3` are unrelated to this work — scope every run to the paths named in each task.
- **Never `git stash` to baseline.** Another session shares this checkout. Use a detached `git worktree` if a baseline is needed.
- **Commits:** no `Co-Authored-By: Claude` lines. Work stays on the `Development` branch.
- **UI repo** is `/home/muthu/PycharmProjects/beyond_procwise_ui`; its branch for this work is `spendiq-ui`.
- **Live reality (verified 2026-08-01):** 5,043 deals exist; only **3** ever came through an upload session. 5,040 deals will legitimately have no analysis history. No existing row exercises the analysis↔deal many-to-many.

---

## File Structure

**BP_Backend — create**

| File | Responsibility |
|---|---|
| `deploy/sql/2026-08-01_bp_analysis.sql` | DDL for the three tables |
| `src/services/analysis_store.py` | All writes: `start`, `freeze`, `sweep`. The only module that inserts into `bp_analysis*` |
| `src/services/analysis_findings.py` | Read-only: builds the frozen `findings` blob and its headline figures from `proc.*` |
| `src/api/routers/analysis.py` | Five HTTP routes. No business logic — delegates to the two services above |
| `scripts/backfill_analysis_events.py` | One-off, idempotent, re-runnable backfill |
| `tests/services/test_analysis_store.py` | Unit tests for the store |
| `tests/services/test_analysis_findings.py` | Unit tests for the findings builder |
| `tests/api/test_analysis_endpoints.py` | Route-level tests |

**BP_Backend — modify**

| File | Change |
|---|---|
| `src/services/session_notify_listener.py:186` | Call `analysis_store.freeze()` between linking and broadcasting |
| `src/services/backend_scheduler.py:383` | Register the sweep job |
| `src/api/main.py:425` | Mount the analysis router |

**spendiq-ui — create**

| File | Responsibility |
|---|---|
| `src/modules/AnalyseUpload/startAnalysis.js` | `POST /analysis` after upload. Never throws |
| `src/modules/AnalyseUpload/startAnalysis.test.js` | Tests for it |
| `src/lib/analysisVersions.js` | Pure version-delta helper, bridged to `engine.js` on `window` |
| `src/lib/analysisVersions.test.js` | Tests for it |
| `src/modules/SpendIQ/analysisEvents.contract.test.js` | Asserts `engine.js` behaviour from its source |

**spendiq-ui — modify**

| File | Change |
|---|---|
| `src/modules/AnalyseUpload/nextRoute.js` | Route every mode to the analysis |
| `src/modules/AnalyseUpload/nextRoute.test.js` | Update expectations |
| `src/modules/AnalyseUpload/index.jsx:343-362` | Call `startAnalysis` before navigating |
| `src/modules/SpendIQ/index.jsx:88` | Expose `window.__SIQ_ANALYSIS__` |
| `src/modules/SpendIQ/engine.js` | List, open, freeze-aware report, deal history |

---

## Task 1: Database tables

**Files:**
- Create: `deploy/sql/2026-08-01_bp_analysis.sql`
- Test: `tests/sql/test_bp_analysis_ddl.py`

**Interfaces:**
- Consumes: nothing.
- Produces: tables `proc.bp_analysis` (PK `analysis_id UUID`, `session_id TEXT UNIQUE`), `proc.bp_analysis_document` (PK `(analysis_id, file_path)`), `proc.bp_analysis_deal` (PK `(analysis_id, deal_id)`, `UNIQUE (deal_id, version)`).

- [ ] **Step 1: Write the migration**

Create `deploy/sql/2026-08-01_bp_analysis.sql`:

```sql
-- 2026-08-01  Analysis events
-- ---------------------------------------------------------------------------
-- An analysis becomes a first-class record instead of being indistinguishable
-- from a draft deal (proc.bp_deal.is_tracked = false). One row per analysis
-- RUN: a name, a date, the documents it read, what it found, and the deals it
-- produced.
--
-- version lives on bp_analysis_deal, NOT on bp_analysis: one run can touch
-- several deals, and each of those deals is at a different point in its own
-- history, so the same run may be v3 for one deal and v1 for another.
--
-- Idempotent. Run against: bp_sqldb.
-- See docs/superpowers/specs/2026-08-01-analysis-events-design.md
-- ---------------------------------------------------------------------------

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_analysis (
    analysis_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    mode            TEXT NOT NULL DEFAULT 'new'
                    CHECK (mode IN ('new', 'amend', 'bulk')),
    -- UNIQUE so creating an event is idempotent: one upload can never produce
    -- two events, whether it is created by the UI or by the sweep.
    session_id      TEXT UNIQUE,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'complete', 'failed')),
    failure_reason  TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    created_by      TEXT,
    document_count  INTEGER,
    value_found     NUMERIC(18,2),
    currency        VARCHAR(8),
    findings        JSONB
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_started
    ON proc.bp_analysis (started_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_analysis_status
    ON proc.bp_analysis (status) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS proc.bp_analysis_document (
    analysis_id   UUID NOT NULL
                  REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    doc_type      TEXT,
    doc_pk        TEXT,
    file_path     TEXT NOT NULL,
    file_name     TEXT,
    outcome       TEXT CHECK (outcome IN ('target', 'discrepancy', 'failed')),
    PRIMARY KEY (analysis_id, file_path)
);

CREATE TABLE IF NOT EXISTS proc.bp_analysis_deal (
    analysis_id  UUID NOT NULL
                 REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    deal_id      VARCHAR(25) NOT NULL,
    version      INTEGER NOT NULL,
    is_latest    BOOLEAN NOT NULL DEFAULT true,
    linked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (analysis_id, deal_id),
    UNIQUE (deal_id, version)
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_deal_deal
    ON proc.bp_analysis_deal (deal_id, version DESC);

COMMENT ON TABLE proc.bp_analysis IS
    'One row per analysis RUN. Frozen at session resolution: findings is a '
    'point-in-time snapshot and is never updated afterwards.';
COMMENT ON COLUMN proc.bp_analysis_deal.version IS
    'The nth analysis for THIS deal. Allocated per deal_id, not globally.';

COMMIT;
```

- [ ] **Step 2: Write the failing test**

Create `tests/sql/test_bp_analysis_ddl.py`:

```python
"""The analysis-event DDL says what the spec says it says.

Asserted against the migration text rather than a live database so the test
runs anywhere. Applying it is verified in Task 13 against bp_sqldb.
"""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-08-01_bp_analysis.sql").read_text()


def test_three_tables_created_idempotently():
    for table in ("bp_analysis", "bp_analysis_document", "bp_analysis_deal"):
        assert f"CREATE TABLE IF NOT EXISTS proc.{table}" in SQL


def test_session_id_is_unique_so_start_is_idempotent():
    assert "session_id      TEXT UNIQUE" in SQL


def test_version_is_unique_per_deal_not_globally():
    assert "UNIQUE (deal_id, version)" in SQL


def test_status_and_mode_are_constrained():
    assert "CHECK (status IN ('running', 'complete', 'failed'))" in SQL
    assert "CHECK (mode IN ('new', 'amend', 'bulk'))" in SQL


def test_child_rows_cascade_so_an_analysis_deletes_cleanly():
    assert SQL.count("ON DELETE CASCADE") == 2
```

- [ ] **Step 3: Run the test**

```bash
./venv/bin/python -m pytest tests/sql/test_bp_analysis_ddl.py -v
```
Expected: PASS (the migration was written in Step 1).

- [ ] **Step 4: Apply the migration to bp_sqldb**

```bash
./venv/bin/python -c "
import os
from dotenv import load_dotenv; load_dotenv()
import psycopg2
c = psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT', 5432),
                     dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
                     password=os.getenv('DB_PASSWORD'))
c.cursor().execute(open('deploy/sql/2026-08-01_bp_analysis.sql').read())
c.commit()
print('applied')
"
```
Expected: `applied`. Run it a second time — it must print `applied` again, proving idempotency.

- [ ] **Step 5: Commit**

```bash
git add deploy/sql/2026-08-01_bp_analysis.sql tests/sql/test_bp_analysis_ddl.py
git commit -m "feat(analysis): tables for first-class analysis events

version lives on bp_analysis_deal, not bp_analysis: one run can touch
several deals, each at a different point in its own history."
```

---

## Task 2: `analysis_store.start()`

**Files:**
- Create: `src/services/analysis_store.py`
- Test: `tests/services/test_analysis_store.py`

**Interfaces:**
- Consumes: Task 1's tables. `from src.services.db import get_conn`.
- Produces: `start(*, session_id: str, name: str | None = None, mode: str = "new", created_by: str | None = None, conn=None) -> str` returning the `analysis_id` as a string. Idempotent on `session_id`: a second call returns the **same** id and does not overwrite `name` or `mode`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_analysis_store.py`:

```python
"""analysis_store writes. Every test drives a fake psycopg2 connection so the
suite never needs a database."""
import uuid

import pytest

from src.services import analysis_store


class FakeCursor:
    """Records every statement and replays queued results in order."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._results.pop(0) if self._results else None

    def fetchall(self):
        return self._results.pop(0) if self._results else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, results=()):
        self.cur = FakeCursor(results)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_start_returns_the_new_analysis_id():
    new_id = uuid.uuid4()
    conn = FakeConn(results=[(new_id,)])

    got = analysis_store.start(session_id="ses-1", name="Q3 renewal",
                               mode="new", conn=conn)

    assert got == str(new_id)


def test_start_is_idempotent_on_session_id():
    """A second call for the same upload must return the SAME id and must not
    overwrite the name the user chose."""
    existing = uuid.uuid4()
    conn = FakeConn(results=[(existing,)])

    got = analysis_store.start(session_id="ses-1", name="", conn=conn)

    sql, _ = conn.cur.calls[0]
    assert "ON CONFLICT (session_id) DO UPDATE" in sql
    assert "name = COALESCE(proc.bp_analysis.name, EXCLUDED.name)" in sql
    assert got == str(existing)


def test_start_rejects_a_blank_session_id():
    with pytest.raises(ValueError, match="session_id is required"):
        analysis_store.start(session_id="  ", conn=FakeConn())


def test_start_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        analysis_store.start(session_id="ses-1", mode="sideways",
                             conn=FakeConn())
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.analysis_store'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/services/analysis_store.py`:

```python
"""Writes for analysis events (proc.bp_analysis + its two child tables).

This is the ONLY module that inserts into bp_analysis*. Three entry points:

    start()   an upload begins           -> status 'running'
    freeze()  its session resolves       -> status 'complete', findings captured
    sweep()   safety net on a timer      -> creates missed events, freezes stuck
                                            ones, fails ones that never resolved

See docs/superpowers/specs/2026-08-01-analysis-events-design.md
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

MODES = ("new", "amend", "bulk")


@contextmanager
def _txn(conn: Optional[Any]) -> Iterator[Any]:
    """Run inside the caller's connection, or own one and commit/rollback.

    Matches the pattern in src/services/deal_lifecycle.py: a caller that is
    already inside a transaction keeps control of it.
    """
    if conn is not None:
        yield conn
        return
    with get_conn() as own:
        own.autocommit = False
        try:
            yield own
            own.commit()
        except Exception:
            own.rollback()
            raise


def start(*, session_id: str, name: Optional[str] = None, mode: str = "new",
          created_by: Optional[str] = None, conn: Optional[Any] = None) -> str:
    """Create (or return) the analysis event for an upload session.

    Idempotent on session_id. A second call NEVER overwrites a name that is
    already set — the UI's call carries the user's chosen name, the sweep's
    fallback call does not, and whichever lands second must not win.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    with _txn(conn) as c:
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO proc.bp_analysis (session_id, name, mode, created_by)
            VALUES (%s, NULLIF(%s, ''), %s, %s)
            ON CONFLICT (session_id) DO UPDATE
               SET name = COALESCE(proc.bp_analysis.name, EXCLUDED.name)
            RETURNING analysis_id
            """,
            (sid, (name or "").strip(), mode, created_by),
        )
        return str(cur.fetchone()[0])
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/analysis_store.py tests/services/test_analysis_store.py
git commit -m "feat(analysis): analysis_store.start(), idempotent per upload session

A second call never overwrites a name that is already set — the UI call
carries the user's chosen name, the sweep's fallback does not."
```

---

## Task 3: `analysis_store.freeze()` — documents, deal links, versions

**Files:**
- Modify: `src/services/analysis_store.py`
- Modify: `tests/services/test_analysis_store.py`

**Interfaces:**
- Consumes: `start()` from Task 2.
- Produces: `freeze(session_id: str, *, findings: dict | None = None, document_count: int | None = None, value_found=None, currency: str | None = None, conn=None) -> str | None` returning the frozen `analysis_id`, or `None` if there was nothing in `running` for that session. Also `_allocate_version(cur, deal_id) -> int`.
- Note: `findings` is passed **in** rather than fetched here. Task 4 builds it; this task only stores it. That keeps the store free of query logic and lets both be tested alone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/test_analysis_store.py`:

```python
def test_freeze_copies_session_documents_onto_the_analysis():
    aid = uuid.uuid4()
    conn = FakeConn(results=[
        (aid,),          # SELECT the running analysis
        [],              # INSERT ... documents (no result)
        [("D-1",)],      # SELECT DISTINCT deal_id
        (2,),            # _allocate_version -> next version for D-1
    ])

    analysis_store.freeze("ses-1", findings={"deal": {}}, conn=conn)

    sqls = [s for s, _ in conn.cur.calls]
    assert any("INSERT INTO proc.bp_analysis_document" in s for s in sqls)
    assert any("proc.session_document_outcome" in s for s in sqls)


def test_freeze_allocates_version_per_deal_not_globally():
    """The same run may be v3 for one deal and v1 for another."""
    aid = uuid.uuid4()
    conn = FakeConn(results=[
        (aid,),
        [],
        [("D-1",), ("D-2",)],
        (3,),            # D-1 already had two analyses
        (1,),            # D-2 has none
    ])

    analysis_store.freeze("ses-1", findings={}, conn=conn)

    link_params = [p for s, p in conn.cur.calls
                   if "INSERT INTO proc.bp_analysis_deal" in s]
    assert [(p[1], p[2]) for p in link_params] == [("D-1", 3), ("D-2", 1)]


def test_freeze_clears_is_latest_on_the_deals_previous_versions():
    aid = uuid.uuid4()
    conn = FakeConn(results=[(aid,), [], [("D-1",)], (2,)])

    analysis_store.freeze("ses-1", findings={}, conn=conn)

    sqls = [s for s, _ in conn.cur.calls]
    assert any("SET is_latest = false" in s and "deal_id = %s" in s
               for s in sqls)


def test_freeze_is_a_noop_when_nothing_is_running():
    """Idempotent: freezing an already-complete analysis must not touch it."""
    conn = FakeConn(results=[None])

    assert analysis_store.freeze("ses-1", findings={}, conn=conn) is None
    assert not any("INSERT INTO proc.bp_analysis_deal" in s
                   for s, _ in conn.cur.calls)


def test_freeze_marks_the_analysis_complete_with_its_headline_figures():
    aid = uuid.uuid4()
    conn = FakeConn(results=[(aid,), [], [], ])

    analysis_store.freeze("ses-1", findings={"a": 1}, document_count=4,
                          value_found=12400, currency="GBP", conn=conn)

    final = [(s, p) for s, p in conn.cur.calls
             if "UPDATE proc.bp_analysis" in s and "status = 'complete'" in s]
    assert len(final) == 1
    assert 4 in final[0][1] and "GBP" in final[0][1]
```

- [ ] **Step 2: Run them to make sure they fail**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py -v -k freeze
```
Expected: FAIL — `AttributeError: module ... has no attribute 'freeze'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/services/analysis_store.py`:

```python
import json


def _allocate_version(cur: Any, deal_id: str) -> int:
    """The next version number for THIS deal.

    Per deal_id, never global — one analysis run can touch several deals and
    each of them is at a different point in its own history.
    """
    cur.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM proc.bp_analysis_deal "
        "WHERE deal_id = %s",
        (str(deal_id),),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 1


def freeze(session_id: str, *, findings: Optional[dict] = None,
           document_count: Optional[int] = None, value_found: Any = None,
           currency: Optional[str] = None,
           conn: Optional[Any] = None) -> Optional[str]:
    """Freeze the running analysis for a resolved upload session.

    Returns the analysis_id, or None when there is nothing in 'running' for
    this session — which is the normal outcome of a second call, and is what
    makes this safe for both the listener and the sweep to invoke.
    """
    sid = (session_id or "").strip()
    if not sid:
        return None

    with _txn(conn) as c:
        cur = c.cursor()

        cur.execute(
            "SELECT analysis_id FROM proc.bp_analysis "
            "WHERE session_id = %s AND status = 'running' FOR UPDATE",
            (sid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        analysis_id = row[0]

        # 1. What this analysis read. Copied from the trigger-written outcome
        #    table, which is the authoritative record of the session.
        cur.execute(
            """
            INSERT INTO proc.bp_analysis_document
                   (analysis_id, doc_type, file_path, file_name, outcome)
            SELECT %s, sdo.document_type, sdo.file_path,
                   regexp_replace(sdo.file_path, '^.*/', ''), sdo.outcome
              FROM proc.session_document_outcome sdo
             WHERE sdo.session_id = %s
            ON CONFLICT (analysis_id, file_path) DO NOTHING
            """,
            (analysis_id, sid),
        )

        # 2. Which deals it produced. process_monitor already carries deal_id.
        cur.execute(
            "SELECT DISTINCT deal_id FROM proc.process_monitor "
            "WHERE session_id = %s AND deal_id IS NOT NULL ORDER BY deal_id",
            (sid,),
        )
        for (deal_id,) in (cur.fetchall() or []):
            cur.execute(
                "UPDATE proc.bp_analysis_deal SET is_latest = false "
                "WHERE deal_id = %s",
                (str(deal_id),),
            )
            cur.execute(
                "INSERT INTO proc.bp_analysis_deal "
                "       (analysis_id, deal_id, version, is_latest) "
                "VALUES (%s, %s, %s, true) "
                "ON CONFLICT (analysis_id, deal_id) DO NOTHING",
                (analysis_id, str(deal_id), _allocate_version(cur, deal_id)),
            )

        # 3. Freeze.
        cur.execute(
            """
            UPDATE proc.bp_analysis
               SET status = 'complete', completed_at = now(),
                   findings = %s, document_count = %s,
                   value_found = %s, currency = %s
             WHERE analysis_id = %s
            """,
            (json.dumps(findings) if findings is not None else None,
             document_count, value_found, currency, analysis_id),
        )
        return str(analysis_id)
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/analysis_store.py tests/services/test_analysis_store.py
git commit -m "feat(analysis): freeze() captures documents, deal links and versions

Version is allocated per deal_id, so one run can be v3 on one deal and v1
on another. A second freeze is a no-op, which is what lets both the
listener and the scheduled sweep call it."
```

---

## Task 4: The frozen findings blob

**Files:**
- Create: `src/services/analysis_findings.py`
- Test: `tests/services/test_analysis_findings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `capture(deal_ids: list[str], *, conn=None) -> dict` and `headline(findings: dict) -> tuple[float | None, str | None]` returning `(value_found, currency)`.
- The returned dict has exactly these keys: `deals`, `discrepancies`, `opportunities`, `summaries`, `benchmarks`, `captured_at`, `sources`. `sources` maps each of the five names to `"ok"` or `"error"` — a source that failed must be recorded as `"error"`, never silently emptied.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_analysis_findings.py`:

```python
"""The frozen findings blob. Fail-closed is the whole point: a source that
errored must be recorded as 'error', never as an empty list, because an empty
list reads as 'we looked and found nothing'."""
from src.services import analysis_findings


class FakeCursor:
    def __init__(self, script):
        self.script = script
        self.last = ""

    def execute(self, sql, params=None):
        self.last = " ".join(sql.split())
        for marker, behaviour in self.script.items():
            if marker in self.last:
                if isinstance(behaviour, Exception):
                    raise behaviour
                self._rows = behaviour
                return
        self._rows = []

    @property
    def description(self):
        return [type("C", (), {"name": n})
                for n in (self._rows[0].keys() if self._rows else [])]

    def fetchall(self):
        return [tuple(r.values()) for r in self._rows]


class FakeConn:
    def __init__(self, script):
        self._cur = FakeCursor(script)

    def cursor(self):
        return self._cur


def test_capture_records_every_source_as_ok_when_all_succeed():
    conn = FakeConn({
        "bp_deal_overview": [{"deal_id": "D-1", "currency": "GBP"}],
        "bp_extraction_discrepancy": [],
        "bp_opportunity": [],
        "bp_analysis_summary": [],
    })
    got = analysis_findings.capture(["D-1"], conn=conn)

    assert set(got["sources"]) == {"deals", "discrepancies", "opportunities",
                                   "summaries", "benchmarks"}
    assert got["sources"]["deals"] == "ok"
    assert got["deals"][0]["deal_id"] == "D-1"


def test_a_failing_source_is_recorded_as_error_not_as_empty():
    conn = FakeConn({
        "bp_deal_overview": [{"deal_id": "D-1"}],
        "bp_opportunity": RuntimeError("relation does not exist"),
    })
    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["opportunities"] == "error"
    assert got["opportunities"] == []


def test_capture_with_no_deals_still_returns_a_usable_blob():
    """A one-off analysis that produced no deal still has a findings record."""
    got = analysis_findings.capture([], conn=FakeConn({}))

    assert got["deals"] == []
    assert got["sources"]["deals"] == "ok"


def test_headline_sums_opportunity_impact_and_recovered_amounts():
    findings = {
        "opportunities": [{"financial_impact_gbp": 10000},
                          {"financial_impact_gbp": 2400}],
        "discrepancies": [{"recovered_amount": 500}],
        "deals": [{"currency": "GBP"}],
    }
    assert analysis_findings.headline(findings) == (12900.0, "GBP")


def test_headline_is_none_when_there_is_nothing_to_add_up():
    """Never 0 — zero means 'we found nothing', None means 'no figure'."""
    findings = {"opportunities": [], "discrepancies": [], "deals": []}
    assert analysis_findings.headline(findings) == (None, None)
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_findings.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.analysis_findings'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/services/analysis_findings.py`:

```python
"""Builds the point-in-time findings snapshot stored on proc.bp_analysis.

Five sources, all read straight out of proc.* — no HTTP call out to the Node
gateway, which the listener thread cannot rely on reaching.

The portfolio compliance measures are deliberately NOT captured: they are
workspace-wide percentages that change for reasons unrelated to this upload.
See §5 of the spec.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

SOURCES = ("deals", "discrepancies", "opportunities", "summaries", "benchmarks")

_DEALS = """
SELECT deal_id, deal_name, supplier_name, supplier_id, deal_date, currency,
       quote_count, po_count, invoice_count,
       quote_total, po_total, invoice_total,
       three_way_match, price_variance_pct, cycle_days_quote_to_po
  FROM proc.bp_deal_overview WHERE deal_id = ANY(%s)
"""

_DISCREPANCIES = """
SELECT id, deal_id, document_type, doc_pk, field_name, severity, description,
       blocks_promotion, status, resolution_outcome, recovered_amount
  FROM proc.bp_extraction_discrepancy WHERE deal_id = ANY(%s)
"""

_OPPORTUNITIES = """
SELECT opportunity_id, deal_id, title, opportunity_type, stage,
       financial_impact_gbp, confidence, detected_on
  FROM proc.bp_opportunity WHERE deal_id = ANY(%s)
"""

_SUMMARIES = """
SELECT deal_id, summary, deal_value, currency, volume, unit_price,
       price_change_pct, volume_change_pct, efficiency_score, item_count
  FROM proc.bp_analysis_summary WHERE deal_id = ANY(%s) AND is_current
"""


def _rows(cur: Any, sql: str, deal_ids: list) -> list[dict]:
    cur.execute(sql, (deal_ids,))
    cols = [c.name for c in (cur.description or [])]
    return [dict(zip(cols, r)) for r in (cur.fetchall() or [])]


def capture(deal_ids: list, *, conn: Optional[Any] = None) -> dict:
    """Snapshot everything this analysis found, per source.

    Each source is fetched independently so one failure cannot empty the
    others, and its status is recorded — an errored source reads as 'error',
    never as an empty result.
    """
    ids = [str(d) for d in (deal_ids or [])]
    out: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "sources": {name: "ok" for name in SOURCES},
    }
    for name in SOURCES:
        out[name] = []

    def _run(c: Any) -> None:
        cur = c.cursor()
        for name, sql in (("deals", _DEALS),
                          ("discrepancies", _DISCREPANCIES),
                          ("opportunities", _OPPORTUNITIES),
                          ("summaries", _SUMMARIES)):
            if not ids:
                continue
            try:
                out[name] = _rows(cur, sql, ids)
            except Exception:
                log.exception("findings capture failed for source=%s", name)
                out["sources"][name] = "error"

        for deal_id in ids:
            try:
                from src.api.routers.benchmark import benchmark_by_deal
                out["benchmarks"].append(
                    {"deal_id": deal_id, "benchmark": benchmark_by_deal(deal_id)})
            except Exception:
                log.exception("benchmark capture failed for deal=%s", deal_id)
                out["sources"]["benchmarks"] = "error"

    if conn is not None:
        _run(conn)
    else:
        with get_conn() as own:
            _run(own)
    return out


def headline(findings: dict) -> tuple:
    """(value_found, currency) for the list view and the version delta.

    None rather than 0 when there is nothing to add up: zero means 'we looked
    and found nothing', None means 'there is no figure'. They are different
    facts and the list view renders them differently.
    """
    total = 0.0
    seen = False
    for opp in (findings.get("opportunities") or []):
        v = opp.get("financial_impact_gbp")
        if v is not None:
            total += float(v)
            seen = True
    for disc in (findings.get("discrepancies") or []):
        v = disc.get("recovered_amount")
        if v is not None:
            total += float(v)
            seen = True
    if not seen:
        return (None, None)
    currency = next((d.get("currency") for d in (findings.get("deals") or [])
                     if d.get("currency")), "GBP")
    return (round(total, 2), currency)
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_findings.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Verify the column names exist against bp_sqldb**

The four queries above name columns that must actually exist. Check before trusting them:

```bash
./venv/bin/python -c "
import os
from dotenv import load_dotenv; load_dotenv()
import psycopg2
c=psycopg2.connect(host=os.getenv('DB_HOST'),port=os.getenv('DB_PORT',5432),dbname=os.getenv('DB_NAME'),user=os.getenv('DB_USER'),password=os.getenv('DB_PASSWORD'))
cur=c.cursor()
for t in ('bp_deal_overview','bp_extraction_discrepancy','bp_opportunity','bp_analysis_summary'):
    cur.execute(\"select column_name from information_schema.columns where table_schema='proc' and table_name=%s\",(t,))
    print(t, sorted(r[0] for r in cur.fetchall()))
"
```
Expected: every column referenced in `_DEALS`, `_DISCREPANCIES`, `_OPPORTUNITIES` and `_SUMMARIES` appears. **If one does not, fix the query — do not leave a column that will fail at runtime.** In particular confirm `bp_extraction_discrepancy` has a `deal_id`; if it does not, scope discrepancies through the deal's documents instead and note the change in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/services/analysis_findings.py tests/services/test_analysis_findings.py
git commit -m "feat(analysis): capture the frozen findings snapshot

Five sources read straight from proc.*, each fetched independently so one
failure cannot empty the others. A failed source is recorded as 'error',
never as an empty list — those are different facts."
```

---

## Task 5: Freeze on session resolution

**Files:**
- Modify: `src/services/session_notify_listener.py:186-201`
- Test: `tests/services/test_analysis_freeze_on_session.py`

**Interfaces:**
- Consumes: `analysis_store.freeze()` (Task 3), `analysis_findings.capture()` / `headline()` (Task 4).
- Produces: `SessionNotifyListener._freeze_analysis(session_id: str) -> None`, called from `_link_then_broadcast` between `_link_session` and `_broadcast`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_analysis_freeze_on_session.py`:

```python
"""The freeze runs between linking and broadcasting, and never blocks the
broadcast. A spinner that never resolves is a worse failure than a report
with no findings."""
import importlib

mod = importlib.import_module("src.services.session_notify_listener")


def _listener():
    return mod.SessionNotifyListener.__new__(mod.SessionNotifyListener)


def test_freeze_runs_after_linking_and_before_broadcasting(monkeypatch):
    order = []
    lis = _listener()
    monkeypatch.setattr(lis, "_link_session", lambda s: order.append("link"),
                        raising=False)
    monkeypatch.setattr(lis, "_freeze_analysis", lambda s: order.append("freeze"),
                        raising=False)
    monkeypatch.setattr(lis, "_broadcast", lambda s, p: order.append("cast"),
                        raising=False)

    lis._link_then_broadcast("ses-1", {})

    assert order == ["link", "freeze", "cast"]


def test_a_failing_freeze_still_broadcasts(monkeypatch):
    order = []
    lis = _listener()
    monkeypatch.setattr(lis, "_link_session", lambda s: None, raising=False)

    def boom(_):
        raise RuntimeError("db gone")

    monkeypatch.setattr(lis, "_freeze_analysis", boom, raising=False)
    monkeypatch.setattr(lis, "_broadcast", lambda s, p: order.append("cast"),
                        raising=False)

    lis._link_then_broadcast("ses-1", {})

    assert order == ["cast"]


def test_freeze_passes_the_captured_findings_to_the_store(monkeypatch):
    captured = {}
    store = importlib.import_module("src.services.analysis_store")
    findings = importlib.import_module("src.services.analysis_findings")

    monkeypatch.setattr(mod, "_deal_ids_for_session", lambda s: ["D-1"],
                        raising=False)
    monkeypatch.setattr(mod, "_document_count_for_session", lambda s: 4,
                        raising=False)
    monkeypatch.setattr(findings, "capture",
                        lambda ids, **k: {"deals": [{"currency": "GBP"}],
                                          "opportunities": [
                                              {"financial_impact_gbp": 900}],
                                          "discrepancies": []})
    monkeypatch.setattr(store, "freeze",
                        lambda sid, **kw: captured.update(kw) or "aid")

    _listener()._freeze_analysis("ses-1")

    assert captured["value_found"] == 900.0
    assert captured["currency"] == "GBP"
    assert captured["document_count"] == 4
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_freeze_on_session.py -v
```
Expected: FAIL — `_freeze_analysis` does not exist and `_link_then_broadcast` does not call it.

- [ ] **Step 3: Write the implementation**

In `src/services/session_notify_listener.py`, change `_link_then_broadcast` (currently at line 186) so the freeze sits between the two existing calls:

```python
    def _link_then_broadcast(self, session_id: str, payload: dict) -> None:
        """Link, freeze, then broadcast — and broadcast whatever happens.

        Fail-open is deliberate: a linking or freezing error must never strand
        the page on a spinner. The client is told the session resolved either
        way; a failure shows up as a report with no documents, which is what it
        would have shown before this existed. A freeze that did not happen is
        picked up by the scheduled sweep.
        """
        try:
            self._link_session(session_id)
        except Exception:
            log.exception(
                "Fast deal-linking failed for session %s — broadcasting anyway; "
                "the scheduled sweep will pick it up",
                session_id,
            )
        try:
            self._freeze_analysis(session_id)
        except Exception:
            log.exception(
                "Freezing the analysis for session %s failed — broadcasting "
                "anyway; the scheduled sweep will retry it",
                session_id,
            )
        self._broadcast(session_id, payload)

    def _freeze_analysis(self, session_id: str) -> None:
        """Turn the running analysis for this session into a frozen record."""
        from src.services import analysis_findings, analysis_store  # noqa: PLC0415

        deal_ids = _deal_ids_for_session(session_id)
        findings = analysis_findings.capture(deal_ids)
        value_found, currency = analysis_findings.headline(findings)
        analysis_store.freeze(
            session_id,
            findings=findings,
            document_count=_document_count_for_session(session_id),
            value_found=value_found,
            currency=currency,
        )
```

Add these two module-level helpers next to the other module-level functions in
the same file. Make sure `from typing import Optional` is imported at the top of
the file; add it if it is not.

```python
def _deal_ids_for_session(session_id: str) -> list:
    """Which deals this session's documents landed on.

    process_monitor already carries deal_id, so this is a direct lookup rather
    than a walk through the _trgt tables.
    """
    from src.services.db import get_conn  # noqa: PLC0415

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT deal_id FROM proc.process_monitor "
            "WHERE session_id = %s AND deal_id IS NOT NULL",
            (session_id,),
        )
        return [r[0] for r in (cur.fetchall() or [])]


def _document_count_for_session(session_id: str) -> Optional[int]:
    """How many documents this analysis read. None on failure, never 0 —
    zero would read as 'it read nothing'."""
    from src.services.db import get_conn  # noqa: PLC0415

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM proc.session_document_outcome "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] else None
    except Exception:
        log.exception("document count failed for session=%s", session_id)
        return None
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_freeze_on_session.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Check nothing else in the listener regressed**

```bash
./venv/bin/python -m pytest tests/ -k "session_notify or session_status" -v
```
Expected: no new failures versus before this task.

- [ ] **Step 6: Commit**

```bash
git add src/services/session_notify_listener.py tests/services/test_analysis_freeze_on_session.py
git commit -m "feat(analysis): freeze the analysis when its upload session resolves

Sits between linking and broadcasting, where the deals already exist but the
client has not yet been told to render. Fails open — a freeze that did not
happen is picked up by the sweep; a spinner that never resolves is worse."
```

---

## Task 6: The sweep

**Files:**
- Modify: `src/services/analysis_store.py`
- Modify: `src/services/backend_scheduler.py:383` (`_register_default_jobs`)
- Modify: `tests/services/test_analysis_store.py`

**Interfaces:**
- Consumes: `start()`, `freeze()`.
- Produces: `sweep(*, stale_minutes: int = 60, conn=None) -> dict` returning `{"created": int, "frozen": int, "failed": int}`. Scheduler job name constant `ANALYSIS_SWEEP_JOB_NAME = "analysis-sweep"`, method `_run_analysis_sweep`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/test_analysis_store.py`:

```python
def test_sweep_creates_events_for_sessions_that_never_got_one(monkeypatch):
    """The UI's POST is an optimisation. If the browser closed before it fired,
    the sweep must still produce the event — just without the chosen name."""
    conn = FakeConn(results=[
        [("ses-9", "Renewal deal")],   # sessions with no bp_analysis row
        [],                            # sessions running but resolved
        [],                            # sessions running past the stale cap
    ])
    started = []
    monkeypatch.setattr(analysis_store, "start",
                        lambda **kw: started.append(kw) or "aid")

    got = analysis_store.sweep(conn=conn)

    assert got["created"] == 1
    assert started[0]["session_id"] == "ses-9"
    assert started[0]["name"] == "Renewal deal"


def test_sweep_freezes_a_resolved_but_stuck_analysis(monkeypatch):
    conn = FakeConn(results=[[], [("ses-9",)], []])
    frozen = []
    monkeypatch.setattr(analysis_store, "_freeze_one",
                        lambda sid: frozen.append(sid))

    got = analysis_store.sweep(conn=conn)

    assert got["frozen"] == 1 and frozen == ["ses-9"]


def test_sweep_fails_an_analysis_whose_session_never_resolved():
    conn = FakeConn(results=[[], [], [("ses-9",)]])

    got = analysis_store.sweep(conn=conn)

    assert got["failed"] == 1
    sqls = [s for s, _ in conn.cur.calls]
    assert any("status = 'failed'" in s for s in sqls)
    assert any("session did not resolve" in str(p) for _, p in conn.cur.calls)


def test_sweep_stale_cap_defaults_to_sixty_minutes():
    """The UI gives up narrating after 6 minutes (REPORT_WAIT_CAP_MS). Failing
    an analysis at that point would kill slow-but-working large uploads."""
    import inspect
    sig = inspect.signature(analysis_store.sweep)
    assert sig.parameters["stale_minutes"].default == 60
```

- [ ] **Step 2: Run them to make sure they fail**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py -v -k sweep
```
Expected: FAIL — `sweep` does not exist.

- [ ] **Step 3: Write the implementation**

Append to `src/services/analysis_store.py`:

```python
def _freeze_one(session_id: str) -> None:
    """Freeze one session the same way the listener does, findings and all."""
    from src.services import analysis_findings  # noqa: PLC0415
    from src.services.session_notify_listener import (  # noqa: PLC0415
        _deal_ids_for_session, _document_count_for_session,
    )

    findings = analysis_findings.capture(_deal_ids_for_session(session_id))
    value_found, currency = analysis_findings.headline(findings)
    freeze(session_id, findings=findings,
           document_count=_document_count_for_session(session_id),
           value_found=value_found, currency=currency)


def sweep(*, stale_minutes: int = 60, conn: Optional[Any] = None) -> dict:
    """Safety net for everything the listener path can miss.

    Three passes:
      1. sessions with documents but no analysis event at all (the browser
         closed before the UI's POST fired)
      2. analyses still 'running' whose session HAS resolved (the listener died
         mid-session)
      3. analyses 'running' past the stale cap whose session never resolved

    stale_minutes defaults to 60 — deliberately far beyond the UI's 6-minute
    patience cap, because a large upload legitimately takes minutes and must
    never be declared failed while it is still working.
    """
    result = {"created": 0, "frozen": 0, "failed": 0}
    with _txn(conn) as c:
        cur = c.cursor()

        cur.execute(
            """
            SELECT DISTINCT pm.session_id, MAX(pm.deal_name)
              FROM proc.process_monitor pm
             WHERE pm.session_id IS NOT NULL
               AND EXISTS (SELECT 1 FROM proc.session_document_outcome sdo
                            WHERE sdo.session_id = pm.session_id)
               AND NOT EXISTS (SELECT 1 FROM proc.bp_analysis a
                                WHERE a.session_id = pm.session_id)
             GROUP BY pm.session_id
            """
        )
        for session_id, deal_name in (cur.fetchall() or []):
            try:
                start(session_id=session_id, name=deal_name, conn=c)
                result["created"] += 1
            except Exception:
                log.exception("sweep could not create event for %s", session_id)

        cur.execute(
            """
            SELECT a.session_id FROM proc.bp_analysis a
             WHERE a.status = 'running'
               AND EXISTS (SELECT 1 FROM proc.process_monitor pm
                            WHERE pm.session_id = a.session_id
                              AND pm.action_status IS NOT NULL)
            """
        )
        for (session_id,) in (cur.fetchall() or []):
            try:
                _freeze_one(session_id)
                result["frozen"] += 1
            except Exception:
                log.exception("sweep could not freeze %s", session_id)

        cur.execute(
            """
            UPDATE proc.bp_analysis
               SET status = 'failed', completed_at = now(),
                   failure_reason = 'session did not resolve'
             WHERE status = 'running'
               AND started_at < now() - (%s || ' minutes')::interval
             RETURNING session_id
            """,
            (str(int(stale_minutes)),),
        )
        result["failed"] = len(cur.fetchall() or [])
    return result
```

- [ ] **Step 4: Register the scheduler job**

In `src/services/backend_scheduler.py`, add to `_register_default_jobs` (line 383), after `self._register_deal_assignment_job()`:

```python
        self._register_analysis_sweep_job()
```

And add the method, following the shape of `_register_deal_assignment_job` at line 611:

```python
    ANALYSIS_SWEEP_JOB_NAME = "analysis-sweep"

    def _register_analysis_sweep_job(self) -> None:
        """Safety net for analysis events the listener path missed. Toggle
        ANALYSIS_SWEEP_ENABLED (default on), interval
        ANALYSIS_SWEEP_INTERVAL_MINUTES (default 15)."""
        import os
        if os.environ.get("ANALYSIS_SWEEP_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("analysis sweep job disabled by ANALYSIS_SWEEP_ENABLED")
            return
        if self.ANALYSIS_SWEEP_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("ANALYSIS_SWEEP_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.ANALYSIS_SWEEP_JOB_NAME,
            self._run_analysis_sweep,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _run_analysis_sweep(self) -> None:
        from src.services import analysis_store  # noqa: PLC0415
        result = analysis_store.sweep()
        if any(result.values()):
            logger.info("analysis sweep: %s", result)
```

- [ ] **Step 5: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_analysis_store.py tests/test_backend_scheduler.py -v
```
Expected: all pass, including the pre-existing scheduler tests.

- [ ] **Step 6: Commit**

```bash
git add src/services/analysis_store.py src/services/backend_scheduler.py tests/services/test_analysis_store.py
git commit -m "feat(analysis): sweep for events the listener path missed

Creates missing events, freezes stuck ones, fails ones whose session never
resolved. 60-minute stale cap, well beyond the UI's 6-minute patience, so a
slow large upload is never declared failed while still working."
```

---

## Task 7: Read endpoints

**Files:**
- Create: `src/api/routers/analysis.py`
- Modify: `src/api/main.py:425`
- Test: `tests/api/test_analysis_endpoints.py`

**Interfaces:**
- Consumes: `analysis_store.start()`.
- Produces: routes under prefix `/analysis` — `POST ""`, `GET ""`, `GET /{analysis_id}`, `GET /by-deal/{deal_id}`, `GET /by-session/{session_id}`. Route-ordering matters: `by-deal` and `by-session` must be declared **before** `/{analysis_id}` or the path param captures them (the same trap the gateway documents at `spendiq.controller.ts:80`).

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_analysis_endpoints.py`:

```python
"""Analysis-event routes."""
import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

mod = importlib.import_module("src.api.routers.analysis")


def _client():
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


def test_start_returns_the_analysis_id(monkeypatch):
    monkeypatch.setattr(mod.analysis_store, "start", lambda **kw: "aid-1")
    r = _client().post("/analysis", json={"session_id": "ses-1",
                                          "name": "Q3", "mode": "new"})
    assert r.status_code == 200
    assert r.json() == {"analysis_id": "aid-1"}


def test_start_rejects_a_missing_session_id():
    r = _client().post("/analysis", json={"name": "Q3"})
    assert r.status_code == 422


def test_by_deal_and_by_session_are_declared_before_the_id_route():
    """Otherwise /{analysis_id} swallows them."""
    paths = [r.path for r in mod.router.routes]
    assert paths.index("/analysis/by-deal/{deal_id}") < paths.index("/analysis/{analysis_id}")
    assert paths.index("/analysis/by-session/{session_id}") < paths.index("/analysis/{analysis_id}")


def test_by_deal_returns_versions_newest_first_with_deltas(monkeypatch):
    monkeypatch.setattr(mod, "_query", lambda sql, params: [
        {"analysis_id": "a2", "version": 2, "name": "v2",
         "started_at": "2026-07-14", "document_count": 2, "value_found": 9100,
         "currency": "GBP", "status": "complete"},
        {"analysis_id": "a1", "version": 1, "name": "v1",
         "started_at": "2026-07-02", "document_count": 6, "value_found": 9100,
         "currency": "GBP", "status": "complete"},
    ])
    body = _client().get("/analysis/by-deal/D-1").json()

    assert [v["version"] for v in body["versions"]] == [2, 1]
    assert body["versions"][0]["delta"] == {"document_count": -4,
                                            "value_found": 0.0}
    assert body["versions"][1]["delta"] is None   # nothing to compare v1 to


def test_a_deal_with_no_analyses_returns_an_empty_list_not_an_error(monkeypatch):
    """5,040 of 5,043 deals are in this state — it is the common path."""
    monkeypatch.setattr(mod, "_query", lambda sql, params: [])
    body = _client().get("/analysis/by-deal/D-1").json()
    assert body == {"deal_id": "D-1", "versions": []}


def test_get_by_session_404s_when_there_is_no_such_analysis(monkeypatch):
    monkeypatch.setattr(mod, "_query", lambda sql, params: [])
    assert _client().get("/analysis/by-session/nope").status_code == 404
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
./venv/bin/python -m pytest tests/api/test_analysis_endpoints.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.routers.analysis'`.

- [ ] **Step 3: Write the implementation**

Create `src/api/routers/analysis.py`:

```python
"""Analysis events — the read surface the Analyse area and the deal history use.

Called directly by spendiq-ui against VITE_AI_API_URL; the Node gateway does not
proxy these. See docs/superpowers/specs/2026-08-01-analysis-events-design.md
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.db import get_conn
from src.services import analysis_store

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

_LIST_COLS = """
    a.analysis_id, a.name, a.mode, a.session_id, a.status, a.failure_reason,
    a.started_at, a.completed_at, a.document_count, a.value_found, a.currency
"""


def _query(sql: str, params: tuple) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c.name for c in (cur.description or [])]
        return [dict(zip(cols, r)) for r in (cur.fetchall() or [])]


class AnalysisStartIn(BaseModel):
    session_id: str = Field(min_length=1)
    name: Optional[str] = None
    mode: str = "new"
    created_by: Optional[str] = None


@router.post("", summary="Start an analysis event for an upload session")
def post_analysis(body: AnalysisStartIn) -> dict:
    try:
        analysis_id = analysis_store.start(
            session_id=body.session_id, name=body.name, mode=body.mode,
            created_by=body.created_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("could not start analysis for session=%s", body.session_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"analysis_id": analysis_id}


@router.get("", summary="List analysis events, newest first")
def list_analyses(status: Optional[str] = None, deal_id: Optional[str] = None,
                  limit: int = 50, offset: int = 0) -> dict:
    where, params = ["1=1"], []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if deal_id:
        where.append("EXISTS (SELECT 1 FROM proc.bp_analysis_deal ad "
                     "WHERE ad.analysis_id = a.analysis_id AND ad.deal_id = %s)")
        params.append(deal_id)
    rows = _query(
        f"SELECT {_LIST_COLS}, "
        "  (SELECT COALESCE(json_agg(json_build_object("
        "        'deal_id', ad.deal_id, 'version', ad.version)), '[]'::json) "
        "     FROM proc.bp_analysis_deal ad "
        "    WHERE ad.analysis_id = a.analysis_id) AS deals "
        "  FROM proc.bp_analysis a "
        f" WHERE {' AND '.join(where)} "
        " ORDER BY a.started_at DESC LIMIT %s OFFSET %s",
        tuple(params) + (max(1, min(int(limit), 200)), max(0, int(offset))),
    )
    return {"analyses": rows, "total": len(rows)}


# Declared BEFORE /{analysis_id}: otherwise the path param captures 'by-deal'
# and 'by-session'. Same trap the gateway documents at spendiq.controller.ts:80.
@router.get("/by-deal/{deal_id}", summary="Version history for one deal")
def get_by_deal(deal_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS}, ad.version, ad.is_latest "
        "  FROM proc.bp_analysis a "
        "  JOIN proc.bp_analysis_deal ad ON ad.analysis_id = a.analysis_id "
        " WHERE ad.deal_id = %s ORDER BY ad.version DESC",
        (deal_id,),
    )
    # Delta against the NEXT row down (the previous version). The oldest version
    # has nothing to compare against, so its delta is None — not zero.
    for i, row in enumerate(rows):
        prev = rows[i + 1] if i + 1 < len(rows) else None
        row["delta"] = None if prev is None else {
            "document_count": _diff(row.get("document_count"),
                                    prev.get("document_count")),
            "value_found": _diff(row.get("value_found"), prev.get("value_found")),
        }
    return {"deal_id": deal_id, "versions": rows}


def _diff(now: Any, before: Any) -> Optional[float]:
    """None when either side is unknown — an unknown is not a zero change."""
    if now is None or before is None:
        return None
    return round(float(now) - float(before), 2)


@router.get("/by-session/{session_id}", summary="The analysis for one upload")
def get_by_session(session_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS} FROM proc.bp_analysis a WHERE a.session_id = %s",
        (session_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="no analysis for that session")
    return _hydrate(rows[0])


@router.get("/{analysis_id}", summary="One analysis event in full")
def get_analysis(analysis_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS}, a.findings FROM proc.bp_analysis a "
        " WHERE a.analysis_id = %s", (analysis_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="no such analysis")
    return _hydrate(rows[0])


def _hydrate(row: dict) -> dict:
    aid = row["analysis_id"]
    row["documents"] = _query(
        "SELECT doc_type, doc_pk, file_path, file_name, outcome "
        "  FROM proc.bp_analysis_document WHERE analysis_id = %s "
        " ORDER BY file_name", (aid,))
    row["deals"] = _query(
        "SELECT deal_id, version, is_latest, linked_at "
        "  FROM proc.bp_analysis_deal WHERE analysis_id = %s "
        " ORDER BY deal_id", (aid,))
    return row
```

- [ ] **Step 4: Mount the router**

In `src/api/main.py`, add the import alongside the other router imports:

```python
from api.routers import analysis as analysis_router
```

and after line 425 (`app.include_router(value_summary_router.router)`) add:

```python
app.include_router(analysis_router.router)
```

If the surrounding imports in that file use a different style (some are plain `from api.routers import documents`), match whichever style its immediate neighbours use.

- [ ] **Step 5: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/api/test_analysis_endpoints.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Confirm the router is actually mounted**

```bash
./venv/bin/python -c "
from src.api.main import app
print(sorted(r.path for r in app.routes if '/analysis' in r.path))
"
```
Expected: all five paths listed, with `by-deal` and `by-session` present.

- [ ] **Step 7: Commit**

```bash
git add src/api/routers/analysis.py src/api/main.py tests/api/test_analysis_endpoints.py
git commit -m "feat(analysis): five read endpoints for analysis events

by-deal/by-session declared before /{analysis_id} so the path param does not
capture them. A deal with no analyses returns an empty list, not an error —
that is the state 5,040 of 5,043 deals are in."
```

---

## Task 8: Backfill

**Files:**
- Create: `scripts/backfill_analysis_events.py`
- Test: `tests/services/test_backfill_analysis_events.py`

**Interfaces:**
- Consumes: `analysis_store.start()`.
- Produces: `backfill(*, dry_run: bool = False, conn=None) -> dict` returning `{"sessions": int, "created": int, "documents": int, "links": int}`.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_backfill_analysis_events.py`:

```python
"""Backfill of pre-existing upload sessions.

Live reality on 2026-08-01: exactly 3 sessions exist, so this creates 3 events.
findings stays NULL — the historical findings were never captured and inventing
them now would be fabrication."""
import importlib

mod = importlib.import_module("scripts.backfill_analysis_events")


class FakeCursor:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._results.pop(0) if self._results else []

    def fetchone(self):
        return self._results.pop(0) if self._results else None


class FakeConn:
    def __init__(self, results):
        self.cur = FakeCursor(results)

    def cursor(self):
        return self.cur

    def commit(self):
        pass

    def rollback(self):
        pass


def test_backfill_uses_created_date_not_start_ts():
    """process_monitor_watcher sets start_ts = NULL on rows it reaps
    (process_monitor_watcher.py:1046), so start_ts is not a durable timestamp."""
    conn = FakeConn([[]])
    mod.backfill(conn=conn)
    sql = conn.cur.calls[0][0]
    assert "MIN(pm.created_date)" in sql
    assert "start_ts" not in sql


def test_backfill_leaves_findings_null(monkeypatch):
    conn = FakeConn([[("ses-1", "Test Deal", "2026-07-29")], [], []])
    monkeypatch.setattr(mod.analysis_store, "start", lambda **kw: "aid-1")

    mod.backfill(conn=conn)

    updates = [s for s, _ in conn.cur.calls if "UPDATE proc.bp_analysis" in s]
    assert updates, "the analysis must be marked complete"
    assert all("findings" not in s for s in updates)


def test_backfill_is_idempotent_via_the_session_id_constraint():
    conn = FakeConn([[]])
    mod.backfill(conn=conn)
    assert "NOT EXISTS" in conn.cur.calls[0][0]


def test_dry_run_writes_nothing(monkeypatch):
    conn = FakeConn([[("ses-1", "Test Deal", "2026-07-29")]])
    monkeypatch.setattr(mod.analysis_store, "start",
                        lambda **kw: (_ for _ in ()).throw(
                            AssertionError("must not write")))

    got = mod.backfill(dry_run=True, conn=conn)

    assert got["sessions"] == 1 and got["created"] == 0
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
./venv/bin/python -m pytest tests/services/test_backfill_analysis_events.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `scripts/backfill_analysis_events.py`:

```python
"""One-off, idempotent backfill of analysis events from existing upload sessions.

Live reality on 2026-08-01: only 3 of 5,043 deals ever came through an upload
session, so this creates 3 events. The other 5,040 deals legitimately have no
analysis history — that is the correct state for them, not a gap to fill.

findings stays NULL. The historical findings were never captured; reading live
data now and presenting it as "what we found then" would be fabrication.

Usage:
    ./venv/bin/python -m scripts.backfill_analysis_events --dry-run
    ./venv/bin/python -m scripts.backfill_analysis_events
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from src.services import analysis_store
from src.services.db import get_conn

log = logging.getLogger(__name__)

# created_date, NOT start_ts: the watcher's stale-row cleanup sets start_ts to
# NULL on rows it reaps (src/services/process_monitor_watcher.py:1046), so it is
# not a durable record of when the upload happened.
_SESSIONS = """
SELECT pm.session_id, MAX(pm.deal_name) AS deal_name,
       MIN(pm.created_date) AS created_date
  FROM proc.process_monitor pm
 WHERE pm.session_id IS NOT NULL
   AND EXISTS (SELECT 1 FROM proc.session_document_outcome sdo
                WHERE sdo.session_id = pm.session_id)
   AND NOT EXISTS (SELECT 1 FROM proc.bp_analysis a
                    WHERE a.session_id = pm.session_id)
 GROUP BY pm.session_id
 ORDER BY MIN(pm.created_date)
"""


def backfill(*, dry_run: bool = False, conn: Optional[Any] = None) -> dict:
    result = {"sessions": 0, "created": 0, "documents": 0, "links": 0}
    own = conn is None
    c = conn if conn is not None else get_conn().__enter__()
    try:
        cur = c.cursor()
        cur.execute(_SESSIONS)
        rows = cur.fetchall() or []
        result["sessions"] = len(rows)
        if dry_run:
            for session_id, deal_name, created in rows:
                log.info("would create: session=%s name=%s at=%s",
                         session_id, deal_name, created)
            return result

        for session_id, deal_name, created in rows:
            # mode is always 'new': process_monitor.is_new_deal exists but is
            # NULL on every row, so it cannot distinguish new from amend.
            analysis_id = analysis_store.start(
                session_id=session_id, name=deal_name, mode="new", conn=c)
            result["created"] += 1

            cur.execute(
                """
                INSERT INTO proc.bp_analysis_document
                       (analysis_id, doc_type, file_path, file_name, outcome)
                SELECT %s, sdo.document_type, sdo.file_path,
                       regexp_replace(sdo.file_path, '^.*/', ''), sdo.outcome
                  FROM proc.session_document_outcome sdo
                 WHERE sdo.session_id = %s
                ON CONFLICT (analysis_id, file_path) DO NOTHING
                """,
                (analysis_id, session_id),
            )
            cur.execute(
                "SELECT DISTINCT deal_id FROM proc.process_monitor "
                "WHERE session_id = %s AND deal_id IS NOT NULL",
                (session_id,))
            for (deal_id,) in (cur.fetchall() or []):
                cur.execute(
                    "UPDATE proc.bp_analysis_deal SET is_latest = false "
                    "WHERE deal_id = %s", (str(deal_id),))
                cur.execute(
                    "INSERT INTO proc.bp_analysis_deal "
                    "       (analysis_id, deal_id, version, is_latest) "
                    "VALUES (%s, %s, %s, true) "
                    "ON CONFLICT (analysis_id, deal_id) DO NOTHING",
                    (analysis_id, str(deal_id),
                     analysis_store._allocate_version(cur, deal_id)))
                result["links"] += 1

            cur.execute(
                """
                UPDATE proc.bp_analysis
                   SET status = 'complete', started_at = %s,
                       completed_at = %s,
                       document_count = (SELECT COUNT(*)
                                           FROM proc.bp_analysis_document
                                          WHERE analysis_id = %s)
                 WHERE analysis_id = %s
                """,
                (created, created, analysis_id, analysis_id),
            )
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        if own:
            c.close()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    print(backfill(dry_run=p.parse_args().dry_run))
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
./venv/bin/python -m pytest tests/services/test_backfill_analysis_events.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Dry-run against bp_sqldb, then run it**

```bash
./venv/bin/python -m scripts.backfill_analysis_events --dry-run
```
Expected: `{'sessions': 3, 'created': 0, ...}` and three "would create" log lines naming `ses-20260729-Z8P4`, `ses-20260729-BPTF`, `ses-20260730-UJF3`.

```bash
./venv/bin/python -m scripts.backfill_analysis_events
```
Expected: `{'sessions': 3, 'created': 3, 'links': 3, ...}`.

Run it a **second** time. Expected: `{'sessions': 0, 'created': 0, ...}` — proving idempotency.

- [ ] **Step 6: Commit**

```bash
git add scripts/backfill_analysis_events.py tests/services/test_backfill_analysis_events.py
git commit -m "feat(analysis): backfill events from the 3 existing upload sessions

Uses created_date, not start_ts — the watcher nulls start_ts on rows it reaps.
findings stays NULL: the historical findings were never captured, and reading
live data now and calling it 'what we found then' would be fabrication."
```

---

## Task 9: Upload starts the analysis and routes to it

**Repo:** `/home/muthu/PycharmProjects/beyond_procwise_ui` (branch `spendiq-ui`)

**Files:**
- Create: `src/modules/AnalyseUpload/startAnalysis.js`
- Create: `src/modules/AnalyseUpload/startAnalysis.test.js`
- Modify: `src/modules/AnalyseUpload/nextRoute.js`
- Modify: `src/modules/AnalyseUpload/nextRoute.test.js`
- Modify: `src/modules/AnalyseUpload/index.jsx:343-362`

**Interfaces:**
- Consumes: `POST /analysis` from Task 7.
- Produces: `startAnalysis({ sessionId, name, mode }) -> Promise<string | null>` — resolves to the `analysis_id`, or `null` on any failure. **Never throws.** And `nextRouteAfterUpload({ mode, analysisId, dealId, sessionId }) -> string`.

- [ ] **Step 1: Write the failing tests**

Create `src/modules/AnalyseUpload/startAnalysis.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { startAnalysis } from './startAnalysis';

vi.mock('axios');

describe('startAnalysis', () => {
  beforeEach(() => vi.resetAllMocks());

  it('returns the analysis id from BP_Backend', async () => {
    axios.post.mockResolvedValue({ data: { analysis_id: 'a-1' } });
    await expect(startAnalysis({ sessionId: 'S9', name: 'Q3', mode: 'new' }))
      .resolves.toBe('a-1');
  });

  it('never throws — a failed start must not lose the upload', async () => {
    axios.post.mockRejectedValue(new Error('network'));
    await expect(startAnalysis({ sessionId: 'S9' })).resolves.toBeNull();
  });

  it('does not call the API without a session id', async () => {
    await expect(startAnalysis({ sessionId: '' })).resolves.toBeNull();
    expect(axios.post).not.toHaveBeenCalled();
  });
});
```

Rewrite `src/modules/AnalyseUpload/nextRoute.test.js` entirely:

```js
import { describe, it, expect } from 'vitest';
import { nextRouteAfterUpload } from './nextRoute';

describe('nextRouteAfterUpload', () => {
  it('routes to the analysis event when there is one', () => {
    expect(nextRouteAfterUpload({ mode: 'new', analysisId: 'a-1', dealId: 'ACME01' }))
      .toBe('/spendiq?view=analytics&analysis=a-1');
  });

  it('amend routes to the analysis too — not straight to the deal', () => {
    // The regression this guards: amending a deal used to skip the report
    // entirely and leave no analysis record at all.
    expect(nextRouteAfterUpload({ mode: 'amend', analysisId: 'a-2', dealId: 'ACME01' }))
      .toBe('/spendiq?view=analytics&analysis=a-2');
  });

  it('bulk routes to the analysis', () => {
    expect(nextRouteAfterUpload({ mode: 'bulk', analysisId: 'a-3' }))
      .toBe('/spendiq?view=analytics&analysis=a-3');
  });

  it('falls back to the old deal route when the analysis could not be started', () => {
    expect(nextRouteAfterUpload({ mode: 'new', dealId: 'ACME01', sessionId: 'S9' }))
      .toBe('/spendiq?view=analysis-report&deal=ACME01&session=S9');
  });

  it('amend falls back to the deal detail', () => {
    expect(nextRouteAfterUpload({ mode: 'amend', dealId: 'ACME01' }))
      .toBe('/spendiq?view=analytics&deal=ACME01');
  });

  it('with nothing at all, still opens Analyse', () => {
    expect(nextRouteAfterUpload({ mode: 'new' })).toBe('/spendiq?view=analysis-report');
  });
});
```

- [ ] **Step 2: Run them to make sure they fail**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
npx vitest run src/modules/AnalyseUpload
```
Expected: FAIL — `startAnalysis` does not exist; `nextRouteAfterUpload` ignores `analysisId`.

- [ ] **Step 3: Write `startAnalysis`**

Create `src/modules/AnalyseUpload/startAnalysis.js`:

```js
// Registers the analysis event for an upload, on BP_Backend (VITE_AI_API_URL).
//
// This call is an OPTIMISATION, not a correctness dependency. If it fails — or
// the browser closes before it fires — the backend's scheduled sweep creates the
// event from process_monitor instead. It just will not carry the name the user
// typed. So this must never throw and never block the navigation.
import axios from 'axios'

export async function startAnalysis({ sessionId, name, mode = 'new' }) {
  if (!sessionId) return null
  try {
    const { data } = await axios.post(
      `${import.meta.env.VITE_AI_API_URL}/analysis`,
      { session_id: sessionId, name: name || null, mode }
    )
    return data?.analysis_id || null
  } catch (err) {
    console.warn('[Analyse] could not register the analysis event —', err?.message)
    return null
  }
}
```

- [ ] **Step 4: Rewrite `nextRoute.js`**

```js
// Where an upload lands.
//
// Every mode now lands on its ANALYSIS EVENT, including amend — the point of
// uploading is to see what the documents revealed, and the analysis header names
// the deal and links to it. Amend used to go straight to the deal detail, which
// is why amending a deal left no visible record at all.
//
// The fallbacks below are the pre-analysis-event routes, used only when the
// analysis could not be registered (see startAnalysis.js). The backend sweep
// creates the event regardless, so the fallback is a worse landing page, not
// lost data.
export function nextRouteAfterUpload({ mode, analysisId, dealId, sessionId }) {
  if (analysisId) {
    return `/spendiq?view=analytics&analysis=${encodeURIComponent(analysisId)}`
  }
  const id = dealId ? encodeURIComponent(dealId) : ''
  if (mode === 'amend') {
    return id ? `/spendiq?view=analytics&deal=${id}` : '/spendiq?view=analytics'
  }
  if (!id) return '/spendiq?view=analysis-report'
  const sess = sessionId ? `&session=${encodeURIComponent(sessionId)}` : ''
  return `/spendiq?view=analysis-report&deal=${id}${sess}`
}
```

- [ ] **Step 5: Wire it into the upload handler**

In `src/modules/AnalyseUpload/index.jsx`, add the import next to the `nextRouteAfterUpload` import:

```js
import { startAnalysis } from './startAnalysis'
```

Then replace lines 359-362 (from `const { dealId: newDealId, sessionId } = ...` through the `navigate(...)` call) with:

```js
      const { dealId: newDealId, sessionId } = lastUploadResult || {}
      const dealId = mode === 'amend' ? (selectedDeal.deal_id || newDealId) : newDealId
      // Register the analysis event before navigating so we can land on it.
      // Returns null on failure; nextRouteAfterUpload falls back in that case.
      const analysisId = await startAnalysis({
        sessionId: batchSessionId || sessionId, name, mode,
      })
      showSnackbar('Documents in — finding opportunities and flagging anomalies…', 'success')
      navigate(nextRouteAfterUpload({ mode, analysisId, dealId, sessionId }))
```

- [ ] **Step 6: Run the tests and make sure they pass**

```bash
npx vitest run src/modules/AnalyseUpload
```
Expected: 9 passed.

- [ ] **Step 7: Commit**

```bash
git add src/modules/AnalyseUpload/
git commit -m "feat(analyse): register an analysis event on upload and land on it

Amend now lands on its analysis too. It used to go straight to the deal
detail, which is why amending a deal left no record at all.

The POST is an optimisation, not a dependency — the backend sweep creates
the event either way, so a failure costs the chosen name, not the data."
```

---

## Task 10: The Analyse area lists analysis events

**Repo:** `beyond_procwise_ui`

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` — `loadLiveAnalyses` (line 1561), `liveAnalysesPanel` (1583), `analysisSubNav` (7361), `openSnapshot` (3856), the URL parser (7953)
- Create: `src/modules/SpendIQ/analysisEvents.contract.test.js`

**Interfaces:**
- Consumes: `GET /analysis` and `GET /analysis/{id}` from Task 7.
- Produces: `openAnalysis(analysisId)` in `engine.js`; the route `?view=analytics&analysis=<id>`.

- [ ] **Step 1: Write the failing contract test**

Create `src/modules/SpendIQ/analysisEvents.contract.test.js`:

```js
// engine.js is a classic script injected via ?raw and cannot be imported, so its
// behaviour is asserted against its source — the convention established by
// valueSummary.contract.test.js.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const engine = readFileSync(join(here, 'engine.js'), 'utf8');

function fnSource(name) {
  const start = engine.indexOf(`function ${name}(`);
  expect(start, `${name}() should exist in engine.js`).toBeGreaterThan(-1);
  let depth = 0;
  for (let j = engine.indexOf('{', start); j < engine.length; j++) {
    if (engine[j] === '{') depth++;
    else if (engine[j] === '}' && --depth === 0) {
      return engine.slice(start, j + 1)
        .replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
    }
  }
  throw new Error(`could not find end of ${name}()`);
}

describe('Analyse area lists analysis events', () => {
  it('loads from BP_Backend /analysis, not the gateway drafts endpoint', () => {
    const src = fnSource('loadLiveAnalyses');
    expect(src).toContain("'/analysis'");
    expect(src).not.toContain('live-analyses');
  });

  it('uses the AI bridge, since /analysis lives on BP_Backend', () => {
    expect(fnSource('loadLiveAnalyses')).toContain('__SPENDIQ_API_AI__');
  });

  it('keeps the three-state contract — an error must not read as "none"', () => {
    const src = fnSource('liveAnalysesPanel');
    expect(src).toMatch(/Couldn't load/i);
    expect(src).toContain('liveAnalyses === null');
  });

  it('no longer calls the list "drafts" — it lists every analysis', () => {
    const src = fnSource('liveAnalysesPanel');
    expect(src).not.toMatch(/not yet promoted/i);
    expect(src).toContain('Analyses');
  });

  it('opens an analysis by analysis id, not by deal id', () => {
    const src = fnSource('openAnalysis');
    expect(src).toContain('currentAnalysis');
    expect(src).toContain('renderBody');
  });

  it('accepts ?view=analytics&analysis= as a deep link', () => {
    expect(engine).toContain("params.get('analysis')");
  });

  it('keeps the legacy analysis-report deep link working as a redirect', () => {
    expect(engine).toContain('analysis-report');
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
npx vitest run src/modules/SpendIQ/analysisEvents.contract.test.js
```
Expected: FAIL — `openAnalysis()` does not exist; `loadLiveAnalyses` still calls `/spendiq/live-analyses`.

- [ ] **Step 3: Repoint the loader**

In `engine.js`, replace the body of `loadLiveAnalyses()` (line 1561) so it calls BP_Backend. Keep the three-state contract exactly as it is:

```js
async function loadLiveAnalyses(){
  if(liveAnalysesLoading) return;
  if(!window.__SPENDIQ_API_AI__){
    liveAnalysesError = new Error('no ai api bridge');
    renderBody();
    return;
  }
  liveAnalysesLoading = true;
  try{
    // GET /analysis on BP_Backend — every analysis event, not just the ones
    // whose deal has not been promoted. The old gateway endpoint was a query
    // for is_tracked=false deals, which is exactly why an analysis vanished
    // from this list the moment its deal became real.
    const r = await window.__SPENDIQ_API_AI__('/analysis', {limit:50});
    liveAnalyses = Array.isArray(r && r.analyses) ? r.analyses : [];
    liveAnalysesError = null;
  }catch(e){
    liveAnalysesError = e instanceof Error ? e : new Error(String((e && e.message) || e));
    console.warn('[SpendIQ] analyses unavailable — GET /analysis failed:', liveAnalysesError.message);
  }
  liveAnalysesLoading = false;
  window.__SPENDIQ_DATA__ && (window.__SPENDIQ_DATA__['report.drafts'] = liveAnalyses || []);
  renderNav();
  renderBody();
}
```

- [ ] **Step 4: Retitle and reshape the panel**

Replace the panel's title block and its row renderer in `liveAnalysesPanel()`. The row now reads from the `/analysis` shape (`analysis_id`, `name`, `started_at`, `document_count`, `value_found`, `currency`, `status`, `deals`):

```js
function liveAnalysesPanel(){
  const fmtValue = a => (a.value_found == null)
    ? '—'
    : ((window.__SIQ_FMT && window.__SIQ_FMT.formatCompactCurrency)
        ? window.__SIQ_FMT.formatCompactCurrency(Number(a.value_found), {currency:a.currency||'GBP'})
        : '£'+Number(a.value_found).toLocaleString('en-GB'));
  let body;
  if(liveAnalysesError){
    // Fail-closed: an error must read as "couldn't load", never as "there are no
    // analyses" — those are different facts.
    body = `<div class="p-sub">Couldn't load analyses. <a href="#" onclick="event.preventDefault();loadLiveAnalyses()">Retry</a></div>`;
  } else if(liveAnalyses === null){
    body = `<div class="p-sub">Loading analyses…</div>`;
  } else if(!liveAnalyses.length){
    body = `<div class="p-sub">No analyses yet. Upload documents from Find an opportunity to run one.</div>`;
  } else {
    const rows = liveAnalyses.map(a => {
      const deals = Array.isArray(a.deals) ? a.deals : [];
      const bits = [];
      if(a.started_at) bits.push(escH(String(a.started_at).slice(0,10)));
      if(a.document_count) bits.push(a.document_count+' document'+(a.document_count===1?'':'s'));
      bits.push(fmtValue(a)+' found');
      if(deals.length) bits.push(deals.length===1
        ? escH(deals[0].deal_id)+' · v'+deals[0].version
        : deals.length+' deals');
      const badge = a.status==='running'
        ? `<span class="tag warn">Running</span>`
        : (a.status==='failed' ? `<span class="tag crit">Failed</span>` : '');
      return `<div class="acti" onclick="openAnalysis(${jsStr(String(a.analysis_id))})">
        <div class="acti-top">
          <div style="flex:1;min-width:0">
            <div class="acti-title">${escH(a.name || 'Untitled analysis')} ${badge}</div>
            <div class="acti-desc">${bits.join(' · ')}</div>
          </div>
        </div>
      </div>`;
    }).join('');
    body = `<div class="acti-list">${rows}</div>`;
  }
  return `<div class="panel2">
    <div class="p-title">Analyses</div>
    <div class="p-sub">Every analysis you have run — one-off, or linked to a deal. Open one to see what it found.</div>
    ${body}
  </div>`;
}
```

- [ ] **Step 5: Add `openAnalysis` and the deep link**

Next to `openSnapshot` (line 3856), replace it with:

```js
let currentAnalysis = null;
// Opens an analysis EVENT by its own id. The old openSnapshot() took a deal id,
// which is why an analysis could not exist without a deal.
function openAnalysis(id){
  currentAnalysis = id || null;
  currentSnapshot = null;
  go('analytics');
  analysisMode = 'compare';
  subTab = 'Overview';
  // reportLoadAnalysis lands in Task 11. Guarded so this task is testable and
  // shippable on its own — until then, opening an analysis navigates but does
  // not load its findings.
  if(typeof reportLoadAnalysis === 'function') reportLoadAnalysis(id);
  renderNav(); renderTabs(); renderBody();
}
```

In the URL parser at line 7953, extend the accepted view set and read the new param:

```js
    var valid={analytics:1,home:1,'analysis-report':1};
    var an = params.get('analysis');
    if(an){ openAnalysis(an); return; }
```

**Leave the `analysis-report` branch intact.** The old `?view=analysis-report&deal=` deep link must keep working: the Actions list still generates it (`engine.js:1601`), and links already sent to users must not 404. It simply stops being how the Analyse list opens things.

- [ ] **Step 6: Point the sub-nav at analysis events**

In `analysisSubNav()` (line 7361), change the row renderer:

```js
  const rows = drafts.slice(0,6).map(a=>{
    const label = escH(a.name || 'Untitled analysis');
    const id = String(a.analysis_id);
    return `<button class="fn-subitem ${currentAnalysis===id?'active':''}" onclick="event.stopPropagation();openAnalysis('${id}')" title="${label}"><span class="dotpin"></span><span class="fn-sublabel">${label}</span></button>`;
  }).join('');
```

- [ ] **Step 7: Run the tests and make sure they pass**

```bash
npx vitest run src/modules/SpendIQ/analysisEvents.contract.test.js
```
Expected: 7 passed.

- [ ] **Step 8: Check nothing else in SpendIQ regressed**

```bash
npx vitest run src/modules/SpendIQ
```
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
git add src/modules/SpendIQ/
git commit -m "feat(analyse): list analysis events, not draft deals

The old list queried is_tracked=false deals, which is exactly why an analysis
vanished from Analyse the moment its deal was promoted. It now lists every
analysis event, opened by its own id via ?view=analytics&analysis=."
```

---

## Task 11: The report reads the frozen record

**Repo:** `beyond_procwise_ui`

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` — report state (5111-5145), `reportLoadData` (5380)
- Modify: `src/modules/SpendIQ/analysisEvents.contract.test.js`

**Interfaces:**
- Consumes: `GET /analysis/{id}` from Task 7, `openAnalysis()` from Task 10.
- Produces: `reportLoadAnalysis(analysisId)`, and the state flag `reportFrozen` (true when `status === 'complete'`).

- [ ] **Step 1: Write the failing tests**

Append to `src/modules/SpendIQ/analysisEvents.contract.test.js`:

```js
describe('a frozen analysis is a record, not a workspace', () => {
  it('loads a complete analysis from its stored findings', () => {
    const src = fnSource('reportLoadAnalysis');
    expect(src).toContain('/analysis/');
    expect(src).toContain('findings');
    expect(src).toContain('reportFrozen');
  });

  it('a running analysis still fetches live, as it does today', () => {
    const src = fnSource('reportLoadAnalysis');
    expect(src).toContain('reportLoadData');
  });

  it('does not offer state-changing controls when frozen', () => {
    // Resolve / approve / propose all mutate the deal. On a frozen record they
    // must link to the deal instead, or the record is neither.
    const src = fnSource('reportActionsAllowed');
    expect(src).toContain('!reportFrozen');
  });

  it('says so when the findings were never captured, rather than showing zero', () => {
    expect(engine).toMatch(/Findings were not captured/i);
  });
});
```

- [ ] **Step 2: Run them to make sure they fail**

```bash
npx vitest run src/modules/SpendIQ/analysisEvents.contract.test.js
```
Expected: FAIL — `reportLoadAnalysis` and `reportActionsAllowed` do not exist.

- [ ] **Step 3: Add the state and the loader**

Next to the other report state declarations (around line 5115), add:

```js
// The analysis event this report is showing, and whether it is frozen.
// A frozen analysis renders stored findings and offers no controls that would
// change anything — the live state belongs to the deal.
let reportAnalysisId = null;
let reportFrozen = false;
let reportAnalysis = null;
```

Add the loader next to `reportLoadData`:

```js
// Loads an analysis EVENT. A complete one renders from its stored snapshot; a
// running one falls through to the live fetches, which is what makes the report
// the working surface immediately after upload.
async function reportLoadAnalysis(analysisId){
  reportAnalysisId = analysisId || null;
  reportAnalysis = null; reportFrozen = false; reportDataLoaded = false;
  if(!analysisId || !window.__SPENDIQ_API_AI__){ renderBody(); return; }
  try{
    const a = await window.__SPENDIQ_API_AI__('/analysis/'+encodeURIComponent(analysisId));
    reportAnalysis = a || null;
    reportFrozen = (a && a.status === 'complete');
    reportDealId = (a && Array.isArray(a.deals) && a.deals.length) ? a.deals[0].deal_id : null;
    reportSession = (a && a.session_id) || null;
    if(reportFrozen){
      const f = a.findings;
      if(!f){
        // Backfilled analyses have no findings — they were never captured.
        // Say so; a zero here would be a fabricated fact.
        reportData = null;
        reportDataLoaded = true;
        renderBody();
        return;
      }
      reportData = {
        deal: (f.deals||[])[0] || null,
        discrepancies: f.discrepancies || [],
        opportunities: f.opportunities || [],
        summary: (f.summaries||[])[0] || null,
        benchmarks: f.benchmarks || [],
        sources: f.sources || {},
      };
      reportDataLoaded = true;
      renderBody();
      return;
    }
    reportLoadData();      // still running — live, exactly as today
  }catch(e){
    console.warn('[SpendIQ] could not load analysis', analysisId, e && e.message);
    reportDataLoaded = true;
    renderBody();
  }
}

// Whether this report may offer controls that change state. Frozen records may
// not: resolving a discrepancy or approving a document mutates the deal, and a
// record you can edit is not a record.
function reportActionsAllowed(){ return !reportFrozen; }
```

- [ ] **Step 4: Gate the action controls**

Find every state-changing control in the report:

```bash
grep -n "reportResolve\|_reportResolveAllConfirm\|confirmUnconnApprove\|unconnLoad\|dealProposal\|proposals/generate\|benchmarkLoad\|__SPENDIQ_API_AI_POST__" src/modules/SpendIQ/engine.js
```

Each hit that renders a button or fires a write must be gated. The pattern for a rendered control is:

```js
${reportActionsAllowed() ? `<button class="btn2" onclick="…">Resolve</button>` : ''}
```

and for a handler, an early return:

```js
  if(!reportActionsAllowed()){ toast('This is a record of a past analysis — open the deal to act on it.'); return; }
```

Where a whole block of controls is hidden, render this in its place:

```js
`<div class="p-sub">This is a record of what was found on ${escH(String(reportAnalysis&&reportAnalysis.completed_at||'').slice(0,10))}. <a href="#" onclick="event.preventDefault();openDeal('${escH(String(reportDealId||''))}')">Open the deal</a> to act on it.</div>`
```

Use `grep -n "reportResolve\|confirmUnconnApprove\|dealProposal\|benchmarkLoad" src/modules/SpendIQ/engine.js` to find them all.

- [ ] **Step 5: Add the not-captured message**

In the report's Overview renderer, when `reportFrozen && !reportData`:

```js
`<div class="panel2"><div class="p-title">Findings were not captured for this analysis</div>
 <div class="p-sub">This analysis predates the analysis-event record, so what it found that day was never stored. <a href="#" onclick="event.preventDefault();openDeal('${escH(String(reportDealId||''))}')">Open the deal</a> for current detail.</div></div>`
```

- [ ] **Step 6: Run the tests and make sure they pass**

```bash
npx vitest run src/modules/SpendIQ
```
Expected: 11 passed in `analysisEvents.contract.test.js`, no new failures elsewhere.

- [ ] **Step 7: Commit**

```bash
git add src/modules/SpendIQ/
git commit -m "feat(analyse): a complete analysis renders its frozen findings

Running analyses stay live — that is the working surface after upload. Once
frozen, state-changing controls link to the deal instead: a record you can
edit is not a record. Backfilled analyses say their findings were never
captured rather than showing a zero."
```

---

## Task 12: Analysis history on the deal

**Repo:** `beyond_procwise_ui`

**Files:**
- Create: `src/lib/analysisVersions.js`
- Create: `src/lib/analysisVersions.test.js`
- Modify: `src/modules/SpendIQ/index.jsx:88`
- Modify: `src/modules/SpendIQ/engine.js` — `dealView()` (3454)
- Modify: `src/modules/SpendIQ/analysisEvents.contract.test.js`

**Interfaces:**
- Consumes: `GET /analysis/by-deal/{deal_id}` from Task 7.
- Produces: `describeDelta(delta, currency) -> string` in `src/lib/analysisVersions.js`, exposed as `window.__SIQ_ANALYSIS__.describeDelta` (the bridge pattern used for `window.__SIQ_FMT` at `index.jsx:88`). In `engine.js`: `dealAnalysisPanel()` and `loadDealAnalyses(dealId)`.

- [ ] **Step 1: Write the failing test**

Create `src/lib/analysisVersions.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { describeDelta } from './analysisVersions';

describe('describeDelta', () => {
  it('describes documents added and value found going up', () => {
    expect(describeDelta({ document_count: 2, value_found: 3300 }, 'GBP'))
      .toBe('+2 documents · +£3,300 found');
  });

  it('describes a drop in value found', () => {
    expect(describeDelta({ document_count: 0, value_found: -500 }, 'GBP'))
      .toBe('No documents added · −£500 found');
  });

  it('says nothing changed when both are zero', () => {
    expect(describeDelta({ document_count: 0, value_found: 0 }, 'GBP'))
      .toBe('No change');
  });

  it('returns empty for the first version, which has nothing to compare to', () => {
    expect(describeDelta(null, 'GBP')).toBe('');
  });

  it('omits a figure that is unknown rather than calling it zero', () => {
    expect(describeDelta({ document_count: 1, value_found: null }, 'GBP'))
      .toBe('+1 document');
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
npx vitest run src/lib/analysisVersions.test.js
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write the helper**

Create `src/lib/analysisVersions.js`:

```js
// What changed between an analysis version and the one before it.
//
// Both sides are frozen snapshots of the same shape, so this compares stored
// numbers — it never recomputes anything. A null part is UNKNOWN, not zero, and
// is left out entirely rather than described as "no change".

const CURRENCY_SIGN = { GBP: '£', USD: '$', EUR: '€' }

function money(n, currency) {
  const sign = CURRENCY_SIGN[currency] || ''
  const abs = Math.abs(n).toLocaleString('en-GB')
  return `${n < 0 ? '−' : '+'}${sign}${abs}`
}

const NO_DOCS = 'No documents added'

export function describeDelta(delta, currency = 'GBP') {
  if (!delta) return ''
  const parts = []

  // null is UNKNOWN and is left out entirely. 0 is a known "nothing added" and
  // is stated, because "we ran an analysis and it added no documents" is a
  // finding; "we don't know how many" is not.
  const docs = delta.document_count
  if (docs != null) {
    if (docs > 0) parts.push(`+${docs} document${docs === 1 ? '' : 's'}`)
    else if (docs < 0) parts.push(`${docs} document${docs === -1 ? '' : 's'}`)
    else parts.push(NO_DOCS)
  }

  const value = delta.value_found
  if (value != null && value !== 0) parts.push(`${money(value, currency)} found`)

  if (!parts.length) return ''
  // The only thing we can say is "nothing was added and the value did not move".
  if (parts.length === 1 && parts[0] === NO_DOCS) return 'No change'
  return parts.join(' · ')
}
```

- [ ] **Step 4: Run the test**

```bash
npx vitest run src/lib/analysisVersions.test.js
```
Expected: 5 passed. If any assertion does not match exactly, rewrite `describeDelta` — **the tests are the contract, not this implementation**.

- [ ] **Step 5: Bridge it to `engine.js`**

In `src/modules/SpendIQ/index.jsx`, next to line 88:

```js
  import { describeDelta } from '../../lib/analysisVersions';
  // engine.js is a classic script and cannot import — same reason as __SIQ_FMT.
  window.__SIQ_ANALYSIS__ = { describeDelta };
```

Place the import with the file's other imports, not inline.

- [ ] **Step 6: Add the panel to `dealView()`**

Add near the other `siqLoad*` calls at the top of `dealView()`:

```js
  loadDealAnalyses(dealId);
```

And define, next to `loadLiveAnalyses`:

```js
// Analysis history for one deal. Three states, same fail-closed contract as
// loadLiveAnalyses: null = not loaded, Error = failed, [] = genuinely none.
// [] is the COMMON case — only 3 of 5,043 deals ever came through an upload.
let dealAnalyses = null, dealAnalysesError = null, dealAnalysesFor = null;
async function loadDealAnalyses(id){
  if(!id || dealAnalysesFor === id) return;
  dealAnalysesFor = id; dealAnalyses = null; dealAnalysesError = null;
  if(!window.__SPENDIQ_API_AI__){ dealAnalysesError = new Error('no ai api bridge'); return; }
  try{
    const r = await window.__SPENDIQ_API_AI__('/analysis/by-deal/'+encodeURIComponent(id));
    dealAnalyses = Array.isArray(r && r.versions) ? r.versions : [];
  }catch(e){
    dealAnalysesError = e instanceof Error ? e : new Error(String((e && e.message) || e));
  }
  renderBody();
}

function dealAnalysisPanel(){
  const D = window.__SIQ_ANALYSIS__ || {};
  let body;
  if(dealAnalysesError){
    body = `<div class="p-sub">Couldn't load the analysis history. <a href="#" onclick="event.preventDefault();dealAnalysesFor=null;loadDealAnalyses(dealId)">Retry</a></div>`;
  } else if(dealAnalyses === null){
    body = `<div class="p-sub">Loading analysis history…</div>`;
  } else if(!dealAnalyses.length){
    // The normal state for 5,040 of 5,043 deals: they were bulk-ingested and
    // never came through an upload. Not an error, and not a zero.
    body = `<div class="p-sub">No analysis has been run on this deal.</div>`;
  } else {
    body = `<div class="acti-list">${dealAnalyses.map(v=>{
      const delta = D.describeDelta ? D.describeDelta(v.delta, v.currency||'GBP') : '';
      const when = String(v.started_at||'').slice(0,10);
      const docs = v.document_count!=null ? v.document_count+' document'+(v.document_count===1?'':'s') : '';
      return `<div class="acti" onclick="openAnalysis(${jsStr(String(v.analysis_id))})">
        <div class="acti-top"><div style="flex:1;min-width:0">
          <div class="acti-title">v${v.version} · ${escH(v.name||'Untitled analysis')}${v.is_latest?' <span class="tag ok">Latest</span>':''}</div>
          <div class="acti-desc">${[escH(when), docs].filter(Boolean).join(' · ')}${delta?` <span style="color:var(--ink-3)">· ${escH(delta)}</span>`:''}</div>
        </div></div></div>`;
    }).join('')}</div>`;
  }
  return `<div class="panel2">
    <div class="p-title">Analysis history</div>
    <div class="p-sub">Every analysis run on this deal, newest first.</div>
    ${body}
    <button class="btn2" onclick="location.href='/analyse?deal='+encodeURIComponent(dealId)">Run a new analysis</button>
  </div>`;
}
```

Insert `${dealAnalysisPanel()}` into `dealView()`'s returned markup, directly after `${stagePanel}`.

- [ ] **Step 7: Add the contract assertions**

Append to `src/modules/SpendIQ/analysisEvents.contract.test.js`:

```js
describe('analysis history on the deal', () => {
  it('fetches the version history for the deal', () => {
    expect(fnSource('loadDealAnalyses')).toContain('/analysis/by-deal/');
  });

  it('renders "no analysis" as a normal state, not an error or a zero', () => {
    const src = fnSource('dealAnalysisPanel');
    expect(src).toContain('No analysis has been run on this deal');
    expect(src).toMatch(/Couldn't load/i);
  });

  it('offers a way to run a new analysis on the deal', () => {
    expect(fnSource('dealAnalysisPanel')).toContain('Run a new analysis');
  });

  it('the deal view renders the panel', () => {
    expect(fnSource('dealView')).toContain('dealAnalysisPanel()');
  });
});
```

- [ ] **Step 8: Preselect the deal on the upload page**

In `src/modules/AnalyseUpload/index.jsx`, after the `deals` list loads (near line 125), read `?deal=` from the URL and preselect it in amend mode:

```js
  // Entered from a deal's "Run a new analysis" — preselect and confirm it so
  // the user is not asked to find a deal they just came from.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get('deal')
    if (!wanted || !deals.length || selectedDeal) return
    const match = deals.find((d) => d.deal_id === wanted)
    if (match) { setMode('amend'); setSelectedDeal(match); setConfirmRef(match.deal_id) }
  }, [deals, selectedDeal])
```

Check the existing `confirmed` guard at `index.jsx:113` and make sure `setConfirmRef` supplies whatever that check compares against; adjust the value if it expects something other than the deal id.

- [ ] **Step 9: Run all UI tests**

```bash
npx vitest run src/lib src/modules/SpendIQ src/modules/AnalyseUpload
```
Expected: all pass, no new failures.

- [ ] **Step 10: Commit**

```bash
git add src/lib/analysisVersions.js src/lib/analysisVersions.test.js src/modules/
git commit -m "feat(deal): analysis history with per-version deltas

Deltas compare stored snapshot numbers, never recompute. A deal with no
history says so plainly — that is the normal state for 5,040 of 5,043 deals,
not an error. 'Run a new analysis' opens the upload page preselected."
```

---

## Task 13: Live verification

**Files:** none — this task proves the feature works on the running local server against live `bp_sqldb`.

**Interfaces:** consumes everything above.

- [ ] **Step 1: Start the stack**

Start BP_Backend (uvicorn) and the gateway. The gateway needs `node --experimental-global-webcrypto`. **Never `pkill -f uvicorn`** — another session may be running one.

- [ ] **Step 2: Confirm the endpoints answer**

```bash
curl -s localhost:8000/analysis | head -c 400
curl -s localhost:8000/analysis/by-deal/TESTDEAL2026072901 | head -c 400
```
Expected: the three backfilled analyses in the first; one version (`v1`, `delta: null`) in the second.

- [ ] **Step 3: New analysis end to end**

Upload a multi-document deal through `/analyse` in the UI. Confirm:
- an analysis appears in the Analyse list immediately, badged **Running**
- it flips to complete and shows a document count and a value-found figure
- the deal shows it as **v1** with no delta

- [ ] **Step 4: Amend produces v2 — the regression this feature exists for**

From that deal, click **Run a new analysis**, upload one more document. Confirm:
- the upload page opens with the deal preselected
- it lands on the **new analysis**, not the bare deal
- the deal now shows **v2** with a headline delta, and v1 is still there

- [ ] **Step 5: Promotion does not hide the analyses**

Promote the deal to Pipeline. Confirm **both** analyses are still in the Analyse list. This is the direct test for the "it disappears once the deal is real" symptom.

- [ ] **Step 6: One-off analysis**

Upload documents in bulk mode that form no single deal. Confirm an analysis appears with its documents listed and no deal link, and that opening it does not error.

- [ ] **Step 7: A backfilled analysis is honest**

Open one of the three backfilled analyses. Confirm it shows *"Findings were not captured for this analysis"* and a working link to the deal — not a zero, not an empty chart.

- [ ] **Step 8: A bulk-ingested deal is honest**

Open any deal that never came through an upload. Confirm its Analysis history reads *"No analysis has been run on this deal"* with the "Run a new analysis" button — not a spinner, not an error.

- [ ] **Step 9: Record the result**

Write what actually happened — including anything that did not work — into the plan file under a "Verification results" heading, then commit.

```bash
git add docs/superpowers/plans/2026-08-01-analysis-events.md
git commit -m "docs(analysis): live verification results"
```

---

## Verification results

_(Filled in during Task 13.)_
