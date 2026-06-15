# Persona Summary Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persona-driven summary service that summarizes a single deal or the whole portfolio over the `bp_*_trgt` target tables, caches/versions results in `proc.bp_summary`, refreshes on demand, and precomputes daily.

**Architecture:** A `summary_agent` service (mirroring the proven `deal_summary.py` — direct SQL, cloud LLM, no GPU) resolves a persona from `bp_prompt` (raw-string fallback), gathers deal data (reusing `gather_deal_context`) or portfolio aggregates, generates a grounded summary via `ollama_cloud_generate`, and persists it to `bp_summary` with an `is_current` flag. A new router exposes refresh/cache/history endpoints; `backend_scheduler` runs a daily precompute.

**Tech Stack:** PostgreSQL (`bp_sqldb`), Python 3.12, psycopg2, FastAPI, pytest, Ollama Cloud API.

**Conventions:**
- `bp_` table prefix; indexes `ix_bp_<table>_<suffix>`. No Claude attribution in commits.
- `docs/superpowers/` is gitignored — do NOT commit plan/spec files; only code/test/SQL.
- pytest: `.venv/bin/pytest`. Repo root + src are already on `sys.path` via `tests/conftest.py`.
- DB: `set -a && . ./.env 2>/dev/null && set +a` then `PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" ...` (a harmless `supplierconnect: command not found` line may print when sourcing `.env` — ignore it).
- Commit ONLY the files each task lists with an explicit `git add` — the working tree has unrelated modified files that must stay out.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deploy/sql/2026-06-08_create_bp_summary.sql` | `bp_summary` DDL + 3 persona seeds into `bp_prompt` | Create |
| `src/services/summary_agent.py` | persona resolve, portfolio gather, prompt build, store, generate, precompute | Create |
| `src/api/routers/summary.py` | `/summary` endpoints | Create |
| `src/api/main.py` | register the summary router | Modify |
| `src/services/backend_scheduler.py` | daily precompute job | Modify |
| `config/settings.py` | two new settings fields | Modify |
| `tests/test_summary_agent.py` | unit tests for the service | Create |
| `tests/test_summary_api.py` | endpoint tests | Create |

Reused unchanged: `src/services/deal_summary.py` (`gather_deal_context`, `_build_prompt`), `src/services/ollama_client.py` (`ollama_cloud_generate`), `src/services/db.py` (`get_conn`).

---

## Task 1: Migration — `bp_summary` table + persona seeds

**Files:**
- Create: `deploy/sql/2026-06-08_create_bp_summary.sql`

- [ ] **Step 1: Write the migration SQL**

Create `deploy/sql/2026-06-08_create_bp_summary.sql` with exactly:

```sql
-- Persona summary cache/history. summary_id is an opaque UUID. Each row stores
-- the generated summary plus the exact data snapshot it was built from (so an
-- as_of request can regenerate over a past data state). Idempotent.

CREATE TABLE IF NOT EXISTS proc.bp_summary (
    summary_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona         TEXT        NOT NULL,
    persona_source  TEXT        NOT NULL,
    scope           TEXT        NOT NULL,
    deal_id         VARCHAR(25),
    summary         TEXT        NOT NULL,
    data_snapshot   JSONB       NOT NULL,
    sources         JSONB,
    model           TEXT,
    is_current      BOOLEAN     NOT NULL DEFAULT true,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL DEFAULT 'system'
);

CREATE INDEX IF NOT EXISTS ix_bp_summary_current
    ON proc.bp_summary (persona, deal_id) WHERE is_current;
CREATE INDEX IF NOT EXISTS ix_bp_summary_lookup
    ON proc.bp_summary (persona, deal_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_summary_deal
    ON proc.bp_summary (deal_id);

-- Seed persona framings into the governance prompt table. Guarded on
-- prompt_name so re-runs are no-ops.
INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'analysis', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement data analyst. Emphasize spend totals, price and volume trends, supplier concentration, and quantitative anomalies."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'analysis' AND prompt_type = 'summary_persona');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'negotiation', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement negotiation strategist. Emphasize leverage points, price gaps between quotes, purchase orders and invoices, contract and renewal timing, and concession opportunities."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'negotiation' AND prompt_type = 'summary_persona');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'compliance', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement compliance auditor. Emphasize policy adherence, discrepancies, missing approvals, tax and currency correctness, and audit flags."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'compliance' AND prompt_type = 'summary_persona');
```

- [ ] **Step 2: Apply to bp_sqldb**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env 2>/dev/null && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -f deploy/sql/2026-06-08_create_bp_summary.sql && echo MIGRATION_OK
```
Expected: `CREATE TABLE`, 3× `CREATE INDEX`, 3× `INSERT 0 1`, `MIGRATION_OK`.

- [ ] **Step 3: Verify table + persona seeds + idempotency**

Run:
```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT to_regclass('proc.bp_summary'); SELECT count(*) FROM proc.bp_prompt WHERE prompt_type='summary_persona';"
```
Expected: `proc.bp_summary` then `3`. Re-run the migration file once more; the persona count must remain `3`.

- [ ] **Step 4: Commit**

```bash
git add deploy/sql/2026-06-08_create_bp_summary.sql
git commit -m "feat(summary): create bp_summary table and seed persona prompts"
```

---

## Task 2: `resolve_persona` — persona from bp_prompt with raw fallback

**Files:**
- Create: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_agent.py` with:

```python
import json
import src.services.summary_agent as sa


class _FakeCursor:
    """Matches a SQL substring -> (columns, rows). Supports fetchone/fetchall."""

    def __init__(self, table_data, recorder=None):
        self._table_data = table_data
        self._recorder = recorder
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        if self._recorder is not None:
            self._recorder.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data, recorder=None):
        self._cur = _FakeCursor(table_data, recorder)
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def test_resolve_persona_hits_bp_prompt():
    conn = _FakeConn({
        "FROM proc.bp_prompt": (
            ["prompts_desc"],
            [({"prompt_template": "You are a compliance auditor."},)],
        ),
    })
    framing, source = sa.resolve_persona("compliance", conn)
    assert framing == "You are a compliance auditor."
    assert source == "bp_prompt"


def test_resolve_persona_falls_back_to_raw():
    conn = _FakeConn({"FROM proc.bp_prompt": (["prompts_desc"], [])})
    framing, source = sa.resolve_persona("some ad-hoc persona", conn)
    assert framing == "some ad-hoc persona"
    assert source == "raw"
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.summary_agent'`.

- [ ] **Step 3: Create the module with `resolve_persona`**

Create `src/services/summary_agent.py` with:

```python
"""Persona-driven summaries over the final (_trgt) procurement tables.

Mirrors ``deal_summary`` (direct SQL, cloud LLM, no local GPU). A persona is
resolved from the ``bp_prompt`` governance table (``prompt_type='summary_persona'``)
with a raw-string fallback. Results are cached and versioned in ``proc.bp_summary``.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.ollama_client import ollama_cloud_generate
from src.services.deal_summary import gather_deal_context, _build_prompt

log = logging.getLogger(__name__)

# Summaries run on the Ollama Cloud API (remote), keeping the local GPU free.
_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "gpt-oss:120b")


class SummarizationError(RuntimeError):
    """Raised when the LLM returns no usable summary."""


class SnapshotNotFound(RuntimeError):
    """Raised when an as_of request finds no snapshot at/before the datetime."""


def resolve_persona(persona: str, conn: Any) -> tuple[str, str]:
    """Return (framing_text, persona_source).

    Looks up ``persona`` in bp_prompt (prompt_type='summary_persona'). On a hit
    returns the stored template and 'bp_prompt'; on a miss returns the persona
    string itself and 'raw'.
    """
    cur = conn.cursor()
    row = None
    try:
        cur.execute(
            "SELECT prompts_desc FROM proc.bp_prompt "
            "WHERE prompt_type = 'summary_persona' AND prompt_name = %s "
            "AND COALESCE(prompts_status, 1) = 1 LIMIT 1",
            (persona,),
        )
        row = cur.fetchone()
    except Exception:  # pragma: no cover - defensive
        log.exception("persona lookup failed for %s", persona)
    if row and row[0]:
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        if isinstance(payload, dict):
            template = payload.get("prompt_template") or payload.get("template")
            if template:
                return str(template), "bp_prompt"
    return persona, "raw"
```

- [ ] **Step 4: Run the test, verify it PASSES**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): resolve_persona from bp_prompt with raw fallback"
```

---

## Task 2 NOTE for later tasks (shared test doubles)
`tests/test_summary_agent.py` now defines `_FakeCursor` / `_FakeConn` (supporting `execute`/`fetchone`/`fetchall`/`commit` and an optional `recorder`). Later tasks REUSE these — do not redefine them.

---

## Task 3: `gather_portfolio_context` — aggregate the target tables

**Files:**
- Modify: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

All three `_trgt` tables have `converted_amount_usd`, `supplier_id`, `currency` (verified). The aggregation summarizes rather than dumping rows.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_agent.py`:

```python
def _portfolio_conn():
    return _FakeConn({
        "count(*) FROM proc.bp_invoice_trgt": (["count"], [(2,)]),
        "count(*) FROM proc.bp_purchase_order_trgt": (["count"], [(1,)]),
        "count(*) FROM proc.bp_quote_trgt": (["count"], [(3,)]),
        "SUM(converted_amount_usd),0) FROM proc.bp_invoice_trgt": (["s"], [(1500.0,)]),
        "FROM proc.bp_purchase_order_trgt t": (["s"], [(800.0,)]),
        "GROUP BY supplier_id": (["supplier_id", "usd"], [("SUP-A", 1200.0), ("SUP-B", 300.0)]),
        "GROUP BY currency": (["currency", "n"], [("USD", 2)]),
        "FROM proc.bp_extraction_discrepancy": (["count"], [(4,)]),
        "FROM proc.bp_agent_actions": (["count"], [(7,)]),
    })


def test_gather_portfolio_context_aggregates():
    ctx = sa.gather_portfolio_context(_portfolio_conn())
    assert ctx is not None
    assert ctx["scope"] == "portfolio"
    assert ctx["totals"]["invoices"] == 2
    assert ctx["totals"]["quotes"] == 3
    assert ctx["totals"]["invoice_spend_usd"] == 1500.0
    assert ctx["top_suppliers"][0]["supplier_id"] == "SUP-A"
    assert ctx["sources"]["discrepancies"] == 4


def test_gather_portfolio_context_empty_returns_none():
    conn = _FakeConn({
        "count(*) FROM proc.bp_invoice_trgt": (["count"], [(0,)]),
        "count(*) FROM proc.bp_purchase_order_trgt": (["count"], [(0,)]),
        "count(*) FROM proc.bp_quote_trgt": (["count"], [(0,)]),
    })
    assert sa.gather_portfolio_context(conn) is None
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `.venv/bin/pytest tests/test_summary_agent.py::test_gather_portfolio_context_aggregates -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'gather_portfolio_context'`.

- [ ] **Step 3: Implement `gather_portfolio_context`**

Append to `src/services/summary_agent.py`:

```python
def gather_portfolio_context(conn: Any) -> Optional[dict]:
    """Aggregate the final (_trgt) tables into a compact portfolio fact dict.

    Returns None when all three document tables are empty.
    """
    cur = conn.cursor()

    def _scalar(sql: str) -> Any:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else None

    inv = _scalar("SELECT count(*) FROM proc.bp_invoice_trgt") or 0
    pos = _scalar("SELECT count(*) FROM proc.bp_purchase_order_trgt") or 0
    quotes = _scalar("SELECT count(*) FROM proc.bp_quote_trgt") or 0
    if (inv + pos + quotes) == 0:
        return None

    inv_usd = _scalar(
        "SELECT COALESCE(SUM(converted_amount_usd),0) FROM proc.bp_invoice_trgt"
    ) or 0
    po_usd = _scalar(
        "SELECT COALESCE(SUM(converted_amount_usd),0) FROM proc.bp_purchase_order_trgt t"
    ) or 0

    cur.execute(
        "SELECT supplier_id, COALESCE(SUM(converted_amount_usd),0) AS usd "
        "FROM proc.bp_invoice_trgt WHERE supplier_id IS NOT NULL "
        "GROUP BY supplier_id ORDER BY usd DESC LIMIT 10"
    )
    top_suppliers = [
        {"supplier_id": s, "invoice_usd": float(u or 0)} for s, u in cur.fetchall()
    ]

    cur.execute("SELECT currency, count(*) FROM proc.bp_invoice_trgt GROUP BY currency")
    currency_mix = {str(c): int(n) for c, n in cur.fetchall() if c is not None}

    disc = _scalar("SELECT count(*) FROM proc.bp_extraction_discrepancy") or 0
    actions = _scalar("SELECT count(*) FROM proc.bp_agent_actions") or 0

    return {
        "scope": "portfolio",
        "totals": {
            "invoices": int(inv),
            "purchase_orders": int(pos),
            "quotes": int(quotes),
            "invoice_spend_usd": float(inv_usd or 0),
            "po_value_usd": float(po_usd or 0),
        },
        "top_suppliers": top_suppliers,
        "currency_mix": currency_mix,
        "sources": {
            "invoices": int(inv),
            "purchase_orders": int(pos),
            "quotes": int(quotes),
            "discrepancies": int(disc),
            "actions": int(actions),
        },
    }
```

- [ ] **Step 4: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): portfolio aggregation over target tables"
```

---

## Task 4: `_build_persona_prompt` — persona framing + grounded rules

**Files:**
- Modify: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_agent.py`:

```python
def test_build_persona_prompt_includes_framing_rules_and_facts():
    facts = {"scope": "portfolio", "totals": {"invoices": 2}}
    prompt = sa._build_persona_prompt("You are a compliance auditor.", facts)
    assert "You are a compliance auditor." in prompt
    # grounded no-fabrication rule reused from deal_summary._build_prompt
    assert "Do not fabricate" in prompt
    # the facts are embedded as JSON
    assert '"invoices": 2' in prompt
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `.venv/bin/pytest tests/test_summary_agent.py::test_build_persona_prompt_includes_framing_rules_and_facts -v`
Expected: FAIL — no attribute `_build_persona_prompt`.

- [ ] **Step 3: Implement `_build_persona_prompt`**

Append to `src/services/summary_agent.py`:

```python
def _build_persona_prompt(framing: str, facts: dict) -> str:
    """Persona framing + the grounded base rules + the fact JSON.

    Reuses ``deal_summary._build_prompt`` for the no-fabrication base so the
    grounding rules stay identical across both summary paths.
    """
    base = _build_prompt(facts)
    return f"{framing.strip()}\n\n{base}"
```

Note: `deal_summary._build_prompt` already emits "Use ONLY the data provided. Do not fabricate ..." and embeds the facts JSON, so the asserted substrings are present.

- [ ] **Step 4: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): persona-framed prompt builder"
```

---

## Task 5: `_store_summary` — insert row + flip is_current

**Files:**
- Modify: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_agent.py`:

```python
def test_store_summary_flips_current_and_inserts():
    rec = []
    conn = _FakeConn({}, recorder=rec)
    out = sa._store_summary(
        conn,
        persona="compliance",
        persona_source="bp_prompt",
        scope="deal",
        deal_id="D-9",
        summary="text",
        data_snapshot={"a": 1},
        sources={"invoices": 1},
        model="gpt-oss:120b",
        is_current=True,
    )
    assert out["summary_id"]
    assert out["persona"] == "compliance"
    assert out["deal_id"] == "D-9"
    assert "generated_at" in out
    assert conn.committed is True
    sqls = " ".join(s for s, _ in rec)
    assert "UPDATE proc.bp_summary SET is_current = false" in sqls
    assert "INSERT INTO proc.bp_summary" in sqls


def test_store_summary_as_of_does_not_flip_current():
    rec = []
    conn = _FakeConn({}, recorder=rec)
    sa._store_summary(
        conn, persona="compliance", persona_source="raw", scope="deal",
        deal_id="D-9", summary="t", data_snapshot={}, sources=None,
        model="m", is_current=False,
    )
    sqls = " ".join(s for s, _ in rec)
    assert "UPDATE proc.bp_summary SET is_current = false" not in sqls
    assert "INSERT INTO proc.bp_summary" in sqls
```

- [ ] **Step 2: Run the tests, verify they FAIL**

Run: `.venv/bin/pytest tests/test_summary_agent.py::test_store_summary_flips_current_and_inserts -v`
Expected: FAIL — no attribute `_store_summary`.

- [ ] **Step 3: Implement `_store_summary`**

Append to `src/services/summary_agent.py`:

```python
def _store_summary(
    conn: Any,
    *,
    persona: str,
    persona_source: str,
    scope: str,
    deal_id: Optional[str],
    summary: str,
    data_snapshot: Any,
    sources: Any,
    model: str,
    is_current: bool = True,
) -> dict:
    """Insert a summary row. When is_current, demote the prior current row of the
    same (persona, scope, deal_id) group first. summary_id/generated_at are set
    in Python so the result is returned without RETURNING parsing.
    """
    sid = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    cur = conn.cursor()
    if is_current:
        cur.execute(
            "UPDATE proc.bp_summary SET is_current = false "
            "WHERE persona = %s AND scope = %s "
            "AND deal_id IS NOT DISTINCT FROM %s AND is_current",
            (persona, scope, deal_id),
        )
    cur.execute(
        "INSERT INTO proc.bp_summary "
        "(summary_id, persona, persona_source, scope, deal_id, summary, "
        " data_snapshot, sources, model, is_current, generated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            sid, persona, persona_source, scope, deal_id, summary,
            json.dumps(data_snapshot, default=str),
            json.dumps(sources, default=str) if sources is not None else None,
            model, is_current, generated_at,
        ),
    )
    conn.commit()
    return {
        "summary_id": sid,
        "persona": persona,
        "persona_source": persona_source,
        "scope": scope,
        "deal_id": deal_id,
        "summary": summary,
        "sources": sources,
        "generated_at": generated_at.isoformat(),
    }
```

- [ ] **Step 4: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): persist summaries with is_current versioning"
```

---

## Task 6: `generate_summary` — orchestrate deal / portfolio / as_of

**Files:**
- Modify: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_agent.py`:

```python
def test_generate_summary_portfolio(monkeypatch):
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "PORTFOLIO SUMMARY")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_portfolio_context", lambda conn: {"scope": "portfolio", "sources": {"invoices": 2}})
    rec = []
    conn = _FakeConn({}, recorder=rec)
    out = sa.generate_summary("analysis", deal_id=None, conn=conn)
    assert out["summary"] == "PORTFOLIO SUMMARY"
    assert out["scope"] == "portfolio"
    assert out["deal_id"] is None
    assert "INSERT INTO proc.bp_summary" in " ".join(s for s, _ in rec)


def test_generate_summary_deal(monkeypatch):
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "DEAL SUMMARY")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "raw"))
    monkeypatch.setattr(sa, "gather_deal_context", lambda deal_id, conn=None: {"deal_id": deal_id, "sources": {"invoices": 1}})
    conn = _FakeConn({})
    out = sa.generate_summary("compliance", deal_id="D-9", conn=conn)
    assert out["scope"] == "deal"
    assert out["deal_id"] == "D-9"
    assert out["persona_source"] == "raw"


def test_generate_summary_no_data_returns_none(monkeypatch):
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_portfolio_context", lambda conn: None)
    assert sa.generate_summary("analysis", conn=_FakeConn({})) is None


def test_generate_summary_empty_llm_raises(monkeypatch):
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_deal_context", lambda deal_id, conn=None: {"deal_id": deal_id, "sources": {}})
    import pytest
    with pytest.raises(sa.SummarizationError):
        sa.generate_summary("analysis", deal_id="D-9", conn=_FakeConn({}))


def test_generate_summary_as_of_uses_snapshot(monkeypatch):
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "HISTORICAL")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    conn = _FakeConn({
        "SELECT data_snapshot FROM proc.bp_summary": (["data_snapshot"], [({"scope": "deal", "old": True},)]),
    })
    out = sa.generate_summary("analysis", deal_id="D-9", as_of="2026-05-01T00:00:00Z", conn=conn)
    assert out["summary"] == "HISTORICAL"


def test_generate_summary_as_of_missing_snapshot_raises(monkeypatch):
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    conn = _FakeConn({"SELECT data_snapshot FROM proc.bp_summary": (["data_snapshot"], [])})
    import pytest
    with pytest.raises(sa.SnapshotNotFound):
        sa.generate_summary("analysis", deal_id="D-9", as_of="2020-01-01T00:00:00Z", conn=conn)
```

- [ ] **Step 2: Run the tests, verify they FAIL**

Run: `.venv/bin/pytest tests/test_summary_agent.py -k generate_summary -v`
Expected: FAIL — no attribute `generate_summary`.

- [ ] **Step 3: Implement `generate_summary`**

Append to `src/services/summary_agent.py`:

```python
def generate_summary(
    persona: str,
    deal_id: Optional[str] = None,
    as_of: Optional[str] = None,
    conn: Any = None,
) -> Optional[dict]:
    """Generate (and persist) a persona summary.

    deal_id present -> per-deal scope; absent -> portfolio. When as_of is set,
    regenerate over the nearest stored snapshot at/before that datetime (the
    result is stored as a historical, non-current row). Returns None when there
    is no underlying data; raises SnapshotNotFound / SummarizationError.
    """
    if conn is None:
        with get_conn() as own:
            return generate_summary(persona, deal_id, as_of, conn=own)

    scope = "deal" if deal_id else "portfolio"
    framing, persona_source = resolve_persona(persona, conn)

    if as_of is not None:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_snapshot FROM proc.bp_summary "
            "WHERE scope = %s AND deal_id IS NOT DISTINCT FROM %s "
            "AND generated_at <= %s ORDER BY generated_at DESC LIMIT 1",
            (scope, deal_id, as_of),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            raise SnapshotNotFound(
                f"no snapshot at/before {as_of} for scope={scope} deal_id={deal_id}"
            )
        facts = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        is_current = False
    else:
        facts = (
            gather_deal_context(deal_id, conn=conn)
            if deal_id
            else gather_portfolio_context(conn)
        )
        if facts is None:
            return None
        is_current = True

    text = ollama_cloud_generate(
        _build_persona_prompt(framing, facts),
        model=_SUMMARY_MODEL,
        temperature=0.0,
        num_predict=1024,
        timeout=120,
        retries=2,
    )
    if not text or not text.strip():
        raise SummarizationError(
            f"empty summary for persona={persona} deal_id={deal_id}"
        )

    sources = facts.get("sources") if isinstance(facts, dict) else None
    return _store_summary(
        conn,
        persona=persona,
        persona_source=persona_source,
        scope=scope,
        deal_id=deal_id,
        summary=text.strip(),
        data_snapshot=facts,
        sources=sources,
        model=_SUMMARY_MODEL,
        is_current=is_current,
    )
```

- [ ] **Step 4: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): generate_summary for deal/portfolio/as_of"
```

---

## Task 7: `precompute_summaries` and cache read/history helpers

**Files:**
- Modify: `src/services/summary_agent.py`
- Test: `tests/test_summary_agent.py`

Adds the precompute driver plus the read helpers the GET endpoints need:
`get_cached_summary`, `list_summary_history`, `get_summary_by_id`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_agent.py`:

```python
def test_precompute_iterates_personas_and_scopes(monkeypatch):
    calls = []
    def fake_generate(persona, deal_id=None, as_of=None, conn=None):
        calls.append((persona, deal_id))
        return {"summary_id": "x"}
    monkeypatch.setattr(sa, "generate_summary", fake_generate)
    # personas from bp_prompt; distinct deal_ids from target tables
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",), ("compliance",)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [("D-1",), ("D-2",)]),
    })
    out = sa.precompute_summaries(conn=conn)
    # 2 personas x (portfolio + 2 deals) = 6
    assert out["generated"] == 6
    assert ("analysis", None) in calls
    assert ("compliance", "D-2") in calls


def test_precompute_continues_past_failures(monkeypatch):
    def fake_generate(persona, deal_id=None, as_of=None, conn=None):
        if deal_id == "D-1":
            raise RuntimeError("boom")
        return {"summary_id": "x"}
    monkeypatch.setattr(sa, "generate_summary", fake_generate)
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [("D-1",)]),
    })
    out = sa.precompute_summaries(conn=conn)
    assert out["failed"] == 1
    assert out["generated"] == 1  # portfolio succeeded


def test_get_cached_summary_returns_current_row():
    conn = _FakeConn({
        "FROM proc.bp_summary": (
            ["summary_id", "persona", "persona_source", "scope", "deal_id", "summary", "sources", "generated_at"],
            [("sid-1", "compliance", "bp_prompt", "deal", "D-9", "cached text", {"invoices": 1}, "2026-06-08T00:00:00Z")],
        ),
    })
    out = sa.get_cached_summary("compliance", "D-9", conn=conn)
    assert out["summary_id"] == "sid-1"
    assert out["summary"] == "cached text"


def test_get_cached_summary_none_when_absent():
    conn = _FakeConn({"FROM proc.bp_summary": (["summary_id"], [])})
    assert sa.get_cached_summary("compliance", "D-9", conn=conn) is None
```

- [ ] **Step 2: Run the tests, verify they FAIL**

Run: `.venv/bin/pytest tests/test_summary_agent.py -k "precompute or cached" -v`
Expected: FAIL — missing attributes.

- [ ] **Step 3: Implement the helpers**

Append to `src/services/summary_agent.py`:

```python
def _rows_as_dicts(cur) -> list[dict]:
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_cached_summary(persona: str, deal_id: Optional[str] = None, conn: Any = None) -> Optional[dict]:
    """Return the current cached summary for (persona, deal_id), or None."""
    if conn is None:
        with get_conn() as own:
            return get_cached_summary(persona, deal_id, conn=own)
    scope = "deal" if deal_id else "portfolio"
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, persona, persona_source, scope, deal_id, summary, "
        "sources, generated_at FROM proc.bp_summary "
        "WHERE persona = %s AND scope = %s AND deal_id IS NOT DISTINCT FROM %s "
        "AND is_current ORDER BY generated_at DESC LIMIT 1",
        (persona, scope, deal_id),
    )
    rows = _rows_as_dicts(cur)
    return rows[0] if rows else None


def list_summary_history(persona: str, deal_id: Optional[str] = None, conn: Any = None) -> list[dict]:
    """Return prior summaries for (persona, deal_id), newest first."""
    if conn is None:
        with get_conn() as own:
            return list_summary_history(persona, deal_id, conn=own)
    scope = "deal" if deal_id else "portfolio"
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, scope, deal_id, generated_at, is_current, "
        "left(summary, 200) AS snippet FROM proc.bp_summary "
        "WHERE persona = %s AND scope = %s AND deal_id IS NOT DISTINCT FROM %s "
        "ORDER BY generated_at DESC",
        (persona, scope, deal_id),
    )
    return _rows_as_dicts(cur)


def get_summary_by_id(summary_id: str, conn: Any = None) -> Optional[dict]:
    """Return a single stored summary by id, or None."""
    if conn is None:
        with get_conn() as own:
            return get_summary_by_id(summary_id, conn=own)
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, persona, persona_source, scope, deal_id, summary, "
        "sources, model, is_current, generated_at FROM proc.bp_summary "
        "WHERE summary_id = %s",
        (summary_id,),
    )
    rows = _rows_as_dicts(cur)
    return rows[0] if rows else None


def _distinct_deal_ids(conn: Any) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT deal_id FROM ("
        " SELECT deal_id FROM proc.bp_invoice_trgt "
        " UNION SELECT deal_id FROM proc.bp_purchase_order_trgt "
        " UNION SELECT deal_id FROM proc.bp_quote_trgt) t "
        "WHERE deal_id IS NOT NULL"
    )
    return [r[0] for r in cur.fetchall()]


def _summary_personas(conn: Any) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT prompt_name FROM proc.bp_prompt "
        "WHERE prompt_type = 'summary_persona' AND COALESCE(prompts_status,1)=1"
    )
    return [r[0] for r in cur.fetchall()]


def precompute_summaries(
    personas: Optional[list[str]] = None,
    deal_ids: Optional[list[str]] = None,
    conn: Any = None,
) -> dict:
    """Generate current summaries for every persona x scope (portfolio + each
    deal). Per-item failures are logged and skipped. Returns counts.
    """
    if conn is None:
        with get_conn() as own:
            return precompute_summaries(personas, deal_ids, conn=own)

    personas = personas or _summary_personas(conn)
    deal_ids = deal_ids if deal_ids is not None else _distinct_deal_ids(conn)
    scopes: list[Optional[str]] = [None] + list(deal_ids)  # None = portfolio
    planned = len(personas) * len(scopes)
    log.info(
        "summary precompute: %d personas x %d scopes = %d generations",
        len(personas), len(scopes), planned,
    )

    generated = 0
    failed = 0
    for persona in personas:
        for deal_id in scopes:
            try:
                generate_summary(persona, deal_id=deal_id, conn=conn)
                generated += 1
            except Exception:  # pragma: no cover - logged, run continues
                failed += 1
                log.exception(
                    "precompute failed for persona=%s deal_id=%s", persona, deal_id
                )
    return {
        "generated": generated,
        "failed": failed,
        "personas": len(personas),
        "scopes": len(scopes),
    }
```

- [ ] **Step 4: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/summary_agent.py tests/test_summary_agent.py
git commit -m "feat(summary): precompute driver + cache/history read helpers"
```

---

## Task 8: Endpoints router

**Files:**
- Create: `src/api/routers/summary.py`
- Modify: `src/api/main.py` (import + include_router, lines 47 and ~271)
- Test: `tests/test_summary_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary_api.py` with:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import summary as summary_router
import src.services.summary_agent as sa


def _client():
    app = FastAPI()
    app.include_router(summary_router.router)
    return TestClient(app)


def test_post_summary_generates(monkeypatch):
    monkeypatch.setattr(
        sa, "generate_summary",
        lambda persona, deal_id=None, as_of=None: {
            "summary_id": "sid-1", "persona": persona, "scope": "portfolio",
            "deal_id": None, "summary": "S", "sources": {}, "generated_at": "t",
        },
    )
    resp = _client().post("/summary", json={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["summary_id"] == "sid-1"


def test_post_summary_404_when_no_data(monkeypatch):
    monkeypatch.setattr(sa, "generate_summary", lambda persona, deal_id=None, as_of=None: None)
    resp = _client().post("/summary", json={"persona": "analysis", "deal_id": "NOPE"})
    assert resp.status_code == 404


def test_get_summary_returns_cache(monkeypatch):
    monkeypatch.setattr(
        sa, "get_cached_summary",
        lambda persona, deal_id=None: {"summary_id": "sid-1", "summary": "cached"},
    )
    resp = _client().get("/summary", params={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["summary"] == "cached"


def test_get_summary_404_when_uncached(monkeypatch):
    monkeypatch.setattr(sa, "get_cached_summary", lambda persona, deal_id=None: None)
    resp = _client().get("/summary", params={"persona": "analysis"})
    assert resp.status_code == 404


def test_get_history(monkeypatch):
    monkeypatch.setattr(sa, "list_summary_history", lambda persona, deal_id=None: [{"summary_id": "a"}])
    resp = _client().get("/summary/history", params={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["history"] == [{"summary_id": "a"}]


def test_post_precompute(monkeypatch):
    monkeypatch.setattr(sa, "precompute_summaries", lambda personas=None, deal_ids=None: {"generated": 4, "failed": 0})
    resp = _client().post("/summary/precompute", json={})
    assert resp.status_code == 200
    assert resp.json()["generated"] == 4
```

- [ ] **Step 2: Run the tests, verify they FAIL**

Run: `.venv/bin/pytest tests/test_summary_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.routers.summary'`.

- [ ] **Step 3: Create the router**

Create `src/api/routers/summary.py` with:

```python
"""Persona-driven summaries over the procurement target tables.

POST /summary            — regenerate + store a summary (on-demand refresh).
GET  /summary            — return the latest cached summary (fast).
GET  /summary/history    — list prior summaries for a persona/scope.
GET  /summary/{id}       — fetch one stored summary.
POST /summary/precompute — warm the cache for personas x scopes.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import src.services.summary_agent as summary_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["Summary"])


class SummaryRequest(BaseModel):
    persona: str
    deal_id: Optional[str] = None
    as_of: Optional[str] = None


class PrecomputeRequest(BaseModel):
    personas: Optional[list[str]] = None
    deal_ids: Optional[list[str]] = None


@router.post("", summary="Generate (refresh) a persona summary")
def post_summary(req: SummaryRequest) -> dict[str, Any]:
    try:
        result = summary_agent.generate_summary(
            req.persona, deal_id=req.deal_id, as_of=req.as_of
        )
    except summary_agent.SnapshotNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except summary_agent.SummarizationError as exc:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}")
    except Exception as exc:  # DB or unexpected
        logger.exception("summary generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for persona={req.persona} deal_id={req.deal_id}",
        )
    return result


@router.get("", summary="Latest cached persona summary")
def get_summary(persona: str, deal_id: Optional[str] = None) -> dict[str, Any]:
    try:
        result = summary_agent.get_cached_summary(persona, deal_id)
    except Exception as exc:
        logger.exception("cached summary read failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached summary for persona={persona} deal_id={deal_id}; POST to generate.",
        )
    return result


@router.get("/history", summary="Historical summaries for a persona/scope")
def get_history(persona: str, deal_id: Optional[str] = None) -> dict[str, Any]:
    try:
        return {"history": summary_agent.list_summary_history(persona, deal_id)}
    except Exception as exc:
        logger.exception("summary history read failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/precompute", summary="Warm the summary cache")
def post_precompute(req: PrecomputeRequest) -> dict[str, Any]:
    try:
        return summary_agent.precompute_summaries(
            personas=req.personas, deal_ids=req.deal_ids
        )
    except Exception as exc:
        logger.exception("summary precompute failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{summary_id}", summary="Fetch one stored summary by id")
def get_one(summary_id: str) -> dict[str, Any]:
    try:
        result = summary_agent.get_summary_by_id(summary_id)
    except Exception as exc:
        logger.exception("summary fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No summary {summary_id}")
    return result
```

Note: the `/precompute` and `/history` routes are declared BEFORE `/{summary_id}` so the static paths win over the path param.

- [ ] **Step 4: Register the router in `api/main.py`**

In `src/api/main.py` line 47, add `summary` to the existing import:
```python
from api.routers import agents as agents_router_mod, documents, email, metrics, run, stream, system, training, vendors, workflows, deal_summary, promotion, summary
```
After the existing `app.include_router(deal_summary.router)` line (~271), add:
```python
app.include_router(summary.router)
```

- [ ] **Step 5: Run the tests, verify they PASS**

Run: `.venv/bin/pytest tests/test_summary_api.py -v`
Expected: all PASS.

Also confirm the app still imports cleanly:
```bash
.venv/bin/python -c "import ast; ast.parse(open('src/api/main.py').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/summary.py src/api/main.py tests/test_summary_api.py
git commit -m "feat(summary): /summary endpoints (refresh, cache, history, precompute)"
```

---

## Task 9: Scheduled daily precompute + settings

**Files:**
- Modify: `config/settings.py`
- Modify: `src/services/backend_scheduler.py`

- [ ] **Step 1: Add settings fields**

In `config/settings.py`, add two fields to the settings model (place them near other feature flags such as `enable_training_scheduler`; match the file's existing field style):
```python
    enable_summary_precompute: bool = True
    summary_precompute_interval_hours: int = 24
```

- [ ] **Step 2: Add the scheduler job (no unit test — exercised live in Task 10)**

In `src/services/backend_scheduler.py`, find `_register_default_jobs` (around line 259) and add a call to a new registration method. Add to `_register_default_jobs`:
```python
        self._register_summary_precompute_job()
```
Then add these two methods to the `BackendScheduler` class (mirror the existing `_register_kg_sync_job` / `_run_kg_sync` pattern; `timedelta`, `logger`, and `settings` are already imported in this file):
```python
    def _register_summary_precompute_job(self) -> None:
        """Register the daily persona-summary precompute job."""
        if not bool(getattr(settings, "enable_summary_precompute", True)):
            logger.info("Summary precompute disabled; skipping job registration")
            return
        hours = int(getattr(settings, "summary_precompute_interval_hours", 24))
        self.register_job(
            name="summary-precompute",
            callback=self._run_summary_precompute,
            interval=timedelta(hours=hours),
        )

    def _run_summary_precompute(self) -> None:
        try:
            from src.services.summary_agent import precompute_summaries
            counts = precompute_summaries()
            logger.info("summary precompute completed: %s", counts)
        except Exception:
            logger.exception("summary precompute job failed")
```
IMPORTANT: check the actual signature of `register_job` in this file (around line 219) and match its parameter names exactly (it takes `name`, `callback`, `interval`). If `register_job` requires additional positional/keyword args, supply them to match the other `_register_*_job` call sites.

- [ ] **Step 3: Verify imports/syntax**

Run:
```bash
.venv/bin/python -c "import ast; [ast.parse(open(f).read()) for f in ['config/settings.py','src/services/backend_scheduler.py']]; print('SYNTAX_OK')"
```
Expected: `SYNTAX_OK`.

- [ ] **Step 4: Commit**

```bash
git add config/settings.py src/services/backend_scheduler.py
git commit -m "feat(summary): daily scheduled precompute job + settings"
```

---

## Task 10: Live verification

**Files:** none (operational).

- [ ] **Step 1: Run the full new test suite**

Run:
```bash
.venv/bin/pytest tests/test_summary_agent.py tests/test_summary_api.py tests/test_deal_summary_api.py -q
```
Expected: all PASS.

- [ ] **Step 2: Restart procwise**

Run: `sudo -n /usr/bin/systemctl restart procwise && echo RESTART_OK && sleep 22 && sudo -n /usr/bin/systemctl is-active procwise`
Expected: `RESTART_OK` then `active`.

- [ ] **Step 3: Generate a per-deal summary live**

Find a real deal_id and generate:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env 2>/dev/null && set +a
DEAL=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT deal_id FROM proc.bp_invoice_trgt WHERE deal_id IS NOT NULL LIMIT 1")
echo "deal=$DEAL"
curl -s -m 180 -X POST http://localhost:8000/summary -H "Content-Type: application/json" -d "{\"persona\":\"compliance\",\"deal_id\":\"$DEAL\"}" | head -c 600; echo
```
Expected: JSON with a `summary_id`, `persona":"compliance"`, `persona_source":"bp_prompt"`, and a non-empty `summary`.

- [ ] **Step 4: Confirm cache read + persistence**

```bash
curl -s "http://localhost:8000/summary?persona=compliance&deal_id=$DEAL" | head -c 400; echo
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT persona, scope, is_current FROM proc.bp_summary WHERE deal_id = '$DEAL';"
```
Expected: GET returns the cached row (same `summary_id`); the DB shows a row with `is_current = t`.

- [ ] **Step 5: Generate a portfolio summary + precompute one persona**

```bash
curl -s -m 180 -X POST http://localhost:8000/summary -H "Content-Type: application/json" -d '{"persona":"analysis"}' | head -c 400; echo
curl -s -m 600 -X POST http://localhost:8000/summary/precompute -H "Content-Type: application/json" -d '{"personas":["analysis"]}' | head -c 300; echo
```
Expected: portfolio summary with a `summary_id` and `scope":"portfolio"`; precompute returns `{"generated": N, "failed": 0, ...}` with N ≥ 1.

- [ ] **Step 6: Confirm no errors in the log**

```bash
sudo -n /usr/bin/journalctl -u procwise --no-pager --since "3 minutes ago" | grep -iE "summary|Traceback|ERROR" | grep -iv "PolicyEngine loaded" | head -20
```
Expected: a `summary precompute` info line (if the scheduler ran) and no tracebacks tied to the summary endpoints.

---

## Self-Review Notes

- **Spec coverage:** `bp_summary` schema + persona seed (T1); persona resolve (T2); portfolio gather (T3); prompt build reusing grounded rules (T4); persist + is_current versioning (T5); generate for deal/portfolio/as_of (T6); precompute + cache/history reads (T7); endpoints incl. precompute (T8); scheduled daily precompute + settings (T9); live verification (T10). All spec sections mapped.
- **Type consistency:** service functions return dicts with `summary_id`/`persona`/`persona_source`/`scope`/`deal_id`/`summary`/`sources`/`generated_at`; endpoints and tests use those exact keys. `generate_summary(persona, deal_id, as_of, conn)`, `get_cached_summary(persona, deal_id, conn)`, `list_summary_history(persona, deal_id, conn)`, `get_summary_by_id(summary_id, conn)`, `precompute_summaries(personas, deal_ids, conn)` signatures are consistent across service, router, and tests. Reused symbols (`gather_deal_context`, `_build_prompt`, `ollama_cloud_generate`, `get_conn`) verified to exist with the used signatures.
- **No placeholders:** every SQL statement, function body, and test is concrete; all table/column names verified against the live DB.
