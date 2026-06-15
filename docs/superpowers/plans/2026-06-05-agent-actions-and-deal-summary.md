# Agent Actions Event Log + Deal Summarization API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fine-grained `proc.agent_actions` event log written by the existing extraction/validation pipeline, plus an AI API that summarizes a procurement deal (by `deal_id`) into clear text.

**Architecture:** A best-effort writer (`record_action`) that never breaks extraction logs one row per action/step into `proc.agent_actions`. The existing `dispatch_document()` and `write_discrepancies()` call it. A read-only gatherer assembles a deal's final (`_trgt`) records + line items + action trail + discrepancies; a grounded LLM summarizer turns that into text behind a FastAPI `GET /deals/{deal_id}/summary` endpoint.

**Tech Stack:** Python 3, psycopg2, FastAPI, Ollama (`qwen2.5:7b` for summarization), pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-agent-actions-and-deal-summary-design.md`

**Conventions verified in repo:**
- DB access: `from src.services.db import get_conn`; psycopg2; `conn.autocommit=False` + explicit `commit`/`rollback`; schema `proc`.
- Under pytest, `get_conn()` returns an in-memory `_FakeConnection` (see `src/services/db.py:975`) — DB writes are no-ops in tests unless you inject your own conn.
- Migrations: plain SQL in `deploy/sql/YYYY-MM-DD_*.sql`, `CREATE TABLE IF NOT EXISTS`, `bigserial` PK, `timestamptz DEFAULT now()`, `jsonb`.
- Final tables are `_trgt` (`bp_invoice_trgt`, `bp_purchase_order_trgt`, `bp_quote_trgt`, `bp_contracts`) keyed by `deal_id` (+`deal_name`, `document_id`).
- FastAPI routers live in `src/api/routers/`, registered with `app.include_router(...)` in `src/api/main.py`.

> **Note on LLM choice:** The spec mentioned `llm_router` for summarization. `LLMRouter` is currently unused anywhere in the codebase and its "cloud" tier requires a separately-configured Ollama endpoint. To stay robust and simple (Avoid-Overengineering), this plan calls the proven `ollama_generate()` helper with model `qwen2.5:7b` — a general model, deliberately NOT the extraction adapter `AgentNick`, so the GPU/adapter stays free for extraction. Same intent, fewer moving parts. Swappable to `llm_router` later behind the `_SUMMARY_MODEL` constant.

---

## File Structure

- Create `deploy/sql/2026-06-05_agent_actions.sql` — the table + indexes.
- Create `src/services/agent_actions.py` — writer (`record_action`, `bulk_record`) + canonical vocab constants. One responsibility: persist action rows, best-effort.
- Modify `src/services/extraction/dispatch.py` — emit `extraction` actions at phase boundaries.
- Modify `src/services/extraction/persistence.py` — emit one `validation` action per discrepancy.
- Create `src/services/deal_summary.py` — `gather_deal_context()` (read-only assembly) + `summarize_deal()` + `_build_prompt()`. Pure-data gather kept separate from LLM call.
- Create `src/api/routers/deal_summary.py` — `GET /deals/{deal_id}/summary`.
- Modify `src/api/main.py` — register the router.
- Create `tests/services/test_agent_actions.py`, `tests/services/test_deal_summary.py`, `tests/test_deal_summary_api.py`.

---

## Task 1: Create the `agent_actions` migration and apply it

**Files:**
- Create: `deploy/sql/2026-06-05_agent_actions.sql`

- [ ] **Step 1: Write the migration SQL**

Create `deploy/sql/2026-06-05_agent_actions.sql`:

```sql
-- Fine-grained, append-only agent action event log.
-- One row per action/step across extraction / validation / consolidation phases.
CREATE TABLE IF NOT EXISTS proc.agent_actions (
    action_id          bigserial PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    deal_id            text,
    document_id        text,
    doc_pk             text,
    doc_type           text,
    process_monitor_id integer,
    trace_id           text,
    phase              text NOT NULL,
    action_type        text NOT NULL,
    agent              text,
    field_name         text,
    status             text,
    summary            text,
    details            jsonb,
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

- [ ] **Step 2: Apply the migration to the live DB**

Run (uses the same env the app uses — `PGHOST`/`PGDATABASE`/`PGUSER`/`PGPASSWORD`/`PGPORT`):

```bash
python3 -c "
from src.services.db import get_conn
sql = open('deploy/sql/2026-06-05_agent_actions.sql').read()
with get_conn() as conn:
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
print('applied')
"
```

Expected output: `applied`

- [ ] **Step 3: Verify the table exists**

Run:

```bash
python3 -c "
from src.services.db import get_conn
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute(\"select column_name from information_schema.columns where table_schema='proc' and table_name='agent_actions' order by ordinal_position\")
    print([r[0] for r in cur.fetchall()])
"
```

Expected: a list containing `action_id, created_at, deal_id, ... pipeline_version` (17 columns).

- [ ] **Step 4: Commit**

```bash
git add deploy/sql/2026-06-05_agent_actions.sql
git commit -m "feat(db): add proc.agent_actions event log table"
```

---

## Task 2: Implement the `record_action` / `bulk_record` writer

**Files:**
- Create: `src/services/agent_actions.py`
- Test: `tests/services/test_agent_actions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_agent_actions.py`:

```python
import src.services.agent_actions as aa


class _RecordingCursor:
    def __init__(self):
        self.executed = []
        self.many = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def executemany(self, sql, params):
        self.many.append((sql, list(params)))


class _RecordingConn:
    def __init__(self):
        self._cur = _RecordingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_record_action_uses_injected_conn_and_does_not_commit():
    conn = _RecordingConn()
    aa.record_action(
        phase=aa.PHASE_EXTRACTION,
        action_type="regex_extract",
        doc_type="invoice",
        doc_pk="INV-1",
        details={"n": 3},
        conn=conn,
    )
    assert len(conn._cur.executed) == 1
    sql, params = conn._cur.executed[0]
    assert "INSERT INTO proc.agent_actions" in sql
    # details serialized to JSON text
    assert '"n": 3' in params[12]
    # caller owns the transaction: writer must NOT commit an injected conn
    assert conn.committed is False


def test_record_action_swallows_db_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(aa, "get_conn", boom)
    # Must not raise — best-effort logging cannot break the pipeline.
    aa.record_action(phase=aa.PHASE_EXTRACTION, action_type="persist")


def test_record_action_swallows_bad_phase_missing():
    # Missing required phase/action_type must be swallowed, not raised.
    aa.record_action(phase=None, action_type=None)  # type: ignore[arg-type]


def test_bulk_record_uses_executemany_on_injected_conn():
    conn = _RecordingConn()
    aa.bulk_record(
        [
            {"phase": aa.PHASE_VALIDATION, "action_type": "discrepancy", "field_name": "tax_amount"},
            {"phase": aa.PHASE_VALIDATION, "action_type": "discrepancy", "field_name": "total"},
        ],
        conn=conn,
    )
    assert len(conn._cur.many) == 1
    sql, rows = conn._cur.many[0]
    assert "INSERT INTO proc.agent_actions" in sql
    assert len(rows) == 2


def test_bulk_record_empty_is_noop():
    conn = _RecordingConn()
    aa.bulk_record([], conn=conn)
    assert conn._cur.many == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_agent_actions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.agent_actions'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/agent_actions.py`:

```python
"""Best-effort writer for the proc.agent_actions event log.

One row per action/step. Writes are best-effort: any failure is logged and
swallowed so a logging problem can never break extraction. Callers may pass an
existing ``conn`` to fold the write into their own transaction (the caller then
owns commit/rollback); otherwise a short autonomous transaction is used.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping

from src.services.db import get_conn

log = logging.getLogger(__name__)

# Canonical phase values (stored as free text in the DB).
PHASE_EXTRACTION = "extraction"
PHASE_VALIDATION = "validation"
PHASE_CONSOLIDATION = "consolidation"

# Canonical action_type values (documentation; DB column is free text).
#   extraction:  parse, regex_extract, engineered_extract, ner_gapfill, judge,
#                context_synthesize, line_recovery, grounding_gate, persist
#   validation:  discrepancy
#   consolidation: reconcile_match, reconcile_mismatch  (writer slot only)

_COLUMNS = (
    "deal_id", "document_id", "doc_pk", "doc_type", "process_monitor_id",
    "trace_id", "phase", "action_type", "agent", "field_name", "status",
    "summary", "details", "confidence", "pipeline_version",
)

_INSERT = (
    "INSERT INTO proc.agent_actions ("
    + ", ".join(_COLUMNS)
    + ") VALUES (" + ", ".join(["%s"] * len(_COLUMNS)) + ")"
)


def _as_text(value: Any) -> Any:
    return str(value) if value is not None else None


def _row_params(fields: Mapping[str, Any]) -> tuple:
    """Build the positional params tuple for one row. Requires phase + action_type."""
    if not fields.get("phase") or not fields.get("action_type"):
        raise ValueError("agent_actions row requires phase and action_type")
    details = fields.get("details")
    if details is not None and not isinstance(details, str):
        details = json.dumps(details, default=str)
    return (
        fields.get("deal_id"),
        fields.get("document_id"),
        fields.get("doc_pk"),
        fields.get("doc_type"),
        fields.get("process_monitor_id"),
        _as_text(fields.get("trace_id")),
        fields["phase"],
        fields["action_type"],
        fields.get("agent"),
        fields.get("field_name"),
        fields.get("status", "ok"),
        fields.get("summary"),
        details,
        fields.get("confidence"),
        fields.get("pipeline_version"),
    )


def record_action(*, phase: str, action_type: str, conn: Any = None, **fields: Any) -> None:
    """Insert one action row. Best-effort: errors are logged, never raised.

    If ``conn`` is provided, the row is written on that connection's cursor and
    the caller owns commit/rollback. Otherwise a short autonomous transaction is
    opened and committed here.
    """
    try:
        fields["phase"] = phase
        fields["action_type"] = action_type
        params = _row_params(fields)
        if conn is not None:
            conn.cursor().execute(_INSERT, params)
            return
        with get_conn() as own:
            own.autocommit = False
            cur = own.cursor()
            try:
                cur.execute(_INSERT, params)
                own.commit()
            except Exception:
                own.rollback()
                raise
    except Exception as exc:  # best-effort: never break the caller
        log.warning("agent_actions.record_action failed (%s/%s): %s", phase, action_type, exc)


def bulk_record(rows: Iterable[Mapping[str, Any]], *, conn: Any = None) -> None:
    """Insert many rows (each mapping must include phase + action_type).

    Best-effort: errors are logged, never raised.
    """
    try:
        params = [_row_params(r) for r in rows]
        if not params:
            return
        if conn is not None:
            conn.cursor().executemany(_INSERT, params)
            return
        with get_conn() as own:
            own.autocommit = False
            cur = own.cursor()
            try:
                cur.executemany(_INSERT, params)
                own.commit()
            except Exception:
                own.rollback()
                raise
    except Exception as exc:  # best-effort
        log.warning("agent_actions.bulk_record failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_agent_actions.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_actions.py tests/services/test_agent_actions.py
git commit -m "feat(actions): best-effort agent_actions writer"
```

---

## Task 3: Emit extraction actions from `dispatch_document()`

**Files:**
- Modify: `src/services/extraction/dispatch.py`
- Test: `tests/services/test_agent_actions.py` (add an integration-style test with a recording conn is NOT possible here because dispatch opens its own; instead verify via a focused unit test on a small helper)

We add a tiny module-private helper in `dispatch.py` so each call site is one line and cannot alter control flow. All calls pass `phase=PHASE_EXTRACTION`. The writer is already best-effort, so no try/except is needed at the call sites.

- [ ] **Step 1: Add the import and helper near the top of dispatch.py**

In `src/services/extraction/dispatch.py`, after the existing imports (the block around line 17-24), add:

```python
from src.services.agent_actions import record_action, PHASE_EXTRACTION
```

- [ ] **Step 2: Emit an action after grounding gate**

Find the line (≈197):

```python
    grounded = [c for c in candidates if _grounded(c)]
```

Immediately after it, add:

```python
    record_action(
        phase=PHASE_EXTRACTION,
        action_type="grounding_gate",
        doc_type=doc_type,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        agent="grounding_gate",
        status="ok" if grounded else "warn",
        summary=f"{len(grounded)} of {len(candidates)} candidates grounded",
        details={"candidates": len(candidates), "grounded": len(grounded)},
    )
```

- [ ] **Step 3: Emit an action after context-layer synthesis**

Find the context-layer block (≈233-244) where `synthesized` is applied. Immediately after the `for k, v in synthesized.items():` loop completes (before the `except` that logs `context_layer synthesis failed`), add inside the `try` after the loop:

```python
            record_action(
                phase=PHASE_EXTRACTION,
                action_type="context_synthesize",
                doc_type=doc_type,
                trace_id=trace_id,
                pipeline_version=pipeline_version,
                agent="AgentNick",
                status="ok",
                summary=f"context_layer synthesized {len(synthesized)} fields",
                details={"fields": sorted(synthesized.keys())},
            )
```

(If the surrounding indentation differs, match the indentation of the `for` loop body. The call must be inside the same `try` so a failure there is already caught by the existing `except`.)

- [ ] **Step 4: Emit an action after persist (write_raw returns raw_id)**

Find (≈410):

```python
    raw_id = persistence.write_raw(
```

After the full `write_raw(...)` call returns (i.e. after the closing `)` of that statement and any line that assigns `raw_id`), add:

```python
    record_action(
        phase=PHASE_EXTRACTION,
        action_type="persist",
        doc_type=doc_type,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        agent="persistence",
        status="ok",
        summary=f"persisted raw_id={raw_id}",
        details={"raw_id": raw_id, "n_fields": len(columns), "n_lines": len(line_items)},
    )
```

(`columns` and `line_items` are already in scope from lines ≈200-201. If a variable name differs in the actual code, use the in-scope header-columns and line-items variables.)

- [ ] **Step 5: Run the existing dispatch tests to confirm nothing breaks**

Run: `pytest tests/extraction/test_dispatch.py -v`
Expected: PASS (same set as before — under pytest the writer's `get_conn()` is the in-memory fake, so the new calls are no-ops and cannot affect results).

- [ ] **Step 6: Commit**

```bash
git add src/services/extraction/dispatch.py
git commit -m "feat(actions): emit extraction actions from dispatch_document"
```

---

## Task 4: Emit one validation action per discrepancy

**Files:**
- Modify: `src/services/extraction/persistence.py`
- Test: `tests/services/test_persistence_actions.py` (new, focused)

`write_discrepancies()` (line 284) already builds `rows` and opens its own transaction. We record one `validation` action per discrepancy on the SAME connection so the action log is atomic with the discrepancy write.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_persistence_actions.py`:

```python
import src.services.agent_actions as aa
from src.services.extraction import persistence
from src.services.extraction.persistence import Discrepancy


def test_write_discrepancies_records_validation_actions(monkeypatch):
    captured = {}

    def fake_bulk_record(rows, *, conn=None):
        captured["rows"] = list(rows)
        captured["conn_passed"] = conn is not None

    monkeypatch.setattr(persistence, "bulk_record", fake_bulk_record)

    discs = [
        Discrepancy(
            field_name="tax_amount", raw_value="10", expected_value="12",
            computed_value="12", issue_type="tax_mismatch", severity="high",
            blocks_promotion=True, evidence_page=1, evidence_bbox=None,
            evidence_text="Tax 12", notes="",
        ),
    ]
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=42, source_file="/tmp/x.pdf",
        doc_pk_candidate="INV-1", discrepancies=discs,
    )
    assert captured["conn_passed"] is True
    assert len(captured["rows"]) == 1
    row = captured["rows"][0]
    assert row["phase"] == aa.PHASE_VALIDATION
    assert row["action_type"] == "discrepancy"
    assert row["field_name"] == "tax_amount"
    assert row["doc_pk"] == "INV-1"
```

> Note: under pytest `get_conn()` returns the in-memory fake, so the real INSERT is a no-op; the test asserts the action ROWS we hand to `bulk_record`, monkeypatching it to capture them. Confirm the `Discrepancy` constructor field names against `persistence.py` before running and adjust the kwargs if they differ.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/services/test_persistence_actions.py -v`
Expected: FAIL — `bulk_record` is not yet imported/called in `persistence.py` (AttributeError on `persistence.bulk_record`, or no rows captured).

- [ ] **Step 3: Add the import to persistence.py**

Near the top of `src/services/extraction/persistence.py` (with the other imports), add:

```python
from src.services.agent_actions import bulk_record, PHASE_VALIDATION
```

- [ ] **Step 4: Record actions inside write_discrepancies, on the same conn**

In `write_discrepancies()` replace the `with get_conn() as conn:` block (≈309-319) so the action rows are written on the same connection before commit:

```python
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.executemany(sql, rows)
            action_rows = [
                {
                    "phase": PHASE_VALIDATION,
                    "action_type": "discrepancy",
                    "doc_type": doc_type,
                    "doc_pk": doc_pk_candidate,
                    "field_name": d.field_name,
                    "status": "error" if d.blocks_promotion else "warn",
                    "summary": f"{d.issue_type} on {d.field_name}",
                    "details": {
                        "issue_type": d.issue_type,
                        "severity": d.severity,
                        "blocks_promotion": d.blocks_promotion,
                        "raw_value": d.raw_value,
                        "expected_value": d.expected_value,
                        "computed_value": d.computed_value,
                    },
                }
                for d in discrepancies
            ]
            bulk_record(action_rows, conn=conn)
            conn.commit()
            return len(rows)
        except Exception:
            conn.rollback()
            raise
```

> `discrepancies` is the original iterable param. If it is a one-shot generator that was already consumed building `rows`, materialize it at the top of the function instead: change the first lines to `discrepancies = list(discrepancies)` before the `rows = []` loop, then the comprehension above is safe.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/services/test_persistence_actions.py -v`
Expected: PASS.

- [ ] **Step 6: Run the broader persistence/extraction suite**

Run: `pytest tests/extraction -q`
Expected: PASS (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction/persistence.py tests/services/test_persistence_actions.py
git commit -m "feat(actions): record a validation action per discrepancy"
```

---

## Task 5: Build the deal-context gatherer

**Files:**
- Create: `src/services/deal_summary.py`
- Test: `tests/services/test_deal_summary.py`

The gatherer is read-only and returns a plain dict. It uses a `_fetch_dicts(cur, sql, params)` helper that maps rows to dicts via `cur.description`, so it is testable with a fake cursor that keys off the table name in the SQL.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_deal_summary.py`:

```python
import src.services.deal_summary as ds


class _FakeCursor:
    """Returns canned (description, rows) based on a substring of the SQL."""

    def __init__(self, table_data):
        # table_data: {sql_substring: (columns, rows)}
        self._table_data = table_data
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data):
        self._cur = _FakeCursor(table_data)

    def cursor(self):
        return self._cur


def _conn_for_deal_with_one_invoice():
    return _FakeConn({
        "bp_invoice_trgt": (
            ["invoice_id", "supplier_id", "invoice_amount", "deal_id", "deal_name", "document_id"],
            [("INV-1", "ACME", 100, "D-9", "Acme Deal", "DOC-1")],
        ),
        "bp_invoice_line_items_trgt": (
            ["invoice_id", "line_number", "item_description", "quantity"],
            [("INV-1", 1, "Widget", 3)],
        ),
        "bp_purchase_order_trgt": (["po_id", "deal_id"], []),
        "bp_quote_trgt": (["quote_id", "deal_id"], []),
        "bp_contracts": (["contract_id", "deal_id"], []),
        "agent_actions": (
            ["phase", "action_type", "summary"],
            [("extraction", "persist", "persisted raw_id=5")],
        ),
        "bp_extraction_discrepancy": (["field_name", "issue_type"], []),
    })


def test_gather_deal_context_assembles_documents_and_trail():
    ctx = ds.gather_deal_context("D-9", conn=_conn_for_deal_with_one_invoice())
    assert ctx is not None
    assert ctx["deal_id"] == "D-9"
    assert ctx["deal_name"] == "Acme Deal"
    invoices = ctx["documents"]["invoices"]
    assert len(invoices) == 1
    assert invoices[0]["invoice_id"] == "INV-1"
    assert invoices[0]["line_items"][0]["item_description"] == "Widget"
    assert ctx["actions"][0]["action_type"] == "persist"
    assert ctx["sources"]["invoices"] == 1
    assert ctx["sources"]["actions"] == 1


def test_gather_deal_context_unknown_deal_returns_none():
    empty = _FakeConn({
        "bp_invoice_trgt": (["invoice_id", "deal_id"], []),
        "bp_purchase_order_trgt": (["po_id", "deal_id"], []),
        "bp_quote_trgt": (["quote_id", "deal_id"], []),
        "bp_contracts": (["contract_id", "deal_id"], []),
        "agent_actions": (["phase"], []),
        "bp_extraction_discrepancy": (["field_name"], []),
    })
    assert ds.gather_deal_context("NOPE", conn=empty) is None


def test_build_prompt_is_grounded_and_factual():
    ctx = ds.gather_deal_context("D-9", conn=_conn_for_deal_with_one_invoice())
    prompt = ds._build_prompt(ctx)
    low = prompt.lower()
    assert "do not fabricate" in low or "only the data" in low
    assert "INV-1" in prompt        # facts are present in the prompt
    assert "Acme Deal" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_deal_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.deal_summary'`.

- [ ] **Step 3: Implement the gatherer (and prompt builder)**

Create `src/services/deal_summary.py`:

```python
"""Assemble and summarize everything known about a procurement deal.

A "deal" is the entity keyed by ``deal_id`` across the final (_trgt) document
tables. ``gather_deal_context`` is read-only and returns a plain dict;
``summarize_deal`` turns that dict into clear text via a grounded LLM call.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

# General model for summarization — deliberately NOT the extraction adapter,
# so the GPU/AgentNick stays free for extraction. Swappable to llm_router later.
_SUMMARY_MODEL = "qwen2.5:7b"

# (final table, line-items table or None, primary-key column)
_DOC_SOURCES = {
    "invoices": ("proc.bp_invoice_trgt", "proc.bp_invoice_line_items_trgt", "invoice_id"),
    "purchase_orders": ("proc.bp_purchase_order_trgt", "proc.bp_po_line_items_trgt", "po_id"),
    "quotes": ("proc.bp_quote_trgt", "proc.bp_quote_line_items_trgt", "quote_id"),
    "contracts": ("proc.bp_contracts", None, "contract_id"),
}


def _fetch_dicts(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _gather(conn, deal_id: str) -> Optional[dict]:
    cur = conn.cursor()
    documents: dict[str, list[dict]] = {}
    total_docs = 0
    deal_name: Optional[str] = None

    for key, (table, lines_table, pk) in _DOC_SOURCES.items():
        rows = _fetch_dicts(cur, f"SELECT * FROM {table} WHERE deal_id = %s", (deal_id,))
        for r in rows:
            if not deal_name and r.get("deal_name"):
                deal_name = r["deal_name"]
            if lines_table is not None and r.get(pk) is not None:
                r["line_items"] = _fetch_dicts(
                    cur, f"SELECT * FROM {lines_table} WHERE {pk} = %s", (r[pk],)
                )
            else:
                r["line_items"] = []
        documents[key] = rows
        total_docs += len(rows)

    if total_docs == 0:
        return None

    actions = _fetch_dicts(
        cur,
        "SELECT * FROM proc.agent_actions WHERE deal_id = %s ORDER BY created_at ASC",
        (deal_id,),
    )
    discrepancies = _fetch_dicts(
        cur,
        "SELECT * FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate IN ("
        " SELECT invoice_id::text FROM proc.bp_invoice_trgt WHERE deal_id = %s"
        " UNION SELECT po_id::text FROM proc.bp_purchase_order_trgt WHERE deal_id = %s"
        " UNION SELECT quote_id::text FROM proc.bp_quote_trgt WHERE deal_id = %s)",
        (deal_id, deal_id, deal_id),
    )

    return {
        "deal_id": deal_id,
        "deal_name": deal_name,
        "documents": documents,
        "actions": actions,
        "discrepancies": discrepancies,
        "sources": {
            "invoices": len(documents["invoices"]),
            "purchase_orders": len(documents["purchase_orders"]),
            "quotes": len(documents["quotes"]),
            "contracts": len(documents["contracts"]),
            "actions": len(actions),
            "discrepancies": len(discrepancies),
        },
    }


def gather_deal_context(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Read-only assembly of a deal. Returns None if no final records exist."""
    if conn is not None:
        return _gather(conn, deal_id)
    with get_conn() as own:
        return _gather(own, deal_id)


def _build_prompt(ctx: dict) -> str:
    facts = json.dumps(ctx, indent=2, default=str)
    return (
        "You are a procurement analyst. Write a clear, plain-English summary of "
        "the deal described by the JSON facts below.\n\n"
        "Rules:\n"
        "- Use ONLY the data provided. Do not fabricate or infer values that are "
        "not present.\n"
        "- If something is absent or null, simply leave it out — do not guess.\n"
        "- Cover: the documents involved (invoices, purchase orders, quotes, "
        "contracts), key amounts and currencies, suppliers, and any "
        "discrepancies or notable actions in the trail.\n"
        "- Be concise and factual; no marketing language.\n\n"
        f"Deal facts (JSON):\n{facts}\n\n"
        "Summary:"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_deal_summary.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_summary.py tests/services/test_deal_summary.py
git commit -m "feat(summary): read-only deal-context gatherer + grounded prompt"
```

---

## Task 6: Add the `summarize_deal` LLM call

**Files:**
- Modify: `src/services/deal_summary.py`
- Test: `tests/services/test_deal_summary.py`

- [ ] **Step 1: Write the failing tests (append to the existing test file)**

Append to `tests/services/test_deal_summary.py`:

```python
def test_summarize_deal_returns_text_and_sources(monkeypatch):
    monkeypatch.setattr(
        ds, "gather_deal_context",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": "Acme Deal",
            "documents": {"invoices": [], "purchase_orders": [], "quotes": [], "contracts": []},
            "actions": [], "discrepancies": [],
            "sources": {"invoices": 1, "purchase_orders": 0, "quotes": 0,
                        "contracts": 0, "actions": 2, "discrepancies": 0},
        },
    )
    monkeypatch.setattr(ds, "ollama_generate", lambda *a, **k: "Acme Deal: one invoice for ACME.")
    out = ds.summarize_deal("D-9")
    assert out["deal_id"] == "D-9"
    assert "Acme Deal" in out["summary"]
    assert out["sources"]["invoices"] == 1


def test_summarize_deal_unknown_returns_none(monkeypatch):
    monkeypatch.setattr(ds, "gather_deal_context", lambda deal_id, conn=None: None)
    assert ds.summarize_deal("NOPE") is None


def test_summarize_deal_raises_on_empty_llm(monkeypatch):
    monkeypatch.setattr(
        ds, "gather_deal_context",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": None,
            "documents": {"invoices": [], "purchase_orders": [], "quotes": [], "contracts": []},
            "actions": [], "discrepancies": [], "sources": {"invoices": 1},
        },
    )
    monkeypatch.setattr(ds, "ollama_generate", lambda *a, **k: "")
    import pytest
    with pytest.raises(ds.SummarizationError):
        ds.summarize_deal("D-9")
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/services/test_deal_summary.py -k summarize -v`
Expected: FAIL — `summarize_deal` / `SummarizationError` / `ollama_generate` not defined in module.

- [ ] **Step 3: Implement the summarizer**

In `src/services/deal_summary.py`, add the import near the top imports:

```python
from src.services.ollama_client import ollama_generate
```

and append at the end of the module:

```python
class SummarizationError(RuntimeError):
    """Raised when the LLM returns no usable summary."""


def summarize_deal(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Summarize a deal into clear text. Returns None if the deal is unknown.

    Raises SummarizationError if the LLM returns nothing.
    """
    ctx = gather_deal_context(deal_id, conn=conn)
    if ctx is None:
        return None
    prompt = _build_prompt(ctx)
    text = ollama_generate(
        prompt,
        model=_SUMMARY_MODEL,
        temperature=0.0,
        num_predict=1024,
        timeout=120,
        retries=2,
    )
    if not text or not text.strip():
        raise SummarizationError(f"empty summary for deal {deal_id}")
    return {
        "deal_id": deal_id,
        "deal_name": ctx.get("deal_name"),
        "summary": text.strip(),
        "sources": ctx["sources"],
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/services/test_deal_summary.py -v`
Expected: all passed (6 total in the file).

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_summary.py tests/services/test_deal_summary.py
git commit -m "feat(summary): grounded summarize_deal via ollama qwen2.5:7b"
```

---

## Task 7: Expose the FastAPI endpoint

**Files:**
- Create: `src/api/routers/deal_summary.py`
- Modify: `src/api/main.py`
- Test: `tests/test_deal_summary_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deal_summary_api.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.deal_summary as router_mod


def _client():
    app = FastAPI()
    app.include_router(router_mod.router)
    return TestClient(app)


def test_get_summary_ok(monkeypatch):
    monkeypatch.setattr(
        router_mod, "summarize_deal",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": "Acme Deal",
            "summary": "One invoice for ACME.",
            "sources": {"invoices": 1, "actions": 2},
        },
    )
    resp = _client().get("/deals/D-9/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deal_id"] == "D-9"
    assert body["summary"] == "One invoice for ACME."
    assert "generated_at" in body


def test_get_summary_unknown_deal_404(monkeypatch):
    monkeypatch.setattr(router_mod, "summarize_deal", lambda deal_id, conn=None: None)
    resp = _client().get("/deals/NOPE/summary")
    assert resp.status_code == 404


def test_get_summary_llm_failure_502(monkeypatch):
    def boom(deal_id, conn=None):
        raise router_mod.SummarizationError("empty")
    monkeypatch.setattr(router_mod, "summarize_deal", boom)
    resp = _client().get("/deals/D-9/summary")
    assert resp.status_code == 502
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_deal_summary_api.py -v`
Expected: FAIL — `No module named 'src.api.routers.deal_summary'`.

- [ ] **Step 3: Implement the router**

Create `src/api/routers/deal_summary.py`:

```python
"""AI summary of a procurement deal.

GET /deals/{deal_id}/summary — consolidates the deal's final (_trgt) records,
line items, action trail and discrepancies, then returns a clear-text summary.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from src.services.deal_summary import summarize_deal, SummarizationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deals", tags=["Deals"])


@router.get("/{deal_id}/summary", summary="AI summary of a procurement deal")
def get_deal_summary(deal_id: str) -> dict[str, Any]:
    try:
        result = summarize_deal(deal_id)
    except SummarizationError as exc:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}")
    except Exception as exc:  # DB or unexpected error
        logger.exception("deal summary failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal found for deal_id={deal_id}")
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_deal_summary_api.py -v`
Expected: 3 passed.

- [ ] **Step 5: Register the router in main.py**

In `src/api/main.py`, line 47, add `deal_summary` to the routers import:

```python
from api.routers import agents as agents_router_mod, documents, email, metrics, run, stream, system, training, vendors, workflows, deal_summary
```

Then near the other `app.include_router(...)` calls (≈261-267) add:

```python
app.include_router(deal_summary.router)
```

- [ ] **Step 6: Verify the app imports cleanly**

Run:

```bash
python3 -c "import src.api.main; print('main imports ok')"
```

Expected: `main imports ok` (it may log startup lines; the import must not raise).

- [ ] **Step 7: Commit**

```bash
git add src/api/routers/deal_summary.py src/api/main.py tests/test_deal_summary_api.py
git commit -m "feat(api): GET /deals/{deal_id}/summary endpoint"
```

---

## Task 8: End-to-end smoke test against the live DB

**Files:** none (manual verification)

- [ ] **Step 1: Confirm a real deal_id exists**

Run:

```bash
python3 -c "
from src.services.db import get_conn
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute(\"select deal_id, count(*) from proc.bp_quote_trgt where deal_id is not null group by deal_id order by 2 desc limit 5\")
    print(cur.fetchall())
"
```

Expected: a list of `(deal_id, count)` tuples. Pick one `deal_id` (call it `<DEAL>`).

- [ ] **Step 2: Gather context for that deal (no LLM)**

Run:

```bash
python3 -c "
from src.services.deal_summary import gather_deal_context
ctx = gather_deal_context('<DEAL>')
print(None if ctx is None else ctx['sources'])
"
```

Expected: a `sources` dict with non-zero counts. If `None`, pick a different deal_id.

- [ ] **Step 3: Generate a summary end-to-end (hits Ollama)**

Run:

```bash
python3 -c "
from src.services.deal_summary import summarize_deal
out = summarize_deal('<DEAL>')
print(out['summary'][:600])
"
```

Expected: a coherent plain-text summary mentioning the deal's documents/amounts. Requires Ollama running with `qwen2.5:7b` available (`ollama pull qwen2.5:7b` if missing).

- [ ] **Step 4: Verify agent_actions rows accrue on the next extraction**

After the next document runs through `dispatch_document` (or re-run an extraction), run:

```bash
python3 -c "
from src.services.db import get_conn
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute('select phase, action_type, count(*) from proc.agent_actions group by 1,2 order by 3 desc')
    print(cur.fetchall())
"
```

Expected: rows for `('extraction','persist')`, `('extraction','grounding_gate')`, `('validation','discrepancy')`, etc.

- [ ] **Step 5: Full test suite**

Run: `pytest tests/services/test_agent_actions.py tests/services/test_deal_summary.py tests/services/test_persistence_actions.py tests/test_deal_summary_api.py tests/extraction -q`
Expected: all pass.

---

## Self-Review Notes

- **Spec coverage:** table (Task 1) ✓; writer best-effort + optional conn (Task 2) ✓; extraction hooks (Task 3) ✓; validation-per-discrepancy on same conn (Task 4) ✓; consolidation writer slot — documented constants `PHASE_CONSOLIDATION` + `reconcile_*` action types, no logic built ✓; gatherer over `_trgt` + line items + actions + discrepancies (Task 5) ✓; grounded summarizer (Task 6) ✓; endpoint with 404/502/500 (Task 7) ✓; tests throughout ✓.
- **LLM deviation from spec** (llm_router → `ollama_generate` + `qwen2.5:7b`) is documented in the header with rationale; behind `_SUMMARY_MODEL` for easy swap.
- **Type consistency:** `gather_deal_context(deal_id, conn=None)`, `summarize_deal(deal_id, conn=None)`, `_build_prompt(ctx)`, `record_action(*, phase, action_type, conn=None, **fields)`, `bulk_record(rows, *, conn=None)`, `SummarizationError` used consistently across service, tests, and router.
- **Known adjustment points flagged inline:** `Discrepancy` constructor field names (Task 4 Step 1), exact in-scope variable names at the dispatch persist call site (Task 3 Step 4), and whether `discrepancies` is a generator (Task 4 Step 4).
