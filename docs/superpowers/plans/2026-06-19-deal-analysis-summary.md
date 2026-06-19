# Deal Analysis Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a deal becomes `Deal_Linked`, automatically produce a structured metrics row (`proc.bp_analysis_summary`) and a narrative summary (`proc.bp_summary`) for it — for all linked deals — and wire the metrics row to the UI's "Detailed Analysis Summary" grid.

**Architecture:** A new deterministic service (`deal_analysis_service.py`) reuses `deal_summary.gather_deal_context()` to compute per-deal metrics with no LLM, persists them to a new `bp_analysis_summary` table, and generates the narrative via the existing local-AgentNick `summarize_deal()`, storing it in `bp_summary`. A sync function processes every `Deal_Linked` deal concurrently and is invoked at the end of the existing `assign_deals()` pipeline (the natural `Deal_Linked` event boundary) plus a manual backfill endpoint. The UI grid is repointed from the legacy AWS `/analyse` to a new local FastAPI endpoint.

**Tech Stack:** Python 3 / FastAPI / psycopg (`src.services.db.get_conn`), PostgreSQL (`proc` schema, `bp_sqldb`), Ollama local `BeyondProcwise/AgentNick:unified`, React + Material-UI (`beyond_procwise_ui`).

## Global Constraints

- All new DB tables use the `bp_` prefix; indexes use `ix_bp_*` naming.
- No fabrication: any metric not derivable from the deal's documents stays `NULL` (UI renders "–").
- Narrative generation uses **local AgentNick** only (`deal_summary.summarize_deal`, model `BeyondProcwise/AgentNick:unified`). Never route to qwen/cloud.
- Never modify source extraction data; this feature is read-only against `_trgt` document/line tables.
- No `Co-Authored-By: Claude` lines in git commits.
- Prove the change on the running local server against live `bp_sqldb`, not just tests/mocks.
- A deal is `Deal_Linked` only when it has quote + PO + invoice (`reconcile_status`).

---

### Task 1: Create `proc.bp_analysis_summary` table

**Files:**
- Create: `deploy/sql/2026-06-19_create_bp_analysis_summary.sql`

**Interfaces:**
- Produces: table `proc.bp_analysis_summary` with columns `analysis_id UUID PK`, `deal_id varchar(25)`, `deal_name varchar`, `supplier text`, `category text`, `deal_value numeric(18,2)`, `currency varchar(8)`, `volume numeric(18,2)`, `unit_price numeric(18,4)`, `price_change_pct numeric(9,2)`, `volume_change_pct numeric(9,2)`, `efficiency_score numeric(18,2)`, `items jsonb`, `item_count int`, `narrative_summary_id uuid`, `data_snapshot jsonb`, `model text`, `is_current bool`, `generated_at timestamptz`; index `ix_bp_analysis_summary_current`.

- [ ] **Step 1: Write the DDL file**

Create `deploy/sql/2026-06-19_create_bp_analysis_summary.sql`:

```sql
-- Structured per-deal analytics row backing the UI "Detailed Analysis Summary"
-- grid. One current row per deal (is_current pattern, mirrors proc.bp_summary).
-- All values are computed deterministically from the deal's _trgt documents;
-- anything not derivable stays NULL (no fabrication).

CREATE TABLE IF NOT EXISTS proc.bp_analysis_summary (
    analysis_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id              VARCHAR(25) NOT NULL,
    deal_name            VARCHAR,
    supplier             TEXT,
    category             TEXT,
    deal_value           NUMERIC(18,2),
    currency             VARCHAR(8),
    volume               NUMERIC(18,2),
    unit_price           NUMERIC(18,4),
    price_change_pct     NUMERIC(9,2),
    volume_change_pct    NUMERIC(9,2),
    efficiency_score     NUMERIC(18,2),
    items                JSONB,
    item_count           INTEGER,
    narrative_summary_id UUID,
    data_snapshot        JSONB,
    model                TEXT,
    is_current           BOOLEAN     NOT NULL DEFAULT true,
    generated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_summary_current
    ON proc.bp_analysis_summary (deal_id) WHERE is_current;
CREATE INDEX IF NOT EXISTS ix_bp_analysis_summary_deal
    ON proc.bp_analysis_summary (deal_id, generated_at DESC);
```

- [ ] **Step 2: Apply the DDL to live `bp_sqldb`**

Run:
```bash
python -c "from src.services.db import get_conn; sql=open('deploy/sql/2026-06-19_create_bp_analysis_summary.sql').read(); c=get_conn(); cur=c.cursor(); cur.execute(sql); c.commit(); print('applied')"
```
Expected: prints `applied` with no error.

- [ ] **Step 3: Verify the table exists**

Run:
```bash
python -c "from src.services.db import get_conn; c=get_conn(); cur=c.cursor(); cur.execute(\"select count(*) from proc.bp_analysis_summary\"); print('rows:', cur.fetchone()[0])"
```
Expected: prints `rows: 0`.

- [ ] **Step 4: Commit**

```bash
git add deploy/sql/2026-06-19_create_bp_analysis_summary.sql
git commit -m "feat(deal-summary): add proc.bp_analysis_summary metrics table"
```

---

### Task 2: Deterministic metric computation (`compute_deal_metrics`)

**Files:**
- Create: `src/services/deal_analysis_service.py`
- Test: `tests/services/test_deal_analysis_service.py`

**Interfaces:**
- Consumes: `deal_summary.gather_deal_context(deal_id, conn) -> dict | None` returning `{"deal_id","deal_name","documents":{"invoices":[...],"purchase_orders":[...],"quotes":[...]},"sources":{...}}` where each document dict has `supplier_name`, `total_amount`, `currency`, and a `line_items` list whose items have `item_description`, `quantity`, `unit_price`.
- Produces: `compute_deal_metrics(deal_id: str, conn=None) -> dict | None` returning keys: `deal_id, deal_name, supplier, category, deal_value, currency, volume, unit_price, price_change_pct, volume_change_pct, efficiency_score, items (list[dict]), item_count (int), data_snapshot (dict)`. Returns `None` if the deal has no final records. Also produces module-level helpers `_doc_total(docs)`, `_sum_qty(docs)`, `_weighted_unit_price(docs)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_deal_analysis_service.py`:

```python
import importlib
import pytest

mod = importlib.import_module("src.services.deal_analysis_service")


def _ctx(invoices=None, pos=None, quotes=None, deal_name="DEAL-1"):
    return {
        "deal_id": "DEAL-1",
        "deal_name": deal_name,
        "documents": {
            "invoices": invoices or [],
            "purchase_orders": pos or [],
            "quotes": quotes or [],
        },
        "sources": {},
    }


def test_full_deal_metrics(monkeypatch):
    inv = {"supplier_name": "Acme", "total_amount": 1000, "currency": "GBP",
           "line_items": [
               {"item_description": "Widget A", "quantity": 100, "unit_price": 8.0},
               {"item_description": "Bolt B", "quantity": 100, "unit_price": 2.0}]}
    quote = {"supplier_name": "Acme", "total_amount": 900, "currency": "GBP",
             "line_items": [
                 {"item_description": "Widget A", "quantity": 120, "unit_price": 7.0},
                 {"item_description": "Bolt B", "quantity": 80, "unit_price": 1.5}]}
    monkeypatch.setattr(mod, "gather_deal_context",
                        lambda deal_id, conn=None: _ctx(invoices=[inv], quotes=[quote]))
    monkeypatch.setattr(mod, "_deal_category", lambda cur, deal_id: "Electronics")
    m = mod.compute_deal_metrics("DEAL-1", conn=object())
    assert m["supplier"] == "Acme"
    assert m["category"] == "Electronics"
    assert m["deal_value"] == 1000          # invoice total
    assert m["currency"] == "GBP"
    assert m["volume"] == 200               # 100 + 100 invoiced qty
    assert m["unit_price"] == pytest.approx(5.0)   # 1000 / 200
    # invoice weighted unit = 1000/200 = 5.0 ; quote weighted = (120*7+80*1.5)/200 = 4.8
    assert m["price_change_pct"] == pytest.approx(4.17, abs=0.01)   # (5.0-4.8)/4.8*100
    assert m["volume_change_pct"] == pytest.approx(0.0)             # 200 vs 200
    assert m["item_count"] == 2
    assert {i["name"] for i in m["items"]} == {"Widget A", "Bolt B"}


def test_missing_quote_leaves_changes_null(monkeypatch):
    inv = {"supplier_name": "Acme", "total_amount": 500, "currency": "USD",
           "line_items": [{"item_description": "X", "quantity": 50, "unit_price": 10.0}]}
    monkeypatch.setattr(mod, "gather_deal_context",
                        lambda deal_id, conn=None: _ctx(invoices=[inv]))
    monkeypatch.setattr(mod, "_deal_category", lambda cur, deal_id: None)
    m = mod.compute_deal_metrics("DEAL-1", conn=object())
    assert m["deal_value"] == 500
    assert m["price_change_pct"] is None
    assert m["volume_change_pct"] is None
    assert m["efficiency_score"] is None


def test_unknown_deal_returns_none(monkeypatch):
    monkeypatch.setattr(mod, "gather_deal_context", lambda deal_id, conn=None: None)
    assert mod.compute_deal_metrics("NOPE", conn=object()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_deal_analysis_service.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (module/functions not defined).

- [ ] **Step 3: Write the implementation**

Create `src/services/deal_analysis_service.py`:

```python
"""Deterministic per-deal analytics + summary sync.

compute_deal_metrics() turns a deal's final (_trgt) documents into the metrics
row backing the UI grid — no LLM, no fabrication. sync_deal_summaries() (Task 3)
persists those rows and the AgentNick narrative for every Deal_Linked deal.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn
from src.services.deal_summary import gather_deal_context

log = logging.getLogger(__name__)


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(docs: list[dict], key: str):
    for d in docs:
        val = d.get(key)
        if val not in (None, ""):
            return val
    return None


def _doc_total(docs: list[dict]) -> Optional[float]:
    """Sum of header total_amount across docs of one type; None if none present."""
    vals = [_num(d.get("total_amount")) for d in docs]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _sum_qty(docs: list[dict]) -> Optional[float]:
    total = 0.0
    seen = False
    for d in docs:
        for li in d.get("line_items") or []:
            q = _num(li.get("quantity"))
            if q is not None:
                total += q
                seen = True
    return total if seen else None


def _weighted_unit_price(docs: list[dict]) -> Optional[float]:
    """Total line value / total qty across a doc type's line items."""
    val = 0.0
    qty = 0.0
    seen = False
    for d in docs:
        for li in d.get("line_items") or []:
            q = _num(li.get("quantity"))
            up = _num(li.get("unit_price"))
            if q is not None and up is not None:
                val += q * up
                qty += q
                seen = True
    if not seen or qty == 0:
        return None
    return val / qty


def _pct_change(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base is None or base == 0:
        return None
    return round((new - base) / base * 100.0, 2)


def _deal_category(cur, deal_id: str) -> Optional[str]:
    cur.execute(
        "select category from proc.process_monitor "
        "where deal_id = %s and category is not null limit 1",
        (deal_id,))
    row = cur.fetchone()
    return row[0] if row else None


def _items_from(docs: list[dict]) -> list[dict]:
    items: list[dict] = []
    for d in docs:
        for li in d.get("line_items") or []:
            name = li.get("item_description")
            if name:
                items.append({"name": name,
                              "qty": _num(li.get("quantity")),
                              "unit_price": _num(li.get("unit_price"))})
    return items


def _compute(ctx: dict, cur) -> dict:
    docs = ctx["documents"]
    inv, po, quote = docs["invoices"], docs["purchase_orders"], docs["quotes"]

    supplier = (_first_present(inv, "supplier_name")
                or _first_present(po, "supplier_name")
                or _first_present(quote, "supplier_name"))

    # deal value / currency: prefer invoice, then PO, then quote
    deal_value = _doc_total(inv)
    currency = _first_present(inv, "currency")
    if deal_value is None:
        deal_value, currency = _doc_total(po), _first_present(po, "currency")
    if deal_value is None:
        deal_value, currency = _doc_total(quote), _first_present(quote, "currency")

    # volume: prefer invoice line qty, then PO, then quote
    volume = _sum_qty(inv) or _sum_qty(po) or _sum_qty(quote)
    unit_price = (deal_value / volume) if (deal_value is not None and volume) else None

    inv_unit = _weighted_unit_price(inv)
    quote_unit = _weighted_unit_price(quote) or _weighted_unit_price(po)
    price_change_pct = _pct_change(inv_unit, quote_unit)

    inv_vol = _sum_qty(inv)
    quote_vol = _sum_qty(quote) or _sum_qty(po)
    volume_change_pct = _pct_change(inv_vol, quote_vol)

    # efficiency = realized savings = (quoted unit - invoiced unit) * invoiced volume
    efficiency_score = None
    if inv_unit is not None and quote_unit is not None and inv_vol is not None:
        efficiency_score = round((quote_unit - inv_unit) * inv_vol, 2)

    items = _items_from(inv) or _items_from(quote) or _items_from(po)

    return {
        "deal_id": ctx["deal_id"],
        "deal_name": ctx.get("deal_name"),
        "supplier": supplier,
        "category": _deal_category(cur, ctx["deal_id"]) if cur is not None else None,
        "deal_value": round(deal_value, 2) if deal_value is not None else None,
        "currency": currency,
        "volume": volume,
        "unit_price": round(unit_price, 4) if unit_price is not None else None,
        "price_change_pct": price_change_pct,
        "volume_change_pct": volume_change_pct,
        "efficiency_score": efficiency_score,
        "items": items,
        "item_count": len(items),
        "data_snapshot": {
            "invoice_total": _doc_total(inv), "po_total": _doc_total(po),
            "quote_total": _doc_total(quote), "invoice_unit": inv_unit,
            "quote_unit": quote_unit, "invoice_volume": inv_vol,
            "quote_volume": quote_vol,
        },
    }


def compute_deal_metrics(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Deterministic metrics for a deal. None if no final records exist."""
    if conn is not None:
        ctx = gather_deal_context(deal_id, conn=conn)
        if ctx is None:
            return None
        return _compute(ctx, conn.cursor())
    with get_conn() as own:
        ctx = gather_deal_context(deal_id, conn=own)
        if ctx is None:
            return None
        return _compute(ctx, own.cursor())
```

Note: the tests monkeypatch `_deal_category`, so the `cur.cursor()` call in `_compute` is never hit with a real DB during unit tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_deal_analysis_service.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_analysis_service.py tests/services/test_deal_analysis_service.py
git commit -m "feat(deal-summary): deterministic compute_deal_metrics"
```

---

### Task 3: Persist metrics + narrative, and `sync_deal_summaries`

**Files:**
- Modify: `src/services/deal_analysis_service.py`
- Test: `tests/services/test_deal_analysis_sync.py`

**Interfaces:**
- Consumes: `compute_deal_metrics` (Task 2); `deal_summary.summarize_deal(deal_id, conn) -> {"summary","deal_name","sources"} | None` (local AgentNick); `summary_agent._store_summary(conn, *, persona, persona_source, scope, deal_id, summary, data_snapshot, sources, model, is_current) -> {"summary_id",...}`.
- Produces: `upsert_analysis_row(conn, metrics: dict, narrative_summary_id: str | None, model: str) -> str` (returns `analysis_id`); `generate_for_deal(deal_id: str, conn) -> dict` (computes metrics, generates+stores narrative, upserts row); `sync_deal_summaries(conn=None, deal_ids: list[str] | None = None, max_workers: int = 4) -> dict` returning `{"processed": int, "skipped": int, "failed": int, "deal_ids": [...]}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_deal_analysis_sync.py`:

```python
import importlib

mod = importlib.import_module("src.services.deal_analysis_service")

SAMPLE = {
    "deal_id": "DEAL-1", "deal_name": "DEAL-1", "supplier": "Acme",
    "category": "Electronics", "deal_value": 1000.0, "currency": "GBP",
    "volume": 200.0, "unit_price": 5.0, "price_change_pct": 4.17,
    "volume_change_pct": 0.0, "efficiency_score": 40.0,
    "items": [{"name": "Widget A", "qty": 100, "unit_price": 8.0}],
    "item_count": 1, "data_snapshot": {"invoice_total": 1000.0},
}


class FakeCur:
    def __init__(self):
        self.executed = []
        self._rows = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if "select distinct deal_id" in sql:
            self._rows = [("DEAL-1",), ("DEAL-2",)]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return None


class FakeConn:
    def __init__(self):
        self._cur = FakeCur()

    def cursor(self):
        return self._cur

    def commit(self):
        pass


def test_upsert_demotes_then_inserts():
    conn = FakeConn()
    aid = mod.upsert_analysis_row(conn, SAMPLE, "sum-123", "BeyondProcwise/AgentNick:unified")
    sqls = " | ".join(s for s, _ in conn._cur.executed)
    assert "UPDATE proc.bp_analysis_summary SET is_current = false" in sqls
    assert "INSERT INTO proc.bp_analysis_summary" in sqls
    assert isinstance(aid, str) and len(aid) > 0


def test_sync_processes_linked_deals(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_linked_deal_ids_needing_summary",
                        lambda cur: ["DEAL-1", "DEAL-2"])
    monkeypatch.setattr(mod, "generate_for_deal",
                        lambda deal_id, conn: calls.append(deal_id) or {"deal_id": deal_id})
    res = mod.sync_deal_summaries(conn=FakeConn(), max_workers=2)
    assert res["processed"] == 2
    assert set(calls) == {"DEAL-1", "DEAL-2"}


def test_sync_one_failure_isolated(monkeypatch):
    def boom(deal_id, conn):
        if deal_id == "DEAL-2":
            raise RuntimeError("llm down")
        return {"deal_id": deal_id}
    monkeypatch.setattr(mod, "_linked_deal_ids_needing_summary",
                        lambda cur: ["DEAL-1", "DEAL-2"])
    monkeypatch.setattr(mod, "generate_for_deal", boom)
    res = mod.sync_deal_summaries(conn=FakeConn(), max_workers=2)
    assert res["processed"] == 1
    assert res["failed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_deal_analysis_sync.py -v`
Expected: FAIL with `AttributeError` (functions not defined).

- [ ] **Step 3: Write the implementation (append to `deal_analysis_service.py`)**

Add these imports at the top of `src/services/deal_analysis_service.py` (alongside existing imports):

```python
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
```

Append to `src/services/deal_analysis_service.py`:

```python
_NARRATIVE_PERSONA = "analysis"
_NARRATIVE_SOURCE = "deal_analysis_service"


def upsert_analysis_row(conn: Any, metrics: dict, narrative_summary_id: Optional[str],
                        model: str) -> str:
    """Demote the deal's prior current row, then insert the new current row."""
    analysis_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    cur = conn.cursor()
    cur.execute(
        "UPDATE proc.bp_analysis_summary SET is_current = false "
        "WHERE deal_id = %s AND is_current", (metrics["deal_id"],))
    cur.execute(
        "INSERT INTO proc.bp_analysis_summary "
        "(analysis_id, deal_id, deal_name, supplier, category, deal_value, currency, "
        " volume, unit_price, price_change_pct, volume_change_pct, efficiency_score, "
        " items, item_count, narrative_summary_id, data_snapshot, model, is_current, "
        " generated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (analysis_id, metrics["deal_id"], metrics.get("deal_name"),
         metrics.get("supplier"), metrics.get("category"), metrics.get("deal_value"),
         metrics.get("currency"), metrics.get("volume"), metrics.get("unit_price"),
         metrics.get("price_change_pct"), metrics.get("volume_change_pct"),
         metrics.get("efficiency_score"),
         json.dumps(metrics.get("items"), default=str),
         metrics.get("item_count"), narrative_summary_id,
         json.dumps(metrics.get("data_snapshot"), default=str),
         model, True, generated_at))
    conn.commit()
    return analysis_id


def generate_for_deal(deal_id: str, conn: Any) -> dict:
    """Compute metrics, generate+store the AgentNick narrative, upsert the row."""
    from src.services.deal_summary import summarize_deal, _SUMMARY_MODEL
    from src.services.summary_agent import _store_summary

    metrics = compute_deal_metrics(deal_id, conn=conn)
    if metrics is None:
        return {"deal_id": deal_id, "status": "no_records"}

    narrative_id = None
    try:
        narr = summarize_deal(deal_id, conn=conn)
        if narr and narr.get("summary"):
            stored = _store_summary(
                conn, persona=_NARRATIVE_PERSONA, persona_source=_NARRATIVE_SOURCE,
                scope="deal", deal_id=deal_id, summary=narr["summary"],
                data_snapshot=metrics.get("data_snapshot"),
                sources=narr.get("sources"), model=_SUMMARY_MODEL, is_current=True)
            narrative_id = stored["summary_id"]
    except Exception as exc:  # narrative is best-effort; metrics still persist
        log.warning("narrative generation failed for %s: %s", deal_id, exc)

    upsert_analysis_row(conn, metrics, narrative_id, _SUMMARY_MODEL)
    return {"deal_id": deal_id, "status": "ok", "narrative_summary_id": narrative_id}


def _linked_deal_ids_needing_summary(cur) -> list[str]:
    """Deals at Deal_Linked status with no current bp_analysis_summary row."""
    cur.execute(
        "select distinct deal_id from proc.process_monitor pm "
        "where pm.status = 'Deal_Linked' and coalesce(pm.deal_id,'') <> '' "
        "and not exists (select 1 from proc.bp_analysis_summary a "
        "                where a.deal_id = pm.deal_id and a.is_current)")
    return [r[0] for r in cur.fetchall()]


def sync_deal_summaries(conn: Any = None, deal_ids: Optional[list[str]] = None,
                        max_workers: int = 4) -> dict:
    """Generate metrics + narrative for every linked deal missing a current summary.

    Each deal is processed on its own connection so failures stay isolated and
    work runs concurrently. Safe to call repeatedly (idempotent).
    """
    if conn is not None:
        ids = deal_ids if deal_ids is not None else _linked_deal_ids_needing_summary(conn.cursor())
    else:
        with get_conn() as own:
            ids = deal_ids if deal_ids is not None else _linked_deal_ids_needing_summary(own.cursor())

    processed = failed = 0
    done: list[str] = []

    def _work(deal_id: str):
        # Each worker uses an independent connection (psycopg connections are
        # not thread-safe to share). When a conn was passed in we still open a
        # fresh one per deal to keep failures from poisoning a shared txn.
        with get_conn() as wc:
            return generate_for_deal(deal_id, wc)

    if not ids:
        return {"processed": 0, "skipped": 0, "failed": 0, "deal_ids": []}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_work, d): d for d in ids}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                fut.result()
                processed += 1
                done.append(d)
            except Exception as exc:
                failed += 1
                log.warning("summary sync failed for deal %s: %s", d, exc)

    return {"processed": processed, "skipped": 0, "failed": failed, "deal_ids": done}
```

Note: the unit tests monkeypatch `generate_for_deal` and `_linked_deal_ids_needing_summary`, so the `ThreadPoolExecutor`/`get_conn` path is exercised against fakes without a real DB; `test_upsert_demotes_then_inserts` drives the real SQL strings through `FakeConn`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_deal_analysis_sync.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_analysis_service.py tests/services/test_deal_analysis_sync.py
git commit -m "feat(deal-summary): persist metrics + AgentNick narrative; sync_deal_summaries"
```

---

### Task 4: Hook into `assign_deals` + UI-shaped API endpoints

**Files:**
- Modify: `src/services/deal_assignment_service.py:756-770` (the `_run` function)
- Modify: `src/api/routers/deal_summary.py`
- Modify: `src/services/deal_analysis_service.py` (add `to_ui_row`)
- Test: `tests/services/test_deal_analysis_ui_row.py`

**Interfaces:**
- Consumes: `sync_deal_summaries` (Task 3); the metrics row columns (Task 1).
- Produces: `to_ui_row(row: dict) -> dict` mapping a DB row to `{id, supplier, category, value, volume, unitPrice, priceChange, volumeChange, efficiency, items}` (all strings, "–" for NULL); endpoints `GET /deals/{deal_id}/analysis-summary`, `GET /deals/analysis-summary`, `POST /deals/analysis-summary/sync`.

- [ ] **Step 1: Write the failing test for `to_ui_row`**

Create `tests/services/test_deal_analysis_ui_row.py`:

```python
import importlib
mod = importlib.import_module("src.services.deal_analysis_service")


def test_to_ui_row_formats_strings():
    row = {"deal_id": "DEAL-1", "supplier": "Acme", "category": "Electronics",
           "deal_value": 470000, "currency": "GBP", "volume": 105000,
           "unit_price": 4.48, "price_change_pct": 7.5, "volume_change_pct": -1.7,
           "efficiency_score": 18.06,
           "items": [{"name": "Widget A"}, {"name": "Bolt B"}]}
    ui = mod.to_ui_row(row)
    assert ui["id"] == "DEAL-1"
    assert ui["value"] == "£470K"
    assert ui["volume"] == "105,000"
    assert ui["unitPrice"] == "£4.48"
    assert ui["priceChange"] == "+7.5%"
    assert ui["volumeChange"] == "-1.7%"
    assert ui["efficiency"] == "18.06"
    assert ui["items"] == "Widget A, Bolt B"   # string, never a list


def test_to_ui_row_nulls_render_dash():
    row = {"deal_id": "DEAL-2", "supplier": None, "category": None,
           "deal_value": None, "currency": None, "volume": None,
           "unit_price": None, "price_change_pct": None, "volume_change_pct": None,
           "efficiency_score": None, "items": None}
    ui = mod.to_ui_row(row)
    assert ui["value"] == "–"
    assert ui["priceChange"] == "–"
    assert ui["items"] == "–"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_deal_analysis_ui_row.py -v`
Expected: FAIL with `AttributeError: ... 'to_ui_row'`.

- [ ] **Step 3: Add `to_ui_row` to `deal_analysis_service.py`**

Append to `src/services/deal_analysis_service.py`:

```python
_CCY = {"GBP": "£", "USD": "$", "EUR": "€"}


def _money(value, currency) -> str:
    if value is None:
        return "–"
    sym = _CCY.get((currency or "").upper(), (currency + " ") if currency else "")
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{sym}{v / 1_000_000:.1f}M".replace(".0M", "M")
    if abs(v) >= 1_000:
        return f"{sym}{round(v / 1_000)}K"
    return f"{sym}{v:,.2f}"


def _signed_pct(value) -> str:
    if value is None:
        return "–"
    return f"{'+' if value >= 0 else ''}{value:g}%"


def to_ui_row(row: dict) -> dict:
    """Map a bp_analysis_summary row to the exact shape AnalysisSummary.jsx wants.

    Every value is a string (the UI search filter lowercases each field), NULL
    numerics render as the en-dash, and items is a comma-joined product string.
    """
    cur = row.get("currency")
    items = row.get("items")
    if isinstance(items, list):
        names = [i.get("name") for i in items if isinstance(i, dict) and i.get("name")]
        items_str = ", ".join(names) if names else "–"
    else:
        items_str = "–"
    unit = row.get("unit_price")
    return {
        "id": row.get("deal_id") or "–",
        "supplier": row.get("supplier") or "–",
        "category": row.get("category") or "–",
        "value": _money(row.get("deal_value"), cur),
        "volume": f"{float(row['volume']):,.0f}" if row.get("volume") is not None else "–",
        "unitPrice": _money(unit, cur) if unit is not None and float(unit) >= 1000
                     else (f"{_CCY.get((cur or '').upper(), (cur + ' ') if cur else '')}{float(unit):,.2f}"
                           if unit is not None else "–"),
        "priceChange": _signed_pct(row.get("price_change_pct")),
        "volumeChange": _signed_pct(row.get("volume_change_pct")),
        "efficiency": f"{float(row['efficiency_score']):g}" if row.get("efficiency_score") is not None else "–",
        "items": items_str,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_deal_analysis_ui_row.py -v`
Expected: 2 passed.

- [ ] **Step 5: Hook the sync into `assign_deals._run`**

In `src/services/deal_assignment_service.py`, the `_run` function ends (around line 766-770) with `status = reconcile_status(cur)` and a returned dict. Modify it so that after the transaction is set to commit, summaries are synced. Replace the body of `assign_deals` (lines 741-753) so the sync runs **after commit** (the new rows it reads must be visible, and it opens its own connections):

```python
def assign_deals(conn: Any = None, limit: Optional[int] = None) -> dict:
    """Run look-forward, look-back, reconcile, and unassigned-flag passes,
    then generate analysis summaries for any newly Deal_Linked deals."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                r = _run(own.cursor())
                own.commit()
            except Exception:
                own.rollback()
                raise
    else:
        r = _run(conn.cursor())

    # Summary generation is post-link and best-effort: it must never fail the
    # linking pipeline. Runs on its own connections (see sync_deal_summaries).
    try:
        from src.services.deal_analysis_service import sync_deal_summaries
        r["summaries"] = sync_deal_summaries()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("deal summary sync after assign_deals failed: %s", exc)
        r["summaries"] = {"error": str(exc)}
    return r
```

Confirm `log` is defined in that module (it uses `logging`); if not, add `import logging` and `log = logging.getLogger(__name__)` near the top.

- [ ] **Step 6: Add the endpoints to `deal_summary.py`**

In `src/api/routers/deal_summary.py`, add after the existing `get_deal_summary` handler:

```python
@router.get("/analysis-summary", summary="Analysis-summary grid rows (all current deals)")
def get_analysis_summary_all() -> dict[str, Any]:
    from src.services.db import get_conn
    from src.services.deal_analysis_service import to_ui_row
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select deal_id, deal_name, supplier, category, deal_value, currency, "
                "volume, unit_price, price_change_pct, volume_change_pct, "
                "efficiency_score, items, item_count from proc.bp_analysis_summary "
                "where is_current order by generated_at desc")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:
        logger.exception("analysis summary list failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"rows": [to_ui_row(r) for r in rows], "count": len(rows),
            "generated_at": datetime.now(timezone.utc).isoformat()}


@router.get("/{deal_id}/analysis-summary", summary="Analysis-summary grid row for one deal")
def get_analysis_summary(deal_id: str) -> dict[str, Any]:
    from src.services.db import get_conn
    from src.services.deal_analysis_service import to_ui_row
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select deal_id, deal_name, supplier, category, deal_value, currency, "
                "volume, unit_price, price_change_pct, volume_change_pct, "
                "efficiency_score, items, item_count from proc.bp_analysis_summary "
                "where deal_id = %s and is_current limit 1", (deal_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404,
                                    detail=f"No analysis summary for deal_id={deal_id}")
            cols = [d[0] for d in cur.description]
            data = dict(zip(cols, row))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("analysis summary read failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"row": to_ui_row(data),
            "generated_at": datetime.now(timezone.utc).isoformat()}


@router.post("/analysis-summary/sync", summary="Backfill analysis summaries for all linked deals")
def post_analysis_summary_sync() -> dict[str, Any]:
    from src.services.deal_analysis_service import sync_deal_summaries
    try:
        result = sync_deal_summaries()
    except Exception as exc:
        logger.exception("analysis summary sync failed")
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
```

Note: register order matters — FastAPI matches `/deals/analysis-summary` before `/deals/{deal_id}/summary` only if the static route is declared first. Declare `get_analysis_summary_all` (static `/analysis-summary`) **before** `get_analysis_summary` ({deal_id}) as written. The router is already included in `main.py:283`; no main.py change needed.

- [ ] **Step 7: Run the full service test suite**

Run: `pytest tests/services/test_deal_analysis_service.py tests/services/test_deal_analysis_sync.py tests/services/test_deal_analysis_ui_row.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/services/deal_analysis_service.py src/services/deal_assignment_service.py src/api/routers/deal_summary.py tests/services/test_deal_analysis_ui_row.py
git commit -m "feat(deal-summary): hook sync into assign_deals + UI-shaped endpoints"
```

---

### Task 5: Wire the UI grid to the live backend + add Items column

**Files:**
- Modify: `beyond_procwise_ui/src/modules/HomeAnalyse/Analyse/AnalysisSummary.jsx`
- Modify: `beyond_procwise_ui/src/modules/HomeAnalyse/CustomTabs.jsx` (the `fetchAnalysisData` query)

**Interfaces:**
- Consumes: `GET {VITE_AI_API_URL}/deals/{deal_id}/analysis-summary -> {row: {id,supplier,category,value,volume,unitPrice,priceChange,volumeChange,efficiency,items}}`.
- Produces: the rendered 10-column grid.

- [ ] **Step 1: Harden the filter and add the Items column header**

In `AnalysisSummary.jsx`, replace the filter (lines 10-12) so non-string values can't throw:

```jsx
    const filteredTable = analysisSummarytable?.filter((row) =>
        Object.values(row).some((v) => String(v ?? '').toLowerCase().includes(search.toLowerCase()))
    )
```

Replace the header array (line 36) to add `'Items'` as the final column:

```jsx
                                {['Deal ID', 'Supplier', 'Category', 'Deal Value', 'Volume', 'Unit Price', 'Price Change', 'Volume Change', 'Efficiency Score', 'Items'].map((h) => (
```

- [ ] **Step 2: Render the Items cell**

In `AnalysisSummary.jsx`, after the Efficiency Score cell (line 57), add the Items cell using the existing token styling. Also guard the change-cell color logic against missing values:

```jsx
                                    <TableCell sx={{ fontSize: 11, py: 0.8, color: (row.priceChange || '').startsWith('+') ? '#16a34a' : '#64748b' }}>{row.priceChange}</TableCell>
                                    <TableCell sx={{ fontSize: 11, py: 0.8, color: (row.volumeChange || '').startsWith('+') ? '#16a34a' : '#dc2626' }}>{row.volumeChange}</TableCell>
                                    <TableCell sx={{ fontSize: 11, py: 0.8 }}>{row.efficiency}</TableCell>
                                    <TableCell sx={{ fontSize: 11, py: 0.8, maxWidth: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={row.items}>{row.items}</TableCell>
```

(The two change-cell lines replace the existing lines 55-56; the efficiency line is unchanged; the items line is new.)

- [ ] **Step 3: Repoint the data source to the live backend**

In `CustomTabs.jsx`, the analysis grid is fed by `fetchAnalysisData` (lines 25-28) which currently hits `${VITE_API_URL}/analyse`. Change it to read the live local backend and map the response to the array the grid expects. Replace `fetchAnalysisData`:

```jsx
const fetchAnalysisData = (parameters) =>
    axios.get(`${import.meta.env.VITE_AI_API_URL}/deals/${parameters.deal_id}/analysis-summary`)
        .then((res) => ({ analysisSummary: res.data?.row ? [res.data.row] : [] }))
        .catch(() => ({ analysisSummary: [] }))
```

If `CustomTabs.jsx` merges `/analyse`'s other fields (graph, topSuppliers, etc.) into the same query object, keep those calls intact and only swap the `analysisSummary` source — i.e. fetch both and spread: `return { ...legacy, analysisSummary }`. Inspect the actual `fetchAnalysisData`/`useQueries` wiring at edit time and preserve sibling panels; only the `analysisSummary` array changes source.

- [ ] **Step 4: Verify the UI builds**

Run:
```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui && npm run build
```
Expected: build completes with no errors referencing `AnalysisSummary.jsx` or `CustomTabs.jsx`.

- [ ] **Step 5: Commit (in the UI repo)**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui && git add src/modules/HomeAnalyse/Analyse/AnalysisSummary.jsx src/modules/HomeAnalyse/CustomTabs.jsx && git commit -m "feat(analysis): add Items column + read live backend analysis-summary"
```

---

### Task 6: Live end-to-end validation on the running server + `bp_sqldb`

**Files:** none (validation only)

- [ ] **Step 1: Backfill summaries for all currently-linked deals**

Run:
```bash
python -c "from src.services.deal_analysis_service import sync_deal_summaries; print(sync_deal_summaries())"
```
Expected: prints `{'processed': N, 'skipped': 0, 'failed': 0, 'deal_ids': [...]}` where N matches the count of `Deal_Linked` deals.

- [ ] **Step 2: Confirm rows landed in `bp_analysis_summary` for every linked deal**

Run:
```bash
python -c "from src.services.db import get_conn; c=get_conn(); cur=c.cursor(); cur.execute(\"select count(distinct deal_id) from proc.process_monitor where status='Deal_Linked'\"); linked=cur.fetchone()[0]; cur.execute('select count(*) from proc.bp_analysis_summary where is_current'); rows=cur.fetchone()[0]; print('linked deals:', linked, 'summary rows:', rows)"
```
Expected: `summary rows` >= `linked deals` (one current row per linked deal).

- [ ] **Step 3: Confirm narratives landed in `bp_summary`**

Run:
```bash
python -c "from src.services.db import get_conn; c=get_conn(); cur=c.cursor(); cur.execute(\"select count(*) from proc.bp_summary where persona_source='deal_analysis_service' and is_current\"); print('narratives:', cur.fetchone()[0])"
```
Expected: a non-zero count (one per deal that produced a narrative).

- [ ] **Step 4: Start the API server and hit the endpoint**

Start the server (background) per the project's run method, then:
```bash
curl -s http://localhost:8000/deals/analysis-summary | python -m json.tool | head -40
```
Expected: JSON with a `rows` array; each row has string fields `id, supplier, category, value, volume, unitPrice, priceChange, volumeChange, efficiency, items`; NULLs render as `–`; `items` is a comma-joined product string (never a list).

- [ ] **Step 5: Confirm the `Deal_Linked` trigger path end-to-end**

Run `assign_deals` directly and confirm it reports a `summaries` block:
```bash
python -c "from src.services.deal_assignment_service import assign_deals; import json; print(json.dumps(assign_deals().get('summaries'), default=str))"
```
Expected: a `{"processed":...,"failed":...}` block (0 processed is valid on a second run — idempotent; the point is the hook fires without error).

- [ ] **Step 6: Spot-check one deal against its documents (no fabrication)**

Pick one `deal_id` from Step 1's output and compare the persisted metrics to the underlying docs:
```bash
python -c "from src.services.deal_analysis_service import compute_deal_metrics; import json; print(json.dumps(compute_deal_metrics('<DEAL_ID>'), default=str, indent=2))"
```
Expected: `deal_value`/`volume`/`items` match the deal's invoice; absent fields are `null`, not invented.

- [ ] **Step 7: Report results**

Summarize: number of linked deals, summary rows created, a sample UI row, and any deals that failed (with reasons). Do not claim success unless Steps 2-4 passed.
```

## Self-Review

**Spec coverage:**
- §3 data model → Task 1 (table) + Task 2 (computation) ✓
- §4 trigger flow (`compute_deal_metrics`, `sync_deal_summaries`, hook in `assign_deals`) → Tasks 2, 3, 4 Step 5 ✓
- §4 narrative via local AgentNick → `bp_summary` → Task 3 `generate_for_deal` ✓
- §5 API (3 endpoints, UI row shape, signed strings, items-as-string, NULL→"–") → Task 4 ✓
- §6 UI (Items column kept + Efficiency, repoint fetch, filter hardening, tokens) → Task 5 ✓
- §7 error handling (per-deal isolation, narrative best-effort, non-fatal hook) → Task 3 `sync`, Task 4 Step 5 ✓
- §8 tests + live validation → Tasks 2,3,4 tests + Task 6 ✓
- Figma alignment (tokens reused, no Figma file present) → Task 5 reuses exact sx tokens ✓

**Placeholder scan:** No TBD/TODO. `<DEAL_ID>` in Task 6 Step 6 is a deliberate runtime value the operator fills from Step 1 output. Task 5 Step 3 instructs inspecting the actual `CustomTabs.jsx` wiring — this is real guidance, not a placeholder, because the legacy multi-panel response shape is unknown until edit time.

**Type consistency:** `compute_deal_metrics` returns the dict consumed by `upsert_analysis_row`, `to_ui_row`, and `generate_for_deal` — keys match across Tasks 2/3/4. `_store_summary` kwargs match the signature read from `summary_agent.py:135`. `summarize_deal(deal_id, conn)` and `_SUMMARY_MODEL` match `deal_summary.py`. Endpoint response keys (`id,supplier,...,items`) match `AnalysisSummary.jsx` row fields and `to_ui_row` output.
