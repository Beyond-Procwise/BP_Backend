# Deal Linking & Deal-Centric Views — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group every Quote/PO/Invoice into a single authoritative `deal_id` (from `process_monitor` for look-forward, derived for look-back), capture `deal_date` (= order expected delivery date), keep `process_monitor.status` current per document, and expose `bp_` views that feed the deal screens and the executive dashboard.

**Architecture:** A new `deal_assignment_service.py` runs after `trgt-promotion` on the existing `backend_scheduler`. Pure, DB-free helpers (document_id minting, filename matching, deal_date resolution, look-back id derivation) are unit-tested in isolation; DB orchestration is tested with a per-test `_FakeConn` (the `test_deal_summary.py` pattern). The look-back grouping reuses `linking_engine.score_link`. Read-only `bp_` views aggregate the three `_trgt` tables. A one-time backfill applies the DDL, runs assignment over existing rows, and reconciles legacy `DEAL-<po>` keys.

**Tech Stack:** Python 3, psycopg2, PostgreSQL 16 (`bp_sqldb`, schema `proc`), pytest, APScheduler-style `backend_scheduler`.

**Spec:** `docs/superpowers/specs/2026-06-11-deal-linking-design.md`

**Live DB access (for backfill/validation):** load `.env`, map `DB_*`→`PG*`, `sslmode=require`. Connection verified working 2026-06-11.

---

## File Structure

- **Create** `deploy/sql/2026-06-11_deal_linking.sql` — additive DDL: `deal_date` columns, `bp_deal_document_map`, indexes.
- **Create** `deploy/sql/2026-06-11_deal_views.sql` — `bp_deal_documents`, `bp_deal_overview`, `bp_deal_kpis`, `bp_process_monitor_status`.
- **Create** `src/services/deal_assignment_service.py` — the engine (helpers + 3 passes + `assign_deals`).
- **Create** `tests/services/test_deal_assignment_service.py` — unit tests.
- **Modify** `src/services/backend_scheduler.py` — register `deal-assignment` job after `trgt-promotion`.
- **Create** `scripts/backfill_deal_linking.py` — one-time DDL apply + assign + views + report.
- **Modify** `tests/services/test_backend_scheduler.py` (or create) — assert the new job registers when enabled.

Status enum additions (no DDL): `Deal_Linked`, `Deal_Unassigned_Review`, `Deal_Conflict_Review`.

---

## Task 1: Additive DDL (deal_date, bp_deal_document_map, indexes)

**Files:**
- Create: `deploy/sql/2026-06-11_deal_linking.sql`

- [ ] **Step 1: Write the DDL file**

```sql
-- 2026-06-11 Deal linking: additive schema. Idempotent. No destructive edits.
BEGIN;

-- deal_date = order expected delivery date, stamped on every doc in the deal.
ALTER TABLE proc.bp_invoice_trgt          ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_trgt            ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_trgt   ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_invoice_stg           ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_stg             ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_stg    ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_invoice_raw           ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_raw             ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_raw    ADD COLUMN IF NOT EXISTS deal_date DATE;

-- Stable per-document identity within a deal. Supersedes proc.deal_document_id_map.
CREATE TABLE IF NOT EXISTS proc.bp_deal_document_map (
    document_id   VARCHAR PRIMARY KEY,
    deal_id       VARCHAR NOT NULL,
    deal_name     VARCHAR,
    doc_type      VARCHAR NOT NULL,    -- 'quote' | 'po' | 'invoice'
    doc_pk        VARCHAR NOT NULL,    -- quote_id / po_id / invoice_id
    source_file   TEXT,
    assigned_at   TIMESTAMPTZ DEFAULT NOW(),
    assigned_by   VARCHAR DEFAULT 'deal_assignment_service'
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_deal_document_map_natural
    ON proc.bp_deal_document_map (deal_id, doc_type, doc_pk);

CREATE INDEX IF NOT EXISTS ix_bp_invoice_trgt_deal_id        ON proc.bp_invoice_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_deal_id          ON proc.bp_quote_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_purchase_order_trgt_deal_id ON proc.bp_purchase_order_trgt (deal_id);

COMMIT;
```

- [ ] **Step 2: Apply to live DB and verify columns exist**

Run (loads `.env`, maps to PG*):
```bash
python3 scripts/_apply_sql.py deploy/sql/2026-06-11_deal_linking.sql   # helper created in Task 9 backfill; for now apply inline:
```
For this step apply inline with a throwaway snippet:
```bash
python3 - <<'PY'
import os
env={}; [env.__setitem__(*l.strip().split('=',1)) for l in open('.env') if '=' in l and not l.startswith('#')]
import psycopg2
c=psycopg2.connect(host=env['DB_HOST'].strip(),dbname=env['DB_NAME'].strip(),user=env['DB_USER'].strip(),
  password=env['DB_PASSWORD'].strip(),port=env.get('DB_PORT','5432').strip(),sslmode='require')
c.autocommit=True
cur=c.cursor(); cur.execute(open('deploy/sql/2026-06-11_deal_linking.sql').read())
cur.execute("""select count(*) from information_schema.columns
  where table_schema='proc' and column_name='deal_date'
  and table_name in ('bp_invoice_trgt','bp_quote_trgt','bp_purchase_order_trgt')""")
print("deal_date on trgt tables:", cur.fetchone()[0])  # expect 3
cur.execute("select to_regclass('proc.bp_deal_document_map')"); print("map table:", cur.fetchone()[0])
c.close()
PY
```
Expected: `deal_date on trgt tables: 3` and `map table: proc.bp_deal_document_map`.

- [ ] **Step 3: Commit**

```bash
git add deploy/sql/2026-06-11_deal_linking.sql
git commit -m "feat(deal-linking): additive DDL for deal_date, bp_deal_document_map, indexes"
```

---

## Task 2: Pure helpers (DB-free) + unit tests

**Files:**
- Create: `src/services/deal_assignment_service.py`
- Test: `tests/services/test_deal_assignment_service.py`

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# tests/services/test_deal_assignment_service.py
import datetime as dt
from src.services import deal_assignment_service as das


def test_basename_match_ignores_directory_and_case():
    assert das.basename_match("documents/po/DUNCAN PO526702.pdf", "/tmp/x/duncan po526702.PDF")
    assert not das.basename_match("a/INV1.pdf", "a/INV2.pdf")


def test_mint_document_id_is_deterministic_and_typed():
    a = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    b = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    assert a == b == "DEAL_A2026052891::invoice::INV610366"


def test_resolve_deal_date_prefers_po_expected_delivery():
    po = {"expected_delivery_date": dt.date(2024, 10, 9)}
    assert das.resolve_deal_date(po, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 10, 9)


def test_resolve_deal_date_falls_back_to_invoice_line_then_none():
    assert das.resolve_deal_date(None, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 11, 1)
    assert das.resolve_deal_date(None, inv_line_delivery=None) is None


def test_lookback_deal_identity_from_canonical_po():
    assert das.lookback_deal_id("526702") == "DEALV2-526702"
    assert das.lookback_deal_name("Duncan LLC", "526702") == "Duncan LLC — PO 526702"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: FAIL — `module 'deal_assignment_service' has no attribute 'basename_match'` (or ImportError).

- [ ] **Step 3: Implement the helpers**

```python
# src/services/deal_assignment_service.py
"""Assign procurement documents to deals.

deal_id is the backend grouping key. For look-forward documents it is taken
verbatim from proc.process_monitor (user-supplied at upload). For look-back
documents it is derived deterministically from the canonical PO, using the
existing linking_engine score to decide membership. deal_name is the
user-facing label; document_id is a stable per-document id within a deal;
deal_date is the order's expected delivery date stamped on every doc.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)


def basename_match(path_a: Optional[str], path_b: Optional[str]) -> bool:
    """True when two file paths share the same case-insensitive basename."""
    if not path_a or not path_b:
        return False
    ba = os.path.basename(str(path_a)).strip().lower()
    bb = os.path.basename(str(path_b)).strip().lower()
    return bool(ba) and ba == bb


def mint_document_id(deal_id: str, doc_type: str, doc_pk: str) -> str:
    """Deterministic per-document identity within a deal."""
    return f"{deal_id}::{doc_type}::{doc_pk}"


def resolve_deal_date(po_row: Optional[dict], inv_line_delivery=None):
    """deal_date = order expected delivery date.

    Prefer the deal's PO expected_delivery_date; fall back to an invoice line
    delivery_date; else None.
    """
    if po_row and po_row.get("expected_delivery_date"):
        return po_row["expected_delivery_date"]
    return inv_line_delivery


def lookback_deal_id(canonical_po: str) -> str:
    """Versioned derived deal_id (avoids colliding with legacy DEAL-<po>)."""
    return f"DEALV2-{canonical_po}"


def lookback_deal_name(supplier_name: Optional[str], canonical_po: str) -> str:
    supplier = (supplier_name or "Unknown Supplier").strip()
    return f"{supplier} — PO {canonical_po}"
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_assignment_service.py tests/services/test_deal_assignment_service.py
git commit -m "feat(deal-linking): pure helpers for document_id, deal_date, look-back identity"
```

---

## Task 3: Doc-table registry + `_persist_deal` writer

**Files:**
- Modify: `src/services/deal_assignment_service.py`
- Test: `tests/services/test_deal_assignment_service.py`

Reuses `linking_engine._rows` / `_table_columns` for column introspection. A small per-test fake cursor records executed SQL.

- [ ] **Step 1: Write failing test using a recording fake cursor**

```python
# add to tests/services/test_deal_assignment_service.py
class _RecCursor:
    def __init__(self, columns):
        self._columns = columns  # {table: [col,...]}
        self.executed = []       # (sql, params)
        self._result = []
        self.description = None
    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))
        s = sql.lower()
        if "information_schema.columns" in s:
            tbl = params[1]
            self._result = [(c,) for c in self._columns.get(tbl, [])]
            self.description = [("column_name",)]
        else:
            self._result = []
            self.description = None
    def fetchall(self): return list(self._result)
    def fetchone(self): return self._result[0] if self._result else None

class _RecConn:
    def __init__(self, cur): self._cur = cur; self.autocommit = True
    def cursor(self): return self._cur
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


def test_persist_deal_writes_deal_cols_to_stg_and_trgt():
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _RecCursor({"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    das._persist_deal(cur, "invoice", "INV610366",
                      deal_id="DEAL_A2026052891", deal_name="deal_a",
                      document_id="DEAL_A2026052891::invoice::INV610366",
                      deal_date=None)
    updates = [e for e in cur.executed if e[0].lower().startswith("update")]
    assert any("bp_invoice_trgt" in e[0] and "deal_id" in e[0] for e in updates)
    assert any("bp_invoice_stg" in e[0] and "deal_id" in e[0] for e in updates)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_deal_assignment_service.py::test_persist_deal_writes_deal_cols_to_stg_and_trgt -q`
Expected: FAIL — `has no attribute '_persist_deal'`.

- [ ] **Step 3: Implement registry + writer**

```python
# add to src/services/deal_assignment_service.py
from src.services.linking_engine import _table_columns  # column introspection

# doc_type -> (pk, raw, stg, trgt, line_stg, line_trgt)
_DOC = {
    "invoice": ("invoice_id",
                "proc.bp_invoice_raw", "proc.bp_invoice_stg", "proc.bp_invoice_trgt",
                "proc.bp_invoice_line_items_stg", "proc.bp_invoice_line_items_trgt"),
    "quote": ("quote_id",
              "proc.bp_quote_raw", "proc.bp_quote_stg", "proc.bp_quote_trgt",
              "proc.bp_quote_line_items_stg", "proc.bp_quote_line_items_trgt"),
    "po": ("po_id",
           "proc.bp_purchase_order_raw", "proc.bp_purchase_order_stg", "proc.bp_purchase_order_trgt",
           "proc.bp_po_line_items_stg", "proc.bp_po_line_items_trgt"),
}
_DEAL_COLS = ("deal_id", "deal_name", "document_id", "deal_date")


def _persist_deal(cur, doc_type, doc_pk, *, deal_id, deal_name, document_id, deal_date):
    """Write deal columns onto the document's stg/trgt rows + their line items,
    only for columns that actually exist on each table."""
    pk, _raw, stg, trgt, line_stg, line_trgt = _DOC[doc_type]
    values = {"deal_id": deal_id, "deal_name": deal_name,
              "document_id": document_id, "deal_date": deal_date}
    for table in (stg, trgt, line_stg, line_trgt):
        present = [c for c in _DEAL_COLS if c in _table_columns(cur, table)]
        if not present or pk not in _table_columns(cur, table):
            continue
        set_clause = ", ".join(f"{c}=%s" for c in present)
        cur.execute(
            f"update {table} set {set_clause} where {pk}=%s",
            [values[c] for c in present] + [doc_pk])
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_assignment_service.py tests/services/test_deal_assignment_service.py
git commit -m "feat(deal-linking): doc-table registry and _persist_deal writer"
```

---

## Task 4: Look-forward pass + process_monitor status

**Files:**
- Modify: `src/services/deal_assignment_service.py`
- Test: `tests/services/test_deal_assignment_service.py`

- [ ] **Step 1: Write failing test (fake conn returning a monitor row + a matching trgt row)**

```python
# add to tests/services/test_deal_assignment_service.py
class _ScriptCursor:
    """Returns canned rows per matched SQL fragment; records writes & status updates."""
    def __init__(self, script, columns):
        self.script = script            # list of (substr, rows[list[dict]])
        self.columns = columns
        self.executed = []
        self._rows = []; self.description = None
    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        self.executed.append((s, params))
        low = s.lower()
        if "information_schema.columns" in low:
            tbl = params[1]; self._rows = [(c,) for c in self.columns.get(tbl, [])]
            self.description = [("column_name",)]; return
        for substr, rows in self.script:
            if substr.lower() in low:
                self._rows = [tuple(r.values()) for r in rows]
                self.description = [(k,) for k in (rows[0].keys() if rows else [])]
                return
        self._rows = []; self.description = None
    def fetchall(self): return list(self._rows)
    def fetchone(self): return self._rows[0] if self._rows else None


def test_look_forward_links_monitor_deal_to_matching_invoice(monkeypatch):
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    monitor = [{"id": 720, "file_path": "documents/Invoice/THRIVE INV103404 for PO502001.pdf",
                "deal_id": "TEST00120260610104", "deal_name": "Test001",
                "category": "Invoice", "document_type": "pdf"}]
    inv = [{"invoice_id": "103404", "source_file": "x/THRIVE INV103404 for PO502001.pdf"}]
    cur = _ScriptCursor(
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv),
                ("from proc.bp_invoice_trgt", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    conn = _RecConn(cur)
    n = das._look_forward(cur)
    assert n >= 1
    # deal columns written to trgt
    assert any("update proc.bp_invoice_trgt" in e[0].lower() and "deal_id" in e[0].lower()
               for e in cur.executed)
    # status advanced to Deal_Linked
    assert any("update proc.process_monitor set status" in e[0].lower()
               and e[1] and "Deal_Linked" in e[1] for e in cur.executed)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_deal_assignment_service.py::test_look_forward_links_monitor_deal_to_matching_invoice -q`
Expected: FAIL — `has no attribute '_look_forward'`.

- [ ] **Step 3: Implement the look-forward pass**

```python
# add to src/services/deal_assignment_service.py
from src.services.linking_engine import _rows

_DOCTYPE_FROM_HINT = {"invoice": "invoice", "quote": "quote", "po": "po",
                      "purchase_order": "po", "purchaseorder": "po"}


def _set_monitor_status(cur, monitor_id, status):
    cur.execute(
        "update proc.process_monitor set status=%s, lastmodified_date=now() where id=%s",
        (status, monitor_id))


def _candidate_doc_types(category, document_type):
    for hint in (category, document_type):
        key = (hint or "").strip().lower().replace(" ", "")
        if key in _DOCTYPE_FROM_HINT:
            return [_DOCTYPE_FROM_HINT[key]]
    return ["invoice", "quote", "po"]   # unknown hint -> search all


def _look_forward(cur) -> int:
    """Stamp process_monitor deals onto matching extracted documents."""
    linked = 0
    monitors = _rows(cur,
        "select id, file_path, deal_id, deal_name, category, document_type "
        "from proc.process_monitor "
        "where deal_id is not null and deal_id <> ''")
    for m in monitors:
        matched = False
        for dt in _candidate_doc_types(m.get("category"), m.get("document_type")):
            pk, raw, stg, trgt, _ls, _lt = _DOC[dt]
            # find the doc whose source_file basename matches the monitor file_path
            src_rows = _rows(cur, f"select {pk}, source_file from {raw}") or []
            hit = next((r for r in src_rows
                        if basename_match(m["file_path"], r.get("source_file"))), None)
            if not hit:
                # raw may be absent; fall back to trgt by basename of any source col if present
                continue
            doc_pk = hit[pk]
            doc_id = mint_document_id(m["deal_id"], dt, str(doc_pk))
            # resolve deal_date from the deal's PO if this is/has one (best-effort)
            deal_date = _deal_date_for_doc(cur, dt, doc_pk)
            _persist_deal(cur, dt, doc_pk, deal_id=m["deal_id"], deal_name=m["deal_name"],
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, m["deal_id"], m["deal_name"], dt, doc_pk, doc_id,
                                 m["file_path"])
            matched = True
            linked += 1
        _set_monitor_status(cur, m["id"], "Deal_Linked" if matched
                            else "Deal_Unassigned_Review")
    return linked
```

Add the two helpers it calls:

```python
def _upsert_document_map(cur, deal_id, deal_name, doc_type, doc_pk, document_id, source_file):
    cur.execute(
        "insert into proc.bp_deal_document_map "
        "(document_id, deal_id, deal_name, doc_type, doc_pk, source_file) "
        "values (%s,%s,%s,%s,%s,%s) "
        "on conflict (document_id) do update set "
        "deal_id=excluded.deal_id, deal_name=excluded.deal_name, "
        "doc_type=excluded.doc_type, doc_pk=excluded.doc_pk, source_file=excluded.source_file",
        (document_id, deal_id, deal_name, doc_type, str(doc_pk), source_file))


def _deal_date_for_doc(cur, doc_type, doc_pk):
    """Resolve the order's expected delivery date for the deal this doc belongs to."""
    from src.services.linking_engine import _PO, _norm_po
    # po doc: its own expected_delivery_date
    if doc_type == "po":
        r = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where po_id=%s", (doc_pk,))
        return r[0]["expected_delivery_date"] if r else None
    # invoice/quote: find parent PO via po_id on the doc, then its delivery date
    pk, _raw, stg, trgt, _ls, _lt = _DOC[doc_type]
    rr = _rows(cur, f"select po_id from {trgt} where {pk}=%s", (doc_pk,)) or \
         _rows(cur, f"select po_id from {stg} where {pk}=%s", (doc_pk,))
    po_ref = rr[0]["po_id"] if rr else None
    if not po_ref:
        return None
    cond = "regexp_replace(regexp_replace(lower(po_id),'[^a-z0-9]','','g'),'^po','')"
    pr = _rows(cur, f"select expected_delivery_date from {_PO['trgt']} where {cond}=%s",
               (_norm_po(po_ref),))
    return pr[0]["expected_delivery_date"] if pr else None
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_assignment_service.py tests/services/test_deal_assignment_service.py
git commit -m "feat(deal-linking): look-forward pass stamps monitor deals onto documents + status"
```

---

## Task 5: Look-back pass (canonical PO + multi-signal membership)

**Files:**
- Modify: `src/services/deal_assignment_service.py`
- Test: `tests/services/test_deal_assignment_service.py`

- [ ] **Step 1: Write failing test**

```python
# add to tests/services/test_deal_assignment_service.py
def test_look_back_groups_invoice_under_canonical_po_deal(monkeypatch):
    # invoice with no deal, references PO 502001; a PO exists in trgt
    inv = [{"invoice_id": "103404", "po_id": "PO502001", "supplier_id": "SUP-Thrive",
            "deal_id": None}]
    po = [{"po_id": "502001", "supplier_id": "SUP-Thrive", "supplier_name": "Thrive Ltd",
           "expected_delivery_date": None, "deal_id": None}]
    columns = {"bp_invoice_stg": ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"],
               "bp_invoice_trgt": ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]}
    cur = _ScriptCursor(
        script=[("from proc.bp_invoice_trgt", inv),
                ("from proc.bp_purchase_order_trgt", po)],
        columns=columns)
    # force the link score to pass
    monkeypatch.setattr(das, "score_link", lambda *a, **k: {"F": 95.0})
    n = das._look_back(cur)
    assert n >= 1
    assert any("dealv2-502001" in (str(e[1]).lower() if e[1] else "") for e in cur.executed)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_deal_assignment_service.py::test_look_back_groups_invoice_under_canonical_po_deal -q`
Expected: FAIL — `has no attribute '_look_back'`.

- [ ] **Step 3: Implement the look-back pass**

```python
# add to src/services/deal_assignment_service.py
from src.services.linking_engine import score_link, _norm_po, _PO

MIN_LINK_SCORE = float(os.getenv("PROMOTE_MIN_LINK_SCORE", "80"))


def _unlinked_docs(cur, doc_type):
    pk, _raw, stg, trgt, _ls, _lt = _DOC[doc_type]
    return _rows(cur, f"select * from {trgt} where deal_id is null or deal_id = ''")


def _look_back(cur) -> int:
    """Group deal-less docs under their canonical PO deal when the link score passes."""
    linked = 0
    for doc_type in ("invoice", "quote"):
        pk = _DOC[doc_type][0]
        for row in _unlinked_docs(cur, doc_type):
            po_ref = row.get("po_id")
            npo = _norm_po(po_ref)
            if not npo:
                continue   # no-PO doc -> left for review by orchestrator
            cond = "regexp_replace(regexp_replace(lower(po_id),'[^a-z0-9]','','g'),'^po','')"
            pos = _rows(cur, f"select * from {_PO['trgt']} where {cond}=%s", (npo,))
            if not pos:
                continue
            po = pos[0]
            profile = "invoice_po" if doc_type == "invoice" else "quote_po"
            link = score_link(row, po, profile)
            if link["F"] < MIN_LINK_SCORE:
                continue
            supplier = po.get("supplier_name") or po.get("supplier_id")
            deal_id = lookback_deal_id(npo)
            deal_name = lookback_deal_name(supplier, npo)
            doc_pk = row[pk]
            doc_id = mint_document_id(deal_id, doc_type, str(doc_pk))
            deal_date = po.get("expected_delivery_date")
            _persist_deal(cur, doc_type, doc_pk, deal_id=deal_id, deal_name=deal_name,
                          document_id=doc_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, doc_type, doc_pk, doc_id,
                                 row.get("source_file"))
            # also stamp the PO itself into the same derived deal
            po_doc_id = mint_document_id(deal_id, "po", str(po["po_id"]))
            _persist_deal(cur, "po", po["po_id"], deal_id=deal_id, deal_name=deal_name,
                          document_id=po_doc_id, deal_date=deal_date)
            _upsert_document_map(cur, deal_id, deal_name, "po", po["po_id"], po_doc_id, None)
            linked += 1
    return linked
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_assignment_service.py tests/services/test_deal_assignment_service.py
git commit -m "feat(deal-linking): look-back pass groups docs under canonical PO deal"
```

---

## Task 6: Reconciliation pass + `assign_deals` orchestrator

**Files:**
- Modify: `src/services/deal_assignment_service.py`
- Test: `tests/services/test_deal_assignment_service.py`

- [ ] **Step 1: Write failing test for orchestrator wiring**

```python
# add to tests/services/test_deal_assignment_service.py
def test_assign_deals_runs_all_passes_and_returns_counts(monkeypatch):
    calls = []
    monkeypatch.setattr(das, "_look_forward", lambda cur: calls.append("fwd") or 2)
    monkeypatch.setattr(das, "_look_back", lambda cur: calls.append("back") or 1)
    monkeypatch.setattr(das, "_reconcile_legacy", lambda cur: calls.append("rec") or 3)
    monkeypatch.setattr(das, "_flag_unassigned", lambda cur: calls.append("flag") or 4)
    cur = _ScriptCursor(script=[], columns={})
    conn = _RecConn(cur)
    result = das.assign_deals(conn=conn)
    assert result == {"forward_linked": 2, "backward_linked": 1,
                      "reconciled": 3, "unassigned_review": 4}
    assert calls == ["fwd", "back", "rec", "flag"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_deal_assignment_service.py::test_assign_deals_runs_all_passes_and_returns_counts -q`
Expected: FAIL — `has no attribute 'assign_deals'`.

- [ ] **Step 3: Implement reconcile, flag, and orchestrator**

```python
# add to src/services/deal_assignment_service.py
from src.services.db import get_conn


def _reconcile_legacy(cur) -> int:
    """Rewrite legacy DEAL-<po> keys (no monitor deal) to the derived DEALV2-<po>
    form so all deal_ids share one scheme. Authoritative monitor deals already
    overwrote their rows in the look-forward pass."""
    reconciled = 0
    for doc_type in ("invoice", "quote", "po"):
        pk, _raw, _stg, trgt, _ls, _lt = _DOC[doc_type]
        rows = _rows(cur,
            f"select {pk}, deal_id, po_id from {trgt} "
            f"where deal_id like 'DEAL-%'")
        for r in rows:
            npo = _norm_po(r.get("po_id")) or r["deal_id"].split("-", 1)[-1]
            new_id = lookback_deal_id(npo)
            if new_id == r["deal_id"]:
                continue
            cur.execute(f"update {trgt} set deal_id=%s where {pk}=%s", (new_id, r[pk]))
            reconciled += 1
    return reconciled


def _flag_unassigned(cur) -> int:
    """Mark monitor rows whose document still has no deal as review-needed."""
    cur.execute(
        "update proc.process_monitor set status='Deal_Unassigned_Review', "
        "lastmodified_date=now() "
        "where (deal_id is null or deal_id='') "
        "and status not in ('Extraction_Failed','Deal_Unassigned_Review') "
        "returning id")
    try:
        return len(cur.fetchall())
    except Exception:
        return 0


def assign_deals(conn: Any = None, limit: Optional[int] = None) -> dict:
    """Run look-forward, look-back, reconcile, and unassigned-flag passes."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                r = _run(own.cursor())
                own.commit()
                return r
            except Exception:
                own.rollback()
                raise
    return _run(conn.cursor())


def _run(cur) -> dict:
    fwd = _look_forward(cur)
    back = _look_back(cur)
    rec = _reconcile_legacy(cur)
    flag = _flag_unassigned(cur)
    return {"forward_linked": fwd, "backward_linked": back,
            "reconciled": rec, "unassigned_review": flag}
```

> Note: the orchestrator test monkeypatches the four passes, so `_run` must call them by module attribute. Adjust `_run` to call `das`-level names (already module-level) — monkeypatch on the module replaces them, and Python resolves `_look_forward` as a global, so the patch is honored.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_deal_assignment_service.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_assignment_service.py tests/services/test_deal_assignment_service.py
git commit -m "feat(deal-linking): reconciliation pass + assign_deals orchestrator"
```

---

## Task 7: `bp_` views

**Files:**
- Create: `deploy/sql/2026-06-11_deal_views.sql`

- [ ] **Step 1: Write the views SQL**

```sql
-- 2026-06-11 Deal-centric read views. CREATE OR REPLACE, read-only.
-- 7a. one row per document in a deal
CREATE OR REPLACE VIEW proc.bp_deal_documents AS
SELECT deal_id, deal_name, document_id, 'quote'::text AS doc_type,
       quote_id AS doc_pk, quote_id AS doc_number, quote_date AS doc_date, deal_date,
       supplier_id, NULL::text AS supplier_name, buyer_id, currency,
       total_amount AS amount, total_amount_incl_tax AS amount_incl_tax,
       converted_amount_usd, country, region, confidence_score,
       status, created_date
FROM proc.bp_quote_trgt
UNION ALL
SELECT deal_id, deal_name, document_id, 'po', po_id, po_id, order_date, deal_date,
       supplier_id, supplier_name, buyer_id, currency,
       total_amount, total_amount_incl_tax, converted_amount_usd,
       ship_to_country, delivery_region, confidence_score, po_status, created_date
FROM proc.bp_purchase_order_trgt
UNION ALL
SELECT deal_id, deal_name, document_id, 'invoice', invoice_id, invoice_id, invoice_date, deal_date,
       supplier_id, NULL, buyer_id, currency,
       invoice_amount, invoice_total_incl_tax, converted_amount_usd,
       country, region, confidence_score, invoice_status, created_date
FROM proc.bp_invoice_trgt;

-- 7b. one row per deal
CREATE OR REPLACE VIEW proc.bp_deal_overview AS
WITH d AS (SELECT * FROM proc.bp_deal_documents WHERE deal_id IS NOT NULL AND deal_id <> '')
SELECT
  deal_id,
  max(deal_name) AS deal_name,
  max(supplier_id) AS supplier_id,
  max(supplier_name) AS supplier_name,
  max(buyer_id) AS buyer_id,
  max(deal_date) AS deal_date,
  min(doc_date) AS first_activity_date,
  max(doc_date) AS last_activity_date,
  count(*) FILTER (WHERE doc_type='quote')   AS quote_count,
  count(*) FILTER (WHERE doc_type='po')      AS po_count,
  count(*) FILTER (WHERE doc_type='invoice') AS invoice_count,
  sum(amount) FILTER (WHERE doc_type='quote')   AS quote_total,
  sum(amount) FILTER (WHERE doc_type='po')      AS po_total,
  sum(amount) FILTER (WHERE doc_type='invoice') AS invoice_total,
  max(currency) AS currency,
  sum(converted_amount_usd) AS converted_total_usd,
  (count(*) FILTER (WHERE doc_type='quote')>0
   AND count(*) FILTER (WHERE doc_type='po')>0
   AND count(*) FILTER (WHERE doc_type='invoice')>0) AS three_way_match,
  CASE WHEN sum(amount) FILTER (WHERE doc_type='po') > 0
       THEN round(100.0*abs(coalesce(sum(amount) FILTER (WHERE doc_type='invoice'),0)
              - sum(amount) FILTER (WHERE doc_type='po'))
              / nullif(sum(amount) FILTER (WHERE doc_type='po'),0), 2)
       END AS price_variance_pct,
  (max(doc_date) FILTER (WHERE doc_type='po')
     - min(doc_date) FILTER (WHERE doc_type='quote')) AS cycle_days_quote_to_po,
  (max(doc_date) FILTER (WHERE doc_type='invoice')
     - min(doc_date) FILTER (WHERE doc_type='po')) AS cycle_days_po_to_invoice
FROM d GROUP BY deal_id;

-- 7c. executive dashboard single-row feed
CREATE OR REPLACE VIEW proc.bp_deal_kpis AS
SELECT
  count(*) AS deal_count,
  sum(invoice_total) AS invoiced_total,
  round(avg(cycle_days_po_to_invoice)::numeric, 1) AS avg_cycle_days,
  round(avg(cycle_days_quote_to_po)::numeric, 1)   AS avg_days_to_po,
  round(100.0*count(*) FILTER (WHERE three_way_match)/nullif(count(*),0),0) AS three_way_match_pct,
  round(avg(price_variance_pct)::numeric, 1) AS price_variance_pct,
  (SELECT count(*) FROM (
      SELECT supplier_id, amount, invoice_date FROM proc.bp_invoice_trgt
      GROUP BY supplier_id, amount, invoice_date HAVING count(*)>1) dup) AS duplicate_count,
  sum(invoice_total) FILTER (WHERE po_count=0) AS no_po_spend
FROM proc.bp_deal_overview;

-- 7d. document processing status board
CREATE OR REPLACE VIEW proc.bp_process_monitor_status AS
SELECT id, file_path, document_type AS doc_type, category, deal_id, deal_name,
       status, start_ts, end_ts, lastmodified_date
FROM proc.process_monitor;
```

- [ ] **Step 2: Apply and verify all four views resolve**

```bash
python3 - <<'PY'
import os
env={}; [env.__setitem__(*l.strip().split('=',1)) for l in open('.env') if '=' in l and not l.startswith('#')]
import psycopg2
c=psycopg2.connect(host=env['DB_HOST'].strip(),dbname=env['DB_NAME'].strip(),user=env['DB_USER'].strip(),
  password=env['DB_PASSWORD'].strip(),port=env.get('DB_PORT','5432').strip(),sslmode='require'); c.autocommit=True
cur=c.cursor(); cur.execute(open('deploy/sql/2026-06-11_deal_views.sql').read())
for v in ('bp_deal_documents','bp_deal_overview','bp_deal_kpis','bp_process_monitor_status'):
    cur.execute(f"select count(*) from proc.{v}"); print(v, cur.fetchone()[0])
c.close()
PY
```
Expected: each view returns a count without error.

- [ ] **Step 3: Commit**

```bash
git add deploy/sql/2026-06-11_deal_views.sql
git commit -m "feat(deal-linking): bp_ deal views (documents, overview, kpis, status)"
```

---

## Task 8: Scheduler registration

**Files:**
- Modify: `src/services/backend_scheduler.py` (after `_register_trgt_promotion_job`, ~line 295)
- Test: `tests/services/test_backend_scheduler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/services/test_backend_scheduler.py  (create if absent)
import os
from datetime import timedelta
from src.services.backend_scheduler import BackendScheduler


def test_deal_assignment_job_registers_when_enabled(monkeypatch):
    monkeypatch.setenv("DEAL_ASSIGNMENT_ENABLED", "1")
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_deal_assignment_job()
    assert BackendScheduler.DEAL_ASSIGNMENT_JOB_NAME in s._jobs


def test_deal_assignment_job_disabled(monkeypatch):
    monkeypatch.setenv("DEAL_ASSIGNMENT_ENABLED", "0")
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_deal_assignment_job()
    assert BackendScheduler.DEAL_ASSIGNMENT_JOB_NAME not in s._jobs
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/services/test_backend_scheduler.py -q`
Expected: FAIL — `DEAL_ASSIGNMENT_JOB_NAME` / `_register_deal_assignment_job` missing.

- [ ] **Step 3: Implement registration**

In `backend_scheduler.py`, add the class constant near `TRGT_PROMOTION_JOB_NAME`:
```python
    DEAL_ASSIGNMENT_JOB_NAME = "deal-assignment"
```
Add a call in `_register_default_jobs` after `self._register_trgt_promotion_job()`:
```python
        self._register_deal_assignment_job()
```
Add the methods after `_run_trgt_promotion`:
```python
    def _register_deal_assignment_job(self) -> None:
        """Assign documents to deals after _stg->_trgt promotion (look-forward +
        look-back + reconcile). Toggle DEAL_ASSIGNMENT_ENABLED (default on),
        interval DEAL_ASSIGNMENT_INTERVAL_MINUTES (default 15)."""
        import os
        if os.environ.get("DEAL_ASSIGNMENT_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("deal assignment job disabled by DEAL_ASSIGNMENT_ENABLED")
            return
        if self.DEAL_ASSIGNMENT_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("DEAL_ASSIGNMENT_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.DEAL_ASSIGNMENT_JOB_NAME,
            self._run_deal_assignment,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _run_deal_assignment(self) -> None:
        """Run the deal assignment passes."""
        try:
            from src.services.deal_assignment_service import assign_deals
            result = assign_deals()
            logger.info("deal assignment completed: %s", result)
        except Exception:
            logger.exception("deal assignment job failed")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/services/test_backend_scheduler.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/backend_scheduler.py tests/services/test_backend_scheduler.py
git commit -m "feat(deal-linking): register deal-assignment scheduler job after trgt-promotion"
```

---

## Task 9: Backfill script + live validation

**Files:**
- Create: `scripts/backfill_deal_linking.py`

- [ ] **Step 1: Write the backfill script**

```python
#!/usr/bin/env python3
"""One-time deal-linking backfill: apply DDL, run assignment over existing rows,
create views, print a reconciliation report. Idempotent; no deletes."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
    for k_pg, k_db in [("PGHOST","DB_HOST"),("PGDATABASE","DB_NAME"),("PGUSER","DB_USER"),
                       ("PGPASSWORD","DB_PASSWORD"),("PGPORT","DB_PORT")]:
        if env.get(k_db): os.environ.setdefault(k_pg, env[k_db])
    os.environ.setdefault("PGSSLMODE", "require")

def main():
    _env()
    import psycopg2
    conn = psycopg2.connect(host=os.environ["PGHOST"], dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT","5432"), sslmode="require")
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(open(os.path.join(ROOT, "deploy/sql/2026-06-11_deal_linking.sql")).read())
    conn.commit()
    from src.services.deal_assignment_service import assign_deals
    result = assign_deals(conn=conn); conn.commit()
    cur.execute(open(os.path.join(ROOT, "deploy/sql/2026-06-11_deal_views.sql")).read()); conn.commit()
    print("ASSIGN:", result)
    for v in ("bp_deal_overview", "bp_deal_kpis"):
        cur.execute(f"select count(*) from proc.{v}"); print(v, cur.fetchone()[0])
    cur.execute("select deal_id, deal_name, quote_count, po_count, invoice_count, deal_date "
                "from proc.bp_deal_overview order by deal_id")
    for r in cur.fetchall(): print("  DEAL", r)
    conn.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full unit suite first**

Run: `pytest tests/services/test_deal_assignment_service.py tests/services/test_backend_scheduler.py -q`
Expected: PASS.

- [ ] **Step 3: Run the backfill against live `bp_sqldb`**

Run: `python3 scripts/backfill_deal_linking.py`
Expected: prints `ASSIGN: {...}` with non-zero `forward_linked`/`reconciled`, then per-deal rows where known chains (PO526702 Duncan, PO502001 Thrive) show correct quote/po/invoice counts.

- [ ] **Step 4: Validate KPIs are non-null**

```bash
python3 - <<'PY'
import os
env={}; [env.__setitem__(*l.strip().split('=',1)) for l in open('.env') if '=' in l and not l.startswith('#')]
import psycopg2
c=psycopg2.connect(host=env['DB_HOST'].strip(),dbname=env['DB_NAME'].strip(),user=env['DB_USER'].strip(),
  password=env['DB_PASSWORD'].strip(),port=env.get('DB_PORT','5432').strip(),sslmode='require')
cur=c.cursor(); cur.execute("select * from proc.bp_deal_kpis")
print([d[0] for d in cur.description]); print(cur.fetchone()); c.close()
PY
```
Expected: a populated KPI row (three_way_match_pct, avg_cycle_days, duplicate_count, etc.).

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_deal_linking.py
git commit -m "feat(deal-linking): one-time backfill script + live validation"
```

---

## Task 10: Full regression + summary

- [ ] **Step 1: Run the relevant suites**

Run: `pytest tests/services/ -q -k "deal or scheduler or linking or promotion"`
Expected: PASS (no regressions in neighbouring suites).

- [ ] **Step 2: Confirm no source data was mutated destructively**

```bash
python3 - <<'PY'
import os
env={}; [env.__setitem__(*l.strip().split('=',1)) for l in open('.env') if '=' in l and not l.startswith('#')]
import psycopg2
c=psycopg2.connect(host=env['DB_HOST'].strip(),dbname=env['DB_NAME'].strip(),user=env['DB_USER'].strip(),
  password=env['DB_PASSWORD'].strip(),port=env.get('DB_PORT','5432').strip(),sslmode='require')
cur=c.cursor()
for t in ('bp_invoice_trgt','bp_quote_trgt','bp_purchase_order_trgt'):
    cur.execute(f"select count(*), count(deal_id), count(deal_date) from proc.{t}")
    print(t, cur.fetchone())
c.close()
PY
```
Expected: row counts unchanged from baseline (2 / 5 / 7), with `deal_id` and `deal_date` now populated.

- [ ] **Step 3: Write the change summary** into `docs/superpowers/plans/2026-06-11-deal-linking-SUMMARY.md` (tables created/changed, views, service, scheduler, backfill results).

---

## Self-Review

- **Spec coverage:** identity model (T2–T6), deal_date (T1,T4,T5), document_id (T2,T4,T5), schema (T1), look-forward (T4), look-back (T5), reconciliation (T6), status lifecycle (T4,T6), views incl. exec KPIs (T7), scheduler (T8), backfill (T9), tests throughout. ✔
- **Placeholder scan:** every code step contains full code; no TBD/TODO. ✔
- **Type consistency:** `_DOC` tuple order `(pk, raw, stg, trgt, line_stg, line_trgt)` used identically in T3–T6; `assign_deals`/`_run` names consistent; helper names (`basename_match`, `mint_document_id`, `resolve_deal_date`, `lookback_deal_id/name`, `_persist_deal`, `_upsert_document_map`, `_deal_date_for_doc`, `_look_forward`, `_look_back`, `_reconcile_legacy`, `_flag_unassigned`) consistent across tasks. ✔
- **Open follow-up:** deal list/detail view columns inferred (exec dashboard only legible screen); reconcile against Figma when accessible.
