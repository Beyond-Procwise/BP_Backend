# Document Analysis Report Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After uploading documents from Home → "Find an opportunity", land the user on a live-data Document Analysis Report; a new analysis is a *draft* until Promoted into a tracked Pipeline deal.

**Architecture:** Three sequenced sub-plans across three repos. **A) BP_Backend** (owns `proc.*`): new `proc.bp_deal` draft/tracked header + promote/save endpoints + opportunity↔deal linkage. **B) Gateway** (`beyond-procwaise-Api`): create the draft row on upload, filter Pipeline to tracked deals, expose a drafts list. **C) UI** (`beyond_procwise_ui`, branch `spendiq-ui`): route new analyses to a new `engine.js` `analysis-report` view, render five live-data tabs reusing existing SpendIQ primitives, wire the six-tab Deal-Detail overlay + Promote/Save actions.

**Tech Stack:** Python 3 / FastAPI / psycopg2 (BP_Backend); NestJS / TypeORM / Postgres (gateway); React 19 / Vite / imperative `engine.js` HTML-string views (UI). Tests: pytest (BP_Backend), jest (gateway), vitest (UI).

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-15-document-analysis-report-design.md` — authoritative; this plan implements Phase 1 only.
- **engine.js-native UI:** no new React component tree; reuse `.hscroll`/`kpiScroll`, `rag()`, `case-grp`, `ac-grid`, `.cmx`, `dealView()`. Tokens already match the mockup (`--brand:#2f6df6`, Inter).
- **No fabrication / fail-closed:** every KPI/status is a deterministic backend read traceable to source docs; missing data surfaces as an exception ("Needs validation" / "Not measured yet"), never a clean default. LLM prose (executive summary) is labelled "✨ Agent draft · AI-generated · review before acting".
- **Never write `deal_id` in app code for this feature** — Promote flips flags only; it must not mint or reassign `deal_id`.
- **Never modify source extraction data.** Linkage backfill sets only `bp_opportunity.deal_id` (a currently-NULL column).
- **Branch discipline:** all work on the current dev branch of each repo (BP_Backend `Development`, UI `spendiq-ui`); never push to `main`.
- **DB naming:** new tables `bp_*`, indexes `ix_bp_*`.
- **Migrations are applied manually via `psql`** (no runtime runner); each new `.sql` gets a text-assertion test under `tests/sql/`.
- **Phase 2 (NOT in this plan):** floating Ask/email agent panels, real Split-PO/Approval-bypass/Duplicate-invoice detectors, Export CSV/PDF, in-report "Add documents".

---

# SUB-PLAN A — BP_Backend (foundation)

Repo: `/home/muthu/PycharmProjects/BP_Backend`. Run tests with `pytest`.

### Task A1: `proc.bp_deal` draft/tracked header + backfill

**Files:**
- Create: `deploy/sql/2026-07-15_bp_deal.sql`
- Test: `tests/sql/test_bp_deal_sql.py`

**Interfaces:**
- Produces: table `proc.bp_deal(deal_id text PK, is_tracked bool, is_saved_reference bool, tracked_at timestamptz, created_at timestamptz, updated_at timestamptz)`; index `ix_bp_deal_is_tracked`. Backfill inserts every existing `bp_deal_overview.deal_id` as `is_tracked=true`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sql/test_bp_deal_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-07-15_bp_deal.sql").read_text().lower()

def test_table_and_columns_present():
    assert "create table if not exists proc.bp_deal" in SQL
    for col in ("deal_id", "is_tracked", "is_saved_reference",
                "tracked_at", "created_at", "updated_at"):
        assert col in SQL, f"missing column {col}"

def test_defaults_are_draft():
    assert "is_tracked" in SQL and "default false" in SQL

def test_index_present():
    assert "ix_bp_deal_is_tracked" in SQL

def test_backfill_marks_existing_tracked():
    # existing deals must be inserted as tracked so nothing vanishes from Pipeline
    assert "insert into proc.bp_deal" in SQL
    assert "bp_deal_overview" in SQL
    assert "true" in SQL  # is_tracked=true for existing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sql/test_bp_deal_sql.py -v`
Expected: FAIL — file `deploy/sql/2026-07-15_bp_deal.sql` does not exist.

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-07-15_bp_deal.sql
-- 2026-07-15 Deal header: draft vs tracked lifecycle for uploaded analyses.
-- A new upload starts as a DRAFT (is_tracked=false); Promote flips it tracked.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_deal (
    deal_id             text PRIMARY KEY,
    is_tracked          boolean NOT NULL DEFAULT false,
    is_saved_reference  boolean NOT NULL DEFAULT false,
    tracked_at          timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_deal_is_tracked ON proc.bp_deal (is_tracked);

-- One-time backfill: every deal that already exists stays visible in Pipeline.
INSERT INTO proc.bp_deal (deal_id, is_tracked, tracked_at)
SELECT deal_id, true, now()
  FROM proc.bp_deal_overview
 WHERE deal_id IS NOT NULL
ON CONFLICT (deal_id) DO NOTHING;

COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sql/test_bp_deal_sql.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Apply the migration to live `bp_sqldb`**

Run (loads `.env` PG vars first per repo convention):
```bash
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 \
  -f deploy/sql/2026-07-15_bp_deal.sql
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -c \
  "select count(*) filter (where is_tracked) tracked, count(*) total from proc.bp_deal;"
```
Expected: `total` equals the current distinct-deal count in `bp_deal_overview`; `tracked == total`.

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-07-15_bp_deal.sql tests/sql/test_bp_deal_sql.py
git commit -m "feat(deal): proc.bp_deal draft/tracked header + backfill existing as tracked"
```

---

### Task A2: `bp_deal` store — promote / save-reference writes

**Files:**
- Create: `src/services/deal_lifecycle.py`
- Test: `tests/services/test_deal_lifecycle.py`

**Interfaces:**
- Consumes: `src.services.db.get_conn` (contextmanager; autocommit True by default, set False for writes) — mirror `opportunity_store.set_stage`'s `conn=None` injection idiom.
- Produces:
  - `promote_deal(deal_id: str, conn=None) -> None` — upserts `bp_deal` to `is_tracked=true, is_saved_reference=false, tracked_at=now()`, then advances linked opportunities `identified -> negotiation`.
  - `save_reference(deal_id: str, conn=None) -> None` — upserts `bp_deal` to `is_saved_reference=true` without changing `is_tracked`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_deal_lifecycle.py
from src.services.deal_lifecycle import promote_deal, save_reference

class _Cur:
    def __init__(self): self.calls = []
    def execute(self, sql, params=()): self.calls.append((" ".join(sql.lower().split()), params))

class _Conn:
    def __init__(self): self._cur = _Cur(); self.autocommit = True; self.committed = False
    def cursor(self): return self._cur
    def commit(self): self.committed = True
    def rollback(self): pass

def test_promote_sets_tracked_and_advances_opps():
    c = _Conn()
    promote_deal("ACME2026071501", conn=c)
    sqls = [s for s, _ in c._cur.calls]
    assert any("insert into proc.bp_deal" in s and "is_tracked" in s for s in sqls)
    assert any("update proc.bp_opportunity set stage='negotiation'" in s
               and "where deal_id" in s and "stage='identified'" in s for s in sqls)
    # deal_id passed as a bound param, never interpolated
    assert any("ACME2026071501" in str(p) for _, p in c._cur.calls)

def test_save_reference_keeps_draft():
    c = _Conn()
    save_reference("ACME2026071501", conn=c)
    s = " ".join(x for x, _ in c._cur.calls)
    assert "is_saved_reference" in s
    assert "is_tracked=true" not in s  # must NOT promote
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_deal_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: src.services.deal_lifecycle`.

- [ ] **Step 3: Write the store**

```python
# src/services/deal_lifecycle.py
"""Draft <-> tracked lifecycle writes for proc.bp_deal (see 2026-07-15 spec)."""
from __future__ import annotations
from typing import Any
from src.services.db import get_conn


def _promote(c: Any, deal_id: str) -> None:
    cur = c.cursor()
    cur.execute(
        "insert into proc.bp_deal (deal_id, is_tracked, is_saved_reference, tracked_at) "
        "values (%s, true, false, now()) "
        "on conflict (deal_id) do update set "
        "is_tracked=true, is_saved_reference=false, tracked_at=now(), updated_at=now()",
        (str(deal_id),))
    # Advance any linked opportunities out of 'identified' when the deal is committed.
    cur.execute(
        "update proc.bp_opportunity set stage='negotiation', stage_updated_at=now(), "
        "updated_at=now() where deal_id=%s and stage='identified'",
        (str(deal_id),))


def _save_ref(c: Any, deal_id: str) -> None:
    cur = c.cursor()
    cur.execute(
        "insert into proc.bp_deal (deal_id, is_tracked, is_saved_reference) "
        "values (%s, false, true) "
        "on conflict (deal_id) do update set is_saved_reference=true, updated_at=now()",
        (str(deal_id),))


def _with_txn(fn, deal_id: str, conn: Any) -> None:
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                fn(own, deal_id); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        fn(conn, deal_id)


def promote_deal(deal_id: str, conn: Any = None) -> None:
    _with_txn(_promote, deal_id, conn)


def save_reference(deal_id: str, conn: Any = None) -> None:
    _with_txn(_save_ref, deal_id, conn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_deal_lifecycle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/deal_lifecycle.py tests/services/test_deal_lifecycle.py
git commit -m "feat(deal): promote_deal / save_reference lifecycle store writes"
```

---

### Task A3: `POST /deals/{id}/promote` and `/save-reference` endpoints

**Files:**
- Modify: `src/api/routers/deal_summary.py` (router `prefix="/deals"`; add two handlers alongside the existing `/{deal_id}/reconcile` at ~line 133)
- Test: `tests/api/test_deal_lifecycle_endpoints.py`

**Interfaces:**
- Consumes: `promote_deal`, `save_reference` from Task A2.
- Produces: `POST /deals/{deal_id}/promote` → `{"status":"ok","deal_id":...,"is_tracked":true}`; `POST /deals/{deal_id}/save-reference` → `{"status":"ok","deal_id":...,"is_saved_reference":true}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_deal_lifecycle_endpoints.py
import importlib
mod = importlib.import_module("src.api.routers.deal_summary")

def test_promote_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(mod, "promote_deal", lambda deal_id: called.setdefault("promote", deal_id))
    res = mod.post_promote_deal("ACME2026071501")
    assert res["status"] == "ok"
    assert res["is_tracked"] is True
    assert called["promote"] == "ACME2026071501"

def test_save_reference_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(mod, "save_reference", lambda deal_id: called.setdefault("save", deal_id))
    res = mod.post_save_reference("ACME2026071501")
    assert res["status"] == "ok"
    assert res["is_saved_reference"] is True
    assert called["save"] == "ACME2026071501"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_deal_lifecycle_endpoints.py -v`
Expected: FAIL — `post_promote_deal` not defined.

- [ ] **Step 3: Add imports + handlers to `deal_summary.py`**

At the top of the file, add:
```python
from src.services.deal_lifecycle import promote_deal, save_reference
```
After the existing `/{deal_id}/reconcile` handler add:
```python
@router.post("/{deal_id}/promote", summary="Commit a draft analysis into a tracked Pipeline deal")
def post_promote_deal(deal_id: str) -> dict:
    try:
        promote_deal(deal_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("promote failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "deal_id": deal_id, "is_tracked": True}


@router.post("/{deal_id}/save-reference", summary="Keep an analysis as a saved (untracked) reference")
def post_save_reference(deal_id: str) -> dict:
    try:
        save_reference(deal_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("save-reference failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "deal_id": deal_id, "is_saved_reference": True}
```
(If `logger` / `HTTPException` are not already imported in this file, add `import logging; logger = logging.getLogger(__name__)` and `from fastapi import HTTPException` — check the existing header first; `deal_summary.py` already raises `HTTPException` in `/reconcile`, so both exist.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_deal_lifecycle_endpoints.py -v`
Expected: PASS (2 tests). No router registration needed — these attach to the already-registered `deal_summary.router`.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/deal_summary.py tests/api/test_deal_lifecycle_endpoints.py
git commit -m "feat(deal): POST /deals/{id}/promote and /save-reference endpoints"
```

---

### Task A4: Opportunity ↔ deal linkage (backfill + on-demand)

**Files:**
- Create: `src/services/opportunity_linkage.py`
- Create: `scripts/backfill_opportunity_deal_link.py`
- Modify: `src/api/routers/opportunities.py` (add `POST /opportunities/link-deals`)
- Test: `tests/services/test_opportunity_linkage.py`

**Interfaces:**
- Produces: `link_opportunities_to_deals(conn=None) -> int` (returns rows updated); route `POST /opportunities/link-deals` → `{"status":"ok","linked":<n>}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_opportunity_linkage.py
from src.services.opportunity_linkage import link_opportunities_to_deals

class _Cur:
    def __init__(self): self.calls = []; self.rowcount = 3
    def execute(self, sql, params=()): self.calls.append(" ".join(sql.lower().split()))

class _Conn:
    def __init__(self): self._cur = _Cur(); self.autocommit = True
    def cursor(self): return self._cur
    def commit(self): pass
    def rollback(self): pass

def test_linkage_joins_quote_to_deal():
    c = _Conn()
    n = link_opportunities_to_deals(conn=c)
    s = c._cur.calls[0]
    assert "update proc.bp_opportunity" in s
    assert "bp_deal_documents" in s
    assert "doc_type = 'quote'" in s
    assert "o.deal_id  is null".replace("  ", " ") in s.replace("  ", " ")
    assert n == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_opportunity_linkage.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the linkage service**

```python
# src/services/opportunity_linkage.py
"""Backfill bp_opportunity.deal_id from its anchoring quote (currently always NULL)."""
from __future__ import annotations
from typing import Any
from src.services.db import get_conn

_LINK_SQL = (
    "update proc.bp_opportunity o "
    "set deal_id = d.deal_id, updated_at = now() "
    "from proc.bp_deal_documents d "
    "where d.doc_type = 'quote' and d.doc_pk = o.quote_id "
    "and o.quote_id is not null and o.deal_id is null"
)


def _run(c: Any) -> int:
    cur = c.cursor()
    cur.execute(_LINK_SQL)
    return int(getattr(cur, "rowcount", 0) or 0)


def link_opportunities_to_deals(conn: Any = None) -> int:
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                n = _run(own); own.commit(); return n
            except Exception:
                own.rollback(); raise
    return _run(conn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_opportunity_linkage.py -v`
Expected: PASS.

- [ ] **Step 5: Add the endpoint to `opportunities.py`**

```python
from src.services.opportunity_linkage import link_opportunities_to_deals

@router.post("/link-deals", summary="Link opportunities to their deal via anchoring quote")
def post_link_deals() -> dict:
    try:
        n = link_opportunities_to_deals()
    except Exception as exc:  # noqa: BLE001
        logger.exception("opportunity->deal linkage failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "linked": n}
```

- [ ] **Step 6: Write the one-time backfill script**

```python
# scripts/backfill_opportunity_deal_link.py
"""One-time: populate bp_opportunity.deal_id. Run: python scripts/backfill_opportunity_deal_link.py"""
from src.services.opportunity_linkage import link_opportunities_to_deals

if __name__ == "__main__":
    n = link_opportunities_to_deals()
    print(f"linked {n} opportunities to deals")
```

- [ ] **Step 7: Run backfill against live DB + verify**

Run:
```bash
python scripts/backfill_opportunity_deal_link.py
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -c \
  "select count(*) filter (where deal_id is not null) linked, count(*) total from proc.bp_opportunity;"
```
Expected: `linked > 0` (any opportunity whose `quote_id` resolves to a deal).

- [ ] **Step 8: Commit**

```bash
git add src/services/opportunity_linkage.py scripts/backfill_opportunity_deal_link.py \
        src/api/routers/opportunities.py tests/services/test_opportunity_linkage.py
git commit -m "feat(opportunity): backfill deal_id from anchoring quote + /link-deals endpoint"
```

---

### Task A5: `GET /opportunities/by-deal/{deal_id}` (report-scoped)

**Files:**
- Modify: `src/api/routers/opportunities.py`
- Test: `tests/api/test_opportunities_by_deal.py`

**Interfaces:**
- Produces: `GET /opportunities/by-deal/{deal_id}` → `{"deal_id":..., "opportunities":[{opportunity_id, detector_type, category_id, supplier_name, item_description, financial_impact_gbp, stage, ml_priority_score, quote_id}]}`. Empty list when none (fail-closed, not an error).

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_opportunities_by_deal.py
import importlib
import src.services.db as db
mod = importlib.import_module("src.api.routers.opportunities")

class _Cur:
    def __init__(self, rows): self._rows = rows
        # description drives column names -> dict rows
    description = [("opportunity_id",),("detector_type",),("category_id",),("supplier_name",),
                   ("item_description",),("financial_impact_gbp",),("stage",),
                   ("ml_priority_score",),("quote_id",)]
    def execute(self, sql, params=()): self.sql = sql; self.params = params
    def fetchall(self): return self._rows

class _Conn:
    def __init__(self, rows): self._c = _Cur(rows)
    def cursor(self): return self._c
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_by_deal_returns_rows(monkeypatch):
    rows = [("OPP-1","price_variance","cat-9","Acme","Widget",8100.0,"identified",0.9,"QA-1042")]
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(rows))
    res = mod.get_opportunities_by_deal("ACME2026071501")
    assert res["deal_id"] == "ACME2026071501"
    assert res["opportunities"][0]["financial_impact_gbp"] == 8100.0
    assert res["opportunities"][0]["detector_type"] == "price_variance"

def test_by_deal_empty_is_not_error(monkeypatch):
    monkeypatch.setattr(db, "get_conn", lambda: _Conn([]))
    res = mod.get_opportunities_by_deal("NOPE")
    assert res["opportunities"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_opportunities_by_deal.py -v`
Expected: FAIL — `get_opportunities_by_deal` not defined.

- [ ] **Step 3: Add the handler** (lazy `get_conn` import so the monkeypatch works, matching `deal_summary.py` style)

```python
@router.get("/by-deal/{deal_id}", summary="Opportunities for one deal (report-scoped)")
def get_opportunities_by_deal(deal_id: str) -> dict:
    from src.services.db import get_conn
    cols = ["opportunity_id", "detector_type", "category_id", "supplier_name",
            "item_description", "financial_impact_gbp", "stage",
            "ml_priority_score", "quote_id"]
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"select {', '.join(cols)} from proc.bp_opportunity "
                "where deal_id = %s order by financial_impact_gbp desc nulls last",
                (deal_id,))
            rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001
        logger.exception("by-deal opportunities failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"deal_id": deal_id, "opportunities": [dict(zip(cols, r)) for r in rows]}
```
(Route ordering: declare `/by-deal/{deal_id}` and `/link-deals` **before** any existing `/{opportunity_id}`-style route in this router so the literal segments aren't captured by the param route. Check the file and reorder if needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_opportunities_by_deal.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Live smoke against a real deal**

Run (pick a real `deal_id` from `bp_deal_overview`):
```bash
curl -s "$AI_API/opportunities/by-deal/<REAL_DEAL_ID>" | head -c 400
```
Expected: JSON `{"deal_id":...,"opportunities":[...]}` (possibly empty — that's valid).

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/opportunities.py tests/api/test_opportunities_by_deal.py
git commit -m "feat(opportunity): GET /opportunities/by-deal/{deal_id} for the analysis report"
```

---

# SUB-PLAN B — Gateway (`beyond-procwaise-Api`)

Repo: `/home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api`. Unit tests: `npm test` (jest, excludes `*.integration.spec.ts`).

### Task B1: Create the draft `bp_deal` row on confirm-upload

**Files:**
- Modify: `src/modules/Data-integration/data-integration.service.ts` (`confirmUpload`, lines 223-246)
- Test: `src/modules/Data-integration/data-integration.service.spec.ts` (new)

**Interfaces:**
- Consumes: `proc.bp_deal` (Task A1). Runs raw SQL via `this.analyseRepo.query(...)` (the `bpsqldbconnection`-bound repo).
- Produces: after a successful confirm, a draft row `INSERT INTO proc.bp_deal (deal_id, is_tracked) VALUES ($1, false) ON CONFLICT DO NOTHING` using `process.deal_id`.

- [ ] **Step 1: Write the failing test**

```ts
// src/modules/Data-integration/data-integration.service.spec.ts
import { DataIntegrationService } from './data-integration.service';

function makeService(process: any) {
  const queries: Array<{ sql: string; params: any[] }> = [];
  const analyseRepo: any = {
    findOne: jest.fn(async () => process),
    save: jest.fn(async () => process),
    query: jest.fn(async (sql: string, params: any[] = []) => { queries.push({ sql, params }); return []; }),
    count: jest.fn(async () => 0),
  };
  const svc = new DataIntegrationService({} as any, { } as any, analyseRepo);
  // silence the client notify side-effect
  (svc as any).notifyClients = jest.fn(async () => {});
  return { svc, queries, analyseRepo };
}

describe('confirmUpload draft bp_deal', () => {
  it('inserts a draft deal row on success', async () => {
    const { svc, queries } = makeService({ id: 1, deal_id: 'ACME2026071501', status: '' });
    await svc.confirmUpload(1, true, 's3/key');
    const ins = queries.find(q => q.sql.includes('proc.bp_deal'));
    expect(ins).toBeTruthy();
    expect(ins!.sql).toContain('is_tracked');
    expect(ins!.sql.toLowerCase()).toContain('on conflict');
    expect(ins!.params).toEqual(['ACME2026071501']);
  });

  it('does not insert when success is false', async () => {
    const { svc, queries } = makeService({ id: 1, deal_id: 'ACME2026071501', status: '' });
    await svc.confirmUpload(1, false, 's3/key');
    expect(queries.find(q => q.sql.includes('proc.bp_deal'))).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- data-integration.service.spec`
Expected: FAIL — no `proc.bp_deal` insert emitted.

- [ ] **Step 3: Add the upsert to `confirmUpload`**

In `confirmUpload`, after `await this.analyseRepo.save(process);` and before `notifyClients`, add:
```ts
if (success && process.deal_id) {
  // Register the deal as a DRAFT (not yet tracked in Pipeline). Idempotent.
  await this.analyseRepo.query(
    `INSERT INTO proc.bp_deal (deal_id, is_tracked) VALUES ($1, false) ON CONFLICT DO NOTHING`,
    [process.deal_id],
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- data-integration.service.spec`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/modules/Data-integration/data-integration.service.ts \
        src/modules/Data-integration/data-integration.service.spec.ts
git commit -m "feat(upload): register each new upload as a draft proc.bp_deal row"
```

---

### Task B2: Pipeline lists only tracked deals

**Files:**
- Modify: `src/modules/spendiq/spendiq.service.ts` (`getDeals`, lines 692-737)
- Test: `src/modules/spendiq/spendiq.service.spec.ts` (add a describe block)

**Interfaces:**
- Produces: `getDeals()` SQL filters to deals with a tracked `bp_deal` row.

- [ ] **Step 1: Write the failing test** (append to the existing spec, reuse its `makeService`)

```ts
describe('SpendIqService.getDeals tracked-only', () => {
  it('filters to tracked deals via proc.bp_deal', async () => {
    const { service, calls } = makeService([]);
    await service.getDeals();
    expect(calls[0].sql).toContain('proc.bp_deal');
    expect(calls[0].sql).toContain('is_tracked');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- spendiq.service.spec`
Expected: FAIL — SQL has no `is_tracked`.

- [ ] **Step 3: Add the tracked filter**

In `getDeals()`, alias the view and add an EXISTS predicate; qualify the now-ambiguous `deal_id` in the SELECT list:
```ts
const rows = await this.dataSource.query(
  `SELECT o.deal_id, deal_name, supplier_name, supplier_id, deal_date, quote_count, po_count,
          invoice_count, quote_total, po_total, invoice_total, currency, three_way_match,
          price_variance_pct, cycle_days_quote_to_po, cycle_days_po_to_invoice, orphaned
     FROM proc.bp_deal_overview o
    WHERE o.deal_id IS NOT NULL
      AND EXISTS (SELECT 1 FROM proc.bp_deal t
                   WHERE t.deal_id = o.deal_id AND t.is_tracked = true)
    ORDER BY GREATEST(COALESCE(invoice_total,0), COALESCE(quote_total,0), COALESCE(po_total,0)) DESC`,
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- spendiq.service.spec`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/spendiq/spendiq.service.ts src/modules/spendiq/spendiq.service.spec.ts
git commit -m "feat(pipeline): list only tracked deals (EXISTS proc.bp_deal is_tracked)"
```

---

### Task B3: `GET /spendiq/live-analyses` (drafts list)

**Files:**
- Modify: `src/modules/spendiq/spendiq.service.ts` (add `getLiveAnalyses`)
- Modify: `src/modules/spendiq/spendiq.controller.ts` (add route, before any `:id` sibling)
- Test: `src/modules/spendiq/spendiq.service.spec.ts`

**Interfaces:**
- Produces: `getLiveAnalyses()` → `{ analyses: [{id,name,supplier,value,currency,...}], total }` for `is_tracked=false` deals; `GET /spendiq/live-analyses`.

- [ ] **Step 1: Write the failing test**

```ts
describe('SpendIqService.getLiveAnalyses', () => {
  it('selects draft (untracked) deals', async () => {
    const { service, calls } = makeService([{ deal_id: 'D1', deal_name: 'Draft', quote_count: 1 }]);
    const res = await service.getLiveAnalyses();
    expect(calls[0].sql).toContain('is_tracked = false');
    expect(res.total).toBe(1);
    expect(res.analyses[0].id).toBe('D1');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- spendiq.service.spec`
Expected: FAIL — `getLiveAnalyses` undefined.

- [ ] **Step 3: Add the service method**

```ts
async getLiveAnalyses() {
  const rows = await this.dataSource.query(
    `SELECT o.deal_id, deal_name, supplier_name, supplier_id, deal_date,
            quote_count, po_count, invoice_count, quote_total, po_total, invoice_total, currency
       FROM proc.bp_deal_overview o
      WHERE o.deal_id IS NOT NULL
        AND EXISTS (SELECT 1 FROM proc.bp_deal t
                     WHERE t.deal_id = o.deal_id AND t.is_tracked = false)
      ORDER BY o.deal_date DESC NULLS LAST`,
  );
  const analyses = rows.map((r: any) => ({
    id: r.deal_id, name: r.deal_name || r.deal_id,
    supplier: r.supplier_name || r.supplier_id || '—',
    value: Number(r.invoice_total ?? r.po_total ?? r.quote_total) || 0,
    currency: r.currency || 'GBP',
    quoteCount: Number(r.quote_count) || 0, poCount: Number(r.po_count) || 0,
    invoiceCount: Number(r.invoice_count) || 0, dealDate: r.deal_date,
  }));
  return { analyses, total: analyses.length };
}
```

- [ ] **Step 4: Add the controller route**

```ts
@Get('live-analyses')
async getLiveAnalyses() {
  return this.spendIqService.getLiveAnalyses();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm test -- spendiq.service.spec`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/modules/spendiq/spendiq.service.ts src/modules/spendiq/spendiq.controller.ts \
        src/modules/spendiq/spendiq.service.spec.ts
git commit -m "feat(spendiq): GET /spendiq/live-analyses lists draft (untracked) analyses"
```

---

# SUB-PLAN C — UI (`beyond_procwise_ui`, branch `spendiq-ui`)

Repo: `/home/muthu/PycharmProjects/beyond_procwise_ui`. Tests: `npm test` (vitest). The engine is imperative HTML-string rendering with no component-test harness; pure helpers get vitest tests, rendering/integration is verified **live against the running stack** per the Global Constraints.

### Task C1: Post-upload routing (draft → report; amend → deal)

**Files:**
- Create: `src/modules/AnalyseUpload/nextRoute.js`
- Modify: `src/modules/AnalyseUpload/index.jsx` (`uploadType` 227-253 to return `sessionId`; `handleAnalyse` navigate 280-286)
- Test: `src/modules/AnalyseUpload/nextRoute.test.js`

**Interfaces:**
- Produces: `nextRouteAfterUpload({ mode, dealId, sessionId }) -> string` — `amend` → `/spendiq?view=analytics&deal=<id>` (existing deal detail); `new`/`bulk` → `/spendiq?view=analysis-report&deal=<id>&session=<id>` (session omitted if absent); no dealId → `/spendiq?view=analysis-report`.

- [ ] **Step 1: Write the failing test**

```js
// src/modules/AnalyseUpload/nextRoute.test.js
import { describe, it, expect } from 'vitest';
import { nextRouteAfterUpload } from './nextRoute';

describe('nextRouteAfterUpload', () => {
  it('new analysis -> report view with deal + session', () => {
    expect(nextRouteAfterUpload({ mode: 'new', dealId: 'ACME01', sessionId: 'S9' }))
      .toBe('/spendiq?view=analysis-report&deal=ACME01&session=S9');
  });
  it('bulk behaves like new', () => {
    expect(nextRouteAfterUpload({ mode: 'bulk', dealId: 'ACME01', sessionId: 'S9' }))
      .toContain('view=analysis-report');
  });
  it('amend -> existing deal detail (analytics)', () => {
    expect(nextRouteAfterUpload({ mode: 'amend', dealId: 'ACME01' }))
      .toBe('/spendiq?view=analytics&deal=ACME01');
  });
  it('report without session omits the param', () => {
    expect(nextRouteAfterUpload({ mode: 'new', dealId: 'ACME01' }))
      .toBe('/spendiq?view=analysis-report&deal=ACME01');
  });
  it('no dealId still opens the report', () => {
    expect(nextRouteAfterUpload({ mode: 'new' })).toBe('/spendiq?view=analysis-report');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- nextRoute`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the helper**

```js
// src/modules/AnalyseUpload/nextRoute.js
// Path A (new/bulk) lands on the Document Analysis Report as a DRAFT.
// Path B (amend) goes straight to the existing tracked deal's detail.
export function nextRouteAfterUpload({ mode, dealId, sessionId }) {
  const id = dealId ? encodeURIComponent(dealId) : '';
  if (mode === 'amend') {
    return id ? `/spendiq?view=analytics&deal=${id}` : '/spendiq?view=analytics';
  }
  if (!id) return '/spendiq?view=analysis-report';
  const sess = sessionId ? `&session=${encodeURIComponent(sessionId)}` : '';
  return `/spendiq?view=analysis-report&deal=${id}${sess}`;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- nextRoute`
Expected: PASS (5 tests).

- [ ] **Step 5: Thread `sessionId` through `uploadType` and use the helper**

In `uploadType` (index.jsx ~227), destructure the top-level `sessionId` from the presigned-url response and return it alongside the deal id:
```js
const { data } = await axios.post(
  `${import.meta.env.VITE_API_URL}/data-integration/presigned-url`,
  { documentType: typeKey, fileNames: files.map((f) => f.name),
    source: 'local', totalFiles: files.length, dealName: name }
)
// ... existing per-file PUT + confirm-upload loop unchanged ...
return { dealId: data?.files?.[0]?.dealId, sessionId: data?.sessionId }
```
Update `handleAnalyse` to capture both (it currently reads the returned dealId) and replace the navigate block at 280-286:
```js
import { nextRouteAfterUpload } from './nextRoute'
// ...inside handleAnalyse, after the upload loop:
const { dealId: newDealId, sessionId } = lastUploadResult  // from the final uploadType(...) call
const dealId = mode === 'amend' ? (selectedDeal.deal_id || newDealId) : newDealId
showSnackbar('Documents in — finding opportunities and flagging anomalies…', 'success')
navigate(nextRouteAfterUpload({ mode, dealId, sessionId }))
```
(Adjust `lastUploadResult` to however the loop currently captures the per-type return — assign each `await uploadType(...)` result to it; the last non-null wins. Keep the existing multi-type loop intact.)

- [ ] **Step 6: Live verification**

Start the stack (UI + gateway + BP_Backend). Upload one quote in **New deal** mode → browser URL becomes `/spendiq?view=analysis-report&deal=...&session=...`. Repeat via a deal deep-link (`/analyse?deal=...`) in **amend** mode → URL becomes `/spendiq?view=analytics&deal=...`.
Expected: correct branch per mode. (The report view itself is built in C2+.)

- [ ] **Step 7: Commit**

```bash
git add src/modules/AnalyseUpload/nextRoute.js src/modules/AnalyseUpload/nextRoute.test.js \
        src/modules/AnalyseUpload/index.jsx
git commit -m "feat(analyse): route new analyses to the report, amend to the deal"
```

---

### Task C2: Register the `analysis-report` view + working-state shell

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (NAV ~133-165; `?view=` validation IIFE ~4791-4800; `renderBody` ~4181-4276; add `analysisReportView()`)
- Modify: `src/modules/SpendIQ/data/endpoints.js` (VIEW_COVERAGE ~11-54: add an `analysis-report` entry)

**Interfaces:**
- Consumes: `current` module state; `SD(key, fallback)`; URL params.
- Produces: `analysisReportView()` returning an HTML string; a reachable `?view=analysis-report`.

- [ ] **Step 1: Register the view id**

In the `?view=` IIFE (~4795) add `analysis-report` to the `valid` map so the deep-link isn't dropped:
```js
var valid={analytics:1,home:1,'analysis-report':1};
```
Add a NAV item (or keep it deep-link-only but valid). If adding to NAV (~133-165), append to an appropriate group:
```js
{id:'analysis-report', label:'Analysis report', icon:'ti-report-analytics', badge:0},
```

- [ ] **Step 2: Add the `renderBody` dispatch branch**

In `renderBody` (~4181), before the `else { body.innerHTML = genericView(...) }` fallback:
```js
} else if(current==='analysis-report'){
  body.innerHTML = analysisReportView();
```

- [ ] **Step 3: Add the shell render function** (tabs are filled in C4–C7; this establishes layout + working state)

```js
// Report state (module scope, near other view state ~4079-4091):
let reportTab = 'overview';           // overview|documents|validate|opportunities|compliance
let reportDealId = null;
let reportSession = null;
let reportReady = false;              // flips true on WS terminal frame (C3)
function setReportTab(t){ reportTab = t; renderBody(); }

function analysisReportView(){
  // deal + session come from the URL on first entry
  if(reportDealId === null){
    const p = new URLSearchParams(location.search);
    reportDealId = p.get('deal') || '';
    reportSession = p.get('session') || '';
  }
  const tabs = [['overview','Overview'],['documents','Documents'],
    ['validate','Data Validation & Actions'],['opportunities','Opportunities'],
    ['compliance','Compliance Check']];
  const tabBar = tabs.map(t =>
    `<button class="tab ${reportTab===t[0]?'on':''}" onclick="setReportTab('${t[0]}')">${t[1]}</button>`
  ).join('');
  const working = !reportReady
    ? `<div class="panel2" role="status"><div class="p-sub">Reading your documents… findings will appear here as analysis completes.</div></div>`
    : '';
  const body = reportReady ? reportTabBody(reportTab) : '';
  return `
    <div class="siq-report">
      <div class="section-h"><h2>Document Analysis Report</h2>
        <span class="rag a">Live analysis · draft</span></div>
      <div class="tabbar">${tabBar}</div>
      ${working}
      ${body}
    </div>`;
}
function reportTabBody(tab){ return '<div class="panel2"><div class="p-sub">Loading…</div></div>'; } // filled in C4+
```

- [ ] **Step 4: Add the VIEW_COVERAGE entry** in `data/endpoints.js` so the status banner recognises the view:
```js
{ nav:'analysis-report', label:'Analysis report', endpoints:['/spendiq/deals/:id (ai)','/spendiq/discrepancies','/compliance/getComplianceData','/opportunities/by-deal/:id (ai)','/deals/:id/summary (ai)'], status:'partial' },
```

- [ ] **Step 5: Live verification**

Navigate to `/spendiq?view=analysis-report&deal=<REAL_DEAL_ID>`.
Expected: the report shell renders with five tab buttons, the "Live analysis · draft" chip, and the "Reading your documents…" working panel. Tab buttons switch `reportTab` without error.

- [ ] **Step 6: Commit**

```bash
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/data/endpoints.js
git commit -m "feat(report): register analysis-report view + working-state shell"
```

---

### Task C3: "Working…" completion via the session WebSocket

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (`analysisReportView` mount; add `subscribeReportSession()`)
- Modify: `src/modules/SpendIQ/index.jsx` (only if a bridge to trigger a re-render/refetch is needed — reuse `__SPENDIQ_REFETCH__` at 260-264)

**Interfaces:**
- Consumes: `AI_API` base (`VITE_AI_API_URL`); the existing WS pattern (`index.jsx:308-326`); `reportSession`.
- Produces: sets `reportReady=true` and re-renders on a terminal frame (`action_status !== 'running'`), with a 45s safety fallback.

- [ ] **Step 1: Add the subscription** (mirror the proven pattern; no new env var — derive ws URL from `AI_API`)

```js
let _reportWs = null;
function subscribeReportSession(){
  if(!reportSession || _reportWs) { if(!reportSession){ reportReady = true; } return; }
  try{
    const base = String(window.__AI_API__ || '').replace(/^http/,'ws').replace(/\/+$/,'');
    const ws = new WebSocket(`${base}/ws/session/${reportSession}`);
    _reportWs = ws;
    let closed = false;
    const done = () => { if(!closed){ closed=true; try{ws.close();}catch{} reportReady=true;
      if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__(); renderBody(); } };
    ws.onmessage = (m) => { let f={}; try{ f=JSON.parse(m.data); }catch{}
      if(f.action_status && f.action_status !== 'running') done(); };
    ws.onerror = () => done();
    setTimeout(done, 45000);
  }catch(e){ reportReady = true; renderBody(); }
}
```
Expose `AI_API` for the engine: in `index.jsx` where other `window.__SPENDIQ_*` bridges are set (~160-170), add `window.__AI_API__ = AI_API;`.

- [ ] **Step 2: Call it when the view first mounts**

In `analysisReportView()`, right after resolving `reportDealId/reportSession` from the URL, call `subscribeReportSession();` once (guard with the `_reportWs` null-check already inside).

- [ ] **Step 3: Live verification**

Do a fresh **New deal** upload → land on the report. While the pipeline runs, the working panel shows; when the session reaches a terminal frame (or after 45s), the tabs render. Confirm via browser devtools Network → WS that `/ws/session/<id>` opened.
Expected: working → ready transition without reload.

- [ ] **Step 4: Commit**

```bash
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx
git commit -m "feat(report): flip working->ready on session WebSocket terminal frame"
```

---

### Task C4: Overview + Documents tabs (live data)

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (`reportTabBody`; add `reportLoadData()`)
- Modify: `src/modules/SpendIQ/data/useSpendData.js` (add `adaptReport*` if using the hook path) — OR use the imperative `window.__SPENDIQ_API__` bridge (preferred, matches `siqLoadDeal`).

**Interfaces:**
- Consumes: `window.__SPENDIQ_API__(path)` (gateway GET), `window.__SPENDIQ_API_AI__(path)` (BP_Backend GET — add if absent, mirroring `__SPENDIQ_API__` with `AI_API`); `rag()`, `kpiScroll`/`.hscroll`, `case-grp`.
- Produces: `reportData` cache `{ deal, discrepancies, compliance, opportunities, summary }`; `reportTabBody('overview')` and `('documents')` render real values.

- [ ] **Step 1: Add an AI GET bridge** (if not present) in `index.jsx` near `__SPENDIQ_API__` (161-162):
```js
window.__SPENDIQ_API_AI__ = (path, params) =>
  axios.get(`${AI_API}${path}`, params ? { params } : undefined).then(r => r.data);
```

- [ ] **Step 2: Load the report data imperatively** (called from `subscribeReportSession` `done()` and on first ready render)

```js
let reportData = { deal:null, discrepancies:[], compliance:null, opportunities:[], summary:null };
async function reportLoadData(){
  const id = reportDealId; if(!id) return;
  const G = window.__SPENDIQ_API__, A = window.__SPENDIQ_API_AI__;
  try{
    const [deal, disc, comp, opps, summ] = await Promise.allSettled([
      G(`/spendiq/deals/${encodeURIComponent(id)}`),
      G(`/spendiq/discrepancies`, { status:'open', limit:200, deal:id }),
      G(`/compliance/getComplianceData`),
      A(`/opportunities/by-deal/${encodeURIComponent(id)}`),
      A(`/deals/${encodeURIComponent(id)}/summary`),
    ]);
    reportData = {
      deal: deal.value || null,
      discrepancies: (disc.value && (disc.value.items||disc.value)) || [],
      compliance: comp.value || null,
      opportunities: (opps.value && opps.value.opportunities) || [],
      summary: summ.value || null,
    };
  }catch(e){ /* fail-closed: tabs show "no data" states, not fabricated values */ }
  renderBody();
}
```
Call `reportLoadData()` inside the WS `done()` (Task C3) before `renderBody()`.

- [ ] **Step 3: Render the Overview tab** — RAG + KPI row (reuse `rag()` and `.hscroll`), highlights/watch capped at 10, executive summary labelled as agent output:

```js
function reportOverview(){
  const d = reportData.deal || {};
  const disc = reportData.discrepancies || [];
  const atRisk = disc.reduce((s,x)=> s + (Number(x.amount)||0), 0);
  const ragCls = atRisk<=0 ? 'g' : (disc.length>5 ? 'r' : 'a');
  const ragTxt = atRisk<=0 ? 'On track' : `£${atRisk.toLocaleString()} at risk across ${disc.length} findings`;
  const kpis = [
    ['Documents analysed', (d.documents&&d.documents.length)||0],
    ['Opportunities', reportData.opportunities.length],
    ['Items to validate', disc.length],
    ['3-way match', d.three_way_match ? 'Yes' : 'Needs validation'],
  ];
  const kpiRow = `<div class="hscroll kpis-row">${kpis.map(k=>
    `<div class="kpi"><div class="kpi-v">${k[1]}</div><div class="kpi-l">${k[0]}</div></div>`).join('')}</div>`;
  const summary = reportData.summary && reportData.summary.summary
    ? `<div class="panel2"><div class="p-h">Executive summary
         <span class="chip">✨ Agent draft · AI-generated · review before acting</span></div>
         <div class="p-body">${reportData.summary.summary}</div></div>`
    : `<div class="panel2"><div class="p-sub">Summary not available yet.</div></div>`;
  return `<div class="section-h"><h2>Overview</h2><span class="rag ${ragCls}">${ragTxt}</span></div>
          ${kpiRow}${summary}`;
}
```

- [ ] **Step 4: Render the Documents tab** — cases from `deal.documents`, grouped by type with `case-grp`; one summary line + a link into Validate:

```js
function reportDocuments(){
  const docs = (reportData.deal && reportData.deal.documents) || [];
  if(!docs.length) return `<div class="panel2"><div class="p-sub">No documents linked to this analysis yet.</div></div>`;
  const byType = {};
  docs.forEach(x => { const t=(x.doc_type||'other'); (byType[t]=byType[t]||[]).push(x); });
  const mix = Object.keys(byType).map(t => `${byType[t].length} ${t}`).join(' · ');
  const rows = docs.map(x =>
    `<tr><td>${x.doc_type||''}</td><td>${x.doc_number||x.doc_pk||''}</td>
         <td>${x.doc_date||''}</td><td class="num">${x.amount!=null?('£'+Number(x.amount).toLocaleString()):'—'}</td></tr>`
  ).join('');
  return `<div class="section-h"><h2>Documents</h2><span class="p-sub">${mix}</span></div>
    <div class="case-grp"><table class="cmx"><thead>
      <tr><th>Type</th><th>Reference</th><th>Date</th><th>Amount</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div style="margin-top:10px"><a href="#" onclick="event.preventDefault();setReportTab('validate')">Review discrepancies in Data Validation & Actions →</a></div>`;
}
```

- [ ] **Step 5: Wire `reportTabBody`**

```js
function reportTabBody(tab){
  if(tab==='overview') return reportOverview();
  if(tab==='documents') return reportDocuments();
  return `<div class="panel2"><div class="p-sub">This tab is wired in a later step.</div></div>`;
}
```

- [ ] **Step 6: Live verification (real deal)**

Open `/spendiq?view=analysis-report&deal=<REAL_DEAL_ID>` (a deal with docs). Overview shows real KPI counts + a real/absent summary; Documents lists the deal's real documents grouped by type.
Expected: values match `curl "$AI_API/deals/<id>/summary"` and the gateway `/spendiq/deals/<id>` payload; no fabricated numbers; empty states where data is absent.

- [ ] **Step 7: Commit**

```bash
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx
git commit -m "feat(report): Overview + Documents tabs on live deal data"
```

---

### Task C5: Data Validation & Actions tab (queue + decision + comparison)

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (`reportValidate()`, decision-type map, action handlers)

**Interfaces:**
- Consumes: `reportData.discrepancies`; `ac-grid`; `.cmx`; `window.__SPENDIQ_API_POST__`.
- Produces: `reportValidate()`; `reportDecide(findingId, action)`; `DECISION_TYPES` map keyed by issue type.

- [ ] **Step 1: Add the centrally-maintained decision-type map** (filtered per issue type — three-way-match ≠ compliance ≠ sourcing):

```js
const DECISION_TYPES = {
  amount_over_po:      [['apply_value','Accept corrected amount'],['flag','Flag for review'],['dismiss','Accept as risk']],
  line_amount_over_po: [['apply_value','Accept corrected line'],['flag','Flag for review'],['dismiss','Accept as risk']],
  po_not_found:        [['flag','Raise PO exception'],['dismiss','Confirm off-PO']],
  po_reference_missing:[['apply_value','Attach PO reference'],['flag','Flag for review']],
  default:             [['flag','Flag for review'],['dismiss','Dismiss']],
};
function decisionTypesFor(issueType){ return DECISION_TYPES[issueType] || DECISION_TYPES.default; }
```

- [ ] **Step 2: Render the queue + detail with `ac-grid`** (grouped by case, prioritised by confidence × impact; selected item shows evidence + a Decision block):

```js
let reportSelFinding = null;
function reportValidate(){
  const items = (reportData.discrepancies||[]).slice()
    .sort((a,b)=> (Number(b.amount)||0)-(Number(a.amount)||0));
  if(!items.length) return `<div class="panel2"><div class="p-sub">No open items to validate.</div></div>`;
  const sel = reportSelFinding || items[0];
  const list = items.map(x =>
    `<div class="ac-row ${sel && sel.id===x.id?'on':''}" onclick="reportSelect('${x.id}')">
       <div class="ac-title">${x.issue_type||'finding'}</div>
       <div class="ac-sub">${x.supplier||''} · £${(Number(x.amount)||0).toLocaleString()}</div></div>`).join('');
  const opts = decisionTypesFor(sel && sel.issue_type).map(o =>
    `<button class="btn" onclick="reportDecide('${sel.id}','${o[0]}')">${o[1]}</button>`).join('');
  const detail = `
    <div class="ac-detail"><div class="p-h">${sel.issue_type||'Finding'} <span class="rag a">Needs validation</span></div>
      <div class="p-body">
        <div class="kv"><span>Supplier</span><b>${sel.supplier||'—'}</b></div>
        <div class="kv"><span>Exposure</span><b>£${(Number(sel.amount)||0).toLocaleString()}</b></div>
        <div class="kv"><span>Why flagged</span><b>${sel.reason||sel.detail||'—'}</b></div>
        <div class="decision"><div class="p-sub">Decision</div>${opts}</div>
      </div></div>`;
  return `<div class="section-h"><h2>Data Validation & Actions</h2></div>
          <div class="ac-grid"><div class="ac-list">${list}</div>${detail}</div>`;
}
function reportSelect(id){ reportSelFinding = (reportData.discrepancies||[]).find(x=>String(x.id)===String(id))||null; renderBody(); }
```

- [ ] **Step 3: Wire the real, audit-logged decision mutation** (`/decisions/finding/{id}/action` on BP_Backend; refetch after):

```js
async function reportDecide(findingId, action){
  try{
    await window.__SPENDIQ_API_AI_POST__(`/decisions/finding/${encodeURIComponent(findingId)}/action`, { action });
    if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__();
    await reportLoadData();
  }catch(e){ /* surface a toast; do not optimistically mutate local state */ }
}
```
Add `window.__SPENDIQ_API_AI_POST__` in `index.jsx` mirroring `__SPENDIQ_API_POST__` but against `AI_API`.

- [ ] **Step 4: Add `reportValidate` to `reportTabBody`** (`if(tab==='validate') return reportValidate();`).

- [ ] **Step 5: Live verification**

Open the tab for a deal with discrepancies. Selecting an item shows real evidence; the Decision buttons match the finding's `issue_type` (three-way-match findings show the amount/line options, not the compliance set). Click a decision → confirm a new `bp_agent_actions` row is written:
```bash
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -c \
  "select action_type, phase, created_at from proc.bp_agent_actions order by created_at desc limit 3;"
```
Expected: a fresh audit row; queue refetches.

- [ ] **Step 6: Commit**

```bash
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx
git commit -m "feat(report): Validation queue + issue-type-filtered decisions (audit-logged)"
```

---

### Task C6: Opportunities + Compliance tabs

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (`reportOpportunities()`, `reportCompliance()`)

**Interfaces:**
- Consumes: `reportData.opportunities`, `reportData.deal.documents`, `reportData.compliance`, `reportData.discrepancies`.
- Produces: both tab renderers; derived levers/coverage/status; fail-closed "Not measured yet" for the three detectorless measures.

- [ ] **Step 1: Opportunities tab** — coverage chips (from deal docs), levers capped at 3 + "+N", Ready/Needs-validation (from open discrepancies), needs-validation rows hide Pursue/Assign/Reject:

```js
function reportOpportunities(){
  const opps = reportData.opportunities || [];
  if(!opps.length) return `<div class="panel2"><div class="p-sub">No opportunities detected for this analysis.</div></div>`;
  const docTypes = new Set(((reportData.deal&&reportData.deal.documents)||[]).map(x=>x.doc_type));
  const cover = ['quote','po','invoice','contract'].map(t =>
    `<span class="cov ${docTypes.has(t)?'on':'off'}">${t}</span>`).join('');
  const needsVal = (reportData.discrepancies||[]).length > 0;
  // group levers by detector_type
  const levers = {};
  opps.forEach(o => { const t=o.detector_type||'other'; levers[t]=(levers[t]||0)+1; });
  const leverKeys = Object.keys(levers);
  const leverChips = leverKeys.slice(0,3).map(k=>`<span class="chip">${k}</span>`).join('')
    + (leverKeys.length>3?`<span class="chip">+${leverKeys.length-3} more</span>`:'');
  const saving = opps.reduce((s,o)=> s + (Number(o.financial_impact_gbp)||0), 0);
  const status = needsVal ? `<span class="rag a">Needs validation</span>` : `<span class="rag g">Ready</span>`;
  const actions = needsVal ? '' :
    `<button class="btn">Pursue</button><button class="btn">Assign</button><button class="btn ghost">Reject</button>`;
  return `<div class="section-h"><h2>Opportunities</h2>${status}</div>
    <div class="panel2"><div class="p-body">
      <div class="cov-row">${cover}</div>
      <div class="lever-row">${leverChips}</div>
      <div class="kv"><span>Potential saving</span><b>£${saving.toLocaleString()}</b></div>
      <div class="act-row">${actions}</div>
    </div></div>`;
}
```

- [ ] **Step 2: Compliance tab** — real measures from `compliance.getComplianceData`; the three detectorless measures rendered fail-closed:

```js
const UNMEASURED = ['Split PO detection','Approval bypass rate','Duplicate invoices'];
function reportCompliance(){
  const c = reportData.compliance || {};
  const measures = (c.measures || c.issues || []);   // adapt to the real payload shape
  const real = measures.map(m =>
    `<tr><td>${m.name||m.issue_type}</td><td class="num">${m.value ?? m.amount ?? '—'}</td>
         <td>${m.description||''}</td></tr>`).join('');
  const stub = UNMEASURED.map(n =>
    `<tr><td>${n}</td><td><span class="rag a">Not measured yet</span></td>
         <td>No detector in this system yet (Phase 2).</td></tr>`).join('');
  return `<div class="section-h"><h2>Compliance Check</h2></div>
    <table class="cmx"><thead><tr><th>Measure</th><th>Value</th><th>What it checks</th></tr></thead>
    <tbody>${real}${stub}</tbody></table>`;
}
```
(Confirm the real `getComplianceData` payload keys during implementation and map exactly — do not invent values; unknown → "—".)

- [ ] **Step 3: Wire both into `reportTabBody`** (`opportunities`, `compliance`).

- [ ] **Step 4: Live verification**

For a real deal: Opportunities shows real savings/levers/coverage and hides Pursue/Assign/Reject when the deal has open discrepancies; Compliance shows the real measures plus the three "Not measured yet" rows (never a green pass).
Expected: matches `curl "$AI_API/opportunities/by-deal/<id>"` and gateway `/compliance/getComplianceData`.

- [ ] **Step 5: Commit**

```bash
git add src/modules/SpendIQ/engine.js
git commit -m "feat(report): Opportunities (derived levers/coverage/status) + fail-closed Compliance"
```

---

### Task C7: Deal Detail overlay + Promote / Save / Assign / Reject

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (open deal from report; draft banner in `dealView`; Promote/Save handlers; shared Assign/Reject popover)

**Interfaces:**
- Consumes: existing `openDeal(id)` (body-swap, sets `analyticsPage='deal'`), `dealView()` six-tab set; `window.__SPENDIQ_API_AI_POST__`; `/spendiq/discrepancies/resolve` (gateway).
- Produces: `goDealFromReport(id)`; `reportPromote(id)`; `reportSaveRef(id)`; `assignRejectPopover(...)` (one component for both).

- [ ] **Step 1: Open the six-tab deal from the report** (reuse `dealView`, do not fork):

```js
function goDealFromReport(id){ go('analytics'); openDeal(id); }   // openDeal sets analyticsPage='deal'
```
Wire the report's "Detail" affordance (e.g. on the Opportunities panel) to `goDealFromReport(reportDealId)`.

- [ ] **Step 2: Draft banner + commit paths in `dealView`** — when the deal is a draft (`is_tracked=false`), show the snapshot banner and the two commit buttons. Detect draft via a lightweight cache set on data load, or by asking the gateway; simplest is to reuse `reportDealId`/`reportReady` context when arriving from the report:

```js
// Inside dealView(), near the header (~2732), when arriving from a live analysis:
const draftBanner = (dealId === reportDealId)
  ? `<div class="snap-banner"><span class="rag a">Live analysis</span>
       Point-in-time read — not yet tracked.
       <button class="btn" onclick="reportPromote('${dealId}')">Promote to tracked deal</button>
       <button class="btn ghost" onclick="reportSaveRef('${dealId}')">Save to references</button></div>`
  : '';
// prepend draftBanner to the dealView header block.
```

- [ ] **Step 3: Promote / Save handlers** (real backend calls; Promote then returns to Pipeline):

```js
async function reportPromote(id){
  try{
    await window.__SPENDIQ_API_AI_POST__(`/deals/${encodeURIComponent(id)}/promote`, {});
    if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__();
    go('pipeline');                     // now visible as a tracked deal
  }catch(e){ /* toast */ }
}
async function reportSaveRef(id){
  try{ await window.__SPENDIQ_API_AI_POST__(`/deals/${encodeURIComponent(id)}/save-reference`, {}); }
  catch(e){ /* toast */ }
}
```

- [ ] **Step 4: Shared Assign/Reject popover** (one component; Reject requires justification, writes via gateway resolve; both audit-logged):

```js
let _arState = null;   // {mode:'assign'|'reject', findingId}
function openAR(mode, findingId){ _arState = {mode, findingId}; renderBody(); }
function closeAR(){ _arState = null; renderBody(); }
async function confirmAR(){
  const s = _arState; if(!s) return;
  const justification = (document.getElementById('ar-just')||{}).value || '';
  if(s.mode==='reject' && !justification.trim()) return;   // required
  try{
    await window.__SPENDIQ_API_POST__(`/spendiq/discrepancies/resolve`,
      { id:s.findingId, action:s.mode, justification });
    if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__();
    await reportLoadData();
  }catch(e){ /* toast */ }
  closeAR();
}
function assignRejectPopover(){
  if(!_arState) return '';
  const isReject = _arState.mode==='reject';
  return `<div class="popover"><div class="p-h">${isReject?'Reject':'Assign'}</div>
    ${isReject?`<textarea id="ar-just" placeholder="Justification (required)"></textarea>`
              :`<input id="ar-just" placeholder="Assign to (user)"/>`}
    <div class="act-row"><button class="btn" onclick="confirmAR()">Confirm</button>
      <button class="btn ghost" onclick="closeAR()">Cancel</button></div></div>`;
}
```
Include `${assignRejectPopover()}` in `analysisReportView()`'s returned markup, and wire the Opportunities/Validation Assign & Reject buttons to `openAR('assign',id)` / `openAR('reject',id)`.

- [ ] **Step 5: Unsaved-close guard** — closing the report/overlay with pending changes prompts Promote / Discard / Cancel (never silent discard). Hook the existing breadcrumb/back handler (`backToAnalyse`) to check `dealId===reportDealId && !tracked` and show the three-way prompt.

- [ ] **Step 6: Live verification (end-to-end)**

1. Fresh **New deal** upload → report (draft). Confirm the deal is **absent** from Pipeline (`/spendiq?view=pipeline`).
2. Open Detail → six tabs render (Overview/Summary/Compliance/Negotiation/Opportunity/Checks) with the draft banner.
3. Click **Promote** → deal now **appears** in Pipeline; verify:
```bash
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -c \
  "select deal_id, is_tracked, tracked_at from proc.bp_deal where deal_id='<ID>';"
```
Expected: `is_tracked=true`. 4. Assign/Reject on an opportunity writes a real resolve + audit row.

- [ ] **Step 7: Commit**

```bash
git add src/modules/SpendIQ/engine.js
git commit -m "feat(report): draft deal overlay + Promote/Save/Assign/Reject (real, audit-logged)"
```

---

### Task C8: "Live analyses" list (reach saved drafts)

**Files:**
- Modify: `src/modules/SpendIQ/engine.js` (a small `liveAnalysesView()` or a panel on Home/Analyse listing drafts)

**Interfaces:**
- Consumes: `GET /spendiq/live-analyses` (Task B3) via `window.__SPENDIQ_API__`.
- Produces: a list of draft analyses, each opening the report at `?view=analysis-report&deal=<id>`.

- [ ] **Step 1: Add a drafts panel** (reachable from the Analyse area; reuse list styling):

```js
async function loadLiveAnalyses(){
  try{ const r = await window.__SPENDIQ_API__(`/spendiq/live-analyses`);
    window.__SPENDIQ_DATA__ && (window.__SPENDIQ_DATA__['report.drafts'] = r.analyses||[]); renderBody(); }
  catch(e){}
}
function liveAnalysesPanel(){
  const rows = (SD('report.drafts', [])||[]).map(a =>
    `<div class="ac-row" onclick="location.href='/spendiq?view=analysis-report&deal='+encodeURIComponent('${a.id}')">
       <div class="ac-title">${a.name}</div><div class="ac-sub">${a.supplier} · £${(a.value||0).toLocaleString()}</div></div>`
  ).join('') || `<div class="p-sub">No draft analyses.</div>`;
  return `<div class="panel2"><div class="p-h">Live analyses (drafts)</div><div class="ac-list">${rows}</div></div>`;
}
```
Call `loadLiveAnalyses()` when the Analyse view (or Home) mounts; render `liveAnalysesPanel()` there.

- [ ] **Step 2: Live verification**

Upload two **New deal** analyses without promoting → both appear in "Live analyses"; clicking one reopens its report; a promoted deal disappears from this list and shows in Pipeline.
Expected: draft/tracked split is coherent end-to-end.

- [ ] **Step 3: Commit**

```bash
git add src/modules/SpendIQ/engine.js
git commit -m "feat(report): Live-analyses drafts list (reopen untracked analyses)"
```

---

## Self-Review

**Spec coverage** (spec §→task):
- §3 engine.js-native + reuse → C2–C8. ✅
- §4 routing Path A/B → C1. ✅
- §5 draft/tracked model (table, draft-on-upload, promote, save-ref, pipeline filter, live-analyses) → A1, B1, A2/A3, A2/A3, B2, B3/C8. ✅
- §6.1 Overview → C4; §6.2 Documents → C4; §6.3 Validation+decision filtering → C5; §6.4 Opportunities (linkage + derived) → A4/A5 + C6; §6.5 Compliance fail-closed → C6. ✅
- §8 working WS → C3. ✅
- §9 Assign/Reject/Decision/Promote/Save audit-logged → C5, C7. ✅
- §11 live verification → verification steps in A5, C4–C8. ✅
- Phase-2 items correctly excluded. ✅

**Placeholder scan:** No "TBD/handle edge cases" left. Two spots say "adapt to the real payload shape / confirm keys" (C6 compliance, C4 discrepancy field names) — these are explicit *verify-against-live-payload* instructions with a concrete fail-closed fallback ("unknown → —"), not deferred work.

**Type/name consistency:** `promote_deal`/`save_reference` (A2) match imports in A3; `link_opportunities_to_deals` (A4) matches A5 usage context; `nextRouteAfterUpload` signature (C1) matches its test; `reportTabBody`/`setReportTab`/`reportLoadData`/`reportData` used consistently across C2–C8; `__SPENDIQ_API_AI__` / `__SPENDIQ_API_AI_POST__` introduced in C4/C5 and reused in C7. `is_tracked`/`is_saved_reference` column names consistent across A1/A2/B1/B2/B3.

**Known implementation-time confirmations (not placeholders):** exact key names in `/spendiq/discrepancies` and `/compliance/getComplianceData` payloads must be read from the live response and mapped; unknown values render as "—" (fail-closed), never fabricated.
