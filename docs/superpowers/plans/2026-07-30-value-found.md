# Value Found (W1 money number) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One evidence-backed headline — "£X value found · £Y recovered" — on Home and the SpendIQ Dashboard, with a findings drawer, a duplicate-invoice detector, a one-click supplier query action, and a weekly digest.

**Architecture:** A read-model. `value_summary_service.py` (BP_Backend) aggregates three existing sources (`proc.bp_extraction_discrepancy`, `proc.bp_opportunity`, benchmark deltas) at query time behind `GET /spendiq/value-summary`. No new store; the only schema change is three nullable columns on the discrepancy table. UI consumes it in two places via the established query/adapter patterns.

**Tech Stack:** FastAPI + psycopg2 (BP_Backend), NestJS/TypeORM (gateway, one endpoint extension), React + engine.js (UI). Spec: `docs/superpowers/specs/2026-07-30-value-found-design.md`.

## Global Constraints

- **Vocabulary:** *value found / recovered / potential*. Never "savings" for over-billing corrections.
- **No fabrication:** never render `£` on an unconverted foreign amount; a failed source is reported `"unavailable"`, never silently zero; empty states are honest, never sample rows.
- **Strict numeric parse:** a value parses only if the WHOLE string (after stripping `£$€¥₹`, commas, spaces, and a leading `+`) is a number. A SHA-256 hash must parse to `None`.
- **Tier rules (from live-verified vocab):** discrepancy tiers use `issue_type IN ('amount_over_po','line_amount_over_po','duplicate_invoice')`, exclude `status IN ('ignored','superseded')`; opportunity stages: verified = `negotiation, agreed`; potential = `identified`; recovered = `realised`; excluded = `rejected, closed`.
- **Delta convention:** for these issue types `computed_value` is the signed delta (e.g. `"+950.00"`); parse it first, fall back to `raw_value − expected_value`.
- **DB:** tables/indexes use `bp_` / `ix_bp_*_*` prefixes; migrations are SQL files in `scripts/migrations/`, applied to BOTH `bp_sqldb` and `bp_testdb`.
- **Git:** commit to `Development` only; NO Co-Authored-By lines, no Claude attribution.
- **Tests:** BP_Backend: `set -a; . ./.env; set +a` first, then `./venv/bin/python -m pytest --import-mode=importlib`. UI (`/home/muthu/PycharmProjects/beyond_procwise_ui`): `npm test -- <file>`. Gateway (`/home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api`): `npm test -- <file>`.
- **Server for live checks:** start BP_Backend WITH `.env` loaded (`EXTRACTION_RENOVATION_ENABLED=1` matters) or it runs the wrong engine.

---

### Task 1: Migration — outcome columns on the discrepancy table

**Files:**
- Create: `scripts/migrations/2026-07-30-value-found-columns.sql`

**Interfaces:**
- Produces: columns `resolution_outcome TEXT`, `recovered_amount NUMERIC(18,2)`, `query_sent_at TIMESTAMPTZ` on `proc.bp_extraction_discrepancy`, used by Tasks 2, 4, 8.

- [ ] **Step 1: Write the migration**

```sql
-- Value Found (W1): distinguish HOW a discrepancy was resolved, and record the
-- Phase-3 "query sent" stamp. All nullable; historical resolved rows keep outcome
-- NULL and deliberately never count as "recovered".
ALTER TABLE proc.bp_extraction_discrepancy
    ADD COLUMN IF NOT EXISTS resolution_outcome TEXT,
    ADD COLUMN IF NOT EXISTS recovered_amount   NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS query_sent_at      TIMESTAMPTZ;

ALTER TABLE proc.bp_extraction_discrepancy
    DROP CONSTRAINT IF EXISTS bp_extraction_discrepancy_resolution_outcome_check;
ALTER TABLE proc.bp_extraction_discrepancy
    ADD CONSTRAINT bp_extraction_discrepancy_resolution_outcome_check
    CHECK (resolution_outcome IS NULL OR resolution_outcome IN ('recovered', 'accepted'));
```

- [ ] **Step 2: Apply to both databases and verify**

Load `.env` (`set -a; . ./.env; set +a`), then apply with `psql` to `bp_sqldb` AND `bp_testdb` using the connection values from `.env` (same host, databases `bp_sqldb` / `bp_testdb`). Verify:

Run: `psql ... -c "\d proc.bp_extraction_discrepancy" | grep -E "resolution_outcome|recovered_amount|query_sent_at"`
Expected: all three columns listed, in both databases. Record `md5sum scripts/migrations/2026-07-30-value-found-columns.sql` in the commit message body.

- [ ] **Step 3: Commit**

```bash
git add -f scripts/migrations/2026-07-30-value-found-columns.sql
git commit -m "feat(value-found): discrepancy outcome + query-sent columns"
```

---

### Task 2: `value_summary_service` — the aggregation core (TDD)

**Files:**
- Create: `src/services/value_summary_service.py`
- Test: `tests/services/test_value_summary_service.py`

**Interfaces:**
- Consumes: `src.services.db.get_conn`, `repositories.fx_rate_repo.get_or_refresh_rates()` (USD-quoted rates dict, shape `{"rates": {"GBP": 0.79, ...}, "fetched_at": ...}` — confirm exact shape by reading `src/api/routers/fx.py` and `repositories/fx_rate_repo.py` before coding `to_gbp`).
- Produces (used by Tasks 3, 8, 10):
  - `build_value_summary(conn=None) -> dict` — the full API payload minus `generated_at`.
  - `parse_amount(v) -> float | None`
  - `discrepancy_delta(row: dict) -> float | None`
  - `classify_discrepancy(row: dict) -> dict | None` (a *finding* dict) and `classify_opportunity(row: dict) -> dict | None`
  - `dedupe(findings: list[dict]) -> list[dict]`
  - Finding dict keys: `id, tier, source, amount_gbp, converted_from, title, supplier_name, deal_id, doc_pk, found_at, age_days, link, status, queryable, superseded_by`.

- [ ] **Step 1: Write failing tests for the pure core**

```python
"""Unit tests for the Value Found aggregation core. Pure functions on dicts —
no DB. Row shapes mirror proc.bp_extraction_discrepancy / proc.bp_opportunity."""
from datetime import datetime, timezone, timedelta

from src.services.value_summary_service import (
    parse_amount, discrepancy_delta, classify_discrepancy,
    classify_opportunity, dedupe, summarise,
)


def _disc(**kw):
    row = {
        "discrepancy_id": 1, "issue_type": "amount_over_po", "status": "open",
        "raw_value": "10950.00", "expected_value": "10000.00", "computed_value": "+950.00",
        "resolution_outcome": None, "recovered_amount": None, "query_sent_at": None,
        "doc_type": "invoice", "doc_pk_candidate": "INV-1042", "deal_id": "D-1",
        "supplier_name": "Techworld", "currency": "GBP", "notes": "",
        "created_at": datetime.now(timezone.utc) - timedelta(days=45),
    }
    row.update(kw)
    return row


def test_parse_amount_strict():
    assert parse_amount("£1,234.50") == 1234.5
    assert parse_amount("+950.00") == 950.0
    assert parse_amount("a" * 64) is None          # SHA-256-like → None, never a number
    assert parse_amount("1042 refund") is None     # partial numbers don't count


def test_delta_prefers_signed_computed_value():
    assert discrepancy_delta(_disc()) == 950.0
    # convention 2: no computed_value → observed − expected
    assert discrepancy_delta(_disc(computed_value=None)) == 950.0
    assert discrepancy_delta(_disc(computed_value="garbage", raw_value="x")) is None


def test_discrepancy_tiering():
    f = classify_discrepancy(_disc())
    assert f["tier"] == "verified" and f["amount_gbp"] == 950.0 and f["queryable"] is True
    assert f["age_days"] == 45
    # ignored (= dismissed false positive) and superseded never count
    assert classify_discrepancy(_disc(status="ignored")) is None
    assert classify_discrepancy(_disc(status="superseded")) is None
    # historical resolved with NULL outcome: stays in found, NOT recovered
    f = classify_discrepancy(_disc(status="resolved"))
    assert f["tier"] == "verified" and f["status"] == "resolved"
    # resolved as recovered: amount falls back to delta when recovered_amount is null
    f = classify_discrepancy(_disc(status="resolved", resolution_outcome="recovered"))
    assert f["tier"] == "verified" and f["recovered_gbp"] == 950.0
    f = classify_discrepancy(_disc(status="resolved", resolution_outcome="recovered",
                                   recovered_amount=500))
    assert f["recovered_gbp"] == 500.0
    # non-value issue types produce no finding
    assert classify_discrepancy(_disc(issue_type="po_not_found")) is None


def test_opportunity_tiering():
    def _opp(**kw):
        row = {"opportunity_id": "O-1", "stage": "identified", "financial_impact_gbp": 1200,
               "realised_savings_gbp": None, "supplier_name": "Acme", "deal_id": "D-2",
               "po_id": None, "quote_id": None, "item_description": "notebooks",
               "created_at": datetime.now(timezone.utc)}
        row.update(kw)
        return row
    assert classify_opportunity(_opp())["tier"] == "potential"
    assert classify_opportunity(_opp(stage="negotiation"))["tier"] == "verified"
    assert classify_opportunity(_opp(stage="agreed"))["tier"] == "verified"
    f = classify_opportunity(_opp(stage="realised", realised_savings_gbp=800))
    assert f["tier"] == "verified" and f["recovered_gbp"] == 800.0
    assert classify_opportunity(_opp(stage="rejected")) is None
    assert classify_opportunity(_opp(stage="closed")) is None
    # unknown stage must raise, never be silently dropped (spec rule)
    try:
        classify_opportunity(_opp(stage="mystery"))
        assert False, "unknown stage must raise"
    except ValueError:
        pass


def test_dedupe_precedence_and_supersede_flag():
    d = classify_discrepancy(_disc(deal_id="D-9", doc_pk_candidate="INV-9"))
    o = classify_opportunity({"opportunity_id": "O-9", "stage": "agreed",
        "financial_impact_gbp": 950, "realised_savings_gbp": None,
        "supplier_name": "Techworld", "deal_id": "D-9", "po_id": None, "quote_id": None,
        "item_description": None, "created_at": datetime.now(timezone.utc),
        "doc_pk": "INV-9"})
    out = dedupe([o, d])
    kept = [f for f in out if f["superseded_by"] is None]
    supp = [f for f in out if f["superseded_by"] is not None]
    assert len(kept) == 1 and kept[0]["source"] == "discrepancy"
    assert len(supp) == 1 and supp[0]["superseded_by"] == kept[0]["id"]


def test_summarise_totals():
    fs = [
        {"tier": "verified", "amount_gbp": 950.0, "recovered_gbp": 950.0,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 100.0, "recovered_gbp": None,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "potential", "amount_gbp": 500.0, "recovered_gbp": None,
         "supplier_name": "Acme", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 999.0, "recovered_gbp": None,
         "supplier_name": "X", "superseded_by": "other"},   # suppressed: excluded from sums
    ]
    t = summarise(fs)
    assert t["verified_found_gbp"] == 1050.0
    assert t["recovered_gbp"] == 950.0
    assert t["potential_gbp"] == 500.0
    assert t["finding_count"] == 3
    assert t["by_supplier"][0] == {"supplier_name": "Techworld",
                                   "verified_found_gbp": 1050.0, "finding_count": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_value_summary_service.py --import-mode=importlib -v`
Expected: FAIL — `ModuleNotFoundError: src.services.value_summary_service`

- [ ] **Step 3: Implement the service**

```python
"""Value Found (W1): read-model aggregation over existing finding sources.

Pure functions do the classification/dedup/summing so they unit-test on dicts;
build_value_summary() does the (thin) SQL. Spec:
docs/superpowers/specs/2026-07-30-value-found-design.md
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

DISCREPANCY_VALUE_TYPES = ("amount_over_po", "line_amount_over_po", "duplicate_invoice")
_EXCLUDED_STATUS = ("ignored", "superseded")
_STAGE_TIER = {"identified": "potential", "negotiation": "verified", "agreed": "verified",
               "realised": "verified", "rejected": None, "closed": None}
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def parse_amount(v: Any) -> Optional[float]:
    """STRICT: whole-string numbers only — a SHA-256 raw_value must return None."""
    s = re.sub(r"[£$€¥₹,\s]", "", str(v if v is not None else "")).strip()
    if not _NUM_RE.match(s):
        return None
    return float(s)


def discrepancy_delta(row: dict) -> Optional[float]:
    # convention 1: computed_value IS the signed delta ("+950.00") for these types
    d = parse_amount(row.get("computed_value"))
    if d is not None:
        return abs(d)
    o, e = parse_amount(row.get("raw_value")), parse_amount(row.get("expected_value"))
    if o is None or e is None:
        return None
    return round(abs(o - e), 2)


def _age_days(created_at) -> Optional[int]:
    if not isinstance(created_at, datetime):
        return None
    now = datetime.now(timezone.utc)
    ca = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return max((now - ca).days, 0)


def classify_discrepancy(row: dict) -> Optional[dict]:
    if row.get("issue_type") not in DISCREPANCY_VALUE_TYPES:
        return None
    if row.get("status") in _EXCLUDED_STATUS:
        return None
    delta = discrepancy_delta(row)
    if delta is None:
        return None
    recovered = None
    if row.get("status") == "resolved" and row.get("resolution_outcome") == "recovered":
        recovered = parse_amount(row.get("recovered_amount"))
        if recovered is None:
            recovered = delta          # spec: reader falls back to the delta
    return {
        "id": f"disc:{row.get('discrepancy_id')}",
        "tier": "verified",
        "source": "discrepancy",
        "amount_gbp": delta,           # converted later if currency != GBP
        "recovered_gbp": recovered,
        "converted_from": None,
        "currency": row.get("currency") or "GBP",
        "title": _disc_title(row, delta),
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id"),
        "doc_pk": row.get("doc_pk_candidate"),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "actions", "id": row.get("discrepancy_id")},
        "status": row.get("status"),
        "queryable": row.get("status") == "open" and bool(row.get("supplier_name")),
        "query_sent_at": row.get("query_sent_at").isoformat() if isinstance(row.get("query_sent_at"), datetime) else None,
        "superseded_by": None,
    }


def _disc_title(row: dict, delta: float) -> str:
    kind = {"duplicate_invoice": "appears to duplicate another invoice"}.get(
        row.get("issue_type"), "bills over its purchase order")
    return f"{row.get('doc_type', 'document').capitalize()} {row.get('doc_pk_candidate')} {kind} by {delta:,.2f}"


def classify_opportunity(row: dict) -> Optional[dict]:
    stage = row.get("stage")
    if stage not in _STAGE_TIER:
        raise ValueError(f"unmapped bp_opportunity stage {stage!r} — assign it a tier")
    tier = _STAGE_TIER[stage]
    if tier is None:
        return None
    amount = parse_amount(row.get("financial_impact_gbp")) or 0.0
    recovered = parse_amount(row.get("realised_savings_gbp")) if stage == "realised" else None
    if amount <= 0 and not recovered:
        return None
    return {
        "id": f"opp:{row.get('opportunity_id')}",
        "tier": tier,
        "source": "opportunity",
        "amount_gbp": amount,          # financial_impact_gbp is already native GBP
        "recovered_gbp": recovered,
        "converted_from": None,
        "currency": "GBP",
        "title": f"Opportunity: {row.get('item_description') or row.get('supplier_name') or 'unnamed'}",
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id"),
        "doc_pk": row.get("doc_pk") or row.get("po_id") or row.get("quote_id"),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "opportunities", "id": row.get("opportunity_id")},
        "status": stage,
        "queryable": False,
        "query_sent_at": None,
        "superseded_by": None,
    }


_PRECEDENCE = {"discrepancy": 0, "opportunity": 1, "benchmark": 2}


def _norm_item(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def dedupe(findings: list[dict]) -> list[dict]:
    """Same £ in several sources counts once, in the strongest source. Suppressed
    duplicates stay in the list flagged superseded_by (the drawer explains, never omits)."""
    best: dict[tuple, dict] = {}
    for f in sorted(findings, key=lambda f: _PRECEDENCE[f["source"]]):
        key = (f.get("deal_id"), f.get("doc_pk"), _norm_item(f.get("title") if f["source"] == "benchmark" else None))
        if f.get("deal_id") is None and f.get("doc_pk") is None:
            best[("solo", f["id"], "")] = f           # nothing to collide on
            continue
        if key in best:
            f["superseded_by"] = best[key]["id"]
        else:
            best[key] = f
    return findings


def summarise(findings: list[dict]) -> dict:
    live = [f for f in findings if f.get("superseded_by") is None]
    verified = [f for f in live if f["tier"] == "verified"]
    by_supplier: dict[str, dict] = {}
    for f in verified:
        name = f.get("supplier_name") or "Unknown supplier"
        g = by_supplier.setdefault(name, {"supplier_name": name, "verified_found_gbp": 0.0,
                                          "finding_count": 0})
        g["verified_found_gbp"] = round(g["verified_found_gbp"] + f["amount_gbp"], 2)
        g["finding_count"] += 1
    return {
        "verified_found_gbp": round(sum(f["amount_gbp"] for f in verified), 2),
        "recovered_gbp": round(sum(f["recovered_gbp"] or 0.0 for f in live), 2),
        "potential_gbp": round(sum(f["amount_gbp"] for f in live if f["tier"] == "potential"), 2),
        "finding_count": len(live),
        "by_supplier": sorted(by_supplier.values(),
                              key=lambda g: -g["verified_found_gbp"]),
    }
```

Then `build_value_summary(conn=None)` in the same file: three source loaders, each in its own `try/except` setting `sources[name] = "unavailable"` on failure (log with `log.exception`), never raising:

```python
def _rows(cur, sql, params=()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


_DISCREPANCY_SQL = """
SELECT e.discrepancy_id, e.doc_type, e.doc_pk_candidate, e.field_name, e.raw_value,
       e.expected_value, e.computed_value, e.issue_type, e.status, e.notes, e.created_at,
       e.resolution_outcome, e.recovered_amount, e.query_sent_at,
       (SELECT d.deal_id FROM proc.bp_deal_documents d
         WHERE d.doc_pk = e.doc_pk_candidate LIMIT 1) AS deal_id,
       i.supplier_name, i.currency
  FROM proc.bp_extraction_discrepancy e
  LEFT JOIN proc.bp_invoice_trgt i ON e.doc_type = 'invoice'
       AND i.invoice_id = e.doc_pk_candidate
 WHERE e.issue_type IN %s
"""
```

(Before coding, confirm `proc.bp_invoice_trgt`'s pk/supplier/currency column names with `\d proc.bp_invoice_trgt` — adjust the join accordingly. If `currency` is absent on a row, set the finding's `currency` to `None` and let the FX step EXCLUDE it from GBP sums, flagging it `converted_from: {"currency": "unknown"}` — never assume GBP for a foreign document.)

FX conversion helper (read `repositories/fx_rate_repo.py` first; rates are USD-quoted):

```python
def _to_gbp(amount: float, currency: Optional[str], rates: Optional[dict]) -> tuple[Optional[float], Optional[dict]]:
    if currency in (None, "", "GBP"):
        return (amount if currency == "GBP" or currency in (None, "") else None), None
    if not rates or currency not in rates or "GBP" not in rates:
        return None, {"currency": currency, "amount": amount, "rate_date": None}  # excluded
    gbp = round(amount / rates[currency] * rates["GBP"], 2)
    return gbp, {"currency": currency, "amount": amount, "rate_date": rates.get("_fetched_at")}
```

Benchmark source: call the existing evidence path (`src/services/benchmark_live.py`) guarded by `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=5)`; on `TimeoutError`/exception → `sources["benchmark"] = "unavailable"`. Findings from it use `source: "benchmark"`, `tier: "potential"`, positive deltas only, threshold ≥3 observations (reuse its existing gate — do not re-implement matching).

`build_value_summary` assembles: classify → FX-convert discrepancy amounts → dedupe → summarise → return dict with `findings`, `sources`, `since: None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_value_summary_service.py --import-mode=importlib -v`
Expected: PASS (all)

- [ ] **Step 5: Add source-failure isolation test and make it pass**

```python
def test_source_failure_isolation(monkeypatch):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "_load_discrepancies", lambda cur: (_ for _ in ()).throw(RuntimeError("db")))
    out = vss.build_value_summary(conn=FakeConn())   # FakeConn: cursor() returns a stub whose execute raises for opportunity SQL too if needed
    assert out["sources"]["discrepancies"] == "unavailable"
    assert out["verified_found_gbp"] == 0.0
```

(Structure `build_value_summary` so `_load_discrepancies(cur)`, `_load_opportunities(cur)`, `_load_benchmark()` are module-level and monkeypatchable.)

Run: same pytest command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/services/value_summary_service.py tests/services/test_value_summary_service.py
git commit -m "feat(value-found): aggregation core - tiers, dedup, FX, failure isolation"
```

---

### Task 3: `GET /spendiq/value-summary` router

**Files:**
- Create: `src/api/routers/value_summary.py`
- Modify: `src/api/main.py` (imports around line 390, `include_router` block ends at line 423)
- Test: `tests/api/test_value_summary_router.py` (create `tests/api/` if absent — check first; other router tests may live in `tests/`)

**Interfaces:**
- Consumes: `build_value_summary()` from Task 2.
- Produces: `GET /spendiq/value-summary` returning the spec's JSON contract + `generated_at`.

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient


def test_value_summary_endpoint(monkeypatch):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "build_value_summary", lambda conn=None: {
        "verified_found_gbp": 950.0, "recovered_gbp": 0.0, "potential_gbp": 0.0,
        "finding_count": 1, "since": None, "by_supplier": [], "findings": [],
        "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
    })
    from src.api.main import app
    client = TestClient(app)
    r = client.get("/spendiq/value-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["verified_found_gbp"] == 950.0
    assert "generated_at" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/api/test_value_summary_router.py --import-mode=importlib -v`
Expected: FAIL — 404 (route not registered)

- [ ] **Step 3: Implement the router and register it**

```python
# src/api/routers/value_summary.py
"""GET /spendiq/value-summary — the Value Found headline (W1).
One read-model over discrepancies + opportunities + benchmark deltas.
Spec: docs/superpowers/specs/2026-07-30-value-found-design.md"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.services import value_summary_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Value Found"])


@router.get("/value-summary", summary="Evidence-backed value found / recovered / potential")
def get_value_summary() -> dict:
    try:
        result = value_summary_service.build_value_summary()
    except Exception as exc:                     # the service isolates per-source failures;
        logger.exception("value-summary failed")  # reaching here means something structural
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
```

In `src/api/main.py`: import alongside the other routers (match the local import style used there, e.g. `from src.api.routers import value_summary as value_summary_router`) and add `app.include_router(value_summary_router.router)` after line 423's block. NOTE: the monkeypatch in the test targets the module attribute, so the router must call `value_summary_service.build_value_summary()` (module-qualified), not a `from`-imported name.

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Live smoke check**

Restart the running procwise service the house way (systemd unit `procwise.service` if active — check `systemctl status procwise`; NEVER `pkill -f uvicorn`), then:
Run: `curl -s http://localhost:8000/spendiq/value-summary | python3 -m json.tool | head -30`
Expected: 200 with real totals; the £950 over-billing finding present in `findings`. Record the actual numbers in the task notes.

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/value_summary.py src/api/main.py tests/api/test_value_summary_router.py
git commit -m "feat(value-found): /spendiq/value-summary endpoint"
```

---

### Task 4: Gateway — resolve endpoint learns outcomes

**Files (repo `/home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api`):**
- Modify: `src/modules/spendiq/spendiq.controller.ts:106-109`
- Modify: `src/modules/spendiq/spendiq.service.ts:1371-1388` (`resolveDiscrepancy`)
- Test: locate with `grep -rl "resolveDiscrepancy\|spendiq" src --include=*.spec.ts`; add cases there, or create `src/modules/spendiq/spendiq.service.spec.ts` following the nearest existing `*.service.spec.ts` pattern.

**Interfaces:**
- Consumes: Task 1's columns.
- Produces: `POST /spendiq/discrepancies/resolve` body gains optional `outcome` (`'recovered' | 'accepted'`) and `recoveredAmount` (number). Used by Task 6's drawer.

- [ ] **Step 1: Write the failing test** — assert that resolving with `{action:'apply_value', outcome:'recovered', recoveredAmount:500}` issues an UPDATE whose parameters include `'recovered'` and `500`, and that an outcome sent with a non-resolving verb (`dismiss`) is written as NULL. Mock `dataSource.query` (capture calls) per the existing spec pattern in that module.

- [ ] **Step 2: Run it to verify it fails**

Run: `npm test -- spendiq` (in the gateway repo). Expected: FAIL.

- [ ] **Step 3: Implement**

```ts
// controller
@Post('discrepancies/resolve')
async resolveDiscrepancy(@Body() body: any) {
  return this.spendIqService.resolveDiscrepancy(
    body?.id, body?.action, body?.resolvedBy, body?.outcome, body?.recoveredAmount);
}
```

```ts
// service — extend the existing method; keep every current behaviour identical
async resolveDiscrepancy(id: number, action?: string, resolvedBy?: string,
                         outcome?: string, recoveredAmount?: number) {
  const verb = String(action || 'apply_value').toLowerCase();
  const status = SpendIqService.RESOLUTION_STATUS[verb] ?? 'open';
  const resolutionAction = verb === 'reject'
    ? 'dismiss'
    : SpendIqService.ALLOWED_RESOLUTION_ACTIONS.has(verb) ? verb : null;
  // outcome only makes sense on a resolving verb; anything else stores NULL.
  const cleanOutcome = status === 'resolved' &&
    (outcome === 'recovered' || outcome === 'accepted') ? outcome : null;
  const cleanAmount = cleanOutcome === 'recovered' &&
    Number.isFinite(Number(recoveredAmount)) ? Number(recoveredAmount) : null;
  await this.dataSource.query(
    `UPDATE proc.bp_extraction_discrepancy
        SET status=$4, resolution_action=$2, resolved_by=$3,
            resolution_outcome=$5, recovered_amount=$6,
            resolved_at = CASE WHEN $4 = 'open' THEN NULL ELSE now() END
      WHERE discrepancy_id=$1`,
    [Number(id), resolutionAction, resolvedBy || 'ui', status, cleanOutcome, cleanAmount],
  );
  return { status: 'ok', id, resolution: verb, newStatus: status, outcome: cleanOutcome };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- spendiq`. Expected: PASS (new cases + the existing suite untouched).

- [ ] **Step 5: Commit (gateway repo, its current working branch)**

```bash
git add src/modules/spendiq/spendiq.controller.ts src/modules/spendiq/spendiq.service.ts <spec file>
git commit -m "feat(value-found): resolve endpoint records outcome + recovered amount"
```

---

### Task 5: Home tile + drawer (UI)

**Files (repo `/home/muthu/PycharmProjects/beyond_procwise_ui`):**
- Modify: `src/modules/ProcurementHome/useHomeData.js` (add the query), `src/modules/ProcurementHome/index.jsx` (tile + drawer), `src/modules/ProcurementHome/procurementHome.css`
- Create: `src/modules/ProcurementHome/ValueFoundDrawer.jsx`
- Test: `src/modules/ProcurementHome/valueFound.test.js` (follow `homeKpis.test.js` conventions)

**Interfaces:**
- Consumes: `GET ${VITE_AI_API_URL}/spendiq/value-summary` (Task 3). Fetch with the app's global-default axios exactly as `useHomeData.js`'s existing queries do.
- Produces: `useValueSummary()` hook returning `{data, isLoading, isError}`; `<ValueFoundDrawer summary={...} onClose={...}>` — reused conceptually (not literally) by Task 6's engine drawer.

- [ ] **Step 1: Write failing contract tests**

Cover, with a mocked summary payload: (a) the tile renders "value found" + "recovered" + quiet potential line, formatted through the display-currency helpers already used by the spend KPI; (b) `sources` containing an `"unavailable"` renders the "excludes N unavailable source(s)" note; (c) an all-zero summary renders the honest empty state "No verified findings yet — upload documents to begin"; (d) the methodology tooltip text is present (title/aria); (e) drawer groups rows under supplier headers from `by_supplier` order and shows "found 45 days ago" from `age_days`; (f) a `superseded_by` row renders muted with "counted under its stronger source".

- [ ] **Step 2: Run to verify they fail** — `npm test -- valueFound`. Expected: FAIL.

- [ ] **Step 3: Implement.** Tile goes next to the spend KPI in the KPI ticker (`index.jsx` ~line 392 block); clicking opens `ValueFoundDrawer`. Vocabulary per Global Constraints. Deep links: `navigate('/spendiq?view=actions')` (existing `gotoSpend` helper) for discrepancies, `view=opportunities` for opportunities; analysis-report only when `deal_id` present.

- [ ] **Step 4: Run to verify they pass** — `npm test -- valueFound`, then the module's full suite `npm test -- ProcurementHome`. Expected: PASS, no regressions.

- [ ] **Step 5: Commit (UI repo, branch `spendiq-ui`)**

```bash
git add src/modules/ProcurementHome
git commit -m "feat(value-found): Home hero tile + findings drawer"
```

---

### Task 6: SpendIQ Dashboard tile + drawer + vocabulary (UI)

**Files (repo `/home/muthu/PycharmProjects/beyond_procwise_ui`):**
- Modify: `src/modules/SpendIQ/data/useSpendData.js` (adapter + AI-API query, keyed `"home.valueSummary"`), `src/modules/SpendIQ/index.jsx` (wire query → `SD` slot), `src/modules/SpendIQ/engine.js` (Dashboard tile + drawer render; Opportunities-view label copy), `src/modules/SpendIQ/styles.css`
- Test: `src/modules/SpendIQ/valueSummary.contract.test.js`

**Interfaces:**
- Consumes: Task 3's endpoint via the existing AI-API query helper in `useSpendData.js` (the `${AI_API}${path}` pattern at line ~92).
- Produces: `adaptValueSummary(res)` → the engine-facing shape `{found, recovered, potential, findingCount, bySupplier, findings, partialNote}`; engine slot `SD('home.valueSummary', null)`.

- [ ] **Step 1: Write failing contract tests** — mirror Task 5's cases (a)-(f) against the engine render path (follow `chartPopOut.contract.test.js` / `reportBuilderLive.contract.test.js` conventions for driving `engine.js`), plus: (g) `SD` fallback is `null` → the Dashboard tile renders its loading/absent state, NEVER sample figures; (h) the Opportunities view's KPI labels read "value found"/"recovered" (copy-only change), not "savings".

- [ ] **Step 2: Run to verify they fail** — `npm test -- valueSummary`. Expected: FAIL.

- [ ] **Step 3: Implement.** `adaptValueSummary` maps GBP figures through the module's `moneyDisplay.js` display-currency helper exactly as the spend adapters do. Engine drawer reuses the module's existing overlay/drawer patterns; deep links call the engine's own `go('<view>')` navigation.

- [ ] **Step 4: Run to verify they pass** — `npm test -- valueSummary`, then the module suite. Expected: PASS; existing SpendIQ tests unaffected.

- [ ] **Step 5: Live check both surfaces.** With BP_Backend + gateway + UI running (gateway needs `node --experimental-global-webcrypto`; reload the page after engine edits — HMR leaves dead listeners): Home tile figure MUST equal Dashboard tile figure exactly.

- [ ] **Step 6: Commit**

```bash
git add src/modules/SpendIQ
git commit -m "feat(value-found): Dashboard tile, drawer, value-found vocabulary"
```

---

### Task 7: Duplicate-invoice detector (Phase 2)

**Files:**
- Create: `src/services/duplicate_invoice_detector.py`, `scripts/backfill_duplicate_invoices.py`
- Modify: `src/services/backend_scheduler.py` (`_run_downstream_chain`, line ~153 — add the detector after the existing downstream steps)
- Test: `tests/services/test_duplicate_invoice_detector.py`

**Interfaces:**
- Consumes: `proc.bp_invoice_trgt` rows; writes `proc.bp_extraction_discrepancy` rows with `issue_type='duplicate_invoice'` (flows into Tasks 2/3 with zero further wiring).
- Produces: `find_duplicates(invoices: list[dict]) -> list[dict]` (pure; each result `{later: row, earlier: row, amount: float}`) and `run_detector(conn=None) -> int` (rows written).

- [ ] **Step 1: Write failing tests for the pure rule**

```python
from datetime import datetime, timezone, timedelta
from src.services.duplicate_invoice_detector import find_duplicates, _refs_near


def _inv(iid, supplier="Techworld", total=1000.0, po="PO-1", days_ago=0, ref=None):
    return {"invoice_id": iid, "supplier_name": supplier, "total_amount": total,
            "po_id": po, "invoice_ref": ref or iid,
            "invoice_date": datetime.now(timezone.utc) - timedelta(days=days_ago)}


def test_same_supplier_total_and_po_within_window_flags_later():
    dups = find_duplicates([_inv("INV-1", days_ago=30), _inv("INV-2", days_ago=1)])
    assert len(dups) == 1
    assert dups[0]["later"]["invoice_id"] == "INV-2"
    assert dups[0]["amount"] == 1000.0


def test_near_identical_ref_counts_even_across_pos():
    a, b = _inv("INV-100", po="PO-1"), _inv("INV-100A", po="PO-2")
    assert find_duplicates([a, b])          # ref edit distance 1
    assert _refs_near("INV-100", "INV-100A") is True
    assert _refs_near("INV-100", "INV-200") is False


def test_recurring_charge_not_flagged():
    # same supplier + same amount but different PO and unrelated refs (monthly fee)
    a = _inv("INV-JAN", po="PO-1", ref="SVC-JAN", days_ago=60)
    b = _inv("INV-FEB", po="PO-2", ref="SVC-FEB", days_ago=30)
    assert find_duplicates([a, b]) == []


def test_outside_90_days_not_flagged():
    assert find_duplicates([_inv("INV-1", days_ago=120), _inv("INV-2", days_ago=1)]) == []


def test_different_supplier_or_amount_not_flagged():
    assert find_duplicates([_inv("INV-1"), _inv("INV-2", supplier="Acme")]) == []
    assert find_duplicates([_inv("INV-1"), _inv("INV-2", total=999.99)]) == []
```

- [ ] **Step 2: Run to verify they fail** — pytest as usual. Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Rule (spec, deliberately conservative): same normalised supplier AND equal totals (2-dp) AND (`same po_id` OR `_refs_near`, Levenshtein ≤ 1 on normalised refs) AND dates within 90 days AND different `invoice_id`. Flag the later invoice; `amount` = its full total. Implement Levenshtein inline (≤1 check is a simple O(n) scan — no dependency). `run_detector`: load invoices, run rule, and for each duplicate INSERT a discrepancy row — `doc_type='invoice'`, `doc_pk_candidate=later.invoice_id`, `field_name='invoice_ref'`, `issue_type='duplicate_invoice'`, `severity='critical'`, `blocks_promotion=False`, `raw_value=f"{later.total_amount:.2f}"`, `computed_value=f"+{amount:.2f}"` (keeps Task 2's delta convention), `notes=f"possible duplicate of {earlier.invoice_id} ({earlier.invoice_date:%Y-%m-%d}): same supplier and amount, matching reference"` — **idempotent**: skip when an un-ignored `duplicate_invoice` row already exists for that `doc_pk_candidate` (`SELECT 1 ... WHERE doc_pk_candidate=%s AND issue_type='duplicate_invoice' AND status != 'ignored'`). An `ignored` (dismissed) pair must NOT be re-raised — check `status='ignored'` rows too and skip the pair entirely.

Scheduler hook: append `run_detector()` at the end of `_run_downstream_chain` inside its own try/except (log, never break the chain). Backfill script: argparse `--dry-run` (default) printing would-be findings; `--apply` writes; prints the count either way.

- [ ] **Step 4: Run tests to verify they pass.** Expected: PASS.

- [ ] **Step 5: Run the backfill live**

Run: `set -a; . ./.env; set +a; ./venv/bin/python scripts/backfill_duplicate_invoices.py` (dry-run first; review every listed pair against the actual documents — with 44 invoices this is minutes) then `--apply`.
Expected: each applied finding traces to two real documents; count reported. Then `curl .../spendiq/value-summary` — the headline includes them.

- [ ] **Step 6: Commit**

```bash
git add src/services/duplicate_invoice_detector.py scripts/backfill_duplicate_invoices.py src/services/backend_scheduler.py tests/services/test_duplicate_invoice_detector.py
git commit -m "feat(value-found): duplicate-invoice detector + backfill"
```

---

### Task 8: Query-it backend (Phase 3)

**Files:**
- Modify: `src/api/routers/value_summary.py`, `src/services/value_summary_service.py`
- Create: `src/services/value_query_service.py`
- Test: `tests/services/test_value_query_service.py`

**Interfaces:**
- Consumes: Task 1's `query_sent_at`; `src/services/email_service.py::send_email`; `bp_prompt` via the governance prompt engine (read how `src/services/governance_tools/tools.py` fetches a prompt and reuse that path); supplier email from `proc.bp_supplier` (confirm the email column with `\d proc.bp_supplier` before coding).
- Produces:
  - `GET /spendiq/value-summary/findings/{finding_id}/query-draft` → `{"to": str|null, "subject": str, "body": str, "figures": {"delta": "950.00", "doc_ref": "INV-1042", "po_ref": "PO-2210"}}`
  - `POST /spendiq/value-summary/findings/{finding_id}/query-send` body `{"to": str, "subject": str, "body": str}` → `{"status": "sent", "query_sent_at": iso}`
  - `finding_id` format is Task 2's `disc:<discrepancy_id>`; only discrepancy findings are queryable.

- [ ] **Step 1: Write failing tests**

```python
def test_draft_figures_are_byte_equal_to_stored_values(...):
    # build a draft from a stubbed discrepancy row; assert the exact strings
    # "950.00", "INV-1042", "PO-2210" appear in the body, and that body figures
    # come from interpolation (template contains {delta}/{doc_ref}/{po_ref}).

def test_draft_refuses_non_queryable(...):
    # resolved finding, or supplier without an email -> 409-shaped error/ValueError

def test_send_stamps_query_sent_and_audits(...):
    # monkeypatch email_service.send_email -> success; assert UPDATE sets query_sent_at
    # and a bp_agent_actions row is written (monkeypatch the audit helper; assert called)

def test_send_failure_leaves_row_untouched(...):
    # send_email raises -> no query_sent_at UPDATE executed, error propagates cleanly
```

Write these as real tests against `value_query_service` with monkeypatched DB/email boundaries, following Task 2's Fake patterns.

- [ ] **Step 2: Run to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement `value_query_service.py`**

- `DEFAULT_TEMPLATE` module constant (subject `"Query on {doc_ref} against {po_ref}"`; body citing `{supplier_name}`, `{doc_ref}`, `{po_ref}`, `{delta}` and asking for a credit note / clarification; a distinct body variant for `duplicate_invoice` citing both invoice refs from the finding's notes). Fetch override from `bp_prompt` key `value_found_query_email` via the governance path; fall back to the constant.
- `build_draft(finding_id, conn=None)`: load the discrepancy row (must be `status='open'`, value issue type), resolve supplier email from `bp_supplier` by supplier name (exact match on the resolver's canonical name; no email → draft returned with `to: None` and the router surfaces 409 on send, not on draft). ALL figures formatted from stored values with the same `parse_amount`/`f"{x:,.2f}"` code — the template is interpolated in Python; no LLM call in v1 (grounding guaranteed by construction).
- `send_query(finding_id, to, subject, body, conn=None)`: re-check the row is still `open`; call `email_service.send_email`; on success `UPDATE ... SET query_sent_at = now()` and write a `proc.bp_agent_actions` row (`agent='value_found_query'`, follow the existing insert helper used elsewhere — locate with `grep -rn "bp_agent_actions" src/services | head`); on send failure, do NOT stamp — let the exception map to a 502 in the router. Per-finding only; no bulk path exists by design.

Router additions in `value_summary.py`: the two endpoints, thin, mapping `ValueError` → 409, send failure → 502.

- [ ] **Step 4: Run tests to verify they pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/value_query_service.py src/api/routers/value_summary.py tests/services/test_value_query_service.py
git commit -m "feat(value-found): query-it draft + send with grounded figures"
```

---

### Task 9: Query-it UI (Phase 3)

**Files (UI repo):**
- Modify: `src/modules/ProcurementHome/ValueFoundDrawer.jsx`, `src/modules/SpendIQ/engine.js` (drawer rows), shared row markup as built in Tasks 5/6
- Test: extend `valueFound.test.js` / `valueSummary.contract.test.js`

**Interfaces:**
- Consumes: Task 8's two endpoints; `queryable` + `query_sent_at` on finding rows.

- [ ] **Step 1: Write failing tests** — (a) "Query it" button renders ONLY on `queryable` rows; (b) clicking fetches the draft and opens a modal with editable to/subject/body prefilled; (c) confirming POSTs the (possibly edited) draft and the row now shows "Query sent today"; (d) a row with `query_sent_at` set 3 days ago shows "Query sent 3 days ago" and no button; (e) send failure shows the error on the row and leaves the button available.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** The review modal IS the approval gate — no send without an explicit human confirm; copy on the confirm button: "Approve & send".

- [ ] **Step 4: Run to verify they pass**, plus both modules' suites.

- [ ] **Step 5: Commit**

```bash
git add src/modules/ProcurementHome src/modules/SpendIQ
git commit -m "feat(value-found): query-it action in the findings drawer"
```

---

### Task 10: Weekly value digest (Phase 4)

**Files:**
- Create: `src/services/value_digest.py`
- Modify: `src/services/backend_scheduler.py` (register a weekly `ScheduledJob` following the existing job pattern at lines ~38-61 and the price-outlier registration)
- Test: `tests/services/test_value_digest.py`

**Interfaces:**
- Consumes: `build_value_summary()` (Task 2), `email_service.send_email`.
- Produces: `compose_digest(summary: dict, now: datetime) -> dict | None` (`{"subject", "body"}` or `None` for an empty week); `run_weekly_digest()` for the scheduler. Env: `VALUE_DIGEST_ENABLED` (default off), `VALUE_DIGEST_RECIPIENTS` (comma-separated; empty → skip with a log line).

- [ ] **Step 1: Write failing tests**

```python
def test_empty_week_sends_nothing():
    # no findings with found_at in the last 7 days AND nothing recovered this week
    assert compose_digest(summary_with_old_findings, now) is None

def test_digest_content():
    d = compose_digest(summary_fixture, now)
    assert "value found this week" in d["body"].lower()
    # top 3 OPEN findings by amount, each with its age
    assert d["body"].count("found ") >= 1 and "£" in d["body"]

def test_recipients_env_gate(monkeypatch):
    monkeypatch.delenv("VALUE_DIGEST_RECIPIENTS", raising=False)
    assert run_weekly_digest() == 0   # skipped, no send attempted
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** Weekly slice = findings whose `found_at` ≥ now−7d; recovered-this-week from findings whose recovery happened in the window (use `resolved_at` where exposed; if not exposed in the summary, extend Task 2's finding dict with `resolved_at` rather than guessing). Plain-text body; one link line to `/spendiq` (the drawer). Subject: `Value found this week: £X · £Y recovered`. Scheduler: weekly interval, guarded by both env vars, wrapped in try/except so a failure never breaks the chain.

- [ ] **Step 4: Run to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add src/services/value_digest.py src/services/backend_scheduler.py tests/services/test_value_digest.py
git commit -m "feat(value-found): weekly value digest (env-gated)"
```

---

### Task 11: End-to-end live verification + docs

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-value-found-design.md` (status line), `/home/muthu/PycharmProjects/beyond_procwise_ui/src/modules/SpendIQ/BACKEND_GAPS.md` (note the new live slot on Dashboard/Home)

- [ ] **Step 1: Full-stack run.** BP_Backend with `.env`; gateway (`node --experimental-global-webcrypto`); UI dev server. Fresh browser session.

- [ ] **Step 2: Verify the £950 case end-to-end.** (a) Home tile and Dashboard tile show identical figures and the drawer lists the £950 over-billing under Techworld's group with its age; (b) Query-it on it produces a draft whose delta/doc/PO refs byte-match the stored finding; approve & send (to a safe internal address configured for the test — do NOT email a real supplier from a dev run; point `bp_supplier`'s test row or the `to` override at your own address); row shows "Query sent today"; (c) resolve it via the gateway (`POST /spendiq/discrepancies/resolve` with `outcome: "recovered"`), reload — headline unchanged, *recovered* now includes £950.

- [ ] **Step 3: Verify honesty paths.** Stop the DB briefly OR monkeypatch-run the endpoint with a failing source: tile shows the partial/unavailable note. Confirm the duplicate-detector findings each trace to two real documents (from Task 7's backfill output).

- [ ] **Step 4: Run every suite touched.** BP_Backend pytest (new tests + `tests/services`), UI `npm test` (both modules), gateway `npm test -- spendiq`. All green, zero regressions.

- [ ] **Step 5: Update docs + final commit.** Spec status → "Implemented (Phases 1-4)"; BACKEND_GAPS note. Commit docs; do NOT push (Development stays local until asked).

---

## Self-review record

- **Spec coverage:** tiers/dedup/FX/time-window (T2), endpoint (T3), schema+resolve (T1/T4), tiles+drawer+tooltip+vocabulary+empty/partial states (T5/T6), duplicate detector+backfill+idempotence (T7), query-it grounded draft/send/audit (T8/T9), digest+empty-week rule (T10), live verification incl. £950 walkthrough (T11). Caveats need no tasks (documentation only).
- **Deliberate deviations:** none from the spec as amended 2026-07-30 (stage `negotiation`→verified; outcome enum `recovered|accepted`; `query_sent_at` column; template-interpolated draft).
- **Type consistency:** finding dict keys defined once in Task 2 and consumed verbatim in Tasks 3/5/6/8/10; `disc:<id>` finding-id format shared by Tasks 2 and 8.
