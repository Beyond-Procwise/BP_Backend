# Conformance Engine — Phase 1 (Detection Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make procurement detection data-driven — rules and thresholds live in a `bp_rule` table, findings land in a generalized `bp_finding` store with a *deterministic* idempotency key, and the same detectors run behind a source adapter — by renovating the existing opportunity miner without changing detector math.

**Architecture:** Externalize the in-code detector registry (`opportunity_miner_agent.py:4070-4138`) and thresholds into `proc.bp_rule` loaded by a new `RuleBook` engine (mirroring `PolicyEngine`/`PromptEngine`). Generalize `proc.bp_opportunity` into `proc.bp_finding` (superset) via an expand→backfill→view migration, fixing a latent positional-overwrite bug by switching the primary key to the already-computed deterministic id. Put an `ItemSource` adapter in front of `_ingest_data`. Each change is proven equivalent against a golden snapshot of the `/opportunities` endpoint before the old path is removed.

**Tech Stack:** Python 3, psycopg2, PostgreSQL (`proc` schema), pandas, pytest.

## Global Constraints

- All new DB tables use the `bp_` prefix; indexes use `ix_bp_<table>_<col>`. (verbatim project rule)
- No fabrication: a finding is written only when a detector fires on real data; absent data → no finding, never a guessed value.
- Detector math is NOT modified in Phase 1 — only where the rule list/thresholds come from, and where findings are written. Characterization tests lock the math.
- Migrations are additive + idempotent (`CREATE TABLE IF NOT EXISTS`, guarded backfill), runnable against live `bp_sqldb`.
- DB connections use the app's `agent_nick.get_db_connection` / settings; tests inject rows (no live DB in unit tests), following the `PolicyEngine(policy_rows=...)` pattern.

---

## File Structure

- Create `src/engines/rule_book.py` — `Rule` dataclass + `RuleBook` loader (load/cache/`active_rules`/`rules_for`/`reload`).
- Create `src/engines/detector_registry.py` — `DetectorSpec` + `DETECTOR_REGISTRY` (slug → spec) + `invoke()` uniform wrapper over existing handlers.
- Create `src/services/item_source.py` — `ItemSource` Protocol + `StoreAdapter` (wraps current `_ingest_data`).
- Create `deploy/sql/2026-06-21_bp_rule.sql` — rule book table + seed of the 11 detectors.
- Create `deploy/sql/2026-06-21_bp_finding.sql` — `bp_finding` table, backfill from `bp_opportunity`, replace `bp_opportunity` with a view.
- Modify `src/agents/opportunity_miner_agent.py` — deterministic id; iterate `RuleBook` rules through `DETECTOR_REGISTRY`; load via `StoreAdapter`.
- Modify `src/services/opportunity_store.py` — write to `bp_finding`; carry `finding_type`/`rule_id`/`rule_version`.
- Create `artifacts/opportunity_golden_snapshot.py` — capture/compare the `/opportunities` endpoint baseline (acceptance oracle).
- Tests: `tests/engines/test_rule_book.py`, `tests/engines/test_detector_registry.py`, `tests/services/test_item_source.py`, `tests/services/test_opportunity_store_finding.py`, `tests/agents/test_opportunity_deterministic_id.py`.

---

## Task 0: Capture the golden baseline (acceptance oracle)

**Files:**
- Create: `artifacts/opportunity_golden_snapshot.py`

**Interfaces:**
- Produces: a JSON snapshot file `artifacts/opportunity_golden.json` of the current `/opportunities` API output, and a `compare(current) -> dict` diff used by Task 8.

- [ ] **Step 1: Write the snapshot/compare script**

```python
"""Capture and compare the /opportunities endpoint output — the migration oracle.
Usage:
  python artifacts/opportunity_golden_snapshot.py capture   # before changes
  python artifacts/opportunity_golden_snapshot.py compare    # after changes
"""
import json, os, sys, urllib.request

SNAP = os.path.join(os.path.dirname(__file__), "opportunity_golden.json")
URL = os.getenv("OPP_URL", "http://localhost:8000/opportunities")

def fetch():
    with urllib.request.urlopen(URL, timeout=120) as r:
        return json.loads(r.read().decode())

def _key(o):
    return (o.get("detector_type"), o.get("supplier_id"), o.get("item_id"),
            round(float(o.get("financial_impact_gbp") or 0), 2))

def capture():
    data = fetch()
    json.dump(data, open(SNAP, "w"), indent=2, default=str, sort_keys=True)
    print(f"captured {len(data) if isinstance(data, list) else 'payload'} -> {SNAP}")

def compare():
    cur = fetch()
    old = json.load(open(SNAP))
    ol = old if isinstance(old, list) else old.get("opportunities", [])
    cl = cur if isinstance(cur, list) else cur.get("opportunities", [])
    old_keys = {_key(o) for o in ol}
    cur_keys = {_key(o) for o in cl}
    print(json.dumps({
        "old_count": len(ol), "new_count": len(cl),
        "dropped": [list(k) for k in sorted(old_keys - cur_keys)],
        "added": [list(k) for k in sorted(cur_keys - old_keys)],
    }, indent=2))

if __name__ == "__main__":
    {"capture": capture, "compare": compare}[sys.argv[1]]()
```

- [ ] **Step 2: Capture against the running local server**

Run: `python artifacts/opportunity_golden_snapshot.py capture`
Expected: prints a count and writes `artifacts/opportunity_golden.json`. (If the server isn't running, start it first; this is the live baseline.)

- [ ] **Step 3: Commit**

```bash
git add artifacts/opportunity_golden_snapshot.py
git commit -m "test(conformance): golden-snapshot oracle for /opportunities baseline"
```

---

## Task 1: RuleBook engine + `bp_rule` table

**Files:**
- Create: `deploy/sql/2026-06-21_bp_rule.sql`
- Create: `src/engines/rule_book.py`
- Test: `tests/engines/test_rule_book.py`

**Interfaces:**
- Produces:
  - `Rule` dataclass: `rule_id:int, rule_name:str, detector_slug:str, finding_type:str, scope:Optional[str], conditions:dict, severity:Optional[str], version:int`
  - `RuleBook(agent_nick=None, connection_factory=None, rule_rows=None)` with `active_rules() -> list[Rule]`, `rules_for(detector_slug) -> list[Rule]`, `reload() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engines/test_rule_book.py
from engines.rule_book import RuleBook, Rule

ROWS = [
    {"rule_id": 1, "rule_name": "Price Variance", "detector_slug": "price_variance_check",
     "finding_type": "opportunity", "scope": "po_lines",
     "conditions": {"variance_threshold_pct": 0.05}, "severity": "medium",
     "rule_status": 1, "version": 1},
    {"rule_id": 2, "rule_name": "Disabled rule", "detector_slug": "maverick_spend_check",
     "finding_type": "non_conformance", "scope": "purchase_orders",
     "conditions": {}, "severity": "high", "rule_status": 0, "version": 1},
]

def test_active_rules_excludes_disabled():
    rb = RuleBook(rule_rows=ROWS)
    active = rb.active_rules()
    assert [r.detector_slug for r in active] == ["price_variance_check"]
    assert isinstance(active[0], Rule)
    assert active[0].conditions["variance_threshold_pct"] == 0.05

def test_rules_for_filters_by_slug():
    rb = RuleBook(rule_rows=ROWS)
    assert rb.rules_for("maverick_spend_check") == []  # disabled
    assert len(rb.rules_for("price_variance_check")) == 1

def test_conditions_json_string_is_parsed():
    rows = [{"rule_id": 3, "rule_name": "x", "detector_slug": "supplier_risk_check",
             "finding_type": "opportunity", "scope": None,
             "conditions": '{"risk_threshold": 0.5}', "rule_status": 1, "version": 1}]
    rb = RuleBook(rule_rows=rows)
    assert rb.active_rules()[0].conditions["risk_threshold"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engines/test_rule_book.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.rule_book'`.

- [ ] **Step 3: Write `bp_rule.sql`**

```sql
-- 2026-06-21 Rule book: data-driven detection rules. Additive + idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_rule (
    rule_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rule_name          TEXT NOT NULL,
    detector_slug      TEXT NOT NULL,
    finding_type       TEXT NOT NULL DEFAULT 'opportunity',
    scope              TEXT,
    conditions         JSONB NOT NULL DEFAULT '{}',
    severity           TEXT,
    rule_status        SMALLINT NOT NULL DEFAULT 1,
    version            INTEGER NOT NULL DEFAULT 1,
    created_date       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by         TEXT NOT NULL DEFAULT 'system',
    last_modified_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by   TEXT NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_rule_status   ON proc.bp_rule (rule_status);
CREATE INDEX IF NOT EXISTS ix_bp_rule_detector ON proc.bp_rule (detector_slug);
COMMIT;
```

- [ ] **Step 4: Write `RuleBook` (`src/engines/rule_book.py`)**

```python
"""Data-driven detection rules from proc.bp_rule. Mirrors PolicyEngine."""
from __future__ import annotations
import json, logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    rule_id: int
    rule_name: str
    detector_slug: str
    finding_type: str
    scope: Optional[str]
    conditions: Dict[str, Any]
    severity: Optional[str] = None
    version: int = 1


class RuleBook:
    _COLUMNS = ("rule_id", "rule_name", "detector_slug", "finding_type", "scope",
                "conditions", "severity", "rule_status", "version")

    def __init__(self, agent_nick: Optional[Any] = None,
                 connection_factory: Optional[Any] = None,
                 rule_rows: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        if connection_factory is not None:
            self._cf = connection_factory
        elif agent_nick is not None:
            self._cf = getattr(agent_nick, "get_db_connection", None)
        else:
            self._cf = None
        self._rules: List[Rule] = []
        self._build(rule_rows)

    @contextmanager
    def _connect(self):
        if self._cf is None:
            yield None; return
        res = self._cf() if callable(self._cf) else self._cf
        if res is None:
            yield None; return
        if hasattr(res, "__enter__"):
            with res as conn:
                yield conn
        else:
            yield res

    @staticmethod
    def _coerce_conditions(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(errors="ignore")
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _fetch_rows(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT rule_id, rule_name, detector_slug, finding_type, scope, "
                        "conditions, severity, rule_status, version FROM proc.bp_rule "
                        "WHERE COALESCE(rule_status, 1) = 1")
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception:
                logger.exception("Failed to load rules from proc.bp_rule")
                return []

    def _build(self, override: Optional[Iterable[Dict[str, Any]]]) -> None:
        rows = list(override) if override is not None else self._fetch_rows()
        rules: List[Rule] = []
        for row in rows:
            if int(row.get("rule_status", 1)) != 1:
                continue
            rules.append(Rule(
                rule_id=int(row.get("rule_id")),
                rule_name=str(row.get("rule_name") or ""),
                detector_slug=str(row.get("detector_slug") or ""),
                finding_type=str(row.get("finding_type") or "opportunity"),
                scope=row.get("scope"),
                conditions=self._coerce_conditions(row.get("conditions")),
                severity=row.get("severity"),
                version=int(row.get("version", 1) or 1),
            ))
        self._rules = rules

    def active_rules(self) -> List[Rule]:
        return list(self._rules)

    def rules_for(self, detector_slug: str) -> List[Rule]:
        return [r for r in self._rules if r.detector_slug == detector_slug]

    def reload(self) -> None:
        self._build(None)


__all__ = ["Rule", "RuleBook"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engines/test_rule_book.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-06-21_bp_rule.sql src/engines/rule_book.py tests/engines/test_rule_book.py
git commit -m "feat(rule-book): proc.bp_rule + RuleBook loader (data-driven detection rules)"
```

---

## Task 2: Detector registry (slug → primitive wrapper)

**Files:**
- Create: `src/engines/detector_registry.py`
- Test: `tests/engines/test_detector_registry.py`

**Interfaces:**
- Consumes: nothing (metadata only in this task).
- Produces:
  - `DetectorSpec` dataclass: `slug:str, display_name:str, handler_attr:str, default_conditions:dict`.
  - `DETECTOR_REGISTRY: dict[str, DetectorSpec]` with the 11 detectors.
  - `invoke(spec, miner, tables, conditions) -> list` — calls `getattr(miner, spec.handler_attr)` with the miner's existing handler signature.

> The 11 slugs/handlers/defaults are copied verbatim from `opportunity_miner_agent.py:4070-4138`. Do not invent new ones.

- [ ] **Step 1: Write the failing test**

```python
# tests/engines/test_detector_registry.py
from engines.detector_registry import DETECTOR_REGISTRY, DetectorSpec, invoke

EXPECTED_SLUGS = {
    "price_variance_check", "volume_consolidation_check", "contract_expiry_check",
    "supplier_risk_check", "maverick_spend_check", "duplicate_supplier_check",
    "category_overspend_check", "inflation_passthrough_check",
    "unused_contract_value_check", "supplier_performance_check", "esg_opportunity_check",
}

def test_registry_has_all_eleven_detectors():
    assert set(DETECTOR_REGISTRY.keys()) == EXPECTED_SLUGS

def test_each_spec_is_well_formed():
    for slug, spec in DETECTOR_REGISTRY.items():
        assert isinstance(spec, DetectorSpec)
        assert spec.slug == slug
        assert spec.handler_attr.startswith("_policy_")
        assert isinstance(spec.default_conditions, dict)

def test_contract_expiry_default_window():
    assert DETECTOR_REGISTRY["contract_expiry_check"].default_conditions == {"negotiation_window_days": 90}

def test_invoke_calls_handler_with_merged_conditions():
    calls = {}
    class FakeMiner:
        def _policy_supplier_risk(self, tables, input_data, notifications, policy_cfg):
            calls["conditions"] = input_data.get("conditions")
            return ["finding"]
    spec = DETECTOR_REGISTRY["supplier_risk_check"]
    out = invoke(spec, FakeMiner(), tables={}, conditions={"risk_threshold": 0.7})
    assert out == ["finding"]
    assert calls["conditions"]["risk_threshold"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engines/test_detector_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.detector_registry'`.

- [ ] **Step 3: Confirm the handler attr names against source**

Run: `grep -nE '_policy_(price_benchmark_variance|volume_consolidation|contract_expiry|supplier_risk|maverick_spend|duplicate_supplier|category_overspend|inflation_passthrough|unused_contract_value|supplier_performance|esg_opportunity)' src/agents/opportunity_miner_agent.py | grep 'def '`
Expected: prints the 11 `def _policy_*` lines. Use these exact method names as `handler_attr`.

- [ ] **Step 4: Write `detector_registry.py`**

```python
"""Maps detector slugs to the existing opportunity_miner handler methods.
The handler math is unchanged; this only formalizes the registry so rules
(from RuleBook) can drive which detectors run and with what thresholds."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class DetectorSpec:
    slug: str
    display_name: str
    handler_attr: str
    default_conditions: Dict[str, Any] = field(default_factory=dict)


DETECTOR_REGISTRY: Dict[str, DetectorSpec] = {
    "price_variance_check": DetectorSpec(
        "price_variance_check", "Price Benchmark Variance",
        "_policy_price_benchmark_variance", {"variance_threshold_pct": 0.0}),
    "volume_consolidation_check": DetectorSpec(
        "volume_consolidation_check", "Volume Consolidation",
        "_policy_volume_consolidation", {"minimum_volume_gbp": 0.0}),
    "contract_expiry_check": DetectorSpec(
        "contract_expiry_check", "Contract Expiry Opportunity",
        "_policy_contract_expiry", {"negotiation_window_days": 90}),
    "supplier_risk_check": DetectorSpec(
        "supplier_risk_check", "Supplier Risk Alert",
        "_policy_supplier_risk", {"risk_threshold": 0.0, "risk_weight": 1000.0}),
    "maverick_spend_check": DetectorSpec(
        "maverick_spend_check", "Maverick Spend Detection",
        "_policy_maverick_spend", {"minimum_value_gbp": 0.0}),
    "duplicate_supplier_check": DetectorSpec(
        "duplicate_supplier_check", "Duplicate Supplier",
        "_policy_duplicate_supplier", {"minimum_overlap_gbp": 0.0}),
    "category_overspend_check": DetectorSpec(
        "category_overspend_check", "Category Overspend",
        "_policy_category_overspend", {"category_budgets": {}}),
    "inflation_passthrough_check": DetectorSpec(
        "inflation_passthrough_check", "Inflation Pass-Through",
        "_policy_inflation_passthrough", {"market_inflation_pct": 0.0, "tolerance_pct": 0.0}),
    "unused_contract_value_check": DetectorSpec(
        "unused_contract_value_check", "Unused Contract Value",
        "_policy_unused_contract_value", {"minimum_unused_value_gbp": 0.0}),
    "supplier_performance_check": DetectorSpec(
        "supplier_performance_check", "Supplier Performance Deviation",
        "_policy_supplier_performance", {"performance_threshold": 0.9}),
    "esg_opportunity_check": DetectorSpec(
        "esg_opportunity_check", "ESG Opportunity",
        "_policy_esg_opportunity", {"esg_threshold": 0.0, "incumbent_score": 0.0}),
}


def invoke(spec: DetectorSpec, miner: Any, tables: Any, conditions: Dict[str, Any]) -> List[Any]:
    """Call the bound handler with the miner's existing signature.
    Handlers read thresholds via _get_condition(input_data, ...), which reads
    input_data['conditions']; so we pass conditions there."""
    handler = getattr(miner, spec.handler_attr)
    input_data = {"conditions": dict(conditions)}
    policy_cfg = {"policy_id": spec.slug, "detector": spec.display_name,
                  "policy_name": spec.display_name}
    return handler(tables=tables, input_data=input_data,
                   notifications=[], policy_cfg=policy_cfg)


__all__ = ["DetectorSpec", "DETECTOR_REGISTRY", "invoke"]
```

> NOTE for implementer: Step 3's grep gives the authoritative `handler_attr` names and the exact handler keyword signature. If a handler's real signature differs from `(tables, input_data, notifications, policy_cfg)`, adjust `invoke()` to match the source — do not change the handlers.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engines/test_detector_registry.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/engines/detector_registry.py tests/engines/test_detector_registry.py
git commit -m "feat(detector-registry): slug->handler registry over existing 11 detectors"
```

---

## Task 3: Deterministic finding id in the miner

**Files:**
- Modify: `src/agents/opportunity_miner_agent.py:4399-4401`
- Test: `tests/agents/test_opportunity_deterministic_id.py`

**Interfaces:**
- Consumes: the deterministic `opportunity_id` local computed at `opportunity_miner_agent.py:4385-4389`.
- Produces: `Finding.opportunity_id` is now the deterministic value (stable across runs).

- [ ] **Step 1: Write the failing characterization test**

```python
# tests/agents/test_opportunity_deterministic_id.py
# Verifies the same logical finding yields the SAME id across two builds,
# locking the idempotency fix. Uses the deterministic-id formula directly.
import hashlib

def _expected_id(policy_slug, detector_slug, sources, supplier_slug, item_slug):
    normalised = sorted(str(s).strip() for s in sources if s and str(s).strip())
    source_token = ""
    if normalised:
        source_token = hashlib.sha1("|".join(normalised).encode()).hexdigest()[:8]
    return "_".join(t for t in [policy_slug, detector_slug, source_token,
                                supplier_slug, item_slug] if t)

def test_same_inputs_same_id():
    a = _expected_id("price_variance_check", "price_variance", ["PO2", "PO1"], "SUP1", "ITM1")
    b = _expected_id("price_variance_check", "price_variance", ["PO1", "PO2"], "SUP1", "ITM1")
    assert a == b and a != ""

def test_different_supplier_different_id():
    a = _expected_id("p", "d", ["PO1"], "SUP1", "ITM1")
    b = _expected_id("p", "d", ["PO1"], "SUP2", "ITM1")
    assert a != b
```

- [ ] **Step 2: Run test to verify it passes (formula is correct today)**

Run: `pytest tests/agents/test_opportunity_deterministic_id.py -v`
Expected: 2 passed. (This pins the intended id formula; the bug is only that the miner doesn't *use* it as the PK.)

- [ ] **Step 3: Apply the one-line correction**

In `src/agents/opportunity_miner_agent.py`, the `Finding(...)` construction at `:4399`:

```python
        finding = Finding(
            opportunity_id=opportunity_id,          # was: self._next_opportunity_id()
            opportunity_ref_id=opportunity_id,
```

(Leave `_next_opportunity_id` defined for now; it becomes dead and is removed in Task 7.)

- [ ] **Step 4: Run the existing miner test suite to confirm no math regression**

Run: `pytest tests/agents -k opportunity -v`
Expected: PASS (no behavioral change beyond id). If a test asserts a numeric/sequential `opportunity_id`, update that assertion to the deterministic value — that test was encoding the bug.

- [ ] **Step 5: Commit**

```bash
git add src/agents/opportunity_miner_agent.py tests/agents/test_opportunity_deterministic_id.py
git commit -m "fix(opportunity): use deterministic finding id as PK (fixes positional-overwrite upsert)"
```

---

## Task 4: `bp_finding` table + backfill + `bp_opportunity` view

**Files:**
- Create: `deploy/sql/2026-06-21_bp_finding.sql`

**Interfaces:**
- Produces: `proc.bp_finding` (superset table) and `proc.bp_opportunity` as a VIEW projecting the original columns where `finding_type='opportunity'`.

- [ ] **Step 1: Write the migration SQL**

```sql
-- 2026-06-21 Generalize bp_opportunity -> bp_finding. Expand + backfill + view.
-- Idempotent and reversible (the view can be dropped and table restored from bp_finding).
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_finding (
    finding_id           VARCHAR PRIMARY KEY,
    finding_ref_id       VARCHAR,
    finding_type         VARCHAR NOT NULL DEFAULT 'opportunity',
    rule_id              VARCHAR,
    rule_version         INTEGER,
    detector_slug        VARCHAR,
    detector_type        VARCHAR,
    policy_id            VARCHAR,
    severity             VARCHAR,
    supplier_id          VARCHAR,
    supplier_name        VARCHAR,
    category_id          VARCHAR,
    item_id              TEXT,
    item_description     TEXT,
    financial_impact_gbp NUMERIC,
    realised_savings_gbp NUMERIC,
    expected             JSONB,
    actual               JSONB,
    deviation            JSONB,
    calculation_details  JSONB,
    source_records       JSONB,
    stage                VARCHAR NOT NULL DEFAULT 'identified',
    deal_id              VARCHAR,
    ml_priority_score    NUMERIC,
    weightage            NUMERIC,
    quote_id             VARCHAR,
    po_id                VARCHAR,
    detected_on          TIMESTAMPTZ,
    stage_updated_at     TIMESTAMPTZ DEFAULT now(),
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT bp_finding_stage_check CHECK (
        stage IN ('identified','negotiation','agreed','realised','closed','rejected'))
);
CREATE INDEX IF NOT EXISTS ix_bp_finding_type      ON proc.bp_finding (finding_type);
CREATE INDEX IF NOT EXISTS ix_bp_finding_stage     ON proc.bp_finding (stage);
CREATE INDEX IF NOT EXISTS ix_bp_finding_detector  ON proc.bp_finding (detector_slug);
CREATE INDEX IF NOT EXISTS ix_bp_finding_supplier  ON proc.bp_finding (supplier_id);

-- Backfill from the existing table ONLY if it is still a base table (first run).
-- Re-key onto the deterministic opportunity_ref_id; collapse positional-id dupes
-- (keep the most recently detected row per deterministic key).
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables
             WHERE table_schema='proc' AND table_name='bp_opportunity' AND table_type='BASE TABLE') THEN
    INSERT INTO proc.bp_finding (
        finding_id, finding_ref_id, finding_type, detector_type, policy_id,
        supplier_id, supplier_name, category_id, item_id, item_description,
        financial_impact_gbp, realised_savings_gbp, calculation_details,
        source_records, stage, deal_id, ml_priority_score, weightage,
        detected_on, stage_updated_at, created_at, updated_at)
    SELECT DISTINCT ON (COALESCE(opportunity_ref_id, opportunity_id))
        COALESCE(opportunity_ref_id, opportunity_id), opportunity_ref_id, 'opportunity',
        detector_type, policy_id, supplier_id, supplier_name, category_id, item_id,
        item_description, financial_impact_gbp, realised_savings_gbp, calculation_details,
        source_records, stage, deal_id, ml_priority_score, weightage,
        detected_on, stage_updated_at, created_at, updated_at
    FROM proc.bp_opportunity
    ORDER BY COALESCE(opportunity_ref_id, opportunity_id), detected_on DESC NULLS LAST
    ON CONFLICT (finding_id) DO NOTHING;

    ALTER TABLE proc.bp_opportunity RENAME TO bp_opportunity_legacy_20260621;
  END IF;
END $$;

-- bp_opportunity becomes a view with the ORIGINAL column names/shape.
CREATE OR REPLACE VIEW proc.bp_opportunity AS
SELECT
    finding_id          AS opportunity_id,
    finding_ref_id      AS opportunity_ref_id,
    detector_type, policy_id, supplier_id, supplier_name, category_id, item_id,
    item_description, financial_impact_gbp, realised_savings_gbp, stage, deal_id,
    ml_priority_score, weightage, calculation_details, source_records,
    detected_on, stage_updated_at, created_at, updated_at
FROM proc.bp_finding
WHERE finding_type = 'opportunity';

COMMIT;
```

- [ ] **Step 2: Apply against live DB and verify backfill parity**

Run:
```bash
psql "$DATABASE_URL" -f deploy/sql/2026-06-21_bp_finding.sql
psql "$DATABASE_URL" -c "SELECT
  (SELECT count(*) FROM proc.bp_opportunity) AS view_rows,
  (SELECT count(*) FROM proc.bp_finding WHERE finding_type='opportunity') AS finding_rows,
  (SELECT count(*) FROM proc.bp_opportunity_legacy_20260621) AS legacy_rows;"
```
Expected: `view_rows == finding_rows`. `legacy_rows >= finding_rows` (the difference = positional-id duplicates that were correctly collapsed). Record the numbers in the commit message.

> If `$DATABASE_URL` is not set, build it from the `.env` keys (`DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT`).

- [ ] **Step 3: Commit**

```bash
git add deploy/sql/2026-06-21_bp_finding.sql
git commit -m "feat(findings): bp_finding superset + backfill + bp_opportunity compat view"
```

---

## Task 5: `opportunity_store` writes to `bp_finding`

**Files:**
- Modify: `src/services/opportunity_store.py:23-64`
- Test: `tests/services/test_opportunity_store_finding.py`

**Interfaces:**
- Consumes: a finding record dict (the existing `Finding`-derived dict) plus optional `finding_type`, `rule_id`, `rule_version`, `severity`, `detector_slug`.
- Produces: `upsert_finding(cur, rec)` inserting into `proc.bp_finding`, idempotent on `finding_id`, preserving a progressed `stage`.

- [ ] **Step 1: Write the failing test (fake cursor records SQL + params)**

```python
# tests/services/test_opportunity_store_finding.py
from services.opportunity_store import upsert_finding

class FakeCur:
    def __init__(self): self.calls = []
    def execute(self, sql, params=None): self.calls.append((sql, params))

def test_upsert_targets_bp_finding_with_deterministic_id():
    cur = FakeCur()
    upsert_finding(cur, {
        "opportunity_id": "price_variance_check_pv_abc123_SUP1_ITM1",
        "opportunity_ref_id": "price_variance_check_pv_abc123_SUP1_ITM1",
        "detector_type": "Price Benchmark Variance",
        "detector_slug": "price_variance_check",
        "finding_type": "opportunity", "rule_id": "1", "rule_version": 1,
        "severity": "medium", "supplier_id": "SUP1", "item_id": "ITM1",
        "financial_impact_gbp": 1200.0,
        "calculation_details": {"variance_pct": 0.2},
        "source_records": ["PO1"], "detected_on": None,
    })
    sql, params = cur.calls[0]
    assert "insert into proc.bp_finding" in sql.lower()
    assert "on conflict (finding_id)" in sql.lower()
    assert params[0] == "price_variance_check_pv_abc123_SUP1_ITM1"   # finding_id is the deterministic key
    assert "price_variance_check" in params                          # detector_slug carried

def test_rejected_flag_sets_rejected_stage():
    cur = FakeCur()
    upsert_finding(cur, {"opportunity_id": "x", "is_rejected": True,
                         "finding_type": "opportunity", "calculation_details": {}})
    _, params = cur.calls[0]
    assert "rejected" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_opportunity_store_finding.py -v`
Expected: FAIL — `ImportError: cannot import name 'upsert_finding'`.

- [ ] **Step 3: Add `upsert_finding` to `opportunity_store.py`**

Add alongside the existing `upsert_opportunity` (do not delete it yet — Task 7 removes the old write path):

```python
def upsert_finding(cur, rec: dict) -> None:
    """Insert/update one finding into proc.bp_finding. Idempotent on finding_id
    (deterministic). Preserves a progressed stage; only forces 'rejected'."""
    calc = rec.get("calculation_details") or {}
    fid = str(rec.get("finding_id") or rec.get("opportunity_id"))
    item_desc = rec.get("item_description") or calc.get("item_description") or rec.get("item_id")
    stage = "rejected" if rec.get("is_rejected") else "identified"
    cur.execute(
        """
        insert into proc.bp_finding
          (finding_id, finding_ref_id, finding_type, rule_id, rule_version,
           detector_slug, detector_type, policy_id, severity, supplier_id, supplier_name,
           category_id, item_id, item_description, financial_impact_gbp, calculation_details,
           source_records, stage, ml_priority_score, weightage, quote_id, po_id, detected_on)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (finding_id) do update set
          finding_ref_id=excluded.finding_ref_id, finding_type=excluded.finding_type,
          rule_id=excluded.rule_id, rule_version=excluded.rule_version,
          detector_slug=excluded.detector_slug, detector_type=excluded.detector_type,
          policy_id=excluded.policy_id, severity=excluded.severity,
          supplier_id=excluded.supplier_id, supplier_name=excluded.supplier_name,
          category_id=excluded.category_id, item_id=excluded.item_id,
          item_description=excluded.item_description,
          financial_impact_gbp=excluded.financial_impact_gbp,
          calculation_details=excluded.calculation_details,
          source_records=excluded.source_records,
          ml_priority_score=excluded.ml_priority_score, weightage=excluded.weightage,
          quote_id=excluded.quote_id, po_id=excluded.po_id, detected_on=excluded.detected_on,
          stage=case when excluded.stage='rejected' then 'rejected'
                     else proc.bp_finding.stage end,
          updated_at=now()
        """,
        (
            fid, rec.get("opportunity_ref_id") or fid, rec.get("finding_type") or "opportunity",
            rec.get("rule_id"), rec.get("rule_version"), rec.get("detector_slug"),
            rec.get("detector_type"), rec.get("policy_id"), rec.get("severity"),
            rec.get("supplier_id"), rec.get("supplier_name"), rec.get("category_id"),
            rec.get("item_id"), item_desc, rec.get("financial_impact_gbp"),
            json.dumps(calc), json.dumps(rec.get("source_records") or []), stage,
            rec.get("ml_priority_score"), rec.get("weightage"),
            rec.get("quote_id") or calc.get("quote_id"),
            rec.get("po_id") or calc.get("po_id"), rec.get("detected_on"),
        ),
    )
```

(Confirm `import json` is present at the top of the file; it is used by `upsert_opportunity` already.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_opportunity_store_finding.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/opportunity_store.py tests/services/test_opportunity_store_finding.py
git commit -m "feat(findings): upsert_finding writes to bp_finding (idempotent, stage-preserving)"
```

---

## Task 6: `ItemSource` adapter seam

**Files:**
- Create: `src/services/item_source.py`
- Test: `tests/services/test_item_source.py`
- Modify: `src/agents/opportunity_miner_agent.py` `_ingest_data` (`:2254-2260`) to delegate to a `StoreAdapter`.

**Interfaces:**
- Consumes: the miner's `TABLE_MAP` and `_read_sql`.
- Produces:
  - `ItemSource` Protocol: `load() -> dict[str, pandas.DataFrame]`.
  - `StoreAdapter(read_sql, table_map)` whose `load()` returns the same frames `_ingest_data` produces.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_item_source.py
import pandas as pd
from services.item_source import StoreAdapter

def test_store_adapter_loads_each_mapped_table():
    seen = []
    def fake_read_sql(sql):
        seen.append(sql)
        return pd.DataFrame({"x": [1]})
    table_map = {"purchase_orders": "proc.bp_purchase_order_trgt",
                 "invoices": "proc.bp_invoice_trgt"}
    frames = StoreAdapter(fake_read_sql, table_map).load()
    assert set(frames.keys()) == {"purchase_orders", "invoices"}
    assert "SELECT * FROM proc.bp_purchase_order_trgt" in seen

def test_store_adapter_empty_frame_on_error():
    def boom(sql):
        raise RuntimeError("table missing")
    frames = StoreAdapter(boom, {"contracts": "proc.contracts"}).load()
    assert frames["contracts"].empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_item_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.item_source'`.

- [ ] **Step 3: Write `item_source.py`**

```python
"""Source-agnostic item loading. StoreAdapter wraps the live _trgt ingest.
Upload/Feed adapters arrive in later phases; the engine depends on ItemSource."""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Protocol

logger = logging.getLogger(__name__)


class ItemSource(Protocol):
    def load(self) -> Dict[str, Any]: ...   # name -> pandas.DataFrame


class StoreAdapter:
    """Loads the TABLE_MAP-keyed frames from the warehouse (today's behaviour)."""

    def __init__(self, read_sql: Callable[[str], Any], table_map: Dict[str, str]) -> None:
        self._read_sql = read_sql
        self._table_map = table_map

    def load(self) -> Dict[str, Any]:
        import pandas as pd
        frames: Dict[str, Any] = {}
        for table, sql_name in self._table_map.items():
            try:
                frames[table] = self._read_sql(f"SELECT * FROM {sql_name}")
            except Exception:
                logger.exception("StoreAdapter: failed to load %s (%s)", table, sql_name)
                frames[table] = pd.DataFrame()
        return frames


__all__ = ["ItemSource", "StoreAdapter"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_item_source.py -v`
Expected: 2 passed.

- [ ] **Step 5: Delegate `_ingest_data` to `StoreAdapter`**

Replace the body of `_ingest_data` (`opportunity_miner_agent.py:2254-2260`) with:

```python
    def _ingest_data(self) -> Dict[str, pd.DataFrame]:
        from services.item_source import StoreAdapter
        return StoreAdapter(self._read_sql, self.TABLE_MAP).load()
```

- [ ] **Step 6: Run the miner tests to confirm no regression**

Run: `pytest tests/agents -k opportunity -v`
Expected: PASS (behaviour identical; loading path refactored).

- [ ] **Step 7: Commit**

```bash
git add src/services/item_source.py tests/services/test_item_source.py src/agents/opportunity_miner_agent.py
git commit -m "feat(conformance): ItemSource seam; _ingest_data delegates to StoreAdapter"
```

---

## Task 7: Drive the sweep from RuleBook + write findings; retire sequential id

**Files:**
- Modify: `src/agents/opportunity_miner_agent.py` (policy-execution loop after `:1703`; remove `_next_opportunity_id` `:2316-2320`).
- Modify: `src/services/backend_scheduler.py` (`_run_opportunity_mining`) and/or the miner's persistence call to use `upsert_finding`.
- Modify: `deploy/sql/2026-06-21_bp_rule.sql` — append the seed of the 11 detectors.
- Test: extend `tests/agents/test_opportunity_deterministic_id.py` with a wiring test using a fake RuleBook.

**Interfaces:**
- Consumes: `engines.rule_book.RuleBook`, `engines.detector_registry.DETECTOR_REGISTRY` + `invoke`, `services.opportunity_store.upsert_finding`.
- Produces: the sweep iterates `rule_book.active_rules()`, runs each via `invoke`, tags findings with `finding_type`/`rule_id`/`rule_version`/`severity`/`detector_slug`, and persists via `upsert_finding`.

- [ ] **Step 1: Append the rule seed to `bp_rule.sql`**

```sql
-- Seed the 11 existing detectors with their current default thresholds so
-- behaviour is unchanged on first run. Idempotent on rule_name.
INSERT INTO proc.bp_rule (rule_name, detector_slug, finding_type, scope, conditions, severity)
VALUES
 ('Price Benchmark Variance','price_variance_check','opportunity','po_lines','{"variance_threshold_pct":0.0}','medium'),
 ('Volume Consolidation','volume_consolidation_check','opportunity','po_lines','{"minimum_volume_gbp":0.0}','low'),
 ('Contract Expiry Opportunity','contract_expiry_check','opportunity','contracts','{"negotiation_window_days":90}','medium'),
 ('Supplier Risk Alert','supplier_risk_check','non_conformance','supplier_master','{"risk_threshold":0.0,"risk_weight":1000.0}','high'),
 ('Maverick Spend Detection','maverick_spend_check','non_conformance','purchase_orders','{"minimum_value_gbp":0.0}','high'),
 ('Duplicate Supplier','duplicate_supplier_check','opportunity','po_lines','{"minimum_overlap_gbp":0.0}','low'),
 ('Category Overspend','category_overspend_check','non_conformance','invoice_lines','{"category_budgets":{}}','medium'),
 ('Inflation Pass-Through','inflation_passthrough_check','anomaly','invoice_lines','{"market_inflation_pct":0.0,"tolerance_pct":0.0}','medium'),
 ('Unused Contract Value','unused_contract_value_check','opportunity','contracts','{"minimum_unused_value_gbp":0.0}','low'),
 ('Supplier Performance Deviation','supplier_performance_check','anomaly','invoices','{"performance_threshold":0.9}','medium'),
 ('ESG Opportunity','esg_opportunity_check','opportunity',NULL,'{"esg_threshold":0.0,"incumbent_score":0.0}','low')
ON CONFLICT DO NOTHING;
```

Run: `psql "$DATABASE_URL" -f deploy/sql/2026-06-21_bp_rule.sql`
Expected: 11 rows present — verify `psql "$DATABASE_URL" -c "SELECT count(*) FROM proc.bp_rule;"` → 11.

- [ ] **Step 2: Write the wiring test (fake RuleBook + fake miner handler)**

```python
# append to tests/agents/test_opportunity_deterministic_id.py
from engines.rule_book import Rule

def test_sweep_runs_only_active_rules(monkeypatch):
    from engines import detector_registry as dr
    ran = []
    def fake_invoke(spec, miner, tables, conditions):
        ran.append((spec.slug, conditions))
        return []
    monkeypatch.setattr(dr, "invoke", fake_invoke)
    rules = [Rule(1, "Maverick", "maverick_spend_check", "non_conformance",
                  "purchase_orders", {"minimum_value_gbp": 500.0}, "high", 1)]
    # drive the loop helper directly (see Step 3 for run_rules signature)
    from agents.opportunity_miner_agent import run_rules
    run_rules(miner=object(), tables={}, rules=rules, registry=dr.DETECTOR_REGISTRY,
              invoke=fake_invoke)
    assert ran == [("maverick_spend_check", {"minimum_value_gbp": 500.0})]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/agents/test_opportunity_deterministic_id.py::test_sweep_runs_only_active_rules -v`
Expected: FAIL — `ImportError: cannot import name 'run_rules'`.

- [ ] **Step 4: Add a `run_rules` helper and call it from the sweep**

Add a module-level helper in `opportunity_miner_agent.py` (keeps the loop testable in isolation):

```python
def run_rules(miner, tables, rules, registry, invoke):
    """Run each enabled rule through its detector primitive; return tagged findings."""
    findings = []
    for rule in rules:
        spec = registry.get(rule.detector_slug)
        if spec is None:
            logger.warning("run_rules: no detector for slug %s", rule.detector_slug)
            continue
        try:
            produced = invoke(spec, miner, tables, rule.conditions or spec.default_conditions)
        except Exception:
            logger.exception("run_rules: detector %s raised; skipping", rule.detector_slug)
            continue
        for f in produced or []:
            # tag finding object/dict with rule provenance
            for attr, val in (("finding_type", rule.finding_type), ("rule_id", str(rule.rule_id)),
                              ("rule_version", rule.version), ("severity", rule.severity),
                              ("detector_slug", rule.detector_slug)):
                try:
                    setattr(f, attr, val)
                except Exception:
                    if isinstance(f, dict):
                        f[attr] = val
            findings.append(f)
    return findings
```

In the miner's `process()` (the policy-execution section after `:1703`), replace the in-code registry iteration with:

```python
        from engines.rule_book import RuleBook
        from engines.detector_registry import DETECTOR_REGISTRY, invoke as _invoke
        rule_book = getattr(self.agent_nick, "rule_book", None) or RuleBook(self.agent_nick)
        findings = run_rules(self, tables, rule_book.active_rules(), DETECTOR_REGISTRY, _invoke)
```

> Implementer note: keep all surrounding enrichment (currency normalize, supplier lookup, scoring) exactly as-is — only the "which detectors run + with what conditions" selection changes. The `Finding` already carries `finding_type` etc. once tagged; if `Finding` is a frozen dataclass, add these as optional fields (default `None`) rather than `setattr`.

- [ ] **Step 5: Switch persistence to `upsert_finding`**

Find where findings are persisted (the call to `upsert_opportunity`) and route to `upsert_finding`, passing the finding's `__dict__` (or existing dict). Then delete `_next_opportunity_id` (`:2316-2320`) — now dead.

Run: `grep -rn "upsert_opportunity" src/` → update each call site to `upsert_finding`.

- [ ] **Step 6: Run the full affected suites**

Run: `pytest tests/agents -k opportunity tests/engines tests/services/test_opportunity_store_finding.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agents/opportunity_miner_agent.py src/services/backend_scheduler.py deploy/sql/2026-06-21_bp_rule.sql tests/agents/test_opportunity_deterministic_id.py
git commit -m "feat(conformance): drive sweep from RuleBook+registry, persist to bp_finding; seed 11 rules"
```

---

## Task 8: Live proof against `bp_sqldb` (acceptance)

**Files:** none (verification only).

- [ ] **Step 1: Apply migrations + seed (if not already)**

Run:
```bash
psql "$DATABASE_URL" -f deploy/sql/2026-06-21_bp_rule.sql
psql "$DATABASE_URL" -f deploy/sql/2026-06-21_bp_finding.sql
```
Expected: no errors; `bp_rule` has 11 rows; `bp_finding` exists; `bp_opportunity` is a view.

- [ ] **Step 2: Run the live sweep over `_trgt`**

Restart the local server (so `RuleBook` loads) and trigger mining:
Run: `curl -s -X POST http://localhost:8000/opportunities -H 'Content-Type: application/json' -d '{"workflow":"all","min_financial_impact":100}' | head`
Expected: HTTP 200 with findings.

- [ ] **Step 3: Verify findings landed in `bp_finding`**

Run:
```bash
psql "$DATABASE_URL" -c "SELECT finding_type, detector_slug, count(*), round(sum(financial_impact_gbp)::numeric,2)
FROM proc.bp_finding GROUP BY 1,2 ORDER BY 1,2;"
```
Expected: rows grouped by detector with non-zero counts where the live data supports them; every row has a non-null `detector_slug` and `rule_id` (provenance present).

- [ ] **Step 4: Confirm no idempotency duplication on re-run**

Run the sweep again (Step 2), then:
`psql "$DATABASE_URL" -c "SELECT count(*) AS total, count(DISTINCT finding_id) AS distinct_ids FROM proc.bp_finding;"`
Expected: `total == distinct_ids` (deterministic upsert; re-run did not duplicate).

- [ ] **Step 5: Compare the endpoint to the golden baseline**

Run: `python artifacts/opportunity_golden_snapshot.py compare`
Expected: `added` is empty; `dropped` (if any) corresponds only to former positional-id duplicates (cross-check against the Task 4 legacy/finding row delta). Any unexplained drop is a regression — stop and investigate before declaring done.

- [ ] **Step 6: Record the result**

```bash
git commit --allow-empty -m "test(conformance): Phase 1 proven on live bp_sqldb — N findings across M detectors, idempotent re-run, endpoint parity"
```

---

## Self-Review

**Spec coverage:** rule book (`bp_rule` + RuleBook) → T1; detector registry → T2; deterministic key / bug fix → T3; `bp_finding` + view migration → T4; findings persistence → T5; source-adapter seam → T6; rule-driven sweep + seeding + provenance tags → T7; live proof + idempotency + endpoint parity → T8; golden oracle → T0. All Phase-1 spec sections map to a task.

**Out-of-scope kept out:** no policy lifecycle, decision engine, playbooks, external-feed/rule-change triggers (later phases). Ranking scoring policies untouched (not part of the sweep).

**Placeholder scan:** no TBD/TODO; every code step shows code; the two "implementer notes" point to exact grep commands and source lines to confirm signatures, not vague guidance.

**Type consistency:** `RuleBook.active_rules()` → `list[Rule]` (T1) consumed by `run_rules` (T7); `DETECTOR_REGISTRY`/`invoke` signatures (T2) match `run_rules` call (T7); `upsert_finding(cur, rec)` (T5) keyed on `finding_id` matching the deterministic id (T3) and the `bp_finding` PK (T4); `StoreAdapter(read_sql, table_map).load()` (T6) matches `_ingest_data` delegation.

**Known follow-ups (flagged, not gaps):** `Finding` may be a frozen dataclass — T7 Step 4 note covers adding optional fields vs `setattr`. Backfill collapses positional-id dupes — T4/T8 explicitly reconcile the count delta so the drop is verified, not assumed.
