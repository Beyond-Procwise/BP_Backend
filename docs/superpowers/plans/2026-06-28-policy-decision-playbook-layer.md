# Decision Layer (Policy / Decision / Playbook) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the conformance decision layer — a pure permission Policy Engine, human-authored Playbooks, and a Decision Engine — all run by an orchestrator envelope that applies policy + decision to every agent action (with or without a playbook), and dissolve the legacy `bp_policy` config store into rules and playbooks.

**Architecture:** Add four data-driven engines that mirror the Phase-1 `RuleBook` pattern (a `bp_` table + a load-once engine + `rows=` test injection). The orchestrator wraps every agent invocation in a `gate → act → observe` envelope; a playbook is an optional multi-step composition, the no-playbook case is a single enveloped step. Generalize the existing (broken) `ReasoningEngine.observe()` into the Decision Engine. Migrate each legacy `bp_policy` row to where it belongs (detection → `bp_rule`, ranking strategy → a playbook, negotiation thresholds → pure gates), repointing every caller before removing dead code.

**Tech Stack:** Python 3, psycopg2, PostgreSQL (`proc` schema), pandas, pytest, FastAPI.

## Prerequisites

This plan builds on Conformance Phase 1. Before starting, confirm Phase 1 is implemented (or implement it first via `docs/superpowers/plans/2026-06-20-conformance-engine-phase1.md`):
- `proc.bp_rule` table + `engines/rule_book.py::RuleBook`
- `proc.bp_finding` table + `engines/detector_registry.py::DETECTOR_REGISTRY`

Run to verify: `python -c "from engines.rule_book import RuleBook; from engines.detector_registry import DETECTOR_REGISTRY; print(len(DETECTOR_REGISTRY))"` → expect `11`. If this errors, build Phase 1 first.

## Global Constraints

- All new DB tables use the `bp_` prefix; indexes use `ix_bp_<table>_<col>`. (verbatim project rule)
- Policies only **gate outcomes** (`allow | require_approval | deny | escalate`); they never tune thresholds or enable/disable rules.
- Governance is a universal envelope: policy (gate) + decision (observe) apply to **every** agent invocation; the playbook is optional. No agent runs without policy and a decision.
- Playbooks are human-authored expert strategy; the system executes them, never invents them.
- Human-in-the-loop by default: playbook steps are `human_gated` unless explicitly whitelisted `auto`.
- No fabrication: a decision/action is recorded only from real findings, real policy rows, and real playbook steps.
- Migrations are additive + idempotent (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, guarded backfill), runnable against live `bp_sqldb`.
- Engines degrade gracefully when the DB is unreachable (the `_safe_engine` pattern): empty sets; decisions default to `escalate` + `requires_human`. Never fail open.
- Tests inject rows (no live DB in unit tests), following `PolicyEngine(policy_rows=...)`.
- DB connections use `agent_nick.get_db_connection`. Engine constructors accept `(agent_nick=None, connection_factory=None, <thing>_rows=None)`.

---

## File Structure

- Modify `src/engines/policy_engine.py` — add `Policy`/`Gate` dataclasses + `applicable_gates()`/`is_allowed()`; later remove the legacy config methods.
- Create `src/engines/playbook.py` — `Playbook`/`PlaybookStep` dataclasses + `PlaybookEngine` loader.
- Create `src/engines/action_registry.py` — `ActionSpec` + `ACTION_REGISTRY` + `invoke()`.
- Create `src/services/playbook_runner.py` — `PlaybookRunner` (gate → act → observe per step).
- Create `src/engines/decision_engine.py` — `Decision` + `DecisionEngine` (generalizes `observe()`).
- Modify `src/orchestration/reasoning_engine.py` — repoint `observe()` at `DecisionEngine`.
- Modify `src/orchestration/orchestrator.py` — universal envelope + default path; repoint legacy policy calls.
- Modify `src/agents/supplier_ranking_agent.py` — read ranking config from the Supplier Ranking playbook.
- Modify `src/engines/negotiation_strategy_engine.py` — `evaluate_policy_rails()` calls `applicable_gates()`.
- Modify `src/agents/opportunity_miner_agent.py` — `iter_policies()` reader → `RuleBook`.
- Modify `src/api/routers/agents.py` — extend `/agents/reload-governance` to reload playbooks.
- Create migrations under `deploy/sql/2026-06-28_*.sql`.
- Tests under `tests/engines/`, `tests/services/`, `tests/agents/`, `tests/orchestration/`.

---

## Task 1: Pure-policy schema + governance audit table

**Files:**
- Create: `deploy/sql/2026-06-28_bp_policy_pure.sql`
- Create: `deploy/sql/2026-06-28_bp_governance_change.sql`

**Interfaces:**
- Produces: `proc.bp_policy` gains structured columns `applies_to JSONB`, `effect TEXT`, `priority INT`, `policy_status TEXT`; new table `proc.bp_governance_change`.

- [ ] **Step 1: Write the pure-policy migration (additive columns + status backfill)**

```sql
-- 2026-06-28 Make bp_policy a pure permission table. Additive + idempotent.
BEGIN;
ALTER TABLE proc.bp_policy ADD COLUMN IF NOT EXISTS applies_to JSONB NOT NULL DEFAULT '{}';
ALTER TABLE proc.bp_policy ADD COLUMN IF NOT EXISTS effect      TEXT;
ALTER TABLE proc.bp_policy ADD COLUMN IF NOT EXISTS priority    INTEGER NOT NULL DEFAULT 100;
-- normalise the legacy smallint status into a text lifecycle column
ALTER TABLE proc.bp_policy ADD COLUMN IF NOT EXISTS policy_lifecycle TEXT NOT NULL DEFAULT 'active';
CREATE INDEX IF NOT EXISTS ix_bp_policy_lifecycle ON proc.bp_policy (policy_lifecycle);
CREATE INDEX IF NOT EXISTS ix_bp_policy_effect    ON proc.bp_policy (effect);
COMMIT;
```

(We keep the legacy columns for now; Task 13 retires the legacy config rows after callers are repointed.)

- [ ] **Step 2: Write the governance-change migration**

```sql
-- 2026-06-28 Governance versioning/approval audit. Additive + idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_governance_change (
    change_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    target_table  TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    version_from  INTEGER,
    version_to    INTEGER,
    old_value     JSONB,
    new_value     JSONB,
    change_status TEXT NOT NULL DEFAULT 'pending_approval',
    changed_by    TEXT NOT NULL DEFAULT 'system',
    approved_by   TEXT,
    changed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_bp_governance_change_target
    ON proc.bp_governance_change (target_table, target_id);
COMMIT;
```

- [ ] **Step 3: Apply (or dry-run) against the DB**

Run: `psql "$DATABASE_URL" -f deploy/sql/2026-06-28_bp_policy_pure.sql -f deploy/sql/2026-06-28_bp_governance_change.sql`
Expected: no errors. If `$DATABASE_URL` is unset, build it from `.env` (`DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT`). If the DB is in REDUCED mode (unreachable), record that the migration is staged and continue — unit tests do not need the DB.

- [ ] **Step 4: Commit**

```bash
git add deploy/sql/2026-06-28_bp_policy_pure.sql deploy/sql/2026-06-28_bp_governance_change.sql
git commit -m "feat(policy): pure bp_policy columns (applies_to/effect/priority/lifecycle) + bp_governance_change"
```

---

## Task 2: Pure Policy API — `applicable_gates` / `is_allowed`

**Files:**
- Modify: `src/engines/policy_engine.py`
- Test: `tests/engines/test_policy_gates.py`

**Interfaces:**
- Produces:
  - `Gate` dataclass: `effect:str, priority:int, policy_id:Any, rationale:Optional[str]`.
  - `PolicyEngine.applicable_gates(context: dict) -> list[Gate]` — active policies whose `applies_to` matches `context`.
  - `PolicyEngine.is_allowed(context: dict) -> Gate` — single resolved verdict; most-restrictive wins (`deny > escalate > require_approval > allow`), tie-broken by `priority`.
- Consumes: rows shaped `{policy_id, policy_name, applies_to, effect, priority, policy_lifecycle}` (via `policy_rows=`).

- [ ] **Step 1: Write the failing test**

```python
# tests/engines/test_policy_gates.py
from engines.policy_engine import PolicyEngine, Gate

ROWS = [
    {"policy_id": 1, "policy_name": "PO over 50k", "policy_lifecycle": "active",
     "applies_to": {"finding_type": ["opportunity"], "financial_impact_gbp": {">": 50000}},
     "effect": "require_approval", "priority": 100},
    {"policy_id": 2, "policy_name": "Low value auto", "policy_lifecycle": "active",
     "applies_to": {"financial_impact_gbp": {"<": 1000}}, "effect": "allow", "priority": 50},
    {"policy_id": 3, "policy_name": "Retired", "policy_lifecycle": "retired",
     "applies_to": {}, "effect": "deny", "priority": 1},
]

def test_applicable_gates_matches_context_and_excludes_retired():
    pe = PolicyEngine(policy_rows=ROWS)
    gates = pe.applicable_gates({"finding_type": "opportunity", "financial_impact_gbp": 60000})
    assert [g.effect for g in gates] == ["require_approval"]
    assert isinstance(gates[0], Gate)

def test_is_allowed_most_restrictive_wins():
    pe = PolicyEngine(policy_rows=ROWS)
    # both a deny-ish and an allow match → restrictive wins
    rows = ROWS + [{"policy_id": 9, "policy_name": "block cat", "policy_lifecycle": "active",
                    "applies_to": {"category_id": ["X"]}, "effect": "deny", "priority": 100}]
    pe = PolicyEngine(policy_rows=rows)
    verdict = pe.is_allowed({"category_id": "X", "financial_impact_gbp": 500})
    assert verdict.effect == "deny"

def test_is_allowed_defaults_to_allow_when_no_match():
    pe = PolicyEngine(policy_rows=ROWS)
    assert pe.is_allowed({"finding_type": "anomaly"}).effect == "allow"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engines/test_policy_gates.py -v`
Expected: FAIL — `ImportError: cannot import name 'Gate'`.

- [ ] **Step 3: Add the pure API to `policy_engine.py`**

Add near the top (after imports) and as new methods on `PolicyEngine` (do NOT delete the legacy methods yet):

```python
from dataclasses import dataclass

_EFFECT_RANK = {"allow": 0, "require_approval": 1, "escalate": 2, "deny": 3}


@dataclass
class Gate:
    effect: str
    priority: int
    policy_id: object = None
    rationale: "Optional[str]" = None


def _match_condition(cond, value) -> bool:
    """cond is a dict like {">":50000} / {"<":1000} / {"in":[...]}, or a list (membership),
    or a scalar (equality)."""
    if isinstance(cond, dict):
        for op, target in cond.items():
            if op == ">" and not (value is not None and value > target):
                return False
            if op == "<" and not (value is not None and value < target):
                return False
            if op == "in" and value not in target:
                return False
            if op == "eq" and value != target:
                return False
        return True
    if isinstance(cond, list):
        return value in cond
    return value == cond
```

Add these methods to the class:

```python
    @staticmethod
    def _coerce_applies_to(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(errors="ignore")
        if isinstance(value, str) and value.strip():
            try:
                import json as _json
                parsed = _json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _gate_rows(self):
        """Active rows that carry an effect (the pure permission policies)."""
        for row in (self._policy_rows_cache or []):
            if (row.get("policy_lifecycle") or "active") != "active":
                continue
            if not row.get("effect"):
                continue
            yield row

    def applicable_gates(self, context: dict):
        gates = []
        for row in self._gate_rows():
            applies = self._coerce_applies_to(row.get("applies_to"))
            if all(_match_condition(cond, context.get(key)) for key, cond in applies.items()):
                gates.append(Gate(effect=row["effect"], priority=int(row.get("priority", 100)),
                                  policy_id=row.get("policy_id"), rationale=row.get("policy_name")))
        return gates

    def is_allowed(self, context: dict):
        gates = self.applicable_gates(context)
        if not gates:
            return Gate(effect="allow", priority=0)
        return sorted(gates, key=lambda g: (_EFFECT_RANK.get(g.effect, 0), g.priority), reverse=True)[0]
```

In `__init__`, cache the raw rows so `_gate_rows` can read them. Find where `policy_rows`/`_load_policies` is handled and add:

```python
        self._policy_rows_cache = list(policy_rows) if policy_rows is not None else self._fetch_policy_rows()
```

(Place this assignment before the existing `_load_policies` call so both share the fetched rows. Reuse `_fetch_policy_rows`; extend its SELECT to also return `applies_to, effect, priority, policy_lifecycle`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engines/test_policy_gates.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the existing policy tests to confirm no regression**

Run: `pytest tests/test_policy_engine.py -v`
Expected: PASS (legacy methods untouched).

- [ ] **Step 6: Commit**

```bash
git add src/engines/policy_engine.py tests/engines/test_policy_gates.py
git commit -m "feat(policy): pure applicable_gates/is_allowed API (effect resolution, most-restrictive wins)"
```

---

## Task 3: Playbook tables

**Files:**
- Create: `deploy/sql/2026-06-28_bp_playbook.sql`

**Interfaces:**
- Produces: `proc.bp_playbook`, `proc.bp_playbook_step`, `proc.bp_playbook_run`, `proc.bp_playbook_run_step`.

- [ ] **Step 1: Write the migration**

```sql
-- 2026-06-28 Playbooks (human-authored response strategy). Additive + idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_playbook (
    playbook_id     VARCHAR PRIMARY KEY,
    playbook_name   TEXT NOT NULL,
    trigger_kind    VARCHAR NOT NULL,            -- finding | event
    trigger_match   JSONB NOT NULL DEFAULT '{}',
    playbook_status VARCHAR NOT NULL DEFAULT 'draft',
    version         INTEGER NOT NULL DEFAULT 1,
    authored_by     TEXT NOT NULL DEFAULT 'system',
    approved_by     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS proc.bp_playbook_step (
    step_id     VARCHAR PRIMARY KEY,
    playbook_id VARCHAR NOT NULL REFERENCES proc.bp_playbook (playbook_id),
    step_no     INTEGER NOT NULL,
    action_slug VARCHAR NOT NULL,
    params      JSONB NOT NULL DEFAULT '{}',
    mode        VARCHAR NOT NULL DEFAULT 'human_gated',  -- human_gated | auto
    condition   JSONB,
    on_success  INTEGER,
    on_failure  INTEGER
);
CREATE TABLE IF NOT EXISTS proc.bp_playbook_run (
    run_id      VARCHAR PRIMARY KEY,
    playbook_id VARCHAR,                          -- NULL/'default' for the no-playbook path
    finding_id  VARCHAR,
    event_ref   VARCHAR,
    run_status  VARCHAR NOT NULL DEFAULT 'running',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS proc.bp_playbook_run_step (
    run_step_id VARCHAR PRIMARY KEY,
    run_id      VARCHAR NOT NULL REFERENCES proc.bp_playbook_run (run_id),
    step_no     INTEGER NOT NULL,
    step_status VARCHAR NOT NULL DEFAULT 'pending',
    result      JSONB,
    acted_by    VARCHAR,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_status   ON proc.bp_playbook (playbook_status);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_step_pb  ON proc.bp_playbook_step (playbook_id);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_run_find ON proc.bp_playbook_run (finding_id);
COMMIT;
```

- [ ] **Step 2: Apply (or stage) against the DB**

Run: `psql "$DATABASE_URL" -f deploy/sql/2026-06-28_bp_playbook.sql`
Expected: no errors (or staged if DB in REDUCED mode).

- [ ] **Step 3: Commit**

```bash
git add deploy/sql/2026-06-28_bp_playbook.sql
git commit -m "feat(playbook): bp_playbook + step + run + run_step tables"
```

---

## Task 4: PlaybookEngine

**Files:**
- Create: `src/engines/playbook.py`
- Test: `tests/engines/test_playbook_engine.py`

**Interfaces:**
- Produces:
  - `PlaybookStep` dataclass: `step_no:int, action_slug:str, params:dict, mode:str, condition:Optional[dict], on_success:Optional[int], on_failure:Optional[int]`.
  - `Playbook` dataclass: `playbook_id:str, playbook_name:str, trigger_kind:str, trigger_match:dict, steps:list[PlaybookStep], version:int`.
  - `PlaybookEngine(agent_nick=None, connection_factory=None, playbook_rows=None, step_rows=None)` with `active_playbooks() -> list[Playbook]`, `playbook_for(trigger_kind, match) -> Optional[Playbook]`, `reload() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engines/test_playbook_engine.py
from engines.playbook import PlaybookEngine, Playbook, PlaybookStep

PB = [{"playbook_id": "pb_supplier_risk", "playbook_name": "Supplier Risk Response",
       "trigger_kind": "finding", "trigger_match": {"finding_type": "non_conformance"},
       "playbook_status": "active", "version": 1},
      {"playbook_id": "pb_draft", "playbook_name": "Disabled", "trigger_kind": "finding",
       "trigger_match": {"finding_type": "opportunity"}, "playbook_status": "draft", "version": 1}]
STEPS = [{"step_id": "s1", "playbook_id": "pb_supplier_risk", "step_no": 1,
          "action_slug": "notify_buyer", "params": {}, "mode": "human_gated"}]

def test_active_playbooks_excludes_non_active():
    eng = PlaybookEngine(playbook_rows=PB, step_rows=STEPS)
    assert [p.playbook_id for p in eng.active_playbooks()] == ["pb_supplier_risk"]
    assert isinstance(eng.active_playbooks()[0].steps[0], PlaybookStep)

def test_playbook_for_matches_finding_type():
    eng = PlaybookEngine(playbook_rows=PB, step_rows=STEPS)
    pb = eng.playbook_for("finding", {"finding_type": "non_conformance"})
    assert pb.playbook_id == "pb_supplier_risk"
    assert eng.playbook_for("finding", {"finding_type": "opportunity"}) is None  # draft excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engines/test_playbook_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.playbook'`.

- [ ] **Step 3: Write `playbook.py`**

```python
"""Human-authored response playbooks from proc.bp_playbook. Mirrors RuleBook."""
from __future__ import annotations
import json, logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlaybookStep:
    step_no: int
    action_slug: str
    params: Dict[str, Any] = field(default_factory=dict)
    mode: str = "human_gated"
    condition: Optional[Dict[str, Any]] = None
    on_success: Optional[int] = None
    on_failure: Optional[int] = None


@dataclass
class Playbook:
    playbook_id: str
    playbook_name: str
    trigger_kind: str
    trigger_match: Dict[str, Any]
    steps: List[PlaybookStep]
    version: int = 1


def _coerce(value, default):
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="ignore")
    if isinstance(value, str) and value.strip():
        try:
            p = json.loads(value)
            return p if isinstance(p, dict) else default
        except Exception:
            return default
    return default


class PlaybookEngine:
    def __init__(self, agent_nick=None, connection_factory=None,
                 playbook_rows=None, step_rows=None):
        self._cf = connection_factory or (getattr(agent_nick, "get_db_connection", None)
                                          if agent_nick is not None else None)
        self._playbooks: List[Playbook] = []
        self._build(playbook_rows, step_rows)

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

    def _fetch(self, sql):
        with self._connect() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception:
                logger.exception("PlaybookEngine fetch failed: %s", sql)
                return []

    def _build(self, pb_rows, step_rows):
        pbs = list(pb_rows) if pb_rows is not None else self._fetch(
            "SELECT playbook_id, playbook_name, trigger_kind, trigger_match, "
            "playbook_status, version FROM proc.bp_playbook WHERE playbook_status='active'")
        steps = list(step_rows) if step_rows is not None else self._fetch(
            "SELECT step_id, playbook_id, step_no, action_slug, params, mode, "
            "condition, on_success, on_failure FROM proc.bp_playbook_step ORDER BY step_no")
        by_pb: Dict[str, List[PlaybookStep]] = {}
        for s in steps:
            by_pb.setdefault(s["playbook_id"], []).append(PlaybookStep(
                step_no=int(s["step_no"]), action_slug=s["action_slug"],
                params=_coerce(s.get("params"), {}), mode=s.get("mode") or "human_gated",
                condition=_coerce(s.get("condition"), None) or None,
                on_success=s.get("on_success"), on_failure=s.get("on_failure")))
        result = []
        for p in pbs:
            if (p.get("playbook_status") or "active") != "active":
                continue
            result.append(Playbook(
                playbook_id=p["playbook_id"], playbook_name=p.get("playbook_name") or "",
                trigger_kind=p.get("trigger_kind") or "finding",
                trigger_match=_coerce(p.get("trigger_match"), {}),
                steps=sorted(by_pb.get(p["playbook_id"], []), key=lambda x: x.step_no),
                version=int(p.get("version", 1) or 1)))
        self._playbooks = result

    def active_playbooks(self) -> List[Playbook]:
        return list(self._playbooks)

    def playbook_for(self, trigger_kind: str, match: Dict[str, Any]) -> Optional[Playbook]:
        for pb in self._playbooks:
            if pb.trigger_kind != trigger_kind:
                continue
            if all(match.get(k) == v for k, v in pb.trigger_match.items()):
                return pb
        return None

    def reload(self) -> None:
        self._build(None, None)


__all__ = ["Playbook", "PlaybookStep", "PlaybookEngine"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engines/test_playbook_engine.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engines/playbook.py tests/engines/test_playbook_engine.py
git commit -m "feat(playbook): PlaybookEngine loader (active_playbooks/playbook_for)"
```

---

## Task 5: ActionRegistry

**Files:**
- Create: `src/engines/action_registry.py`
- Test: `tests/engines/test_action_registry.py`

**Interfaces:**
- Produces:
  - `ActionSpec` dataclass: `slug:str, display_name:str, handler_attr:str, default_params:dict`.
  - `ACTION_REGISTRY: dict[str, ActionSpec]` — the seed actions.
  - `invoke(spec, agents, context, params) -> dict` — calls the wrapped agent and returns a result dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/engines/test_action_registry.py
from engines.action_registry import ACTION_REGISTRY, ActionSpec, invoke

def test_registry_has_seed_actions():
    assert {"rank_suppliers", "draft_supplier_email", "open_negotiation",
            "request_approval", "notify_buyer", "update_finding_stage"} <= set(ACTION_REGISTRY)

def test_each_spec_well_formed():
    for slug, spec in ACTION_REGISTRY.items():
        assert isinstance(spec, ActionSpec) and spec.slug == slug
        assert isinstance(spec.default_params, dict)

def test_invoke_calls_mapped_agent():
    calls = {}
    class FakeAgent:
        def run_action(self, context, params):
            calls["params"] = params
            return {"ok": True}
    spec = ActionSpec("notify_buyer", "Notify Buyer", "notify_buyer", {})
    out = invoke(spec, agents={"notify_buyer": FakeAgent()}, context={}, params={"msg": "hi"})
    assert out["ok"] is True and calls["params"]["msg"] == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engines/test_action_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Confirm the agent keys against the live registry**

Run: `grep -n "self.agents\[" src/agents/base_agent.py src/orchestration/orchestrator.py | head`
Use the real agent keys (e.g. `email_drafting`, `negotiation`, `approvals`, `supplier_ranking`) as `handler_attr`. If an agent exposes a specific method rather than a generic `run_action`, set that in `invoke` (see note).

- [ ] **Step 4: Write `action_registry.py`**

```python
"""Maps action slugs to existing agents. Mirror of the Phase-1 DetectorRegistry.
The agent logic is unchanged; this only formalizes what a playbook step can call."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionSpec:
    slug: str
    display_name: str
    handler_attr: str          # key into agent_nick.agents
    default_params: Dict[str, Any] = field(default_factory=dict)


ACTION_REGISTRY: Dict[str, ActionSpec] = {
    "rank_suppliers":       ActionSpec("rank_suppliers", "Rank Suppliers", "supplier_ranking", {}),
    "draft_supplier_email": ActionSpec("draft_supplier_email", "Draft Supplier Email", "email_drafting", {}),
    "open_negotiation":     ActionSpec("open_negotiation", "Open Negotiation", "negotiation", {}),
    "request_approval":     ActionSpec("request_approval", "Request Approval", "approvals", {}),
    "notify_buyer":         ActionSpec("notify_buyer", "Notify Buyer", "notify_buyer", {}),
    "create_task":          ActionSpec("create_task", "Create Task", "notify_buyer", {}),
    "update_finding_stage": ActionSpec("update_finding_stage", "Update Finding Stage", "finding_store", {}),
    "link_to_deal":         ActionSpec("link_to_deal", "Link To Deal", "deal_assignment", {}),
}


def invoke(spec: ActionSpec, agents: Dict[str, Any], context: Dict[str, Any],
           params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the mapped agent. Agents expose run_action(context, params); if a
    specific agent uses a different entrypoint, adapt here (do not change agents)."""
    agent = agents.get(spec.handler_attr)
    if agent is None:
        logger.warning("ActionRegistry: no agent for %s (%s)", spec.slug, spec.handler_attr)
        return {"ok": False, "error": f"no agent for {spec.handler_attr}"}
    merged = {**spec.default_params, **(params or {})}
    runner = getattr(agent, "run_action", None)
    if callable(runner):
        return runner(context=context, params=merged)
    # fallback to the BaseAgent.run(context) contract
    from agents.base_agent import AgentContext  # local import to avoid cycles
    out = agent.run(AgentContext(input_data={**context, **merged}))
    return {"ok": True, "data": getattr(out, "data", None)}


__all__ = ["ActionSpec", "ACTION_REGISTRY", "invoke"]
```

> Implementer note: Step 3's grep gives the authoritative agent keys. If the live registry uses different keys (e.g. `email_drafting_agent`), update `handler_attr` to match — do not rename the agents.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engines/test_action_registry.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/engines/action_registry.py tests/engines/test_action_registry.py
git commit -m "feat(playbook): ActionRegistry mapping action slugs to existing agents"
```

---

## Task 6: bp_decision table + DecisionEngine

**Files:**
- Create: `deploy/sql/2026-06-28_bp_decision.sql`
- Create: `src/engines/decision_engine.py`
- Test: `tests/engines/test_decision_engine.py`

**Interfaces:**
- Consumes: `engines.policy_engine.Gate` (list of gates for the context).
- Produces:
  - `Decision` dataclass: `action:str, resolution_path:str, requires_human:bool, rationale:str, confidence:float, chosen_option:Optional[str], llm_brief:Optional[dict]`.
  - `DecisionEngine(llm=None)` with `decide(outputs: dict, gates: list[Gate], confidence: float) -> Decision`.

- [ ] **Step 1: Write the migration**

```sql
-- 2026-06-28 Decision records. Additive + idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_decision (
    decision_id     VARCHAR PRIMARY KEY,
    run_id          VARCHAR, finding_id VARCHAR, step_no INTEGER,
    action          VARCHAR NOT NULL,
    chosen_option   VARCHAR,
    resolution_path VARCHAR NOT NULL,
    rationale       TEXT, confidence NUMERIC,
    requires_human  BOOLEAN NOT NULL DEFAULT false,
    llm_brief       JSONB,
    final_choice    VARCHAR, decided_by VARCHAR,
    status          VARCHAR NOT NULL DEFAULT 'open',
    decided_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_decision_run    ON proc.bp_decision (run_id);
CREATE INDEX IF NOT EXISTS ix_bp_decision_status ON proc.bp_decision (status);
COMMIT;
```

- [ ] **Step 2: Write the failing test**

```python
# tests/engines/test_decision_engine.py
from engines.decision_engine import DecisionEngine, Decision
from engines.policy_engine import Gate

def test_clear_gates_resolve_deterministically():
    eng = DecisionEngine(llm=None)
    d = eng.decide(outputs={"ok": True}, gates=[Gate("allow", 0)], confidence=0.95)
    assert d.resolution_path == "deterministic"
    assert d.requires_human is False
    assert d.action in ("complete", "choose")

def test_conflicting_gates_escalate_to_human_with_brief():
    captured = {}
    def fake_llm(prompt):
        captured["prompt"] = prompt
        return {"recommended": "block", "reasoning": "risk too high"}
    eng = DecisionEngine(llm=fake_llm)
    d = eng.decide(outputs={"ok": True},
                   gates=[Gate("allow", 10), Gate("deny", 10)], confidence=0.9)
    assert d.resolution_path == "llm_assisted_human"
    assert d.requires_human is True
    assert d.llm_brief["recommended"] == "block"

def test_low_confidence_escalates_even_without_conflict():
    eng = DecisionEngine(llm=lambda p: {"recommended": "retry", "reasoning": "unsure"})
    d = eng.decide(outputs={}, gates=[Gate("allow", 0)], confidence=0.2)
    assert d.requires_human is True

def test_llm_unavailable_fails_safe():
    eng = DecisionEngine(llm=None)  # cannot build a brief
    d = eng.decide(outputs={}, gates=[Gate("allow", 0), Gate("deny", 0)], confidence=0.9)
    assert d.action == "escalate" and d.requires_human is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/engines/test_decision_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write `decision_engine.py`**

```python
"""DecisionEngine — generalizes ReasoningEngine.observe(): given outputs + the
applicable policy gates + a confidence score, decide what happens next. Resolves
deterministically when clear; otherwise builds an LLM brief and routes to a
human. Fails safe (escalate + human) when it cannot resolve and cannot brief."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from engines.policy_engine import Gate

logger = logging.getLogger(__name__)

CONFIDENCE_FLOOR = 0.4


@dataclass
class Decision:
    action: str                      # complete | retry | escalate | adapt | choose
    resolution_path: str             # deterministic | llm_assisted_human
    requires_human: bool
    rationale: str
    confidence: float
    chosen_option: Optional[str] = None
    llm_brief: Optional[Dict[str, Any]] = None


class DecisionEngine:
    def __init__(self, llm: Optional[Callable[[str], Dict[str, Any]]] = None):
        self._llm = llm

    @staticmethod
    def _gates_conflict(gates: List[Gate]) -> bool:
        effects = {g.effect for g in gates}
        # a permissive and a restrictive effect both present = conflict
        return bool(effects & {"allow"}) and bool(effects & {"deny", "escalate"})

    def decide(self, outputs: Dict[str, Any], gates: List[Gate], confidence: float) -> Decision:
        unclear = self._gates_conflict(gates) or confidence < CONFIDENCE_FLOOR
        if not unclear:
            # deterministic: the resolved gate drives the action
            effect = max(gates, key=lambda g: ({"allow": 0, "require_approval": 1,
                         "escalate": 2, "deny": 3}.get(g.effect, 0), g.priority)).effect \
                     if gates else "allow"
            action = {"allow": "complete", "require_approval": "choose",
                      "escalate": "escalate", "deny": "complete"}.get(effect, "complete")
            return Decision(action=action, resolution_path="deterministic",
                            requires_human=(effect in ("require_approval", "escalate", "deny")),
                            rationale=f"deterministic from gate effect '{effect}'",
                            confidence=confidence)
        # unclear → need a human; try to build an LLM brief first
        if self._llm is None:
            return Decision(action="escalate", resolution_path="llm_assisted_human",
                            requires_human=True, rationale="unclear; no LLM available — failing safe",
                            confidence=confidence, llm_brief=None)
        try:
            brief = self._llm(self._build_prompt(outputs, gates, confidence))
        except Exception:
            logger.exception("DecisionEngine: LLM brief failed; failing safe")
            return Decision(action="escalate", resolution_path="llm_assisted_human",
                            requires_human=True, rationale="LLM brief failed — failing safe",
                            confidence=confidence, llm_brief=None)
        return Decision(action="escalate", resolution_path="llm_assisted_human",
                        requires_human=True,
                        rationale=brief.get("reasoning", "needs human decision"),
                        confidence=confidence, llm_brief=brief)

    @staticmethod
    def _build_prompt(outputs, gates, confidence) -> str:
        lines = ["A procurement decision needs review. Conflicting or low-confidence signals.",
                 f"Confidence: {confidence:.2f}", "Applicable policy gates:"]
        for g in gates:
            lines.append(f"  - {g.effect} (priority {g.priority}): {g.rationale or ''}")
        lines.append(f"Step outputs: {outputs}")
        lines.append("Return JSON: {\"recommended\": <option>, \"reasoning\": <text>}.")
        return "\n".join(lines)


__all__ = ["Decision", "DecisionEngine"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engines/test_decision_engine.py -v`
Expected: 4 passed.

- [ ] **Step 6: Apply migration + commit**

```bash
psql "$DATABASE_URL" -f deploy/sql/2026-06-28_bp_decision.sql   # or stage if DB down
git add deploy/sql/2026-06-28_bp_decision.sql src/engines/decision_engine.py tests/engines/test_decision_engine.py
git commit -m "feat(decision): DecisionEngine (deterministic / llm-assisted-human, fail-safe) + bp_decision"
```

---

## Task 7: PlaybookRunner (the gate → act → observe envelope)

**Files:**
- Create: `src/services/playbook_runner.py`
- Test: `tests/services/test_playbook_runner.py`

**Interfaces:**
- Consumes: `PlaybookEngine`/`Playbook`, `PolicyEngine.applicable_gates`, `ACTION_REGISTRY`+`invoke`, `DecisionEngine`, and `services/agent_actions` for logging.
- Produces: `PlaybookRunner(agents, policy_engine, decision_engine, action_invoke=invoke, action_log=None)` with `run_step(run_id, step, context) -> dict` and `run(playbook, context) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_playbook_runner.py
from services.playbook_runner import PlaybookRunner
from engines.playbook import PlaybookStep
from engines.policy_engine import Gate

class FakePolicy:
    def __init__(self, gate): self._g = gate
    def applicable_gates(self, context): return self._g

class FakeDecision:
    def decide(self, outputs, gates, confidence):
        from engines.decision_engine import Decision
        return Decision("complete", "deterministic", False, "ok", 0.9)

def test_human_gated_step_pauses_for_approval():
    logs = []
    runner = PlaybookRunner(agents={}, policy_engine=FakePolicy([Gate("allow", 0)]),
                            decision_engine=FakeDecision(),
                            action_invoke=lambda *a, **k: {"ok": True},
                            action_log=lambda **kw: logs.append(kw))
    step = PlaybookStep(step_no=1, action_slug="notify_buyer", mode="human_gated")
    out = runner.run_step("run1", step, context={})
    assert out["step_status"] == "awaiting_approval"  # paused, not executed

def test_gate_deny_blocks_execution():
    runner = PlaybookRunner(agents={}, policy_engine=FakePolicy([Gate("deny", 0)]),
                            decision_engine=FakeDecision(),
                            action_invoke=lambda *a, **k: {"ok": True},
                            action_log=lambda **kw: None)
    step = PlaybookStep(step_no=1, action_slug="open_negotiation", mode="auto")
    out = runner.run_step("run1", step, context={})
    assert out["step_status"] in ("failed", "skipped")
    assert out.get("blocked_by") == "policy"

def test_auto_step_executes_and_logs():
    logs = []
    runner = PlaybookRunner(agents={}, policy_engine=FakePolicy([Gate("allow", 0)]),
                            decision_engine=FakeDecision(),
                            action_invoke=lambda *a, **k: {"ok": True},
                            action_log=lambda **kw: logs.append(kw))
    step = PlaybookStep(step_no=1, action_slug="notify_buyer", mode="auto")
    out = runner.run_step("run1", step, context={})
    assert out["step_status"] == "done"
    assert len(logs) == 1  # one bp_agent_actions row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_playbook_runner.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `playbook_runner.py`**

```python
"""Executes a playbook (or a single default step) through the universal envelope:
gate (policy) -> act (agent) -> observe (decision). Human-gated by default."""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional

from engines.action_registry import ACTION_REGISTRY, invoke as _default_invoke

logger = logging.getLogger(__name__)

_RESTRICTIVE = {"deny", "escalate", "require_approval"}


class PlaybookRunner:
    def __init__(self, agents: Dict[str, Any], policy_engine, decision_engine,
                 action_invoke: Callable = _default_invoke,
                 action_log: Optional[Callable] = None):
        self.agents = agents
        self.policy = policy_engine
        self.decision = decision_engine
        self._invoke = action_invoke
        self._log = action_log or (lambda **kw: None)

    def run_step(self, run_id: str, step, context: Dict[str, Any]) -> Dict[str, Any]:
        # GATE
        gates = self.policy.applicable_gates({**context, "action_slug": step.action_slug})
        verdict = max(gates, key=lambda g: ({"allow": 0, "require_approval": 1,
                      "escalate": 2, "deny": 3}.get(g.effect, 0), g.priority)).effect \
                  if gates else "allow"
        if verdict == "deny":
            return {"run_id": run_id, "step_no": step.step_no,
                    "step_status": "skipped", "blocked_by": "policy"}
        # human gating: either the policy requires it, or the step is human_gated
        if verdict in _RESTRICTIVE or step.mode == "human_gated":
            return {"run_id": run_id, "step_no": step.step_no,
                    "step_status": "awaiting_approval", "gate": verdict}
        # ACT (auto + allowed)
        spec = ACTION_REGISTRY.get(step.action_slug)
        if spec is None:
            return {"run_id": run_id, "step_no": step.step_no, "step_status": "failed",
                    "error": f"unknown action {step.action_slug}"}
        try:
            result = self._invoke(spec, self.agents, context, step.params)
        except Exception as exc:
            logger.exception("step %s failed", step.action_slug)
            self._log(phase="action", action_type=step.action_slug, status="failed",
                      summary=str(exc))
            return {"run_id": run_id, "step_no": step.step_no, "step_status": "failed"}
        self._log(phase="action", action_type=step.action_slug, status="done",
                  summary=f"ran {step.action_slug}")
        # OBSERVE
        decision = self.decision.decide(outputs=result, gates=gates, confidence=0.9)
        return {"run_id": run_id, "step_no": step.step_no, "step_status": "done",
                "result": result, "decision": decision.action}

    def run(self, playbook, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"{getattr(playbook, 'playbook_id', 'default')}:{context.get('finding_id', '')}"
        steps = getattr(playbook, "steps", [])
        out_steps: List[dict] = []
        for step in steps:
            res = self.run_step(run_id, step, context)
            out_steps.append(res)
            if res["step_status"] in ("awaiting_approval", "skipped", "failed"):
                break  # pause/stop; orchestrator resumes after human input
        return {"run_id": run_id, "steps": out_steps}


__all__ = ["PlaybookRunner"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_playbook_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/playbook_runner.py tests/services/test_playbook_runner.py
git commit -m "feat(playbook): PlaybookRunner — gate->act->observe envelope, human-gated default"
```

---

## Task 8: Orchestrator universal envelope + default path; wire ReasoningEngine

**Files:**
- Modify: `src/orchestration/reasoning_engine.py` (`observe` at `:337`)
- Modify: `src/orchestration/orchestrator.py`
- Test: `tests/orchestration/test_universal_envelope.py`

**Interfaces:**
- Consumes: `PlaybookEngine`, `PlaybookRunner`, `DecisionEngine`, `PolicyEngine`.
- Produces: `Orchestrator.handle(trigger_kind, context) -> dict` — selects a playbook or runs the single default step; always applies policy + decision.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_universal_envelope.py
from services.playbook_runner import PlaybookRunner
from engines.playbook import PlaybookStep
from engines.policy_engine import Gate

class FakePolicy:
    def applicable_gates(self, context): return [Gate("allow", 0)]
class FakeDecision:
    def decide(self, outputs, gates, confidence):
        from engines.decision_engine import Decision
        return Decision("complete", "deterministic", False, "ok", 0.9)

def test_no_playbook_runs_single_default_step_through_envelope():
    logs = []
    runner = PlaybookRunner(agents={}, policy_engine=FakePolicy(),
                            decision_engine=FakeDecision(),
                            action_invoke=lambda *a, **k: {"ok": True},
                            action_log=lambda **kw: logs.append(kw))
    # the default path = an implicit one-step playbook for a direct agent call
    default_step = PlaybookStep(step_no=1, action_slug="notify_buyer", mode="auto")

    class DefaultPB:
        playbook_id = "default"
        steps = [default_step]

    out = runner.run(DefaultPB(), context={"finding_id": "f1"})
    assert out["steps"][0]["step_status"] == "done"   # governance ran even with no real playbook
    assert len(logs) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestration/test_universal_envelope.py -v`
Expected: PASS for the runner-level default path (this characterizes the contract). If it fails, fix the runner from Task 7.

- [ ] **Step 3: Add the envelope to the orchestrator**

In `src/orchestration/orchestrator.py`, add a method that implements "playbook first, else default path." Use the engines already on `agent_nick` (add `playbook_engine`/`decision_engine` in `base_agent.py` `_safe_engine` block alongside the others):

```python
    def handle(self, trigger_kind: str, context: dict) -> dict:
        """Universal entry: select a governing playbook, else run a single
        default step. Policy gate + decision always apply (PlaybookRunner)."""
        from services.playbook_runner import PlaybookRunner
        from engines.playbook import Playbook, PlaybookStep
        runner = PlaybookRunner(
            agents=self.agents,
            policy_engine=self.agent_nick.policy_engine,
            decision_engine=self.agent_nick.decision_engine,
        )
        pb = self.agent_nick.playbook_engine.playbook_for(trigger_kind, context)
        if pb is None:
            action = context.get("action_slug", "notify_buyer")
            pb = Playbook(playbook_id="default", playbook_name="Default",
                          trigger_kind=trigger_kind, trigger_match={},
                          steps=[PlaybookStep(step_no=1, action_slug=action, mode="human_gated")],
                          version=1)
        return runner.run(pb, context)
```

- [ ] **Step 4: Repoint `ReasoningEngine.observe()` at the DecisionEngine**

In `src/orchestration/reasoning_engine.py` `observe()` (`:337`), replace its ad-hoc confidence/escalation branching with a call to the shared engine, preserving the `Observation(action=...)` return:

```python
    def observe(self, results: dict) -> Observation:
        from engines.decision_engine import DecisionEngine
        from engines.policy_engine import Gate
        wf_ctx = results.get("workflow_context")
        confidence = self._aggregate_confidence(results)  # keep existing helper
        gates = []  # ReasoningEngine has no per-action gates; pass confidence only
        decision = DecisionEngine(llm=self._llm_brief).decide(results, gates, confidence)
        return Observation(action=decision.action, reason=decision.rationale)
```

(Keep the existing `RECOMMEND_ESCALATION`/`SUGGEST_AGENT` signal checks ahead of this call — they map to `escalate`/`adapt` and remain. `_llm_brief` is a thin wrapper over the existing AgentNick call used elsewhere in the file.)

- [ ] **Step 5: Run the affected suites**

Run: `pytest tests/orchestration/ tests/engines/test_decision_engine.py -v`
Expected: PASS. If a ReasoningEngine test asserted the old branch wording, update it to the new `action` values (same set: complete/retry/escalate/adapt).

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/orchestrator.py src/orchestration/reasoning_engine.py tests/orchestration/test_universal_envelope.py
git commit -m "feat(orchestrator): universal gate->act->observe envelope + default path; wire observe() to DecisionEngine"
```

---

## Task 9: Register the new engines on AgentNick + extend reload-governance

**Files:**
- Modify: `src/agents/base_agent.py` (engine init block ~`:1500-1520`)
- Modify: `src/api/routers/agents.py` (`:72-88`)
- Test: `tests/test_reload_governance.py`

**Interfaces:**
- Produces: `agent_nick.playbook_engine`, `agent_nick.decision_engine`; `/agents/reload-governance` reloads policies + playbooks and returns counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reload_governance.py
def test_reload_governance_counts(monkeypatch):
    from api.routers import agents as agents_router
    class FakePE:
        def reload_policies(self): pass
        def list_policies(self): return [1, 2]
    class FakePB:
        def reload(self): pass
        def active_playbooks(self): return [1]
    class FakeNick:
        policy_engine = FakePE(); playbook_engine = FakePB()
    body = agents_router._reload_governance_payload(FakeNick())
    assert body["policies"] == 2 and body["playbooks"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reload_governance.py -v`
Expected: FAIL — `AttributeError: _reload_governance_payload`.

- [ ] **Step 3: Register engines on AgentNick**

In `base_agent.py`, alongside the existing `_safe_engine(...)` calls, add:

```python
        from engines.playbook import PlaybookEngine
        from engines.decision_engine import DecisionEngine
        self.playbook_engine = _safe_engine(
            "PlaybookEngine", lambda: PlaybookEngine(self),
            lambda: PlaybookEngine(agent_nick=None, playbook_rows=[], step_rows=[]))
        self.decision_engine = _safe_engine(
            "DecisionEngine", lambda: DecisionEngine(llm=self._decision_llm),
            lambda: DecisionEngine(llm=None))
```

(Define `self._decision_llm` as a small wrapper over the existing AgentNick generate call; if none is handy, pass `llm=None` — the engine fails safe.)

- [ ] **Step 4: Extend the reload endpoint**

In `src/api/routers/agents.py`, factor the body into a helper and call it from the existing route:

```python
def _reload_governance_payload(agent_nick) -> dict:
    agent_nick.policy_engine.reload_policies()
    if getattr(agent_nick, "playbook_engine", None):
        agent_nick.playbook_engine.reload()
    return {
        "policies": len(agent_nick.policy_engine.list_policies()),
        "playbooks": len(agent_nick.playbook_engine.active_playbooks())
                     if getattr(agent_nick, "playbook_engine", None) else 0,
    }
```

Call `_reload_governance_payload(agent_nick)` inside the `/agents/reload-governance` handler and return it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_reload_governance.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/agents/base_agent.py src/api/routers/agents.py tests/test_reload_governance.py
git commit -m "feat(governance): register PlaybookEngine/DecisionEngine; reload-governance includes playbooks"
```

---

## Task 10: Legacy dissolution — opportunity policies → Rule Book

**Files:**
- Create: `deploy/sql/2026-06-28_migrate_opportunity_policies.sql`
- Modify: `src/agents/opportunity_miner_agent.py:742`
- Test: `tests/agents/test_opportunity_policy_reader.py`

**Interfaces:**
- Produces: opportunity `bp_policy` rows represented as `bp_rule` rows (Phase-1 seed already covers the 11 detectors; this verifies parity and retires the policy mirror). `opportunity_miner_agent` reads detectors from `RuleBook`, not `policy_engine.iter_policies()`.

- [ ] **Step 1: Verify Phase-1 already seeds the detectors as rules**

Run: `psql "$DATABASE_URL" -c "SELECT detector_slug FROM proc.bp_rule ORDER BY 1;"`
Expected: the 11 detector slugs. If present, no data migration is needed — only the reader repoint + retiring the legacy policy rows.

- [ ] **Step 2: Write the migration that retires the opportunity policy rows**

```sql
-- 2026-06-28 Retire legacy opportunity policy mirrors (now owned by bp_rule).
BEGIN;
UPDATE proc.bp_policy SET policy_lifecycle = 'retired'
WHERE policy_type = 'opportunity';
COMMIT;
```

- [ ] **Step 3: Write the failing reader test**

```python
# tests/agents/test_opportunity_policy_reader.py
def test_miner_reads_detectors_from_rulebook(monkeypatch):
    from agents import opportunity_miner_agent as oma
    class FakeRuleBook:
        def active_rules(self):
            from engines.rule_book import Rule
            return [Rule(1, "Price", "price_variance_check", "opportunity", "po_lines", {}, "medium", 1)]
    slugs = oma.detector_slugs_from_rulebook(FakeRuleBook())
    assert slugs == ["price_variance_check"]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/agents/test_opportunity_policy_reader.py -v`
Expected: FAIL — `AttributeError: detector_slugs_from_rulebook`.

- [ ] **Step 5: Add the reader + repoint line 742**

Add a module-level helper in `opportunity_miner_agent.py`:

```python
def detector_slugs_from_rulebook(rule_book) -> list:
    return [r.detector_slug for r in rule_book.active_rules()]
```

Replace the `policy_engine.iter_policies()` usage at `:742` with reading from `self.agent_nick.rule_book` (Phase-1 engine), e.g. `detector_slugs_from_rulebook(self.agent_nick.rule_book)`.

- [ ] **Step 6: Run tests + miner suite**

Run: `pytest tests/agents/test_opportunity_policy_reader.py tests/agents -k opportunity -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add deploy/sql/2026-06-28_migrate_opportunity_policies.sql src/agents/opportunity_miner_agent.py tests/agents/test_opportunity_policy_reader.py
git commit -m "refactor(policy): opportunity policies owned by RuleBook; retire legacy mirrors"
```

---

## Task 11: Legacy dissolution — ranking config → Supplier Ranking playbook

**Files:**
- Create: `deploy/sql/2026-06-28_seed_supplier_ranking_playbook.sql`
- Modify: `src/agents/supplier_ranking_agent.py:522`
- Modify: `src/orchestration/orchestrator.py:219`
- Test: `tests/agents/test_ranking_config_from_playbook.py`

**Interfaces:**
- Produces: a `bp_playbook` "Supplier Ranking" with a `rank_suppliers` step whose `params` carry the migrated weights/normalization/categorical maps; `supplier_ranking_agent` reads them from the playbook step params.

- [ ] **Step 1: Seed the Supplier Ranking playbook from the legacy weights**

```sql
-- 2026-06-28 Move ranking config out of bp_policy into a human-authored playbook.
BEGIN;
INSERT INTO proc.bp_playbook (playbook_id, playbook_name, trigger_kind, trigger_match, playbook_status, version, authored_by)
VALUES ('pb_supplier_ranking', 'Supplier Ranking', 'event',
        '{"event":"rank_suppliers"}', 'active', 1, 'migration')
ON CONFLICT (playbook_id) DO NOTHING;

INSERT INTO proc.bp_playbook_step (step_id, playbook_id, step_no, action_slug, params, mode)
VALUES ('pb_supplier_ranking_s1', 'pb_supplier_ranking', 1, 'rank_suppliers',
        '{"default_weights":{"price":0.4,"delivery":0.3,"risk":0.2,"payment_terms":0.1},
          "directions":{"price":"lower_is_better","delivery":"lower_is_better","risk":"lower_is_better","payment_terms":"higher_is_better"},
          "categorical_maps":{}}',
        'human_gated')
ON CONFLICT (step_id) DO NOTHING;

UPDATE proc.bp_policy SET policy_lifecycle = 'retired'
WHERE policy_type = 'supplier_ranking';
COMMIT;
```

- [ ] **Step 2: Write the failing test**

```python
# tests/agents/test_ranking_config_from_playbook.py
def test_ranking_weights_read_from_playbook():
    from agents.supplier_ranking_agent import ranking_config_from_playbook
    from engines.playbook import Playbook, PlaybookStep
    pb = Playbook("pb_supplier_ranking", "Supplier Ranking", "event", {"event": "rank_suppliers"},
                  [PlaybookStep(1, "rank_suppliers",
                   {"default_weights": {"price": 0.4, "delivery": 0.3, "risk": 0.2, "payment_terms": 0.1}})], 1)
    cfg = ranking_config_from_playbook(pb)
    assert cfg["default_weights"]["price"] == 0.4
    assert abs(sum(cfg["default_weights"].values()) - 1.0) < 1e-6
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/agents/test_ranking_config_from_playbook.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 4: Add the reader + repoint callers**

Add a module-level helper to `supplier_ranking_agent.py`:

```python
def ranking_config_from_playbook(playbook) -> dict:
    for step in getattr(playbook, "steps", []):
        if step.action_slug == "rank_suppliers":
            return step.params or {}
    return {}
```

Replace `self.policy_engine.supplier_policies` at `:522` with config read from the Supplier Ranking playbook:
`ranking_config_from_playbook(self.agent_nick.playbook_engine.playbook_for("event", {"event": "rank_suppliers"}))`.
Do the same at `orchestrator.py:219` (the `WeightAllocationPolicy` lookup) — read `default_weights` from that config dict.

- [ ] **Step 5: Run ranking parity tests**

Run: `pytest tests/agents/test_ranking_config_from_playbook.py tests/ -k "rank" -v`
Expected: PASS — ranking output unchanged (weights identical, just sourced from the playbook).

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-06-28_seed_supplier_ranking_playbook.sql src/agents/supplier_ranking_agent.py src/orchestration/orchestrator.py tests/agents/test_ranking_config_from_playbook.py
git commit -m "refactor(policy): ranking config moves to Supplier Ranking playbook; repoint callers"
```

---

## Task 12: Legacy dissolution — negotiation thresholds → pure gate policies

**Files:**
- Create: `deploy/sql/2026-06-28_seed_negotiation_gates.sql`
- Modify: `src/engines/negotiation_strategy_engine.py` (`evaluate_policy_rails`, `:274-307`)
- Test: `tests/test_negotiation_gates_parity.py`

**Interfaces:**
- Produces: two gate policies (`allow` for low value, `escalate` for high value); `evaluate_policy_rails` resolves via `PolicyEngine.applicable_gates`, preserving the `RailDecision` shape.

- [ ] **Step 1: Seed the negotiation gate policies**

```sql
-- 2026-06-28 Negotiation thresholds become pure permission gates.
BEGIN;
INSERT INTO proc.bp_policy (policy_name, policy_type, applies_to, effect, priority, policy_lifecycle)
SELECT 'NegotiationLowValueAuto', 'gate',
       '{"action_slug":"open_negotiation","order_value":{"<":5000}}'::jsonb, 'allow', 50, 'active'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name='NegotiationLowValueAuto');
INSERT INTO proc.bp_policy (policy_name, policy_type, applies_to, effect, priority, policy_lifecycle)
SELECT 'NegotiationHighValueEscalate', 'gate',
       '{"action_slug":"open_negotiation","order_value":{">":50000}}'::jsonb, 'escalate', 100, 'active'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name='NegotiationHighValueEscalate');
COMMIT;
```

- [ ] **Step 2: Write the parity test**

```python
# tests/test_negotiation_gates_parity.py
from engines.policy_engine import PolicyEngine

ROWS = [
    {"policy_id": 1, "policy_name": "low", "policy_lifecycle": "active",
     "applies_to": {"action_slug": "open_negotiation", "order_value": {"<": 5000}},
     "effect": "allow", "priority": 50},
    {"policy_id": 2, "policy_name": "high", "policy_lifecycle": "active",
     "applies_to": {"action_slug": "open_negotiation", "order_value": {">": 50000}},
     "effect": "escalate", "priority": 100},
]

def test_low_value_allows():
    pe = PolicyEngine(policy_rows=ROWS)
    assert pe.is_allowed({"action_slug": "open_negotiation", "order_value": 1000}).effect == "allow"

def test_high_value_escalates():
    pe = PolicyEngine(policy_rows=ROWS)
    assert pe.is_allowed({"action_slug": "open_negotiation", "order_value": 90000}).effect == "escalate"
```

- [ ] **Step 3: Run test to verify it passes (the engine already supports this)**

Run: `pytest tests/test_negotiation_gates_parity.py -v`
Expected: 2 passed (verifies the gates reproduce the old auto-approve/escalation thresholds).

- [ ] **Step 4: Refactor `evaluate_policy_rails` to use the gates**

In `negotiation_strategy_engine.py`, replace the hardcoded threshold checks (`:291`, `:301`) with a call to the policy engine, mapping the gate effect back to the existing `RailDecision.action`:

```python
    def evaluate_policy_rails(self, ctx, policy_engine=None):
        if ctx.round_number > self.max_rounds:                      # keep: rounds exhausted
            return RailDecision(action="escalate", reason="rounds exhausted")
        if policy_engine is not None:
            gate = policy_engine.is_allowed(
                {"action_slug": "open_negotiation", "order_value": ctx.order_value})
            if gate.effect == "allow":
                return RailDecision(action="accept", reason="auto-approve (policy gate)")
            if gate.effect in ("escalate", "deny", "require_approval"):
                return RailDecision(action="escalate", reason="requires approval (policy gate)")
        return RailDecision(action="continue", reason="within rails")
```

Pass `self.agent_nick.policy_engine` from `negotiation_agent` where `evaluate_policy_rails` is called. The hardcoded `auto_approve_threshold`/`escalation_threshold` attributes become dead defaults (leave for one release, remove in cleanup).

- [ ] **Step 5: Run the negotiation suite**

Run: `pytest tests/test_negotiation_agent.py tests/test_negotiation_skills.py tests/test_negotiation_gates_parity.py -v`
Expected: PASS (decisions unchanged).

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-06-28_seed_negotiation_gates.sql src/engines/negotiation_strategy_engine.py tests/test_negotiation_gates_parity.py
git commit -m "refactor(policy): negotiation thresholds become pure gates via applicable_gates"
```

---

## Task 13: Legacy cleanup — repoint remaining callers + remove dead methods

**Files:**
- Modify: `src/orchestration/orchestrator.py:312,1419` (`validate_workflow`)
- Modify: `src/agents/base_agent.py:385` (`get_policy`)
- Modify: `src/engines/policy_engine.py` (remove dead legacy methods)
- Test: `tests/engines/test_policy_engine_pure.py`

**Interfaces:**
- Produces: `PolicyEngine` exposes only the pure API (`applicable_gates`, `is_allowed`, `reload_policies`, `list_policies`); legacy `validate_workflow`, `supplier_policies`, `opportunity_policies`, `get_policy`, weight-normalization are removed.

- [ ] **Step 1: Repoint `validate_workflow` callers**

Ranking validation now lives in the Supplier Ranking playbook config. At `orchestrator.py:1419` and `:312`, replace the `policy_engine.validate_workflow(...)` call with a check that the ranking config has weights for the requested criteria:

```python
        cfg = ranking_config_from_playbook(
            self.agent_nick.playbook_engine.playbook_for("event", {"event": "rank_suppliers"}))
        weights = (cfg or {}).get("default_weights", {})
        allowed = all(c in weights for c in criteria) if criteria else True
```

- [ ] **Step 2: Repoint `get_policy` in base_agent**

At `base_agent.py:385`, `governing_policy(name)` should resolve a pure permission policy by name. Replace `engine.get_policy(name)` with a lookup over `applicable_gates` by policy name, or return `None` if not a gate policy (governing policies are now gates):

```python
            gates = engine.applicable_gates({"policy_name": name}) if engine else []
            return gates[0] if gates else None
```

- [ ] **Step 3: Write the pure-engine test**

```python
# tests/engines/test_policy_engine_pure.py
import pytest
from engines.policy_engine import PolicyEngine

def test_legacy_methods_removed():
    pe = PolicyEngine(policy_rows=[])
    for dead in ("validate_workflow", "get_policy", "supplier_policies", "opportunity_policies"):
        assert not hasattr(pe, dead), f"{dead} should be removed"

def test_pure_api_present():
    pe = PolicyEngine(policy_rows=[])
    assert hasattr(pe, "applicable_gates") and hasattr(pe, "is_allowed")
    assert hasattr(pe, "reload_policies") and hasattr(pe, "list_policies")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/engines/test_policy_engine_pure.py -v`
Expected: FAIL — legacy methods still present.

- [ ] **Step 5: Remove the dead methods**

Delete from `policy_engine.py`: `validate_workflow`, `validate_and_apply`, `_collect_supplier_policies`, `_collect_opportunity_policies`, `_normalise_weight_policy`, `get_policy`, the `supplier_policies`/`opportunity_policies` attributes, and the `SUPPLIER_POLICY_SLUGS`/`OPPORTUNITY_KEYWORDS` constants. Keep `reload_policies`/`list_policies` (repoint them to rebuild `_policy_rows_cache`).

- [ ] **Step 6: Run the full affected suite**

Run: `pytest tests/engines/test_policy_engine_pure.py tests/engines/test_policy_gates.py tests/orchestration tests/agents -k "policy or rank or opportunity" -v`
Expected: PASS. Fix any remaining import of a removed symbol.

- [ ] **Step 7: Commit**

```bash
git add src/orchestration/orchestrator.py src/agents/base_agent.py src/engines/policy_engine.py tests/engines/test_policy_engine_pure.py
git commit -m "refactor(policy): repoint last callers; remove legacy config methods (pure engine)"
```

---

## Task 14: Live proof against bp_sqldb (acceptance)

**Files:** none (verification only).

- [ ] **Step 1: Apply all migrations**

Run:
```bash
for f in deploy/sql/2026-06-28_*.sql; do psql "$DATABASE_URL" -f "$f"; done
```
Expected: no errors; `bp_policy` has only gate/active rows for permissions, legacy config rows `retired`; `bp_playbook` has the Supplier Ranking playbook; `bp_decision`/`bp_playbook_*` exist.

- [ ] **Step 2: Reload governance and confirm counts**

Run: `curl -s -X POST http://localhost:8000/agents/reload-governance`
Expected: HTTP 200 with `{"policies": N, "playbooks": M}` (M ≥ 1).

- [ ] **Step 3: Prove the no-playbook default path runs governance**

Trigger a direct agent action with no matching playbook and confirm a `bp_playbook_run` (`playbook_id` NULL/`'default'`), a `bp_decision`, and `bp_agent_actions` rows were written:

```bash
psql "$DATABASE_URL" -c "SELECT count(*) FROM proc.bp_playbook_run WHERE playbook_id IS NULL OR playbook_id='default';"
psql "$DATABASE_URL" -c "SELECT count(*) FROM proc.bp_decision;"
```
Expected: non-zero for the actions just triggered — governance ran without a playbook.

- [ ] **Step 4: Prove a human-gated playbook pauses**

Trigger a finding that matches a seeded playbook; confirm a `bp_playbook_run_step` row at `awaiting_approval`:

```bash
psql "$DATABASE_URL" -c "SELECT run_id, step_no, step_status FROM proc.bp_playbook_run_step WHERE step_status='awaiting_approval' LIMIT 5;"
```
Expected: at least one paused step (human-in-the-loop default holds).

- [ ] **Step 5: Confirm parity — ranking & negotiation unchanged**

Run: `pytest tests/agents -k rank tests/test_negotiation_agent.py -v`
Expected: PASS — behaviour identical, definitions now sourced from playbook/gates.

- [ ] **Step 6: Record the result**

```bash
git commit --allow-empty -m "test(decision-layer): proven on live bp_sqldb — governance runs with/without playbook, human-gated pauses, ranking+negotiation parity"
```

---

## Self-Review

**Spec coverage:** pure policy + gates → T1, T2; governance versioning/audit → T1 (+ lifecycle used throughout); playbook tables/engine → T3, T4; ActionRegistry → T5; decision engine (generalizes observe) → T6, wired in T8; PlaybookRunner envelope (gate→act→observe, human-gated) → T7; orchestrator universal envelope + no-playbook default path → T8; engine registration + reload-governance → T9; legacy dissolution (opportunity→rule, ranking→playbook, negotiation→gates, caller repoints, dead-method removal) → T10–T13; live acceptance incl. the no-playbook governance proof and parity → T14. All spec sections map to a task.

**Placeholder scan:** no TBD/TODO; every code step shows code; implementer notes point to exact grep commands/source lines to confirm signatures, not vague guidance.

**Type consistency:** `Gate(effect, priority, policy_id, rationale)` (T2) consumed by `DecisionEngine.decide` (T6) and `PlaybookRunner.run_step` (T7); `Policy`/`PlaybookStep`/`Playbook` (T4) consumed by runner (T7) and orchestrator (T8); `applicable_gates(context)`/`is_allowed(context)` (T2) used by runner (T7), decision (T6), negotiation (T12), base_agent (T13); `ranking_config_from_playbook` (T11) reused in T13; `ACTION_REGISTRY`/`invoke` (T5) used by runner (T7); `reload_policies`/`list_policies` retained for the router (T9, T13).

**Known follow-ups (flagged, not gaps):** Phase 1 (`bp_rule`/`RuleBook`/`DETECTOR_REGISTRY`) is a prerequisite (stated up top); if not built, run its plan first. Agent keys in the ActionRegistry are confirmed by grep at build time (T5 Step 3). `auto_approve_threshold`/`escalation_threshold` left as dead defaults in T12, removable in a later cleanup once nothing reads them.
```
