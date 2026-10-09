# Agent Policy Governance — Stage 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent policies can be created, reviewed through an example check, versioned, activated or retired in the form. Each saved version is compiled by plain code into validated `hard-policy/2` JSON, and the orchestrator reads only live, valid policies from one stable endpoint.

**Architecture:**
- **Logic** is pure Python in `src/services/agent_policy/`: conditions, compiler, contract validation, readiness and confidence. No database or model is involved.
- **Data** goes in new `proc.bp_*` tables written through `src/repositories/agent_policy_repo.py`, and is served by `src/api/routers/agent_policies.py`.
- **The screens** reach it only through a new gateway module, `agent-policy`, which checks sign-in and permission and forwards the verified identity with a shared key. The backend re-checks the role and audits every write.
- **The form** is new functions in `engine.js` plus a pure module, `src/modules/SpendIQ/agentPolicy/`. It is built from the existing `fm-*` form styles.

**Tech Stack:** Python 3 / FastAPI / psycopg2 / jsonschema 4.26 (in both venvs); NestJS gateway (jest); React+vanilla `engine.js` UI (vitest).

**Spec:** `specs/2026-10-08-agent-policy-governance-brief.md` (requirements, verbatim) and `specs/2026-10-08-agent-policy-governance-design.md` (rulings + decisions; wins on conflict).

## Global Constraints

- `proc.bp_policy` is never written, altered or migrated. It may be read only, for roles.
- New tables use the `bp_` prefix; indexes are `ix_bp_*`.
- Every migration has a `_rollback.sql` and is applied to **both** `bp_testdb` and `bp_sqldb`. Announce DDL before applying, because the databases are shared with other sessions.
- Status values: Draft / Active / Retired in the UI; `draft` / `live` / `retired` in the JSON and the database.
- Policy IDs: `<PREFIX>-<4 digits>`, e.g. `FIN-0012`. They are never reused and never change after the first save.
- Response-time default: `PT4H`, clock time (design §3.1).
- `engine.js` changes are confined to new functions whose names start `ap` or `agentPolicy`, plus the minimum wiring lines. There are no unrelated refactors (ruling A).
- The screens never call the Python backend for this feature. They call the gateway only (ruling C).
- Every write endpoint audits through `agent_actions.record_action_or_fail` before returning.
- In the shared checkout, stage your own hunks only. Never use `git add -A`, `git stash`, or a Haiku implementer.
- Tests: `./venv/bin/python -m pytest` with `.env` loaded. Database tests need `PROCWISE_TEST_LIVE_DB=1`. Isolate from the GPU with `CUDA_VISIBLE_DEVICES=""`.

## Review Focus

1. **Two people edit the same policy.** The second save must be refused with 409 ("someone saved a newer version"), not silently overwrite the first. Covered in Task 6 by `test_stale_base_version_is_refused`.
2. **The backend is reached without the gateway.** A request with no `X-Gateway-Key`, a wrong key, or an unset key environment variable must be refused, and must never fall back to a default identity. Covered in Task 7 by `test_missing_or_wrong_gateway_key_is_refused` and `test_unset_key_env_refuses_everything`.
3. **A live policy is later found to be invalid** because the registry changed. The orchestrator feed must leave it out and list it under `refused`; it must never ship it. Covered in Task 7 by `test_feed_refuses_policy_whose_tool_left_registry`.
4. **Pasted text or CSV cells that start with formula characters** (`=`, `+`, `-`, `@`, and also tab or carriage return). The export must neutralise every one, including a value that has leading whitespace. Covered in Task 9 by `neutralises leading whitespace before a formula`.
5. **Switching outcome back and forth in the form** (approve → block → approve). This must never carry stale levels into the saved JSON, and must never lose the levels the user re-enters. Covered in Task 3 by `test_switching_outcome_drops_previous_fields_from_json` and Task 9 by `switchOutcome round trip`.

---

## File Structure

**BP_Backend**

| Path | Responsibility |
|---|---|
| `deploy/sql/2026-10-08_bp_agent_policy.sql` (+ `_rollback.sql`) | taxonomy, registry, policy, version (immutable), settings row |
| `src/services/agent_policy/__init__.py` | package marker |
| `src/services/agent_policy/conditions.py` | operator translation, field listing, example results |
| `src/services/agent_policy/registry.py` | `RegistrySnapshot` + loader |
| `src/services/agent_policy/settings.py` | company settings with defaults |
| `src/services/agent_policy/compiler.py` | pure form → `hard-policy/2` |
| `src/services/agent_policy/hard-policy-2.schema.json` | JSON Schema, checked in |
| `src/services/agent_policy/contract.py` | schema + cross-field validation |
| `src/services/agent_policy/readiness.py` | Active checks, How-it-is-enforced text, extraction confidence |
| `src/repositories/agent_policy_repo.py` | IDs, versions, transitions, reads |
| `src/api/routers/agent_policies.py` | screen endpoints + orchestrator feed |
| `scripts/agent_policy/seed_registry.py` | registry seed from the real tool list |
| `tests/agent_policy/…` | one test file per module |

**Gateway** (`beyond-procwaise-Api/beyond_procwaise_api`, branch `spendiq-ui`)

| Path | Responsibility |
|---|---|
| `src/modules/agent-policy/agent-policy.controller.ts` | routes, role check |
| `src/modules/agent-policy/agent-policy.service.ts` | forward to BP_Backend with key + identity |
| `src/modules/agent-policy/agent-policy.module.ts` | module |
| `src/modules/agent-policy/agent-policy.yml` | serverless HTTP events |
| `src/modules/agent-policy/agent-policy.controller.spec.ts` | jest |
| `src/app.module.ts` | register module (1 import + 1 array entry) |

**UI** (`beyond_procwise_ui`, branch `spendiq-ui`)

| Path | Responsibility |
|---|---|
| `src/modules/SpendIQ/agentPolicy/model.js` | pure form state: empty form, outcome switch, clearing confirmation, legacy field mapping |
| `src/modules/SpendIQ/agentPolicy/inventory.js` | group by source, group by area/sub-area, CSV with formula guard |
| `src/modules/SpendIQ/agentPolicy/*.test.js` | vitest |
| `src/modules/SpendIQ/engine.js` | new `ap*` functions: list, inventory, form, confirm dialogs; tab on the Policies screen |

---

### Task 1: Migration — taxonomy, registry, policy, immutable version, settings

**Files:**
- Create: `deploy/sql/2026-10-08_bp_agent_policy.sql`
- Create: `deploy/sql/2026-10-08_bp_agent_policy_rollback.sql`
- Test: `tests/migrations/test_2026_10_08_bp_agent_policy.py`

**Interfaces:**
- Produces these tables, used by every later task:
  - `proc.bp_business_area`
  - `proc.bp_orchestrator_registry`
  - `proc.bp_agent_policy`
  - `proc.bp_agent_policy_version`
  - the `proc.bp_admin_config` row `agent_policy_settings`

- [ ] **Step 1: Write the failing live test**

```python
"""proc.bp_agent_policy* in both live databases. Needs PROCWISE_TEST_LIVE_DB=1."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


@pytest.mark.parametrize("dbname", DATABASES)
def test_tables_exist(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        for t in ("bp_business_area", "bp_orchestrator_registry",
                  "bp_agent_policy", "bp_agent_policy_version"):
            cur.execute("SELECT to_regclass(%s)", (f"proc.{t}",))
            assert cur.fetchone()[0] is not None, t


@pytest.mark.parametrize("dbname", DATABASES)
def test_every_area_has_general_and_finance_needs_second_reviewer(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT area_name, sub_areas, second_reviewer, never_suggest FROM proc.bp_business_area")
        rows = {r[0]: r for r in cur.fetchall()}
    assert all("General" in r[1] for r in rows.values())
    assert rows["Finance"][2] is True
    assert not any(r[2] for n, r in rows.items() if n != "Finance")
    assert rows["Legal and compliance"][3] is True and rows["Security"][3] is True


@pytest.mark.parametrize("dbname", DATABASES)
def test_saved_version_cannot_be_changed_or_deleted(dbname):
    import psycopg2
    conn = _connect(dbname)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO proc.bp_agent_policy (policy_key, area_name, status, latest_version, created_by)"
                    " VALUES ('TST-9999', NULL, 'draft', 1, 'test') ")
        cur.execute("INSERT INTO proc.bp_agent_policy_version (policy_key, version, saved_as, form_state, saved_by)"
                    " VALUES ('TST-9999', 1, 'draft', '{}'::jsonb, 'test')")
        with pytest.raises(psycopg2.Error):
            cur.execute("UPDATE proc.bp_agent_policy_version SET change_note='x' WHERE policy_key='TST-9999'")
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_policy_untouched(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_policy' ORDER BY 1")
        cols = [r[0] for r in cur.fetchall()]
    assert cols == sorted(["created_by", "created_date", "last_modified_by", "last_modified_date",
                           "policy_desc", "policy_details", "policy_id", "policy_linked_agents",
                           "policy_name", "policy_status", "policy_type", "version"])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/migrations/test_2026_10_08_bp_agent_policy.py -v`
Expected: `test_tables_exist` FAILS (the tables don't exist yet) and `test_bp_policy_untouched` PASSES. If `test_bp_policy_untouched` fails, stop: its column list is wrong, so fix the test to match the live table before continuing.

- [ ] **Step 3: Write the migration**

```sql
-- 2026-10-08  Agent policy governance, stage 1.
-- Agent policies are NOT proc.bp_policy rows (ruling B, 2026-10-08): that table holds
-- role permissions and tuning settings and is left untouched. Additive, idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_business_area (
    area_name        TEXT PRIMARY KEY,
    id_prefix        TEXT NOT NULL UNIQUE CHECK (id_prefix ~ '^[A-Z]{3}$'),
    sub_areas        TEXT[] NOT NULL DEFAULT ARRAY['General']
        CHECK ('General' = ANY (sub_areas)),
    -- Highest number ever issued under this prefix. Only ever increases, so an id is never reused.
    last_number      INTEGER NOT NULL DEFAULT 0 CHECK (last_number >= 0),
    never_suggest    BOOLEAN NOT NULL DEFAULT FALSE,
    second_reviewer  BOOLEAN NOT NULL DEFAULT FALSE,
    is_unassigned    BOOLEAN NOT NULL DEFAULT FALSE,
    last_modified_by TEXT NOT NULL DEFAULT 'seed',
    last_modified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE proc.bp_business_area IS
    'Where an agent policy comes from (as its document states), never what it applies to. Admin-editable.';

INSERT INTO proc.bp_business_area (area_name, id_prefix, sub_areas, never_suggest, second_reviewer, is_unassigned) VALUES
  ('Unassigned',            'GEN', ARRAY['General'], FALSE, FALSE, TRUE),
  ('Finance',               'FIN', ARRAY['General','Refunds and credits','Payments','Expenses'], FALSE, TRUE, FALSE),
  ('Procurement',           'PRC', ARRAY['General','Sourcing','Purchase orders','Suppliers'], FALSE, FALSE, FALSE),
  ('Customer operations',   'CUS', ARRAY['General','Refunds','Complaints'], FALSE, FALSE, FALSE),
  ('Legal and compliance',  'LEG', ARRAY['General','Contracts','Data protection'], TRUE, FALSE, FALSE),
  ('Security',              'SEC', ARRAY['General','Access','Data handling'], TRUE, FALSE, FALSE),
  ('People',                'PPL', ARRAY['General'], FALSE, FALSE, FALSE),
  ('Operations',            'OPS', ARRAY['General'], FALSE, FALSE, FALSE)
ON CONFLICT (area_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS proc.bp_orchestrator_registry (
    registry_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('checkpoint','action','input')),
    name         TEXT NOT NULL,
    -- For 'action' and 'input': the checkpoint they belong to. NULL for a checkpoint.
    checkpoint   TEXT,
    plain        TEXT NOT NULL,
    value_type   TEXT CHECK (value_type IN ('string','number','boolean','date','list')),
    -- Where an input comes from. This release registers 'action' only (ruling: lookups and
    -- running totals belong to the orchestrator team and are not registered yet).
    source       TEXT CHECK (source IS NULL OR source = 'action' OR source LIKE 'lookup:%' OR source LIKE 'total:%'),
    -- 'live' = the orchestrator supplies it today. 'planned' = named but not yet supplied;
    -- a policy needing it shows "Can't be enforced yet".
    status       TEXT NOT NULL DEFAULT 'live' CHECK (status IN ('live','planned')),
    seeded_from  TEXT NOT NULL DEFAULT 'manual',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_orch_registry_checkpoint CHECK ((kind = 'checkpoint') = (checkpoint IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_orchestrator_registry_key
    ON proc.bp_orchestrator_registry (kind, name, COALESCE(checkpoint, ''));

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy (
    policy_key      TEXT PRIMARY KEY CHECK (policy_key ~ '^[A-Z]{3}-[0-9]{4,}$'),
    area_name       TEXT REFERENCES proc.bp_business_area (area_name),
    status          TEXT NOT NULL CHECK (status IN ('draft','live','retired')),
    live_version    INTEGER,
    latest_version  INTEGER NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_agent_policy_live CHECK ((status = 'live') = (live_version IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_status ON proc.bp_agent_policy (status);

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_version (
    policy_key     TEXT NOT NULL REFERENCES proc.bp_agent_policy (policy_key),
    version        INTEGER NOT NULL CHECK (version >= 1),
    saved_as       TEXT NOT NULL CHECK (saved_as IN ('draft','live','retired')),
    form_state     JSONB NOT NULL,
    compiled       JSONB,
    problems       JSONB NOT NULL DEFAULT '[]',
    confidence     JSONB,
    change_note    TEXT,
    saved_by       TEXT NOT NULL,
    saved_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (policy_key, version)
);

CREATE OR REPLACE FUNCTION proc.bp_agent_policy_version_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_agent_policy_version rows cannot be changed once saved (%)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_version_immutable ON proc.bp_agent_policy_version;
CREATE TRIGGER tr_bp_agent_policy_version_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_agent_policy_version
    FOR EACH ROW EXECUTE FUNCTION proc.bp_agent_policy_version_immutable();

-- An id, once issued, is never freed: policies are retired, not deleted.
CREATE OR REPLACE FUNCTION proc.bp_agent_policy_no_delete() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'agent policies are retired, never deleted';
END $$;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_no_delete ON proc.bp_agent_policy;
CREATE TRIGGER tr_bp_agent_policy_no_delete
    BEFORE DELETE ON proc.bp_agent_policy
    FOR EACH ROW EXECUTE FUNCTION proc.bp_agent_policy_no_delete();

INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) VALUES
  ('agent_policy_settings', '{
     "response_time": "PT4H",
     "response_time_basis": "clock",
     "on_missing_data": {"approve": "fail_closed", "block": "fail_closed", "notify": "fail_closed"},
     "live_conflict_repeat": 5,
     "learning": {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
                  "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30}
   }'::jsonb, 'seed')
ON CONFLICT (config_key) DO NOTHING;

COMMIT;
```

Rollback file:

```sql
-- Rollback for 2026-10-08_bp_agent_policy.sql. Destroys agent policies; run only to undo stage 1.
BEGIN;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_version_immutable ON proc.bp_agent_policy_version;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_no_delete ON proc.bp_agent_policy;
DROP TABLE IF EXISTS proc.bp_agent_policy_version;
DROP TABLE IF EXISTS proc.bp_agent_policy;
DROP TABLE IF EXISTS proc.bp_orchestrator_registry;
DROP TABLE IF EXISTS proc.bp_business_area;
DROP FUNCTION IF EXISTS proc.bp_agent_policy_version_immutable();
DROP FUNCTION IF EXISTS proc.bp_agent_policy_no_delete();
DELETE FROM proc.bp_admin_config WHERE config_key = 'agent_policy_settings';
COMMIT;
```

- [ ] **Step 4: Apply to both databases, then run the tests**

Announce the DDL first, because the databases are shared. Then:
`for db in bp_testdb bp_sqldb; do PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $db -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-08_bp_agent_policy.sql; done`
Re-run the Step 2 command. Expected: all 9 parametrised tests PASS.

- [ ] **Step 5: Prove the immutability guard fails when it should**

Temporarily comment out the `CREATE TRIGGER tr_bp_agent_policy_version_immutable` statement in a scratch copy, apply it to `bp_testdb` only, and run `test_saved_version_cannot_be_changed_or_deleted`. Expected: FAIL. Re-apply the real file and the test is back to PASS. Record both outputs in the task report.

- [ ] **Step 6: Commit** (stage only these three files)

```bash
git add deploy/sql/2026-10-08_bp_agent_policy.sql deploy/sql/2026-10-08_bp_agent_policy_rollback.sql tests/migrations/test_2026_10_08_bp_agent_policy.py
git commit -m "feat(agent-policy): tables for agent policies, taxonomy, registry and immutable versions"
```

---

### Task 2: Settings and the orchestrator registry (loader + seed)

**Files:**
- Create: `src/services/agent_policy/__init__.py` (empty)
- Create: `src/services/agent_policy/settings.py`
- Create: `src/services/agent_policy/registry.py`
- Create: `scripts/agent_policy/seed_registry.py`
- Test: `tests/agent_policy/test_settings_registry.py`, `tests/agent_policy/__init__.py` (empty)

**Interfaces:**
- Produces:
  - `settings.DEFAULTS: dict`
  - `settings.load_settings(conn=None) -> dict` (deep-merges over DEFAULTS; returns DEFAULTS if the row is missing or unreadable, and logs a warning)
  - `registry.RegistrySnapshot(checkpoints: dict[str, dict], actions: dict[str, set[str]], inputs: dict[str, dict[str, dict]])` with methods:
    - `.checkpoint_live(cp) -> bool`
    - `.knows_action(cp, name) -> bool`
    - `.input_row(cp, field) -> dict | None`
    - `.available(cp, field) -> bool` (exists AND `status=='live'`)
    - `.plain(cp) -> str`
  - `registry.load_registry(conn=None) -> RegistrySnapshot`
  - `registry.snapshot_from_rows(rows: list[dict]) -> RegistrySnapshot`
  - `seed_registry.registry_rows() -> list[dict]` and `seed_registry.main()`

- [ ] **Step 1: Write the failing tests**

```python
from services.agent_policy import registry, settings
from scripts.agent_policy import seed_registry


def test_settings_defaults_are_the_ruled_values():
    d = settings.DEFAULTS
    assert d["response_time"] == "PT4H" and d["response_time_basis"] == "clock"
    assert d["live_conflict_repeat"] == 5
    assert d["learning"] == {"min_decisions": 30, "min_days": 30, "min_approvers": 3,
                             "wilson_lower": 0.85, "median_seconds_floor": 30,
                             "not_yet_more": 30, "dismiss_more": 30}


def test_settings_merge_keeps_defaults_for_missing_keys():
    merged = settings.merge({"response_time": "PT2H", "learning": {"min_decisions": 40}})
    assert merged["response_time"] == "PT2H"
    assert merged["learning"]["min_decisions"] == 40
    assert merged["learning"]["min_days"] == 30


def _snap():
    return registry.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs", "status": "live"},
        {"kind": "checkpoint", "name": "message.send.before", "checkpoint": None, "plain": "before a message is sent", "status": "planned"},
        {"kind": "action", "name": "supplier_ranking", "checkpoint": "tool.call.before", "plain": "rank suppliers", "status": "live"},
        {"kind": "input", "name": "tool.name", "checkpoint": "tool.call.before", "plain": "tool name", "value_type": "string", "source": "action", "status": "live"},
        {"kind": "input", "name": "agg.refunds_30d", "checkpoint": "tool.call.before", "plain": "refunds in 30 days", "value_type": "number", "source": "total:refunds_30d", "status": "planned"},
    ])


def test_snapshot_answers_availability():
    s = _snap()
    assert s.checkpoint_live("tool.call.before")
    assert not s.checkpoint_live("message.send.before")
    assert s.knows_action("tool.call.before", "supplier_ranking")
    assert not s.knows_action("tool.call.before", "refund.issue")
    assert s.available("tool.call.before", "tool.name")
    assert not s.available("tool.call.before", "agg.refunds_30d")   # named, not supplied
    assert s.input_row("tool.call.before", "agg.refunds_30d")["status"] == "planned"
    assert not s.available("tool.call.before", "customer.country")  # not named at all


def test_seed_matches_the_tools_the_agent_loop_really_offers():
    """Guard: the registry must name exactly the tools build_tools() hands AgentNick."""
    from agents.auto_registry import AutoRegistry
    reg = AutoRegistry.from_json()
    real = {s["function"]["name"] for s in reg.tool_schemas()} | set(seed_registry.FIXED_TOOLS)
    seeded = {r["name"] for r in seed_registry.registry_rows()
              if r["kind"] == "action" and r["checkpoint"] == "tool.call.before"}
    assert seeded == real


def test_seed_registers_only_action_inputs_as_live():
    rows = seed_registry.registry_rows()
    live_inputs = [r for r in rows if r["kind"] == "input" and r["status"] == "live"]
    assert live_inputs and all(r["source"] == "action" for r in live_inputs)
    cps = {r["name"]: r["status"] for r in rows if r["kind"] == "checkpoint"}
    assert cps == {"tool.call.before": "live", "message.send.before": "planned",
                   "data.egress.before": "planned", "record.write.before": "planned"}
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_settings_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: services.agent_policy`.

- [ ] **Step 3: Implement**

`src/services/agent_policy/settings.py`:

```python
"""Company settings for agent policies, read from proc.bp_admin_config['agent_policy_settings'].

Defaults are the values ruled on 2026-10-08. A missing or unreadable row falls back to them;
it never falls back to something more permissive, because every default here is the safe one.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "response_time": "PT4H",
    "response_time_basis": "clock",
    "on_missing_data": {"approve": "fail_closed", "block": "fail_closed", "notify": "fail_closed"},
    "live_conflict_repeat": 5,
    "learning": {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
                 "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30},
}


def merge(stored: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(DEFAULTS)
    for key, value in (stored or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


def load_settings(conn: Any = None) -> Dict[str, Any]:
    from services.db import get_conn

    try:
        if conn is None:
            with get_conn() as own:
                return load_settings(own)
        cur = conn.cursor()
        cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = 'agent_policy_settings'")
        row = cur.fetchone()
        value = row[0] if row else None
        if isinstance(value, str):
            value = json.loads(value)
        return merge(value)
    except Exception as exc:  # unreadable settings -> ruled defaults, loudly
        logger.warning("agent_policy_settings unreadable, using defaults: %s", exc)
        return merge(None)
```

`src/services/agent_policy/registry.py`:

```python
"""What the orchestrator can see, at which checkpoint. The single source for every
"does the orchestrator recognise this name" and "is this input available here" question."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class RegistrySnapshot:
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actions: Dict[str, Set[str]] = field(default_factory=dict)
    inputs: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def checkpoint_live(self, cp: Optional[str]) -> bool:
        return bool(cp) and (self.checkpoints.get(cp) or {}).get("status") == "live"

    def knows_checkpoint(self, cp: Optional[str]) -> bool:
        return bool(cp) and cp in self.checkpoints

    def knows_action(self, cp: Optional[str], name: str) -> bool:
        return name in self.actions.get(cp or "", set())

    def input_row(self, cp: Optional[str], fld: str) -> Optional[Dict[str, Any]]:
        return self.inputs.get(cp or "", {}).get(fld)

    def available(self, cp: Optional[str], fld: str) -> bool:
        row = self.input_row(cp, fld)
        return bool(row) and row.get("status") == "live" and self.checkpoint_live(cp)

    def plain(self, cp: Optional[str]) -> str:
        return (self.checkpoints.get(cp or "") or {}).get("plain") or (cp or "")

    def as_dict(self) -> Dict[str, Any]:
        return {"checkpoints": self.checkpoints,
                "actions": {k: sorted(v) for k, v in self.actions.items()},
                "inputs": self.inputs}


def snapshot_from_rows(rows: List[Dict[str, Any]]) -> RegistrySnapshot:
    cps: Dict[str, Dict[str, Any]] = {}
    acts: Dict[str, Set[str]] = {}
    ins: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if r["kind"] == "checkpoint":
            cps[r["name"]] = {"plain": r["plain"], "status": r.get("status", "live")}
        elif r["kind"] == "action":
            acts.setdefault(r["checkpoint"], set()).add(r["name"])
        elif r["kind"] == "input":
            ins.setdefault(r["checkpoint"], {})[r["name"]] = {
                "plain": r["plain"], "type": r.get("value_type"),
                "source": r.get("source"), "status": r.get("status", "live")}
    return RegistrySnapshot(cps, acts, ins)


def load_registry(conn: Any = None) -> RegistrySnapshot:
    from services.db import get_conn

    if conn is None:
        with get_conn() as own:
            return load_registry(own)
    cur = conn.cursor()
    cur.execute("SELECT kind, name, checkpoint, plain, value_type, source, status "
                "FROM proc.bp_orchestrator_registry")
    cols = ("kind", "name", "checkpoint", "plain", "value_type", "source", "status")
    return snapshot_from_rows([dict(zip(cols, row)) for row in cur.fetchall()])
```

`scripts/agent_policy/seed_registry.py` (also create an empty `scripts/agent_policy/__init__.py`; if `scripts/` has no `__init__.py`, check with `ls scripts/__init__.py` and create an empty one so `from scripts.agent_policy import seed_registry` imports):

```python
"""Seed proc.bp_orchestrator_registry from the tools the agent loop really offers.

Only tool.call.before is 'live'. Stage 3 builds the gate there, and it supplies exactly the
inputs registered here: tool.name, agent.name, agent.reason and each tool's own arguments.
The other three checkpoints are named so a policy can say where it belongs, but are
'planned': nothing checks them yet, so a policy pinned to one shows "Can't be enforced yet".

Run: PYTHONPATH=.:src ./venv/bin/python -m scripts.agent_policy.seed_registry [--apply]
Without --apply it prints the rows and writes nothing.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

# Tools build_tools() adds besides the agents (orchestration/agentnick_control.py).
FIXED_TOOLS = {
    "list_governance": ["agent"],
    "get_policy": ["query"],
    "get_prompt": ["query"],
    "get_corpus_facts": ["question"],
}

CHECKPOINTS = [
    ("tool.call.before", "before a tool runs", "live"),
    ("message.send.before", "before a message is sent", "planned"),
    ("data.egress.before", "before data leaves the system", "planned"),
    ("record.write.before", "before a record is written", "planned"),
]
COMMON_INPUTS = [
    ("tool.name", "the tool being used", "string"),
    ("agent.name", "the agent acting", "string"),
    ("agent.reason", "the agent's stated reason", "string"),
]


def _agent_tools() -> Dict[str, List[str]]:
    from agents.auto_registry import AutoRegistry

    reg = AutoRegistry.from_json()
    out: Dict[str, List[str]] = {}
    for schema in reg.tool_schemas():
        fn = schema["function"]
        out[fn["name"]] = sorted(fn.get("parameters", {}).get("properties", {}).keys())
    return out


def registry_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, plain, status in CHECKPOINTS:
        rows.append({"kind": "checkpoint", "name": name, "checkpoint": None, "plain": plain,
                     "value_type": None, "source": None, "status": status})
    tools = {**_agent_tools(), **FIXED_TOOLS}
    cp = "tool.call.before"
    for field_name, plain, vtype in COMMON_INPUTS:
        rows.append({"kind": "input", "name": field_name, "checkpoint": cp, "plain": plain,
                     "value_type": vtype, "source": "action", "status": "live"})
    seen_args = set()
    for tool, params in sorted(tools.items()):
        rows.append({"kind": "action", "name": tool, "checkpoint": cp, "plain": tool.replace("_", " "),
                     "value_type": None, "source": None, "status": "live"})
        for p in params:
            if p in seen_args:
                continue
            seen_args.add(p)
            rows.append({"kind": "input", "name": f"args.{p}", "checkpoint": cp,
                         "plain": p.replace("_", " "), "value_type": "string",
                         "source": "action", "status": "live"})
    return rows


def main(argv: List[str]) -> int:
    rows = registry_rows()
    if "--apply" not in argv:
        print(json.dumps(rows, indent=1))
        return 0
    from services.db import get_conn

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                "INSERT INTO proc.bp_orchestrator_registry (kind, name, checkpoint, plain, value_type, source, status, seeded_from)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,'seed_registry')"
                " ON CONFLICT (kind, name, COALESCE(checkpoint, '')) DO NOTHING",
                (r["kind"], r["name"], r["checkpoint"], r["plain"], r["value_type"], r["source"], r["status"]))
        conn.commit()
    print(f"seeded {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run the tests, prove the seed guard, and seed both databases**

Run the Step 2 command. Expected: PASS.
**Prove the guard:** add a fake `"made_up_tool": []` to `FIXED_TOOLS` and confirm `test_seed_matches_the_tools_the_agent_loop_really_offers` FAILS. Then remove it.
**Seed:** run `--apply` against each database by setting `DB_NAME=bp_testdb` and then `DB_NAME=bp_sqldb`. First check how `get_conn` picks the database name: `grep -n "_pg_dsn" -A15 src/services/db.py`.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/__init__.py src/services/agent_policy/settings.py src/services/agent_policy/registry.py scripts/agent_policy/__init__.py scripts/agent_policy/seed_registry.py tests/agent_policy/__init__.py tests/agent_policy/test_settings_registry.py
git commit -m "feat(agent-policy): company settings and an orchestrator registry seeded from the real tool list"
```

---

### Task 3: Conditions, example results and the JSON compiler

**Files:**
- Create: `src/services/agent_policy/conditions.py`
- Create: `src/services/agent_policy/compiler.py`
- Test: `tests/agent_policy/test_conditions.py`, `tests/agent_policy/test_compiler.py`

**Interfaces:**
- Consumes: `services.policy_condition.evaluate / validate / MissingField / ConditionError` (existing).
- Produces, in `conditions.py`:
  - `OPS: dict[str, str]`
  - `to_engine(cond) -> dict`
  - `condition_fields(cond) -> set[str]`
  - `tool_names(cond) -> set[str]` (values of `tool.name` `eq`/`in` leaves)
  - `nest(flat: dict) -> dict`
  - `example_result(condition, outcome, example_input, on_missing="fail_closed") -> str` (one of `approve|block|notify|none`)
  - `RESULT_LABEL: dict[str, str]`
  - `reviewer_view(form, settings) -> list[dict]`, giving each example `{input, computed, label, flipped, reviewer_expects, agent_expected}`
- Produces, in `compiler.py`:
  - `compile_policy(form: dict, *, policy_key: str, version: int, status: str, settings: dict, never_suggest: bool, conflicts: list | None = None) -> dict`
  - `on_missing_for(form, settings) -> str`
  - `response_time(form, settings) -> tuple[str, str]` (duration, source)

**The form state** is the canonical shape stored in `bp_agent_policy_version.form_state`. Every later task and the UI use these exact keys:

```python
FORM_EXAMPLE = {
  "name": "Refund or credit over $500", "category": "Approval",
  "businessArea": "Finance", "subArea": "Refunds and credits",
  "situation": "The agent is about to issue a refund or credit above $500.",
  "source": {"document": "Finance Payments Policy", "documentVersion": 1, "reference": "1.1",
             "excerpt": "Refunds or credits above $500 need approval from the Finance Manager."},
  "outcome": "approve",                      # approve | block | notify | None
  "outcomeBecause": "need approval from",    # the document phrase behind the suggestion, or None
  "deciders": ["Finance Manager", "CFO"],    # approve only, ordered
  "responseTime": None,                      # None = company default, else ISO 8601 duration
  "notify": [],                              # notify (required) / block (optional)
  "limit": {"on": False, "text": ""},
  "owner": "Chief Financial Officer", "effectiveFrom": "2026-02-01", "reviewBy": None,
  "messageForAgent": "Refunds over $500 need Finance approval.",
  "messageForPerson": "Your refund needs a manager's approval. You will hear back within 4 hours.",
  "hidden": {
    "checkpoint": "tool.call.before",
    "actions": {"tools": ["refund.issue", "credit.issue"], "plain": "issuing a refund or credit"},
    "timeWindow": None,                      # or {"days": [...], "from": "18:00", "to": "08:00", "timeZone": "Europe/London"}
    "units": {"currency": "USD", "convertOther": "rate_on_action_date", "amountsIncludeTax": True},
    "inputs": [
      {"name": "Refund amount", "field": "args.amount", "type": "number", "isAmount": True, "unit": "USD",
       "from": "action", "showApprover": True, "sensitive": False},
      {"name": "Tool", "field": "tool.name", "type": "string", "from": "action", "showApprover": False, "sensitive": False},
      {"name": "Agent's reason", "field": "agent.reason", "type": "string", "from": "action", "showApprover": True, "sensitive": False},
    ],
    "missingInputs": [],                     # [{"name": "...", "reason": "..."}] from the agent
    "unknownNames": [],                      # names the agent needed but the registry lacks
    "condition": {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                          {"field": "args.amount", "op": "gt", "value": 500}]},
    "onMissingData": None,                   # None = company setting
    "reasonCode": "over_limit",
    "whilePaused": "no_retry",
    "setBy": "extraction_agent",             # or "person"
  },
  "examples": [
    {"input": {"tool.name": "refund.issue", "args.amount": 501}, "agentExpected": "approve", "flipped": False},
    {"input": {"tool.name": "refund.issue", "args.amount": 500}, "agentExpected": "none", "flipped": False},
    {"input": {"tool.name": "refund.issue", "args.amount": 499}, "agentExpected": "none", "flipped": False},
    {"input": {"tool.name": "supplier_ranking", "args.amount": 900}, "agentExpected": "none", "flipped": False},
  ],
  "checked": None,                           # {"by": "<subject>", "at": "<ISO>"} once confirmed
  "changeNote": "",
}
```

- [ ] **Step 1: Write the failing tests**

`tests/agent_policy/test_conditions.py`:

```python
import copy
import pytest

from services.agent_policy import conditions as C
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS


def test_translation_uses_one_evaluator():
    eng = C.to_engine(FORM_EXAMPLE["hidden"]["condition"])
    assert eng == {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                           {"field": "args.amount", "op": ">", "value": 500}]}


def test_unknown_operator_is_refused():
    with pytest.raises(C.ConditionError):
        C.to_engine({"field": "args.amount", "op": "greater", "value": 1})


def test_fields_and_tool_names():
    cond = FORM_EXAMPLE["hidden"]["condition"]
    assert C.condition_fields(cond) == {"tool.name", "args.amount"}
    assert C.tool_names(cond) == {"refund.issue", "credit.issue"}


@pytest.mark.parametrize("amount,expected", [(501, "approve"), (500, "none"), (499, "none")])
def test_boundary_results_are_computed_by_code(amount, expected):
    out = C.example_result(FORM_EXAMPLE["hidden"]["condition"], "approve",
                           {"tool.name": "refund.issue", "args.amount": amount})
    assert out == expected


def test_missing_input_fails_closed_by_default():
    out = C.example_result(FORM_EXAMPLE["hidden"]["condition"], "block", {"tool.name": "refund.issue"})
    assert out == "block"


def test_reviewer_view_labels_and_flip():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["examples"][1]["flipped"] = True
    rows = C.reviewer_view(form, SETTINGS)
    assert [r["label"] for r in rows] == ["A person decides", "Nothing happens", "Nothing happens", "Nothing happens"]
    assert rows[1]["flipped"] and rows[1]["reviewer_expects"] == "approve"
    assert rows[0]["reviewer_expects"] == "approve"
```

Create `tests/agent_policy/fixtures.py`, holding `FORM_EXAMPLE` exactly as above, `SETTINGS = settings.merge(None)`, and a `REGISTRY` built with `registry.snapshot_from_rows`. The registry has:
- the live checkpoint `tool.call.before`, with plain text `"before a tool runs"`;
- the actions `refund.issue`, `credit.issue` and `supplier_ranking` at that checkpoint;
- live action inputs `tool.name`, `agent.name`, `agent.reason`, `args.amount` and `args.currency`;
- a planned input `agg.refunds_30d`.

`tests/agent_policy/test_compiler.py`:

```python
import copy

from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS


def _compile(form, **kw):
    args = dict(policy_key="FIN-0012", version=2, status="live", settings=SETTINGS, never_suggest=False)
    args.update(kw)
    return compile_policy(form, **args)


def test_compiles_the_brief_example_shape():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    doc = _compile(form)
    assert doc["schema"] == "hard-policy/2" and doc["id"] == "FIN-0012" and doc["status"] == "live"
    assert doc["businessArea"] == {"primary": "Finance", "subArea": "Refunds and credits"}
    assert doc["scope"] == {"agents": ["*"], "tools": ["*"], "skills": ["*"], "limit": None}
    assert doc["trigger"]["events"] == ["tool.call.before"]
    assert doc["trigger"]["onMissingData"] == "fail_closed"
    assert doc["trigger"]["checkedBy"] == "user_8841"
    assert doc["enforcement"] == {"outcome": "approve", "intervention": {
        "escalateTo": [{"type": "role", "name": "Finance Manager"}, {"type": "role", "name": "CFO"}],
        "sla": {"source": "company_default", "respondWithin": "PT4H", "onTimeout": "escalate_next"}}}
    assert doc["outputs"]["toAgent"] == {"onMatch": "paused_for_approval", "reasonCode": "FIN-0012.over_limit",
                                         "reason": "Refunds over $500 need Finance approval.",
                                         "whilePaused": "no_retry",
                                         "messageForPerson": form["messageForPerson"]}
    assert doc["outputs"]["toApprover"]["show"] == ["args.amount", "agent.reason"]
    assert doc["outputs"]["audit"] == {"logInputs": True, "mask": []}
    assert doc["learning"] == {"eligible": True}
    assert doc["conflicts"] == []


def test_single_level_times_out_to_reject():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["deciders"] = ["Finance Manager"]
    assert _compile(form)["enforcement"]["intervention"]["sla"]["onTimeout"] == "reject"


def test_override_response_time_is_stored_as_iso_duration():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["responseTime"] = "PT6H"
    sla = _compile(form)["enforcement"]["intervention"]["sla"]
    assert sla == {"source": "policy", "respondWithin": "PT6H", "onTimeout": "escalate_next"}


def test_switching_outcome_drops_previous_fields_from_json():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["outcome"] = "block"          # deciders + responseTime still in form state on purpose
    form["responseTime"] = "PT6H"
    doc = _compile(form)
    assert doc["enforcement"] == {"outcome": "block"}
    assert doc["outputs"]["toApprover"] is None
    assert "whilePaused" not in doc["outputs"]["toAgent"]
    assert doc["outputs"]["toAgent"]["onMatch"] == "blocked"
    assert doc["learning"] == {"eligible": False}
    form["outcome"] = "notify"
    form["notify"] = ["Finance Manager"]
    doc = _compile(form)
    assert doc["enforcement"] == {"outcome": "notify", "notify": [{"type": "role", "name": "Finance Manager"}]}
    assert doc["outputs"]["toAgent"]["onMatch"] == "allowed"
    assert doc["outputs"]["toNotify"] == {"to": ["Finance Manager"]}


def test_never_suggest_turns_learning_off():
    assert _compile(copy.deepcopy(FORM_EXAMPLE), never_suggest=True)["learning"] == {"eligible": False}


def test_limit_text_and_sensitive_mask():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["limit"] = {"on": True, "text": "EU support agents only"}
    form["hidden"]["inputs"][2]["sensitive"] = True
    doc = _compile(form)
    assert doc["scope"]["limit"] == "EU support agents only"
    assert doc["outputs"]["audit"]["mask"] == ["agent.reason"]


def test_compile_is_pure():
    form = copy.deepcopy(FORM_EXAMPLE)
    before = copy.deepcopy(form)
    assert _compile(form) == _compile(form)
    assert form == before
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_conditions.py tests/agent_policy/test_compiler.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`src/services/agent_policy/conditions.py`:

```python
"""Agent-policy conditions, evaluated by the ONE existing evaluator (services.policy_condition).

Stored conditions use the brief's operator names; to_engine() is the only translation.
Example results are computed here, by code, so an example can never disagree with what the
orchestrator will enforce: the model proposes inputs, never results.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from services import policy_condition as pc
from services.policy_condition import ConditionError, MissingField  # re-exported

OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "==", "ne": "!=",
       "in": "in", "not_in": "not_in", "exists": "exists"}
RESULT_LABEL = {"approve": "A person decides", "block": "Blocked",
                "notify": "Someone is told", "none": "Nothing happens"}


def to_engine(cond: Any) -> Dict[str, Any]:
    if not isinstance(cond, dict) or not cond:
        raise ConditionError("a condition must be a non-empty object")
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        return {key: [to_engine(c) for c in cond[key]]}
    if "not" in cond:
        return {"not": to_engine(cond["not"])}
    op = cond.get("op")
    if op not in OPS:
        raise ConditionError(f"unknown operator {op!r}")
    out = {"field": cond.get("field"), "op": OPS[op]}
    if op != "exists":
        out["value"] = cond.get("value")
    pc.validate(out)
    return out


def _leaves(cond: Any) -> List[Dict[str, Any]]:
    if not isinstance(cond, dict):
        return []
    for key in ("all", "any"):
        if key in cond:
            return [leaf for c in cond[key] for leaf in _leaves(c)]
    if "not" in cond:
        return _leaves(cond["not"])
    return [cond]


def condition_fields(cond: Any) -> Set[str]:
    return {leaf["field"] for leaf in _leaves(cond) if leaf.get("field")}


def tool_names(cond: Any) -> Set[str]:
    out: Set[str] = set()
    for leaf in _leaves(cond):
        if leaf.get("field") != "tool.name":
            continue
        value = leaf.get("value")
        out.update(value if isinstance(value, list) else [value] if value is not None else [])
    return out


def nest(flat: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, value in (flat or {}).items():
        node = out
        parts = str(path).split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def example_result(condition: Any, outcome: Any, example_input: Dict[str, Any],
                   on_missing: str = "fail_closed") -> str:
    if outcome not in ("approve", "block", "notify"):
        return "none"
    try:
        hit = pc.evaluate(to_engine(condition), nest(example_input))
    except MissingField:
        hit = on_missing == "fail_closed"
    return outcome if hit else "none"


def reviewer_view(form: Dict[str, Any], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    from services.agent_policy.compiler import on_missing_for

    hidden = form.get("hidden") or {}
    outcome = form.get("outcome")
    rows = []
    for ex in form.get("examples") or []:
        try:
            computed = example_result(hidden.get("condition"), outcome, ex.get("input") or {},
                                      on_missing_for(form, settings))
        except ConditionError:
            computed = "invalid"
        flipped = bool(ex.get("flipped"))
        expects = computed
        if flipped and computed in ("none",) and outcome:
            expects = outcome
        elif flipped:
            expects = "none"
        rows.append({"input": ex.get("input") or {}, "computed": computed,
                     "label": RESULT_LABEL.get(computed, "The condition could not be read"),
                     "flipped": flipped, "reviewer_expects": expects,
                     "agent_expected": ex.get("agentExpected")})
    return rows
```

`src/services/agent_policy/compiler.py`:

```python
"""Form state -> hard-policy/2. Pure: no database, no model, no clock, no mutation of the input.

Only the current outcome's fields are emitted, so switching outcome in the form can never
leave a stale level, response time or approver block in the JSON.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

_ON_MATCH = {"approve": "paused_for_approval", "block": "blocked", "notify": "allowed"}


def on_missing_for(form: Dict[str, Any], settings: Dict[str, Any]) -> str:
    explicit = (form.get("hidden") or {}).get("onMissingData")
    if explicit:
        return explicit
    return (settings.get("on_missing_data") or {}).get(form.get("outcome") or "", "fail_closed")


def response_time(form: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[str, str]:
    override = form.get("responseTime")
    if override:
        return override, "policy"
    return settings["response_time"], "company_default"


def _roles(names: List[str]) -> List[Dict[str, str]]:
    return [{"type": "role", "name": n} for n in names if str(n).strip()]


def compile_policy(form: Dict[str, Any], *, policy_key: str, version: int, status: str,
                   settings: Dict[str, Any], never_suggest: bool,
                   conflicts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    f = copy.deepcopy(form)
    h = f.get("hidden") or {}
    outcome = f.get("outcome")
    inputs = h.get("inputs") or []
    checkpoint = h.get("checkpoint")
    limit = f.get("limit") or {}
    checked = f.get("checked") or {}

    json_inputs = []
    for i in inputs:
        row = {"name": i.get("name"), "field": i.get("field"), "type": i.get("type"),
               "from": i.get("from"), "showApprover": bool(i.get("showApprover")),
               "sensitive": bool(i.get("sensitive"))}
        if i.get("unit"):
            row["unit"] = i["unit"]
        json_inputs.append(row)

    to_agent: Dict[str, Any] = {"onMatch": _ON_MATCH.get(outcome),
                                "reasonCode": f"{policy_key}.{h.get('reasonCode') or 'policy'}",
                                "reason": f.get("messageForAgent"),
                                "messageForPerson": f.get("messageForPerson")}
    if outcome == "approve":
        to_agent["whilePaused"] = h.get("whilePaused") or "no_retry"
        # keep the brief's key order: onMatch, reasonCode, reason, whilePaused, messageForPerson
        to_agent = {k: to_agent[k] for k in ("onMatch", "reasonCode", "reason", "whilePaused", "messageForPerson")}

    enforcement: Dict[str, Any] = {"outcome": outcome}
    to_approver = None
    to_notify = None
    if outcome == "approve":
        deciders = [d for d in (f.get("deciders") or []) if str(d).strip()]
        within, source = response_time(f, settings)
        enforcement["intervention"] = {
            "escalateTo": _roles(deciders),
            "sla": {"source": source, "respondWithin": within,
                    "onTimeout": "escalate_next" if len(deciders) > 1 else "reject"}}
        to_approver = {"show": [i["field"] for i in inputs if i.get("showApprover")],
                       "options": ["approve", "reject"], "reasonRequiredOn": ["reject"]}
    elif outcome in ("block", "notify"):
        told = [n for n in (f.get("notify") or []) if str(n).strip()]
        if told:
            enforcement["notify"] = _roles(told)
            to_notify = {"to": told}

    return {
        "schema": "hard-policy/2",
        "id": policy_key,
        "version": version,
        "status": status,
        "title": f.get("name"),
        "category": f.get("category"),
        "owner": f.get("owner"),
        "businessArea": {"primary": f.get("businessArea"), "subArea": f.get("subArea")},
        "effective": {"from": f.get("effectiveFrom") or None, "reviewBy": f.get("reviewBy") or None},
        "scope": {"agents": ["*"], "tools": ["*"], "skills": ["*"],
                  "limit": (limit.get("text") or None) if limit.get("on") else None},
        "source": f.get("source"),
        "context": {"checkpoint": checkpoint, "actions": h.get("actions"),
                    "timeWindow": h.get("timeWindow"), "units": h.get("units")},
        "inputs": json_inputs,
        "outputs": {"toAgent": to_agent, "toApprover": to_approver, "toNotify": to_notify,
                    "audit": {"logInputs": True,
                              "mask": [i["field"] for i in inputs if i.get("sensitive")]}},
        "trigger": {"plain": f.get("situation"),
                    "events": [checkpoint] if checkpoint else [],
                    "condition": h.get("condition"),
                    "onMissingData": on_missing_for(f, settings),
                    "setBy": h.get("setBy") or "person",
                    "checkedBy": checked.get("by"),
                    "checkedAt": checked.get("at")},
        "enforcement": enforcement,
        "conflicts": list(conflicts or []),
        "learning": {"eligible": outcome == "approve" and not never_suggest},
    }
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/conditions.py src/services/agent_policy/compiler.py tests/agent_policy/fixtures.py tests/agent_policy/test_conditions.py tests/agent_policy/test_compiler.py
git commit -m "feat(agent-policy): example results computed by code, and a pure compiler to hard-policy/2"
```

---

### Task 4: JSON Schema and contract validation

**Files:**
- Create: `src/services/agent_policy/hard-policy-2.schema.json`
- Create: `src/services/agent_policy/contract.py`
- Test: `tests/agent_policy/test_contract.py`

**Interfaces:**
- Consumes: `compile_policy` and the `RegistrySnapshot` methods.
- Produces: `contract.validate(doc: dict, registry: RegistrySnapshot) -> list[str]`. An empty list means valid. Each message is plain English and names the field.

- [ ] **Step 1: Write the failing tests**

```python
import copy

from services.agent_policy import contract
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS


def _doc(mutate=None, status="live"):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    if mutate:
        mutate(form)
    return compile_policy(form, policy_key="FIN-0012", version=1, status=status,
                          settings=SETTINGS, never_suggest=False)


def test_the_example_is_valid():
    assert contract.validate(_doc(), REGISTRY) == []


def test_condition_field_missing_from_inputs_fails():
    def drop(form):
        form["hidden"]["inputs"] = [i for i in form["hidden"]["inputs"] if i["field"] != "args.amount"]
    problems = contract.validate(_doc(drop), REGISTRY)
    assert any("args.amount" in p and "inputs" in p for p in problems)


def test_live_without_checked_by_is_refused():
    problems = contract.validate(_doc(lambda f: f.update(checked=None)), REGISTRY)
    assert any("checkedBy" in p for p in problems)


def test_unknown_tool_is_refused():
    problems = contract.validate(_doc(lambda f: f["hidden"]["condition"]["all"][0].update(value=["refund.isue"])), REGISTRY)
    assert any("refund.isue" in p for p in problems)


def test_input_not_available_at_checkpoint_is_refused():
    def planned(form):
        form["hidden"]["inputs"].append({"name": "Refunds in 30 days", "field": "agg.refunds_30d", "type": "number",
                                         "from": "total:refunds_30d", "showApprover": False, "sensitive": False})
    problems = contract.validate(_doc(planned), REGISTRY)
    assert any("agg.refunds_30d" in p for p in problems)


def test_events_must_equal_checkpoint():
    doc = _doc()
    doc["trigger"]["events"] = ["message.send.before"]
    assert any("events" in p for p in contract.validate(doc, REGISTRY))


def test_block_with_intervention_fails_schema():
    doc = _doc(lambda f: f.update(outcome="block"))
    doc["enforcement"]["intervention"] = {"escalateTo": [], "sla": {}}
    assert contract.validate(doc, REGISTRY)


def test_notify_without_notify_fails_schema():
    doc = _doc(lambda f: f.update(outcome="notify", notify=[]))
    assert contract.validate(doc, REGISTRY)
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_contract.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`src/services/agent_policy/hard-policy-2.schema.json` (Draft 2020-12):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "hard-policy/2",
  "type": "object",
  "required": ["schema","id","version","status","title","category","owner","businessArea","effective","scope","source","context","inputs","outputs","trigger","enforcement","conflicts","learning"],
  "additionalProperties": false,
  "properties": {
    "schema": {"const": "hard-policy/2"},
    "id": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4,}$"},
    "version": {"type": "integer", "minimum": 1},
    "status": {"enum": ["draft","live","retired"]},
    "title": {"type": "string", "minLength": 1},
    "category": {"type": ["string","null"]},
    "owner": {"type": ["string","null"]},
    "businessArea": {"type": "object", "required": ["primary","subArea"],
      "properties": {"primary": {"type": ["string","null"]}, "subArea": {"type": ["string","null"]}}, "additionalProperties": false},
    "effective": {"type": "object", "required": ["from","reviewBy"],
      "properties": {"from": {"type": ["string","null"], "format": "date"}, "reviewBy": {"type": ["string","null"], "format": "date"}}, "additionalProperties": false},
    "scope": {"type": "object", "required": ["agents","tools","skills","limit"],
      "properties": {"agents": {"const": ["*"]}, "tools": {"const": ["*"]}, "skills": {"const": ["*"]}, "limit": {"type": ["string","null"]}}, "additionalProperties": false},
    "source": {"type": ["object","null"], "required": ["document","documentVersion","reference","excerpt"],
      "properties": {"document": {"type": "string"}, "documentVersion": {"type": ["integer","null"]}, "reference": {"type": ["string","null"]}, "excerpt": {"type": "string"}}, "additionalProperties": false},
    "context": {"type": "object", "required": ["checkpoint","actions","timeWindow","units"],
      "properties": {
        "checkpoint": {"type": ["string","null"]},
        "actions": {"type": ["object","null"], "properties": {"tools": {"type": "array", "items": {"type": "string"}}, "plain": {"type": "string"}}},
        "timeWindow": {"type": ["object","null"], "required": ["timeZone"], "properties": {"timeZone": {"type": "string", "minLength": 1}}},
        "units": {"type": ["object","null"]}
      }, "additionalProperties": false},
    "inputs": {"type": "array", "items": {"type": "object", "required": ["name","field","type","from","showApprover","sensitive"],
      "properties": {"name": {"type": "string"}, "field": {"type": "string"}, "type": {"enum": ["string","number","boolean","date","list"]},
        "unit": {"type": "string"}, "from": {"type": "string", "pattern": "^(action|lookup:.+|total:.+)$"},
        "showApprover": {"type": "boolean"}, "sensitive": {"type": "boolean"}}, "additionalProperties": false}},
    "outputs": {"type": "object", "required": ["toAgent","toApprover","toNotify","audit"]},
    "trigger": {"type": "object", "required": ["plain","events","condition","onMissingData","setBy","checkedBy","checkedAt"],
      "properties": {"onMissingData": {"enum": ["fail_closed","fail_open"]}, "setBy": {"enum": ["extraction_agent","person"]}}},
    "enforcement": {"type": "object", "required": ["outcome"],
      "oneOf": [
        {"properties": {"outcome": {"const": "approve"},
                        "intervention": {"type": "object", "required": ["escalateTo","sla"],
                          "properties": {"escalateTo": {"type": "array", "minItems": 1},
                                         "sla": {"type": "object", "required": ["source","respondWithin","onTimeout"],
                                                 "properties": {"source": {"enum": ["company_default","policy"]},
                                                                "respondWithin": {"type": "string", "pattern": "^P(?!$)(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+S)?)?$"},
                                                                "onTimeout": {"enum": ["escalate_next","reject"]}}}}}},
         "required": ["intervention"], "not": {"required": ["notify"]}},
        {"properties": {"outcome": {"const": "block"}}, "not": {"required": ["intervention"]}},
        {"properties": {"outcome": {"const": "notify"}, "notify": {"type": "array", "minItems": 1}},
         "required": ["notify"], "not": {"required": ["intervention"]}}
      ]},
    "conflicts": {"type": "array", "items": {"type": "object", "required": ["with","rule","caseId","decidedAt"]}},
    "learning": {"type": "object", "required": ["eligible"], "properties": {"eligible": {"type": "boolean"}}}
  }
}
```

This schema file is read by Python's `jsonschema` only, never by Ollama, so `oneOf` is safe here (see memory: Ollama ignores unions).

`src/services/agent_policy/contract.py`:

```python
"""Is a hard-policy/2 document one the orchestrator can enforce without guessing?

JSON Schema covers shape. The cross-field rules the brief adds (§7 "Contract") cannot be
expressed in JSON Schema, so they are checked here, in the same call: one function, one
answer, used by activation AND by the orchestrator feed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

from services.agent_policy import conditions
from services.agent_policy.registry import RegistrySnapshot

_SCHEMA = json.loads((Path(__file__).with_name("hard-policy-2.schema.json")).read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA)


def validate(doc: Dict[str, Any], registry: RegistrySnapshot) -> List[str]:
    problems = [f"{'/'.join(str(p) for p in e.path) or 'document'}: {e.message}"
                for e in _VALIDATOR.iter_errors(doc)]
    trigger = doc.get("trigger") or {}
    context = doc.get("context") or {}
    cp = context.get("checkpoint")
    input_fields = {i.get("field") for i in doc.get("inputs") or []}

    if trigger.get("events") != ([cp] if cp else []):
        problems.append("trigger.events must equal [context.checkpoint]")
    if not registry.knows_checkpoint(cp):
        problems.append(f"context.checkpoint {cp!r} is not a checkpoint the orchestrator recognises")

    cond = trigger.get("condition")
    try:
        conditions.to_engine(cond)
    except conditions.ConditionError as exc:
        problems.append(f"trigger.condition cannot be read: {exc}")
    for fld in sorted(conditions.condition_fields(cond) - input_fields):
        problems.append(f"condition field {fld} is not listed in inputs")
    for tool in sorted(conditions.tool_names(cond)):
        if not registry.knows_action(cp, tool):
            problems.append(f"tool {tool} is not something the orchestrator recognises")
    for tool in (context.get("actions") or {}).get("tools") or []:
        if not registry.knows_action(cp, tool):
            problems.append(f"tool {tool} is not something the orchestrator recognises")

    show = ((doc.get("outputs") or {}).get("toApprover") or {}).get("show") or []
    for fld in show:
        if fld not in input_fields:
            problems.append(f"approver field {fld} is not listed in inputs")
    for i in doc.get("inputs") or []:
        if not registry.available(cp, i.get("field")):
            problems.append(f"input {i.get('field')} is not available at {cp}")

    if doc.get("status") == "live" and not trigger.get("checkedBy"):
        problems.append("a live policy needs trigger.checkedBy")
    return sorted(set(problems))
```

- [ ] **Step 4: Run the tests, and prove each cross-field check fails when broken**

Run the Step 2 command. Expected: PASS.
Then, one at a time, delete each of these blocks in `contract.py` and confirm its named test goes red:
- the events check;
- the condition-field loop;
- the tool loop;
- the availability loop;
- the `checkedBy` check.

Restore each block after its check. Record each red/green pair in the task report.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/hard-policy-2.schema.json src/services/agent_policy/contract.py tests/agent_policy/test_contract.py
git commit -m "feat(agent-policy): hard-policy/2 JSON Schema and the cross-field contract checks"
```

---

### Task 5: What Active requires, How it is enforced, and extraction confidence

**Files:**
- Create: `src/services/agent_policy/readiness.py`
- Test: `tests/agent_policy/test_readiness.py`

**Interfaces:**
- Consumes: `conditions.reviewer_view`, `conditions.condition_fields`, `conditions.tool_names`, `compiler.compile_policy`, `contract.validate`, and the `RegistrySnapshot` methods.
- Produces:
  - `Problem = dict(field: str, message: str)`
  - `activation_problems(form, registry, settings) -> list[Problem]`, in form order, so `[0]` is the field to focus;
  - `how_enforced(form, registry, settings) -> dict`, which is either `{"ok": True, "checkedWhen": str, "needsToKnow": str, "then": str}` or `{"ok": False, "cantEnforce": [str]}`;
  - `extraction_confidence(form, document_text, registry, settings) -> dict | None`, which is `{"level": "High"|"Medium"|"Low", "failed": [str]}`, or `None` when the policy has no source document;
  - `confirmation_cleared(old_form, new_form) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
import copy

from services.agent_policy import readiness as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

DOC_TEXT = "1.1 Refunds or credits above $500 need approval from the Finance Manager.\n1.2 ..."


def _ready_form():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["hidden"]["condition"]["all"][0]["value"] = ["refund.issue", "credit.issue"]
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    return form


def _fields(problems):
    return [p["field"] for p in problems]


def test_complete_policy_has_no_problems():
    assert R.activation_problems(_ready_form(), REGISTRY, SETTINGS) == []


def test_draft_with_only_a_name_lists_every_problem_in_form_order():
    problems = R.activation_problems({"name": "Only a name"}, REGISTRY, SETTINGS)
    fields = _fields(problems)
    assert fields[:3] == ["businessArea", "subArea", "situation"]
    for f in ("examples", "checked", "checkpoint", "outcome", "owner"):
        assert f in fields
    # form order: The policy -> What happens -> Governance
    assert fields.index("situation") < fields.index("examples") < fields.index("outcome") < fields.index("owner")


def test_approve_needs_a_decider_and_a_positive_response_time():
    form = _ready_form()
    form["deciders"] = []
    form["responseTime"] = "PT0H"
    fields = _fields(R.activation_problems(form, REGISTRY, SETTINGS))
    assert "deciders" in fields and "responseTime" in fields


def test_notify_needs_someone_to_tell():
    form = _ready_form()
    form["outcome"] = "notify"
    form["notify"] = []
    assert "notify" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_limit_on_needs_text():
    form = _ready_form()
    form["limit"] = {"on": True, "text": "  "}
    assert "limit" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_flipped_example_blocks_active():
    form = _ready_form()
    form["examples"][0]["flipped"] = True
    assert "examples" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_unknown_tool_is_routed_to_an_administrator():
    form = _ready_form()
    form["hidden"]["condition"]["all"][0]["value"] = ["refund.isue"]
    msgs = [p for p in R.activation_problems(form, REGISTRY, SETTINGS) if p["field"] == "registry"]
    assert msgs and msgs[0]["message"].startswith("This policy refers to something the orchestrator does not recognise")
    assert msgs[0].get("routeTo") == "administrator"


def test_amount_without_currency_blocks_active():
    form = _ready_form()
    form["hidden"]["units"]["currency"] = None
    assert "units" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_approve_and_block_need_a_message_to_the_agent():
    form = _ready_form()
    form["messageForAgent"] = ""
    assert "messageForAgent" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_unavailable_input_cannot_be_enforced_yet():
    form = _ready_form()
    form["hidden"]["inputs"].append({"name": "refunds to this customer in the last 30 days", "field": "agg.refunds_30d",
                                     "type": "number", "from": "total:refunds_30d", "showApprover": False, "sensitive": False})
    he = R.how_enforced(form, REGISTRY, SETTINGS)
    assert he["ok"] is False
    assert he["cantEnforce"] == ["Can't be enforced yet: the orchestrator does not receive "
                                 "refunds to this customer in the last 30 days at this point"]
    assert "inputs" in _fields(R.activation_problems(form, REGISTRY, SETTINGS))


def test_agent_reported_missing_input_also_cannot_be_enforced():
    form = _ready_form()
    form["hidden"]["missingInputs"] = [{"name": "customer's country", "reason": "no registry field"}]
    assert R.how_enforced(form, REGISTRY, SETTINGS)["ok"] is False


def test_how_enforced_reads_in_plain_words():
    he = R.how_enforced(_ready_form(), REGISTRY, SETTINGS)
    assert he == {"ok": True,
                  "checkedWhen": "the agent is about to do this: issuing a refund or credit (before a tool runs)",
                  "needsToKnow": "Refund amount (USD, from the action), Tool (from the action), Agent's reason (from the action)",
                  "then": "the action pauses and Finance Manager is asked to approve, then CFO if there is no answer "
                          "within 4 hours. The agent tells the person: \"Your refund needs a manager's approval. "
                          "You will hear back within 4 hours.\""}


def test_confirmation_is_cleared_by_situation_or_flip_but_not_by_owner():
    old = _ready_form()
    new = copy.deepcopy(old); new["situation"] += " Today."
    assert R.confirmation_cleared(old, new)
    new = copy.deepcopy(old); new["examples"][1]["flipped"] = True
    assert R.confirmation_cleared(old, new)
    new = copy.deepcopy(old); new["owner"] = "CFO"
    assert not R.confirmation_cleared(old, new)


def test_extraction_confidence_high_medium_low():
    form = _ready_form()
    form["hidden"]["setBy"] = "extraction_agent"
    for ex, agent in zip(form["examples"], ["approve", "none", "none", "none"]):
        ex["agentExpected"] = agent
    assert R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS) == {"level": "High", "failed": []}
    form["examples"][0]["agentExpected"] = "none"            # agent disagrees with code
    assert R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)["level"] == "Medium"
    form["source"]["excerpt"] = "Refunds above $500 need approval."   # not word for word
    low = R.extraction_confidence(form, DOC_TEXT, REGISTRY, SETTINGS)
    assert low["level"] == "Low"
    assert set(low["failed"]) == {"The excerpt does not appear word for word in the document",
                                  "The agent's expected result differs from the computed one for 1 example"}


def test_no_source_means_no_extraction_confidence():
    form = _ready_form(); form["source"] = None
    assert R.extraction_confidence(form, None, REGISTRY, SETTINGS) is None
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_readiness.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`src/services/agent_policy/readiness.py`:

```python
"""What stands between a policy and Active, said in the form's own words.

activation_problems() returns EVERY problem at once, ordered as the form is laid out, so the
UI can show one summary and focus problems[0]. A Draft never calls this: a Draft saves with
only a name.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.agent_policy import conditions
from services.agent_policy.compiler import response_time
from services.agent_policy.registry import RegistrySnapshot

UNKNOWN_NAME = "This policy refers to something the orchestrator does not recognise"
_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
_CONFIRM_KEYS = ("situation", "outcome", "hidden", "messageForPerson", "deciders", "notify")


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and not v)


def _seconds(duration: Optional[str]) -> Optional[int]:
    m = _DURATION.match(duration or "")
    if not m or duration in ("P", "PT"):
        return None
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _human(duration: str) -> str:
    secs = _seconds(duration) or 0
    hours, rem = divmod(secs, 3600)
    if rem == 0 and hours:
        return f"{hours} hour" + ("s" if hours != 1 else "")
    return duration


def _cant_enforce(form: Dict[str, Any], registry: RegistrySnapshot) -> List[str]:
    h = form.get("hidden") or {}
    cp = h.get("checkpoint")
    out = []
    for i in h.get("inputs") or []:
        if not registry.available(cp, i.get("field")):
            out.append(f"Can't be enforced yet: the orchestrator does not receive {i.get('name') or i.get('field')} at this point")
    for m in h.get("missingInputs") or []:
        out.append(f"Can't be enforced yet: the orchestrator does not receive {m.get('name')} at this point")
    if cp and registry.knows_checkpoint(cp) and not registry.checkpoint_live(cp):
        out.append(f"Can't be enforced yet: nothing checks policies {registry.plain(cp)} yet")
    return out


def how_enforced(form: Dict[str, Any], registry: RegistrySnapshot, settings: Dict[str, Any]) -> Dict[str, Any]:
    cant = _cant_enforce(form, registry)
    if cant:
        return {"ok": False, "cantEnforce": cant}
    h = form.get("hidden") or {}
    cp = h.get("checkpoint")
    plain = (h.get("actions") or {}).get("plain") or "act"
    needs = ", ".join(
        f"{i.get('name')} ({i['unit']}, from the action)" if i.get("unit") else f"{i.get('name')} (from the action)"
        for i in h.get("inputs") or [])
    outcome = form.get("outcome")
    person = form.get("messageForPerson")
    tail = f' The agent tells the person: "{person}"' if person else ""
    if outcome == "approve":
        deciders = [d for d in form.get("deciders") or [] if str(d).strip()]
        within, _ = response_time(form, settings)
        chain = deciders[0] if deciders else "nobody yet"
        if len(deciders) > 1:
            chain += ", then " + ", then ".join(deciders[1:])
            then = f"the action pauses and {deciders[0]} is asked to approve, then {', then '.join(deciders[1:])} if there is no answer within {_human(within)}."
        else:
            then = f"the action pauses and {chain} is asked to approve within {_human(within)}; with no answer it is rejected."
    elif outcome == "block":
        then = "the action is refused. Nobody is asked to approve it."
    elif outcome == "notify":
        then = "the action goes ahead and " + ", ".join(form.get("notify") or ["nobody yet"]) + " is told."
    else:
        then = "nothing yet: choose what happens."
    return {"ok": True,
            "checkedWhen": f"the agent is about to do this: {plain} ({registry.plain(cp)})",
            "needsToKnow": needs,
            "then": then + tail}


def activation_problems(form: Dict[str, Any], registry: RegistrySnapshot,
                        settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    p: List[Dict[str, Any]] = []
    add = lambda f, m, **kw: p.append({"field": f, "message": m, **kw})  # noqa: E731
    h = form.get("hidden") or {}
    outcome = form.get("outcome")

    # Form order (brief §3.1): Identity, The policy (situation, how enforced, examples),
    # What happens, Applies to, Governance. problems[0] is therefore the first field to fix.
    if _blank(form.get("name")): add("name", "Name is required.")
    if _blank(form.get("businessArea")): add("businessArea", "Business area is required.")
    if _blank(form.get("subArea")): add("subArea", "Sub-area is required.")
    if _blank(form.get("situation")): add("situation", "The situation is required.")

    cp = h.get("checkpoint")
    if _blank(cp): add("checkpoint", "The policy has no checkpoint.")
    cond = h.get("condition")
    unknown = [t for t in sorted(conditions.tool_names(cond)) if not registry.knows_action(cp, t)]
    unknown += [t for t in (h.get("actions") or {}).get("tools") or [] if not registry.knows_action(cp, t)]
    unknown += [f for f in sorted(conditions.condition_fields(cond)) if not registry.input_row(cp, f)]
    unknown += list(h.get("unknownNames") or [])
    if cp and not registry.knows_checkpoint(cp):
        unknown.append(cp)
    if unknown:
        add("registry", f"{UNKNOWN_NAME}: {', '.join(sorted(set(unknown)))}.", routeTo="administrator")
    cant = _cant_enforce(form, registry)
    if cant:
        add("inputs", " ".join(cant), routeTo="administrator")
    inputs = h.get("inputs") or []
    if any(i.get("isAmount") for i in inputs):
        currency = (h.get("units") or {}).get("currency")
        if _blank(currency) or any(i.get("isAmount") and i.get("unit") != currency for i in inputs):
            add("units", "Amounts need a stated currency.")
    tw = h.get("timeWindow")
    if tw and _blank(tw.get("timeZone")):
        add("timeWindow", "A time window needs a time zone.")

    rows = conditions.reviewer_view(form, settings)
    if not rows:
        add("examples", "There are no examples to check.")
    elif any(r["flipped"] for r in rows):
        add("examples", "An example was marked wrong, so the condition is wrong. Ask the agent to fix it.")
    elif any(r["computed"] == "invalid" for r in rows):
        add("examples", "The condition could not be read. Ask the agent to fix it.")
    if not form.get("checked"):
        add("checked", "Confirm the examples and how it is enforced.")

    if outcome not in ("approve", "block", "notify"): add("outcome", "Choose what happens.")
    if outcome == "approve":
        if not [d for d in form.get("deciders") or [] if str(d).strip()]:
            add("deciders", "Add at least one person or role who decides.")
        if form.get("responseTime") is not None and (_seconds(form["responseTime"]) or 0) < 1:
            add("responseTime", "A custom response time must be at least 1.")
    if outcome == "notify" and not [n for n in form.get("notify") or [] if str(n).strip()]:
        add("notify", "Add at least one person to tell.")
    if outcome in ("approve", "block") and _blank(form.get("messageForAgent")):
        add("messageForAgent", "Write the message the agent receives.")

    limit = form.get("limit") or {}
    if limit.get("on") and _blank(limit.get("text")):
        add("limit", "Say how the policy is limited, or turn Limit it off.")

    if _blank(form.get("owner")): add("owner", "Owner is required.")
    return p


def confirmation_cleared(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    if any(old.get(k) != new.get(k) for k in _CONFIRM_KEYS):
        return True
    return [bool(e.get("flipped")) for e in old.get("examples") or []] != \
           [bool(e.get("flipped")) for e in new.get("examples") or []] or \
           [e.get("input") for e in old.get("examples") or []] != [e.get("input") for e in new.get("examples") or []]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extraction_confidence(form: Dict[str, Any], document_text: Optional[str], registry: RegistrySnapshot,
                          settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not form.get("source"):
        return None
    failed: List[str] = []
    excerpt = _normalise((form.get("source") or {}).get("excerpt"))
    if not excerpt or excerpt not in _normalise(document_text or ""):
        failed.append("The excerpt does not appear word for word in the document")
    required = ("name", "category", "businessArea", "subArea", "situation", "outcome", "owner")
    if any(_blank(form.get(k)) for k in required) or _blank((form.get("hidden") or {}).get("condition")):
        failed.append("Not every field is filled")
    probs = activation_problems(form, registry, settings)
    if any(p["field"] == "registry" for p in probs):
        failed.append("The condition uses names the orchestrator does not recognise")
    rows = conditions.reviewer_view(form, settings)
    disagree = sum(1 for r in rows if r["agent_expected"] != r["computed"])
    if disagree:
        failed.append(f"The agent's expected result differs from the computed one for {disagree} example"
                      + ("s" if disagree != 1 else ""))
    if _cant_enforce(form, registry):
        failed.append("An input is not available when the policy is checked")
    level = "High" if not failed else "Medium" if len(failed) == 1 else "Low"
    return {"level": level, "failed": failed}
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run the Step 2 command. Expected: PASS. If `test_how_enforced_reads_in_plain_words` fails only on wording, fix the code, not the test. The test's wording is the agreed text.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/readiness.py tests/agent_policy/test_readiness.py
git commit -m "feat(agent-policy): what Active requires, how it is enforced, extraction confidence"
```

---

### Task 6: Repository — IDs, versions, transitions

**Files:**
- Create: `src/repositories/agent_policy_repo.py`
- Test: `tests/agent_policy/test_repo_live.py` (live, `bp_testdb`)

**Interfaces:**
- Consumes: Task 1 tables, plus `compile_policy`, `contract.validate`, `readiness.activation_problems`, `readiness.extraction_confidence`, `load_settings` and `load_registry`.
- Produces:
  - `class StaleVersion(Exception)` → HTTP 409
  - `class NotReady(Exception)` with `.problems` → HTTP 422
  - `class NotFound(Exception)` → 404
  - `create_draft(conn, form, *, actor) -> dict` (`{"policyKey", "version"}`)
  - `save_version(conn, policy_key, form, *, base_version, intent, actor, change_note, document_text=None) -> dict`, where `intent` is `draft | activate`
  - `retire(conn, policy_key, *, base_version, actor, change_note) -> dict`
  - `get_policy(conn, policy_key) -> dict`, giving `{policyKey, status, liveVersion, latestVersion, areaName, versions:[{version, savedAs, savedBy, savedAt, changeNote, form, compiled, problems, confidence}]}`
  - `list_policies(conn) -> list[dict]`, one row per policy built from its latest version: `{policyKey, status, liveVersion, latestVersion, name, category, businessArea, subArea, outcome, source:{document, documentVersion, reference}, confidence, problemsCount}`
  - `live_documents(conn) -> list[dict]` (compiled JSON of each policy's live version)

- [ ] **Step 1: Write the failing live tests**

```python
"""Repository against bp_testdb. Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb.
Every test uses a fresh policy under the Unassigned (GEN) prefix; versions are immutable,
so nothing is cleaned up. That is the point of the table."""
import copy
import os

import pytest

from repositories import agent_policy_repo as repo
from tests.agent_policy.fixtures import FORM_EXAMPLE

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


@pytest.fixture
def conn():
    from services.db import get_conn
    with get_conn() as c:
        yield c


def test_draft_saves_with_only_a_name_and_gets_a_stable_id(conn):
    out = repo.create_draft(conn, {"name": "Name only"}, actor="test")
    assert out["version"] == 1 and out["policyKey"].startswith("GEN-")
    again = repo.save_version(conn, out["policyKey"], {"name": "Name only, renamed"},
                              base_version=1, intent="draft", actor="test", change_note="rename")
    assert again == {"policyKey": out["policyKey"], "version": 2}
    got = repo.get_policy(conn, out["policyKey"])
    assert [v["version"] for v in got["versions"]] == [1, 2] and got["status"] == "draft"


def test_ids_are_never_reused(conn):
    a = repo.create_draft(conn, {"name": "A"}, actor="test")["policyKey"]
    b = repo.create_draft(conn, {"name": "B"}, actor="test")["policyKey"]
    assert int(b.split("-")[1]) > int(a.split("-")[1])


def test_stale_base_version_is_refused(conn):
    key = repo.create_draft(conn, {"name": "Race"}, actor="test")["policyKey"]
    repo.save_version(conn, key, {"name": "Race 2"}, base_version=1, intent="draft", actor="a", change_note="")
    with pytest.raises(repo.StaleVersion):
        repo.save_version(conn, key, {"name": "Race 2b"}, base_version=1, intent="draft", actor="b", change_note="")


def test_activate_refused_with_every_problem(conn):
    key = repo.create_draft(conn, {"name": "Not ready"}, actor="test")["policyKey"]
    with pytest.raises(repo.NotReady) as err:
        repo.save_version(conn, key, {"name": "Not ready"}, base_version=1, intent="activate", actor="t", change_note="")
    assert len(err.value.problems) > 3
    assert repo.get_policy(conn, key)["latestVersion"] == 1   # nothing written on refusal


def test_live_stays_live_while_a_new_draft_exists(conn, monkeypatch):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["businessArea"] = None  # keep it in GEN for tests; activation needs an area, so patch readiness
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    key = repo.create_draft(conn, form, actor="test")["policyKey"]
    repo.save_version(conn, key, form, base_version=1, intent="activate", actor="t", change_note="go")
    repo.save_version(conn, key, {**form, "name": "edited"}, base_version=2, intent="draft", actor="t", change_note="edit")
    got = repo.get_policy(conn, key)
    assert got["status"] == "live" and got["liveVersion"] == 2 and got["latestVersion"] == 3
    assert key in {d["id"] for d in repo.live_documents(conn)}
    repo.retire(conn, key, base_version=3, actor="t", change_note="done")
    got = repo.get_policy(conn, key)
    assert got["status"] == "retired" and got["liveVersion"] is None and got["latestVersion"] == 4
    assert key not in {d["id"] for d in repo.live_documents(conn)}
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `set -a; . ./.env; set +a; PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_repo_live.py -v`
Expected: FAIL (`ModuleNotFoundError`). Confirm `.env` points at `bp_testdb` before running.

- [ ] **Step 3: Implement**

`src/repositories/agent_policy_repo.py`:

```python
"""Agent policies and their versions.

    draft --activate--> live --retire--> retired
      ^                  |  (a new draft beside a live version leaves the live one live)
      +----- save -------+

Every save inserts a version row; rows are immutable (DB trigger). get_conn() is AUTOCOMMIT,
so each transition switches autocommit off and commits once: the version row and the
pointer move together or not at all. base_version is the optimistic lock: a save based on
anything but the latest version is refused, never merged silently.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.agent_policy import contract, readiness
from services.agent_policy.compiler import compile_policy
from services.agent_policy.registry import load_registry
from services.agent_policy.settings import load_settings


class StaleVersion(Exception):
    pass


class NotReady(Exception):
    def __init__(self, problems: List[Dict[str, Any]]):
        super().__init__(f"{len(problems)} problem(s)")
        self.problems = problems


class NotFound(Exception):
    pass


_activation_problems = readiness.activation_problems
_contract_problems = contract.validate


def _area(cur, name: Optional[str]) -> Dict[str, Any]:
    if name:
        cur.execute("SELECT area_name, never_suggest FROM proc.bp_business_area WHERE area_name = %s", (name,))
        row = cur.fetchone()
        if row:
            return {"area_name": row[0], "never_suggest": row[1]}
    cur.execute("SELECT area_name, never_suggest FROM proc.bp_business_area WHERE is_unassigned")
    row = cur.fetchone()
    return {"area_name": row[0], "never_suggest": row[1]}


def _allocate(cur, area_name: str) -> str:
    cur.execute("UPDATE proc.bp_business_area SET last_number = last_number + 1 "
                "WHERE area_name = %s RETURNING id_prefix, last_number", (area_name,))
    prefix, number = cur.fetchone()
    return f"{prefix}-{number:04d}"


def _write_version(cur, key, version, saved_as, form, actor, note, document_text):
    settings = load_settings(cur.connection)
    registry = load_registry(cur.connection)
    area = _area(cur, form.get("businessArea"))
    status = {"draft": "draft", "live": "live", "retired": "retired"}[saved_as]
    compiled = compile_policy(form, policy_key=key, version=version, status=status,
                              settings=settings, never_suggest=area["never_suggest"])
    problems = _contract_problems(compiled, registry)
    confidence = readiness.extraction_confidence(form, document_text, registry, settings)
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_version (policy_key, version, saved_as, form_state, compiled,"
        " problems, confidence, change_note, saved_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (key, version, saved_as, json.dumps(form), json.dumps(compiled), json.dumps(problems),
         json.dumps(confidence) if confidence else None, note, actor))
    return compiled, problems


def _txn(conn):
    conn.autocommit = False
    return conn.cursor()


def create_draft(conn, form: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    cur = _txn(conn)
    try:
        area = _area(cur, form.get("businessArea"))
        key = _allocate(cur, area["area_name"])
        cur.execute("INSERT INTO proc.bp_agent_policy (policy_key, area_name, status, latest_version, created_by)"
                    " VALUES (%s,%s,'draft',1,%s)", (key, form.get("businessArea") and area["area_name"], actor))
        _write_version(cur, key, 1, "draft", form, actor, form.get("changeNote") or "", None)
        conn.commit()
        return {"policyKey": key, "version": 1}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def _lock(cur, key):
    cur.execute("SELECT status, live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s FOR UPDATE", (key,))
    row = cur.fetchone()
    if not row:
        raise NotFound(key)
    return {"status": row[0], "live_version": row[1], "latest_version": row[2]}


def save_version(conn, policy_key: str, form: Dict[str, Any], *, base_version: int, intent: str,
                 actor: str, change_note: str, document_text: Optional[str] = None) -> Dict[str, Any]:
    if intent not in ("draft", "activate"):
        raise ValueError(intent)
    cur = _txn(conn)
    try:
        row = _lock(cur, policy_key)
        if row["latest_version"] != base_version:
            raise StaleVersion(f"latest is {row['latest_version']}, edit was based on {base_version}")
        version = base_version + 1
        if intent == "activate":
            settings, registry = load_settings(conn), load_registry(conn)
            problems = _activation_problems(form, registry, settings)
            if problems:
                raise NotReady(problems)
            _, contract_problems = _write_version(cur, policy_key, version, "live", form, actor, change_note, document_text)
            if contract_problems:
                raise NotReady([{"field": "registry", "message": m, "routeTo": "administrator"} for m in contract_problems])
            cur.execute("UPDATE proc.bp_agent_policy SET status='live', live_version=%s, latest_version=%s,"
                        " area_name=COALESCE(%s, area_name) WHERE policy_key=%s",
                        (version, version, form.get("businessArea"), policy_key))
        else:
            _write_version(cur, policy_key, version, "draft", form, actor, change_note, document_text)
            cur.execute("UPDATE proc.bp_agent_policy SET latest_version=%s, area_name=COALESCE(%s, area_name)"
                        " WHERE policy_key=%s", (version, form.get("businessArea"), policy_key))
        conn.commit()
        return {"policyKey": policy_key, "version": version}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def retire(conn, policy_key: str, *, base_version: int, actor: str, change_note: str) -> Dict[str, Any]:
    cur = _txn(conn)
    try:
        row = _lock(cur, policy_key)
        if row["latest_version"] != base_version:
            raise StaleVersion(f"latest is {row['latest_version']}, retire was based on {base_version}")
        cur.execute("SELECT form_state FROM proc.bp_agent_policy_version WHERE policy_key=%s AND version=%s",
                    (policy_key, base_version))
        form = cur.fetchone()[0]
        form = json.loads(form) if isinstance(form, str) else form
        version = base_version + 1
        _write_version(cur, policy_key, version, "retired", form, actor, change_note, None)
        cur.execute("UPDATE proc.bp_agent_policy SET status='retired', live_version=NULL, latest_version=%s"
                    " WHERE policy_key=%s", (version, policy_key))
        conn.commit()
        return {"policyKey": policy_key, "version": version}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def get_policy(conn, policy_key: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT status, live_version, latest_version, area_name FROM proc.bp_agent_policy WHERE policy_key=%s", (policy_key,))
    head = cur.fetchone()
    if not head:
        raise NotFound(policy_key)
    cur.execute("SELECT version, saved_as, saved_by, saved_at, change_note, form_state, compiled, problems, confidence"
                " FROM proc.bp_agent_policy_version WHERE policy_key=%s ORDER BY version", (policy_key,))
    versions = [{"version": r[0], "savedAs": r[1], "savedBy": r[2], "savedAt": r[3].isoformat(),
                 "changeNote": r[4], "form": _j(r[5]), "compiled": _j(r[6]), "problems": _j(r[7]),
                 "confidence": _j(r[8])} for r in cur.fetchall()]
    return {"policyKey": policy_key, "status": head[0], "liveVersion": head[1], "latestVersion": head[2],
            "areaName": head[3], "versions": versions}


def list_policies(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT p.policy_key, p.status, p.live_version, p.latest_version, v.form_state, v.confidence, v.problems"
        " FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v"
        "   ON v.policy_key = p.policy_key AND v.version = p.latest_version ORDER BY p.policy_key")
    out = []
    for key, status, live, latest, form, conf, problems in cur.fetchall():
        form = _j(form) or {}
        src = form.get("source") or {}
        out.append({"policyKey": key, "status": status, "liveVersion": live, "latestVersion": latest,
                    "name": form.get("name"), "category": form.get("category"),
                    "businessArea": form.get("businessArea"), "subArea": form.get("subArea"),
                    "outcome": form.get("outcome"),
                    "source": {"document": src.get("document"), "documentVersion": src.get("documentVersion"),
                               "reference": src.get("reference")},
                    "confidence": _j(conf), "problemsCount": len(_j(problems) or [])})
    return out


def live_documents(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT v.compiled FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v"
                " ON v.policy_key = p.policy_key AND v.version = p.live_version"
                " WHERE p.status = 'live' ORDER BY p.policy_key")
    return [_j(r[0]) for r in cur.fetchall()]
```

When a version is activated, its compiled JSON is written with `status: "live"`. When the policy is later retired, that row is unchanged (rows are immutable), but `live_documents` reads only through `live_version`, which is now NULL. A retired policy therefore never reaches the feed.

- [ ] **Step 4: Run the tests to make sure they pass**

Run the Step 2 command. Expected: PASS. Then prove the optimistic lock: comment out the `StaleVersion` raise in `save_version`, and confirm `test_stale_base_version_is_refused` FAILS. Then restore it.

- [ ] **Step 5: Commit**

```bash
git add src/repositories/agent_policy_repo.py tests/agent_policy/test_repo_live.py
git commit -m "feat(agent-policy): versioned repository with stable ids, optimistic lock, live/draft/retired"
```

---

### Task 7: Backend endpoints, gateway trust, and the orchestrator feed

**Files:**
- Create: `src/api/routers/agent_policies.py`
- Modify: `src/api/main.py` (import + two `app.include_router(...)` lines next to `app.include_router(ws_router_mod.router)` at ~line 531, NOT in `_AUTHENTICATED_ROUTERS`)
- Modify: `src/services/actions.py` (add `"agent_policy.read": "read"`, `"agent_policy.write": "write"`, `"agent_policy.activate": "configure"`, `"agent_policy.admin": "configure"`)
- Test: `tests/agent_policy/test_router.py`

**Interfaces:**
- Consumes:
  - Task 6 repository;
  - `readiness.*`, `conditions.reviewer_view`, `load_registry`, `load_settings`, `contract.validate`;
  - `rbac.effective_role`, `rbac.role_rank`;
  - `agent_actions.record_action_or_fail`;
  - `src/api/auth.Principal`.
- Produces HTTP endpoints. All take headers `X-Gateway-Key`, `X-User-Sub`, `X-User-Email` and `X-User-Groups` (a JSON array):

| Method + path | Minimum role | Body | Returns |
|---|---|---|---|
| GET `/agent-policies` | Viewer | — | `{"policies": list_policies}` |
| GET `/agent-policies/{key}` | Viewer | — | `get_policy`. `compiled` is removed from versions unless the caller is Admin. |
| POST `/agent-policies` | Buyer | `{form}` | `{policyKey, version}` |
| POST `/agent-policies/{key}/versions` | Buyer (draft), Approver (activate) | `{form, baseVersion, intent, changeNote}` | `{policyKey, version}`; 409 if stale; 422 `{problems}` |
| POST `/agent-policies/{key}/retire` | Approver | `{baseVersion, changeNote}` | `{policyKey, version}` |
| POST `/agent-policies/preview` | Viewer | `{form, policyKey?, version?}` | `{examples, howEnforced, problems, compiled?}`. `compiled` is Admin only. |
| GET `/agent-policies/taxonomy` | Viewer | — | `{areas:[{areaName, prefix, subAreas, neverSuggest, secondReviewer}]}` |
| PUT `/agent-policies/taxonomy/{area}` | Admin | `{subAreas, neverSuggest, secondReviewer}` | updated row. The prefix and name can't change. |
| GET `/orchestrator/agent-policies/v2/live` | service key `X-Orchestrator-Key` | — | `{"feed": "hard-policy-feed/2", "generatedAt", "policies": [...], "refused": [{"id","problems"}]}` |

Environment: `AGENT_POLICY_GATEWAY_KEY` and `AGENT_POLICY_ORCHESTRATOR_KEY`. When either is unset, its endpoints return 503. They never fall open.

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

GOOD = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@x", "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    roles = {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer", "PROCWISE_PROCUMENT_BUYER_ANALYST": "Buyer"}
    monkeypatch.setattr(R, "_role_of", lambda principal: roles.get((principal.claims or {}).get("cognito:groups", [""])[0], "Viewer"))
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R, "load_registry", lambda conn=None: REGISTRY)
    monkeypatch.setattr(R, "load_settings", lambda conn=None: SETTINGS)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    app = FastAPI(); app.include_router(R.router); app.include_router(R.orchestrator_router)
    c = TestClient(app); c.audits = audits
    return c


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def test_missing_or_wrong_gateway_key_is_refused(client):
    assert client.get("/agent-policies").status_code == 401
    assert client.get("/agent-policies", headers={**GOOD, "X-Gateway-Key": "nope"}).status_code == 401


def test_unset_key_env_refuses_everything(client, monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_GATEWAY_KEY")
    assert client.get("/agent-policies", headers=GOOD).status_code == 503


def test_viewer_cannot_create(client, monkeypatch):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    assert client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr).status_code == 403
    assert client.audits and client.audits[-1]["status"] == "denied"


def test_buyer_creates_and_write_is_audited(client, monkeypatch):
    monkeypatch.setattr(R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    r = client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr)
    assert r.status_code == 200 and r.json() == {"policyKey": "GEN-0001", "version": 1}
    assert client.audits[-1]["action_type"] == "agent_policy.write" and client.audits[-1]["status"] == "allowed"


def test_buyer_cannot_activate(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    body = {"form": FORM_EXAMPLE, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=hdr).status_code == 403


def test_not_ready_returns_every_problem(client, monkeypatch):
    def refuse(*a, **k): raise R.repo.NotReady([{"field": "owner", "message": "Owner is required."}])
    monkeypatch.setattr(R.repo, "save_version", refuse)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    r = client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "owner"


def test_stale_save_is_409(client, monkeypatch):
    def stale(*a, **k): raise R.repo.StaleVersion("latest is 3")
    monkeypatch.setattr(R.repo, "save_version", stale)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "draft", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD).status_code == 409


def test_preview_hides_json_from_non_admins(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=hdr).json()
    assert "compiled" not in r and r["examples"][0]["label"] == "A person decides"
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert r["compiled"]["schema"] == "hard-policy/2"


def test_feed_needs_its_own_key(client):
    assert client.get("/orchestrator/agent-policies/v2/live").status_code == 401
    assert client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "k1"}).status_code == 401


def test_feed_refuses_policy_whose_tool_left_registry(client, monkeypatch):
    from services.agent_policy.compiler import compile_policy
    good = dict(FORM_EXAMPLE, checked={"by": "u", "at": "2026-10-08T00:00:00Z"})
    bad = json.loads(json.dumps(good)); bad["hidden"]["condition"]["all"][0]["value"] = ["gone.tool"]
    docs = [compile_policy(good, policy_key="FIN-0001", version=1, status="live", settings=SETTINGS, never_suggest=False),
            compile_policy(bad, policy_key="FIN-0002", version=1, status="live", settings=SETTINGS, never_suggest=False)]
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: docs)
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"}).json()
    assert [p["id"] for p in r["policies"]] == ["FIN-0001"]
    assert r["refused"][0]["id"] == "FIN-0002" and r["feed"] == "hard-policy-feed/2"
```

The `FORM_EXAMPLE` fixture names `refund.issue`, and the fixture `REGISTRY` knows it. So `FIN-0001` is valid in this test.

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy/test_router.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

`src/api/routers/agent_policies.py`:

```python
"""Agent policies: screen endpoints (via the gateway only) and the orchestrator feed.

WHY A GATEWAY KEY (design §3.2, ruling C 2026-10-08): this service's own sign-in check
(ASK_AUTH_MODE) is off in development and unconfirmed in production, so a direct browser
call could not be shown to carry the gateway's checks. These routes therefore trust ONLY
the gateway: it verifies the Cognito token and the role, then forwards the verified
identity with a shared key. No key, wrong key, or unset key -> refused. The role is
re-derived here from the forwarded groups through the same role table every other gate
uses, and every write is audited before it returns.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth import Principal
from repositories import agent_policy_repo as repo
from services import agent_actions, rbac
from services.agent_policy import conditions, contract, readiness
from services.agent_policy.compiler import compile_policy
from services.agent_policy.registry import load_registry
from services.agent_policy.settings import load_settings
from services.db import get_conn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent-policies", tags=["Agent policies"])
orchestrator_router = APIRouter(prefix="/orchestrator/agent-policies", tags=["Agent policies"])

_RANK = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}


def _conn():
    return get_conn()


def _key_ok(given: Optional[str], env: str) -> None:
    expected = os.getenv(env)
    if not expected:
        raise HTTPException(status_code=503, detail="this endpoint is not configured")
    if not given or not hmac.compare_digest(given, expected):
        raise HTTPException(status_code=401, detail="not accepted")


def _role_of(principal: Principal) -> str:
    return rbac.effective_role(principal)


def gateway_principal(request: Request) -> Principal:
    _key_ok(request.headers.get("x-gateway-key"), "AGENT_POLICY_GATEWAY_KEY")
    sub = (request.headers.get("x-user-sub") or "").strip()
    if not sub:
        raise HTTPException(status_code=401, detail="not accepted")
    try:
        groups = json.loads(request.headers.get("x-user-groups") or "[]")
    except ValueError:
        groups = []
    return Principal(subject=sub, email=request.headers.get("x-user-email"),
                     claims={"cognito:groups": [str(g) for g in groups if isinstance(g, str)]})


def _require(principal: Principal, minimum: str, action: str, details: Dict[str, Any]) -> str:
    role = _role_of(principal)
    allowed = _RANK.get(role, 0) >= _RANK[minimum]
    if action != "agent_policy.read":
        agent_actions.record_action_or_fail(
            phase="authorize", action_type=action, agent="agent_policy_api",
            status="allowed" if allowed else "denied",
            summary=f"{role} {'may' if allowed else 'may not'} {action}",
            details={**details, "principal": principal.subject, "role": role, "minimum": minimum})
    if not allowed:
        raise HTTPException(status_code=403, detail=f"{action} needs the {minimum} role or higher")
    return role


class CreateBody(BaseModel):
    form: Dict[str, Any]


class VersionBody(BaseModel):
    form: Dict[str, Any]
    baseVersion: int
    intent: str = Field(pattern="^(draft|activate)$")
    changeNote: str = ""


class RetireBody(BaseModel):
    baseVersion: int
    changeNote: str = ""


class PreviewBody(BaseModel):
    form: Dict[str, Any]
    policyKey: Optional[str] = None
    version: Optional[int] = None


class AreaBody(BaseModel):
    subAreas: List[str]
    neverSuggest: bool
    secondReviewer: bool


@router.get("")
def list_policies(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"policies": repo.list_policies(conn)}


@router.get("/taxonomy")
def taxonomy(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT area_name, id_prefix, sub_areas, never_suggest, second_reviewer, is_unassigned"
                    " FROM proc.bp_business_area ORDER BY is_unassigned, area_name")
        return {"areas": [{"areaName": r[0], "prefix": r[1], "subAreas": list(r[2]), "neverSuggest": r[3],
                           "secondReviewer": r[4], "unassigned": r[5]} for r in cur.fetchall()]}


@router.put("/taxonomy/{area}")
def update_area(area: str, body: AreaBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Admin", "agent_policy.admin", {"area": area})
    subs = [s.strip() for s in body.subAreas if s.strip()]
    if "General" not in subs:
        subs.insert(0, "General")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE proc.bp_business_area SET sub_areas=%s, never_suggest=%s, second_reviewer=%s,"
                    " last_modified_by=%s, last_modified_at=now() WHERE area_name=%s RETURNING area_name",
                    (subs, body.neverSuggest, body.secondReviewer, p.subject, area))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="no such business area")
    return {"areaName": area, "subAreas": subs, "neverSuggest": body.neverSuggest, "secondReviewer": body.secondReviewer}


@router.post("/preview")
def preview(body: PreviewBody, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {})
    registry, settings = load_registry(), load_settings()
    out: Dict[str, Any] = {"examples": conditions.reviewer_view(body.form, settings),
                           "howEnforced": readiness.how_enforced(body.form, registry, settings),
                           "problems": readiness.activation_problems(body.form, registry, settings)}
    if role == "Admin":
        doc = compile_policy(body.form, policy_key=body.policyKey or "GEN-0000", version=body.version or 1,
                             status="draft", settings=settings, never_suggest=False)
        out["compiled"] = doc
        out["compiledProblems"] = contract.validate(doc, registry)
    return out


@router.get("/{key}")
def get_one(key: str, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {"policy": key})
    with _conn() as conn:
        try:
            got = repo.get_policy(conn, key)
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")
    if role != "Admin":
        for v in got["versions"]:
            v.pop("compiled", None)
    return got


@router.post("")
def create(body: CreateBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Buyer", "agent_policy.write", {"intent": "create"})
    with _conn() as conn:
        return repo.create_draft(conn, body.form, actor=p.subject)


@router.post("/{key}/versions")
def save(key: str, body: VersionBody, p: Principal = Depends(gateway_principal)):
    minimum, action = ("Approver", "agent_policy.activate") if body.intent == "activate" else ("Buyer", "agent_policy.write")
    _require(p, minimum, action, {"policy": key, "intent": body.intent, "baseVersion": body.baseVersion})
    with _conn() as conn:
        try:
            return repo.save_version(conn, key, body.form, base_version=body.baseVersion, intent=body.intent,
                                     actor=p.subject, change_note=body.changeNote)
        except repo.StaleVersion as exc:
            raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
        except repo.NotReady as exc:
            raise HTTPException(status_code=422, detail={"problems": exc.problems})
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")


@router.post("/{key}/retire")
def retire(key: str, body: RetireBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Approver", "agent_policy.activate", {"policy": key, "intent": "retire"})
    with _conn() as conn:
        try:
            return repo.retire(conn, key, base_version=body.baseVersion, actor=p.subject, change_note=body.changeNote)
        except repo.StaleVersion as exc:
            raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")


@orchestrator_router.get("/v2/live")
def live_feed(request: Request):
    _key_ok(request.headers.get("x-orchestrator-key"), "AGENT_POLICY_ORCHESTRATOR_KEY")
    registry = load_registry()
    with _conn() as conn:
        docs = repo.live_documents(conn)
    good, refused = [], []
    for doc in docs:
        problems = contract.validate(doc, registry)
        (refused.append({"id": doc.get("id"), "problems": problems}) if problems else good.append(doc))
    if refused:
        logger.warning("agent-policy feed refused %d live policies: %s", len(refused), [r["id"] for r in refused])
    return {"feed": "hard-policy-feed/2", "generatedAt": datetime.now(timezone.utc).isoformat(),
            "policies": good, "refused": refused}
```

The test monkeypatches `R.repo.save_version` and similar names, so the router must call them through the `repo` module, as shown. It must not import them by name.

`src/api/main.py`: add `from api.routers import agent_policies as agent_policies_router`. Match the import style the file already uses for routers (check with `grep -n "routers import" src/api/main.py | head`). Then, after `app.include_router(ws_router_mod.router)`, add:

```python
# Agent policies trust only the gateway's key + forwarded identity (design 2026-10-08 §3.2),
# so they are NOT in _AUTHENTICATED_ROUTERS: the browser never calls them directly.
app.include_router(agent_policies_router.router)
app.include_router(agent_policies_router.orchestrator_router)
```

`.env`: add `AGENT_POLICY_GATEWAY_KEY=<random 32 bytes hex>` and `AGENT_POLICY_ORCHESTRATOR_KEY=<different random>`. Generate them with `./venv/bin/python -c "import secrets;print(secrets.token_hex(32))"`. Never commit `.env`.

- [ ] **Step 4: Run the tests, and the existing gate tests, to make sure nothing regressed**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy -v`, then `./venv/bin/python -m pytest tests/guardrails tests/governance tests/approvals tests/engines -q`.
Expected: the new tests PASS, and the existing suites show the same counts as before. Capture the "before" counts first with `git stash`-free checking: run the existing suites before Step 3.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/agent_policies.py src/api/main.py src/services/actions.py tests/agent_policy/test_router.py
git commit -m "feat(agent-policy): endpoints behind the gateway key, and a versioned orchestrator feed"
```

---

### Task 8: Gateway module `agent-policy`

**Files** (repo `beyond-procwaise-Api/beyond_procwaise_api`, branch `spendiq-ui`; stage only these hunks, because another session keeps edits in `spendiq.service.ts`):
- Create: `src/modules/agent-policy/agent-policy.service.ts`
- Create: `src/modules/agent-policy/agent-policy.controller.ts`
- Create: `src/modules/agent-policy/agent-policy.module.ts`
- Create: `src/modules/agent-policy/agent-policy.yml`
- Create: `src/modules/agent-policy/agent-policy.controller.spec.ts`
- Modify: `src/app.module.ts` (import + `AgentPolicyModule` in `imports`)
- Modify: `serverless.yml` (include `agent-policy.yml` the way `policy.yml` is included; check with `grep -n "policy.yml" serverless.yml`)

**Interfaces:**
- Consumes: `CognitoGuard` (`src/auth/guards/cognito.guard.ts`), `productRole` (`src/modules/user/user-roles.ts`; check its exact signature with `grep -n "export function productRole" -A8 src/modules/user/user-roles.ts`), and the Task 7 endpoints.
- Produces the same paths as Task 7 under the gateway: `/agent-policies`, `/agent-policies/taxonomy`, `/agent-policies/preview`, `/agent-policies/:key`, `/agent-policies/:key/versions`, `/agent-policies/:key/retire`, and `PUT /agent-policies/taxonomy/:area`.
- Environment: `BP_BACKEND_URL` and `AGENT_POLICY_GATEWAY_KEY`. When either is missing, the gateway returns 503.

- [ ] **Step 1: Write the failing jest spec**

```ts
import { AgentPolicyController } from './agent-policy.controller';
import { AgentPolicyService } from './agent-policy.service';

const req = (groups: string[]) => ({ user: { sub: 'u1', email: 'u1@x', 'cognito:groups': groups } });

describe('AgentPolicyController', () => {
  const env = process.env;
  let fetchMock: jest.Mock;
  beforeEach(() => {
    process.env = { ...env, BP_BACKEND_URL: 'http://py', AGENT_POLICY_GATEWAY_KEY: 'k1' };
    fetchMock = jest.fn().mockResolvedValue({ status: 200, json: async () => ({ ok: true }) });
    (global as any).fetch = fetchMock;
  });
  afterEach(() => { process.env = env; });

  const ctl = () => new AgentPolicyController(new AgentPolicyService());

  it('forwards the verified identity and the key, never the browser token', async () => {
    await ctl().list(req(['PROCWISE_VIEWER']));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('http://py/agent-policies');
    expect(init.headers['X-Gateway-Key']).toBe('k1');
    expect(init.headers['X-User-Sub']).toBe('u1');
    expect(JSON.parse(init.headers['X-User-Groups'])).toEqual(['PROCWISE_VIEWER']);
    expect(init.headers.Authorization).toBeUndefined();
  });

  it('refuses a viewer creating a policy before calling the backend', async () => {
    await expect(ctl().create(req(['PROCWISE_VIEWER']), { form: { name: 'x' } })).rejects.toMatchObject({ status: 403 });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses a buyer activating', async () => {
    await expect(ctl().saveVersion(req(['PROCWISE_PROCUMENT_BUYER_ANALYST']), 'FIN-0001',
      { form: {}, baseVersion: 1, intent: 'activate', changeNote: '' })).rejects.toMatchObject({ status: 403 });
  });

  it('passes the backend status through (409 stale, 422 problems)', async () => {
    fetchMock.mockResolvedValue({ status: 422, json: async () => ({ detail: { problems: [{ field: 'owner' }] } }) });
    await expect(ctl().saveVersion(req(['PROCWISE_ADMIN']), 'FIN-0001',
      { form: {}, baseVersion: 1, intent: 'activate', changeNote: '' })).rejects.toMatchObject({ status: 422 });
  });

  it('is 503 when not configured', async () => {
    delete process.env.AGENT_POLICY_GATEWAY_KEY;
    await expect(ctl().list(req(['PROCWISE_ADMIN']))).rejects.toMatchObject({ status: 503 });
  });

  it('rejects a path key that is not a policy id', async () => {
    await expect(ctl().getOne(req(['PROCWISE_ADMIN']), '../system')).rejects.toMatchObject({ status: 400 });
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `npx jest src/modules/agent-policy`
Expected: FAIL (`Cannot find module`).

- [ ] **Step 3: Implement**

`agent-policy.service.ts`:

```ts
import { HttpException, Injectable } from '@nestjs/common';

/** Forwards to BP_Backend with the gateway key and the identity CognitoGuard verified.
 *  The browser's own token is NOT forwarded: the backend trusts the key, not the caller. */
@Injectable()
export class AgentPolicyService {
  async forward(user: any, method: string, path: string, body?: unknown): Promise<any> {
    const base = process.env.BP_BACKEND_URL;
    const key = process.env.AGENT_POLICY_GATEWAY_KEY;
    if (!base || !key) throw new HttpException('agent policies are not configured', 503);
    const groups = user?.['cognito:groups'];
    const res = await fetch(`${base.replace(/\/$/, '')}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-Gateway-Key': key,
        'X-User-Sub': String(user?.sub ?? ''),
        'X-User-Email': String(user?.email ?? ''),
        'X-User-Groups': JSON.stringify(Array.isArray(groups) ? groups : groups ? [groups] : []),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (res.status >= 400) throw new HttpException(data?.detail ?? data, res.status);
    return data;
  }
}
```

`agent-policy.controller.ts`:

```ts
import { BadRequestException, Body, Controller, ForbiddenException, Get, Param, Post, Put, Req, UseGuards } from '@nestjs/common';
import { CognitoGuard } from 'src/auth/guards/cognito.guard';
import { productRole } from '../user/user-roles';
import { AgentPolicyService } from './agent-policy.service';

const RANK = { Viewer: 1, Buyer: 2, Approver: 3, Admin: 4 } as const;
type Role = keyof typeof RANK;
const KEY = /^[A-Z]{3}-[0-9]{4,}$/;

function need(req: any, minimum: Role): void {
  const role = productRole(req.user?.['cognito:groups']) as Role;
  if ((RANK[role] ?? 0) < RANK[minimum]) throw new ForbiddenException(`needs the ${minimum} role or higher`);
}
function key(k: string): string {
  if (!KEY.test(k)) throw new BadRequestException('not a policy id');
  return k;
}

@UseGuards(CognitoGuard)
@Controller('agent-policies')
export class AgentPolicyController {
  constructor(private readonly svc: AgentPolicyService) {}

  @Get() list(@Req() req) { need(req, 'Viewer'); return this.svc.forward(req.user, 'GET', '/agent-policies'); }
  @Get('taxonomy') taxonomy(@Req() req) { need(req, 'Viewer'); return this.svc.forward(req.user, 'GET', '/agent-policies/taxonomy'); }
  @Put('taxonomy/:area') updateArea(@Req() req, @Param('area') area: string, @Body() body: any) {
    need(req, 'Admin'); return this.svc.forward(req.user, 'PUT', `/agent-policies/taxonomy/${encodeURIComponent(area)}`, body);
  }
  @Post('preview') preview(@Req() req, @Body() body: any) { need(req, 'Viewer'); return this.svc.forward(req.user, 'POST', '/agent-policies/preview', body); }
  @Get(':key') getOne(@Req() req, @Param('key') k: string) { need(req, 'Viewer'); return this.svc.forward(req.user, 'GET', `/agent-policies/${key(k)}`); }
  @Post() create(@Req() req, @Body() body: any) { need(req, 'Buyer'); return this.svc.forward(req.user, 'POST', '/agent-policies', body); }
  @Post(':key/versions') saveVersion(@Req() req, @Param('key') k: string, @Body() body: any) {
    need(req, body?.intent === 'activate' ? 'Approver' : 'Buyer');
    return this.svc.forward(req.user, 'POST', `/agent-policies/${key(k)}/versions`, body);
  }
  @Post(':key/retire') retire(@Req() req, @Param('key') k: string, @Body() body: any) {
    need(req, 'Approver'); return this.svc.forward(req.user, 'POST', `/agent-policies/${key(k)}/retire`, body);
  }
}
```

In the spec, the controller methods are called directly, so the `need()` and `key()` checks run inside the methods; the spec doesn't depend on guards.

The spec calls `getOne(req, '../system')` and expects 400 before any backend call. So `need()` runs first (an Admin passes), and then `key()` throws.

`agent-policy.module.ts`:

```ts
import { Module } from '@nestjs/common';
import { AgentPolicyController } from './agent-policy.controller';
import { AgentPolicyService } from './agent-policy.service';

@Module({ controllers: [AgentPolicyController], providers: [AgentPolicyService] })
export class AgentPolicyModule {}
```

`agent-policy.yml`: copy the `getAllPolicies` block from `src/modules/policy/policy.yml` (same `cors` origins and headers, same `authorizer`), once per route in the table above. Use `path: agent-policies`, `agent-policies/taxonomy`, `agent-policies/taxonomy/{area}`, `agent-policies/preview`, `agent-policies/{key}`, `agent-policies/{key}/versions`, `agent-policies/{key}/retire`, with matching methods.

`src/app.module.ts`: add `import { AgentPolicyModule } from './modules/agent-policy/agent-policy.module';` and add `AgentPolicyModule` to the `imports` array next to `PolicyModule`.

Gateway `.env` (local, not committed): set `BP_BACKEND_URL=http://localhost:8000`, and set `AGENT_POLICY_GATEWAY_KEY` to the same value as in BP_Backend `.env`.

- [ ] **Step 4: Run the spec and the whole gateway suite**

Run: `npx jest src/modules/agent-policy`, then `npx jest`.
Expected: the new spec PASSES, and the full-suite counts match a run taken before Step 3.

- [ ] **Step 5: Commit** (only these paths; check `git status` and `git diff --cached --stat` before committing)

```bash
git add src/modules/agent-policy src/app.module.ts serverless.yml
git diff --cached --stat
git commit -m "feat(agent-policy): gateway routes that check sign-in and role, then forward with the service key"
```

---

### Task 9: UI pure module — form model, inventory, CSV

**Files** (repo `beyond_procwise_ui`, branch `spendiq-ui`):
- Create: `src/modules/SpendIQ/agentPolicy/model.js`
- Create: `src/modules/SpendIQ/agentPolicy/inventory.js`
- Test: `src/modules/SpendIQ/agentPolicy/model.test.js`, `src/modules/SpendIQ/agentPolicy/inventory.test.js`

**Interfaces:**
- Produces, in `model.js`:
  - `emptyForm()`
  - `switchOutcome(form, outcome)`: returns a new form. It keeps the other outcomes' fields in a private `_stash` so switching back restores what the user typed, and `toSaveable()` strips `_stash`.
  - `toSaveable(form)`
  - `editClearsConfirmation(oldForm, newForm) -> bool`, which mirrors the backend's `confirmation_cleared`
  - `flipExample(form, i)`: returns a new form with `checked: null`
  - `confirm(form, subject, nowIso)`
  - `responseTimeLabel(form, companyDefault)`, e.g. `'Response time: company default, 4 hours'`
  - `LEGACY_FIELD_MAP`
  - `OUTCOMES = [{key:'approve', label:'Needs approval: a person decides'}, {key:'block', label:'Not allowed'}, {key:'notify', label:'Notify only'}]`
  - `STATUS_LABEL = {draft:'Draft', live:'Active', retired:'Retired'}`
- Produces, in `inventory.js`:
  - `groupBySource(rows) -> [{document, policies:[...]}]` ("No source document" last)
  - `groupByArea(rows) -> [{area, subAreas:[{subArea, policies}]}]`
  - `csvCell(v) -> string`
  - `inventoryCsv(rows) -> string`

- [ ] **Step 1: Write the failing vitest tests**

```js
// model.test.js
import { describe, it, expect } from 'vitest';
import { emptyForm, switchOutcome, toSaveable, editClearsConfirmation, flipExample, confirm, responseTimeLabel, LEGACY_FIELD_MAP } from './model';

const approve = () => ({ ...emptyForm(), name: 'R', outcome: 'approve', deciders: ['Finance Manager', 'CFO'], responseTime: 'PT6H',
  checked: { by: 'u', at: 't' }, examples: [{ input: { 'args.amount': 501 }, agentExpected: 'approve', flipped: false }] });

describe('agent policy form model', () => {
  it('switchOutcome round trip keeps what was typed but saves only the current outcome', () => {
    let f = switchOutcome(approve(), 'block');
    expect(toSaveable(f).deciders).toEqual([]);
    expect(toSaveable(f).responseTime).toBeNull();
    f = switchOutcome(f, 'approve');
    expect(toSaveable(f).deciders).toEqual(['Finance Manager', 'CFO']);
    expect(toSaveable(f).responseTime).toBe('PT6H');
    expect(toSaveable(f)._stash).toBeUndefined();
  });
  it('switching outcome clears the confirmation', () => {
    expect(switchOutcome(approve(), 'block').checked).toBeNull();
  });
  it('editing the situation clears the confirmation; owner does not', () => {
    const a = approve();
    expect(editClearsConfirmation(a, { ...a, situation: 'changed' })).toBe(true);
    expect(editClearsConfirmation(a, { ...a, owner: 'CFO' })).toBe(false);
  });
  it('flipping an example marks it and clears the confirmation', () => {
    const f = flipExample(approve(), 0);
    expect(f.examples[0].flipped).toBe(true);
    expect(f.checked).toBeNull();
  });
  it('confirm records who and when', () => {
    expect(confirm({ ...approve(), checked: null }, 'user_8841', '2026-10-08T09:14:00Z').checked)
      .toEqual({ by: 'user_8841', at: '2026-10-08T09:14:00Z' });
  });
  it('response time label', () => {
    expect(responseTimeLabel({ responseTime: null }, 'PT4H')).toBe('Response time: company default, 4 hours');
    expect(responseTimeLabel({ responseTime: 'PT6H' }, 'PT4H')).toBe('Response time: 6 hours');
  });
  it('every legacy field has a destination', () => {
    expect(Object.keys(LEGACY_FIELD_MAP).sort()).toEqual(
      ['desc', 'effective_from', 'enforcement', 'name', 'note', 'owner', 'review_by', 'scope', 'status', 'type'].sort());
  });
});
```

```js
// inventory.test.js
import { describe, it, expect } from 'vitest';
import { groupBySource, groupByArea, csvCell, inventoryCsv } from './inventory';

const rows = [
  { policyKey: 'FIN-0012', name: 'Refund over $500', businessArea: 'Finance', subArea: 'Refunds and credits', outcome: 'approve', status: 'live', source: { document: 'Finance Payments Policy', reference: '1.1' } },
  { policyKey: 'FIN-0013', name: 'Refund over $10,000', businessArea: 'Finance', subArea: 'Refunds and credits', outcome: 'block', status: 'draft', source: { document: 'Finance Payments Policy', reference: '1.1' } },
  { policyKey: 'GEN-0001', name: '=HYPERLINK("x")', businessArea: null, subArea: null, outcome: null, status: 'draft', source: { document: null } },
];

describe('inventory', () => {
  it('groups tiered policies under one source, unsourced last', () => {
    const g = groupBySource(rows);
    expect(g.map((x) => x.document)).toEqual(['Finance Payments Policy', 'No source document']);
    expect(g[0].policies.map((p) => p.policyKey)).toEqual(['FIN-0012', 'FIN-0013']);
  });
  it('groups by area and sub-area', () => {
    const g = groupByArea(rows);
    expect(g[0]).toMatchObject({ area: 'Finance', subAreas: [{ subArea: 'Refunds and credits' }] });
    expect(g[g.length - 1].area).toBe('Unassigned');
  });
  it.each(['=1+1', '+1', '-1', '@SUM(A1)', '\t=1', '\r=1'])('neutralises %j', (v) => {
    expect(csvCell(v).replace(/^"|"$/g, '').startsWith("'")).toBe(true);
  });
  it('neutralises leading whitespace before a formula', () => {
    expect(csvCell('  =1+1').replace(/^"|"$/g, '').startsWith("'")).toBe(true);
  });
  it('quotes commas and quotes', () => {
    expect(csvCell('a,"b"')).toBe('"a,""b"""');
  });
  it('csv has a header and one line per policy', () => {
    const lines = inventoryCsv(rows).trim().split('\r\n');
    expect(lines[0]).toBe('Policy ID,Name,Business area,Sub-area,What happens,Status,Source document,Section');
    expect(lines).toHaveLength(4);
    expect(lines[3]).toContain(`"'=HYPERLINK(""x"")"`);
  });
});
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `npx vitest run src/modules/SpendIQ/agentPolicy`
Expected: FAIL (cannot resolve `./model`).

- [ ] **Step 3: Implement**

`model.js`:

```js
/* Agent policy form state. Pure: no DOM, no fetch. Keys match BP_Backend's form_state
 * (services/agent_policy, fixtures FORM_EXAMPLE) exactly; the backend is the authority on
 * results, readiness and JSON, this module only keeps the form honest while it is edited. */

export const OUTCOMES = [
  { key: 'approve', label: 'Needs approval: a person decides' },
  { key: 'block', label: 'Not allowed' },
  { key: 'notify', label: 'Notify only' },
];
export const STATUS_LABEL = { draft: 'Draft', live: 'Active', retired: 'Retired' };
export const RESULT_LABEL = { approve: 'A person decides', block: 'Blocked', notify: 'Someone is told', none: 'Nothing happens' };

/* Where each field of the old Policies form went (brief §3.1). Shown to nobody; asserted
 * by test so a field cannot be silently dropped. */
export const LEGACY_FIELD_MAP = {
  name: 'name', type: 'category', desc: 'situation', enforcement: 'outcome', scope: 'limit',
  owner: 'owner', effective_from: 'effectiveFrom', review_by: 'reviewBy', status: 'status', note: 'changeNote',
};

const OUTCOME_FIELDS = { approve: ['deciders', 'responseTime'], block: ['notify'], notify: ['notify'] };
const BLANK = { deciders: [], responseTime: null, notify: [] };
const CONFIRM_KEYS = ['situation', 'outcome', 'hidden', 'messageForPerson', 'deciders', 'notify'];

export function emptyForm() {
  return {
    name: '', category: 'Approval', businessArea: null, subArea: null, situation: '', source: null,
    outcome: null, outcomeBecause: null, deciders: [], responseTime: null, notify: [],
    limit: { on: false, text: '' }, owner: '', effectiveFrom: null, reviewBy: null,
    messageForAgent: '', messageForPerson: '', hidden: null, examples: [], checked: null, changeNote: '',
  };
}

export function switchOutcome(form, outcome) {
  const stash = { ...(form._stash || {}) };
  if (form.outcome) stash[form.outcome] = Object.fromEntries((OUTCOME_FIELDS[form.outcome] || []).map((k) => [k, form[k]]));
  const restored = stash[outcome] || {};
  const next = { ...form, ...BLANK, ...restored, outcome, checked: null, _stash: stash };
  return next;
}

export function toSaveable(form) {
  const { _stash, ...rest } = form;
  const keep = new Set(OUTCOME_FIELDS[rest.outcome] || []);
  for (const k of Object.keys(BLANK)) if (!keep.has(k)) rest[k] = BLANK[k];
  return rest;
}

export function editClearsConfirmation(oldForm, newForm) {
  if (CONFIRM_KEYS.some((k) => JSON.stringify(oldForm[k] ?? null) !== JSON.stringify(newForm[k] ?? null))) return true;
  const sig = (f) => JSON.stringify((f.examples || []).map((e) => [e.input, !!e.flipped]));
  return sig(oldForm) !== sig(newForm);
}

export function flipExample(form, i) {
  const examples = (form.examples || []).map((e, j) => (j === i ? { ...e, flipped: !e.flipped } : e));
  return { ...form, examples, checked: null };
}

export function confirm(form, subject, nowIso) {
  return { ...form, checked: { by: subject, at: nowIso } };
}

function hours(duration) {
  const m = /^PT(\d+)H$/.exec(duration || '');
  return m ? `${m[1]} hour${m[1] === '1' ? '' : 's'}` : duration;
}

export function responseTimeLabel(form, companyDefault) {
  return form.responseTime
    ? `Response time: ${hours(form.responseTime)}`
    : `Response time: company default, ${hours(companyDefault)}`;
}
```

`inventory.js`:

```js
/* Inventory views and the CSV export. The export is guarded against formula injection:
 * a cell that a spreadsheet would read as a formula (=, +, -, @, or a leading tab/CR,
 * even after leading spaces) gets a leading apostrophe. */

const UNSOURCED = 'No source document';
const UNASSIGNED = 'Unassigned';

export function groupBySource(rows) {
  const map = new Map();
  for (const r of rows || []) {
    const doc = (r.source && r.source.document) || UNSOURCED;
    if (!map.has(doc)) map.set(doc, []);
    map.get(doc).push(r);
  }
  const groups = [...map.entries()].map(([document, policies]) => ({
    document,
    policies: policies.slice().sort((a, b) =>
      String(a.source?.reference || '').localeCompare(String(b.source?.reference || ''), undefined, { numeric: true })
      || a.policyKey.localeCompare(b.policyKey)),
  }));
  return groups.sort((a, b) => (a.document === UNSOURCED) - (b.document === UNSOURCED) || a.document.localeCompare(b.document));
}

export function groupByArea(rows) {
  const areas = new Map();
  for (const r of rows || []) {
    const area = r.businessArea || UNASSIGNED;
    const sub = r.subArea || 'General';
    if (!areas.has(area)) areas.set(area, new Map());
    const subs = areas.get(area);
    if (!subs.has(sub)) subs.set(sub, []);
    subs.get(sub).push(r);
  }
  return [...areas.entries()]
    .map(([area, subs]) => ({ area, subAreas: [...subs.entries()].map(([subArea, policies]) => ({ subArea, policies })) }))
    .sort((a, b) => (a.area === UNASSIGNED) - (b.area === UNASSIGNED) || a.area.localeCompare(b.area));
}

export function csvCell(v) {
  let s = v == null ? '' : String(v);
  if (/^[\s]*[=+\-@]/.test(s) || /^[\t\r]/.test(s)) s = `'${s}`;
  return `"${s.replace(/"/g, '""')}"`;
}

const OUTCOME = { approve: 'Needs approval', block: 'Not allowed', notify: 'Notify only' };
const STATUS = { draft: 'Draft', live: 'Active', retired: 'Retired' };

export function inventoryCsv(rows) {
  const head = 'Policy ID,Name,Business area,Sub-area,What happens,Status,Source document,Section';
  const body = groupByArea(rows).flatMap((a) => a.subAreas.flatMap((s) => s.policies.map((p) => [
    p.policyKey, p.name, a.area, s.subArea, OUTCOME[p.outcome] || '', STATUS[p.status] || p.status,
    p.source?.document || '', p.source?.reference || '',
  ].map(csvCell).join(','))));
  return [head, ...body].join('\r\n') + '\r\n';
}
```

The header row is written as plain text, without `csvCell`, so the test's exact header string matches. Every data cell goes through `csvCell`.

- [ ] **Step 4: Run them to make sure they pass, and prove the CSV guard**

Run: `npx vitest run src/modules/SpendIQ/agentPolicy`
Expected: PASS. Then remove the `s = \`'${s}\`` assignment, confirm the `neutralises` tests go red, and restore it.

- [ ] **Step 5: Commit** (stage only these paths)

```bash
git add src/modules/SpendIQ/agentPolicy/model.js src/modules/SpendIQ/agentPolicy/inventory.js src/modules/SpendIQ/agentPolicy/model.test.js src/modules/SpendIQ/agentPolicy/inventory.test.js
git commit -m "feat(agent-policy): form model, inventory grouping and a formula-safe CSV export"
```

---

### Task 10: UI wiring in `engine.js` — list, inventory, form, confirmations

**Files:**
- Modify: `src/modules/SpendIQ/engine.js`. The changes are additive: new `ap*` functions placed directly after `policyDelete`, plus a tab switch at the top of `policiesView`. **Do not change `policyEdit`, `policyDelete`, or `openFormModal`** (ruling A).
- Modify: `src/modules/SpendIQ/index.jsx`. Add `window.__SPENDIQ_AP__ = { model, inventory }` beside the existing `window.__SPENDIQ_API_WRITE__`, imported from `./agentPolicy/model` and `./agentPolicy/inventory`. `engine.js` is a script context, not a module; check how it reaches other modules with `grep -n "window.__SPENDIQ" src/modules/SpendIQ/engine.js | head`, and follow that pattern.
- Test: `src/modules/SpendIQ/agentPolicy/engineWiring.contract.test.js`

**Interfaces:**
- Consumes:
  - `window.__SPENDIQ_API_WRITE__(method, path, body) -> Promise<data>` (existing; GET works through it);
  - `window.__SPENDIQ_AP__.model` and `window.__SPENDIQ_AP__.inventory`;
  - the gateway paths from Task 8;
  - existing CSS classes `fm-ov fm-card fm-title fm-body fm-section fm-row fm-lbl fm-help fm-field fm-area fm-err fm-foot btn primary tablecard section-h statusTag`, plus `escH`, `TB`, `toast`, `wfRerender` and `svg`.
- Produces these engine functions:
  - `apPoliciesTab()`
  - `apListHTML()`
  - `apInventoryHTML()`
  - `apExportCsv()`
  - `apOpen(key|null)`
  - `apFormHTML(state)`
  - `apBind(ov, state)`
  - `apRefreshPreview(state)`
  - `apSave(state, intent)`
  - `apConfirmTwoStep(title, body, onYes)`
  - `apRetire(state)`
  - `apCancel(state)`

**Behaviour the wiring must implement (brief §3):**

1. **Two tabs.** `policiesView` gets two tabs: **Agent policies** (default) and **System settings**. The second renders the existing table exactly as today.
2. **Agent policies tab.** It has a "List" / "Inventory" toggle.
   - **List:** `groupBySource`. One card per source document; tiered policies sit under their document. Columns: ID, Name, What happens, Status (Draft/Active/Retired), Extraction confidence. Each row has "Open".
   - **Inventory:** `groupByArea`, with an **Export CSV** button. The button builds `inventoryCsv(rows)` into a `Blob` and downloads `agent-policy-inventory.csv`.
3. **Form** (`apOpen`). It is a modal built with the `fm-*` classes, with sections in this order:
   - **Identity:**
     - Name;
     - Category (free list, `CA_POLICY_CATS` + in use);
     - Business area `<select>` from `/agent-policies/taxonomy`;
     - Sub-area `<select>`, refreshed when the area changes.
   - **The policy:**
     - Situation (textarea);
     - Source excerpt (read-only `<blockquote>` with "Section X of Document"; hidden when there is no source);
     - **How it is enforced** (read-only, three lines from `preview.howEnforced`, or the `cantEnforce` lines in its place);
     - **Example check:** a table of `preview.examples`. Each row shows the input as plain "name: value" pairs, the computed label, and a "That's wrong" toggle that calls `flipExample`. Flipped rows get class `is-changed`. When any row is flipped, show "Ask the agent to fix it" disabled, with the help text "Available once documents are uploaded (stage 2)". Below the table, the checkbox "The examples and how it is enforced are right. This is what the policy says." Ticking it calls `confirm(state.form, currentUserSub, new Date().toISOString())`. Unticking sets `checked: null`.
   - **What happens:**
     - three radio buttons from `OUTCOMES`; changing one calls `switchOutcome`;
     - when the agent suggested the outcome, the line "Suggested by the agent because the document says '<outcomeBecause>'";
     - **approve:** ordered "Who decides" levels (add / remove / move up / move down) and a `responseTimeLabel` line with a "Set a different time" link. The link reveals an hours number input, which is stored as `PT<n>H`;
     - **block:** optional "Who is told" list;
     - **notify:** required "Who is told" list;
     - for approve and block, "Message to the agent";
     - for all outcomes, an editable "What the agent tells the person" field.
   - **Applies to:** the line "Applies to all agents, tools and skills" plus a **Limit it** toggle, which reveals one text field.
   - **Governance:** Owner (help text "Who answers for this policy." with no exception wording), Effective from, Review by, and Change note.
   - **Health line:** shown only when `status === 'live'`. Stage 1 text: "Active since <saved date>. Firing data arrives in stage 3."
   - **Technical view (administrators):** a `<details>`, collapsed, rendered only when the preview response contains `compiled`. It shows `<pre>` with `JSON.stringify(compiled, null, 2)` and any `compiledProblems`.
   - **Header:** `<ID> · Version <n> · <Draft|Active|Retired> · Extraction confidence: <level or "not extracted">`. When the level is not High, show a "Check before this goes Active" notice listing `confidence.failed`.
   - **Footer:** Cancel, Save draft, Activate… (Approver+), and Retire… (Approver+, only when live or draft).
4. **Preview.** Every edit that changes `situation`, `outcome`, `deciders`, `notify`, `responseTime`, `hidden` or `examples` calls `POST /agent-policies/preview`, debounced 300 ms. Before the call, if `editClearsConfirmation(prev, next)`, set `checked = null` and untick the box.
5. **Save draft.** `POST /agent-policies` for a new policy, otherwise `POST /agent-policies/{key}/versions` with `intent: 'draft'` and `baseVersion`. Toast "Saved as version N". Reload the policy.
6. **Activate….** `apConfirmTwoStep('Make this policy Active?', 'Agents will be held to it from now on.', …)`. Show the confirmation modal; on Yes, show a second modal "Confirm: activate <ID>?" with the buttons "Activate" and "Go back". Only then POST with `intent: 'activate'`.
   - **On 422:** render every `problems[].message` in one `fm-err` summary list, and focus the element `[data-ap-field="<problems[0].field>"]`. Problems with `routeTo: 'administrator'` get the suffix " An administrator has been told." (stage 1 states this; the admin task is created in stage 3).
   - **On 409:** toast the detail and reload.
7. **Retire….** The same two-step confirm, then `POST /{key}/retire`.
8. **Cancel.** Discard edits and re-render from the last saved version, `versions[latestVersion-1].form`. For a never-saved new policy, Cancel closes the modal.
9. **Hidden from everyone except the Technical view:** events, hook points, condition rows, all/any, per-tool or per-agent scope lists, audit settings and test cases. They appear only inside the Technical view's JSON.
10. **Every element** that a problem can name gets `data-ap-field="<problems.field key>"`. The keys are: `name`, `businessArea`, `subArea`, `situation`, `outcome`, `deciders`, `responseTime`, `notify`, `messageForAgent`, `limit`, `examples`, `checked`, `checkpoint`, `registry`, `inputs`, `units`, `timeWindow`, `owner`. Field-less problems (`checkpoint`, `registry`, `inputs`, `units`, `timeWindow`) map to the How-it-is-enforced block.

- [ ] **Step 1: Write the failing contract test**

Follow the existing pattern in `src/modules/SpendIQ/canvasAssets.contract.test.js`: read `engine.js` as text and assert its structure.

```js
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const src = fs.readFileSync(path.join(__dirname, '..', 'engine.js'), 'utf8');
const between = (a, b) => src.slice(src.indexOf(a), src.indexOf(b, src.indexOf(a)));

describe('agent policy wiring in engine.js', () => {
  it('leaves the old policy form untouched', () => {
    expect(src).toContain("function policyEdit(i){const p=i>=0?POLICIES[i]:{id:'POL-'+(501+POLICIES.length)");
    expect(src).toContain("function policyDelete(i){");
  });
  it('form sections in brief order', () => {
    const f = between('function apFormHTML', '\nfunction ');
    const order = ['Identity', 'The policy', 'What happens', 'Applies to', 'Governance', 'Technical view (administrators)'];
    const idx = order.map((s) => f.indexOf(s));
    expect(idx.every((i) => i >= 0)).toBe(true);
    expect([...idx].sort((a, b) => a - b)).toEqual(idx);
  });
  it('uses the agreed wording', () => {
    for (const s of ['The examples and how it is enforced are right. This is what the policy says.',
      'Applies to all agents, tools and skills', 'Limit it', 'How it is enforced', 'Extraction confidence',
      'Check before this goes Active', 'Set a different time']) expect(src).toContain(s);
    expect(between('function apFormHTML', '\nfunction ')).not.toMatch(/approves an exception/i);
  });
  it('hidden controls are not rendered outside the technical view', () => {
    const f = between('function apFormHTML', '\nfunction ');
    const visible = f.slice(0, f.indexOf('Technical view (administrators)'));
    for (const s of ['hook point', 'Fires when', 'onMissingData', 'condition rows', 'test case']) expect(visible).not.toContain(s);
  });
  it('activate and retire go through the two-step confirm', () => {
    expect(between('function apSave', '\nfunction ')).toContain('apConfirmTwoStep');
    expect(between('function apRetire', '\nfunction ')).toContain('apConfirmTwoStep');
  });
  it('talks only to the gateway', () => {
    const block = between('function apPoliciesTab', '/* The small edit form');
    expect(block).not.toContain('__SPENDIQ_AGENT_RUN__');
    expect(block).not.toContain('AI_API');
    expect(block).toContain("__SPENDIQ_API_WRITE__");
  });
});
```

These assertions need the new `ap*` functions to sit together, between `policyDelete` and the `/* The small edit form` comment that precedes `openFormModal`. Place them there.

- [ ] **Step 2: Run it to make sure it fails**

Run: `npx vitest run src/modules/SpendIQ/agentPolicy/engineWiring.contract.test.js`
Expected: FAIL (the `apFormHTML` section isn't found).

- [ ] **Step 3: Implement the `ap*` functions**

Write them to the behaviour list above, using the existing `openFormModal` markup conventions (`fm-ov`, `fm-card`, `fm-section`, `fm-row`, `fm-help`, `fm-err`, `fm-foot`). Build the overlay the way `openFormModal` does: append it inside `.siq-root`, honour `fm-midnight` when `current==='workspace'`, close on Escape, and set the initial focus. Escape every user string with `escH`. Every outcome-specific block is rendered from `state.form.outcome`, so an unselected outcome's fields are absent from the DOM, not hidden with CSS.

Then make `policiesView()` begin with:

```js
  if((window.__apTab||'agent')==='agent') return apPoliciesTab();
```

Add a tab strip to both renders: `apPoliciesTab()` draws it, and the existing body is wrapped by a prefix string only. The strip has two buttons, "Agent policies" and "System settings", and sets `window.__apTab`, then calls `wfRerender()`.

Put the Policies view back in the nav, gated by the existing `NAV_PERMISSION.policies`. First check `canvasAssets.contract.test.js:305-320`, which asserts the rail does **not** list Policies. If adding the nav entry would break that test, leave the rail alone and add a "Policies" link to the Agent canvas side panel's policy library header instead. Report which option you took.

- [ ] **Step 4: Run the UI tests, and check the screen by hand on the local stack**

Run: `npx vitest run src/modules/SpendIQ`
Expected: the new tests PASS, and the existing SpendIQ tests show the same pass count as before (capture it before Step 3).

Start the stack, following memory's local-stack note: the gateway needs `node --experimental-global-webcrypto`, and you must never `pkill -f uvicorn`. Then open the Policies screen. Chrome MCP cannot reach this host's localhost (see memory), so use the `run` skill's browser-driven pattern from this host, or ask the user to click through. Walk the "Live demonstration" script in Task 11 and record any console errors. There must be none (brief test 22).

- [ ] **Step 5: Commit** (stage only your hunks; `engine.js` and `index.jsx` may carry another session's edits)

```bash
git diff src/modules/SpendIQ/engine.js | grep -c '^[-+]' # sanity: only ap* additions + the 1-line tab switch
git add -p src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx
git add src/modules/SpendIQ/agentPolicy/engineWiring.contract.test.js
git diff --cached --stat
git commit -m "feat(agent-policy): agent policies tab, inventory, the new form with example check and two-step confirm"
```

---

### Task 11: Live demonstration and diff summary

**Files:**
- Create: `specs/2026-10-08-agent-policy-governance-stage1-verification.md`

- [ ] **Step 1: Run every new and touched suite in one go, and record the counts**

- BP_Backend: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src ./venv/bin/python -m pytest tests/agent_policy tests/guardrails tests/governance tests/approvals tests/engines -q`, then the same with `PROCWISE_TEST_LIVE_DB=1` for `tests/migrations/test_2026_10_08_bp_agent_policy.py tests/agent_policy/test_repo_live.py`.
- Gateway: `npx jest`.
- UI: `npx vitest run src/modules/SpendIQ`.

- [ ] **Step 2: Demonstrate on the local server against `bp_testdb`, through the gateway**

Use `curl` against the gateway with a real token, or with the local `AUTH_BYPASS` (Admin), and record each response:
1. Create a draft with only a name. It returns `GEN-000n` v1.
2. Save it with Finance / Refunds and credits and the `FORM_EXAMPLE` content. It returns v2, with the ID unchanged.
3. Preview. The examples show "A person decides" for 501 and "Nothing happens" for 500 and 499.
4. Activate without confirming. Expect 422 listing `checked` **and** `registry`, because no `refund.issue` tool exists. That is correct.
5. Change the condition's tool to a real registered tool, e.g. `supplier_ranking`, and the input to `args.payload_json`. Confirm, then activate. Expect v3 live.
6. `GET /orchestrator/agent-policies/v2/live` on BP_Backend with the orchestrator key. The policy appears, and its JSON passes `contract.validate`.
7. Save a new draft (v4). The feed still serves v3 (live stays live).
8. Retire. The feed no longer contains the policy.
9. Run a second save with a stale `baseVersion`. Expect 409.
10. Call BP_Backend directly without the gateway key. Expect 401.

- [ ] **Step 3: Write the diff summary (ruling A)**

In the verification file, list each repo's commits (`git log --oneline` for the range) and, per file, what was added or changed. For `engine.js`, list:
- every new function name;
- the line count added;
- confirmation, by `git diff` of the old form's line range, that `policyEdit`, `policyDelete` and `openFormModal` are byte-identical.

- [ ] **Step 4: Commit the verification note and stop for review**

```bash
git add specs/2026-10-08-agent-policy-governance-stage1-verification.md
git commit -m "docs(agent-policy): stage 1 verification and diff summary"
```

Do not merge or push. Hand the diff summary to the user (ruling A), and write the stage 2 plan only after stage 1 is approved.

---

## Self-review notes (written while planning)

- **Brief coverage in stage 1:**
  - §3.1 field mapping: Tasks 9 and 10.
  - §3.2 example check: Tasks 3, 5, 9 and 10. The "Ask the agent to fix it" re-run is stage 2; it is shown disabled with an honest note.
  - §3.3 outcome: Tasks 3, 9 and 10.
  - §3.4 status and versioning: Tasks 1, 6 and 10.
  - §3.5 what Active requires: Task 5.
  - §3.6 extraction confidence: Task 5. It is computed, but only extracted policies (stage 2) have a document to check against.
  - §3.7 the contract: Tasks 3, 4 and 5.
  - §4 list, inventory and CSV: Tasks 9 and 10. Conflicts are stage 4.
  - §7 JSON: Tasks 3 and 4.
  - Acceptance tests covered now: 3, 4, 5, 7, 8, 9, 10, 12 (contract parts), 20, 21 and 22. The remaining tests are in stages 2 to 5.
- **Deliberately not in stage 1:** the extraction agent, enforcement, the decision engine, conflicts and learning (design §4). Policy owners cannot yet make a policy that matches a refund, because no refund tool exists in the registry. The form says so through the registry problem, which is the brief's intended behaviour.
