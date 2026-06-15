# bp_prompt / bp_policy Governance Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create DB-governed `proc.bp_prompt` / `proc.bp_policy` tables seeded from current agent defaults, point the existing engines at them, and give agents a uniform resolver + a reload endpoint.

**Architecture:** Two existing engines (`PromptEngine`, `PolicyEngine`) already load-once-and-cache rows from `proc.prompt` / `proc.policy` (which never existed). We create the `bp_`-prefixed tables with the exact column shape the engines parse, seed them from values agents currently hardcode, repoint the two `SELECT`s, add three thin agent-scoped helper methods on `BaseAgent`, and extend the reload endpoint to cover prompts as well as policies.

**Tech Stack:** PostgreSQL (`bp_sqldb` on RDS), Python 3.12, psycopg2, FastAPI, pytest.

**Conventions:**
- All new tables use the `bp_` prefix; indexes `ix_bp_<table>_<col>` (per repo convention).
- No Claude attribution in commit messages.
- `docs/superpowers/` is gitignored — the plan/spec files are NOT committed; only code/test/SQL files are.
- DB connection: `set -a && . ./.env 2>/dev/null && set +a` then
  `PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" ...`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql` | DDL + idempotent seed for both tables | Create |
| `src/orchestration/prompt_engine.py` | repoint `FROM proc.prompt` → `proc.bp_prompt` (line 128) | Modify |
| `src/engines/policy_engine.py` | repoint `FROM proc.policy` → `proc.bp_policy` (line 189) | Modify |
| `src/agents/base_agent.py` | add `resolve_prompt` / `governing_policies` / `governing_policy` | Modify |
| `src/api/routers/agents.py` | add `POST /agents/reload-governance` (+ keep `reload-policies`) | Modify |
| `tests/test_prompt_engine.py` | assert query targets `proc.bp_prompt` | Modify |
| `tests/test_policy_engine.py` | assert query targets `proc.bp_policy` | Modify |
| `tests/test_base_agent_governance.py` | resolver unit tests | Create |
| `tests/test_agent_endpoints.py` | reload-governance endpoint test | Modify |

---

## Task 1: Migration — create and seed `bp_prompt` / `bp_policy`

**Files:**
- Create: `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql`

- [ ] **Step 1: Write the migration SQL**

Create `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql` with exactly:

```sql
-- Create proc.bp_prompt and proc.bp_policy (bp_ naming convention) and seed
-- them from the defaults agents currently hardcode. Idempotent: safe to re-run.

-- ============================ bp_prompt ============================
CREATE TABLE IF NOT EXISTS proc.bp_prompt (
    prompt_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prompt_name          TEXT        NOT NULL,
    prompt_type          TEXT,
    prompt_linked_agents TEXT,
    prompts_desc         JSONB,
    prompts_status       SMALLINT    NOT NULL DEFAULT 1,
    version              INTEGER     NOT NULL DEFAULT 1,
    created_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    last_modified_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by     TEXT        NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_status
    ON proc.bp_prompt (prompts_status) WHERE prompts_status = 1;
CREATE INDEX IF NOT EXISTS ix_bp_prompt_name
    ON proc.bp_prompt (prompt_name);

-- ============================ bp_policy ============================
CREATE TABLE IF NOT EXISTS proc.bp_policy (
    policy_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name          TEXT        NOT NULL,
    policy_type          TEXT,
    policy_desc          TEXT,
    policy_details       JSONB,
    policy_linked_agents TEXT,
    policy_status        SMALLINT    NOT NULL DEFAULT 1,
    version              INTEGER     NOT NULL DEFAULT 1,
    created_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    last_modified_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by     TEXT        NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_status
    ON proc.bp_policy (policy_status) WHERE policy_status = 1;
CREATE INDEX IF NOT EXISTS ix_bp_policy_name
    ON proc.bp_policy (policy_name);
CREATE INDEX IF NOT EXISTS ix_bp_policy_type
    ON proc.bp_policy (policy_type);

-- ============================ seed prompts ============================
-- Guarded on prompt_name so re-runs are no-ops (no unique constraint added,
-- so the governance UI stays free to create name variants/versions later).
INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'supplier_ranking_justification', 'justification', 'supplier_ranking_agent',
       '{"prompt_template": "Supplier {supplier_name} achieved a final score of {final_score:.2f}.\n{score_breakdown}"}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'supplier_ranking_justification');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'rank_by_criteria', 'decomposition', 'supplier_ranking_agent',
       '{"templates": [{"template_id": "rank_by_criteria", "parameters": {"category": null, "criteria": ["price", "delivery", "risk"], "time_period": null, "filters": null}}]}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'rank_by_criteria');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'negotiation_message_default', 'message', 'negotiation_agent',
       '{"prompt_template": "{header}\n{details}{context_sections}"}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'negotiation_message_default');

-- ============================ seed policies ============================
INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'WeightAllocationPolicy', 'supplier_ranking', 'Default supplier ranking weights', 'supplier_ranking_agent',
       '{"policy_identifier": "weight_allocation_policy", "rules": {"default_weights": {"price": 0.4, "delivery": 0.3, "risk": 0.2, "payment_terms": 0.1}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'WeightAllocationPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'CategoricalScoringPolicy', 'supplier_ranking', 'Categorical scoring maps', 'supplier_ranking_agent',
       '{"policy_identifier": "categorical_scoring_policy", "rules": {"categorical_maps": {}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'CategoricalScoringPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'NormalizationDirectionPolicy', 'supplier_ranking', 'Per-metric normalization direction', 'supplier_ranking_agent',
       '{"policy_identifier": "normalization_direction_policy", "rules": {"directions": {"price": "lower_is_better", "delivery": "lower_is_better", "risk": "lower_is_better", "payment_terms": "higher_is_better"}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'NormalizationDirectionPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'ContractExpiryOpportunity', 'opportunity', 'Flag contracts nearing expiry', 'opportunity_miner_agent',
       '{"policy_identifier": "contract_expiry_check", "required_fields": ["negotiation_window_days"], "default_conditions": {"negotiation_window_days": 90}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'ContractExpiryOpportunity');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'PriceBenchmarkVariance', 'opportunity', 'Flag price variance vs benchmark', 'opportunity_miner_agent',
       '{"policy_identifier": "price_variance_check", "required_fields": ["supplier_id", "item_id", "actual_price", "benchmark_price"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'PriceBenchmarkVariance');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'VolumeConsolidation', 'opportunity', 'Flag volume consolidation opportunities', 'opportunity_miner_agent',
       '{"policy_identifier": "volume_consolidation_check", "required_fields": ["minimum_volume_gbp"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'VolumeConsolidation');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'SupplierRiskAlert', 'opportunity', 'Flag elevated supplier risk', 'opportunity_miner_agent',
       '{"policy_identifier": "supplier_risk_check", "required_fields": ["risk_threshold"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'SupplierRiskAlert');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'MaverickSpend', 'opportunity', 'Flag off-contract maverick spend', 'opportunity_miner_agent',
       '{"policy_identifier": "maverick_spend_check", "required_fields": [], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'MaverickSpend');
```

- [ ] **Step 2: Apply the migration to `bp_sqldb`**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env 2>/dev/null && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql && echo MIGRATION_OK
```
Expected: a series of `CREATE TABLE` / `CREATE INDEX` / `INSERT 0 1` lines ending with `MIGRATION_OK`.

- [ ] **Step 3: Verify tables, indexes, and seed counts**

Run:
```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
  "SELECT (SELECT count(*) FROM proc.bp_prompt) AS prompts,
          (SELECT count(*) FROM proc.bp_policy) AS policies;"
```
Expected: `3|8` (3 prompts, 8 policies).

Run (re-apply to confirm idempotency — counts must not change):
```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql >/dev/null
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
  "SELECT (SELECT count(*) FROM proc.bp_prompt), (SELECT count(*) FROM proc.bp_policy);"
```
Expected: `3|8` again.

- [ ] **Step 4: Commit**

```bash
git add deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql
git commit -m "feat(governance): create and seed bp_prompt/bp_policy tables"
```

---

## Task 2: Repoint `PromptEngine` to `proc.bp_prompt`

**Files:**
- Modify: `src/orchestration/prompt_engine.py:128`
- Test: `tests/test_prompt_engine.py`

- [ ] **Step 1: Add a failing test that the query targets `proc.bp_prompt`**

Append to `tests/test_prompt_engine.py`:

```python
def test_prompt_engine_queries_bp_prompt_table():
    captured = {}

    class CapturingCursor(DummyCursor):
        def execute(self, query, params=None):
            captured["query"] = query

    class CapturingConn(DummyConn):
        def cursor(self):
            return CapturingCursor(self._rows)

    PromptEngine(connection_factory=lambda: CapturingConn([]))
    assert "proc.bp_prompt" in captured["query"]
    assert "proc.prompt " not in captured["query"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_prompt_engine.py::test_prompt_engine_queries_bp_prompt_table -v`
Expected: FAIL — assertion error, query still contains `proc.prompt`.

- [ ] **Step 3: Repoint the query**

In `src/orchestration/prompt_engine.py`, in `_fetch_prompt_rows` (around line 128), change:

```python
                        FROM proc.prompt
```
to:

```python
                        FROM proc.bp_prompt
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_prompt_engine.py -v`
Expected: PASS (new test + all existing prompt-engine tests).

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/prompt_engine.py tests/test_prompt_engine.py
git commit -m "feat(governance): point PromptEngine at proc.bp_prompt"
```

---

## Task 3: Repoint `PolicyEngine` to `proc.bp_policy`

**Files:**
- Modify: `src/engines/policy_engine.py:189`
- Test: `tests/test_policy_engine.py`

- [ ] **Step 1: Add a failing test that the query targets `proc.bp_policy`**

Append to `tests/test_policy_engine.py`:

```python
def test_policy_engine_queries_bp_policy_table():
    captured = {}

    class CapturingCursor:
        def __init__(self):
            self.description = None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def execute(self, query, params=None):
            captured["query"] = query
        def fetchall(self):
            return []

    class CapturingConn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def cursor(self):
            return CapturingCursor()

    PolicyEngine(connection_factory=lambda: CapturingConn())
    assert "proc.bp_policy" in captured["query"]
    assert "proc.policy " not in captured["query"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_policy_engine.py::test_policy_engine_queries_bp_policy_table -v`
Expected: FAIL — query still contains `proc.policy`.

- [ ] **Step 3: Repoint the query**

In `src/engines/policy_engine.py`, in `_fetch_policy_rows` (around line 189), change:

```python
                        FROM proc.policy
```
to:

```python
                        FROM proc.bp_policy
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_policy_engine.py -v`
Expected: PASS (new test + all existing policy-engine tests).

- [ ] **Step 5: Commit**

```bash
git add src/engines/policy_engine.py tests/test_policy_engine.py
git commit -m "feat(governance): point PolicyEngine at proc.bp_policy"
```

---

## Task 4: Add agent-scoped resolver helpers to `BaseAgent`

**Files:**
- Modify: `src/agents/base_agent.py` (add methods after `get_workflow_context`, around line 294)
- Test: `tests/test_base_agent_governance.py` (create)

Note: `BaseAgent.__init__` sets `self.prompt_engine` (line 248). The policy engine
is NOT on the agent — it lives on the container as `agent_nick.policy_engine`.
The helpers use `self.prompt_engine` and `getattr(self.agent_nick, "policy_engine", None)`.
The module already defines `_slugify_agent_name` (line 65) and imports `Optional`,
`List`, `Dict`, `Any`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_base_agent_governance.py`:

```python
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine


def _make_agent():
    prompt_rows = [
        {
            "prompt_id": 1,
            "prompt_name": "supplier_ranking_justification",
            "prompt_type": "justification",
            "prompt_linked_agents": "supplier_ranking_agent",
            "prompts_desc": '{"prompt_template": "Score for {name} is {score}."}',
        },
        {
            "prompt_id": 2,
            "prompt_name": "global_only",
            "prompt_type": "info",
            "prompt_linked_agents": "",
            "prompts_desc": '{"prompt_template": "Global template."}',
        },
    ]
    policy_rows = [
        {
            "policy_id": 1,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "weights",
            "policy_details": '{"rules": {"default_weights": {"price": 1.0}}}',
            "policy_linked_agents": "supplier_ranking_agent",
        },
        {
            "policy_id": 2,
            "policy_name": "UnrelatedPolicy",
            "policy_type": "other",
            "policy_desc": "n/a",
            "policy_details": "{}",
            "policy_linked_agents": "some_other_agent",
        },
    ]
    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="gpt-oss"),
        prompt_engine=PromptEngine(prompt_rows=prompt_rows),
        policy_engine=PolicyEngine(policy_rows=policy_rows),
        learning_repository=None,
    )

    class SupplierRankingAgent(base_agent.BaseAgent):
        def run(self, *a, **k):
            return None

    return SupplierRankingAgent(agent_nick)


def test_resolve_prompt_agent_scoped_with_formatting():
    agent = _make_agent()
    out = agent.resolve_prompt("supplier_ranking_justification", name="ACME", score=9)
    assert out == "Score for ACME is 9."


def test_resolve_prompt_falls_back_to_global_then_none():
    agent = _make_agent()
    assert agent.resolve_prompt("global_only") == "Global template."
    assert agent.resolve_prompt("does_not_exist") is None


def test_governing_policies_returns_only_linked():
    agent = _make_agent()
    names = {p.get("policyName") for p in agent.governing_policies()}
    assert names == {"WeightAllocationPolicy"}


def test_governing_policy_by_name():
    agent = _make_agent()
    policy = agent.governing_policy("weight_allocation_policy")
    assert policy is not None
    assert policy.get("policyName") == "WeightAllocationPolicy"
    assert agent.governing_policy("nonexistent_policy") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_base_agent_governance.py -v`
Expected: FAIL — `AttributeError: 'SupplierRankingAgent' object has no attribute 'resolve_prompt'`.

- [ ] **Step 3: Implement the helpers**

In `src/agents/base_agent.py`, insert these methods into `BaseAgent` immediately
after the `get_workflow_context` method (after line 294):

```python
    # ------------------------------------------------------------------
    # Governance resolver — uniform, agent-scoped access to prompts/policies
    # ------------------------------------------------------------------
    def _governance_slug(self) -> str:
        """Slug identifying THIS agent for prompt/policy linkage lookups."""
        return _slugify_agent_name(self.__class__.__name__)

    def resolve_prompt(self, name: str, **fmt: Any) -> Optional[str]:
        """Return the active prompt template governing this agent, by name.

        Resolution order: prompt linked to this agent whose name matches →
        global prompt with that name → None. When ``fmt`` is supplied the
        template is ``str.format``-ed; a formatting error returns the raw
        template rather than raising.
        """
        engine = getattr(self, "prompt_engine", None)
        if engine is None:
            return None
        target = _slugify_agent_name(name)

        def _match(prompts):
            for prompt in prompts or []:
                if _slugify_agent_name(prompt.get("promptName")) == target:
                    return prompt.get("template")
            return None

        template = None
        try:
            template = _match(engine.prompts_for_agent(self.__class__.__name__))
        except Exception:  # pragma: no cover - defensive
            logger.debug("resolve_prompt: agent-scoped lookup failed", exc_info=True)
        if template is None:
            try:
                template = _match(engine.all_prompts())
            except Exception:  # pragma: no cover - defensive
                logger.debug("resolve_prompt: global lookup failed", exc_info=True)
        if not template:
            return None
        if fmt:
            try:
                return str(template).format(**fmt)
            except (KeyError, IndexError, ValueError):
                return str(template)
        return str(template)

    def governing_policies(self) -> List[Dict[str, Any]]:
        """Return all policies linked to this agent (parsed)."""
        engine = getattr(self.agent_nick, "policy_engine", None)
        if engine is None:
            return []
        slug = self._governance_slug()
        try:
            policies = engine.list_policies()
        except Exception:  # pragma: no cover - defensive
            logger.debug("governing_policies: list_policies failed", exc_info=True)
            return []
        result: List[Dict[str, Any]] = []
        for policy in policies:
            aliases = policy.get("aliases") or set()
            linked = set(policy.get("policy_linked_agents") or [])
            if slug in aliases or slug in linked:
                result.append(policy)
        return result

    def governing_policy(self, name: str) -> Optional[Dict[str, Any]]:
        """Return one policy by slug/name: agent-scoped → global → None."""
        engine = getattr(self.agent_nick, "policy_engine", None)
        if engine is None:
            return None
        target = _slugify_agent_name(name)
        for policy in self.governing_policies():
            if target in (policy.get("aliases") or set()):
                return policy
        try:
            return engine.get_policy(name)
        except Exception:  # pragma: no cover - defensive
            logger.debug("governing_policy: get_policy failed", exc_info=True)
            return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_base_agent_governance.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Run the existing base-agent tests for regressions**

Run: `.venv/bin/pytest tests/test_base_agent_ollama.py -v`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/agents/base_agent.py tests/test_base_agent_governance.py
git commit -m "feat(governance): add agent-scoped prompt/policy resolver to BaseAgent"
```

---

## Task 5: Extend the reload endpoint to cover prompts

**Files:**
- Modify: `src/api/routers/agents.py` (after the existing `reload_policies`, line 68-76)
- Test: `tests/test_agent_endpoints.py`

The existing `POST /agents/reload-policies` (lines 68-76) reloads policies only and
stays as a backward-compatible alias. We add `POST /agents/reload-governance` that
reloads both engines and returns row counts.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_endpoints.py` (this test builds a minimal app with a
stub `agent_nick` exposing the two engines):

```python
def test_reload_governance_reloads_both_engines():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from api.routers import agents as agents_router

    calls = {"prompts": 0, "policies": 0}

    prompt_engine = SimpleNamespace(
        refresh=lambda: calls.__setitem__("prompts", calls["prompts"] + 1),
        all_prompts=lambda: [{"promptId": 1}, {"promptId": 2}],
    )
    policy_engine = SimpleNamespace(
        reload_policies=lambda: calls.__setitem__("policies", calls["policies"] + 1),
        list_policies=lambda: [{"policyId": "a"}],
    )

    app = FastAPI()
    app.include_router(agents_router.router)
    app.state.agent_nick = SimpleNamespace(
        prompt_engine=prompt_engine,
        policy_engine=policy_engine,
        agents={},
    )

    client = TestClient(app)
    resp = client.post("/agents/reload-governance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["prompts"] == 2
    assert body["policies"] == 1
    assert calls == {"prompts": 1, "policies": 1}
```

(If `tests/test_agent_endpoints.py` does not import `pytest`/paths the way other
tests do, mirror the import header used by neighbouring tests in that file.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent_endpoints.py::test_reload_governance_reloads_both_engines -v`
Expected: FAIL — `404 Not Found` (route does not exist yet).

- [ ] **Step 3: Implement the endpoint**

In `src/api/routers/agents.py`, immediately after the existing `reload_policies`
function (after line 76), add:

```python
@router.post("/reload-governance")
async def reload_governance(agent_nick=Depends(get_agent_nick)):
    """Reload both prompt and policy governance from the bp_ tables."""
    try:
        agent_nick.policy_engine.reload_policies()
        agent_nick.prompt_engine.refresh()
        return {
            "status": "success",
            "prompts": len(agent_nick.prompt_engine.all_prompts()),
            "policies": len(agent_nick.policy_engine.list_policies()),
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to reload governance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_agent_endpoints.py::test_reload_governance_reloads_both_engines -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/agents.py tests/test_agent_endpoints.py
git commit -m "feat(governance): add reload-governance endpoint (prompts + policies)"
```

---

## Task 6: Live verification — restart and confirm clean governance load

**Files:** none (operational).

- [ ] **Step 1: Run the full affected test suite**

Run:
```bash
.venv/bin/pytest tests/test_prompt_engine.py tests/test_policy_engine.py \
  tests/test_base_agent_governance.py tests/test_base_agent_ollama.py \
  tests/test_agent_endpoints.py -v
```
Expected: all PASS.

- [ ] **Step 2: Restart procwise**

Run: `sudo -n /usr/bin/systemctl restart procwise && echo RESTART_OK && sleep 18 && sudo -n /usr/bin/systemctl is-active procwise`
Expected: `RESTART_OK` then `active`.

- [ ] **Step 3: Confirm the governance-load errors are gone and rows loaded**

Run:
```bash
sudo -n /usr/bin/journalctl -u procwise --no-pager --since "40 seconds ago" \
  | grep -iE "Failed to load prompts|Failed to load policies|PolicyEngine loaded"
```
Expected: NO `Failed to load prompts/policies` lines; a `PolicyEngine loaded 8 policies` line (count ≥ 8).

- [ ] **Step 4: Confirm the reload endpoint works against the live service**

Run: `curl -s -X POST http://localhost:8000/agents/reload-governance`
Expected: JSON `{"status":"success","prompts":3,"policies":8}` (counts may grow as
governance rows are added later).

---

## Self-Review Notes

- **Spec coverage:** schema (Task 1), engine repoint (Tasks 2-3), resolver (Task 4),
  reload endpoint (Task 5), seed-from-defaults (Task 1 seeds), startup-error
  clearance + integration (Task 6). All spec sections mapped.
- **Type consistency:** engines return `promptName` / `policyName` /
  `aliases` / `policy_linked_agents` / `template` keys (verified against
  `prompt_engine._normalise_prompt` and `policy_engine._normalise_policy_row`);
  the resolver and tests use those exact keys. `prompts_for_agent` /
  `all_prompts` / `list_policies` / `get_policy` / `refresh` / `reload_policies`
  all exist on the current engines.
- **No placeholders:** every SQL value, template string, and method body is
  concrete and lifted from the live source.
