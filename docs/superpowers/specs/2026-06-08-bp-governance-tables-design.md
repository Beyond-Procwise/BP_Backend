# Design: `bp_prompt` / `bp_policy` Governance Tables + Agent Resolver

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Author:** muthu

## Problem

`PromptEngine` (`src/orchestration/prompt_engine.py`) and `PolicyEngine`
(`src/engines/policy_engine.py`) query `proc.prompt` and `proc.policy`, but those
tables **have never existed** — no DDL anywhere in the repo. As a result:

- Both engines load 0 rows at startup and log `Failed to load prompts/policies
  from database` (the two remaining ERROR lines after the recent reranker fix).
- Every agent silently falls back to hardcoded defaults (e.g. supplier-ranking
  weights, the `rank_by_criteria` template, opportunity-policy conditions).
- Governance is not editable at runtime; behavior is frozen in Python.

## Goals

1. Create `proc.bp_prompt` and `proc.bp_policy` (per the `bp_` table convention),
   seeded from the values agents currently hardcode so behavior is **identical**
   but now DB-governed.
2. Point the two existing engines at the new tables with no change to their
   parsing/caching internals.
3. Give agents one **clear, uniform, efficient** way to pull their governing
   prompts/policies, scoped to the calling agent.
4. Allow runtime reload so governance-UI edits propagate without a restart.

## Non-Goals

- No schema normalization — keep the JSONB-blob shape the engines already parse.
- No auto-injection of prompts/policies into every LLM call — agents opt in.
- No rewrite of existing agent call sites — they keep working unchanged.
- No data migration — the old `proc.prompt`/`proc.policy` names never existed and
  are simply abandoned.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Engine scope | Repoint engines + add a clean agent-facing resolver |
| Seed data | Seed from current hardcoded defaults |
| Schema shape | Keep the existing shape the engines parse (JSONB blobs) |
| Reload endpoint | In scope (extend existing `reload-policies`) |

## Schema

All columns chosen to match what the engines already SELECT and parse. Extra
governance/audit columns are additive and harmless (engines select explicit
column lists).

### `proc.bp_prompt`

| column | type | notes |
|---|---|---|
| `prompt_id` | `BIGINT GENERATED ALWAYS AS IDENTITY` PRIMARY KEY | engine does `int(prompt_id)` |
| `prompt_name` | `TEXT NOT NULL` | |
| `prompt_type` | `TEXT` | |
| `prompt_linked_agents` | `TEXT` | comma/JSON list of agent slugs; engine coerces to list |
| `prompts_desc` | `JSONB` | `{prompt_template, templates[], metadata, instructions, prompt_config}` |
| `prompts_status` | `SMALLINT NOT NULL DEFAULT 1` | engine filters `COALESCE(prompts_status,1)=1` |
| `version` | `INTEGER NOT NULL DEFAULT 1` | |
| `created_date` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `created_by` | `TEXT NOT NULL DEFAULT 'system'` | |
| `last_modified_date` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `last_modified_by` | `TEXT NOT NULL DEFAULT 'system'` | |

Indexes:
- `ix_bp_prompt_status` — partial `WHERE prompts_status = 1` (the hot filter)
- `ix_bp_prompt_name`

### `proc.bp_policy`

| column | type | notes |
|---|---|---|
| `policy_id` | `BIGINT GENERATED ALWAYS AS IDENTITY` PRIMARY KEY | |
| `policy_name` | `TEXT NOT NULL` | |
| `policy_type` | `TEXT` | |
| `policy_desc` | `TEXT` | |
| `policy_details` | `JSONB` | `{rules:{default_weights:{...}}, policy_identifier, ...}` |
| `policy_linked_agents` | `TEXT` | agent slugs; engine coerces to list |
| `policy_status` | `SMALLINT NOT NULL DEFAULT 1` | engine filters `COALESCE(policy_status,1)=1` |
| `version` | `INTEGER NOT NULL DEFAULT 1` | |
| `created_date` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `created_by` | `TEXT NOT NULL DEFAULT 'system'` | |
| `last_modified_date` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `last_modified_by` | `TEXT NOT NULL DEFAULT 'system'` | |

Indexes:
- `ix_bp_policy_status` — partial `WHERE policy_status = 1`
- `ix_bp_policy_name`
- `ix_bp_policy_type`

## Component 1 — Engine repoint

**`src/orchestration/prompt_engine.py`** — change the one query in
`_fetch_prompt_rows`: `FROM proc.prompt` → `FROM proc.bp_prompt`.

**`src/engines/policy_engine.py`** — change the one query in
`_fetch_policy_rows`: `FROM proc.policy` → `FROM proc.bp_policy`.

No other engine changes. All normalization, slug indexing, caching, and the
`refresh()` / `reload_policies()` methods stay as-is.

**Efficiency:** Both engines are constructed once per process and cached on the
`AgentNick` container (`agent_nick.prompt_engine`, `agent_nick.policy_engine`),
so the DB is read once at startup. Each engine builds in-memory indexes
(`_agent_index` keyed by agent slug, `_slug_index` keyed by policy alias), so all
agent lookups are O(1) dict hits with zero per-call DB traffic.

## Component 2 — Agent-facing resolver (BaseAgent)

Add three thin helpers to `BaseAgent` (`src/agents/base_agent.py`). They wrap the
existing engine methods and add agent-scoping using the calling agent's class
name slug. ~40 lines total.

```python
def resolve_prompt(self, name: str, **fmt) -> Optional[str]:
    """Return the active prompt template governing THIS agent, by name,
    optionally formatted with **fmt.

    Resolution order:
      1. prompt linked to this agent (prompt_linked_agents) whose name matches
      2. global prompt with that name (no linked agents)
      3. None
    The returned string is the template (`prompts_desc.prompt_template` or the
    raw template); .format(**fmt) is applied when fmt is supplied and the
    template contains placeholders.
    """

def governing_policies(self) -> List[Dict[str, Any]]:
    """All policies linked to this agent (via policy_linked_agents), parsed."""

def governing_policy(self, name: str) -> Optional[Dict[str, Any]]:
    """Single policy by slug/name: agent-scoped match → global → None."""
```

Implementation notes:
- Agent slug derived from `self.__class__.__name__` via the same `_slugify`
  convention the engines use, so linkage matches.
- `resolve_prompt` uses `prompt_engine.prompts_for_agent(slug)` then falls back to
  `prompt_engine.all_prompts()` for a global name match.
- `governing_policies` filters `policy_engine.list_policies()` where the agent
  slug is in the policy's `aliases`/`policy_linked_agents`.
- All helpers are defensive: return `None`/`[]` on any miss or engine absence
  (tests construct agents without a live DB).

**Adoption is incremental and opt-in.** Existing call sites
(`supplier_policies`, `get_prompt`, `prompts_for_agent`, etc.) are untouched.

## Component 3 — Reload endpoint

`src/api/routers/agents.py` already exposes `POST /agents/reload-policies` which
calls `agent_nick.policy_engine.reload_policies()`. Extend governance reload to
cover prompts too:

- Add `POST /agents/reload-governance` that calls both
  `agent_nick.policy_engine.reload_policies()` and
  `agent_nick.prompt_engine.refresh()`, returning per-engine row counts.
- Keep `POST /agents/reload-policies` as a backward-compatible alias (policies
  only) to avoid breaking existing callers.

Response shape:
```json
{"status": "success", "prompts": <int>, "policies": <int>}
```

## Component 4 — Migration + seed

New file `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql`, idempotent:

- `CREATE TABLE IF NOT EXISTS proc.bp_prompt (...)` + indexes (`IF NOT EXISTS`).
- `CREATE TABLE IF NOT EXISTS proc.bp_policy (...)` + indexes.
- Seed `INSERT`s guarded so re-running is a no-op. Because `prompt_id`/`policy_id`
  are IDENTITY, seeds are keyed on a natural unique (`prompt_name` / `policy_name`)
  using `INSERT ... SELECT ... WHERE NOT EXISTS (...)` (no unique constraint is
  added on name, to keep the governance UI free to create name variants/versions).

### Seed contents (from current hardcoded defaults)

**Prompts (`bp_prompt`):**
- Supplier-ranking justification template — `prompts_desc.prompt_template` =
  the default in `supplier_ranking_agent.py` (`"Supplier {supplier_name}
  achieved a final score of {final_score:.2f}.\n{score_breakdown}"`);
  `prompt_linked_agents = 'supplier_ranking_agent'`.
- `rank_by_criteria` query-decomposition template — seeded into
  `prompts_desc.templates[]` so `prompt_library()` / `deconstruct_query()` have a
  template; linked to the orchestrator/supplier-ranking flow.
- Key negotiation prompts referenced by `negotiation_agent.py`
  (`prompts_for_agent`/`get_prompt`) — enumerated during implementation;
  `prompt_linked_agents = 'negotiation_agent'`.

**Policies (`bp_policy`):**
- `weight_allocation_policy` — `policy_details.rules.default_weights` from the
  current supplier-ranking defaults (engine re-normalizes to sum 1).
- `categorical_scoring_policy`, `normalization_direction_policy` — supplier
  policy slugs the engine looks for (`SUPPLIER_POLICY_SLUGS`).
- Opportunity-policy metadata from `opportunity_miner_agent._get_policy_registry`:
  `price_variance_check`, `volume_consolidation_check`,
  `contract_expiry_check` (`default_conditions.negotiation_window_days = 90`),
  `supplier_risk_check`, `maverick_spend_check`, and the remaining registry
  entries — seeded as `policy_details` metadata (handlers stay in Python; only the
  governable conditions/thresholds live in the table). `policy_linked_agents`
  set to `opportunity_miner_agent`.

Exact values are lifted verbatim from the agent source during implementation so
behavior is unchanged.

### Apply

Run against `bp_sqldb` (same pattern as the recent rename migration):
```
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -v ON_ERROR_STOP=1 \
  -f deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql
```

## Testing

- **Existing engine unit tests** inject `prompt_rows` / `policy_rows` fixtures, so
  the table-name change does not affect them — they must keep passing.
- **New BaseAgent resolver tests** (`tests/agents/`): with a stub engine,
  `resolve_prompt` returns the agent-linked template, falls back to global, then
  `None`; `governing_policies` returns only linked policies; `governing_policy`
  honors the scope order.
- **Integration (post-seed)**: against a seeded DB, assert `PromptEngine` and
  `PolicyEngine` load `> 0` rows; assert `get_policy('weight_allocation_policy')`
  returns weights; assert `resolve_prompt` on the supplier-ranking agent returns
  the seeded justification template.
- **Startup verification**: after applying the migration and restarting procwise,
  confirm the journal no longer logs `Failed to load prompts/policies from
  database` and that `PolicyEngine loaded N policies` shows `N > 0`.

## Rollout / Blast Radius

| Change | Files |
|---|---|
| 2 engine one-line query repoints | `prompt_engine.py`, `policy_engine.py` |
| ~40-line resolver | `base_agent.py` |
| 1 reload endpoint (+ alias) | `api/routers/agents.py` |
| 1 idempotent migration + seed | `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql` |
| Tests | `tests/agents/`, `tests/...` engine/integration |

No existing agent behavior changes; resolver and endpoint are additive.

## Future (out of scope)

- `LISTEN/NOTIFY` channel (`bp_governance_changed`) for automatic hot-reload
  instead of the manual endpoint.
- Gradual migration of scattered agent call sites onto the resolver helpers.
- Auto-injection of the resolved system prompt into agent LLM calls.
