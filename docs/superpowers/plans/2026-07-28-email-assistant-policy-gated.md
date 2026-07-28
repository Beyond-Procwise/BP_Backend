# Policy-Gated Email Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a supplier replies, decide against a governed policy whether the agent answers it unattended or escalates it as a single item in the existing Action Centre Todo list, whose detail pane is the restored email review panel with attachments.

**Architecture:** A new `bp_policy` row holds the autonomy rules (shipped with auto-reply disabled). A new parallel, fail-closed `resolve_authority()` resolves that policy per agent and the orchestrator injects it as `context.input_data["authority"]`. `DecisionEngine.decide_email_reply()` gathers facts from `proc.supplier_response` + the matched draft, classifies intent with AgentNick behind a sentence-level grounding guard, applies the authority, and either dispatches through the existing threaded-send path or persists an escalation to `proc.bp_decision`. Escalations surface as Action Centre cards; the restored panel is the detail pane.

**Tech Stack:** Python 3.12 / FastAPI / psycopg2 / pytest (backend); PostgreSQL `proc` schema on `bp_sqldb`; vanilla-JS `engine.js` classic script + React data hook (SpendIQ UI, separate repo `/home/muthu/PycharmProjects/beyond_procwise_ui`).

**Spec:** `docs/superpowers/specs/2026-07-28-email-assistant-policy-gated-design.md`

## Global Constraints

- **AgentNick is the only base model.** Never propose, configure or fall back to qwen or any non-AgentNick model for the intent classification.
- **Never fabricate.** If a value is not in the source, leave it NULL/blank. No prefilled recipient addresses (`bp_supplier` contact emails are empty; all 18 live drafts have no recipient).
- **Authority fails closed.** Missing policy, unparseable rules, ungrounded classification, or low confidence → escalate. The existing fail-open `governed` prompt envelope in `orchestrator._apply_governance_envelope` must not change behaviour.
- **`deal_id` is owned by a DB stored procedure** — never assign it in app code. Read it only.
- **New tables take the `bp_` prefix** (this plan adds no tables; one column and one data row).
- **Migrations** live in `deploy/sql/YYYY-MM-DD_<name>.sql` with a matching `_rollback.sql`, wrapped in `BEGIN; … COMMIT;`, and must be idempotent.
- **`docs/` is gitignored** — commit plan/spec files with `git add -f`.
- **Branch is `Development`.** Never push to `main`.
- **No Claude attribution** in commit messages.
- **Run against `bp_sqldb`.** `.env` currently points `DB_NAME` at `bp_testdb`, where every table in this plan is empty. For live verification, override `DB_NAME=bp_sqldb` for that command only — do not edit `.env`.
- **Test command:** `venv/bin/python -m pytest <path> -v` from `/home/muthu/PycharmProjects/BP_Backend`.

## File Structure

**Backend — create**
- `deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql` (+ `_rollback.sql`) — the policy row.
- `deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql` (+ `_rollback.sql`) — the attachments column.
- `src/services/governance_tools/authority.py` — parallel, fail-closed authority resolution. Separate from `envelope.py` because the two have deliberately opposite failure semantics and mixing them in one module invites someone "tidying" one into the other.
- `src/services/email_intent.py` — AgentNick intent classification + grounded quote check for a supplier reply.
- `tests/governance/test_authority_resolution.py`
- `tests/services/test_email_intent.py`
- `tests/engines/test_decide_email_reply.py`
- `tests/api/test_email_attachments.py`

**Backend — modify**
- `src/engines/decision_engine.py` — add `_fetch_email_reply()`, `decide_email_reply()`, `execute_email_reply()`.
- `src/orchestration/orchestrator.py` — inject `authority` alongside the existing `governed` envelope.
- `src/api/routers/decisions.py` — `POST /decisions/email-reply/{response_id}`, `GET /decisions`.
- `src/api/routers/workflows.py` — attachment upload/delete endpoints.
- `src/services/email_dispatch_service.py` — read stored attachments and pass them to the MIME path.
- `src/services/obligations/grounding.py` — add a `min_words` parameter (default 8, so contract behaviour is byte-identical).

**UI — modify** (`/home/muthu/PycharmProjects/beyond_procwise_ui`)
- `src/modules/SpendIQ/data/useSpendData.js` — fetch the escalated email decisions, adapt them into the "Approve agent" queue.
- `src/modules/SpendIQ/engine.js` — email branch in `actionDetail()`: the restored panel.

---

### Task 1: The autonomy policy row

**Files:**
- Create: `deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql`
- Create: `deploy/sql/2026-07-28_bp_policy_email_reply_autonomy_rollback.sql`
- Test: `tests/governance/test_authority_resolution.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a `proc.bp_policy` row resolvable as `PolicyEngine.get_policy("email_reply_autonomy")`, returning `{"policy_type": "email_autonomy", "slug": "email_reply_autonomy", "details": {"rules": {...}}, "policyName": "EmailReplyAutonomyPolicy", "raw_row": {"policy_id": int, ...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/governance/test_authority_resolution.py
"""The autonomy policy must resolve by slug, and ship with auto-reply disabled."""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.policy_engine import PolicyEngine

# The rules body exactly as the migration writes it. If the migration and this
# literal drift, the engine reads something the test never checked.
AUTONOMY_RULES = {
    "auto_reply_intents": [],
    "escalate_intents": [
        "price_change", "terms_change", "contract_variation",
        "liability", "dispute", "new_commitment",
    ],
    "defer_value_limit_to": "approval_threshold",
    "max_auto_replies_per_thread": 2,
    "min_intent_confidence": 0.8,
    "on_missing_policy": "escalate",
    "on_ungrounded_facts": "escalate",
}


def _autonomy_row():
    return {
        "policy_id": 11,
        "policy_name": "EmailReplyAutonomyPolicy",
        "policy_type": "email_autonomy",
        "policy_desc": "When the email agent may reply unattended",
        "policy_details": json.dumps(
            {"policy_identifier": "email_reply_autonomy", "rules": AUTONOMY_RULES}
        ),
        "policy_linked_agents": "email_drafting_agent, negotiation_agent, supplier_interaction_agent",
    }


def test_autonomy_policy_resolves_by_slug():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    policy = engine.get_policy("email_reply_autonomy")
    assert policy is not None
    assert policy["policy_type"] == "email_autonomy"


def test_autonomy_policy_ships_with_auto_reply_disabled():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    rules = engine.get_policy("email_reply_autonomy")["details"]["rules"]
    # The conservative default IS the safety property: nothing auto-sends until a
    # human widens this list in the Policies screen.
    assert rules["auto_reply_intents"] == []
    assert rules["on_missing_policy"] == "escalate"


def test_autonomy_policy_resolves_by_linked_agent_alias():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    assert engine.get_policy("email_drafting_agent") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/governance/test_authority_resolution.py -v`
Expected: the first two PASS (they only exercise `PolicyEngine` with injected rows), `test_autonomy_policy_resolves_by_linked_agent_alias` may pass or fail depending on alias folding. **Confirm which**, then treat the failing one as the target. If all three pass, the injected-row path is already sound and the real gap is the missing database row — proceed to Step 3 and verify against the live database in Step 5.

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql
BEGIN;
-- The email agent's autonomy limit is governed DATA, not code: which supplier
-- replies it may answer unattended, and which must go to a human, is read from
-- this row at decision time (DecisionEngine.decide_email_reply via
-- resolve_authority). No threshold is hardcoded anywhere in the email path.
--
-- auto_reply_intents ships EMPTY on purpose. On day one every reply escalates to
-- the Action Centre; widening the list is a Policies-screen edit, not a deploy.
-- Money authority is NOT duplicated here -- defer_value_limit_to points at the
-- existing ApprovalThresholdPolicy so there is exactly one spend limit.
--
-- Keyed on (policy_type, policy_name), the natural key enforced by
-- ux_bp_governance_active_unique. policy_id is environment-specific.
-- Idempotent: re-running updates the same row rather than inserting a second.

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
VALUES (
    'EmailReplyAutonomyPolicy',
    'email_autonomy',
    'When the email agent may reply to a supplier unattended, and when it must escalate to a human.',
    jsonb_build_object(
      'policy_identifier', 'email_reply_autonomy',
      'rules', jsonb_build_object(
        'auto_reply_intents', '[]'::jsonb,
        'escalate_intents', '["price_change","terms_change","contract_variation","liability","dispute","new_commitment"]'::jsonb,
        'defer_value_limit_to', 'approval_threshold',
        'max_auto_replies_per_thread', 2,
        'min_intent_confidence', 0.8,
        'on_missing_policy', 'escalate',
        'on_ungrounded_facts', 'escalate'
      )
    ),
    'email_drafting_agent, negotiation_agent, supplier_interaction_agent',
    1,
    1,
    now(), 'implementation-plan', now(), 'implementation-plan'
)
ON CONFLICT (policy_type, policy_name) WHERE policy_status = 1
DO UPDATE SET
    policy_details     = EXCLUDED.policy_details,
    policy_linked_agents = EXCLUDED.policy_linked_agents,
    policy_desc        = EXCLUDED.policy_desc,
    version            = proc.bp_policy.version + 1,
    last_modified_date = now(),
    last_modified_by   = 'implementation-plan';

COMMIT;
```

```sql
-- deploy/sql/2026-07-28_bp_policy_email_reply_autonomy_rollback.sql
BEGIN;
-- Deactivate rather than delete: bp_decision rows reference policy_name/policy_id
-- and an audit trail that points at a vanished policy cannot be re-derived.
UPDATE proc.bp_policy
   SET policy_status = 0,
       last_modified_date = now(),
       last_modified_by = 'rollback-2026-07-28'
 WHERE policy_type = 'email_autonomy'
   AND policy_name = 'EmailReplyAutonomyPolicy';
COMMIT;
```

- [ ] **Step 4: Verify the ON CONFLICT target exists**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv("/home/muthu/PycharmProjects/BP_Backend/.env")
c = psycopg2.connect(host=os.getenv("DB_HOST"), dbname="bp_sqldb", user=os.getenv("DB_USER"),
                     password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT", "5432"))
cur = c.cursor()
cur.execute("""select indexname, indexdef from pg_indexes
               where schemaname='proc' and tablename='bp_policy'""")
for r in cur.fetchall(): print(r[0], "|", r[1])
PY
```
Expected: an index on `(policy_type, policy_name)` filtered on active rows (created by `deploy/sql/2026-07-18_bp_governance_active_unique.sql`). **If the index definition differs from the `ON CONFLICT` clause above, change the migration to match the real index** — a mismatched inference clause fails at runtime, not at write time.

- [ ] **Step 5: Apply the migration and confirm the row resolves live**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2, json
from dotenv import load_dotenv
load_dotenv("/home/muthu/PycharmProjects/BP_Backend/.env")
c = psycopg2.connect(host=os.getenv("DB_HOST"), dbname="bp_sqldb", user=os.getenv("DB_USER"),
                     password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT", "5432"))
cur = c.cursor()
cur.execute(open("deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql").read())
c.commit()
cur.execute("""select policy_id, policy_name, policy_linked_agents,
                      policy_details->'rules'->'auto_reply_intents'
                 from proc.bp_policy
                where policy_type='email_autonomy' and policy_status=1""")
print(cur.fetchall())
PY
```
Expected: one row, `auto_reply_intents` = `[]`, linked agents listing the three email agents. Re-run the same command: still exactly one row (idempotence proven, `version` incremented).

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql \
        deploy/sql/2026-07-28_bp_policy_email_reply_autonomy_rollback.sql \
        tests/governance/test_authority_resolution.py
git commit -m "feat(governance): email autonomy limit as policy data, auto-reply off by default"
```

---

### Task 2: Parallel, fail-closed authority resolution

**Files:**
- Create: `src/services/governance_tools/authority.py`
- Test: `tests/governance/test_authority_resolution.py` (extend)

**Interfaces:**
- Consumes: `PolicyEngine.get_policy(slug)` → normalised policy dict (Task 1).
- Produces:
  `resolve_authority(policy_engine, agents: Sequence[str], *, autonomy_slug: str = "email_reply_autonomy", approval_slug: str = "approval_threshold") -> Dict[str, Dict[str, Any]]`
  keyed by agent name, each value:
  ```python
  {"agent": str, "governed": bool, "slug": str|None, "policy_id": int|None,
   "policy_name": str|None, "auto_intents": list[str], "escalate_intents": list[str],
   "limit_gbp": str|None, "limit_currency": str|None,
   "max_auto_replies_per_thread": int|None, "min_intent_confidence": float|None,
   "reason": str}
  ```
  `governed=False` always means *escalate* — there is no other reading of it.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/governance/test_authority_resolution.py
import json

from src.services.governance_tools.authority import resolve_authority

_APPROVAL_ROW = {
    "policy_id": 10,
    "policy_name": "ApprovalThresholdPolicy",
    "policy_type": "approval",
    "policy_desc": "Spend authority",
    "policy_details": json.dumps({
        "policy_identifier": "approval_threshold",
        "rules": {"currency": "GBP", "default_threshold_gbp": 10000,
                  "on_above": "escalate", "on_at_or_below": "approve"},
    }),
    "policy_linked_agents": "approvals_agent",
}

AGENTS = ["email_drafting_agent", "negotiation_agent"]


def test_resolves_every_agent_with_the_governed_limit():
    engine = PolicyEngine(policy_rows=[_autonomy_row(), _APPROVAL_ROW])
    out = resolve_authority(engine, AGENTS)
    assert set(out) == set(AGENTS)
    for agent in AGENTS:
        block = out[agent]
        assert block["governed"] is True
        assert block["slug"] == "email_reply_autonomy"
        assert block["policy_name"] == "EmailReplyAutonomyPolicy"
        # The money limit is READ from the approval policy, never restated here.
        assert block["limit_gbp"] == "10000"
        assert block["limit_currency"] == "GBP"
        assert block["auto_intents"] == []
        assert "price_change" in block["escalate_intents"]
        assert block["max_auto_replies_per_thread"] == 2
        assert block["min_intent_confidence"] == 0.8


def test_missing_autonomy_policy_fails_closed():
    engine = PolicyEngine(policy_rows=[_APPROVAL_ROW])  # autonomy row absent
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    assert block["governed"] is False
    assert "email_reply_autonomy" in block["reason"]


def test_missing_approval_policy_fails_closed():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])  # approval row absent
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    # An autonomy rule that defers its money limit to a policy that does not exist
    # has no limit at all. Ungoverned money must not be spendable.
    assert block["governed"] is False
    assert "approval_threshold" in block["reason"]


def test_unparseable_rules_fail_closed():
    bad = dict(_autonomy_row())
    bad["policy_details"] = json.dumps({"policy_identifier": "email_reply_autonomy",
                                        "rules": "not-an-object"})
    engine = PolicyEngine(policy_rows=[bad, _APPROVAL_ROW])
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    assert block["governed"] is False


def test_a_raising_policy_engine_fails_closed_not_open():
    class Exploding:
        def get_policy(self, slug):
            raise RuntimeError("policy table unreachable")

    out = resolve_authority(Exploding(), AGENTS)
    assert out["email_drafting_agent"]["governed"] is False
    assert out["negotiation_agent"]["governed"] is False


def test_resolution_runs_concurrently():
    # Each agent's resolution is an independent read; serial round-trips are the
    # only cost. Assert overlap rather than wall-clock, which is flaky.
    import threading
    import time

    inside = []
    barrier_hit = threading.Event()

    class Slow:
        def get_policy(self, slug):
            inside.append(slug)
            if len(inside) >= 2:
                barrier_hit.set()
            time.sleep(0.15)
            return None

    resolve_authority(Slow(), ["a_agent", "b_agent", "c_agent"])
    assert barrier_hit.is_set(), "agents were resolved one after another"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/governance/test_authority_resolution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.governance_tools.authority'`

- [ ] **Step 3: Write the implementation**

```python
# src/services/governance_tools/authority.py
"""What is this agent allowed to send on its own — resolved, in parallel, fail-closed.

This is deliberately NOT `envelope.resolve_governance`, and the difference is the
whole point of a separate module:

  * `envelope` is FAIL-OPEN. A governance error there must never break a workflow,
    because what it carries is prompts and descriptive policy summaries.
  * this is FAIL-CLOSED. What it carries is an authority limit. A missing limit is
    not "carry on unlimited", it is "stop and ask a human" -- the same rule
    DecisionEngine already applies to a missing approval threshold, and the exact
    mistake ApprovalsAgent once made by defaulting a spend limit to 1000.

Do not "tidy" these two into one module. `governed: False` here means escalate.

It also resolves through `PolicyEngine.get_policy()` -- the engine's own selection
logic -- rather than a static workflow->agent map, so the row reported is the row
that would genuinely apply at run time. `node_governance` says plainly that it
cannot make that claim; this can, because it asks the engine.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_AUTONOMY_SLUG = "email_reply_autonomy"
DEFAULT_APPROVAL_SLUG = "approval_threshold"

# Resolution is IO-bound (one policy lookup per agent) and the agent count per run
# is small; a handful of threads is plenty and keeps the DB pool calm.
_MAX_WORKERS = 8


def _ungoverned(agent: str, reason: str) -> Dict[str, Any]:
    """The fail-closed block. Every field a caller reads is present and empty."""
    return {
        "agent": agent,
        "governed": False,
        "slug": None,
        "policy_id": None,
        "policy_name": None,
        "auto_intents": [],
        "escalate_intents": [],
        "limit_gbp": None,
        "limit_currency": None,
        "max_auto_replies_per_thread": None,
        "min_intent_confidence": None,
        "reason": reason,
    }


def _rules(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Rule body out of PolicyEngine's normalised shape (same accessor order as
    DecisionEngine._rules -- one convention, not two)."""
    if not policy:
        return {}
    details = policy.get("details") or policy.get("policy_details") or {}
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules") or policy.get("rules") or {}
    return rules if isinstance(rules, dict) else {}


def _ids(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = (policy or {}).get("raw_row") or {}
    return {
        "policy_id": raw.get("policy_id"),
        "policy_name": raw.get("policy_name") or (policy or {}).get("policyName"),
    }


def _str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v) for v in value if v is not None]


def _resolve_one(
    policy_engine: Any, agent: str, autonomy_slug: str, approval_slug: str
) -> Dict[str, Any]:
    try:
        autonomy = policy_engine.get_policy(autonomy_slug)
    except Exception:  # noqa: BLE001 - fail CLOSED, and say why
        log.exception("authority: autonomy policy lookup failed for %s", agent)
        return _ungoverned(agent, f"policy lookup for '{autonomy_slug}' raised")

    rules = _rules(autonomy)
    if not autonomy or not rules:
        return _ungoverned(
            agent,
            f"no usable governed policy '{autonomy_slug}' (rules absent or not an object)",
        )

    limit_gbp: Optional[str] = None
    limit_currency: Optional[str] = None
    deferred = rules.get("defer_value_limit_to")
    if deferred:
        try:
            approval = policy_engine.get_policy(str(deferred) or approval_slug)
        except Exception:  # noqa: BLE001
            log.exception("authority: approval policy lookup failed for %s", agent)
            return _ungoverned(agent, f"policy lookup for '{deferred}' raised")
        approval_rules = _rules(approval)
        threshold = approval_rules.get("default_threshold_gbp")
        if threshold is None:
            return _ungoverned(
                agent,
                f"autonomy defers its value limit to '{deferred}', which has no "
                "default_threshold_gbp -- there is no limit to enforce",
            )
        limit_gbp = str(threshold)
        limit_currency = str(approval_rules.get("currency") or "GBP")

    ids = _ids(autonomy)
    confidence = rules.get("min_intent_confidence")
    cap = rules.get("max_auto_replies_per_thread")
    return {
        "agent": agent,
        "governed": True,
        "slug": autonomy.get("slug") or autonomy_slug,
        "policy_id": ids["policy_id"],
        "policy_name": ids["policy_name"],
        "auto_intents": _str_list(rules.get("auto_reply_intents")),
        "escalate_intents": _str_list(rules.get("escalate_intents")),
        "limit_gbp": limit_gbp,
        "limit_currency": limit_currency,
        "max_auto_replies_per_thread": int(cap) if cap is not None else None,
        "min_intent_confidence": float(confidence) if confidence is not None else None,
        "reason": "resolved from governed policy",
    }


def resolve_authority(
    policy_engine: Any,
    agents: Sequence[str],
    *,
    autonomy_slug: str = DEFAULT_AUTONOMY_SLUG,
    approval_slug: str = DEFAULT_APPROVAL_SLUG,
) -> Dict[str, Dict[str, Any]]:
    """Resolve each agent's send authority concurrently. Never raises."""
    names = [str(a) for a in agents if a]
    if not names:
        return {}
    workers = min(_MAX_WORKERS, len(names))
    out: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="authority") as pool:
        futures = {
            pool.submit(_resolve_one, policy_engine, name, autonomy_slug, approval_slug): name
            for name in names
        }
        for future, name in futures.items():
            try:
                out[name] = future.result()
            except Exception:  # noqa: BLE001 - a thread that died is not a licence to send
                log.exception("authority resolution thread failed for %s", name)
                out[name] = _ungoverned(name, "authority resolution failed")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/governance/test_authority_resolution.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/governance_tools/authority.py tests/governance/test_authority_resolution.py
git commit -m "feat(governance): parallel fail-closed authority resolution for the email path

resolve_authority asks PolicyEngine for the row that would actually apply, rather
than reading the static workflow->agent map, and resolves the run's agents
concurrently. Missing policy, missing deferred spend limit, unparseable rules or a
raising engine all yield governed=False, which means escalate. The existing
fail-open prompt envelope is untouched."
```

---

### Task 3: Orchestrator injects the authority block

**Files:**
- Modify: `src/orchestration/orchestrator.py` (add `_apply_authority`; call it beside `_apply_governance_envelope` at line ~366)
- Test: `tests/governance/test_orchestrator_injects_authority.py`

**Interfaces:**
- Consumes: `resolve_authority()` (Task 2).
- Produces: `context.input_data["authority"]` and `enriched_input["authority"]` — the dict from Task 2, keyed by agent name — for email-path workflows.

- [ ] **Step 1: Write the failing test**

```python
# tests/governance/test_orchestrator_injects_authority.py
"""The orchestrator must hand the agent its limit, not merely resolve one.

Only 1 of 14 agents reads the existing `governed` envelope, because that envelope is
additive and nothing obliges anyone to read it. `authority` is different: the email
path reads it and refuses to send without it, so the injection has to be real.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from orchestration.orchestrator import Orchestrator

EMAIL_WORKFLOWS = ["supplier_interaction", "negotiation"]


def _orchestrator():
    # Construct without __init__: the real one builds AgentNick, a DB pool and every
    # agent. This test is about one method's behaviour, not wiring.
    orch = Orchestrator.__new__(Orchestrator)
    return orch


def test_authority_injected_for_email_workflows():
    orch = _orchestrator()
    resolved = {"email_drafting_agent": {"agent": "email_drafting_agent", "governed": True}}
    orch._resolve_authority_for = lambda agents: resolved  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    for workflow in EMAIL_WORKFLOWS:
        ctx, enriched = Ctx(), {}
        ctx.input_data = enriched
        out = orch._apply_authority(workflow, ctx, enriched)
        assert out == resolved
        assert enriched["authority"]["email_drafting_agent"]["governed"] is True
        assert ctx.input_data["authority"] is enriched["authority"]


def test_no_authority_for_extraction():
    orch = _orchestrator()
    orch._resolve_authority_for = lambda agents: {"x": {}}  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    ctx, enriched = Ctx(), {}
    ctx.input_data = enriched
    # document_extraction must stay deterministic and unaffected, exactly as the
    # existing governance envelope excludes it.
    assert orch._apply_authority("document_extraction", ctx, enriched) is None
    assert "authority" not in enriched


def test_resolution_failure_injects_ungoverned_not_nothing():
    orch = _orchestrator()

    def boom(agents):
        raise RuntimeError("resolver down")

    orch._resolve_authority_for = boom  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    ctx, enriched = Ctx(), {}
    ctx.input_data = enriched
    orch._apply_authority("negotiation", ctx, enriched)
    # Absent authority and ungoverned authority must be indistinguishable downstream,
    # and both must mean escalate. Injecting nothing would let a reader that forgets
    # to check treat it as "no restriction".
    block = enriched["authority"]["email_drafting_agent"]
    assert block["governed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/governance/test_orchestrator_injects_authority.py -v`
Expected: FAIL — `AttributeError: 'Orchestrator' object has no attribute '_apply_authority'`

- [ ] **Step 3: Write the implementation**

Add to `src/orchestration/orchestrator.py`, immediately after `_apply_governance_envelope`:

```python
    # Workflows whose agents can put mail in front of a supplier. Extraction is
    # excluded for the same reason the governance envelope excludes it: it must stay
    # deterministic.
    _AUTHORITY_WORKFLOWS = {
        "supplier_interaction": ["email_drafting_agent", "negotiation_agent",
                                 "supplier_interaction_agent"],
        "negotiation": ["email_drafting_agent", "negotiation_agent"],
        "email_drafting": ["email_drafting_agent"],
    }

    def _resolve_authority_for(self, agents):
        """Seam for tests; production path goes to the parallel resolver."""
        from src.services.governance_tools.authority import resolve_authority
        return resolve_authority(self.policy_engine, agents)

    def _apply_authority(self, workflow_name, context, enriched_input):
        """Resolve + inject each email agent's send authority. FAIL-CLOSED.

        Unlike _apply_governance_envelope (fail-open, prompts), a failure here still
        injects a block -- an ungoverned one. Downstream, `governed: False` means
        escalate, so a resolver outage stops sends rather than permitting them.
        """
        agents = self._AUTHORITY_WORKFLOWS.get(workflow_name)
        if not agents:
            return None
        try:
            resolved = self._resolve_authority_for(agents)
        except Exception:  # noqa: BLE001
            logger.exception("authority resolution failed for %s", workflow_name)
            resolved = {
                agent: {
                    "agent": agent,
                    "governed": False,
                    "slug": None,
                    "policy_id": None,
                    "policy_name": None,
                    "auto_intents": [],
                    "escalate_intents": [],
                    "limit_gbp": None,
                    "limit_currency": None,
                    "max_auto_replies_per_thread": None,
                    "min_intent_confidence": None,
                    "reason": "authority resolution failed",
                }
                for agent in agents
            }
        if isinstance(enriched_input, dict):
            enriched_input["authority"] = resolved
        try:
            context.input_data["authority"] = resolved
        except Exception:  # noqa: BLE001
            pass
        ungoverned = [a for a, b in resolved.items() if not b.get("governed")]
        logger.info(
            "authority resolved for %s: %d agents, %d ungoverned%s",
            workflow_name, len(resolved), len(ungoverned),
            f" ({', '.join(ungoverned)})" if ungoverned else "",
        )
        return resolved
```

Then call it directly after the existing envelope call (around line 366):

```python
            governance_applied = self._apply_governance_envelope(
                workflow_name, context, enriched_input)

            # Send authority for email-capable workflows. Separate from the envelope
            # above and fail-closed -- see services/governance_tools/authority.py.
            self._apply_authority(workflow_name, context, enriched_input)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/governance/test_orchestrator_injects_authority.py tests/governance/ -v`
Expected: all PASS, and no previously passing governance test regresses.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/orchestrator.py tests/governance/test_orchestrator_injects_authority.py
git commit -m "feat(orchestration): inject fail-closed send authority for email workflows"
```

---

### Task 4: Sentence grounding for short replies

**Files:**
- Modify: `src/services/obligations/grounding.py`
- Test: `tests/services/test_email_intent.py` (grounding cases; the module lands in Task 5)

**Interfaces:**
- Produces: `is_quote_grounded(quote: str, full_text: str, *, min_words: int = 8) -> bool` — default unchanged, so every existing obligation caller behaves byte-identically.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_email_intent.py
"""Grounding a supplier's sentence, without opening the digit hole.

The obligations guard requires 8+ words because a contract sentence is never
shorter. A supplier reply legitimately is: "We can offer 94,000.00 GBP." is five
words and is the single most important sentence in the thread. The floor has to be
a parameter, not a constant -- and lowering it must not reintroduce
extraction_v3.is_value_grounded's digit-signature fallback, which passes wholly
invented sentences.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from src.services.obligations.grounding import is_quote_grounded

REPLY = (
    "Thank you for the proposal. We can offer 94,000.00 GBP with 45 day payment "
    "terms and a 14 day lead time."
)


def test_short_supplier_sentence_grounds_with_a_lower_floor():
    assert is_quote_grounded("We can offer 94,000.00 GBP.", REPLY, min_words=4) is False
    # The trailing full stop is not in the source; the sentence itself is.
    assert is_quote_grounded("We can offer 94,000.00 GBP", REPLY, min_words=4) is True


def test_default_floor_is_unchanged_for_contract_callers():
    eight_words = "The Contractor shall indemnify the Authority in full"
    assert is_quote_grounded(eight_words, eight_words) is True
    assert is_quote_grounded("shall indemnify the Authority", eight_words) is False


def test_a_fabricated_sentence_never_grounds():
    # Digits present in the source, sentence not. This is the exact failure mode the
    # obligations guard exists to prevent -- it must survive the new parameter.
    assert is_quote_grounded(
        "We will absorb the 94,000.00 GBP increase entirely", REPLY, min_words=4
    ) is False


def test_empty_source_never_grounds():
    assert is_quote_grounded("We can offer 94,000.00 GBP", "", min_words=4) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/services/test_email_intent.py -v`
Expected: FAIL — `TypeError: is_quote_grounded() got an unexpected keyword argument 'min_words'`

- [ ] **Step 3: Write the implementation**

Replace the function in `src/services/obligations/grounding.py` (keep the module docstring, extend it):

```python
def is_quote_grounded(quote: str, full_text: str, *, min_words: int = MIN_QUOTE_WORDS) -> bool:
    """True only if ``quote`` appears verbatim (modulo case/whitespace) in ``full_text``.

    ``min_words`` defaults to the contract floor, so every existing caller is
    unchanged. The email path passes a lower floor because a supplier's most
    consequential sentence is often five words ("We can offer 94,000.00 GBP"),
    while a bare clause reference is still refused. What does NOT change with the
    floor is the rule that makes this guard safe: whole-quote containment, with no
    digit fallback, no date fallback, and no "cannot verify -> allow".
    """
    q = _norm(quote)
    if len(q.split()) < max(1, int(min_words)):
        return False
    if not _norm(full_text):
        return False
    return q in _norm(full_text)
```

- [ ] **Step 4: Run tests to verify they pass, and that obligations did not regress**

Run: `venv/bin/python -m pytest tests/services/test_email_intent.py -v`
Expected: PASS.
Run: `venv/bin/python -m pytest tests/ -k "obligation or grounding" -v`
Expected: PASS — no existing obligation test changes behaviour.

- [ ] **Step 5: Commit**

```bash
git add src/services/obligations/grounding.py tests/services/test_email_intent.py
git commit -m "feat(grounding): parameterise the quote-length floor for supplier replies"
```

---

### Task 5: Intent classification for a supplier reply

**Files:**
- Create: `src/services/email_intent.py`
- Test: `tests/services/test_email_intent.py` (extend)

**Interfaces:**
- Consumes: `is_quote_grounded(..., min_words=…)` (Task 4); AgentNick via `agent_nick` (the only permitted model).
- Produces:
  ```python
  @dataclass
  class ReplyIntent:
      intent: str          # e.g. "price_change" | "acknowledge" | "unclassified"
      confidence: float    # 0.0-1.0; 0.0 when unusable
      quote: str           # verbatim supporting sentence from the reply
      grounded: bool       # quote verified against the reply body
      reason: str          # why unusable, when it is
  classify_reply(body: str, *, agent_nick: Any, min_quote_words: int = 4) -> ReplyIntent
  KNOWN_INTENTS: tuple[str, ...]
  ```

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/services/test_email_intent.py
import json
from types import SimpleNamespace

from src.services.email_intent import ReplyIntent, classify_reply


def _nick(payload):
    """AgentNick stand-in: one chat call, returns a JSON string."""
    calls = []

    def chat(*args, **kwargs):
        calls.append((args, kwargs))
        return payload

    return SimpleNamespace(chat=chat, _calls=calls)


def test_classifies_a_price_change_with_a_grounded_quote():
    nick = _nick(json.dumps({
        "intent": "price_change",
        "confidence": 0.94,
        "quote": "We can offer 94,000.00 GBP",
    }))
    out = classify_reply(REPLY, agent_nick=nick)
    assert out.intent == "price_change"
    assert out.confidence == 0.94
    assert out.grounded is True


def test_an_ungrounded_quote_is_reported_not_trusted():
    nick = _nick(json.dumps({
        "intent": "price_change",
        "confidence": 0.99,
        "quote": "We will absorb the increase entirely",
    }))
    out = classify_reply(REPLY, agent_nick=nick)
    assert out.grounded is False
    # The claim is preserved for inspection; the caller decides (it escalates).
    assert out.intent == "price_change"
    assert "not found" in out.reason.lower()


def test_an_unknown_intent_becomes_unclassified():
    nick = _nick(json.dumps({"intent": "vibes", "confidence": 0.9, "quote": "We can offer 94,000.00 GBP"}))
    out = classify_reply(REPLY, agent_nick=nick)
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_unparseable_model_output_is_unusable_not_guessed():
    out = classify_reply(REPLY, agent_nick=_nick("I think it's a price change!"))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0
    assert out.grounded is False


def test_a_raising_model_is_unusable_not_fatal():
    def boom(*a, **k):
        raise RuntimeError("ollama down")

    out = classify_reply(REPLY, agent_nick=SimpleNamespace(chat=boom))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_empty_body_is_unusable():
    out = classify_reply("", agent_nick=_nick("{}"))
    assert out.intent == "unclassified"
    assert isinstance(out, ReplyIntent)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/services/test_email_intent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.email_intent'`

- [ ] **Step 3: Write the implementation**

```python
# src/services/email_intent.py
"""What is this supplier actually asking for -- and can we prove it from their words.

Two rules, inherited from the decision engine and the grounding work:

1. The model classifies; it does not decide. This returns an intent and a confidence,
   and the caller applies the governed policy. Nothing here sends anything.
2. Every classification must carry a verbatim sentence from the supplier's own reply,
   checked against the stored body. A classification we cannot point at is reported
   as ungrounded and the caller escalates. It is never quietly trusted, and never
   quietly discarded either -- an ungrounded claim is itself worth showing a human.

AgentNick is the only model used here. There is no fallback to another model: a
second model would answer differently and no one would know which one spoke.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.services.obligations.grounding import is_quote_grounded

log = logging.getLogger(__name__)

# The vocabulary the policy is written against. An intent outside this set is
# 'unclassified' -- policy cannot reason about a label it has never heard of, and a
# label the model invented is not evidence of anything.
KNOWN_INTENTS = (
    # consequential
    "price_change", "terms_change", "contract_variation",
    "liability", "dispute", "new_commitment",
    # routine
    "acknowledge", "confirm_receipt", "request_missing_document",
    "chase_no_response", "clarify_lead_time", "out_of_office",
)

UNCLASSIFIED = "unclassified"

_SYSTEM = (
    "You classify one inbound supplier email for a procurement team. "
    "Reply with JSON only, no prose, with exactly these keys: "
    '{"intent": <one of ' + "|".join(KNOWN_INTENTS) + '>, '
    '"confidence": <number between 0 and 1>, '
    '"quote": <one sentence copied VERBATIM from the email that supports the intent>}. '
    "The quote must be copied character-for-character from the email. Do not "
    "paraphrase, summarise, correct or complete it. If no sentence supports a "
    "classification, return intent 'unclassified' with confidence 0."
)


@dataclass
class ReplyIntent:
    intent: str
    confidence: float
    quote: str
    grounded: bool
    reason: str = ""


def _unusable(reason: str) -> ReplyIntent:
    return ReplyIntent(intent=UNCLASSIFIED, confidence=0.0, quote="", grounded=False, reason=reason)


def _extract_json(text: str) -> dict:
    """The first JSON object in the response. Models add prose despite instructions."""
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group(0))


def classify_reply(body: str, *, agent_nick: Any, min_quote_words: int = 4) -> ReplyIntent:
    """Classify ``body``, requiring a grounded verbatim quote. Never raises."""
    text = (body or "").strip()
    if not text:
        return _unusable("the reply has no body to classify")

    try:
        raw = agent_nick.chat(
            system=_SYSTEM,
            user=f"Email:\n{text}",
            think=False,          # reasoning models return an empty `response` otherwise
            temperature=0,
        )
    except Exception:  # noqa: BLE001
        log.exception("email intent classification failed")
        return _unusable("the classifier was unreachable")

    try:
        payload = _extract_json(raw if isinstance(raw, str) else str(raw))
    except Exception:  # noqa: BLE001
        return _unusable("the classifier did not return usable JSON")

    intent = str(payload.get("intent") or "").strip()
    if intent not in KNOWN_INTENTS:
        return _unusable(f"'{intent or 'missing'}' is not a governed intent")

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return _unusable("the classifier returned no usable confidence")
    confidence = max(0.0, min(1.0, confidence))

    quote = str(payload.get("quote") or "").strip()
    grounded = is_quote_grounded(quote, text, min_words=min_quote_words)
    reason = (
        "classified with a grounded quote" if grounded
        else "the supporting sentence was not found verbatim in the reply"
    )
    return ReplyIntent(
        intent=intent, confidence=confidence, quote=quote, grounded=grounded, reason=reason
    )
```

- [ ] **Step 4: Verify the AgentNick call signature before running**

Run: `grep -rn "def chat" src/agents/base_agent.py src/services/model_selector.py | head`
The `agent_nick.chat(...)` call above must match the real signature. **If it differs** (different parameter names, a client object rather than a method), adapt `classify_reply`'s call site and the test's `_nick` stand-in together, keeping `think=False` and `temperature=0`. Reasoning models return an empty `response` without `think=False`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/services/test_email_intent.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/services/email_intent.py tests/services/test_email_intent.py
git commit -m "feat(email): grounded intent classification for inbound supplier replies

AgentNick classifies; policy decides. Every classification must carry a sentence
copied verbatim from the supplier's reply, verified against the stored body. An
ungrounded claim is returned as ungrounded rather than trusted or dropped."
```

---

### Task 6: `DecisionEngine.decide_email_reply()`

**Files:**
- Modify: `src/engines/decision_engine.py`
- Test: `tests/engines/test_decide_email_reply.py`

**Interfaces:**
- Consumes: `resolve_authority()` block shape (Task 2, via `context.input_data["authority"]` or resolved directly); `classify_reply()` → `ReplyIntent` (Task 5); existing `Decision`, `Evidence`, `RESOLVED`, `ESCALATED`, `record()`.
- Produces:
  - `DecisionEngine._fetch_email_reply(response_id) -> Optional[Dict[str, Any]]`
  - `DecisionEngine.decide_email_reply(response_id, *, authority=None, requested=None) -> Decision` with `subject_type="email_reply"`, `subject_id=<draft unique_id or response id>`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engines/test_decide_email_reply.py
"""Every escalate trigger, in isolation, and the one path that auto-sends.

The default policy ships with auto_reply_intents empty, so the auto-send case here
uses a widened policy on purpose: it proves the mechanism works and that widening it
is the ONLY thing that enables a send.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from engines.decision_engine import DecisionEngine, ESCALATED, RESOLVED
from src.services.email_intent import ReplyIntent

GOVERNED = {
    "agent": "email_drafting_agent",
    "governed": True,
    "slug": "email_reply_autonomy",
    "policy_id": 11,
    "policy_name": "EmailReplyAutonomyPolicy",
    "auto_intents": [],
    "escalate_intents": ["price_change", "terms_change"],
    "limit_gbp": "10000",
    "limit_currency": "GBP",
    "max_auto_replies_per_thread": 2,
    "min_intent_confidence": 0.8,
    "reason": "resolved from governed policy",
}

REPLY_ROW = {
    "id": 1,
    "workflow_id": "wf-1",
    "unique_id": "wf-1-PeopleFirst",
    "supplier_id": "PeopleFirst HR Solutions Ltd",
    "response_subject": "RE: Negotiation",
    "response_text": "Thank you. We can offer 94,000.00 GBP with 45 day payment terms.",
    "response_from": "billing@peoplefirst.invalid",
    "round_number": 1,
    "match_confidence": 1.0,
    "price": 94000,
    "currency": "GBP",
    "payment_terms": "45 Days",
    "lead_time": 14,
    "prior_price": 90000,       # from the matched draft
    "prior_currency": "GBP",
    "deal_id": "DEAL-1",
    "auto_replies_on_thread": 0,
}


def _engine(*, row=None, intent=None):
    nick = SimpleNamespace(
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
        get_db_connection=MagicMock(side_effect=RuntimeError("db off in test")),
        chat=lambda **k: "{}",
    )
    eng = DecisionEngine(nick)
    eng._fetch_email_reply = lambda _id: (REPLY_ROW if row is None else row)  # type: ignore
    eng._classify = lambda body: (  # type: ignore
        intent or ReplyIntent("price_change", 0.95, "We can offer 94,000.00 GBP", True)
    )
    return eng


def test_a_consequential_intent_escalates():
    d = _engine().decide_email_reply("1", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert d.subject_type == "email_reply"
    assert d.subject_id == "wf-1-PeopleFirst"
    assert "price_change" in d.rationale


def test_value_over_the_governed_limit_escalates_even_for_a_routine_intent():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": "1000"}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    # 94,000 vs a prior 90,000 is 4,000 at stake, over a 1,000 limit.
    assert "4000" in d.rationale.replace(",", "") or "4,000" in d.rationale


def test_low_confidence_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.4, "Thank you.", True))
    assert eng.decide_email_reply("1", authority=auth).resolution == ESCALATED


def test_an_ungrounded_classification_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "invented sentence", False))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "ground" in d.rationale.lower()


def test_missing_authority_escalates_and_names_the_policy():
    d = _engine().decide_email_reply("1", authority=None)
    assert d.resolution == ESCALATED
    assert "email_reply_autonomy" in d.rationale


def test_ungoverned_authority_escalates():
    auth = {"agent": "email_drafting_agent", "governed": False,
            "auto_intents": [], "escalate_intents": [],
            "reason": "no usable governed policy 'email_reply_autonomy'"}
    d = _engine().decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "no usable governed policy" in d.rationale


def test_thread_cap_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "auto_replies_on_thread": 2}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "2" in d.rationale


def test_a_missing_reply_escalates_rather_than_inventing_a_subject():
    eng = _engine(row=None)
    eng._fetch_email_reply = lambda _id: None  # type: ignore
    d = eng.decide_email_reply("999", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert "999" in d.rationale


def test_a_widened_policy_permits_an_auto_send():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": None, "prior_price": None}   # nothing at stake
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.decision == "send"


def test_every_fact_carries_a_source():
    d = _engine().decide_email_reply("1", authority=GOVERNED)
    assert d.evidence, "a decision with no evidence is a guess"
    for item in d.evidence:
        assert item.source, f"fact {item.fact} has no source"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/engines/test_decide_email_reply.py -v`
Expected: FAIL — `AttributeError: 'DecisionEngine' object has no attribute 'decide_email_reply'`

- [ ] **Step 3: Write the implementation**

Add to `src/engines/decision_engine.py` after `decide_finding`:

```python
    # ------------------------------------------------------------------
    # Email replies
    # ------------------------------------------------------------------
    _EMAIL_SUBJECT_TYPE = "email_reply"

    def _fetch_email_reply(self, response_id: str) -> Optional[Dict[str, Any]]:
        """The supplier's reply, plus the offer it is replying to.

        Column names are the REAL ones on proc.supplier_response (checked against
        bp_sqldb): the body is `response_text`, the key is `id`, and the link back to
        what we sent is `unique_id` / `matched_sent_email_id`. `auto_replies_on_thread`
        is derived, not stored -- it counts what the agent has already sent
        unattended on this thread, which is what the per-thread cap governs.
        """
        sql = """
            SELECT sr.id, sr.workflow_id, sr.unique_id, sr.supplier_id,
                   sr.response_subject, sr.response_text, sr.response_from,
                   sr.round_number, sr.match_confidence,
                   sr.price, sr.currency, sr.payment_terms, sr.lead_time,
                   d.subject           AS draft_subject,
                   d.recipient_email   AS draft_recipient,
                   d.payload           AS draft_payload
              FROM proc.supplier_response sr
              LEFT JOIN proc.draft_rfq_emails d
                     ON d.unique_id = sr.unique_id
             WHERE sr.id::text = %s
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(response_id),))
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    record = dict(zip(cols, row))
                    # Prior offer: the price we last put to this supplier, read off the
                    # draft payload when the drafting agent recorded one. Absent is
                    # absent -- do not substitute the supplier's own number, which
                    # would make every reply look like it changed nothing.
                    payload = record.get("draft_payload")
                    if isinstance(payload, dict):
                        record["prior_price"] = payload.get("target_price") or payload.get("offer_price")
                        record["prior_currency"] = payload.get("currency")
                    cur.execute(
                        """
                        SELECT count(*) FROM proc.bp_decision
                         WHERE subject_type = %s AND subject_id = %s
                           AND resolution = %s AND decision = 'send'
                        """,
                        (self._EMAIL_SUBJECT_TYPE, record.get("unique_id"), RESOLVED),
                    )
                    record["auto_replies_on_thread"] = int((cur.fetchone() or [0])[0] or 0)
                    return record
        except Exception:
            logger.exception("failed to read supplier reply %s", response_id)
            return None

    def _classify(self, body: str):
        """Seam for tests; production path goes to the grounded classifier."""
        from src.services.email_intent import classify_reply
        return classify_reply(body, agent_nick=self.agent_nick)

    def decide_email_reply(
        self,
        response_id: str,
        *,
        authority: Optional[Dict[str, Any]] = None,
        requested: Optional[str] = None,
    ) -> Decision:
        """Answer it ourselves, or put it in front of a human -- and say which, and why.

        Deterministic in the part that matters: the model contributes an intent label
        and a quoted sentence, and every gate below is arithmetic and set membership
        over governed values. No model is asked whether to send.
        """
        facts: Dict[str, Any] = {}
        evidence: List[Evidence] = []

        row = self._fetch_email_reply(response_id)
        if not row:
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE,
                subject_id=str(response_id),
                decision="escalate",
                resolution=ESCALATED,
                rationale=(
                    f"Supplier reply {response_id} was not found in "
                    "proc.supplier_response, so there are no facts to decide on."
                ),
            )

        ref = str(row.get("id"))
        subject_id = str(row.get("unique_id") or response_id)
        for key in ("supplier_id", "response_subject", "response_from", "round_number",
                    "match_confidence", "price", "currency", "payment_terms", "lead_time"):
            value = row.get(key)
            if value is None:
                continue
            facts[key] = str(value) if isinstance(value, Decimal) else value
            evidence.append(Evidence(fact=key, value=facts[key],
                                     source=f"proc.supplier_response.{key}", reference=ref))

        policy_name = (authority or {}).get("policy_name")
        policy_id = (authority or {}).get("policy_id")

        def _escalate(rationale: str) -> Decision:
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE, subject_id=subject_id,
                decision="escalate", resolution=ESCALATED, rationale=rationale,
                policy_id=policy_id, policy_name=policy_name,
                facts=facts, evidence=evidence,
                deal_id=row.get("deal_id"), supplier_id=row.get("supplier_id"),
            )

        # 1. Authority. No governed limit means no unattended send -- the same rule
        #    that stops a missing approval threshold from auto-approving money.
        if not authority or not authority.get("governed"):
            reason = (authority or {}).get("reason") or (
                "no send authority was resolved for policy 'email_reply_autonomy'"
            )
            facts["authority"] = "ungoverned"
            evidence.append(Evidence(fact="authority", value="ungoverned",
                                     source="proc.bp_policy(email_reply_autonomy)",
                                     reference=reason))
            return _escalate(
                f"This reply needs a human because {reason}. Nothing is sent on an "
                "unknown limit."
            )

        # 2. Classification, with its quote checked against the supplier's own words.
        intent = self._classify(str(row.get("response_text") or ""))
        facts["intent"] = intent.intent
        facts["intent_confidence"] = intent.confidence
        evidence.append(Evidence(fact="intent", value=intent.intent,
                                 source="AgentNick classification of supplier_response.response_text",
                                 reference=ref))
        if intent.quote:
            evidence.append(Evidence(fact="supporting_sentence", value=intent.quote,
                                     source="proc.supplier_response.response_text",
                                     reference="verbatim" if intent.grounded else "NOT FOUND in source"))
        if not intent.grounded:
            return _escalate(
                f"The classification '{intent.intent}' could not be grounded: "
                f"{intent.reason}. An ungrounded reading of a supplier's message is not "
                "a basis for replying unattended."
            )

        min_conf = authority.get("min_intent_confidence")
        if min_conf is not None and intent.confidence < float(min_conf):
            return _escalate(
                f"Confidence in '{intent.intent}' is {intent.confidence:.2f}, below the "
                f"governed minimum of {float(min_conf):.2f}."
            )

        # 3. Governed intent lists.
        if intent.intent in (authority.get("escalate_intents") or []):
            return _escalate(
                f"'{intent.intent}' is a governed escalate-only intent under "
                f"{policy_name or 'the autonomy policy'}: a human decides this one."
            )
        if intent.intent not in (authority.get("auto_intents") or []):
            return _escalate(
                f"'{intent.intent}' is not on the governed auto-reply list, so it goes "
                "to a human. Widen auto_reply_intents in the policy to change that."
            )

        # 4. Value at stake against the governed spend limit. Derived, and the
        #    derivation is stated: two numbers, both cited above.
        price = self._num(row.get("price"))
        prior = self._num(row.get("prior_price"))
        limit = self._num(authority.get("limit_gbp"))
        if price is not None and prior is not None:
            at_stake = abs(price - prior)
            facts["value_at_stake"] = str(at_stake)
            evidence.append(Evidence(
                fact="value_at_stake", value=str(at_stake),
                source="abs(supplier_response.price - draft offer price)", reference=ref))
            if limit is not None and at_stake > limit:
                return _escalate(
                    f"The reply moves {at_stake} {row.get('currency') or ''} "
                    f"(supplier {price} against our {prior}), above the governed limit "
                    f"of {limit} {authority.get('limit_currency') or 'GBP'}."
                )
        elif price is not None and prior is None:
            # We can see their number but not ours. That is not "nothing at stake".
            return _escalate(
                f"The supplier quotes {price} {row.get('currency') or ''} but no prior "
                "offer is recorded on the draft, so the amount at stake cannot be "
                "computed. A human should compare these."
            )

        # 5. Per-thread cap on unattended replies.
        cap = authority.get("max_auto_replies_per_thread")
        already = int(row.get("auto_replies_on_thread") or 0)
        if cap is not None and already >= int(cap):
            facts["auto_replies_on_thread"] = already
            evidence.append(Evidence(fact="auto_replies_on_thread", value=already,
                                     source="proc.bp_decision (prior sends on this thread)",
                                     reference=subject_id))
            return _escalate(
                f"The agent has already answered this thread {already} time(s) "
                f"unattended, at the governed cap of {cap}. A human takes it from here."
            )

        return Decision(
            subject_type=self._EMAIL_SUBJECT_TYPE, subject_id=subject_id,
            decision="send", resolution=RESOLVED,
            rationale=(
                f"'{intent.intent}' is on the governed auto-reply list under "
                f"{policy_name or 'the autonomy policy'}, the supporting sentence is "
                f"verbatim from the supplier's reply, confidence is "
                f"{intent.confidence:.2f}, and nothing exceeds the governed limit."
            ),
            policy_id=policy_id, policy_name=policy_name,
            facts=facts, evidence=evidence,
            deal_id=row.get("deal_id"), supplier_id=row.get("supplier_id"),
        )
```

- [ ] **Step 4: Verify the join column before running the live path**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv("/home/muthu/PycharmProjects/BP_Backend/.env")
c = psycopg2.connect(host=os.getenv("DB_HOST"), dbname="bp_sqldb", user=os.getenv("DB_USER"),
                     password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT", "5432"))
cur = c.cursor()
cur.execute("""SELECT sr.id, sr.unique_id, d.unique_id, d.subject
                 FROM proc.supplier_response sr
                 LEFT JOIN proc.draft_rfq_emails d ON d.unique_id = sr.unique_id""")
print(cur.fetchall())
PY
```
Expected: the one live reply row. **If `d.unique_id` comes back NULL**, the two tables do not share that key for this row — fall back to joining on `sr.matched_sent_email_id = d.payload->>'message_id'` and record which join worked in the method's docstring. Do not silently leave a join that never matches.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/engines/test_decide_email_reply.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/engines/decision_engine.py tests/engines/test_decide_email_reply.py
git commit -m "feat(decisions): decide_email_reply — send unattended or escalate, on governed facts

Sibling to decide_finding, reusing Decision/Evidence/record. Every gate is
arithmetic or set membership over governed values; the model only labels the reply
and must quote it verbatim. Missing authority, ungrounded quote, low confidence, a
consequential intent, an over-limit move, an uncomputable amount, or the per-thread
cap each escalate with the reason stated."
```

---

### Task 7: API — decide a reply, and read the escalation queue

**Files:**
- Modify: `src/api/routers/decisions.py`
- Test: `tests/api/test_decisions_email_endpoints.py`

**Interfaces:**
- Consumes: `decide_email_reply()` (Task 6), `resolve_authority()` (Task 2), `record()`.
- Produces:
  - `POST /decisions/email-reply/{response_id}` → `{"decision": {...}, "decision_id": int|None}`
  - `GET /decisions?subject_type=email_reply&status=open&limit=100` → `{"data": [...], "total": int}` where each row carries `decision_id, subject_id, supplier_id, deal_id, decision, resolution, rationale, policy_name, facts, created_at`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_decisions_email_endpoints.py
"""The queue endpoint feeds the Todo list. It must return escalations only."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engines.decision_engine import Decision, ESCALATED
import api.routers.decisions as decisions_router


ROWS = [
    (7, "email_reply", "wf-1-PeopleFirst", "PeopleFirst HR Solutions Ltd", "DEAL-1",
     "escalate", "escalated", "price_change is escalate-only", "EmailReplyAutonomyPolicy",
     {"intent": "price_change"}, "2026-07-28T10:00:00+00:00"),
]


class _Cur:
    description = [("decision_id",), ("subject_type",), ("subject_id",), ("supplier_id",),
                   ("deal_id",), ("decision",), ("resolution",), ("rationale",),
                   ("policy_name",), ("facts",), ("created_at",)]

    def __init__(self):
        self.sql = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params or ()

    def fetchall(self):
        return ROWS

    def fetchone(self):
        return (len(ROWS),)


class _Conn:
    def __init__(self, cur):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur


@pytest.fixture()
def client():
    cur = _Cur()
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _Conn(cur),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    app.state._cur = cur
    return TestClient(app)


def test_queue_returns_escalated_email_decisions(client):
    res = client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1
    row = body["data"][0]
    assert row["decision_id"] == 7
    assert row["subject_id"] == "wf-1-PeopleFirst"
    assert row["policy_name"] == "EmailReplyAutonomyPolicy"


def test_queue_filters_on_escalations_in_sql_not_in_python(client):
    client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    sql = client.app.state._cur.sql.lower()
    # Filtering after the LIMIT would silently drop escalations off the end of a
    # busy queue -- the exact bug that pinned the findings badge at its page size.
    assert "resolution" in sql
    assert "where" in sql


def test_decide_endpoint_returns_the_decision_and_records_it(client, monkeypatch):
    captured = {}

    def fake_decide(self, response_id, *, authority=None, requested=None):
        captured["response_id"] = response_id
        captured["authority"] = authority
        return Decision(subject_type="email_reply", subject_id="wf-1-PeopleFirst",
                        decision="escalate", resolution=ESCALATED,
                        rationale="needs a human")

    monkeypatch.setattr("engines.decision_engine.DecisionEngine.decide_email_reply",
                        fake_decide, raising=True)
    monkeypatch.setattr("engines.decision_engine.DecisionEngine.record",
                        lambda self, d, **k: 42, raising=True)

    res = client.post("/decisions/email-reply/1")
    assert res.status_code == 200
    assert res.json()["decision_id"] == 42
    assert res.json()["decision"]["resolution"] == "escalated"
    assert captured["response_id"] == "1"
    # The endpoint must resolve authority itself: a caller-supplied limit would be a
    # limit chosen by the requester.
    assert captured["authority"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/api/test_decisions_email_endpoints.py -v`
Expected: FAIL — 404 on both routes.

- [ ] **Step 3: Write the implementation**

Add to `src/api/routers/decisions.py`:

```python
from fastapi import Query

_EMAIL_AGENT = "email_drafting_agent"


@router.post("/email-reply/{response_id}")
def decide_email_reply(
    response_id: str,
    body: DecideRequest | None = None,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Decide a supplier reply: answer it unattended, or escalate it to a human.

    Authority is resolved HERE, from the governed policy -- never taken from the
    request. A limit supplied by the caller is a limit chosen by the caller.
    """
    from engines.decision_engine import DecisionEngine
    from src.services.governance_tools.authority import resolve_authority

    authority = resolve_authority(agent_nick.policy_engine, [_EMAIL_AGENT]).get(_EMAIL_AGENT)
    engine = DecisionEngine(agent_nick)
    decision = engine.decide_email_reply(
        response_id,
        authority=authority,
        requested=(body.requested if body else None),
    )
    decision_id = engine.record(
        decision,
        workflow_id=(body.workflow_id if body else None),
        agent=_EMAIL_AGENT,
        created_by=(body.user_id if body and body.user_id else "system"),
    )
    payload = decision.to_dict()
    payload["decision_id"] = decision_id
    return {"decision": payload, "decision_id": decision_id}


@router.get("")
def list_decisions(
    subject_type: Optional[str] = Query(default=None),
    status: str = Query(default="open"),
    limit: int = Query(default=100, ge=1, le=500),
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Escalated decisions awaiting a human — the rows behind the Todo list.

    Only escalations are returned. A decision the agent resolved itself is not a
    task; it is an audit record, and putting it here would fill the list with work
    nobody has to do.

    The filters are in SQL, before the LIMIT, and `total` is the true server-side
    count rather than the page size.
    """
    where = ["d.resolution = 'escalated'"]
    params: list[Any] = []
    if subject_type:
        where.append("d.subject_type = %s")
        params.append(subject_type)
    if status:
        where.append("d.status = %s")
        params.append(status)
    clause = " AND ".join(where)

    sql = f"""
        SELECT d.decision_id, d.subject_type, d.subject_id, d.supplier_id, d.deal_id,
               d.decision, d.resolution, d.rationale, d.policy_name, d.facts, d.created_at
          FROM proc.bp_decision d
         WHERE {clause}
         ORDER BY d.created_at DESC
         LIMIT %s
    """
    count_sql = f"SELECT count(*) FROM proc.bp_decision d WHERE {clause}"
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params) + (limit,))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                cur.execute(count_sql, tuple(params))
                total = int((cur.fetchone() or [0])[0] or 0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to list decisions")
        raise HTTPException(status_code=500, detail=str(exc))

    for row in rows:
        created = row.get("created_at")
        if created is not None and not isinstance(created, str):
            row["created_at"] = created.isoformat()
    return {"data": rows, "total": total}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/api/test_decisions_email_endpoints.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/decisions.py tests/api/test_decisions_email_endpoints.py
git commit -m "feat(api): decide a supplier reply, and read the escalation queue

The queue returns escalations only -- an auto-resolved reply is an audit record,
not a task. Authority is resolved server-side from the governed policy so a caller
cannot supply its own limit."
```

---

### Task 8: Attachments — column, upload, delete, dispatch

**Files:**
- Create: `deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql` (+ `_rollback.sql`)
- Modify: `src/api/routers/workflows.py`, `src/services/email_dispatch_service.py`
- Test: `tests/api/test_email_attachments.py`

**Interfaces:**
- Produces:
  - `POST /workflows/email/{unique_id}/attachments` (multipart, field `files`) → `{"unique_id": str, "attachments": [{"filename","content_type","bytes","s3_key","added_by","added_at"}]}`
  - `DELETE /workflows/email/{unique_id}/attachments/{index}` → the same shape, minus the removed entry.
  - `EmailDispatchService._load_attachments(draft) -> list[tuple[bytes, str]]`, consumed by the existing `attachments` parameter of `send_draft`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_email_attachments.py
"""A user's attachment must reach the MIME path, and never the extraction pipeline.

The SMTP layer has always supported attachments (EmailService.send_email builds a
MIMEBase part per file). What was missing was any way to get one in: the endpoint
had no field and the draft table had no column.
"""
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from src.services.email_dispatch_service import EmailDispatchService


def test_dispatch_loads_stored_attachments_as_bytes_and_filename(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": [
        {"filename": "terms.pdf", "s3_key": "email-attachments/wf-1/terms.pdf", "bytes": 4},
    ]}
    monkeypatch.setattr(svc, "_read_s3_bytes", lambda key: b"PDF!", raising=False)
    assert svc._load_attachments(draft) == [(b"PDF!", "terms.pdf")]


def test_an_unreadable_attachment_is_skipped_not_faked(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": [
        {"filename": "gone.pdf", "s3_key": "email-attachments/wf-1/gone.pdf", "bytes": 9},
        {"filename": "ok.pdf", "s3_key": "email-attachments/wf-1/ok.pdf", "bytes": 2},
    ]}

    def read(key):
        if key.endswith("gone.pdf"):
            raise RuntimeError("no such key")
        return b"OK"

    monkeypatch.setattr(svc, "_read_s3_bytes", read, raising=False)
    # An empty part named terms.pdf would be a lie about what was sent.
    assert svc._load_attachments(draft) == [(b"OK", "ok.pdf")]


def test_no_attachments_is_none_not_an_empty_list(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    assert svc._load_attachments({}) == []


def test_a_json_string_column_is_tolerated(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": json.dumps([{"filename": "a.txt", "s3_key": "k", "bytes": 1}])}
    monkeypatch.setattr(svc, "_read_s3_bytes", lambda key: b"A", raising=False)
    assert svc._load_attachments(draft) == [(b"A", "a.txt")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/api/test_email_attachments.py -v`
Expected: FAIL — `AttributeError: 'EmailDispatchService' object has no attribute '_load_attachments'`

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql
BEGIN;
-- Attachments a human added to a draft before sending. The SMTP layer has always
-- been able to attach files (EmailService.send_email -> MIMEBase per file) and
-- EmailDispatchService.send_draft has always accepted them; there was simply
-- nowhere to record one, so no UI could offer it.
--
-- Bytes live in S3 under an email-attachments/ prefix -- deliberately NOT the
-- data-integration document path, which would ingest the file into the extraction
-- pipeline and raise findings against a supplier's own signed PDF.
--
-- Shape: [{"filename","content_type","bytes","s3_key","added_by","added_at"}]
ALTER TABLE proc.draft_rfq_emails
  ADD COLUMN IF NOT EXISTS attachments JSONB;
COMMIT;
```

```sql
-- deploy/sql/2026-07-28_draft_rfq_emails_attachments_rollback.sql
BEGIN;
ALTER TABLE proc.draft_rfq_emails DROP COLUMN IF EXISTS attachments;
COMMIT;
```

- [ ] **Step 4: Write the dispatch-side implementation**

In `src/services/email_dispatch_service.py`, add to `EmailDispatchService`:

```python
    # Attachment limits are configuration, not literals buried in a handler.
    _ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
    _ATTACHMENT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

    def _read_s3_bytes(self, s3_key: str) -> bytes:
        """Fetch one attachment's bytes. Separated so tests can stand it in."""
        import boto3
        from src.config.settings import get_settings  # existing settings accessor

        settings = get_settings()
        client = boto3.client("s3", region_name=getattr(settings, "aws_region", "eu-west-1"))
        bucket = getattr(settings, "s3_bucket", None) or getattr(settings, "aws_s3_bucket", None)
        obj = client.get_object(Bucket=bucket, Key=s3_key)
        return obj["Body"].read()

    def _load_attachments(self, draft: Dict[str, Any]) -> List[Tuple[bytes, str]]:
        """Stored attachments as (bytes, filename), ready for the MIME path.

        An attachment we cannot read is SKIPPED and logged, never sent as an empty
        part: a zero-byte file named terms.pdf is a false statement about what the
        supplier received.
        """
        raw = (draft or {}).get("attachments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning("draft attachments column held unparseable JSON")
                return []
        if not isinstance(raw, list):
            return []

        loaded: List[Tuple[bytes, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            key, filename = item.get("s3_key"), item.get("filename")
            if not key or not filename:
                continue
            try:
                data = self._read_s3_bytes(str(key))
            except Exception:  # noqa: BLE001
                logger.exception("attachment %s could not be read; not sending it", key)
                continue
            if not data:
                logger.warning("attachment %s read as empty; not sending it", key)
                continue
            loaded.append((data, str(filename)))
        return loaded
```

Then, in `send_draft`, default the parameter from storage when the caller passed none:

```python
        if attachments is None:
            attachments = self._load_attachments(draft) or None
```

Place this immediately after the draft has been hydrated (the `_hydrate_draft` result is in scope) and before the send call that receives `attachments`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/api/test_email_attachments.py -v`
Expected: all PASS. **Verify the settings accessor and bucket attribute names** with `grep -rn "s3_bucket\|aws_region" src/config/*.py | head` and correct `_read_s3_bytes` if they differ.

- [ ] **Step 6: Add the upload and delete endpoints**

In `src/api/routers/workflows.py`:

```python
_ATTACHMENT_ALLOWED_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".txt",
}


@router.post(
    "/email/{unique_id}/attachments",
    summary="Attach files to a persisted draft (no send)",
)
async def add_email_attachments(
    unique_id: str,
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="api"),
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Store attachments against a draft, in S3 plus a record on the draft row.

    The bytes go to an email-attachments/ prefix, NOT through
    /data-integration/presigned-url: that path feeds document extraction, and a
    supplier's countersigned contract arriving as an "uploaded document" would
    raise discrepancies against itself.
    """
    import json as _json
    from datetime import datetime, timezone

    from repositories import draft_rfq_emails_repo

    draft = draft_rfq_emails_repo.load_by_unique_id(unique_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"No draft for unique_id {unique_id}")

    existing = draft.get("attachments")
    if isinstance(existing, str):
        try:
            existing = _json.loads(existing)
        except (TypeError, ValueError):
            existing = []
    records: List[Dict[str, Any]] = list(existing or [])
    total = sum(int(r.get("bytes") or 0) for r in records)

    service = EmailDispatchService(agent_nick)
    rejected: List[Dict[str, str]] = []
    for upload in files:
        name = os.path.basename(upload.filename or "")
        suffix = os.path.splitext(name)[1].lower()
        if not name or suffix not in _ATTACHMENT_ALLOWED_SUFFIXES:
            rejected.append({"filename": name, "reason": f"file type {suffix or 'unknown'} not allowed"})
            continue
        data = await upload.read()
        if len(data) > service._ATTACHMENT_MAX_BYTES:
            rejected.append({"filename": name, "reason": "file exceeds the per-file limit"})
            continue
        if total + len(data) > service._ATTACHMENT_MAX_TOTAL_BYTES:
            rejected.append({"filename": name, "reason": "message would exceed the total attachment limit"})
            continue
        key = f"email-attachments/{unique_id}/{name}"
        try:
            await run_in_threadpool(service._write_s3_bytes, key, data, upload.content_type)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"filename": name, "reason": f"upload failed: {exc}"})
            continue
        total += len(data)
        records.append({
            "filename": name,
            "content_type": upload.content_type or "application/octet-stream",
            "bytes": len(data),
            "s3_key": key,
            "added_by": user_id,
            "added_at": datetime.now(timezone.utc).isoformat(),
        })

    _persist_draft_attachments(agent_nick, unique_id, records)
    return {"unique_id": unique_id, "attachments": records, "rejected": rejected}


@router.delete(
    "/email/{unique_id}/attachments/{index}",
    summary="Remove one attachment from a draft before sending",
)
def remove_email_attachment(
    unique_id: str,
    index: int,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    import json as _json

    from repositories import draft_rfq_emails_repo

    draft = draft_rfq_emails_repo.load_by_unique_id(unique_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"No draft for unique_id {unique_id}")
    existing = draft.get("attachments")
    if isinstance(existing, str):
        try:
            existing = _json.loads(existing)
        except (TypeError, ValueError):
            existing = []
    records = list(existing or [])
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail=f"No attachment at index {index}")
    records.pop(index)
    _persist_draft_attachments(agent_nick, unique_id, records)
    return {"unique_id": unique_id, "attachments": records}


def _persist_draft_attachments(agent_nick, unique_id: str, records: List[Dict[str, Any]]) -> None:
    """Write the attachment list onto the draft row. Raises on failure — a stored
    file with no record is invisible, and silence here would produce exactly that."""
    import json as _json

    with agent_nick.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE proc.draft_rfq_emails SET attachments = %s::jsonb, updated_on = now()"
                " WHERE unique_id = %s",
                (_json.dumps(records), unique_id),
            )
        conn.commit()
```

Add the matching writer beside `_read_s3_bytes` in `EmailDispatchService`:

```python
    def _write_s3_bytes(self, s3_key: str, data: bytes, content_type: Optional[str]) -> None:
        import boto3
        from src.config.settings import get_settings

        settings = get_settings()
        client = boto3.client("s3", region_name=getattr(settings, "aws_region", "eu-west-1"))
        bucket = getattr(settings, "s3_bucket", None) or getattr(settings, "aws_s3_bucket", None)
        client.put_object(
            Bucket=bucket, Key=s3_key, Body=data,
            ContentType=content_type or "application/octet-stream",
        )
```

Ensure `os`, `UploadFile`, `File`, `Form` and `run_in_threadpool` are imported in `workflows.py` (`UploadFile`/`File`/`Form` follow `src/api/routers/documents.py:15`).

- [ ] **Step 7: Apply the migration and prove the round trip live**

Run:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv("/home/muthu/PycharmProjects/BP_Backend/.env")
c = psycopg2.connect(host=os.getenv("DB_HOST"), dbname="bp_sqldb", user=os.getenv("DB_USER"),
                     password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT", "5432"))
cur = c.cursor()
cur.execute(open("deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql").read())
c.commit()
cur.execute("""select column_name, data_type from information_schema.columns
               where table_schema='proc' and table_name='draft_rfq_emails'
                 and column_name='attachments'""")
print(cur.fetchall())
PY
```
Expected: `[('attachments', 'jsonb')]`.

Then, with the server running (`DB_NAME=bp_sqldb`), upload against a real draft:
```bash
printf 'test attachment' > /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/46a3e1ba-afa1-428a-9ef1-23d04bdea1ee/scratchpad/att.txt
UID=$(DB_NAME=bp_sqldb venv/bin/python -c "
import os,psycopg2
from dotenv import load_dotenv; load_dotenv('.env')
c=psycopg2.connect(host=os.getenv('DB_HOST'),dbname='bp_sqldb',user=os.getenv('DB_USER'),password=os.getenv('DB_PASSWORD'),port=os.getenv('DB_PORT','5432'))
cur=c.cursor(); cur.execute('select unique_id from proc.draft_rfq_emails where unique_id is not null order by id desc limit 1'); print(cur.fetchone()[0])")
curl -s -X POST "http://localhost:8000/workflows/email/$UID/attachments" \
  -F "files=@/tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/46a3e1ba-afa1-428a-9ef1-23d04bdea1ee/scratchpad/att.txt" | head -20
```
Expected: JSON listing one attachment with an `s3_key`, `rejected: []`. Re-query the row and confirm the JSON persisted.

- [ ] **Step 8: Commit**

```bash
git add deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql \
        deploy/sql/2026-07-28_draft_rfq_emails_attachments_rollback.sql \
        src/api/routers/workflows.py src/services/email_dispatch_service.py \
        tests/api/test_email_attachments.py
git commit -m "feat(email): user attachments on a draft, stored and sent

Bytes to S3 under email-attachments/, recorded on the draft row, handed to the MIME
path that already worked. Deliberately not the document-ingestion upload route,
which would extract the attachment and raise findings against it. An attachment
that cannot be read is skipped and logged, never sent as an empty part."
```

---

### Task 9: UI — escalations in the Todo list

**Files:**
- Modify: `/home/muthu/PycharmProjects/beyond_procwise_ui/src/modules/SpendIQ/data/useSpendData.js`
- Modify: `/home/muthu/PycharmProjects/beyond_procwise_ui/src/modules/SpendIQ/data/endpoints.js` (coverage note)

**Interfaces:**
- Consumes: `GET /decisions?subject_type=email_reply&status=open` (Task 7) via the `qAI` helper (BP_Backend base URL).
- Produces: email items appended to `actions.queue.agent` and `actions.queue.all`, each shaped like `findingToAction`'s output plus `_kind: 'email_reply'`, `_decisionId`, `_subjectId`, and an `email` object `{to, from, re, body}` for `actionDetail`.

- [ ] **Step 1: Write the adapter**

In `useSpendData.js`, beside `adaptActionQueue`:

```javascript
// Escalated email decisions -> Todo items. GET /decisions returns escalations ONLY,
// so an auto-answered reply never lands here: it is an audit record, not a task.
// The mail icon already exists in the engine's Action Centre icon set (ACicon.mail).
function emailDecisionToAction(d) {
  const facts = d.facts || {};
  const supplier = d.supplier_id || 'Unknown supplier';
  const intent = facts.intent || 'reply';
  const stake = facts.value_at_stake;
  const tags = [['warn', 'Approve agent'], ['mut', intent]];
  if (d.policy_name) tags.push(['mut', d.policy_name]);
  if (stake) tags.push(['bad', `${stake} at stake`]);
  return {
    _id: d.decision_id, _type: 'agent', _kind: 'email_reply',
    _decisionId: d.decision_id, _subjectId: d.subject_id,
    ic: 'warn', icon: 'mail',
    title: `${supplier} — ${String(intent).replace(/_/g, ' ')}`,
    // The agent's own rationale, verbatim. Do not summarise it here: the whole
    // point is that the human reads why it escalated.
    desc: d.rationale || 'Escalated for review.',
    tags,
    btns: [['Review', ''], ['Send', 'primary']],
    case: d.deal_id || supplier,
    fields: [
      ['Supplier', supplier],
      ['Deal', d.deal_id],
      ['Intent', intent],
      ['Confidence', facts.intent_confidence],
      ['At stake', stake],
      ['Policy', d.policy_name],
      ['Raised', d.created_at],
    ].filter((r) => r[1] != null && r[1] !== ''),
    // The supplier's own sentence, as cited by the decision. Absent stays absent.
    quote: facts.supporting_sentence || null,
    dTitle: 'Supplier reply — approve agent', dSub: `${supplier} · ${intent}`,
    dec: ['Send reply', 'Reject'], acts: ['send', 'reject'],
  };
}

function adaptEmailDecisions(res) {
  const rows = res?.data;
  if (!Array.isArray(rows)) return [];
  return rows.map(emailDecisionToAction);
}
```

- [ ] **Step 2: Wire the query and merge into the queue**

Beside the existing `obligationSummary` / `obligationList` queries (~line 1424):

```javascript
  const emailDecisions = useQuery(
    qAI('email-decisions', '/decisions?subject_type=email_reply&status=open'),
  );
```

Where `adaptActionQueue(...)`'s output is spread into the data map, merge the email items:

```javascript
  const emailItems = adaptEmailDecisions(emailDecisions.data);
  const queue = adaptActionQueue(actions.data /* existing argument */);
  if (emailItems.length && queue['actions.queue.all']) {
    queue['actions.queue.all'] = [...emailItems, ...queue['actions.queue.all']];
    queue['actions.queue.agent'] = [...emailItems, ...(queue['actions.queue.agent'] || [])];
    queue['actions.queue.counts'] = {
      ...(queue['actions.queue.counts'] || {}),
      agent: (queue['actions.queue.counts']?.agent || 0) + emailItems.length,
    };
    queue['actions.queue.total'] = (queue['actions.queue.total'] || 0) + emailItems.length;
  }
```

Match the surrounding code's existing variable names when merging — read the block before editing rather than assuming `actions.data` is the argument in use.

- [ ] **Step 3: Update the coverage map**

In `endpoints.js`, extend the `Actions` view entry's `endpoints` array with `'GET /decisions?subject_type=email_reply (ai)'` and add to its `note`: `Supplier replies that need a person appear in Approve agent; ones the agent handled itself do not.`

- [ ] **Step 4: Verify in the browser**

Run the stack per `LOCAL_RUN.md` (BP_Backend on 8000 with `DB_NAME=bp_sqldb`, gateway 3001 with `node --experimental-global-webcrypto`, UI 3000). Open `http://localhost:3000/spendiq`, go to **Actions → Approve agent**.
Expected: with at least one escalated email decision in `proc.bp_decision`, a mail-icon card appears showing the supplier, the intent, and the agent's rationale. With none, the tab is unchanged. Confirm via the browser console that no request 404s.

- [ ] **Step 5: Commit (UI repo)**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/SpendIQ/data/useSpendData.js src/modules/SpendIQ/data/endpoints.js
git commit -m "feat(actions): supplier replies needing a person appear in Approve agent

Escalated email decisions only. A reply the agent answered itself is an audit
record, not a task, and does not enter the queue."
```

---

### Task 10: UI — the restored review panel as the detail pane

**Files:**
- Modify: `/home/muthu/PycharmProjects/beyond_procwise_ui/src/modules/SpendIQ/engine.js` (`actionDetail`, ~line 1964)

**Interfaces:**
- Consumes: the item shape from Task 9 (`_kind`, `_decisionId`, `_subjectId`, `email`, `quote`, `fields`).
- Produces: `siqEmailReplyPanel(item)` rendering the supplier's message beside the editable draft; handlers `siqEmailAttach(uniqueId, files)`, `siqEmailRemoveAttachment(uniqueId, index)`, `siqEmailRegenerate(uniqueId)`, `siqEmailSend(uniqueId, decisionId)`, `siqEmailReject(decisionId)`.

- [ ] **Step 1: Read the original before porting**

Run:
```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git show f989fab^:src/modules/HomeActions/Details/EmailDraft.jsx > /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/46a3e1ba-afa1-428a-9ef1-23d04bdea1ee/scratchpad/EmailDraft.original.jsx
wc -l /tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/46a3e1ba-afa1-428a-9ef1-23d04bdea1ee/scratchpad/EmailDraft.original.jsx
```
Keep from it: the supplier selector, sender/subject/recipient rows, the two-column layout, the contentEditable body with Edit/Save, Regenerate, Send. **Do not port** its Tone select (it was `readOnly` and never sent), its References select (bound to the `tone` field, options mislabelled "View Refrences"/"Informal"), its dead Regenerate handler, or its placeholder recipient/sender defaults.

- [ ] **Step 2: Add the email branch to `actionDetail`**

Immediately before the existing `if (s.email) {` block in `actionDetail(s)`, add:

```javascript
  // Supplier-reply escalations get the restored review panel instead of the generic
  // finding layout: the point of this card is a message and a reply to it, side by
  // side, which the generic evidence grid cannot show.
  if(s._kind==='email_reply'){ return siqEmailReplyPanel(s); }
```

Then add the panel and its handlers near the other Action Centre helpers:

```javascript
/* ---------------- Restored supplier-reply review panel ----------------
   Ported from the retired MUI panel (f989fab^ HomeActions/Details/EmailDraft.jsx)
   into engine.js, which is a classic script: every handler below has to be a global
   because the markup uses inline on* attributes.

   Three of the original's controls were cosmetic and are NOT reproduced: Tone was a
   readOnly select the backend never received, References was bound to the tone field
   with mislabelled options, and Regenerate had no handler. Tone is a real control
   here (it maps to the style engine's intent on the draft); References is replaced by
   the decision's cited sentence, which is a fact rather than an empty dropdown.

   What the original never had, and this does: the supplier's own message, beside the
   reply. Reviewing a response without its question is not reviewing. */
let siqEmailBody = {};      // {unique_id: edited html} — survives re-render
let siqEmailSubject = {};
let siqEmailTo = {};
let siqEmailTone = {};
let siqEmailAtt = {};       // {unique_id: [{filename,bytes,s3_key}]}
let siqEmailEditing = {};
let siqEmailBusy = {};
let siqEmailError = {};     // honest backend error text, never a fabricated success

function siqEmailReplyPanel(s){
  const uid=String(s._subjectId||'');
  const body=siqEmailBody[uid]!=null?siqEmailBody[uid]:(s.email&&s.email.body)||'';
  const subject=siqEmailSubject[uid]!=null?siqEmailSubject[uid]:(s.email&&s.email.re)||'';
  const to=siqEmailTo[uid]!=null?siqEmailTo[uid]:(s.email&&s.email.to)||'';
  const tone=siqEmailTone[uid]||'formal';
  const atts=siqEmailAtt[uid]||[];
  const editing=!!siqEmailEditing[uid];
  const busy=!!siqEmailBusy[uid];
  const err=siqEmailError[uid];
  const inbound=(s.email&&s.email.inbound)||'';
  const quote=s.quote?`<div class="policy"><div class="pc">Cited from the supplier's reply</div><div class="pt">"${s.quote}"</div></div>`:'';
  const attList=atts.length
    ? atts.map((a,i)=>`<div class="frow"><span class="fl">${a.filename} <span style="color:var(--ink-3)">${Math.round((a.bytes||0)/1024)} KB</span></span><button class="btn sm" onclick="siqEmailRemoveAttachment('${uid}',${i})">Remove</button></div>`).join('')
    : '<div class="p-sub">No attachments.</div>';
  return `<div class="p-title">${s.dTitle||'Supplier reply'}</div><div class="p-sub">${s.dSub||''}</div>
    <div class="dbox"><div class="dbox-t">Why this needs you</div><div class="rsn"><span class="rd"></span><span>${s.desc||''}</span></div></div>
    ${quote}
    <div class="ac-grid" style="grid-template-columns:1fr 1fr;gap:12px">
      <div class="dbox"><div class="dbox-t">The supplier wrote</div>
        <div class="email-b" style="max-height:240px;overflow:auto">${inbound||'<span class="p-sub">The message body was not stored with this decision.</span>'}</div></div>
      <div class="dbox"><div class="dbox-t">Your reply</div>
        <div class="frow"><span class="fl">To</span><input class="siq-in" value="${to}" placeholder="No address on file — type one" oninput="siqEmailTo['${uid}']=this.value"></div>
        <div class="frow"><span class="fl">Subject</span><input class="siq-in" value="${subject}" oninput="siqEmailSubject['${uid}']=this.value"></div>
        <div class="frow"><span class="fl">Tone</span>
          <select class="siq-in" onchange="siqEmailTone['${uid}']=this.value">
            <option value="formal" ${tone==='formal'?'selected':''}>Formal</option>
            <option value="direct" ${tone==='direct'?'selected':''}>Direct</option>
            <option value="warm" ${tone==='warm'?'selected':''}>Warm</option>
          </select></div>
        <div class="email-b" contenteditable="${editing}" oninput="siqEmailBody['${uid}']=this.innerHTML"
             style="max-height:240px;overflow:auto;${editing?'outline:2px solid var(--brand)':''}">${body}</div>
      </div>
    </div>
    <div class="dbox"><div class="dbox-t">Attachments</div>${attList}
      <input type="file" multiple onchange="siqEmailAttach('${uid}', this.files)" style="font-size:.72rem;margin-top:8px"></div>
    ${err?`<div class="dbox" style="border-color:#f3c0c0"><div class="dbox-t" style="color:#c22b2b">Not sent</div><div class="p-sub">${err}</div></div>`:''}
    <div class="dbox" style="margin-bottom:0"><div style="font-size:.8rem;font-weight:700;margin-bottom:9px">Decision</div>
      <div class="dec">
        <button class="btn primary" ${busy?'disabled':''} onclick="siqEmailSend('${uid}',${s._decisionId})">${busy?'Sending…':'Send reply'}</button>
        <button class="btn" ${busy?'disabled':''} onclick="siqEmailReject(${s._decisionId})">Reject</button>
        <button class="btn" onclick="siqEmailEditing['${uid}']=!siqEmailEditing['${uid}'];renderBody()">${editing?'Save':'Edit'}</button>
        <button class="btn" ${busy?'disabled':''} onclick="siqEmailRegenerate('${uid}')">Regenerate</button>
      </div></div>`;
}

async function siqEmailAttach(uid, files){
  if(!files||!files.length) return;
  if(typeof window.__SPENDIQ_API_AI_UPLOAD__!=='function'){ toast('Attachment bridge unavailable'); return; }
  siqEmailBusy[uid]=true; siqEmailError[uid]=null; renderBody();
  try{
    const res=await window.__SPENDIQ_API_AI_UPLOAD__(`/workflows/email/${encodeURIComponent(uid)}/attachments`, files);
    siqEmailAtt[uid]=res.attachments||[];
    // Rejections are shown, not swallowed: a file the user believes they attached
    // and we silently dropped is the worst possible outcome here.
    (res.rejected||[]).forEach(r=>toast(`${r.filename}: ${r.reason}`));
    if(!(res.rejected||[]).length) toast('Attached.');
  }catch(e){ siqEmailError[uid]='Attachment failed — '+((e&&e.message)||e); }
  siqEmailBusy[uid]=false; renderBody();
}

async function siqEmailRemoveAttachment(uid, index){
  if(typeof window.__SPENDIQ_API_AI_DELETE__!=='function'){ toast('Delete bridge unavailable'); return; }
  try{
    const res=await window.__SPENDIQ_API_AI_DELETE__(`/workflows/email/${encodeURIComponent(uid)}/attachments/${index}`);
    siqEmailAtt[uid]=res.attachments||[]; renderBody();
  }catch(e){ toast('Could not remove — '+((e&&e.message)||e)); }
}

async function siqEmailRegenerate(uid){
  if(typeof window.__SPENDIQ_API_AI_POST__!=='function'){ toast('Agent bridge unavailable'); return; }
  siqEmailBusy[uid]=true; renderBody();
  try{
    const res=await window.__SPENDIQ_API_AI_POST__('/workflows/email-drafting', {unique_id:uid, tone:siqEmailTone[uid]||'formal'});
    const draft=(res&&res.drafts&&res.drafts[0])||null;
    if(draft&&draft.body){ siqEmailBody[uid]=draft.body; if(draft.subject) siqEmailSubject[uid]=draft.subject; toast('Re-drafted.'); }
    else toast('The agent did not return a draft');
  }catch(e){ siqEmailError[uid]='Re-draft failed — '+((e&&e.message)||e); }
  siqEmailBusy[uid]=false; renderBody();
}

async function siqEmailSend(uid, decisionId){
  const to=(siqEmailTo[uid]||'').trim();
  if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)){ toast('Enter a recipient address first'); return; }
  siqEmailBusy[uid]=true; siqEmailError[uid]=null; renderBody();
  try{
    await window.__SPENDIQ_API_AI_POST__('/workflows/email', {
      unique_id: uid, recipients: [to],
      subject: siqEmailSubject[uid], body: siqEmailBody[uid],
    });
    // Only record the human action once the send actually returned.
    await window.__SPENDIQ_API_AI_POST__(`/decisions/finding/${decisionId}/action`, {action:'send', user_id:'ui'});
    toast('Reply sent.');
    if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__();
  }catch(e){ siqEmailError[uid]='Send failed — '+((e&&e.message)||e); }
  siqEmailBusy[uid]=false; renderBody();
}

async function siqEmailReject(decisionId){
  try{
    await window.__SPENDIQ_API_AI_POST__(`/decisions/finding/${decisionId}/action`, {action:'reject', user_id:'ui'});
    toast('Rejected — nothing was sent.');
    if(window.__SPENDIQ_REFETCH__) window.__SPENDIQ_REFETCH__();
  }catch(e){ toast('Could not record the rejection — '+((e&&e.message)||e)); }
}
```

- [ ] **Step 3: Confirm or add the bridges this panel needs**

Run: `grep -n "__SPENDIQ_API_AI_POST__\|__SPENDIQ_API_AI_UPLOAD__\|__SPENDIQ_API_AI_DELETE__\|siq-in" src/modules/SpendIQ/index.jsx src/modules/SpendIQ/styles.css | head -20`

`__SPENDIQ_API_AI_POST__` already exists (the current email panel uses it). If `__SPENDIQ_API_AI_UPLOAD__` / `__SPENDIQ_API_AI_DELETE__` do not, add them in `index.jsx` beside the existing bridges — multipart POST and DELETE against `AI_API`, **not** through `__SPENDIQ_UPLOAD__`, which presigns S3 for document extraction. If the `siq-in` input class does not exist in `styles.css`, add a minimal rule scoped under `.siq-root` matching the neighbouring form controls.

Also confirm the endpoint used by `siqEmailRegenerate` exists (`grep -n '"/email-drafting"\|email_drafting' ../BP_Backend/src/api/routers/*.py`). If there is no such route, point Regenerate at the existing agent route that returns drafts and adjust the payload accordingly — do not leave a button wired to a 404, which is the exact failure this port exists to remove.

- [ ] **Step 4: Verify in the browser**

With the stack running against `bp_sqldb`: open **Actions → Approve agent**, select the email card.
Expected: supplier's message on the left, editable reply on the right, tone selector, attachment list with a file picker, and Send disabled until a valid recipient is typed. Attach a small file and confirm it appears with its size; remove it and confirm it disappears. Trigger a deliberate failure (stop BP_Backend, click Send) and confirm the honest error renders inline rather than a success toast.

- [ ] **Step 5: Commit (UI repo)**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx src/modules/SpendIQ/styles.css
git commit -m "feat(actions): restore the supplier-reply review panel as the escalation detail pane

Ported from the retired MUI panel into engine.js. The supplier's message now sits
beside the draft, which the original never showed. Tone is a real control bound to
the style engine; the cosmetic References dropdown and the dead Regenerate handler
are gone. Failures render verbatim — no fabricated 'Sent!'."
```

---

### Task 11: End-to-end live demonstration

**Files:**
- Create: `scripts/demo_email_assistant.py`

**Interfaces:**
- Consumes: every prior task.
- Produces: a repeatable live proof against `bp_sqldb` — not tests.

- [ ] **Step 1: Write the demonstration script**

```python
# scripts/demo_email_assistant.py
"""Live proof of the policy-gated email assistant, against bp_sqldb.

Run with the API up:  DB_NAME=bp_sqldb venv/bin/python scripts/demo_email_assistant.py

Prints, in order: the governed policy that was resolved, the authority handed to the
agent, the decision made about a real supplier reply, and where it landed. Nothing
here is mocked; if a step cannot be proven it says so and stops.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB = dict(
    host=os.getenv("DB_HOST"), dbname="bp_sqldb", user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT", "5432"),
)


def main() -> int:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    print("== 1. governed policy ==")
    cur.execute(
        """SELECT policy_id, policy_name, policy_details->'rules'
             FROM proc.bp_policy
            WHERE policy_type='email_autonomy' AND policy_status=1"""
    )
    row = cur.fetchone()
    if not row:
        print("FAIL: EmailReplyAutonomyPolicy is not present or not active.")
        return 1
    print(f"   policy {row[0]} {row[1]}")
    print(f"   rules  {json.dumps(row[2])}")

    print("== 2. authority handed to the agent ==")
    from engines.policy_engine import PolicyEngine
    from src.services.governance_tools.authority import resolve_authority

    engine = PolicyEngine()
    authority = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    print(f"   governed={authority['governed']} limit={authority['limit_gbp']} "
          f"{authority['limit_currency']} auto={authority['auto_intents']}")
    if not authority["governed"]:
        print(f"   (fail-closed: {authority['reason']}) -- every reply will escalate")

    print("== 3. a real supplier reply ==")
    cur.execute("""SELECT id, supplier_id, left(response_text, 90)
                     FROM proc.supplier_response ORDER BY id DESC LIMIT 1""")
    reply = cur.fetchone()
    if not reply:
        print("SKIP: proc.supplier_response is empty — run the watcher or inject a reply.")
        return 0
    print(f"   reply {reply[0]} from {reply[1]}: {reply[2]!r}")

    print("== 4. the decision ==")
    from engines.decision_engine import DecisionEngine

    class Nick:
        policy_engine = engine

        def get_db_connection(self):
            return psycopg2.connect(**DB)

    decision = DecisionEngine(Nick()).decide_email_reply(str(reply[0]), authority=authority)
    print(f"   {decision.resolution.upper()}: {decision.rationale}")
    print(f"   facts: {json.dumps(decision.facts, default=str)}")
    for item in decision.evidence:
        print(f"   evidence: {item.fact} = {str(item.value)[:60]!r}  <- {item.source}")

    print("== 5. where it lands ==")
    if decision.escalated:
        print("   -> Action Centre 'Approve agent' (GET /decisions?subject_type=email_reply)")
    else:
        print("   -> sent unattended; audit only, no Todo item")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && DB_NAME=bp_sqldb venv/bin/python scripts/demo_email_assistant.py`
Expected: sections 1–5 print; the policy resolves with `auto_reply_intents: []`; the decision on the live reply is **ESCALATED** with the reason naming the governed intent list; every fact prints a source.

- [ ] **Step 3: Prove the fail-closed path**

Deactivate the policy, re-run, and confirm the decision still escalates and names the missing policy:
```bash
cd /home/muthu/PycharmProjects/BP_Backend
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv(".env")
c=psycopg2.connect(host=os.getenv("DB_HOST"),dbname="bp_sqldb",user=os.getenv("DB_USER"),
                   password=os.getenv("DB_PASSWORD"),port=os.getenv("DB_PORT","5432"))
cur=c.cursor()
cur.execute("update proc.bp_policy set policy_status=0 where policy_type='email_autonomy'")
c.commit(); print("policy deactivated")
PY
DB_NAME=bp_sqldb venv/bin/python scripts/demo_email_assistant.py
DB_NAME=bp_sqldb venv/bin/python - <<'PY'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv(".env")
c=psycopg2.connect(host=os.getenv("DB_HOST"),dbname="bp_sqldb",user=os.getenv("DB_USER"),
                   password=os.getenv("DB_PASSWORD"),port=os.getenv("DB_PORT","5432"))
cur=c.cursor()
cur.execute("update proc.bp_policy set policy_status=1 where policy_type='email_autonomy'")
c.commit(); print("policy restored")
PY
```
Expected: with the policy off, `governed=False` and the decision escalates citing `email_reply_autonomy`. **No send occurs in either run.**

- [ ] **Step 4: Prove the widened path sends (once, deliberately)**

Only if the user asks for a live send: widen `auto_reply_intents` to include a single routine intent, re-run, and confirm the decision resolves to `send` and the reply lands in the supplier thread. Then restore the empty list. Do not leave a widened policy behind.

- [ ] **Step 5: Commit**

```bash
git add scripts/demo_email_assistant.py
git commit -m "test(email): live demonstration of the policy-gated decision path"
```

---

## Self-Review

**Spec coverage:** §4.1 policy → Task 1. §4.2 parallel fail-closed resolution → Task 2. Orchestrator injection (§3 gap 2, §4.3) → Task 3. §4.4 decision, incl. grounded classification → Tasks 4, 5, 6. §4.5 auto path → Task 6 (`RESOLVED`/`send`) + existing dispatch. §4.6 queue + restored panel → Tasks 7, 9, 10. §4.7 attachments → Task 8. §4.8 inbound on demand → existing `POST /emailwatcher`, surfaced by the panel's refetch; **no new work, by design**. §5 data changes → Tasks 1, 8. §6 endpoints → Tasks 7, 8. §7 error handling → covered by the escalate branches (Task 6), rejected-attachment reporting (Task 8), inline send errors (Task 10). §8 testing → every task, plus Task 11 for the live demonstration.

**Known deviations from the spec, deliberate:** the spec listed `POST /decisions/email-reply/{response_id}` and `GET /decisions` as separate items; the human action on an escalation reuses the existing `POST /decisions/finding/{id}/action` rather than a new route, because `execute()` already records `actioned_by` / `override_reason` and a parallel route would fork that audit path. If that endpoint proves to be finding-specific in a way that rejects an email decision id, Task 10 Step 3's verification will catch it and a sibling route is then in scope.

**Type consistency:** the authority block keys defined in Task 2 (`governed`, `auto_intents`, `escalate_intents`, `limit_gbp`, `limit_currency`, `max_auto_replies_per_thread`, `min_intent_confidence`, `reason`) are the exact keys read in Task 3's fallback block and Task 6's gates. `ReplyIntent` fields (`intent`, `confidence`, `quote`, `grounded`, `reason`) defined in Task 5 are the ones consumed in Task 6. `_kind`/`_decisionId`/`_subjectId` produced in Task 9 are the ones read in Task 10.

**Unverified assumptions, each with a check step in the task that depends on it:** the `bp_policy` unique index shape (Task 1 Step 4), the `supplier_response`↔`draft_rfq_emails` join key (Task 6 Step 4), `agent_nick.chat`'s signature (Task 5 Step 4), the settings names for the S3 bucket and region (Task 8 Step 5), the UI bridge helpers and the re-draft route (Task 10 Step 3).
