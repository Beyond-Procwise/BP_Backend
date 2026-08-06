# Guardrail Enforcement Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one enforcement seam that irreversible actions must pass through, with all rule content stored in `proc.bp_policy`, and use it to close the four authorised holes: no role model, two HITL bypasses, droppable audit, and an ungoverned email send path.

**Architecture:** Python establishes facts (caller's role, content class, approval on file); `bp_policy` decides what those facts permit. A single `authorize()` gate evaluates policy with deny-beats-allow, default-deny for irreversible action classes, and fail-closed on any error. The email send path then runs five ordered checks inside `EmailDispatchService.send_draft`, the one choke point every caller passes through.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, PostgreSQL (`proc` schema on `bp_testdb`), pytest, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-08-06-guardrail-enforcement-layer-design.md`

## Global Constraints

- **Run tests with `./venv/bin/python -m pytest`**, never bare `pytest`. The venv is at `./venv`. Tests load credentials via `load_dotenv()` — follow the existing pattern in `tests/governance/test_active_governance_is_unique.py`.
- **Roughly 250 pre-existing collection errors** in the suite relate to `extraction_v3` and are unrelated to this work. Run only the test paths named in each task; do not try to fix unrelated collectors.
- **New tables and columns use the `bp_` prefix**; indexes use `ix_bp_<table>_<cols>`.
- **Never modify source documents or extracted source data.**
- **No `Co-Authored-By` lines in commit messages.**
- **Stay on the `Development` branch.** Never push to `main`.
- **Every guard must be proven to fail before it is trusted.** Each task writes the test that breaks the guard on purpose, runs it, and watches it go red before the implementation lands. A guard that has never gone red has not been shown to work.
- **No LLM call may appear inside a gate or detector.** Every decision path is deterministic and reproducible.
- **Fail closed everywhere.** A missing, unparseable or unreachable policy is a deny, never a pass.
- Existing behaviour classes, copied verbatim from the spec: `read`, `compute`, `write`, `communicate`, `transact`, `share`, `configure`, `delegate`. Irreversible set: `communicate`, `transact`, `share`, `configure`, `delegate`.
- Sensitivity classes and order, verbatim: `public`=1, `internal`=2, `commercial_confidential`=3, `personal`=4.
- Roles and ranks, verbatim: `Viewer`=1, `Buyer`=2, `Approver`=3, `Admin`=4.

---

## File Structure

| File | Responsibility |
|---|---|
| `deploy/sql/2026-08-06_guardrail_enforcement.sql` | Schema + policy rows |
| `deploy/sql/2026-08-06_guardrail_enforcement_rollback.sql` | Reverse of the above |
| `src/services/rbac.py` | Resolve a principal to a role; answer "may this role do this action class" |
| `src/services/guardrail.py` | The `authorize()` gate and its `Decision` type |
| `src/services/email_sensitivity.py` | Deterministic content detectors and `classify()` |
| `src/services/approval_store.py` | Read and write `proc.bp_approval` |
| `src/engines/policy_engine.py` | Extended with `policies_for_action()` |
| `src/services/agent_actions.py` | Extended with `record_action_or_fail()` |
| `src/api/auth.py` | `Principal` gains `roles` / `role` |
| `src/services/email_dispatch_service.py` | The five ordered checks |
| `src/agents/negotiation_agent.py` | HITL bypass removal |

Tests mirror the source path under `tests/guardrails/`.

---

### Task 1: Schema and policy rows

**Files:**
- Create: `deploy/sql/2026-08-06_guardrail_enforcement.sql`
- Create: `deploy/sql/2026-08-06_guardrail_enforcement_rollback.sql`
- Test: `tests/guardrails/test_guardrail_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `proc.bp_supplier.clearance_level` (text, NOT NULL, default `'internal'`); table `proc.bp_role_assignment(assignment_id, subject, role, granted_by, granted_at, revoked_at)`; six active rows in `proc.bp_policy` with `policy_identifier` values `role_definition`, `role_assignment`, `email_dispatch_approval`, `email_recipient_allowlist`, `email_sensitivity`, `email_volume`.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/__init__.py` (empty file) and `tests/guardrails/test_guardrail_schema.py`:

```python
"""The guardrail layer's storage must exist before anything can enforce it.

These assert the deploy SQL has been applied: without the clearance column
every supplier is unclassifiable, and without the six policy rows the gate
has nothing to read and (correctly) denies everything.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()

REQUIRED_POLICIES = {
    "role_definition",
    "role_assignment",
    "email_dispatch_approval",
    "email_recipient_allowlist",
    "email_sensitivity",
    "email_volume",
}


@pytest.fixture(scope="module")
def conn():
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )
    yield connection
    connection.close()


def test_supplier_has_clearance_level_defaulting_to_internal(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_supplier' "
        "AND column_name='clearance_level'"
    )
    row = cur.fetchone()
    assert row is not None, "bp_supplier.clearance_level is missing"
    data_type, is_nullable, default = row
    assert data_type == "text"
    assert is_nullable == "NO"
    assert "internal" in (default or "")


def test_every_existing_supplier_has_a_clearance(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FROM proc.bp_supplier "
        "WHERE clearance_level IS NULL OR clearance_level = ''"
    )
    assert cur.fetchone()[0] == 0


def test_role_assignment_table_exists(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_role_assignment'"
    )
    columns = {r[0] for r in cur.fetchall()}
    assert {
        "assignment_id",
        "subject",
        "role",
        "granted_by",
        "granted_at",
        "revoked_at",
    } <= columns


def test_six_guardrail_policies_are_active(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT policy_details->>'policy_identifier' FROM proc.bp_policy "
        "WHERE policy_status = 1"
    )
    identifiers = {r[0] for r in cur.fetchall() if r[0]}
    missing = REQUIRED_POLICIES - identifiers
    assert not missing, f"missing guardrail policies: {sorted(missing)}"


def test_guardrail_policies_declare_a_required_role(conn):
    """The RBAC ruling is a first-class key, not an afterthought."""
    cur = conn.cursor()
    cur.execute(
        "SELECT policy_details->>'policy_identifier', policy_details->>'required_role' "
        "FROM proc.bp_policy WHERE policy_status = 1 "
        "AND policy_details->>'policy_identifier' = ANY(%s)",
        (sorted(REQUIRED_POLICIES),),
    )
    for identifier, required_role in cur.fetchall():
        assert required_role in {"Viewer", "Buyer", "Approver", "Admin"}, (
            f"{identifier} has no usable required_role"
        )
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_guardrail_schema.py -v`
Expected: all five tests FAIL — the column, table and policy rows do not exist yet.

- [ ] **Step 3: Write the deploy SQL**

Create `deploy/sql/2026-08-06_guardrail_enforcement.sql`:

```sql
-- Guardrail enforcement layer: storage + policy content.
-- Spec: docs/superpowers/specs/2026-08-06-guardrail-enforcement-layer-design.md
BEGIN;

-- 1. Supplier clearance. Existing rows default to 'internal', which keeps
-- routine RFQ traffic flowing while blocking commercial-confidential and
-- personal content to suppliers nobody has cleared.
ALTER TABLE proc.bp_supplier
    ADD COLUMN IF NOT EXISTS clearance_level text NOT NULL DEFAULT 'internal';

-- 2. Direct subject -> role overrides. Permissions live in bp_policy; only
-- the mapping is tabular.
CREATE TABLE IF NOT EXISTS proc.bp_role_assignment (
    assignment_id bigserial PRIMARY KEY,
    subject       text NOT NULL,
    role          text NOT NULL,
    granted_by    text,
    granted_at    timestamptz NOT NULL DEFAULT now(),
    revoked_at    timestamptz
);

CREATE INDEX IF NOT EXISTS ix_bp_role_assignment_subject
    ON proc.bp_role_assignment (subject)
    WHERE revoked_at IS NULL;

-- 3. Policy content. Six rows, all active.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
VALUES
(
 'RoleDefinitionPolicy', 'security',
 'The four roles and the action classes each may perform.',
 '{
   "policy_identifier": "role_definition",
   "required_role": "Admin",
   "rules": {
     "roles": {
       "Viewer":   {"rank": 1, "allow": ["read"]},
       "Buyer":    {"rank": 2, "allow": ["read","compute","write"]},
       "Approver": {"rank": 3, "allow": ["read","compute","write","communicate","transact"]},
       "Admin":    {"rank": 4, "allow": ["read","compute","write","communicate","transact","share","configure","delegate"]}
     },
     "irreversible_classes": ["communicate","transact","share","configure","delegate"],
     "on_missing_role": "deny"
   }
 }'::jsonb,
 '', 1, 1, 'guardrail_migration', now()
),
(
 'RoleAssignmentPolicy', 'security',
 'Cognito group to role mapping and the no-principal default.',
 '{
   "policy_identifier": "role_assignment",
   "required_role": "Admin",
   "rules": {
     "claim": "cognito:groups",
     "group_to_role": {
       "bp-viewers": "Viewer",
       "bp-buyers": "Buyer",
       "bp-approvers": "Approver",
       "bp-admins": "Admin"
     },
     "no_principal_role": "Viewer",
     "unmapped_group_role": "Viewer",
     "multiple_groups": "highest_rank"
   }
 }'::jsonb,
 '', 1, 1, 'guardrail_migration', now()
),
(
 'EmailDispatchApprovalPolicy', 'email',
 'Outbound mail requires an approval verified against the store.',
 '{
   "policy_identifier": "email_dispatch_approval",
   "required_role": "Approver",
   "rules": {
     "approval_required": true,
     "verify_against": "proc.bp_approval",
     "accepted_status": ["approved"],
     "require_actioned_by": true,
     "trust_input_payload": false,
     "on_missing_approval": "deny"
   }
 }'::jsonb,
 'email_dispatch_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailRecipientAllowlistPolicy', 'email',
 'Recipients must appear on the supplier master.',
 '{
   "policy_identifier": "email_recipient_allowlist",
   "required_role": "Approver",
   "rules": {
     "sources": ["proc.bp_supplier.contact_email_1","proc.bp_supplier.contact_email_2"],
     "match": "exact_casefold",
     "on_unknown_recipient": "deny_and_raise_review",
     "allow_recipients_from_email_body": false,
     "new_domain_requires_confirmation": true
   }
 }'::jsonb,
 'email_dispatch_agent, supplier_interaction_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailSensitivityPolicy', 'email',
 'Outbound content class must not exceed the recipient supplier clearance.',
 '{
   "policy_identifier": "email_sensitivity",
   "required_role": "Admin",
   "rules": {
     "classes": ["public","internal","commercial_confidential","personal"],
     "order": {"public": 1, "internal": 2, "commercial_confidential": 3, "personal": 4},
     "default_supplier_clearance": "internal",
     "detectors": {
       "third_party_price":        {"enabled": true, "raises_to": "commercial_confidential"},
       "contract_prose":           {"enabled": true, "raises_to": "commercial_confidential"},
       "internal_staff_contact":   {"enabled": true, "raises_to": "personal"},
       "source_document_attached": {"enabled": true, "raises_to": "commercial_confidential"}
     },
     "rule": "content_class <= recipient_clearance",
     "on_undetermined_class": "deny",
     "on_missing_clearance": "use_default"
   }
 }'::jsonb,
 'email_dispatch_agent, email_drafting_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailVolumePolicy', 'email',
 'Per-run and per-user outbound caps.',
 '{
   "policy_identifier": "email_volume",
   "required_role": "Admin",
   "rules": {
     "max_per_run": 20,
     "max_per_user_per_day": 50,
     "on_exceeded": "pause_and_raise_review"
   }
 }'::jsonb,
 'email_dispatch_agent', 1, 1, 'guardrail_migration', now()
);

COMMIT;
```

- [ ] **Step 4: Write the rollback SQL**

Create `deploy/sql/2026-08-06_guardrail_enforcement_rollback.sql`:

```sql
BEGIN;

DELETE FROM proc.bp_policy
WHERE policy_details->>'policy_identifier' IN (
    'role_definition',
    'role_assignment',
    'email_dispatch_approval',
    'email_recipient_allowlist',
    'email_sensitivity',
    'email_volume'
);

DROP INDEX IF EXISTS proc.ix_bp_role_assignment_subject;
DROP TABLE IF EXISTS proc.bp_role_assignment;

ALTER TABLE proc.bp_supplier DROP COLUMN IF EXISTS clearance_level;

COMMIT;
```

- [ ] **Step 5: Apply the migration**

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-08-06_guardrail_enforcement.sql
```

Expected: `COMMIT` with no errors.

- [ ] **Step 6: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_guardrail_schema.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add deploy/sql/2026-08-06_guardrail_enforcement.sql \
        deploy/sql/2026-08-06_guardrail_enforcement_rollback.sql \
        tests/guardrails/__init__.py \
        tests/guardrails/test_guardrail_schema.py
git commit -m "feat(guardrails): supplier clearance, role assignment table, six policy rows"
```

---

### Task 2: Role resolution (`rbac.py`)

**Files:**
- Create: `src/services/rbac.py`
- Test: `tests/guardrails/test_rbac.py`

**Interfaces:**
- Consumes: `PolicyEngine.get_policy(slug)` from `src/engines/policy_engine.py`, which returns a dict shaped `{"policyId","policyName","policy_desc","policy_type","details","aliases","slug","policy_linked_agents","raw_row"}` — the rules live at `policy["details"]["rules"]`.
- Produces:
  - `ROLE_UNKNOWN: str = "Viewer"`
  - `resolve_roles(principal, policy_engine=None) -> list[str]`
  - `effective_role(principal, policy_engine=None) -> str`
  - `role_rank(role, policy_engine=None) -> int`
  - `may(role, action_class, policy_engine=None) -> bool`
  - `is_irreversible(action_class, policy_engine=None) -> bool`

  `principal` is an `api.auth.Principal` or `None`. Every function is safe to call with `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_rbac.py`:

```python
"""Roles come from policy, never from Python.

The load-bearing cases are the unhappy ones: no principal at all, a group
nobody mapped, and a policy that failed to load. All three must land on the
least-privileged answer rather than the most convenient one.
"""

import pytest

from src.services import rbac


class FakePolicyEngine:
    """Stands in for PolicyEngine with the row shape it really returns."""

    def __init__(self, policies):
        self._policies = policies

    def get_policy(self, slug):
        return self._policies.get(slug)


class FakePrincipal:
    def __init__(self, subject, claims=None):
        self.subject = subject
        self.claims = claims or {}


ROLE_DEFINITION = {
    "policyId": "role_definition",
    "details": {
        "policy_identifier": "role_definition",
        "required_role": "Admin",
        "rules": {
            "roles": {
                "Viewer": {"rank": 1, "allow": ["read"]},
                "Buyer": {"rank": 2, "allow": ["read", "compute", "write"]},
                "Approver": {
                    "rank": 3,
                    "allow": ["read", "compute", "write", "communicate", "transact"],
                },
                "Admin": {
                    "rank": 4,
                    "allow": [
                        "read", "compute", "write", "communicate",
                        "transact", "share", "configure", "delegate",
                    ],
                },
            },
            "irreversible_classes": [
                "communicate", "transact", "share", "configure", "delegate",
            ],
            "on_missing_role": "deny",
        },
    },
}

ROLE_ASSIGNMENT = {
    "policyId": "role_assignment",
    "details": {
        "policy_identifier": "role_assignment",
        "required_role": "Admin",
        "rules": {
            "claim": "cognito:groups",
            "group_to_role": {
                "bp-viewers": "Viewer",
                "bp-buyers": "Buyer",
                "bp-approvers": "Approver",
                "bp-admins": "Admin",
            },
            "no_principal_role": "Viewer",
            "unmapped_group_role": "Viewer",
            "multiple_groups": "highest_rank",
        },
    },
}


@pytest.fixture
def engine():
    return FakePolicyEngine(
        {"role_definition": ROLE_DEFINITION, "role_assignment": ROLE_ASSIGNMENT}
    )


def test_no_principal_is_viewer(engine):
    assert rbac.effective_role(None, policy_engine=engine) == "Viewer"


def test_group_maps_to_role(engine):
    principal = FakePrincipal("sub-1", {"cognito:groups": ["bp-approvers"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Approver"


def test_multiple_groups_take_the_highest_rank(engine):
    principal = FakePrincipal(
        "sub-2", {"cognito:groups": ["bp-viewers", "bp-admins", "bp-buyers"]}
    )
    assert rbac.effective_role(principal, policy_engine=engine) == "Admin"


def test_unmapped_group_falls_back_to_viewer(engine):
    principal = FakePrincipal("sub-3", {"cognito:groups": ["some-other-group"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


def test_principal_with_no_groups_claim_is_viewer(engine):
    principal = FakePrincipal("sub-4", {})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


def test_viewer_may_read_but_not_communicate(engine):
    assert rbac.may("Viewer", "read", policy_engine=engine) is True
    assert rbac.may("Viewer", "communicate", policy_engine=engine) is False


def test_approver_may_communicate(engine):
    assert rbac.may("Approver", "communicate", policy_engine=engine) is True


def test_approver_may_not_configure(engine):
    assert rbac.may("Approver", "configure", policy_engine=engine) is False


def test_irreversible_classes_come_from_policy(engine):
    assert rbac.is_irreversible("communicate", policy_engine=engine) is True
    assert rbac.is_irreversible("read", policy_engine=engine) is False


def test_missing_policy_denies_everything(engine):
    """A policy that failed to load must not silently grant."""
    empty = FakePolicyEngine({})
    assert rbac.effective_role(None, policy_engine=empty) == "Viewer"
    assert rbac.may("Admin", "communicate", policy_engine=empty) is False


def test_unknown_role_may_do_nothing(engine):
    assert rbac.may("Wizard", "read", policy_engine=engine) is False


def test_unknown_action_class_is_treated_as_irreversible(engine):
    """An action class nobody classified must not slip through as safe."""
    assert rbac.is_irreversible("teleport", policy_engine=engine) is True
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_rbac.py -v`
Expected: collection error — `No module named 'src.services.rbac'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/rbac.py`:

```python
"""Resolve an authenticated caller to a role, and answer what that role may do.

Both answers come from proc.bp_policy, never from constants in this file. A
role's powers change by editing a policy row, which is how the rest of the
platform's governance already works.

Every helper is safe to call with ``principal=None`` and every failure path
lands on the least-privileged answer. A policy that will not load must not
become an accidental grant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROLE_UNKNOWN = "Viewer"

_ROLE_DEFINITION_SLUG = "role_definition"
_ROLE_ASSIGNMENT_SLUG = "role_assignment"


def _engine(policy_engine: Optional[Any]) -> Optional[Any]:
    if policy_engine is not None:
        return policy_engine
    try:
        from src.engines.policy_engine import PolicyEngine
        from src.services.db import get_conn

        return PolicyEngine(connection_factory=get_conn)
    except Exception as exc:  # noqa: BLE001 - resolved to deny by the callers
        logger.error("rbac: could not construct a PolicyEngine: %s", exc)
        return None


def _rules(slug: str, policy_engine: Optional[Any]) -> Dict[str, Any]:
    engine = _engine(policy_engine)
    if engine is None:
        return {}
    try:
        policy = engine.get_policy(slug)
    except Exception as exc:  # noqa: BLE001
        logger.error("rbac: get_policy(%s) failed: %s", slug, exc)
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def _roles_table(policy_engine: Optional[Any]) -> Dict[str, Any]:
    roles = _rules(_ROLE_DEFINITION_SLUG, policy_engine).get("roles")
    return roles if isinstance(roles, dict) else {}


def role_rank(role: Optional[str], policy_engine: Optional[Any] = None) -> int:
    """Rank of ``role``; 0 for anything the policy does not define."""

    entry = _roles_table(policy_engine).get(str(role or ""))
    if not isinstance(entry, dict):
        return 0
    try:
        return int(entry.get("rank") or 0)
    except (TypeError, ValueError):
        return 0


def resolve_roles(
    principal: Optional[Any], policy_engine: Optional[Any] = None
) -> List[str]:
    """Roles carried by ``principal``, mapped from its identity-provider groups."""

    rules = _rules(_ROLE_ASSIGNMENT_SLUG, policy_engine)
    if not rules:
        return []
    if principal is None:
        return []

    claim = str(rules.get("claim") or "cognito:groups")
    claims = getattr(principal, "claims", None)
    raw = claims.get(claim) if isinstance(claims, dict) else None
    if isinstance(raw, str):
        groups = [raw]
    elif isinstance(raw, (list, tuple, set)):
        groups = [str(g) for g in raw]
    else:
        groups = []

    mapping = rules.get("group_to_role")
    mapping = mapping if isinstance(mapping, dict) else {}
    unmapped = str(rules.get("unmapped_group_role") or ROLE_UNKNOWN)

    resolved: List[str] = []
    for group in groups:
        resolved.append(str(mapping.get(group) or unmapped))
    return resolved


def effective_role(
    principal: Optional[Any], policy_engine: Optional[Any] = None
) -> str:
    """The single role ``principal`` acts with.

    No principal, no groups, or an unloadable policy all resolve to the
    policy's ``no_principal_role`` (Viewer), never to something permissive.
    """

    rules = _rules(_ROLE_ASSIGNMENT_SLUG, policy_engine)
    fallback = str(rules.get("no_principal_role") or ROLE_UNKNOWN)

    roles = resolve_roles(principal, policy_engine=policy_engine)
    if not roles:
        return fallback

    if str(rules.get("multiple_groups") or "highest_rank") == "highest_rank":
        best = max(roles, key=lambda r: role_rank(r, policy_engine=policy_engine))
        return best if role_rank(best, policy_engine=policy_engine) else fallback
    return roles[0]


def may(
    role: Optional[str], action_class: str, policy_engine: Optional[Any] = None
) -> bool:
    """True when ``role`` is permitted ``action_class`` by policy."""

    entry = _roles_table(policy_engine).get(str(role or ""))
    if not isinstance(entry, dict):
        return False
    allowed = entry.get("allow")
    if not isinstance(allowed, (list, tuple, set)):
        return False
    return str(action_class) in {str(a) for a in allowed}


def is_irreversible(action_class: str, policy_engine: Optional[Any] = None) -> bool:
    """True when the action class is one policy marks irreversible.

    An action class the policy does not list at all is treated as
    irreversible. Anything unclassified gets the stricter handling, not the
    looser one.
    """

    rules = _rules(_ROLE_DEFINITION_SLUG, policy_engine)
    listed = rules.get("irreversible_classes")
    if not isinstance(listed, (list, tuple, set)):
        return True

    irreversible = {str(c) for c in listed}
    if str(action_class) in irreversible:
        return True

    known = set()
    for entry in _roles_table(policy_engine).values():
        if isinstance(entry, dict) and isinstance(entry.get("allow"), (list, tuple, set)):
            known.update(str(a) for a in entry["allow"])
    return str(action_class) not in known
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_rbac.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/rbac.py tests/guardrails/test_rbac.py
git commit -m "feat(guardrails): resolve principals to policy-defined roles"
```

---

### Task 3: The gate (`guardrail.py`)

**Files:**
- Create: `src/services/guardrail.py`
- Modify: `src/engines/policy_engine.py` (add `policies_for_action`)
- Test: `tests/guardrails/test_guardrail_gate.py`

**Interfaces:**
- Consumes: `rbac.effective_role`, `rbac.may`, `rbac.is_irreversible` from Task 2.
- Produces:
  - `@dataclass(frozen=True) Decision` with fields `allowed: bool`, `reason: str`, `policy_id: Optional[str]`, `policy_name: Optional[str]`, `policy_version: Optional[int]`, `evidence: dict`.
  - `authorize(action: str, action_class: str, principal, context: dict, policy_engine=None) -> Decision`
  - `PolicyEngine.policies_for_action(action: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_guardrail_gate.py`:

```python
"""The gate must be provably closed before it is trusted.

Each test here removes one thing the gate depends on and asserts it denies.
A gate that only ever says yes has not been shown to work.
"""

import pytest

from src.services import guardrail
from tests.guardrails.test_rbac import (
    ROLE_ASSIGNMENT,
    ROLE_DEFINITION,
    FakePolicyEngine,
    FakePrincipal,
)


ALLOW_SEND = {
    "policyId": "email_dispatch_approval",
    "policyName": "EmailDispatchApprovalPolicy",
    "details": {
        "policy_identifier": "email_dispatch_approval",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {"approval_required": True},
    },
    "raw_row": {"version": 1},
}


class GateEngine(FakePolicyEngine):
    """Adds the action lookup the gate needs."""

    def policies_for_action(self, action):
        return [
            p
            for p in self._policies.values()
            if action in (p.get("details", {}).get("applies_to") or [])
        ]


def engine_with(*extra):
    policies = {
        "role_definition": ROLE_DEFINITION,
        "role_assignment": ROLE_ASSIGNMENT,
    }
    for policy in extra:
        policies[policy["policyId"]] = policy
    return GateEngine(policies)


def approver():
    return FakePrincipal("sub-approver", {"cognito:groups": ["bp-approvers"]})


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


def test_approver_with_an_allowing_policy_is_permitted():
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is True
    assert decision.policy_name == "EmailDispatchApprovalPolicy"


def test_viewer_may_not_communicate():
    decision = guardrail.authorize(
        "email.send", "communicate", viewer(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is False
    assert "Viewer" in decision.reason


def test_no_principal_is_denied_an_irreversible_action():
    decision = guardrail.authorize(
        "email.send", "communicate", None, {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is False
    assert "no authenticated principal" in decision.reason


def test_irreversible_action_with_no_policy_is_denied():
    """Default-deny: silence is not permission."""
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with()
    )
    assert decision.allowed is False
    assert "no policy" in decision.reason.lower()


def test_reversible_action_with_no_policy_is_allowed():
    decision = guardrail.authorize(
        "deal.read", "read", viewer(), {}, policy_engine=engine_with()
    )
    assert decision.allowed is True


def test_deny_beats_allow():
    denying = {
        "policyId": "email_block",
        "policyName": "EmailBlockPolicy",
        "details": {
            "policy_identifier": "email_block",
            "required_role": "Approver",
            "applies_to": ["email.send"],
            "rules": {"effect": "deny", "reason": "dispatch frozen"},
        },
        "raw_row": {"version": 1},
    }
    decision = guardrail.authorize(
        "email.send",
        "communicate",
        approver(),
        {},
        policy_engine=engine_with(ALLOW_SEND, denying),
    )
    assert decision.allowed is False
    assert "dispatch frozen" in decision.reason


def test_required_role_on_the_policy_is_enforced():
    admin_only = {
        "policyId": "email_admin_only",
        "policyName": "EmailAdminOnlyPolicy",
        "details": {
            "policy_identifier": "email_admin_only",
            "required_role": "Admin",
            "applies_to": ["email.send"],
            "rules": {},
        },
        "raw_row": {"version": 1},
    }
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(admin_only)
    )
    assert decision.allowed is False
    assert "Admin" in decision.reason


def test_an_exploding_engine_denies_rather_than_raises():
    class Exploding:
        def get_policy(self, slug):
            raise RuntimeError("database is gone")

        def policies_for_action(self, action):
            raise RuntimeError("database is gone")

    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=Exploding()
    )
    assert decision.allowed is False
    assert decision.evidence.get("error")


def test_decision_carries_the_policy_version_for_audit():
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.policy_version == 1
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_guardrail_gate.py -v`
Expected: collection error — `No module named 'src.services.guardrail'`.

- [ ] **Step 3: Add the action lookup to PolicyEngine**

In `src/engines/policy_engine.py`, add this method immediately after `get_policy` (which ends at line 357):

```python
    def policies_for_action(self, action: str) -> List[Dict[str, Any]]:
        """Every active policy that declares it applies to ``action``.

        A policy opts in by listing the action in ``details.applies_to``. This
        is the gate's only lookup path, so policies continue to load from
        exactly one place.
        """

        wanted = str(action or "").strip()
        if not wanted:
            return []
        matched: List[Dict[str, Any]] = []
        for policy in self._policies:
            details = policy.get("details")
            if not isinstance(details, dict):
                continue
            applies = details.get("applies_to")
            if isinstance(applies, str):
                applies = [applies]
            if not isinstance(applies, (list, tuple, set)):
                continue
            if wanted in {str(a) for a in applies}:
                matched.append(policy)
        return matched
```

- [ ] **Step 4: Write the gate**

Create `src/services/guardrail.py`:

```python
"""The single seam an irreversible action must pass through.

Evaluation order, and the reasoning behind it:

1. Resolve the caller's role. No principal means Viewer, so an unauthenticated
   environment cannot send mail just because authentication happens to be off.
2. Role cap. An agent executes as the person who invoked it and can never
   exceed them.
3. Every applicable policy is evaluated and any denial wins. Deny beats allow
   so that adding a restriction never depends on removing a permission.
4. Default-deny for irreversible classes. Silence is not permission.

Any exception is converted to a denial with the error recorded as evidence.
The gate never raises into its caller and never fails open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services import rbac

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    policy_id: Optional[str] = None
    policy_name: Optional[str] = None
    policy_version: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


def _deny(reason: str, **evidence: Any) -> Decision:
    return Decision(allowed=False, reason=reason, evidence=dict(evidence))


def _version_of(policy: Dict[str, Any]) -> Optional[int]:
    raw_row = policy.get("raw_row")
    if isinstance(raw_row, dict):
        try:
            return int(raw_row.get("version"))
        except (TypeError, ValueError):
            return None
    return None


def authorize(
    action: str,
    action_class: str,
    principal: Optional[Any],
    context: Optional[Dict[str, Any]] = None,
    policy_engine: Optional[Any] = None,
) -> Decision:
    """Decide whether ``principal`` may perform ``action``."""

    context = dict(context or {})
    try:
        role = rbac.effective_role(principal, policy_engine=policy_engine)
        irreversible = rbac.is_irreversible(action_class, policy_engine=policy_engine)

        if principal is None and irreversible:
            return _deny(
                "no authenticated principal: irreversible actions are refused",
                action=action,
                action_class=action_class,
            )

        if not rbac.may(role, action_class, policy_engine=policy_engine):
            return _deny(
                f"role {role} may not perform {action_class}",
                action=action,
                role=role,
            )

        engine = policy_engine
        if engine is None:
            from src.engines.policy_engine import PolicyEngine
            from src.services.db import get_conn

            engine = PolicyEngine(connection_factory=get_conn)

        policies: List[Dict[str, Any]] = engine.policies_for_action(action) or []

        allowing: Optional[Dict[str, Any]] = None
        for policy in policies:
            details = policy.get("details") or {}
            rules = details.get("rules") or {}

            required_role = details.get("required_role")
            if required_role and rbac.role_rank(
                role, policy_engine=policy_engine
            ) < rbac.role_rank(required_role, policy_engine=policy_engine):
                return _deny(
                    f"{policy.get('policyName')} requires role {required_role}; "
                    f"caller is {role}",
                    action=action,
                    role=role,
                )

            if str(rules.get("effect") or "").lower() == "deny":
                return Decision(
                    allowed=False,
                    reason=str(rules.get("reason") or "denied by policy"),
                    policy_id=policy.get("policyId"),
                    policy_name=policy.get("policyName"),
                    policy_version=_version_of(policy),
                    evidence={"action": action, "role": role},
                )

            if allowing is None:
                allowing = policy

        if allowing is None:
            if irreversible:
                return _deny(
                    f"no policy permits {action}; irreversible actions are "
                    "default-deny",
                    action=action,
                    action_class=action_class,
                    role=role,
                )
            return Decision(
                allowed=True,
                reason=f"{action_class} is not irreversible and no policy denies it",
                evidence={"action": action, "role": role},
            )

        return Decision(
            allowed=True,
            reason=f"permitted by {allowing.get('policyName')}",
            policy_id=allowing.get("policyId"),
            policy_name=allowing.get("policyName"),
            policy_version=_version_of(allowing),
            evidence={"action": action, "role": role},
        )

    except Exception as exc:  # noqa: BLE001 - a broken gate is a closed gate
        logger.error("guardrail.authorize(%s) failed: %s", action, exc)
        return _deny(
            "policy evaluation failed; denying",
            action=action,
            error=str(exc),
        )
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_guardrail_gate.py -v`
Expected: 9 passed.

- [ ] **Step 6: Confirm nothing existing broke**

Run: `./venv/bin/python -m pytest tests/test_policy_engine.py tests/guardrails -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/services/guardrail.py src/engines/policy_engine.py \
        tests/guardrails/test_guardrail_gate.py
git commit -m "feat(guardrails): default-deny authorize() gate with deny-beats-allow"
```

---

### Task 4: Content sensitivity (`email_sensitivity.py`)

**Files:**
- Create: `src/services/email_sensitivity.py`
- Test: `tests/guardrails/test_email_sensitivity.py`

**Interfaces:**
- Consumes: the `email_sensitivity` policy row from Task 1.
- Produces:
  - `CLASS_UNDETERMINED: str = "undetermined"`
  - `classify(subject, body, attachments, recipient_supplier_id, peer_prices, internal_domains, policy_engine=None) -> ClassificationResult`
  - `@dataclass(frozen=True) ClassificationResult` with `content_class: str`, `detectors_fired: list[str]`, `evidence: dict`
  - `clearance_permits(content_class, clearance, policy_engine=None) -> bool`

  `peer_prices` is a list of `{"supplier_id": str, "amount": str}` for suppliers **other than** the recipient, supplied by the caller. `internal_domains` is a list of the buying organisation's own email domains.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_email_sensitivity.py`:

```python
"""Sensitivity classification, proven by the cases that must be blocked.

Every detector gets a test that fires it and a test that does not, because a
detector that fires on everything is as useless as one that never fires.
"""

import pytest

from src.services import email_sensitivity as sens
from tests.guardrails.test_rbac import FakePolicyEngine


SENSITIVITY_POLICY = {
    "policyId": "email_sensitivity",
    "policyName": "EmailSensitivityPolicy",
    "details": {
        "policy_identifier": "email_sensitivity",
        "required_role": "Admin",
        "rules": {
            "classes": ["public", "internal", "commercial_confidential", "personal"],
            "order": {
                "public": 1,
                "internal": 2,
                "commercial_confidential": 3,
                "personal": 4,
            },
            "default_supplier_clearance": "internal",
            "detectors": {
                "third_party_price": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
                "contract_prose": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
                "internal_staff_contact": {"enabled": True, "raises_to": "personal"},
                "source_document_attached": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
            },
            "rule": "content_class <= recipient_clearance",
            "on_undetermined_class": "deny",
            "on_missing_clearance": "use_default",
        },
    },
}


@pytest.fixture
def engine():
    return FakePolicyEngine({"email_sensitivity": SENSITIVITY_POLICY})


def classify(engine, **kwargs):
    defaults = dict(
        subject="Request for quotation",
        body="Please quote for 100 units.",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=["ourcompany.com"],
    )
    defaults.update(kwargs)
    return sens.classify(policy_engine=engine, **defaults)


def test_plain_rfq_is_internal(engine):
    result = classify(engine)
    assert result.content_class == "internal"
    assert result.detectors_fired == []


def test_competitor_price_raises_to_commercial_confidential(engine):
    result = classify(
        engine,
        body="Supplier B quoted 12,450.00 for the same line.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert result.content_class == "commercial_confidential"
    assert "third_party_price" in result.detectors_fired


def test_own_price_does_not_fire_the_peer_detector(engine):
    """Quoting the recipient their own number is not a leak."""
    result = classify(
        engine,
        body="You quoted 12,450.00 last month.",
        peer_prices=[{"supplier_id": "SUP-1", "amount": "12450.00"}],
    )
    assert "third_party_price" not in result.detectors_fired


def test_contract_prose_raises_to_commercial_confidential(engine):
    result = classify(
        engine,
        body=(
            "6.2 Limitation of Liability. Neither party shall be liable for "
            "indirect or consequential loss arising under this Agreement."
        ),
    )
    assert result.content_class == "commercial_confidential"
    assert "contract_prose" in result.detectors_fired


def test_internal_staff_contact_raises_to_personal(engine):
    result = classify(
        engine, body="Call Jane on +44 20 7946 0812 or jane.doe@ourcompany.com."
    )
    assert result.content_class == "personal"
    assert "internal_staff_contact" in result.detectors_fired


def test_supplier_own_address_is_not_internal_staff_contact(engine):
    result = classify(engine, body="Reply to sales@supplier-b.com.")
    assert "internal_staff_contact" not in result.detectors_fired


def test_attached_source_document_raises_to_commercial_confidential(engine):
    result = classify(
        engine, attachments=[{"filename": "PO-10021.pdf", "is_source_document": True}]
    )
    assert result.content_class == "commercial_confidential"
    assert "source_document_attached" in result.detectors_fired


def test_highest_firing_detector_wins(engine):
    result = classify(
        engine,
        body="Supplier B quoted 12,450.00. Call jane.doe@ourcompany.com.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert result.content_class == "personal"
    assert set(result.detectors_fired) >= {"third_party_price", "internal_staff_contact"}


def test_a_disabled_detector_does_not_fire(engine):
    policy = {
        "policyId": "email_sensitivity",
        "policyName": "EmailSensitivityPolicy",
        "details": {
            "policy_identifier": "email_sensitivity",
            "rules": {
                **SENSITIVITY_POLICY["details"]["rules"],
                "detectors": {
                    **SENSITIVITY_POLICY["details"]["rules"]["detectors"],
                    "internal_staff_contact": {
                        "enabled": False,
                        "raises_to": "personal",
                    },
                },
            },
        },
    }
    disabled = FakePolicyEngine({"email_sensitivity": policy})
    result = sens.classify(
        subject="s",
        body="Call jane.doe@ourcompany.com.",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=["ourcompany.com"],
        policy_engine=disabled,
    )
    assert "internal_staff_contact" not in result.detectors_fired


def test_missing_policy_yields_undetermined(engine):
    empty = FakePolicyEngine({})
    result = sens.classify(
        subject="s",
        body="b",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=[],
        policy_engine=empty,
    )
    assert result.content_class == sens.CLASS_UNDETERMINED


def test_clearance_comparison(engine):
    assert sens.clearance_permits("internal", "internal", policy_engine=engine) is True
    assert sens.clearance_permits("public", "internal", policy_engine=engine) is True
    assert (
        sens.clearance_permits("commercial_confidential", "internal", policy_engine=engine)
        is False
    )
    assert (
        sens.clearance_permits(
            "commercial_confidential", "commercial_confidential", policy_engine=engine
        )
        is True
    )


def test_undetermined_is_never_permitted(engine):
    assert (
        sens.clearance_permits(sens.CLASS_UNDETERMINED, "personal", policy_engine=engine)
        is False
    )


def test_missing_clearance_uses_the_policy_default(engine):
    assert sens.clearance_permits("internal", None, policy_engine=engine) is True
    assert (
        sens.clearance_permits("commercial_confidential", None, policy_engine=engine)
        is False
    )
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_email_sensitivity.py -v`
Expected: collection error — `No module named 'src.services.email_sensitivity'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/email_sensitivity.py`:

```python
"""Classify outbound email content, deterministically.

The detectors here are code because a gate must be reproducible: the same
message must classify the same way every time, and it must still classify when
the model host is down. Which detectors run, and what class each raises the
content to, is policy — so tightening a rule is a row edit, not a deploy.

An error inside a detector yields ``undetermined``, which policy maps to a
denial. A classifier that cannot decide must not be read as "safe".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CLASS_UNDETERMINED = "undetermined"

_SLUG = "email_sensitivity"

# A clause number followed by a contract-style heading, or the stock phrases
# that only appear in contractual prose.
_CONTRACT_PATTERNS = (
    re.compile(r"\b\d+\.\d+\s+[A-Z][A-Za-z ]{3,40}\.", re.MULTILINE),
    re.compile(
        r"\b(limitation of liability|indemnif(y|ication)|termination for convenience"
        r"|governing law|confidentiality obligations|force majeure"
        r"|consequential loss|this agreement)\b",
        re.IGNORECASE,
    ),
)

_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?){2,4}\d{2,4}")


@dataclass(frozen=True)
class ClassificationResult:
    content_class: str
    detectors_fired: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


def _rules(policy_engine: Optional[Any]) -> Dict[str, Any]:
    engine = policy_engine
    if engine is None:
        try:
            from src.engines.policy_engine import PolicyEngine
            from src.services.db import get_conn

            engine = PolicyEngine(connection_factory=get_conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("email_sensitivity: no PolicyEngine: %s", exc)
            return {}
    try:
        policy = engine.get_policy(_SLUG)
    except Exception as exc:  # noqa: BLE001
        logger.error("email_sensitivity: get_policy failed: %s", exc)
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def _order(rules: Dict[str, Any]) -> Dict[str, int]:
    order = rules.get("order")
    if not isinstance(order, dict):
        return {}
    out: Dict[str, int] = {}
    for name, rank in order.items():
        try:
            out[str(name)] = int(rank)
        except (TypeError, ValueError):
            continue
    return out


def _digits(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def _detect_third_party_price(
    text: str, recipient_supplier_id: Optional[str], peer_prices: Iterable[Dict[str, Any]]
) -> Optional[str]:
    """Fires when a figure belonging to another supplier appears in the text."""

    recipient = str(recipient_supplier_id or "").strip()
    haystack = _digits(text)
    for entry in peer_prices or []:
        if not isinstance(entry, dict):
            continue
        owner = str(entry.get("supplier_id") or "").strip()
        if owner and owner == recipient:
            continue
        amount = _digits(entry.get("amount"))
        if len(amount) >= 3 and amount in haystack:
            return f"{owner or 'another supplier'}:{entry.get('amount')}"
    return None


def _detect_contract_prose(text: str) -> Optional[str]:
    for pattern in _CONTRACT_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(0)[:80]
    return None


def _detect_internal_staff_contact(
    text: str, internal_domains: Iterable[str]
) -> Optional[str]:
    domains = {str(d).strip().lower() for d in (internal_domains or []) if str(d).strip()}
    for match in _EMAIL_RE.finditer(text or ""):
        if match.group(1).lower() in domains:
            return match.group(0)
    if domains and _PHONE_RE.search(text or ""):
        # A direct dial only counts as internal when the message is otherwise
        # ours to leak -- an internal address alongside it is the signal.
        for match in _EMAIL_RE.finditer(text or ""):
            if match.group(1).lower() in domains:
                return match.group(0)
    return None


def _detect_source_document_attached(
    attachments: Optional[Iterable[Any]],
) -> Optional[str]:
    for attachment in attachments or []:
        if isinstance(attachment, dict) and attachment.get("is_source_document"):
            return str(attachment.get("filename") or "attachment")
    return None


def classify(
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]],
    recipient_supplier_id: Optional[str],
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
    internal_domains: Optional[Iterable[str]] = None,
    policy_engine: Optional[Any] = None,
) -> ClassificationResult:
    """Classify a message. Returns ``undetermined`` rather than guessing."""

    rules = _rules(policy_engine)
    if not rules:
        return ClassificationResult(
            content_class=CLASS_UNDETERMINED,
            evidence={"error": "email_sensitivity policy unavailable"},
        )

    detectors = rules.get("detectors")
    detectors = detectors if isinstance(detectors, dict) else {}
    order = _order(rules)
    text = f"{subject or ''}\n{body or ''}"

    fired: List[str] = []
    evidence: Dict[str, Any] = {}
    best_class = "internal"
    best_rank = order.get("internal", 2)

    checks = {
        "third_party_price": lambda: _detect_third_party_price(
            text, recipient_supplier_id, peer_prices or []
        ),
        "contract_prose": lambda: _detect_contract_prose(text),
        "internal_staff_contact": lambda: _detect_internal_staff_contact(
            text, internal_domains or []
        ),
        "source_document_attached": lambda: _detect_source_document_attached(attachments),
    }

    for name, check in checks.items():
        config = detectors.get(name)
        if not isinstance(config, dict) or not config.get("enabled"):
            continue
        try:
            hit = check()
        except Exception as exc:  # noqa: BLE001 - cannot decide means deny
            logger.error("email_sensitivity: detector %s failed: %s", name, exc)
            return ClassificationResult(
                content_class=CLASS_UNDETERMINED,
                detectors_fired=fired,
                evidence={"error": f"detector {name} failed: {exc}"},
            )
        if not hit:
            continue
        fired.append(name)
        evidence[name] = hit
        raised = str(config.get("raises_to") or "")
        rank = order.get(raised, 0)
        if rank > best_rank:
            best_class, best_rank = raised, rank

    return ClassificationResult(
        content_class=best_class, detectors_fired=fired, evidence=evidence
    )


def clearance_permits(
    content_class: str,
    clearance: Optional[str],
    policy_engine: Optional[Any] = None,
) -> bool:
    """True when ``content_class`` may be sent to a supplier at ``clearance``."""

    rules = _rules(policy_engine)
    if not rules:
        return False
    if str(content_class) == CLASS_UNDETERMINED:
        return False

    order = _order(rules)
    effective = str(clearance or "").strip() or str(
        rules.get("default_supplier_clearance") or ""
    )

    content_rank = order.get(str(content_class))
    clearance_rank = order.get(effective)
    if content_rank is None or clearance_rank is None:
        return False
    return content_rank <= clearance_rank
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_email_sensitivity.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/email_sensitivity.py tests/guardrails/test_email_sensitivity.py
git commit -m "feat(guardrails): deterministic outbound content classification"
```

---

### Task 5: Mandatory audit for irreversible actions

**Files:**
- Modify: `src/services/agent_actions.py` (add after `record_action`, which ends at line 127)
- Test: `tests/guardrails/test_mandatory_audit.py`

**Interfaces:**
- Consumes: existing `_row_params`, `_INSERT`, `_write_on_shared_conn`, `get_conn` in `agent_actions.py`.
- Produces: `record_action_or_fail(*, phase: str, action_type: str, conn=None, **fields) -> None` — raises `AuditWriteError` when the row cannot be written. Also exports `class AuditWriteError(RuntimeError)`.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_mandatory_audit.py`:

```python
"""An irreversible action that cannot be logged must not happen.

The existing record_action is deliberately best-effort so a logging blip
cannot halt extraction. That trade is wrong for sending mail or approving
spend, so those callers use record_action_or_fail instead. This proves the
two behave differently under exactly the same failure.
"""

import pytest

from src.services import agent_actions


class ExplodingConn:
    """A connection whose cursor always fails, like a dropped session."""

    def cursor(self):
        raise RuntimeError("connection is gone")


def test_best_effort_writer_swallows_a_failure():
    agent_actions.record_action(
        phase="extraction",
        action_type="parse",
        conn=ExplodingConn(),
        summary="should not raise",
    )


def test_mandatory_writer_raises_on_a_failure():
    with pytest.raises(agent_actions.AuditWriteError):
        agent_actions.record_action_or_fail(
            phase="communicate",
            action_type="email.send",
            conn=ExplodingConn(),
            summary="must raise",
        )


def test_mandatory_writer_names_the_action_in_the_error():
    with pytest.raises(agent_actions.AuditWriteError) as excinfo:
        agent_actions.record_action_or_fail(
            phase="communicate",
            action_type="email.send",
            conn=ExplodingConn(),
        )
    assert "email.send" in str(excinfo.value)
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_mandatory_audit.py -v`
Expected: `test_best_effort_writer_swallows_a_failure` passes; the other two FAIL with `AttributeError: module 'src.services.agent_actions' has no attribute 'AuditWriteError'`.

- [ ] **Step 3: Write the implementation**

In `src/services/agent_actions.py`, add after `record_action` (which ends at line 127) and before `bulk_record`:

```python
class AuditWriteError(RuntimeError):
    """Raised when an action that must be audited could not be recorded."""


def record_action_or_fail(
    *, phase: str, action_type: str, conn: Any = None, **fields: Any
) -> None:
    """Insert one action row, or raise.

    The counterpart to :func:`record_action` for irreversible actions —
    sending, approving, configuring. Auditing cannot be switched off, so an
    action whose audit row cannot be written does not proceed. Reads and
    computes keep the best-effort writer, because a transient database blip
    should not halt an extraction backlog.
    """

    try:
        fields["phase"] = phase
        fields["action_type"] = action_type
        params = _row_params(fields)
        if conn is not None:
            _write_on_shared_conn(conn, lambda cur: cur.execute(_INSERT, params))
            return
        with get_conn() as own:
            own.autocommit = False
            cur = own.cursor()
            try:
                cur.execute(_INSERT, params)
                own.commit()
            except Exception:
                own.rollback()
                raise
    except Exception as exc:
        log.error(
            "agent_actions.record_action_or_fail failed (%s/%s): %s",
            phase,
            action_type,
            exc,
        )
        raise AuditWriteError(
            f"could not audit {phase}/{action_type}: {exc}"
        ) from exc
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_mandatory_audit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_actions.py tests/guardrails/test_mandatory_audit.py
git commit -m "feat(guardrails): mandatory audit writer for irreversible actions"
```

---

### Task 6: Approval store — write and verify `bp_approval`

**Files:**
- Create: `src/services/approval_store.py`
- Test: `tests/guardrails/test_approval_store.py`

**Interfaces:**
- Consumes: `proc.bp_approval` (exists, currently empty) with columns `approval_id, deal_id, rfq_id, finding_id, supplier_id, amount, currency, threshold, decision, decision_reason, policy_id, policy_name, grounding, status, actioned_by, actioned_at, workflow_id, created_by, created_date`.
- Produces:
  - `record_approval(*, rfq_id, workflow_id, unique_id, supplier_id, actioned_by, policy_id=None, policy_name=None, amount=None, currency=None, conn=None) -> int` returning `approval_id`.
  - `find_dispatch_approval(*, rfq_id, workflow_id, unique_id, conn=None) -> Optional[dict]` returning the approval row as a dict, or `None`.

  Matching rule, verbatim from the spec: match on `rfq_id` **and** `workflow_id`; where a draft carries a `unique_id` but no `rfq_id`, match on `workflow_id` plus the `unique_id` recorded in the approval's `grounding`. An approval matching neither is treated as absent.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_approval_store.py`:

```python
"""bp_approval is the only place a dispatch approval is believed.

The table exists but has never been written to, so this builds both halves.
The tests that matter are the ones proving a near-miss does not count:
right rfq wrong workflow, pending status, and no actioned_by.
"""

import os
import uuid

import psycopg2
import psycopg2.extras
import pytest
from dotenv import load_dotenv

from src.services import approval_store

load_dotenv()


@pytest.fixture
def conn():
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )
    connection.autocommit = False
    yield connection
    connection.rollback()  # never leave test rows behind
    connection.close()


@pytest.fixture
def ids():
    token = uuid.uuid4().hex[:10]
    return {
        "rfq_id": f"RFQ-{token}",
        "workflow_id": f"WF-{token}",
        "unique_id": f"PROC-WF-{token}",
    }


def test_recorded_approval_is_found(conn, ids):
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        policy_id="email_dispatch_approval",
        policy_name="EmailDispatchApprovalPolicy",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is not None
    assert found["status"] == "approved"
    assert found["actioned_by"] == "buyer@ourcompany.com"


def test_nothing_recorded_means_nothing_found(conn, ids):
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None


def test_right_rfq_wrong_workflow_does_not_match(conn, ids):
    approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id="WF-someone-elses",
        unique_id="PROC-WF-someone-elses",
        conn=conn,
    )
    assert found is None


def test_unique_id_only_draft_matches_via_grounding(conn, ids):
    approval_store.record_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=None,
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is not None


def test_pending_approval_is_not_an_approval(conn, ids):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())",
        (
            ids["rfq_id"],
            ids["workflow_id"],
            "SUP-1",
            "pending",
            "pending",
            "buyer@ourcompany.com",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None


def test_approval_with_no_actioned_by_does_not_count(conn, ids):
    """An approval nobody signed is not a human approval."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, decision, status, actioned_by, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,NULL,%s,%s, now())",
        (
            ids["rfq_id"],
            ids["workflow_id"],
            "SUP-1",
            "approved",
            "approved",
            psycopg2.extras.Json({"unique_id": ids["unique_id"]}),
            "test",
        ),
    )
    found = approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        conn=conn,
    )
    assert found is None
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_approval_store.py -v`
Expected: collection error — `No module named 'src.services.approval_store'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/approval_store.py`:

```python
"""Read and write proc.bp_approval, the record of who approved what.

The table has the right shape but has never been written to, so dispatch had
nothing to verify against. Both halves live here: the write a human approval
produces, and the lookup the send path trusts.

Only a row that is approved AND carries the name of the person who approved it
counts. An approval nobody signed is not a human approval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import psycopg2.extras

from src.services.db import get_conn

logger = logging.getLogger(__name__)

_STATUS_APPROVED = "approved"


def _dict_cursor(conn: Any):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def record_approval(
    *,
    rfq_id: Optional[str],
    workflow_id: Optional[str],
    unique_id: Optional[str],
    supplier_id: Optional[str],
    actioned_by: str,
    policy_id: Optional[str] = None,
    policy_name: Optional[str] = None,
    amount: Optional[Any] = None,
    currency: Optional[str] = None,
    conn: Any = None,
) -> int:
    """Record a human approval. Returns the new ``approval_id``."""

    signer = str(actioned_by or "").strip()
    if not signer:
        raise ValueError("actioned_by is required: an approval must name a person")

    grounding = psycopg2.extras.Json({"unique_id": unique_id})
    sql = (
        "INSERT INTO proc.bp_approval "
        "(rfq_id, workflow_id, supplier_id, amount, currency, decision, "
        " status, actioned_by, actioned_at, policy_id, policy_name, "
        " grounding, created_by, created_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now(), %s,%s,%s,%s, now()) "
        "RETURNING approval_id"
    )
    params = (
        rfq_id,
        workflow_id,
        supplier_id,
        amount,
        currency,
        _STATUS_APPROVED,
        _STATUS_APPROVED,
        signer,
        policy_id,
        policy_name,
        grounding,
        signer,
    )

    if conn is not None:
        cur = conn.cursor()
        cur.execute(sql, params)
        return int(cur.fetchone()[0])

    with get_conn() as own:
        own.autocommit = False
        cur = own.cursor()
        try:
            cur.execute(sql, params)
            approval_id = int(cur.fetchone()[0])
            own.commit()
            return approval_id
        except Exception:
            own.rollback()
            raise


def find_dispatch_approval(
    *,
    rfq_id: Optional[str],
    workflow_id: Optional[str],
    unique_id: Optional[str],
    conn: Any = None,
) -> Optional[Dict[str, Any]]:
    """The approval permitting this draft to be sent, or ``None``.

    Matched on rfq_id AND workflow_id. A draft carrying a unique_id but no
    rfq_id matches on workflow_id plus the unique_id recorded in grounding.
    Anything matching neither is absent, not approved.
    """

    workflow = str(workflow_id or "").strip()
    if not workflow:
        return None

    rfq = str(rfq_id or "").strip()
    unique = str(unique_id or "").strip()

    if rfq:
        sql = (
            "SELECT * FROM proc.bp_approval "
            "WHERE rfq_id = %s AND workflow_id = %s "
            "AND status = %s AND actioned_by IS NOT NULL "
            "ORDER BY approval_id DESC LIMIT 1"
        )
        params: tuple = (rfq, workflow, _STATUS_APPROVED)
    elif unique:
        sql = (
            "SELECT * FROM proc.bp_approval "
            "WHERE workflow_id = %s AND grounding->>'unique_id' = %s "
            "AND status = %s AND actioned_by IS NOT NULL "
            "ORDER BY approval_id DESC LIMIT 1"
        )
        params = (workflow, unique, _STATUS_APPROVED)
    else:
        return None

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = _dict_cursor(connection)
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_approval_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/approval_store.py tests/guardrails/test_approval_store.py
git commit -m "feat(guardrails): write and verify dispatch approvals in bp_approval"
```

---

### Task 7: Gate the send path

**Files:**
- Modify: `src/services/email_dispatch_service.py` (inside `send_draft`, after `body_text` is computed at line ~187)
- Create: `src/services/email_dispatch_guard.py`
- Test: `tests/guardrails/test_send_path_gate.py`

**Interfaces:**
- Consumes: `guardrail.authorize` (Task 3), `email_sensitivity.classify` / `clearance_permits` (Task 4), `approval_store.find_dispatch_approval` (Task 6), `agent_actions.record_action_or_fail` (Task 5).
- Produces:
  - `class DispatchDenied(PermissionError)` with attribute `decision: guardrail.Decision`
  - `check_dispatch(*, conn, draft, recipients, subject, body, attachments, principal, run_count, policy_engine=None) -> guardrail.Decision` — raises nothing; returns a `Decision`.
  - `resolve_recipients(draft, requested) -> list[str]` — the stored draft is authoritative; a caller-supplied list may only narrow it.

- [ ] **Step 1: Write the failing test**

Create `tests/guardrails/test_send_path_gate.py`:

```python
"""The five ordered checks, each proven by making it fail.

These are unit tests over the guard, not over SES. Each one removes exactly
one precondition and asserts the send is refused, and the last asserts the
happy path still passes so the guard is not simply always-deny.
"""

import pytest

from src.services import email_dispatch_guard as guard
from tests.guardrails.test_email_sensitivity import SENSITIVITY_POLICY
from tests.guardrails.test_guardrail_gate import GateEngine
from tests.guardrails.test_rbac import (
    ROLE_ASSIGNMENT,
    ROLE_DEFINITION,
    FakePrincipal,
)


ALLOWLIST_POLICY = {
    "policyId": "email_recipient_allowlist",
    "policyName": "EmailRecipientAllowlistPolicy",
    "details": {
        "policy_identifier": "email_recipient_allowlist",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {
            "match": "exact_casefold",
            "on_unknown_recipient": "deny_and_raise_review",
            "allow_recipients_from_email_body": False,
        },
    },
    "raw_row": {"version": 1},
}

APPROVAL_POLICY = {
    "policyId": "email_dispatch_approval",
    "policyName": "EmailDispatchApprovalPolicy",
    "details": {
        "policy_identifier": "email_dispatch_approval",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {
            "approval_required": True,
            "require_actioned_by": True,
            "trust_input_payload": False,
            "on_missing_approval": "deny",
        },
    },
    "raw_row": {"version": 1},
}

VOLUME_POLICY = {
    "policyId": "email_volume",
    "policyName": "EmailVolumePolicy",
    "details": {
        "policy_identifier": "email_volume",
        "required_role": "Admin",
        "rules": {"max_per_run": 2, "max_per_user_per_day": 50},
    },
    "raw_row": {"version": 1},
}


def engine():
    return GateEngine(
        {
            "role_definition": ROLE_DEFINITION,
            "role_assignment": ROLE_ASSIGNMENT,
            "email_sensitivity": SENSITIVITY_POLICY,
            "email_recipient_allowlist": ALLOWLIST_POLICY,
            "email_dispatch_approval": APPROVAL_POLICY,
            "email_volume": VOLUME_POLICY,
        }
    )


def approver():
    return FakePrincipal("sub-approver", {"cognito:groups": ["bp-approvers"]})


class FakeConn:
    """Answers only the two lookups the guard makes against the database."""

    def __init__(self, allowlist=("buyer@supplier-b.com",), clearance="internal"):
        self.allowlist = {a.casefold() for a in allowlist}
        self.clearance = clearance

    def lookup_supplier_emails(self, supplier_id):
        return set(self.allowlist)

    def lookup_supplier_clearance(self, supplier_id):
        return self.clearance


def base_kwargs(**overrides):
    kwargs = dict(
        conn=FakeConn(),
        draft={
            "unique_id": "PROC-WF-1",
            "rfq_id": "RFQ-1",
            "workflow_id": "WF-1",
            "supplier_id": "SUP-1",
            "recipients": ["buyer@supplier-b.com"],
        },
        recipients=["buyer@supplier-b.com"],
        subject="Request for quotation",
        body="Please quote for 100 units.",
        attachments=None,
        principal=approver(),
        run_count=0,
        policy_engine=engine(),
        approval_lookup=lambda **_: {
            "approval_id": 1,
            "status": "approved",
            "actioned_by": "buyer@ourcompany.com",
        },
        internal_domains=["ourcompany.com"],
        peer_prices=[],
    )
    kwargs.update(overrides)
    return kwargs


def test_happy_path_is_allowed():
    decision = guard.check_dispatch(**base_kwargs())
    assert decision.allowed is True, decision.reason


def test_check_1_no_approval_denies():
    decision = guard.check_dispatch(**base_kwargs(approval_lookup=lambda **_: None))
    assert decision.allowed is False
    assert "approval" in decision.reason.lower()


def test_check_1_input_payload_claiming_approval_is_ignored():
    """Trusting the caller's own word is exactly the hole being closed."""
    draft = dict(base_kwargs()["draft"])
    draft["approved"] = True
    draft["sent_status"] = True
    decision = guard.check_dispatch(
        **base_kwargs(draft=draft, approval_lookup=lambda **_: None)
    )
    assert decision.allowed is False


def test_check_2_recipient_not_on_supplier_master_denies():
    decision = guard.check_dispatch(
        **base_kwargs(recipients=["stranger@elsewhere.com"])
    )
    assert decision.allowed is False
    assert "allow-list" in decision.reason.lower()


def test_check_3_competitor_price_to_internal_supplier_denies():
    decision = guard.check_dispatch(
        **base_kwargs(
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
        )
    )
    assert decision.allowed is False
    assert "clearance" in decision.reason.lower()


def test_check_3_cleared_supplier_may_receive_commercial_content():
    decision = guard.check_dispatch(
        **base_kwargs(
            conn=FakeConn(clearance="commercial_confidential"),
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
        )
    )
    assert decision.allowed is True, decision.reason


def test_check_4_viewer_is_denied():
    decision = guard.check_dispatch(
        **base_kwargs(
            principal=FakePrincipal("sub-v", {"cognito:groups": ["bp-viewers"]})
        )
    )
    assert decision.allowed is False


def test_check_4_no_principal_is_denied():
    decision = guard.check_dispatch(**base_kwargs(principal=None))
    assert decision.allowed is False
    assert "no authenticated principal" in decision.reason


def test_check_5_run_cap_denies():
    decision = guard.check_dispatch(**base_kwargs(run_count=2))
    assert decision.allowed is False
    assert "volume" in decision.reason.lower()


def test_stored_draft_is_authoritative_for_recipients():
    draft = {"recipients": ["buyer@supplier-b.com"]}
    assert guard.resolve_recipients(draft, None) == ["buyer@supplier-b.com"]


def test_caller_may_narrow_recipients_but_not_add():
    draft = {"recipients": ["a@supplier-b.com", "b@supplier-b.com"]}
    assert guard.resolve_recipients(draft, ["a@supplier-b.com"]) == ["a@supplier-b.com"]
    assert guard.resolve_recipients(draft, ["new@elsewhere.com"]) == []
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_send_path_gate.py -v`
Expected: collection error — `No module named 'src.services.email_dispatch_guard'`.

- [ ] **Step 3: Write the guard**

Create `src/services/email_dispatch_guard.py`:

```python
"""The five checks every outbound message passes before it reaches SES.

Order matters and is deliberate: cheapest and most decisive first. An
unapproved draft is refused before anything is classified, and no message is
classified for a recipient that is not on the supplier master.

The stored draft is authoritative for recipients. A caller may narrow that
list; it may never add to it. Accepting caller-supplied addresses is the
specific hole this closes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.services import approval_store, email_sensitivity, guardrail

logger = logging.getLogger(__name__)


class DispatchDenied(PermissionError):
    """Raised by the send path when a guard refuses the message."""

    def __init__(self, decision: guardrail.Decision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


def _normalise(values: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def resolve_recipients(
    draft: Dict[str, Any], requested: Optional[Iterable[Any]]
) -> List[str]:
    """Recipients for this send: the stored draft's list, optionally narrowed."""

    stored = _normalise(draft.get("recipients"))
    if not stored and draft.get("receiver"):
        stored = _normalise([draft.get("receiver")])
    if requested is None:
        return stored
    asked = {r.casefold() for r in _normalise(requested)}
    return [r for r in stored if r.casefold() in asked]


def _rules(policy_engine: Optional[Any], slug: str) -> Dict[str, Any]:
    if policy_engine is None:
        return {}
    try:
        policy = policy_engine.get_policy(slug)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def check_dispatch(
    *,
    conn: Any,
    draft: Dict[str, Any],
    recipients: Optional[Iterable[str]],
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]],
    principal: Optional[Any],
    run_count: int = 0,
    policy_engine: Optional[Any] = None,
    approval_lookup: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    internal_domains: Optional[Iterable[str]] = None,
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
) -> guardrail.Decision:
    """Run the five checks. Returns a Decision; never raises."""

    try:
        supplier_id = draft.get("supplier_id")
        recipient_list = _normalise(recipients)

        # --- 1. Approval, verified against the store ---------------------
        lookup = approval_lookup or approval_store.find_dispatch_approval
        approval = lookup(
            rfq_id=draft.get("rfq_id"),
            workflow_id=draft.get("workflow_id"),
            unique_id=draft.get("unique_id"),
            conn=conn,
        )
        if not approval:
            return guardrail.Decision(
                allowed=False,
                reason="no recorded human approval for this draft",
                policy_name="EmailDispatchApprovalPolicy",
                evidence={"unique_id": draft.get("unique_id")},
            )

        # --- 2. Recipient allow-list -------------------------------------
        if not recipient_list:
            return guardrail.Decision(
                allowed=False,
                reason="no recipient survived allow-list resolution",
                policy_name="EmailRecipientAllowlistPolicy",
            )
        known = {str(a).casefold() for a in (conn.lookup_supplier_emails(supplier_id) or [])}
        unknown = [r for r in recipient_list if r.casefold() not in known]
        if unknown:
            return guardrail.Decision(
                allowed=False,
                reason=f"recipient not on the supplier allow-list: {unknown[0]}",
                policy_name="EmailRecipientAllowlistPolicy",
                evidence={"unknown_recipients": unknown},
            )

        # --- 3. Sensitivity versus supplier clearance --------------------
        clearance = conn.lookup_supplier_clearance(supplier_id)
        classification = email_sensitivity.classify(
            subject=subject,
            body=body,
            attachments=attachments,
            recipient_supplier_id=supplier_id,
            peer_prices=peer_prices or [],
            internal_domains=internal_domains or [],
            policy_engine=policy_engine,
        )
        if not email_sensitivity.clearance_permits(
            classification.content_class, clearance, policy_engine=policy_engine
        ):
            return guardrail.Decision(
                allowed=False,
                reason=(
                    f"content is {classification.content_class}; supplier clearance "
                    f"is {clearance or 'default'}"
                ),
                policy_name="EmailSensitivityPolicy",
                evidence={
                    "detectors_fired": classification.detectors_fired,
                    "detector_evidence": classification.evidence,
                },
            )

        # --- 4. Policy ----------------------------------------------------
        decision = guardrail.authorize(
            "email.send",
            "communicate",
            principal,
            {
                "unique_id": draft.get("unique_id"),
                "supplier_id": supplier_id,
                "recipients": recipient_list,
            },
            policy_engine=policy_engine,
        )
        if not decision.allowed:
            return decision

        # --- 5. Volume ----------------------------------------------------
        volume = _rules(policy_engine, "email_volume")
        try:
            max_per_run = int(volume.get("max_per_run"))
        except (TypeError, ValueError):
            max_per_run = None
        if max_per_run is not None and int(run_count) >= max_per_run:
            return guardrail.Decision(
                allowed=False,
                reason=f"volume cap reached: {run_count}/{max_per_run} for this run",
                policy_name="EmailVolumePolicy",
            )

        return guardrail.Decision(
            allowed=True,
            reason="all dispatch checks passed",
            policy_id=decision.policy_id,
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            evidence={
                "approval_id": approval.get("approval_id"),
                "approved_by": approval.get("actioned_by"),
                "content_class": classification.content_class,
                "recipients": recipient_list,
            },
        )

    except Exception as exc:  # noqa: BLE001 - a broken guard is a closed guard
        logger.error("email_dispatch_guard.check_dispatch failed: %s", exc)
        return guardrail.Decision(
            allowed=False,
            reason="dispatch guard failed; denying",
            evidence={"error": str(exc)},
        )
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_send_path_gate.py -v`
Expected: 11 passed.

- [ ] **Step 5: Wire the guard into the real send path**

In `src/services/email_dispatch_service.py`, add to the imports at the top of the file:

```python
from src.services import email_dispatch_guard
from src.services.agent_actions import record_action_or_fail
```

Then replace the recipient block at lines 160-167:

```python
            recipient_list = self._normalise_recipients(
                recipients if recipients is not None else draft.get("recipients")
            )
            if not recipient_list and draft.get("receiver"):
                recipient_list = self._normalise_recipients([draft["receiver"]])

            if not recipient_list:
                raise ValueError("At least one recipient email is required to send the draft")
```

with:

```python
            # The stored draft is authoritative. A caller may narrow this list
            # but may never introduce an address of its own.
            recipient_list = self._normalise_recipients(
                email_dispatch_guard.resolve_recipients(draft, recipients)
            )

            if not recipient_list:
                raise ValueError("At least one recipient email is required to send the draft")
```

Then, immediately after `body_text` is assigned (line ~187, `body_text = str(body_source).strip() if body_source else ""`), insert the gate:

```python
            # Nothing reaches SES until all five checks pass. Deny is recorded
            # with the same weight as a send: an attempted send that was
            # refused is exactly the event an auditor needs to see.
            gate = email_dispatch_guard.check_dispatch(
                conn=conn,
                draft=draft,
                recipients=recipient_list,
                subject=subject,
                body=body_text,
                attachments=attachments,
                principal=(workflow_dispatch_context or {}).get("principal"),
                run_count=(workflow_dispatch_context or {}).get("run_count", 0),
                internal_domains=self._internal_domains(),
                peer_prices=self._peer_prices(conn, draft),
            )
            record_action_or_fail(
                phase="communicate",
                action_type="email.send",
                conn=conn,
                agent="EmailDispatchAgent",
                status="allowed" if gate.allowed else "denied",
                summary=gate.reason,
                details={
                    "unique_id": unique_id,
                    "supplier_id": draft.get("supplier_id"),
                    "recipients": recipient_list,
                    "principal": getattr(
                        (workflow_dispatch_context or {}).get("principal"),
                        "subject",
                        None,
                    ),
                    "policy_name": gate.policy_name,
                    "policy_version": gate.policy_version,
                    "decision": "allow" if gate.allowed else "deny",
                    "evidence": gate.evidence,
                    "egress": "amazon_ses",
                },
            )
            if not gate.allowed:
                raise email_dispatch_guard.DispatchDenied(gate)
```

Finally add the two helpers to the class, immediately before `send_draft`:

```python
    def _internal_domains(self) -> List[str]:
        """Our own email domains, used to spot internal staff contact details."""

        sender = str(getattr(self.settings, "ses_default_sender", "") or "")
        domain = sender.partition("@")[2].strip()
        return [domain] if domain else []

    def _peer_prices(self, conn: Any, draft: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Figures belonging to suppliers other than this draft's recipient.

        Used by the third_party_price detector. Scoped to the deal so the
        comparison is against genuine competitors on the same requirement.
        """

        deal_id = draft.get("deal_id")
        supplier_id = draft.get("supplier_id")
        if not deal_id:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT supplier_id, total_amount FROM proc.bp_quote_trgt "
                "WHERE deal_id = %s AND supplier_id IS DISTINCT FROM %s "
                "AND total_amount IS NOT NULL",
                (deal_id, supplier_id),
            )
            return [
                {"supplier_id": row[0], "amount": row[1]} for row in cur.fetchall()
            ]
        except Exception as exc:  # noqa: BLE001 - absence of peers is not a pass
            logger.warning("peer price lookup failed for deal %s: %s", deal_id, exc)
            return []
```

The guard calls `conn.lookup_supplier_emails` and `conn.lookup_supplier_clearance`. Add them as module-level helpers in `email_dispatch_guard.py`, replacing the two `conn.*` calls in `check_dispatch` with:

```python
        known = {
            str(a).casefold() for a in (_supplier_emails(conn, supplier_id) or [])
        }
```
and
```python
        clearance = _supplier_clearance(conn, supplier_id)
```

with these helpers added above `check_dispatch`:

```python
def _supplier_emails(conn: Any, supplier_id: Optional[str]) -> List[str]:
    """Addresses on the supplier master for this supplier.

    A test connection may answer directly; a real one is queried.
    """

    if hasattr(conn, "lookup_supplier_emails"):
        return list(conn.lookup_supplier_emails(supplier_id) or [])
    if not supplier_id:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT contact_email_1, contact_email_2 FROM proc.bp_supplier "
        "WHERE supplier_id = %s",
        (supplier_id,),
    )
    out: List[str] = []
    for row in cur.fetchall():
        for value in row:
            text = str(value or "").strip()
            if text:
                out.append(text)
    return out


def _supplier_clearance(conn: Any, supplier_id: Optional[str]) -> Optional[str]:
    """This supplier's clearance level, or None to use the policy default."""

    if hasattr(conn, "lookup_supplier_clearance"):
        return conn.lookup_supplier_clearance(supplier_id)
    if not supplier_id:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT clearance_level FROM proc.bp_supplier WHERE supplier_id = %s",
        (supplier_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None
```

- [ ] **Step 6: Re-run the guard tests and the dispatch tests**

Run: `./venv/bin/python -m pytest tests/guardrails/test_send_path_gate.py -v`
Expected: 11 passed.

Run: `./venv/bin/python -m pytest tests/ -k "dispatch or email_dispatch" -v`
Expected: existing dispatch tests either pass, or fail only where they send without an approval — which is the intended new behaviour. Update any such test to record an approval first; do not weaken the guard to make a test pass.

- [ ] **Step 7: Commit**

```bash
git add src/services/email_dispatch_guard.py src/services/email_dispatch_service.py \
        tests/guardrails/test_send_path_gate.py
git commit -m "feat(guardrails): gate the send path on approval, allow-list, sensitivity, policy and volume"
```

---

### Task 8: Close the HITL bypasses

**Files:**
- Modify: `src/agents/negotiation_agent.py:2918-3007`
- Test: `tests/guardrails/test_hitl_bypasses_closed.py`

**Interfaces:**
- Consumes: `guardrail`, `rbac` from Tasks 2-3.
- Produces: no new public API. `_hitl_auto_approved` (the `hitl_enabled` reader at line 2922) now consults policy before the setting.

- [ ] **Step 1: Read the current code**

Run: `sed -n '2915,3010p' src/agents/negotiation_agent.py`

Identify the two bypasses: the `hitl_enabled` read at line 2922 and the `hitl_auto_approve` block at lines 2999-3001.

- [ ] **Step 2: Write the failing test**

Create `tests/guardrails/test_hitl_bypasses_closed.py`:

```python
"""Neither bypass may return an approval.

A caller waiving its own checkpoint and a global off-switch are the same
failure wearing different clothes: automation deciding it does not need a
human. Both must produce 'pending', and the attempt must be visible.
"""

import pytest

from src.agents.negotiation_agent import NegotiationAgent


class Ctx:
    def __init__(self, input_data):
        self.input_data = input_data
        self.workflow_id = "WF-1"


def _agent():
    return NegotiationAgent.__new__(NegotiationAgent)


def test_payload_auto_approve_does_not_approve():
    agent = _agent()
    state = {}
    result = agent._resolve_hitl_decision(
        context=Ctx({"hitl_auto_approve": True}),
        shared_context={},
        negotiation_state=state,
        round_num=1,
    )
    assert result["status"] == "pending"
    assert result["source"] != "auto_approved"


def test_shared_context_auto_approve_does_not_approve():
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={"hitl_auto_approve": True},
        negotiation_state={},
        round_num=1,
    )
    assert result["status"] == "pending"


def test_the_attempt_is_recorded():
    """An attempted bypass is an event an auditor needs to see."""
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({"hitl_auto_approve": True}),
        shared_context={},
        negotiation_state={},
        round_num=1,
    )
    assert result.get("bypass_attempted") is True


def test_an_explicit_human_decision_still_works():
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={},
        negotiation_state={"hitl_decisions": {"1": "approved"}},
        round_num=1,
    )
    assert result["status"] == "approved"
    assert result["source"] == "provided"


def test_hitl_enabled_false_does_not_auto_approve():
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={},
        negotiation_state={},
        round_num=1,
        hitl_enabled=False,
    )
    assert result["status"] == "pending"
```

> **Note for the implementer:** the method name and signature in the test above must match the real method at `negotiation_agent.py:2985+`. Read the actual method first (Step 1) and adjust the test's call signature to the real one before running it. Do not rename the production method to fit the test.

- [ ] **Step 3: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/guardrails/test_hitl_bypasses_closed.py -v`
Expected: the auto-approve tests FAIL, returning `{"status": "approved", "source": "auto_approved"}`.

- [ ] **Step 4: Remove the payload bypass**

Replace the block at `negotiation_agent.py:2996-3007`:

```python
        auto_flag: Optional[bool] = None
        if raw_value is None:
            if isinstance(shared_context, dict):
                auto_flag = shared_context.get("hitl_auto_approve")
            if isinstance(context.input_data, dict):
                inherited = context.input_data.get("hitl_auto_approve")
                if inherited is not None:
                    auto_flag = bool(inherited)

        if raw_value is None:
            if isinstance(auto_flag, bool) and auto_flag:
                return {"status": "approved", "source": "auto_approved"}
            return {"status": "pending", "source": "awaiting_review"}
```

with:

```python
        # A caller cannot waive its own human checkpoint. The flag is read
        # only so its use can be recorded -- an attempted bypass is exactly
        # the event an auditor needs to see -- and is never acted on.
        bypass_attempted = False
        if raw_value is None:
            if isinstance(shared_context, dict) and shared_context.get(
                "hitl_auto_approve"
            ):
                bypass_attempted = True
            if isinstance(context.input_data, dict) and context.input_data.get(
                "hitl_auto_approve"
            ):
                bypass_attempted = True

        if raw_value is None:
            if bypass_attempted:
                logger.warning(
                    "hitl_auto_approve was supplied for workflow %s round %s and "
                    "was ignored; approval requires a named human",
                    getattr(context, "workflow_id", None),
                    round_num,
                )
            return {
                "status": "pending",
                "source": "awaiting_review",
                "bypass_attempted": bypass_attempted,
            }
```

- [ ] **Step 5: Make `hitl_enabled` narrow-only**

Replace the body of the settings reader at line 2922:

```python
            return bool(getattr(self.agent_nick.settings, "hitl_enabled", True))
```

with:

```python
            # hitl_enabled may narrow which rounds need review; it can no
            # longer switch review off. Where policy requires approval, a
            # false setting fails closed rather than auto-approving.
            setting = bool(getattr(self.agent_nick.settings, "hitl_enabled", True))
            if setting:
                return True
            try:
                from src.services import guardrail

                decision = guardrail.authorize(
                    "negotiation.release_round",
                    "communicate",
                    None,
                    {"round": round_num},
                )
                # A denial means policy still requires a human, so HITL stays on.
                return not decision.allowed
            except Exception:  # noqa: BLE001 - unresolvable means keep the human
                return True
```

- [ ] **Step 6: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/guardrails/test_hitl_bypasses_closed.py -v`
Expected: 5 passed.

- [ ] **Step 7: Confirm the negotiation suite still passes**

Run: `./venv/bin/python -m pytest tests/ -k "negotiation" -v`
Expected: pass, except any test that asserted the old auto-approve behaviour. Update those to assert `pending`; do not restore the bypass.

- [ ] **Step 8: Commit**

```bash
git add src/agents/negotiation_agent.py tests/guardrails/test_hitl_bypasses_closed.py
git commit -m "fix(guardrails): a caller can no longer waive its own HITL checkpoint"
```

---

### Task 9: Live verification on the running server

**Files:**
- Create: `docs/guardrail_live_verification_2026-08-06.md`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: a verification record with real command output.

- [ ] **Step 1: Run the whole guardrail suite**

```bash
./venv/bin/python -m pytest tests/guardrails -v
```
Expected: all pass. Record the counts.

- [ ] **Step 2: Confirm the policies are live and authoritative**

```bash
set -a && . ./.env && set +a
./venv/bin/python -c "
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services import rbac
e = PolicyEngine(connection_factory=get_conn)
for slug in ('role_definition','role_assignment','email_dispatch_approval',
             'email_recipient_allowlist','email_sensitivity','email_volume'):
    p = e.get_policy(slug)
    print(slug, 'LOADED' if p else 'MISSING')
print('no-principal role:', rbac.effective_role(None, policy_engine=e))
print('Viewer may communicate:', rbac.may('Viewer','communicate',policy_engine=e))
print('Approver may communicate:', rbac.may('Approver','communicate',policy_engine=e))
"
```
Expected: six `LOADED`, no-principal role `Viewer`, Viewer `False`, Approver `True`.

- [ ] **Step 3: Prove the gate denies against the live database**

```bash
./venv/bin/python -c "
from src.services import guardrail
d = guardrail.authorize('email.send','communicate',None,{})
print('allowed:', d.allowed)
print('reason:', d.reason)
"
```
Expected: `allowed: False`, reason naming the missing principal.

- [ ] **Step 4: Prove policy is authoritative without a deploy**

Deactivate the sensitivity policy, confirm classification goes `undetermined` (and therefore denies), then reactivate:

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "UPDATE proc.bp_policy SET policy_status=0 WHERE policy_details->>'policy_identifier'='email_sensitivity';"

./venv/bin/python -c "
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services import email_sensitivity as s
e = PolicyEngine(connection_factory=get_conn)
r = s.classify(subject='x', body='y', attachments=None, recipient_supplier_id='SUP-1',
               peer_prices=[], internal_domains=[], policy_engine=e)
print('class with policy off:', r.content_class)
print('permitted:', s.clearance_permits(r.content_class,'personal',policy_engine=e))
"

PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "UPDATE proc.bp_policy SET policy_status=1 WHERE policy_details->>'policy_identifier'='email_sensitivity';"
```
Expected: `class with policy off: undetermined`, `permitted: False`. Then confirm the reactivation restored `policy_status=1`.

- [ ] **Step 3a: Start the server and confirm it boots**

Use the same invocation the systemd unit uses (`procwise.service`) — note it is `.venv`, not `venv`, the module path is `api.main:app`, and `PYTHONPATH` must include `src`:

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PYTHONPATH=/home/muthu/PycharmProjects/BP_Backend:/home/muthu/PycharmProjects/BP_Backend/src \
  ./.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
```

If the service is already running on port 8000, use that instance rather than starting a second one.

Confirm startup logs show `PolicyEngine loaded N policies` with N ≥ 18 (12 pre-existing + 6 new). **Never `pkill -f uvicorn`** — another session may share this checkout; stop it with Ctrl-C in its own terminal.

- [ ] **Step 5: Write the verification record**

Create `docs/guardrail_live_verification_2026-08-06.md` containing, for each step above, the command run and its **actual output pasted verbatim**. Where something did not work, record that rather than omitting it. State plainly which of the four authorised fixes are proven live and which are proven only by test.

- [ ] **Step 6: Commit**

```bash
git add -f docs/guardrail_live_verification_2026-08-06.md
git commit -m "docs(guardrails): live verification results"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §5.1 six policy rows | Task 1 |
| §5.2 roles, principal, `bp_role_assignment`, no-principal = Viewer | Tasks 1, 2 |
| §5.3 `authorize()`, `Decision`, `policies_for_action` | Task 3 |
| §5.4 detectors, `clearance_level`, comparison rule | Tasks 1, 4 |
| §5.5 five ordered checks, approval write path, recipient resolution | Tasks 6, 7 |
| §5.6 `record_action_or_fail`, G8 audit fields | Tasks 5, 7 |
| §5.7 both HITL bypasses | Task 8 |
| §6 data model | Task 1 |
| §7 testing (13 named guards) | Tasks 1-8, each breaking its guard first |
| §8 rollout order | Task order 1 → 9 |

**Known deviation from spec §5.2:** the spec lists `Principal` gaining `roles`/`role` fields. The plan instead computes the role via `rbac.effective_role(principal)` at each call site and leaves `Principal` unchanged. This keeps `api/auth.py` free of a policy-engine dependency at import time and avoids a database read during token verification. Behaviour is identical; if you would rather have the fields on `Principal`, say so and Task 2 changes.

**Type consistency:** `Decision` is constructed in `guardrail.py`, `email_dispatch_guard.py` and asserted in three test files with the same field names throughout. `classify()` keyword arguments match between `email_sensitivity.py`, its tests and `email_dispatch_guard.check_dispatch`. `find_dispatch_approval` uses the same `rfq_id`/`workflow_id`/`unique_id`/`conn` keywords in `approval_store.py`, its tests, and the `approval_lookup` injection point.

**One flagged risk:** Task 8's test calls `_resolve_hitl_decision` with an assumed signature. Step 1 of that task requires reading the real method first and adjusting the test to the real signature — the production method must not be renamed to fit the test.
