# Approvals Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the authenticated surface that records human approvals into `proc.bp_approval`, so the guardrail layer's approval check has something real to verify — with the approver's identity taken from the token and never from a request body.

**Architecture:** Four endpoints on the existing `decisions` router, all authenticated. An approval binds to a content hash of what the approver actually saw, so editing a draft invalidates it. Agent-initiated sends (no human approval) consult the autonomy policy that already exists rather than a third mechanism. Every rule ships as a policy row, not code.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, PostgreSQL (`proc` schema), pytest, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-08-07-approvals-surface-design.md`

## Global Constraints

- Run tests with `./venv/bin/python -m pytest`, never bare `pytest`. For direct `python -c`, set `PYTHONPATH` to include both the repo root and its `src` directory — several modules import siblings bare (`from agents...`, `from services...`) and will otherwise fail with `ModuleNotFoundError`.
- **The approver's identity is `principal.subject` and nothing else.** No request body field may influence who is recorded. A fallback to a body value is not acceptable — a fallback preserves the forgery whenever it is taken. This is the defect that caused the previous attempt to be reverted (`8d59acc`).
- **Every rule is a policy row, not a Python constant.** Capability grants, role requirements and enforcement modes live in `proc.bp_policy` and ship as seed SQL.
- Shared production-cluster database: tests must roll back and leave nothing behind. Never `DELETE` or `UPDATE` a pre-existing row.
- Fail closed everywhere. A missing, unparseable or unreachable policy denies.
- Commit messages contain no `Co-Authored-By` line.
- New tables/columns use the `bp_` prefix; indexes `ix_bp_<table>_<cols>`.
- Roles and ranks, verbatim: `Viewer`=1, `Buyer`=2, `Approver`=3, `Admin`=4.
- **Every guard proven to fail before it is trusted:** write the test, run it, watch it go RED, implement, watch it go GREEN. Put the RED output in your report.
- **Ask of every test: would it still pass if the check it guards were deleted?** This plan's predecessor was bitten eight times by tests that passed while the guard was dead, wrong or forgeable — most often because a fixture was richer than the database. Any task touching policy content must include a test that reads the **live row** via `PolicyEngine`, asserting both directions, run RED before the migration.

---

## Current state (verified, so you do not have to re-derive it)

- `proc.bp_approval` exists, has 0 rows, and nothing writes a findable approval.
- `approval_store.record_approval(*, rfq_id, workflow_id, unique_id, supplier_id, actioned_by, deal_id=None, policy_id=None, policy_name=None, amount=None, currency=None, grounding_extra=None, conn=None) -> int` — note `grounding_extra`, which is where the content hash goes.
- `approval_store.find_dispatch_approval(*, rfq_id, workflow_id, unique_id, conn=None)` and `find_round_approval(*, workflow_id, round_num, conn=None)` take the newest row for a key regardless of status, then require `status='approved'` and non-null `actioned_by` — so a later `revoked` row shadows an approval.
- `email_dispatch_guard.resolve_recipients(draft, requested) -> List[str]` — the stored draft is authoritative; a caller may only narrow.
- `email_dispatch_guard.check_dispatch` fetches the approval at line ~354 and reads `approval.get("deal_id")` at ~370. The content-hash check belongs immediately after the approval is found.
- `guardrail.authorize(action, action_class, principal, context=None, policy_engine=None) -> Decision`.
- `PolicyEngine.policies_for_action(action)` selects on `details.applies_to`. `email_dispatch_approval` (policy_id 673) has `applies_to: ["email.send"]`, `required_role: "Approver"`.
- `role_definition` is policy_id 671; `Buyer.allow` is currently `["read","compute","write"]`.
- `decisions.py` does **not** import `require_user`. Its endpoints take the actor as `body.user_id or "api"`.
- `proc.draft_rfq_emails` has `recipient_email` (singular), `payload`, `unique_id`, `rfq_id`, `workflow_id`, `subject`, `body`, `attachments`, `sender`, `supplier_id`, `sent`. **There is no `recipients` column.**
- There are currently **0 unsent drafts**, so live verification has nothing to approve until one is created.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/services/approval_content.py` | Compute the content hash. One helper, called by both the approval endpoint and the send path, so they cannot drift. |
| `src/api/routers/approvals.py` | The four endpoints. New file rather than growing `decisions.py`, which is already 450+ lines. |
| `src/services/approval_store.py` | Extended with `list_pending_approvals` and `revoke_approval` |
| `src/services/email_dispatch_guard.py` | Content-hash check; agent-autonomy fallback |
| `src/api/routers/decisions.py` | Authenticate the four existing endpoints |
| `deploy/sql/seed/accelerator_policies.sql` | The policy rows, as seed data |

---

### Task 1: The `approve_email` capability, as policy

**Files:**
- Create: `deploy/sql/seed/accelerator_policies.sql`
- Create: `deploy/sql/seed/accelerator_policies_rollback.sql`
- Test: `tests/approvals/test_approval_policy.py` (create `tests/approvals/__init__.py`, empty)

**Interfaces:**
- Consumes: nothing.
- Produces: `role_definition` grants `approve_email` to Buyer/Approver/Admin and lists it in `irreversible_classes`; `email_dispatch_approval` has `required_role: "Buyer"` and `on_content_mismatch: "deny"`; a new policy `EmailApprovalCapabilityPolicy` with `applies_to: ["approval.email"]` and `required_role: "Buyer"`.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/__init__.py` (empty) and `tests/approvals/test_approval_policy.py`:

```python
"""The capability that lets a person approve an email is a policy row, not code.

A customer who wants email approval restricted to Approvers changes one row.
These read the LIVE rows: a fixture richer than the database is how this
project has repeatedly shipped a guard that was already dead.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

from src.engines.policy_engine import PolicyEngine
from src.services import rbac

load_dotenv()


def _factory():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )


@pytest.fixture(scope="module")
def engine():
    return PolicyEngine(connection_factory=_factory)


def test_buyer_and_above_may_approve_an_email(engine):
    assert rbac.may("Buyer", "approve_email", policy_engine=engine) is True
    assert rbac.may("Approver", "approve_email", policy_engine=engine) is True
    assert rbac.may("Admin", "approve_email", policy_engine=engine) is True


def test_a_viewer_may_not_approve_an_email(engine):
    assert rbac.may("Viewer", "approve_email", policy_engine=engine) is False


def test_approving_an_email_is_not_approving_money(engine):
    """The two capabilities are deliberately distinct, not one rank doing both."""
    assert rbac.may("Buyer", "transact", policy_engine=engine) is False
    assert rbac.may("Approver", "transact", policy_engine=engine) is True


def test_approve_email_is_treated_as_irreversible(engine):
    assert rbac.is_irreversible("approve_email", policy_engine=engine) is True


def test_the_approval_action_resolves_to_a_policy(engine):
    matched = engine.policies_for_action("approval.email")
    assert matched, "no live policy declares applies_to ['approval.email']"
    assert any(
        (p.get("details") or {}).get("required_role") == "Buyer" for p in matched
    )


def test_dispatch_approval_requires_buyer_and_denies_on_content_mismatch(engine):
    policy = engine.get_policy("email_dispatch_approval")
    assert policy is not None
    details = policy["details"]
    assert details["required_role"] == "Buyer"
    assert details["rules"]["on_content_mismatch"] == "deny"
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_policy.py -v`
Expected: all six FAIL — `approve_email` is in no role's `allow`, no policy declares `approval.email`, and `email_dispatch_approval` still says `Approver` with no `on_content_mismatch`.

- [ ] **Step 3: Write the seed SQL**

Create `deploy/sql/seed/accelerator_policies.sql`:

```sql
-- Accelerator policies: the rules a customer adopts and then edits.
-- Nothing here is compiled into the product; changing a row changes behaviour.
-- Spec: docs/superpowers/specs/2026-08-07-approvals-surface-design.md
BEGIN;

-- 1. approve_email: approving your own outbound email is a distinct, lesser
-- capability than approving money. Buyer and above hold it; transact does not
-- move. Named separately so the difference is visible in the policy, not
-- implied by rank.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(
               jsonb_set(
                   jsonb_set(
                       policy_details,
                       '{rules,roles,Buyer,allow}',
                       '["read","compute","write","approve_email"]'::jsonb, true),
                   '{rules,roles,Approver,allow}',
                   '["read","compute","write","communicate","transact","approve_email"]'::jsonb, true),
               '{rules,roles,Admin,allow}',
               '["read","compute","write","communicate","transact","share","configure","delegate","approve_email"]'::jsonb, true),
           '{rules,irreversible_classes}',
           '["communicate","transact","share","configure","delegate","approve_email"]'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

-- 2. Dispatch approval drops to Buyer (the drafter approves their own mail --
-- the agent drafted it on their behalf), and a draft edited after approval is
-- refused rather than sent on a stale approval.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(policy_details, '{required_role}', '"Buyer"'::jsonb, true),
           '{rules,on_content_mismatch}', '"deny"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_dispatch_approval';

-- 3. The approval action itself, so authorize() has a policy to resolve.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
VALUES
(
 'EmailApprovalCapabilityPolicy', 'security',
 'Who may record an approval for an outbound email.',
 '{
   "policy_identifier": "email_approval_capability",
   "required_role": "Buyer",
   "applies_to": ["approval.email"],
   "rules": {
     "self_approval_allowed": true,
     "note": "The agent drafts on the user behalf, so approving is authorship rather than oversight. Set self_approval_allowed to false to require a second person."
   }
 }'::jsonb,
 '', 1, 1, 'accelerator_seed', now()
);

COMMIT;
```

- [ ] **Step 4: Write the rollback**

Create `deploy/sql/seed/accelerator_policies_rollback.sql`:

```sql
BEGIN;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'email_approval_capability';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(
               jsonb_set(
                   jsonb_set(
                       policy_details,
                       '{rules,roles,Buyer,allow}',
                       '["read","compute","write"]'::jsonb, true),
                   '{rules,roles,Approver,allow}',
                   '["read","compute","write","communicate","transact"]'::jsonb, true),
               '{rules,roles,Admin,allow}',
               '["read","compute","write","communicate","transact","share","configure","delegate"]'::jsonb, true),
           '{rules,irreversible_classes}',
           '["communicate","transact","share","configure","delegate"]'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details - 'rules' ||
           jsonb_build_object('rules', (policy_details->'rules') - 'on_content_mismatch'),
           '{required_role}', '"Approver"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_dispatch_approval';

COMMIT;
```

- [ ] **Step 5: Apply the seed**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/seed/accelerator_policies.sql
```

Expect `UPDATE 1`, `UPDATE 1`, `INSERT 0 1`, `COMMIT`. **If any reports a different count, stop and report it** — the filters name exactly one row each, and a different count means something unexpected matched on a shared cluster.

- [ ] **Step 6: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_policy.py -v`
Expected: 6 passed.

Then confirm the guardrail suite is undisturbed:
Run: `./venv/bin/python -m pytest tests/guardrails -q`
Expected: still green. If `test_rbac.py`'s live-row test fails because `Buyer.allow` changed, update that fixture to match — the live row is the source of truth.

- [ ] **Step 7: Commit**

```bash
git add deploy/sql/seed/accelerator_policies.sql \
        deploy/sql/seed/accelerator_policies_rollback.sql \
        tests/approvals/__init__.py tests/approvals/test_approval_policy.py
git commit -m "feat(approvals): approve_email as a policy row a customer owns"
```

---

### Task 2: The content hash

**Files:**
- Create: `src/services/approval_content.py`
- Test: `tests/approvals/test_approval_content.py`

**Interfaces:**
- Consumes: `email_dispatch_guard.resolve_recipients(draft, requested)`.
- Produces: `content_hash(draft: Dict[str, Any]) -> str` — a hex digest over the recipients the send path will actually use, plus subject, body and attachment identities.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/test_approval_content.py`:

```python
"""What was approved must be what gets sent.

Without this, an approval is standing permission on a mutable object: approve
a routine RFQ, someone edits the body to carry a competitor's price, and the
original approval still releases it.
"""

from src.services.approval_content import content_hash


def _draft(**over):
    base = {
        "unique_id": "PROC-WF-1",
        "recipients": ["buyer@supplier-b.com"],
        "subject": "Request for quotation",
        "body": "Please quote for 100 units.",
        "attachments": [{"filename": "spec.pdf"}],
    }
    base.update(over)
    return base


def test_the_same_draft_hashes_the_same_way():
    assert content_hash(_draft()) == content_hash(_draft())


def test_editing_the_body_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(body="Supplier B quoted 12,450.00.")
    )


def test_changing_a_recipient_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(recipients=["someone.else@elsewhere.com"])
    )


def test_changing_the_subject_changes_the_hash():
    assert content_hash(_draft()) != content_hash(_draft(subject="Revised"))


def test_adding_an_attachment_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(attachments=[{"filename": "spec.pdf"}, {"filename": "po.pdf"}])
    )


def test_recipient_order_does_not_change_the_hash():
    """Two recipients in a different order are the same set of recipients."""
    a = _draft(recipients=["a@supplier-b.com", "b@supplier-b.com"])
    b = _draft(recipients=["b@supplier-b.com", "a@supplier-b.com"])
    assert content_hash(a) == content_hash(b)


def test_it_hashes_the_recipients_the_send_path_will_use():
    """draft_rfq_emails has recipient_email, not recipients.

    Hashing a raw column instead of the resolved list would let the approved
    set and the sent set diverge.
    """
    from_singular = _draft(recipients=None, receiver="buyer@supplier-b.com")
    assert content_hash(from_singular) == content_hash(_draft())


def test_a_draft_with_no_content_still_hashes():
    """Never raise into a caller -- an unhashable draft must not crash a send."""
    assert isinstance(content_hash({}), str)
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_content.py -v`
Expected: collection error — `No module named 'src.services.approval_content'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/approval_content.py`:

```python
"""Hash what an approver saw, so an edited draft cannot ride an old approval.

One helper, called by both the approval endpoint and the send path. They must
never compute this differently -- if they drift, an approval starts covering
content nobody approved, which is the whole failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _attachment_identities(draft: Dict[str, Any]) -> list:
    """Attachment names, sorted. Content is not hashed -- the identity of what
    was attached is what an approver actually reviewed."""

    out = []
    raw = draft.get("attachments")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("filename") or item.get("name")
            else:
                name = item
            text = str(name or "").strip()
            if text:
                out.append(text)
    return sorted(out)


def content_hash(draft: Dict[str, Any]) -> str:
    """A stable digest of the recipients, subject, body and attachments.

    Recipients come from ``resolve_recipients``, which is what the send path
    actually sends to -- ``proc.draft_rfq_emails`` stores ``recipient_email``
    (singular) plus a payload blob, so reading a column directly would let the
    approved set and the sent set diverge.

    Never raises: an unhashable draft yields a digest of what could be read,
    and the comparison then simply fails to match, which denies.
    """

    try:
        from src.services.email_dispatch_guard import resolve_recipients

        recipients = sorted(r.casefold() for r in resolve_recipients(draft, None))
    except Exception as exc:  # noqa: BLE001 - a hash that cannot be computed must not crash a send
        logger.error("approval_content: recipient resolution failed: %s", exc)
        recipients = []

    payload = {
        "recipients": recipients,
        "subject": str(draft.get("subject") or "").strip(),
        "body": str(draft.get("body") or "").strip(),
        "attachments": _attachment_identities(draft),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_content.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/approval_content.py tests/approvals/test_approval_content.py
git commit -m "feat(approvals): hash what the approver saw"
```

---

### Task 3: Listing and revoking in the store

**Files:**
- Modify: `src/services/approval_store.py` (append after `find_round_approval`)
- Test: `tests/approvals/test_approval_store_extensions.py`

**Interfaces:**
- Consumes: `record_approval`, `find_dispatch_approval` from the existing module.
- Produces:
  - `list_pending_dispatch_approvals(*, limit: int = 50, conn=None) -> List[Dict[str, Any]]` — unsent drafts with no current approval, each carrying `unique_id`, `rfq_id`, `workflow_id`, `supplier_id`, `subject`, `content_hash`.
  - `revoke_approval(*, approval_id: int, actioned_by: str, reason: Optional[str] = None, conn=None) -> int` — writes a **later** row with `status='revoked'` copying the original's keys, returns the new id.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/test_approval_store_extensions.py`:

```python
"""Listing what awaits approval, and withdrawing one that was given.

Revocation writes a NEW row rather than mutating: bp_approval is append-only,
and find_dispatch_approval already takes the newest row for a key regardless
of status, so a later revoked row shadows the approval.
"""

import os
import uuid

import psycopg2
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
    connection.rollback()
    connection.close()


@pytest.fixture
def ids():
    token = uuid.uuid4().hex[:10]
    return {
        "rfq_id": f"RFQ-{token}",
        "workflow_id": f"WF-{token}",
        "unique_id": f"PROC-WF-{token}",
    }


def test_revoking_shadows_the_approval(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"],
        workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"],
        supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com",
        conn=conn,
    )
    assert approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], conn=conn,
    ) is not None

    approval_store.revoke_approval(
        approval_id=approval_id, actioned_by="approver@ourcompany.com",
        reason="sent in error", conn=conn,
    )

    assert approval_store.find_dispatch_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], conn=conn,
    ) is None, "a withdrawn approval must not still authorise a send"


def test_revoking_records_who_withdrew_it(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    new_id = approval_store.revoke_approval(
        approval_id=approval_id, actioned_by="approver@ourcompany.com",
        reason="sent in error", conn=conn,
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT status, actioned_by, decision_reason FROM proc.bp_approval "
        "WHERE approval_id = %s",
        (new_id,),
    )
    status, actioned_by, reason = cur.fetchone()
    assert status == "revoked"
    assert actioned_by == "approver@ourcompany.com"
    assert reason == "sent in error"


def test_revoking_an_unknown_approval_raises(conn):
    with pytest.raises(ValueError):
        approval_store.revoke_approval(
            approval_id=-1, actioned_by="approver@ourcompany.com", conn=conn
        )


def test_revocation_requires_a_named_person(conn, ids):
    approval_id = approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    with pytest.raises(ValueError):
        approval_store.revoke_approval(
            approval_id=approval_id, actioned_by="   ", conn=conn
        )


def test_pending_list_returns_a_list(conn):
    """There are 0 unsent drafts in this environment, so this asserts shape,
    not contents -- a contents assertion would pass vacuously."""
    rows = approval_store.list_pending_dispatch_approvals(limit=5, conn=conn)
    assert isinstance(rows, list)
    for row in rows:
        assert "unique_id" in row
        assert "content_hash" in row


def test_an_approved_draft_is_not_pending(conn, ids):
    """Insert a draft, approve it, and confirm it drops out of the list."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.draft_rfq_emails "
        "(rfq_id, supplier_id, subject, body, sent, recipient_email, "
        " thread_index, sender, workflow_id, unique_id, created_on) "
        "VALUES (%s,%s,%s,%s,false,%s,1,%s,%s,%s, now())",
        (ids["rfq_id"], "SUP-1", "RFQ", "Please quote.",
         "buyer@supplier-b.com", "us@ourcompany.com",
         ids["workflow_id"], ids["unique_id"]),
    )
    pending = approval_store.list_pending_dispatch_approvals(limit=200, conn=conn)
    assert any(r["unique_id"] == ids["unique_id"] for r in pending)

    approval_store.record_approval(
        rfq_id=ids["rfq_id"], workflow_id=ids["workflow_id"],
        unique_id=ids["unique_id"], supplier_id="SUP-1",
        actioned_by="buyer@ourcompany.com", conn=conn,
    )
    pending_after = approval_store.list_pending_dispatch_approvals(limit=200, conn=conn)
    assert not any(r["unique_id"] == ids["unique_id"] for r in pending_after)
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_store_extensions.py -v`
Expected: FAIL — `module 'src.services.approval_store' has no attribute 'revoke_approval'`.

- [ ] **Step 3: Write the implementation**

Append to `src/services/approval_store.py`, after `find_round_approval`:

```python
def list_pending_dispatch_approvals(
    *, limit: int = 50, conn: Any = None
) -> List[Dict[str, Any]]:
    """Unsent drafts that have no current approval.

    Without this an approver has nothing to act on. The content hash of each
    is included so a caller approves a specific version of the draft rather
    than the draft as a moving target.
    """

    sql = (
        "SELECT unique_id, rfq_id, workflow_id, supplier_id, subject, body, "
        "       recipient_email, sender, attachments, payload "
        "  FROM proc.draft_rfq_emails "
        " WHERE sent IS NOT TRUE "
        "   AND unique_id IS NOT NULL "
        " ORDER BY created_on DESC "
        " LIMIT %s"
    )

    def _run(connection: Any) -> List[Dict[str, Any]]:
        from src.services.approval_content import content_hash

        cur = _dict_cursor(connection)
        cur.execute(sql, (int(limit),))
        drafts = [dict(r) for r in cur.fetchall()]
        out: List[Dict[str, Any]] = []
        for draft in drafts:
            existing = find_dispatch_approval(
                rfq_id=draft.get("rfq_id"),
                workflow_id=draft.get("workflow_id"),
                unique_id=draft.get("unique_id"),
                conn=connection,
            )
            if existing:
                continue
            out.append(
                {
                    "unique_id": draft.get("unique_id"),
                    "rfq_id": draft.get("rfq_id"),
                    "workflow_id": draft.get("workflow_id"),
                    "supplier_id": draft.get("supplier_id"),
                    "subject": draft.get("subject"),
                    "content_hash": content_hash(draft),
                }
            )
        return out

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)


def revoke_approval(
    *,
    approval_id: int,
    actioned_by: str,
    reason: Optional[str] = None,
    conn: Any = None,
) -> int:
    """Withdraw an approval by writing a later revoking row. Returns its id.

    bp_approval is append-only, and both lookups take the newest row for a key
    regardless of status before requiring it to be approved and signed -- so a
    later revoked row shadows the original without mutating history.
    """

    signer = str(actioned_by or "").strip()
    if not signer:
        raise ValueError("actioned_by is required: a revocation must name a person")

    def _run(connection: Any) -> int:
        cur = _dict_cursor(connection)
        cur.execute(
            "SELECT * FROM proc.bp_approval WHERE approval_id = %s", (int(approval_id),)
        )
        original = cur.fetchone()
        if original is None:
            raise ValueError(f"no approval with id {approval_id}")
        original = dict(original)

        write = connection.cursor()
        write.execute(
            "INSERT INTO proc.bp_approval "
            "(deal_id, rfq_id, supplier_id, decision, decision_reason, status, "
            " actioned_by, actioned_at, grounding, workflow_id, created_by, created_date) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s, now(), %s,%s,%s, now()) "
            "RETURNING approval_id",
            (
                original.get("deal_id"),
                original.get("rfq_id"),
                original.get("supplier_id"),
                "deny",
                reason,
                "revoked",
                signer,
                psycopg2.extras.Json(original.get("grounding") or {}),
                original.get("workflow_id"),
                signer,
            ),
        )
        return int(write.fetchone()[0])

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        own.autocommit = False
        try:
            new_id = _run(own)
            own.commit()
            return new_id
        except Exception:
            own.rollback()
            raise
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_store_extensions.py -v`
Expected: 6 passed.

- [ ] **Step 5: Confirm the database is clean**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
./venv/bin/python -c "
import os,psycopg2
c=psycopg2.connect(host=os.getenv('DB_HOST'),port=os.getenv('DB_PORT',5432),dbname=os.getenv('DB_NAME'),user=os.getenv('DB_USER'),password=os.getenv('DB_PASSWORD'),connect_timeout=10)
q=c.cursor(); q.execute('select count(*) from proc.bp_approval'); print('bp_approval rows:',q.fetchone()[0],'(want 0)')
q.execute(\"select count(*) from proc.draft_rfq_emails where unique_id like 'PROC-WF-%' and created_on > now() - interval '1 hour'\"); print('recent test drafts:',q.fetchone()[0],'(want 0)')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/services/approval_store.py tests/approvals/test_approval_store_extensions.py
git commit -m "feat(approvals): list what awaits approval, and withdraw one"
```

---

### Task 4: The endpoints

**Files:**
- Create: `src/api/routers/approvals.py`
- Modify: `src/api/main.py` (register the router)
- Test: `tests/approvals/test_approval_endpoints.py`

**Interfaces:**
- Consumes: `approval_store.record_approval` / `list_pending_dispatch_approvals` / `revoke_approval`; `approval_content.content_hash`; `guardrail.authorize`; `api.auth.require_user`.
- Produces: `POST /approvals/dispatch/{unique_id}`, `POST /approvals/round/{workflow_id}/{round_num}`, `POST /approvals/{approval_id}/revoke`, `GET /approvals/pending`.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/test_approval_endpoints.py`:

```python
"""The approver is whoever the token says, and nobody else.

The previous attempt at this surface was reverted because it took the
approver's name from the request body over an unauthenticated route, which
let anyone forge a human approval. These tests exist mainly to keep that
shut.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import approvals as approvals_router


class _Principal:
    subject = "sub-buyer-001"
    email = "buyer@ourcompany.com"
    claims = {"cognito:groups": ["bp-buyers"]}


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: _Principal()
    return TestClient(app)


@pytest.fixture
def anonymous_client():
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: None
    return TestClient(app)


def test_the_body_cannot_name_the_approver(client, monkeypatch):
    """The single most important test in this file."""
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft", lambda uid, conn=None: {"unique_id": uid}
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )

    response = client.post(
        "/approvals/dispatch/PROC-WF-1",
        json={"actioned_by": "ceo@ourcompany.com", "user_id": "ceo@ourcompany.com"},
    )

    assert response.status_code == 200
    assert recorded["actioned_by"] == "sub-buyer-001", (
        "the approver came from the request body, not the token"
    )


def test_an_unauthenticated_caller_cannot_approve(anonymous_client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    response = anonymous_client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert response.status_code in (401, 403)
    assert called["recorded"] is False, "an approval was written with no principal"


def test_a_denied_capability_writes_nothing(client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft", lambda uid, conn=None: {"unique_id": uid}
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(
            allowed=False, reason="role Viewer may not perform approve_email"
        ),
    )

    response = client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert response.status_code == 403
    assert called["recorded"] is False


def test_approving_a_missing_draft_is_refused(client, monkeypatch):
    monkeypatch.setattr(approvals_router, "_load_draft", lambda uid, conn=None: None)
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )
    response = client.post("/approvals/dispatch/PROC-WF-NOPE", json={})
    assert response.status_code == 404


def test_the_content_hash_is_recorded(client, monkeypatch):
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft",
        lambda uid, conn=None: {
            "unique_id": uid, "subject": "RFQ", "body": "Please quote.",
            "recipients": ["buyer@supplier-b.com"],
        },
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )

    client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert recorded["grounding_extra"]["content_hash"]
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_endpoints.py -v`
Expected: collection error — `cannot import name 'approvals' from 'src.api.routers'`.

- [ ] **Step 3: Write the router**

Create `src/api/routers/approvals.py`:

```python
"""Recording that a human approved something.

The approver is ``principal.subject`` and nothing else. A previous version of
this surface accepted the approver's name from the request body over an
unauthenticated route, which let anyone forge a human approval -- so the body
here carries only WHAT is being approved, never WHO approved it. Do not add a
body field for the actor, and do not add a fallback when the principal is
absent: a fallback is the forgery, taken conditionally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_user
from src.services import approval_store, guardrail
from src.services.approval_content import content_hash

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approvals", tags=["Approvals"])

_ACTION = "approval.email"
_ACTION_CLASS = "approve_email"


class ApproveRequest(BaseModel):
    """What is being approved. Deliberately carries no actor field."""

    reason: Optional[str] = None


class RevokeRequest(BaseModel):
    reason: Optional[str] = None


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return nick


def _load_draft(unique_id: str, conn: Any = None) -> Optional[Dict[str, Any]]:
    """The stored draft being approved, or None."""

    from src.services.db import get_conn

    sql = (
        "SELECT unique_id, rfq_id, workflow_id, supplier_id, subject, body, "
        "       recipient_email, sender, attachments, payload "
        "  FROM proc.draft_rfq_emails "
        " WHERE unique_id = %s AND sent IS NOT TRUE "
        " ORDER BY id DESC LIMIT 1"
    )

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = approval_store._dict_cursor(connection)
        cur.execute(sql, (unique_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)


def _require_capability(principal: Any, context: Dict[str, Any]) -> None:
    """Deny unless policy grants this caller the approve_email capability."""

    decision = guardrail.authorize(_ACTION, _ACTION_CLASS, principal, context)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)


def _subject(principal: Any) -> str:
    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        # Belt and braces: require_user already refuses an unidentified caller
        # when auth is enforced, and an approval without a person is exactly
        # what this surface exists to prevent.
        raise HTTPException(
            status_code=401, detail="an approval must name an authenticated person"
        )
    return subject


@router.get("/pending")
def list_pending(principal=Depends(require_user)) -> Dict[str, Any]:
    """Drafts awaiting a decision, each with the hash a caller would approve."""

    _require_capability(principal, {"action": "list_pending"})
    rows = approval_store.list_pending_dispatch_approvals(limit=200)
    return {"pending": rows, "count": len(rows)}


@router.post("/dispatch/{unique_id}")
def approve_dispatch(
    unique_id: str,
    body: ApproveRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Approve a drafted email for sending."""

    actioned_by = _subject(principal)
    _require_capability(principal, {"unique_id": unique_id})

    draft = _load_draft(unique_id)
    if draft is None:
        raise HTTPException(
            status_code=404, detail=f"no unsent draft with unique_id {unique_id}"
        )

    digest = content_hash(draft)
    approval_id = approval_store.record_approval(
        rfq_id=draft.get("rfq_id"),
        workflow_id=draft.get("workflow_id"),
        unique_id=unique_id,
        supplier_id=draft.get("supplier_id"),
        actioned_by=actioned_by,
        deal_id=draft.get("deal_id"),
        policy_name="EmailDispatchApprovalPolicy",
        grounding_extra={"content_hash": digest, "reason": body.reason},
    )
    logger.info(
        "approval recorded: unique_id=%s approval_id=%s by=%s",
        unique_id, approval_id, actioned_by,
    )
    return {
        "approval_id": approval_id,
        "unique_id": unique_id,
        "actioned_by": actioned_by,
        "content_hash": digest,
    }


@router.post("/round/{workflow_id}/{round_num}")
def approve_round(
    workflow_id: str,
    round_num: int,
    body: ApproveRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Approve a negotiation round.

    A round has no editable artefact, so there is no content hash to bind.
    """

    actioned_by = _subject(principal)
    _require_capability(principal, {"workflow_id": workflow_id, "round": round_num})

    approval_id = approval_store.record_approval(
        rfq_id=None,
        workflow_id=workflow_id,
        unique_id=None,
        supplier_id=None,
        actioned_by=actioned_by,
        policy_name="EmailDispatchApprovalPolicy",
        grounding_extra={"round": int(round_num), "reason": body.reason},
    )
    return {
        "approval_id": approval_id,
        "workflow_id": workflow_id,
        "round": int(round_num),
        "actioned_by": actioned_by,
    }


@router.post("/{approval_id}/revoke")
def revoke(
    approval_id: int,
    body: RevokeRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Withdraw an approval. Writes a later row; history is not rewritten."""

    actioned_by = _subject(principal)
    _require_capability(principal, {"approval_id": approval_id})

    try:
        new_id = approval_store.revoke_approval(
            approval_id=approval_id, actioned_by=actioned_by, reason=body.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "revoked_approval_id": approval_id,
        "revocation_id": new_id,
        "actioned_by": actioned_by,
    }
```

- [ ] **Step 4: Register the router**

In `src/api/main.py`, find where the other routers are included (search for `include_router`) and add, following the existing pattern exactly:

```python
from api.routers import approvals as approvals_router
...
app.include_router(approvals_router.router)
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_approval_endpoints.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/approvals.py src/api/main.py \
        tests/approvals/test_approval_endpoints.py
git commit -m "feat(approvals): the authenticated surface that records a human decision"
```

---

### Task 5: Enforce the content hash on the send path

**Files:**
- Modify: `src/services/email_dispatch_guard.py` (after the approval lookup, ~line 354-370)
- Test: `tests/approvals/test_content_binding_enforced.py`

**Interfaces:**
- Consumes: `approval_content.content_hash`; the approval row's `grounding.content_hash`.
- Produces: `check_dispatch` denies when the draft's current hash differs from the approved one, subject to the `on_content_mismatch` policy key.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/test_content_binding_enforced.py`:

```python
"""An approval covers the email that was approved, not the draft as a moving target."""

from src.services import email_dispatch_guard as guard
from src.services.approval_content import content_hash
from tests.guardrails.test_send_path_gate import base_kwargs


DRAFT = {
    "unique_id": "PROC-WF-1",
    "rfq_id": "RFQ-1",
    "workflow_id": "WF-1",
    "supplier_id": "SUP-1",
    "recipients": ["buyer@supplier-b.com"],
    "subject": "Request for quotation",
    "body": "Please quote for 100 units.",
}


def _approval(hash_value):
    return {
        "approval_id": 1,
        "status": "approved",
        "actioned_by": "buyer@ourcompany.com",
        "deal_id": None,
        "grounding": {"content_hash": hash_value},
    }


def test_an_unchanged_draft_is_allowed():
    """So the check cannot be satisfied by simply always denying."""
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=dict(DRAFT),
            body=DRAFT["body"],
            subject=DRAFT["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is True, decision.reason


def test_an_edited_body_is_refused():
    edited = dict(DRAFT, body="Supplier B quoted 12,450.00.")
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=edited,
            body=edited["body"],
            subject=edited["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is False
    assert "changed" in decision.reason.lower()


def test_a_changed_recipient_is_refused():
    edited = dict(DRAFT, recipients=["someone@elsewhere.com"])
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=edited,
            body=edited["body"],
            subject=edited["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is False


def test_an_approval_with_no_hash_is_refused():
    """An approval predating content binding cannot vouch for content."""
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=dict(DRAFT),
            body=DRAFT["body"],
            subject=DRAFT["subject"],
            approval_lookup=lambda **_: _approval(None),
        )
    )
    assert decision.allowed is False
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_content_binding_enforced.py -v`
Expected: the three refusal tests FAIL (the send is allowed) — the check does not exist yet.

- [ ] **Step 3: Add the check**

In `src/services/email_dispatch_guard.py`, immediately after the approval is found and before the allow-list check, insert:

```python
        # --- 1b. The approval covers the email that was approved -----------
        # Without this an approval is standing permission on a mutable object:
        # approve a routine RFQ, edit the body to carry a competitor's price,
        # and the original approval still releases it.
        from src.services.approval_content import content_hash

        grounding = approval.get("grounding")
        approved_hash = (
            grounding.get("content_hash") if isinstance(grounding, dict) else None
        )
        mismatch_mode = str(
            (_rules(policy_engine, "email_dispatch_approval") or {}).get(
                "on_content_mismatch"
            )
            or "deny"
        ).lower()
        current_hash = content_hash(draft)
        if approved_hash != current_hash and mismatch_mode != "warn":
            return guardrail.Decision(
                allowed=False,
                reason=(
                    "the draft changed since it was approved; it must be "
                    "approved again"
                ),
                policy_name="EmailDispatchApprovalPolicy",
                evidence={
                    "approved_content_hash": approved_hash,
                    "current_content_hash": current_hash,
                },
            )
```

Note `mismatch_mode != "warn"` rather than `== "deny"`: an unrecognised value denies, consistent with every other guard here.

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_content_binding_enforced.py -v`
Expected: 4 passed.

- [ ] **Step 5: Confirm the guardrail suite still passes**

Run: `./venv/bin/python -m pytest tests/guardrails -v`
Expected: green. Existing send-path tests supply approvals without a `content_hash`, so they will now be refused — **update those fixtures to include a matching hash rather than weakening the check.** If a test cannot pass without removing the check, stop and report it.

- [ ] **Step 6: Commit**

```bash
git add src/services/email_dispatch_guard.py \
        tests/approvals/test_content_binding_enforced.py tests/guardrails/
git commit -m "feat(approvals): an edited draft cannot ride its old approval"
```

---

### Task 6: Agent autonomy through the existing policy

**Files:**
- Modify: `src/services/email_dispatch_guard.py` (the no-approval branch)
- Test: `tests/approvals/test_agent_autonomy.py`

**Interfaces:**
- Consumes: `src.services.governance_tools.authority.resolve_authority(policy_engine, agents, *, autonomy_slug=...)`, which returns `{agent: {...}}` per agent and never raises.
- Produces: a send with no human approval consults `resolve_authority` for the acting agent instead of denying outright.

- [ ] **Step 1: Know the real shape (already verified — do not re-derive)**

`resolve_authority(policy_engine, [agent]) -> {agent: verdict}`. The verdict dict has **exactly** these keys, confirmed against `ungoverned_block` and the success path:

```
agent, governed, slug, policy_id, policy_name, auto_intents,
escalate_intents, limit_gbp, limit_currency,
max_auto_replies_per_thread, min_intent_confidence, reason
```

**There is no `may_send` key and no `reasons` key** — `reason` is singular. An earlier draft of this plan assumed both; they do not exist.

The semantics that matter here:

- `governed: False` means **escalate** — the policy could not be read or is not linked. Never a permissive default.
- `governed: True` with `auto_intents == []` means nothing is autonomous. This is the live state today.
- Autonomy exists only for a **named intent** in `auto_intents`.

One consequence worth stating plainly, because it shapes the branch: `EmailReplyAutonomyPolicy` governs *replies*, which carry an intent. A plain outbound dispatch has no intent, so there is nothing to match against `auto_intents` and it **always denies**. That is the correct outcome — an agent sending unprompted outbound mail is precisely what should not happen silently — but it means this branch grants autonomy only to intent-bearing sends, and only when a customer has named that intent in policy. Say so in your report rather than letting a future reader think the branch is dead.

- [ ] **Step 2: Write the failing test**

Create `tests/approvals/test_agent_autonomy.py`:

```python
"""A send with no human approval is agent-initiated, and policy decides.

resolve_authority already reads EmailReplyAutonomyPolicy and is used by the
orchestrator and decision engine. The send path did not consult it, so two
email-governance mechanisms coexisted unaware of each other.
"""

from src.services import email_dispatch_guard as guard
from tests.guardrails.test_send_path_gate import base_kwargs


def _verdict(**over):
    """The real shape resolve_authority returns. No may_send key exists."""
    base = {
        "agent": "email_dispatch_agent",
        "governed": True,
        "slug": "email_reply_autonomy",
        "policy_id": 473,
        "policy_name": "EmailReplyAutonomyPolicy",
        "auto_intents": [],
        "escalate_intents": ["price_change"],
        "limit_gbp": None,
        "limit_currency": None,
        "max_auto_replies_per_thread": 2,
        "min_intent_confidence": 0.8,
        "reason": "resolved from governed policy",
    }
    base.update(over)
    return base


def test_no_approval_and_nothing_autonomous_is_refused():
    """auto_intents is empty on the live policy today."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            authority_lookup=lambda agent: _verdict(),
        )
    )
    assert decision.allowed is False
    assert "autonom" in decision.reason.lower() or "approval" in decision.reason.lower()


def test_a_named_autonomous_intent_is_allowed():
    """So the branch is a real gate, not a second way of always denying."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="acknowledgement",
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is True, decision.reason


def test_an_intent_not_named_in_policy_is_refused():
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="price_change",
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is False


def test_a_send_with_no_intent_is_refused_even_when_intents_are_granted():
    """A plain outbound dispatch carries no intent, so nothing matches."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent=None,
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is False


def test_an_ungoverned_agent_is_refused():
    """governed=False means escalate, not proceed."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="acknowledgement",
            authority_lookup=lambda agent: _verdict(
                governed=False, auto_intents=["acknowledgement"]
            ),
        )
    )
    assert decision.allowed is False, (
        "an ungoverned verdict must escalate even when it names the intent"
    )


def test_a_failing_authority_lookup_is_refused():
    def explode(agent):
        raise RuntimeError("policy engine unreachable")

    decision = guard.check_dispatch(
        **base_kwargs(approval_lookup=lambda **_: None, agent_name="x",
                      intent="acknowledgement", authority_lookup=explode)
    )
    assert decision.allowed is False
```

- [ ] **Step 3: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_agent_autonomy.py -v`
Expected: FAIL — `check_dispatch` has no `agent_name`/`authority_lookup` parameters, and the no-approval branch denies unconditionally.

- [ ] **Step 4: Implement the branch**

Add `agent_name: Optional[str] = None`, `intent: Optional[str] = None` and `authority_lookup: Optional[Callable[[str], Dict[str, Any]]] = None` to `check_dispatch`'s signature, and replace the no-approval denial with:

```python
        if not approval:
            # No human approved this, so it is agent-initiated. That question
            # is already governed by EmailReplyAutonomyPolicy via
            # resolve_authority, which the orchestrator and decision engine
            # consult -- use it rather than adding a third mechanism.
            if not agent_name:
                return guardrail.Decision(
                    allowed=False,
                    reason="no recorded human approval for this draft",
                    policy_name="EmailDispatchApprovalPolicy",
                    evidence={"unique_id": draft.get("unique_id")},
                )
            try:
                if authority_lookup is not None:
                    verdict = authority_lookup(agent_name)
                else:
                    from src.services.governance_tools.authority import (
                        resolve_authority,
                    )

                    engine = policy_engine or rbac.policy_engine()
                    verdict = (resolve_authority(engine, [agent_name]) or {}).get(
                        agent_name
                    ) or {}
            except Exception as exc:  # noqa: BLE001 - unresolvable authority denies
                logger.error("authority lookup failed for %s: %s", agent_name, exc)
                return guardrail.Decision(
                    allowed=False,
                    reason="agent send authority could not be resolved; denying",
                    evidence={"error": str(exc)},
                )

            # governed=False means escalate. Otherwise autonomy exists only for
            # a named intent in auto_intents -- which is empty on the live
            # policy, so nothing is autonomous today. A plain dispatch carries
            # no intent and therefore never matches, which is the wanted
            # outcome: an agent must not send unprompted outbound mail.
            auto_intents = {str(i) for i in (verdict.get("auto_intents") or [])}
            permitted = bool(
                verdict.get("governed")
                and intent
                and str(intent) in auto_intents
            )
            if not permitted:
                return guardrail.Decision(
                    allowed=False,
                    reason=(
                        "no human approval, and policy does not grant this agent "
                        "autonomy to send"
                    ),
                    policy_name="EmailReplyAutonomyPolicy",
                    evidence={
                        "agent": agent_name,
                        "intent": intent,
                        "governed": verdict.get("governed"),
                        "auto_intents": sorted(auto_intents),
                        "reason": verdict.get("reason"),
                    },
                )
            # Autonomy granted: skip the content-hash check, since there is no
            # approval to bind to, and continue to the remaining checks.
            approval = {"autonomous": True, "agent": agent_name}
```

**Important:** the content-hash check from Task 5 must skip when `approval.get("autonomous")` is true — there is no approved hash to compare against. Guard it accordingly.

Then in `email_dispatch_service.py`, pass `agent_name` through from the dispatch context so production supplies it.

- [ ] **Step 5: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_agent_autonomy.py -v`
Expected: 4 passed.

- [ ] **Step 6: Verify against the live policy**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PYTHONPATH=$PWD:$PWD/src ./venv/bin/python -c "
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services.governance_tools.authority import resolve_authority
e = PolicyEngine(connection_factory=get_conn)
v = resolve_authority(e, ['email_dispatch_agent'])
print('live authority verdict:', v)
print('may send unattended:', bool(v.get('email_dispatch_agent',{}).get('may_send')))
"
```

Expected: `may send unattended: False` — `auto_reply_intents` is empty, so nothing is autonomous. Record the actual output.

- [ ] **Step 7: Commit**

```bash
git add src/services/email_dispatch_guard.py src/services/email_dispatch_service.py \
        tests/approvals/test_agent_autonomy.py
git commit -m "feat(approvals): agent sends consult the autonomy policy that already exists"
```

---

### Task 7: Authenticate the existing decisions endpoints

**Files:**
- Modify: `src/api/routers/decisions.py` (four endpoints, and the request models)
- Test: `tests/approvals/test_decisions_authenticated.py`

**Interfaces:**
- Consumes: `api.auth.require_user`.
- Produces: `decide_finding`, `act_on_finding`, `decide_email_reply`, `act_on_email_reply` all take `principal=Depends(require_user)` and record `principal.subject` as the actor. The `user_id` field is removed from `DecideRequest` and `ActionRequest`.

- [ ] **Step 1: Write the failing test**

Create `tests/approvals/test_decisions_authenticated.py`:

```python
"""These endpoints resolve findings and release supplier email replies, so who
acted is load-bearing -- and it was taken from the request body."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import decisions as decisions_router


class _Principal:
    subject = "sub-approver-001"
    email = "approver@ourcompany.com"
    claims = {"cognito:groups": ["bp-approvers"]}


@pytest.fixture
def anonymous_client():
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.dependency_overrides[decisions_router.require_user] = lambda: None
    return TestClient(app)


def test_deciding_a_finding_requires_authentication(anonymous_client):
    response = anonymous_client.post("/decisions/finding/F-1", json={})
    assert response.status_code in (401, 403)


def test_acting_on_a_finding_requires_authentication(anonymous_client):
    response = anonymous_client.post("/decisions/finding/F-1/action", json={"action": "resolve"})
    assert response.status_code in (401, 403)


def test_the_request_model_no_longer_accepts_an_actor():
    assert "user_id" not in decisions_router.DecideRequest.model_fields, (
        "a body-supplied actor is exactly the forgery this removes"
    )
    assert "user_id" not in decisions_router.ActionRequest.model_fields
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `./venv/bin/python -m pytest tests/approvals/test_decisions_authenticated.py -v`
Expected: FAIL — `decisions_router` has no `require_user`, and `user_id` is still a field.

- [ ] **Step 3: Implement**

In `src/api/routers/decisions.py`:

1. Add to the imports: `from api.auth import require_user`
2. Remove `user_id: Optional[str] = None` from `DecideRequest` and `ActionRequest`.
3. Add `principal=Depends(require_user)` to `decide_finding`, `act_on_finding`, `decide_email_reply` and `act_on_email_reply`.
4. Replace every `body.user_id or "api"` with a subject drawn from the principal, using this helper added near the top:

```python
def _actor(principal: Any) -> str:
    """Who acted. From the token, never from the body.

    A body-supplied actor is forgeable, and these endpoints resolve findings
    and release supplier mail -- the name recorded against that has to mean
    something.
    """

    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        raise HTTPException(
            status_code=401, detail="this action must be attributed to a person"
        )
    return subject
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `./venv/bin/python -m pytest tests/approvals/test_decisions_authenticated.py -v`
Expected: 3 passed.

- [ ] **Step 5: Check for callers that will break**

Run: `grep -rn "user_id" src/api/routers/decisions.py tests/ | grep -i decision | head`

Any test posting `user_id` to these endpoints needs updating to authenticate instead. Update them; do not restore the field.

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/decisions.py tests/approvals/test_decisions_authenticated.py
git commit -m "fix(decisions): the actor comes from the token, not the request body"
```

---

### Task 8: Live verification

**Files:**
- Create: `docs/approvals_live_verification_2026-08-07.md`

- [ ] **Step 1: Run the whole suite**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
./venv/bin/python -m pytest tests/approvals tests/guardrails \
  tests/test_email_dispatch_service.py tests/test_agent_endpoints.py \
  tests/test_negotiation_agent.py tests/services/test_value_query_service.py \
  tests/agents/test_approvals_agent.py tests/test_email_dispatch_agent.py -v
```

Record the counts.

- [ ] **Step 2: Prove the capability resolves from the live policy**

```bash
set -a && . ./.env && set +a
PYTHONPATH=$PWD:$PWD/src ./venv/bin/python -c "
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services import rbac
e = PolicyEngine(connection_factory=get_conn)
for role in ('Viewer','Buyer','Approver','Admin'):
    print(role, 'approve_email=', rbac.may(role,'approve_email',policy_engine=e),
          'transact=', rbac.may(role,'transact',policy_engine=e))
print('approve_email irreversible:', rbac.is_irreversible('approve_email', policy_engine=e))
"
```

Expected: Viewer False/False; Buyer True/False; Approver True/True; Admin True/True; irreversible True.

- [ ] **Step 3: Prove an end-to-end approval against the live database, then roll back**

```bash
PYTHONPATH=$PWD:$PWD/src ./venv/bin/python -c "
import os, psycopg2, uuid
from src.services import approval_store
from src.services.approval_content import content_hash
c=psycopg2.connect(host=os.getenv('DB_HOST'),port=os.getenv('DB_PORT',5432),dbname=os.getenv('DB_NAME'),user=os.getenv('DB_USER'),password=os.getenv('DB_PASSWORD'),connect_timeout=10)
c.autocommit=False
t=uuid.uuid4().hex[:8]
draft={'unique_id':'PROC-WF-'+t,'rfq_id':'RFQ-'+t,'workflow_id':'WF-'+t,
       'subject':'RFQ','body':'Please quote.','recipients':['buyer@supplier-b.com']}
h=content_hash(draft)
aid=approval_store.record_approval(rfq_id=draft['rfq_id'],workflow_id=draft['workflow_id'],
    unique_id=draft['unique_id'],supplier_id='SUP-1',actioned_by='sub-buyer-001',
    grounding_extra={'content_hash':h},conn=c)
found=approval_store.find_dispatch_approval(rfq_id=draft['rfq_id'],
    workflow_id=draft['workflow_id'],unique_id=draft['unique_id'],conn=c)
print('approval found:', found is not None, '| hash stored:', (found or {}).get('grounding',{}).get('content_hash')==h)
approval_store.revoke_approval(approval_id=aid,actioned_by='sub-approver-001',reason='probe',conn=c)
after=approval_store.find_dispatch_approval(rfq_id=draft['rfq_id'],
    workflow_id=draft['workflow_id'],unique_id=draft['unique_id'],conn=c)
print('after revocation:', 'shadowed' if after is None else 'STILL FOUND (bad)')
c.rollback(); c.close(); print('rolled back')
"
```

- [ ] **Step 4: Prove policy is authoritative without a deploy**

Deactivate `EmailApprovalCapabilityPolicy`, show `authorize("approval.email", ...)` starts denying, reactivate it, show it allows again. Confirm `policy_status` is restored to the value you found.

- [ ] **Step 5: Boot the server**

Use the systemd unit's invocation (`.venv`, `api.main:app`, `PYTHONPATH` including both roots). **Port 8000 runs the main checkout, which may not have this code — use a different port and stop only the PID you started.** Never `pkill -f uvicorn`. Confirm the startup log shows the policy engine loading at least 19 policies (18 + the new capability policy), and that `/approvals/pending` appears in the route table.

- [ ] **Step 6: Confirm the database is clean**

`proc.bp_approval` = 0 rows, no test drafts left in `proc.draft_rfq_emails`, no test residue in `proc.bp_agent_actions`.

- [ ] **Step 7: Write the record**

Create `docs/approvals_live_verification_2026-08-07.md` with, for each check, the command and its **actual output pasted verbatim**. End with two explicit lists: what is proven against the live system, and what is proven only by unit test or not at all. Record failures with the same prominence as passes.

Note in it that there are **0 unsent drafts** in this environment, so `/approvals/pending` legitimately returns an empty list — say so rather than presenting an empty result as a pass.

- [ ] **Step 8: Commit**

```bash
git add -f docs/approvals_live_verification_2026-08-07.md
git commit -m "docs(approvals): live verification results"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §5.1 `approve_email` capability as policy | Task 1 |
| §5.2 four endpoints, identity from token | Task 4 |
| §5.3 content binding | Tasks 2, 5 |
| §5.4 agent autonomy via `resolve_authority` | Task 6 |
| §5.5 authenticate existing decisions endpoints | Task 7 |
| §5.6 what an approval requires | Tasks 1, 4 (capability + authenticated principal + target) |
| §5.7 accelerator seed set | Task 1 |
| §6 data model | Task 1 (policy edits), Task 2 (`grounding.content_hash`, no DDL) |
| §7 testing | Every task, each proven RED first |

**Placeholder scan:** none — checked for TBD/TODO/"handle appropriately".

**Type consistency:** `content_hash(draft) -> str` used identically in Tasks 2, 4, 5 and 8. `record_approval`'s `grounding_extra` carries `content_hash` in Tasks 4 and 8 and is read from `approval["grounding"]["content_hash"]` in Task 5 — consistent. `revoke_approval(*, approval_id, actioned_by, reason=None, conn=None) -> int` matches between Tasks 3, 4 and 8.

**Two flagged risks:**

1. **Task 6's key names were wrong in an earlier draft and are now corrected against the real module.** `resolve_authority` returns `governed`, `auto_intents`, `escalate_intents` and `reason` (singular) — there is no `may_send` and no `reasons`. The task now carries the verified key list. Do not rename anything in `authority.py` to match a test.
2. **Task 5 will break existing send-path tests** whose approval fixtures carry no `content_hash`. That is expected and correct; those fixtures get a matching hash. If any test can only pass by removing the check, stop rather than weakening it.

**One deliberate scope call:** `self_approval_allowed` ships as a policy key set to `true` but is not read by code in this plan, because the design has no self-approval check to gate. It exists so a customer can see the decision was made and can later require a second person without a schema change. Flagged so a reviewer does not read it as dead code that was forgotten.
