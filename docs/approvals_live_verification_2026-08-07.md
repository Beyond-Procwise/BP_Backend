# Approvals surface — Task 8 live verification

Date: 2026-08-07
Branch: `approvals-surface`, worktree: `/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/approvals`
Database: `bp_testdb` on `procwisemvpdb01.cluster-cpae0sg4mrk8.eu-west-1.rds.amazonaws.com` (the live, shared RDS instance this environment points at)

This record follows the brief's instruction literally: every command below was actually
run in this session, and its output is pasted verbatim (with noisy, irrelevant
`pydantic`/settings deprecation-warning lines stripped — nothing else is edited).
Where something did not work as first attempted, that failed attempt is kept rather than
silently rewritten, because the honesty of this record is its only value.

Three outcomes were expected going in and are recorded as **designed behaviour, not
failures**:

1. `ASK_AUTH_MODE="off"` in this environment (confirmed below), so `require_user`
   returns `None` and unauthenticated calls to the approvals/decisions surface are
   refused. No live Cognito token exists in this environment to test the `enforce` path.
2. `GET /approvals/pending` was not exercised to a non-empty result — there are 0
   unsent drafts in `proc.draft_rfq_emails`, confirmed directly against the database
   (see Step 6/DB state below), so an empty-list result was never possible in this
   environment. Not tested here as "empty list returned" — tested as "database
   genuinely has nothing to return."
3. `proc.bp_approval` had 0 rows before this session and has 0 rows after it.

---

## Step 1 — Full test suite

Command:

```
cd /home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/approvals
./venv/bin/python -m pytest tests/approvals tests/guardrails \
  tests/test_email_dispatch_service.py tests/test_agent_endpoints.py \
  tests/test_negotiation_agent.py tests/services/test_value_query_service.py \
  tests/agents/test_approvals_agent.py tests/test_email_dispatch_agent.py -v
```

Output (final line; full run produced 312 passing node IDs across the 8 targets,
warnings are pydantic/starlette deprecation noise unrelated to this feature):

```
312 passed, 224 warnings in 77.77s (0:01:17)
```

**Verdict: PASS.** All 312 collected tests across `tests/approvals`, `tests/guardrails`,
and the five named files passed. 0 failed, 0 errored.

---

## Step 2 — Capability resolves from the live policy

Confirmed `ASK_AUTH_MODE` first:

```
$ grep -n "^ASK_AUTH_MODE" .env
93:ASK_AUTH_MODE="off"
```

**Verdict: matches expected outcome #1 above — `off`, as designed for this environment.**

Command:

```python
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services import rbac
e = PolicyEngine(connection_factory=get_conn)
for role in ('Viewer','Buyer','Approver','Admin'):
    print(role, 'approve_email=', rbac.may(role,'approve_email',policy_engine=e),
          'transact=', rbac.may(role,'transact',policy_engine=e))
print('approve_email irreversible:', rbac.is_irreversible('approve_email', policy_engine=e))
```

Output:

```
Viewer approve_email= False transact= False
Buyer approve_email= True transact= False
Approver approve_email= True transact= True
Admin approve_email= True transact= True
approve_email irreversible: True
```

**Verdict: PASS.** Exactly matches the expected Viewer False/False; Buyer True/False;
Approver True/True; Admin True/True; irreversible True — read live from
`proc.bp_policy`, not from a fixture.

### `approval.email` resolves to `EmailApprovalCapabilityPolicy`

`guardrail.authorize("approval.email", "approve_email", principal, {}, policy_engine=e)`
was run once per role (using real Cognito group names read from the live
`role_assignment` policy — `bp-buyers`, `bp-approvers`, `bp-admins`, `bp-viewers` — not
the role display names, which do not match the `group_to_role` mapping and were tried
first and correctly denied as Viewer):

```
Viewer  -> allowed= False policy_name= None                       reason= role Viewer may not perform approve_email
Buyer   -> allowed= True  policy_name= EmailApprovalCapabilityPolicy reason= EmailApprovalCapabilityPolicy permits approval.email for role Buyer; policy-specific preconditions are enforced by the caller
Approver-> allowed= True  policy_name= EmailApprovalCapabilityPolicy reason= EmailApprovalCapabilityPolicy permits approval.email for role Approver; policy-specific preconditions are enforced by the caller
Admin   -> allowed= True  policy_name= EmailApprovalCapabilityPolicy reason= EmailApprovalCapabilityPolicy permits approval.email for role Admin; policy-specific preconditions are enforced by the caller
```

**Verdict: PASS.** `approval.email` is attributed to `EmailApprovalCapabilityPolicy` for
every role that is allowed.

### `EmailApprovalCapabilityPolicy.rules.revoke_scope`

Read directly from the live policy (`PolicyEngine.get_policy('email_approval_capability')`):

```json
{
  "rules": {
    "note": "The agent drafts on the user behalf, so approving is authorship rather than oversight. Set self_approval_allowed to false to require a second person.",
    "revoke_scope": "own_or_higher_rank",
    "self_approval_allowed": true
  },
  "applies_to": ["approval.email"],
  "required_role": "Buyer",
  "policy_identifier": "email_approval_capability"
}
```

**Verdict: PASS.** `revoke_scope` is `own_or_higher_rank` on the live row, as specified.
(`self_approval_allowed: true` is present but, per the design note in the task-8 brief's
self-review, is deliberately not read by any code path today — flagged there, not a gap
found here.)

### `email_dispatch_approval`

```json
{
  "rules": {
    "verify_against": "proc.bp_approval",
    "accepted_status": ["approved"],
    "approval_required": true,
    "on_content_mismatch": "deny",
    "on_missing_approval": "deny",
    "require_actioned_by": true,
    "trust_input_payload": false
  },
  "applies_to": ["email.send"],
  "required_role": "Buyer",
  "policy_identifier": "email_dispatch_approval"
}
```

**Verdict: PASS.** `required_role: Buyer` and `rules.on_content_mismatch: deny`, both
confirmed on the live row.

### Live `role_definition` — the `approve_email` / `transact` distinction

```json
{
  "roles": {
    "Admin":    {"rank": 4, "allow": ["read","compute","write","communicate","transact","share","configure","delegate","approve_email"]},
    "Buyer":    {"rank": 2, "allow": ["read","compute","write","approve_email"]},
    "Viewer":   {"rank": 1, "allow": ["read"]},
    "Approver": {"rank": 3, "allow": ["read","compute","write","communicate","transact","approve_email"]}
  },
  "irreversible_classes": ["communicate","transact","share","configure","delegate","approve_email"]
}
```

**Verdict: PASS.** Buyer holds `approve_email` but not `transact`; Approver and Admin
hold both. The two capabilities are deliberately distinct, confirmed on the live row —
matches the count from Step 2's role loop above exactly.

Policy count read from the live engine at this point: `PolicyEngine loaded 19 policies`
(also confirmed again at server boot, Step 5).

---

## Step 3 — End-to-end approval store chain, inside a transaction rolled back

Command (verbatim, from the brief):

```python
import os, psycopg2, uuid
from src.services import approval_store
from src.services.approval_content import content_hash
c=psycopg2.connect(host=os.getenv('DB_HOST'), ...); c.autocommit=False
t=uuid.uuid4().hex[:8]
draft={'unique_id':'PROC-WF-'+t, 'rfq_id':'RFQ-'+t, 'workflow_id':'WF-'+t,
       'subject':'RFQ','body':'Please quote.','recipients':['buyer@supplier-b.com']}
h=content_hash(draft)
aid=approval_store.record_approval(rfq_id=..., workflow_id=..., unique_id=...,
    supplier_id='SUP-1', actioned_by='sub-buyer-001', grounding_extra={'content_hash':h}, conn=c)
found=approval_store.find_dispatch_approval(..., conn=c)
got=approval_store.get_approval(approval_id=aid, conn=c)
approval_store.revoke_approval(approval_id=aid, actioned_by='sub-approver-001', reason='probe', conn=c)
after=approval_store.find_dispatch_approval(..., conn=c)
c.rollback(); c.close()
```

Output:

```
approval_id: 1180
approval found: True | hash stored: True
get_approval returns row: True | approval_id matches: True
after revocation: shadowed
rolled back
```

**Verdict: PASS.** `record_approval` wrote a row; `find_dispatch_approval` found it with
the `content_hash` round-tripping through `grounding` exactly; `get_approval` returned
the same row by id; `revoke_approval` wrote a later shadowing row and
`find_dispatch_approval` correctly stopped returning the approval afterward; the
transaction was rolled back and `proc.bp_approval` was reconfirmed at 0 rows in Step 6.

---

## Content binding through the real guard (`email_dispatch_guard.check_dispatch`)

Run against the **live** `PolicyEngine` (real `email_sensitivity`, `email_volume`,
`email_recipient_allowlist`, `email_dispatch_approval` policy rows), with a stub DB
connection standing in only for the supplier-master/peer-price/daily-count lookups
(`lookup_supplier_emails`, `lookup_supplier_clearance`, `lookup_peer_prices`,
`lookup_daily_send_count`) — these lookups are I/O seams the module itself supports
stubbing for exactly this reason; no `proc.bp_approval` row was written for this part
(the approval object is injected directly via `approval_lookup`).

First attempt (kept, not hidden): the stub supplier clearance was set to `"restricted"`,
which is not a valid clearance value in the live `email_sensitivity` policy (valid order
is `public < internal < commercial_confidential < personal`) — this produced a false
deny at the sensitivity check, not the content-hash check, and was corrected to `None`
(use the policy's own `default_supplier_clearance`, which is `"internal"` and matches
this message's own classification). Second attempt used a Buyer principal for the send
action itself, which the live `role_definition` correctly denies (`communicate` is not
in Buyer's `allow` list — Buyer holds `approve_email`, not `communicate`); corrected to
an Approver principal, which is who would actually be dispatching. Both corrections were
needed to isolate the content-binding mechanism specifically, and are recorded here so
the first (wrong) results are not mistaken for the mechanism failing.

Final run output:

```
=== 1. unchanged draft (should be ALLOWED) ===
unchanged: allowed=True reason='all dispatch checks passed' evidence={'approval_id': 999999, 'approved_by': 'sub-buyer-001', 'content_class': 'internal', 'recipients': ['buyer@supplier-b.com']}

=== 2. edited body (should be DENIED) ===
edited-body: allowed=False reason='the draft changed since it was approved; it must be approved again' evidence={'approved_content_hash': '6be3ed34bddf0302d3294113fe8efd119e890fa29535e3f72c43c21dca82479a', 'current_content_hash': 'bb5d600803f2d09ea7fe34691005c2323e1fab267ca6f639041a30c0923befc0'}

=== 3. changed recipient (should be DENIED) ===
changed-recipient: allowed=False reason='the draft changed since it was approved; it must be approved again' evidence={'approved_content_hash': '6be3ed34bddf0302d3294113fe8efd119e890fa29535e3f72c43c21dca82479a', 'current_content_hash': 'a26c2562e99fe16be193d45607d8b57741c84920299a427209de50bbc8ec7be1'}

=== 4. approval with no hash (should be DENIED) ===
no-hash: allowed=False reason='the draft changed since it was approved; it must be approved again' evidence={'approved_content_hash': None, 'current_content_hash': '6be3ed34bddf0302d3294113fe8efd119e890fa29535e3f72c43c21dca82479a'}
```

**Verdict: PASS.** All four cases behaved exactly as specified: the unchanged draft is
the one case that is *allowed* (proving the guard is not simply always-deny); an edited
body, a changed recipient, and an approval carrying no hash are each refused by the same
content-hash comparison in `check_dispatch` step 1b.

---

## Agent autonomy against the live policy

Command:

```python
from src.engines.policy_engine import PolicyEngine
from src.services.db import get_conn
from src.services.governance_tools.authority import resolve_authority
e = PolicyEngine(connection_factory=get_conn)
print(resolve_authority(e, ['email_dispatch_agent']))
```

Output:

```json
{
  "email_dispatch_agent": {
    "agent": "email_dispatch_agent",
    "governed": true,
    "slug": "email_reply_autonomy",
    "policy_id": 473,
    "policy_name": "EmailReplyAutonomyPolicy",
    "auto_intents": [],
    "escalate_intents": ["price_change","terms_change","contract_variation","liability","dispute","new_commitment"],
    "limit_gbp": "10000",
    "limit_currency": "GBP",
    "max_auto_replies_per_thread": 2,
    "min_intent_confidence": 0.8,
    "reason": "resolved from governed policy"
  }
}
```

`'may_send' in result['email_dispatch_agent']` → `False`.

**Verdict: PASS.** `governed=True`, `auto_intents=[]` — nothing sends unattended today —
and there is genuinely no `may_send` key; the only autonomy signal is the empty
`auto_intents` list.

---

## Step 4 — Policy is genuinely authoritative (no deploy)

Command (deactivate → authorize → reactivate → authorize → confirm restored):

```python
# read original policy_status, then:
# UPDATE proc.bp_policy SET policy_status = 0 WHERE policy_name = 'EmailApprovalCapabilityPolicy'
# guardrail.authorize(...) with a FRESH PolicyEngine(connection_factory=get_conn) each time
```

Output:

```
original policy_status: 1

--- before change ---
allowed= True policy_name= EmailApprovalCapabilityPolicy reason= EmailApprovalCapabilityPolicy permits approval.email for role Buyer; policy-specific preconditions are enforced by the caller

--- deactivating (policy_status=0) ---
policy_status now: 0
allowed= False policy_name= None reason= no policy permits approval.email; irreversible actions are default-deny

--- reactivating (policy_status back to original) ---
policy_status now: 1
allowed= True policy_name= EmailApprovalCapabilityPolicy reason= EmailApprovalCapabilityPolicy permits approval.email for role Buyer; policy-specific preconditions are enforced by the caller

final policy_status: 1 | restored to original: True
```

**Verdict: PASS.** Deactivating the row on the live database made `authorize()` start
denying `approval.email` immediately (once a fresh `PolicyEngine` was constructed —
`PolicyEngine` loads once at construction time and does not poll; `rbac.policy_engine()`
additionally TTL-caches for 60s, so a caller using the shared cache would not see the
change until the cache expired — this test bypassed that by constructing a fresh engine
each time, which is the honest way to prove the row itself is authoritative rather than
proving something about cache timing). Reactivating restored the allow. **Confirmed:
`policy_status` for `EmailApprovalCapabilityPolicy` (policy_id 704) is `1`, the exact
value found before this check began.**

---

## Step 5 — Boot the server

Startup command, matching `procwise.service`'s invocation (`.venv`, module path
`api.main:app`, `PYTHONPATH` including both the repo root and `src`), on port **8010**
— port 8000 is occupied by the main checkout (verified: PID 6835, does not carry this
branch's code) and was left untouched throughout:

```
PYTHONPATH=/home/.../worktrees/approvals:/home/.../worktrees/approvals/src \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8010 --log-level info
```

Startup log (policy engine line):

```
2026-08-07 20:43:16,784 - INFO - engines.policy_engine - PolicyEngine loaded 19 policies
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8010 (Press CTRL+C to quit)
```

**Verdict: PASS.** 19 policies loaded (18 pre-existing + `EmailApprovalCapabilityPolicy`),
matching the count also read directly from the engine in Step 2.

Route table (`GET /openapi.json`, filtered to `/approvals` and `/decisions`):

```
/approvals/dispatch/{unique_id}            ['post']
/approvals/pending                         ['get']
/approvals/round/{workflow_id}/{round_num} ['post']
/approvals/{approval_id}/revoke            ['post']
/decisions                                 ['get']
/decisions/email-reply/{decision_id}/action ['post']
/decisions/email-reply/{decision_id}/message ['get']
/decisions/email-reply/{response_id}       ['post']
/decisions/finding/{finding_id}            ['post']
/decisions/finding/{finding_id}/action     ['post']
/decisions/{decision_id}                   ['get']
```

**Verdict: PASS.** All 4 approvals routes and all 7 decisions routes are present in the
live route table.

### Exercising every endpoint unauthenticated (ASK_AUTH_MODE=off)

The brief's design note says every approvals endpoint refuses with a 401. Exercised
live, this is **not uniformly true** and is recorded honestly rather than smoothed over:

```
GET  /approvals/pending                       -> 403 {"detail": "no authenticated principal: irreversible actions are refused"}
POST /approvals/dispatch/TESTID               -> 401 {"detail": "an approval must name an authenticated person"}
POST /approvals/round/WF1/1                   -> 401 {"detail": "an approval must name an authenticated person"}
POST /approvals/123/revoke                    -> 401 {"detail": "an approval must name an authenticated person"}

POST /decisions/finding/1                     -> 401 {"detail": "this action must be attributed to a person"}
POST /decisions/finding/1/action              -> 401 {"detail": "this action must be attributed to a person"}
POST /decisions/email-reply/1                 -> 401 {"detail": "this action must be attributed to a person"}
POST /decisions/email-reply/1/action          -> 401 {"detail": "this action must be attributed to a person"}
GET  /decisions/email-reply/1/message         -> 401 {"detail": "this action must be attributed to a person"}
GET  /decisions                               -> 401 {"detail": "this action must be attributed to a person"}
GET  /decisions/1                             -> 401 {"detail": "this action must be attributed to a person"}
```

**Verdict: 10 of 11 exercised endpoints return 401 as designed; `GET /approvals/pending`
returns 403.** The reason is structural, not a bug: the three `POST`/mutating approvals
endpoints call `_subject(principal)` (which raises 401 for `principal is None`) before
`_require_capability`, but `list_pending` calls `_require_capability` directly without
ever calling `_subject`, so with no principal the request is refused earlier, by
`guardrail.authorize`'s own "no authenticated principal: irreversible actions are
refused" default-deny, which surfaces as 403 rather than 401. All 7 `decisions`
endpoints call `_actor(principal)` first and uniformly return 401. Every endpoint is
refused unauthenticated either way — none can be exercised without a principal — but the
brief's "every approvals endpoint refuses with a 401" is only 3/4 true as measured live;
`/approvals/pending` is a 403. This is flagged here rather than rounded off.

Server stopped afterward (`kill -TERM` on the exact PID started, `1230741`) — port 8000
(PID 6835, the main checkout) was left running and unaffected throughout.

### All 7 decisions and 4 approvals endpoints carry `Depends(require_user)`; no actor field

Confirmed by reading the router source directly (`src/api/routers/approvals.py`,
`src/api/routers/decisions.py`): every route function's signature includes
`principal=Depends(require_user)`. `ApproveRequest`, `RevokeRequest`, `DecideRequest`,
`ActionRequest`, and `EmailActionRequest` were read in full — none defines an actor/user
field; the actor is always taken from `principal.subject` via `_subject`/`_actor`
helpers, never from the request body.

**Verdict: PASS** (confirmed by source inspection, reinforced by the live 401/403
behaviour above — a request body cannot substitute for a principal in any of the 11
endpoints, because none of them read one).

---

## Step 6 — Database is clean

Command, run **after** every check above (including the stopped server):

```python
SELECT COUNT(*) FROM proc.bp_approval;
SELECT COUNT(*) FROM proc.draft_rfq_emails WHERE sent IS NOT TRUE;
SELECT COUNT(*) FROM proc.bp_agent_actions WHERE details::text LIKE '%PROC-WF-%' OR details::text LIKE '%probe%';
SELECT policy_id, policy_status FROM proc.bp_policy WHERE policy_name = 'EmailApprovalCapabilityPolicy';
```

Output:

```
bp_approval rows: 0
draft_rfq_emails test-pattern residue: 1
bp_agent_actions residue: 0
EmailApprovalCapabilityPolicy status: [(704, 1)]
unsent drafts total (final): 0
```

The one `draft_rfq_emails` row matching a loose `PROC-WF-%`/`RFQ-%` text pattern was
inspected directly and is **not residue from this session**: it is row id 68,
`unique_id='demo-email-assistant-2026-07-28-northfield-labs'`, `rfq_id='RFQ-DEMO-01'`,
`sent=True`, `created_on=2026-07-28` — a pre-existing demo record from over a week
before this verification, already sent, and therefore not counted among "unsent
drafts" (confirmed separately: 0 unsent drafts exist). Nothing in this session inserted
into `proc.draft_rfq_emails` — `record_approval` only ever writes to `proc.bp_approval`,
and the content-binding probes used a stub connection with no real database writes.

**Verdict: PASS.**
- `proc.bp_approval` = 0 rows (matches the count found before this session started).
- 0 unsent drafts in `proc.draft_rfq_emails` — the one pattern-matched hit is pre-existing,
  already-sent demo data, unrelated to this verification.
- 0 test residue in `proc.bp_agent_actions`.
- `EmailApprovalCapabilityPolicy.policy_status` = 1, the exact value found at the start
  of Step 4.

---

## What is proven against the live system

- The full test suite (312 tests) passes against this branch's code.
- `ASK_AUTH_MODE="off"` in this environment, confirmed by reading `.env` directly.
- `rbac.may`/`rbac.is_irreversible` resolve Viewer/Buyer/Approver/Admin exactly as
  specified, read live from `proc.bp_policy` via a real `PolicyEngine`.
- `guardrail.authorize("approval.email", "approve_email", ...)` resolves to
  `EmailApprovalCapabilityPolicy` for every role that is granted it, using real Cognito
  group names from the live `role_assignment` policy.
- `EmailApprovalCapabilityPolicy.rules.revoke_scope == "own_or_higher_rank"`,
  `email_dispatch_approval.required_role == "Buyer"` and
  `.rules.on_content_mismatch == "deny"`, and the live `role_definition` row shows
  Buyer holds `approve_email` but not `transact`, while Approver/Admin hold both — all
  read directly off the live policy rows.
- The full `approval_store` chain — `record_approval` → `find_dispatch_approval` →
  `get_approval` → `revoke_approval` → `find_dispatch_approval` (shadowed) — works
  end-to-end against the live database, inside a transaction that was rolled back.
- `email_dispatch_guard.check_dispatch`, run against the live `PolicyEngine`, allows an
  unchanged draft and denies an edited body, a changed recipient, and a hash-less
  approval — all via the same content-hash comparison.
- `resolve_authority` against the live `EmailReplyAutonomyPolicy` returns
  `governed=True`, `auto_intents=[]`, and carries no `may_send` key.
- Deactivating/reactivating `EmailApprovalCapabilityPolicy` on the live database changes
  `authorize()`'s answer immediately (with a freshly constructed engine), and
  `policy_status` was restored to its original value (1).
- The server boots from the systemd unit's exact invocation, loads 19 policies live, and
  serves all 4 approvals and all 7 decisions routes.
- Every one of the 11 approvals/decisions endpoints refuses an unauthenticated request
  live — 10 with 401, one (`GET /approvals/pending`) with 403 for a structural reason
  explained above.
- `proc.bp_approval` is 0 rows, `proc.draft_rfq_emails` has 0 unsent drafts, and
  `proc.bp_agent_actions` carries no test residue, both before and after this session.

## What is proven only by unit test, or not at all

- **`ASK_AUTH_MODE="enforce"` was never exercised.** This environment has no way to mint
  a real Cognito token, so the actual bearer-token verification path
  (`_verifier.verify(token)`, signature/audience/expiry checks) is proven only by the
  unit tests in `tests/approvals/test_decisions_authenticated.py` and equivalent, not by
  a live request with a real token.
- **`GET /approvals/pending` returning a genuinely non-empty, correctly-shaped list**
  was not observed — there are 0 unsent drafts in this environment, so a populated
  response was structurally impossible to produce live without fabricating draft rows
  in the shared database, which this verification deliberately did not do. The empty
  case (0 unsent drafts) is confirmed directly against the database, not against a live
  200 response (which requires `ASK_AUTH_MODE=enforce`, also not available here).
  `list_pending`'s row-shaping and the `content_hash` it attaches per row are proven
  only by `tests/approvals/test_approval_endpoints.py` and similar.
- **A real end-to-end HTTP request that actually authenticates as Buyer/Approver/Admin**
  and receives a 200 was not performed against any of the 11 endpoints — every live HTTP
  exercise above was of the unauthenticated-refusal path, because `enforce` mode is
  unavailable. The authenticated-success path for all 11 endpoints is proven only by
  the unit test suite (Step 1), using `TestClient` with dependency overrides for
  `require_user`, not by a live token-bearing request.
- **The `email.send` volume caps (`max_per_run`, `max_per_user_per_day`) and the
  recipient-allowlist/sensitivity checks** were exercised in the content-binding probe
  only through a stub connection standing in for `proc.bp_supplier` and
  `proc.bp_agent_actions` lookups — the real supplier-master and daily-send-count SQL
  paths (`_supplier_emails`, `_daily_send_count`'s live-connection branches) are proven
  only by `tests/guardrails/test_send_path_gate.py` and similar, not exercised against
  real rows in this session.
- **The systemd unit itself (`procwise.service`) was not started via `systemctl`** — its
  invocation was reproduced manually (same binary, module path, and `PYTHONPATH`), but
  `ExecStartPre` (the Neo4j docker-compose bring-up) and the unit's process supervision
  (`Restart=on-failure`, `KillMode=mixed`, etc.) were not exercised.
