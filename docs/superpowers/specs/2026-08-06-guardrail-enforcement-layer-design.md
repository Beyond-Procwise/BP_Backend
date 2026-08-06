# Guardrail Enforcement Layer — Design Spec

**Date:** 2026-08-06
**Branch:** `Development`
**Status:** Approved for planning
**Discovery input:** `docs/guardrail_policy_discovery_2026-08-06.md`

---

## 1. Problem

The discovery pass found that BP_Backend has policies but no enforcement. Four specific holes were authorised for closure:

1. **No role model.** `api/auth.py` verifies a Cognito ID token correctly but produces no role. The principle *"an agent can never exceed the permissions of the human it acts for"* has nothing to inherit from.
2. **Two open HITL bypasses.** `HITL_ENABLED=false` (`config/settings.py:347` → `negotiation_agent.py:2922`) switches human approval off globally; `hitl_auto_approve: true` in a request payload (`negotiation_agent.py:2999–3001`) lets a caller waive its own checkpoint.
3. **Droppable audit.** `services/agent_actions.py` writes are best-effort and savepoint-wrapped; a failed audit write is swallowed and the action proceeds unlogged (`agent_actions.py:125,150`).
4. **An ungoverned send path.** `EmailDispatchAgent.run()` accepts a caller-supplied `drafts` list including arbitrary `recipients` and passes them to SES. `_send_draft` and `EmailDispatchService.send_draft` validate only that a recipient *exists* (`email_dispatch_service.py:167`). No approval is checked, no allow-list consulted, no policy read, no content inspected.

The underlying cause is common to all four: **there is no seam where an irreversible action must stop and ask permission.** `PolicyEngine.validate_workflow()` implements exactly one workflow (`supplier_ranking`) and every other call falls through to `return {"allowed": True, "reason": "No policy checks"}` (`policy_engine.py:383`).

## 2. Goals

- One enforcement seam that every irreversible action passes through.
- All rule content stored in `proc.bp_policy`, changeable without a deploy.
- A role model bound to the authenticated principal, capping what any agent acting for that person may do.
- Email dispatch gated on approval, recipient allow-list, content sensitivity versus supplier clearance, policy, and volume.
- Audit that cannot be silently skipped for irreversible actions.

## 3. Non-goals

- **Tenant scoping.** No `tenant`/`customer` column exists on any of the 105 `bp_` tables. Multi-tenant isolation is out of scope and stays a recorded gap (G-c).
- **UI or gateway changes.** BP_Backend only.
- **Classifying every column in the database.** Content classification here applies to outbound email content, not to a corpus-wide labelling exercise.
- **Retro-fitting the gate to read/compute paths.** Reads stay ungated in this phase.

## 4. Design principles

| Principle | How it shows up |
|---|---|
| Policy decides, code detects | Python establishes facts (content class, caller role, approval on file). `bp_policy` decides what those facts permit. |
| Deny beats allow | Any single denying rule stops the action regardless of other permits. |
| Default-deny for irreversible | `communicate`, `transact`, `share`, `configure`, `delegate` require an explicit allow. |
| Fail closed | A missing, unparseable, or unreachable policy is a **deny**, never a pass. |
| Agents inherit, never hold | An agent has no role of its own; it executes as the invoking principal and is capped by that role. |
| No LLM in a gate | Every detector is deterministic and reproducible. A gate that can hallucinate or time out is not a gate. |

## 5. Component design

### 5.1 Policy content — six new `bp_policy` rows

No schema change. `bp_policy` already provides `policy_type`, `policy_details` (jsonb), `policy_linked_agents`, `version`, and the unique index on active (type, name). `PolicyEngine` already loads every active row generically, slugifies the name, and indexes it by alias — new types need no loader change.

**RBAC assignment ruling.** Any policy may carry `"required_role": "<Role>"` in `policy_details`, naming the minimum role needed to perform or waive what it governs. The engine reads this key uniformly, so future policies gain RBAC without new code.

#### `RoleDefinitionPolicy` — type `security`

```json
{
  "policy_identifier": "role_definition",
  "required_role": "Admin",
  "rules": {
    "roles": {
      "Viewer":   { "rank": 1, "allow": ["read"] },
      "Buyer":    { "rank": 2, "allow": ["read", "compute", "write"] },
      "Approver": { "rank": 3, "allow": ["read", "compute", "write", "communicate", "transact"] },
      "Admin":    { "rank": 4, "allow": ["read", "compute", "write", "communicate", "transact", "share", "configure", "delegate"] }
    },
    "irreversible_classes": ["communicate", "transact", "share", "configure", "delegate"],
    "on_missing_role": "deny"
  }
}
```

#### `RoleAssignmentPolicy` — type `security`

```json
{
  "policy_identifier": "role_assignment",
  "required_role": "Admin",
  "rules": {
    "group_to_role": {
      "bp-viewers": "Viewer",
      "bp-buyers": "Buyer",
      "bp-approvers": "Approver",
      "bp-admins": "Admin"
    },
    "claim": "cognito:groups",
    "no_principal_role": "Viewer",
    "unmapped_group_role": "Viewer",
    "multiple_groups": "highest_rank"
  }
}
```

#### `EmailDispatchApprovalPolicy` — type `email`

```json
{
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
}
```

#### `EmailRecipientAllowlistPolicy` — type `email`

```json
{
  "policy_identifier": "email_recipient_allowlist",
  "required_role": "Approver",
  "rules": {
    "sources": ["proc.bp_supplier.contact_email_1", "proc.bp_supplier.contact_email_2"],
    "match": "exact_casefold",
    "on_unknown_recipient": "deny_and_raise_review",
    "allow_recipients_from_email_body": false,
    "new_domain_requires_confirmation": true
  }
}
```

#### `EmailSensitivityPolicy` — type `email`

```json
{
  "policy_identifier": "email_sensitivity",
  "required_role": "Admin",
  "rules": {
    "classes": ["public", "internal", "commercial_confidential", "personal"],
    "order": { "public": 1, "internal": 2, "commercial_confidential": 3, "personal": 4 },
    "default_supplier_clearance": "internal",
    "detectors": {
      "third_party_price":       { "enabled": true, "raises_to": "commercial_confidential" },
      "contract_prose":          { "enabled": true, "raises_to": "commercial_confidential" },
      "internal_staff_contact":  { "enabled": true, "raises_to": "personal" },
      "source_document_attached":{ "enabled": true, "raises_to": "commercial_confidential" }
    },
    "rule": "content_class <= recipient_clearance",
    "on_undetermined_class": "deny",
    "on_missing_clearance": "use_default"
  }
}
```

#### `EmailVolumePolicy` — type `email`

```json
{
  "policy_identifier": "email_volume",
  "required_role": "Admin",
  "rules": {
    "max_per_run": 20,
    "max_per_user_per_day": 50,
    "on_exceeded": "pause_and_raise_review"
  }
}
```

All six link to the relevant agents via `policy_linked_agents` so `node_governance.governance_for()` resolves them.

### 5.2 Roles and the principal

**`Principal` gains `roles: list[str]` and `role: str`** (the highest-ranked). Derived from the `cognito:groups` claim the token already carries, mapped through `RoleAssignmentPolicy`.

**New module `src/services/rbac.py`:**

- `resolve_roles(principal) -> list[str]` — reads groups from the principal's claims, maps via policy, applies `multiple_groups: highest_rank`.
- `effective_role(principal) -> str` — returns `no_principal_role` (Viewer) when `principal is None`.
- `may(role, action_class) -> bool` — consults `RoleDefinitionPolicy`.

**New table `proc.bp_role_assignment`** — direct `subject → role` overrides for cases where a Cognito group is impractical. Permissions stay in policy; only the mapping is tabular.

| Column | Type | Note |
|---|---|---|
| `assignment_id` | bigserial PK | |
| `subject` | text | Cognito `sub` |
| `role` | text | must name a role in `RoleDefinitionPolicy` |
| `granted_by` | text | |
| `granted_at` | timestamptz | |
| `revoked_at` | timestamptz | null = active |

Index: `ix_bp_role_assignment_subject` (matches the `bp_`/`ix_bp_*` convention).

**No-principal behaviour.** With `ASK_AUTH_MODE=off`, `require_user` returns `None`. `effective_role(None)` is **Viewer**, so every irreversible action is refused with reason `"no authenticated principal"`. Read paths continue to work, so local demonstration of read surfaces is unaffected.

### 5.3 The gate — `src/services/guardrail.py`

Single entry point:

```python
authorize(action: str,
          action_class: str,
          principal: Principal | None,
          context: dict) -> Decision
```

`Decision` is a frozen dataclass: `allowed: bool`, `reason: str`, `policy_id: int | None`, `policy_name: str | None`, `policy_version: int | None`, `evidence: dict`.

Evaluation order:

1. Resolve `effective_role(principal)`.
2. If `action_class` is in `irreversible_classes` and the role does not allow it → **deny**.
3. Evaluate every policy applying to this action. Any denial → **deny** (deny beats allow).
4. If `action_class` is irreversible and no policy explicitly allowed it → **deny** (default-deny).
5. Otherwise allow, carrying the deciding policy's id and version.

Any exception during evaluation is caught and converted to a **deny** with the exception recorded in `evidence`. The gate never raises into the caller and never fails open.

`PolicyEngine` gains `policies_for_action(action)` so the gate has one lookup path. This extends the existing engine rather than introducing a parallel one — policies continue to load from exactly one place.

### 5.4 Email sensitivity

Two independent facts, checked together.

**Content class** — `src/services/email_sensitivity.py`, deterministic detectors, no model call:

| Detector | Fires when |
|---|---|
| `third_party_price` | The body or attachments carry a monetary figure attributable to a supplier other than the recipient (resolved via the deal's quote/PO rows). |
| `contract_prose` | Content matches contract-clause structure sourced from the existing obligations extractors. |
| `internal_staff_contact` | An internal email address or direct dial appears that the human did not add. Reuses the patterns already proven in `services/style/redaction.py`. |
| `source_document_attached` | An attachment resolves to a `bp_*_trgt`/`_stg` source document. |

The detectors are code — testable and reproducible. *Which detectors run, and what class each raises the content to,* is policy. Content class is the **highest** class any firing detector raises to; absent any firing detector the class is `internal`. An error inside a detector yields `undetermined`, which policy maps to **deny**.

**Supplier clearance** — new column:

```sql
ALTER TABLE proc.bp_supplier
  ADD COLUMN clearance_level text NOT NULL DEFAULT 'internal';
```

All 5,028 existing suppliers default to `internal`. Consequence: routine RFQs, acknowledgements and requirement text flow exactly as today; a competitor's price, contract prose, staff personal data or an attached source document to an uncleared supplier stops.

**Decision rule:** send proceeds only when `order[content_class] <= order[recipient_clearance]`.

### 5.5 The send path

`EmailDispatchAgent` stops treating caller-supplied input as authoritative. Recipients are **resolved from the stored draft record** (`proc.draft_rfq_emails`) and then validated. Five checks run in order before anything reaches SES; the first failure stops the send, records the reason, and raises a review item:

| # | Check | Source of truth | On failure |
|---|---|---|---|
| 1 | **Approval** | Row in `proc.bp_approval` with `status='approved'` and non-null `actioned_by`, matched on `rfq_id` **and** `workflow_id` (both present on the draft row and on `bp_approval`). Where a draft carries a `unique_id` but no `rfq_id`, the approval is matched on `workflow_id` plus the draft's `unique_id` recorded in the approval's `grounding`. An approval that matches on neither is treated as absent. | deny |
| 2 | **Allow-list** | Every recipient present in `bp_supplier.contact_email_1/2` | deny + review |
| 3 | **Sensitivity** | content class ≤ `bp_supplier.clearance_level` | deny + review |
| 4 | **Policy** | `authorize('email.send', 'communicate', principal, ctx)` | deny |
| 5 | **Volume** | `EmailVolumePolicy` counters. `max_per_run` is counted per dispatch invocation; `max_per_user_per_day` is counted against the principal's Cognito `sub` over a rolling 24 hours, derived from `bp_agent_actions` rows for `email.send`. Since check 4 already denies without a principal, there is always a subject to count against by the time this runs. | pause + review |

**Approval write path.** `proc.bp_approval` exists with the right shape but holds **zero rows** — nothing writes to it. This design therefore also builds the write: `ApprovalsAgent` and the decisions surface record an approval row when a human approves a draft for dispatch, carrying `actioned_by`, `actioned_at`, `policy_id` and `policy_name`. Verification reads that row; it never trusts a `sent_status`-style field on the input.

Placement: checks live in `EmailDispatchService.send_draft` — the single choke point both the agent and any other caller pass through — so a future caller cannot route around them.

### 5.6 Mandatory audit

`services/agent_actions.py` gains:

```python
def record_action_or_fail(*, phase: str, action_type: str, conn=None, **fields) -> None
```

Identical row shape to `record_action`, but **no savepoint swallow**: a failed write raises, and the calling action aborts. Irreversible actions use it. `record_action` keeps today's best-effort behaviour for read/compute, so a transient database blip cannot halt an extraction backlog.

Every gate decision — allow or deny — writes an audit row carrying the G8 fields: timestamp, agent, human principal, action verb, target ids, policy consulted and its version, decision, evidence refs, and egress destination where one applies.

### 5.7 HITL closure

- **Remove** the `hitl_auto_approve` payload path at `negotiation_agent.py:2999–3001`. A caller can no longer waive its own checkpoint; the field is ignored if present, and its presence is recorded as an audit event so attempted use is visible.
- **`hitl_enabled` may only narrow.** It can restrict which rounds require approval; it can no longer switch approval off. Where policy requires approval, `hitl_enabled=false` fails closed rather than auto-approving. `negotiation_agent.py:2922` reads the policy first and the setting second.

## 6. Data model changes

| Change | Object |
|---|---|
| New column | `proc.bp_supplier.clearance_level text NOT NULL DEFAULT 'internal'` |
| New table | `proc.bp_role_assignment` (+ `ix_bp_role_assignment_subject`) |
| New rows | 6 rows in `proc.bp_policy` |
| First writes | `proc.bp_approval` (table exists, currently empty) |

Deploy SQL goes in `deploy/sql/2026-08-06_guardrail_enforcement.sql` with a matching `_rollback.sql`, following the existing convention.

## 7. Testing

Per the standing rule, **every guard is broken on purpose and watched go red before the fix lands.** A guard that has never failed has not been shown to work.

| Test | Breaks it by |
|---|---|
| Approval gate | Dispatching a draft with no `bp_approval` row |
| Approval not trustable from input | Dispatching with an input payload claiming approval while the store has none |
| Allow-list | Recipient not on the supplier master |
| Body-introduced recipient | Address appearing only in an email body |
| Sensitivity | Competitor price to an `internal`-clearance supplier |
| Sensitivity fail-closed | Detector raising, asserting `undetermined` → deny |
| Role cap | Viewer attempting a send |
| No principal | `ASK_AUTH_MODE=off` attempting a send |
| Default-deny | Irreversible action with no policy present |
| Mandatory audit | Audit write forced to fail; assert the action aborts |
| HITL payload bypass | Request carrying `hitl_auto_approve: true` |
| HITL global switch | `hitl_enabled=false` where policy requires approval |
| Policy is authoritative | Flipping a `bp_policy` row changes behaviour with no code change |

Then verified on the running local server against the live database — not tests alone.

## 8. Rollout

1. Deploy SQL (column, table, policy rows). Policies land **active**; with `default_supplier_clearance: internal` the operational effect on existing traffic is nil.
2. Land `rbac.py`, `guardrail.py`, `email_sensitivity.py` with tests. Nothing calls them yet.
3. Wire the audit split (`record_action_or_fail`) on irreversible call sites.
4. Wire the send path checks in `EmailDispatchService.send_draft`.
5. Close the HITL bypasses.
6. Live verification on the local server.

Steps 2–5 are independently revertible. The riskiest step is 4, because it is the one that can stop mail leaving; it lands last and behind policy rows that can be relaxed without a deploy.

## 9. Dependencies on you

- **Cognito groups** `bp-viewers`, `bp-buyers`, `bp-approvers`, `bp-admins` must be created in the pool and users assigned. Until they exist every authenticated caller resolves to **Viewer** and irreversible actions are refused — the design degrades safely, but outbound mail will not flow until at least one Approver exists.
- **`ASK_AUTH_MODE`** stays `off` in this environment. Send remains blocked here for want of a principal; live send verification needs it set to `enforce` with a real token.

## 10. Accepted consequences

- Outbound email cannot be sent by an unauthenticated caller. This is the intended effect and it will be visible immediately in this environment.
- Any irreversible action whose audit write fails now fails. This is a deliberate reversal of current behaviour.
- Suppliers needing more than `internal` clearance must be raised explicitly — there is no automatic promotion.

## 11. Known gaps this does not close

Carried forward from discovery and **not** addressed here: tenant scoping (G-c), corpus-wide data classification (G-a), rate limits outside email (G-g), prompt-injection handling (G-h), the ungoverned web-research egress (E4), and the live Ollama Cloud credential in `.env` (E6).
