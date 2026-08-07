# Approvals Surface — Design Spec

**Date:** 2026-08-07
**Branch target:** `Development`
**Status:** Approved for planning
**Predecessor:** `docs/superpowers/specs/2026-08-06-guardrail-enforcement-layer-design.md` (merged as `18eb8e3`)

---

## 1. Problem

The guardrail layer verifies that a human approved an outbound email before it sends. Nothing can record that a human approved anything, so nothing sends.

`proc.bp_approval` has the right shape, the lookups work, revocation shadowing works, and all of it is tested. The table has zero rows and no authenticated surface writes to it.

A first attempt at a writer was reverted (`8d59acc`) because it was forgeable: it treated a caller-supplied `actioned_by` in a request body as proof of a human decision, over an unauthenticated route. That failure defines the central requirement here — **the approver's identity comes from the authenticated token and from nowhere else.**

## 2. The governing idea

An agent drafting an email is assisting the person who would otherwise have written it. So approving one is **authorship, not oversight**: the person approving is the person whose email it is. There is deliberately no segregation-of-duties rule for email — the drafter may approve.

What must not happen is an **agent** sending without a human, and that is a separate question governed by its own policy.

## 3. Goals

- An authenticated surface that records approvals for email dispatch and for negotiation rounds.
- Approver identity taken from the token only.
- An approval binds to what the approver actually saw, so editing a draft invalidates it.
- Agent-initiated sends governed by the autonomy policy that already exists, rather than a third mechanism.
- Every rule expressed as a policy row, not code (see §4).

## 4. Policy-as-configuration

A standing product principle, applied here and to be applied product-wide:

> Every decision about how the product should behave toward a customer's business is a policy row the customer owns. Decisions about how the software operates stay in code.

Concretely for this work: the `approve_email` capability, which roles hold it, whether an agent may send unattended, and the content-binding enforcement mode are **all policy rows**, shipped as seed data a customer adopts and edits. None is a Python constant.

The line matters in both directions. A retry count, a backoff timer or a connection pool size is not a policy — exposing it hands a customer a way to break their own deployment with no upside, and turns an implementation detail into a supported interface.

**Forward note (not in scope here):** small-business customers will share a deployment while larger ones get their own tenant. `bp_policy` and `bp_approval` have no customer column, so in a shared deployment one customer's approver could clear another's email. Scoping policy and approvals per customer is a hard requirement of the tenancy work that follows; nothing in this spec should assume global rows remain global.

## 5. Design

### 5.1 The `approve_email` capability — a policy row, not code

`RoleDefinitionPolicy` gains an action class `approve_email`, granted to **Buyer, Approver and Admin**, and listed in `irreversible_classes` so it inherits default-deny.

It sits alongside `transact` (spend approval), which stays **Approver and above**. The policy row then says plainly that approving your own outbound email and approving money are different capabilities, rather than one rank doing double duty. A customer who wants email approval restricted to Approvers changes one row.

`EmailDispatchApprovalPolicy.required_role` moves from `Approver` to `Buyer`, consistent with the above.

Delivered as a migration, and as an entry in the accelerator seed set (§5.7).

### 5.2 Endpoints

On the existing `decisions` router, which is already the human-in-the-loop surface.

| Endpoint | Purpose |
|---|---|
| `GET /decisions/approvals/pending` | Drafts and rounds awaiting a decision |
| `POST /decisions/approvals/dispatch/{unique_id}` | Approve a drafted email for sending |
| `POST /decisions/approvals/round/{workflow_id}/{round_num}` | Approve a negotiation round |
| `POST /decisions/approvals/{approval_id}/revoke` | Withdraw an approval |

All four take `principal=Depends(require_user)`. **`actioned_by` is `principal.subject`.** No request body field may influence who is recorded — the body carries what is being approved, never who approved it.

All four are gated by `guardrail.authorize("approval.email", "approve_email", principal, ...)`, so the capability check goes through the same gate as everything else.

`GET /pending` returns unsent drafts from `proc.draft_rfq_emails` and negotiation rounds sitting `pending`, each with the content hash a caller would be approving. Without it an approver has nothing to act on.

Revoke writes a **later row** with `status='revoked'` rather than mutating the original. `find_dispatch_approval` and `find_round_approval` already take the newest row for a key regardless of status and then require it to be approved and signed, so a revocation shadows the original. That behaviour is built and tested; this endpoint is its first real producer.

### 5.3 Content binding

At approval time, compute a stable hash over the draft's **recipients, subject, body and attachment identities**, and store it in `grounding.content_hash`.

The recipients hashed must be the ones the send path will actually use — that is, the output of `email_dispatch_guard.resolve_recipients(draft, None)`, not a raw column read. `proc.draft_rfq_emails` has `recipient_email` (singular) plus a `payload` blob, and the resolved list is what `send_draft` sends to; hashing anything else would let the approved set and the sent set diverge. The hash is computed by one shared helper called from both the approval endpoint and the send path, so the two can never drift.

The send path recomputes the hash from the draft it is about to send and refuses when it differs from the approval's.

Without this an approval is standing permission on a mutable object: approve a routine RFQ, someone edits the body to carry a competitor's price, and the original approval still releases it. With it, editing a draft invalidates its approval and it must be approved again — which is what "I approved that email" is normally taken to mean.

Enforcement mode is a policy key (`on_content_mismatch`), defaulting to `deny`. A customer may set it to `warn`, and the seed set ships `deny`.

Round approvals bind to the round number only; a negotiation round has no equivalent editable artefact.

### 5.4 Agent autonomy, through the policy that already exists

`resolve_authority` (`src/services/governance_tools/authority.py`) already reads `EmailReplyAutonomyPolicy` (#473) and is consulted by the orchestrator, the decision engine and `decisions.py`. **The send path added in the guardrail layer never consults it**, so two email-governance mechanisms currently coexist unaware of each other.

A send with no human approval is by definition agent-initiated. The send path consults `resolve_authority` for the acting agent:

- Authority granted → the send proceeds under that grant, recorded as such in the audit.
- Not granted → deny, as today.

`auto_reply_intents` is currently empty, so nothing is autonomous. Granting a specific agent autonomy for a specific intent is a policy edit, not a code change. This closes the split rather than adding a third mechanism.

### 5.5 Authenticating the existing decisions endpoints

`decide_finding`, `act_on_finding`, `decide_email_reply` and `act_on_email_reply` currently take the actor as `body.user_id or "api"` and have no authentication. These endpoints resolve findings and release supplier email replies, so the actor is load-bearing and presently forgeable.

Each gains `principal=Depends(require_user)`; the actor becomes `principal.subject`; the `user_id` body field is removed rather than kept as a fallback, since a fallback preserves the forgery whenever it is taken.

### 5.6 What an approval requires, stated plainly

1. An authenticated principal whose role holds `approve_email`.
2. A target that exists — a draft, or a workflow round.
3. A content hash captured at approval time (dispatch only).

A row satisfying `find_dispatch_approval` can be produced by no other path. `ApprovalsAgent` remains unable to produce one; its automated verdict stays non-findable, exactly as restored in `8d59acc`.

### 5.7 Accelerator seed set

The policy rows this work introduces ship as a seed file — `deploy/sql/seed/accelerator_policies.sql` — that a customer applies and then edits, rather than as behaviour compiled into the product. It carries the `approve_email` capability grant, the `EmailDispatchApprovalPolicy` role change, and the content-mismatch mode.

This is the first instalment of the wider accelerator pack; the pack itself and the product-wide sweep are separate work.

## 6. Data model changes

| Change | Object |
|---|---|
| New key | `grounding.content_hash` on dispatch approvals (jsonb, no DDL) |
| Policy edit | `RoleDefinitionPolicy`: add `approve_email` to Buyer/Approver/Admin `allow` and to `irreversible_classes` |
| Policy edit | `EmailDispatchApprovalPolicy`: `required_role` `Approver` → `Buyer`; add `on_content_mismatch: "deny"` |
| New file | `deploy/sql/seed/accelerator_policies.sql` (+ rollback) |

No new tables and no new columns.

## 7. Testing

Every guard broken on purpose and watched fail before it is trusted.

| Test | Breaks it by |
|---|---|
| Identity is not forgeable | Posting `actioned_by`/`user_id` in the body and asserting the token's subject is recorded |
| Unauthenticated approval | No principal → refused, nothing written |
| Capability enforced | A Viewer attempting to approve |
| Self-approval permitted | The drafter approving their own draft succeeds — the rule is deliberate, so it gets a test |
| Content binding | Approve, edit the body, attempt to send → refused |
| Content binding is real | Approve, send unchanged → allowed (so the check is not simply always-deny) |
| Revocation | Approve, revoke, attempt to send → refused |
| Agent autonomy denied | Agent-initiated send with empty `auto_reply_intents` → refused |
| Agent autonomy granted | Same with the intent granted by policy → allowed, and recorded as autonomous |
| Existing endpoints | `decide_finding` with no principal → refused |
| Policy is authoritative | Flipping the capability row changes who may approve, with no deploy |
| Live rows, not fixtures | The capability resolves from the live `bp_policy`, asserted in both directions |

The last two exist because this plan's predecessor was bitten eight times by tests that passed while the guard was dead, wrong or forgeable — most often because a fixture was richer than the database.

## 8. Scope boundary

BP_Backend only. No UI; that is a separate repository.

Out of scope and following separately: policy scoping (universal / role / agent / customer), the product-wide sweep converting hardcoded behavioural decisions to policy, and the full accelerator pack.

## 9. Accepted consequences

- Email cannot send until `ASK_AUTH_MODE=enforce` and a caller resolves to a role holding `approve_email`.
- Editing an approved draft invalidates its approval. Intended, and it will be visible to users as re-approval.
- The existing decisions endpoints stop accepting a body-supplied actor, which is a breaking change for any caller relying on it.
