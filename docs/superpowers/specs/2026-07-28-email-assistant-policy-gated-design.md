# Policy-gated email assistant — design

**Date:** 2026-07-28
**Branch:** Development
**Status:** approved, ready for implementation planning

## 1. What this is

An executive assistant for supplier email, not an email client.

When a supplier replies, the system decides — against a policy held in the policy
engine — whether the reply is small enough to answer by itself, or consequential
enough that a human must see it. Small things are answered and never surface.
Consequential things become **one item in the existing Action Centre queue**, whose
detail pane is a restored version of the email review panel (supplier's message,
editable draft, tone, attachments, Send).

Explicitly out of scope: a bespoke email module, a mailbox UI, a new nav item, an
inbox. Email is a channel the agent works in; the human's surface is the Todo list
they already use.

## 2. Why this shape

The pieces already exist and are load-bearing elsewhere:

| Capability | Where it already lives | State |
|---|---|---|
| Todo list with an "Approve agent" tab | `engine.js:1861` `actionCentreView()` | live; its buttons already route to the decision engine |
| Mail icon for such a card | `engine.js:1674` `ACicon.mail` | defined, unused |
| Fact-gathering + evidence + policy → resolve/escalate | `src/engines/decision_engine.py` `decide_finding()` | live; 9 decisions in `proc.bp_decision` (5 escalated) |
| Governed limits as data | `proc.bp_policy` (10 rows), `ApprovalThresholdPolicy` | live; £10,000, `on_above: escalate` |
| Reply drafting that quotes the thread | `src/agents/email_drafting_agent.py:733` negotiation mode | live; 18 drafts persisted |
| Inbound capture + thread matching | `src/services/email_watcher.py`, `email_watcher_agent.py` | live; matches on `In-Reply-To`/`References` |
| Threaded send with edits | `POST /workflows/email` + `EmailDispatchService` | live; `subject_override`/`body_override`, sets reply headers |
| MIME attachments | `src/services/email_service.py:93-102` | live at the SMTP layer |
| HITL halt + resume | `src/api/routers/agent_workflows.py:132-141` | live; persists before returning |

## 3. Audit: how the product actually works today

This design was approved on the understanding that the intended architecture —
*orchestrator resolves the applicable policy, tells the agent its limit, agent then
sends or escalates* — is **partially built**. Verified against the code:

**True today**
- The orchestrator resolves and injects governance per workflow:
  `_apply_governance_envelope()` (`orchestrator.py:286`) writes a `governed` block into
  `enriched_input` and `context.input_data`, and audits the fact.
- HITL halt/resume works and persists (`agent_workflows.py:132-141`).
- One agent genuinely pulls its own governed limit: `ApprovalsAgent`
  (`approvals_agent.py:75`, slug `approval_threshold`).

**Not true today — the gaps this work closes**
1. **Resolution is sequential**, a single `resolve_governance(workflow_name)` call.
   Parallelism exists elsewhere (`dag_scheduler.py`, `reasoning_engine` `parallel_group`)
   but never for policy; and `use_dag_scheduler` is read as
   `getattr(self.settings, 'use_dag_scheduler', False)` with no such setting defined
   anywhere, so that path is dead code.
2. **Agents are not informed.** Exactly 1 of 14 agents reads the envelope
   (`supplier_ranking_agent.py:572`). The orchestrator says so: *"additive — agents
   unaffected unless they read `governed`"*.
3. **"Which policy applies" is a static map, not a resolution.** `envelope.py` hardcodes
   5 workflow→agent pairs; `node_governance.py` states in its own docstring that it is
   *"a STATIC LOOKUP … not a resolution through PromptEngine/PolicyEngine's selection
   logic"*, and that only 5 of 14 agents have any governance.
4. **Workflow policy validation is a no-op** outside supplier_ranking:
   `PolicyEngine.validate_workflow()` returns `{"allowed": True, "reason": "No policy checks"}`.
5. **The email path has no policy at all.** Live query over `bp_policy`: zero policies
   linked to `negotiation_agent`, `email_drafting_agent`, `email_dispatch_agent`,
   `supplier_interaction_agent`, `email_watcher_agent`. (Negotiation has a linked prompt,
   no policy.)
6. **Governance is fail-open by design** — "a governance problem never breaks a workflow".
   Correct for prompts, wrong for an authority limit. `DecisionEngine` already sets the
   right precedent: no threshold policy → escalate and say why.

Live data state (bp_sqldb, 2026-07-28): `draft_rfq_emails` 18 rows / **0 sent**;
`supplier_response` 1 row and it is **simulated** (`<simulated-…@example.invalid>`);
`bp_decision` 9 rows; `bp_policy` 10 rows. `.env` currently points at `bp_testdb`, where
all of these are empty — implementation and demo must run against `bp_sqldb`.

## 4. Architecture

```
inbound reply (IMAP)                     human-initiated draft
   │ email_watcher → proc.supplier_response      │
   ▼                                             ▼
DecisionEngine.decide_email_reply(response_id) ──┘
   │  facts: supplier_response + matched draft + deal
   │  intent + commercial deltas: AgentNick, grounded (must quote source sentence)
   │  authority: resolve_authority() → email_reply_autonomy (+ approval_threshold)
   ├── resolution = resolved  → EmailDispatchService.send_draft (threaded)
   │                            audit only; NEVER enters the Todo list
   └── resolution = escalated → proc.bp_decision (subject_type='email_reply', status='open')
                                → Action Centre card, "Approve agent" tab
                                → detail pane = restored email panel
                                → human edits + attaches + Send
                                → POST /workflows/email (overrides) + human action recorded
```

### 4.1 Policy — the limit lives in the engine, as data

One new `proc.bp_policy` row. Nothing about the limit is hardcoded in app code.

- `policy_name`: `EmailReplyAutonomyPolicy`
- `policy_type`: `email_autonomy`
- `policy_linked_agents`: `email_drafting_agent, negotiation_agent, supplier_interaction_agent`
- `policy_details.policy_identifier`: `email_reply_autonomy`
- `policy_details.rules`:

```json
{
  "auto_reply_intents": [],
  "escalate_intents": ["price_change", "terms_change", "contract_variation",
                       "liability", "dispute", "new_commitment"],
  "defer_value_limit_to": "approval_threshold",
  "max_auto_replies_per_thread": 2,
  "min_intent_confidence": 0.8,
  "on_missing_policy": "escalate",
  "on_ungrounded_facts": "escalate"
}
```

`auto_reply_intents` ships **empty**: on day one every reply escalates, so the behaviour
is conservative by configuration rather than by code. Widening it is a Policies-screen
edit, no deployment. Money authority is **not** duplicated — `defer_value_limit_to`
points at the existing `approval_threshold` policy so there remains exactly one limit.

Candidate auto intents, for when the user chooses to widen (documented, not enabled):
`acknowledge`, `confirm_receipt`, `request_missing_document`, `chase_no_response`,
`clarify_lead_time`.

### 4.2 Authority resolution — orchestrator-side, parallel, fail-closed

New `resolve_authority(workflow_name, agents)` in `src/services/governance_tools/envelope.py`
(or a sibling module if that file grows past clarity):

- Resolves **through `PolicyEngine.get_policy()`'s own selection logic**, not the static
  `_WORKFLOW_AGENT` map, so the row reported is the row that would actually apply.
- Resolves the agents of the run **concurrently** (independent DB reads,
  `ThreadPoolExecutor`), since resolution per agent is independent and serial DB
  round-trips are the only cost.
- Injects `context.input_data["authority"] = {<agent>: {...}}` with: `slug`, `version`,
  `limit_gbp` (from the deferred approval policy), `auto_intents`, `escalate_intents`,
  `max_auto_replies_per_thread`, `min_intent_confidence`, `fail_mode`.
- **Fails closed.** Any error, missing policy, or unparseable rules yields an authority
  block with `fail_mode: "escalate"` and a reason string. The existing fail-open
  `governed` envelope is left exactly as it is (prompts must not start breaking runs).

The existing envelope and the new authority block are separate keys with separate failure
semantics; that difference is the point, and must be stated in the module docstring.

### 4.3 Agents read their limit

The email path reads `context.input_data["authority"]["email_drafting_agent"]`. Absent,
malformed, or `fail_mode: escalate` → escalate, with that recorded as the decision
rationale. Scope discipline: only this path is changed. The other 12 agents that ignore
governance are a known, documented gap and are not touched here.

### 4.4 The decision

`DecisionEngine.decide_email_reply(response_id, *, requested=None)`, a sibling to
`decide_finding`, returning the same `Decision` dataclass so `record()`, `execute()`,
`trace()` and the router are reused unchanged.

Facts and evidence (every fact carries `source` and `reference`, as `decide_finding` does):
- from `proc.supplier_response`: `supplier_id`, `subject`, `body`, `from_address`,
  `round_number`, `match_confidence`, `price`, `currency`, `payment_terms`, `lead_time`
- from the matched draft (`proc.draft_rfq_emails` via `matched_sent_email_id` /
  `unique_id`): prior offer, subject, thread headers, reply count on the thread
- derived: value at stake — the delta between the supplier's number and the prior offer,
  in GBP. **The `computed_value` two-conventions trap does not apply here** (that is a
  `bp_extraction_discrepancy` concern), but the same discipline does: state the
  derivation in the rationale and cite both numbers.
- classification: AgentNick (the only base model — never another model) returns
  `intent`, `confidence`, and a **verbatim quoted sentence** from the supplier's body
  supporting it. The quote is checked against the stored body with the format-tolerant
  grounding guard; a quote that cannot be matched → escalate. Note
  `extraction_v3.is_value_grounded` is unsafe for sentences and must not be used here.

Outcome:
- `intent ∈ escalate_intents`, or `value_at_stake > limit_gbp`, or
  `confidence < min_intent_confidence`, or ungrounded, or authority missing,
  or thread already at `max_auto_replies_per_thread` → `resolution = escalated`
- `intent ∈ auto_reply_intents` and every above check passes → `resolution = resolved`,
  `decision = "send"`

Persisted to `proc.bp_decision` with `subject_type='email_reply'`,
`subject_id=<draft unique_id>`, `deal_id`, `supplier_id`, plus facts and evidence JSON.
No schema change: the table already carries everything needed.

### 4.5 Auto path

`DecisionEngine.execute()` dispatches via `EmailDispatchService.send_draft()` — the
existing threaded-reply path, `In-Reply-To`/`References` already handled. Audited to
`bp_agent_actions` and the decision trace. It does **not** create a Todo item; the human
sees it only in Alerts / Audit logs, or by opening the decision trace.

### 4.6 Escalated path — the Todo item and the restored panel

**Queue.** Escalated email decisions surface as cards in the Action Centre's existing
"Approve agent" tab, fetched by a new read endpoint
`GET /decisions?subject_type=email_reply&status=open` (list + count). Card: mail icon
(`ACicon.mail`), title = supplier + intent, description = the agent's one-line rationale,
tags = value at stake and policy name, buttons Review / Send / Reject.

**Detail pane.** `actionDetail()` gains an email branch — the restored panel, ported
**engine.js-native** (SpendIQ is a classic script whose functions hang off `window`;
the original was MUI/React in a module tree that no longer exists). Restored from
`f989fab^:src/modules/HomeActions/Details/EmailDraft.jsx`, keeping: supplier context,
sender, subject, recipient, editable body, Edit/Save, Regenerate, Send. Fixing what was
cosmetic in the original:

| Original | Restored |
|---|---|
| Tone `<Select readOnly>`, never sent | Real control, bound to the style engine's intent/mode (`draft_rfq_emails.style_*`) |
| References select bound to the `tone` field, options "View Refrences"/"Informal" | Removed; replaced by the decision's cited evidence (the supplier's own quoted sentence) |
| Regenerate with no handler | Re-drafts via the drafting agent for the same thread |
| Recipient/sender placeholder defaults | Real values from the draft; blank when none is on file, never invented |

**New: the supplier's message is shown beside the draft** — the thing the original panel
never did. Reply on the left, draft response on the right.

**Send.** `POST /workflows/email` with `subject_override`/`body_override` (existing),
behind the explicit confirm convention already used by SpendIQ's email panel, then
records the human action against the decision (`actioned_by`, `override_reason` when the
human overrides the agent's view). Recipient must be typed when none is on file:
`bp_supplier` contact emails are empty and all 18 live drafts have no recipient — a
prefilled address would be fabricated.

### 4.7 Attachments

Supported at the SMTP layer, unreachable today: `POST /workflows/email` has no
attachments field and `draft_rfq_emails` has no attachments column.

- New `POST /workflows/email/{unique_id}/attachments` (multipart, `List[UploadFile]`),
  following the `documents.py` upload conventions.
- Bytes to S3 under a dedicated `email-attachments/` prefix — deliberately **not** the
  `data-integration/presigned-url` document path, which would ingest the file into the
  extraction pipeline and raise findings against it.
- Recorded in a new `attachments JSONB` column on `proc.draft_rfq_emails`:
  `[{filename, content_type, bytes, s3_key, added_by, added_at}]`.
- `EmailDispatchService.send_draft()` reads them back and passes `[(bytes, filename)]` to
  the MIME code that already works.
- Limits: per-file and per-message size caps read from settings, extension allowlist,
  rejection surfaced honestly in the panel (never a silent drop).
- `DELETE .../attachments/{index}` so a mistake can be removed before sending.

### 4.8 Inbound

"Check for new replies" in the Action Centre calls the existing `POST /emailwatcher`,
then runs `decide_email_reply` over replies with no decision yet. On-demand only: no
background polling, no timer, nothing touches the mailbox unasked.

## 5. Data changes

| Change | Type | Note |
|---|---|---|
| `proc.bp_policy` + 1 row `EmailReplyAutonomyPolicy` | data | migration-inserted, idempotent |
| `proc.draft_rfq_emails.attachments JSONB` | column | additive, nullable |
| `proc.bp_decision` | none | `subject_type='email_reply'` fits as-is |
| `proc.supplier_response` | none | already holds body, headers, match confidence |

No new tables. (If one were needed it would take the `bp_` prefix; none is.)

## 6. Endpoints

| Endpoint | Purpose | New? |
|---|---|---|
| `GET /decisions?subject_type=&status=` | the escalated queue for the Todo list | new |
| `POST /workflows/email/{unique_id}/attachments` | add attachments to a draft | new |
| `DELETE /workflows/email/{unique_id}/attachments/{index}` | remove one | new |
| `POST /decisions/email-reply/{response_id}` | decide a reply (or re-decide) | new |
| `POST /workflows/email` | send, with overrides | existing, unchanged |
| `POST /emailwatcher` | fetch inbound | existing, unchanged |
| `GET /decisions/{decision_id}` | full trace | existing, unchanged |

## 7. Error handling

- **No authority resolved** → escalate; rationale names the missing policy slug. Never
  send on an unknown limit.
- **Classification ungrounded or low-confidence** → escalate; the failed quote is
  recorded so the model's mistake is inspectable.
- **Dispatch failure** → the decision stays open, the failure is surfaced verbatim in the
  panel (existing convention: never a fabricated "Sent!"), draft remains unsent.
- **Attachment upload failure** → per-file error in the panel; the draft is still sendable
  without it, and nothing is half-recorded.
- **Watcher failure** → the queue still renders from stored decisions; the fetch error is
  shown, not swallowed.
- **Policy present but rules unparseable** → treated as missing (escalate), logged with
  the policy id.

## 8. Testing

- **Policy resolution:** unit tests over `resolve_authority` — policy present, absent,
  unparseable, multiple linked rows (selection must match `PolicyEngine`'s own choice),
  and concurrency (N agents resolve in parallel, results keyed correctly).
- **Decision:** table-driven tests over `decide_email_reply` — each escalate trigger in
  isolation (intent, value over limit, low confidence, ungrounded quote, thread cap,
  missing authority) and the one auto-send case, asserting `resolution`, rationale text,
  and that every fact carries a source.
- **Fail-closed proof:** with the policy row deleted, no send occurs and the decision is
  escalated with the reason.
- **Attachments:** round-trip test — upload, recorded on the draft, dispatch receives
  `(bytes, filename)`; oversize and disallowed extension rejected; delete removes it.
- **Panel:** the queue renders escalated decisions and nothing else; auto-resolved
  decisions never appear as Todo items.
- **Live demonstration (required, per project convention):** against `bp_sqldb` on the
  running local stack — inject a supplier reply, show it escalates with the policy cited,
  open the Todo card, edit the draft, attach a file, send, and show the threaded message
  plus the recorded human action. Not tests alone.

## 9. Scope guard

In: one policy row, one column, one parallel resolver, one engine method, one agent read,
one panel port into the existing detail pane, four new endpoints.

Out: a nav item, an inbox, a mailbox browser, background polling, autonomy learning
("earned trust" tiers), fixing governance for the other 12 agents, and any change to the
fail-open prompt envelope.
