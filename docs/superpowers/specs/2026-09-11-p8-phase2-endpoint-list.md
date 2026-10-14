# P8 phase 2 — the 49 write/send endpoints that still do not resolve a caller

Produced 2026-09-11, before any code change. Re-derived by running the committed
scan (`python3 scripts/p8_endpoint_scan.py`) on `Development` at `615da42`, not
copied from the phase-1 prose.

| | count |
|---|---|
| write/send endpoints (POST/PUT/PATCH/DELETE) in `src/api/routers` | 87 |
| …taking `Depends(require_user)` | 38 |
| …**not** taking it | **49** |

The 57 → 49 drop is phase 1's eight, nothing else. The GA-3 gate commits
(69f693e, 789cc56) were already counted in the 30 phase 1 started from.

Every one of the 49 is still *authenticated*: `api/main.py` mounts every
router behind `require_user`. What they don't do is **resolve** the caller into
the handler.

## Phase 1's list was wrong about one thing

Phase 1 said none of the remaining 49 "accepts a caller-supplied identity
today". Reading the handlers and what they write shows **eleven** that do.
Each takes a name from the request and writes it to an actor column, or
carries it into an agent payload:

| Router | Endpoint | Caller-typed field | Where it lands |
|---|---|---|---|
| `agent_workflows.py` | `POST /agent-workflows/runs/{run_id}/input` | `body.answered_by` (default `"human"`) | `proc.bp_workflow_input_request.answered_by`, the HITL audit trail |
| `deal_proposals.py` | `POST /deals/proposals/{id}/confirm` | `body.confirmed_by` (required) | `proc.bp_deal_proposal.confirmed_by` **and** `proc.bp_agent_actions.agent` |
| `deal_proposals.py` | `POST /deals/proposals/{id}/reject` | `body.rejected_by` (required) | `proc.bp_deal_proposal.confirmed_by` |
| `extraction_feedback.py` | `POST /extraction/proposals/{id}/approve` | `body.approver` (default `"api"`) | `proc.bp_prompt.created_by` / `last_modified_by` (the governance prompt table, which overrides code), `proc.bp_extraction_hint_proposal.reviewed_by` |
| `extraction_feedback.py` | `POST /extraction/proposals/{id}/reject` | `body.approver` (default `"api"`) | `proc.bp_extraction_hint_proposal.reviewed_by` |
| `requirements.py` | `POST /requirements/message` | `body.created_by`, else hardcoded `"api"` | `proc.bp_requirement.created_by`, and `specifications.accepted_by` |
| `requirements.py` | `POST /requirements/run-workflow` | `body.created_by` | same, via the `requirements_to_ranking` payload |
| `workflows.py` | `POST /workflows/negotiate` | `req.user_id` | the agent payload (echoed into `proc.routing.process_details`; a negotiation batch entry reads it as its sub-run `user_id`) |
| `workflows.py` | `POST /workflows/approvals` | `req.user_id` | the agent payload |
| `workflows.py` | `POST /workflows/supplier-interaction` | `req.user_id` | the agent payload |
| `workflows.py` | `POST /workflows/discrepancy` | `req.user_id` | the agent payload |

Phase 1's "identity-taking" scan matched on field names it had already seen
(`reviewer`, `created_by`, `user_id`). `answered_by`, `confirmed_by`,
`rejected_by` and `approver` slipped past it.

## Seven more write an actor column from a hardcoded string

| Router | Endpoint | Column | Written today |
|---|---|---|---|
| `agent_groups.py` | `POST /agent-groups` | `proc.bp_agent_group.created_by` | `"system"` (repo default) |
| `negotiate.py` | `POST /deals/{id}/advice/message` | `proc.bp_negotiation_advice_fact.stated_by`, `proc.bp_negotiation_advice.created_by` | `"buyer"` / NULL |
| `negotiate.py` | `DELETE /deals/{id}/advice/fact/{key}` | `proc.bp_negotiation_advice.created_by` (when the turn seeds advice) | NULL |
| `agents.py` | `POST /agents/execute` | `proc.routing.created_by`, `user_id` | `"AgentNick"` (`settings.script_user`) / NULL |
| `workflows.py` | `POST /workflows/opportunities` | `proc.routing.created_by`, `user_id` | `"AgentNick"` / NULL |
| `workflows.py` | `POST /workflows/extract` | `proc.routing.created_by`, `user_id` | `"AgentNick"` / NULL |
| `run.py` | `POST /run` | `proc.routing.modified_by` | `"AgentNick"` |

**Eighteen of the 49 write an actor.** The other 31 write no actor column at
all.

## The 31 that write no actor

Phase 2 still resolves the principal on these, so identity reaches the handler.
It does not invent new attribution storage for them.

- `agent_groups.py`: `PUT /{group_id}`, `DELETE /{group_id}`. `bp_agent_group` has no `updated_by`.
- `agent_workflows.py`: none beyond the one above.
- `agents.py`: `POST /reason` (no write), `POST /process-document` (the orchestrator logs under its own context).
- `benchmark.py`: `POST /preview`, a pure calculation.
- `deal_proposals.py`: `POST /proposals/generate`, `PATCH /proposals/{id}/members`.
- `deal_summary.py`: `POST /analysis-summary/sync`, `POST /{deal_id}/promote`, `POST /{deal_id}/reconcile`, `POST /{deal_id}/save-reference`. `bp_deal` has no actor column.
- `email.py`: `POST /emailwatcher`.
- `extraction_feedback.py`: `POST /proposals/run`. Its audit row's `agent` is `"extraction_feedback"`, which names an agent, not a person.
- `governance.py`: `POST /govern`. Its audit row's `agent` is the agent the task is governed *as* (`body.agent`, else `"agentnick"`). That is an agent slug the reasoning is scoped by, not a human actor, so it is left as it is.
- `negotiate.py`: none beyond the two above.
- `opportunities.py`: `POST /link-deals`, `POST /sync`, `POST /{id}/stage`. `set_stage` writes no actor.
- `promotion.py`: `POST /canonicalize-po`.
- `requirements.py`: none beyond the two above.
- `stream.py`: `POST /plan`, no write.
- `summary.py`: `POST /summary`, `POST /summary/precompute`. `bp_summary` has no actor column.
- `supplier_review.py`: `POST /reviews/sweep`.
- `support.py`: `POST /contact`, `POST /contact/stream`, `POST /{reference}/confirm`. See "Not changed" below.
- `vendors.py`: `POST /onboard/upload`, `POST /onboard/{sid}/correct`, `POST /onboard/{sid}/save`. Template corrections carry no actor.
- `workflows.py`: `POST /rank`, `POST /quotes/evaluate`, `DELETE /email/{uid}/attachments/{index}`.

## The rule applied to the eighteen

The actor is `principal.subject`, and nothing else is. With no principal
(`ASK_AUTH_MODE=off`, as in this sandbox) the record names nobody. There is no
fallback to the typed value. A typed value survives only as an explicitly named
`*_label` where the record already has room for one. None of these eighteen
has such room, so the typed values are dropped. The request fields stay on the
models because clients send them.

Two columns cannot hold NULL, and those are handled explicitly:

- `proc.bp_prompt.created_by` / `last_modified_by` are `NOT NULL DEFAULT
  'system'`. With no principal, the approval writes the column's own default
  (`'system'`) there, and NULL to the nullable `reviewed_by`.
- `proc.routing.created_by` / `modified_by` are filled by
  `ProcessRoutingService` as `created_by or settings.script_user`. The router
  now passes the subject. With no principal the service still writes its
  system user, which is the service's own identity, not a caller-typed one.

## Not changed, deliberately

- **The orchestrator's `AgentContext.user_id`.** `execute_workflow(...)` falls
  back to `settings.script_user` (`"AgentNick"`) for every API-started
  workflow. That value reaches `proc.workflow_execution.user_id`, which
  `approval_store.workflow_initiator` reads for the self-approval bar. Feeding
  the principal into it would change what a policy decides, and this phase
  does not change gate or policy behaviour. The same applies to the
  `"human"` user passed when `/agent-workflows/runs/{id}/input` resumes a run.
- **`support.py`'s `user_name` / `user_email`.** These are the reply-to
  details on a support ticket, not attribution. Replacing them under the
  no-fallback rule would leave every escalation with nobody to reply to while
  auth is off. This needs its own decision.

## Already resolving a principal, but still reading a typed identity

These are outside the 49, so they are not changed here. They were found while
deriving the list:

- `supplier_research.py` `POST /suppliers/enrichment/{id}/apply`:
  `getattr(principal, "subject", None) or body.reviewer`. That is the
  conditional forgery phase 1 removed from promotion.
- `agent_workflows.py` `POST /agent-workflows/{id}/run`: passes
  `body.user_id` (default `"system"`) as the run's user. That becomes
  `proc.workflow_execution.user_id`, which is exactly what the self-approval
  bar reads.
