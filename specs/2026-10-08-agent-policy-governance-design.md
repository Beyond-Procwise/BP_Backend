# Agent policy governance — design

**Date:** 2026-10-08
**Brief:** `specs/2026-10-08-agent-policy-governance-brief.md` (verbatim; the requirements live there).
This note records what the code actually contains, the rulings given on 2026-10-08, and
the decisions taken while planning. Where this note and the brief differ, this note wins,
because it carries the rulings.

## 1. What exists today (read 2026-10-08)

| Piece the brief assumes | Reality |
|---|---|
| Policy upload + extraction | Does not exist in any repo or branch. |
| Policy list grouped by source | Flat table, `policiesView` (`engine.js:22587`), no longer in the nav rail. |
| Policy edit form | `policyEdit` (`engine.js:22603`) on `openFormModal`. 10 fields; only name, type, desc, status are saved. |
| Inventory, CSV export, overlap check, JSON export | Do not exist. |
| Status / versioning | `proc.bp_policy.policy_status` 1/0. Gateway edit retires the old row and inserts a new one (`policy.service.ts:157`). No Draft, no Retired, no two-step confirm. |
| Orchestrator enforcement | Agent tool loop (`src/services/tool_runtime.py`) calls tools with no gate. `guardrail.authorize` is called at ~7 fixed doors only. Precedence there is "deny beats allow"; no block>approve>notify rule exists. |
| Orchestrator registry | Does not exist. Tools come from `auto_registry.tool_schemas()`; action names from `services/actions.py`. |
| Decision engine | `src/engines/decision_engine.py`, `proc.bp_decision`. Synchronous, polled, no levels, no SLA, no timeouts, no callback, no "approve with changes". |
| Company settings | `proc.bp_admin_config` (key/JSONB). |
| Business-area taxonomy | Does not exist. |
| Mock `hard-policy-form-simple.html` | Not on this machine. Build follows the brief's described behaviour. |

## 2. Rulings (user, 2026-10-08)

- **A.** Edit `engine.js`, but keep the changes in their own functions or modules. Make no unrelated refactors, and give a diff summary before merging.
- **B.** Use new tables and leave `proc.bp_policy` untouched. The tables are: agent policy (stable ID, current version), policy version (unchangeable once saved), source document and its versions, extraction run, conflict case, firing and decision log, and learning suggestion. List any `bp_policy` rows that look like agent policies (§6), but don't migrate them.
- **C.** Put the logic in the Python backend, but don't bypass the gateway's sign-in, permission or audit checks. Route through the gateway unless we can show the screens already call the backend directly with the same checks. Give the orchestrator one stable, versioned endpoint for reading live policies.
- **D.** Extend the decision engine. Existing behaviour must not change, and existing tests must still pass.
- **Revised documents:**
  - an unchanged clause gets no new version;
  - a changed clause becomes a new draft, and the live version stays live until that draft is activated;
  - a new clause becomes a new draft;
  - a removed clause becomes "Proposed retire" and is never retired automatically;
  - the screen shows the document before and after.
- **Second reviewer:** a setting per business area, on by default for Finance only. The second reviewer must be a different person from the owner.
- **Not allowed in a conflict:** still blocks immediately.
- **Live conflict approver:** the engine's routing. The fallback is the last escalation level of each matching approve policy, and all of them must approve. If no approve policy matches, no approver is needed.
- **Lookups and running totals:** owned by the orchestrator team. This version uses inputs from the action only, and anything else shows "Can't be enforced yet".
- **Approve with changes:** not in this version. Approve or reject only. Reject requires a reason, and the agent sees it.
- **Defaults:** the response time is set below (§3.1). Learning needs 30 decisions, 30 days, 3 approvers, a Wilson lower bound of 0.85, none edited or reversed, and a median decision time of at least 30 s. Five same-way live decisions raise a policy case. All of these are company settings.

## 3. Decisions taken in planning

1. **Response time is 4 hours of clock time (`PT4H`).** Business hours would need a working calendar and a time zone for every approver, and none exists. Clock time is unambiguous. It also fails safe, because a timeout rejects and never approves.
2. **The gateway route goes through the gateway.** The backend's own sign-in check (`ASK_AUTH_MODE`) is `off` in `.env`, and whether it is on in production is an open question (see memory: COGNITO_USER_POOL_ID). So we cannot *show* that a direct call carries the same checks. Every new screen endpoint therefore goes UI → gateway → backend:
   - **Gateway:** checks sign-in with `CognitoGuard` and permission with `productRole`, then forwards the call with a shared secret, `X-Gateway-Key`, and the verified user's subject, email and groups.
   - **Backend:** refuses any call without a valid key. With the key set, it fails closed: when it is unset, it returns 503. It re-checks the role from the forwarded groups, and audits every write through `agent_actions.record_action_or_fail`, so a write whose audit cannot be stored does not happen.
   - **The orchestrator's read endpoint:** `GET /orchestrator/agent-policies/v2/live` requires the same service key and is versioned by path.
3. **Permissions do not add rows to `bp_policy`.** That table stays untouched (ruling B). Role checks use the existing role ranks in `RoleDefinitionPolicy` (read-only):
   - Viewer: read.
   - Buyer: create drafts and edit.
   - Approver: activate, retire, and act as second reviewer.
   - Admin: technical view, taxonomy, registry, plus everything above.
4. **The Policies screen.** It now shows **Agent policies**, the new feature. The existing `bp_policy` system settings stay reachable unchanged under a second tab, **System settings**, which uses the old form as it is. The canvas library keeps listing `bp_policy`.
5. **Operator names in conditions.** Stored conditions use the brief's operator names (`gt gte lt lte eq ne in not_in exists`). One function translates them to `services/policy_condition.py`'s grammar, so there is one evaluator and its three-valued "cannot tell" logic is reused.
6. **"One agent call" means one extraction run** of one agent. A large upload is read in parts inside that run, because the local model's context is finite. Test 1 asserts one run, not one model call.
7. **Delivering "who is told".** `notify` uses the existing governed email path. It is never a new sender.
8. **Pausing an agent action.** It is not resumed in the agent's session. The agent receives `paused_for_approval` and does not retry. On approval, the stored action is checked against the policy again and executed by the system.

## 4. Stages (one plan each, each shown working on the local server before the next)

| Stage | Plan | Delivers |
|---|---|---|
| 1 | `specs/2026-10-08-agent-policy-governance-plan-1-foundation.md` | Tables, taxonomy, registry, settings, example check, JSON compiler + schema, Active checks, extraction confidence, versioning, backend + gateway endpoints, the form, list, inventory, CSV export, orchestrator read endpoint. |
| 2 | written when 1 ships | Extraction agent: upload many documents, one run, JSON Lines streaming, "Not enforceable", stable IDs on re-extraction, revised-document before/after, "Ask the agent to fix it". |
| 3 | written when 2 ships | Enforcement: gate in the tool loop, results to the agent, approvals through the extended decision engine (levels, timeout sweeper), notify, firing + audit log, orchestrator refusals. |
| 4 | written when 3 ships | Conflicts: design-time detection, `policy-conflict/1` cases, applying decisions, live multi-match pause, fail-safe, repeat-5 feedback. |
| 5 | written when 4 ships | Learning: patterns, Wilson bound, guardrails, suggestion card and queue, second reviewer. |

## 5. `policy-conflict/1` ↔ decision engine field mapping (built in stage 4)

**Inbound** (case → `proc.bp_decision`):

| `policy-conflict/1` field | `bp_decision` column |
|---|---|
| `caseId` | `decision_id`, displayed as `pc_<id>` |
| `kind` | `subject_type` (`policy_conflict` or `live_conflict`) |
| `action`, `policies`, `standingRules`, `priorDecisions` | `facts` |
| `overlap` | `evidence` |
| `options`, `respondWithin`, `onTimeout` | new columns `options`, `respond_by`, `on_timeout` |

**Returned decision:**

| Decision field | Where it comes from |
|---|---|
| `decision` | the action's value |
| `scope` | new column `decision_scope` |
| `decidedBy` | `actioned_by` |
| `decidedAt` | `actioned_at` |
| `reason` | `override_reason` |

## 6. `bp_policy` rows that look like agent policies (listed, NOT migrated)

These describe "a situation an agent meets, and what must happen". The same ids appear in both databases unless noted.

| Row | Reads as |
|---|---|
| `ApprovalThresholdPolicy` (testdb 10 / sqldb 10) | approve: spend above £10,000 escalates |
| `EmailDispatchApprovalPolicy` (673 / 102) | approve: outbound mail needs a recorded approval |
| `EmailRecipientAllowlistPolicy` (674 / 103) | block: recipient not on the supplier master |
| `EmailSensitivityPolicy` (675 / 104) | block: content class above recipient clearance |
| `EmailVolumePolicy` (676 / 105) | approve: over 20 per run / 50 per user per day pauses for review |
| `EmailReplyAutonomyPolicy` (473 / 38) | approve: listed reply intents escalate to a person |
| `NegotiationBoundsPolicy` (903 / 139) | block: bounds an agent may put to a supplier |
| `ReportSignoffPolicy` (1298 / 298) | approve: a report needs sign-off before it leaves the company |

Every other active row is a role permission (`*AuthorityPolicy`, `RoleDefinitionPolicy`, `RoleAssignmentPolicy`) or a tuning setting (`limit`, `supplier_ranking`, `critique`, `email_family`, `email_learning`, `requirements`, `negotiation`, `governance`, `ShadowModePolicy`).
