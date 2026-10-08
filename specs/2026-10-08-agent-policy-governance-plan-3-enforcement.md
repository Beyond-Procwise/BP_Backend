# Agent Policy Governance — Stage 3 (Enforcement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Live agent policies are enforced on every agent tool call:
- **Not allowed:** the action is refused.
- **Needs approval:** the action pauses and becomes an approval case in the existing decision engine, with ordered levels, response times and timeouts. A timeout never approves. When it's approved, the system re-checks the policy and runs the stored action.
- **Notify only:** the action runs, and an in-product notification is written.

Every check is logged. The agent receives the documented result, reason code and message.

**Architecture:**
- **Live policies:** an in-process loader reads them the same way the orchestrator feed does (`repo.live_documents` plus `contract.validate`), with a short cache.
- **The check:** a pure `enforcement.check(context, policies)` decides the verdict. `tool_runtime.run_tools` and `run_tools_stream` call a gate before every `tool.handler(**args)`.
- **Approvals:** approval cases are `proc.bp_decision` rows (new `subject_type='agent_policy_approval'`, plus additive columns). This is no second decision service.
- **Timeouts:** a `BackendScheduler` job sweeps them.
- **Approved actions:** re-run through `build_tools(agent_nick, workflow_id, user_id)`, by name and arguments.
- **Deciders:** names are linked to people through an admin-editable map.
- **Screens and gateway:** the screens get an Approvals list and a Notifications list on the Agent policies tab, through the gateway.

**Spec:** `specs/2026-10-08-agent-policy-governance-brief.md`:
- §3.3 (outcomes, levels, response time, timeout);
- §3.7 (outputs to the agent, approver, people told and audit; approval details; masking; onMissingData);
- §4.3 (a Not allowed policy still blocks; the user's live-conflict fallback);
- §6.1 (the firing log fields learning needs);
- §7 (orchestrator refusals);
- §8 tests 6, 12, 14, 20.

The design note's rulings bind (§3.8: no session resume; a no-retry agent; the system runs the action on approval).

## Global Constraints

- Everything in stages 1 and 2 still binds.
- **One evaluator:** `conditions.to_engine` + `policy_condition.evaluate`. `MissingField` is resolved by the policy's `trigger.onMissingData`, where `fail_closed` means the condition counts as met.
- **Only valid live policies are enforced:** the same filter as the orchestrator feed (`contract.validate`, which refuses a live policy with no `checkedBy` and any unknown name).
- **Fail closed.** If live policies can't be loaded or evaluated, the tool call is refused with reason code `policy_check_unavailable`, and the agent is told so. If there are no live policies, nothing changes; behaviour is identical to today.
- **Precedence:**
  - Any matching `block` policy blocks immediately (user ruling).
  - Matching `approve` policies each need their own approval, and the action runs only when every one approves (user ruling: the fallback is that all of them must approve).
  - Matching `notify` policies always send their notifications, whatever else is decided.
  - There is no other precedence logic. Live-conflict cases are stage 4.
- **Timeouts:**
  - A timeout escalates to the next level.
  - At the last level, or when there is only one level, the case is rejected.
  - A timeout never approves.
  - Response time is the policy's `respondWithin`, otherwise the company default (`PT4H`, clock time).
- **Approvers:**
  - Only people linked to the current level's decider name may approve or reject.
  - Never the person whose request triggered the action, if known (self-approval bar).
  - A reject requires a reason, and the agent's record shows it.
  - Approve or reject only; no "approve with changes" (user ruling).
- **Masking:** sensitive inputs are masked in every response and log, except for an approver eligible at the case's current level.
- **Kill switch:** `AGENT_POLICY_ENFORCEMENT=off` disables the gate. Default is on. Any other value means on.
- **New tables:** use the `bp_` prefix, are applied to both databases, and are additive only. `bp_decision` gains nullable columns only; existing rows and behaviour are unchanged, and the existing decision tests must stay green.
- **Never call the real model in tests.** The live demonstration drives `run_tools` with a scripted chat stand-in, so enforcement is shown deterministically while the shared model is busy.

## Review Focus

1. **A model that loops on a refused tool.** The second identical call while paused also comes back `paused_for_approval` with the same request ID, and no second case is created. Test: `test_repeat_call_while_paused_reuses_the_case` (Task 6).
2. **Approving a case after the policy was retired or edited.** The re-check runs against the current live policies, and the replay is refused if a block now matches. Test: `test_replay_rechecks_current_policies` (Task 5).
3. **The sweeper running twice at once, or after a restart.** A level escalates exactly once, using a row lock. Test: `test_concurrent_sweeps_escalate_once` (Task 4).
4. **A decider name with nobody linked.** It blocks Active through readiness. If it is unlinked at run time because the map changed, the case is created as `unroutable` and an administrator task is raised. It never auto-approves. Test (Task 3 and Task 4).
5. **A sensitive input in the agent's result, the logs or a non-approver's view.** It is masked everywhere. Test (Task 2 and Task 7).

---

### Task 1: Migration

Files: `deploy/sql/2026-10-10_bp_agent_policy_enforcement.sql` (+ `_rollback.sql`) and `tests/migrations/test_2026_10_10_bp_agent_policy_enforcement.py` (live, both databases). The DDL:

```sql
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_policy_decider_map (
    decider_name  TEXT PRIMARY KEY,               -- exactly as written in policies, e.g. 'Finance Manager'
    groups        TEXT[] NOT NULL DEFAULT '{}',   -- Cognito groups whose members may act
    emails        TEXT[] NOT NULL DEFAULT '{}',   -- individual people (lower-case)
    notes         TEXT,
    last_modified_by TEXT NOT NULL,
    last_modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_policy_decider_map_someone CHECK (cardinality(groups) + cardinality(emails) > 0)
);

CREATE TABLE IF NOT EXISTS proc.bp_policy_firing (
    firing_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_key    TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    checkpoint    TEXT NOT NULL,
    action_name   TEXT NOT NULL,                  -- tool name
    agent         TEXT,
    workflow_id   TEXT,
    requested_by  TEXT,                           -- the user the agent served, if known
    outcome       TEXT NOT NULL CHECK (outcome IN ('approve','block','notify')),
    result        TEXT NOT NULL CHECK (result IN ('allowed','paused_for_approval','approved','rejected','blocked','timed_out','error')),
    matched_values JSONB NOT NULL DEFAULT '{}',   -- condition inputs, masked where sensitive
    missing_inputs TEXT[] NOT NULL DEFAULT '{}',
    decision_id   BIGINT,                         -- the approval case, for approve
    decided_level INTEGER,
    decided_by    TEXT,
    decided_at    TIMESTAMPTZ,
    reason        TEXT,
    duration_ms   INTEGER,
    reversal_of   BIGINT REFERENCES proc.bp_policy_firing (firing_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_firing_policy ON proc.bp_policy_firing (policy_key, created_at);
CREATE INDEX IF NOT EXISTS ix_bp_policy_firing_decision ON proc.bp_policy_firing (decision_id);
-- append-only: an UPDATE may only fill the decision columns of a row whose result is
-- 'paused_for_approval', and DELETE is refused. The trigger enforces both.

CREATE TABLE IF NOT EXISTS proc.bp_policy_notification (
    notification_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    firing_id     BIGINT NOT NULL REFERENCES proc.bp_policy_firing (firing_id),
    recipient     TEXT NOT NULL,                  -- decider/notify name as written
    message       TEXT NOT NULL,                  -- action, policy, outcome (no sensitive values)
    link          TEXT NOT NULL,                  -- 'agent-policy:<KEY>' or 'decision:<id>'
    read_by       TEXT[] NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_notification_recipient ON proc.bp_policy_notification (recipient, created_at);

ALTER TABLE proc.bp_decision
    ADD COLUMN IF NOT EXISTS options       JSONB,
    ADD COLUMN IF NOT EXISTS respond_by    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS on_timeout    TEXT,
    ADD COLUMN IF NOT EXISTS decision_scope TEXT,
    ADD COLUMN IF NOT EXISTS levels        JSONB,     -- [{"name":"Finance Manager","respondWithin":"PT4H"}, ...]
    ADD COLUMN IF NOT EXISTS current_level INTEGER;
CREATE INDEX IF NOT EXISTS ix_bp_decision_open_respond_by ON proc.bp_decision (respond_by) WHERE status = 'open';
COMMIT;
```

Tests:
- every table and column exists in both databases;
- the firing trigger refuses DELETE, and refuses UPDATE of anything but the decision columns on a paused row (prove it red);
- the decider-map CHECK holds;
- `bp_decision`'s existing columns are unchanged (copy the column list from the live DB first);
- `proc.bp_policy` is untouched.

Apply to both databases.

---

### Task 2: Enforcement core (pure) and the live-policy loader

Files: `src/services/agent_policy/enforcement.py`, `src/services/agent_policy/live_policies.py`, `tests/agent_policy/test_enforcement.py`.

**`live_policies.load(conn=None, *, ttl=30) -> list[dict]`**
- Returns the valid live documents (`repo.live_documents` filtered by `contract.validate` against `load_registry()`), cached for `ttl` seconds per process.
- `invalidate()` clears the cache. Saves and retires call it, through the repo or the router.
- If loading raises, it raises `PolicyStoreUnavailable`.

**`enforcement.check(ctx: dict, policies: list[dict]) -> Verdict`.** Pure: no DB, no clock beyond the passed `now`.
- `ctx` contains `checkpoint`, `tool.name`, `agent.name`, `agent.reason`, and `args`, the dict of call arguments.
- Considered policies: `context.checkpoint == ctx.checkpoint`, plus `limit` (stage 1 stores free text that isn't machine-enforced; ignore it and record that).
- For each considered policy, evaluate `trigger.condition` on `nest`-ed ctx:
  - `MissingField` → `onMissingData == 'fail_closed'` means matched, and the missing field is recorded.
  - `ConditionError` → matched as **block**, with reason code `<id>.condition_unreadable` (fail closed).
- The `Verdict` holds:
  - `blocks`: matched block policies;
  - `approvals`: matched approve policies;
  - `notifies`: matched notify policies, plus blocks that carry a notify list;
  - `result`:
    - `blocked` if there are any blocks;
    - `paused_for_approval` if there are any approvals;
    - `allowed` otherwise;
  - `to_agent`, the dict below;
  - per policy, `matched_values` (the condition fields' values, masked where the input is sensitive) and `missing`.
- **`to_agent`:**
  - **Blocked:** `{"result":"blocked","reasonCode": first block's outputs.toAgent.reasonCode, "reason": its reason, "messageForPerson": ..., "policies":[ids]}`.
  - **Paused:** `{"result":"paused_for_approval","requestIds":[...filled by the gate...],"respondWithin":..., "whilePaused":"no_retry", "reasonCode", "reason", "messageForPerson"}`.
  - **Allowed:** None. The tool runs normally and its real result is returned.
- `mask(values, policy)`: replaces the value of every input with `sensitive: true` by `"•••"`.

Tests:
- block wins over approve;
- notify still listed when blocked;
- two approve policies give two approvals;
- a missing field with fail_closed matches and is recorded;
- an unreadable condition blocks;
- a different checkpoint is ignored;
- sensitive values are masked in `matched_values`;
- no live policies gives allowed with `to_agent` None;
- `PolicyStoreUnavailable` from the loader is surfaced by the gate (Task 6), not swallowed here.

---

### Task 3: Decider map and readiness

Files: `src/services/agent_policy/deciders.py`, an extension to `readiness.activation_problems`, and tests.

- `load_map(conn) -> dict[name, {"groups":[], "emails":[]}]`.
- `eligible(principal, decider_name, mapping) -> bool`:
  - the principal's `cognito:groups` intersect the entry's groups, **or** the principal's email (lower-case) is in its emails;
  - Admin is NOT automatically eligible (separation of duties).
- `unmapped(names, mapping) -> list[str]`.
- **Readiness (stage 1 extension):**
  - For `approve`, every name in `deciders` must be mapped. For `notify` (and block with notify), every name in `notify` must be mapped.
  - Otherwise add the problem `{"field":"deciders"|"notify","code":"decider_unmapped","names":[...],"routeTo":"administrator","message":"Nobody is linked to <names> yet. An administrator must link them before this policy can be Active."}`.
  - The UI renders it through `problemMessage` (add the code).
  - `activation_problems` receives the mapping as a new optional argument. `None` means "load from the DB", and existing callers must keep working (tests pass a map).

Tests: the eligibility table (group match, email match, Admin not eligible, no match); readiness refuses an unmapped decider and accepts a mapped one; existing readiness tests stay green.

---

### Task 4: Approval cases, decisions and the timeout sweeper

Files: `src/services/agent_policy/approvals.py`, a registration in `src/services/backend_scheduler.py` (`_register_agent_policy_approvals_job`, interval 60 s, env toggle `AGENT_POLICY_APPROVAL_SWEEP=on|off`, default on), and tests (live, bp_testdb).

**`open_case(conn, *, policy_doc, firing_id, action: {tool, args, agent, workflowId, userId, reason}, requested_by, now) -> int`**
- Inserts a `bp_decision` row:
  - `subject_type='agent_policy_approval'`, `subject_id=<policyKey>:<firing_id>`, `decision='approve_or_reject'`, `resolution='escalated'`, `status='open'`;
  - `policy_name=<key>`, `policy_id=NULL` (the key is text; the existing column is the bp_policy id);
  - `facts={action (args masked for storage? NO: store the full args, because the replay needs them; masking applies when displaying), policy: {id, version, situation, excerpt, reference, document}, approvalInputs: outputs.toApprover.show, requestedBy}`;
  - `levels=[{name, respondWithin}]`, built from `enforcement.intervention.escalateTo` and `sla.respondWithin`;
  - `current_level=0`, `respond_by=now+respondWithin`, `on_timeout` (`escalate_next` or `reject`), `options=["approve","reject"]`;
  - `created_by=requested_by or "agent:<agent>"`.
- If any level name is unmapped at this moment, also set `facts.unroutable=[names]`. The case still opens, and the screen shows "Nobody can act on this until an administrator links <name>". It never auto-approves.
- Returns `decision_id`.

**`act(conn, decision_id, *, principal, verb, reason, now) -> dict`**
- `verb` is `approve` or `reject`. `reject` without a non-blank reason raises a 422-style `ApprovalRefused`.
- Lock the row (`FOR UPDATE`, explicit transaction; `get_conn` is autocommit).
- Refuse when: the case is not open; the principal is not eligible at `current_level` (`deciders.eligible`); or the principal's subject or email equals `facts.requestedBy`.
- Record through the decision engine's existing human-action pattern: a new `bp_decision` row (`status` `actioned`, `actioned_by`, `actioned_at`, `override_reason`=reason), and close the original (`status` `actioned`). Mirror `_close_original_email_decision`; read `decision_engine._record_human_action` and reuse it if its signature allows.
- Update the firing row's decision columns: `decided_level`, `decided_by`, `decided_at`, `reason`, `result` approved or rejected.
- On approve, hand off to `replay.run(decision_id)` (Task 5). The replay runs **after** commit, never inside the lock.

**`sweep(conn, now) -> dict`**
- For each open `agent_policy_approval` row with `respond_by <= now`, lock it (`FOR UPDATE SKIP LOCKED`):
  - if `current_level < len(levels)-1`: `current_level += 1`, `respond_by = now + levels[current_level].respondWithin`, and write a notification to the new level ("<action> needs your decision; the previous approver did not answer in time");
  - else: close as rejected (`actioned_by='system:timeout'`, reason "No decision in time; a timeout never approves"), set the firing result to `timed_out`, and notify the first level and the requester.
- Returns the counts.

**Required tests:**
- `test_concurrent_sweeps_escalate_once` (two threads);
- levels keep their order;
- the last level rejects and never approves;
- `act` refuses ineligible, self-approval, reject-without-reason, and already-closed;
- `act` approve records the actor, level and time;
- the existing `tests/engines` decision tests stay green.

---

### Task 5: Replay on approval

Files: `src/services/agent_policy/replay.py` and tests.

**`run(decision_id, *, agent_nick=None)`**
1. Load the case. If every approval case of the same `firing_group` is approved, carry on. Otherwise stop: the action waits for all of them (user fallback rule).
   - `firing_group` is stored in `facts` when the gate opens several cases for one tool call.
2. **Re-check** with `enforcement.check` against the **current** live policies, using the stored ctx:
   - if a block now matches → do not run; firing result `rejected`, reason "A policy now forbids this action";
   - if a new approve policy matches that this group did not cover → open a new case for it and stop.
3. Run the tool through `agentnick_control.build_tools(agent_nick, workflow_id=..., user_id=...)` and the handler named `tool`, with the stored args. `agent_nick` comes from `app.state.agent_nick`, or `BackendScheduler`'s.
4. Store the outcome (`ok`, result summary or error) on a new `note` row in `bp_decision`, linked by `subject_id`, and on the firing row. The original chat session is not resumed (design §3.8).

**Required tests** (with a stub `agent_nick` whose tool records calls):
- `test_replay_rechecks_current_policies` (a block policy activated after the case opened → no run);
- two-case group → the tool runs only after both are approved;
- a replay error is recorded and never raises.

---

### Task 6: The gate in the tool loop

Files: `src/services/tool_runtime.py` (both loops), `src/orchestration/agentnick_control.py` (pass the calling agent), `src/services/agent_policy/gate.py`, and tests.

**`gate.before_tool(*, tool_name, args, agent, reason, workflow_id, user_id) -> GateResult`**
- `GateResult` is `{allow: bool, to_agent: dict|None, firing_ids: [...], case_ids: [...]}`.
- **Kill switch:** `AGENT_POLICY_ENFORCEMENT=off` returns allow.
- Loads live policies. `PolicyStoreUnavailable` (or any error) refuses with `to_agent={"result":"blocked","reasonCode":"policy_check_unavailable","reason":"Policy checks are unavailable, so this action was not run."}`, written to the firing log best-effort (`policy_key='*'`).
- Runs `enforcement.check`. Writes one firing row per matched policy (outcome, result, masked values, missing inputs).
- **Notifications:** for each notify recipient, writes `bp_policy_notification` rows. These are written whatever the result.
- **Approvals:** opens one case per matched approve policy, all sharing a `firing_group` id. Before opening, it checks for an open case for the same (policy, tool, args digest, workflow_id). If one exists, it reuses it and opens no new one (`test_repeat_call_while_paused_reuses_the_case`).
- Fills `requestIds`. Returns.
- **Duration:** records `duration_ms` for the check.

**`tool_runtime`:** in both loops, immediately before `tool.handler(**args)`:
- call the gate;
- if it does not allow, do **not** run the handler;
- record a `ToolCall` with `ok=False`, `result=to_agent`, `error=None`;
- send the model `{"role":"tool","name":name,"content": json(to_agent)}`.

**Reason and agent:**
- `run_tools(...)` gains an optional `agent: str | None = None`. The callers pass it: `agentnick_control.reason` passes the agent slug from `BaseAgent.reason`; `/agents/reason` passes `"agent_nick"`.
- `agent.reason` is the assistant message text that accompanies the tool call in that round (`message.get("content")`, stripped and capped at 500 characters), or absent.
- Support chat (`describe_platform` only) and evals pass nothing, so they are still gated with `agent=None`.

**Tests** (scripted chat stand-in, no model):
- a blocked tool is never executed and the model sees the documented block result;
- a paused tool is not executed, the result has `requestIds`, and a repeat call reuses the case;
- a notify-only policy runs the tool and writes notifications;
- no live policies → identical behaviour to today (assert the messages list equals a run with the gate patched out);
- store unavailable → refused;
- the kill switch works;
- both `run_tools` and `run_tools_stream` are gated.

---

### Task 7: Endpoints and gateway routes

**Backend** (`agent_policies.py`; same `gateway_principal`; writes audited first):

| Method + path | Role | Returns |
|---|---|---|
| GET `/agent-policies/approvals?status=open` | Viewer | the cases where the caller is eligible at the current level, plus all cases for Admins (read-only for Admins unless they are eligible): `{id, policyKey, actionPlain, inputs (masked unless the caller is eligible), agentReason, situation, excerpt, reference, document, level, levelName, respondBy, onTimeout, unroutable, requestedBy}` |
| GET `/agent-policies/approvals/{id}` | Viewer | one case, same masking, plus its history (decisions, notes, firing rows) |
| POST `/agent-policies/approvals/{id}/decide` | Viewer (eligibility decides) | `{verb, reason}` → `act`; 403 if not eligible; 422 for a reject with no reason |
| GET `/agent-policies/notifications?mine=1` | Viewer | notifications whose recipient name is mapped to the caller (the decider map), newest first, with read state |
| POST `/agent-policies/notifications/{id}/read` | Viewer | marks it read for the caller |
| GET `/agent-policies/deciders` | Viewer | the map |
| PUT `/agent-policies/deciders/{name}` | Admin | `{groups, emails, notes}` |
| GET `/agent-policies/{key}/firings?limit=50` | Viewer | the policy's recent firing rows (masked) |

- Add these GET paths to the user-approved OutputSafety screen-read exemption list? **No.** That would widen the user's ruling. Instead, check the responses through `scrub_payload` in a test, and **if** anything is withheld, stop and report it for a user ruling.
- **Gateway:** the same routes with the same `need()` roles, an id allow-list (`^[0-9]{1,18}$`), and a decider name allow-list (`^[A-Za-z][A-Za-z0-9 &,'.-]{0,63}$`).

---

### Task 8: Screens

New `ap*` functions, plus pure modules with vitest tests.
- **Approvals** sub-tab on Agent policies (a badge shows the open count for the caller):
  - **Each card shows:** the action in one sentence; the approval-detail inputs (masked values show "Hidden"); the agent's stated reason ("No reason given" when absent); the situation; the source excerpt word for word, with section and document; the level ("Level 1 of 2: Finance Manager"); the deadline and what happens on timeout; and "Nobody can act yet: link <name>" when unroutable.
  - **Actions:** Approve, and Reject, which needs a reason before the button enables. Both use the existing two-step confirm pattern for Approve.
  - **After a decision:** the card shows the outcome, and later the replay outcome note.
- **Notifications** sub-tab with unread markers; opening one marks it read and follows its link.
- **Deciders** admin editor (Admin only): name, groups, emails, notes; validation mirrors the gateway.
- **Policy form:**
  - The health line for Active policies: "Checked N times in the last 30 days: A allowed, P paused, B blocked" from `/firings`.
  - The `decider_unmapped` problem message.
- Behaviour tests in the no-DOM harness:
  - Reject is disabled without a reason;
  - masked inputs never render their value;
  - Approve goes through the two-step confirm;
  - polling, if any, stops on close.

---

### Task 9: Live demonstration and verification

On the local stack: backend :8010 from the worktree, gateway :3011 with `AUTH_BYPASS_GROUPS=PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER`, UI :3010, bp_testdb. Then:
1. **Mapping:** link "Finance Manager" → group `PROCWISE_FINANCE_REVIEWER_APPROVER`, and "Head of Ops" → a test email.
2. **Policies:** create and activate three policies on a real registered tool (e.g. `run_supplier_ranking`, with `args.deal_id` conditions): one block, one approve with two levels and a short response time (`PT2M`), and one notify.
3. **Tool loop:** drive `run_tools` with a scripted chat stand-in that calls the tool three times with arguments hitting each policy.
4. **Record:**
   - what the "model" received each time;
   - the tool never ran for block or approve;
   - the notify notification;
   - the firing rows.
5. **Approvals:** approve the case through the gateway and screen as an eligible user. The replay runs the tool; record the result note.
6. **Timeout:** for a second approval, let level 1 time out. Run the sweeper manually and see it escalate, then time out again to reject. Record it.
7. **Fail closed:** stop the DB connection by pointing the loader at a bad DSN in the test process only, and see `policy_check_unavailable`.
8. **Screens:** headless Chrome: the Approvals, Notifications and Deciders tabs, with zero console errors.

Write `specs/2026-10-08-agent-policy-governance-stage3-verification.md` with every request and response (keys redacted), and the diff summary per repo. Stop only your own processes; delete your test policies' S3 objects if you created any.
