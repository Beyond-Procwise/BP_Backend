# Agent policy governance, stage 3 (enforcement): verification and diff summary

Date: 2026-10-08, 23:29 to 23:36 UTC. Branch `agent-policy-stage3` in three worktrees. Database: `bp_testdb`
(checked with `select current_database()` before every write step). Nothing has been merged or pushed.

## In plain English

- **The rules now bite.** Three demo rules were switched on for one real agent tool, `get_policy`.
  - The "block" rule stopped the tool, and the agent was told why.
  - The "approve" rule paused the tool and opened a request for a person to decide.
  - The "notify" rule let the tool run and told the right person.
- **Asking twice does not open a second request.** The same paused call, made again, reused the open request.
- **Approving runs the action once.** An eligible person approved the request through the gateway. The
  backend then ran the stored tool call once, and wrote down what it returned.
- **Nobody answering is never a yes.** A second request was left unanswered. The timer passed it to the next
  person (the "Demo CFO"). When that time ran out as well, the request was rejected as "system:timeout".
  The action never ran.
- **When rules cannot be checked, nothing runs.** With the rule store made unreachable inside the test
  process, the tool was refused with `policy_check_unavailable`. With the off switch set, the tool ran.
- **The screens work.** Approvals (badge, card, approve with the two-step confirm), Notifications
  (marks read, then opens), Deciders, and the health line on a policy were all used in headless Chrome.
  There were no console errors and no uncaught exceptions.
- **Clean-up.** All three demo rules are retired, so no demo rule is live in `bp_testdb`.
- **Three small faults were found.** None blocks the feature. They are listed under "Defects found" below.

## How the stack was run

| Part | Port | How |
|---|---|---|
| Backend | 127.0.0.1:8010 | `.venv/bin/uvicorn api.main:app --workers 1` from the BP worktree, `PYTHONPATH=<worktree>:<worktree>/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`, `AGENT_POLICY_ENFORCEMENT=on`, `AGENT_POLICY_APPROVAL_SWEEP=off`. To make sure the real model could not be called, `OLLAMA_BASE_URL=http://127.0.0.1:9` (a dead port) and `OLLAMA_CLOUD_*` were unset. The only model traffic in its log is the start-up preload, which got "connection refused" from port 9. Nothing reached :11434. |
| Gateway | :3011 | `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development AUTH_BYPASS_GROUPS=PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER BP_BACKEND_URL=http://127.0.0.1:8010`, gateway key read from the BP `.env` (never printed). Bypass sub `c2b5b404-40c1-7047-71c4-dd8093ecf25d`. |
| UI | 127.0.0.1:3010 | Vite from the UI worktree, `VITE_API_URL=http://127.0.0.1:3011 VITE_DEV_AUTH_BYPASS=true`. The untracked `mockPipelineDeals.js` was copied in for the run (stage 1 finding 6) and deleted afterwards. |
| Headless Chrome | :9223 | `google-chrome --headless=new` (Chrome 150), driven over CDP with `/usr/share/nodejs/ws`, 1440×900. |
| Driver | none | Small Python scripts in the scratchpad (not committed) that import the worktree's own modules (`services.tool_runtime`, `services.agent_policy.*`) with the same `.env`, dead Ollama port and `AGENT_POLICY_ENFORCEMENT=on`. |

All gateway requests carried `x-customer-id: 001`. At the end, all four processes were stopped by
process group, and ports 8010, 3011, 3010 and 9223 were confirmed closed. The shared :8000, :3001, :3000,
`procwise.service` (still `active`) and Ollama (:11434) were never touched.

**Choice of tool (why `get_policy`, not `run_supplier_ranking`).** After an approval, the backend replays the
stored call through the real `agentnick_control.build_tools`. `run_supplier_ranking` would run the whole
ranking agent, which writes routing rows and can ask the model for justifications. `get_policy` is a real,
registered, live action (`proc.bp_orchestrator_registry`, checkpoint `tool.call.before`) that only reads
governance rows. Its argument `query` is the registered string input `args.query`, so the conditions are
`all[tool.name in ["get_policy"], args.query eq "DEMO-…"]`.

## Step 1: decider map, through the gateway. PASS (one defect)

| # | Request | Response |
|---|---|---|
| 1a | `PUT /agent-policies/deciders/Demo%20Finance%20Manager` `{groups:["PROCWISE_FINANCE_REVIEWER_APPROVER"], emails:[], notes:"Stage 3 demo (2026-10-08)"}` | 200 `{"name":"Demo Finance Manager","groups":["[withheld]"],…,"lastModifiedBy":"c2b5b404-…"}` (**see defect D1**) |
| 1b | `PUT …/deciders/Demo%20CFO` `{groups:[], emails:["demo-cfo@example.test"]}` | 200, echoed as sent |
| 1c | `PUT …/deciders/Demo%20Ops` `{groups:["PROCWISE_ADMIN"]}` | 200, echoed as sent |
| 1d | `GET /agent-policies/deciders` | 200, the three rows with the **real** group names. So the value was stored correctly, and only the PUT echo was scrubbed. |

## Step 2: three policies, created, confirmed and activated through the gateway. PASS

Business area Operations / General (no second reviewer), owner "Demo Ops", inputs Query (`args.query`), Tool, Agent's reason.

| # | Request | Response |
|---|---|---|
| 2a | `POST /agent-policies/preview` ×3 (block, approve, notify) | 201. The examples computed exactly as expected: `DEMO-X` → Blocked / A person decides / Someone is told; `OTHER-1` → Nothing happens; `run_rag` + `DEMO-X` → Nothing happens. The only problem left was `checked` / `not_confirmed`. The approve policy's "then" reads: "the action pauses and Demo Finance Manager is asked to approve, then Demo CFO if there is no answer within PT2M; if the last level does not answer, the action is rejected." |
| 2b | `POST /agent-policies {form}` ×3 | 201 `OPS-0001` (block), `OPS-0002` (approve, deciders ["Demo Finance Manager","Demo CFO"], responseTime PT2M), `OPS-0003` (notify ["Demo Ops"]), each version 1 |
| 2c | `POST /agent-policies/OPS-0002/versions {baseVersion:1, intent:"activate"}` with `checked:null` | **422** `{"problems":[{"field":"checked","code":"not_confirmed","message":"Confirm the examples and how it is enforced."}]}` (readiness refusal, verbatim) |
| 2d | Preview of the approve form with deciders ["Demo Finance Manager","Demo Unlinked Person"] | 201, `problems:[{"field":"deciders","code":"decider_unmapped","message":"Nobody is linked to Demo Unlinked Person yet. An administrator must link them before this policy can be Active.","names":["Demo Unlinked Person"],"routeTo":"administrator"}]` (verbatim) |
| 2e | `POST …/OPS-000{1,2,3}/versions {baseVersion:1, intent:"activate"}` with a browser-supplied `checked:{by:"browser-supplied",…}` | 201 `{version:2}` ×3. `GET /agent-policies/OPS-000n` shows `status:"live"`, `liveVersion:2`, and `checked.by` = the bypass sub with server time (the browser value was replaced). |
| 2f | `live_policies.load()` in a separate process | `[('OPS-0001',2),('OPS-0002',2),('OPS-0003',2)]`. The 33 older `GEN-*` "live" rows in bp_testdb are skipped as invalid (see observation O1). |

The policy ids carry the Operations prefix (`OPS-`). The names start with "DEMO stage 3 …", so they are recognisable as demo data.

## Step 3: the tool loop with a scripted chat stand-in. PASS

`tool_runtime.run_tools("Stage 3 demo task", [stub get_policy], …, agent="agent_nick", workflow_id="demo-s3-wf-1",
user_id="demo-requester@example.test")`, with `tool_runtime._chat` replaced by a script. No model was called.
The tool was a **stub** `get_policy` whose handler records each run, not the BackendScheduler's agent_nick.
The script made four calls: `DEMO-BLOCK`, `DEMO-APPROVE`, `DEMO-NOTIFY`, then `DEMO-APPROVE` again. Then it answered "Done.".

| Call | What the "model" received next (the tool message) | Handler ran? |
|---|---|---|
| DEMO-BLOCK | `{"result":"blocked","reasonCode":"OPS-0001.demo_blocked","reason":"DEMO: this lookup is not allowed.","messageForPerson":"This lookup is not allowed.","policies":["OPS-0001"]}` | **no** |
| DEMO-APPROVE | `{"result":"paused_for_approval","requestIds":[4293],"respondWithin":"PT2M","whilePaused":"no_retry","reasonCode":"OPS-0002.demo_needs_approval","reason":"DEMO: this lookup needs a Finance decision first.","messageForPerson":"This lookup needs approval. You will hear back within 2 minutes."}` | **no** |
| DEMO-NOTIFY | `{"stub":true,"found":false,"query":"DEMO-NOTIFY"}` (the tool's own result) | **yes** |
| DEMO-APPROVE (repeat) | identical to call 2, **same `requestIds:[4293]`** | **no** |

The handler-run record was `['DEMO-NOTIFY']`. The loop took 5 rounds and returned no error.

Firing rows (`proc.bp_policy_firing`; no input is marked sensitive, so nothing is masked):

| firing_id | policy | outcome | result | matched_values | decision_id | reason |
|---|---|---|---|---|---|---|
| 1280 | OPS-0001 v2 | block | blocked | `{tool.name:get_policy, args.query:DEMO-BLOCK}` | – | – |
| 1281 | OPS-0002 v2 | approve | paused_for_approval | `{…, args.query:DEMO-APPROVE}` | 4293 | – |
| 1282 | OPS-0003 v2 | notify | allowed | `{…, args.query:DEMO-NOTIFY}` | – | – |
| 1283 | OPS-0002 v2 | approve | paused_for_approval | `{…, args.query:DEMO-APPROVE}` | 4293 | "Repeat call while request 4293 is open; no new request opened" |

All four rows have agent `agent_nick`, workflow `demo-s3-wf-1`, requested_by `demo-requester@example.test`.

Notifications (`proc.bp_policy_notification`):
- 574 → Demo Ops: "Looking up a governed policy (policy OPS-0001) was blocked." (`agent-policy:OPS-0001`)
- 575 → Demo Finance Manager: "Looking up a governed policy (policy OPS-0002) needs your decision." (`decision:4293`)
- 576 → Demo Ops: "Looking up a governed policy (policy OPS-0003) was allowed to run." (`agent-policy:OPS-0003`)

No notification text contains an input value.

Case 4293:
- status `open`, `current_level 0`;
- levels `[{Demo Finance Manager, PT2M}, {Demo CFO, PT2M}]`;
- `respond_by` = opened + 2 min;
- `on_timeout escalate_next`;
- requestedBy `demo-requester@example.test`;
- one firing_group.

## Step 4: approve through the gateway as an eligible person. PASS

The bypass identity holds `PROCWISE_FINANCE_REVIEWER_APPROVER` (linked to "Demo Finance Manager") and is not the requester.

| # | Request | Response |
|---|---|---|
| 4a | `GET /agent-policies/approvals?status=open` | 200, one case: `{id:4293, policyKey:"OPS-0002", actionPlain:"looking up a governed policy", inputs:[Query=DEMO-APPROVE, Agent's reason="Now the DEMO-APPROVE policy."], level:0, levelName:"Demo Finance Manager", levels:[…,"Demo CFO"], onTimeout:"escalate_next", requestedBy:"demo-requester@example.test", canDecide:true}` |
| 4b | `GET /agent-policies/approvals/4293` | 200, the same fields, plus `history.notes` (notification 575) and `history.firings` (1281, 1283; both paused), `replay:null` |
| 4c | `POST /agent-policies/approvals/4293/decide {verb:"approve", reason:"Stage 3 demo: approved at level 1"}` | **201** in 0.44 s: `{"decisionId":4293,"actionId":4294,"verb":"approve","result":"approved","decidedBy":"c2b5b404-…","level":0,"levelName":"Demo Finance Manager"}` |
| 4d | `GET /agent-policies/approvals/4293` 4 s later | `status:"actioned"`. One decision: approve by the bypass sub at level 0. Firings 1281 and 1283 are both `approved`, decided_level 0, decided_by the bypass sub. `history.replay = {outcome:"ran", resultSummary:"{\"found\": false, \"note\": \"No governed policy matches 'DEMO-APPROVE'. …\", \"available\": [\"weight_allocation_policy\", …]}", error:null}` |
| 4e | `POST …/approvals/4293/decide {verb:"approve"}` again | 409 `{"statusCode":409,"message":"This request has already been decided."}` |

**Two-level chain, as designed:** an approval at the current level closes the case. Level 2 ("Demo CFO")
is only used when level 1 times out (`approvals._insert_case`: `on_timeout = "escalate_next"` while a
next level exists; `act()` records the decision and closes the case). That is what happened.

**Replay ran once.** In the DB:
- the group has exactly one `agent_policy_replay` row (4295, subject `OPS-0002:1281`, `caseIds:[4293]`, outcome `ran`, actor `system:replay`);
- the real `get_policy` ran inside :8010 through `build_tools`;
- the repeat call (1283) did not cause a second run.

## Step 5: timeout, escalation, then rejection. PASS

A second `DEMO-APPROVE` call was made with `workflow_id demo-s3-wf-2`. The model received `paused_for_approval` with `requestIds:[4296]`, and the handler did not run.
Case 4296 opened at level 0, and notification 577 went to Demo Finance Manager.

| Round | Action (bp_testdb, only row 4296) | `approvals.sweep(conn, now, decision_ids=[4296])` | State after |
|---|---|---|---|
| 1 | `UPDATE proc.bp_decision SET respond_by = now() - 1 min WHERE decision_id = 4296 AND status='open'` | `{escalated:1, rejected:0, skipped:0, errors:0}` | `open`, `current_level 1`, respond_by = now + 2 min. Notification 578 → **Demo CFO**: "…(policy OPS-0002) needs your decision; the previous approver did not answer in time." Firing 1284 is still `paused_for_approval`. |
| 2 | same UPDATE again | `{escalated:0, rejected:1, skipped:0, errors:0}` | `actioned`. Action row 4297: `reject` by **`system:timeout`** at level 1. Firing 1284: **`timed_out`**, decided_level 1, decided_by `system:timeout`, reason "No decision in time; a timeout never approves". Notifications 579 (Demo Finance Manager) and 580 (the requester): "…was rejected: nobody decided in time, and a timeout never approves." **No replay row; it was never approved.** |

The sweep was run by hand and limited to `decision_ids=[4296]`, so no other session's open cases in the shared DB were touched.

## Step 6: fail closed, and the kill switch. PASS

All of these changes were made in the driver process only.

| Case | Set-up | Model received | Handler ran? | Firing row |
|---|---|---|---|---|
| 6a | `live_policies.load` monkeypatched to raise `PolicyStoreUnavailable` | `{"result":"blocked","reasonCode":"policy_check_unavailable","reason":"Policy checks are unavailable, so this action was not run."}` | no | 1285: policy_key `*`, outcome block, result `error`, reason `policy_check_unavailable: PolicyStoreUnavailable` |
| 6b | Bad DSN: after import, `PGHOST=127.0.0.1 PGPORT=1` in the process (the plan's wording) | same refusal, in 0.1 s | no | none could be written; it logged "could not log the policy_check_unavailable refusal for get_policy" (best effort, as designed) |
| 6c | `AGENT_POLICY_ENFORCEMENT=off`, loader still raising, call `DEMO-BLOCK` | `{"stub":true,"found":false,"query":"DEMO-BLOCK"}` | **yes** | none (0 rows for `demo-s3-wf-4`) |

Observation O2: a bad DSN **at import time** fails earlier. `tool_runtime` reads its round limit from
governance when it is imported, so the import itself raises `LimitUnavailable`. This is pre-existing and
not stage-3 code. 6b therefore broke the DSN after import.

## Step 7: screens in headless Chrome. PASS (two UI defects)

A fresh case 4298 was opened first (`workflow_id demo-s3-wf-6`; agent reason "The buyer asked me to check the
DEMO-APPROVE rule before ranking suppliers.").

**What was exercised (three passes, in order):**
1. Loaded `/spendiq` (re-navigating past the first-screen redirect), ran `go('workspace')` and
   `wfSetAgentTab('policy')`, and **clicked the "Policies" button**. The segments were
   List | Inventory | Documents | **Approvals 1** | Notifications | Deciders. The badge read **1**.
2. **Approvals.** One card (4298). It showed:
   - "looking up a governed policy";
   - "OPS-0002 · Level 1 of 2: Demo Finance Manager";
   - What the agent wants to do: Query DEMO-APPROVE, plus the agent's reason;
   - the situation and excerpt, and "Section demo of Stage 3 demo (no document)";
   - "If nobody answers by Oct 8, 2026, 11:34 PM, it goes to Demo CFO.";
   - the reason box, Reject… (disabled), and Approve….
3. **Clicked Approve….**
   - Confirm 1 read "Approve this request? / The action goes ahead as the agent asked. Your name is recorded with the decision. / Cancel / Yes".
   - A second Approve click while the confirm was open did nothing (one overlay).
   - Clicked Yes. Confirm 2 read "Confirm: approve request 4298? / Go back / Approve".
   - **No `/decide` call had been sent yet.**
   - Clicked Approve. One `POST /agent-policies/approvals/4298/decide` returned 201, and the card read "· Approved / You approved this request."
   - Its "What happened next" read "Waiting for the other approvals before the action runs." (**defect D2**).
   - After **Check for updates** it read "Action run: {"found": false, "note": "No governed policy matches 'DEMO-APPROVE'…".
   - The DB shows replay 4300, outcome `ran`.
4. **Notifications.** Six rows, all bold with "New". **Clicked Open on 579** (the timed-out case):
   - `POST /notifications/579/read` returned 201, **then** `GET /approvals/4296` returned 200;
   - the view switched to Approvals with "Opened from a notification";
   - the card read "OPS-0002 · Level 2 of 2: Demo CFO · **actioned**" (**defect D3**);
   - back on Notifications, row 579 was no longer bold or "New".
5. **Clicked Open on 576** (policy OPS-0003). Mark-read returned 201, then the policy form opened: "OPS-0003 · Version 2 · Active".
   Its health line read "Active since 2026-10-08. Checked 1 time in the last 30 days: 1 allowed, 0 paused, 0 blocked". The form was closed.
6. **Deciders** (Admin). It listed the three demo rows with their real groups/emails, notes, "2026-10-08 · c2b5b404-…" and Edit.
7. **List → Open OPS-0002.** Health line: "Active since 2026-10-08. **Checked 4 times in the last 30 days: 0 allowed, 4 paused, 0 blocked**".
   That matches firings 1281, 1283, 1284 and 1286: a paused check counts as paused whatever was decided later (Task 8 choice).
8. (Step 8 below) Retire ×3 from the screen.

Screenshots are in the scratchpad (`s3demo/shots/01…12*.png`).

**Console, over all three passes:**
- **0 `console.error` calls and 0 uncaught exceptions.** No failed `:3011` request.
- Every `/agent-policies` call returned 2xx: approvals, decide, approvals/{id}, notifications, notifications/{id}/read, deciders, taxonomy, preview, {key}, {key}/firings, retire.
- Network noise not from this feature:
  - the same `GET :3011/users/me/preferences` 404 as stages 1 and 2;
  - hundreds of resource-load entries (495 in the second pass alone) of **401 from `http://16.61.116.180:8000`** (`/agent-workflows`, `/workflows/types`, `/agent-groups`, `/i18n/*`, `/spendiq/*`, `/decisions`). The UI's `.env` points its AI API at that host, and the workspace screen polls it.

**Not exercised:**
- Reject from the screen.
- The Deciders editor's save.
- A non-eligible viewer's card. The bypass is eligible, so the "you cannot decide" text was not seen live.
- The badge refreshing on its own (it refreshes only on Refresh or reload, by design).

## Step 8: retire the demo policies. PASS

For OPS-0001, OPS-0002 and OPS-0003 in turn, from the screen:
- Open → **Retire…**;
- confirm 1: "Retire this policy? / Agents will no longer be held to it. It stays in the inventory as Retired. / Cancel / Yes";
- confirm 2: "Confirm: retire OPS-000n? / Go back / Retire";
- `POST /agent-policies/OPS-000n/retire` returned 201, and the form showed "Version 3 · Retired".

Afterwards:
- in `bp_testdb`, all three are `retired` with live_version NULL;
- `live_policies.load()` returns `[]`;
- all demo cases (4293, 4296, 4298) are closed (`actioned`). None is left open.

**Left in `bp_testdb` as the record:**
- the decider rows "Demo Finance Manager", "Demo CFO" and "Demo Ops". Retired policies still name them, so they were not deleted;
- policies OPS-0001..0003 (retired, v1–v3);
- cases and action rows 4293–4300;
- firing rows 1280–1286 (append-only by design);
- notifications 574–581;
- the audit rows.

No S3 objects were created.

## Defects found (recorded, not patched)

**D1. `PUT /agent-policies/deciders/{name}` echoes a group name as "[withheld]".**
- Request: `PUT …/deciders/Demo%20Finance%20Manager {groups:["PROCWISE_FINANCE_REVIEWER_APPROVER"]}`.
- Response: 200 `groups:["[withheld]"]`.
- Expected: the saved row as stored.
- Cause: the user's OutputSafety exemption covers the GET paths only. The PUT response is a 2xx echo of a Cognito group name, which reads like an env var.
- Impact today: none on screen. The UI ignores the PUT body and reloads with GET, which shows the real value. Any other client of the PUT would see "[withheld]".
- Decision needed: add the exact PUT path to the exemption, or return no body.

**D2. Right after an approval, the card says "Waiting for the other approvals before the action runs." even when the case is alone in its group.**
- The UI reads `GET /approvals/{id}` straight after the decide.
- The replay runs in the background, so its row is not written yet, and `history.replay` is `null`.
- `replayLine(null, 'approved')` cannot tell "running soon" from "waiting for other cases".
- After "Check for updates" it showed "Action run: …".
- Expected: wording that is true in both cases, or an API field saying whether other cases in the group are still open.

**D3. A closed case shows the raw status "actioned" and not what happened.**
- Opening notification 579 showed "OPS-0002 · Level 2 of 2: Demo CFO · actioned".
- The case was rejected by `system:timeout`. `history.decisions` holds `{verb:"reject", by:"system:timeout", timeout:true}`.
- The card does not show that history, and `STATUS_WORDS` has no "actioned" entry.
- Expected: "Rejected: nobody decided in time" (or the decision and who made it).

**Observations (not stage-3 defects):**
- O1. `bp_testdb` holds 33 policies marked `live` (GEN-0285 … GEN-2460) that fail validation. Each `live_policies.load()` logs a warning for every one of them, so every gated tool call in a process with a cold cache pays for 33 validations. They look like leftovers from earlier live tests.
- O2. A bad DSN at import time makes `tool_runtime` itself fail to import (`LimitUnavailable` from governed limits). This is pre-existing.
- O3. The approve policy's "how it is enforced" text says "within PT2M" (a raw ISO duration). The agent's own message uses the person-written "2 minutes".
- O4. A policy typed in by a person still shows "Extraction confidence: Medium" and "Check before this goes Active / The excerpt does not appear word for word in the document" while Active. This is stage 1/2 form behaviour.
- O5. No notification tells the requester that a person approved the request. The requester hears only of a timeout rejection. This is not required by the plan; noted for stage 4.
- O6. The excerpt "…is approveed." is a typo in my demo fixture, not the product's.

## Rulings (every "Ruling:" line in the stage-3 ledger)

1. decider names map to people via an admin-editable table (bp_policy_decider_map: groups/emails); unmapped deciders/notify names block Active; Admin is not automatically an approver — cost if wrong: admins must map themselves to approve
2. notifications are in-product rows (no email in stage 3; internal email is refused by the supplier allow-list path) — cost if wrong: people must open the screen to see them
3. agent.reason = the assistant text accompanying the tool call (may be absent → "No reason given"); agent.name = calling agent passed into run_tools — cost if wrong: approvers sometimes see no reason
4. multiple matching approve policies → one case each; the action runs only when all approve (user's fallback); block always wins (user ruling); full live-conflict cases are stage 4 — cost if wrong: stage 4 replaces this path
5. enforcement fails closed (store unavailable → refuse); kill switch AGENT_POLICY_ENFORCEMENT=off; no live policies → behaviour identical to today — cost if wrong: a DB outage stops agent tool use
6. live demo drives run_tools with a scripted chat stand-in (shared model saturated) — cost if wrong: the real model's handling of the refusal text is unverified
7. Task 2 — masking uses the union of sensitive fields across all policies; unparseable durations count as longest; missing onMissingData defaults to fail_closed — cost if wrong: some non-sensitive values hidden; longer quoted waits
8. one shared ISO duration parser for enforcement + approvals; unreadable → company default (agent told the same wait the timer applies)
9. Task 4 — act() lazily imports replay.run (no mutable hook; dual import paths src.services vs services would silently drop approved actions); missing replay logs ERROR — cost if wrong: none
10. Task 5 — the firing row records the human decision (approved/rejected/timed_out); the replay outcome (ran/blocked at re-check/error/no_runtime) lives on the agent_policy_replay row (the append-only trigger forbids changing a resolved firing) — cost if wrong: learning sees "approved" for an action later blocked at replay
11. Task 5 — a missing agent runtime must not consume the exactly-once claim; replay's new cases are inserted inside its single transaction via a transaction-less approvals._insert_case — cost if wrong: none
12. Task 6 — kill switch is read before importing the gate (operator escape hatch survives a broken module); gate writes atomic per call; every firing row linked to a case is settled; case reuse keyed on requester too (closes a self-approval bypass); notify rows record the overall result and are settled with the case — cost if wrong: none
13. Task 8 — replay outcome surfaces as history.replay on GET /approvals/{id} (backend, Task 7) and the card renders it (UI fix round) — cost if wrong: none
14. an approved action whose background replay is lost (restart, no runtime) is retried by the approval sweeper (approved group, no replay row, bounded attempts) — final wave — cost if wrong: an action could run minutes after approval rather than at once
15. USER RULING: extend the OutputSafety 2xx exemption to GET approvals list/detail, notifications, deciders, and /{key}/firings (exact paths; errors still filtered)
16. approvers who can decide see unmasked values in history firings too (consistent with inputs) — cost if wrong: approver sees the same values in two places

Ruling 6 still stands. The real model's handling of the refusal text is unverified, because no model was called.
Rulings 4, 5, 10, 12 and 15 were exercised live:
- 4: block wins, one case per approve policy;
- 5: fail closed and the kill switch;
- 10: firing `approved`/`timed_out`, replay outcome on its own row;
- 12: repeat reuse, linked rows settled;
- 15: the GET reads were not scrubbed; the PUT echo was (D1).

## Open findings carried from the ledger (not re-tested here)

- Task 1 minors: TRUNCATE is not blocked on the firing log; a paused row may move back to paused; the rollback has an unconditional DROP COLUMN. The scratch `task-1-report.md` is committed on the branch, to be removed with `git rm --cached` in the final wave.
- Task 4 minors: a grace window of up to 60 s between `respond_by` and the next sweep, during which the current level can still act; the `_after_lock` seam is module state; ImportError inside replay is caught as "missing".
- Task 5 minors: `mask_text` does not mask a sensitive value glued to word characters; the stored-policy mask SQL is only tested via a seam.
- Task 6 minors:
  - store-outage latency (failures are not cached);
  - case reuse ignores the policy version;
  - test firing rows accumulate in bp_testdb;
  - a notify row in a two-approval group is settled by the first decision.
- Deploy note: confirm `live_policies.load()` against bp_sqldb before switching enforcement on there.

## Diff summary

### BP_Backend: `0d682122..HEAD` (17 commits before this note; 37 files, +5042 / −34)

```
0794a350 fix(agent-policy): approval endpoints review round 1
eb37a838 feat(agent-policy): an approval case's history carries its replay outcome
0b508ee8 fix(agent-policy): gate writes are atomic, linked firing rows settle with their case, reuse keys on the requester
7a5a617f test(p8): the write-endpoint guard accepts the gateway's verified identity
9cae612d feat(agent-policy): approval, notification, decider and firing endpoints
d06d83f0 fix(agent-policy): replay logs no input values, opens new cases in its own transaction, never spends the claim without a runtime
6e01aef9 feat(agent-policy): the gate in the tool loop -- live policies now constrain agent tool calls
b492abe4 refactor(agent-policy): open_case's writes as _insert_case, usable inside a caller's transaction
cd5d3d3c feat(agent-policy): replay an approved action once its whole group approves, re-checked against current policies
7d48ac96 fix(agent-policy): an approval always reaches the replay or says it cannot; timeouts escalate while a level remains
28b5f32b feat(agent-policy): approval cases, decisions and the timeout sweeper
9260cfc1 feat(agent-policy): one ISO 8601 duration reader; the agent is told the wait the timer applies
870a69ed fix(agent-policy): enforcement masks across policies, never under-states the wait, fails closed by default
fc6904b2 fix(agent-policy): decider readiness follow-ups, live-test setup links deciders
b60d4960 feat(agent-policy): enforcement check (pure) and the live-policy loader
442f1ba0 feat(agent-policy): decider map eligibility and unmapped-decider readiness problem
fd382b48 feat(agent-policy): enforcement tables, append-only firing log, bp_decision approval columns
```

| Area | Files (lines added) |
|---|---|
| Schema | `deploy/sql/2026-10-10_bp_agent_policy_enforcement.sql` (+81) and its rollback (+11) |
| New services | `agent_policy/approval_views.py` (+495), `approvals.py` (+419), `replay.py` (+395), `gate.py` (+301), `enforcement.py` (+141), `replay_retry.py` (+88), `live_policies.py` (+72), `deciders.py` (+58), `durations.py` (+39) |
| Wiring | `tool_runtime.py` (+70: gate before each handler), `backend_scheduler.py` (+51: sweep + replay retry), `routers/agent_policies.py` (+203), `api/main.py` (+9: user-approved GET exemptions), `readiness.py` (+41), `agentnick_control.py` (+7), `routers/agents.py` (+3), `base_agent.py` (+1), `scripts/p8_endpoint_scan.py` (+6) |
| Tests | `agent_policy/test_gate.py`, `test_approval_endpoints_live.py`, `test_approvals_live.py`, `test_replay_live.py`, `test_approval_endpoints_scrub.py`, `test_enforcement.py`, `test_deciders.py`, `test_approvals_job.py`, `test_durations.py`, the migration test, plus small edits in 5 others |
| Scratch (to remove) | `.superpowers/…/task-1-report.md` (+8) |

### Gateway: `068d84e..HEAD` (2 commits; 3 files, +351)

```
6b60ffc docs(agent-policy): decide's timeout comment matches the background replay
5213d99 feat(agent-policy): approval, notification, decider and firing routes
```

The files are `agent-policy.controller.ts` (+48), `agent-policy.yml` (+200) and `agent-policy.controller.spec.ts` (+103).

### UI: `b2c3e2a..HEAD` (2 commits; 7 files, +1092 / −6)

```
49e3940 fix(agent-policy): the approval card shows what came of the approved action
0c3b172 feat(agent-policy): approvals, notifications, deciders and the health line
```

| File | Lines added |
|---|---|
| `agentPolicy/approvals.js` | +212 |
| `approvals.test.js` | +160 |
| `engineWiring.stage3.contract.test.js` | +347 |
| `model.js` | +12 |
| `model.test.js` | +15 |
| `engine.js` | +349 |
| `index.jsx` | 3 lines changed |

**`engine.js` guarded functions are byte-identical to base `b2c3e2a`.** Each function body was extracted at
both commits and hashed (sha256, first 16 hex characters):

| Function | Bytes | Hash at base and at HEAD |
|---|---|---|
| `policyEdit` | 3281 | `46bfb5ad40f48960` |
| `policyDelete` | 183 | `30ffba85689e2a22` |
| `openFormModal` | 4645 | `73afdb556aa98a0d` |

The `b2c3e2a..HEAD` diff of `engine.js` does not mention any of the three names.
