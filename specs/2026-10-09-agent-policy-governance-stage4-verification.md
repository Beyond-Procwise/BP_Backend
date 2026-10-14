# Agent policy governance, stage 4 (conflicts): verification and diff summary

Date: 2026-10-09, 13:31 to 13:52 UTC. Branch `agent-policy-stage4` in three worktrees (BP `e9f2134b`, gateway
`e7d9c99`, UI `b943567`). Database: `bp_testdb` (checked with `select current_database()` before the one
direct write step). Nothing has been merged or pushed.

## In plain English

- **Contradicting rules are now caught when they are saved.** A "needs approval" rule from one document and a
  "not allowed" rule from another document, both covering the same lookup, raised one conflict case the moment
  the second was saved. Code found the example that proves the clash, and it re-checks true against both rules.
  Both owners were told. The screens show "Conflict sent for decision" on both rules, in the list and in the
  inventory.
- **Rules from the same document never count as a clash** ("tiered" rules). Two such rules that overlap raised
  nothing.
- **The owners decide, and nothing is changed behind their backs.**
  - "Keep both: B takes priority" wrote a standing rule, and the orchestrator feed now carries it on both rules.
  - "Change E" changed nothing. The screen offered "Open as a draft", which filled in the change note only.
    Cancelling left the rule at exactly one version.
- **At run time, two approval rules that disagree pause the action and ask both sides.** The agent was told
  it is paused. Each rule's last approver got their own one-step request. Only after both approved did the
  action run, once.
- **"Not allowed" still wins.** A call matching an approval rule and a "not allowed" rule was blocked at once;
  the clash was written down and closed.
- **Nobody answering is never a yes.** One side approved, the other side ran out of time; the sweeper rejected
  the action and the clash was closed as rejected. The action never ran.
- **Five same-way decisions raise a suggestion.** After the same clash had been approved by people five times,
  a case was raised proposing a standing rule. Not before.
- **The generic Decisions screen never shows these cases.**
- **Screens:** the Conflicts tab (with a real decide through the two-step confirm), the policy form's
  Conflicts section, the approval card's "Policies in conflict" block and a notification opening a conflict
  were all used in headless Chrome. **0 console errors and 0 uncaught exceptions.**
- **One fault was found (D1).** A policy's own history shows a settled run-time clash as
  "approve_or_reject" with no decider, instead of what was decided. Details below.
- **Clean-up.** All 12 demo rules are retired (none is live in `bp_testdb`), my 4 decider-map rows are
  deleted, and no company setting was changed.

## How the stack was run

| Part | Port | How |
|---|---|---|
| Backend | 127.0.0.1:8010 | `.venv/bin/uvicorn api.main:app --workers 1` from the BP worktree, `PYTHONPATH=<worktree>:<worktree>/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`, `AGENT_POLICY_ENFORCEMENT=on`, `AGENT_POLICY_APPROVAL_SWEEP=off`, **`AGENT_POLICY_CONFLICT_SCAN=off`**, `OLLAMA_BASE_URL=OLLAMA_HOST=http://127.0.0.1:9` (dead port), `OLLAMA_CLOUD_*` unset. The only model traffic in its log is the start-up preload, refused by 127.0.0.1:9. PID 337146. |
| Gateway | :3011 | `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development AUTH_BYPASS_GROUPS=PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER BP_BACKEND_URL=http://127.0.0.1:8010`; gateway key read from the BP `.env`, never printed. Bypass sub `c2b5b404-40c1-7047-71c4-dd8093ecf25d`. PID 338337. |
| UI | 127.0.0.1:3010 | Vite from the UI worktree, `VITE_API_URL=http://127.0.0.1:3011 VITE_DEV_AUTH_BYPASS=true`. The untracked `mockPipelineDeals.js` was copied in for the run (stage 1 finding 6) and deleted afterwards; the UI worktree is clean. PID 342998. |
| Headless Chrome | :9223 | `google-chrome --headless=new` (Chrome 150), driven over CDP with `/usr/share/nodejs/ws`, 1440×900. PID 343235. |
| Driver | none | Scratch Python scripts (not committed) that import the worktree's own modules with the same `.env`, dead Ollama port and `AGENT_POLICY_ENFORCEMENT=on`. `tool_runtime._chat` is replaced by a script, so **no model was called**. The tool is a stub `get_policy` that records each run. After approval, the backend replays through the real `get_policy` (`agentnick_control.build_tools`), as in stage 3. |

All gateway requests carried `x-customer-id: 001`. The orchestrator feed was read straight from :8010 with
`x-orchestrator-key` from the `.env` (never printed; without it the feed answers 401). At the end the four
processes were stopped by process group (`kill -TERM -- -<pgid>`, my PIDs only), and ports 8010, 3011, 3010 and
9223 were confirmed closed. `procwise.service` was never touched (still `active`), nor were :8000/:3000/:3001
or Ollama.

### Keeping the shared database safe

- **Unique demo scope.** Every demo rule is `tool.name in ["get_policy"]` and `args.query eq "DEMO-S4V-<pair>"`
  (one query value per pair), named "DEMO stage 4 …", in Operations / General (ids `OPS-0004` … `OPS-0015`).
  Owners and deciders are "Demo S4 Finance Owner", "Demo S4 Customer Owner", "Demo S4 Finance Approver" and
  "Demo S4 Ops Approver".
- **Dry run before every create.** `bp_testdb` holds 4,470 non-retired policy versions (4,362 drafts, 54 "live"
  rows that all fail validation). Before any save, a read-only script compiled each demo form and ran
  `conflict_cases._pairs` against all of them. The only witnessed pairs were the intended demo pairs
  (A|B, E|F, G|H); every check reported 0 errors (A: 902 candidate pairs, B: 2,008, …).
- **Result:** every conflict row in `bp_testdb` after the run (12 rows) names only demo policies. **No save raised
  a case against a non-demo policy**, so no other rows had to be removed. The live test suites run afterwards
  also left 0 stray conflict rows.
- The scan stayed off; detection ran only through my own saves (each capped at 25 new cases).
- `live_policies.load()` returned `[]` before the run (no valid live policy in `bp_testdb`), so no other policy
  could take part in the demo calls.

## The demo policies

| Pair | Key | Outcome | Source document | Owner | Deciders | Query |
|---|---|---|---|---|---|---|
| A/B | OPS-0004 (A) | approve | Demo Finance | Demo S4 Finance Owner | Demo S4 Finance Approver | DEMO-S4V-AB |
| A/B | OPS-0005 (B) | block | Demo Customer | Demo S4 Customer Owner | – | DEMO-S4V-AB |
| tiered | OPS-0006 (T1) | approve | Demo S4 Tiered | Demo S4 Finance Owner | Demo S4 Finance Approver | in [TIER-A, TIER-B] |
| tiered | OPS-0007 (T2) | block | Demo S4 Tiered | Demo S4 Finance Owner | – | TIER-B |
| E/F | OPS-0008 (E) | approve | Demo Finance | Demo S4 Finance Owner | Demo S4 Finance Approver | DEMO-S4V-EF |
| E/F | OPS-0009 (F) | block | Demo Customer | Demo S4 Customer Owner | – | DEMO-S4V-EF |
| live | OPS-0010 (L1) | approve, PT10M | Demo Finance | Demo S4 Finance Owner | Demo S4 Finance Approver | DEMO-S4V-LIVE |
| live | OPS-0011 (L2) | approve, PT10M | Demo Ops | Demo S4 Customer Owner | Demo S4 Ops Approver | DEMO-S4V-LIVE |
| repeat | OPS-0012 (R1) | approve, PT10M | Demo Finance | Demo S4 Finance Owner | Demo S4 Finance Approver | DEMO-S4V-REP |
| repeat | OPS-0013 (R2) | approve, PT10M | Demo Ops | Demo S4 Customer Owner | Demo S4 Ops Approver | DEMO-S4V-REP |
| screen | OPS-0014 (G) | approve | Demo Finance | Demo S4 Finance Owner | Demo S4 Finance Approver | DEMO-S4V-GH |
| screen | OPS-0015 (H) | block | Demo Customer | Demo S4 Customer Owner | – | DEMO-S4V-GH |

G/H repeats step 2 for the screens: A/B had already been decided (step 4) before the UI was started, so a fresh
open pair was needed to see the tag and to decide from the Conflicts tab.

## Step 1: owners and deciders in the decider map. PASS

`PUT /agent-policies/deciders/<name>` ×4, each `{groups:["PROCWISE_FINANCE_REVIEWER_APPROVER"], emails:[], notes:"Stage 4 demo (2026-10-09)"}`
→ 200 `{"name":…,"savedAt":…}` each. `GET /agent-policies/deciders` → 200, the four rows with the real group
name. (Stage 3's D1, the "[withheld]" echo, no longer happens: the PUT now answers name and savedAt only.)
All four names map to a group the bypass identity holds, so one person can act for every owner and decider.

## Step 2: design time, A then B. PASS

| # | Request | Response |
|---|---|---|
| 2a | `POST /agent-policies {form A}` | 201 `{"policyKey":"OPS-0004","version":1}`; no case (B did not exist yet) |
| 2b | `POST /agent-policies {form B}` | 201 `{"policyKey":"OPS-0005","version":1}`; detection after the save raised case **20593** |
| 2c | `GET /agent-policies/conflicts` | 200, one open case `pc_20593`, `raisedBy:"save"` |
| 2d | `GET /agent-policies/conflicts/20593` | 200: policies OPS-0004 v1 (approve, Demo Finance §4.1, excerpt verbatim) and OPS-0005 v1 (block, Demo Customer §4.1); `example {"tool.name":"get_policy","args.query":"DEMO-S4V-AB"}`; `why` "One policy needs approval from Demo S4 Finance Approver; the other does not allow this at all."; `options ["keep_both:OPS-0005","change:OPS-0004","change:OPS-0005","limit:OPS-0004","limit:OPS-0005","retire:OPS-0004","retire:OPS-0005"]` (Q1: only the block can be kept as the winner); `respondWithin null`; `unroutable []`; `canDecide true` |
| 2e | `GET /agent-policies` | OPS-0004 and OPS-0005 both carry `openConflicts:["pc_20593"]` |
| 2f | `GET /agent-policies/OPS-0004` and `/OPS-0005` | `conflicts:[{caseId:"pc_20593",kind:"policy",isOpen:true,otherPolicies:[the other]}]`, `pendingConflictAction:null` |

Database:
- `bp_agent_policy_conflict` 20593: kind `policy`, pair `OPS-0004|OPS-0005`, versions `{OPS-0004:1, OPS-0005:1}`, raised_by `save`, open.
- `bp_decision` 20593: subject_type `policy_conflict`, decision `resolve_conflict`, status `open`, `respond_by` NULL, on_timeout `none`, created_by `system:conflict_detector`.
- Owner notifications (firing_id NULL, link `conflict:20593`): 14554 → Demo S4 Finance Owner and 14555 → Demo S4 Customer Owner, "Policies OPS-0004 and OPS-0005 conflict; a decision is needed."

**The witness, re-evaluated** from the stored row with the one evaluator (`conditions.to_engine` +
`policy_condition.evaluate`) against each policy's stored compiled version:

```
stored witness: {'tool.name': 'get_policy', 'args.query': 'DEMO-S4V-AB'}
OPS-0004 v 1 approve matches witness: True tool listed: True
   control input {'tool.name': 'get_policy', 'args.query': 'OTHER-S4'} -> False
OPS-0005 v 1 block matches witness: True tool listed: True
   control input {'tool.name': 'get_policy', 'args.query': 'OTHER-S4'} -> False
```

**The tag on screen** (pair G/H, the same path: `POST` G then H raised case 20644, `openConflicts:["pc_20644"]` on both):
- List: OPS-0014 "Draft · **Conflict sent for decision**", OPS-0015 "Draft · **Conflict sent for decision**" (also OPS-0012/0013 for the open repeat proposal of step 9).
- Inventory: the same four rows, e.g. "General OPS-0014 DEMO stage 4 G approve (Demo Finance) Needs approval: a person decides Draft Conflict sent for decision Demo Finance · 8.1 Open".
- Policies without an open case (OPS-0004, 0005, 0008 …) show no tag.

## Step 3: tiered pair from one document. PASS

`POST /agent-policies` T1 → 201 OPS-0006 v1; T2 → 201 OPS-0007 v1. No conflict row, no case; `GET` of each shows `conflicts:[]`.
To show this is the same-source rule and not a lack of overlap:

```
sources: Demo S4 Tiered | Demo S4 Tiered same_source: True
outcomes: approve block
witness exists: {'args.query': 'DEMO-S4V-TIER-B', 'tool.name': 'get_policy'}
design_time_pair: False
```

## Step 4: standing rule `keep_both:B`, decided as an owner. PASS

| # | Request | Response |
|---|---|---|
| 4a | `POST /agent-policies/conflicts/20593/decide {option:"keep_both:OPS-0005"}` (no reason) | **400** `{"message":"reason must be text of at most 2000 characters","error":"Bad Request","statusCode":400}` (the gateway's type check; see O2) |
| 4b | `… {option:"keep_both:OPS-0004", reason:"x"}` | **422** `{"problems":[{"field":"option","code":"unknown_option","message":"That is not one of this case's options."}]}` (Q1: the approve policy can never be kept over the block) |
| 4c | `… {option:"keep_both:OPS-0005", reason:"Stage 4 demo: the customer block stands; …"}` | **201** `{"caseId":"pc_20593","decision":"keep_both:OPS-0005","scope":"standing_rule","decidedBy":"c2b5b404-…","decidedAt":"2026-10-09T13:35:32.719955+00:00","reason":"…","actionId":20594,"applied":"standing_rule"}` |
| 4d | the same again | **409** `{"statusCode":409,"message":"This conflict has already been decided."}` |
| 4e | `POST /agent-policies/OPS-0004/versions` and `/OPS-0005/versions` `{baseVersion:1, intent:"activate", form with a browser-supplied checked}` | 201 `{version:2}` each; the re-detection after each save raised nothing (the rule covers the pair) |

Database: rule 232 `OPS-0004|OPS-0005`, prevails `OPS-0005`, yields `OPS-0004`, "OPS-0005 takes priority over
OPS-0004", decision 20593, superseded NULL. Conflict 20593 closed, outcome `keep_both:OPS-0005`, by_person true.
Notifications 14556/14557 to both owners: "The conflict between OPS-0004 and OPS-0005 was decided: Keep both: OPS-0005 takes priority."

`GET /orchestrator/agent-policies/v2/live` (with the orchestrator key; 401 without it):

```
hard-policy-feed/2 policies: ['OPS-0004', 'OPS-0005'] refused: 54
OPS-0004 v 2 [{"with": "OPS-0005", "rule": "OPS-0005 takes priority over OPS-0004", "caseId": "pc_20593", "decidedAt": "2026-10-09T13:35:32.719955+00:00", "prevails": "OPS-0005"}]
OPS-0005 v 2 [{"with": "OPS-0004", "rule": "OPS-0005 takes priority over OPS-0004", "caseId": "pc_20593", "decidedAt": "2026-10-09T13:35:32.719955+00:00", "prevails": "OPS-0005"}]
```

(The 54 refused are the old invalid `GEN-*` "live" rows in bp_testdb, as in stage 3's O1.)

## Step 5: `change:<key>` on a second pair; open the draft in the UI, cancel. PASS

| # | Request | Response |
|---|---|---|
| 5a | `POST /agent-policies` E, then F | 201 OPS-0008 v1, OPS-0009 v1; case **20595** raised (`example {"tool.name":"get_policy","args.query":"DEMO-S4V-EF"}`) |
| 5b | `POST /agent-policies/conflicts/20595/decide {option:"change:OPS-0008", reason:"Stage 4 demo: narrow E so it no longer overlaps F."}` | 201 `{"caseId":"pc_20595","decision":"change:OPS-0008","scope":"this_action",…,"actionId":20596,"applied":"draft_pending"}` |
| 5c | `GET /agent-policies/OPS-0008` | `latestVersion 1`, versions `[1]`, `pendingConflictAction {caseId:"pc_20595", action:"change", changeNote:"Conflict decision pc_20595: Change OPS-0008 — Stage 4 demo: narrow E so it no longer overlaps F.", limitText:null}` |

In headless Chrome (List → Open OPS-0008):
- the form's **Conflicts** section read "Conflict decision pc_20595 asks for this policy to be changed. Open it as a draft, change it, then save. [Open as a draft]" and the history line "pc_20595 with OPS-0009 · Between policies · Decided: Change OPS-0008 — Stage 4 demo: narrow E so it no longer overlaps F.";
- the Change note was empty; after **Open as a draft** it read "Conflict decision pc_20595: Change OPS-0008 — Stage 4 demo: narrow E so it no longer overlaps F." and had the focus;
- the first **Cancel** discarded the edit (Change note empty again, form still open), the second closed the form;
- the only agent-policy calls during this were `GET /agent-policies/taxonomy`, `GET /agent-policies/OPS-0008` and two `POST /agent-policies/preview`. **No `/versions`, `/retire` or `/decide` call.**

Afterwards `GET /agent-policies/OPS-0008` → `latestVersion 1`, versions `[1]`, and `select count(*) from
proc.bp_agent_policy_version where policy_key='OPS-0008'` → **1**. The version count is unchanged.

## Step 6: a live conflict between two approve policies. PASS

L1 (OPS-0010, Demo Finance, decider Demo S4 Finance Approver) and L2 (OPS-0011, Demo Ops, decider Demo S4 Ops
Approver) were created and activated (v2). No design-time case: both are approve (design-time compares outcomes only).

Driver: `run_tools(…, agent="agent_nick", workflow_id="demo-s4-wf-6", user_id="demo-s4-requester@example.test")`;
the script called `get_policy {query:"DEMO-S4V-LIVE"}` twice, then answered "Done.".

| Call | What the "model" received next | Handler ran? |
|---|---|---|
| 1 | `{"result":"paused_for_approval","requestIds":[20598,20599],"respondWithin":"PT10M","whilePaused":"no_retry","reasonCode":"OPS-0010.demo_s4_approve","reason":"DEMO S4: this lookup needs approval.","messageForPerson":"This lookup needs approval.","conflictCaseId":"pc_20597"}` | no |
| 2 (repeat) | identical, **same `requestIds` and the same `conflictCaseId`** | no |

The live case (decision **20597**, conflict row kind `live`, pair `OPS-0010|OPS-0011`, versions `{OPS-0010:2, OPS-0011:2}`, raised_by `live`), with
`decision approve_or_reject`, `options ["approve","reject"]`, `on_timeout reject`, `respond_by − created_at = 0:09:59.9`,
`pairs [["OPS-0010","OPS-0011"]]`, `requestedBy demo-s4-requester@example.test`, and **args (condition fields only)**:

```
facts.action = {"args": {"query": "DEMO-S4V-LIVE"}, "tool": "get_policy", "agent": "agent_nick",
                "plain": "Looking up a governed policy", "workflowId": "demo-s4-wf-6"}
```

The agent's reason ("Checking the DEMO-S4V-LIVE rule before I continue.") is not a condition field: it is in the
member cases' `facts.action.reason`, not in the live case. (`get_policy` takes only `query`, so this run cannot
show an extra argument being dropped; `test_conflict_live_gate.py` covers that with a non-condition argument.)

The two **one-level member cases**:

| Case | Policy | levels | current_level | on_timeout | liveConflict | Notification |
|---|---|---|---|---|---|---|
| 20598 | OPS-0010 | `[{"name":"Demo S4 Finance Approver","respondWithin":"PT10M"}]` | 0 | reject | 20597 | 14562 → Demo S4 Finance Approver, "Looking up a governed policy (policy OPS-0010) needs your decision." |
| 20599 | OPS-0011 | `[{"name":"Demo S4 Ops Approver","respondWithin":"PT10M"}]` | 0 | reject | 20597 | 14563 → Demo S4 Ops Approver |

Firings 4324/4325 (call 1) and 4326/4327 (repeat, reason "Repeat call while request 2059n is open; no new request opened"). Only one live record was written.

| # | Request | Response |
|---|---|---|
| 6a | `GET /agent-policies/approvals?status=open` | 200, 20598 (levelName/levels "Demo S4 Finance Approver") and 20599 ("Demo S4 Ops Approver"), canDecide true |
| 6b | `GET /agent-policies/approvals/20598` | 200, `conflict {caseId:"pc_20597", why:"The policies name different approvers: Demo S4 Finance Approver and Demo S4 Ops Approver.", policies:[OPS-0010 v2, OPS-0011 v2], prior:{sameConflict:0}, options:["approve","reject"], respondWithin:"PT10M", actionPlain:"Looking up a governed policy", args:{query:"DEMO-S4V-LIVE"}, example:{…}}` |
| 6c | `POST /agent-policies/approvals/20598/decide {verb:"approve"}` | 201, `result approved`, `group {open:1, approved:1, rejected:0, total:2}` |
| 6d | `POST /agent-policies/approvals/20599/decide {verb:"approve"}` | 201, `group {open:0, approved:2, rejected:0, total:2}` |
| 6e | `GET /agent-policies/approvals/20599` 5 s later | `history.replay = {outcome:"ran", resultSummary:"{\"found\": false, \"note\": \"No governed policy matches 'DEMO-S4V-LIVE'. …"}` |

**The replay ran once:** one `agent_policy_replay` row (20603, subject `OPS-0010:4324`, `caseIds [20598, 20599]`,
outcome `ran`, actor `system:replay`). The live case settled as action row 20602 `approve` by the bypass sub
(the last approver, "Stage 4 demo: ops approves"), `memberCases [20598, 20599]`; conflict 20597 closed `approve`,
by_person true. Firings 4324–4327 are `approved`, decided_level 0.

## Step 7: block record. PASS

`run_tools` (workflow `demo-s4-wf-7`), one call `get_policy {query:"DEMO-S4V-AB"}`, matching A (approve,
Demo Finance) and B (block, Demo Customer), both live:

```
{"result":"blocked","reasonCode":"OPS-0005.demo_s4_block","reason":"DEMO S4: this lookup is not allowed.",
 "messageForPerson":"This lookup is not allowed.","policies":["OPS-0005"],"conflictCaseId":"pc_20604"}
```

The handler did not run. The closed live case 20604: decision `block`, status `actioned`, actioned_by
`system:not_allowed`, decision_scope `this_action`, `facts.action.args {"query":"DEMO-S4V-AB"}`; conflict row
kind `live`, outcome `block`, by_person false. Firings 4328 (OPS-0005 block, `blocked`) and 4329 (OPS-0004
approve, `blocked`), neither linked to a case. **No approval case was opened**, and no new policy case: the
pair's standing rule (step 4) already covers it.

## Step 9: repeat-5 (run before step 8, so step 8's open case could be shown on screen). PASS

`live_conflict_repeat` was **already 5** in `bp_admin_config.agent_policy_settings` (the ruled default), so no
setting was changed or needed restoring (re-read as 5 at the end).

R1 (OPS-0012) and R2 (OPS-0013) were created and activated. Five identical calls `get_policy {query:"DEMO-S4V-REP"}`
(workflows `demo-s4-wf-9-1` … `-5`, same requester), each approved by both members through the gateway:

| Run | Paused with | Live case | After both approvals |
|---|---|---|---|
| 1 | `[20606, 20607]` | pc_20605 | settled `approve`, by_person |
| 2 | `[20613, 20614]` | pc_20612 | settled `approve`, by_person |
| 3 | `[20620, 20621]` | pc_20619 | settled `approve`, by_person |
| 4 | `[20627, 20628]` | pc_20626 | settled `approve`, by_person; **still no policy case** |
| 5 | `[20634, 20635]` | pc_20633 | settled `approve`; **policy case 20639 raised** |

Case 20639: kind `policy`, raised_by `repeat`, open, `facts.proposal {"from":"repeat","count":5,"outcome":"approve"}`,
options `keep_both:OPS-0012`, `keep_both:OPS-0013`, `change:…`, `limit:…`, `retire:…` (both keep_both options:
no block is involved), evidence `overlap example {"tool.name":"get_policy","args.query":"DEMO-S4V-REP"}`.
Owner notifications 14574/14575. Five replay rows ran, one per run. On screen the Conflicts card read
"Decided the same way 5 times: make it a standing rule?". (It was closed as `moot` by the retire at clean-up.)

## Step 8: a live member case times out. PASS

A third call `get_policy {query:"DEMO-S4V-LIVE"}` (workflow `demo-s4-wf-8`) paused with `requestIds [20642, 20643]`,
`conflictCaseId pc_20641`. This case was used for the screens first (step 11). Then:

1. `POST /agent-policies/approvals/20643/decide {verb:"approve"}` (the Ops side) → 201, `group {open:1, approved:1, rejected:0, total:2}`.
2. `UPDATE proc.bp_decision SET respond_by = now() - interval '1 minute' WHERE decision_id = 20642 AND status = 'open'` → 1 row (bp_testdb).
3. `approvals.sweep(conn, now, decision_ids=[20642])` → `{escalated:0, rejected:1, skipped:0, errors:0}` (limited to my case: no other session's case was touched).

Result:
- 20647: `reject` by **`system:timeout`**, "No decision in time; a timeout never approves" (member 20642).
- **The live case settled `reject`**: action row 20648 `reject` by `system:timeout`; conflict 20641 outcome `reject`, by_person **false** (so it does not count towards repeat-5, ruling Q4), even though the other side had approved.
- Firing 4340 (OPS-0010) `timed_out`; 4341 (OPS-0011) `approved` (the firing records the person's decision; stage 3 ruling 10).
- **No replay row**: the action never ran.
- Notifications 14582 (Demo S4 Finance Approver) and 14583 (the requester): "Looking up a governed policy (policy OPS-0010) was rejected: nobody decided in time, and a timeout never approves."

## Step 10: `/decisions` never shows the new subject types. PASS

Against :8010 (`ASK_AUTH_MODE=off` in this `.env`), while the repeat proposal 20639 was open:

```
GET /decisions?subject_type=policy_conflict -> total 0 rows 0
GET /decisions?subject_type=live_conflict -> total 0 rows 0
GET /decisions?subject_type=policy_conflict&status=actioned -> total 0 rows 0
GET /decisions?subject_type=live_conflict&status=actioned -> total 0 rows 0
GET /decisions?limit=500 -> total 51 rows 51 | conflict rows: 0 | types: ['authorization', 'email_reply', 'finding']
GET /decisions?status=actioned&limit=500 -> total 3 rows 3 | conflict rows: 0 | types: ['email_reply', 'finding']
GET /decisions/1356 (control, an ordinary 'authorization' row) -> 200
GET /decisions/{20593, 20594, 20597, 20602, 20604, 20639, 20641, 20644} -> 404 each ("decision N not found")
```

**Red/green.** In a separate process only (no file changed), the router's hidden list was set back to stage 3's
two names, then restored:

```
RED (stage 3 hidden list, in this process only): list policy_conflict total=1 ids=[20639] ; by-id 20639 -> 200
GREEN (code as committed): list policy_conflict total=0 ids=[] ; by-id 20639 -> 404
```

## Step 11: screens in headless Chrome. PASS

Loaded `/spendiq` (re-navigating past the first-screen redirect), `go('workspace')`, `wfSetAgentTab('policy')`,
then **clicked the workspace "Policies" button** (`.wf-panel-create`, which opens the agent-policy screen).
Segments: Agent policies | System settings | List | Inventory | Documents | **Approvals 2** | **Conflicts 2** | Notifications | Deciders.

1. **List and Inventory:** the "Conflict sent for decision" tag on OPS-0012/0013/0014/0015 and on no other row (step 2).
2. **Conflicts tab** (badge 2): card pc_20644 in brief §4.5 order — the example sentence "Example action that triggers both policies: tool.name get_policy, args.query DEMO-S4V-GH" and its witness table; each policy (id · version, situation, "Needs approval: a person decides · Owner: Demo S4 Finance Owner", the excerpt in a blockquote, "Section 8.1 of Demo Finance"); "Why they conflict"; "Previous decisions: First time"; the seven options; "No deadline"; Reason (required); Decide…. Card pc_20639 also showed the proposal banner.
   - Decide… was disabled until an option and a reason were given, then enabled.
   - Chose "Keep both: OPS-0015 takes priority", typed a reason, clicked **Decide…**: confirm 1 "Decide this conflict? / Your decision and your reason are recorded against both policies, and both owners are told. / Cancel / Yes". A second Decide click while it was open left **1** overlay.
   - Yes → confirm 2 "Confirm: Keep both: OPS-0015 takes priority? / Go back / Decide". **No `/decide` call had been sent yet.**
   - Decide → one `POST /agent-policies/conflicts/20644/decide` 201, then `GET /agent-policies/conflicts/20644` 200 and `GET /agent-policies` 200. The card read "· Decided: Keep both: OPS-0015 takes priority … You decided: Keep both: OPS-0015 takes priority." Rule 233 (`OPS-0015` prevails) was written.
3. **Policy form section** (OPS-0008): as in step 5 (pending change, Open as a draft, Cancel, no write).
4. **Approval card conflict block** (Approvals, member cases 20642/20643 of pc_20641): "Policies in conflict / pc_20641 · More than one policy matched this action and they say different things. / Looking up a governed policy / tool.name get_policy / args.query DEMO-S4V-LIVE / OPS-0010 · Version 2 … Section 6.1 of Demo Finance / OPS-0011 · Version 2 … Section 6.2 of Demo Ops / Why they conflict: The policies name different approvers: Demo S4 Finance Approver and Demo S4 Ops Approver. / Previous decisions: Decided 1 time before; last: approved / Options: Approve, Reject. Respond within 10 minutes." followed by the stage 3 card (What the agent wants to do, the agent's reason, the policy).
5. **Notification → conflict:** Notifications, Open on 14579 ("Policies OPS-0014 and OPS-0015 conflict; a decision is needed."): `POST /agent-policies/notifications/14579/read` 201, **then** `GET /agent-policies/conflicts/20644` 200; the view switched to Conflicts with that case.

Screenshots are in the scratchpad (`s4demo/shots/01…10*.png`, not committed).

**Console, over every pass that recorded it (5 browser sessions; two short inspection-only passes did not record):**
- **0 `console.error` calls and 0 uncaught exceptions.** Every `/agent-policies` call returned 2xx (approvals, conflicts, conflicts/{id}, conflicts/{id}/decide, notifications/{id}/read, the list, taxonomy, {key}, preview).
- Network noise not from this feature, logged by the browser (not by page code): the same `GET :3011/users/me/preferences` 404 as stages 1–3, and a CORS refusal of `GET :3011/user?page=1&limit=100` (the bypass gateway answers it with `Access-Control-Allow-Origin: null`).

**Not exercised on screen:** Retire from a conflict decision (two-step retire; the retires at clean-up went through the API), a limit decision with its text, a non-eligible viewer's card (the bypass identity is linked to every demo name).

## Clean-up. DONE

- `POST /agent-policies/OPS-00nn/retire` for all 12 demo policies (OPS-0004 … OPS-0015) → 201 each. All are
  `retired`, live_version NULL. The open repeat proposal 20639 closed as `moot` by `system:retired` (close_moot).
  No demo case is open (0 open `policy_conflict` / `live_conflict` / `agent_policy_approval` rows for them).
- The orchestrator feed then served `policies: []`, and `live_policies.load()` returned `[]`.
- `DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(<my 4 names>)` → 4 rows. The map now holds
  only stage 3's three rows.
- `live_conflict_repeat` = 5, unchanged throughout.
- **Left in `bp_testdb` as the record** (ids are never deleted): the 12 retired policies and their versions; conflict
  rows 20593–20644 (12) and their decision/action rows; standing rules 232 and 233 (still in force, naming retired
  policies; the known Task 5 minor); firing rows (append-only by design); the notifications; the audit rows.
- Processes: only my four PIDs were stopped. The scratch worktrees used for the baseline suites (`base-bp`,
  `base-gw`, `base-ui`, detached, in the scratchpad) are not part of any branch.

## Defects found (recorded, not patched)

**D1. A policy's conflict history shows a settled live conflict as "approve_or_reject" with no decider.**
- Where: `conflict_cases.history_for`, which feeds `GET /agent-policies/{key}` `conflicts[]` and the form's
  history lines.
- Evidence (after steps 6 and 8):
  ```
  GET /agent-policies/OPS-0010 conflicts[]:
    {"caseId": "pc_20641", "kind": "live", "isOpen": false, …, "decision": {"caseId": "pc_20641", "decision": "approve_or_reject", "scope": null, "decidedBy": null, "decidedAt": null, "reason": null}}
    {"caseId": "pc_20597", "kind": "live", "isOpen": false, …, "decision": {"caseId": "pc_20597", "decision": "approve_or_reject", "scope": null, "decidedBy": null, "decidedAt": null, "reason": null}}
  ```
  Expected: pc_20597 → `approve` by the bypass sub (action row 20602); pc_20641 → `reject` by `system:timeout`
  (action row 20648). The block record pc_20604 and every policy case show correctly.
- Cause: the action rows are found by `facts->>'caseId'` with `status = 'actioned'`, ordered `actioned_at,
  decision_id`, and "the latest wins". A live case's own row also carries `facts.caseId` (insert_live adds it)
  and becomes `actioned` when settled, but its `actioned_at` stays NULL. Postgres sorts NULL last in ascending
  order, so the original row overwrites the real action:
  ```
  decision_id | decision          | status   | actioned_by     | actioned_at                | caseId
  20597       | approve_or_reject | actioned | NULL            | NULL                       | pc_20597
  20641       | approve_or_reject | actioned | NULL            | NULL                       | pc_20641
  20602       | approve           | actioned | c2b5b404-…      | 2026-10-09 13:37:20.468570 | pc_20597
  20648       | reject            | actioned | system:timeout  | 2026-10-09 13:46:34.752548 | pc_20641
  ```
- Impact: display only. Enforcement, settling and repeat-5 read `bp_agent_policy_conflict.outcome`, which is
  right (`approve` / `reject`).
- Fix to decide: exclude the case's own row (`decision_id <> case id`), or order `actioned_at NULLS FIRST`.

## Observations (not defects)

- O1. The repeat proposal's owner notification reads exactly like a design-time one ("Policies OPS-0012 and
  OPS-0013 conflict; a decision is needed."). Only the screen says it is a 5-times proposal.
- O2. A decide with no `reason` key gets the gateway's 400 "reason must be text of at most 2000 characters",
  not the backend's 422 `reason_required` (the Task 9 deferred minor). The UI never sends one: Decide… stays
  disabled without a reason.
- O3. In step 8 the Ops approver who said yes is not told that the action was rejected after the other side
  timed out; only the timed-out level and the requester are (stage 3's O5 pattern).
- O4. The live member cases' approval detail answered `group: null` on `GET /approvals/{id}`; the decide
  answer carries the group counts. Not needed by any screen used here.
- O5. The 54 invalid `GEN-*` "live" rows in bp_testdb are refused by the feed on every read (stage 3 O1 counted 33).

## Test suites, against baseline

Baseline = the stage 4 base commits, run in detached scratch worktrees (BP `5e0edfcf` = origin/Development
after stages 1–3, gateway `45d4d8c`, UI `faf1f2b`), the same day, same interpreter (`venv/bin/python`) and `.env`.

| Repo | Suite | Baseline | HEAD |
|---|---|---|---|
| BP | `tests/agent_policy`, fake DB | 434 passed, 128 skipped | 544 passed, 180 skipped |
| BP | `tests/agent_policy`, live (`PROCWISE_TEST_LIVE_DB=1`, bp_testdb) | 557 passed, 5 skipped | 719 passed, 5 skipped |
| BP | `tests/engines` + `tests/approvals` | 303 passed, 1 skipped | 303 passed, 1 skipped |
| BP | decisions router tests (`tests/api/test_decisions_email_endpoints.py`, `tests/approvals/test_decisions_authenticated.py`, `tests/api/test_card_action_gates.py`, `tests/engines/test_email_decision_action.py`) | 62 passed | 62 passed |
| BP | `tests/migrations/test_2026_10_12_bp_agent_policy_conflicts.py`, live | n/a (new) | 16 passed |
| BP | `scripts/p8_endpoint_scan.py` | – | 166 write endpoints, 0 without an identity |
| Gateway | `npx jest` (all) | 16 suites, 323 passed | 16 suites, 356 passed |
| UI | `npx vitest run src/modules/SpendIQ` | 1 file failed / 100 passed; 4 failed, 2395 passed | 1 file failed / 102 passed; 4 failed, 2428 passed |

No new failures anywhere. The only failures are the 4 known ones in `atb/composedPages.contract.test.js`,
identical at base and HEAD.

## Red/green captures

- This run: `/decisions` hiding (step 10 above), broken on purpose in-process and restored.
- Per task (in each task's report; summarised here):

| Task | Guard broken on purpose | Red | Restored |
|---|---|---|---|
| 1 | open-pair partial unique index dropped inside a rolled-back transaction, both DBs | "second open policy row refused with index dropped: False" | index present after rollback |
| 2 | witness search's tool-list check disabled | 1 failed, 8 passed (`test_no_witness_when_tool_lists_are_disjoint`) | green |
| 3 | `onTimeout` removed from `build()` | "Extra items in the right set: 'onTimeout'" | green |
| 4 | module moved aside / not wired | collection errors; 4 router/extraction wiring tests failed | green |
| 6 | old `len(losers)==n-1` auto rule restored | the winner tests failed | green |
| 8 | masking removed / Q5 pairs removed / 2xx-only check removed / hidden types reverted / routes after `/{key}` / authorize audit downgraded | 6 / 6 / 2 / 2 / 11 / 1 failed | green |
| 9 | unknown-key check and option regex dropped | 8 failed, 107 passed | 115/115 |
| 10 | 11 UI guards (m3 ticket, masking, draft writes nothing, limit text, escaping, tag, guarded-function hashes, two-step retire/decide) | 1–5 tests red each | 225/225 |

## Diff summary (ruling A)

### BP_Backend: `5e0edfcf..e9f2134b` (15 commits; 39 files, +6150 / −59; nothing under `.superpowers/`)

```
e9f2134b feat(agent-policy): conflict endpoints; conflict cases hidden from /decisions
7c3e78eb fix(agent-policy): a live conflict settles as the member decision that decided it
2b3c86c8 feat(agent-policy): live conflicts pause the action and go to the decision engine
78ac566f fix(agent-policy): auto conflict needs one winner that beats every other involved policy
ea859f95 feat(agent-policy): classify a multi-policy match as a live conflict
18b78068 fix(agent-policy): a retired policy's conflict case can never stay open or be decided
91965a29 feat(agent-policy): owners decide policy conflicts; standing rules reach conflicts[]
24c03700 fix(agent-policy): conflict scan on its own lane, reads once, capped per run
660ee381 feat(agent-policy): policy conflicts raised on save, extraction and hourly scan
21bc9e1d feat(agent-policy): policy-conflict/1 payload and approver summary
3e558d40 fix(agent-policy): a candidate the evaluator cannot compare is no match, not an error
b9e7a65f feat(agent-policy): code-found witness for policy conflicts
54206b57 feat(agent-policy): conflict case and standing rule tables
fb2361d0 fix(agent-policy): stage 3 deferred minors — reserved decider name, lock after permission, no-runtime gives up loudly
25d6b0aa docs(agent-policy): stage 4 (conflicts) implementation plan
```

| Area | Files (lines added / removed) |
|---|---|
| Schema | `deploy/sql/2026-10-12_bp_agent_policy_conflicts.sql` (+53) and its rollback (+9) |
| New services | `conflict_cases.py` (+736), `conflict_live.py` (+348), `conflict_views.py` (+153), `conflict_detect.py` (+148), `conflict_payload.py` (+144), `conflict_engine.py` (+77) |
| Changed services | `approval_views.py` (+95/−8), `gate.py` (+82/−7), `replay.py` (+57/−8), `approvals.py` (+40/−18), `conditions.py` (+8/−5), `extraction_run.py` (+3/−1), `settings.py` (+1), `repositories/agent_policy_repo.py` (+15/−5), `backend_scheduler.py` (+41: the hourly conflict scan, `AGENT_POLICY_CONFLICT_SCAN`) |
| Routers | `routers/agent_policies.py` (+98/−6: conflict list/detail/decide, detection after create/save, moot close after retire), `routers/decisions.py` (+3/−1: two hidden subject types), `api/main.py` (+5: the Q5 exemption for exactly two GET paths) |
| Tests | `tests/agent_policy/test_conflict_*.py` (11 files), `conftest.py` (+23: after_save off unless marked), small additions to `test_approvals_job.py`, `test_approvals_live.py`, `test_decisions_hide_approvals_live.py`, `test_final_wave*.py`, and the migration test (+201) |
| Plan | `specs/2026-10-09-agent-policy-governance-plan-4-conflicts.md` (+838) |

### Gateway: `45d4d8c..e7d9c99` (1 commit; 3 files, +214)

```
e7d9c99 feat(agent-policy): conflict routes
```

`agent-policy.controller.ts` (+32: GET conflicts, GET conflicts/:decisionId, POST conflicts/:decisionId/decide),
`agent-policy.yml` (+75), `agent-policy.controller.spec.ts` (+107).

### UI: `faf1f2b..b943567` (1 commit; 5 files, +987 / −15)

```
b943567 feat(agent-policy): conflicts tab, conflict tag, decision-to-draft
```

| File | Lines |
|---|---|
| `agentPolicy/conflicts.js` (new, pure) | +174 |
| `agentPolicy/conflicts.test.js` | +95 |
| `agentPolicy/engineWiring.stage4.contract.test.js` | +422 |
| `engine.js` | +294 / −14 |
| `index.jsx` | +2 / −1 (`conflicts` on `window.__SPENDIQ_AP__`) |

New `engine.js` functions: apPcLib, apPcUi, apPcFind, apPcOpenCount, apPcLoad, apPcBind, apPcSelect, apPcReason,
apPcLimit, apPcReady, apPcLabel, apPcCardHTML, apPcPoliciesHTML, apPcHTML, apPcDecide, apPcSend, apPcLoadDetail,
apPcOpen, apPcTag, apPcPolicySection, apPcOpenDraft, apPcRetire, apPcApprovalBlock, apPcNoteTarget. Changed
existing `ap*` functions (minimal wiring): apPoliciesTab (Conflicts view and badge), apListHTML and
apInventoryHTML (one `apPcTag(p)` each), apFormHTML (`apPcPolicySection`), apApprovalCardHTML
(`apPcApprovalBlock`), apNotificationOpen (the `conflict` target), and apApprovalDecide / apApprovalLoadDetail
(the m3 fresh re-read).

**Guarded functions are byte-identical to base `faf1f2b`** (each body extracted at both commits, sha256 first 16 hex):

| Function | Bytes | Hash at base and at HEAD |
|---|---|---|
| `policyEdit` | 3287 | `46bfb5ad40f48960` |
| `policyDelete` | 848 | `9e650ea2eb32051e` |
| `openFormModal` | 4854 | `ecf76d1a16845451` |

(The byte counts differ from stage 3's table because this extraction runs to the next top-level declaration; the
point is base = HEAD.) No removed `engine.js` line mentions any of the three names.

## Appendix: every gateway request and response in this run

Recorded by the driver as it ran (abridged where long; no key was ever sent by the driver or logged). The
feed and `/decisions` reads went straight to :8010 and are quoted in steps 4 and 10.

| # | Step | Request | Status | Response (abridged) |
|---|---|---|---|---|
| 1 | pre | `POST /agent-policies/preview` form "DEMO stage 4 A approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-AB) | 201 | {"examples": [{"input": {"tool.name": "get_policy", "args.query": "DEMO-S4V-AB"}, "computed": "approve", "label": "A person decides", "flipped": false, "reviewer_expects": "approve", "agent_expected": "approve"}, {"input": {"tool.… |
| 2 | 1 | `PUT /agent-policies/deciders/Demo%20S4%20Finance%20Owner` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Stage 4 demo (2026-10-09)"} | 200 | {"name": "Demo S4 Finance Owner", "savedAt": "2026-10-09T13:34:32.952861+00:00"} |
| 3 | 1 | `PUT /agent-policies/deciders/Demo%20S4%20Customer%20Owner` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Stage 4 demo (2026-10-09)"} | 200 | {"name": "Demo S4 Customer Owner", "savedAt": "2026-10-09T13:34:33.409976+00:00"} |
| 4 | 1 | `PUT /agent-policies/deciders/Demo%20S4%20Finance%20Approver` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Stage 4 demo (2026-10-09)"} | 200 | {"name": "Demo S4 Finance Approver", "savedAt": "2026-10-09T13:34:33.755824+00:00"} |
| 5 | 1 | `PUT /agent-policies/deciders/Demo%20S4%20Ops%20Approver` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Stage 4 demo (2026-10-09)"} | 200 | {"name": "Demo S4 Ops Approver", "savedAt": "2026-10-09T13:34:34.099226+00:00"} |
| 6 | 1 | `GET /agent-policies/deciders`  | 200 | {"deciders": [{"name": "Demo S4 Customer Owner", "groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Stage 4 demo (2026-10-09)", "lastModifiedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "lastModifiedAt": "20… |
| 7 | 2 | `POST /agent-policies` form "DEMO stage 4 A approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-AB) | 201 | {"policyKey": "OPS-0004", "version": 1} |
| 8 | 2 | `POST /agent-policies` form "DEMO stage 4 B block (Demo Customer)" (block, doc Demo Customer, query DEMO-S4V-AB) | 201 | {"policyKey": "OPS-0005", "version": 1} |
| 9 | 2 | `GET /agent-policies/conflicts`  | 200 | {"conflicts": [{"caseId": "pc_20593", "decisionId": 20593, "status": "open", "raisedAt": "2026-10-09T13:34:41.409316+00:00", "raisedBy": "save", "policies": [{"id": "OPS-0004", "owner": "Demo S4 Finance Owner", "source": {"excerpt… |
| 10 | 2 | `GET /agent-policies/conflicts/20593`  | 200 | {"caseId": "pc_20593", "decisionId": 20593, "status": "open", "raisedAt": "2026-10-09T13:34:41.409316+00:00", "raisedBy": "save", "policies": [{"id": "OPS-0004", "owner": "Demo S4 Finance Owner", "source": {"excerpt": "A lookup of… |
| 11 | 2 | `GET /agent-policies`  | 200 | {"policies": "[5550 rows; demo rows quoted in the step]"} |
| 12 | 2 | `GET /agent-policies/OPS-0004`  | 200 | {"policyKey": "OPS-0004", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20593", "kind": "policy", "isOpen": true, "otherPolicies": ["OPS-0005"], "raisedAt": "2026-10-09T13:34:41.409316+00… |
| 13 | 2 | `GET /agent-policies/OPS-0005`  | 200 | {"policyKey": "OPS-0005", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20593", "kind": "policy", "isOpen": true, "otherPolicies": ["OPS-0004"], "raisedAt": "2026-10-09T13:34:41.409316+00… |
| 14 | 3 | `POST /agent-policies` form "DEMO stage 4 T1 tiered approve (Demo Tiered)" (approve, doc Demo S4 Tiered, query ['DEMO-S4V-TIER-A', 'DEMO-S4V-TIER-B']) | 201 | {"policyKey": "OPS-0006", "version": 1} |
| 15 | 3 | `POST /agent-policies` form "DEMO stage 4 T2 tiered block (Demo Tiered)" (block, doc Demo S4 Tiered, query DEMO-S4V-TIER-B) | 201 | {"policyKey": "OPS-0007", "version": 1} |
| 16 | 3 | `GET /agent-policies/OPS-0006`  | 200 | {"policyKey": "OPS-0006", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 17 | 3 | `GET /agent-policies/OPS-0007`  | 200 | {"policyKey": "OPS-0007", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 18 | 4 | `POST /agent-policies/conflicts/20593/decide` {"option": "keep_both:OPS-0005"} | 400 | {"message": "reason must be text of at most 2000 characters", "error": "Bad Request", "statusCode": 400} |
| 19 | 4 | `POST /agent-policies/conflicts/20593/decide` {"option": "keep_both:OPS-0004", "reason": "x"} | 422 | {"problems": [{"field": "option", "code": "unknown_option", "message": "That is not one of this case's options."}]} |
| 20 | 4 | `POST /agent-policies/conflicts/20593/decide` {"option": "keep_both:OPS-0005", "reason": "Stage 4 demo: the customer block stands; the finance approval applies only where B does not."} | 201 | {"caseId": "pc_20593", "decision": "keep_both:OPS-0005", "scope": "standing_rule", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:35:32.719955+00:00", "reason": "Stage 4 demo: the customer block s… |
| 21 | 4 | `POST /agent-policies/conflicts/20593/decide` {"option": "keep_both:OPS-0005", "reason": "again"} | 409 | {"statusCode": 409, "message": "This conflict has already been decided."} |
| 22 | 4 | `GET /agent-policies/OPS-0004`  | 200 | {"policyKey": "OPS-0004", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20593", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0005"], "raisedAt": "2026-10-09T13:34:41.409316+0… |
| 23 | 4 | `POST /agent-policies/OPS-0004/versions` form "DEMO stage 4 A approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-AB) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0004", "version": 2} |
| 24 | 4 | `GET /agent-policies/OPS-0005`  | 200 | {"policyKey": "OPS-0005", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20593", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0004"], "raisedAt": "2026-10-09T13:34:41.409316+0… |
| 25 | 4 | `POST /agent-policies/OPS-0005/versions` form "DEMO stage 4 B block (Demo Customer)" (block, doc Demo Customer, query DEMO-S4V-AB) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0005", "version": 2} |
| 26 | 5 | `POST /agent-policies` form "DEMO stage 4 E approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-EF) | 201 | {"policyKey": "OPS-0008", "version": 1} |
| 27 | 5 | `POST /agent-policies` form "DEMO stage 4 F block (Demo Customer)" (block, doc Demo Customer, query DEMO-S4V-EF) | 201 | {"policyKey": "OPS-0009", "version": 1} |
| 28 | 5 | `GET /agent-policies/conflicts`  | 200 | {"conflicts": [{"caseId": "pc_20595", "decisionId": 20595, "status": "open", "raisedAt": "2026-10-09T13:36:05.076369+00:00", "raisedBy": "save", "policies": [{"id": "OPS-0008", "owner": "Demo S4 Finance Owner", "source": {"excerpt… |
| 29 | 5 | `POST /agent-policies/conflicts/20595/decide` {"option": "change:OPS-0008", "reason": "Stage 4 demo: narrow E so it no longer overlaps F."} | 201 | {"caseId": "pc_20595", "decision": "change:OPS-0008", "scope": "this_action", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:36:06.061964+00:00", "reason": "Stage 4 demo: narrow E so it no longer … |
| 30 | 5 | `GET /agent-policies/OPS-0008`  | 200 | {"policyKey": "OPS-0008", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20595", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0009"], "raisedAt": "2026-10-09T13:36:05.076369+0… |
| 31 | 6 | `POST /agent-policies` form "DEMO stage 4 L1 approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-LIVE) | 201 | {"policyKey": "OPS-0010", "version": 1} |
| 32 | 6 | `POST /agent-policies` form "DEMO stage 4 L2 approve (Demo Ops)" (approve, doc Demo Ops, query DEMO-S4V-LIVE) | 201 | {"policyKey": "OPS-0011", "version": 1} |
| 33 | 6 | `GET /agent-policies/OPS-0010`  | 200 | {"policyKey": "OPS-0010", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 34 | 6 | `POST /agent-policies/OPS-0010/versions` form "DEMO stage 4 L1 approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-LIVE) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0010", "version": 2} |
| 35 | 6 | `GET /agent-policies/OPS-0011`  | 200 | {"policyKey": "OPS-0011", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 36 | 6 | `POST /agent-policies/OPS-0011/versions` form "DEMO stage 4 L2 approve (Demo Ops)" (approve, doc Demo Ops, query DEMO-S4V-LIVE) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0011", "version": 2} |
| 37 | 6 | `GET /agent-policies/approvals?status=open`  | 200 | {"approvals": [{"id": 20598, "policyKey": "OPS-0010", "levelName": "Demo S4 Finance Approver", "levels": ["Demo S4 Finance Approver"], "canDecide": true}, {"id": 20599, "policyKey": "OPS-0011", "levelName": "Demo S4 Ops Approver",… |
| 38 | 6 | `GET /agent-policies/approvals/20598`  | 200 | {"id": 20598, "policyKey": "OPS-0010", "policyVersion": 2, "status": "open", "outcome": null, "actionPlain": "looking up a governed policy", "inputs": [{"field": "args.query", "name": "Query", "value": "DEMO-S4V-LIVE", "missing": … |
| 39 | 6 | `POST /agent-policies/approvals/20598/decide` {"verb": "approve", "reason": "Stage 4 demo: finance approves"} | 201 | {"decisionId": 20598, "actionId": 20600, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:37:19.691301+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 40 | 6 | `GET /agent-policies/approvals/20598`  | 200 | {"id": 20598, "policyKey": "OPS-0010", "policyVersion": 2, "status": "actioned", "outcome": "approved", "actionPlain": "looking up a governed policy", "inputs": [{"field": "args.query", "name": "Query", "value": "DEMO-S4V-LIVE", "… |
| 41 | 6 | `POST /agent-policies/approvals/20599/decide` {"verb": "approve", "reason": "Stage 4 demo: ops approves"} | 201 | {"decisionId": 20599, "actionId": 20601, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:37:20.468570+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 42 | 6 | `GET /agent-policies/approvals/20599`  | 200 | {"id": 20599, "policyKey": "OPS-0011", "policyVersion": 2, "status": "actioned", "outcome": "approved", "actionPlain": "looking up a governed policy", "inputs": [{"field": "args.query", "name": "Query", "value": "DEMO-S4V-LIVE", "… |
| 43 | 9 | `POST /agent-policies` form "DEMO stage 4 R1 approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-REP) | 201 | {"policyKey": "OPS-0012", "version": 1} |
| 44 | 9 | `POST /agent-policies` form "DEMO stage 4 R2 approve (Demo Ops)" (approve, doc Demo Ops, query DEMO-S4V-REP) | 201 | {"policyKey": "OPS-0013", "version": 1} |
| 45 | 9 | `GET /agent-policies/OPS-0012`  | 200 | {"policyKey": "OPS-0012", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 46 | 9 | `POST /agent-policies/OPS-0012/versions` form "DEMO stage 4 R1 approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-REP) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0012", "version": 2} |
| 47 | 9 | `GET /agent-policies/OPS-0013`  | 200 | {"policyKey": "OPS-0013", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 48 | 9 | `POST /agent-policies/OPS-0013/versions` form "DEMO stage 4 R2 approve (Demo Ops)" (approve, doc Demo Ops, query DEMO-S4V-REP) {"baseVersion": 1, "intent": "activate", "changeNote": "Stage 4 demo: activate"} | 201 | {"policyKey": "OPS-0013", "version": 2} |
| 49 | 9 | `POST /agent-policies/approvals/20606/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 1"} | 201 | {"decisionId": 20606, "actionId": 20608, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:01.875687+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 50 | 9 | `POST /agent-policies/approvals/20607/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 1"} | 201 | {"decisionId": 20607, "actionId": 20609, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:02.349407+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 51 | 9 | `POST /agent-policies/approvals/20613/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 2"} | 201 | {"decisionId": 20613, "actionId": 20615, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:04.231661+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 52 | 9 | `POST /agent-policies/approvals/20614/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 2"} | 201 | {"decisionId": 20614, "actionId": 20616, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:04.699242+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 53 | 9 | `POST /agent-policies/approvals/20620/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 3"} | 201 | {"decisionId": 20620, "actionId": 20622, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:06.618953+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 54 | 9 | `POST /agent-policies/approvals/20621/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 3"} | 201 | {"decisionId": 20621, "actionId": 20623, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:07.093327+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 55 | 9 | `POST /agent-policies/approvals/20627/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 4"} | 201 | {"decisionId": 20627, "actionId": 20629, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:09.005384+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 56 | 9 | `POST /agent-policies/approvals/20628/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 4"} | 201 | {"decisionId": 20628, "actionId": 20630, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:09.676057+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 57 | 9 | `POST /agent-policies/approvals/20634/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 5"} | 201 | {"decisionId": 20634, "actionId": 20636, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:21.259731+00:00", "level": 0, "levelName": "Demo S4 Finance Appr… |
| 58 | 9 | `POST /agent-policies/approvals/20635/decide` {"verb": "approve", "reason": "Stage 4 demo repeat 5"} | 201 | {"decisionId": 20635, "actionId": 20637, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:38:21.736744+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 59 | 2b | `POST /agent-policies` form "DEMO stage 4 G approve (Demo Finance)" (approve, doc Demo Finance, query DEMO-S4V-GH) | 201 | {"policyKey": "OPS-0014", "version": 1} |
| 60 | 2b | `POST /agent-policies` form "DEMO stage 4 H block (Demo Customer)" (block, doc Demo Customer, query DEMO-S4V-GH) | 201 | {"policyKey": "OPS-0015", "version": 1} |
| 61 | 2b | `GET /agent-policies`  | 200 | {"policies": "[5560 rows; demo rows quoted in the step]"} |
| 62 | 5-after-ui | `GET /agent-policies/OPS-0008`  | 200 | {"policyKey": "OPS-0008", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20595", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0009"], "raisedAt": "2026-10-09T13:36:05.076369+0… |
| 63 | 8 | `POST /agent-policies/approvals/20643/decide` {"verb": "approve", "reason": "Stage 4 demo: ops approves; finance will time out"} | 201 | {"decisionId": 20643, "actionId": 20646, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T13:46:34.212493+00:00", "level": 0, "levelName": "Demo S4 Ops Approver… |
| 64 | cleanup | `GET /agent-policies/OPS-0004`  | 200 | {"policyKey": "OPS-0004", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20604", "kind": "live", "isOpen": false, "otherPolicies": ["OPS-0005"], "raisedAt": "2026-10-09T13:37:37.994573+00:00",… |
| 65 | cleanup | `POST /agent-policies/OPS-0004/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0004", "version": 3} |
| 66 | cleanup | `GET /agent-policies/OPS-0005`  | 200 | {"policyKey": "OPS-0005", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20604", "kind": "live", "isOpen": false, "otherPolicies": ["OPS-0004"], "raisedAt": "2026-10-09T13:37:37.994573+00:00",… |
| 67 | cleanup | `POST /agent-policies/OPS-0005/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0005", "version": 3} |
| 68 | cleanup | `GET /agent-policies/OPS-0006`  | 200 | {"policyKey": "OPS-0006", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 69 | cleanup | `POST /agent-policies/OPS-0006/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0006", "version": 2} |
| 70 | cleanup | `GET /agent-policies/OPS-0007`  | 200 | {"policyKey": "OPS-0007", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [], "pendingConflictAction": null, "versions": "(omitted)"} |
| 71 | cleanup | `POST /agent-policies/OPS-0007/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0007", "version": 2} |
| 72 | cleanup | `GET /agent-policies/OPS-0008`  | 200 | {"policyKey": "OPS-0008", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20595", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0009"], "raisedAt": "2026-10-09T13:36:05.076369+0… |
| 73 | cleanup | `POST /agent-policies/OPS-0008/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0008", "version": 2} |
| 74 | cleanup | `GET /agent-policies/OPS-0009`  | 200 | {"policyKey": "OPS-0009", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20595", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0008"], "raisedAt": "2026-10-09T13:36:05.076369+0… |
| 75 | cleanup | `POST /agent-policies/OPS-0009/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0009", "version": 2} |
| 76 | cleanup | `GET /agent-policies/OPS-0010`  | 200 | {"policyKey": "OPS-0010", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20641", "kind": "live", "isOpen": false, "otherPolicies": ["OPS-0011"], "raisedAt": "2026-10-09T13:38:38.446478+00:00",… |
| 77 | cleanup | `POST /agent-policies/OPS-0010/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0010", "version": 3} |
| 78 | cleanup | `GET /agent-policies/OPS-0011`  | 200 | {"policyKey": "OPS-0011", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20641", "kind": "live", "isOpen": false, "otherPolicies": ["OPS-0010"], "raisedAt": "2026-10-09T13:38:38.446478+00:00",… |
| 79 | cleanup | `POST /agent-policies/OPS-0011/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0011", "version": 3} |
| 80 | cleanup | `GET /agent-policies/OPS-0012`  | 200 | {"policyKey": "OPS-0012", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20639", "kind": "policy", "isOpen": true, "otherPolicies": ["OPS-0013"], "raisedAt": "2026-10-09T13:38:21.751669+00:00"… |
| 81 | cleanup | `POST /agent-policies/OPS-0012/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0012", "version": 3} |
| 82 | cleanup | `GET /agent-policies/OPS-0013`  | 200 | {"policyKey": "OPS-0013", "status": "live", "liveVersion": 2, "latestVersion": 2, "conflicts": [{"caseId": "pc_20639", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0012"], "raisedAt": "2026-10-09T13:38:21.751669+00:00… |
| 83 | cleanup | `POST /agent-policies/OPS-0013/retire` {"baseVersion": 2, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0013", "version": 3} |
| 84 | cleanup | `GET /agent-policies/OPS-0014`  | 200 | {"policyKey": "OPS-0014", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20644", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0015"], "raisedAt": "2026-10-09T13:41:02.229709+0… |
| 85 | cleanup | `POST /agent-policies/OPS-0014/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0014", "version": 2} |
| 86 | cleanup | `GET /agent-policies/OPS-0015`  | 200 | {"policyKey": "OPS-0015", "status": "draft", "liveVersion": null, "latestVersion": 1, "conflicts": [{"caseId": "pc_20644", "kind": "policy", "isOpen": false, "otherPolicies": ["OPS-0014"], "raisedAt": "2026-10-09T13:41:02.229709+0… |
| 87 | cleanup | `POST /agent-policies/OPS-0015/retire` {"baseVersion": 1, "changeNote": "Stage 4 demo finished: retired"} | 201 | {"policyKey": "OPS-0015", "version": 2} |
