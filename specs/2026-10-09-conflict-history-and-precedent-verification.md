# Conflict history and precedent: live verification and diff summary

Date: 2026-10-09, 20:27 to 20:56 UTC. Branch `agent-policy-stage4` in three worktrees (BP `4d6cf48d`, gateway
`0723d05`, UI `ef5583b`). Database: `bp_testdb` (checked with `select current_database()` before every direct
write). Nothing has been merged or pushed. Design: `specs/2026-10-09-conflict-history-and-precedent-design.md`;
plan: `specs/2026-10-09-conflict-history-and-precedent-plan.md` (Task 11).

## In plain English

- **After five people approved the same clash, the sixth identical action ran on its own.** Two approval policies
  (A and B) named different approvers for the same lookup, so each call paused and asked both sides. People
  approved it five times. The sixth identical call ran straight away. It was recorded as decided "on precedent"
  and cited the five earlier cases by number.
- **The engine said why it was not deciding yet.** Calls 1 to 5 each recorded "only 0 of 5 … only 4 of 5 decisions
  by people on this exact clash". The number 5 came from the governed policy row, which was not changed.
- **After the fifth decision, the owners were asked to make it a standing rule** (case pc_40622, "Decided the same
  way 5 times").
- **The owners and approvers were told,** in four notices that hold no input values.
- **The history can be read and exported.** It appears on the policy page, in the Conflicts screen and on the
  approval card, and two CSV exports carry it.
  - A person linked to the policies sees everything.
  - A Viewer who is not linked sees the same rows, but the query value is replaced by `•••`, also inside a
    reason that quoted it. The reason is masked, never dropped.
- **Screens:** headless Chrome showed:
  - the policy form's history ("by a person" and "on precedent"), with Export pressed (a file was saved);
  - a Conflicts case with "History of this clash";
  - an approval card with "Earlier decisions on this clash".

  There were 0 console errors and 0 uncaught exceptions.
- **The generic Decisions screen shows none of these cases.**
- **Three faults were found (D1 to D3)**, all about what is shown, not about what is enforced:
  - D1: the agent is not told that its action ran on precedent;
  - D2: an open case on the Conflicts screen shows its history only after someone decides it;
  - D3: the 5-times proposal still says "Previous decisions: First time".
- **Clean-up.**
  - All four demo policies are retired.
  - My four decider-map rows are deleted.
  - The governed row still reads `{"precedent_count": 5}`.
  - No company setting was changed.
  - Only my own processes were stopped.

## How the stack was run

| Part | Port | How |
|---|---|---|
| Backend | 127.0.0.1:8010 | `/home/muthu/PycharmProjects/BP_Backend/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8010 --workers 1` from the BP worktree, `PYTHONPATH=<worktree>:<worktree>/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`, `AGENT_POLICY_ENFORCEMENT=on`, `AGENT_POLICY_APPROVAL_SWEEP=off`, **`AGENT_POLICY_CONFLICT_SCAN=off`**, `OLLAMA_BASE_URL=OLLAMA_HOST=http://127.0.0.1:9` (dead port), `OLLAMA_CLOUD_*` unset. The only model traffic in its log is the start-up preload, refused by the dead port. PID/PGID 1187727. |
| Gateway | :3011 | `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development BP_BACKEND_URL=http://127.0.0.1:8010`. The gateway key was read from the BP `.env` and never printed. `AUTH_BYPASS_GROUPS` was `PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER` (PGID 1187728, later 1213613), and it was restarted twice on the same port for step 5: once as `PROCWISE_FINANCE_REVIEWER_APPROVER` only (linked, role Approver) and once as `PROCWISE_VIEWER` only (unlinked, role Viewer). The bypass sub is `c2b5b404-40c1-7047-71c4-dd8093ecf25d`; `proc.bp_role_assignment` is empty, so the groups alone set the role. |
| UI | 127.0.0.1:3010 | Vite from the UI worktree, `VITE_API_URL=http://127.0.0.1:3011 VITE_DEV_AUTH_BYPASS=true`. The untracked `mockPipelineDeals.js` was copied in for the run and deleted afterwards; the UI worktree is clean. PGID 1217170. |
| Headless Chrome | :9223 | `google-chrome --headless=new` driven over CDP (`/usr/share/nodejs/ws`), 1440×900, downloads allowed into a scratch folder. PGID 1217175. |
| Driver | none | Scratch Python (not committed) importing the worktree's modules with the same `.env`, dead Ollama port and `AGENT_POLICY_ENFORCEMENT=on`. `tool_runtime._chat` was replaced by a script, so **no model was called**. The tool was a stub `get_policy` that records each run. |

All gateway requests carried `x-customer-id: 001`. At the end the four process groups were stopped
(`kill -TERM -- -<pgid>`, my PGIDs only). Ports 8010, 3011, 3010 and 9223 were then confirmed closed.
`procwise.service` stayed `active` and was never touched, and neither were :8000, :3000, :3001 or Ollama.
(While switching the gateway's groups, one stop failed: my scratch script used `kill --`, which `/bin/sh` does
not accept. I stopped my own gateway group with `kill -KILL -<pgid>` instead. No other process was signalled.)

### Keeping the shared database safe

- **Before any write** (`out/prep.txt`):
  - `current_database` is `bp_testdb`;
  - the governed row reads `[{"precedent_count": 5}]`;
  - none of my decider names were in the map;
  - no policy carried my tag `9782f2`;
  - `live_policies.load()` returned **0 valid live policies**, so no other policy could match the demo calls;
  - both demo queries matched nothing.
- **The governed row was never edited.** N = 5 was demonstrated through the real row as it stands. It still read
  `{"precedent_count": 5}` at the end.
- **Unique demo scope.**
  - Every demo policy is `tool.name in ["get_policy"]` and `args.query eq "DEMO-PREC-9782f2"`. Pair C/D uses
    `"DEMO-PREC-9782f2-CD"`.
  - The policies are named "DEMO precedent A/B/C/D 9782f2".
  - The owners are "Demo Prec Finance Owner" and "Demo Prec Customer Owner". The deciders are "Demo Prec Finance
    Approver" and "Demo Prec Customer Approver".
  - The Query input is marked `sensitive`, so that masking has something to mask.
- **Dry run before saving** (`out/dryrun.txt`):
  - The check was read-only. It compiled each demo form and ran `conflict_cases._pairs` against all 6,384
    non-retired policy versions.
  - Each form had 1,298 candidate pairs, 0 errors and **0 witnessed pairs**. All four forms are "approve", and
    the design-time check compares outcomes, so even A/B is no design-time pair. The clash happens only at run
    time, by design.
  - As a result, no save raised a case against any policy. Every conflict row the run created names only
    OPS-0016…OPS-0019.

## The demo policies

| Key | Form | Outcome | Source document (section) | Owner | Decider | Query |
|---|---|---|---|---|---|---|
| OPS-0016 | A | approve, PT10M | Demo Finance (9.1) | Demo Prec Finance Owner | Demo Prec Finance Approver | DEMO-PREC-9782f2 |
| OPS-0017 | B | approve, PT10M | Demo Customer (9.2) | Demo Prec Customer Owner | Demo Prec Customer Approver | DEMO-PREC-9782f2 |
| OPS-0018 | C | approve, PT10M | Demo Finance (9.3) | Demo Prec Finance Owner | Demo Prec Finance Approver | DEMO-PREC-9782f2-CD |
| OPS-0019 | D | approve, PT10M | Demo Customer (9.4) | Demo Prec Customer Owner | Demo Prec Customer Approver | DEMO-PREC-9782f2-CD |

## Step 1: baselines and ports. PASS

The baselines ran in detached scratch worktrees at the base commits (BP `f2bab3ac`, gateway `e7d9c99`, UI
`8991c0b`), and HEAD ran in the three worktrees. Both used the same interpreter (`venv/bin/python`), the same
`.env` and the same day. They ran one after the other, before any demo write.

| Repo | Suite | Base | HEAD |
|---|---|---|---|
| BP | `PROCWISE_TEST_LIVE_DB=1 pytest tests/agent_policy tests/engines tests/approvals tests/governance tests/migrations -q` | 5 failed, 1227 passed, 5 skipped | 5 failed, **1340 passed**, 5 skipped |
| Gateway | `npx jest src/modules/agent-policy` | 1 suite, 115 passed | 1 suite, **128 passed** |
| UI | `npx vitest run src/modules/SpendIQ` | 1 file failed / 102 passed; 4 failed, 2429 passed | 1 file failed / 102 passed; 4 failed, **2443 passed** |

**No new failures.** The failures are the same at base and HEAD, and all were there before this work:

- BP:
  - `tests/agent_policy/test_approvals_job.py::test_conflict_scan_has_its_own_lane` (fails only in a full run;
    noted in Tasks 1-3 as pre-existing);
  - `tests/governance/test_governed_limits.py::test_every_governed_limit_is_present_in_the_live_policy_set` and
    `::test_the_in_memory_seed_matches_the_live_rows`: the Task 2 Step 1 baseline. Another session's
    `supplier_info_request` row in bp_testdb makes the counts differ.
  - `tests/migrations/test_2026_05_16_discrepancy_hitl.py::test_existing_rows_default_blocks_false`;
  - `tests/migrations/test_2026_09_27_bp_rule.py::test_every_seeded_rule_is_present_and_active[bp_testdb]`.
- UI: the 4 known `atb/composedPages.contract.test.js` failures.

Ports: `ss -ltn` showed 8010, 3011, 3010 and 9223 free before the start.

## Step 2: the stack. PASS

As in the table above. `GET /agent-policies/deciders` through :3011 answered 200 before the first write.

## Step 3: shared database safety. PASS

As in "Keeping the shared database safe" above.

## Step 4.1: owners, deciders, A and B. PASS

| # | Request | Response |
|---|---|---|
| 1-4 | `PUT /agent-policies/deciders/Demo%20Prec%20{Finance Owner, Customer Owner, Finance Approver, Customer Approver}` `{groups:["PROCWISE_FINANCE_REVIEWER_APPROVER"], emails:[], notes:"Precedent demo (2026-10-09)"}` | 200 `{name, savedAt}` each |
| 5 | `GET /agent-policies/deciders` | 200; the four rows with that group |
| 6 | `POST /agent-policies {form A}` | 201 `{"policyKey":"OPS-0016","version":1}` |
| 7 | `POST /agent-policies {form B}` | 201 `{"policyKey":"OPS-0017","version":1}`; `conflicts: []` on both (no design-time case, as expected) |
| 8-9 | `POST /agent-policies/OPS-0016/versions` and `/OPS-0017/versions` `{baseVersion:1, intent:"activate"}` | 201 `{version:2}` each |

`live_policies.load()` then held exactly `[('OPS-0016', 2), ('OPS-0017', 2)]`.

## Step 4.2: five identical clashes, each approved by people. PASS

Driver: `run_tools(…, agent="agent_nick", workflow_id="demo-prec-9782f2-wf-<n>",
user_id="demo-prec-9782f2-requester@example.test")`. The script makes one `get_policy {query:"DEMO-PREC-9782f2"}`
call and then answers "Done.". Both member cases were approved through the gateway, as the bypass identity,
which is linked to both deciders.

| Run | What the agent got | Live case | `facts.precedent.why` | `facts.history` entries | Member approvals | Conflict row after |
|---|---|---|---|---|---|---|
| 1 | `paused_for_approval`, requestIds [40589, 40590] | pc_40588 | only 0 of 5 decisions by people on this exact clash | 0 | 201 approved ×2 | closed, approve, by_person true, `{OPS-0016:2, OPS-0017:2}` |
| 2 | paused, [40596, 40597] | pc_40595 | only 1 of 5 … | 1 | 201 ×2 | the same |
| 3 | paused, [40603, 40604] | pc_40602 | only 2 of 5 … | 2 | 201 ×2, reason **"Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine"** | the same |
| 4 | paused, [40610, 40611] | pc_40609 | only 3 of 5 … | 3 | 201 ×2 | the same |
| 5 | paused, [40617, 40618] | pc_40616 | only 4 of 5 … | 4 | 201 ×2 | the same |

The handler did not run in any of the five (`handler_ran: []`). After approval, the backend replayed through the
real `get_policy`, as in stage 3. Full text: `out/runs1-5.txt` (scratch).

Run 1's full answer to the agent:
`{"result":"paused_for_approval","requestIds":[40589,40590],"respondWithin":"PT10M","whilePaused":"no_retry","reasonCode":"OPS-0016.demo_prec_approve","reason":"DEMO PREC: this lookup needs approval.","messageForPerson":"This lookup needs approval.","conflictCaseId":"pc_40588"}`

The settling rows record who decided:

```
40593 live_conflict approve actioned c2b5b404-… "Precedent demo run 1"  decidedBy {"kind":"person","name":"c2b5b404-…"}  caseId pc_40588
40607 live_conflict approve actioned c2b5b404-… "Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine"  decidedBy {"kind":"person",…}  caseId pc_40602
```

## Step 4.3: the standing-rule proposal. PASS

After the fifth settlement, the system raised policy case **40622**:

- `kind policy`, `raised_by repeat`, open;
- versions `{OPS-0016:2, OPS-0017:2}`;
- `subject_type policy_conflict`, created_by `system:conflict_detector`;
- **`facts.proposal = {"from":"repeat","count":5,"outcome":"approve"}`**.

Before the fifth settlement there was no proposal (it is the only `policy` row on the pair).

## Step 4.4: the sixth identical call runs on precedent. PASS (see D1)

The same call (`workflow demo-prec-9782f2-wf-6`):

```
{"rounds": 2, "error": null, "handler_ran": ["DEMO-PREC-9782f2"],
 "tool_messages": ["{\"stub\": true, \"found\": false, \"query\": \"DEMO-PREC-9782f2\"}"], "answer": "Done."}
```

**The tool ran.** No approval case was opened: there are 0 `agent_policy_approval` rows for this workflow.

**The live record, decision 40624** (`out/step4.4.txt`):

- `subject_type live_conflict`, subject `OPS-0016|OPS-0017`, `decision approve`, `status actioned`;
- **`actioned_by system:precedent`**, `decision_scope this_action`;
- `override_reason`: "Decided the same way (approve) 5 times before by people: pc_40616, pc_40609, pc_40602,
  pc_40595, pc_40588.";
- **`facts.decidedBy = {"kind":"precedent","name":"system:precedent"}`**,
  `facts.versionsAtDecision = {"OPS-0016":2,"OPS-0017":2}`, `facts.priorDecisions = {"sameConflict":5,"lastOutcome":"approve"}`;
- `evidence`: the overlap example plus **five `precedent` entries**, one per cited case. For example:
  `{"kind":"precedent","caseId":"pc_40616","source":"proc.bp_agent_policy_conflict","outcome":"approve","actioned_by":"c2b5b404-…","actioned_at":"2026-10-09T20:38:52.040587+00:00","decision_id":40616}`.
  The others are pc_40609, pc_40602, pc_40595 and pc_40588, the same five cases as runs 1 to 5.
- Conflict row: `kind live`, `is_open false`, `outcome approve`, **`decided_by system:precedent`, `by_person false`**.
  Because it is not by a person, it can never count toward a later precedent.

**The firing rows linked to it:**

```
11014 OPS-0016 v2 approve  result allowed  decision_id 40624  "Decided on precedent (pc_40624)"  matched {"tool.name":"get_policy","args.query":"•••"}
11015 OPS-0017 v2 approve  result allowed  decision_id 40624  "Decided on precedent (pc_40624)"  matched {…,"args.query":"•••"}
```

**The notifications** (no input values, the query is not in any of them):

```
23529 → Demo Prec Finance Owner     "Looking up a governed policy (policy OPS-0016) ran on precedent: decided the same way 5 times before (pc_40624)."  link agent-policy:OPS-0016
23530 → Demo Prec Finance Approver  (the same, OPS-0016)
23531 → Demo Prec Customer Owner    "… (policy OPS-0017) ran on precedent: decided the same way 5 times before (pc_40624)."  link agent-policy:OPS-0017
23532 → Demo Prec Customer Approver (the same, OPS-0017)
```

## Step 4.5: `GET /agent-policies/OPS-0016` `conflicts[]`, linked and unlinked. PASS

Three callers, all through :3011 (`out/view-*.txt`):

| Caller (gateway groups) | Role | Linked? | `conflicts[]` |
|---|---|---|---|
| `PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER` | Admin | yes | 7 entries, unmasked |
| `PROCWISE_FINANCE_REVIEWER_APPROVER` | Approver | **yes (linked owner/decider, not Admin)** | 7 entries, unmasked, identical to Admin's |
| `PROCWISE_VIEWER` | Viewer | **no** | 7 entries, **masked** |

The newest entry, for the linked caller:

```
{"caseId":"pc_40624","kind":"live","isOpen":false,"policies":[{"id":"OPS-0016","version":2},{"id":"OPS-0017","version":2}],
 "example":{"tool.name":"get_policy","args.query":"DEMO-PREC-9782f2"},
 "decision":{"option":"approve","scope":"this_action","decidedBy":{"kind":"precedent","name":"system:precedent"},
             "decidedAt":"2026-10-09T20:39:03.819822+00:00","reason":"Decided the same way (approve) 5 times before by people: pc_40616, …"},
 "citedCases":["pc_40616","pc_40609","pc_40602","pc_40595","pc_40588"],"proposal":null,"otherPolicies":["OPS-0017"]}
```

The other entries: pc_40622 (policy, open, `proposal {from:"repeat",count:5,outcome:"approve"}`). After it,
pc_40616 … pc_40588 (live, `decidedBy {kind:"person", name:"c2b5b404-…"}`, reasons "Precedent demo run n").

Linked caller (Approver) and unlinked caller (Viewer), side by side:

| Field | Linked | Unlinked |
|---|---|---|
| `example` (all 7) | `{"tool.name":"get_policy","args.query":"DEMO-PREC-9782f2"}` | `{"tool.name":"get_policy","args.query":"•••"}` |
| pc_40602 reason | "Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine" | "Precedent demo run 3: approved, the lookup of **•••** is routine" |
| other reasons | as written | as written (no sensitive value in them) |

The unlinked Viewer's `GET /agent-policies/conflicts/40622` gave the same answer: `canDecide false`, example
masked, `conflictHistory` 7 entries, and pc_40602's reason masked.

## Step 4.6: the two CSV exports. PASS

`GET /agent-policies/OPS-0016/conflicts/history.csv` →
200, `Content-Type: text/csv; charset=utf-8`, **`Content-Disposition: attachment; filename="conflict-history-OPS-0016.csv"`**.
`GET /agent-policies/conflicts/history.csv?pair=OPS-0016%7COPS-0017` →
200, the same type, **`Content-Disposition: attachment; filename="conflict-history-OPS-0016_OPS-0017.csv"`**.
For these demo policies, both files hold the same rows.

As the linked caller:

```
"Case","Kind","Raised","Policies","Decided by","Name","Decision","Scope","Decided at","Reason","Cited cases"
"pc_40624","During an action","2026-10-09T20:39:03.911983+00:00","OPS-0016 v2; OPS-0017 v2","Precedent","system:precedent","Approve","this_action","2026-10-09T20:39:03.819822+00:00","Decided the same way (approve) 5 times before by people: pc_40616, pc_40609, pc_40602, pc_40595, pc_40588.","pc_40616 pc_40609 pc_40602 pc_40595 pc_40588"
"pc_40622","Between policies","2026-10-09T20:38:52.055270+00:00","OPS-0016 v2; OPS-0017 v2","","","Waiting for a decision","","","",""
"pc_40616","During an action","2026-10-09T20:38:50.596918+00:00","OPS-0016 v2; OPS-0017 v2","Person","c2b5b404-40c1-7047-71c4-dd8093ecf25d","Approve","this_action","2026-10-09T20:38:52.040587+00:00","Precedent demo run 5",""
"pc_40609","During an action","2026-10-09T20:38:47.587715+00:00","OPS-0016 v2; OPS-0017 v2","Person","c2b5b404-40c1-7047-71c4-dd8093ecf25d","Approve","this_action","2026-10-09T20:38:49.072843+00:00","Precedent demo run 4",""
"pc_40602","During an action","2026-10-09T20:38:44.527936+00:00","OPS-0016 v2; OPS-0017 v2","Person","c2b5b404-40c1-7047-71c4-dd8093ecf25d","Approve","this_action","2026-10-09T20:38:46.036328+00:00","Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine",""
"pc_40595","During an action","2026-10-09T20:38:41.493859+00:00","OPS-0016 v2; OPS-0017 v2","Person","c2b5b404-40c1-7047-71c4-dd8093ecf25d","Approve","this_action","2026-10-09T20:38:42.980706+00:00","Precedent demo run 2",""
"pc_40588","During an action","2026-10-09T20:38:38.487803+00:00","OPS-0016 v2; OPS-0017 v2","Person","c2b5b404-40c1-7047-71c4-dd8093ecf25d","Approve","this_action","2026-10-09T20:38:39.957762+00:00","Precedent demo run 1",""
```

- There are **six decision rows** (five by a person, one on precedent). There is also a seventh row: the open
  5-times proposal, "Waiting for a decision".
- **The masking difference:** the unlinked Viewer's files are byte-identical except for one cell, pc_40602's
  Reason, which reads `"Precedent demo run 3: approved, the lookup of ••• is routine"`.
- The Admin and Approver files are identical.

## Step 4.7: screens in headless Chrome. PASS (see D2, D3)

Navigation was the same as in stage 4: `/spendiq`, `go('workspace')`, `wfSetAgentTab('policy')`, then a click on
the workspace "Policies" button. The caller was the Admin+Approver bypass identity.

1. **The policy form's Conflicts section** (List → Open OPS-0016; `shots/01-form-conflicts-section.png`):
   ```
   CONFLICTS
   pc_40624 with OPS-0017 · During an action · Decided: Approve · on precedent (pc_40616, pc_40609, pc_40602, pc_40595, pc_40588) — Decided the same way (approve) 5 times before by people: pc_40616, …
   pc_40622 with OPS-0017 · Between policies · Waiting for a decision
   pc_40616 with OPS-0017 · During an action · Decided: Approve · by a person — Precedent demo run 5
   … (runs 4, 3, 2, 1; run 3 shows its reason with the query, as this reader is linked)
   [Export conflict history]
   ```
   Pressing **Export conflict history** made one call, `200 XHR /agent-policies/OPS-0016/conflicts/history.csv`.
   The browser saved **`conflict-history-OPS-0016.csv`**, which is identical to the file in step 4.6. Nothing was
   written during the form pass (no `/versions`, `/retire` or `/decide`), and Cancel closed the form.
2. **A Conflicts-screen case with "History of this clash"** (case pc_40622):
   - In the list view (`shots/03-…`), the card shows the proposal banner "Decided the same way 5 times: make it a
     standing rule?", the example, both policies, the options and **Export conflict history**, but **no "History
     of this clash"** (D2). It also shows "Previous decisions: First time" (D3).
   - Export from the card made one call, `200 /agent-policies/conflicts/history.csv?pair=OPS-0016%7COPS-0017`,
     and saved `conflict-history-OPS-0016_OPS-0017.csv`.
   - To make the history appear, I decided the case on screen: "Keep both: OPS-0016 takes priority", with a
     reason. Confirm 1 read "Decide this conflict? …" and confirm 2 read "Confirm: Keep both: OPS-0016 takes
     priority?". Then came `201 /conflicts/40622/decide`, `200 /conflicts/40622` and `200 /agent-policies`.
   - The card then read **"History of this clash"** (`shots/04-conflicts-case-history.png`):
     ```
     pc_40624 · During an action · Decided: Approve · on precedent (pc_40616, pc_40609, pc_40602, pc_40595, pc_40588) — Decided the same way (approve) 5 times before …
     pc_40622 · Between policies · Decided: Keep both: OPS-0016 takes priority · by a person — Precedent demo: make the five approvals a standing rule.
     pc_40616 … pc_40588 · During an action · Decided: Approve · by a person — Precedent demo run 5 … 1
     ```
3. **An approval card of a paused clash with "Earlier decisions on this clash"** (fresh pair C/D, OPS-0018/0019):
   - Both policies were created and activated. Call 1 (`wf-CD-1`) was paused (pc_40625, "only 0 of 5"), and both
     sides approved it with the reason "Precedent demo C/D first call: approved for DEMO-PREC-9782f2-CD".
   - Call 2 (`wf-CD-2`) was paused (pc_40632, requestIds [40633, 40634], "only 1 of 5"), with 1 entry in
     `facts.history`.
   - The Approvals tab showed the card's conflict block (`shots/05-…`, `06-…`):
     ```
     Policies in conflict  pc_40632 · … Previous decisions: Decided 1 time before; last: approved
     Options: Approve, Reject. Respond within 10 minutes.
     Not decided on precedent: only 1 of 5 decisions by people on this exact clash
     Earlier decisions on this clash
       pc_40625 · During an action · Decided: Approve · by a person — Precedent demo C/D first call: approved for DEMO-PREC-9782f2-CD
     ```

**Console, over all four browser sessions:**
- **0 `console.error` calls and 0 uncaught exceptions.** Every `/agent-policies` call returned 2xx.
- There was some network noise that does not come from this feature. The browser logged it, not page code. It is
  the same as in stages 1 to 4: `GET :3011/users/me/preferences` 404, and a CORS refusal of
  `GET :3011/user?page=1&limit=100` (the bypass gateway answers it with `Access-Control-Allow-Origin: null`).

## Step 4.8: `/decisions` shows neither type. PASS

Read straight from :8010:

```
GET /decisions?subject_type=policy_conflict              -> total 0
GET /decisions?subject_type=live_conflict                -> total 0
GET /decisions?subject_type=policy_conflict&status=actioned -> total 0
GET /decisions?subject_type=live_conflict&status=actioned   -> total 0
GET /decisions?limit=500                -> total 51, rows 51, conflict rows 0, types ['authorization','email_reply','finding']
GET /decisions?status=actioned&limit=500 -> total 3, rows 3, conflict rows 0, types ['email_reply','finding']
GET /decisions/{40622, 40624, 40632, 40588} -> 404 each;  GET /decisions/1356 (control) -> 200
```

## Step 5: clean-up. DONE

- `POST /agent-policies/OPS-00{16,17,18,19}/retire` → 201 each (version 3). All four are `retired`, with
  live_version NULL, and `live_policies.load()` returns `[]`.
- Retiring did **not** close the paused C/D call: live case 40632 and member cases 40633/40634 stayed open
  (see O1). I rejected it through the gateway, `POST /agent-policies/approvals/40633/decide {verb:"reject"}` → 201
  `rejected`, group `{rejected:2,total:2}`. The second member was closed by the same reject (a repeat answered
  409). Live case 40632 is now closed `reject`, by_person true.
- Afterwards, 0 open `bp_decision` rows name OPS-0016…0019 or the demo workflows.
- `DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(<my 4 names>)` → 4 rows. The map now holds
  `Demo CFO`, `Demo Finance Manager`, `Demo Ops` (stage 3's rows). The `TST RP e925bde3` row seen before the run
  was removed by another session, not by me: my delete matched names only.
- The governed row reads `{"precedent_count": 5}` and was never edited. No company setting was changed.
- **Left in `bp_testdb` as the record** (ids are never deleted):
  - the four retired policies and their versions;
  - conflict rows 40588, 40595, 40602, 40609, 40616, 40622, 40624, 40625 and 40632, with their decision and action
    rows;
  - the standing rule from the screen decision (OPS-0016 prevails over OPS-0017; both are retired);
  - the firing rows (append-only by design), the notifications and the audit rows.
- Processes: only my four process groups were stopped. Ports 8010, 3011, 3010 and 9223 are closed.
  `procwise.service` is `active`. The scratch base worktrees were removed.

## Defects found (recorded, not patched)

**D1. The agent is never told that its action ran on precedent.**
- Spec §3.2: "`to_agent` carries `conflictCaseId` and `precedent: true`". The gate builds it
  (`gate._precedent_answer` → `GateResult(allow=True, to_agent={"result":"allowed","conflictCaseId":…,"precedent":True})`).
- `services/tool_runtime.py:184` returns `None if verdict.allow else (…to_agent…)`, so on allow the `to_agent` is
  dropped.
- Evidence: in step 4.4 the only tool message the "model" received was the stub's own result
  `{"stub": true, "found": false, "query": "DEMO-PREC-9782f2"}`. It had no `conflictCaseId` and no `precedent`.
- Impact: the agent, and through it the person, cannot say that no human approved this run. The record,
  firings and notifications are correct.
- This was already known as a deferred minor from Task 7 ("tool_runtime drops to_agent on allow"). The live run
  confirms it.

**D2. An open case on the Conflicts screen does not show "History of this clash" until someone decides it.**
- `GET /agent-policies/conflicts` (the list) carries no `conflictHistory`; only `GET /conflicts/{id}` does.
- `apPcCardHTML` reads `ui.detail || c`, and `ui.detail` is loaded only after a decide (`apPcLoadDetail`). A case
  opened from a tag or a notification uses the list entry when it is in the list (`apPcOpen`).
- Evidence: step 4.7.2. The open proposal pc_40622 showed no history section (`history section in list view:
  false`). After the decide, the detail was fetched and the section appeared with 7 entries.
- Impact: the owners deciding a "make it a standing rule?" proposal, the case where the history matters most,
  cannot see the decisions it is based on. They can still export them.
- Possible fixes (for a later decision): fetch the detail when a card is shown, or add the history to the list.

**D3. The 5-times proposal says "Previous decisions: First time".**
- The proposal card (pc_40622) shows the banner "Decided the same way 5 times: make it a standing rule?" and, in
  the same card, "Previous decisions: First time". After the decide, the card's history listed 6 earlier
  decisions on this clash.
- Cause (likely): a policy case's `prior` counts only earlier policy cases for the pair, not live ones. This is
  stage 4 behaviour, not new code, but it now sits next to the history and contradicts it.
- Impact: display only.

## Observations (not defects)

- O1. Retiring both policies of a paused live clash leaves the live case and its member approval cases open. An
  approver could still approve an action of a retired policy, and the replay would then re-check against
  policies that are no longer live. I closed it by rejecting. This is outside this feature's scope.
- O2. On the precedent record, `actioned_at` (20:39:03.819, the gate's clock when the call began) is earlier than
  `created_at` (20:39:03.911, the database's `now()`). So the CSV's "Decided at" comes before its "Raised" for
  pc_40624.
- O3. The brief expected "six rows" in the export. There are six decision rows plus the open proposal row, seven
  in all. One row per case is the ruled shape (plan ruling "one CSV row per case").
- O4. The CSV's "Decided by" has no words for kind `system` (the known final-wave item, conflict_history.py
  `DECIDED_BY_WORDS`). No case in this run was closed that way, so it did not show.
- O5. An unlinked Viewer still sees who decided (the deciding person's sub, in `decidedBy.name` and the CSV "Name"
  column). Only values from the action are masked, which matches design §3.1.

## Red/green captures

This run broke no guard (no product code was touched). The per-task captures, from each task's report:

| Task | Guard broken on purpose | Red | Restored |
|---|---|---|---|
| 1 | `_rules` forced to the cached path (`if True:`) | `test_a_fresh_read_sees_an_edit_without_a_restart`: `assert 5 == 2` | `-k fresh` 5 passed |
| 2 | migration's `WHERE NOT EXISTS (…)` removed (file only, never applied) | `test_applying_again_adds_nothing…` failed (idempotence) | 6 passed; 1 row per DB |
| 3 | `threshold()` returns 5 when the row is missing | 2 failed (`…_is_none_and_warns_when_the_row_is_missing`, `…_not_a_number`) | 9 passed |
| 4 | (red before code) | 5 unit + 5 live failed (`no attribute 'PRECEDENT'`, …) | 6 + 6 passed |
| 5 | `AND c.by_person` removed from `PRECEDENT_SQL` | 1 failed (`AssertionError: c.by_person`) | 13 passed |
| 6 | `shown()` masking disabled (`if False:`); fix round: `_shown_actions` `if True:` | 4 failed (stranger reads masked, eligible sees all, whole-token, live owner/decider/admin); 1 failed (`'10001' not in …`) | 15 passed; 6 passed |
| 7 | G1 `by_person` removed (the call RAN on 2 precedent rows, `KeyError 'result'`); a block outside the clash; G3 `shown(raw())` stored; G6 commit right after the precedent insert | each red as reported (G6: `system:precedent` row left behind) | step-4 suite 75 passed |
| 8 | `csv_cell` skips the apostrophe | 10 unit failures + `test_a_formula_in_a_reason_is_neutralised` | 22 + 11 passed |
| 9 | gateway `PAIR = /.*/` | 6 failed (bad pair forms accepted) | 369 passed (full jest) |
| 10 | G1 raw reason in `apPcHistoryList`; G2 raw precedent note; G3/G4 `system` kind; fix round G5 `retired` kind, G6 moot suppression | 1-2 failed each | 239 / 240 passed |

## Diff summary

### BP_Backend: `f2bab3ac..4d6cf48d` (11 commits; 33 files, +5413 / −152; nothing under `.superpowers/`)

```
4d6cf48d feat(agent-policy): export a policy's or a pair's conflict history as CSV, masked per caller
db13af6f feat(agent-policy): the gate lets the decision engine decide a live clash on precedent
b58ec4ea fix(agent-policy): the Conflicts screen masks a case's decision and history like its conflict history
282ef3dc feat(agent-policy): one conflict history reader, masked per reader, for every screen
2edf08c0 feat(decisions): the decision engine decides a live policy clash on clear precedent, else escalates
180b7f28 fix(agent-policy): a live clash is a timeout only when the sweep decided it, never by actor prefix
d75a5807 feat(agent-policy): every conflict decision records what decided it and the versions it was about
791b60cd feat(agent-policy): the standing-rule proposal reads N from the governed precedent count
65695a42 feat(agent-policy): the precedent count is a governed policy row a customer can change
708c6ffa feat(governance): read a governed limit fresh, so a policy-admin edit applies without a restart
1373e486 docs(agent-policy): conflict history and precedent implementation plan
```

| Area | Files (lines added / removed) |
|---|---|
| Schema | `deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql` (+38) and its rollback (+7); applied to both DBs in Task 2 |
| New service | `services/agent_policy/conflict_history.py` (+245: reader, masking, CSV) |
| Decision engine | `engines/decision_engine.py` (+101: `decide_live_conflict`, `PRECEDENT_SQL`) |
| Changed services | `gate.py` (+150 net: precedent consult/record/notify/answer), `conflict_live.py` (+76), `conflict_cases.py` (70 changed: `decided_by`), `conflict_views.py` (+41), `settings.py` (+18: `precedent_count`, `live_conflict_repeat` removed), `governed_limits.py` (+36: `fresh=True`), `approval_views.py` (+6), `repositories/agent_policy_repo.py` (+4) |
| Router | `routers/agent_policies.py` (+57: two CSV routes, viewer-aware reads) |
| Tests | 7 new test files (`test_conflict_precedent_live.py` +382, `test_conflict_history*.py`, `test_conflict_export_live.py`, `test_conflict_decided_by*.py`, `test_precedent_count.py`, `tests/engines/test_decide_live_conflict.py`, the migration test) and small edits to 8 others |
| Plan | `specs/2026-10-09-conflict-history-and-precedent-plan.md` (+3293) |

### Gateway: `e7d9c99..0723d05` (1 commit; 4 files, +154 / −9)

```
0723d05 feat(agent-policy): forward the conflict history CSV exports
```

`agent-policy.controller.ts` (+23), `agent-policy.service.ts` (+32/−9, `forwardText`), `agent-policy.yml` (+50),
`agent-policy.controller.spec.ts` (+58).

### UI: `8991c0b..ef5583b` (2 commits; 4 files, +208 / −9)

```
ef5583b fix(agent-policy): a case closed by a policy's retirement names its decider, and says retired once
869f815 feat(agent-policy): conflict history shows who decided, lists a clash's history, and exports it
```

`agentPolicy/conflicts.js` (+49: `decidedByText`, `reasonText`, `pairOf`, `historyExportPath`, `historyFileName`),
`conflicts.test.js` (+63), `engineWiring.stage4.contract.test.js` (+76), `engine.js` (+29: new
`apPcHistoryList`, `apPcExportHistory`, `apPcDownload`, plus minimal wiring in `apPcBind`, `apPcCardHTML`,
`apPcPolicySection`, `apPcApprovalBlock`). `policyEdit`, `policyDelete` and `openFormModal` are not in the diff.

## Appendix: every gateway request and response in this run

The driver recorded these as it ran. Long answers are abridged. The driver never sent or logged a key. The
`/decisions` reads went straight to :8010 and are quoted in step 4.8. CSV answers are summarised here; their full
text is in steps 4.6 and 4.7.

| # | Step | Request | Status | Response (abridged) |
|---|---|---|---|---|
| 1 | 4.1-deciders | `PUT /agent-policies/deciders/Demo%20Prec%20Finance%20Owner` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Precedent demo (2026-10-09)"} | 200 | {"name": "Demo Prec Finance Owner", "savedAt": "2026-10-09T20:38:21.351584+00:00"} |
| 2 | 4.1-deciders | `PUT /agent-policies/deciders/Demo%20Prec%20Customer%20Owner` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Precedent demo (2026-10-09)"} | 200 | {"name": "Demo Prec Customer Owner", "savedAt": "2026-10-09T20:38:21.814300+00:00"} |
| 3 | 4.1-deciders | `PUT /agent-policies/deciders/Demo%20Prec%20Finance%20Approver` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Precedent demo (2026-10-09)"} | 200 | {"name": "Demo Prec Finance Approver", "savedAt": "2026-10-09T20:38:22.163777+00:00"} |
| 4 | 4.1-deciders | `PUT /agent-policies/deciders/Demo%20Prec%20Customer%20Approver` {"groups": ["PROCWISE_FINANCE_REVIEWER_APPROVER"], "emails": [], "notes": "Precedent demo (2026-10-09)"} | 200 | {"name": "Demo Prec Customer Approver", "savedAt": "2026-10-09T20:38:22.513408+00:00"} |
| 5 | 4.1-deciders | `GET /agent-policies/deciders`  | 200 | {"deciders": [{"name": "Demo CFO", "groups": [], "emails": ["demo-cfo@example.test"], "notes": "Stage 3 demo (2026-10-08)", "lastModifiedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "lastModifiedAt": "2026-10-08T23:29:50… |
| 6 | 4.1-create | `POST /agent-policies` form "DEMO precedent A 9782f2" | 201 | {"policyKey": "OPS-0016", "version": 1} |
| 7 | 4.1-create | `POST /agent-policies` form "DEMO precedent B 9782f2" | 201 | {"policyKey": "OPS-0017", "version": 1} |
| 8 | 4.1-create | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 9 | 4.1-create | `GET /agent-policies/OPS-0017`  | 200 | {"policyKey": "OPS-0017", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 10 | 4.1-activate | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 11 | 4.1-activate | `POST /agent-policies/OPS-0016/versions` form "DEMO precedent A 9782f2" {"baseVersion": 1, "intent": "activate", "changeNote": "Precedent demo: activate"} | 201 | {"policyKey": "OPS-0016", "version": 2} |
| 12 | 4.1-activate | `GET /agent-policies/OPS-0017`  | 200 | {"policyKey": "OPS-0017", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 13 | 4.1-activate | `POST /agent-policies/OPS-0017/versions` form "DEMO precedent B 9782f2" {"baseVersion": 1, "intent": "activate", "changeNote": "Precedent demo: activate"} | 201 | {"policyKey": "OPS-0017", "version": 2} |
| 14 | 4.2-1 | `POST /agent-policies/approvals/40589/decide` {"verb": "approve", "reason": "Precedent demo run 1"} | 201 | {"decisionId": 40589, "actionId": 40591, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:39.481270+00:00", "level": 0, "levelName": "Demo Prec … |
| 15 | 4.2-1 | `POST /agent-policies/approvals/40590/decide` {"verb": "approve", "reason": "Precedent demo run 1"} | 201 | {"decisionId": 40590, "actionId": 40592, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:39.957762+00:00", "level": 0, "levelName": "Demo Prec … |
| 16 | 4.2-2 | `POST /agent-policies/approvals/40596/decide` {"verb": "approve", "reason": "Precedent demo run 2"} | 201 | {"decisionId": 40596, "actionId": 40598, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:42.503173+00:00", "level": 0, "levelName": "Demo Prec … |
| 17 | 4.2-2 | `POST /agent-policies/approvals/40597/decide` {"verb": "approve", "reason": "Precedent demo run 2"} | 201 | {"decisionId": 40597, "actionId": 40599, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:42.980706+00:00", "level": 0, "levelName": "Demo Prec … |
| 18 | 4.2-3 | `POST /agent-policies/approvals/40603/decide` {"verb": "approve", "reason": "Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine"} | 201 | {"decisionId": 40603, "actionId": 40605, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:45.554907+00:00", "level": 0, "levelName": "Demo Prec … |
| 19 | 4.2-3 | `POST /agent-policies/approvals/40604/decide` {"verb": "approve", "reason": "Precedent demo run 3: approved, the lookup of DEMO-PREC-9782f2 is routine"} | 201 | {"decisionId": 40604, "actionId": 40606, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:46.036328+00:00", "level": 0, "levelName": "Demo Prec … |
| 20 | 4.2-4 | `POST /agent-policies/approvals/40610/decide` {"verb": "approve", "reason": "Precedent demo run 4"} | 201 | {"decisionId": 40610, "actionId": 40612, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:48.593759+00:00", "level": 0, "levelName": "Demo Prec … |
| 21 | 4.2-4 | `POST /agent-policies/approvals/40611/decide` {"verb": "approve", "reason": "Precedent demo run 4"} | 201 | {"decisionId": 40611, "actionId": 40613, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:49.072843+00:00", "level": 0, "levelName": "Demo Prec … |
| 22 | 4.2-5 | `POST /agent-policies/approvals/40617/decide` {"verb": "approve", "reason": "Precedent demo run 5"} | 201 | {"decisionId": 40617, "actionId": 40619, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:51.566267+00:00", "level": 0, "levelName": "Demo Prec … |
| 23 | 4.2-5 | `POST /agent-policies/approvals/40618/decide` {"verb": "approve", "reason": "Precedent demo run 5"} | 201 | {"decisionId": 40618, "actionId": 40620, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:38:52.040587+00:00", "level": 0, "levelName": "Demo Prec … |
| 24 | 4.5-admin-linked | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 25 | 4.6-admin-linked | `GET /agent-policies/OPS-0016/conflicts/history.csv`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016.csv"; 7 data rows |
| 26 | 4.6-admin-linked | `GET /agent-policies/conflicts/history.csv?pair=OPS-0016%7COPS-0017`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016_OPS-0017.csv"; 7 data rows |
| 27 | 4.5-approver-linked | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 28 | 4.6-approver-linked | `GET /agent-policies/OPS-0016/conflicts/history.csv`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016.csv"; 7 data rows |
| 29 | 4.6-approver-linked | `GET /agent-policies/conflicts/history.csv?pair=OPS-0016%7COPS-0017`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016_OPS-0017.csv"; 7 data rows |
| 30 | 4.5-viewer-unlinked | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 31 | 4.6-viewer-unlinked | `GET /agent-policies/OPS-0016/conflicts/history.csv`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016.csv"; 7 data rows |
| 32 | 4.6-viewer-unlinked | `GET /agent-policies/conflicts/history.csv?pair=OPS-0016%7COPS-0017`  | 200 | text/csv; charset=utf-8; attachment; filename="conflict-history-OPS-0016_OPS-0017.csv"; 7 data rows |
| 33 | 4.7-viewer | `GET /agent-policies/conflicts/40622`  | 200 | {"caseId": "pc_40622", "decisionId": 40622, "pairKey": "OPS-0016\|OPS-0017", "status": "open", "raisedAt": "2026-10-09T20:38:52.055270+00:00", "raisedBy": "repeat", "policies": [{"id": "OPS-0016", "owner": "Demo Prec Fina… |
| 34 | 4.7-create | `POST /agent-policies` form "DEMO precedent C 9782f2" | 201 | {"policyKey": "OPS-0018", "version": 1} |
| 35 | 4.7-create | `POST /agent-policies` form "DEMO precedent D 9782f2" | 201 | {"policyKey": "OPS-0019", "version": 1} |
| 36 | 4.7-create | `GET /agent-policies/OPS-0018`  | 200 | {"policyKey": "OPS-0018", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 37 | 4.7-create | `GET /agent-policies/OPS-0019`  | 200 | {"policyKey": "OPS-0019", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 38 | 4.7-activate | `GET /agent-policies/OPS-0018`  | 200 | {"policyKey": "OPS-0018", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 39 | 4.7-activate | `POST /agent-policies/OPS-0018/versions` form "DEMO precedent C 9782f2" {"baseVersion": 1, "intent": "activate", "changeNote": "Precedent demo: activate"} | 201 | {"policyKey": "OPS-0018", "version": 2} |
| 40 | 4.7-activate | `GET /agent-policies/OPS-0019`  | 200 | {"policyKey": "OPS-0019", "status": "draft", "liveVersion": null, "latestVersion": 1, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt"… |
| 41 | 4.7-activate | `POST /agent-policies/OPS-0019/versions` form "DEMO precedent D 9782f2" {"baseVersion": 1, "intent": "activate", "changeNote": "Precedent demo: activate"} | 201 | {"policyKey": "OPS-0019", "version": 2} |
| 42 | 4.2-CD-1 | `POST /agent-policies/approvals/40626/decide` {"verb": "approve", "reason": "Precedent demo C/D first call: approved for DEMO-PREC-9782f2-CD"} | 201 | {"decisionId": 40626, "actionId": 40628, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:50:39.828929+00:00", "level": 0, "levelName": "Demo Prec … |
| 43 | 4.2-CD-1 | `POST /agent-policies/approvals/40627/decide` {"verb": "approve", "reason": "Precedent demo C/D first call: approved for DEMO-PREC-9782f2-CD"} | 201 | {"decisionId": 40627, "actionId": 40629, "verb": "approve", "result": "approved", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:50:40.310332+00:00", "level": 0, "levelName": "Demo Prec … |
| 44 | cleanup | `GET /agent-policies/OPS-0016`  | 200 | {"policyKey": "OPS-0016", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 45 | cleanup | `POST /agent-policies/OPS-0016/retire` {"baseVersion": 2, "changeNote": "Precedent demo finished: retired"} | 201 | {"policyKey": "OPS-0016", "version": 3} |
| 46 | cleanup | `GET /agent-policies/OPS-0017`  | 200 | {"policyKey": "OPS-0017", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 47 | cleanup | `POST /agent-policies/OPS-0017/retire` {"baseVersion": 2, "changeNote": "Precedent demo finished: retired"} | 201 | {"policyKey": "OPS-0017", "version": 3} |
| 48 | cleanup | `GET /agent-policies/OPS-0018`  | 200 | {"policyKey": "OPS-0018", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 49 | cleanup | `POST /agent-policies/OPS-0018/retire` {"baseVersion": 2, "changeNote": "Precedent demo finished: retired"} | 201 | {"policyKey": "OPS-0018", "version": 3} |
| 50 | cleanup | `GET /agent-policies/OPS-0019`  | 200 | {"policyKey": "OPS-0019", "status": "live", "liveVersion": 2, "latestVersion": 2, "areaName": "Operations", "versions": [{"version": 1, "savedAs": "draft", "savedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "savedAt": "2… |
| 51 | cleanup | `POST /agent-policies/OPS-0019/retire` {"baseVersion": 2, "changeNote": "Precedent demo finished: retired"} | 201 | {"policyKey": "OPS-0019", "version": 3} |
| 52 | cleanup | `POST /agent-policies/approvals/40633/decide` {"verb": "reject", "reason": "Precedent demo finished: the paused C/D call is rejected"} | 201 | {"decisionId": 40633, "actionId": 40636, "verb": "reject", "result": "rejected", "decidedBy": "c2b5b404-40c1-7047-71c4-dd8093ecf25d", "decidedAt": "2026-10-09T20:54:24.466975+00:00", "level": 0, "levelName": "Demo Prec F… |
| 53 | cleanup | `POST /agent-policies/approvals/40634/decide` {"verb": "reject", "reason": "Precedent demo finished: the paused C/D call is rejected"} | 409 | {"statusCode": 409, "message": "This request has already been decided."} |
