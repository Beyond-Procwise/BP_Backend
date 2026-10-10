# Agent policy governance, stage 5 (learning): verification and diff summary

Date: 2026-10-10, 12:06 to 12:41 UTC. Branch `agent-policy-stage5` in three worktrees (BP `f0a55677`, gateway
`2c5f799`, UI `5251664`). Database: `bp_testdb` (checked with `select current_database()` before the first
write). Nothing has been merged, pushed or deployed.

## In plain English

**The evidence in this run is made up.** The rules need 30 decisions, over 30 days, by 3 different people. No
real traffic of that kind exists yet, so my script inserted backdated "approved" rows into the decision log for
nine demo policies named "DEMO stage 5 …". Real suggestions on real traffic cannot appear until 30 days of
real decisions exist. Everything else in the run is real: the backend, the gateway, the screens in headless
Chrome, the database.

**What works**

- **A suggestion is raised only when the evidence supports it.** Six demo policies with 33 approvals each got
  one card each: "Raise the limit on Payment amount from 500 USD to 800 USD?". The policy with 29 approvals got
  none. The "not allowed" policy got none. The policy in Legal and compliance ("Never suggest changes") got
  none.
- **The card says what it should.** The change, the evidence sentence, four statistics and "Suggestion
  confidence: High". The same card sits on the policy's own form. The form's heading still says "Extraction
  confidence: Medium", so the two confidences are never mixed up.
- **"Draft this change" writes nothing.** As a Buyer, the form opened with the limit at 800, the situation
  sentence updated, three new examples (799, 800, 801) reading "Nothing happens / Nothing happens / A person
  decides", the confirmation unticked and the change note filled in. The database still showed the policy at
  version 2. The only requests were the read of the draft and the example check.
- **A form with unsaved edits is not overwritten.** It answered "Save or cancel your edits first."
- **Saving the draft closes the card.** The suggestion became "saved" and left the queue and the form. An
  Approver then activated it the normal way. 40 more approvals above the new limit did not bring the old card
  back (they raised a new, different one: 800 to 1100).
- **A saved draft that is put back brings the card back.** Save the draft at 800, save another draft back at
  500, scan: the card returned.
- **Finance needs a second person.** In Finance the person who saved the draft could not activate it ("A
  second reviewer must review this change before it goes Active."), could not review it themselves ("You saved
  this change, so someone else must review it."), and the policy's owner could not review it either ("The
  policy's owner cannot be its second reviewer."). A second Approver recorded the review; the form then showed
  "Reviewed by demo-s5-p2 on 2026-10-10." and the first person activated it.
- **The second review does not depend on the card.** After the card was dismissed, a limit typed by hand
  was refused the same way, whether it was tighter (400) or looser (800). A reviewed draft could not be swapped
  for a different one (900) at activation. A wording-only edit of the same policy went Active with no review.
- **"Not yet" hides the card until 30 more decisions count.** Hidden at 30 decisions; still hidden at 59;
  back at 60.
- **A reversal weakens the card.** An Approver marked one approval as reversed, with a reason, through the
  two-step confirm. The row then read "Reversed on 2026-10-10: …". After a scan the card read "Suggestion
  confidence: Not enough yet" and its Draft button was off. A second reversal of the same row was refused.
- **A Viewer can read and nothing else.** The queue showed with no buttons. The policy form showed only
  Cancel. Seven direct write requests through the gateway each answered 403.
- **An Admin can switch learning off for a business area.** With "Never suggest changes" on for Operations,
  the Operations demo policies were no longer looked at and their four cards were closed. Both changes were
  written to the audit log with the values before and after. **Operations was switched back off 14 seconds
  later** and confirmed off in the database.
- **Nothing became Active on its own.** Every version of every demo policy was saved by a demo person through
  the gateway. The list is in step 18.
- **Screens:** 13 browser sessions, 6 identities. **0 `console.error` calls and 0 uncaught exceptions.**
- **Suites:** no new failure in any of the four (numbers below).

**What was not shown, and why**

- **The real, whole-database scan was not run.** `bp_testdb` holds 92 other "live" test policies, and a real
  scan would write suggestion rows for them. Every scan here was narrowed to the demo policies. For step 15
  the scan's whole-database path was run once with two changes made in my process only (see step 15), so that
  it could only touch demo rows.
- **Typing a limit by hand on the screen.** The form has no control for the limit (see O1). Step 11 was done
  through the gateway instead.
- **The card "directly above Conflicts".** No demo policy had a conflict, so there was no Conflicts section to
  sit above. The card is the last section of the form. The order is pinned by a UI test.
- **An owner linked to nobody, a sign-in with no email, and a "remove a value from a list" suggestion.**
  Covered by tests only.
- **The scheduler job running.** Its registration was checked without starting it (step 16).

**Faults and surprises found (recorded, not patched)**

- **D1.** On the form, the label "Second review" and its sentence run together: "Second reviewA second
  reviewer must review…".
- **D2.** After a draft is saved from a form opened over the queue, the queue behind it and the count on the
  Suggestions tab still show the card until Refresh is pressed.
- **D3.** "Reviewed by …" shows the reviewer's sign-in id, not a name. In production that id is a long code.
- **D4.** A weakened card does not say why. It still reads "30 of 30 requests … were approved" although one
  was reversed.
- **D5.** No automated test proves that a "not allowed" policy or a "Never suggest changes" area gets no
  suggestion. This run showed it; the test suite would not notice if that rule were removed.
- Smaller points are listed under Observations (O1 to O12).

**Clean-up.** All nine demo policies are retired. My 7 suggestion rows, 2 review rows, 3 decider-map rows and
10 registry rows are deleted. The 360 demo decision rows and the 1 reversal row stay (they cannot be deleted
by design). Only my own four processes were stopped.

## How the stack was run

| Part | Port | How |
|---|---|---|
| Backend | 127.0.0.1:8010 | `/home/muthu/PycharmProjects/BP_Backend/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8010 --workers 1` from the BP worktree, `PYTHONPATH=<worktree>:<worktree>/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`, `AGENT_POLICY_ENFORCEMENT=on`, **`AGENT_POLICY_LEARNING_SCAN=off`**, `AGENT_POLICY_CONFLICT_SCAN=off`, `AGENT_POLICY_APPROVAL_SWEEP=off`, `OLLAMA_BASE_URL=OLLAMA_HOST=http://127.0.0.1:9` (dead port), `OLLAMA_CLOUD_*` unset. Its log says "agent policy learning scan not registered (AGENT_POLICY_LEARNING_SCAN=off)". PGID 430629. |
| Gateway | :3011 | `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development BP_BACKEND_URL=http://127.0.0.1:8010`. The gateway key was read from the BP `.env` and never printed. It was stopped and started again on the same port at every change of identity (12 starts in all; the last PGID was 445618), each time with its own `AUTH_BYPASS_SUB`, `AUTH_BYPASS_EMAIL` and `AUTH_BYPASS_GROUPS` (table below). |
| UI | 127.0.0.1:3010 | Vite from the UI worktree, `VITE_API_URL=http://127.0.0.1:3011 VITE_AI_API_URL=http://127.0.0.1:8010 VITE_DEV_AUTH_BYPASS=true`. The untracked `mockPipelineDeals.js` was copied in for the run and deleted afterwards; the UI worktree is clean. PGID 431719. |
| Headless Chrome | :9223 | `google-chrome --headless=new` (Chrome 150), driven over CDP with `/usr/share/nodejs/ws`, 1440×900. Clicks were real mouse events on the element's centre. PGID 431853. |
| Driver | none | Scratch Python and Node scripts (not committed) in the session scratchpad, `stage5-demo/`. Python imports the worktree's own modules with the same `.env` and the dead Ollama port. **No model was called**: nothing in this stage calls one. |

The demo identities (no user rows were created; the gateway's bypass supplies them):

| Persona | Subject | Email | Group | Role |
|---|---|---|---|---|
| admin | `demo-s5-admin` | admin@demo-s5.example | PROCWISE_ADMIN | Admin |
| p1 | `demo-s5-p1` | p1@demo-s5.example | PROCWISE_FINANCE_REVIEWER_APPROVER | Approver |
| p2 | `demo-s5-p2` | p2@demo-s5.example | PROCWISE_FINANCE_REVIEWER_APPROVER | Approver |
| owner | `demo-s5-owner` | owner@demo-s5.example | PROCWISE_FINANCE_REVIEWER_APPROVER | Approver |
| buyer | `demo-s5-buyer` | buyer@demo-s5.example | PROCWISE_PROCUMENT_BUYER_ANALYST | Buyer |
| viewer | `demo-s5-viewer` | viewer@demo-s5.example | PROCWISE_VIEWER | Viewer |

Decider map (written with `PUT /agent-policies/deciders/<name>` as the Admin, deleted at clean-up):
"Demo S5 Finance Owner" → email owner@demo-s5.example; "Demo S5 Ops Owner" → opsowner@demo-s5.example;
"Demo S5 Decider" → decider@demo-s5.example. The owner is linked **by email, not by group**, so p1 and p2 (same
group as the owner persona) are not linked to the owner.

**The screen's role.** With `VITE_DEV_AUTH_BYPASS` the UI hard-codes the role "super admin" whoever the gateway
says the person is. After each page load the driver therefore set `window.__SPENDIQ_RBAC__.role` to the
persona's group, so the screen offers what that person would be offered. In production this value comes from
the sign-in. The gateway enforces the role on every call regardless.

All gateway requests carried `x-customer-id: 001`. At the end the four process groups were stopped with
`kill -TERM -<pgid>` (my PGIDs only). Ports 8010, 3011, 3010 and 9223 were confirmed closed.
`procwise.service` stayed `active` and was never touched; neither were :8000, :3000, :3001 or Ollama.

### Keeping the shared database safe

- **Before any write** (`out/s00_prep.txt`): `current_database` is `bp_testdb`; the suggestion table held
  **0 rows** and the second-review table 0 rows; nothing carried my tag `5d10a7`; `learning.targets` returned
  92 policies, none of them mine.
- **Own tool names, so no conflict cases.** Each demo policy has its own tool, `demo_s5_5d10a7_<letter>`, and
  reads its own input, `args.demo_s5_amount` (not `args.amount`, so nothing shared was added to). These 10
  names were added to `proc.bp_orchestrator_registry` with `seeded_from = 'demo-s5-5d10a7'` and deleted at
  clean-up. All nine policies also name the same source document. **No conflict row names a demo policy**
  (checked after creation and again before clean-up: 0).
- **Every scan was narrowed.** `learning.scan(conn, now, only=[…])` with the targets `learning.targets(cur)`
  returns, filtered to the demo keys. The unscoped scan was never run as shipped (see step 15).
- **Operations** is a real shared business area. Its "Never suggest changes" setting was on from 12:27:06 to
  12:27:20 UTC and is off again. Its `last_modified_by` now reads `demo-s5-admin` instead of `seed`; that is the
  true record and was left.
- **Policy numbers consumed** (ids are never reused): Finance 566 → 568, Operations 540 → 546, Legal and
  compliance 0 → 1 (`LEG-0001` is the first Legal number ever issued in `bp_testdb`). The live test suites run
  afterwards moved Finance to 612 and Operations to 571.

## The demo policies

All are "tool is `demo_s5_5d10a7_<letter>` and Payment amount (`args.demo_s5_amount`) is greater than 500",
checked before a tool runs, decider "Demo S5 Decider", source "DEMO stage 5 policy 5d10a7". Each was created
as a draft (version 1) and activated (version 2) by the Admin through the gateway.

| Letter | Key | Area | Outcome | Owner | Backdated approvals | Used for |
|---|---|---|---|---|---|---|
| F | FIN-0567 | Finance | approve | Demo S5 Finance Owner | 33 | steps 4, 7 to 10 |
| O | OPS-0541 | Operations | approve | Demo S5 Ops Owner | 33, then 40 more | steps 5, 6, 15 |
| B | OPS-0542 | Operations | **block** | Demo S5 Ops Owner | 33 | no card: block |
| L | LEG-0001 | Legal and compliance | approve | Demo S5 Ops Owner | 33 | no card: never suggest |
| T | OPS-0543 | Operations | approve | Demo S5 Ops Owner | **29** | no card: too few |
| H | FIN-0568 | Finance | approve | Demo S5 Finance Owner | 33 | steps 11, 11b |
| N | OPS-0544 | Operations | approve | Demo S5 Ops Owner | 33, then 26, then 1 | step 12 |
| R | OPS-0545 | Operations | approve | Demo S5 Ops Owner | 33 | step 13 |
| S | OPS-0546 | Operations | approve | Demo S5 Ops Owner | 33 | step 6b, the output filter |

O's situation reads "From 01/04/2026, the agent is about to issue a demo payment above 500 USD." S was later
renamed "DEMO stage 5 S 5d10a7 (from 01/04/2026)" with the same situation (a wording-only version 3).

## The evidence, worked by hand

Each backdated row looks like a real one: `action_name` and `matched_values["tool.name"]` are the policy's own
tool, `outcome 'approve'`, `result 'approved'`, `decision_id` NULL, `policy_version` the live version. Example:

```
(119527, 'FIN-0567', 2, 'tool.call.before', 'demo_s5_5d10a7_f', 'approve', 'approved',
 {'tool.name': 'demo_s5_5d10a7_f', 'args.demo_s5_amount': 510}, None, 'demo-s5-approver-a',
 '2026-08-26 00:00:00+00', '2026-08-26 00:01:00+00')
```

The layout is the one in `tests/agent_policy/test_learning_scan_live.py::fire`. Decision *i* has the amount it
is given, is asked on slot (7·*i*) mod 30 of 30 slots spread over 40 days from 2026-08-26, and is decided 60
seconds later by person *i* mod 3 (`demo-s5-approver-a/b/c`). 7 and 30 share no factor, so **any 30 consecutive
*i*** cover all 30 slots (first to last: 40 days) and all 3 people. Time and decider do not follow the amount.

The rule: decisions above the limit are ordered by amount and cut into bands of 30, walking away from the
limit. Each band must pass every guardrail alone. A tail shorter than 30 is not evidence.

- **33 approvals, 510, 520, … 830** (F, O, H, N, R, S, B, L). Band 1 = 510…800 = 30 decisions, 30 approved,
  3 people, 40 days, median 60 s, none reversed. Wilson lower bound of 30 out of 30 = 30 / (30 + 1.96²) =
  30 / 33.84 = **0.8865**, which is at least 0.85. The 3 left over (810, 820, 830) are not a band. So the
  suggestion is **500 → 800, 30 decisions, 1 band**. (The plan's 830 / 33 / 829-830-831 predates the banding
  rule; the examples are 799, 800, 801.)
- **29 approvals, 510 … 790** (T): fewer than one band. Nothing.
- **O after going live at 800: 40 more, 810 … 1200, on version 4.** Band 1 = 810…1100 = 30. Tail of 10.
  New suggestion **800 → 1100, 30 decisions**.
- **N: 26 more, 840 … 1090** (59 in all): band 1 = 510…800, tail of 29. Still 30 counted. Hidden at 30,
  needs 30 + 30 = 60. **One more, 1100** (60 in all): band 2 = 810…1100 = *i* 30…59, 30 consecutive. Now
  **500 → 1100, 60 decisions, 2 bands**: 60 ≥ 60, the card returns.
- **R: one reversal at 680.** 680 is in band 1, so the band fails. Nothing is proposed; the card stays as
  "Not enough yet" with 30 decisions, 30 approved, 1 reversed.

## Step 1: four policies live, and what the technical view says. PASS

`POST /agent-policies {form}` → 201, then `POST /agent-policies/<key>/versions {…, baseVersion 1, intent
activate}` → 201 version 2, for each of the nine (appendix rows 4 to 21). `GET /agent-policies/<key>` as the
Admin:

```
F FIN-0567 live v2 area Finance              outcome approve compiled.learning {'eligible': True}
O OPS-0541 live v2 area Operations           outcome approve compiled.learning {'eligible': True}
B OPS-0542 live v2 area Operations           outcome block   compiled.learning {'eligible': False}
L LEG-0001 live v2 area Legal and compliance outcome approve compiled.learning {'eligible': False}
```

On screen, the form's "Technical view (administrators)" showed the same `learning` value for F (true), B
(false), L (false) and T (true) (`01-technical-view-*.png`). Every one had `conflicts: []`, `suggestions: []`
and `secondReview.required: false`.

## Step 2: the backdated decisions. PASS

```
F FIN-0567 [(33, 510, 830, 3 deciders, '2026-08-26', '2026-10-05', firing ids 119527..119559)]
O OPS-0541 [(33, 510, 830, 3, …, 119560..119592)]      B OPS-0542 [(33, …, 119593..119625)]
L LEG-0001 [(33, …, 119626..119658)]                   H FIN-0568 [(33, …, 119659..119691)]
N OPS-0544 [(33, …, 119692..119724)]                   R OPS-0545 [(33, …, 119725..119757)]
S OPS-0546 [(33, …, 119758..119790)]                   T OPS-0543 [(29, 510, 790, 3, …, 119791..119819)]
```

## Step 3: the scan, narrowed to the demo policies. PASS

```
are they targets? {'F': True, 'O': True, 'B': False, 'L': False, 'T': True, 'H': True, 'N': True, 'R': True, 'S': True}
scan: 99 targets in the database, 7 of them mine [...] ->
  {'policies': 7, 'raised': 6, 'updated': 0, 'reopened': 0, 'weakened': 0, 'superseded': 0, 'errors': 0, 'unreadable': 0, 'skipped': 0}
```

Six rows, ids 2476 to 2481, one each for F, O, H, N, R, S: `threshold:args.demo_s5_amount:gt:500`, open,
confident, from 500 to 800, 1 band, 30 decisions, 30 approved, 3 approvers, 40 days, 0 reversed.

- **B** (block) and **L** (Legal: never suggest) are not targets at all, although each has the same 33
  approvals.
- **T** is a target and was scanned, and got no row: 29 decisions.
- The plan expected 2 cards (F and O). There are 6 because H, N, R and S need a card for their own steps.
- **Not run:** `learning.scan(conn, now)` with no `only`. It would have written rows for the other 92 targets.

## Step 4: the queue and the card. PASS

As the Admin (`04a-queue-six-cards.png`, `04b-F-form-card-above-conflicts.png`):

- Tabs: Agent policies | System settings | List | Inventory | Documents | Approvals | Conflicts |
  **Suggestions 6** | Notifications | Deciders | Business areas.
- Each card, for example: "FIN-0567 · DEMO stage 5 F 5d10a7 / **Raise the limit on Payment amount from 500 USD
  to 800 USD?** / 30 of 30 requests between 500 USD and 800 USD were approved, by 3 different people over 40
  days. / 30 decisions · 3 approvers · 40 days · typical decision time 1 min / **Suggestion confidence: High**
  / Draft this change / Not yet / Dismiss".
- F's form: heading "FIN-0567 · Version 2 · Active · **Extraction confidence: Medium**"; sections Identity,
  The policy, What happens, Applies to, Governance, **Suggested change**; the same card under it.
- B, L and T's forms have no "Suggested change" section.
- **Not shown literally:** "above Conflicts". F has no conflict, so there is no Conflicts section.

## Step 5: "Draft this change" on O, as a Buyer. PASS

`out/b05_buyer.txt`, `05a…05c*.png`. The Buyer sees no Deciders or Business areas tab, and the form offers
"Cancel | Save draft" only.

1. **A dirty form first.** Opened O's form, typed "an unsaved edit" into the change note, clicked the card's
   Draft this change. Toast: **"Save or cancel your edits first."** The limit was still 500 and the typed note
   still there. No request was sent. Cancel → "Edits discarded".
2. **Then the draft.** Clicked Draft this change on the clean form:
   ```
   requests since the click: ["GET /agent-policies/suggestions/2477/draft 200","POST /agent-policies/preview 201"]
   drafted form: {"limit":800,
     "situation":"From 01/04/2026, the agent is about to issue a demo payment above 800 USD.",
     "agent":"Demo payments over 800 USD need approval.", "checked":null,
     "note":"Suggested by learning: Raise the limit on Payment amount from 500 USD to 800 USD? Evidence: 30 of 30 requests between 500 USD and 800 USD were approved, by 3 different people over 40 days.",
     "examples":[799,800,801], "latest":2}
   examples on screen: Payment amount: 799 Nothing happens | 800 Nothing happens | 801 A person decides
   checkbox ticked: false
   notice on screen: Opened as a draft from a suggestion. Nothing is saved until you press Save draft.
                     The number was updated in: the situation, the message to the agent. Read it before you save.
                     Check the wording of: the message to the person.
   O after Draft (database): latest_version 2, live_version 2, 2 versions
   ```
   The only POST is `/agent-policies/preview`, which works out the examples and writes nothing. The date in
   the situation came through whole.

## Step 6: save the draft, then activate as an Approver. PASS

- **Buyer, Save draft:** toast "Saved as version 3"; `POST /agent-policies/OPS-0541/versions` 201.
  Database: `3:draft:demo-s5-buyer:800`; suggestion 2477 `saved`, `saved_version 3`. The card left the form.
  After Refresh the tab read **Suggestions 5** and O was not in the queue (`06a…png`).
- **P1 (Approver), on screen:** opened O ("Version 3 · Draft · Active: v2"; no Second review section), ticked
  the confirmation, Activate…: "Make this policy Active? / Agents will be held to it from now on." then
  "Confirm: activate OPS-0541?" → toast "Active. Saved as version 4". Database: `4:live:demo-s5-p1:800`
  (`06b…png`). **No second review was asked for: Operations.**
- **40 more approvals** (810…1200, on version 4) and a scan of O: `raised 1`. Row 2477 (500 → 800) is still
  `saved` and untouched. The new row 2482 is a different change, `…:gt:800`, 800 → 1100, 30 decisions. Nothing
  was raised for the old change.

## Step 6b: a saved draft that is put back. PASS

`out/s06b_S.txt`, through the gateway as the Buyer, on S:

| Request | Answer |
|---|---|
| `GET /agent-policies/suggestions/2481/draft` | 200, base v3, limit 800 |
| `POST /agent-policies/OPS-0546/versions {that form, baseVersion 3, intent draft}` | 201 version 4; row 2481 → `saved`, `saved_version 4`; gone from the queue |
| `GET /agent-policies/suggestions/2481/draft` | 409 `policy_changed` "This policy has changed since the suggestion was made." |
| scan of S | nothing changes: still `saved` (the latest draft holds the change) |
| `POST …/versions {the live form, limit 500, baseVersion 4, intent draft}` | 201 version 5 |
| scan of S | `reopened 1`; row 2481 `open`, `saved_version` NULL; back in the queue |

## Step 7: F in Finance, activation refused. PASS

P1, on screen (`out/b06_p1.txt`, `07-F-activation-refused.png`):

- From the queue, Draft this change on F: the form opened with the limit at 800.
- Save draft → "Saved as version 3" (`3:draft:demo-s5-p1:800`; suggestion 2476 `saved`). The form now shows
  "Second review: A second reviewer must review this change before it goes Active. Version 3" and the button
  "Record my review…".
- Ticked the confirmation, Activate…, both confirms: `POST /agent-policies/FIN-0567/versions` **422**
  ```
  [{"field":"secondReview","code":"second_review_required","message":"A second reviewer must review this change before it goes Active."}]
  ```
  The message showed in the form's error box, and **the focus moved into the Second review section**. The
  policy stayed at live v2, latest v3. Nothing was written.

## Step 8: P1 tries to review their own change. PASS

"Record my review…" → "Record your review of this version? / You are confirming you have read the change and
the evidence. You must not be the policy's owner or the person who saved it." → "Confirm: record your review
of this version?" → `POST /agent-policies/FIN-0567/second-review` **409**. Under the line: **"You saved this
change, so someone else must review it."** No review row (`08…png`).

## Step 9: the owner tries. PASS

As `demo-s5-owner` (linked by email to "Demo S5 Finance Owner"): the same two confirms, then 409. Under the
line: **"The policy's owner cannot be its second reviewer."** No review row (`09…png`).

## Step 10: P2 reviews, P1 activates. PASS

- **P2, on screen:** Record my review… → both confirms (nothing sent before the second yes) →
  `POST /agent-policies/FIN-0567/second-review` 201, then `GET /agent-policies/FIN-0567` 200. The section
  now reads **"Reviewed by demo-s5-p2 on 2026-10-10. Version 3"** and the button is gone (`10a…png`).
  Row: `('FIN-0567', 3, 2476, 'demo-s5-p2', '2026-10-10 12:23:19.841808+00', None)`.
- **P1, on screen:** the form showed the "Reviewed by demo-s5-p2…" line before anything was pressed
  (`10b…png`). Ticked, Activate…, both confirms → 201, toast "Active. Saved as version 4". Database:
  `1:draft:admin:500, 2:live:admin:500, 3:draft:p1:800, 4:live:p1:800` (`10c…png`).

## Step 11: H, the card dismissed, the limit changed by hand. PASS (through the gateway; see O1)

- **Buyer, on screen:** Dismiss on H → "Dismiss this suggestion? / It is not suggested again unless the
  evidence becomes much stronger." → "Confirm: dismiss this suggestion?" (no request before the second yes)
  → `POST /agent-policies/suggestions/2478/act` 201. Row 2478: `dismissed`, `n_at_action 30`, by
  `demo-s5-buyer` (`11a…png`).
- **P1, through the gateway** (`out/s06_scans.txt`, `out/s10_h_review.txt`):

| # | Request | Answer |
|---|---|---|
| 1 | save draft v3, limit **400** (tighter) | 201; the policy page says `secondReview {required: true, version: 3}` |
| 2 | activate that form | **422** `second_review_required` |
| 3 | save draft v4, limit **800** (looser, typed by hand) | 201; `secondReview {required: true, version: 4}` |
| 4 | activate that form | **422** `second_review_required` |
| 5 | P1: `POST /FIN-0568/second-review {version 4}` | 409 `own_change` |
| 6 | P2: `…/second-review {version 3}` | 409 `not_latest` "This is no longer the latest draft of the policy. Reload and try again." |
| 7 | P2: `…/second-review {version 4, note}` | 201 `{"reviewedBy":"demo-s5-p2", …}` |
| 8 | P2: the same again | 409 `already_reviewed` |
| 9 | P1: activate a form that is **not** the reviewed one (limit 900) | **422** `second_review_required` |
| 10 | P1: activate the reviewed form | 201 version 5 |

H's versions: `3:draft:p1:400, 4:draft:p1:800, 5:live:p1:800`. The dismissed card stayed dismissed throughout.

## Step 11b: a wording-only edit of H needs no review. PASS

P1, on screen: changed the situation to "The agent is about to issue a demo payment of more than 800 USD to a
supplier.", Save draft → version 6. The form showed **no Second review section**
(`secondReview {required: false}`). Ticked, Activate… → 201 "Active. Saved as version 7". The only review row
for H is still the one for version 4 (`11b…png`).

## Step 12: "Not yet". PASS

- **Buyer, on screen:** Not yet on N. No confirm. Toast **"Hidden until more decisions arrive."**
  `POST /agent-policies/suggestions/2479/act` 201. Row: `not_yet`, `n_at_action 30`.
- 26 more approvals (59 in all), scan: `raised 0, reopened 0`. Still `not_yet`, proposal still 500 → 800 on 30.
- 1 more (60 in all), scan: **`reopened 1`**. Row `open`, 500 → **1100**, 2 bands, 60 decisions. The card was
  back in the queue (`13b…png`, top card).

The plan's "29 more, then 1 more" is the same rule seen through the banding: the card returns when the
proposal's released band counts `n_at_action + 30`, which here is at the 60th decision.

## Step 13: a reversal. PASS

P1, on screen, on R (`13a…png`):

- "Recent decisions (20)", newest first. Each approved row has "Mark as reversed…".
- Picked "2026-10-05 · Approved by demo-s5-approver-c · args.demo_s5_amount 680". The confirm button was off
  until a reason was typed. Reason: "Demo: the payment was clawed back on 02/10/2026."
- "Mark this approval as reversed? / This cannot be undone. Learning will not suggest loosening the policy on
  the strength of it." → "Confirm: mark this approval as reversed?" (nothing sent before the second yes) →
  `POST /agent-policies/firings/119742/reversal` 201, then the firings were read again.
- The row: "… args.demo_s5_amount 680 · **Reversed on 2026-10-10: Demo: the payment was clawed back on
  02/10/2026.**" Database: `(119742, 'demo-s5-p1', '2026-10-10 12:20:02.107453+00', 'Demo: …')`.
- Scan of R: **`weakened 1`**. Row 2480: `open`, `confident false`, `reversed 1`.
- Queue card: `"suggestionConfidence": "Not enough yet", "canDraft": false`. On screen the tag is amber and
  the Draft button is disabled with the hint "There is not enough evidence yet"; a click on it does nothing
  (`13b…png`). `GET /agent-policies/suggestions/2480/draft` → 409 `not_confident`.
- A second reversal of the same row: 409 `already_reversed` "This request is already marked as reversed."
  Still one reversal row.

## Step 14: a Viewer. PASS

As `demo-s5-viewer`, with the page's role set to the Viewer group (`14a…14c*.png`):

- Tabs: no Deciders, no Business areas. **Suggestions 4** readable; 0 buttons on the cards.
- F's form (then a draft waiting for review): footer "Cancel" only; the Second review sentence with **no**
  "Record my review…".
- R's form: 20 decision rows, **0** "Mark as reversed…" buttons; the weak card with 0 buttons.
- Direct requests through the gateway (`s14_viewer_posts.py`):

| Request | Answer |
|---|---|
| `GET /agent-policies/suggestions` | 200, 4 cards |
| `GET /agent-policies/FIN-0567` | 200, no `compiled` in any version |
| `GET /agent-policies/suggestions/2479/draft` | 403 "needs the Buyer role or higher" |
| `POST …/suggestions/2479/act {not_yet}` and `{dismiss}` | 403, 403 |
| `POST /agent-policies/FIN-0567/second-review` | 403 "needs the Approver role or higher" |
| `POST /agent-policies/firings/119725/reversal` | 403 |
| `POST /agent-policies/FIN-0567/versions {draft}` | 403 |
| `PUT /agent-policies/taxonomy/Operations` | 403 "needs the Admin role or higher" |

Nothing changed in the database afterwards.

## Step 15: "Never suggest changes" for Operations. PASS, with the scan narrowed

Admin, on screen, Business areas (`15a…15e*.png`, `out/b15_admin.txt`):

- Ticking Operations' "Never suggest changes" put the box back at once and asked: "Change this setting? /
  Learning will stop suggesting changes to policies from this area." → "Confirm: change this setting?" (no PUT
  before the second yes) → `PUT /agent-policies/taxonomy/Operations` 200, `GET …/taxonomy` 200. The box was
  then ticked.
- Database: `('Operations', never_suggest True, …, 'demo-s5-admin', '2026-10-10 12:27:06')`.
- **`learning.targets(cur)`: 99 → 94.** Demo policies still targets: F and H (Finance) only. O, T, N, R, S
  (Operations) are no longer returned.
- **The closing step.** The scan's whole-database path (`learning.scan(conn, now)`, no `only`) was run once,
  with two changes made in my process only, after checking that all suggestion rows in the table were mine
  (0 others):
  1. `learning._scan_one` was wrapped so every non-demo policy is left unjudged (nothing is written for them);
  2. `learning._GONE_SQL` had one more condition added, `AND g.policy_key = ANY(<the nine demo keys>)`.
  Everything else is the product's own code and statement. Result: `{'policies': 94, …, 'superseded': 4,
  'errors': 0}`. Rows 2479 (N), 2480 (R), 2481 (S, hidden) and 2482 (O) became `superseded`. F's and O's
  `saved` rows were untouched.
- Queue on screen after Refresh: **"No suggestion is open."**, no count on the tab (`15c…png`).
- **Turned back off at once**, the same way on screen ("Learning will be able to suggest loosening policies
  from this area."). Database: `('Operations', never_suggest False, second_reviewer False, 'demo-s5-admin',
  '2026-10-10 12:27:20')`. `learning.targets`: 99 again.
- **Both audit rows** (`proc.bp_agent_actions`, `agent_policy.admin`, phase `act`):
  ```
  93571 12:27:06 before {neverSuggest: False, secondReviewer: False} after {neverSuggest: True,  secondReviewer: False} principal demo-s5-admin
  93573 12:27:20 before {neverSuggest: True,  secondReviewer: False} after {neverSuggest: False, secondReviewer: False} principal demo-s5-admin
  ```
  This also ran the taxonomy update's combined statement against a real database for the first time.
- A narrowed scan afterwards: `raised 2`. O (2482) and N (2479) came back. R's weak card and S's hidden
  card stayed `superseded` (see O6).

## Step 16: the scheduler job. PASS (by a snippet, not by a start-up)

I did not start a backend without the switch. A snippet (`s16_job.py`) built the real `BackendScheduler`
class without starting it and called the real registration:

```
AGENT_POLICY_LEARNING_SCAN=None:  registered=True {'name': 'agent-policy-learning-scan', 'interval': '1:00:00', 'lane': 'agent-policy-learning', 'next_run': 10 minutes later}
AGENT_POLICY_LEARNING_SCAN='off': registered=False   (log: "agent policy learning scan not registered (AGENT_POLICY_LEARNING_SCAN=off)")
called at start-up: True
```

The private backend's own start-up log carries the same "not registered" line.

## Step 17: the console. PASS

13 browser sessions (admin ×2, buyer ×3, p1 ×2, owner, viewer, p2, and 3 exploration passes), 729 gateway
requests seen by the browser.

- **0 `console.error` calls and 0 uncaught exceptions.**
- Every `/agent-policies` call: 113 GET 200, 35 POST 201, 2 PUT 200, and the three expected refusals (one 422,
  two 409).
- Network lines logged by the browser itself, not by page code: the three refusals above; and noise that is
  not this feature's: `GET /users/me/preferences` 404 (as in stages 1 to 4), a CORS refusal of
  `GET /user?page=1&limit=100`, and `GET /agents/AllAgents` 404 (new here: my demo subjects match no user row).

## Step 18: nothing became Active on its own. PASS

Every version, with who saved it (all are demo people, through the gateway; 0 saved by anything else):

```
F FIN-0567  1 draft admin 500 | 2 live admin 500 | 3 draft p1 800 (from the card) | 4 live p1 800 (after P2's review)
O OPS-0541  1 draft admin 500 | 2 live admin 500 | 3 draft buyer 800 (from the card) | 4 live p1 800
B OPS-0542  1 draft admin 500 | 2 live admin 500
L LEG-0001  1 draft admin 500 | 2 live admin 500
T OPS-0543  1 draft admin 500 | 2 live admin 500
H FIN-0568  1 draft admin 500 | 2 live admin 500 | 3 draft p1 400 | 4 draft p1 800 | 5 live p1 800 | 6 draft p1 800 (wording) | 7 live p1 800 (wording)
N OPS-0544  1 draft admin 500 | 2 live admin 500
R OPS-0545  1 draft admin 500 | 2 live admin 500
S OPS-0546  1 draft admin 500 | 2 live admin 500 | 3 live admin 500 (wording) | 4 draft buyer 800 | 5 draft buyer 500
```

N, R and T each had an open or weak suggestion for most of the run and stayed at version 2. Clean-up then
added one `retired` version to each policy.

## The output filter (ruling Q1). PASS

- **The reads pass the wording through.** As the Buyer, `GET /agent-policies/suggestions` carried
  `"policyName": "DEMO stage 5 S 5d10a7 (from 01/04/2026)"`, and `GET …/suggestions/2481/draft` carried the
  name and `"situation": "From 01/04/2026, the agent is about to issue a demo payment above 800 USD."`
  whole.
- **The POST answer is still scrubbed.** `POST …/suggestions/2481/act {not_yet}` → 201 with
  `"policyName": "[withheld]"`. The screen re-reads the queue after acting, so nobody sees this.
- Evidence carries across a wording-only version: after S's version 3 the scan answered `updated 1`, the row's
  `policy_version` moved to 3 and it still counted the 30 decisions made under version 2.

## Clean-up. DONE

- `POST /agent-policies/<key>/retire` for all nine → 201 each. All are `retired`, `live_version` NULL.
- Deleted: 2 second-review rows, 7 suggestion rows, 3 decider-map rows, 10 registry rows. Afterwards the
  suggestion and second-review tables are empty again, the decider map holds stage 3's three rows, the
  registry holds its 35 rows, and `learning.targets` is back to 92 with no demo key.
- **Left in `bp_testdb` as the record** (append-only by design, or never deleted):
  - 360 decision rows: FIN-0567 (33, ids 119527–119559), FIN-0568 (33), LEG-0001 (33), OPS-0541 (73),
    OPS-0542 (33), OPS-0543 (29), OPS-0544 (60), OPS-0545 (33), OPS-0546 (33); firing ids 119527 to 119886.
  - 1 reversal row: firing 119742 (OPS-0545).
  - The nine retired policies and their versions; the audit rows.
- Business areas: Finance and Legal unchanged apart from `last_number`; Operations `never_suggest` false,
  `second_reviewer` false, `last_modified_by demo-s5-admin`.
- Processes: only my four process groups were stopped. Ports 8010, 3011, 3010, 9223 closed.
  `procwise.service` is `active`. All three worktrees are clean.

## Defects found (recorded, not patched)

**D1. The "Second review" label runs into its sentence.**
- Where: `apLnReviewSection` in `engine.js`. The label `<span class="fm-lbl">` is followed directly by the
  sentence's `<span>` inside a `div.fm-row`.
- Seen: "Second reviewA second reviewer must review this change before it goes Active." and "Second
  reviewReviewed by demo-s5-p2 on 2026-10-10." (`08…png`, `09…png`, `10a…png`). The label is small and the
  sentence starts on the same line with no gap.
- To reproduce: open any Finance policy that has a suggestion row and a draft that changes its limit.
- Impact: display only.

**D2. The queue behind the form is stale after a save.**
- Seen: P1 opened F from the queue with Draft this change, pressed Save draft (the suggestion became `saved`
  in the database), and the queue behind the form still showed F's card with all three buttons and the tab
  still read "Suggestions 3" (`08…png`). It corrects itself on Refresh or a page reload.
- Cause: a save re-reads the policy and the list (`apReload`), not the suggestions (`AP.ln`).
- Impact: a person can click a card that is no longer open; the server then answers "This policy has changed
  since the suggestion was made." and the queue reloads. Nothing wrong is written.

**D3. "Reviewed by" shows a sign-in id, not a name.**
- Seen: "Reviewed by demo-s5-p2 on 2026-10-10." The value is `reviewed_by`, the reviewer's subject. In
  production that is a Cognito id such as `c2b5b404-40c1-7047-71c4-dd8093ecf25d`.
- The same holds for "Approved by …" in Recent decisions (the firing's `decided_by`).
- Impact: an owner cannot tell from the form who reviewed the change.

**D4. A weakened card does not say what weakened it.**
- Seen: after the reversal on R the card read "30 of 30 requests between 500 USD and 800 USD were approved,
  by 3 different people over 40 days. / 30 decisions · 3 approvers · 40 days · typical decision time 1 min /
  Suggestion confidence: Not enough yet". The row's stats hold `reversed: 1`, but the card's sentence and
  statistics do not mention it (`13b…png`).
- Impact: the owner sees evidence that looks perfect beside "Not enough yet", with no reason given.

**D5. No automated test proves "no suggestion for a block policy or a 'Never suggest changes' area".**
- The rule lives in one SQL statement, `learning._TARGETS_SQL`. The tests that call `learning.targets` either
  use a fake connection (`test_learning_scan.py`) or only check that an approve policy IS returned
  (`test_learning_endpoints_live.py`). Every scan test injects its targets, so the statement's two exclusions
  are never run against a database with a block policy or a never-suggest area in it.
- This run is the only proof: B and L, each with 33 approvals, were not targets (step 3), and Operations'
  five demo policies left the targets when its setting was turned on (step 15).
- To see the gap: remove `AND COALESCE(a.never_suggest, FALSE) = FALSE` from `_TARGETS_SQL` and run
  `LIVE=1 bptest tests/agent_policy -q`. I expect no test to go red; I did not try it (no product code was
  changed in this task).
- Impact: brief §8 test 19's clause has no guard against a later edit of that statement.

## Observations (not defects)

- **O1. The form has no control for the limit.** The condition can only change through "Draft this change" or
  "Ask the agent to fix it" (which calls the model, so it was not used). "Typing 800 by hand" therefore means a
  direct gateway request, as in step 11.
- **O2. A hidden card can come back bigger.** N was hidden as "500 → 800"; it came back as "500 → 1100",
  because the same row follows the evidence.
- **O3. Dismiss gives no toast**, and neither does a business-area switch. The card or the box simply changes.
- **O4. The gateway answers the 403s itself.** The backend's own role check was not reached through the
  gateway for these calls (it is covered by the backend tests).
- **O5. A dismissed card is superseded once the limit moves.** H's row went `dismissed` → `superseded` at the
  first scan after H went live at 800 (it keeps `n_at_action 30`).
- **O6. An area switched on and off loses some cards.** After step 15, R's weak card stayed `superseded` (no
  passing proposal brings it back) and S's hidden card stayed `superseded` with its count, so it waits the
  longer of the two waits.
- **O7. One Admin retired both protected Finance policies at clean-up** with no second person. This is the
  known limit the build recorded: retiring needs only an Approver.
- **O8. The health line counts approved requests as "paused"**: "Checked 19 times in the last 30 days: 0
  allowed, 19 paused, 0 blocked" for 19 backdated approvals.
- **O9. The "Not yet" record keeps its actor after the card returns**: `acted_by demo-s5-buyer` with
  `n_at_action` NULL on the reopened row.
- **O10. The refusal for a review of an older draft** (`not_latest`) says "Reload and try again", which is
  right for a stale page but not for a caller who named the wrong version.
- **O11. The draft asked the owner to check "the message to the person"** although that message holds no
  number. That is the rule working as ruled (wording with no single match is flagged), but on a real form it
  will be listed often.
- **O12. `AGENT_POLICY_LEARNING_SCAN=off` is needed on any server pointed at `bp_testdb`.** That database now
  holds 92 "live" test policies that are scan targets.

## Test suites, against baseline

Run after clean-up, same interpreter (`venv/bin/python`), `.env` and day.

| Repo | Suite | Baseline | This run |
|---|---|---|---|
| BP | `bptest tests/agent_policy tests/engines/test_decide_live_conflict.py -q` (fake DB) | 1 failed, 818 passed | **1 failed, 873 passed**, 389 skipped |
| BP | the same with `LIVE=1` (bp_testdb) | 1 failed, 1257 passed | **1 failed, 1257 passed**, 5 skipped (10 min 57 s) |
| Gateway | `npm test` | 388 passed | 17 suites, **388 passed** |
| UI | `npx vitest run` | 7 known failures | 2 files failed / 164 passed; **7 failed, 3545 passed** |

- BP: the one failure in both runs is the known
  `tests/agent_policy/test_approvals_job.py::test_conflict_scan_has_its_own_lane`.
- The fake-DB count is 873 where the dispatch quoted 818. The failure is the same single one, and 873 passed +
  389 skipped + 1 failed is the same 1,263 tests as the live run (1257 + 5 + 1). I take the 818 to be a count
  from before the last fix rounds added tests; I did not re-measure the base commit.
- UI: the 7 are the known ones, 4 in `atb/composedPages.contract.test.js` and 3 in
  `lib/i18n/language.contract.test.js`.
- After the live run the suggestion and second-review tables were empty again.

## Red/green captures

Guards broken on purpose during the build, each restored and re-run green (from the task reports):

| Task | Guard broken on purpose | Red | Restored |
|---|---|---|---|
| 2 | Wilson comparison replaced by the plain approval rate | `test_each_guardrail_alone[…]`: `assert [] == ['wilson_lower']`; 1 failed, 17 passed | 18 passed |
| 2 (fix) | rounded Wilson compared again | `test_the_wilson_guardrail_is_judged_before_rounding`; 1 failed, 86 passed | 88 passed |
| 3 | rounding direction swapped in `_nice` | 3 failed, 10 passed (840 for 830; 810 for 803.9; 630 for 629) | 13 passed |
| 3 (fix) | the band walk keeps going past a failed band | 5 failed, 82 passed (set 3; a rejection in the first band; in the second band; good bands beyond a failed one; the days guardrail per band) | 88 passed |
| 3 (fix) | ties split between bands | the original tie test did **not** go red (it checked nothing); a new test was added and fails `assert (760, 1, 50, 49) == (790, 2, 71, 70)` | 88 passed |
| 5 | one decision per approval case (`DISTINCT ON`) removed | a suggestion to 800 raised from 6 decisions; the sibling test counted 60 for 30 | 14 passed |
| 5 | reversal on a linked row; blank decider; `system:` decider; the `saved` skip | one named test red for each | 14 passed |
| 5 (fix) | reopen does not set confident; closing statement loses the real-key pattern; closing statement also takes `saved` | one named test red for each (a fourth break, hard-coding `kind`, went green and is not relied on) | green |
| 8 | own-change check; owner check; fail-closed answer; area read; owner-unmapped check | "DID NOT RAISE" in the named test for each | green |
| 8 (fix 3) | `hidden` dropped from the compared parts; plain `==` for canonical text; leaving-protection clause skipped; only non-superseded suggestions counted | pure 31 failed / 4 failed / 1 failed; live routes 1, 8, 9, 4, 3, 2: `DID NOT RAISE NotReady` | green |
| 8 (fix 4) | `source` removed from the compared parts | pure 3 failed, 83 passed; live route 11 `DID NOT RAISE NotReady` | green |

This run added none: I changed no product code.

## Diff summary (ruling A)

### BP_Backend: `2b5c6764..f0a55677` (19 commits; 27 files, +7921 / −15; nothing under `.superpowers/`)

```
f0a55677 fix(agent-policy): a policy's source document and start date are part of what it enforces
098147e5 fix(agent-policy): once learning has spoken, any change to what a protected policy enforces needs a second person
fda4f08e fix(agent-policy): the second review also covers missing-data loosenings and policies that read a hidden value
095dc35f fix(agent-policy): the second review is triggered by what the change stops asking people about, not by the condition's shape
46b4d16a feat(agent-policy): a suggested loosening in a second-reviewer area needs a second person before it goes Active
a90545e8 fix(agent-policy): a reversal lands on the approved row a person decided, never on a linked notify row
50646dcb fix(agent-policy): one unreadable suggestion row no longer stops a policy's scan, and the scan judges 'in force' under its lock
dea07962 feat(agent-policy): an approver records that an approved action was reversed
75bbcc26 fix(agent-policy): a saved suggestion comes back when no version in force holds its change
d9dd58a7 feat(agent-policy): the suggestion reads pass the customer's own policy wording through (ruling Q1)
ca68cf1a feat(agent-policy): the suggestion card, the queue, and Draft / Not yet / Dismiss
742f640b fix(agent-policy): a dismissed suggestion survives a superseded round trip; newest evidence is kept at the cap
e80bffc0 feat(agent-policy): the hourly learning scan raises suggestions from people's decisions
2fa17104 fix(agent-policy): a suggested limit needs evidence at its far end, exact band edges and unrounded Wilson
0838b229 feat(agent-policy): a suggestion becomes an unsaved draft form, built by code
859fbd18 feat(agent-policy): what the evidence supports loosening, never beyond what people decided
80e81720 feat(agent-policy): the learning statistics and guardrails, with the Wilson lower bound
e1e1119c feat(agent-policy): tables for learning suggestions, second reviews and reversals
3441ab6e docs(agent-policy): stage 5 (learning) implementation plan
```

| File | Change |
|---|---|
| `deploy/sql/2026-10-15_bp_agent_policy_learning.sql` (+ rollback) | new: `bp_agent_policy_suggestion`, `bp_agent_policy_second_review`, `bp_policy_firing_reversal` (append-only) |
| `src/services/agent_policy/learning_stats.py` | new, 84 lines: Wilson bound, the statistics, the guardrails |
| `src/services/agent_policy/learning_patterns.py` | new, 226 lines: what can be loosened, the bands, what the evidence supports |
| `src/services/agent_policy/learning_draft.py` | new, 165 lines: a suggestion as an unsaved form |
| `src/services/agent_policy/learning.py` | new, 674 lines: decisions, scan, cards, act, draft, note_save, record_reversal |
| `src/services/agent_policy/second_review.py` | new, 357 lines: the any-change rule, record, status |
| `src/repositories/agent_policy_repo.py` | +34: second-review check on activate, `note_save` after a save, `suggestions` and `secondReview` on the policy read |
| `src/api/routers/agent_policies.py` | +184: six endpoints; the taxonomy update is audited with before and after |
| `src/services/agent_policy/approval_views.py` | +32: `reversed` on a policy's firings |
| `src/services/backend_scheduler.py` | +42: the hourly learning scan on its own lane |
| `src/api/main.py` | +5: two pass-through entries (ruling Q1) |
| `tests/agent_policy/…` | 14 new test files and `test_router.py` |
| `specs/2026-10-10-agent-policy-governance-plan-5-learning.md` | the plan |

### Gateway: `657d5d8..2c5f799` (1 commit; 3 files, +252)

```
2c5f799 feat(agent-policy): gateway routes for learning suggestions, reversals and the second review
```

`agent-policy.controller.ts` +50 (five forwarding routes: list suggestions, draft, act, reversal, second
review; role floors Viewer / Buyer / Buyer / Approver / Approver), `agent-policy.yml` +125 (five function
blocks), `agent-policy.controller.spec.ts` +77.

### UI: `9ad813d..5251664` (6 commits; 6 files, +2148 / −6)

```
5251664 fix(agent-policy): the second-review wording says any change, not only a suggested loosening
088d2ce feat(agent-policy): an administrator sets Never suggest changes and the second reviewer per business area
8ffffb0 fix(agent-policy): the decisions list is refreshed by a plain call, and the review confirm title is a literal
5bb0cf6 feat(agent-policy): the second review, recent decisions and marking an approval as reversed
58e8104 fix(agent-policy): Draft this change never overwrites unsaved edits, and says which wording it changed
c140e66 feat(agent-policy): the suggestion card and the Suggestions queue
```

New files: `agentPolicy/suggestions.js` (pure, 159 lines), `suggestions.test.js`,
`engineWiring.stage5.contract.test.js`; `index.jsx` (+3 −1: the import and `suggestions` on
`window.__SPENDIQ_AP__`); `model.test.js` +7.

**`engine.js`** (`git diff -U0 9ad813d..5251664`: 9 hunks, +388 −5).

Functions added, all in one new block "Agent policies (stage 5): suggested changes" (lines 24744–25121),
placed directly before "The small edit form":

| Function | What it does |
|---|---|
| `apLnLib` | the pure module |
| `apLnLoad` | `GET /agent-policies/suggestions?status=open` into `AP.ln` |
| `apLnBind` | one delegated click handler for every stage 5 control, one `change` handler for the area boxes, one `input` handler for the reversal reason |
| `apLnOpenState`, `apLnDirty`, `apLnDrafted` | the open form; whether it holds unsaved edits; whether it still holds a draft from a card |
| `apLnCardHTML`, `apLnHTML`, `apLnPolicySection` | the card; the queue; the form's "Suggested change" section |
| `apLnDraft` | "Draft this change": refuses a dirty form, reads the draft, fills the form, saves nothing |
| `apLnAct` | Not yet (no confirm) and Dismiss (two-step) |
| `apLnReviewSection`, `apLnRecordReview` | the Second review line and "Record my review…" (two-step) |
| `apLnDecisionRows`, `apLnReasonReady`, `apLnDecisionsBody`, `apLnDecisionsRefresh`, `apLnDecisionsHTML` | "Recent decisions" |
| `apLnReverse` | "Mark as reversed" (reason, two-step) |
| `apLnAreasLoad`, `apLnAreasHTML`, `apLnAreaToggle` | the Business areas table and its two switches (two-step) |

Constants added in the block: `lnAreaFields`, `AP_LN_DECISIONS_MAX`.

Lines touched inside existing functions:

| Function | Lines |
|---|---|
| `apPoliciesTab` | **added:** `if(apLnLib()&&AP.ln==null&&!AP.lnErr) apLnLoad();` and `if(AP.view==='areas'&&!apIsAdmin()) AP.view='list';`. **changed:** the counts line gained `nLn`; the `views` line gained the `suggestions` entry and, for an Admin, the `areas` entry; the tab template gained the two labels and the open-suggestions count; the `body` line gained the `areas` and `suggestions` branches |
| `apFormHTML` | **changed:** `+apPcPolicySection(state)` became `+apLnReviewSection(state)+apLnPolicySection(state)+apPcPolicySection(state)`. **added:** `+apLnDecisionsHTML(state)` after the health line |
| `apLoadHealth` | **added:** `state.firings=(r&&r.firings)||[];` and `try{ apLnDecisionsRefresh(state); }catch(e){ console.warn(e); }` |

`policyEdit`, `policyDelete` and `openFormModal` are byte-identical (their SHA-256 hashes are pinned in the
stage 4 and stage 5 contract tests, both green in this run). `apHealthHTML` is untouched.

## Brief §8 test 19, clause by clause

"Learning raises a suggestion only when every guardrail holds. Draft, Not yet and Dismiss behave as specified.
There is no suggestion for block or 'Never suggest changes' policies, and none after the change is saved. A
suggested value never goes beyond the evidence."

| Clause | Test that proves it | In this run |
|---|---|---|
| A suggestion only when every guardrail holds | `test_learning_stats.py::test_each_guardrail_alone`, `::test_exactly_on_each_threshold_passes`, `::test_the_wilson_guardrail_is_judged_before_rounding`; `test_learning_patterns.py::test_too_little_evidence_proposes_nothing`, `::test_a_band_must_meet_the_days_guardrail_on_its_own`; `test_learning_scan_live.py::test_thirty_approvals_by_three_people_over_thirty_days_raise_one_suggestion`, `::test_a_reversal_blocks_the_band_and_shows_not_enough_yet`, `::test_a_non_numeric_setting_raises_no_suggestion` | steps 3 (T: 29), 13 |
| Only people's decisions, one per case | `test_learning_scan_live.py::test_one_decision_on_several_linked_rows_counts_once`, `::test_timeouts_and_system_decisions_are_not_evidence`, `::test_a_masked_value_is_not_evidence` | tests only |
| Draft: unsaved, condition and situation updated, examples recomputed, confirmation cleared, note filled | `test_learning_draft.py::test_the_draft_changes_the_condition_the_wording_and_clears_the_confirmation`, `::test_the_examples_sit_on_the_new_boundary_and_are_computed_by_code`; `test_learning_endpoints_live.py::test_draft_returns_a_form_and_writes_nothing`, `::test_nothing_in_this_stage_activates_a_policy`; UI `engineWiring.stage5.contract.test.js` | step 5 |
| Draft off when the evidence is weak | `test_learning_cards_live.py::test_a_weak_card_cannot_be_drafted`, `::test_only_an_open_confident_card_can_be_drafted`; `test_learning_endpoints_live.py::test_a_weak_card_says_not_enough_yet_with_todays_evidence_and_refuses_the_draft` | step 13 |
| Not yet: hidden until 30 more | `test_learning_scan_live.py::test_not_yet_hides_until_thirty_more_decisions`; `test_learning_cards_live.py::test_not_yet_and_dismiss_record_who_when_and_how_many` | step 12 |
| Dismiss: not again unless materially stronger | `test_learning_scan_live.py::test_dismissed_returns_only_on_materially_stronger_evidence`, `::test_a_dismissed_card_does_not_return_when_the_condition_goes_away_and_comes_back`; `test_learning_cards_live.py::test_dismiss_closes_the_card_and_the_queue_shows_it_only_under_all` | step 11 (dismissed; its return was not shown) |
| None for block | **No automated test (D5).** The rule is the line `form_state ->> 'outcome' = 'approve'` in `learning._TARGETS_SQL`. `test_learning_scan_live.py::test_cards_of_policies_no_longer_scanned_are_closed_and_no_others` covers only the closing of cards once a policy is no longer a target | **steps 1, 3 (B): this run is the proof** |
| None for "Never suggest changes" | **No automated test (D5).** The rule is `COALESCE(a.never_suggest, FALSE) = FALSE` and the compiled `learning.eligible` in the same statement; `test_compiler.py` covers only the compiled flag | **steps 1, 3 (L), 15: this run is the proof** |
| None after the change is saved | `test_learning_endpoints_live.py::test_saving_the_draft_marks_the_suggestion_saved_and_it_is_not_suggested_again`; `test_learning_scan_live.py::test_a_saved_suggestion_stays_saved_while_a_version_in_force_holds_it_and_returns_when_none_does`; `test_learning_cards_live.py::test_a_save_marks_the_suggestion_saved_only_when_the_saved_form_holds_the_change` | steps 6, 6b |
| Never beyond the evidence | `test_learning_patterns.py::test_raise_the_limit_to_what_people_approved_rounded_inside_the_evidence`, `::test_never_beyond_the_evidence_when_rounding_would_drop_below_the_guardrails`, `::test_set_1_a_single_far_outlier_is_not_evidence`, `::test_set_5_one_decision_beyond_the_band_does_not_move_the_limit`, `::test_a_rejection_in_the_range_stops_the_suggestion_below_it`, `::test_group_for_is_exactly_what_the_change_releases` | steps 3, 6, 12 (800 with approvals up to 830; 1100 with approvals up to 1200) |
| Design §2: a second person in Finance, never the owner | `test_second_review_live.py::test_finance_activation_of_the_suggested_loosening_is_refused_without_a_review`, `::test_the_person_who_saved_the_draft_cannot_review_it`, `::test_the_owner_cannot_review_it`, `::test_after_another_approver_reviews_it_activation_succeeds`, `::test_typing_the_loosening_by_hand_still_needs_the_review`, `::test_route_2_a_tightening_needs_the_review_itself_so_the_two_step_is_closed_at_step_one`, `::test_a_wording_only_edit_needs_no_review_in_finance`, `::test_operations_needs_no_second_review_whatever_changes` | steps 7 to 11b, 6 |
| §2: nothing becomes Active automatically | `test_learning_endpoints_live.py::test_nothing_in_this_stage_activates_a_policy` | step 18 |
| §3.6: the two confidences are never confused | `test_learning_cards_live.py::test_the_card_names_suggestion_confidence_and_never_extraction_confidence`; UI `suggestions.test.js` | step 4 |

## Appendix: every gateway request the driver sent in this run

The browser's own requests (729, all listed in the scratch file `out/browser.jsonl`) are summarised in each
step above. Long answers are cut at 230 characters.

| # | UTC | As | Request | Answer |
|---|---|---|---|---|
| 1 | 12:09:13 | admin | `PUT /agent-policies/deciders/Demo%20S5%20Finance%20Owner` `{"groups": [], "emails": ["owner@demo-s5.example"], "notes": "Stage 5 demo (2026-10-10)"}` | 200 `{"name": "Demo S5 Finance Owner", "savedAt": "2026-10-10T12:09:13.204583+00:00"}` |
| 2 | 12:09:13 | admin | `PUT /agent-policies/deciders/Demo%20S5%20Ops%20Owner` `{"groups": [], "emails": ["opsowner@demo-s5.example"], "notes": "Stage 5 demo (2026-10-10)"}` | 200 `{"name": "Demo S5 Ops Owner", "savedAt": "2026-10-10T12:09:13.695189+00:00"}` |
| 3 | 12:09:14 | admin | `PUT /agent-policies/deciders/Demo%20S5%20Decider` `{"groups": [], "emails": ["decider@demo-s5.example"], "notes": "Stage 5 demo (2026-10-10)"}` | 200 `{"name": "Demo S5 Decider", "savedAt": "2026-10-10T12:09:14.052823+00:00"}` |
| 4 | 12:09:30 | admin | `POST /agent-policies` {form "DEMO stage 5 F 5d10a7", limit 500} | 201 `{"policyKey": "FIN-0567", "version": 1}` |
| 5 | 12:09:31 | admin | `POST /agent-policies/FIN-0567/versions` {form "DEMO stage 5 F 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "FIN-0567", "version": 2}` |
| 6 | 12:09:33 | admin | `POST /agent-policies` {form "DEMO stage 5 O 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0541", "version": 1}` |
| 7 | 12:09:36 | admin | `POST /agent-policies/OPS-0541/versions` {form "DEMO stage 5 O 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0541", "version": 2}` |
| 8 | 12:09:38 | admin | `POST /agent-policies` {form "DEMO stage 5 B 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0542", "version": 1}` |
| 9 | 12:09:40 | admin | `POST /agent-policies/OPS-0542/versions` {form "DEMO stage 5 B 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0542", "version": 2}` |
| 10 | 12:09:42 | admin | `POST /agent-policies` {form "DEMO stage 5 L 5d10a7", limit 500} | 201 `{"policyKey": "LEG-0001", "version": 1}` |
| 11 | 12:09:44 | admin | `POST /agent-policies/LEG-0001/versions` {form "DEMO stage 5 L 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "LEG-0001", "version": 2}` |
| 12 | 12:09:46 | admin | `POST /agent-policies` {form "DEMO stage 5 T 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0543", "version": 1}` |
| 13 | 12:09:48 | admin | `POST /agent-policies/OPS-0543/versions` {form "DEMO stage 5 T 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0543", "version": 2}` |
| 14 | 12:09:50 | admin | `POST /agent-policies` {form "DEMO stage 5 H 5d10a7", limit 500} | 201 `{"policyKey": "FIN-0568", "version": 1}` |
| 15 | 12:09:52 | admin | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "FIN-0568", "version": 2}` |
| 16 | 12:09:54 | admin | `POST /agent-policies` {form "DEMO stage 5 N 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0544", "version": 1}` |
| 17 | 12:09:55 | admin | `POST /agent-policies/OPS-0544/versions` {form "DEMO stage 5 N 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0544", "version": 2}` |
| 18 | 12:09:57 | admin | `POST /agent-policies` {form "DEMO stage 5 R 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0545", "version": 1}` |
| 19 | 12:09:59 | admin | `POST /agent-policies/OPS-0545/versions` {form "DEMO stage 5 R 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0545", "version": 2}` |
| 20 | 12:10:01 | admin | `POST /agent-policies` {form "DEMO stage 5 S 5d10a7", limit 500} | 201 `{"policyKey": "OPS-0546", "version": 1}` |
| 21 | 12:10:03 | admin | `POST /agent-policies/OPS-0546/versions` {form "DEMO stage 5 S 5d10a7", limit 500, baseVersion 1, intent activate} | 201 `{"policyKey": "OPS-0546", "version": 2}` |
| 22 | 12:10:03 | admin | `GET /agent-policies/FIN-0567` | 200 policy read (live, live v2, latest v2) |
| 23 | 12:10:04 | admin | `GET /agent-policies/OPS-0541` | 200 policy read (live, live v2, latest v2) |
| 24 | 12:10:04 | admin | `GET /agent-policies/OPS-0542` | 200 policy read (live, live v2, latest v2) |
| 25 | 12:10:04 | admin | `GET /agent-policies/LEG-0001` | 200 policy read (live, live v2, latest v2) |
| 26 | 12:10:04 | admin | `GET /agent-policies/OPS-0543` | 200 policy read (live, live v2, latest v2) |
| 27 | 12:10:04 | admin | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v2) |
| 28 | 12:10:05 | admin | `GET /agent-policies/OPS-0544` | 200 policy read (live, live v2, latest v2) |
| 29 | 12:10:05 | admin | `GET /agent-policies/OPS-0545` | 200 policy read (live, live v2, latest v2) |
| 30 | 12:10:05 | admin | `GET /agent-policies/OPS-0546` | 200 policy read (live, live v2, latest v2) |
| 31 | 12:10:23 | admin | `GET /agent-policies/suggestions` | 200 6 cards: 2481 OPS-0546 (High), 2480 OPS-0545 (High), 2479 OPS-0544 (High), 2478 FIN-0568 (High), 2477 OPS-0541 (High), 2476 FIN-0567 (High) |
| 32 | 12:15:29 | admin | `GET /agent-policies/OPS-0546` | 200 policy read (live, live v2, latest v2) |
| 33 | 12:15:30 | admin | `POST /agent-policies/OPS-0546/versions` {form "DEMO stage 5 S 5d10a7 (from 01/04/2026)", limit 500, baseVersion 2, intent activate} | 422 `{"problems": [{"field": "checked", "code": "not_confirmed", "message": "Confirm the examples and how it is enforced."}]}` |
| 34 | 12:15:35 | buyer | `GET /agent-policies/suggestions` | 200 6 cards: 2481 OPS-0546 (High), 2480 OPS-0545 (High), 2479 OPS-0544 (High), 2478 FIN-0568 (High), 2477 OPS-0541 (High), 2476 FIN-0567 (High) |
| 35 | 12:15:35 | buyer | `GET /agent-policies/suggestions/2481/draft` | 200 draft form for OPS-0546, base v2, limit 800 |
| 36 | 12:15:45 | admin | `GET /agent-policies/OPS-0546` | 200 policy read (live, live v2, latest v2) |
| 37 | 12:15:46 | admin | `POST /agent-policies/OPS-0546/versions` {form "DEMO stage 5 S 5d10a7 (from 01/04/2026)", limit 500, baseVersion 2, intent activate} | 201 `{"policyKey": "OPS-0546", "version": 3}` |
| 38 | 12:15:52 | buyer | `GET /agent-policies/suggestions` | 200 6 cards: 2481 OPS-0546 (High), 2480 OPS-0545 (High), 2479 OPS-0544 (High), 2478 FIN-0568 (High), 2477 OPS-0541 (High), 2476 FIN-0567 (High) |
| 39 | 12:15:52 | buyer | `GET /agent-policies/suggestions/2481/draft` | 200 draft form for OPS-0546, base v3, limit 800 |
| 40 | 12:20:56 | p1 | `POST /agent-policies/firings/119742/reversal` `{"reason": "a second try"}` | 409 `{"code": "already_reversed", "message": "This request is already marked as reversed."}` |
| 41 | 12:20:56 | p1 | `GET /agent-policies/suggestions` | 200 3 cards: 2480 OPS-0545 (Not enough yet), 2482 OPS-0541 (High), 2481 OPS-0546 (High) |
| 42 | 12:20:56 | p1 | `GET /agent-policies/suggestions/2480/draft` | 409 `{"code": "not_confident", "message": "There is not enough evidence yet."}` |
| 43 | 12:20:59 | p1 | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v2) |
| 44 | 12:21:01 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 400, baseVersion 2, intent draft} | 201 `{"policyKey": "FIN-0568", "version": 3}` |
| 45 | 12:21:01 | p1 | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v3) |
| 46 | 12:21:02 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 400, baseVersion 3, intent activate} | 422 `{"problems": [{"field": "secondReview", "code": "second_review_required", "message": "A second reviewer must review this change before it goes Active."}]}` |
| 47 | 12:21:02 | p1 | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v3) |
| 48 | 12:21:04 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 800, baseVersion 3, intent draft} | 201 `{"policyKey": "FIN-0568", "version": 4}` |
| 49 | 12:21:04 | p1 | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v4) |
| 50 | 12:21:05 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 800, baseVersion 4, intent activate} | 422 `{"problems": [{"field": "secondReview", "code": "second_review_required", "message": "A second reviewer must review this change before it goes Active."}]}` |
| 51 | 12:21:05 | p1 | `POST /agent-policies/FIN-0568/second-review` `{"version": 4}` | 409 `{"code": "own_change", "message": "You saved this change, so someone else must review it."}` |
| 52 | 12:22:47 | viewer | `GET /agent-policies/FIN-0567` | 200 policy read (live, live v2, latest v3) |
| 53 | 12:22:48 | viewer | `GET /agent-policies/suggestions` | 200 4 cards: 2479 OPS-0544 (High), 2480 OPS-0545 (Not enough yet), 2482 OPS-0541 (High), 2481 OPS-0546 (High) |
| 54 | 12:22:48 | viewer | `GET /agent-policies/suggestions/2479/draft` | 403 `{"message": "needs the Buyer role or higher", "error": "Forbidden", "statusCode": 403}` |
| 55 | 12:22:48 | viewer | `POST /agent-policies/suggestions/2479/act` `{"verb": "not_yet"}` | 403 `{"message": "needs the Buyer role or higher", "error": "Forbidden", "statusCode": 403}` |
| 56 | 12:22:48 | viewer | `POST /agent-policies/suggestions/2479/act` `{"verb": "dismiss"}` | 403 `{"message": "needs the Buyer role or higher", "error": "Forbidden", "statusCode": 403}` |
| 57 | 12:22:48 | viewer | `POST /agent-policies/FIN-0567/second-review` `{"version": 3}` | 403 `{"message": "needs the Approver role or higher", "error": "Forbidden", "statusCode": 403}` |
| 58 | 12:22:48 | viewer | `POST /agent-policies/firings/119725/reversal` `{"reason": "viewer try"}` | 403 `{"message": "needs the Approver role or higher", "error": "Forbidden", "statusCode": 403}` |
| 59 | 12:22:48 | viewer | `POST /agent-policies/FIN-0567/versions` {form "DEMO stage 5 F 5d10a7", limit 800, baseVersion 3, intent draft} | 403 `{"message": "needs the Buyer role or higher", "error": "Forbidden", "statusCode": 403}` |
| 60 | 12:22:48 | viewer | `PUT /agent-policies/taxonomy/Operations` `{"subAreas": ["General"], "neverSuggest": true, "secondReviewer": false}` | 403 `{"message": "needs the Admin role or higher", "error": "Forbidden", "statusCode": 403}` |
| 61 | 12:23:41 | p2 | `POST /agent-policies/FIN-0568/second-review` `{"version": 3}` | 409 `{"code": "not_latest", "message": "This is no longer the latest draft of the policy. Reload and try again."}` |
| 62 | 12:23:41 | p2 | `POST /agent-policies/FIN-0568/second-review` `{"version": 4, "note": "Read the change and the evidence."}` | 201 `{"policyKey": "FIN-0568", "version": 4, "reviewedBy": "demo-s5-p2", "reviewedAt": "2026-10-10T12:23:41.264525+00:00"}` |
| 63 | 12:23:42 | p2 | `POST /agent-policies/FIN-0568/second-review` `{"version": 4}` | 409 `{"code": "already_reviewed", "message": "This draft has already been reviewed."}` |
| 64 | 12:23:46 | p1 | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v2, latest v4) |
| 65 | 12:23:46 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 900, baseVersion 4, intent activate} | 422 `{"problems": [{"field": "secondReview", "code": "second_review_required", "message": "A second reviewer must review this change before it goes Active."}]}` |
| 66 | 12:23:48 | p1 | `POST /agent-policies/FIN-0568/versions` {form "DEMO stage 5 H 5d10a7", limit 800, baseVersion 4, intent activate} | 201 `{"policyKey": "FIN-0568", "version": 5}` |
| 67 | 12:25:39 | buyer | `GET /agent-policies/suggestions/2481/draft` | 200 draft form for OPS-0546, base v3, limit 800 |
| 68 | 12:25:40 | buyer | `POST /agent-policies/OPS-0546/versions` {form "DEMO stage 5 S 5d10a7 (from 01/04/2026)", limit 800, baseVersion 3, intent draft} | 201 `{"policyKey": "OPS-0546", "version": 4}` |
| 69 | 12:25:40 | buyer | `GET /agent-policies/suggestions` | 200 3 cards: 2479 OPS-0544 (High), 2480 OPS-0545 (Not enough yet), 2482 OPS-0541 (High) |
| 70 | 12:25:40 | buyer | `GET /agent-policies/suggestions/2481/draft` | 409 `{"code": "policy_changed", "message": "This policy has changed since the suggestion was made."}` |
| 71 | 12:25:42 | buyer | `GET /agent-policies/OPS-0546` | 200 policy read (live, live v3, latest v4) |
| 72 | 12:25:44 | buyer | `POST /agent-policies/OPS-0546/versions` {form "DEMO stage 5 S 5d10a7 (from 01/04/2026)", limit 500, baseVersion 4, intent draft} | 201 `{"policyKey": "OPS-0546", "version": 5}` |
| 73 | 12:25:46 | buyer | `GET /agent-policies/suggestions` | 200 4 cards: 2481 OPS-0546 (High), 2479 OPS-0544 (High), 2480 OPS-0545 (Not enough yet), 2482 OPS-0541 (High) |
| 74 | 12:25:46 | buyer | `POST /agent-policies/suggestions/2481/act` `{"verb": "not_yet"}` | 201 `{"id": 2481, "policyKey": "OPS-0546", "policyName": "[withheld]", "status": "not_yet", "kind": "threshold", "change": "Raise the limit on Payment amount from 500 USD to 800 USD?", "evidence": "30 of 30 requests between 500 USD and` |
| 75 | 12:25:47 | buyer | `POST /agent-policies/suggestions/2481/act` `{"verb": "not_yet"}` | 409 `{"code": "not_open", "message": "This suggestion is no longer open."}` |
| 76 | 12:28:23 | admin | `GET /agent-policies/FIN-0567` | 200 policy read (live, live v4, latest v4) |
| 77 | 12:28:23 | admin | `POST /agent-policies/FIN-0567/retire` `{"baseVersion": 4, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "FIN-0567", "version": 5}` |
| 78 | 12:28:23 | admin | `GET /agent-policies/OPS-0541` | 200 policy read (live, live v4, latest v4) |
| 79 | 12:28:24 | admin | `POST /agent-policies/OPS-0541/retire` `{"baseVersion": 4, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0541", "version": 5}` |
| 80 | 12:28:24 | admin | `GET /agent-policies/OPS-0542` | 200 policy read (live, live v2, latest v2) |
| 81 | 12:28:24 | admin | `POST /agent-policies/OPS-0542/retire` `{"baseVersion": 2, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0542", "version": 3}` |
| 82 | 12:28:25 | admin | `GET /agent-policies/LEG-0001` | 200 policy read (live, live v2, latest v2) |
| 83 | 12:28:25 | admin | `POST /agent-policies/LEG-0001/retire` `{"baseVersion": 2, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "LEG-0001", "version": 3}` |
| 84 | 12:28:25 | admin | `GET /agent-policies/OPS-0543` | 200 policy read (live, live v2, latest v2) |
| 85 | 12:28:26 | admin | `POST /agent-policies/OPS-0543/retire` `{"baseVersion": 2, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0543", "version": 3}` |
| 86 | 12:28:26 | admin | `GET /agent-policies/FIN-0568` | 200 policy read (live, live v7, latest v7) |
| 87 | 12:28:26 | admin | `POST /agent-policies/FIN-0568/retire` `{"baseVersion": 7, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "FIN-0568", "version": 8}` |
| 88 | 12:28:27 | admin | `GET /agent-policies/OPS-0544` | 200 policy read (live, live v2, latest v2) |
| 89 | 12:28:27 | admin | `POST /agent-policies/OPS-0544/retire` `{"baseVersion": 2, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0544", "version": 3}` |
| 90 | 12:28:27 | admin | `GET /agent-policies/OPS-0545` | 200 policy read (live, live v2, latest v2) |
| 91 | 12:28:28 | admin | `POST /agent-policies/OPS-0545/retire` `{"baseVersion": 2, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0545", "version": 3}` |
| 92 | 12:28:28 | admin | `GET /agent-policies/OPS-0546` | 200 policy read (live, live v3, latest v5) |
| 93 | 12:28:28 | admin | `POST /agent-policies/OPS-0546/retire` `{"baseVersion": 5, "changeNote": "Stage 5 demo clean-up"}` | 201 `{"policyKey": "OPS-0546", "version": 6}` |

## After this note: the final review and its fix wave (2026-10-10)

The demonstration above ran on BP `f0a55677`, gateway `2c5f799`, UI `5251664`. A whole-branch review followed
and a fix wave landed afterwards (BP `9fbee672..a87b160a`, gateway `2d26337`, UI `8e2c25d`), each re-reviewed.
What changed relative to the note above:

- **The hourly learning scan now ships OFF.** It registers only when `AGENT_POLICY_LEARNING_SCAN` is `on`, `1`
  or `true`. Wherever this note or plan decision 13 says it is on unless switched off, that is superseded. Run
  it once under supervision on each database before switching it on.
- **D1, D2, D4 and D5 are fixed**: the "Second review" label no longer runs into its sentence; the Suggestions
  queue and count refresh after a save; a weakened card says "N of them were later marked as reversed"; a live
  test proves a block policy and a policy in a never-suggest area are not scanned. **D3 is open**: "Reviewed by"
  and "Approved by" still show a sign-in id (a name needs the email stored with the record).
- The policy page opens without suggestions when learning cannot be read; a proposal on a field marked
  sensitive in the live form is never raised; a stale Business areas screen can no longer overwrite the other
  switch; the gateway's area-settings update accepts only the three settings.
- These later fixes were covered by automated tests and review, not by a second run of the private stack.
- Suites after the fix wave: BP fake 1 failed / 845 passed, live 1 failed / 1290 passed (the known
  `test_conflict_scan_has_its_own_lane`); gateway 395 passed; UI agentPolicy 391/391, whole suite the known 7.
  One stage 4 test (`test_conflict_export_live.py::test_the_policy_export_masks_per_caller`) failed once in one
  of three full live runs and passed otherwise; unexplained, in code this branch does not touch.
