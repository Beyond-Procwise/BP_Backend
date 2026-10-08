# Agent policy governance, stage 2 (extraction agent): verification and diff summary

Date: 2026-10-08, 21:16 to 21:40 UTC. Branch `agent-policy-stage2` in three worktrees. Database: `bp_testdb`
(checked with `select current_database()`). Nothing has been merged or pushed.

## In plain English

- **Uploading through the gateway works for real.** The gateway's own S3 signer was used for the first
  time: it handed out upload links, the files were sent to S3 with those links, and the backend then
  fetched each file back from S3, measured it and recorded it. Byte sizes match the files on disk.
  The same worked from the browser, through the real Upload panel.
- **Revisions and the before/after view work.** A changed Finance document was recorded as version 2
  of the same document. The compare view showed 1.1 as "changed" ($500 → $750) and 1.2 as "removed",
  everything else "unchanged". No text was replaced by "[withheld]" this time.
- **The extraction run did not produce anything, because the shared AI model was busy.** The run
  started and stayed alive (its heartbeat kept ticking), but the model never answered: two requests
  of 10 minutes each timed out. This is the known problem with the shared model server, not a fault
  found in this branch. The run was left alone; it heals to "failed" on its own (see step 3).
- **Because no policies were produced, the step that needs the live model (acceptance test 5) and the
  "open a policy, flip an example, Ask the agent to fix it" walk-through were not done.**
- **The screens showed no console errors and no uncaught exceptions.** The only browser errors were the
  same pre-existing ones as in stage 1 (endpoints this feature does not call).

## How the stack was run

| Part | Port | How |
|---|---|---|
| Backend | 127.0.0.1:8010 | `.venv/bin/uvicorn api.main:app --workers 1` from the BP worktree, `PYTHONPATH=<worktree>:<worktree>/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`. PID 1283715. At startup it logged "Ollama already holds 'BeyondProcwise/AgentNick:unified'; not preloading", so it did not reload the shared model. |
| Gateway | :3011 | `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development AUTH_BYPASS_GROUPS=PROCWISE_ADMIN BP_BACKEND_URL=http://127.0.0.1:8010`, `AGENT_POLICY_GATEWAY_KEY` read from the BP `.env` (never printed), `AWS_BUCKET_NAME`/`AWS_REGION` from the gateway `.env`. PID 1284949. |
| UI | 127.0.0.1:3010 | Vite from the UI worktree, `VITE_API_URL=http://127.0.0.1:3011 VITE_DEV_AUTH_BYPASS=true`. PID 1302517. The untracked `src/modules/SpendIQ/data/mockPipelineDeals.js` was copied in for the run (stage 1 finding 6) and deleted afterwards. |
| Headless Chrome | :9223 | `google-chrome --headless=new`, driven over CDP with `/usr/share/nodejs/ws`, 1440×900. PID 1303159. |

All requests carried `x-customer-id: 001`. Signatures and access-key ids in S3 links are redacted.
All four processes were stopped by PID at the end; ports 8010, 3011, 3010 and 9223 were confirmed closed.
The shared :8000, :3001, :3000, `procwise.service` and Ollama were never touched and were still up.

## Step 1: upload three documents through the gateway (the real signer). PASS

| # | Request | Response | Time |
|---|---|---|---|
| 1a | `POST :3011/agent-policies/documents/upload-urls` `{files:[finance_payments_policy.md 485 B, customer_refund_standard.md 371 B, security_data_handling.md 364 B], contentType text/markdown}` | 201. Three uploads, each `{uploadId, name, url, headers:{"Content-Type":"text/markdown"}}`. The URL is `https://procwisemvp.s3.eu-west-1.amazonaws.com/agent-policy-documents/uploads/<uploadId>/<name>?AWSAccessKeyId=<redacted>&Content-Type=text%2Fmarkdown&Expires=…&Signature=<redacted>`. No key or URL came from the backend. | 0.73 s |
| 1b | Plain HTTP `PUT <url>` ×3, with exactly the returned headers | 200 ×3 | 0.09 s each |
| 1c | `POST :3011/agent-policies/documents` `{uploads:[{uploadId,name}×3]}` | 201. Documents **314, 315, 316**, each version 1, `isRevision:false`, `duplicate:false` | 0.74 s |
| 1d | `GET :3011/agent-policies/documents` | 200. 314/315/316 listed, byte sizes 485/371/364 (match the files), content hashes recorded, uploader = the bypass sub | 0.17 s |

The backend's egress log shows it fetched each object back from S3 during register. So the gateway's
signature, the content type it signed, and the key the backend rebuilds from `{uploadId, name}` all agree.

Not covered: the link was signed with long-lived credentials from the host's default AWS chain
(`AWSAccessKeyId=` form). In Lambda the role's temporary credentials add a session token to the link;
that form was not exercised here.

## Step 2: revised Finance document and compare. PASS

The revision was built as the acceptance test builds it: `above $500 ` → `above $750 `, and clause 1.2
removed (330 bytes).

| # | Request | Response | Time |
|---|---|---|---|
| 2a | `POST …/documents/upload-urls` (1 file, 330 B) | 201, one signed upload | 0.27 s |
| 2b | `PUT <url>` | 200 | 0.09 s |
| 2c | `POST …/documents` `{uploads:[{uploadId,name,revisionOf:314}]}` | 201 `{documentId:314, version:2, isRevision:true, duplicate:false}` | 0.43 s |
| 2d | `GET …/documents/314/compare?from=1&to=2` | 200, 6 sections (below) | 0.32 s |

| Section | Status | Before → after |
|---|---|---|
| Finance: Payments and Refunds Policy (preamble) | unchanged | same text |
| 1 | unchanged | "1. Refunds and credits" |
| 1.1 | **changed** | "…above **$500** need approval…" → "…above **$750** need approval…" |
| 1.2 | **removed** | "1.2 The agent must get approval from the Finance Manager before it runs email dispatch (the run_email_dispatch tool)…" → null |
| 2 | unchanged | "2. Good practice" |
| 2.1 | unchanged | "2.1 Finance staff should review the refunds…" |

**No "[withheld]" text appeared** in any section, in the API response or on screen. The output-safety
scrubbing of compare text stays an open question (see findings), because these documents happen not to
contain words it flags.

## Step 3: extraction run on the three v1 documents. NO ITEMS (model not available)

| # | Request | Response |
|---|---|---|
| 3a | `POST :3011/agent-policies/extraction-runs` `{documents:[{314,1},{315,1},{316,1}]}` | **202** `{runId: 702}` (0.36 s) |
| 3b | `GET …/extraction-runs/702` every 15 s, 40 polls, 21:18:19 → 21:28:30 | Always 200, `status:"running"`, owner `proc-b21b8781986c`, `heartbeat_at` advancing every ~30 s, `counts:{}`, `error:null`, **0 items** |
| 3d | Final `GET …/extraction-runs/702` at 21:38:44 | `running`, heartbeat 21:38:24, 0 items |

What the backend log shows for this run:
- 21:18:10: it read the three documents from S3.
- 21:28:20: `Ollama read timeout (attempt 1/3, timeout=600s)`.
- 21:38:30: `Ollama read timeout (attempt 2/3)`.

Throughout, `/api/ps` showed AgentNick loaded at **context 8192**. Our extractor asks for 12288, and the
shared `procwise.service` kept the model busy. The GPU swung between 0 % and 84 %. Nothing in Ollama was
cancelled or restarted.

**Per-chunk timings: none this time**, because no chunk was answered. The only real-model timings for
these fixtures are the ones from Task 9a (runs 549/591): Finance 21.5 s / 26.2 s, Customer operations
16.2 s / 15.1 s, Security 16.6 s / 16.7 s; 55.8 s and 58.8 s per run.

**Where the runs were left.** Stopping my backend ended the third attempt. In `bp_testdb`:
- Run 702 is `running`, with its last heartbeat at 21:38:24.
- Run 727 is `queued`. The UI upload in step 5 started it, and it waited behind 702.

Both runs are owned by the stopped process, so the run store heals them to `failed` on the next read
or list by any backend (the Task 5 heal rule). No policies were created by this demonstration.

## Step 4: acceptance test on the live model. SKIPPED

The model never answered in step 3, so `tests/agent_policy/test_acceptance_live_model.py`
(`AGENT_POLICY_LIVE_MODEL=1 PROCWISE_TEST_LIVE_DB=1`) was not run. Acceptance points 1 to 4 passed in
Task 9a (runs 549 and 591). Point 5 is still not re-verified since its test-setup fix (f4dae53b).

## Step 5: headless Chrome on the UI

**Exercised (in order):**
1. Loaded `/spendiq`, re-navigating until the first-screen redirect let it stay.
2. Opened the Agent canvas with `go('workspace')` and selected the Policies panel tab with
   `wfSetAgentTab('policy')`.
3. **Clicked the "Policies" link button.** The Policies screen opened with the segments
   List / Inventory / **Documents** and an **Upload documents** button. 1114 Open links were listed.
4. **Clicked "Upload documents".** The panel opened with the file picker and the help text.
5. **Picked a file** with CDP `DOM.setFileInputFiles` on the panel's real `<input type=file>`. The file
   was `ui_probe_stage2.txt` (88 B). The change event built the plan row: "ui_probe_stage2.txt — New
   document", status Ready, and "Upload and read" became enabled.
6. **Clicked "Upload and read".** The browser made these calls:
   - `POST :3011/…/upload-urls` 201;
   - the browser's CORS preflight to S3, 200;
   - `PUT` to S3, **200**;
   - `POST :3011/…/documents` 201 (document **331** v1);
   - `POST :3011/…/extraction-runs` **202** (run **727**).

   The panel switched to "Reading policy documents — Waiting to start — Nothing found yet", and the
   run was polled (`GET …/extraction-runs/727?afterSeq=0`, 200).
7. Closed the panel, then opened **Documents**. 215 documents were listed, with Compare links (including
   314) and Recent runs (727, 702, …).
8. **Clicked View on run 702.** The run panel showed "Reading policy documents — Reading the documents —
   Nothing found yet", polling `…/extraction-runs/702?afterSeq=0` (200).
9. Closed it, then **clicked "Compare versions" on finance_payments_policy (314)**. The Compare panel
   showed "finance_payments_policy · Compare versions" with version pickers v1/v2. Its sections were
   Unchanged / Unchanged / **Changed 1.1 ($500 → $750)** / Removed 1.2 …, with no "[withheld]". Then
   it was closed.

Screenshots were taken at each stage (scratchpad `s2demo/cdp-*.png`, `cdp3-*.png`). The second pass
(steps 7 to 9 alone) was needed because the first pass clicked Compare while the Documents list was
still reloading after the run panel closed.

**Not exercised:**
- Opening an extracted policy, flipping an example and "Ask the agent to fix it": no extracted
  policy exists, because the model never answered.
- Retire from the run list.
- "Treat as a new document" and "Read again".

**Console:**
- 0 `console.error` calls and 0 uncaught exceptions, in both passes.
- The log recorded the same pre-existing noise as stage 1: `GET :3011/users/me/preferences` 404, and
  `GET :3011/user?page=1&limit=100` blocked by CORS (`Access-Control-Allow-Origin: null`).
- One more network entry: the S3 `PUT` was logged as `net::ERR_ABORTED` **after** its 200 response
  arrived. The upload itself succeeded: register found the object and recorded 88 bytes. The UI checks
  only `res.ok` and never reads the body, so this looks like Chrome abandoning an unread response body
  rather than a failed upload. It is recorded here because it was observed.
- No `/agent-policies` call failed.

## Step 6: clean-up. PASS

- My processes were stopped by PID: Chrome 1303159, Vite 1302517, gateway 1284949 and uvicorn 1283715,
  plus its wrapper shell.
- The UI worktree's copied `mockPipelineDeals.js` was deleted, and `git status` there is clean.
- **S3:** every object under `agent-policy-documents/uploads/<uploadId>/` for the five issued upload ids
  was listed and deleted. A re-list of each prefix returned `KeyCount 0`. The ids were:
  - 1145d97c-3acf-43ad-86e2-5975eb9830da
  - 05118cd1-c4e5-404a-a47c-5f5b0922f474
  - e28bddaa-6337-46ce-8769-f21149bc4573
  - 623d01cc-bdc1-4f4d-83d2-83b2a291d21a
  - 6279c6ed-2356-4c6c-a861-79cd0e030377
- **Left in `bp_testdb` as the record:**
  - documents 314 (v1, v2), 315, 316 and 331;
  - runs 702 and 727;
  - the audit rows.

  The document rows now point at deleted S3 objects. Reading them again would fail to fetch, so treat
  them as demo records only.

## Rulings (every "Ruling:" line in the ledger)

1. interpret user's "keep going" as option 3 for stage 1 (keep branches) and authorisation to plan + execute stage 2 on new branches agent-policy-stage2 stacked on stage 1 — cost if wrong: stage 2 must be rebased if stage 1 changes on review
2. model replies are schema-constrained JSON per chunk (not raw JSON Lines); each policy is written as its own item row as soon as validated, which is what "streams into the list" needs — cost if wrong: brief's literal JSONL wording not followed
3. ollama_generate gains an opt-in use_load_options flag instead of changing its default (changing the default would alter the extraction specialist's context — accuracy risk) — cost if wrong: one flag to delete later
4. upload via backend-issued presigned S3 PUT forwarded by the gateway (gateway Lambda can't carry binary multipart; the existing SpendIQ upload triggers procurement extraction) — cost if wrong: S3 permission must exist for the backend user
5. local AUTH_BYPASS gets an opt-in AUTH_BYPASS_GROUPS env (default unchanged) so local gateway writes can be demonstrated — cost if wrong: one env var in shared guard code, inert unless set in bypass mode
6. drafts are created automatically from extraction (Draft ≠ Active; brief forbids only auto-Active and pre-confirmation) — cost if wrong: drafts appear without a human click
7. Task 2 — normalise suffix strip must require a separator (plan regex cut "Overdraft"→"over"; a false revision match is harmful under the revised-document ruling) — cost if wrong: none
8. Task 3 — clause markers tightened (dotted number, or ≤3-digit number with ./) , or ≤3-digit number + Uppercase); plan regex made "2026 budget"/"3 days" into clauses, and references drive stable-ID matching — cost if wrong: a document numbering clauses like "1 refunds" (lowercase, no dot) is read as one section
9. Task 4 — code flags misfits the model won't (number op on a text field, the payload_json catch-all, no action at tool checkpoint, is_amount on non-number) as unknownNames so such policies show "Can't be enforced yet"; registry digest now gives each action's purpose — cost if wrong: some legitimate string comparisons on args.* could be flagged
10. Task 4 — accept per-chunk schema JSON (not JSON Lines); items stream per chunk via the run store — cost if wrong: brief's literal JSONL wording not followed
11. Task 5 — stranded queued runs (owner gone) heal to failed after the stale limit; finish() reports whether it changed the row so a healed-then-finished run is visible — cost if wrong: a queued run on a slow-starting worker could be failed after 120 s
12. Task 6 — re-extraction compares against the last agent-produced version (from run items), so an unchanged document never overwrites a person's draft edits; a real document change still writes a new draft on top, noting "Replaces the edits in vN" — cost if wrong: a person's edit that diverges from the document stays until the document changes
13. Task 6 — a failed conversion holds back Proposed retire for that document, like a failed chunk — cost if wrong: a truly removed clause waits for the next clean run
14. Task 7 — OutputSafety withholds presigned URLs/keys; instead of a second boundary exemption, the GATEWAY signs S3 PUTs (it already does in prod; Lambda role has s3:PutObject on procwisemvp/*); backend issues {uploadId, safeName} and rebuilds the key on register — cost if wrong: two places know the key format
15. Task 6 — "Replaces the edits in vN" only when the latest version's substance differs from the agent's last version (narrower than literal) — cost if wrong: a non-substantive edit isn't mentioned in the note
16. Task 8 — "Treat as a new document" added (UI option + register asNew, routed to the Task 7 fixer who owns documents.py) — cost if wrong: two documents can share a match name; later uploads match the oldest
17. Task 8 — run-list Retire… opens the policy form and uses its two-step retire so errors show — cost if wrong: one extra modal
18. Task 8 — agent notes shown in the form and run list, not in policy list rows — cost if wrong: owners see notes one click later
19. Task 7 — Viewer compare caching parsed_text is a derived cache, not a governed write (no audit) — cost if wrong: one unaudited cache write per document version
20. Task 7 — live round trip through the gateway's real signer is verified in the Task 9 live demo, not a unit test — cost if wrong: a signer mismatch shows up only in the demo

Ruling 20 is now discharged: the round trip was verified in step 1, and again from the browser in step 5.

## Open findings

1. **The shared Ollama runs at context 8192, and the main service truncates prompts.**
   - `procwise.service` calls Ollama without `num_ctx`, so the runner sits at the Modelfile default
     8192. Prompts longer than that are cut. This is an accuracy issue in the shared service, and it
     is another session's code.
   - Our extractor asks for 12288 through `load_options()`, which would need a reload. Under that load
     the reload never happens: today two 600 s timeouts and no answer.
   - Nothing was changed: that is other callers' behaviour, and the accuracy rule applies.
   - Until this is settled, extraction runs on this host only finish when the shared service is idle.
2. **Output-safety on compare text.** The ledger (Task 7) records that compare section text may be
   scrubbed by output-safety, so a policy document that contains words it flags (for example
   "orchestrator", or a route or table name) could show "[withheld]" in the before/after. Today's fixtures did not trigger it, so this
   demonstration neither confirms nor clears it. This is the same pending decision as stage 1's
   finding 2.
3. **The registry has no refund tool and no amount field.** The tiered refund clause (1.1) can never
   go live: the amount has no registered input, and code correctly flags it as "Can't be enforced
   yet". This is a decision for the registry owner.
4. **Compare scrubbing (UI).** The Compare panel shows whatever text the API returns. If output-safety
   replaces a section, the owner sees "[withheld]" with no explanation. This was not seen today; it
   depends on finding 2.
5. Pending from Task 9a/9b: acceptance test 5 still needs one green run on an idle model.
6. Observed today:
   - the browser S3 PUT logs `ERR_ABORTED` after a 200, which looks benign (step 5);
   - the signer was proven only with long-lived local credentials, not Lambda session credentials
     (step 1).

## Diff summary

### BP_Backend: `ec6ebd0..HEAD` (17 commits)

```
b461af0a fix(agent-policy): wording table, decider/owner roles, and junk-free unknown names
f4dae53b test(agent-policy): three fixture policy documents and the acceptance test on the real AgentNick model
1b0949b4 fix(agent-policy): long names keep their extension; issued upload ids are audited; asNew retries are duplicates
f5a795bf feat(agent-policy): register an upload as a new document with asNew
0b492bd7 fix(agent-policy): the gateway signs document uploads; the backend never returns a URL or key
0266cb7c fix(agent-policy): extraction review round 1 - agent baseline, incomplete holds retires, per-document isolation
1f3eeda7 feat(agent-policy): document, extraction-run and agent-fix endpoints
388c43dd feat(agent-policy): matching, revision decisions, and the extraction run
b5f267c1 fix(agent-policy): extraction review round 1 - code-verified missing inputs, misfit checks, action purposes
b6c24962 fix(agent-policy): run store heals orphaned queued runs, finish reports whether it changed the row, deterministic lock test
d90fd918 feat(agent-policy): extraction run store and single-worker runner with heartbeat and heal-on-read
d12fe0bc feat(agent-policy): extraction schema, governed prompts, and proposal-to-form converter
20f08389 fix(agent-policy): numbered-clause markers no longer match prose; single GPU read
f0a5f457 feat(agent-policy): sections, chunking, diff, and opt-in load options for model calls
614aa0ed fix(agent-policy): documents review round 1 - separator-bound suffixes, issued keys, race-safe register
037c1fa8 feat(agent-policy): policy documents - presign, register, version, parse
ef6af8a0 feat(agent-policy): stage 2 migration - policy documents, versions, extraction runs and items
```

`git diff --stat ec6ebd0..HEAD`: **37 files, +5097 / −10** (before this note).

| Area | Files |
|---|---|
| Migrations | `deploy/sql/2026-10-09_bp_agent_policy_extraction.sql` (+63) and `_rollback` (+14); `2026-10-09_agent_policy_extraction_prompts.sql` (+23) and `_rollback` (+3); `2026-10-10_agent_policy_prompts_wording.sql` (+15) and `_rollback` (+10) |
| New services | `src/services/agent_policy/`: `documents.py` (+437), `extraction_run.py` (+316), `converter.py` (+219), `run_store.py` (+186), `matching.py` (+178), `sections.py` (+175), `extractor.py` (+158), `extraction_schema.py` (+126), `run_runner.py` (+96) |
| Changed | `src/api/routers/agent_policies.py` (+169), `src/repositories/agent_policy_repo.py` (±15), `src/services/agent_policy/registry.py` (±10), `src/services/ollama_client.py` (+21, the opt-in `use_load_options`) |
| Tests | `tests/agent_policy/*`: fixtures ×3, acceptance, converter, documents (+live), extraction_run, extractor, matching, prompt_text, router, run_runner, run_store_live, sections. Also `tests/api/test_every_router_is_authenticated.py` (+18), `tests/migrations/test_2026_10_09_…` (+72) and `tests/test_ollama_instance_consistency.py` (+39) |

### Gateway: `e8c815c..HEAD` (3 commits)

```
cfc7fce test(agent-policy): register forwards asNew unchanged
50ad6dd fix(agent-policy): the gateway signs agent-policy document uploads; register gets 25 s
4fb9707 feat(agent-policy): document, extraction-run and agent-fix routes; local bypass groups
```

**6 files, +485 / −7:**
- `agent-policy.controller.ts` (+43);
- `agent-policy.service.ts` (+58: the signer and the 25 s register timeout);
- `agent-policy.yml` (+200);
- `agent-policy.controller.spec.ts` (+160);
- `src/auth/guards/cognito.guard.ts` (±8: `AUTH_BYPASS_GROUPS`, bypass mode only);
- `src/auth/token-verifier.spec.ts` (+23).

### UI: `6e34455..HEAD` (2 commits)

```
02db583 fix(agent-policy): screens review round 1 - fix poll on close, retire in the form, treat as new
5f953af feat(agent-policy): upload documents, live run list, before/after, Ask the agent to fix it
```

**7 files, +1466 / −8:**
- `agentPolicy/upload.js` (+190) and its test (+135);
- `agentPolicy/runView.js` (+120) and its test (+79);
- `engineWiring.stage2.contract.test.js` (+434);
- `index.jsx` (±4);
- `engine.js` (+512 / −8).

**`engine.js` hunks** (`git diff -U0 6e34455..HEAD`) all sit inside `ap*` functions or after
`apCancel`:

```
@@ -22492 +22492 @@ function apPoliciesTab(){
@@ -22497 +22497 @@ function apPoliciesTab(){
@@ -22499 +22499 @@ function apPoliciesTab(){
@@ -22515 +22515 @@ function apListHTML(){
@@ -22642 +22642 @@ function apFormHTML(state){
@@ -22715,2 +22715 @@ function apExamplesHTML(state){
@@ -22772,0 +22772,3 @@ function apBind(ov,state){
@@ -22784,0 +22787,2 @@ function apClose(state){
@@ -22936,0 +22941 @@ function apCancel(state){
@@ -22940,0 +22946,493 @@ function apCancel(state){
```

**Byte-identical check.** Each function was extracted by brace matching from `git show 6e34455:` and
`git show HEAD:` and compared:

| Function | Base | HEAD | sha256 (first 16) | Identical |
|---|---|---|---|---|
| `policyEdit` | 39 lines @22377 | 39 lines @22377 | `46bfb5ad40f48960` both | yes |
| `policyDelete` | 1 line @22416 | 1 line @22416 | `30ffba85689e2a22` both | yes |
| `openFormModal` | 78 lines @22957 | 78 lines @23455 | `73afdb556aa98a0d` both | yes |

These hashes differ from stage 1's table because the extraction boundaries differ (stage 1 cut by line
range); the comparison here is base vs HEAD of stage 2 with the same method.
