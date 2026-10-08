# Agent policy governance, stage 1: verification and diff summary

Date: 2026-10-08. Branch `agent-policy-stage1` in three worktrees. Database: `bp_testdb` (from `.env`).
Nothing has been merged or pushed.

## In plain English

- **The backend works end to end on live data.** All 10 demonstration points behaved as the plan says:
  create, save, preview, a refused activation, a real activation, the orchestrator feed, a draft beside
  a live version, retire, a stale-save refusal and a refused direct call.
- **The gateway could not do the writes in local development.** The local sign-in bypass gives the
  group `Admin`. The gateway's role table only knows `PROCWISE_*` groups, so it treats that user as a
  **Viewer**. Reads went through the gateway. Every write was refused there with 403 "needs the Buyer
  role or higher" (or Approver). The writes were then sent straight to the backend on :8010, exactly
  as the gateway would forward them for a `PROCWISE_ADMIN` user. Each request is labelled below.
  A real Cognito Admin token was not available for this check.
- **Owners would not see the registry problem message.** In point 4 the backend's output-safety layer
  replaced the `registry` and `inputs` problem messages with "I couldn't retrieve that. I've raised it
  with the team." The cause is that those messages contain the word "orchestrator", which
  `output_safety` flags as a `mechanism` leak. The field names (`registry`, `inputs`) still arrive, so
  the form still points at the right field. The words that say *what* is missing do not arrive.
  The screen therefore cannot say what brief §3.5 asks it to say. This is not fixed here.
- **The Policies screen loaded and opened policies with no console errors and no uncaught exceptions**
  (brief §8 test 22). The only browser errors were network errors from endpoints this feature does not
  call, plus the expected 403 when a Viewer clicked Save draft.

## Step 1: test suites

| Repo | Command | Result |
|---|---|---|
| BP_Backend | `pytest tests/agent_policy tests/guardrails tests/governance tests/approvals tests/engines -q` | 698 passed, 60 skipped, **4 failed**. These are the known baseline failures, none in agent_policy: `guardrails/test_policy_action_vocabulary.py` ×2 and `governance/test_governed_limits.py` ×2 |
| BP_Backend | `PROCWISE_TEST_LIVE_DB=1 pytest tests/migrations/test_2026_10_08_bp_agent_policy.py tests/agent_policy/test_repo_live.py -q` | **19 passed** |
| Gateway | `npx jest` | 13 suites, **226 passed** |
| UI | `npx vitest run src/modules/SpendIQ` | 2236 passed, **4 failed**. These are the known baseline failures, all in `atb/composedPages.contract.test.js` |

## Step 2: live demonstration

Stack: the backend from the worktree on 127.0.0.1:8010, the gateway built from its worktree on
127.0.0.1:3011 (`AUTH_BYPASS=true`, `BP_BACKEND_URL=http://127.0.0.1:8010`) and the UI on
127.0.0.1:3010. Gateway requests carried `x-customer-id: 001`. Key values are redacted.

The policy created was **GEN-0112** (it is now retired in `bp_testdb`).

**Labels used below.** "GW" means a request through the gateway on :3011. "BE-as-GW" means the
same request sent to :8010 with `X-Gateway-Key: <redacted>`, `X-User-Sub` set to the bypass sub,
and `X-User-Groups: ["PROCWISE_ADMIN"]`.

| # | Request | Response | Pass |
|---|---|---|---|
| 1 | GW `POST /agent-policies {"form":{"name":"Task 11 demo - refund or credit over $500"}}` | 403 "needs the Buyer role or higher" | see note |
|   | BE-as-GW, same body | 200 `{"policyKey":"GEN-0112","version":1}` | ✔ |
| 2 | BE-as-GW `POST /agent-policies/GEN-0112/versions` `{form: FORM_EXAMPLE (Finance / Refunds and credits), baseVersion:1, intent:"draft"}` (GW: 403) | 200 `{"policyKey":"GEN-0112","version":2}`. The ID is unchanged. | ✔ |
| 3 | GW `POST /agent-policies/preview` (FORM_EXAMPLE) | 201. 501 → "A person decides"; 500 → "Nothing happens"; 499 → "Nothing happens". The problems are `registry`, `inputs` and `checked`. | ✔ |
| 4 | BE-as-GW `POST …/versions {baseVersion:2, intent:"activate"}` with no confirmation (GW: 403 "needs the Approver role") | 422. The problems listed `registry` (×5), `inputs` and `checked`. **The registry and inputs messages were withheld by output-safety**, as described above. | ✔ (with defect) |
| 5 | GW preview of the changed form, then BE-as-GW `POST …/versions {baseVersion:2, intent:"activate"}` with a browser-supplied `checked.by:"forged-by-browser"` | Preview examples: "A person decides" / "Nothing happens" / "Nothing happens", and the only problem left was `checked`. The activation returned 200 `{"version":3}`. GW `GET /agent-policies/GEN-0112` then showed `status:"live"`, `liveVersion:3`, and `checked.by` set to the bypass sub with server time (the forged value was replaced). | ✔ |
| 6 | `GET :8010/orchestrator/agent-policies/v2/live` with the orchestrator key | 200. `policies` held one entry, GEN-0112 v3, and `refused` was empty. `contract.validate(doc, load_registry())` returned `[]`. | ✔ |
| 7 | BE-as-GW `POST …/versions {baseVersion:3, intent:"draft"}` | 200 `{"version":4}`. The feed still served GEN-0112 **v3**. | ✔ |
| 8 | BE-as-GW `POST …/retire {baseVersion:4}` | 200 `{"version":5}`. The feed's `policies` was then `[]`. | ✔ |
| 9 | BE-as-GW `POST …/versions {baseVersion:3}` (stale) | 409 "Someone saved a newer version (latest is 5, edit was based on 3). Reload and try again." | ✔ |
| 10 | `:8010 GET /agent-policies` with no key; `POST /agent-policies` with `x-gateway-key: wrong`; `GET /orchestrator/agent-policies/v2/live` with no key | 401 `{"detail":"not accepted"}` for all three | ✔ |

**Note on step 5 (ruling: real tool).** The registry has no `supplier_ranking` action. The real action
is `run_supplier_ranking`. The registry has **no numeric `args.*` input**: every `args.*` input is
typed `string`. The condition was therefore
`all[tool.name in ["run_supplier_ranking"], args.deal_id eq "DEMO-T11"]`. Alongside it,
`actions.tools` was set to `["run_supplier_ranking"]`, the inputs list became Deal / Tool / Agent's
reason, and the examples became DEMO-T11 → approve, OTHER-1 → none, and `run_rag` → none.
`args.deal_id` was used instead of `args.payload_json` because it reads plainly in an `eq` condition.
Both are registered string inputs.

## Headless screen check (brief §8 test 22)

The check used headless Chrome 150 driven over CDP: `/usr/share/nodejs/ws`, 1440×900.

**What was exercised:**
1. Loaded `/spendiq`. The first load is bounced to `/home` by the first-screen redirect, so the script
   navigated a second time.
2. Opened the Agent canvas with `go('workspace')`.
3. Selected the Policies panel tab with `wfSetAgentTab('policy')`.
4. **Clicked the "Policies" link button** in the panel header. The Policies screen opened on the
   "Agent policies" tab and listed 112 policies.
5. Clicked **Open** on GEN-0112. The form showed "GEN-0112 · Version 5 · Retired · Extraction
   confidence: Medium" with 16 fields. A screenshot was taken.
6. Clicked Save draft. As a Viewer at the gateway this gave a 403, and the toast read "Could not save:
   needs the Buyer role or higher".
7. Clicked Cancel, and the form closed.
8. Opened a different policy, GEN-0006, then clicked Cancel.
9. Switched to Inventory. It showed the areas and an Export CSV button.

**Not exercised:**
- editing fields;
- the two-step confirm;
- Activate and Retire from the screen;
- the CSV download itself;
- any successful write from the screen, because of the gateway role note above.

**Result:**
- **0 `console.error` calls and 0 uncaught exceptions** across both runs.
- The browser's network log recorded:
  - `GET :3011/users/me/preferences` 404;
  - `GET :3011/user?page=1&limit=100` blocked by CORS (the `Access-Control-Allow-Origin` value was `null`);
  - the expected 403 on `POST /agent-policies/GEN-0112/versions` (Save draft as a Viewer).

  The first two are endpoints this feature does not call, so they are pre-existing local-setup noise.
  No `/agent-policies` read failed.

**Pre-existing setup gap (not from this branch).** `src/modules/SpendIQ/index.jsx` on `spendiq-ui`
(commit 17b73b0) imports `./data/mockPipelineDeals`. That file is **untracked**: it exists only in the
main UI checkout. A clean checkout of `spendiq-ui` therefore fails to load `/spendiq` in dev, with a
Vite 500. For this check the file was copied into the worktree as an untracked file and deleted
afterwards. No tracked file was edited.

## Step 3: diff summary

### BP_Backend: `99f837e..HEAD` (13 commits)

```
c77109c fix(agent-policy): an edit that clears the examples' confirmation clears it, and a missing base row is NotFound
3af8a55 fix(agent-policy): the server records who confirmed a policy's examples, not the browser
abd3dee fix(agent-policy): guard test knows the gateway-keyed routes; more router tests; feed survives a non-dict row
dde074b feat(agent-policy): endpoints behind the gateway key, and a versioned orchestrator feed
ce28bc3 fix(agent-policy): activate refuses with the full problem list; retire state rules
cba2e72 feat(agent-policy): versioned repository with stable ids, optimistic lock, live/draft/retired
ab35b7b fix(agent-policy): condition fields must be available at the checkpoint, not just known
d7db403 feat(agent-policy): what Active requires, how it is enforced, extraction confidence
cfb69c1 feat(agent-policy): hard-policy/2 JSON Schema and the cross-field contract checks
4a4ce2d feat(agent-policy): example results computed by code, and a pure compiler to hard-policy/2
60718b6 feat(agent-policy): company settings and an orchestrator registry seeded from the real tool list
ad29356 feat(agent-policy): tables for agent policies, taxonomy, registry and immutable versions
35bb63c docs(agent-policy): brief, design and stage 1 plan
```

| File | Change |
|---|---|
| `deploy/sql/2026-10-08_bp_agent_policy.sql` (+112) / `_rollback.sql` (+12) | New tables `proc.bp_agent_policy`, `bp_agent_policy_version` (immutable through a trigger), `bp_business_area`, `bp_orchestrator_registry`, plus the settings row. Applied to both DBs in Task 1. |
| `scripts/agent_policy/seed_registry.py` (+95) | Seeds the registry from the real tool list (35 rows). |
| `src/services/agent_policy/{conditions,compiler,contract,readiness,registry,settings}.py`, `hard-policy-2.schema.json` (+642) | New pure logic: condition translation and example results, the hard-policy/2 compiler, the schema and contract checks, the Active requirements, how-enforced, extraction confidence, registry snapshot and settings. |
| `src/repositories/agent_policy_repo.py` (+267) | New versioned repo. It allocates stable IDs, takes an optimistic lock, handles draft/live/retired, retires, and attributes the confirmation on the server. |
| `src/api/routers/agent_policies.py` (+235) | New gateway-keyed screen endpoints and the orchestrator feed `/orchestrator/agent-policies/v2/live`. |
| `src/api/main.py` (+5) | Mounts the two routers. They stay outside `_AUTHENTICATED_ROUTERS` because they are gateway-keyed. |
| `src/services/actions.py` (+6) | Adds 4 action names, `agent_policy.read/write/activate/admin`, to the action-class table. |
| `tests/agent_policy/*` (+923), `tests/migrations/test_2026_10_08_bp_agent_policy.py` (+67) | New tests. |
| `tests/api/test_every_router_is_authenticated.py` (+24/−1) | Gateway-keyed prefixes must answer 401 when no key is sent. |
| `specs/…brief.md`, `…design.md`, `…plan-1-foundation.md` | Docs. |

### Gateway: `a0af842..HEAD` (2 commits)

```
2e58063 fix(agent-policy): refuse unverified sign-ins and validate the business-area path segment
1dd6ea1 feat(agent-policy): gateway routes that check sign-in and role, then forward with the service key
```

| File | Change |
|---|---|
| `src/modules/agent-policy/agent-policy.controller.ts` (+45) | 8 routes. Each requires a verified token and the product role (`productRole`), checks the key and area path params against an allow-list, then forwards. |
| `src/modules/agent-policy/agent-policy.service.ts` (+32) | Forwards with `X-Gateway-Key` and the verified identity. The browser token is not forwarded. A 422 `problems` payload is passed through intact. |
| `src/modules/agent-policy/agent-policy.module.ts` (+6), `src/app.module.ts` (+2) | Module wiring. |
| `src/modules/agent-policy/agent-policy.yml` (+199), `serverless.yml` (+1) | 8 Lambda function and route definitions. |
| `src/modules/agent-policy/agent-policy.controller.spec.ts` (+80) | Tests. |

### UI: `c76e309..HEAD` (4 commits)

```
2b676c7 fix(agent-policy): a refused first activation leaves a saved policy; one write at a time; one form at a time
d5d43b0 feat(agent-policy): agent policies tab, inventory, the new form with example check and two-step confirm
70fc7fb fix(agent-policy): CSV guard also neutralises a leading tab or carriage return
5daf3c7 feat(agent-policy): form model, inventory grouping and a formula-safe CSV export
```

| File | Change |
|---|---|
| `src/modules/SpendIQ/agentPolicy/model.js` (+87) | New tested pure module: `emptyForm`, `switchOutcome`, `toSaveable`, `editClearsConfirmation`, `flipExample`, `confirm`, `responseTimeLabel`, `offersApproverActions`, plus the label maps. |
| `src/modules/SpendIQ/agentPolicy/inventory.js` (+55) | New: `groupBySource`, `groupByArea`, `csvCell`, `inventoryCsv`. |
| `*.test.js` ×3 (+239) | Tests, including `engineWiring.contract.test.js`. |
| `src/modules/SpendIQ/index.jsx` (+4) | Imports the two modules and exposes them on `window.__SPENDIQ_AP__`. |
| `src/modules/SpendIQ/engine.js` (+507/−1) | See below. |

#### `engine.js` in detail

**Lines.** 507 added and 1 removed, in 4 hunks:
- `agentPanel`: +2. The "Policies" link button in the policy-library header.
- `policiesView`: +1 and ±1. The agent tab guard and the `apTabStrip()` prefix on the existing return line.
- After `policyDelete`: +503. The new block.

**New functions (36, all `ap*`):**
- `apLib`, `apErrText`, `apCanApprove`, `apCssEsc`, `apUserSub`, `apTabStrip`, `apLoadList`
- `apStatusTag`, `apOutcomeLabel`, `apPoliciesTab`, `apApi`, `apListHTML`, `apInventoryHTML`, `apExportCsv`
- `apLatest`, `apCategories`, `apOpen`, `apOpenNow`, `apFormHTML`, `apEnforcedHTML`, `apExamplesHTML`
- `apTechProblemsHTML`, `apEdit`, `apRender`, `apBind`, `apClose`, `apRefreshPreview`, `apShowProblems`
- `apReload`, `apBusy`, `apSetWriteButtons`, `apHandleSaveError`, `apSave`, `apConfirmTwoStep`, `apRetire`, `apCancel`

**New top-level constants:** `AP`, `AP_COMPANY_RESPONSE`, `AP_ENFORCED_ALSO`, `AP_PREVIEW_KEYS`.
No non-`ap` function was added.

**Byte-identical proof.** The hunk headers of `git diff -U0 c76e309..HEAD -- engine.js` are:

```
@@ -21544,0 +21545,2 @@ function agentPanel(){
@@ -22358,0 +22361 @@ function policiesView(){
@@ -22361 +22364 @@ function policiesView(){
@@ -22414,0 +22418,503 @@ function policyDelete(i){...
```

At base, `policyEdit` occupies lines 22374–22412, `policyDelete` is line 22413, and `openFormModal`
occupies lines 22431–22508. No hunk's old range touches any of these lines.

Each function was also extracted from `git show c76e309:` and `git show HEAD:` and compared with `cmp`:

| Function | Base | HEAD | sha256 (first 16) | cmp |
|---|---|---|---|---|
| `policyEdit` | 39 lines @22374 | 39 lines @22377 | `3b30cd19055048ee` both | identical |
| `policyDelete` | 2 lines @22413 | 2 lines @22416 | `4ea8f9fb609da1ea` both | identical |
| `openFormModal` | 78 lines @22431 | 78 lines @22937 | `d88868bc25b18dc2` both | identical |

## Open findings for review

1. **The gateway's local bypass means Viewer for agent policies.**
   - `CognitoGuard` bypass sets `cognito:groups: ['Admin']`, which is pre-existing code.
   - `productRole()` maps only `PROCWISE_*` groups, so that user is a Viewer. Every write through the
     gateway in local dev is refused.
   - The UI, meanwhile, believes the user is `PROCWISE_TENANT_SUPER_ADMIN`. It therefore offers Save and
     Activate, which then fail with a toast.
   - This does not affect production with real tokens. It does block local demos of the write path
     through the gateway.
2. **Output-safety withholds problem messages that contain "orchestrator".**
   - This hits the `registry` and `inputs` problems, and the contract messages "tool X is not something
     the orchestrator recognises".
   - Owners see a generic sentence instead of what is missing.
   - Oddly, the same text inside `howEnforced.cantEnforce` was not withheld.
3. **The 422 list mixes owner wording with contract internals.** It contains "a live policy needs
   trigger.checkedBy" and "input args.amount is not available at tool.call.before", and it repeats the
   `registry` field 5 times.
4. **A retired policy's form still shows "Activate…".** The repo does not refuse a save on a retired
   policy. Whether re-activating a retired policy is intended is not stated in the plan.
5. **`bp_testdb` holds 112 agent policies**, mostly leftovers from earlier tasks' live tests and probes.
6. **A clean `spendiq-ui` checkout cannot load `/spendiq` in dev**, because the untracked
   `mockPipelineDeals.js` is missing. This is pre-existing.
