# INTEGRATION_PLAN.md

**Phase 2 — Integration plan.** Follows `INTEGRATION_ANALYSIS.md`. No code written yet.
Date: 2026-07-10

---

## 0. Decisions locked in Phase 1 review

| # | Decision |
|---|---|
| Q1 | **Phase 4 = run the SPA in Chrome.** No extension, no PWA. |
| Q3 | **Repoint Analyse / Negotiate / Opportunities to BP_Backend.** |
| Q5 | **Detection + 7-stage Pipeline are out of scope.** Leave undeployed. |
| Q7 | **No auth on BP_Backend for the local test.** Set `PROCWISE_CORS_ORIGINS` explicitly; stop the UI leaking the Cognito token. |

---

## 1. Headline: most of Q3 is already written, and it was never lost

The 2026-07-08 repoint exists as **three commits on the local branch `Procwise2.0-Sprint-02`**, sitting directly on top of `Nick_UI`'s tip:

```
a21ecc5  feat(analysis): add Items column + read live backend analysis-summary
5a77bd7  feat(analysis): show pre-stored deal summary in Summary panel from /deals/{id}/summary
d34149b  fix(analyse): point Negotiate & Opportunities at real BP_Backend data
```

`git merge-base Nick_UI Procwise2.0-Sprint-02` == `Nick_UI` tip (`2dec9a8`). It is a **clean fast-forward**: 10 files, +101/−44, zero conflicts.

What those commits already do:
- `CustomTabs.jsx` — Negotiate now calls `{AI}/deals/{deal_id}/negotiate`; Opportunities calls `{AI}/opportunities/dashboard`; Analyse merges the legacy `/analyse` payload with `{AI}/deals/{deal_id}/analysis-summary`.
- They alias the backend's `versionHistory` onto the component's misspelled `VerisonHistory` key.
- `Analyse/index.jsx` — replaced the bogus `/Summary/get_summary_summary_get` call with `{AI}/deals/{deal_id}/summary`.

**This means gap A3 from the analysis is fixed by merging, not by new code.**

I verified all three endpoints live against the running backend using a real deal (`DEAL_PERRY2026071061`):

| Endpoint | Returns |
|---|---|
| `GET /deals/{id}/negotiate` | `deal_id, deal_name, orphaned, hasQuoteAnchor, summary, proposalSnapshot, versionHistory, negotiationData, negotiationStrategy, costOverTime, volumeTrend, proposalSummary, demandVsVolumeData` |
| `GET /deals/{id}/analysis-summary` | `row: {id, supplier, category, value, volume, unitPrice, priceChange, volumeChange, efficiency, items}` |
| `GET /opportunities/dashboard` | `opportunitiesData, savingsPipeline, savingsIdentifiedVsCompleted, opportunityTrends, detailedOpportunities` |

> ⚠️ One caveat to fix, not inherit: every repointed fetch ends `.catch(() => ({}))`. Failures render as an empty tab with no error. That will actively hide breakage during Phase 4 testing.

---

## 2. What is actually left to build

After the fast-forward, the **only** broken BP_Backend surface is the chat upload flow in one file, `src/modules/Home/ViewerHomePage.jsx`.

| Gap | Line | Fix |
|---|---|---|
| A1 | `:204` | `POST {AI}/documents/embed-document` → `/document/embed-document` (singular) |
| A2 | `:201`, `:207` | `formData.append('file', …)` → `'files'`; stop reading `res.data.data.fileUrl` (does not exist) |
| A4 | `:156` | `data['files'] = [url]` — `AskRequest` has no `files` field; pydantic silently drops it |
| A5 | `:154` | `model_name: 'llama3.2'` → not installed; violates the AgentNick-only rule |

A2 and A4 are coupled and are the one place I need a business decision — see §6, Q9.

---

## 3. The `beyond-procwaise-Api` change that matters

> **Correction (2026-07-10).** An earlier draft of this section called the `sessionId` situation a
> *regression*. That was wrong, and the error is recorded here rather than quietly deleted.
> `sessionId` did **not** exist before `4bb2bb9` (`git show 4bb2bb9^:…/analyse.service.ts | grep -c sessionId` → `0`).
> The commit **added** it, already commented out. It is unfinished code, not a regression.
> The *deal-scoping* removal in the same commit **is** real. Both are verified below.

Commit `4bb2bb9` ("ip address and demo changes added") did two unrelated things to
`dashboard2.0/analyse/analyse.service.ts`.

### 3.1 It added `sessionId` plumbing — disabled

```ts
private generateSessionId(): string { … }        // :523-528  mints `ses-YYYYMMDD-XXXX`
// const sessionId = this.generateSessionId();   // :551      ← never called
      // session_id: sessionId                   // :568      ← process_monitor row
      // sessionId,                              // :590      ← per-file response
  // sessionId,                                  // :597      ← top-level response
```

So on this branch `generateSessionId()` is **dead code**: uploads insert `proc.process_monitor`
rows with `session_id = NULL`, and the UI never receives a `sessionId`.

**A working implementation already exists on `origin/feature/UAT-branch`** (a separate lineage),
where all four lines are live:

```
UAT analyse.service.ts:583   const sessionId = this.generateSessionId();
UAT analyse.service.ts:600         session_id: sessionId
UAT analyse.service.ts:622         sessionId,          // per-file
UAT analyse.service.ts:629     sessionId,              // top-level response
```

That is why the live database shows **0 NULL session_ids across 61 rows** (`ses-20260710-E2SG`, …):
the deployed Lambda is UAT-lineage. **We port those four lines from UAT rather than inventing them.**

> Implication for Q10/Q11: because the *deployed* gateway appears to return `sessionId` already,
> the WebSocket repoint can likely be exercised against the deployed gateway with **no deploy**.
> Unverified (the endpoint needs Cognito auth), so the UI must treat `sessionId` as optional.

### 3.2 It removed deal scoping — a real regression

Verified against the parent `0982ea8`, which had **6 live filters** (lines 120, 135, 150, 171, 186, 195)
plus 2 phase filters (269, 273):

```sql
WHERE ($1::text IS NULL OR deal_id = $1::text)     -- ×6, all deleted by 4bb2bb9
WHERE LOWER(phase) = 'validation' AND deal_id = $1 -- deleted
WHERE LOWER(phase) = 'extraction' AND deal_id = $1 -- deleted
```

and `async getAnalyseData(deal_id: string)` → `async getAnalyseData()`.

Analyse therefore returns the whole dataset regardless of the selected deal. Per Q12 we restore these
from `4bb2bb9^`.

---

## 4. WebSocket strategy

Today: the UI opens `wss://olop83hwf5…`, and on *any* message just increments a refresh key. That socket is **not deployed** from this branch (`serverless.yml:166` commented; `websocket.yml` points at `dist/modules/websocket/notify.main`, and `notify.ts` does not exist — the real file is `sendMessage.ts`).

Meanwhile BP_Backend already broadcasts exactly the right thing on `WS /ws/session/{session_id}`, driven by Postgres `pg_notify`:

```json
{ "session_id":"ses-20260710-E2SG", "action_status":"completed",
  "total":4, "target":3, "discrepancy":0, "failed":1,
  "duplicate":0, "updated":0, "needs_review":0, "unsupported":0,
  "documents":[{"file_path":"…","doc_action":"duplicate|updated|needs_review|unsupported"}],
  "resolved_at":"…", "category":["…"], "deal_name":["…"] }
```

**Proposed:** repoint `DataIntegration` to BP_Backend's socket.

1. Gateway: un-comment the 4 `sessionId` lines so `/analyse/presigned-url` (and `/data-integration/presigned-url`) return `sessionId` and stamp `process_monitor.session_id`.
2. UI: `new WebSocket(`${AI_WS}/ws/session/${sessionId}`)` after upload; render the real `doc_action` counts instead of blindly refetching.
3. Add `VITE_AI_WS_URL` (derive `ws://` from `VITE_AI_API_URL`).

This retires the dead API Gateway socket and finally consumes the `doc_action` work. It is the largest item in the plan and the one I'd most want your sign-off on (Q11).

---

## 5. Execution plan — one commit per logical unit

| # | Commit | Repo | Risk |
|---|---|---|---|
| 0 | `merge --ff-only Procwise2.0-Sprint-02` (recovers the repoint) | UI | none |
| 1 | `fix(analyse): surface fetch errors instead of swallowing them` — replace `.catch(() => ({}))` with error state | UI | low |
| 2 | `fix(chat): correct embed-document path, field name, and response read` (A1, A2) | UI | low |
| 3 | `fix(chat): send file_path/session per AskRequest; use AgentNick model` (A4, A5) | UI | **needs Q9** |
| 4 | `refactor(api): dedicated aiApi axios client; stop global auth-header leak` | UI | medium — touches 8 call sites |
| 5 | `fix(api.js): 401 interceptor fires on every first error and swallows rejections` | UI | low |
| 6 | `fix(analyse): restore sessionId stamping and return it from presigned-url` (revert `4bb2bb9` regression) | **Api** | **needs Q11** |
| 7 | `feat(data-integration): subscribe to BP_Backend /ws/session/{id}; render doc_action` | UI | **needs Q11** |
| 8 | `chore(env): local run topology (.env.local, PROCWISE_CORS_ORIGINS)` | UI + BP_Backend | low |

Steps 0–5 are safe and unblock a real Chrome test on their own. Steps 6–7 depend on Q11.

### Fix detail for step 5

```js
// src/services/api.js:44  — current
if (err.response?.status === 401 || err?.message === '…expired' || !originalRequest._retry) { … }
//                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
// fires on ANY first-time error (500s, network, CORS), and returns undefined when
// newToken is falsy → the promise RESOLVES, silently swallowing the error.
```
Correct predicate: `if ((err.response?.status === 401 || isExpired(err)) && !originalRequest._retry)`, with an explicit `return Promise.reject(err)` on every non-retry path.

---

## 6. Local run topology for Phase 4

- BP_Backend: already live on `:8000` (systemd `procwise`). Set `PROCWISE_CORS_ORIGINS=http://localhost:3000` and restart.
- UI: `yarn dev` → `:3000`. Add **`.env.local`** (gitignored, overrides `.env.development`) with `VITE_AI_API_URL=http://localhost:8000`.
- **Port collision:** the Nest API also defaults to `:3000` locally. If we run it, it moves to `:3001`.
- `VITE_API_URL` (the ~61 gateway paths) — see Q10: run Nest locally, or keep pointing at the deployed AWS gateway.

Primary flows to exercise in Chrome:
1. Cognito login → `/home`
2. Chat: ask a question; upload a document and ask about it *(the A1–A5 fixes)*
3. `/data-integration`: upload → live progress *(the WebSocket work)*
4. `/home` → Analyse / Negotiate / Opportunities tabs against real deals *(the repoint)*
5. `/actions`: email draft dispatch → `POST {AI}/workflows/email`
6. `/dashboard`, `/users`, `/policies`, `/prompts` (gateway regression check)

---

## 7. Risks and things I will not guess

- **R1 — Analyse "demo mode".** `4bb2bb9` also stripped `deal_id` filtering from the gateway's Analyse queries. After the repoint, the Analyse tab merges a BP_Backend `analysis-summary` (correctly deal-scoped) with a legacy `/analyse` payload (**not** deal-scoped). Numbers on one screen will disagree. See Q12.
- **R2 — `compliance.getDateRange` is pinned to `new Date('2025-12-01')`.** Compliance stays on the gateway (out of scope per Q5) and will show sample-data months. Cosmetic but confusing during testing.
- **R3 — `GET /summary?persona=…` 404s when nothing is cached.** The repoint uses `/deals/{id}/summary` instead, which returns `{summary: null, message: "Summary not available"}` rather than 404. Safe, but tabs may render empty for deals with no precomputed summary.
- **R4 — Step 4 touches 8 call sites** across modules. Mechanical, but it is the largest diff and the easiest place to introduce a regression. I will do it as its own commit so it can be reverted alone.
- **R5 — Running Nest locally needs VPC access** to the RDS hosts. BP_Backend reaches the DB from this machine, so the DSN is obtainable, but the gateway uses a *second* datasource (`BP_SQL_DATABASE_*`) I have not tested.
- **R6 — `QuoteUploadModal.jsx:147` sends `documentType: data.documentType.deal_name`** — a deal name in a field every other caller fills with `po|invoice|contract|spend|quote`. I believe it is a bug; I have not touched it. See Q13.

---

## 8. Questions before I write code

- **Q9 (chat attachments — business logic).** `POST /document/embed-document` returns `{status, total_documents, total_chunks, processed:[{document_id, collection, chunk_count, metadata}], failed:[]}`. There is **no URL**. `AskRequest` has no `files` field; it has `file_path` (a *local* path) and `session_id`. How should an uploaded document reach the answer? Most likely: embed it, then send `session_id` on `/workflows/ask` and let the RAG pipeline retrieve it. Confirm, or tell me the intended semantics.
- **Q10 (gateway for Phase 4).** Run NestJS locally on `:3001` (needs `BP_SQL_DATABASE_*` creds, VPC reachability), or leave `VITE_API_URL` pointed at the deployed AWS gateway? The deployed one works today and keeps Cognito intact — but then step 6 can't be exercised without a deploy.
- **Q11 (WebSocket).** Proceed with steps 6–7 (un-regress `sessionId`, repoint the socket to BP_Backend, render `doc_action`)? Or leave the socket dead for now and let `/data-integration` refresh on a timer?
- **Q12 (Analyse demo mode).** Revert `4bb2bb9`'s removal of `WHERE deal_id = $1` so Analyse is deal-scoped again, or is whole-dataset Analyse a deliberate demo behaviour I should preserve?
- **Q13 (`documentType: deal_name`).** Bug, or a gateway convention?

---

## 9. Outcome (Phases 3–4, 2026-07-10)

**Live-verified against a local BP_Backend (:8000), a locally-run NestJS gateway (:3001), and headless Chrome driving the SPA (:3000).**

All 7 authenticated routes render with **zero console errors and zero failed requests**:
`/home`, `/dashboard`, `/data-integration`, `/actions`, `/users`, `/policies`, `/prompts`.

### Defects found and fixed (none of which were in the original plan)

| # | Defect | Where |
|---|---|---|
| 1 | Chat attachments never retrievable. `search()` filters the uploaded collection on a `session_id` payload field that `embed-document` never wrote; adding it then tripped `_is_internal_payload()`, which discards any key containing `"session_"`. Mutually exclusive. | BP_Backend |
| 2 | Keyword auto-conditions (`"invoice"` → `source_type='Invoice'`) applied to the uploaded collection, excluding the user's own attachment when the classifier disagreed with their wording. | BP_Backend |
| 3 | `DummyQdrant` stuck on the old `search()` API after production moved to `query_points()`. 4 tests failed with `AttributeError`. Now 1 failed / 12 passed. | BP_Backend |
| 4 | `useQueries()` called after a conditional early return — React throws once the analysis query resolves. | UI |
| 5 | 401 interceptor fired on *every* first error and returned `undefined`, resolving the promise and swallowing failures. | UI |
| 6 | Cognito idToken sent in cleartext to BP_Backend over `http://` via global axios defaults. Now isolated behind `services/aiApi`. | UI |
| 7 | `assets/agent.png` vs `Agent.png` — `vite build` broken on Linux. | UI |
| 8 | `fileUploadPromises` created and never awaited; `confirm-upload` could race the dialog close. | UI |
| 9 | Home-page greeting maps `word.length` over an array containing `user?.username?.toUpperCase()`; a user without a username white-screens the page. | UI |
| 10 | `dealName` sent as an object, `documentType` as `undefined` → S3 keys under `documents/undefined/`. | UI |
| 11 | `DetectionFinding` declared nine columns as `T \| null` with a bare `@Column()`. TypeORM cannot map `Object`, so **`feature/nick-branch` could not boot at all**. | API |
| 12 | The first hand-revert of the deal filters produced invalid SQL (`syntax error at or near "WHERE"`) and a parameter-count mismatch. Replaced by a verbatim splice of `4bb2bb9^`, which also restored `topSuppliersQuery`. | API |

### Corrections to §3 of this document
- The `sessionId` plumbing was **not** a regression: it never existed before `4bb2bb9`. A working version lives on `origin/feature/UAT-branch`, which is what the deployed Lambda runs. Ported from there.
- The deal-scoping removal **was** real and is reverted.

### Still open
- `DataIntegration` live progress needs the API deployed: the deployed gateway returns `sessionId` on `/analyse/presigned-url` but not `/data-integration/presigned-url`. The UI detects this and skips the subscription.
- `compliance.getDateRange` remains pinned to `new Date('2025-12-01')`.
- Detection + Pipeline modules remain undeployed and unused (agreed out of scope).
- `tests/test_rag_service.py::test_search_prioritises_target_documents_over_policies` still fails; it needs the test double to return synthetic hits, and fails identically before and after these changes.

### Local run
```
BP_Backend   systemd `procwise`            :8000
gateway      node --experimental-global-webcrypto dist/main.js   :3001   (Node 18 lacks global crypto)
UI           npx vite --port 3000 --host   :3000
```
`beyond_procwise_ui/.env.development.local` and `beyond_procwaise_api/.env` are gitignored and hold the local overrides, including `VITE_DEV_AUTH_BYPASS` / `AUTH_BYPASS`.
