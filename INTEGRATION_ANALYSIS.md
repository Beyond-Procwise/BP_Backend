# INTEGRATION_ANALYSIS.md

**Phase 1 — End-to-end, read-only analysis of the three repositories.**
Date: 2026-07-10 · No code was modified to produce this document.

---

## 0. Confirmed repository paths

| Repo | Path | Branch | State |
|---|---|---|---|
| Backend | `/home/muthu/PycharmProjects/BP_Backend` | `Development` | Clean (data/training artifacts untracked) |
| UI | `/home/muthu/PycharmProjects/beyond_procwise_ui` | `Nick_UI` | Clean (`yarn.lock` modified only) |
| API layer | `/home/muthu/PycharmProjects/beyond-procwaise-Api` | `feature/nick-branch` | Clean (`.idea/` untracked) |

**BP_Backend is already running locally**: `uvicorn api.main:app --host 0.0.0.0 --port 8000` (PID 6601, systemd `procwise.service`). `GET /health` returns `{"status":"ok", agent_nick:true, orchestrator:true, ...}`. Ollama is up on `:11434`. Postgres on `:5432`, Qdrant `:6333`, Neo4j `:7474/:7687`.

---

## 1. Three premise corrections (read this first)

The brief contains three assumptions that the code does not support. Each changes the shape of Phases 2–4, so I have **not** acted on any of them.

### 1.1 There is no Chrome extension — anywhere, in any branch, ever

- No `manifest.json` exists in `beyond_procwise_ui` (working tree, `public/`, or **any commit on any branch** — verified with `git log --all --diff-filter=A --name-only | grep -i manifest` → no results).
- No `chrome.runtime`, `browser_action`, `service_worker`, `content_script`, or `crx` references anywhere in `src/`.
- `vite-plugin-pwa` **is declared in `package.json` devDependencies but is never imported or registered in `vite.config.js`** — so not even a PWA web-app-manifest is generated today.
- The app is a plain Vite + React 19 SPA: `vite --port=3000 --host` for dev, `vite build` → `dist/` for production. `base: './'` (relative), which is at least extension-friendly if we ever went that way.
- `BP_Backend/google-chrome-stable_current_amd64.deb` is an unrelated Chrome *browser* install (used for headless document rendering), not an extension artifact.

**Consequence:** Phase 4 as written ("load unpacked … per the repo's existing extension setup — inspect manifest.json") cannot be executed. There is nothing to inspect. See Q1 in §8.

### 1.2 The UI is not a from-scratch redesign; it is an incremental branch

`Nick_UI` is **3 commits / 49 files / +2,541 −251 lines** ahead of `origin/main`:

```
2dec9a8 Analysis page All graph api integration completed and create file upload modal
6982fc5 Analysis graph functionality
56f2dcf New UI error fixes and UI changes
```

`Nick_UI` is the tip of the `Procwise2.0-Sprint-02` line (last commit 2026-06-10); `origin/main` already merged an earlier cut of that same line on 2026-04-17. The delta is concentrated in `modules/HomeAnalyse/{Analyse,Negotiate,Opportunities}`, `pages/PostExtractionActionCenter.jsx`, `routes/appRoutes.jsx`, and `utils/`.

The "Procwise 2.0" work *is* a substantial redesign relative to the pre-2.0 UI (`Procwise-Sprint-07` era), and old modules do still sit alongside new ones (`src/Appold.jsx`, `src/modules/ActionsOld/`, both `modules/Actions` and `modules/HomeActions`). But "all UI components have changed; treat the old UI as obsolete" is not literally true of the checked-out branch. See Q2 in §8.

### 1.3 BP_Backend is *not* the UI's main backend

This is the single most important finding for planning.

| Target | Env var | Distinct paths the UI calls |
|---|---|---|
| **AWS API Gateway → `beyond-procwaise-Api` (NestJS)** | `VITE_API_URL` | **~61** |
| **BP_Backend (FastAPI)** | `VITE_AI_API_URL` | **8 live + 2 commented-out** |
| AWS API Gateway WebSocket | `VITE_WEBSOCKET_URL` | 1 connection, payload ignored |

Essentially the entire product surface — Dashboard, Analyse, Negotiate, Opportunities, Compliance, Actions, Approvals, Contracts, Invoices, Purchase Orders, Users, Roles, Policies, Prompts, Processes, Data Integration, file upload — is served by the **NestJS API**, reading Postgres directly.

BP_Backend is used only as an "AI sidecar" for: agent-type/model listings, chat (`/workflows/ask`), chat history, document embedding, workflow execution (`/run`), and email dispatch.

**And the two backends do not talk to each other over HTTP at all.** `beyond-procwaise-Api` contains no HTTP client pointed at BP_Backend — no AI-API URL, no `:8000`, no EC2 IP as a *target* (`54.170.112.116` and `16.61.116.180` appear **only** in CORS allow-lists). Integration is via a **shared PostgreSQL `proc` schema**. This is deliberate and documented in the API repo:

> `pipeline.constants.ts:5-6` — tables are "owned by BP_Backend/deploy/sql … there is no HTTP hop between them"

The API even opens a **second TypeORM datasource** named `bpsqldbconnection` (`app.module.ts:108-123`, env `BP_SQL_DATABASE_*`) specifically to read BP_Backend's database for the dashboard2.0 / compliance / analyse modules.

**Consequence:** "integrate the new UI with the existing backend" is ambiguous. Most of what the UI needs already has a backend — it's just a *different* backend than BP_Backend. See Q3 in §8.

---

## 2. Architecture map

### 2.1 BP_Backend (Python / FastAPI)

- **Entry point:** `src/api/main.py:297` → `app = FastAPI(title="ProcWise API v4 (Definitive)", version="4.0", lifespan=lifespan)`
- **Local run:** `python src/api/main.py` (self-hosts `uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)`, `main.py:350`)
- **Prod run:** `procwise.service:22-27`, `PYTHONPATH=<root>:<root>/src`, `EnvironmentFile=.env`, docker-compose pre-start for Neo4j.
- **Required config** (`config/settings.py`, all `Field(...)`): `DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT, S3_BUCKET_NAME, S3_PREFIXES, QDRANT_URL, QDRANT_API_KEY, SES_DEFAULT_SENDER`.
- **79 routes** across 21 routers. Full inventory in §4.
- **Module boundary:** `src/api/routers/*` (thin) → `src/services/*` (business logic) → Postgres `proc.*` / Qdrant / Ollama / S3.
- **WebSocket:** `WS /ws/session/{session_id}` (`src/api/routers/ws.py:131`), fed by a Postgres `LISTEN`/`pg_notify` bridge (`session_notify_listener.py`) → `ws_manager.broadcast_to_session()`.

### 2.2 beyond-procwaise-Api (TypeScript / NestJS 11 on AWS Lambda)

- **Entry point:** `beyond_procwaise_api/src/main.ts:39-49` — `handler` = Nest over Express, wrapped by `@vendia/serverless-express`.
- **Local run:** `npm run build && npm run start` → listens on **port 3000** when `IS_OFFLINE=true` or `AWS_LAMBDA_FUNCTION_NAME` unset (`main.ts:28-37`). No `/dev` stage prefix locally.
- **Deploy:** Serverless Framework v3, `nodejs20.x`, `eu-west-1`, stage `dev`. **Every** function shares one handler (`dist/main.handler`); each REST module contributes an API Gateway path via its own `src/modules/**/<name>.yml`, listed in `serverless.yml:150-177`.
- **⚠️ There is no `{proxy+}` catch-all** (it's commented out at `serverless.yml:58-69`). A Nest controller with no corresponding `.yml` entry **is not reachable through API Gateway**.
- **Data:** PostgreSQL via TypeORM (two datasources) + raw parameterised SQL against schema `proc`. S3 bucket `procwisemvp`. No DynamoDB.

### 2.3 beyond_procwise_ui (React 19 / Vite 6 SPA)

- **Entry:** `src/main.jsx` → `src/App.jsx` → `src/routes/appRoutes.jsx` (lazy-loaded pages, `ProtectedRoute` + role gating).
- **Build:** `vite build`, `base: './'`, one manual chunk for `DynamicTable`.
- **Auth:** AWS Amplify → Cognito user pool `eu-west-1_3rDtdvAh1` (`main.jsx:42-43`).
- **State:** a single zustand store (`src/stores/authStore.js`) holding `{ roles }`; `roles.routeAccess` gates routing, `roles.roleAccess` gates permissions. AES-encrypted into `sessionStorage`.
- **Data fetching:** `@tanstack/react-query` over **global `axios`** (see §3.1).

### 2.4 Data flow (as-built)

```
                 Cognito (eu-west-1_3rDtdvAh1)
                        │ idToken
                        ▼
   ┌──────────────────────────────────────────────┐
   │  beyond_procwise_ui  (React SPA, :3000)      │
   └───────┬───────────────────────┬──────────────┘
           │ VITE_API_URL          │ VITE_AI_API_URL
           │ (~61 paths, Cognito)  │ (8 paths, NO auth, plain http)
           ▼                       ▼
   ┌────────────────────┐   ┌──────────────────────┐
   │ beyond-procwaise-  │   │ BP_Backend (FastAPI) │
   │ Api (NestJS/Lambda)│   │ :8000                │
   └─────────┬──────────┘   └──────────┬───────────┘
             │  TypeORM +              │ psycopg
             │  raw SQL                │
             ▼                         ▼
        ┌──────────────────────────────────┐
        │  PostgreSQL — schema `proc.*`    │  ◀── the ONLY integration point
        └──────────────────────────────────┘

   VITE_WEBSOCKET_URL ──▶ wss://olop83hwf5… (API Gateway WS)   [see §6.2 — likely dead]
   BP_Backend WS       ──▶ ws://…:8000/ws/session/{id}          [unused by UI]
```

---

## 3. UI contract inventory

### 3.1 The API layer is not actually centralised

`src/services/api.js` creates `const api = axios.create({ baseURL: VITE_API_URL })` — **and `api` is never imported anywhere** (0 hits for `api.get(`/`api.post(`). Every call in the app uses the **global `axios`** with a fully-interpolated template-literal URL.

`setAuthToken()` (`api.js:8-13`) therefore sets headers on `axios.defaults.headers.common`, which are **global**:

```js
axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;  // Cognito idToken
axios.defaults.headers.common["x-customer-id"] = `001`;              // hardcoded
```

**Consequences (both real, both need a decision):**
1. The Cognito **idToken is transmitted to BP_Backend** on every AI call — over **plain `http://`** (`VITE_AI_API_URL=http://54.170.112.116:8000`). Cleartext bearer token on the wire. BP_Backend ignores it entirely.
2. If the UI is ever served over `https://`, every `http://` call to BP_Backend is **blocked as mixed content**. Today it "works" only because the SPA is served over plain http.

There is also a latent bug in the 401 interceptor (`api.js:44-70`):

```js
if (err.response?.status === 401 || err?.message === '…expired' || !originalRequest._retry) { … }
```

The trailing `|| !originalRequest._retry` makes the branch fire on **any first-time failure** (500s, network errors, CORS failures), triggering a spurious `fetchAuthSession()` + retry. And when `newToken` is falsy the function **returns `undefined`**, resolving the promise instead of rejecting — errors are silently swallowed. This will actively confuse Phase 4 debugging.

### 3.2 Routes / views

| Path | Module | Backend it depends on |
|---|---|---|
| `/` | `LandingPage` | — |
| `/home` | `Home/ViewerHomePage` | **both** (gateway + BP_Backend chat) |
| `/dashboard` | `Dashboard` (Spend/PO/Contract/Invoice tabs) | gateway |
| `/data-integration` | `DataIntegration` | gateway + **WebSocket** |
| `/extraction-action-center` | `ExtractionActionCenter` | **none — makes zero API calls** |
| `/actions` | `Actions` | gateway (+ BP_Backend `/workflows/email`) |
| `/approvals` | `Approvals` | **none — API commented out, static page** |
| `/events` | `Events` | **none — fully static** |
| `/users` | `Users` | gateway |
| `/policies`, `/prompts` | `Policies`, `Prompts` | gateway (+ BP_Backend `/workflows/types`) |
| `/my-agents` | `CreateWorkFlow/WorkSpace` | gateway (+ BP_Backend `/run`, `/system/models`) |
| `/audit-logs` | `AuditLog` | gateway |
| `/contracts`, `/profile` | — | **route commented out / disabled** |

`HomeAnalyse` (Analyse / Negotiate / Opportunities / Compliance tabs) is reached from `/home`, not a top-level route.

### 3.3 UI → BP_Backend calls (the 8 that matter)

| Method | UI path | Backend reality | Verdict |
|---|---|---|---|
| GET | `/workflows/types` | ✅ exists | **OK** — `agentId/agentType/description` all present |
| GET | `/system/models` | ✅ exists | **OK** — `.models[].model`, `.details.parameter_size` present |
| GET | `/system/history/{email}` | ✅ exists | **OK** |
| POST | `/run` | ✅ exists (`RunRequest{process_id:int, payload?}`) | **OK** |
| POST | `/workflows/email` | ✅ exists (accepts form **or** JSON) | **OK** |
| POST | `/workflows/ask` | ✅ path exists, ⚠️ payload wrong | **BROKEN (silent)** — see below |
| POST | `/documents/embed-document` | ❌ **404** | **BROKEN** — path is `/document/…` (singular) |
| GET | `/Summary/get_summary_summary_get` | ❌ **404** | **BROKEN** — that's an OpenAPI *operationId*, not a path |

All verified live against `http://localhost:8000`:

```
GET  /documents/embed-document          -> 404      GET /document/embed-document  -> 405 (exists, POST-only)
POST /documents/embed-document          -> 404      POST /document/embed-document -> 422 (exists)
GET  /Summary/get_summary_summary_get   -> 404      GET /summary?persona=buyer    -> 404 (exists; nothing cached)
GET  /system/models                     -> 200      GET /workflows/types          -> 200
```

**Commented-out (intended, not live):** `POST /workflows/rank` (`SupplierPopup.jsx:24`), `POST /workflows/opportunities/{id}/reject` (`OpportunitiesPopup.jsx:43`).

### 3.4 The presigned-URL upload flow

Three near-identical copies (`Home/DataUpload.jsx:124-160`, `DataIntegration/index.jsx:89-123`, `HomeAnalyse/Analyse/QuoteUploadModal.jsx:84-124`). All hit the **gateway**, never BP_Backend:

1. `POST {VITE_API_URL}/analyse/presigned-url` (or `/data-integration/presigned-url`)
   - local: `{dealName?, documentType, fileNames[], source}` · s3: `{dealName?, documentType, source, accessKey, secretKey, bucketName, region}`
   - → `res.data.files[] = { fileName, url, processId, key }`
2. `PUT <presigned url>` direct to S3, `Content-Type: file.type`, **no auth header**.
3. `POST {VITE_API_URL}/analyse/confirm-upload` — `{ processId, success: true, key }` per file.

`documentType` ∈ `po | invoice | contract | spend | quote`. Sources: `local`, `s3` (sap/oracle disabled).

> ⚠️ `QuoteUploadModal.jsx:147` sends `documentType: data.documentType.deal_name` — it puts a **deal name** into the `documentType` field. Almost certainly a bug; flagged as Q6.

### 3.5 WebSocket usage

`modules/DataIntegration/index.jsx:53-78` is the **only** WebSocket consumer:

```js
new WebSocket(import.meta.env.VITE_WEBSOCKET_URL)
onmessage: () => setRefreshKey(prev => prev + 1)   // payload parsed then DISCARDED
```

No subscribe frame, no message shape, no reconnect, no heartbeat. Any message = "refetch the monitor table."

### 3.6 Static / dead data still in the tree

Relevant because the brief says "prioritize wiring the new UI to real backend data":

- **Fully static, no API:** `modules/Events/index.jsx:27-105`, `modules/Approvals/index.jsx` (API call commented at `:59`), `modules/ExtractionReport/Pages.jsx:44-241` (`trendData`, `pieData`, `opportunityBars`).
- **Dead leftovers (harmless):** hardcoded `data = [...]` arrays at the top of `Actions/Tables/{Opportunities,Quotes,Approvals}Table.jsx:8-37` — unused; the tables render server data via `DynamicTable`.
- **Disabled:** `modules/Contracts/index.jsx` (route commented out; only console-logs `GET /user/profile`).
- **Legacy duplicate:** `modules/ActionsOld/*` (calls `/actions`, `/actions/table`, `/actions/update-status`), superseded by `modules/Actions/*`.
- **Hardcoded values:** `x-customer-id: '001'`, `model_name: 'llama3.2'`, `recipients: 'procurement.team@company.com'`, `sender: 'supplier@example.com'`.

### 3.7 Standard table contract

`components/common/DynamicTable.jsx:52-67` (and `HomeDynamicTable.jsx:50`) send `{ page (1-based), limit, sortField?, sortOrder?, search?, ...defaultParams }` and expect `{ data: [...], total }` unless a per-table `mapResponse` overrides it.

**Response nesting is inconsistent.** `res.data.data` (double-nested) is required for `agents/AllAgents`, `process/*`, `prompts/names-and-ids`, `policies/id-name`; everything else is single `res.data`.

---

## 4. BP_Backend endpoint inventory (79 routes)

Grouped; full request/response shapes were captured per-router. Highlights and traps:

| Router (prefix) | Routes | Notes |
|---|---|---|
| root | `GET /`, `GET /health` | |
| `/agents` | list, status, manifest, reload-policies, reload-governance, process-document, execute | **two routers share `/agents`** (`agents.py` + `governance.py` → `POST /agents/govern`) |
| `/document` | `extract-from-s3`, `embed-document` | **singular** `document` |
| `/email` | `emailwatcher` | |
| `/workflows` | ask, rank, quotes/evaluate, opportunities, opportunities/{id}/reject, extract, email, email/batch, {id}/email/dispatch-all, negotiate, approvals, supplier-interaction, discrepancy, types | ⚠️ **double-prefix**: real paths are `/workflows/workflows/{id}/status`, `/workflows/workflows/{id}/events`, `/workflows/system/workflows/active` |
| `/system` | models, history/{user_id} | |
| `/` | `POST /run` | |
| `/stream` | `POST /stream/plan` | **SSE** `text/event-stream`; events `connected, planning, plan_created, step_start, agent_thinking, step_complete, execution_complete, error` |
| `/training` | dispatch | |
| `/vendors` | onboard (HTML), onboard/upload, {sid}/correct, {sid}/save | |
| `/metrics` | extraction, extraction/recent, extraction/templates | |
| `/extraction` | proposals (list/get/approve/reject/run) | the human-in-the-loop hint governance loop |
| `/suppliers` | reviews/queue, reviews, reviews/sweep, reviews/{id}/confirm\|reject, research/batch, enrichment/reviews, enrichment/{id}/apply\|reject, {id}/research, {id}/enrichment | **two routers share `/suppliers`** |
| `/deals` | orphans, {id}/summary, analysis-summary, {id}/analysis-summary, analysis-summary/sync, {id}/reconcile, **{id}/negotiate** | **two routers share `/deals`** (`deal_summary.py` + `negotiate.py`) |
| `/opportunities` | dashboard, list, {id}/stage, sync | |
| `/promotion` | run, canonicalize-po, quote-chains, review-queue, review/{doc_type}/{doc_pk}/approve | |
| `/summary` | `POST /summary`, `GET /summary?persona=…`, history, precompute, {summary_id} | `persona` is **required** on GET |
| `/session` | extraction-status | |
| `ws` | `WS /ws/session/{session_id}` | |

**`requirements.py` exists but is NOT mounted** in the live `src/api/main.py` (only in the `.claude/worktrees/conformance-phase1` copy). Any `/requirements/*` call 404s.

Most handlers return **bare dicts** with no `response_model`, so shapes are unenforced except on `/document/embed-document`, `/email/emailwatcher`, `/training/dispatch`, `/system/history/{user_id}`, `/workflows/types`.

### WebSocket payload (BP_Backend), `ws.py:110-125`

```json
{ "session_id":"…", "action_status":"completed|partially_completed|failed",
  "total":4, "target":3, "discrepancy":0, "failed":1,
  "duplicate":0, "updated":0, "needs_review":0, "unsupported":0,
  "documents":[{"file_path":"…","doc_action":"duplicate|updated|needs_review|unsupported"}],
  "resolved_at":"2026-06-30T12:34:56Z", "category":["…"], "deal_name":["…"] }
```

---

## 5. beyond-procwaise-Api inventory

~55 deployed REST routes across 25 modules. Full list captured; the load-bearing ones for the UI are `dashboard`, `analyse/*`, `negotiate/getNegotiateData`, `compliance/getComplianceData`, `opportunitiesnew`, `contract-dashboard/*`, `invoices/*`, `purchasedashboard`, `purchaseorder/list`, `quotes`, `opportunities`, `approvals`, `email-actions/*`, `agent-actions/*`, `actions/*`, `data-integration/*`, `user/*`, `roles`, `policies/*`, `prompts/*`, `process/*`, `agents/*`, `home-page`, `landing-count/summary`, `document/*`, `upload`.

### 5.1 The recent changes, precisely

| Commit | What it did |
|---|---|
| `ec81c9d` | Added **Detection** module (`@Controller('discrepancies')`: `GET /discrepancies`, `GET /discrepancies/metrics`, `POST /discrepancies/resolve`) over `proc.bp_detection_finding`; and **Pipeline** module (7-stage: `demand, strategy, sourcing, negotiation, award, execution, supplier_mgmt`) over `proc.bp_pipeline_record/_stage/_stage_gate/_stage_blocking_issue`. Gate approval is **fail-closed**: a blocked approval returns **409 `{error:'gate_blocked', …}`**, and the DB trigger `proc.bp_stage_gate_guard` (PG `23514`) is caught and re-mapped to the same 409. |
| `ed84156` | Bug fix: `PATCH /pipeline/issues/:findingId` patched `lifecycle_status` but not `status`, so clearing an issue unblocked the stage gate while `GET /discrepancies` still showed it open. Added `STATUS_FOR_LIFECYCLE` map (`open→open, remediating→open, resolved→resolved, accepted_risk→ignored`) + `resolved_at` toggling. |
| `9a1db4a` | Renamed the issues controller `@Controller('compliance')` → `@Controller('pipeline/issues')` to stop it colliding with dashboard2.0's `complianceController`. `GET /compliance/issues` → **`GET /pipeline/issues`**; `PATCH /compliance/issues/{id}` → **`PATCH /pipeline/issues/{id}`**. |
| `4bb2bb9` | (a) Added CORS origin `http://16.61.116.180` to every module `.yml` and `response.util.ts:33`. (b) **"Demo mode"**: `analyse.service.ts` `getAnalyseData(deal_id)` → `getAnalyseData()`, with **every `WHERE deal_id = $1` commented out** — analyse endpoints now return the whole dataset regardless of deal. |
| `0982ea8` (context) | Replaced hardcoded compliance arrays with live SQL against `bpsqldbconnection`. Compliance is now **real, not stubbed** — except `getDateRange` is pinned to `new Date('2025-12-01')` ("latest month available in your sample data", `compliance.service.ts:638-679`). |

### 5.2 ⚠️ Detection and Pipeline are built but unreachable and unused

- There is **no `detection.yml` and no `pipeline.yml`**, and neither is listed in `serverless.yml` functions. With no `{proxy+}` catch-all, **neither module is exposed through API Gateway**. They only work against a local Nest server on `:3000`.
- The UI calls **none** of them (`grep -rn "discrepancies|pipeline/records|pipeline/issues" src` → 0 hits). The `9a1db4a` commit message itself says "Nothing calls the endpoints yet."
- `subscription.yml` exists and `SubscriptionModule` is imported, but the yml is **not** in `serverless.yml` functions either — also undeployed.

So the "smaller set of API-layer changes to reconcile" is, concretely: **a 7-stage procurement pipeline + a detection-findings surface that are finished server-side, deployed nowhere, and consumed by nobody.** The UI page that would obviously consume them — `/extraction-action-center` — currently makes **zero API calls**.

---

## 6. Gap analysis

### (a) UI-expected contracts with no backend counterpart

| # | UI expects | Reality | Severity |
|---|---|---|---|
| A1 | `POST {AI}/documents/embed-document` | **404.** Real path `POST /document/embed-document` | **High** — chat file upload is dead |
| A2 | form field `file` (singular), reads `res.data.data.fileUrl` | Backend takes `files: List[UploadFile]` and returns `{status,total_documents,total_chunks,processed[],failed[]}` — **there is no `data.fileUrl`** | **High** — broken even after fixing A1 |
| A3 | `GET {AI}/Summary/get_summary_summary_get?deal_id=` | **404.** An OpenAPI **operationId** was pasted in as a URL. Real: `GET /summary?persona=<required>&deal_id=` | **High** — Analyse summary tab is dead |
| A4 | `POST {AI}/workflows/ask` body `{query, user_id, model_name:'llama3.2', files:[url]}` | `AskRequest` has **no `files` field**, and no `model_config` → pydantic **silently drops** it. So attached files are ignored with no error. Also `file_path` (a *local path*) is the real field — not a URL list. | **High (silent)** |
| A5 | `model_name: 'llama3.2'` | Not an installed model. Ollama has `BeyondProcwise/AgentNick:{latest,unified,extract}`, `qwen3:30b`, `nuextract:3.8b`. Also violates the standing "AgentNick is the only base model" constraint. | **High** |
| A6 | `/extraction-action-center` page | Makes **no API calls**; the API's `/discrepancies` exists but is undeployed | **Medium** — feature is a shell |
| A7 | `/events`, `/approvals`, `ExtractionReport` | Fully static; no backend wired | **Medium** |
| A8 | `wss://olop83hwf5…` | The WS function is **commented out** of `serverless.yml:166`, and `websocket.yml` points at `dist/modules/websocket/notify.main` — **`notify.ts` does not exist** (the real file is `sendMessage.ts`) | **High** — DataIntegration live refresh likely never fires |
| A9 | `res.data.data` double-nesting | Only some gateway endpoints do this | Low (already handled per-call) |

### (b) Backend / API capability the new UI does not use

**BP_Backend (large, mature, unconsumed):**
- `GET /deals/analysis-summary`, `/deals/{id}/analysis-summary`, `/deals/{id}/summary`, `/deals/orphans`, `POST /deals/{id}/reconcile` — verified live, returning **real deals**.
- `GET /deals/{id}/negotiate` — a complete Negotiate dashboard (KPIs, offer version history, cost-over-time, baseline-vs-current, demand-vs-volume).
- `GET /opportunities/dashboard` — verified live: `{totalOpportunity:24, identified:24, potential:"£195k", …}`, **already camelCase and UI-shaped**.
- `GET /summary`, `POST /summary`, `/summary/history`, `/summary/precompute` — persona-driven AI summaries.
- `POST /stream/plan` — SSE agent planning stream (no UI consumer).
- `WS /ws/session/{session_id}` — rich per-document `doc_action` breakdown (`duplicate/updated/needs_review/unsupported`), **exactly** what `/data-integration` and `/extraction-action-center` need. Unused.
- `/promotion/*`, `/suppliers/reviews*`, `/suppliers/enrichment*`, `/extraction/proposals*`, `/metrics/extraction*`, `/vendors/onboard*`, `/agents/govern`, `/training/dispatch`.

**beyond-procwaise-Api:** `/discrepancies*`, `/pipeline/*` (built, undeployed, unused); `/analyse/top-suppliers`, `/analyse/reports`, `/analyse/top-deals`, `/compliance/compliance-trends`, `/compliance/flagged-cases-by-month`, `/subscription/*`.

> **Note the duplication:** BP_Backend and the gateway **both** implement Opportunities, Negotiate and Analyse, from the same `proc` tables, with different shapes. The UI currently uses the gateway's. My notes record a 2026-07-08 repoint of Negotiate/Opportunities onto BP_Backend — **that work is not on `Nick_UI`** (`CustomTabs.jsx:31` still calls `${VITE_API_URL}/negotiate/getNegotiateData`). Either it lives on an unmerged branch or it was lost. This is Q3/Q4.

### (c) Mismatches in schema / naming / auth / versioning

**Auth — the big one.**

| | Gateway | BP_Backend |
|---|---|---|
| Transport | `https://` | **`http://`** |
| Gate | API Gateway **Cognito User Pool authorizer** per function + Nest `CognitoGuard` | **None** |
| Headers required | `Authorization: Bearer <idToken>` + `x-customer-id` | none |
| CORS | per-`.yml` allow-list, echoes origin, `Allow-Credentials: true` | `allow_origins=["*"]`, `allow_credentials=False` |

- `src/api/auth.py` defines `verify_api_key(x_api_key)` but **nothing imports it**. Even if wired, it no-ops when `PROCWISE_API_KEY` is unset — and it is unset. **BP_Backend is fully open.**
- `main.py:302-303` comments claim the WS route handles "auth … via `token=` query param". **`ws.py:131-135` accepts no token.** The comment is false; the socket is unauthenticated.
- Nest's `CognitoGuard` (`cognito.guard.ts:32-39`) calls `jwt.decode` — it **does not verify the signature**. Security relies entirely on the API Gateway authorizer in front of it. Locally (`npm run start`, no gateway), the guard is therefore trivially bypassable with any well-formed JWT.
- `allow_credentials` is `false` on BP_Backend precisely because origins are `*` — so cookie/credentialed requests would fail; the UI happens not to need them.

**Naming / shape:**
- Gateway is mostly `snake_case` (`prompt_id`, `policy_id`, `total_spend`) with camelCase pockets (`opportunityReviewCount`, `totalPotentialSavings`, `poVolume`).
- BP_Backend is camelCase for `/workflows/types` (`agentId`, `agentType`) and `/opportunities/dashboard`, snake_case for `/workflows/ask` (`follow_ups`).
- Any UI move from gateway→BP_Backend for Analyse/Negotiate/Opportunities is a **response-shape rewrite**, not a URL swap.

**Versioning:** none anywhere. No `/v1`. BP_Backend calls itself "v4.0" in its title only. Gateway pins stage `/dev` into `VITE_API_URL`, so promoting to another stage means editing the env file.

### (d) Missing modules

| Side | Missing |
|---|---|
| UI | Chrome-extension packaging **in its entirety** (§1.1). PWA plugin declared but unregistered. No consumer for `/discrepancies`, `/pipeline/*`, `/stream/plan`, BP_Backend WS. `/extraction-action-center` is a shell. |
| API | `detection.yml`, `pipeline.yml` (modules undeployable). `notify.ts` (referenced by `websocket.yml`, does not exist). `subscription.yml` unwired. No `.env`/`.env.example` committed — only `DATABASE_*` + `NODE_ENV` are Joi-validated. |
| BP_Backend | `requirements.py` router not mounted. `auth.py` written but never wired. |

---

## 7. Other findings worth knowing before Phase 2

1. **Analyse "demo mode"** (`4bb2bb9`) removed all `deal_id` filtering from `analyse.service.ts`. Any Analyse work will show whole-dataset numbers until this is reverted. `compliance.getDateRange` is hardcoded to `2025-12-01`.
2. **RDS is inside a VPC** (`serverless.yml:100-106`). Running the Nest API locally needs network reachability to those Postgres hosts. BP_Backend already reaches `bp_sqldb` (it is live), so the DSN is obtainable.
3. **`GET /summary?persona=buyer` returns 404** on the live server — nothing is cached. `POST /summary` or `/summary/precompute` must run first. Any UI wiring must handle the empty-cache path.
4. The API's local port (**3000**) **collides with the UI dev server's port (3000)**. One must move.

---

## 8. Open questions — I need answers before Phase 2

I will not guess business logic. These are the decisions I can't make from the code:

- **Q1 (Chrome extension).** No extension has ever existed in this repo. Did you mean (a) build a real MV3 extension from scratch, (b) enable the already-installed-but-unused `vite-plugin-pwa` so Chrome can "Install app", or (c) simply run the SPA in Chrome and test the flows? These are very different amounts of work.
- **Q2 (which UI is "the redesign").** `Nick_UI` is 3 commits ahead of `main`, not a rewrite. Is `Nick_UI` the intended target, or is the redesign on a branch/working copy I haven't found?
- **Q3 (who owns Analyse / Negotiate / Opportunities / Compliance).** Both backends implement these from the same tables. Do we (a) keep the UI on the gateway and leave BP_Backend's versions unused, (b) repoint the UI to BP_Backend (my notes say this was done on 2026-07-08 but it is **not on this branch**), or (c) have the gateway proxy BP_Backend? Today there is **no HTTP link between the two services at all** — only the shared `proc` schema.
- **Q4 (the lost repoint).** Do you know where the 2026-07-08 Negotiate/Opportunities repoint work went? If it exists somewhere, reusing it beats redoing it.
- **Q5 (Detection/Pipeline).** The API's new 7-stage pipeline + detection findings are finished, undeployed, and have no UI. Is wiring `/extraction-action-center` to them part of this job, or out of scope?
- **Q6 (`documentType: deal_name`).** `QuoteUploadModal.jsx:147` sends a **deal name** in the `documentType` field, where every other caller sends `po|invoice|contract|spend|quote`. Bug, or an intentional gateway convention I shouldn't touch?
- **Q7 (auth for BP_Backend).** BP_Backend is unauthenticated and receives your Cognito idToken in cleartext over `http://`. For the local Phase 4 test this is harmless. Do you want me to (a) leave it, (b) wire the existing `verify_api_key`, or (c) validate the Cognito JWT properly? Note (c) is the only option that makes BP_Backend safe to expose directly.
- **Q8 (`model_name`).** The UI hardcodes `llama3.2`, which isn't installed and contradicts the "AgentNick is the only base model" rule. Confirm I should replace it with `BeyondProcwise/AgentNick:unified`.

---

## 9. Assumptions made in this document

1. **`Nick_UI`, `Development`, and `feature/nick-branch` are the branches in scope.** No other checkout of these repos was inspected.
2. **The live `bp_sqldb` that BP_Backend (PID 6601) is connected to is the same Postgres the Nest API's `bpsqldbconnection` targets.** Strongly implied by `pipeline.constants.ts` and the shared `proc.bp_*` table names, but I did not connect as the API to prove it.
3. **The deployed `wss://olop83hwf5…` gateway was deployed from an earlier state of the API repo.** The current `feature/nick-branch` config would not deploy it (`serverless.yml:166` commented; handler `notify.main` missing). I could not verify what is actually live behind that URL.
4. **`.serverless/` and `procwaise-beyond.zip` are stale build artifacts**, not authoritative deploy state.
5. **`git diff main...feature/nick-branch` = 214 files / +76,319 lines** because `main` in the API repo is essentially empty — so "changes vs main" is the entire codebase. I therefore scoped "recent API changes" to the four commits you'd expect (§5.1) rather than the whole diff.
6. **Commented-out UI code represents intent, not contract.** I listed it (`/workflows/rank`, `/workflows/opportunities/{id}/reject`) but did not treat it as required.
7. **I did not exercise any mutating endpoint.** All live verification was `GET`, plus `POST` with `{}` bodies purely to distinguish 404 (absent) from 422 (present, bad payload).

---

## 10. What Phase 2 will need to cover

Sketch only — the actual plan comes after you answer §8.

1. **Trivial, unambiguous fixes** (safe regardless of Q1–Q8): A1 path, A2 field name + response read, A3 path + required `persona`, A4 `files`→`file_path`, A5 model name, the `api.js` 401-interceptor bug.
2. **A decision on backend ownership** (Q3) → then either an adapter layer mapping BP_Backend's camelCase dashboards onto the UI's current expectations, or leave the gateway in place.
3. **WebSocket strategy** (A8): fix/deploy the API Gateway socket, or point `/data-integration` at BP_Backend's `WS /ws/session/{session_id}` and actually *use* the `doc_action` payload.
4. **Detection/Pipeline exposure** (Q5): add `detection.yml` + `pipeline.yml`, or a `{proxy+}`, and build the Extraction Action Center UI.
5. **Local run topology**: move the Nest API off `:3000` (UI owns it); point `VITE_AI_API_URL` at `http://localhost:8000`; set `PROCWISE_CORS_ORIGINS` explicitly.
6. **Phase 4 packaging** — entirely dependent on Q1.
