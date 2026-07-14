# Task 3a — UI-to-Backend Coverage Matrix

Date: 2026-07-14. Verified by **loading each page in Chrome and watching the network**, then
cross-checking every payload against the live `bp_sqldb`. Not by reading code.

Stack under test: UI `localhost:3000` → Node gateway `localhost:3001` (28 endpoints) →
BP_Backend `localhost:8000` (2 endpoints).

---

## How the UI actually loads (this shapes everything below)

SpendIQ fetches **all data up front on page load** — 30 endpoints in one burst — into
`window.__SPENDIQ_DATA__`. Every view then renders from that object. Clicking a view fires
**no network request at all**.

The consequence is a trap: `SD(key, fallback)` returns a **hardcoded fallback** whenever a key
is missing. A failed endpoint therefore does not produce an error or an empty screen — it
produces a screen full of plausible sample data. The UI has a `SD_GAP()` marker and a
`__SPENDIQ_GAPS__` register for exactly this, and to its credit: **after visiting all 23 views,
`__SPENDIQ_GAPS__` is empty — no view fell back to sample data.** 33 of 34 data keys were
populated from real responses.

---

## The matrix

| Page | Module | What it should show | Endpoint | Data source | Status |
|---|---|---|---|---|---|
| **Dashboard** | KPI tiles | spend, savings, suppliers | `GET /spendiq/metrics` | `bp_*_trgt` | **LIVE** |
| | Category rows | spend by category | `/spendiq/metrics` → `spendByCategory` | — | **STUB** — returns `[]`; renders empty |
| | Trends charts | committed spend, quote volume | `GET /spendiq/trends` | `bp_*_trgt` | **LIVE** |
| | Executive summary | AI narrative | `GET /spendiq/analyse` | `bp_analysis_summary` | **LIVE** |
| | Ticker / attention | live KPIs, alerts | `/home-page`, `/home-page/attention` | `bp_agent_actions` | **LIVE** |
| **Analyse** | KPIs, top suppliers, levers | analytics over corpus | `GET /spendiq/analyse` | `bp_*_trgt` | **LIVE** |
| | Spend series | spend over time | `/spendiq/spend-series?range=FY26` | `bp_*_trgt` | **LIVE** |
| | Maverick spend | off-contract spend | `/spendiq/analyse` → `maverick` | — | **MISSING** — key absent from response |
| **Pipeline** | Deals, KPIs, stages | deal pipeline | `GET /spendiq/deals` | `bp_deal_overview` | **LIVE** (1 deal) |
| **Actions** | KPIs + work queue | discrepancies to action | `GET /actions`, `/spendiq/discrepancies` | `bp_extraction_discrepancy` | **LIVE** (797) |
| **Quotes** | Quote table | every quote | `GET /spendiq/quotes` | `bp_quote_trgt` | **LIVE** — 4 = 4 in DB |
| **Purchase orders** | PO table | every PO | `GET /spendiq/purchase-orders` | `bp_purchase_order_trgt` | **LIVE** — 21 = 21 |
| **Invoices** | Invoice table | every invoice | `GET /spendiq/invoices` | `bp_invoice_trgt` | **LIVE** — 19 = 19 |
| **Suppliers** | Supplier master + spend | suppliers | `GET /spendiq/suppliers` | `bp_supplier` | **LIVE** — 50 of 114 returned, `total:114` (paginated) |
| **Compliance** | Stats, issues, cases | compliance posture | `GET /compliance/getComplianceData` | `bp_extraction_discrepancy` | **LIVE** |
| **Obligations** | Runs, failures | contract obligations | `GET /obligations`, `/obligations/summary` (**BP_Backend**) | `bp_contract_obligation` | **LIVE** |
| **Negotiations** | Opportunities in play | negotiations | `GET /spendiq/negotiations` | `bp_opportunity` | **LIVE** (4) |
| **Alerts** | Event feed | recent agent events | `GET /spendiq/alerts` | `bp_agent_actions` | **LIVE** (100) |
| **Audit logs** | Event log | who did what | `GET /spendiq/audit-logs` | `bp_agent_actions` | **LIVE** (100) |
| **Prompts** | Prompt library | governed prompts | `GET /prompts/Allprompts` | `bp_prompt` | **LIVE** (100) |
| **Policies** | Policy library | governed policies | `GET /policies/Allpolicies` | `bp_policy` | **LIVE** (75) |
| **Users** | User list | users | `GET /user` | user table | **LIVE** (15) |
| **Admin** | Authority bands, cycles | config | `GET /spendiq/admin/config` | `bp_admin_config` | **LIVE** (read-only) |
| **Agent workspace** | Agent catalogue, canvas, run | build + run workflows | `/workflows/types`, `/agent-workflows/*` (**BP_Backend**) | `bp_agent_workflow` | **LIVE** — 14 agents |
| | **"Your agents" library** | user's saved agents | `GET /agents/AllAgents` | — | **BROKEN — 404** (see below) |
| **Demand intake** | Demand records | demand pipeline | `GET /spendiq/demand` | `bp_demand` | **MOCK** — seeded (see below) |
| **Requirements** | Requirement records | requirements | `GET /spendiq/requirements` | `bp_requirement` | **MOCK** — seeded |
| **Third-party risk** | Risk profiles | supplier risk | `GET /spendiq/tprm/suppliers` | `bp_tprm_supplier` | **MOCK** — seeded |
| **Report builder** | Composer + saved reports | build reports | `bp_reports` (persistence only) | `bp_reports` | **STUB** — self-declared "partial"; renders 0 rows |
| **Settings** | Profile | settings | — | — | **STUB** — self-declared "profile only" |

### Pages outside SpendIQ

| Page | Module | What it should show | Endpoint | Data source | Status |
|---|---|---|---|---|---|
| **/home** (Procurement home) | Live KPI ticker | spend, suppliers, discrepancies | `/spendiq/metrics`, `/spendiq/discrepancies/metrics` | `bp_*_trgt`, `bp_extraction_discrepancy` | **LIVE** — 114 suppliers, 797 findings, £175.0K |
| | Needs attention | approvals / open findings | same two endpoints, templated (`"{n} open discrepancies"`) | live counts | **LIVE** |
| | Tools & agents launcher | pick your tools | none (static catalogue) | `ProcurementHome/data.js` | **STATIC** — a fixed menu, not data. Fine. |
| | Ask bar | ask a question | `POST /workflows/ask` (**BP_Backend**) | RAG + corpus facts | **LIVE** |
| | Contact support | support agent | `POST /support/contact` (**BP_Backend**) | platform KG | **LIVE** |
| | User preferences | saved layout | `GET /users/me/preferences` | user table | **LIVE** |
| **/analyse** | Upload + deal routing | upload a document | presign on upload; `/analyse/deals` on demand | S3 + `process_monitor` | **LIVE** — no calls on mount by design |
| **/** (landing), `/login`, `/forgot-password`, … | auth + marketing | — | Cognito | — | out of scope |

**Checked and cleared:** `ProcurementHome/data.js` carries fabricated alert strings in its
Spanish and French locale blocks — "£240k PO pending your approval", "Acme missing ISO 27001
docs", "RFP — IT Hardware". They look alarming and they are **dead**: the keys (`al1`–`al4`,
`rc1`, `rc2`) are referenced by no component. English renders live templated counts instead
(`"{n} open discrepancies"` → 797). Worth deleting so nobody wires them back up; not a live
defect.

---

## The three findings that matter

### 1. MOCK presented as LIVE — Demand, Requirements, Third-party risk

The UI's own coverage map calls all three **"live"**. They are not. They are real endpoints
running real queries against tables whose **contents are fabricated**:

| Table | Rows | Anything tying it to a real uploaded document? |
|---|---|---|
| `proc.bp_demand` | 5 | **NONE** — no `document_id`, no `deal_id`, no `source_file` |
| `proc.bp_requirement` | 4 | **NONE** |
| `proc.bp_tprm_supplier` | 6 | **NONE** |

Nothing in these tables derives from anything a user ever uploaded. This is the most dangerous
shape of mock data, because it passes every naive check: the endpoint exists, returns 200, and
the query is genuine. Only the *contents* are invented. Three whole pages of the product are
telling a story about data that does not exist. (The UI's own notes do say "(seeded)" — but the
status next to it says `live`, and the status is what anyone reads.)

### 2. BROKEN and silent — the agent library

`GET /agents/AllAgents` returns **404 `{"message":"User not found"}`**, and the UI calls it
**12 times per page load**. The failure is swallowed by an empty `.catch(function(){})`
(`engine.js:1589`), so the "Your agents" section simply never renders and nobody is told. A
module that fails silently is worse than one that errors.

### 3. Empty where it should have data

* `spendByCategory` → `[]` — the Dashboard's category breakdown renders blank.
* `analyse.maverick` → key absent — the only one of 34 keys missing entirely; the maverick-spend
  module has no data source.

---

## What is genuinely healthy

The core procurement spine is **real and verified against the database, row for row**:
invoices 19 = 19, purchase orders 21 = 21, quotes 4 = 4, suppliers 114, discrepancies 797,
policies 75, prompts 100. The Agent workspace runs the real 14-agent catalogue against
BP_Backend. Obligations is the only other module reaching BP_Backend directly.

No view fell back to hardcoded sample data. That is a real credit to whoever built `SD_GAP`.
