# Report Builder: reports index, live data, real export

**Date:** 2026-07-13
**Status:** Draft — awaiting review
**Repos touched:** `beyond_procwise_ui` (spendiq-ui), `beyond-procwaise-Api` (Node gateway), `BP_Backend` (FastAPI)

---

## 1. The problem

Click **Reports** on the home page, fill in the report name / type / period, click **Generate**, and you land in the report builder on a blank "Untitled report". There is no way to reach a report you saved earlier. The three fields you just filled in are silently thrown away.

Underneath that, two further problems mean the feature has no working data path at all:

- Every number in the builder is a **hardcoded constant in the browser**, frozen at Jan–Mar 2025. There is no endpoint behind the tiles, so there is nothing to point at a customer's data when they arrive — only a rewrite.
- **Export PDF** is `window.print()`. It prints the screen. No file is produced, nothing is stored, and nothing can be shared.

This spec covers all three: the missing front door, the missing data path, and the fake export.

**Scope decision (2026-07-13):** the tiles **keep showing demo values for now** — a customer's real numbers only exist once their documents are flowing in. But the demo values move *behind the real API*, so the UI, gateway, period filter and export all run end-to-end against a real data path from day one. Going live for a customer becomes a config flip plus SQL, not a rewrite. See Stage 2.

---

## 2. What already exists (verified, not assumed)

| Thing | State |
|---|---|
| `GET/POST/DELETE /spendiq/reports` | **Live.** Gateway, `spendiq.controller.ts:144-157` |
| `proc.bp_reports` table | **Live.** Stores each save as a row with a JSONB `definition` |
| Builder loads saved reports on mount | **Yes** — into a *hidden* History panel (`engine.js:5682`) |
| Reports list anywhere in the UI | **No** |
| Home modal "Recent reports" table | **Hardcoded empty** (`ProcurementHome/index.jsx:394`: `const reportRows = []`) |
| Generate button reads its form inputs | **No** — inputs are uncontrolled; it just navigates |
| Builder KPI/chart data | **Hardcoded demo constants** (`METRICS`, `GRAPHS`) |
| Export | `window.print()` (`engine.js:5627`) |
| PDF library in either repo | **None** |
| S3 client + presigner in gateway | **Yes** (`@aws-sdk/client-s3`, `s3-request-presigner`) |
| `proc.bp_reports.report_url` column | **Exists, never written by any code** |

**The single most important fact for this design:** a saved report definition stores only *references*, never values —
`{type:'kpi', metric:'savings', period:'mar-2025'}`, not `£525,700`. Charts likewise: `{type:'graph', key:'spendSave', range:'FY'}`.

Consequence: **swapping demo data for live data requires no migration of saved reports.** They already resolve against a data source at render time; today that source happens to be a hardcoded object. There are exactly two lookups to redirect.

---

## 3. Design

Three stages. Each one lands independently, is verifiable on the live server, and is useful on its own. Later stages do not require earlier ones to be perfect.

### Stage 1 — The reports index (the front door)

**Behaviour.** `/spendiq?view=reportbuilder` no longer opens a blank canvas. It opens **My reports**: one row per report, showing its version count and when it was last touched, with a primary **Generate** button. Click a report → the composer opens with that report restored. Click Generate → the composer opens on a new report. The composer gains a **← All reports** link back to the index.

Deep-link from home still works: **Generate** on the home modal goes straight to the composer (skipping the index) and now **carries the name, type and period** you typed, so the report opens pre-titled and pre-scoped instead of as "Untitled report".

The home modal's **Recent reports** table is wired to the same endpoint and stops being permanently empty.

**Data model.** Today every click of *Save version* writes a new row named `"Q1 Review · v3"`, so "report" and "version" are the same thing to the database. To show distinct reports with versions nested underneath, add two columns:

```sql
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS report_key text;
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS version    integer;
CREATE INDEX IF NOT EXISTS ix_bp_reports_report_key ON proc.bp_reports (report_key);
```

Backfill existing rows by parsing the `· vN` suffix off `report_name` (base name → `report_key`, N → `version`). New saves write both columns directly, so the grouping stops depending on string parsing. `report_type` finally stores the Type chosen on the home modal (Executive / Compliance / Category / Savings) instead of the constant `'spendiq'`.

**API (gateway).** Two new endpoints; the two existing ones are unchanged:

- `GET /spendiq/reports/index` → one row per report:
  `{ data: [{ key, name, type, versions, latest_id, latest_ts }], total }`
- `GET /spendiq/reports/:id` → a single report's snapshot: `{ id, name, when, snap }`
  (needed so opening one report doesn't mean downloading all of them)

`GET /spendiq/reports` (flat list) stays as-is — it still backs the per-report History panel.

**Also in scope:** delete a report from the index. The `DELETE` endpoint already exists and has never been called from anywhere.

---

### Stage 2 — Make the data flow real (values stay demo for now)

**The decision.** The tiles keep showing demo numbers for now — a customer's real figures only exist once their documents are flowing in, and today's corpus is too thin to demo against (see the £0 note below). What changes is **where those numbers come from**. Today they are hardcoded in the browser, which means there is no data path at all: nothing to switch on, nothing to test, and a rewrite of the builder on the day a customer arrives.

So: **move the demo data out of the browser and behind the real endpoint**, serving it through the exact contract the live query will implement. Everything — UI, gateway, period filtering, export — runs end-to-end against a real API from day one. Going live for a customer then means implementing the live provider and flipping a flag, not touching the builder.

**Approach.** One new gateway endpoint returns everything a report needs in a single round-trip, rather than the builder firing twenty calls:

`GET /spendiq/report-data?from=<ISO date>&to=<ISO date>` →
```
{ source: 'demo' | 'live',
  period: { from, to },
  kpis:   { <key>: { big, delta, tone, bullets[], sparkline[] } },
  series: { <key>: { labels[], data[] | bars[]+line[] } } }
```

The shape deliberately mirrors the existing `METRICS` / `GRAPHS` objects, so the builder's renderers barely change — they stop reading a constant and start reading a fetched payload.

**Two providers, one contract.** Behind that endpoint sit two implementations selected by config (`REPORTS_DATA_SOURCE=demo|live`, defaulting to `demo`):

- **`demo`** — serves the current constants, relocated server-side and extended to cover all 8 KPIs and all 12 charts (including the four that have no live source today). It respects the `from`/`to` window by slicing its series, so period switching genuinely exercises the same code path the live provider will.
- **`live`** — the real SQL, per the mapping table below. Implemented for the tiles that *can* be real; the rest raise a clear "no data source" for that tile rather than silently returning demo figures. Nothing in the UI changes when the flag flips.

**`source` is returned in the payload, and the UI renders a `Demo data` badge whenever it is `demo`** — on screen and on the exported PDF. Demo numbers are fine; demo numbers that a customer mistakes for their own are not. This is the one non-negotiable in this stage.

**Period becomes real plumbing.** Today the selector only picks which hardcoded slice to read, and `/spendiq/metrics` and `/spendiq/trends` accept **no date filter at all**. The new endpoint takes a real `from`/`to`, the demo provider honours it, and the live provider plumbs it into the SQL. The fixed Mar/Feb/Jan 2025 options are replaced by real ranges (This month / This quarter / Year to date / Custom).

**Live-readiness of each tile.** This is what the `live` provider can and cannot do today — i.e. what a customer actually gets when the flag flips. Verified against the gateway's SQL and the `bp_*_trgt` schema:

*KPIs (8):*

| Tile | Live source | Ready? |
|---|---|---|
| Savings Secured | `bp_opportunity.realised_savings_gbp` | Query works — **but returns £0 today** (see below) |
| In-Flight Negotiations | `bp_opportunity` (open stages) | Yes |
| Opportunity Pipeline | `bp_opportunity.financial_impact_gbp` | Yes |
| Cycle Time to PO | `bp_deal_overview.cycle_days_quote_to_po` | Yes |
| 3-Way Match | `bp_deal_overview.three_way_match` | Yes |
| Non-PO Spend | `bp_invoice_trgt` vs `bp_purchase_order_trgt` (PO-backed share) | Yes |
| Duplicate Risk | `bp_extraction_discrepancy` (duplicate findings) | Yes |
| **Tail Spend Visibility** | needs a spend-category / contract taxonomy | **No — data does not exist** |

*Charts (12):*

| Tile | Live source | Ready? |
|---|---|---|
| Committed spend & savings | `trends.spendByMonth` + `trends.savings` | Yes |
| Realised savings trend | `trends.savings` | Yes |
| Committed spend by month | `trends.spendByMonth` | Yes |
| Quote volume | `trends.quoteVolume` | Yes |
| 3-way match trend | `trends.matchRate` | Yes |
| Cycle time to PO | `trends.cycleTime` | Yes |
| Off-contract spend | `trends.offContract` | Yes |
| Top suppliers by spend | `metrics.topSuppliers` | Yes |
| Compliance rate | no distinct compliance series exists; would duplicate 3-way match | **No — needs contract linkage** |
| Tail spend breakdown | only maverick findings exist, not a 4-way composition | **No — partial data only** |
| **Spend by category** | no category column exists in **any** extracted table (`spendiq.service.ts:566`) | **No — data does not exist** |
| **Supplier risk profile** (radar) | only a single `risk_score`; the radar needs 6 dimensions | **No — data does not exist** |

So when a customer's data arrives, **7 of 8 KPIs and 8 of 12 charts light up immediately**. The other five need data the platform does not yet capture — a spend-category taxonomy, contract linkage, and multi-dimension supplier risk. Those are data-capture projects, not UI work, and they are out of scope here. Until then those five tiles keep serving demo values from the `demo` provider, clearly badged.

**The £0 that is coming.** Worth knowing now, even though it doesn't bite while we're on demo data: "Savings Secured" shows a demo **£525,700**, but the live query against today's corpus returns **£0**. All 24 rows in `bp_opportunity` sit at stage `identified`; nothing has ever been marked realised. The *pipeline* figure is real and healthy — realised savings is genuinely zero until opportunities are progressed through their stages. That is a data/process gap, not a bug, and it is exactly the kind of thing the badge exists to stop us papering over.

**Dead code.** Roughly half the RB6 module is an unreachable earlier generation (`SECTION_DEFS`, the hero/breakdown/cycle/compliance card renderers, the word-budget enforcer and its canned narrative drafts, `HERO`, `SUMMARIES`, `HERO_TREND_VALUES`, `CYCLE_TREND`, `TEMPLATE_DEFAULTS`). It is where much of the fake data lives. It gets deleted as part of this stage.

---

### Stage 3 — Real export

**Why it's cheap.** The builder already rasterises every chart to a PNG data-URI in the browser (`graphPNG` → `toBase64Image`) and `presentationHTML(true)` already assembles a complete, self-contained print document. So the server does **not** need a headless browser or Chart.js. It needs HTML + CSS → PDF, nothing more.

**Flow.**
1. Builder POSTs the rendered document to BP_Backend: `POST /reports/export` with `{ report_id, name, format: 'pdf'|'xlsx', html, css }`.
2. BP_Backend renders it (WeasyPrint for PDF — pure Python, no browser; `openpyxl` for XLSX, exporting the underlying tables rather than the layout).
3. The artifact is uploaded to S3 (bucket `procwisemvp`, `eu-west-1` — the same one document ingest already uses).
4. Download is served by the gateway's **existing** presigned-URL helper (`Documents/s3.service.ts:42-67`), which already sets `Content-Disposition: attachment` and a 1-hour expiry. It needs generalising — it currently hardcodes a `Static Policy/` key prefix — but it does not need writing.
5. The gateway writes the S3 key to `proc.bp_reports.report_url` — the column that has existed all along and never been populated by anything.
6. The reports index gains a download affordance per report; the home modal's download icon (currently a toast that says "Downloading report…" and does nothing) does a real download.

**Why BP_Backend and not the gateway:** the gateway runs on Lambda, where bundling a PDF renderer is awkward. BP_Backend is a long-running server and is the primary repo.

**Dependencies.** New in BP_Backend: `weasyprint` (plus its cairo/pango system libraries — this is the one install that can bite on a fresh box) and `boto3`'s `generate_presigned_url`, which BP_Backend does not currently call anywhere.

`openpyxl` is a trap: it is already **imported and used** by the extraction parsers (`xlsx_parser.py`, `spreadsheet_backend.py`) but is **not declared in `requirements.txt`** — it arrives transitively today. Declare it explicitly as part of this work rather than leaning on a transitive dependency that could vanish under us.

**Two known layout gaps to fix while doing this:** the `stages` and `donut` KPI display kinds currently render nothing at all in the print path, so they silently vanish from any exported report.

---

## 4. Verification

Per the project's standing requirement, each stage is proved on the **running local stack against live `bp_sqldb`**, not only by tests:

- **Stage 1:** save two reports with several versions each; confirm the index groups them correctly, opens the right snapshot, and that delete works. Confirm Generate from the home modal carries the name/type/period through.
- **Stage 2:** confirm the builder renders with **zero hardcoded values left in `engine.js`** — every figure on screen must have arrived over `GET /spendiq/report-data`. Prove it by changing a value in the demo provider server-side and seeing the UI change without touching the frontend. Confirm changing the period re-fetches and re-renders. Confirm the `Demo data` badge appears on screen and in the exported PDF.
  Then flip `REPORTS_DATA_SOURCE=live` on the local stack and confirm: the ready tiles render real figures that match a direct SQL query against `bp_sqldb`; the five not-ready tiles show an explicit "no data source" state rather than silently falling back to demo numbers; and the badge disappears. Flip it back to `demo`. **This flip is the whole point of the stage — if it isn't exercised, the stage isn't done.**
- **Stage 3:** export a report to PDF, download it from the returned URL, open it, and confirm every block present on screen is present in the file (including `stages`/`donut` KPIs). Confirm `report_url` is populated.

Local stack notes: start BP_Backend **with `.env`** (extraction path depends on it) and the gateway with `node --experimental-global-webcrypto`. Never `pkill -f uvicorn`.

---

## 5. Out of scope

- Building a spend **category taxonomy**, contract linkage, or multi-dimension **supplier risk** scoring. All are data-capture work. Until they exist, the five tiles that depend on them keep serving demo values (badged) and their `live` implementations stay unwritten.
- Replacing `agentCompose` — the builder's "✨ Generate" narrative button — with a real LLM call. It is currently keyword matching over hardcoded numbers, not AI. Once Stage 2 lands it would at least be keyword matching over *real* numbers. Pointing it at the AgentNick control plane is a natural follow-on, deliberately not bundled here.
- Scheduled / emailed reports.
- Sharing and permissions on individual reports (the `/spendiq` route is gated by `routeAccess['dashboard']`; reports have no permission of their own).

---

## 6. Risks

- **The main risk of keeping demo data is that the demo path becomes the only path that ever gets exercised.** The `live` provider would rot quietly and we'd discover it on a customer's first day. Two mitigations, both cheap, both mandatory: the live flip is part of Stage 2's definition of done (above), and the demo provider must not be a special case in the UI — same endpoint, same contract, same renderers, one config value apart.

- **When the flip does happen, the numbers will look worse, and that is correct.** The extracted corpus is thin (7 deals, 24 opportunities), so real charts will be sparse where demo charts are smooth, and Savings Secured drops from a fictional £525,700 to a true £0. A sparse honest series beats a smooth invented one. Expect it; don't treat it as a regression.

- **Do not feed report tiles from the wrong sources.** Three surfaces in the gateway look like analytics but are not usable here, and wiring a tile to one of them would reintroduce exactly the fabrication this stage removes:
  - `GET /dashboard` and `GET /invoices/getAllInvoiceData` read the **seeded `uicanvas` demo tables**, whose "savings" is a flat 6% of spend and whose currency mix is fake.
  - `GET /compliance/compliance-trends` and `/compliance/flagged-cases-by-month` are **hardcoded literal series** in the source (`compliance.service.ts:589-660`) — the `timeRange` parameter only filters invented rows.
  - `proc.bp_detection_finding` (behind the `Detection` module) is **empty**; discrepancy tiles must read `bp_extraction_discrepancy`.

- **`/spendiq/spend-series` anchors its window on `MAX(invoice_date)` in the corpus, not on `CURRENT_DATE`** (`spendiq.service.ts:1058-1060`). So "last 30 days" means 30 days back from the newest invoice, not from today. Whatever the new `report-data` endpoint does about date windows, it must be deliberate and consistent about this, or "This month" will quietly mean different things in different tiles.
- The `bp_reports` table has **no `CREATE TABLE` in version control** — it exists in the database only. The migration must therefore be defensive (`ADD COLUMN IF NOT EXISTS`) and must not assume the legacy rows' shape.
- Legacy `bp_reports` rows with `definition IS NULL` predate this feature and are filtered out of every query. The backfill must leave them alone.
