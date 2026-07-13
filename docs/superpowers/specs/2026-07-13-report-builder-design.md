# Report Builder: reports index, live data, real export

**Date:** 2026-07-13
**Status:** Draft — awaiting review
**Repos touched:** `beyond_procwise_ui` (spendiq-ui), `beyond-procwaise-Api` (Node gateway), `BP_Backend` (FastAPI)

---

## 1. The problem

Click **Reports** on the home page, fill in the report name / type / period, click **Generate**, and you land in the report builder on a blank "Untitled report". There is no way to reach a report you saved earlier. The three fields you just filled in are silently thrown away.

Underneath that, two further problems make the feature dishonest rather than merely awkward:

- Every number in the builder is **fabricated**. The KPIs and charts are hardcoded constants frozen at Jan–Mar 2025. A "report" generated today reflects nothing in the database.
- **Export PDF** is `window.print()`. It prints the screen. No file is produced, nothing is stored, and nothing can be shared.

This spec covers all three: the missing front door, the fake data, and the fake export.

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

### Stage 2 — Live data

Replace the two lookups (`METRICS[period][metric]`, `GRAPHS[key]`) with real figures from the database.

**Approach.** One new gateway endpoint returns everything a report needs in a single round-trip, rather than the builder firing twenty calls:

`GET /spendiq/report-data?from=<ISO date>&to=<ISO date>` →
`{ period: {from, to}, kpis: { <key>: {big, delta, tone, bullets[], sparkline[]} }, series: { <key>: {labels[], data[]|bars[]+line[]} } }`

The builder fetches this once per period change and renders from it. The shape deliberately mirrors the existing `METRICS`/`GRAPHS` objects so the renderers barely change.

**Period becomes real.** Today the period selector only picks which hardcoded slice to read, and the existing `/spendiq/metrics` and `/spendiq/trends` endpoints accept **no date filter at all**. Both need `from`/`to` plumbed through their SQL. The selector's fixed Mar/Feb/Jan 2025 options are replaced by real ranges (This month / This quarter / Year to date / Custom).

**Honest tile-by-tile mapping.** This is the part that determines what "live data" actually means. Verified against the gateway's SQL and the `bp_*_trgt` schema:

*KPIs (8):*

| Tile | Source | Status |
|---|---|---|
| Savings Secured | `bp_opportunity.realised_savings_gbp` | Live — **but see below: currently £0** |
| In-Flight Negotiations | `bp_opportunity` (open stages) | Live |
| Opportunity Pipeline | `bp_opportunity.financial_impact_gbp` | Live |
| Cycle Time to PO | `bp_deal_overview.cycle_days_quote_to_po` | Live |
| 3-Way Match | `bp_deal_overview.three_way_match` | Live |
| Non-PO Spend | `bp_invoice_trgt` vs `bp_purchase_order_trgt` (PO-backed share) | Live |
| Duplicate Risk | `bp_extraction_discrepancy` (duplicate findings) | Live |
| **Tail Spend Visibility** | **needs a spend category / contract taxonomy — none exists** | **Blocked** |

*Charts (12):*

| Tile | Source | Status |
|---|---|---|
| Committed spend & savings | `trends.spendByMonth` + `trends.savings` | Live |
| Realised savings trend | `trends.savings` | Live |
| Committed spend by month | `trends.spendByMonth` | Live |
| Quote volume | `trends.quoteVolume` | Live |
| 3-way match trend | `trends.matchRate` | Live |
| Cycle time to PO | `trends.cycleTime` | Live |
| Off-contract spend | `trends.offContract` | Live |
| Top suppliers by spend | `metrics.topSuppliers` | Live |
| Compliance rate | no distinct compliance series; would duplicate 3-way match | **Cut** |
| Tail spend breakdown | only maverick findings exist, not a full 4-way composition | **Cut** |
| **Spend by category** | **no category column exists in any extracted table** (`spendiq.service.ts:566`) | **Blocked** |
| **Supplier risk profile** (radar) | only a single `risk_score`; the radar needs 6 dimensions | **Blocked** |

**How blocked tiles are handled.** They are **removed from the palette and Graph Library** — not left in place showing invented numbers. Any *already-saved* report that references one renders an explicit "No data source" placeholder rather than a fabricated figure. This follows the project's standing rule: if the data isn't there, show nothing, never fabricate.

Net: **6 of 8 KPIs and 8 of 12 charts go live.** Making the remaining four real requires capturing data the platform does not currently hold (a spend category taxonomy; multi-dimension supplier risk). That is a data-capture project, not a UI one, and it is out of scope here.

**Expect the headline number to fall off a cliff.** "Savings Secured" currently shows a fabricated **£525,700**. The real figure is **£0** — all 24 rows in `bp_opportunity` sit at stage `identified`, and nothing has ever been marked realised (the gateway already measures this and returns zero). The *pipeline* figure is real and non-zero; realised savings is not. Going live means the flagship KPI on the Executive template reads £0 until opportunities are actually progressed through their stages.

This is the correct behaviour and the whole point of the exercise, but it is a visible, board-facing change and should not be a surprise on the day it ships. If the Executive template needs a credible headline before then, the honest candidates are Opportunity Pipeline (real) or Non-PO Spend (real).

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
- **Stage 2:** for each live tile, cross-check the rendered figure against a direct SQL query on `bp_sqldb`. A tile is only "live" if the number matches. Confirm blocked tiles are absent from the palette and that a saved report referencing one shows the placeholder, not a number.
- **Stage 3:** export a report to PDF, download it from the returned URL, open it, and confirm every block present on screen is present in the file (including `stages`/`donut` KPIs). Confirm `report_url` is populated.

Local stack notes: start BP_Backend **with `.env`** (extraction path depends on it) and the gateway with `node --experimental-global-webcrypto`. Never `pkill -f uvicorn`.

---

## 5. Out of scope

- Building a spend **category taxonomy**, or multi-dimension **supplier risk** scoring. Both are data-capture work; until they exist, the four blocked tiles stay out.
- Replacing `agentCompose` — the builder's "✨ Generate" narrative button — with a real LLM call. It is currently keyword matching over hardcoded numbers, not AI. Once Stage 2 lands it would at least be keyword matching over *real* numbers. Pointing it at the AgentNick control plane is a natural follow-on, deliberately not bundled here.
- Scheduled / emailed reports.
- Sharing and permissions on individual reports (the `/spendiq` route is gated by `routeAccess['dashboard']`; reports have no permission of their own).

---

## 6. Risks

- **Stage 2 is the risky one.** The figures behind the live tiles are only as good as the extracted corpus. Some series are thin (7 deals, 24 opportunities), so charts will look sparse where the demo data looked smooth. A sparse true series beats a smooth invented one, but it will look worse, and that should be expected rather than treated as a bug. And, as above, Savings Secured drops from a fictional £525,700 to a true £0.

- **Do not feed report tiles from the wrong sources.** Three surfaces in the gateway look like analytics but are not usable here, and wiring a tile to one of them would reintroduce exactly the fabrication this stage removes:
  - `GET /dashboard` and `GET /invoices/getAllInvoiceData` read the **seeded `uicanvas` demo tables**, whose "savings" is a flat 6% of spend and whose currency mix is fake.
  - `GET /compliance/compliance-trends` and `/compliance/flagged-cases-by-month` are **hardcoded literal series** in the source (`compliance.service.ts:589-660`) — the `timeRange` parameter only filters invented rows.
  - `proc.bp_detection_finding` (behind the `Detection` module) is **empty**; discrepancy tiles must read `bp_extraction_discrepancy`.

- **`/spendiq/spend-series` anchors its window on `MAX(invoice_date)` in the corpus, not on `CURRENT_DATE`** (`spendiq.service.ts:1058-1060`). So "last 30 days" means 30 days back from the newest invoice, not from today. Whatever the new `report-data` endpoint does about date windows, it must be deliberate and consistent about this, or "This month" will quietly mean different things in different tiles.
- The `bp_reports` table has **no `CREATE TABLE` in version control** — it exists in the database only. The migration must therefore be defensive (`ADD COLUMN IF NOT EXISTS`) and must not assume the legacy rows' shape.
- Legacy `bp_reports` rows with `definition IS NULL` predate this feature and are filtered out of every query. The backfill must leave them alone.
