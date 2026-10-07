# Report Builder: reports index, metric registry, live data, presentation mode, real export

**Date:** 2026-07-13 (revised 2026-10-07)
**Status:** Draft — awaiting review
**Repos touched:** `beyond_procwise_ui` (spendiq-ui), `beyond-procwaise-Api` (Node gateway), `BP_Backend` (FastAPI)

---

## 1. The problem

Click **Reports** on the home page, fill in the report name / type / period, click **Generate**, and you land in the report builder on a blank "Untitled report". There is no way to reach a report you saved earlier. The three fields you just filled in are silently thrown away.

Underneath that, three further problems mean the feature has no flexible, trustworthy data path:

- Every number in the builder comes from a **closed, hardcoded catalog** (8 KPIs, 12 charts) living in the browser. Nothing server-side defines what a tile *is*, so adding a metric means editing the frontend, and there is no way to say which tiles a customer's data can actually support.
- **Export PDF** is `window.print()`. It prints the screen. No file is produced, nothing is stored, and nothing can be shared.
- There is no safe way to **show the product with dummy data**. Demo numbers and real numbers are indistinguishable in the UI, which is how a customer ends up mistaking one for the other.

This spec covers the missing front door, a **metric registry** that makes reports flexible without making the builder an arbitrary query tool, a governed **presentation data mode** for demos, and a real export.

**Design principles (apply to every stage):**

1. **A tile is a query spec over an allowlisted registry, not a free query.** The allowlist is a **safety boundary, not a size limit**: it exists so no SQL is ever built from user input, and it grows by reviewed entry as far as the data supports. Flexibility comes from combining registered metrics, dimensions and filters freely, never from user-supplied SQL.
2. **Drag and drop is the layout layer only.** It decides order, size and section. It never defines what a tile can contain.
3. **Real data and dummy data never mix, and dummy data is never silent.** Every surface that can show a presentation number marks it, server-side, in a way the user cannot remove.
4. **No LLM ever produces or alters a number.** (Binding on every future agent in §5.)
5. **A report shows what its reader is entitled to read, and nothing else.** A **Buyer sees only their own deals; an Admin sees all; a Viewer is scoped exactly like a Buyer.** Live data is read under the caller's own rights and scope (Stage 2b, "Access rights and data scope").
6. **A page, a print and an export are one report.** All three are rendered from the same payload and carry the same parameters; none may silently drop a tile (Stage 3, "Fidelity").

---

## 1a. As built (2026-10-07) — where this differs from the text below

The backend of Stages 2 and 3 is built and tested. Three things differ from how this document first described them; the design rules are unchanged.

- **Host.** The registry, providers, scope, presentation mode and checks live in **BP_Backend** (`src/services/report_data/`), not the gateway. Roles, the permission gate, audit logging and the export service are already there, so rights, scope and presentation checks have one enforcement point and an export is computed from the same code as the page. Wherever the text below says `GET /spendiq/report-data`, read **`POST /reports/data`**. Also built: `GET/POST /reports/presentation-mode`, `GET/POST/DELETE /reports/scope` (Admin only), `POST /reports/export` (accepts the report-data request so the file carries every tile and its parameters).
- **Gate.** A data request and an export are gated as `report.read` (the file returns only to the person composing it, so anyone may export what they can see). Scope management is a new `report.scope.write` action (class `configure`). `report.export` (a share) is unchanged.
- **Export delivery.** `POST /reports/export` returns the file directly; a copy is stored in S3 and its key written to `bp_reports.report_url` (never for a presentation export). The gateway presigned-URL step in Stage 3 is not used.
- **UI, built (2026-10-07):** the builder has a **Data tile** block (`reportData.js`, drawn as inline SVG so print and PDF match). A tile is a spec over the registry: metric, up to two groupings, show-as (single value, line, bars, table, top findings), comparison, top-N; the report has one **data period**. A report is **bound to one data mode** (`STATE.dataMode`, saved in the definition and in `bp_reports.data_mode` by the gateway): the Admin-only **Presentation data** toggle decides who may open or start a presentation report and never converts a saved report. The banner, per-tile badge and a print watermark show whenever presentation data is in play; Export sends the data request so the server builds the file from the same tiles under the exporter's rights. The reports index, version list and "open one" pass `?presentation=1` only while the toggle is on; the gateway hides presentation reports from everyone but an Admin with it on.
- **Also built:** a **filter** control on each tile (choose a dimension, search, pick values; the choices come from `POST /reports/values`, scoped like a tile, synthetic in presentation mode) and a **Report access** panel for an Admin (`GET/POST/DELETE /reports/scope`, `GET /reports/scope/buyers`): pick a person, see and change the buyers they can see.
- **Not built yet:** removal of the legacy hardcoded KPI/graph blocks and the dead RB6 code, the `stages`/`donut` print gap in those legacy blocks, `POST /spendiq/reports/:id/run` (§6) and the agents (§5) (design only), and any check in a real browser: this has been verified by unit tests, a rendered PDF of real tiles, and a clean checkout of each commit, not by clicking through the screen.
- **Zero vs gap.** An empty month is a zero for a count or sum and a **gap** for a rate or average (a 0 % would assert something the data did not say). Presentation mode refuses any breakdown the synthetic data has no attribute for (it never invents one).
- **Applied to `bp_testdb` only** (the database `.env` points at). `deploy/sql/2026-10-07_report_data.sql` has not been applied to `bp_sqldb`; until it is, the gateway serves and saves live reports as before and refuses to save a presentation one.

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

Consequence: **swapping the data source requires no migration of saved reports.** They already resolve against a data source at render time. The query-spec model in Stage 2 extends that same idea (see §3, Stage 2b for the one-time reading of old definitions).

---

## 3. Design

Four stages. Each lands independently, is verifiable on the live server, and is useful on its own. Later stages do not require earlier ones to be perfect.

### Stage 1 — The reports index (the front door)

**Behaviour.** `/spendiq?view=reportbuilder` no longer opens a blank canvas. It opens **My reports**: one row per report, showing its version count and when it was last touched, with a primary **Generate** button. Click a report → the composer opens with that report restored. Click Generate → the composer opens on a new report. The composer gains a **← All reports** link back to the index.

Deep-link from home still works: **Generate** on the home modal goes straight to the composer (skipping the index) and now **carries the name, type and period** you typed, so the report opens pre-titled and pre-scoped instead of as "Untitled report".

The home modal's **Recent reports** table is wired to the same endpoint and stops being permanently empty.

*Revised:* the index and the Recent reports table **list live reports only** unless the admin presentation toggle is on (Stage 2c, §2c.e). A presentation report is never listed for anyone else.

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

### Stage 2 — A metric registry, query specs, and one data contract

Stage 2 is split so each part can be reviewed and verified alone: **2a** the registry, **2b** the tile spec and the report-data endpoint, **2c** presentation data mode, **2d** the checks layer.

#### Stage 2a — The metric registry

**The rule.** *All measures, dimensions and filters are allowlisted in the registry. SQL is never built from user input.* A request names registry keys; the server maps each key to a fixed, reviewed SQL fragment. Anything not in the registry is rejected, not interpreted.

**Where it lives.** Server-side, in the gateway, as reviewed code/config. It is **not** user-editable and **not** agent-editable (§5). Adding or changing a metric is a normal reviewed change.

**Entry shape.**

| Field | Meaning |
|---|---|
| `key` | stable identifier saved in definitions, e.g. `committed_spend` |
| `label` | display name |
| `measure` | the SQL expression (fixed, reviewed) |
| `dimensions` | allowlist of dimensions this metric may be grouped by |
| `filters` | allowlist of filters this metric accepts |
| `default_comparison` | `prior_period` \| `prior_year` \| `none` |
| `format` | `currency` \| `percent` \| `count` \| `days` |
| `target` | optional target value or reference |
| `requires` | the read action(s) the caller must hold for the source data (e.g. `invoice.read`, `deal.read`, `finding.read`) — see "Access rights and data scope" |
| `scope` | how a Buyer's rows are selected for this metric: the column or reviewed join to `buyer_id`/deal ownership, **or `unscopable`** (then non-admins get `forbidden`, never unscoped numbers) |
| `availability` | `live` \| `presentation-only` \| `unavailable` |

**Availability semantics.**

- `live` — the live provider can compute it from data that exists today. Shown in the picker in live mode.
- `presentation-only` — has presentation values (§2c.d) but no live source. **Hidden from the live picker**; selectable only in presentation mode. Never shown as a dead tile in live mode.
- `unavailable` — needs data the platform does not capture and has no presentation values either. Hidden everywhere; a spec naming it is rejected.

**Seed set (11 metrics, live-ready today).** Derived from the live-readiness tables in this document; no metric is added that those tables do not support.

| Registry key | Source (from the live-readiness verification) |
|---|---|
| `savings_secured` | `bp_opportunity.realised_savings_gbp` (returns £0 today — see Risks) |
| `in_flight_negotiations` | `bp_opportunity` (open stages) |
| `opportunity_pipeline` | `bp_opportunity.financial_impact_gbp` |
| `cycle_time_to_po` | `bp_deal_overview.cycle_days_quote_to_po` |
| `value_reconciled_rate` | `bp_deal_overview.value_reconciled` — do quote, PO and invoice **amounts** agree |
| `three_way_match_rate` | `bp_deal_overview.three_way_matched` — did what was billed **arrive**. **Tri-state: NULL = not assessed** (no goods receipt on the deal). The rate is taken over assessed deals only and the tile states how many were assessed. **Verified 2026-10-07: NULL on all 5,042 deals in the seeded corpus**, so today it renders "not assessed", never 0 % |
| `non_po_spend` | `bp_invoice_trgt` vs `bp_purchase_order_trgt` (PO-backed share) |
| `duplicate_risk` | `bp_extraction_discrepancy` (duplicate findings) |
| `committed_spend` | `trends.spendByMonth`; grouped by supplier it is the "top suppliers" chart |
| `quote_volume` | `trends.quoteVolume` |
| `off_contract_spend` | `trends.offContract` |

The seed is a **starting point, not a ceiling** — see "Admission criteria" below. (The earlier single `three_way_match` column was split into the two rows above; the original design predates that change.) The 12 charts collapse onto these: the realised-savings, 3-way-match and cycle-time **trend** charts are the same three metrics grouped by month; "Top suppliers by spend" is `committed_spend` grouped by supplier; "Committed spend & savings" is two metrics on one line viz. This is the point of the registry — one definition per measure, many ways to show it.

**Not-ready tiles.** Registered, never silently dropped:

| Tile | Registry availability | Why |
|---|---|---|
| Tail Spend Visibility (KPI) | `presentation-only` | needs a spend-category / contract taxonomy — data does not exist |
| Compliance rate | `presentation-only` | no distinct compliance series; would duplicate 3-way match; needs contract linkage |
| Tail spend breakdown | `presentation-only` | only maverick findings exist, not a 4-way composition |
| Spend by category | `presentation-only` | no category column in **any** extracted table (`spendiq.service.ts:566`) |
| Supplier risk profile (radar) | `presentation-only` | only a single `risk_score`; the radar needs 6 dimensions |

Where a not-ready tile has **no** defensible presentation values either, it is registered `unavailable` instead and a spec naming it is rejected.

**Dimensions.** A dimension is registered when it maps to a column (or a reviewed join) that exists **and is populated**. Verified against the schema on 2026-10-07 (seeded corpus `bp_testdb`; **re-verify on the target database before registering**):

| Dimension | Maps to | State |
|---|---|---|
| time (month / quarter / year) | `invoice_date`, `order_date`, `quote_date`; deals: `COALESCE(deal_date, first_activity_date)` (`deal_date` is NULL on every deal row) | register |
| supplier | `supplier_id` / `supplier_name` (deals, invoices, POs, quotes, opportunities) | register |
| buyer | `buyer_id` (deals, invoices, POs, quotes) | register |
| currency | `currency` (5 distinct on deals) | register |
| country, region | `country`, `region` on invoices and quotes; `ship_to_country`, `delivery_region` on POs | register (8 regions / 6 countries on invoices) |
| payment terms | `payment_terms` on invoices and POs | register (9 distinct) |
| document status | `po_status`, `approval_status`, quote `status`, `award_status` | register where populated (`invoice_status` has **0** distinct values today → hold) |
| deal | `deal_id` / `deal_name` | register |
| opportunity stage, detector type | `bp_opportunity.stage`, `.detector_type` | register |
| finding type, severity, status | `bp_extraction_discrepancy.issue_type`, `.severity`, `.status` | register |
| item / UoM | `item_description`, `uom_normalised` on opportunities and invoice lines | register for the sources that carry them |
| **category** | `bp_opportunity.category_id` **exists but is empty (0 of 308 rows)** | **hold — column exists, data does not** |
| **contract** | contract tables exist (`bp_contract_master`, `bp_contracts`, …) but **spend is not linked to a contract** | **hold — linkage does not exist** (re-verify; contract extraction has moved on since this design was first written) |

*The earlier version of this document listed only five dimensions. That was caution about not asserting columns it had not checked, not a design limit; the table above replaces it, and every row is checkable against the schema.*

**Beyond one metric per tile.** All of these stay inside the allowlist because they combine registered pieces and add no new SQL surface:

- **Multi-series tiles** — several registered metrics on one chart (e.g. spend and savings by month).
- **Derived metrics** — a spec may define a ratio, difference, sum or share-of-total **over registered metrics** (e.g. savings ÷ spend). Evaluated by the server from the same registered measures; never from typed expressions.
- **Top-N / sort / limit** on any grouped result, with an explicit "other" bucket so the total still ties (Stage 2d).
- **Several group-by dimensions** where the metric permits both (e.g. supplier × month for a table).
- **Any number of tiles, including the same metric more than once** with different filters, periods or groupings.

**Admission criteria (replaces the earlier size cap).** There is no fixed limit on metrics or dimensions. An entry is admitted by review when it (1) maps to real, populated columns, (2) declares `requires`, (3) has a test showing its result equals a direct SQL query, and (4) states how it behaves when its data is absent. The registry is code reviewed like any other change; agents never edit it (§5).

#### Stage 2b — The query spec, and `GET /spendiq/report-data`

**A tile is a spec.** The fixed `{type:'kpi', metric, period}` / `{type:'graph', key, range}` tile model is replaced by:

```
Tile = {
  metric:     <registry key>,
  groupBy:    <registry dimension> | null,
  filters:    { <registry filter>: <typed value>, ... },
  period:     { from, to } | <named range>,
  comparison: 'prior_period' | 'prior_year' | 'none' | null,   // null → metric default
  target:     <number> | null,                                  // null → metric default
  viz:        'kpi' | 'line' | 'bar' | 'table' | 'findings'
}
```

`viz` is a presentation choice over the same spec. **`table` is new** (rows of the grouped result, with totals); **`findings`** is a text tile (§5) whose content is produced from numbers the server computed, never typed or invented.

**Saved definitions store the spec plus layout**, not the spec alone and not values:

```
definition = {
  tiles:   [ { id, spec, layout: { section, order, size } } ],
  sections:[ ... ],
  template: 'executive' | 'compliance' | 'category' | 'savings' | null,
  cadence: null            // nullable; unused until scheduling exists
}
```

`cadence` is added to the definition **now** so scheduling later needs no migration. Nothing reads it today.

**Reading old definitions.** Existing saved definitions use the old reference shape. The server maps each old reference onto a registry spec at read time (e.g. `savings` → `savings_secured`, `spendSave` → two `committed_spend`/`savings_secured` specs on a line viz). An old reference with no registry equivalent is rendered as an explicit "no data source" tile; it is never dropped and never given invented values. No stored row is rewritten by the read.

**Endpoint.** One round-trip for the whole report:

`GET /spendiq/report-data` — request: a list of tile specs plus `from` / `to`; response: one result per tile.

```
Request:  { from, to, tiles: [ <Tile spec>, ... ], data_mode?: 'live' | 'presentation' }
Response: { period: { from, to },
            tiles: [ { id, data_mode: 'live' | 'presentation',
                       status: 'ok' | 'no_data_source' | 'rejected',
                       result: { big, delta, tone, bullets[], sparkline[] }   // kpi
                             | { labels[], data[] | bars[]+line[] }           // line / bar
                             | { columns[], rows[], total }                   // table
                             | { items[] },                                   // findings
                       checks: [ ... ],        // Stage 2d
                       marker?: 'PRESENTATION DATA - NOT REAL' } ] }
```

(Sent as `POST` if the spec list exceeds URL limits; the contract is the same.)

**`data_mode` is returned PER TILE, not per payload.** The UI and every export read it from the payload. **No frontend constant decides whether data is live or presentation.** This is the mechanism that makes a screenshot of a single tile self-describing.

**Two providers, one contract.** Behind the endpoint:

- **`live`** — maps each spec to registry SQL (parameter-bound) against `bp_sqldb` and honours `from`/`to`.
- **`presentation`** — serves synthetic values (§2c.d) through the **identical** contract, honouring the same `from`/`to` by slicing its series so period switching exercises the same code path.

Selection is by `REPORTS_DATA_SOURCE` (default **`live`**) together with the admin/session rules in §2c.a. The provider never falls back from one to the other.

**Live mode and unready tiles — one rule, everywhere.** In live mode a tile whose metric is `presentation-only`/`unavailable`, or whose live query is not implemented, returns `status: 'no_data_source'` and renders an explicit **"no data source"** state. It **never** returns presentation values. Because such tiles are hidden from the live picker (2a), this state is reachable only from an old saved definition. *This replaces the earlier text under which unready tiles "keep serving demo values", which contradicted Verification.*

**Access rights and data scope.** Anyone can produce a report, and what it shows is bounded by what **they** may read.

*Rule (decided 2026-10-07):* **a Buyer sees only their own deals; an Admin sees all; a Viewer is scoped like a Buyer** (sees only the deals assigned to them, never more than a Buyer would).

- **Two checks, both server-side, both per tile.** (1) *Action right*: the caller holds the entry's `requires` (e.g. `invoice.read`). (2) *Row scope*: the rows are limited to the caller's scope — Admin: none (all rows); Buyer and Viewer: their assigned deals only.
- **Scope is a mandatory predicate the server adds, not a filter the user chooses.** It is applied inside every live query **before** any aggregation, so totals, groups, "other" buckets, comparisons, sparklines and targets are all computed over the caller's rows only. A user-supplied filter on the same dimension (e.g. a buyer filter) **intersects** with scope and can never widen it. Nothing computed over all rows is ever shown and then "trimmed".
- **Fail closed.** A Buyer or Viewer with no scope assigned sees **nothing** (empty, with an explicit "no deals assigned to you" state), never everything. A metric whose `scope` is `unscopable` returns `forbidden` to a non-admin until a verified scoping join exists. *(No registered metric is `unscopable` today: findings are scopable, below. The state remains for any future metric whose scoping join cannot be verified.)*
- **A tile the caller may not read** returns `status: 'forbidden'` and renders "you don't have access to this data" — never zero, blank or approximate, and **not omitted**, so a reader can tell a gap from a zero. A tile that is *in scope but empty* shows its true zero.
- **Saved reports are definitions, not value snapshots.** Opening, running or exporting one re-evaluates every tile under the **opener's** rights and scope. Two people opening the same report see different numbers, each correct for them. Results are never cached across scopes (the cache key includes the scope).
- The registry picker lists only metrics the caller may read.
- **Presentation data** (§2c) is fictional, not corporate data, and is **not** scoped; it stays admin-only.

*What "their own deals" means in the data — and what is missing.* On documents, `buyer_id` is a **customer company code** (e.g. `CC000109`; 502 distinct in the seeded corpus; present on deals, invoices, POs, quotes, deal-documents), **not a person**. **Nothing today links a signed-in user to a deal or to a code.** `bp_deal` has no owner column, and `bp_role_assignment` holds roles only (subject → role). So the rule needs one new piece, and the design proposes it:

- A **user scope table** (`bp_user_buyer_scope`: subject, `buyer_id`, granted_by, granted_at, revoked_at — modelled on `bp_role_assignment`, appended and revoked, never edited) saying which `buyer_id` codes a Buyer may see. "Their deals" = deals whose `buyer_id` is in that set. **Only an Admin grants or revokes scope**, through the existing admin area; every grant and revoke is audited, and a user cannot grant themselves scope.
- Sources scope as follows (verified against the schema 2026-10-07; re-verify on the target DB):
  - **Direct `buyer_id`:** `bp_deal_overview`, `bp_deal_documents`, `bp_invoice_trgt`, `bp_purchase_order_trgt`, `bp_quote_trgt` — scoped on the column.
  - **Via `deal_id`:** `bp_opportunity` (has `deal_id`) — scoped by joining to `bp_deal_overview.buyer_id`.
  - **Findings (decided: Buyers see their own findings):** `bp_extraction_discrepancy` has no `buyer_id` or `deal_id`, but **every finding names its document** (`doc_pk_candidate`, populated on all rows) and `bp_deal_documents` maps each document (`doc_type`, `doc_pk`) to its `deal_id`. A finding is therefore scoped **document → deal → the deal's `buyer_id`** (taken from `bp_deal_overview`, not from the document row, whose `buyer_id` is empty on many quotes). `doc_type` is normalised across the two tables (`po` / `purchase_order`). Verified on the seeded corpus 2026-10-07 (**re-verify on the target DB**): no finding maps to more than one deal; coverage is **invoices 4,475 / 4,515 (99 %), purchase orders 472 / 478 (99 %), quotes 257 / 379 (68 %), contracts 0 / 4, `po`-typed 0 / 2**. Quote coverage is weak and maps to very few deals, so it must be understood before quote-finding metrics are relied on.
  - **Findings that cannot be attributed to a deal** (no matching document) are **visible to Admin only** and never to a Buyer or Viewer; they are not counted in a Buyer's totals. They are reported to Admin as a **data-quality item** ("N findings not attributable to a deal"), so the gap is visible to the person who can fix it. Scoping coverage is a registered check: if a source's attributable share drops below a set threshold, that metric's tile says so rather than presenting a quietly smaller number.
- One deal with a NULL `buyer_id` exists in the corpus; NULL is visible to **Admin only**.

**Rejection.** A spec naming an unregistered metric/dimension/filter, an `unavailable` metric, or a dimension not permitted for that metric is rejected (`status: 'rejected'`, with the reason), never partially executed.

**Period becomes real plumbing.** Today `/spendiq/metrics` and `/spendiq/trends` accept **no date filter at all**. The new endpoint takes a real `from`/`to`; the live provider plumbs it into the SQL. The fixed Mar/Feb/Jan 2025 options are replaced by real ranges (This month / This quarter / Year to date / Custom). **All tiles use one date anchor** (Stage 2d).

**Dead code.** Roughly half the RB6 module is an unreachable earlier generation (`SECTION_DEFS`, the hero/breakdown/cycle/compliance card renderers, the word-budget enforcer and its canned narrative drafts, `HERO`, `SUMMARIES`, `HERO_TREND_VALUES`, `CYCLE_TREND`, `TEMPLATE_DEFAULTS`). It is where much of the fake data lives. It gets deleted as part of this stage.

#### Stage 2c — Presentation data mode (dummy data, presentation only)

Dummy data exists so the product can be **shown**, never so a report can be **trusted**. Everything in this section exists to keep those two apart. (Replaces the earlier "demo provider" and its `Demo data` badge.)

**2c.a Default and activation**

- `REPORTS_DATA_SOURCE` **defaults to `live`**.
- Presentation mode can be enabled **only by users with the admin role**. This is enforced **server-side on every `report-data`, export and headless-run request**, not by hiding the toggle. A non-admin request carrying `data_mode=presentation` is rejected **403**, whatever the client sends or however the request was hand-edited.
- The toggle is **per session**: activating it records a session-scoped activation (user, time, session id) whose lifetime is the session. It **resets to live on sign-out or session expiry**. It is never stored as a user or tenant default.
- A presentation request additionally requires an **active activation for that session**; an admin whose session has no activation is also refused. (Being admin is necessary, not sufficient.)
- Presentation data is **never a fallback** for a failed or unready live tile. A live failure shows an error or "no data source", never dummy values.

**2c.b No mixing**

A report renders **fully live or fully presentation**. A request whose tiles resolve to both modes, or a saved spec that would combine them, is **rejected**. There is no "live with a few demo tiles".

**2c.c Visibility — mandatory, and not removable by the user**

- A **persistent banner** on every screen in this mode: **"PRESENTATION DATA - NOT REAL"**.
- A **per-tile badge**, so a tile is still marked if screenshotted alone (`marker` in the payload, rendered from the payload).
- A **watermark on every PDF page**; on XLSX, a **first-tab note** and a **`-DEMO` filename suffix**; the **same marker on any headless-run output**.
- Markers are produced **server-side from `data_mode`**. They are not CSS the user can hide, and an export request cannot ask for them to be omitted.

**2c.d Content**

- **One fictional organisation** and **fictional suppliers with clearly synthetic names**. **No real customer, supplier or person names.**
- **Internally consistent**: totals tie to their groups, comparisons are plausible, periods line up.
- Covers **all 8 KPIs and 12 charts**, including the five tiles with no live source.
- Lives **server-side** behind the same contract as the live provider; it is not shipped to the browser as a constant.

**2c.e Containment**

- Add `bp_reports.data_mode`, default `'live'`, and **backfill existing rows as `'live'`**. Legacy rows with `definition IS NULL` are not touched.

```sql
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS data_mode text NOT NULL DEFAULT 'live';
```
  (`ADD COLUMN IF NOT EXISTS` — `bp_reports` has no `CREATE TABLE` in version control; see Risks. The backfill is a no-op beyond the default and must leave `definition IS NULL` rows alone.)
- Presentation reports are **visible and openable only to admins**; anyone else with a link gets **not-found** (not "forbidden", so existence is not disclosed). They are **hidden from the normal index unless the toggle is on**, and **cannot be scheduled, shared, or converted to live**.
- **Log each activation** (user, time, session) **and each presentation export.**

#### Stage 2d — Checks layer (runs on every render)

Every tile result passes through checks before it is returned; results are listed in the tile's `checks`.

- **Partial current period** is labelled, or compared like-for-like (same elapsed span of the prior period).
- **Empty groups appear with a zero**, not as a missing row. A top-N result carries an "other" row so the groups still sum to the total.
- **Grouped rows that do not sum to the total are flagged, not published.** The tile returns the flag and withholds the figure rather than showing an inconsistent number.
- **One date anchor for every tile.** `/spendiq/spend-series` today anchors its window on `MAX(invoice_date)` in the corpus, not on `CURRENT_DATE` (`spendiq.service.ts:1058-1060`). **Decision: anchor on `CURRENT_DATE` for every tile**, and for the "latest period" label state the newest period that actually has data. With a thin corpus this makes a "this month" tile honestly empty rather than quietly shifted to a different month than another tile. The anchor is one function in the provider, used by all tiles, so "This month" cannot mean different things in different tiles.

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

**Two known layout gaps to fix while doing this:** the `stages` and `donut` KPI display kinds currently render nothing at all in the print path, so they silently vanish from any exported report. (In presentation mode these must also carry the badge and watermark — see Verification. Fidelity, above, makes any recurrence of this a loud failure rather than a silent one.)

**Fidelity — a print and an export carry the whole report, with its parameters (added).** The page, the print and the exports are rendered from **one payload** (the same per-tile `report-data` result), not re-derived separately.

- **Every tile appears.** Each tile in the payload appears in the PDF and in the XLSX. A tile whose status is `no_data_source`, `forbidden` or `rejected`, or whose checks withheld its figure, appears **in that state, labelled** — never omitted. A visualisation the export cannot draw **fails the export loudly** instead of vanishing (this is how `stages` and `donut` disappeared).
- **Every tile states its parameters.** Metric, group-by, filters, period (`from`–`to`), comparison and target, `data_mode`, and the **as-of date** and the anchor used (Stage 2d). A reader of the printed page can reproduce the number.
- **A report parameters block** (title, period, template, generated-at, generated-by, `data_mode`, registry/definition version) opens the PDF and is the **first sheet** of the XLSX.
- **XLSX** carries one sheet per tile (the underlying table, plus the parameters beneath it) in addition to the parameters sheet — the data, not a picture of the layout.
- **Print and export use identical rendering inputs.** What the browser's print produces and what the server produces differ only in the renderer, not in content or parameters.
- **A saved report's export reflects the opener's rights** (Stage 2b): a tile the exporter may not read exports as "no access", not as the author's numbers.

**Presentation exports (added for Stage 2c).** An export of a presentation report carries the same `data_mode` the payload carried: the **PDF watermark on every page**, the **XLSX first-tab note and `-DEMO` filename suffix** (§2c.c). The marker is applied by the export service from the data it is given, and an export request cannot suppress it. Admin-only enforcement and activation logging (§2c.a, §2c.e) apply to the export endpoint as they do to `report-data`.

---

## 4. Layout vs definition

- **Drag and drop controls layout only**: order, size, sections, and **adding tiles from the registry** (the picker lists registry metrics available to this caller in the current mode). It does **not** define what a tile can contain — that is the spec.
- **The builder is not narrowed by this split.** The user is free to: place any number of tiles; repeat a metric with different filters or periods; choose any visualisation a result can be drawn as (a grouped result can be a line, bar, table or findings; a single value a KPI); size and section tiles freely; and add **non-data blocks** (headings, notes, free text, images) that carry no number at all.
- A tile's content is edited by choosing among registry options (metric(s), derived metric, group-by, filters, period, comparison, top-N, viz). There is no free-form query box.
- A user can **save a tile spec for reuse** across their own reports. A saved tile is just a spec; it gains no capability the registry does not already give.
- **Templates** — Executive, Compliance, Category, Savings — are **saved layouts** (a set of specs plus placement) and map to the home-modal **Type** field, which already carries those four values into `report_type`. Choosing a Type at Generate pre-loads that template; the user can then rearrange or change anything. A template can include `presentation-only` tiles **only** when opened in presentation mode; in live mode those tiles are omitted from the template, not shown dead.

---

## 5. Future agents (design only — nothing here is built)

**Invariant, binding on every agent below: no LLM ever produces or alters a number.** Numbers come from the registry/provider; an LLM may choose among registry options or put words around numbers it is handed.

- **Spec agent.** Natural language → tile spec. Constrained to the registry (it is given the registry's keys and allowed combinations, nothing else). **Returns JSON only, never numbers.** The server validates its output exactly as it validates a human-built spec, so an agent cannot reach anything a user could not.
- **Findings agent.** A **rules-based ranking** orders the biggest deltas and outliers first; the **LLM only words the top three findings from numbers it is given**. It does not select what is notable, compute, round or restate figures beyond what it was handed.
- Both run on the **AgentNick control plane** (the project's single local model).
- **Scheduler/delivery is a plain job runner, not an agent.** It reads `cadence` from the definition and runs the saved report; no model decides what or when.
- **Registry changes are a reviewed human/dev process. Agents never edit the registry.**

---

## 6. Headless run endpoint (design only)

`POST /spendiq/reports/:id/run` — runs a saved definition and returns **summary-only** results (per-tile status, headline value, delta, checks), **no export**, so other products can call a saved report without the interactive flow.

- Resolves the saved spec through the same providers and checks as `report-data`; nothing is computed a second way.
- **Identity (decided): a run is performed as the user who calls it**, with that user's rights and scope, exactly like the interactive path. Another product calling the endpoint passes the **end user's** identity; it does not get a wider, shared one.
- **The exception is a run an automatic schedule triggers** (the `cadence` field, Stage 2b; scheduling itself is out of scope here). A scheduled run has no caller, so it runs **as the schedule's owner (the user who created the schedule)**, with that owner's rights and scope **re-resolved at every run**, never snapshotted: if the owner's scope is reduced or revoked, or the owner is deactivated, the next run returns less or nothing, or fails closed. A schedule can never return more than its owner could see interactively.
- **Delivery to other people** (assumption, to confirm when scheduling is designed): a scheduled report sent to someone other than its owner would disclose the owner's rows to a reader whose scope may be narrower. So each recipient's copy is **evaluated under that recipient's own scope**, not the owner's; a recipient with no scope receives the empty state.
- Returns `data_mode` per tile and carries the **same presentation markers** (§2c.c). A presentation report can be run only by an admin with an active activation; for anyone else it is not-found (§2c.e).
- A report cannot be run in a mode other than the one it is saved in; there is no way to run a live report on presentation data or the reverse.

---

## 7. Verification

Per the project's standing requirement, each stage is proved on the **running local stack against live `bp_sqldb`**, not only by tests.

- **Stage 1:** save two reports with several versions each; confirm the index groups them correctly, opens the right snapshot, and that delete works. Confirm Generate from the home modal carries the name/type/period through. Confirm a presentation report is absent from the index for a non-admin and for an admin with the toggle off.
- **Registry:**
  - Every registered dimension and metric has a test showing its result equals a direct SQL query; a metric whose source data is absent reports that state instead of 0.
  - **A new metric can be added by registry entry alone** — no frontend change — and appears in the live picker.
  - **A spec using an unavailable metric is rejected**, as is an unregistered dimension/filter and a dimension not permitted for that metric. Prove it by hand-editing a request.
  - Confirm no SQL fragment is built from request input (inject SQL-looking text into a filter value and see it bound, not executed).
- **Live flip (part of Stage 2's definition of done):** with `REPORTS_DATA_SOURCE=live`, the **ready tiles match direct SQL** against `bp_sqldb`; **unready tiles show "no data source"**; **no presentation markers appear anywhere**. If this is not exercised, the stage is not done.
- **Toggle off:** **no presentation value is reachable from any endpoint** — `report-data`, export, run, and the index/open endpoints all refuse or return live.
- **Toggle on (admin):** the **banner, per-tile badges and export watermark appear on every surface**, **including `stages`/`donut` KPIs in print**, and a screenshot of one tile alone is still marked.
- **As a non-admin, presentation requests are refused on `report-data`, export and `run`** (403), **including with a hand-edited request**, and an admin with no active session activation is also refused.
- **Presentation mode switches off on sign-out and on session expiry.**
- **Saving, exporting and headless-running a presentation report each carry the marker end to end** (stored `data_mode`, PDF watermark, XLSX note + `-DEMO` filename, run output).
- **No mixing:** a request or saved spec combining live and presentation tiles is rejected; a live tile that fails never returns presentation values.
- **Checks:** a partial current period is labelled or compared like-for-like; an empty group appears as zero; grouped rows that do not sum to the total are flagged and the figure withheld; every tile in one report resolves "this month" to the same dates.
- **Data scope (Buyer / Admin):**
  - A Buyer's every tile — totals, groups, top-N "other", comparisons, sparklines — equals direct SQL restricted to that Buyer's `buyer_id` set. The sum over all Buyers' scopes equals the Admin figure.
  - Two Buyers opening the **same saved report** see different, correct numbers; neither sees the other's suppliers, deals or totals anywhere on the page, in a tooltip, in the XLSX, or in the PDF.
  - A Buyer with **no** scope assigned sees an explicit empty state, not all data. A Buyer hand-editing a request to add or remove a buyer filter, change `from`/`to`, or name a deal outside their scope still gets only their rows.
  - **Findings are scoped:** a Buyer's findings tiles equal direct SQL over findings whose document maps (via `bp_deal_documents`) to one of their deals; the sum over all Buyers' scopes plus the unattributed remainder equals the Admin total. Unattributed findings never appear for a Buyer or Viewer and are reported to Admin as a count.
  - **Viewer** behaves exactly like a Buyer: same scope table, same fail-closed empty state, never more rows than a Buyer with the same grants.
  - A grant can be made only by an Admin; a Buyer cannot grant or widen their own scope (refused, audited).
  - Revoking a grant removes access on the next request.
- **Access rights:** as a user without `invoice.read`, a spend tile returns `forbidden` (not zero, not omitted); opening and exporting a report saved by a more privileged user shows "no access" tiles; the picker lists only metrics that user can read.
- **Fidelity:** for a report containing every viz kind, a forbidden tile, a no-data-source tile and a withheld (non-summing) tile, assert that **each tile id in the payload appears in the PDF and in the XLSX**, in its true state, with the same metric, group-by, filters, period, comparison, `data_mode` and as-of date as the payload; and that the parameters block and parameters sheet are present. An undrawable viz must fail the export, not disappear. Print and server export are compared tile for tile.
- **Stage 3:** export a report to PDF, download it from the returned URL, open it, and confirm every block present on screen is present in the file (including `stages`/`donut` KPIs). Confirm `report_url` is populated. Confirm an XLSX export carries one sheet per table.

Local stack notes: start BP_Backend **with `.env`** (extraction path depends on it) and the gateway with `node --experimental-global-webcrypto`. Never `pkill -f uvicorn`.

---

## 8. Out of scope

- **Building the agents** in §5 (design only here).
- **Scheduling / delivery** (the `cadence` field is added now; nothing reads it).
- **Arbitrary SQL** — free-form queries or user-defined measures outside the registry. (Combining registered metrics, many dimensions, top-N and derived ratios *is* in scope; see 2a.)
- **New data capture:** a spend **category taxonomy**, **contract linkage**, and multi-dimension **supplier risk** scoring. All are data-capture work. Until they exist, the tiles that depend on them stay `presentation-only`/`unavailable` and their live implementations stay unwritten.
- **Per-report sharing.** Who may *open* a particular saved report is not designed here (a saved report is re-evaluated under the opener's rights and scope, so opening one never exposes data the opener may not read). Row-level scoping of the *rest* of the platform (outside report-data, export and run) is also out of scope. The `/spendiq` route is gated by `routeAccess['dashboard']`; reports have no permission of their own. Note the narrow exception in §2c.e: presentation reports are admin-only, and cannot be shared.
- Replacing `agentCompose` — the builder's "✨ Generate" narrative button — with a real LLM call (it is keyword matching over numbers today). Pointing it at the AgentNick control plane is covered, in constrained form, by the findings agent in §5, not built here.

---

## 9. Risks

- **Presentation data leaks into a real report or export.** The most damaging failure this design has: a customer mistakes dummy numbers for their own. Mitigated by **§2c.c** (marker produced server-side per tile, not removable, on every output) and **§2c.e** (stored `data_mode`, admin-only visibility, no sharing/scheduling/conversion, logged activation and export), plus the no-mixing rule (§2c.b) and the rule that presentation is never a fallback (§2c.a).
- **The live path rots if only the presentation path is exercised.** Keep the **live-flip verification** as part of Stage 2's definition of done; the two providers share one contract, one endpoint and the same renderers, so they are one config value apart.
- **The registry becomes a way around access rights or a source of wrong numbers.** With no size cap, safety rests on the admission criteria (2a): real populated columns, a declared `requires`, a parity test against direct SQL, and review. Every entry also widens what a permitted user can slice, so `requires` is checked per tile and never inferred. Agents never edit the registry (§5).
- **Scope enforced only in the report layer.** The databases have no row-level security, so Buyer scoping lives in the registry's scope predicate for report-data, export and run. Any *other* endpoint a Buyer can reach still sees what it sees today; this design does not fix that, and a report must not be assumed to be the only door. A new metric added without a correct `scope` is a data leak, which is why `scope` is a mandatory registry field, reviewed, and tested (a Buyer's totals must equal direct SQL restricted to their deals).
- **The scope mapping is new data that someone must maintain.** A Buyer with no mapping sees nothing; a wrong mapping shows the wrong deals. Grants are audited and revocable, and a mapping change takes effect on the next request (no cached scope).
- **When the live flip happens, the numbers will look worse, and that is correct.** The extracted corpus is thin (7 deals, 24 opportunities in the original audit), so real charts will be sparse where presentation charts are smooth, and Savings Secured drops from a fictional figure to a true **£0**. A sparse honest series beats a smooth invented one. Expect it; don't treat it as a regression.
- **The £0 that is coming.** "Savings Secured" shows a presentation figure, but the live query against the corpus returned **£0** when audited: all 24 rows in `bp_opportunity` sat at stage `identified`; nothing had been marked realised. The *pipeline* figure is real and healthy — realised savings is genuinely zero until opportunities are progressed through their stages. That is a data/process gap, not a bug, and it is exactly what the presentation marker exists to stop us papering over.
- **Do not feed report tiles from the wrong sources.** Three surfaces in the gateway look like analytics but are not usable here, and wiring a tile to one of them would reintroduce exactly the fabrication this design removes:
  - `GET /dashboard` and `GET /invoices/getAllInvoiceData` read the **seeded `uicanvas` demo tables**, whose "savings" is a flat 6% of spend and whose currency mix is fake.
  - `GET /compliance/compliance-trends` and `/compliance/flagged-cases-by-month` are **hardcoded literal series** in the source (`compliance.service.ts:589-660`) — the `timeRange` parameter only filters invented rows.
  - `proc.bp_detection_finding` (behind the `Detection` module) is **empty**; discrepancy tiles must read `bp_extraction_discrepancy`.
- The `bp_reports` table has **no `CREATE TABLE` in version control** — it exists in the database only. Every migration (`report_key`, `version`, `data_mode`) must therefore be defensive (`ADD COLUMN IF NOT EXISTS`) and must not assume the legacy rows' shape.
- Legacy `bp_reports` rows with `definition IS NULL` predate this feature and are filtered out of every query. Neither the backfill nor the `data_mode` default may touch them.
