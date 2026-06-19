# Deal Analysis Summary — Design Spec

**Date:** 2026-06-19
**Status:** Awaiting user review
**Author:** Nick (+ Claude)

## 1. Goal

When a deal reaches `Deal_Linked` status in `proc.process_monitor`, automatically and
**in parallel** produce, for that deal:

1. A **structured metrics row** (the "Detailed Analysis Summary" grid the UI shows), persisted
   to a new `proc.bp_analysis_summary` table.
2. A **narrative text summary**, persisted to the existing `proc.bp_summary` table.

Run this for **all** linked deals (idempotent backfill), wire the metrics row to the live UI
grid, and keep the UI aligned with the existing design.

## 2. Background (current state)

- `process_monitor.status` flips to **`Deal_Linked`** inside `reconcile_status()`
  (`src/services/deal_assignment_service.py:687-738`) — only when a deal has all three doc types
  (quote + PO + invoice). It is run by the periodic `deal-assignment` scheduler job
  (`backend_scheduler.py`), **not** event-triggered.
- Two summary paths already exist:
  - `src/services/deal_summary.py` → `summarize_deal(deal_id)` — **local AgentNick**, returns
    narrative text, **not persisted**.
  - `src/services/summary_agent.py` → persona summaries, **persisted to `proc.bp_summary`**
    (`deploy/sql/2026-06-08_create_bp_summary.sql`), runs on a **daily** precompute job, uses
    cloud qwen.
- **No structured metrics table exists.** `proc.bp_analysis_summary` is referenced only in
  `deploy/sql/truncate_for_fresh_extraction.sql` (no DDL).
- UI grid `beyond_procwise_ui/src/modules/HomeAnalyse/Analyse/AnalysisSummary.jsx` renders
  columns: Deal ID, Supplier, Category, Deal Value, Volume, Unit Price, Price Change,
  Volume Change, Efficiency Score. It is currently fed from the **legacy AWS API gateway**
  (`VITE_API_URL` → `/analyse`), not this Python backend.

## 3. Data model — `proc.bp_analysis_summary`

One **current** row per deal (`is_current` flag pattern, mirroring `bp_summary`). All values
computed deterministically from the deal's own `_trgt` documents + line items. Per the
no-fabrication rule, anything not derivable stays `NULL` (rendered as "–").

| Column | Type | Definition |
|---|---|---|
| `analysis_id` | UUID PK | generated |
| `deal_id` | varchar | deal key |
| `deal_name` | varchar | deal name |
| `supplier` | text | supplier from the deal's docs |
| `category` | text | deal category (`process_monitor.category`) |
| `deal_value` | numeric | final **invoice** total (fallback PO → quote) |
| `currency` | varchar(8) | currency of `deal_value` |
| `volume` | numeric | sum of line-item quantities (invoice; fallback PO/quote) |
| `unit_price` | numeric | `deal_value / volume` (weighted average) |
| `price_change_pct` | numeric | invoiced unit price vs **quoted** unit price, signed |
| `volume_change_pct` | numeric | invoiced volume vs **quoted/ordered** volume, signed |
| `efficiency_score` | numeric | realized savings = favorable price change × volume (best-effort; `NULL` when not computable) |
| `items` | jsonb | list of `{name, qty, unit_price}` for products covered |
| `item_count` | int | number of distinct line items |
| `narrative_summary_id` | UUID | FK → `bp_summary.summary_id` |
| `data_snapshot` | jsonb | raw figures used (provenance) |
| `model` | text | model used for the narrative |
| `is_current` | bool | only one current row per deal |
| `generated_at` | timestamptz | generation time |

Index: `ix_bp_analysis_summary_deal_id` on `(deal_id, is_current)`.

### Metric derivation rules
- **Within-deal comparison** (confirmed): Price Change and Volume Change compare the deal's
  **invoice** against its **quote** (fallback to PO when no quote line is matchable).
- Unit prices are computed as weighted averages (total value / total qty) so a deal with mixed
  line items still yields one comparable figure.
- If a deal lacks a quote (should not happen at `Deal_Linked`, but defensively), change columns
  are `NULL`.

## 4. Trigger flow

New service **`src/services/deal_analysis_service.py`**:

- `compute_deal_metrics(deal_id, conn) -> dict` — deterministic, **no LLM**. Reads the deal's
  `_trgt` headers + line items, returns the metric dict above.
- `sync_deal_summaries(conn=None, deal_ids=None, max_workers=4) -> dict` — finds every deal at
  status `Deal_Linked` that lacks a *current* `bp_analysis_summary` row (or `deal_ids` if given),
  and processes each deal **concurrently**:
  1. `compute_deal_metrics()` → upsert `bp_analysis_summary` (flip prior row's `is_current`).
  2. Generate **narrative** → upsert `bp_summary` (scope=deal, persona=analysis); set
     `narrative_summary_id`.
  Idempotent: re-running only (re)generates deals missing a current summary. Returns counts.

**Hook point:** call `sync_deal_summaries()` at the **end of `assign_deals()`**, immediately
after `reconcile_status()` flips statuses — so a deal's summary is produced as soon as it becomes
`Deal_Linked`, in parallel, without blocking the linking job (failures are logged, non-fatal).
This is the natural event boundary and reuses the existing scheduler cadence; no new DB trigger
or LISTEN/NOTIFY needed.

### Narrative model decision (CONFIRMED 2026-06-19)
Generate the narrative with **local AgentNick** via `deal_summary.summarize_deal()`, and
**persist the result into `bp_summary`** (the established summary table) so it is cached and
UI-reachable. This honors both the **AgentNick-is-the-only-base-model** hard rule and the
**Single-Model Consolidation** direction (AgentNick:unified is the single brain for all
non-extraction tasks; the older cloud-qwen summary routing is superseded). The cloud
`summary_agent` path is NOT used for this feature.

## 5. API

New endpoints on this backend (FastAPI, alongside `src/api/routers/deal_summary.py`):

- `GET /deals/{deal_id}/analysis-summary` → the current `bp_analysis_summary` row mapped to the
  **exact UI row shape**:
  ```json
  {
    "id": "DEALV2-...", "supplier": "...", "category": "...",
    "value": "£470K", "volume": "105,000", "unitPrice": "£4.48",
    "priceChange": "+7.5%", "volumeChange": "-1.7%",
    "efficiency": "18.06",
    "items": "Widget A, Bolt B, Cable C"
  }
  ```
  - `priceChange`/`volumeChange` are **signed strings** (`"+7.5%"` / `"-1.7%"`) because the UI
    keys row color off `.startsWith('+')`.
  - `items` is a **pre-formatted string** (not an array) because the UI search filter calls
    `.toLowerCase()` on every field value; an array would crash it. (We still store structured
    `items` jsonb in the table; the endpoint flattens to a string.)
  - `NULL` numerics render as `"–"`.
- `GET /deals/analysis-summary` → list of current rows (for the grid when no single deal scoped).
- `POST /deals/analysis-summary/sync` → manual backfill = run `sync_deal_summaries()` for all
  linked deals.

## 6. UI changes (`beyond_procwise_ui`)

`src/modules/HomeAnalyse/Analyse/AnalysisSummary.jsx`:
- Add **Items** column (10 columns total: keep Efficiency Score, add Items as the last column).
- Match existing design tokens exactly: header `#09608B` / cell color `#64748b` / zebra
  `#f8fafc` / `fontSize: 11` body, `15` title, `whiteSpace: 'nowrap'` headers,
  `border: '1px solid #e0e7ef'` card.
- Harden the search filter (line 10-12) so non-string values can't throw (`String(v ?? '')`).
- Repoint the data source from the legacy AWS `/analyse` to the **local backend**
  (`VITE_AI_API_URL` → `/deals/{deal_id}/analysis-summary`) so it reads live `bp_sqldb`. The
  feeding query lives in `CustomTabs.jsx` (`fetchAnalysisData`) — update its URL/mapping.

### Figma alignment
The UI repo contains **no Figma file reference** (no `*figma*` files, no embedded links). The
existing `AnalysisSummary.jsx` component is therefore treated as the authoritative,
Figma-aligned implementation, and the new Items column will reuse its exact tokens/spacing so it
remains visually consistent. **If a Figma file URL is provided**, we will cross-check the column
layout and tokens against it via the Figma MCP before finalizing; until then, alignment is to the
in-code design system.

## 7. Error handling

- Metric computation failure for one deal → log + skip that deal; never abort the batch or the
  linking job.
- Narrative LLM failure → still persist the metrics row; leave `narrative_summary_id` NULL and
  retry on the next sync.
- The table holds exactly ONE row per deal: `upsert_analysis_row` deletes any prior row(s) for the deal and inserts the new one, both committed together (one transaction), so a reader never sees the deal with zero rows mid-update. The narrative (`bp_summary`) and the metrics row (`bp_analysis_summary`) are committed in separate transactions and reconciled best-effort — a re-sync heals any orphaned narrative.

## 8. Testing & validation

- Unit tests for `compute_deal_metrics` (deal with quote+PO+invoice; missing-quote defensive
  path; multi-line weighted averages; no fabrication on absent fields).
- Test that `sync_deal_summaries` is idempotent and concurrent (only fills deals lacking a
  current row).
- Endpoint test for the UI row-shape mapping (signed strings, items-as-string, NULL→"–").
- **Live validation (required by project rule):** run against the running local server + live
  `bp_sqldb`: trigger `assign_deals()`, confirm `Deal_Linked` deals get both a
  `bp_analysis_summary` row and a `bp_summary` narrative, and confirm the UI grid renders them.

## 9. Out of scope (YAGNI)

- Historical / cross-deal trend baselines for Price/Volume change (within-deal only for now).
- New summary model/routing beyond the existing AgentNick + `bp_summary` paths.
- Rebuilding the legacy AWS `/analyse` payload's other panels (graph, proposal snapshot, etc.).
