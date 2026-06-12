# Deal Linking — Implementation Summary (2026-06-11)

Status: **Implemented, tested, and backfilled against live `bp_sqldb`.** Branch `Development`.

## What was built

A deal-linking layer that groups Quotes/POs/Invoices into deals and exposes deal-centric `bp_` views for the ProcWise redesign UI (incl. the executive dashboard).

### Identity model
- **`deal_id`** — backend grouping key. Authoritative source = `proc.process_monitor.deal_id` (look-forward). Derived `DEALV2-<canonical_po>` for look-back. One `deal_id` ↔ many docs.
- **`deal_name`** — user-facing label (from `process_monitor`, or `"<supplier> — PO <po>"` when derived).
- **`document_id`** — `<deal_id>::<doc_type>::<doc_pk>`, recorded in `proc.bp_deal_document_map`.
- **`deal_date`** — the order's **expected delivery date** (from the PO's `expected_delivery_date`; invoice-line `delivery_date` fallback), stamped on every doc in the deal.

## Database changes (live `bp_sqldb`, schema `proc`)

**DDL** — `deploy/sql/2026-06-11_deal_linking.sql` (additive, idempotent):
- Added `deal_date DATE` to `bp_{invoice,quote,purchase_order}_{raw,stg,trgt}` (9 tables).
- New table `proc.bp_deal_document_map` (document_id PK; deal_id, deal_name, doc_type, doc_pk, source_file, assigned_at/by) + unique `(deal_id, doc_type, doc_pk)`.
- Indexes `ix_bp_{invoice,quote,purchase_order}_trgt_deal_id`.

**Views** — `deploy/sql/2026-06-11_deal_views.sql` (read-only):
- `bp_deal_documents` — one row per document in a deal (normalized union of the 3 `_trgt` tables).
- `bp_deal_overview` — one row per deal: counts, totals, deal_date, first/last activity, 3-way-match flag, price-variance %, quote→PO and PO→invoice cycle days.
- `bp_deal_kpis` — single-row executive-dashboard feed (3-way-match %, avg cycle days, days-to-PO, price variance, duplicate count, invoiced total, no-PO spend).
- `bp_process_monitor_status` — per-document processing/deal status board.

## Code changes

- **`src/services/deal_assignment_service.py`** (new) — the engine. Passes run in order:
  1. `_look_forward` — match `process_monitor` rows (with deal_id) to extracted docs by **exact `process_monitor_id`** (set by the RENOVATION extraction path), falling back to `basename(file_path)==basename(source_file)`; stamp deal columns, set `process_monitor.status='Deal_Linked'`.
  2. `_look_back` — for deal-less invoices/quotes, find canonical PO, score membership with `linking_engine.score_link` (supplier/buyer/amount/line-item/temporal/location); join when `F ≥ PROMOTE_MIN_LINK_SCORE` (80). **Joins the PO's existing deal if present — never overwrites an authoritative deal_id.** No-PO docs left unassigned.
  3. `_reconcile_legacy` — rewrite legacy `DEAL-<po>` → `DEALV2-<po>` (parameterized LIKE).
  4. `_propagate_deal_along_po` — spread a known deal across the **full PO chain**: any Quote/PO/Invoice sharing a canonical PO joins the single deal present on any of them, so the complete chain lands in one deal even if only one doc was tagged. Conflicting deals on a PO are left for review.
  5. `_backfill_deal_metadata` — stamp `document_id` + `deal_date` on any deal-assigned doc missing them.
  6. `_flag_unassigned` — mark deal-less monitor rows `Deal_Unassigned_Review`.
  - `assign_deals(conn=None)` orchestrates; idempotent (re-run is a clean no-op).

## End-to-end fresh-extraction validation (2026-06-12)

Verified the full pipeline holds for a truncate-and-re-extract, via live transactional simulations (rolled back):
- **`deal_date` is protected** across re-extraction: added to `linking_engine._DEAL_COLS` so stg→trgt promotion never clobbers the assigned value. `raw→stg` and `stg→trgt` promotions both carry/ignore `deal_date` safely (dynamic column detection; no hardcoded lists).
- **Look-forward works on fresh data**: extraction writes `raw.source_file == process_monitor.file_path` and `raw.process_monitor_id`, so the exact-id match links the deal. Simulated invoice → its PO + quote all joined the deal (`forward_linked:1, propagated:2`), with `deal_date` resolved from the PO's `expected_delivery_date`.
- **FK-safe truncation**: `deploy/sql/truncate_for_fresh_extraction.sql` (single `TRUNCATE … RESTART IDENTITY CASCADE`) — tested transactionally, no FK violation on the `bp_*_raw.process_monitor_id → process_monitor` constraint.

### Fresh-extraction runbook
1. `psql -f deploy/sql/truncate_for_fresh_extraction.sql` (review counts, then COMMIT).
2. Ensure deal columns/views exist: `deploy/sql/2026-06-11_deal_linking.sql` + `deploy/sql/2026-06-11_deal_views.sql` (idempotent — safe to re-apply).
3. Upload + extract documents (tag `process_monitor.deal_id`/`deal_name` for look-forward deals).
4. The `backend_scheduler` runs `trgt-promotion` then `deal-assignment` automatically; or run `python3 scripts/backfill_deal_linking.py` once to force it immediately.
5. Read deals from `proc.bp_deal_overview` / `bp_deal_kpis` / `bp_deal_documents` / `bp_process_monitor_status`.
- **`src/services/backend_scheduler.py`** — registers a `deal-assignment` job after `trgt-promotion` (toggle `DEAL_ASSIGNMENT_ENABLED`, interval `DEAL_ASSIGNMENT_INTERVAL_MINUTES` default 15).
- **`scripts/backfill_deal_linking.py`** (new) — one-time apply DDL → assign → views → report.

## Tests
- `tests/services/test_deal_assignment_service.py` — 13 tests (helpers, persist, look-forward incl. unmatched-but-tagged, look-back incl. PO-deal-preservation invariant, reconcile LIKE-param regression, metadata backfill, orchestrator wiring/order). 
- `tests/services/test_backend_scheduler.py` — 2 tests (job registers when enabled / skipped when disabled).
- Neighbouring regression (deal/scheduler/linking/promotion): **20 passed**.

## Live result (verified, persisted)
- `_trgt` tables: **100%** of rows carry `deal_id`, `document_id`, `deal_date` (invoice 2/2, quote 5/5, PO 7/7).
- **7 deals** in `bp_deal_overview`; **14** rows in `bp_deal_document_map`; **4** `bp_` views live.
- `process_monitor` status: `Deal_Linked` 11 · `Deal_Unassigned_Review` 90 · `Extraction_Failed` 8.
- KPIs populated (invoiced £3,371.11, avg cycle 3.0 days, price variance 71.4%).

## Known limitations / follow-ups
1. **Look-forward matched 0 on current data** — the demo `process_monitor` deal rows point at filenames not present in the (near-empty) `_raw` tables, so the filename join finds nothing. It will link once real uploads flow through extraction with `_raw.source_file` set. The 11 deal-tagged monitor rows are still correctly `Deal_Linked` (the deal_id is on the monitor row itself).
2. **`_flag_unassigned` keys on `process_monitor.deal_id`**, not the `_trgt`/map deal. A doc linked *only* via look-back (deal on the `_trgt` row, monitor row still null) can show `Deal_Unassigned_Review`. Acceptable now (monitor↔trgt join is filename-only and look-back map rows carry no source_file); revisit if the monitor↔trgt linkage is strengthened.
3. **Deal list/detail view columns are inferred** — only the executive-dashboard screen was legible; the Figma MCP is blocked by account seat/plan. Reconcile `bp_deal_documents`/`bp_deal_overview` columns against the deal-list/detail frames when available.
4. **`deal_name`/`supplier_id` in `bp_deal_overview` use `max()`** to collapse per deal — assumes uniformity across a deal's docs.
