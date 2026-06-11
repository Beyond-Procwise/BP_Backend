# Deal Linking & Deal-Centric Views — Design Spec

**Date:** 2026-06-11
**Branch:** Development
**Author:** muthu (with Claude)
**Status:** Draft — pending user review

---

## 1. Context & Problem

ProcWise extracts procurement documents (Quotes, Purchase Orders, Invoices) through a
`raw → _stg → _trgt` pipeline. The redesigned UI (ProcWise-Redesign-2026) is **deal-centric**:
documents are grouped into a **deal** (one Quote → PO → Invoice(s) chain for a supplier), and the
executive dashboard ("Procwise Tail — Executive Intelligence Report") renders KPIs that are only
computable once documents are correctly linked into deals — 3-way match, cycle time (quote→PO→invoice),
price variance across the chain, duplicates, tail-spend visibility, savings.

Today the deal layer is **half-built and internally inconsistent** (verified against live `bp_sqldb`,
2026-06-11):

| Area | Live reality |
|---|---|
| `proc.process_monitor` | 109 rows; has `deal_id` + `deal_name` (11 populated). Look-forward id format `<NAME_UPPER><YYYYMMDD><pm_id>` e.g. `DEAL_A2026052891` |
| `bp_*_raw` / `_stg` / `_trgt` (+ line items) | All already carry `deal_id`, `deal_name`, `document_id` columns |
| `_trgt` deal values | Use a **different** scheme `DEAL-<po_id>` (e.g. `DEAL-526702`, name `"Duncan LLC — PO 526702"`) — conflicts with `process_monitor` |
| `document_id` | Empty everywhere in `_trgt`; `deal_document_id_map` holds 2 junk rows |
| `deal_date` | **Does not exist** on any table |
| `process_monitor_id` FK on `_raw` | Present but never populated |
| Views in `proc` | None exist |
| Linkage signals available | `po_line_items.quote_number` (PO→Quote), `invoice_line_items.po_id` (Invoice→PO), `item_id`/`item_description` (product), `supplier_id`, `buyer_id`, filenames encoding `"X PO526702 for QUT128234.pdf"` |

## 2. Goals / Non-Goals

**Goals**
1. One authoritative deal identity: `deal_id` = backend key from `process_monitor`; `deal_name` = user-facing label. One `deal_id` ↔ many Quotes/POs/Invoices.
2. **Look-forward**: documents inherit the deal already stamped on `process_monitor`.
3. **Look-back** (bulk / pre-existing): derive deal grouping deterministically from multi-signal linkage (canonical PO + supplier + buyer + product/line items). No-PO docs left **unassigned for review**.
4. Add and populate `deal_date` (= order's expected delivery date) on the three `_trgt` tables.
5. Populate `document_id` (stable per-document id within a deal).
6. Keep `process_monitor.status` current per document (deal-linking lifecycle).
7. Provide `bp_`-prefixed views that feed the deal list/detail screens and the executive dashboard KPIs.
8. Reconcile existing inconsistent `DEAL-<po>` values to the authoritative scheme. No source data destroyed.

**Non-Goals**
- No change to extraction accuracy / the L0–L3 extraction pipeline.
- No frontend code; backend tables, views, service, scheduler wiring, and a backfill only.
- No new deal *creation* UX — `deal_id`/`deal_name` for look-forward originate upstream in `process_monitor`.

## 3. Deal Identity Model

- **`deal_id`** — backend grouping key. Authoritative source = `process_monitor.deal_id`. Verbatim, propagated to every related Quote/PO/Invoice (+ their line items) raw→stg→trgt. Never re-minted when a monitor value exists.
- **`deal_name`** — user-facing label from `process_monitor.deal_name`.
- **`document_id`** — stable identifier of a document *within* a deal: `<deal_id>::<doctype>::<doc_pk>` (deterministic, idempotent). Backed by a cleaned `proc.bp_deal_document_map` table (supersedes `deal_document_id_map`).
- **`deal_date`** — the order's **expected delivery date** (per user: "the date when the order will be delivered"). Sourced from the deal's PO `expected_delivery_date`; stamped identically on every doc in the deal. Falls back to `invoice_line_items.delivery_date` then NULL when no PO/delivery date exists. *(Assumption flagged — differs from "document transaction date"; confirm at review.)*

### Look-back deterministic deal_id

When no `process_monitor` deal exists, derive a deal from the **canonical PO** (the join hub):
- `deal_id = 'DEALV2-' || <canonical_po>` (e.g. `DEALV2-526702`). Versioned prefix avoids colliding with legacy `DEAL-<po>`.
- `deal_name = '<supplier_name> — PO <canonical_po>'`.
- Membership decided by the **existing `linking_engine.score_link`** signals (PO ref, supplier identity, amount, line-set/product, temporal, location) — not PO string alone. A doc joins the PO's deal only when its link score clears the configured promote threshold; below that it is held for review.
- **No-PO docs** (no PO reference and no scored parent): `deal_id` stays NULL, `process_monitor.status = 'Deal_Unassigned_Review'`.

## 4. Schema Changes (all additive, `bp_` prefixed)

DDL file: `deploy/sql/2026-06-11_deal_linking.sql` (idempotent — `IF NOT EXISTS` / guarded `ADD COLUMN`).

1. **`deal_date DATE`** added to: `bp_invoice_trgt`, `bp_quote_trgt`, `bp_purchase_order_trgt`, and the matching `_stg` and `_raw` tables (so it propagates through promotion). Line-item tables not required.
2. **`proc.bp_deal_document_map`** — `deal_id varchar, deal_name varchar, document_id varchar PK, doc_type varchar, doc_pk varchar, source_file text, assigned_at timestamptz, assigned_by varchar`. Unique on `(deal_id, doc_type, doc_pk)`.
3. **`process_monitor.status`** — no schema change; new enum values used (see §6).
4. Index: `ix_bp_<doc>_trgt_deal_id` on each `_trgt(deal_id)` for view performance.
5. Legacy `proc.deal_document_id_map` left in place but no longer written (deprecated; documented).

## 5. Linking Engine — `src/services/deal_assignment_service.py`

A new, self-contained service. Public entry: `assign_deals(conn=None, limit=None) -> dict` returning
`{"forward_linked": n, "backward_linked": n, "unassigned_review": n, "reconciled": n, "by_reason": {...}}`.

### 5a. Look-forward pass
1. Select `process_monitor` rows where `deal_id` is non-blank.
2. Match each to extracted documents by **`basename(file_path) == basename(source_file)`** across `bp_*_raw` (and directly against `_trgt` source where raw is absent). Where a monitor row's `category`/`document_type` disambiguates, prefer that doc type.
3. Stamp `deal_id`, `deal_name`, `document_id`, `deal_date` onto the matched doc's `_raw`, `_stg`, `_trgt`, and line-item rows. Set `_raw.process_monitor_id` for future-proofing.
4. Idempotent: re-running overwrites only when the monitor value changed; audited to `bp_agent_actions`.

### 5b. Look-back pass
1. For every `_stg`/`_trgt` doc with **no `deal_id`**, find its canonical PO (`linking_engine._norm_po`) from: own `po_id`, `po_line_items.quote_number` (quote↔PO), `invoice_line_items.po_id` (invoice↔PO), or filename `"… for PO#### / QUT####"`.
2. Score candidate membership with `linking_engine.score_link` (supplier + buyer + amount + product/line + temporal + location). Join when `F ≥ PROMOTE_MIN_LINK_SCORE`.
3. Assign derived `deal_id`/`deal_name`/`document_id`/`deal_date`; persist across raw/stg/trgt + line items.
4. No-PO / unscored → leave unassigned, mark for review.

### 5c. Reconciliation pass
- Where a `_trgt` row currently carries a legacy `DEAL-<po>` **and** a `process_monitor` deal now resolves for the same document, overwrite with the authoritative `process_monitor` `deal_id`/`deal_name` (old value logged to `bp_agent_actions` for audit). Legacy rows with no monitor deal are migrated to the `DEALV2-<po>` derived form for consistency.

## 6. `process_monitor` Status Lifecycle

Existing statuses (`Extracted`, `Completed`, `Extraction_Failed`, `Extraction_InReview`) are preserved.
The deal service advances/sets, per document:

| Status | Meaning |
|---|---|
| `Deal_Linked` | Document assigned to a deal (forward or backward) and persisted to `_trgt` |
| `Deal_Unassigned_Review` | Extracted but no deal could be derived (no PO / below link threshold) — needs human assignment |
| `Deal_Conflict_Review` | Document matched >1 deal with comparable scores — needs disambiguation |

Status writes are idempotent and update `lastmodified_date`. The watcher/NOTIFY trigger is unaffected.

## 7. `bp_` Views for the UI

Created in `deploy/sql/2026-06-11_deal_views.sql` (CREATE OR REPLACE VIEW; read-only).

### 7a. `proc.bp_deal_documents` — one row per document in a deal
Union of the three `_trgt` tables, normalized:
`deal_id, deal_name, document_id, doc_type ('quote'|'po'|'invoice'), doc_pk, doc_number, doc_date (quote/order/invoice date), deal_date, supplier_id, supplier_name, buyer_id, currency, amount, amount_incl_tax, converted_amount_usd, country, region, confidence_score, status, created_date`.

### 7b. `proc.bp_deal_overview` — one row per deal (deal list screen)
Aggregates `bp_deal_documents` by `deal_id`:
`deal_id, deal_name, supplier_id, supplier_name, buyer_id, deal_date, first_activity_date (MIN doc_date), last_activity_date (MAX doc_date), quote_count, po_count, invoice_count, quote_total, po_total, invoice_total, currency, converted_total_usd, three_way_match_state (has Q&PO&Inv), price_variance_pct (invoice vs po vs quote), cycle_days_quote_to_po, cycle_days_po_to_invoice, duplicate_flag, deal_status`.

### 7c. `proc.bp_deal_kpis` — single-row executive dashboard feed
Powers the screenshot's tiles:
`savings_secured, savings_rate_pct, in_flight_count, opportunity_pipeline, tail_spend_total, tail_spend_pct, under_contract_pct, avg_cycle_days, avg_days_to_po, price_variance_pct, duplicate_count, three_way_match_pct, no_po_spend, compliance_score`.
Derivations documented inline in the SQL (e.g. `three_way_match_pct` = deals with Q+PO+Inv ÷ total deals; `price_variance_pct` = mean abs % diff invoice_total vs po_total; `duplicate_count` = invoices sharing supplier+amount+near date).

### 7d. `proc.bp_process_monitor_status` — document processing status board
`id, file_path, doc_type, category, deal_id, deal_name, status, start_ts, end_ts, lastmodified_date` — drives any "documents being processed / needs review" panel.

> Deal list/detail column sets are a best-effort match to the redesign; to be reconciled against the Figma frames (only the exec dashboard screenshot was legible locally).

## 8. Scheduler Integration

Register `deal-assignment` job in `backend_scheduler` (mirrors `trgt-promotion`):
- Runs **after** `trgt-promotion` (so `_trgt` is populated first). Interval via `DEAL_ASSIGNMENT_INTERVAL_MINUTES` (default 15), toggle `DEAL_ASSIGNMENT_ENABLED` (default on), `initial_delay` 5 min.
- Calls `deal_assignment_service.assign_deals()` and logs the result dict.

## 9. Backfill / One-Time Migration

Script `scripts/backfill_deal_linking.py`:
1. Apply DDL (§4).
2. Run `assign_deals()` once over all existing `_trgt` rows (forward + backward + reconcile).
3. Refresh views.
4. Print a reconciliation report (rows linked, reconciled, unassigned). No deletes; legacy values archived in `bp_agent_actions`.

## 10. Testing

- **Unit** (`tests/services/test_deal_assignment_service.py`, fake DB): filename→source_file matching; canonical-PO grouping; multi-signal join threshold; no-PO → review; document_id determinism; idempotent re-run; reconciliation overwrite.
- **View** (`tests/sql/test_deal_views.py` or live read-only): `bp_deal_overview` counts/totals reconcile against `bp_deal_documents`; 3-way-match and cycle-time math on a seeded deal.
- **Live validation**: run backfill against `bp_sqldb`, verify the known chains (PO526702 / Duncan, PO502001 / Thrive invoices) group correctly and KPIs are non-null.

## 11. Assumptions & Open Questions

1. **`deal_date` = expected delivery date** (from PO), stamped on all docs — confirm vs document transaction date.
2. Deal list/detail **view columns** are inferred; the only legible screenshot was the executive dashboard. Will reconcile against Figma frames when accessible.
3. Look-back threshold reuses `PROMOTE_MIN_LINK_SCORE` (default 80). Acceptable, or a separate `DEAL_LINK_MIN_SCORE`?
4. Legacy `DEAL-<po>` rows with no monitor deal are migrated to `DEALV2-<po>`; acceptable to rewrite these keys?
