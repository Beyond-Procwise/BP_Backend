# Cross-Document Reconciliation → Consolidation Actions — Design

**Date:** 2026-06-06
**Status:** Approved; live-tested (no unit tests, per user)
**Builds on:** `2026-06-05-agent-actions-and-deal-summary-design.md` (the consolidation writer slot)

## Problem

`proc.agent_actions` has a documented `phase='consolidation'` writer slot, but no
producer. Wire up cross-document reconciliation that compares the final (`_trgt`)
documents within a deal and writes consolidation actions describing matches and
mismatches.

## Scope

In scope: a `reconcile_deal()` service, a `POST /deals/{deal_id}/reconcile`
endpoint, and an auto-reconcile step before summarization. Validation is **live**
against the running service and DB — no unit tests (per user 2026-06-06).

Out of scope: line-level matching, status transitions / promotion changes,
mutating any `_trgt` record.

## Verified schema facts (live, 2026-06-06)

The `_trgt` docs share linking + amount fields:
- invoice_trgt: `invoice_total_incl_tax`, `converted_amount_usd`, `tax_percent`,
  `currency`, `supplier_id`, `po_id`, `deal_id`.
- purchase_order_trgt: `total_amount_incl_tax`, `converted_amount_usd`,
  `tax_percent`, `currency`, `supplier_id`, `po_id`, `deal_id`.
- quote_trgt: `total_amount_incl_tax`, `converted_amount_usd`, `tax_percent`,
  `currency`, `supplier_id`, `po_id`, `quote_id`, `deal_id`.
Deals `1619539563`, `523733748` each have 1 invoice + 1 PO + 1 quote;
`837265240` has 2 invoices + 1 PO + 1 quote.

## 1. Reconciler — `src/services/reconciliation.py`

`reconcile_deal(deal_id, conn=None) -> dict | None`
- Loads docs via `gather_deal_context(deal_id, conn)` (DRY). Returns `None` if the
  deal is unknown.
- Flattens the deal's invoices + POs + quotes into one list of doc summaries, each
  with: `doc_kind` (invoice/purchase_order/quote), `doc_pk`, `amount_usd`
  (`converted_amount_usd`), `currency`, `supplier_id`, `tax_percent`.
- Runs four independent checks, each returning a verdict
  `{dimension, status, values, detail}` where `status ∈ {match, mismatch, skipped}`:
  - **amount_usd** — `match` if `max−min ≤ max(PCT·max, ABS)` over non-null
    `amount_usd` values; `skipped` if <2 values.
  - **currency** — `match` if ≤1 distinct non-null currency; `skipped` if <2 docs
    have currency.
  - **supplier** — `match` if ≤1 distinct non-null `supplier_id`; `skipped` if <2.
  - **tax** — `match` if `max−min ≤ TAX_PCT` over non-null `tax_percent`;
    `skipped` if <2.
- Writes one consolidation action per dimension via `bulk_record(..., conn=conn)`:
  `phase='consolidation'`,
  `action_type`: `reconcile_match` (status match) / `reconcile_mismatch`
  (mismatch) / `reconcile_skipped` (skipped),
  `status`: `ok` / `warn` / `skipped`,
  `field_name`: the dimension, `deal_id`, `summary` (human-readable),
  `details`: the per-document values compared. Atomic with the caller's conn.
- Returns `{deal_id, checks: {<dimension>: verdict, ...}, actions_written}`.

Tolerances (env-overridable): `RECON_AMOUNT_TOLERANCE_PCT=0.01`,
`RECON_AMOUNT_TOLERANCE_ABS=1.00`, `RECON_TAX_TOLERANCE_PCT=0.1`.

## 2. Trigger points

- **Endpoint** `POST /deals/{deal_id}/reconcile` added to the existing
  `src/api/routers/deal_summary.py` router (same `/deals` prefix). Runs
  `reconcile_deal`, returns the verdicts. 404 if unknown deal; 500 on error.
- **Before summary** — `summarize_deal` calls `reconcile_deal(deal_id, conn)`
  first, best-effort (a reconcile failure is caught/logged and never blocks the
  summary). Because reconciliation writes consolidation actions before
  `gather_deal_context` reads the action trail, the summary reflects the fresh
  matches/mismatches.

## 3. Behavior

- Read-only on `_trgt`; only writes to `proc.agent_actions`.
- Append-only event log: each run appends a fresh dimension set (an "as-of"
  record); repeated summaries grow the trail — intended.
- New canonical action_type `reconcile_skipped` (insufficient data) — transparency
  over silent omission.

## 4. Testing (live)

- DB: run `reconcile_deal` for a 1-inv/1-PO/1-quote deal and a mismatch deal;
  confirm consolidation rows land in `proc.agent_actions` with correct
  action_type/field_name/status/details.
- Service: restart procwise; `POST /deals/{deal_id}/reconcile` → 200 verdicts and
  404 for unknown; then `GET /deals/{deal_id}/summary` narrates the reconciliation.
