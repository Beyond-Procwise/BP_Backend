# Quote-Anchored Deal Model with Orphan Handling — Design Spec

**Date:** 2026-06-15 · **Branch:** Development · **Author:** muthu (with Claude)
**Status:** Approved (3 sections) — proceeding to implementation.

## 1. Context & Problem

The deal-linking layer currently anchors deals on the canonical **PO** (`DEALV2-<po>`):
any invoice/quote that resolves a parent PO joins that PO's deal. The user redefines
the procurement flow so the **QUOTE is the anchor**:

- Flow is **Quote → PO → Invoice**.
- A PO or invoice that has no relevant quote is **orphaned** until a relevant quote
  arrives — *even if it carries an upload-time `deal_id`* (it stays linked to that
  deal but flagged orphaned).
- Once a relevant quote appears, the chain auto-completes.

Downstream — analysis (exec KPIs / 3-way match), negotiation, and opportunity — must
compute against this orphan-aware, quote-anchored model.

**Live-data reality:** explicit quote↔PO references (`po_line_items.quote_number`,
`quote.po_id`) are NULL today; only `invoice.po_id` is reliably populated.

## 2. Decisions (confirmed)

1. **Quote matching = relationship score.** A quote is "relevant" to a PO when
   `linking_engine.score_link(quote, po, "quote_po")` ≥ `PROMOTE_MIN_LINK_SCORE`
   (supplier identity + line-item/product overlap + amount + temporal), or an explicit
   ref exists (`quote.po_id` / `po_line_items.quote_number` / canonical-PO / filename
   `for QUT####`). An invoice attaches transitively: `invoice.po_id` → PO → that PO's
   anchoring quote.
2. **Orphan model = process_monitor status + derived view flags.** New status
   `Orphaned_Awaiting_Quote`; views derive `has_quote_anchor` / `orphaned`. No new
   `_trgt` columns.
3. **Deal key = PO-keyed, quote-gated.** Auto-formed deals stay `DEALV2-<po>`, but a
   deal only **forms** when a quote score-matches the PO. POs/invoices with no matching
   quote stay orphaned (no `DEALV2` minted). Look-forward upload `deal_id` is preserved.

## 3. Core model

- **A deal is *complete* iff its canonical-PO chain contains ≥1 quote.**
- **A quote is never orphaned** (it is the anchor).
- A **PO or invoice** whose chain has **no quote** is `Orphaned_Awaiting_Quote`
  (keeps its `deal_id` if it had one; otherwise no deal).
- Orphans **auto-heal** to `Deal_Linked` on the next assignment pass once a quote
  score-matches their PO. This runs via the existing event chain
  (`_run_downstream_chain`) and the periodic deal-assignment backstop.

## 4. Deal assignment (`src/services/deal_assignment_service.py`)

### 4a. `_look_back` — quote-gated (rewrite)
For each canonical PO present in `_trgt`:
- Determine whether **any quote anchors it**: a quote with `score_link(quote, po,
  "quote_po").F ≥ MIN_LINK_SCORE` or an explicit/canonical/filename PO ref.
- **Anchored** → form/confirm `DEALV2-<po>` (or honor an existing authoritative deal):
  assign the quote(s), the PO, and the PO's invoices to the deal (existing
  `_persist_deal` + map upsert).
- **Not anchored** → do **not** mint a deal; leave the PO/invoices for the orphan pass.

Helper `_quote_anchor_for_po(cur, canonical_po) -> Optional[quote_row]` encapsulates
the matching (explicit ref first, then score).

### 4b. `_flag_orphans` (new pass)
Set `process_monitor.status='Orphaned_Awaiting_Quote'` for every PO/invoice document
whose canonical-PO chain has **no quote** in `_trgt`. Set-based, mapped via
`raw.process_monitor_id`. Skips `Extraction_Failed` and `Deal_Conflict_Review`.
Returns count. Runs before `reconcile_status`.

### 4c. `reconcile_status` — precedence update
Per doc, by true stage (set-based, one UPDATE per doc type):
1. keep `Extraction_Failed`, `Deal_Conflict_Review`
2. **`Orphaned_Awaiting_Quote`** — doc is PO/invoice, in `_trgt`, chain has no quote
3. `Deal_Linked` — quote-with-deal, or PO/invoice whose chain has a quote
4. `Deal_Unassigned_Review` — in `_trgt`, no deal, not orphan-eligible
5. `Staged` / `Discrepancy_Review` / `Extracted` (unchanged)

`_run` adds `_flag_orphans`; result dict gains `orphans_flagged`.

## 5. Views (`deploy/sql/2026-06-15_quote_anchor_views.sql`)

- `bp_deal_overview` — add `has_quote_anchor BOOLEAN` (`quote_count>0`) and
  `orphaned BOOLEAN` (`NOT has_quote_anchor AND (po_count>0 OR invoice_count>0)`).
- `bp_deal_kpis` — headline metrics computed over **complete** deals
  (`WHERE has_quote_anchor`); add `complete_deal_count`, `orphaned_deal_count`,
  `orphaned_spend` (sum of po/invoice totals on orphaned deals).
- `bp_deal_orphans` (new) — one row per orphaned PO/invoice doc: `deal_id, doc_type,
  doc_pk, supplier, amount, currency, po_id, status, last_activity_date`.

## 6. Dashboards (computed accordingly)

- **Negotiate** (`negotiate_dashboard.py`): add `orphaned` to the payload. Savings
  baseline is the **quote** (`savings = quote_total − po/invoice total`). When orphaned
  (no quote), `proposalSnapshot.savingsVsBaseline` and `negotiationData.Savings` return
  `"awaiting quote"` / null instead of a misleading figure.
- **Opportunity** (`opportunity_dashboard.py`, `opportunity_miner_agent.py`,
  `bp_opportunity`): add `quote_id` column to `bp_opportunity`; price-benchmark
  detection uses the **quote price** as the reference baseline where a quote anchors;
  opportunities on orphaned chains are flagged (`orphaned=true` in the detail row).
- **Analysis / exec** (`bp_deal_kpis`): orphaned spend is its own tile; 3-way-match %,
  cycle time, variance over complete deals. Maps to the Figma exec report
  (no-PO/maverick spend ↔ orphaned spend; 3-way match %; savings from quote baseline).

## 7. KG (`procurement_kg_builder.py`)
Add an `orphaned` boolean property to PO/Invoice nodes (derived from the deal state).
The `QUOTE_FOR_PO` / `INVOICE_REFERENCES_PO` edges already encode the chain; no new
edge types.

## 8. API (`src/api/routers/deal_summary.py` or new)
- Add `orphaned` (+ `has_quote_anchor`) to deal payloads where a deal is returned.
- New `GET /deals/orphans` — list orphaned PO/invoice docs awaiting a quote.

## 9. Implementation phases (built + verified in order)
1. **Foundation** — `_quote_anchor_for_po`, quote-gated `_look_back`, `_flag_orphans`,
   `reconcile_status` precedence, `Orphaned_Awaiting_Quote`.
2. **Views** — `has_quote_anchor`/`orphaned`/orphan KPIs + `bp_deal_orphans`.
3. **Dashboards** — negotiate (orphan + quote baseline), opportunity (`quote_id` +
   quote benchmark), analysis KPIs.
4. **KG + API** — orphaned property, `/deals/orphans`.

## 10. Testing
- Unit (fake-cursor, per `test_deal_assignment_service.py`): quote-gated look-back
  (anchored vs not), `_flag_orphans` (PO/invoice without quote → flagged; quote never
  flagged), reconcile precedence (orphan above Deal_Linked, below conflict), orphan
  auto-heal when a quote is added.
- View tests / live read: `bp_deal_overview.orphaned` matches the rule;
  `bp_deal_kpis` complete vs orphaned counts reconcile.
- Dashboard tests: negotiate payload shows `orphaned` + "awaiting quote" baseline;
  opportunity row carries `quote_id`.

## 11. End-to-end live validation (acceptance)
Against live `bp_sqldb`: run `assign_deals`, confirm the current PO-only chains
(PO405867 / PO519829, no matching quote) become `Orphaned_Awaiting_Quote`; deals with
a quote (deal_perry, test0015) are `Deal_Linked`/complete; `bp_deal_kpis` splits
complete vs orphaned; negotiate/opportunity endpoints reflect orphan state over HTTP.
Restart `procwise`; verify the event chain + scheduler keep the model consistent.

## 12. Orchestration efficiency (cross-cutting)
The chain is event-driven (promotion NOTIFY → `_run_downstream_chain`) with periodic
backstops. The new passes are set-based (one UPDATE per doc type), so adding
`_flag_orphans` + orphan precedence does not add per-row work. Mining stays chained to
deal-change. Confirm no redundant full-table scans are introduced; keep all new passes
idempotent so the event chain and the timers can both run them safely.

## Assumptions
- "Relevant quote" threshold reuses `PROMOTE_MIN_LINK_SCORE` (80). A separate
  `QUOTE_ANCHOR_MIN_SCORE` env can override if the quote↔PO bar should differ.
- Competing quotes: multiple quotes may anchor one `DEALV2-<po>` deal; any one of them
  makes the deal complete.
