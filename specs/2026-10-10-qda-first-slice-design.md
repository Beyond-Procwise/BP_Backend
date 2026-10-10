# QDA backend, first slice: B1, B1b, B4, B11

**Date:** 2026-10-10
**Status:** approved by Nick 2026-10-10 (new tables on both databases, the two fixtures, this spec as written). Built the same day; see "As built" at the end.
**Source:** `beyond_procwise_ui/docs/qda/backend-build-instructions.md` §2 and `product-review.md` (Phase 0).
**Base:** `src/services/sell_side/`, `src/api/routers/sales.py`. No new quote namespace.

## What this slice does

Today a customer quote and what it costs us are one database row, and the only thing that
keeps cost off the customer's page is a list of forbidden field names. This slice makes the
two separate records, adds a check that looks for our figures (not just their names) in what
the customer is sent, records who did what to a quote, and closes one ungated route.

It does not touch pricing, numbering, the renderer, the pack, or anything on the blocked list.

## B1. Cost and margin move to their own tables

Two new tables, one row per quote and one per line:

- `proc.bp_sales_quote_internal` (`sales_quote_id` PK, FK cascade): `total_cost`, `total_margin`, `margin_pct`
- `proc.bp_sales_quote_line_internal` (`sales_quote_line_id` PK, FK cascade): `unit_cost`, `cost_tier_applied`, `line_margin`, `line_margin_pct`

Code changes:

- `quotes.create_draft` writes customer fields to the existing tables and internal figures to the new ones, in the same transaction.
- `quotes.get_quote` selects a named list of customer columns (no `SELECT *`), so it cannot return an internal figure. `_transition` likewise.
- New `quotes.get_quote_internal(conn, id)` returns the internal record, keyed by quote and by line id.
- `approve()` (below-cost bar) and `outcomes.record_outcome` (`won_margin`) read margin through the internal reader.
- `GET /sales/quotes/{id}` returns the customer record with no permit beyond `quote.read`. New `GET /sales/quotes/{id}/internal` returns the internal record behind the existing `margin.read` permit. This changes the response of an existing route; nothing calls it today (the gateway has no sell-side proxy).
- `quote_render.customer_view` takes the customer record only.

Migration, in two separate steps:

1. **Now:** create the two tables, copy the existing figures (bp_testdb: 3 quotes, 4 lines; bp_sqldb: none). Idempotent, with a rollback file. The old columns stay, untouched and no longer read or written.
2. **Later, separately approved:** drop the seven old columns, after a check that both copies agree.

Between the steps, a quote created by a not-yet-restarted server lands in the old columns only; the copy script is re-runnable to pick it up.

Left where they are, on purpose: `bp_sales_opportunity.expected_cost/expected_margin` and
`bp_sales_quote_outcome.won_margin`. Neither table is a customer document and both are
already behind `margin.read`. `money.py` and `costing.py` compute in memory and do not change.

**Done when:** a test reads a quote through the customer path and the query it runs names no internal column; the 81 sell-side tests pass.

## B1b. A leak check on figures, not only names

A new check runs over the finished HTML, given the quote's internal record:

- Each internal money figure (`unit_cost`, `line_margin`, `total_cost`, `total_margin`) and each margin percentage is searched for as a number, in the forms it could be printed: as stored, to 2 places, without trailing zeros, with thousands separators, and for percentages both as a fraction and times 100.
- A figure may appear only as many times as the customer's own structured fields (quantities, prices, totals, dates) account for. Free text (description, claims, names) accounts for none.
- Internal field names are searched for in the text as well.
- Any hit raises `InternalFieldLeak` (finding `INTERNAL_FIELD_LEAK`). The page is not returned.

Two consequences to accept:

- If a description happens to contain a number equal to our cost ("50mm bolt", cost 50.00), the render is refused. That is the check failing closed. It names the line and the figure.
- `cost_tier_applied` is checked by name only. It is a small quantity break (1, 10, 100) that would collide with ordinary text constantly, and it is not a money figure.

**Done when:** an internal figure placed under an allowed key is caught; one placed inside a claim is caught; a clean quote whose price equals its own cost on another line still renders.

## B4. Lifecycle audit events

Written through `agent_actions` on the same connection as the status change, so the event and the change commit or fail together:

| Event | Writer | Carries |
|---|---|---|
| `sales_quote.created`, `.submitted`, `.superseded` | `record_action` | actor, quote id, ref |
| `sales_quote.approved`, `.issued`, `.outcome` | `record_action_or_fail` | actor, quote id, ref, from/to status |

`submit` and `issue` start recording their `actor`. No cost or margin goes into an event.
`floor_override` and `accepted`-by-portal events wait for B7a and the portal.

**Done when:** one query returns a quote's lifecycle in order, with who.

## B11. Gate the account read

`GET /sales/accounts/{account_id}` takes `require_user` and calls the gate with a new read action `account.read`. Under the existing rules a read with no policy row is allowed and audited, so this adds identity and a trail, not a restriction. A restriction would need an authority policy row, which is a separate decision.

## Tests

- No sell-side quote fixture exists. Proposed: two labelled synthetic fixtures under `tests/sell_side/fixtures/` (one multi-line goods quote, one with an unknown-cost line), marked synthetic in the file.
- Each new guard is broken on purpose once and shown to go red.
- Baseline first: sell-side and RGA suites run before any change, results recorded.

## Needs Nick's yes before any code

1. New tables on bp_testdb and bp_sqldb (both on the production RDS cluster), step 1 only.
2. The two synthetic fixtures.
3. This spec, including the two B1b consequences and the changed response of `GET /sales/quotes/{id}`.

## Differences from the review

- B1 as written names only `quotes.get_quote`. `_transition`, `approve`, `outcomes.record_outcome` and the router also read the internal columns.
- B11 has no existing action to gate with; `account.read` is new (B8a's territory, pulled forward by one name).
- Review B1 lists Q11 (render DB role) as a dependency. This slice does not need it: the split is in tables and queries. A denied DB role would harden it later.

## As built (2026-10-10)

Migration `deploy/sql/2026-10-10_bp_sales_quote_internal.sql` applied to bp_testdb (3 quotes
and 4 lines copied, zero differences from the old columns) and bp_sqldb (no quotes). The
seven old columns are still there, unread and unwritten.

Three things differ from the text above:

- **Internal-only justifications moved too.** `get_quote` used to return every justification,
  including ones marked not safe for a customer ("we make 29% here"). Once the quote read
  dropped to `quote.read`, those would have been readable without the margin permit. The
  customer record now carries only `customer_safe` ones; the internal record carries all.
- **A quote with no internal row cannot be approved or won.** Unknown cost passes the
  below-cost bar by design; a missing row is not the same thing and is refused.
- **Zero is not searched for** by the leak check. A bare 0 identifies nothing; a zero margin
  is caught by its field name, not its value.

Results: sell-side and sales-router suites 158 passed (129 before; 29 new). Fourteen
deliberate breaks, one per guard, each turned at least one test red. RGA suite 370 passed,
1 failed, the same one as before the change (`test_board_paper_live::test_the_deal_is_measured`).
The three real quotes on bp_testdb read through the new path with no internal key on the
customer record; the two issued ones render.

Not done here: the server still runs the old code until it is deployed and restarted. A
quote it drafts in the meantime lands in the old columns only; re-run the migration (it is
idempotent) at deploy to copy it across. Live tests leave `LIVETEST` lifecycle rows in
`bp_agent_actions`, which is append-only.
