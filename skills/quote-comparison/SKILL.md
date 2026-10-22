---
name: quote-comparison
description: Compare the quotes received for one requirement or deal on like-for-like terms and on the organisation's criteria - normalised prices through bids.normalise, scores through quote.weighting_score with the owner's weights, the suppliers' recorded profiles and screening status - and show which quote leads, by how much, and what is unconfirmed. Use for everyday three-quote decisions, "which quote is best", ranking or comparing received quotes. Not for a formal tender award memo (award-recommendation) or for judging one price against the market (price-benchmarking).
---

# Quote Comparison

## Purpose

Turn a handful of received quotes into a clear, like-for-like comparison a buyer can act on: the same lines priced on the same terms, scores on the stated criteria with the owner's weights, each supplier's recorded status, and a leader with its margin and its open points. The comparison is evidence for a decision; the decision and any approval follow the organisation's policy.

## When to use and when not

Use when the user has two or more quotes for one requirement or deal and asks which is best, cheapest on like terms, or how they rank, or wants the quotes compared before raising an order.

Not for a formal tender with published criteria and an award memo (award-recommendation), for a single price against the market (price-benchmarking), for lifecycle cost (total-cost-of-ownership) or for negotiating after the comparison (negotiation-planning).

## Inputs and preconditions

- The quotes: supplier, each line's description, quantity, unit of measure, unit price, currency, Incoterm, payment days, lead time, validity, and any stated extras; from the user or from the deal's quotes on record via `tool:get_corpus_facts` for the named supplier or deal.
- The line master: what was asked for (lines, quantities, units), from the requirement or the user.
- The criteria and weights the owner wants (price, lead time, quality, service, sustainability); without weights the comparison is on normalised price with the other criteria shown unweighted.
- Base currency, target Incoterm and payment days; exchange rates with their date from finance when currencies differ; the cost of capital from the governed `negotiation_bounds` policy via `tool:get_policy` if payment terms are to be adjusted.

Use `tool:run_calculation` with `bids.normalise` to put the quotes on the same lines, currency, Incoterm and payment days (with the user's rates and their date; without a rate a currency is not converted), with `quote.weighting_score` to score each normalised quote on total cost, lead time and volume with the owner's weights, with `supplier.deal_price_score` for each supplier's price position on the deal, and with `deal.pct_change` for the gap between quotes. Use `tool:supplier_lookup` for each supplier's recorded profile and `tool:screening_summary` for its screening status (not performed is not clear), `tool:search_contracts` for an existing agreement with a quoting supplier, and `tool:check_policy` with the leading quote's value for the approval route.

NOT AVAILABLE: build_xlsx. Give the comparison as a text table.

## Workflow

1. Gather. List the quotes with every attribute; a quote missing a line, a unit or a term is noted. Confirm the line master.
2. Normalise. Run `bids.normalise` with the line master, the quotes, the base currency, the rates and their date, the target Incoterm and payment days; report per quote the normalised total, what was adjusted, and what could not be (an unpriced Incoterm gap, an unconverted currency, a missing line under the stated missing-line policy). A quote with a missing line is compared on the lines it priced and flagged, never completed by the skill.
3. Score. With the owner's weights, run `quote.weighting_score` per quote on its normalised total, lead time and volume; without weights, rank on normalised total and show lead time and the other criteria unweighted. Run `supplier.deal_price_score` for each supplier's price position.
4. Suppliers. For each quoting supplier: recorded profile (preferred, currency, Incoterms, lead time on record), screening status from the summary, and any live contract that already covers the lines.
5. Lead and margin. Name the leading quote, the gap to the second from `deal.pct_change`, and whether the lead is inside `config.thresholds.close_call_pct` percent (a close call: the open points decide). A lead that depends on an unconfirmed item (a gap not priced, a missing line, a currency not converted) is "leading subject to" that item.
6. Open points and route. List what to confirm with each supplier before ordering (validity beyond `config.thresholds.validity_min_days` days, missing lines, terms), the screening gaps, and the approval route from `tool:check_policy` for the leading quote's value. Hand a formal award to award-recommendation and any follow-up haggle to negotiation-planning.

## Frameworks and heuristics

- Normalise before you rank: a cheaper unit price on different terms is not cheaper.
- Weights are the owner's; without them the ranking is on price and says so.
- A quote is compared on what it priced; the skill never fills a missing line with another quote's price or a last-paid price unless the stated missing-line policy says so, and then it says so.
- Screening and an existing contract are shown beside the price: a leading quote from an unscreened supplier leads subject to screening.
- Fewer than `config.thresholds.min_quotes` quotes is a comparison of what there is, with the policy on minimum quotes reported.

## Policy hooks (SC-007)

The minimum number of quotes comes from `tool:get_policy` (quoted, or 'not governed') and the approval route for the value from `tool:check_policy`, as replied. The comparison changes nothing; the order and any award carry their own approval.

## Output specification

1. Quotes gathered: supplier, lines, terms, validity; gaps noted.
2. Normalisation table: per quote, normalised total in base currency, adjustments made, items not normalised.
3. Scores: weights used (or "price only"), score per quote, price position per supplier.
4. Supplier table: profile facts, screening status, existing contract.
5. Lead, gap to second, close-call flag, "subject to" items.
6. Open points per supplier, approval route, hand-offs.

## Quality bar

- Every normalised figure and score comes from a calculation; the model computes no totals or conversions.
- Every "subject to" item is named; no quote is completed by the skill.
- Screening status is shown for every supplier; not performed is never clear.
- Weights are stated with their owner, or the ranking is declared price-only.

## Escalation and approval

Escalate to the category lead for a close call or a lead subject to an unpriced gap, and to compliance for a leading supplier with a flagged or unperformed screening. Ordering and award follow the approval route the policy reply states.

## Worked example

Given three quotes for warehouse racking (two in GBP DAP, one in EUR EXW), the skill normalises them with finance's dated rate to GBP DAP and sixty days (the EXW quote's freight is unpriced: not normalised, flagged), scores them on the owner's weights, finds the EUR quote leading on price subject to freight, with the second GBP quote inside the close-call band, shows that the second supplier is preferred and screened while the leader has no screening recorded, and lists the open points: freight for the leader, validity for the third, screening for the leader, and the Approver route for the value.

## Anti-patterns

- Ranking on quoted unit price across different currencies, Incoterms or payment terms.
- Filling a missing line or a freight gap with a guess.
- Choosing weights for the owner.
- Declaring a winner that depends on an unconfirmed item without saying so.
- Treating the comparison as the award.
