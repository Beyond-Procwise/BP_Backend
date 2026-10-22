---
name: price-benchmarking
description: Judge whether a quoted or paid price is in line with evidence - what we paid for the same or near-identical items on our own lines, the price points and indices the user supplies, the registered benchmark and outlier calculations with their confidence, and a loaded commodity index - and say when there is not enough evidence to judge. Use when asked whether a price is fair, high or competitive, whether we are overpaying, for a price check or benchmark, or how a quote compares with what we paid before. Not for setting negotiation targets (negotiation-planning) or for lifecycle cost (total-cost-of-ownership).
---

# Price Benchmarking

## Purpose

Answer "is this price reasonable" with evidence and a confidence, never with an opinion: our own paid prices for the same or near-identical item, the external points the user supplies, index movement where a series is loaded, and the registered benchmark and outlier calculations. When the evidence is too thin to judge, the answer is "not enough evidence", with what would be enough.

## When to use and when not

Use when the user asks whether a quote or paid price is fair, high, low or competitive, for a price check or benchmark, whether we are overpaying a supplier, or how a price compares with our history or an index.

Not for setting targets or walk-away points (negotiation-planning, which takes this benchmark as input), for lifecycle cost (total-cost-of-ownership), for validating a saving (savings-methodology), or for designing an index-linked formula (pricing-mechanisms).

## Inputs and preconditions

- The price under test: item description or item code, unit price, currency, unit of measure, quantity, supplier, Incoterm and payment terms, date.
- External evidence if the user has it: price points with their source, date, quantity and terms; an index code where the price is index-driven.
- The comparison basis the user wants: our own history, the user's points, an index, or all three.

Use `tool:similar_items` to find the same and near-identical descriptions on our PO, invoice and quote lines with their price ranges as billed, supplier and buyer counts; `tool:get_corpus_facts` for a named supplier's documents; `tool:commodity_index_lookup` for a loaded index and its move between two dates; `tool:run_calculation` with `benchmark.adjusted_price` to adjust the user's historical points to the quote's basis, with `benchmark.evidence_confidence` to state how much evidence stands behind the result against the minimum, with `price_outlier.verdict` to test the price against comparable peer prices, and with `deal.pct_change` for the gap between the price and the reference.

NOT AVAILABLE: build_xlsx. Give the evidence table as a text table.
NOT AVAILABLE: render_chart. Describe the price history as a table.

## Workflow

1. Frame. State the price under test with every attribute given; ask for any missing one that changes comparability (currency, unit of measure, Incoterm, quantity). The benchmark is on like-for-like terms or it is not a benchmark.
2. Own history. Run `tool:similar_items` on the description (and code); keep only matches at or above `config.thresholds.similarity_min` and, for a strict comparison, only those with the same item code or a confirmed specification; report each match's price range per currency, line count, suppliers and buyers, and last date. Ranges in another currency or on other terms are shown, not compared.
3. Peers. Build the peer set: own prices on the same terms and currency, plus the user's points adjusted with `benchmark.adjusted_price` (with the user's index and location tables where given). Fewer than `config.thresholds.min_peer_prices` peers means no verdict; say what is missing.
4. Verdict. Run `price_outlier.verdict` on the price against the peers and `benchmark.evidence_confidence` on the peer count; report the verdict, the reference price, the gap from `deal.pct_change`, the confidence and the peer count. A gap inside `config.thresholds.in_line_band_pct` percent is "in line"; outside it, "above" or "below" with the evidence.
5. Index. Where the item is index-driven and a series is loaded, read the index move between the date of the reference prices and the quote date; say how much of the gap the move explains, from the calculation's figures only; no series means "index not available".
6. Caveats. State what the comparison does not cover: specification differences, service levels, volume, terms, age of evidence older than `config.thresholds.evidence_max_age_days` days.
7. Hand-off. A price judged above the evidence goes to negotiation-planning with the benchmark; a saving claim goes to savings-methodology.

## Frameworks and heuristics

- Like-for-like or nothing: a lower price on different terms, units or specification is not a benchmark until adjusted.
- Evidence count before verdict: one old invoice is not a benchmark; the confidence calculation decides what is enough.
- Own history is the strongest evidence we hold; the user's external points are next, with their source; the model's memory is never evidence.
- Index explains movement, not level: a rise in the index justifies part of an increase, not the starting price.
- Ranges, not points: when our history spans a range, the reference is the calculation's, not the minimum.

## Policy hooks (SC-007)

No policy governs a benchmark itself. Where a benchmark feeds an award, a renewal or a price acceptance, the deciding skill checks `tool:check_policy`; the price-tolerance rules a policy states for those decisions are read with `tool:get_policy` and reported, not applied here as a verdict.

## Output specification

1. The price under test with its attributes and what was missing.
2. Own history table: description, item code, similarity, price range per currency, lines, suppliers, buyers, last date, comparable (yes/no and why).
3. Peer set and adjustments made; peer count; confidence.
4. Verdict: in line / above / below / not enough evidence, reference price, gap, confidence.
5. Index move and what it explains, or "index not available".
6. Caveats and hand-offs.

## Quality bar

- Every price in the evidence table comes from a tool result or the user's points with a source; none from memory.
- No verdict below the minimum peer count; the confidence calculation is reported with every verdict.
- Terms, units and currency differences are stated for every comparison.
- The gap and the index effect come from the calculations, never from the model's arithmetic.

## Escalation and approval

Escalate to the category lead when the verdict is "above" on a price about to be committed, and to finance when exchange rates or index series are missing for an index-driven item. The benchmark is advice; the decision it feeds carries its own approval.

## Worked example

Asked whether a quote for laptops is fair, the skill finds the same item code on our purchase-order lines from several suppliers in two currencies, keeps the GBP lines on the same terms as peers, adds the user's two external points adjusted to the quote's basis, finds enough peers for a verdict, runs the outlier calculation (above the reference by a stated gap) with its confidence, reads the loaded index (no series for laptops: not available), lists the caveats (volume and warranty differ) and hands the result to negotiation-planning.

## Anti-patterns

- Calling a price fair or high from memory of market prices.
- Comparing across currencies, units or Incoterms without adjustment.
- Giving a verdict on one or two data points.
- Using the index to justify the price level rather than a movement.
- Turning a benchmark into a target without negotiation-planning.
