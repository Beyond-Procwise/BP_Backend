---
name: demand-aggregation
description: Find where the same or substitutable demand is bought separately across business units, sites or legal entities, size the value of consolidating volume or aligning prices, and sequence the opportunities. Use when asked where we can leverage scale, reduce supplier fragmentation or align prices paid for the same item. Advisory only. Not for a full category strategy (use category strategy) or for one-off tail-spend clean-up.
---

# Demand Aggregation

## Purpose

Show, from the user's own spend data, where the same or near-identical item is bought by several units at different prices or from many suppliers, what consolidating it might be worth, and in what order to pursue it. Every figure carries its method and its data, or it is not given.

## When to use and when not

Use for volume consolidation, supplier fragmentation and price harmonisation across business units, sites, legal entities or suppliers.

Not for building a whole category strategy, not for clean-up of one-off tail purchases, not for validating a saving someone already claims (use savings-methodology), and not for negotiation tactics (use negotiation-planning).

## Inputs and preconditions

- Scope: categories, entities and period. If the user gives no period, use the trailing `config.thresholds.default_period_months` months and say so.
- Spend lines with item or description, unit of measure, quantity, unit price, currency, supplier, business unit, site or country, Incoterm, payment terms, and whether the line was on contract.
- Reporting currency. Mixed currencies without a stated rate are not converted.
- If the user's message already gives prices, quantities and buyers (for example three plants with a price and a yearly volume each), those are the data: work from them and do not search the corpus for an item the user did not name. Never make up an item code, description or unit of measure; call the item what the user called it. A handful of lines is enough for a first view; say it is only those lines.
- Use `tool:get_corpus_facts` to read what the user's own documents hold for a supplier or item. It returns facts from stored documents, not a full spend cube: it cannot give complete category totals, BU splits or classification codes. Say what it could not give and ask for an extract.

Use `tool:build_spend_cube` to cut invoiced spend by supplier, month, currency, country, region, buyer, payment terms or status for a period; its totals equal the dashboards. Invoices carry no category, no business unit and no price-per-item, so it cannot give category totals or item prices: say so, and use `tool:get_corpus_facts` for item prices or ask the user for an extract.
Use `tool:classify_spend` with the item descriptions: a 'recorded' item carries the organisation's own product-to-category mapping; an 'unclassified' item has none, and its candidates are name matches only. Treat candidates and any labels the user gives as "draft classification", never as fact, and state the tool's coverage (recorded share) against `config.thresholds.min_classified_coverage_pct`. Use `query` to look up the taxonomy's levels, UNSPSC codes and owners.
Use `tool:search_contracts` to find contracts by supplier, text or expiry window, and `tool:extract_clauses` for the stored clauses (minimum commitments, exclusivity). It covers only contracts that were uploaded; if it finds none, or cannot be read, ask the user for end dates, lock-ins and minimum commitments and write "contract position not available" for what stays unknown.
NOT AVAILABLE: landed_cost_calc. Use `tool:run_calculation` with `landed_cost.total_cost` when the user supplies freight, duty and other cost inputs; otherwise say logistics cost is not normalised.
NOT AVAILABLE: build_xlsx. Give the opportunity register as a table.
NOT AVAILABLE: build_pptx. Give the wave plan as text.
NOT AVAILABLE: render_chart. Describe the heatmap as a category by business unit table and the spread as a small table of low, reference and high prices.
Use `tool:similar_items` with an item description to find the near-identical descriptions on this organisation's PO, invoice and quote lines, with their line, supplier and buyer counts and price ranges as billed. Its name similarity is not identity: treat a match as the same item only at or above `config.thresholds.min_match_confidence` AND after the specification and unit of measure are confirmed (or the item codes are the same); list every match with its evidence for review.

## Workflow

1. Scope. Fix categories, entities, period and reporting currency. Compute the classified share of spend in scope. If it is below `config.thresholds.min_classified_coverage_pct` percent, say so, continue only on codes the user accepts or on lines labelled "draft classification", and show which opportunities depend on those lines.
2. Fragmentation. Per L4 category, from the data supplied: supplier count, spend per supplier, concentration (HHI), number of business units buying, and the share of spend off contract. A candidate needs at least `config.thresholds.min_bus_buying` business units, at least `config.thresholds.min_suppliers` suppliers and combined spend of at least `config.thresholds.min_combined_spend`. These are configurable defaults, not policy. Check that misclassified lines are not inflating the candidate; show it with and without them.
3. Price harmonisation. Only for identical items: the same item code, or a description, specification and unit of measure match at confidence of at least `config.thresholds.min_match_confidence`. Show the evidence for every match. With fewer than `config.thresholds.min_price_observations` prices, give no spread. Normalise first for Incoterm, payment terms and order size; a cheaper price under a different Incoterm is not a saving until landed cost is compared (see incoterms-selection and `landed_cost.total_cost`). Reference price is the `config.thresholds.reference_percentile`th percentile price, not the minimum. Work out the reference price and the saving with `tool:run_calculation` and `demand.price_harmonisation`: pass the purchase lines (id, price, quantity) and `config.thresholds.reference_percentile` as the percentile. It returns the reference price, each line's saving and the total; quote those, and do not work them out by hand. If the user supplies their own reference price, use it and say so. If it refuses, say why and ask for what it needs. Use `tool:run_calculation` with `price_outlier.verdict` on a price that looks wrong, so one bad record does not set the spread.
4. Volume leverage. Value consolidated volume only from tier prices or supplier quotes the user supplies, cited. With none, give a low to high range from the user's own historical tier discounts and label it an estimate; with no history either, give no figure. Never count price harmonisation and volume leverage on the same volume: assign each unit of volume to one method and show the split.
5. Spec harmonisation. Cluster near-identical items and list the specification differences for engineering or quality review. Do not assume they are interchangeable, and do not count their saving in the base case.
6. Feasibility. Check contract lock-ins and expiry, minimum commitments, logistics distance and lead times, local regulatory or quality requirements, and entity tax or legal constraints. Score feasibility from `config.thresholds.feasibility_min` to `config.thresholds.feasibility_max` with a one-line reason; an unchecked factor is named "not checked" and lowers confidence, not the score silently.
7. Prioritise. Rank by value times feasibility, with time to value. Sequence waves around contract expiries; an opportunity locked in until a stated date waits for it.
8. Policy hook. Call `tool:check_policy` for competitive-bid rules triggered by the new aggregated value and for preferred-supplier rules; see `procurement-policy-compliance`. Name who must approve.

## Frameworks and heuristics

- HHI for concentration; Pareto (80/20) by spend to focus effort.
- Price-spread analysis with interquartile range, reference at a percentile rather than the minimum.
- Landed-cost normalisation before any price comparison.
- One volume, one method: no double counting.
- Low, base and high cases from stated assumptions, never a single point without data.

## Policy hooks (SC-007)

Competitive-bid thresholds, preferred-supplier lists and cross-entity purchasing rules come from governed policy via `tool:check_policy` and `tool:get_policy`. If no row governs, say it is not governed. The thresholds in config are this skill's configurable defaults for screening, not policy.

## Output specification

An opportunity register as a table: category, business units, suppliers, spend (with currency and period), saving low, base and high, method, evidence, feasibility with reason, next step and suggested owner. Then a category by business unit view, a price-spread table for each harmonisation item, and a wave plan tied to contract expiries. State currency, period, data gaps, and which figures are the model's own arithmetic. A request for one savings number without data gets the range and caveats, or "not available".

## Quality bar

- Every saving shows its method and its data.
- Identical-item matches show their evidence.
- No volume is counted twice.
- Currency and period are stated.
- Absent data is "not available", never guessed.

## Escalation and approval

Advisory: nothing is awarded, tendered, signed or changed. Flag opportunities that cross legal entities, that need a competitive tender above the governed limit, or that depend on a specification change, and name who must approve per `procurement-policy-compliance`. Escalate if asked to publish a saving that has no supporting data.

## Worked example

The user says three plants buy the same bearing, item code B-100, in GBP over the last quarter, and gives these purchases: Plant A 1,000 units at 10.00, Plant B 500 at 13.00, Plant C 500 at 10.00. All are on the same Incoterm. They ask what harmonising is worth, with a reference price of 10.00 that they say is their own 25th percentile. Plant B alone is above reference: (13.00 minus 10.00) times 500 = 1,500 GBP, the model's own arithmetic from the figures supplied, for one quarter and one item. Nothing is stated for volume leverage, so none is given. Next step: confirm Plant B's contract position; the user supplied none, so it is "contract position not available".

## Anti-patterns

- Using the minimum price as the reference.
- Comparing prices under different Incoterms or payment terms without normalising.
- Counting harmonisation and volume leverage on the same volume.
- Treating near-identical items as interchangeable without review.
- Giving a savings total with no data behind it.
- Inflating an opportunity with misclassified spend.
