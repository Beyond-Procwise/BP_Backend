---
name: total-cost-of-ownership
description: Compare options on what they really cost over their life - landed cost to our door, then acquisition, running, maintenance, financing and disposal over a horizon at the governed cost of capital - and, where a part is manufactured, a should-cost from a bill of materials, with every rate named with its source and every unstated cost reported as missing rather than zero. Use when asked for a TCO, lifecycle or whole-life cost, a cost model or should-cost, or which option is cheaper over time. Not for choosing the award (award-recommendation) or the Incoterm itself (incoterms-selection).
---

# Total Cost of Ownership

## Purpose

Answer "which option costs less over its life, and by how much" with the registered calculations and only the figures the user, finance or governed policy supplied: landed cost first, then total cost of ownership over the horizon, and a should-cost where the user has a bill of materials. The result is a comparison with its assumptions in the open, not a price.

## When to use and when not

Use when the user wants to compare quotes or options on total cost rather than price, asks for a TCO, lifecycle, whole-life or landed-cost comparison, a cost model, a should-cost or a cost breakdown of a quote.

Not for the award decision (award-recommendation, which takes this comparison as input), for choosing the delivery term (incoterms-selection, which compares landed cost by rule), for negotiation targets (negotiation-planning), or for validating a saving (savings-methodology).

## Inputs and preconditions

- The options: each quote's price lines (unit price, quantity, currency), Incoterm and named place, payment terms, and the stated extra charges (freight, insurance, duty, brokerage, installation, training, licences, support, consumables, energy, disposal).
- The horizon in years and the annual volumes or usage; the cost of capital from governed policy via `tool:get_policy` (the negotiation bounds policy states one) or from finance; exchange rates with their date from finance when currencies differ.
- Where a should-cost is wanted: the bill of materials with material prices and their source, the routing with cycle times and machine or labour rates, batch size and annual volume, and the overhead, SG&A and margin shares from finance.

For a comparison of options use `tool:compare_total_cost` ONCE: it runs landed cost and then ownership cost for every option through the registered calculations and ranks them; pass each option's price lines, Incoterm, `cross_border`, stated components, payment terms and its lifetime items (maintenance, support, consumables, energy, disposal go in `lifetime_items` with a year or day, never in the lines), plus `horizon_years` and `cost_of_capital`. Use `tool:run_calculation` directly only for a single option or a should-cost, with these inputs: `landed_cost.total_cost` takes `lines` (unit_price, quantity, currency), `buyer_currency`, `cross_border`, `incoterm` (rule and place), `components` (key, amount, currency, or status included / not_applicable), optional `duty_rate` with `customs_basis`, `default_insurance_rate`, `fx_rates` with `fx_table_date`; `tco.total_cost` takes `landed` (the whole result of landed_cost.total_cost), `cost_of_capital`, `horizon_years`, `payment_terms` (net_days, discount_pct, discount_days), `ap_can_take_discount`, `lifetime_items` (key, amount, year or day, direction); `cost.should_cost` takes `bom`, `routing`, `batch_size`, `overhead_pct`, `sga_pct`, `margin_pct`, `logistics_per_unit`, `currency`, `annual_volume`. The cost of capital is read ONCE with `tool:get_policy` using the query `negotiation_bounds` (rule `cost_of_capital_apr`); if that policy does not state it and finance has not given one, STOP and ask: never assume a rate and never compute a present value by hand. Use `tool:list_calculations` only if a call is refused for its inputs; `bids.normalise` where bids must first be put on the same terms; `deal.pct_change` for the difference between options. Use `tool:get_policy` for the cost of capital and any governed tolerance, `tool:commodity_index_lookup` for a loaded material index, `tool:similar_items` and `tool:get_corpus_facts` for what we paid before as context (never as a quote), and `tool:check_policy` when the comparison feeds a commitment.

NOT AVAILABLE: get_freight_quote. Freight and insurance figures come from the user's quotes or forwarder; a missing one is reported as missing, not estimated.
NOT AVAILABLE: duty_calc. The duty rate comes from the user or customs broker with its source; without it duty is missing.
NOT AVAILABLE: hs_code_lookup. Ask for the HS code or the duty rate; do not infer either.
NOT AVAILABLE: build_xlsx. Give the comparison and sensitivity as text tables.

## Workflow

1. Frame. List the options, the horizon, the volumes, the currency the answer is in and where each rate comes from; stop and ask for anything missing from the inputs above. The number of options compared is at most `config.thresholds.max_options` in one pass.
2. Same terms. If the quotes differ in currency, Incoterm or payment days, run `bids.normalise` first with the user's rates and their date; report what it could not normalise.
3. Landed cost. Through `tool:compare_total_cost` (or, for one option, `landed_cost.total_cost`) with the lines, the Incoterm, the stated components, the duty rate and customs basis from the user, and the insurance rate from policy or the user. If the origin is not stated, ask whether the goods are imported; a delivered quote (DAP, DDP) from a domestic supplier has no import components (`cross_border` false). Report each component's status (stated, included, derived, estimated, missing, not stated); a total with anything missing is a lower bound and is labelled so. A component missing for EVERY option alike (for example duty with the rate unknown on all of them) does not change the ranking: continue, rank on the lower bounds, say the totals are lower bounds, and ask for the figure in the open items rather than stopping.
4. Ownership cost. From the same `tool:compare_total_cost` call (or `tco.total_cost` for one option), read the present value per option with the same horizon, cost of capital, payment terms and lifetime items (maintenance, support, consumables, energy, training, disposal) for every option; an item one quote states and another omits is "not stated" for the second, never zero. Report the present value per option and the difference with `deal.pct_change`.
5. Should-cost, where a BOM exists. Run `cost.should_cost` with the user's BOM, routing, rates and finance's shares; label the result a modelled estimate; compare it with the quote breakdown and name the drivers that differ by more than `config.thresholds.driver_gap_flag_pct` percent.
6. Sensitivity. Re-run the comparison with the cost of capital, volume and the largest variable component moved by plus and minus `config.thresholds.sensitivity_shock_pct` percent; say whether the ranking changes and at what point.
7. Verdict. Rank the options by present value with the status of each total (complete or lower bound), the assumptions table, and what would change the answer. Hand the award to award-recommendation and the delivery-term choice to incoterms-selection.

## Frameworks and heuristics

- Landed before lifetime: nothing is compared over years until the delivered cost is on the same terms.
- Unstated is not zero: a quote silent on support, consumables or disposal gets "not stated" and the total becomes a lower bound.
- Rates have owners: cost of capital from policy or finance, exchange rates dated from finance, duty from the broker, insurance from policy; the model never fills one in.
- Present value, one horizon, one cost of capital for every option; a longer-lived option is compared over the same horizon with its residual stated by the user or left as not stated.
- A should-cost is a modelled estimate of what it ought to cost, never evidence of a supplier's actual cost.

## Policy hooks (SC-007)

The cost of capital and the default insurance rate come from governed policy via `tool:get_policy` where stated; where not, they come from finance or the user and are labelled so. A comparison that feeds a commitment is checked with `tool:check_policy` by the deciding skill. No tolerance or rate in config is policy.

## Output specification

1. Frame: options, horizon, volumes, currency, source of each rate.
2. Like-for-like step: what was normalised and what could not be.
3. Landed cost table per option: components with status, total, status (complete or lower bound).
4. Ownership table per option: lifetime items with status, present value, difference between options.
5. Should-cost (when run): modelled cost per unit, drivers, gaps against the quote breakdown, labelled estimate.
6. Sensitivity: what moves the ranking and at what point.
7. Verdict with assumptions and open items; hand-offs.

## Quality bar

- Every figure comes from a calculation result or the user's inputs; no freight, duty, insurance, rate or horizon is assumed.
- Every total carries its status; a lower bound is never presented as a total.
- Options are compared on the same horizon, cost of capital, currency and terms.
- The should-cost is labelled a modelled estimate every time it is stated.

## Escalation and approval

Escalate to finance when the cost of capital, exchange rates or residual values are missing or disputed, and to the category lead when a lower-bound total decides the ranking. The comparison is advice; the award and any commitment go through their own skills and approvals.

## Worked example

Asked which of two forklift quotes is cheaper over five years, the skill reads the cost of capital from policy, normalises the EUR quote with finance's dated rate, runs landed cost for both (one quote states delivery, the other is silent: missing, lower bound), runs the ownership calculation with the user's maintenance contracts, energy use and disposal values over five years, finds the dearer-to-buy option cheaper in present value by a stated percentage, shows that the ranking flips if energy price moves by less than the configured shock, and reports the second quote's total as a lower bound pending its delivery charge.

## Anti-patterns

- Treating a charge the quote does not mention as zero.
- Estimating freight, duty, insurance or a cost of capital.
- Comparing options over different horizons or in different currencies.
- Presenting a should-cost as the supplier's cost or as a price target.
- Calling a lower-bound total "the total".
