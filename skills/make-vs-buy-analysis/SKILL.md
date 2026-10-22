---
name: make-vs-buy-analysis
description: "Compare making or keeping something in-house with buying it - the full cost of each over the same horizon (the should-cost of making, the landed and ownership cost of buying), capacity and capability, risk and control, the switching cost and the exit, and the strategic reasons that outweigh cost - with every figure from a quote, a record, the registry or the user, and a recommendation that says what would change it. Use when asked whether to make or buy, insource or outsource, or bring something in-house. Dormant pending review. Not for the business case document itself (business-case) or the category plan (category-strategy)."
---

# Make-versus-Buy Analysis

## Purpose

Give the decision a fair comparison: the make option costed as honestly as the buy option, the non-cost factors stated and weighted by the owner, and a recommendation with its flip point.

## When to use and when not

Use when the question is make or buy, insource or outsource, in-house or contracted, for a part, a service or a function.

Not for the funding case (business-case), the category strategy (category-strategy), the supplier choice once buying is decided (supplier-selection) or the cost comparison of two suppliers (total-cost-of-ownership).

## Inputs and preconditions

- The item or service, the volume and the horizon.
- For make: the bill of materials, routing, labour and overhead rates, capacity and investment, from the user or finance.
- For buy: quotes or prices on record, lead times, switching and exit costs.

- Use `tool:run_calculation` with `cost.should_cost` for the make cost from the user's bill of materials and rates, `tco.total_cost` for the buy option over the horizon, `finance.net_present_value` for the net position of each option's dated flows, and `deal.pct_change` for the gap
- Use `tool:compare_total_cost` for the buy options' landed and ownership cost on one horizon
- Use `tool:similar_items` for prices paid on record for the item (range and last date)
- Use `tool:supplier_discovery` for suppliers on record for the item
- Use `tool:search_contracts` for an agreement already covering the item
- Use `tool:get_policy` for the cost of capital, horizon and any insourcing rule
- Use `tool:check_policy` for the approval the chosen option's commitment needs
NOT AVAILABLE: simulate_scenario. no planning model exists; sensitivity is one re-run of each cost with the largest input moved by the configured share

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.sensitivity_pct`.

## Workflow

1. Frame: item, volume, horizon, what is being decided and why now.
2. Make option: should-cost from the user's bill of materials, routing and rates via the registry; investment and capacity needs as the user states them; the dated flows into the net present value.
3. Buy option: quotes or prices on record, landed and ownership cost via the comparison tool, switching and exit costs the user states; dated flows into the net present value.
4. Non-cost factors: capability, control of quality and intellectual property, supply risk, flexibility, time to switch; stated for each option, weighted only as the owner weights them.
5. Compare on one horizon at the governed rate; sensitivity with the largest input of each option moved by `config.thresholds.sensitivity_pct` percent; the flip point stated as the input that changes the answer.
6. Recommend, with the assumptions table and the approval route for the commitment.

## Frameworks and heuristics

- Cost the make option with the same rigour as the buy option; overhead forgotten is the classic bias.
- A horizon shorter than the investment's life favours buying; longer favours making; state the horizon's effect.
- Control and capability are reasons; they are stated and weighted, not hidden in the cost.
- The switching cost belongs to whichever option changes the status quo.

## Policy hooks (SC-007)

Cost of capital, horizon and any insourcing rule come from `tool:get_policy` and are quoted; the commitment's approval comes from `tool:check_policy` as replied. Nothing is decided or committed by the skill.

## Output specification

1. Frame and the status quo.
2. Cost table per option over the horizon with sources; net present value per option.
3. Non-cost factors per option with the owner's weights.
4. Sensitivity and flip point.
5. Recommendation, assumptions table, approval route.

## Quality bar

- Every cost figure has a source; the registry does the arithmetic.
- Both options are costed on the same horizon and rate.
- Non-cost factors are stated, never folded into a number without the owner's weight.
- Nothing is decided by the skill.

## Escalation and approval

Escalate to finance for rates and the investment figures, to operations for capacity, and to the category lead for the supply-risk judgement.

## Worked example

Asked whether to bring machined brackets in-house, the skill costs the make option from the user's bill of materials and rates via the should-cost calculation plus the stated machine investment, costs the buy option from the two suppliers' prices on record via the comparison tool plus the stated switching cost, puts both through the net present value at the governed rate over the policy horizon, lists capability and control factors with the owner's weights, finds buying cheaper unless volume rises by more than the configured share, and recommends buying with the flip point stated.

## Anti-patterns

- Costing make without overhead or investment.
- Different horizons for the two options.
- Hiding a control preference inside the cost.
- Deciding from the skill.
