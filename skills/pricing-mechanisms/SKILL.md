---
name: pricing-mechanisms
description: Design or check a pricing mechanism for a contract - index-linked price adjustment formulas, review cadence, caps, floors and collars, rebates, volume tiers and payment-linked discounts. Use for volatile-cost categories, long-term supply agreements and renewals. Review-gated, so the category lead and finance approve before any mechanism is used. Not for negotiation strategy (use negotiation-planning), one-off should-cost, or savings claims.
---

# Pricing Mechanisms

## Purpose

Design contract pricing structures that share market risk fairly between buyer and supplier and can be administered without disputes: an index-linked formula, a dead band, a collar, volume tiers and rebates, each traceable to a named source and reproducible from the stated inputs.

## When to use and when not

Use when a contract needs a price adjustment formula, a cap or collar, a rebate or volume tier, or when an existing mechanism is to be checked for fairness and administrability. Typical for volatile-cost categories, long-term supply agreements and renewals.

Not for negotiation strategy or targets (use negotiation-planning), for a one-off should-cost (use `tool:run_calculation` with `cost.should_cost` directly), or for validating a saving (use savings-methodology).

## Inputs and preconditions

Ask for what is missing; do not assume it.

- Annual spend and a volume forecast, the contract term and the current price.
- The supplier's cost breakdown: material, energy, labour, fixed overhead and margin shares. A user-supplied breakdown, or a should-cost from `tool:run_calculation` with `cost.should_cost`. With no breakdown, ask for one; if the user cannot give it, label every weight as an estimate and say the formula is provisional.
- Currency of the price and currency of each candidate index.
- Candidate indices with their codes, publisher and historical values, supplied by the user or the contract.
- Supplier market power. A Kraljic quadrant can come from `tool:run_calculation` with `negotiation.kraljic_quadrant`, or from the user.
- Who in the business can calculate and check the adjustment each period.

NOT AVAILABLE: commodity_index_lookup. There is no index lookup. Ask the user for the index code, source and series; without a series say "index history not available" and do not estimate correlation, base values or any simulation.
NOT AVAILABLE: build_spend_cube. Use `tool:get_corpus_facts` for invoiced spend and volumes, or ask the user.
NOT AVAILABLE: build_docx. Give the recommendation and clause text as plain text.
NOT AVAILABLE: build_xlsx. Give the sensitivity table as a text table.
NOT AVAILABLE: render_chart. Describe the price paths in a table; do not claim a chart was made.
NOT AVAILABLE: obligation_tracker. Give the administration calendar as a table; nothing is registered or scheduled.

## Workflow

1. Gather the inputs above. Compute the price share of each cost driver from the breakdown. Without a breakdown, stop and ask, or label as estimate.
2. Choose the indices. Every cost driver at or above `config.thresholds.material_driver_min_share_pct` percent of price needs a published, independent index that the supplier does not control. Check the index against the supplier's cost history; the target correlation is at least `config.thresholds.index_min_correlation`, but only compute it from a series the user supplies. Record index code, source, publisher licence needs and a fallback index if the first is discontinued. Keep to at most `config.thresholds.max_indices` indices.
3. Write the formula: P_t = P_0 x [a + sum of b_i x (I_i,t / I_i,0)], where a is the fixed, non-indexed share and a plus all b_i equals one. Weights come from the cost breakdown, never from preference. Fix the base period, the averaging window, the lag between the index month and the price month, FX treatment, rounding and review frequency. Check that the formula returns P_0 when every index equals its base value; use `tool:run_calculation` with `deal.pct_change` for percentage moves.
4. Share the risk. Set a dead band between `config.thresholds.dead_band_min_pct` and `config.thresholds.dead_band_max_pct` percent inside which the price does not move, a supplier sharing ratio of `config.thresholds.sharing_ratio_supplier_pct` percent beyond the band, an annual collar of `config.thresholds.collar_annual_pct` percent, and a re-opener when an index moves by `config.thresholds.reopener_index_move_pct` percent. Make adjustment symmetric: the price falls when the index falls. Simulate the user's index history over `config.thresholds.history_months_min` to `config.thresholds.history_months_max` months and a stress case of plus and minus `config.thresholds.stress_shock_pct` percent; show the price path and the cost to each party against a fixed price. If there is no series, say the simulation is not run.
5. Volume tiers and rebates. Choose retrospective (all-units) or incremental tiers. Compute the marginal price at each threshold and reject any structure where buying more costs more in total or where the marginal price is negative (a cliff). Set thresholds against the forecast P10, P50 and P90 so they are achievable but meaningful. Define rebate base, accrual, settlement timing and claw-back. Payment-linked discounts: value with `tool:run_calculation` with `tco.total_cost` only if finance gives the cost of capital.
6. Check against policy. Allowed index-linking, rebate accounting (accrual) and approval for uncapped indexation come from governed policy; use `tool:get_policy` and `tool:check_policy`, and confirm who approves with `procurement-policy-compliance`.
7. Write the clause text: definitions, formula, worked example, data sources, fallback if publication fails, dispute and audit rights. Hand it to the contract drafting owner as a draft; it is not signed or sent.

## Frameworks and heuristics

- Index only what the supplier cannot control. An index the supplier influences is a conflict.
- The fixed share should be at least the margin plus overhead share.
- Symmetric adjustment, up and down; one-way adjustment is not risk sharing.
- Fewer indices is better; each one adds a monthly calculation.
- Kraljic: a leverage position supports tighter bands; a bottleneck position supports a longer term with collars for supply security.
- Tiers must be monotone: more volume never costs more in total.

## Policy hooks (SC-007)

Allowed index-linking, the accounting treatment of rebates and approval to accept uncapped indexation come from governed policy. If a policy row is missing, say it is not governed. The thresholds in config are this skill's configurable defaults, not policy.

## Output specification

A recommended structure with reasons; the formula with weights, base values and a worked example; the index table (code, source, fallback, correlation if supplied); the simulation and sensitivity table (or "not run" with the reason); tier and rebate table with the marginal price at each threshold; clause text; an administration calendar (which index, when, who calculates). Every input is labelled as given, calculated or estimate. Status is a draft for review.

## Quality bar

- Weights sum to one and the formula reproduces the current price at base.
- Every index has a code, source and date, supplied by the user or contract.
- The simulation can be reproduced from the stated series and rules.
- Tier marginal prices are not negative.
- No invented index values, rates, caps or prices.

## Escalation and approval

Review-gated: the named category lead and finance approve before the mechanism is used. Escalate when no reliable index exists, when the supplier proposes an index they influence, or when uncapped exposure exceeds `config.thresholds.uncapped_exposure_escalate_pct` percent of annual spend. Never present a mechanism as approved or agreed.

## Worked example

A buyer supplies: current price 100 per unit; cost breakdown of 50 percent aluminium, 20 percent energy, 10 percent labour and 20 percent fixed. Base index values: aluminium 2,000, energy 80, labour 100. The formula is P_t = 100 x [0.20 + 0.50 x (Al_t / 2,000) + 0.20 x (En_t / 80) + 0.10 x (Lab_t / 100)]. Weights sum to 1.00. At base values the bracket is 0.20 + 0.50 + 0.20 + 0.10 = 1.00, so the price is 100, which reproduces the current price. If aluminium reaches 2,400 and the others stay at base, the bracket is 0.20 + 0.60 + 0.20 + 0.10 = 1.10 and the price is 110 before any dead band or collar. The user gave no index series, so correlation and the 24 to 36 month simulation are reported as not run, and the user is asked for the series and the index codes.

## Anti-patterns

- Weights chosen by preference rather than from the cost breakdown.
- An index the supplier influences, or an unpublished one.
- One-way (upward only) adjustment presented as risk sharing.
- A volume tier cliff that makes a higher volume cost more in total.
- Inventing index values, correlations or simulation results.
- Presenting the mechanism as approved or agreed.
