---
name: business-case
description: "Build the business case for a purchase or a procurement project - the need and the options including doing nothing, the costs over the horizon from quotes and records, the benefits with their basis and owner, the payback and the net position at the governed cost of capital, the risks, the assumptions listed, and the approval the value needs. Use when asked to justify a purchase, write an investment or capex case, or show the return on a procurement initiative. Not for comparing options on cost alone (total-cost-of-ownership) or for savings validation (savings-methodology)."
---

# Business Case

## Purpose

Give the approver a case they can test: options including the status quo, costs traced to quotes and records, benefits with an owner who will stand behind them, the arithmetic done by the registry at the governed rate, and every assumption written down. The skill drafts the case; the sponsor owns the benefits and the approver decides.

## When to use and when not

Use when the request is to justify spending, write a business or investment case, show payback or return, or compare the case for an initiative against doing nothing.

Not for the cost comparison of options alone (total-cost-of-ownership), for claiming savings after the fact (savings-methodology), for the award of a tender (award-recommendation) or for the finance template's formatting.

## Inputs and preconditions

- The need, the options the sponsor is considering, the horizon, and the quotes or estimates with sources.
- The benefits the sponsor claims, with the basis and the owner of each.
- Constraints: budget, timing, policy.

- Use `tool:get_policy` for the cost of capital, the horizon convention, the approval bands and the business-case rules, quoted
- Use `tool:similar_items` for the descriptions we bought under before (line counts and a price range per source; it names no supplier and gives no dated price)
- Use `tool:compare_total_cost` for the lifetime cost of each option on one horizon
- Use `tool:run_calculation` with `tco.total_cost` for present-value cost over the horizon at the governed rate, `deal.pct_change` for the difference between options and `deal.scaled_total` for a benefit over the horizon from an annual figure the owner states; payback and net position are not registry calculations, so they are shown with their inputs and labelled unverified
- Use `tool:build_spend_cube` for the current spend the case would change
- Use `tool:calc_supplier_scorecard` for the incumbent's record where the case is a supplier change
- Use `tool:check_policy` for the approval route the value needs
NOT AVAILABLE: build_docx. the case is given as structured text for the finance template
NOT AVAILABLE: build_chart. figures are given as tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.sensitivity_pct`.

## Workflow

1. Frame: need, options including doing nothing, horizon, sponsor, constraints.
2. Policy: the cost of capital and horizon convention quoted (none means ask finance; never assume a rate); the approval band for the value.
3. Costs per option over the horizon: one-off and recurring from quotes, prices paid and the cube; items not stated are not zero and are listed; lifetime cost via the comparison tool or the registry at the governed rate.
4. Benefits per option: each with its basis (a figure from the cube, a contract price, a headcount the sponsor states), its owner, and its status (contracted, evidenced, estimated); a benefit over the horizon via `deal.scaled_total` from the annual figure the owner states. A benefit with no owner is listed as unowned and excluded from the net position.
5. Net position per option: the registry's present-value cost set against the owned benefits over the horizon; a net or payback figure is not a registry calculation, so it is shown with its inputs and labelled unverified; sensitivity at `config.thresholds.sensitivity_pct` on the largest benefit and the largest cost, shown as the inputs that change; the option that wins and under what assumption it stops winning.
6. Risks with owners; assumptions listed in one place; the approval route as replied; the recommendation and what the approver is asked to decide.

## Frameworks and heuristics

- Doing nothing is an option with a cost; the case is relative to it.
- A benefit has an owner or it is a hope.
- Costs come from quotes and records; a cost nobody quoted is a gap, not a zero.
- The rate and the horizon come from policy; the cost arithmetic comes from the registry, and a net figure says it is unverified.
- One table of assumptions beats ten footnotes.

## Policy hooks (SC-007)

Cost of capital, horizon, business-case rules and approval bands come from `tool:get_policy` and are quoted; the route for the value comes from `tool:check_policy` as replied. The skill commits nothing and approves nothing; benefits are the sponsor's claims with owners, and the savings-methodology rules apply when they are later claimed as savings.

## Output specification

1. Frame: need, options, horizon, sponsor, constraints.
2. Policy inputs quoted: rate, horizon, bands.
3. Cost table per option with sources and the gaps.
4. Benefit table per option with basis, owner and status; unowned benefits excluded.
5. Present-value cost per option via the registry; benefits over the horizon; net and payback labelled unverified with their inputs; sensitivity.
6. Risks, assumptions, approval route, recommendation and the decision asked for.

## Quality bar

- Every cost and benefit traces to a quote, a record or a named owner's statement.
- Rate and horizon are quoted from policy or the case says they are missing.
- Cost arithmetic is from the registry or the comparison tool; any net or payback figure is labelled unverified with its inputs.
- Doing nothing is costed.

## Escalation and approval

Escalate to finance for the rate, the horizon and the template, to the sponsor for unowned benefits, and to the approver the policy names for the decision.

## Worked example

Asked to justify replacing the forklift fleet, the skill frames three options (keep and maintain, replace with the lower quote, replace with the dearer quote), reads the governed cost of capital and horizon, costs each option from the quotes, the maintenance spend in the cube and the sponsor's stated energy figures, lists the benefits with owners (maintenance spend reduction evidenced from the cube, downtime reduction estimated by operations), computes present-value cost per option via the registry, sets the owned benefits against it with the net and payback labelled unverified, runs sensitivity on the downtime benefit, finds the lower quote wins unless the downtime benefit is less than half the estimate, reports the approval route, and lists the assumptions in one table.

## Anti-patterns

- Assuming a cost of capital or a horizon.
- Counting a benefit nobody owns.
- Costing the preferred option fully and the status quo at zero.
- Presenting a net or payback figure as verified when the registry did not produce it.
