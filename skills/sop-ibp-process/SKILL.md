---
name: sop-ibp-process
description: "Run the monthly planning cycle on paper - the demand review, the supply review, the reconciliation of the two and the decisions for the executive meeting - from the forecasts, capacity and inventory figures the user supplies and the records we hold, producing the agenda, the gaps, the options with their cost and risk, and the decisions needed, each with an owner. Use when asked to prepare, run or write up an S&OP or IBP cycle, meeting or pack. Dormant until planning data can be read. Not for the forecast itself (consensus-forecasting) or the supply constraint maths (constrained-supply-planning)."
---

# Sales and Operations Planning

## Purpose

Make the cycle a sequence of decisions with evidence: where demand and supply disagree, what it costs to close the gap each way, and who decides. The skill prepares and records; the executive meeting decides.

## When to use and when not

Use when the request is the cycle, a step of it (demand review, supply review, reconciliation, executive meeting) or the pack for one.

Not for building the consensus forecast (consensus-forecasting), for the constrained plan maths (constrained-supply-planning), for inventory policy (inventory-policy-setting) or for the one-off business case (business-case).

## Inputs and preconditions

- The consensus demand per family and period, the supply plan or capacity per period, inventory and backlog, from the user.
- The cycle calendar and the decision owners.

- Use `tool:run_calculation` with `forecast.accuracy` for last cycle's forecast performance, `deal.pct_change` for plan-to-plan movements and `deal.share_pct` for a gap's share of demand
- Use `tool:build_spend_cube` for the spend by supplier and month that the supply side rests on
- Use `tool:contract_pipeline` for supply agreements ending in the horizon
- Use `tool:savings_tracker` for value delivered that the executive review reports
- Use `tool:build_pptx` to assemble the executive pack when asked
- Use `tool:get_policy` for the planning calendar, the decision rights and the inventory targets if governed
NOT AVAILABLE: run_mrp. no materials plan exists; supply figures are the user's
NOT AVAILABLE: check_capacity. no capacity data; constraints are the user's
NOT AVAILABLE: simulate_scenario. no scenario model; options are costed with the registry one by one

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.escalation_gap_pct`.

## Workflow

1. Frame: the cycle's period, the families in scope, the calendar and the owners; the decision rights from policy.
2. Demand review: the consensus forecast per family with last cycle's accuracy via the registry; changes since last cycle via the registry; the assumptions behind the changes as the owners state them.
3. Supply review: capacity and supply plan per period as supplied; agreements ending in the horizon from the pipeline; supplier risks from the records.
4. Reconciliation: the gaps per family and period (demand above supply, supply above demand); each gap's share via the registry; options to close each gap (overtime, alternate supplier, inventory build, demand shaping, accept the shortfall) with cost and risk as stated or costed with the registry; a gap above `config.thresholds.escalation_gap_pct` percent goes to the executive meeting.
5. Executive pack: the decisions needed with options, recommendation and owner; the metrics (forecast accuracy, inventory versus target, service); built with `tool:build_pptx` when asked.
6. Record the decisions taken and the actions with owners for the next cycle.

## Frameworks and heuristics

- One number per family: the consensus, not three views.
- A gap is a decision, not a surprise; it goes to the meeting with options.
- Cost the options the same way; the cheap one is often the one with the hidden risk.
- Last cycle's accuracy is the first slide.

## Policy hooks (SC-007)

Calendar, decision rights and governed targets come from `tool:get_policy` and are quoted. The skill decides nothing; a decision that commits spend runs through `tool:check_policy` in the carrying skill.

## Output specification

1. Cycle frame: period, families, calendar, owners.
2. Demand review: consensus, accuracy, changes, assumptions.
3. Supply review: capacity, plan, agreements ending, risks.
4. Reconciliation: gaps with shares, options with cost and risk, escalations.
5. Executive pack contents (and export id) and the decision log.

## Quality bar

- Every figure is the user's, a record, or the registry's.
- Every gap has options and an owner.
- Nothing is decided by the skill.

## Escalation and approval

Escalate to the executive meeting every gap above the configured share, to finance for the cost of options, and to supply-continuity-planning where a supply risk is behind a gap.

## Worked example

Asked to prepare this month's S&OP, the skill frames the three families and the calendar, reports last cycle's accuracy via the registry, lists the consensus changes with the owners' assumptions, sets the supplied capacity against demand, finds one family short by more than the escalation share in two months, costs the overtime and the alternate-supplier options with the user's figures, writes the decision for the executive meeting with a recommendation, and builds the pack.

## Anti-patterns

- Three demand numbers in the pack.
- A gap with no options.
- Capacity from memory.
- Deciding in the pack what the meeting should decide.
