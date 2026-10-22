---
name: labour-planning
description: "Plan the warehouse workforce against the workload - the volumes per period and activity (receiving, putaway, picking, packing, dispatch) from the forecast the user supplies, the productivity rates per activity from the user's records, the hours and heads per shift the workload needs, the gap to the current roster, and the options to close it (overtime, temporary staff, shift changes) with cost. Use when asked how many people a period needs, to plan shifts for a peak, or whether the roster covers the forecast. Dormant until workload data can be read. Not for the layout (warehouse-layout-design)."
---

# Warehouse Labour Planning

## Purpose

Turn a volume forecast into heads and hours per shift with the arithmetic from the registry and the rates from the site, so the peak is staffed on numbers and the quiet weeks are not.

## When to use and when not

Use when staffing a warehouse for a period, a peak or a change in volume.

Not for the layout (warehouse-layout-design), for the demand forecast itself (demand-sensing, consensus-forecasting), for buying temporary labour (rfx-design), or for HR rules on shifts.

## Inputs and preconditions

- Volumes per period and activity (orders, lines, pallets in and out) from the user's forecast.
- Productivity rates per activity (units per hour), the current roster (heads and hours per shift), absence rate, from the user.
- Cost rates for overtime and temporary staff, from the user or finance.

- Use `tool:run_calculation` with `finance.average_per_period` for the rate per hour from a period's units and hours, `deal.scaled_total` for hours needed from volume and the inverse rate (hours per unit the user states), heads from hours and shift length, and cost from hours and rate, `deal.pct_change` for the gap to the roster and `deal.share_pct` for an activity's share of hours
- Use `tool:get_policy` for working-time, overtime and temporary-labour rules if governed
- Use `tool:build_xlsx` for the staffing plan workbook
NOT AVAILABLE: query_wms. no warehouse connection; volumes and productivity are the user's
NOT AVAILABLE: statistical_forecast. no planning engine; the volume forecast is the user's (or forecast.exponential_smoothing on a series they supply, through run_calculation)

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.gap_alert_pct`.

## Workflow

1. Frame: the period, the activities, the volumes per period and activity as supplied, the productivity rates with their source and date, the roster, the absence rate, the rules.
2. Hours per activity per period via the registry from volume and hours per unit; total hours; heads per shift from hours and shift length with absence added as the user's rate; each figure's inputs shown.
3. Gap to the roster per shift via the registry; a gap above `config.thresholds.gap_alert_pct` percent is a staffing decision.
4. Options per gap: overtime (within the rules), temporary staff, shift pattern change, moving work between days; cost per option via the registry from hours and the user's rates; the cheapest within the rules first.
5. Plan per shift per period with the chosen options for the site manager; the workbook built when asked.

## Frameworks and heuristics

- Productivity rates are measured, not assumed; a rate from last peak is better than a rate from a textbook.
- Absence is a cost line, not a surprise.
- Overtime is the cheapest option until it is not; the rules say where.
- Plan the quiet weeks too; overstaffing is a cost that does not complain.

## Policy hooks (SC-007)

Working-time, overtime and temporary-labour rules come from `tool:get_policy` and are quoted. The skill plans; hiring, rostering and overtime are the site manager's and HR's acts.

## Output specification

1. Frame with sources.
2. Hours and heads per activity, shift and period via the registry.
3. Gap to roster per shift; flagged gaps.
4. Options per gap with cost; the plan; workbook export id when built.

## Quality bar

- Every hour, head and cost comes from the registry on the user's inputs.
- Every rate has a source and a date.
- Rules are quoted; options outside them are not proposed.
- Nothing is rostered or hired.

## Escalation and approval

Escalate to HR for working-time rules and temporary labour, to the site manager for the roster, and to finance for the overtime budget.

## Worked example

Asked to staff the November peak, the skill takes the user's forecast per week and activity, the site's measured productivity rates, the roster and the absence rate, computes hours and heads per shift via the registry, finds the pick shift short by more than the alert in weeks three and four, costs overtime within the quoted rules and temporary staff at the user's rate, proposes temps for the two weeks as the cheaper option, and builds the plan workbook for the site manager.

## Anti-patterns

- A productivity rate from memory.
- Forgetting absence.
- Overtime beyond the working-time rules.
- Rostering from the skill.
