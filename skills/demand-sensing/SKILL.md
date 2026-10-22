---
name: demand-sensing
description: "Adjust a short-term demand forecast for the signals that have arrived since it was made - recent orders, point-of-sale or consumption data, promotions, market signals the user supplies - by measuring the forecast's recent error, deciding whether the signal is a step or a blip, and proposing a sensed forecast with its basis for the planner to accept. Use when asked whether a forecast still holds, what recent orders say about demand, or to sense demand for the next weeks. Dormant until a demand series can be read; until then the series is the user's. Not for the consensus process (consensus-forecasting) or the statistical baseline alone (the forecast calculation)."
---

# Demand Sensing

## Purpose

Keep the near-term forecast honest against what is actually happening, with the error measured and the adjustment proposed, never applied; the planner owns the number.

## When to use and when not

Use when recent actuals, orders or signals have arrived and the question is whether the forecast for the next few periods should move.

Not for the monthly consensus (consensus-forecasting), for the long-range plan (sop-ibp-process), for supply constraints (constrained-supply-planning) or for a demand pooling across sites (demand-aggregation).

## Inputs and preconditions

- The current forecast per period and the actuals so far, from the user.
- The recent signals: open orders, consumption, promotions, market news, labelled by source and date.

- Use `tool:run_calculation` with `forecast.accuracy` for the forecast's error over the recent periods, `forecast.exponential_smoothing` for a smoothed baseline from the actuals, and `deal.pct_change` for the movement between the forecast and the sensed view
- Use `tool:open_orders` for our own open orders on the item's suppliers as a demand signal we hold
- Use `tool:supplier_items` for the item's order quantities over recent periods as the actuals we hold
- Use `tool:commodity_index_lookup` for a market signal on record that may move demand
- Use `tool:get_policy` for the planning rules (who may change a forecast, within what band)
NOT AVAILABLE: statistical_forecast. there is no planning engine; the baseline is the registered exponential smoothing on the series the user supplies
NOT AVAILABLE: simulate_scenario. no scenario model; a sensed view is one adjusted series, stated with its basis

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.sensing_horizon_periods`, `config.thresholds.bias_alert_pct`.

## Workflow

1. Frame: item or family, horizon (next `config.thresholds.sensing_horizon_periods` periods), the forecast and the actuals as supplied.
2. Measure: the forecast's error over the recent periods via `forecast.accuracy` (MAPE, weighted MAPE, bias); a bias beyond `config.thresholds.bias_alert_pct` points is the first finding.
3. Signals: each recent signal with source, date and direction; our own open orders and recent order quantities from the records; a market index movement if one is on record.
4. Judge step or blip: a signal that persists across periods or has a stated cause is a step; a single period is a blip until repeated; the judgement names its evidence.
5. Sense: a smoothed baseline via `forecast.exponential_smoothing` with the planner's alpha, adjusted for a step only as the planner states the size; the sensed view against the forecast via `deal.pct_change`.
6. Propose, do not apply: the sensed forecast per period, its basis, the error it answers, and the planning rule it must satisfy; the planner accepts or rejects.

## Frameworks and heuristics

- Measure the error before touching the forecast.
- One surprising week is a blip; two with a cause is a step.
- A signal without a date is not a signal.
- The planner changes the number; the skill proposes it.

## Policy hooks (SC-007)

Planning rules (who may change a forecast and within what band) come from `tool:get_policy` and are quoted. The alpha and any step size are the planner's inputs, never chosen by the skill; nothing is applied to any plan.

## Output specification

1. Frame and the supplied forecast and actuals.
2. Error measures from the registry; bias finding.
3. Signals table with source, date, direction, our records' signals.
4. Step or blip judgement with evidence.
5. Proposed sensed forecast per period with basis and movement; the rule it must satisfy.

## Quality bar

- Every error figure and movement comes from the registry.
- Every signal has a source and a date.
- No step size is invented; the planner states it.
- Nothing is applied.

## Escalation and approval

Escalate to the demand planner for the decision, to sales for a signal's cause, and to constrained-supply-planning where a sensed rise exceeds supply.

## Worked example

Asked whether the gloves forecast still holds after a jump in orders, the skill measures the last eight weeks' error via the accuracy calculation (bias low by more than the alert), lists the signals (two weeks of higher orders from our records, a site opening the user names and dates), judges a step on the stated cause, smooths the actuals with the planner's alpha, proposes the sensed forecast with the planner's step size and the movement via the registry, and leaves the change to the planner.

## Anti-patterns

- Changing the forecast on one week's data.
- Choosing an alpha or a step size for the planner.
- Using a market index as if it were demand.
- Applying the sensed view to a plan.
