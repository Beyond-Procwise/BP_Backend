---
name: consensus-forecasting
description: "Bring a statistical baseline and the commercial inputs together into one agreed forecast - the smoothed baseline from the series supplied, each function's adjustment with its reason and owner, the accuracy of last cycle's inputs by source, the reconciliation of the differences, and the consensus number with its assumptions logged. Use when asked to build, reconcile or document a consensus forecast. Dormant until series can be read; until then the series is the user's. Not for near-term sensing (demand-sensing) or the planning cycle around it (sop-ibp-process)."
---

# Consensus Forecasting

## Purpose

Produce one forecast everyone can defend: the baseline from the data, the overrides with names and reasons, and a record of who was right last time, so the next consensus leans on the better source.

## When to use and when not

Use when several views of demand (statistical, sales, marketing, finance, customer) must become one number.

Not for sensing short-term signals (demand-sensing), for the cycle (sop-ibp-process), for supply constraints (constrained-supply-planning) or for ordering patterns (ordering-and-inventory).

## Inputs and preconditions

- The historical series per family and the actuals for last cycle, from the user.
- Each function's forecast or adjustment with its reason and owner.

- Use `tool:run_calculation` with `forecast.exponential_smoothing` for the baseline, `forecast.accuracy` for each source's accuracy last cycle, `deal.pct_change` for each adjustment's size and `deal.share_pct` for a family's share of the total
- Use `tool:supplier_items` for our own order quantities per period as an actuals series we hold
- Use `tool:get_policy` for the forecasting rules (who may override, override limits, documentation required)
NOT AVAILABLE: statistical_forecast. no planning engine; the baseline is the registered smoothing on the supplied series

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.override_limit_pct`.

## Workflow

1. Frame: families, horizon, cycle, the functions contributing and the rules on overrides.
2. Baseline: smoothed via the registry with the planner's alpha from the supplied series (or our order quantities where that is the series we hold).
3. Inputs: each function's number per family with reason and owner; its size against the baseline via `deal.pct_change`; an override beyond `config.thresholds.override_limit_pct` percent needs the reason the rules require.
4. Track record: each source's accuracy last cycle via `forecast.accuracy`; the source with the lower error carries more weight in the reconciliation, stated.
5. Reconcile: where inputs differ, the evidence for each; the consensus number per family with the assumptions logged; unresolved differences named for the planning meeting.
6. Document: the consensus, the assumptions, the overrides and the owners, for the cycle's record.

## Frameworks and heuristics

- The baseline is what the data says; overrides are what people know; both are written down.
- Last cycle's accuracy by source is the fairest referee.
- An override without a reason is a wish.
- Agree the number, not the method, in the meeting.

## Policy hooks (SC-007)

Override rights, limits and documentation come from `tool:get_policy` and are quoted. The alpha is the planner's; no override is applied by the skill; the consensus is proposed for the planning meeting to agree.

## Output specification

1. Frame and rules.
2. Baseline per family with the alpha used.
3. Inputs table: source, number, size versus baseline, reason, owner.
4. Accuracy by source last cycle.
5. Consensus per family with assumptions, unresolved differences, owners.

## Quality bar

- Baseline, sizes and accuracy come from the registry.
- Every override has a reason and an owner.
- Unresolved differences are named, not averaged away.
- Nothing is applied.

## Escalation and approval

Escalate unresolved differences to the planning meeting, overrides beyond the limit to the planning owner, and a source with persistent bias to its function head.

## Worked example

Asked to build the consensus for three packaging families, the skill smooths each supplied series with the planner's alpha, tables sales' and finance's numbers with their reasons and sizes via the registry, reports that sales ran high by a stated bias last cycle and finance was closer, reconciles two families on the evidence and names the third as unresolved for the meeting, and logs the consensus with its assumptions and owners.

## Anti-patterns

- Averaging three views and calling it consensus.
- Accepting an override without a reason.
- Ignoring who was right last time.
- Applying the consensus to a plan from the skill.
