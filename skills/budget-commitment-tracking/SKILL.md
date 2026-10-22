---
name: budget-commitment-tracking
description: "Report where a budget, category or cost centre stands against its commitments - invoiced spend to date, open order value still to be invoiced, contracted commitments ahead, the run rate and the position at year end if it continues - with every figure from the records and the budget figure from the user or policy. Use when asked how much budget is left, what is committed, whether we will overspend, or what the run rate is. Not for the spend analysis of what was bought (spend-analysis) or for savings (savings-methodology)."
---

# Budget and Commitment Tracking

## Purpose

Answer 'where are we against the budget?' with figures the records hold: what is invoiced, what is ordered but not yet invoiced, what the contracts will bring, and what the year-end looks like on the current run rate. The skill reports; it commits, releases and moves nothing.

## When to use and when not

Use when the question is about budget remaining, committed spend, open order value, accruals, run rate, forecast to year end or overspend risk.

Not for what was bought and from whom (spend-analysis), for value claims (savings-methodology), for the ledger of opportunities (opportunity-pipeline-review) or for finance's own accrual postings.

## Inputs and preconditions

- The scope: budget, category, cost centre, supplier or project; and the period or fiscal year.
- The budget figure, from the user or policy, labelled by source.

- Use `tool:build_spend_cube` for invoiced spend in the period by the dimension that matches the scope
- Use `tool:invoice_reconciliation` for open orders and their invoiced versus uninvoiced value for the scope
- Use `tool:contract_pipeline` for the contracted commitments ahead in the period
- Use `tool:obligation_tracker` for scheduled payments and milestones under a contract
- Use `tool:run_calculation` with `deal.pct_change` for the movement against the prior period, `deal.scaled_total` for the year-end projection from the monthly rate and the months remaining, and `deal.share_pct` for the share of budget used
- Use `tool:get_policy` for a governed budget or limit for the scope
- Use `tool:get_corpus_facts` for the period and totals the corpus holds, so scope and period are stated once
NOT AVAILABLE: post_accrual. accruals are finance's posting; the skill reports the uninvoiced open order value finance may accrue
NOT AVAILABLE: build_chart. figures are given as tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.overspend_alert_pct`.

## Workflow

1. Frame: scope, period, budget figure and its source. Where no budget is given or governed, the report is a commitment position only, and says so.
2. Invoiced to date from the cube for the scope and period.
3. Open commitments: orders not yet fully invoiced (order value less invoiced) from the reconciliation; contracted amounts due in the rest of the period from the pipeline and trackers; each listed with its source.
4. Position: budget, invoiced, open orders, contracted ahead, remaining; the run rate from the elapsed part of the period via the registry calculation; the year-end projection if it continues; a projection above the budget by more than `config.thresholds.overspend_alert_pct` is flagged.
5. Movements against the prior period via the registry calculation; the largest open orders and commitments named.
6. Caveats: scope mismatches between the records' dimensions and the budget's; currencies without rates; orders with no value.

## Frameworks and heuristics

- Committed is ordered-not-invoiced plus contracted-ahead; invoiced is spent; the sum against the budget is the position.
- A run rate is the elapsed period's rate, not last month's; say which.
- The budget is the user's or the policy's figure; the skill never sets one.
- The records' dimensions (supplier, buyer, cost centre) rarely match a budget line exactly; say what was used.

## Policy hooks (SC-007)

A governed budget or limit comes from `tool:get_policy` and is quoted; otherwise the budget figure is the user's and labelled. Nothing is committed, released, accrued or approved by the skill; a projected overspend is reported to the owner, not acted on.

## Output specification

1. Frame: scope, period, budget and source.
2. Position table: budget, invoiced, open orders, contracted ahead, remaining, with sources.
3. Run rate and year-end projection via the calculations; overspend flag against the configured share.
4. Movements and the largest commitments.
5. Caveats on scope, currency and missing values.

## Quality bar

- Every figure traces to a tool result or the user's budget statement.
- The run rate and movements come from the calculation registry.
- Scope mapping between the records and the budget is stated.
- Nothing is posted or committed.

## Escalation and approval

Escalate to the budget owner where the projection exceeds the budget by the configured share, to finance for accruals and scope reconciliation, and to the category lead where one commitment dominates.

## Worked example

Asked how much of the facilities budget is left this fiscal year, the skill takes the budget from the user, reads invoiced spend for the facilities buyer from the cube, open orders not yet invoiced from the reconciliation, the maintenance contract's remaining scheduled payments from its tracker, computes the run rate over the elapsed months and the year-end projection, finds the projection above the budget by more than the configured share, and reports the position with the three largest commitments and the caveat that the budget line and the buyer dimension do not match exactly.

## Anti-patterns

- Inventing a budget figure.
- Counting an order both as open commitment and as invoiced.
- Projecting from one month's spend as if it were the run rate.
- Posting an accrual or releasing a commitment from the skill.
