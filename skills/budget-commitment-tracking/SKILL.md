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

- The scope, mapped to what the records hold: a list of suppliers, a buyer, a country or region; and the period or fiscal year.
- The budget figure, from the user or policy, labelled by source.
- Any contracted amounts due in the period, from the user, labelled (the records hold contract totals, not schedules).

- Use `tool:build_spend_cube` for invoiced spend in the period by supplier, buyer, month, country or region (it has no cost centre, category or project: the scope must be mapped to one of these or to a list of suppliers, and the mapping is stated)
- Use `tool:open_orders` for each supplier in scope, the orders with the invoices booked against them and the balance still open (the open commitment, with the tool's tax-basis caveat)
- Use `tool:contract_pipeline` for the contracts ending in the period with their total value as stored (no amounts due by period are recorded: a contract's remaining value is the user's figure or not available)
- Use `tool:run_calculation` with `finance.average_per_period` for the monthly run rate from the invoiced total and the months elapsed, `deal.scaled_total` for the year-end projection from that rate and the months remaining, `deal.share_pct` for the share of budget used and `deal.pct_change` for the movement against the prior period
- Use `tool:get_policy` for a governed budget or limit for the scope
- Use `tool:get_corpus_facts` for the period and totals the corpus holds, so scope and period are stated once
NOT AVAILABLE: post_accrual. accruals are finance's posting; the skill reports the uninvoiced open order value finance may accrue
NOT AVAILABLE: build_chart. figures are given as tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.overspend_alert_pct`.

## Workflow

1. Frame: scope mapped to a records dimension (suppliers, buyer, country, region) by the user, never inferred from supplier names; period; budget figure and its source. Where no budget is given or governed, the report is a commitment position only, and says so.
2. Invoiced to date from the cube for the scope and period.
3. Open commitments: for each supplier in scope, the open balance per order from open_orders (order total minus invoices booked, same currency, as stored); contracts ending in the period from the pipeline with their stored total value, and any amounts due that the user states, labelled.
4. Position: budget, invoiced, open order balances, contracted amounts the user states, remaining; the monthly run rate via `finance.average_per_period` from the invoiced total and the months elapsed, the year-end projection via `deal.scaled_total` from that rate and the months remaining; the share of budget used via `deal.share_pct`; a projection above the budget by more than `config.thresholds.overspend_alert_pct` is flagged.
5. Movements against the prior period via `deal.pct_change`; the largest open balances named.
6. Caveats: the scope mapping; order and invoice tax basis; currencies without rates; orders with no value.

## Frameworks and heuristics

- Committed is the open order balance plus what the user states as contracted and due; invoiced is spent; the sum against the budget is the position.
- The run rate is the registry's average over the months elapsed, and the projection scales it by the months remaining; say how many months it rests on.
- The budget is the user's or the policy's figure; the skill never sets one.
- Budgets live by cost centre or category and the records do not; the mapping to suppliers or buyers is stated, never silent.

## Policy hooks (SC-007)

A governed budget or limit comes from `tool:get_policy` and is quoted; otherwise the budget figure is the user's and labelled. Nothing is committed, released, accrued or approved by the skill; a projected overspend is reported to the owner, not acted on.

## Output specification

1. Frame: scope and its mapping, period, budget and source.
2. Position table: budget, invoiced, open order balances, contracted amounts stated, remaining, with sources.
3. Projection and share of budget via the calculations; overspend flag against the configured share.
4. Movements and the largest open balances.
5. Caveats on mapping, tax basis, currency and missing values.

## Quality bar

- Every figure traces to a tool result or the user's budget statement.
- The run rate, the projection, the share of budget and the movements come from the calculation registry (average_per_period, scaled_total, share_pct, pct_change); no arithmetic in the answer.
- Scope mapping between the records and the budget is the user's and is stated; suppliers are never assigned to a budget by their names.
- Nothing is posted or committed.

## Escalation and approval

Escalate to the budget owner where the projection exceeds the budget by the configured share, to finance for accruals and scope reconciliation, and to the category lead where one commitment dominates.

## Worked example

Asked how much of the facilities budget is left this fiscal year, the skill takes the budget from the user, maps the scope to the facilities buyer, reads invoiced spend for that buyer from the cube, the open balances on the facilities suppliers' orders from open_orders, the maintenance contract ending this year from the pipeline with its stored total and the user's statement of what is still due, takes the run rate over the months elapsed and projects year end over the months remaining via the registry, finds the projection above the budget by more than the configured share, and reports the position with the three largest open balances and the caveat that the budget line and the buyer dimension do not match exactly.

## Anti-patterns

- Inventing a budget figure.
- Counting an order both as open commitment and as invoiced.
- Posting an accrual or releasing a commitment from the skill.
- Projecting from a figure the registry did not compute.
- Deciding which suppliers belong to a budget from their names.
