---
name: control-tower-monitoring
description: "Watch the supply chain's live signals and turn them into alerts with owners - late orders against expected dates, receipts missing, invoices blocked, findings open, contracts near expiry, suppliers flagged, stock cover below policy - from the records we hold and the shipment and stock feeds the user supplies, each alert graded by the configured rules with what it means and who acts. Use when asked what needs attention today, for the supply chain dashboard's signals, or to set up monitoring thresholds. Dormant until shipment and stock feeds can be read; until then the records we hold are the signals. Not for the period report (executive-supply-chain-reporting)."
---

# Supply Chain Control Tower

## Purpose

Make the day's attention list from facts: each signal a record, each alert a rule, each alert an owner; nothing predicted, nothing invented.

## When to use and when not

Use when the question is what needs attention now, which orders or shipments are at risk, or how alerts should be set.

Not for the period report (executive-supply-chain-reporting, procurement-reporting), for the fix itself (delivery-issue-handling, p2p-exception-handling, supply-continuity-planning), or for the forecast (demand-sensing).

## Inputs and preconditions

- The scope (suppliers, sites, categories) and the signals the user can supply beyond our records (shipment tracking, stock cover).
- The alert rules from policy or the defaults here.

- Use `tool:open_orders` for orders past their expected date (where recorded) and orders with no receipt, per supplier in scope
- Use `tool:invoice_status` for invoices overdue by stored due date, with open findings, or with no order
- Use `tool:invoice_reconciliation` for open findings by kind
- Use `tool:contract_pipeline` for contracts ending within the alert horizon and never read for clauses
- Use `tool:screening_summary` for suppliers with flagged, expired or not-performed checks
- Use `tool:monitor_risk_events` for risk events recorded against suppliers in scope
- Use `tool:calc_supplier_scorecard` for suppliers whose on-time rate fell below the rule
- Use `tool:run_calculation` with `deal.share_pct` for an alert class's share of orders or spend and `deal.pct_change` for a signal's movement against the prior period
- Use `tool:get_policy` for the alert rules and thresholds if governed
- Use `tool:render_chart` for the alert counts by class when asked
NOT AVAILABLE: track_shipment. no carrier connection; shipment status is the user's feed, labelled
NOT AVAILABLE: query_wms. no warehouse connection; stock cover is the user's feed, labelled
NOT AVAILABLE: query_erp. our records are read through open_orders, invoice_status and the reconciliation

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.late_days`, `config.thresholds.contract_horizon_days`.

## Workflow

1. Scope and rules: the suppliers, sites and categories in scope; the alert rules from policy or the defaults (an order `config.thresholds.late_days` days past its expected date; an invoice overdue; a contract ending within `config.thresholds.contract_horizon_days` days unread; a flagged or expired check; stock cover below the user's policy days).
2. Signals from our records: late and unreceipted orders, overdue and blocked invoices, open findings, contracts near expiry, screening flags, risk events, scorecards below the rule; signals from the user's feeds labelled by source and time.
3. Grade each alert by the rules (red, amber, watch) with the fact behind it and its date; a signal the records do not hold (an order with no expected date) is reported as unmonitorable, not green.
4. Owners: each alert class to its owner (buyer, site, finance, contract owner, compliance, category lead) and the skill that works it; counts and shares via the registry; movement against yesterday or last week via the registry where the user has the prior list.
5. The attention list ordered by grade then value; the chart of counts by class when asked; nothing actioned.

## Frameworks and heuristics

- An alert is a fact plus a rule; a feeling is not an alert.
- No expected date, no lateness: say unmonitorable.
- Red is for today; amber is for this week; watch is for the trend.
- The owner is named on every line.

## Policy hooks (SC-007)

Alert rules and thresholds come from `tool:get_policy` and are quoted; defaults are named as defaults. The skill actions nothing: each alert goes to its owner and its working skill.

## Output specification

1. Scope and rules with sources.
2. Attention list: alert, grade, fact, date, owner, working skill; ordered.
3. Counts and shares by class via the registry; movement where a prior list exists.
4. Unmonitorable items with the data gap named.
5. Chart export id when drawn.

## Quality bar

- Every alert cites a record or a labelled feed and a rule.
- Nothing without an expected date is called late or on time.
- Counts, shares and movements come from the registry.
- Nothing is actioned.

## Escalation and approval

Escalate red alerts to their owners the same day, a class whose share jumps to the process owner, and data gaps that leave orders unmonitorable to the data owner.

## Worked example

Asked what needs attention today across the top suppliers, the skill reads open orders past their expected dates (and reports the ones with no expected date as unmonitorable), overdue invoices by stored due date, open findings by kind, contracts ending within the horizon never read for clauses, and two suppliers with expired screenings, grades each by the rules, names the owner and working skill per line, and gives the counts by class via the registry with the chart.

## Anti-patterns

- Calling an order on time because no date exists.
- An alert without an owner.
- Predicting a delay from a feeling.
- Actioning an alert from the skill.
