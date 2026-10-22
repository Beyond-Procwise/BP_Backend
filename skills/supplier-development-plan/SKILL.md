---
name: supplier-development-plan
description: "Turn an underperforming supplier's record into an improvement plan - the failures by type and date from the scorecard and findings, the root causes to ask for, the contract levers (review meetings, remedies, step-in, termination for persistent breach), the measurable targets, the review cadence and the consequence ladder. Use when a supplier keeps missing, needs a corrective action plan or a performance improvement plan, or when asked how to get them to improve. Not for the periodic review itself (supplier-performance-review) or for the exit decision (renewal-and-exit-strategy)."
---

# Supplier Development Plan

## Purpose

Give a failing relationship a structured path: evidence the supplier cannot argue with, causes asked not assumed, targets that are measurable from the records, dates, and a ladder that ends somewhere. The skill drafts the plan; the owner agrees it with the supplier.

## When to use and when not

Use when a supplier underperforms repeatedly and the user wants a corrective or improvement plan, a recovery plan, a set of targets and reviews, or the consequence ladder.

Not for the quarterly review pack (supplier-performance-review), for deciding to leave (renewal-and-exit-strategy, supplier-offboarding), for a single failed delivery (delivery-issue-handling) or for claims (dispute-and-claims).

## Inputs and preconditions

- The supplier and the period of concern.
- What the user has already done (meetings, warnings), labelled user-supplied.
- The outcome the user needs (on-time rate, quality level) if known.

- Use `tool:calc_supplier_scorecard` for on-time delivery, price variance and findings over the period, as the evidence base
- Use `tool:invoice_reconciliation` for open findings and their types in the period
- Use `tool:search_contracts` for the agreement
- Use `tool:extract_clauses` for the performance review, remedy, step-in, improvement and termination-for-breach clauses, quoted
- Use `tool:obligation_tracker` for review dates already set
- Use `tool:run_calculation` with `otif.delivery_performance` for the delivery measure and `deal.pct_change` for the gap to target
- Use `tool:get_policy` for any supplier-performance rule or escalation ladder the organisation has set
NOT AVAILABLE: send_message. the plan is drafted; the owner takes it to the supplier
NOT AVAILABLE: build_docx. the plan is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.improvement_trigger_pct`, `config.thresholds.review_interval_weeks`.

## Workflow

1. Frame: supplier, period, what the user needs to improve, what has been done.
2. Evidence: the scorecard measures and the findings by type and month; the trend; the contract's performance clauses quoted (targets, remedies, review rights, step-in, termination for persistent breach).
3. Gap: current measure versus the contract target or the user's required level; the gap via the registry calculation. A supplier below `config.thresholds.improvement_trigger_pct` on-time over the period, or with repeated findings of one type, is in scope.
4. Causes: list the questions to put to the supplier per failure type (capacity, materials, forecasting, logistics, quality system); causes are the supplier's to give, not the skill's to assume.
5. Plan: targets per measure with dates; actions per cause as the supplier proposes them; our actions (forecast sharing, order discipline) where the findings show our part; review cadence of `config.thresholds.review_interval_weeks` weeks; owner each side; evidence to be reviewed each time.
6. Ladder: review, formal notice under the clause, remedy or step-in, re-source decision (renewal-and-exit-strategy); each step tied to a measurable trigger and a date.

## Frameworks and heuristics

- Evidence by type and date, not adjectives; the supplier should be able to reproduce the figures.
- Causes are asked, not assigned; our own part is looked for in the findings.
- Targets are what the scorecard can measure; a target nobody can measure is a wish.
- The ladder is written down before the first review so nobody is surprised.
- A plan with no end date is a relationship on life support.

## Policy hooks (SC-007)

Performance rules and any escalation ladder come from `tool:get_policy` and are quoted. Formal notices and termination are legal acts a person takes under the clause; the plan names them as steps. A re-source that commits spend is checked with `tool:check_policy` when it is reached. Nothing is sent or recorded by the skill.

## Output specification

1. Frame and evidence: measures, findings by type and month, trend, clauses quoted.
2. Gap to target per measure.
3. Cause questions per failure type.
4. Plan table: target, measure, date, action, owner each side, evidence.
5. Review cadence and the consequence ladder with triggers and dates.
6. Draft agenda for the first meeting.

## Quality bar

- Every failure cited is a finding or a scorecard measure with a date.
- Targets are measurable with the scorecard.
- Clauses are quoted; steps that need a clause have one or are marked not provided for.
- No cause is asserted as fact.

## Escalation and approval

Escalate to the category lead where the ladder reaches formal notice, to legal before any notice or step-in, and to renewal-and-exit-strategy when the plan's end date arrives without the target.

## Worked example

Asked for an improvement plan for a packaging supplier that keeps delivering late, the skill reads the scorecard (on-time well below the contract target over six months, late deliveries concentrated in one plant), the findings (short deliveries clustered at month end), the contract's performance review and remedy clauses, and drafts: targets by month back to the contract level, cause questions on month-end capacity and forecasting, our action to share a rolling forecast, fortnightly reviews with the scorecard as evidence, and a ladder ending in a re-source decision at a stated date.

## Anti-patterns

- Assigning a cause the supplier has not given.
- Targets the scorecard cannot measure.
- A ladder with no dates or triggers.
- Sending the plan or a notice from the skill.
