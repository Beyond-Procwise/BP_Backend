---
name: capa-management
description: "Define and track corrective and preventive actions for a problem with an evidenced cause - the action that removes the cause, the preventive action that stops recurrence elsewhere, owners, dates, the verification measure and its target, and the effectiveness check after a set period - as a register the user keeps, with the measures from our records and the registered calculations. Use when asked to set up, track or verify corrective actions. Dormant until an action register exists; until then the register is the user's. Not for finding the cause (root-cause-analysis) or the supplier's 8D (eight-d-problem-solving)."
---

# Corrective and Preventive Action

## Purpose

Make an action a commitment with a test: who does what by when, how we will know it worked, and when we check; closed only on evidence.

## When to use and when not

Use when a cause is known and actions are being defined, tracked or verified for effectiveness.

Not for the cause (root-cause-analysis), for the supplier's response (eight-d-problem-solving), for an audit's findings as such (supplier-audit) or for the performance plan (supplier-development-plan).

## Inputs and preconditions

- The problem and its evidenced cause.
- The actions proposed, owners and dates; the current register, from the user.

- Use `tool:calc_supplier_scorecard` for the delivery and findings measures an action aims to move
- Use `tool:invoice_reconciliation` for the findings pattern before and after an action
- Use `tool:run_calculation` with `quality.ppm` for a defect rate before and after from the user's counts, `deal.pct_change` for the movement, and `deal.share_pct` for an action's coverage of occurrences
- Use `tool:get_policy` for the CAPA rules (who approves closure, verification period)
- Use `tool:build_xlsx` for the action register as a workbook when asked
NOT AVAILABLE: log_non_conformance. no non-conformance or action register exists; the register is the user's workbook built here

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.verification_days`.

## Workflow

1. Frame: the problem, the cause with its evidence, the measure that shows the problem (findings, rate, on-time) with its current value from the records or the registry.
2. Corrective action: removes the cause; preventive action: applies the fix wherever the cause could recur; each with owner, date, resources and the risk of the action itself.
3. Verification: the measure, its target, the data source, and the check date at least `config.thresholds.verification_days` days after implementation; an action without a measure is not accepted.
4. Track: status per action (open, implemented, verified, closed, overdue); overdue actions first; the register built as a workbook when asked.
5. Effectiveness: the measure after the period via the registry against before; closed only when the target is met; otherwise back to root-cause-analysis.

## Frameworks and heuristics

- An action without a measure is a hope; a measure without a date is a wish.
- Corrective fixes this one; preventive fixes the others.
- Verify after the period, not the week after the fix.
- Overdue is a status, not a secret.

## Policy hooks (SC-007)

CAPA rules, closure authority and the verification period come from `tool:get_policy` and are quoted. The skill defines and tracks; closure is the quality owner's act on evidence.

## Output specification

1. Problem, cause, measure before.
2. Action register: corrective and preventive actions with owners, dates, measures, targets, status.
3. Verification results via the registry; closure recommendations.
4. Workbook export id when built.

## Quality bar

- Every action has a measure, a target and a date.
- Before and after come from the records or the registry.
- Closure is on evidence only.
- Nothing is closed by the skill.

## Escalation and approval

Escalate overdue actions to the owner's manager, failed verifications to root-cause-analysis, and closure to the quality owner.

## Worked example

Given the cause of the film breaks (gauge below specification from one supplier), the skill defines the corrective action (specification tightened and incoming gauge check, owner quality, date) and the preventive action (the same check on all film suppliers), sets the measure (breaks per thousand rolls via the registry from the user's log) with a target and a check date after the verification period, tracks the actions in the register workbook, and at the check date computes the movement via the registry and recommends closure only if the target is met.

## Anti-patterns

- An action with no measure.
- Closing on implementation instead of verification.
- Only corrective, no preventive.
- Closing from the skill.
