---
name: sla-design
description: "Design or review the service levels for a contract - the few measures that matter for the service, how each is measured and from what data, the targets with their basis, the reporting cadence, the remedies (service credits, step-in, termination for persistent failure) and their formulas, and the governance meetings - from the specification, what the records can measure, comparable contracts on the register and the policy. Use when asked what KPIs or SLAs a contract should have, how to measure a supplier's service, or whether a proposed SLA schedule is any good. Not for the dispute over a missed level (dispute-and-claims) or for the periodic review (supplier-performance-review)."
---

# Service Level Design

## Purpose

Give a contract service levels that can be measured from data we will actually have, targets with a stated basis, and remedies that bite without being absurd. The skill drafts the schedule's substance; legal writes the wording and the owner sets the targets.

## When to use and when not

Use when the question is which service levels or KPIs to put in a contract, how to measure them, what targets and remedies to set, or whether a supplier's proposed SLA schedule is acceptable.

Not for claiming service credits (dispute-and-claims), for the quarterly review (supplier-performance-review), for the improvement plan (supplier-development-plan) or for the clause wording (contract-clause-review and legal).

## Inputs and preconditions

- The service and its specification or the draft SLA schedule, labelled user-supplied.
- What matters most to the business about the service, from the owner.
- What data we will hold about the service (orders, receipts, tickets, meter readings).

- Use `tool:search_contracts` for comparable agreements on the register
- Use `tool:extract_clauses` for the service-level, measurement, credit, step-in and termination clauses in those agreements and in the draft, quoted
- Use `tool:calc_supplier_scorecard` for what the platform already measures (on-time delivery where expected dates and receipts exist, findings) and the supplier's current record; quality, responsiveness and credits are not recorded there
- Use `tool:get_policy` for any service-level standards or credit limits the organisation has set
- Use `tool:run_calculation` with `deal.pct_change` for a target's gap to current performance
NOT AVAILABLE: build_docx. the schedule is given as structured text for legal

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_measures`, `config.thresholds.max_credit_pct`.

## Workflow

1. Frame: the service, what matters to the owner, what data we will hold.
2. Measures: at most `config.thresholds.max_measures` service levels, each tied to an outcome the owner named, each measurable from data we will hold (named), each with a definition (what counts, what is excluded, the period). A measure without a data source is dropped or becomes a reporting obligation on the supplier with audit rights.
3. Targets: each with a basis (the supplier's current record from the scorecard where it assesses the measure, a comparable contract on the register, the specification, the owner's requirement); the gap to current performance via `deal.pct_change` where a record exists; no target from memory.
4. Remedies: service credits with a formula on the invoice value and a cap within `config.thresholds.max_credit_pct`, escalating with persistence; step-in and termination-for-persistent-failure triggers stated as counts over periods; credits as the sole remedy or not, as legal advises; earn-back only if the owner wants it.
5. Governance: reporting by the supplier at a stated cadence from stated data; review meetings; dispute route for measurement disagreements.
6. Review of a draft: each proposed level tested against the rules above; gaps, unmeasurable levels, targets with no basis, remedies with no formula or cap, named for legal and the owner.

## Frameworks and heuristics

- Few measures, measured well; a schedule with twenty levels measures none.
- If we will not hold the data, the supplier reports it and we audit it, or the level goes.
- A target has a basis or it is an opinion.
- Credits deter; they do not compensate; the cap says how much deterrence we bought.
- Persistence triggers (three misses in six months) matter more than any single month.

## Policy hooks (SC-007)

Service-level standards and credit limits come from `tool:get_policy` and are quoted. The schedule's wording is legal's; the targets are the owner's; nothing is agreed or signed by the skill. Where a credit cap or step-in right commits or releases value, the policy route is checked with `tool:check_policy` and reported as replied.

## Output specification

1. Frame: service, outcomes that matter, data we will hold.
2. Measure table: level, definition, data source, period, exclusions.
3. Target table: level, target, basis, gap to current via the registry.
4. Remedy table: credit formula, cap, persistence triggers, step-in, termination.
5. Governance: reporting, meetings, dispute route.
6. For a draft: the review findings for legal and the owner.

## Quality bar

- Every measure names its data source and definition.
- Every target has a stated basis; none from memory.
- Credit formulas have a cap and persistence triggers are counts over periods.
- Nothing is agreed or signed.

## Escalation and approval

Escalate to legal for the wording and the sole-remedy question, to the owner for targets with no basis, and to the policy owner where no credit limit is set.

## Worked example

Asked what SLAs the new logistics contract should have, the skill reads the specification, finds the platform measures on-time delivery and findings for the incumbent, reads two comparable contracts' service-level clauses on the register, proposes four measures (on-time delivery as the scorecard measures it where expected dates and receipts exist, damage rate from findings, booking response time reported by the supplier with audit, invoice accuracy from findings), sets targets with bases (the incumbent's record plus the owner's requirement, the comparable contracts), credits as a share of the monthly invoice with a cap within the configured limit and a persistence trigger, step-in and termination triggers as counts, monthly reporting and quarterly reviews, and lists the open points for legal.

## Anti-patterns

- A measure nobody can take from data.
- Targets copied from a textbook.
- Credits with no cap or no persistence trigger.
- Agreeing the schedule from the skill.
