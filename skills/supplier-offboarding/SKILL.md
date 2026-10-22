---
name: supplier-offboarding
description: Plan the safe exit from a supplier once the decision to leave has been taken - the notice the contract requires (quoted, never computed from an unstated period), the exit-assistance, data-return and transition clauses, the open orders and invoices to settle and reconcile, the alternative suppliers already on record, the risks of the handover and a dated plan with owners. Use when asked to offboard, exit, leave, stop using, wind down or transition away from a supplier. Not for deciding whether to leave (renewal-and-exit-strategy) or for the new supplier's setup (supplier-onboarding-checklist).
---

# Supplier Offboarding

## Purpose

Turn "we are leaving this supplier" into an exit that costs nothing it need not: notice given on time under the clause the contract actually holds, obligations on both sides listed and dated, open commercial matters closed, the handover to the next supplier sequenced, and the risks named with an owner. The skill reads the records and drafts; it sends no notice and closes no order.

## When to use and when not

Use when the decision to leave a supplier has been made, or is being assumed for planning, and the user asks how to offboard, exit, transition away, wind down or stop using them, or what the exit will take.

Not for the decision itself (renewal-and-exit-strategy, which hands the chosen exit here), for the notice wording as a legal document (contract-clause-review and legal), for the replacement supplier's onboarding (supplier-onboarding-checklist) or for the new sourcing event (rfx-design).

## Inputs and preconditions

- The supplier (name or supplier_id) and the reason and target date for the exit, if known.
- Whether a replacement is chosen, in progress or not started.
- Any service, data or stock held by the supplier that the user knows of, labelled user-supplied.

Use `tool:search_contracts` by supplier for the live agreements and `tool:obligation_tracker` per contract for the stored calendar and the notice, termination, renewal and review clauses quoted word for word; `tool:extract_clauses` once per contract with `containing` set to one word at a time, at most three calls ("exit", "transition", "data"), for the exit obligations on each side, and `clause_type` must_perform for the rest; `tool:invoice_reconciliation` for open invoice lines, lines with no order and open findings to settle before the last payment; `tool:calc_supplier_scorecard` for the spend at stake and open findings; `tool:supplier_discovery` and `tool:similar_items` for the alternative suppliers already on our records for the same items; `tool:screening_summary` for the replacement candidate's status; and `tool:check_policy` where the exit itself, an early-termination payment or the replacement commits spend.

NOT AVAILABLE: send_message. The notice and the handover requests are drafted as text; a person sends them.
NOT AVAILABLE: build_docx. Give the plan as text.

## Workflow

1. Frame. Confirm the supplier, the reason, the target date and the replacement status. The round budget is small: contracts and trackers, clauses, reconciliation, discovery, then the plan.
2. Contract position. For each live agreement: expiry, auto-renewal and renewal term as stored; the notice and termination clauses quoted; the notice period read from the clause, never computed from a period the records do not state; where no clause is stored the notice is "unknown: clause not read" and the plan asks for it first. Flag an exclusivity, minimum-volume or take-or-pay clause as a cost of leaving.
3. Exit obligations. From the clauses: exit assistance, transition support, data return or destruction, equipment or stock return, knowledge transfer, final reconciliation; each with the party bound, the clause reference and the wording; obligations the contract is silent on are "not provided for" and become requests to negotiate.
4. Commercial close-out. From the reconciliation: open invoice lines, lines with no order on record, open findings and disputes to settle before the final payment; from the scorecard the spend at stake over the period. Nothing is paid, held or disputed here.
5. Continuity. Alternatives already on our records for the same items (discovery, similar items) with their screening status; where none exists, the gap is the first risk and goes to rfx-design or urgent-sourcing. Stock, service and data continuity risks with owners.
6. Plan. Backward from the target date: notice by the deadline with `config.thresholds.notice_buffer_days` business days of buffer, transition period of at least `config.thresholds.min_transition_days` days unless the clause states another, data return, final reconciliation, last order date, access and systems removal, relationship close. Each step with an owner and a dependency.
7. Approval. Where an early-termination payment, a run-off order or the replacement commits spend, report the route from `tool:check_policy` as replied.
8. Output the plan, the draft notice text (a draft for legal to check), and the handover requests.

## Frameworks and heuristics

- The clause governs the exit: the notice, the assistance and the costs come from the contract's own words or are "not provided for".
- Pay for what was delivered, dispute what was not: the final reconciliation runs before the last payment, not after.
- Continuity before closure: the last order date comes after the replacement can deliver, with buffer.
- Data, access and stock are obligations with deadlines, not afterthoughts.
- A supplier told early is a supplier that cooperates; the plan sequences the notice at the deadline with buffer, not at the last hour.

## Policy hooks (SC-007)

Early-termination payments, run-off orders and the replacement's first order commit spend and are checked with `tool:check_policy`, reported as replied. The notice is a legal act a person sends; nothing is sent, paid or closed here.

## Output specification

1. Frame: supplier, reason, target date, replacement status.
2. Contract table: agreement, expiry, auto-renewal, notice and termination clauses quoted, notice status (quoted / not stored / clause not read), cost-of-leaving clauses.
3. Exit obligations by party with clause and wording; items not provided for.
4. Commercial close-out: open lines, no-order lines, findings, spend at stake.
5. Continuity: alternatives on record with screening status; risks with owners.
6. Dated plan with owners and dependencies; approval routes.
7. Draft notice text and handover requests; open questions for legal and the supplier.

## Quality bar

- Every clause is quoted with its reference; no notice period is computed from an unstated period.
- Every open commercial item comes from the reconciliation or the scorecard.
- Alternatives are from our records with their screening status; none from memory.
- Nothing is sent, paid, held or closed by the skill.

## Escalation and approval

Escalate to legal when the notice clause is not stored or the contract holds exclusivity or take-or-pay terms, to the category lead when no alternative is on record, and to finance for the final reconciliation and any termination payment. The notice and the approvals are people's acts.

## Worked example

Asked to offboard a logistics provider by March, the skill finds one live agreement (expiry in June, no auto-renewal, termination clause quoted: ninety days' notice), exit-assistance and data-return clauses stored, a reconciliation showing open invoice lines and three open findings, two alternative providers on record (one screened), and builds the plan backward from March: notice by the deadline with buffer, transition with the screened alternative, data return and final reconciliation before the last payment, last order date after the alternative's first delivery, access removal, and the approval route for the run-off order; the notice text is drafted for legal.

## Anti-patterns

- Computing the notice deadline from a "usual" period when the clause is not stored.
- Paying the final invoice before the reconciliation.
- Stopping orders before the replacement can deliver.
- Leaving data, access or stock return out of the plan.
- Sending the notice or closing the relationship from the skill.
