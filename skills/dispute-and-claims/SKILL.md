---
name: dispute-and-claims
description: "Build the case for a commercial claim against a supplier or a defence against theirs - breach, late delivery loss, service credits, liquidated damages, chargebacks, recovery of costs - from the contract's remedy, liability and dispute clauses, the facts on record and the loss figures the user can evidence, and lay out the dispute-resolution route the contract sets. Use when asked what we can claim, whether they are in breach, how to recover a cost or how to respond to their claim. Not for the first handling of a bad delivery (delivery-issue-handling) or for invoice overcharges (overbilling-investigation)."
---

# Dispute and Claims

## Purpose

Put the facts, the clauses and the figures of a dispute in one place so the owner and legal can decide: what the contract gives us, what we can evidence, what route the contract requires, and what to ask for. The skill assembles; it claims nothing, admits nothing and sends nothing.

## When to use and when not

Use when the question is whether a supplier is in breach, what we are entitled to claim, how to recover a loss or a cost, whether service credits or liquidated damages apply, or how to answer a claim the supplier has made against us.

Not for the operational handling of a delivery failure (delivery-issue-handling), for invoice overcharges (overbilling-investigation), for the negotiation of a settlement (negotiation-planning) or for legal advice; legal owns the claim.

## Inputs and preconditions

- The supplier, the contract and the events in the user's words with dates.
- The loss or cost figures the user can evidence, labelled user-supplied, with how they were measured.
- Any claim the supplier has made, as written.

- Use `tool:search_contracts` for the agreement and its stored summary
- Use `tool:extract_clauses` for the remedy, service-credit, liquidated-damages, liability-cap, exclusion, force-majeure, notice and dispute-resolution clauses, quoted
- Use `tool:obligation_tracker` for the dates and obligations in play
- Use `tool:invoice_reconciliation` for the order, invoice and finding facts the claim rests on
- Use `tool:calc_supplier_scorecard` for the delivery and quality record over the period
- Use `tool:run_calculation` with `deal.pct_change` for a shortfall share and plain arithmetic for a credit or damages figure from the clause's own formula and the user's evidenced inputs
- Use `tool:get_policy` for who may raise a claim and the approval a settlement needs
- Use `tool:check_policy` where a settlement, write-off or credit moves spend
NOT AVAILABLE: send_message. any letter is drafted as text; legal or the owner sends it
NOT AVAILABLE: build_docx. the case file is given as text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.legal_referral_gbp`.

## Workflow

1. Frame: parties, contract, events with dates, what the user wants (money, performance, defence).
2. Facts from the records: orders, receipts, invoices, findings, scorecard history for the period; the user's evidence listed separately with its basis.
3. Clauses quoted: the obligation said to be breached; the remedy for that breach; service credits or liquidated damages and their formula; the liability cap and exclusions; force majeure; notice requirements for a claim; the dispute-resolution ladder (escalation, mediation, arbitration or courts) and time limits.
4. Compute only what a clause's formula and evidenced inputs allow (a credit per day late at the clause's rate; a cap). Loss figures beyond a formula are the user's and are shown as claimed, not verified. A claim above `config.thresholds.legal_referral_gbp` goes to legal before anything is sent.
5. Assess: is the breach evidenced; is the remedy available; is notice in time; is the amount within the cap; what the ladder requires first. For a supplier's claim: the same questions in reverse.
6. Output the case summary, the ask or the defence, the route and the first step with owner; drafts handed to supplier-communication-drafting or legal.

## Frameworks and heuristics

- Clause first, then facts, then figures; a figure without a clause is a request, not a claim.
- Notice and time limits decide more disputes than merits do; check them first.
- The liability cap and the exclusions bound every number.
- Follow the ladder the contract sets; skipping a step can forfeit the remedy.
- Never admit, never set off against an undisputed invoice without the clause, never settle from the skill.

## Policy hooks (SC-007)

Who may raise a claim and the approval a settlement, write-off or credit needs come from `tool:get_policy` and `tool:check_policy`, reported as replied. Nothing is sent, admitted, set off or settled by the skill; legal owns the claim above the configured referral value and whenever the supplier disputes the facts.

## Output specification

1. Case summary: parties, contract, events, dates.
2. Facts from the records and the user's evidence, separated.
3. Clause table: obligation, remedy, credits or damages formula, cap and exclusions, notice, dispute ladder, each quoted.
4. Figures: computed from clause formulas with inputs shown; claimed figures labelled.
5. Assessment and the ask or defence; route and first step with owner; legal referral where due.

## Quality bar

- Every entitlement cites a quoted clause; none from general law.
- Every figure is a clause formula applied to evidenced inputs, or is labelled as claimed.
- Notice and time limits are checked against stored dates.
- Nothing is sent, admitted or settled.

## Escalation and approval

Escalate to legal above the configured referral value, where the supplier disputes the facts, where force majeure is raised or where the ladder reaches mediation; to finance for any set-off, credit or write-off; to the category lead for the relationship consequence.

## Worked example

Asked what we can claim for a logistics provider's missed deliveries that stopped a line for two days, the skill reads the delivery record (four late deliveries in the month), quotes the service-level clause and its credit formula, the liability cap, the exclusion of indirect loss and the dispute ladder (written notice, then a meeting within a stated period), computes the service credit from the clause's rate and the recorded late days, shows the line-stoppage cost as the user's claimed figure likely excluded as indirect loss, and sets the first step: written notice under the clause within its time limit, drafted for legal.

## Anti-patterns

- Claiming consequential loss the contract excludes.
- Computing damages from the user's estimate as if it were a clause formula.
- Setting off a claim against an undisputed invoice without the clause.
- Sending a letter before action from the skill.
