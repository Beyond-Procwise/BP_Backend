---
name: invoice-query-handling
description: "Answer a question about where an invoice stands - on hold, blocked, unmatched, overdue, rejected, awaiting receipt, paid - from the invoice, order and receipt records and the findings against it, say what is holding it and who clears it, and draft the reply for the supplier or the requester. Use when a supplier chases payment, an invoice is stuck, or someone asks why an invoice has not been paid. Not for whether the invoice overcharges (overbilling-investigation) or for the payment terms themselves (payment-terms-optimisation)."
---

# Invoice Query Handling

## Purpose

Turn 'why has this not been paid?' into a factual status with the blocker and its owner, in words the supplier or the requester can act on. The skill reads and drafts; it releases no hold, approves no payment and promises no date.

## When to use and when not

Use when the question is the status of one or more invoices, why one is on hold or blocked, when it will be paid, what a supplier must do to be paid, or what a rejection means.

Not for a price or quantity overcharge (overbilling-investigation), for terms negotiation (payment-terms-optimisation), for a delivery dispute behind the hold (delivery-issue-handling) or for the finance payment run itself.

## Inputs and preconditions

- The invoice (number, supplier and amount or date) or the supplier whose invoices are in question.
- Who is asking (supplier, requester, finance) so the reply fits.

- Use `tool:invoice_reconciliation` for the invoice, its order and receipt match, open findings, hold reasons and the lines affected
- Use `tool:supplier_lookup` for the supplier's record and status (never bank details)
- Use `tool:search_contracts` for the agreement's payment terms
- Use `tool:extract_clauses` for the payment, invoicing and dispute clauses, quoted
- Use `tool:get_policy` for the invoice rules (no-PO-no-pay, receipt before payment, tolerance, who clears a hold)
NOT AVAILABLE: release_hold. a hold is cleared by the owner the policy names; the skill says who and what is needed
NOT AVAILABLE: send_message. the reply is drafted; a person sends it

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.overdue_grace_days`.

## Workflow

1. Find the invoice or the supplier's invoices in the period; state each invoice's status as the record holds it.
2. For each one not paid: the blocker (no order, no receipt, quantity or price finding, missing fields, duplicate suspicion, awaiting approval) from the reconciliation; the rule behind it quoted from policy; the owner who clears it.
3. Terms: the agreement's payment terms quoted; the due date only if the record and the terms give one; an invoice past its terms by more than `config.thresholds.overdue_grace_days` days is overdue and said so.
4. What the supplier must do (resubmit with the order number, issue a credit note, provide the delivery note) and what we must do (record the receipt, clear the finding, approve), each with an owner.
5. Draft the reply for the asker: status, blocker, what is needed from whom; no payment date promised unless the record holds a scheduled one; hand the text to supplier-communication-drafting where tone matters.
6. Patterns: several invoices stuck for the same reason are reported as one finding for the process owner.

## Frameworks and heuristics

- A status is the record's word; a blocker is a rule plus a fact, with an owner.
- No promise of a date the record does not hold.
- The supplier's fix and our fix are listed separately; most holds need one of each.
- Repeated holds for one reason are a process problem, not a supplier problem.

## Policy hooks (SC-007)

Invoice rules and who clears each kind of hold come from `tool:get_policy` and are quoted. The skill releases nothing, approves nothing and promises nothing; the reply states facts and asks for what is needed. No bank details or personal contacts appear in a reply.

## Output specification

1. Invoice table: number, supplier, amount, date, status, blocker, rule, owner.
2. Terms and overdue position where the record supports it.
3. Actions: the supplier's and ours, with owners.
4. Draft reply for the asker.
5. Patterns for the process owner.

## Quality bar

- Every status and blocker is from the reconciliation; every rule is quoted.
- No payment date is promised that the record does not hold.
- No bank details or personal contacts in the reply.
- Nothing released, approved or sent.

## Escalation and approval

Escalate to finance to clear a hold or confirm a payment date, to the buyer to record a receipt or clear a finding, to overbilling-investigation where the finding is a price or quantity overcharge, and to the process owner for patterns.

## Worked example

Asked why Acme is chasing three invoices, the skill reads them: one is on hold with a quantity finding against the order (buyer to clear or dispute), one has no order on record (supplier to resubmit quoting the order number under the no-PO-no-pay rule, quoted), and one is approved and inside terms; it drafts a reply stating each status and what is needed from whom, without a payment date for the two on hold, and notes that no-order invoices from this supplier are a recurring pattern.

## Anti-patterns

- Promising a payment date to end the chase.
- Releasing a hold or approving from the skill.
- Quoting bank details back to the supplier.
- Treating a price finding as a query to answer rather than an investigation to route.
