---
name: goods-receipt-recording
description: "Carry out the recording of a goods receipt against an order - confirm the order and its lines, take what the receiver reports (quantities, condition, delivery note, date), compare it with the order, record the receipt through the platform's receiving action, and confirm what was recorded and what remains open. Use when asked to book in, receipt or record a delivery. Dormant until a receiving tool exists; until then the skill prepares the receipt and a person records it. Not for a delivery that went wrong (delivery-issue-handling), which this skill hands over to."
---

# Goods Receipt Recording

## Purpose

Make the receipt the truth the invoice match rests on: what arrived, when, in what condition, against which order lines, recorded once by the person who received it, with any shortfall or damage left open rather than papered over.

## When to use and when not

Use when goods or a service have arrived and the user asks to record, book in or receipt them against an order.

Not for handling a short, damaged or wrong delivery beyond recording the facts (delivery-issue-handling), for the invoice match (invoice-query-handling, overbilling-investigation) or for placing the order (purchase-order-placement).

## Inputs and preconditions

- The order reference and the receiver's report: quantities per line, condition, delivery note reference, date received, who received it.
- Evidence the receiver holds (photos, carrier reference), labelled user-supplied.

- Use `tool:open_orders` for the order's lines, expected date and the receipts already recorded against it
- Use `tool:invoice_reconciliation` for invoices already booked against the order, so the receipt lands before the match runs
- Use `tool:get_policy` for the receiving rules (who may record, inspection before receipt, tolerance on over-delivery)
- Use `tool:supplier_lookup` for the supplier on the order
NOT AVAILABLE: record_goods_receipt. the receiving action does not exist yet; the skill prepares the receipt line by line, and a person records it from the receiving screen

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.over_delivery_tolerance_pct`.

## Workflow

1. Find the order and its lines; list receipts already recorded against it so a second receipt is not a duplicate.
2. Take the receiver's report line by line: quantity received, condition (good, damaged, rejected), delivery note reference, date, receiver; nothing is filled in that the receiver did not report.
3. Compare with the order: short (received below ordered), over (above ordered by more than the policy tolerance, or `config.thresholds.over_delivery_tolerance_pct` percent where policy states none), wrong item, damaged; each difference is recorded as a fact on the receipt, not resolved here.
4. Quote the receiving rules: who may record, inspection before receipt, what an over-delivery needs.
5. Record the receipt once through the platform's receiving action under the agent-policy gate, with the receiver's name and date; the action returns the receipt reference.
6. Confirm what was recorded per line, what remains open on the order (quantity still due), and hand any difference to delivery-issue-handling.

## Frameworks and heuristics

- Record what arrived, not what was ordered.
- A damaged unit is received as damaged, not as missing and not as good.
- The receiver's name and the date are part of the record.
- An over-delivery is recorded as received; whether it is accepted or returned is a separate decision.

## Policy hooks (SC-007)

Receiving rules come from `tool:get_policy` and are quoted. The receiving action runs under the agent-policy gate and is recorded with the receiver's name; the skill records nothing by any other route and reports a receipt only with the action's reference.

## Output specification

1. The order and the receipts already recorded.
2. Receipt lines as reported: quantity, condition, note reference, date, receiver.
3. Differences against the order by kind.
4. The receipt record: reference and lines (or the ready-to-record state while the action does not exist).
5. What remains open on the order and the hand-offs.

## Quality bar

- Every quantity and condition is the receiver's report.
- Differences are recorded as facts, never adjusted away.
- The receipt is reported only with the action's reference.
- No receipt is recorded twice for the same delivery note.

## Escalation and approval

Escalate to delivery-issue-handling for any short, over, wrong or damaged line, to quality where inspection is required before receipt, and to finance where an invoice is already booked against the order.

## Worked example

Asked to book in a pallet delivery, the skill finds the order (five hundred ordered, none received), takes the receiver's report (four hundred and eighty good, ten damaged, delivery note reference, today's date, the receiver's name), records the two conditions as two receipt lines, notes ten still due, quotes the policy's inspection rule, and, while the receiving action does not exist, reports the receipt as ready to record for the receiver to enter from the receiving screen, handing the shortfall and the damage to delivery-issue-handling.

## Anti-patterns

- Recording the ordered quantity because the delivery note says so.
- Booking damaged units as good to clear the invoice.
- Recording a receipt the receiver did not report.
- Reporting a receipt as recorded without a reference.
