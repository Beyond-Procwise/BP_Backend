---
name: purchase-order-placement
description: "Carry out the placement of a purchase order that has been approved - confirm the requisition, the supplier's status, the agreement or quote the price rests on and the approval on record, assemble the order lines exactly as approved, place the order through the platform's ordering action, and confirm the order reference and what it commits. Use when asked to place, raise or release an order that is ready. Dormant until an ordering tool exists; until then the skill prepares the order and a person places it. Not for whether the purchase is allowed (procurement-policy-compliance) or which channel to use (buying-channel-guidance)."
---

# Purchase Order Placement

## Purpose

Turn an approved requisition into an order without drift: the same supplier, lines, prices and terms the approval covered, placed once, recorded with a reference, and confirmed only as the ordering action returns it.

## When to use and when not

Use when the purchase is approved and the user asks to place, raise, release or confirm the order with the supplier.

Not for the compliance ruling (procurement-policy-compliance), the channel (buying-channel-guidance), a retrospective order (policy-exception-request) or an urgent buy still being sourced (urgent-sourcing).

## Inputs and preconditions

- The requisition or approved request: supplier, lines (item, quantity, unit price, currency), delivery date and place, terms.
- The approval record, as the policy replied.
- The quote or agreement the prices rest on.

- Use `tool:check_policy` for the approval the order needs, as replied; an order is placed only when it says approved or the approval record is on file
- Use `tool:supplier_lookup` for the supplier's status, currency, Incoterms and lead time on record
- Use `tool:screening_summary` for the supplier's screening; an unscreened supplier gets no order
- Use `tool:search_contracts` for the agreement the prices and terms come from
- Use `tool:extract_clauses` for the price and payment terms clause, quoted, when the order sits under an agreement
- Use `tool:open_orders` for the supplier's open orders, to spot a duplicate of the same lines
- Use `tool:supplier_items` for the last price paid for each line, as a check against the quote
NOT AVAILABLE: create_purchase_order. the ordering action does not exist yet; the skill assembles the order exactly as approved, checks it, and a person places it from the ordering screen
NOT AVAILABLE: send_message. the order is not sent by the skill; the ordering action, when it exists, transmits it under the policy gate

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.duplicate_window_days`, `config.thresholds.price_variance_alert_pct`.

## Workflow

1. Confirm the preconditions: approval as `tool:check_policy` replied (or the approval record on file), supplier status and screening, the agreement or quote the prices rest on with its validity date.
2. Assemble the order lines exactly as approved: item, quantity, unit price, currency, delivery date and place, Incoterm, payment terms from the agreement; any difference from the approved request stops the placement and goes back to the approver.
3. Check for a duplicate: the supplier's open orders with the same lines within `config.thresholds.duplicate_window_days` days; a match is reported before placing.
4. Compare each line's price with the last price paid and the agreement's price; a line above either by more than `config.thresholds.price_variance_alert_pct` percent is flagged to the user before placing.
5. Place the order once through the platform's ordering action, under the agent-policy gate, with the user's name; record the order reference, the lines, the total and the terms.
6. Confirm: the reference, supplier, total, delivery date and terms as the action returned them; what the order commits; what happens next (acknowledgement, receipt, invoice match).

## Frameworks and heuristics

- The order is the approval, line for line; a changed line is a new approval.
- Price validity is a date: an expired quote is not a price.
- One order per approved request; a duplicate is cheaper to catch than to cancel.
- The confirmation says what the action returned, never what was intended.

## Policy hooks (SC-007)

The approval comes from `tool:check_policy` as replied and is the condition of placing. The ordering action runs under the agent-policy gate and is recorded with the user's name. Nothing is placed by any other route and no order is reported placed without the action's reference.

## Output specification

1. Precondition table: approval, supplier status and screening, price source and validity; stop reasons.
2. The order lines as assembled, with the approved request beside them.
3. Duplicate and price checks with their evidence.
4. The placement record: reference, total, terms, date (or the ready-to-place state while the action does not exist).
5. What happens next and who watches it.

## Quality bar

- Every line matches the approved request exactly.
- Every price has a source and a validity date.
- The placement is reported only with the action's reference.
- Nothing is placed for an unscreened supplier.

## Escalation and approval

Escalate to the approver for any line change, to compliance for a screening gap, to the category lead for a price variance above the configured share, and to finance where the terms differ from the agreement.

## Worked example

Asked to place the approved order for five hundred pallets, the skill confirms the approval as check_policy replied, the supplier's clear screening, the quote's validity, assembles the one line at the quoted price with the agreement's payment terms and the site's delivery date, finds no open duplicate and a price within the configured variance of the last paid, and, while the ordering action does not exist, reports the order as ready to place with the lines and references for a person to place from the ordering screen.

## Anti-patterns

- Changing a quantity or price at placement.
- Placing on an expired quote.
- Reporting an order as placed without a reference.
- Placing a second order for the same lines.
