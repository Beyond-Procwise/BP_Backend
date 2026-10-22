---
name: p2p-exception-handling
description: "Work an exception in the purchase-to-pay flow to resolution - an invoice that does not match its order or receipt, a receipt with no order, an order with no receipt and an invoice waiting, a duplicate suspicion, a blocked payment - from the records and the findings, deciding which fact must change (the order, the receipt, the invoice or nothing), who changes it, and what the supplier must be told. Use when asked to clear, resolve or work a match exception. Dormant pending review; the read side runs today through overbilling-investigation and invoice-query-handling. Not for an overcharge investigation in depth (overbilling-investigation)."
---

# Purchase-to-Pay Exception Handling

## Purpose

Resolve exceptions by fixing the right record once: identify which of order, receipt and invoice is wrong, route the correction to its owner, and keep the supplier informed; the skill decides nothing on its own and changes no record.

## When to use and when not

Use when a match exception, blocked invoice or duplicate suspicion needs to be worked to a resolution rather than just explained.

Not for investigating an overcharge (overbilling-investigation), for a status answer (invoice-query-handling), for recording a receipt (goods-receipt-recording) or for placing an order (purchase-order-placement).

## Inputs and preconditions

- The exception: invoice, order or receipt reference and the finding.
- What the user knows about the delivery or the agreement, labelled user-supplied.

- Use `tool:invoice_reconciliation` for the line matches and findings behind the exception
- Use `tool:invoice_status` for the invoice's status, due date and open findings
- Use `tool:open_orders` for the order's lines, invoiced balance, receipts and buyer id
- Use `tool:duplicate_invoices` for invoices from the same supplier with the same amount within a window, when one exists
- Use `tool:search_contracts` for the agreement's price and tolerance terms
- Use `tool:extract_clauses` for the price, tolerance and invoicing clauses, quoted
- Use `tool:get_policy` for the tolerance and exception rules (who clears which kind, within what limits)
- Use `tool:check_policy` where the resolution commits or releases spend
NOT AVAILABLE: duplicate_invoices. not built yet; until then duplicate suspicion comes from invoice_status by scanning the supplier's invoices for the same amount and date, labelled a suspicion
NOT AVAILABLE: release_hold. finance releases a hold; the skill says what must be true first

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.pattern_window_days`.

## Workflow

1. Read the exception: the finding kind (price above order, quantity above ordered or received, no order line, duplicate suspicion, no receipt) and the three records behind it.
2. Quote the rules: tolerance, who clears each kind, limits.
3. Decide which record is wrong, from evidence: the order (price or quantity agreed differently, per the agreement or the buyer), the receipt (not recorded or recorded short, per the receiver), the invoice (over-billed, per the agreement), or none (within tolerance). Each decision names its evidence.
4. Route the correction: buyer amends the order through approval, receiver records the receipt, supplier issues a credit note or corrected invoice, finance clears within tolerance; each with the role responsible.
5. Supplier communication: what they must do, drafted for supplier-communication-drafting; nothing sent.
6. Pattern: the same exception kind repeated for a supplier or buyer within `config.thresholds.pattern_window_days` days is a finding for the process owner.

## Frameworks and heuristics

- Three records, one truth: decide which one is wrong before anyone changes anything.
- Within tolerance is cleared, not investigated.
- The supplier fixes the invoice; we fix the order and the receipt; nobody fixes the other side's record.
- A duplicate suspicion blocks payment until a person confirms it.

## Policy hooks (SC-007)

Tolerance and clearing rules come from `tool:get_policy` and are quoted. A resolution that commits or releases spend goes through `tool:check_policy` as replied. The skill changes no record, releases no hold and sends nothing.

## Output specification

1. The exception and the three records behind it.
2. Rules quoted.
3. The decision (which record is wrong) with evidence, or within tolerance.
4. Corrections routed to roles; the supplier's action drafted.
5. Patterns for the process owner.

## Quality bar

- Every decision names the record and the evidence.
- No record is changed and no hold released by the skill.
- Within-tolerance cases are cleared, not escalated.
- The supplier is told only what the records support.

## Escalation and approval

Escalate to the buyer for order changes, to the receiver for receipts, to finance for holds and tolerances, to overbilling-investigation where the pattern suggests systematic over-billing, and to the process owner for repeated exceptions.

## Worked example

Asked to clear a blocked invoice, the skill reads the finding (quantity invoiced above ordered), the order (five hundred), the receipt (five hundred and twenty recorded by the site) and the invoice (five hundred and twenty), quotes the tolerance rule, decides the order is the record that differs because the site accepted an over-delivery, routes an order amendment to the buyer through approval, drafts nothing for the supplier since the invoice matches the receipt, and notes this is the third over-delivery from this supplier this quarter.

## Anti-patterns

- Changing the receipt to match the invoice.
- Investigating a case within tolerance.
- Telling the supplier to credit what we actually received.
- Releasing a hold from the skill.
