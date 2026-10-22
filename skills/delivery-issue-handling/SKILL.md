---
name: delivery-issue-handling
description: "Work through a delivery that went wrong - late, short, damaged, wrong item, over-delivered, rejected - from the order and receipt on record, the contract's delivery, acceptance and remedy clauses, the supplier's delivery history and the invoice position, to the steps: what to record, what to hold, what to ask the supplier for, and whether a claim or a finding follows. Use when goods or a service did not arrive as ordered. Not for chasing a delivery that is simply awaited (supplier-communication-drafting) or for the formal claim (dispute-and-claims)."
---

# Delivery Issue Handling

## Purpose

Get a failed delivery handled in the right order: evidence first, hold the payment where the contract allows, tell the supplier what is required, and route anything larger to the right place. The skill reads and advises; it records no receipt, holds no invoice and rejects no goods.

## When to use and when not

Use when a delivery is late, short, damaged, wrong, over-delivered, failed inspection or was rejected, and the user asks what to do, what we are entitled to, or how to record it.

Not for an urgent replacement (urgent-sourcing), for the money recovery once the facts are settled (dispute-and-claims), for the invoice overcharge itself (overbilling-investigation) or for the supplier's trend over time (supplier-performance-review).

## Inputs and preconditions

- The order (number or supplier and item) and what went wrong, in the user's words.
- What was received, when, and any evidence (photos, delivery note), labelled user-supplied.

- Use `tool:invoice_reconciliation` for the order, its lines, what was invoiced and any open finding for the supplier
- Use `tool:search_contracts` for the agreement with the supplier
- Use `tool:extract_clauses` for the delivery, acceptance, inspection, rejection, remedy and risk-transfer clauses, quoted
- Use `tool:calc_supplier_scorecard` for the supplier's on-time record and open findings (quality is not recorded there: say so)
- Use `tool:get_policy` for the goods-receipt and inspection rules and who may reject
- Use `tool:check_policy` where a replacement order or a credit commits or releases spend
- Use `tool:open_orders` for the order's stored expected delivery date and the receipts recorded against it ('not recorded' is said as such)
NOT AVAILABLE: record_goods_receipt. the receipt is recorded by the user in the receiving system; the skill says what to record
NOT AVAILABLE: hold_invoice. an invoice hold is finance's act; the skill says whether the contract supports one



## Workflow

1. Frame: order, item, what was ordered versus what arrived, when, and the evidence.
2. Read the order and invoice position: lines, quantities, whether an invoice is already in, open findings.
3. Quote the contract's delivery, inspection, rejection and remedy clauses: the inspection window, the rejection notice period, the remedy (replace, credit, repair), who bears freight on return, where risk passed. Where no contract is stored, policy or the order's terms govern and the gap is named.
4. Classify the issue (late, short, damaged, wrong, over, quality) and the remedy the clause gives for it. The rejection notice period is read from the clause; where no clause is stored, no window is assumed: notify the supplier now, say the contractual window is unknown, and ask legal.
5. Steps in order: record what was received (quantities, condition, photos); notify the supplier within the window with the remedy asked (hand the text to supplier-communication-drafting); whether the invoice may be held under the clause (finance's act); replacement route if needed (urgent-sourcing); what becomes a finding on the supplier's record.
6. Where a loss beyond the goods arises (line stopped, re-work), hand to dispute-and-claims with the facts gathered.

## Frameworks and heuristics

- Evidence before everything: quantities, condition, time, photos, delivery note.
- The inspection and rejection windows are real deadlines read from the clause; a late rejection may be an acceptance, and an unknown window is a reason to notify now, not to assume days.
- Hold only what the contract allows; the undisputed part of an invoice is usually still due.
- Over-delivery is the supplier's problem to collect, not ours to pay for, unless accepted.
- A single failure gets fixed; a pattern gets the scorecard and supplier-development-plan.

## Policy hooks (SC-007)

Receipt, inspection and rejection authority come from `tool:get_policy` and are quoted. A replacement order or a credit that commits or releases spend is checked with `tool:check_policy` and reported as replied. The skill records nothing, holds nothing and rejects nothing; it says what to do and who does it.

## Output specification

1. Frame: order, item, ordered versus received, dates, evidence.
2. Contract position: clauses quoted; windows and remedies; gaps named.
3. Issue class and the remedy the contract gives.
4. Steps in order with deadlines, owners and the text to send (drafted elsewhere).
5. Invoice position: what may be held under the clause, what is still due.
6. Follow-ups: finding on the supplier, claim, replacement.

## Quality bar

- Quantities and dates come from the order and receipt records or are marked user-supplied.
- Every remedy cited is a quoted clause or a policy rule; none from general practice unless the records are silent, and then it is labelled.
- Deadlines are computed only from a stated window and a stated receipt date.
- Nothing is recorded, held or rejected by the skill.

## Escalation and approval

Escalate to quality for an inspection failure, to finance for an invoice hold, to legal where the supplier disputes the rejection, and to the category lead where the supplier's record shows a pattern.

## Worked example

Told that a pallet order arrived two hundred short and ten damaged, the skill reads the order and the invoice already received for the full quantity, quotes the delivery and rejection clauses (inspection window, remedy by replacement or credit, supplier collects rejects), lists the steps: record the received and damaged quantities today, notify the supplier within the window asking for replacement of the shortfall and collection of the damaged pallets, finance holds only the short and damaged part of the invoice under the clause, and the shortfall becomes a finding on the supplier's record.

## Anti-patterns

- Rejecting goods after the inspection window without noting the risk.
- Holding the whole invoice when only part is disputed.
- Paying for an over-delivery nobody accepted.
- Recording a receipt or holding an invoice from the skill.
