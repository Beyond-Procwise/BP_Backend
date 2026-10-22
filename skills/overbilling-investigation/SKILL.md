---
name: overbilling-investigation
description: Investigate a suspected overbilling by a supplier - invoices above the purchase order, unit prices above the agreed price, quantities billed above what was ordered or received, duplicate or unexpected charges - from our own PO, invoice and receipt records, the triage findings already raised, the contract's price and payment clauses, and the supplier's price variance, and prepare the evidence, the amount in question and the query to the supplier. Use when asked whether a supplier has overcharged us, to check an invoice against its PO, or to chase a credit note. Not for approving or disputing payment (a person does that under policy) or for judging a price against the market (price-benchmarking).
---

# Overbilling Investigation

## Purpose

Put a suspected overcharge on a sound footing before anyone writes to the supplier: what was ordered, what was received, what was invoiced, what the contract says, what our triage already found, and the amount in question with how it was computed by the registered calculations. The skill assembles evidence and drafts the query; the hold, dispute, payment or credit note is a person's decision under policy.

## When to use and when not

Use when the user asks whether a supplier has overbilled or overcharged, wants an invoice checked against its PO or contract, asks about a three-way-match exception, a price or quantity mismatch, a duplicate charge, or wants a credit note chased.

Not for judging whether a price is fair against the market (price-benchmarking), for a supplier's general performance (supplier-performance-review), for the contract wording review beyond the price and payment clauses (contract-clause-review), or for the approval of a payment hold or dispute (procurement-policy-compliance and the approvals route).

## Inputs and preconditions

- The supplier and, where known, the invoice number, PO number or deal; the period if several invoices are in scope.
- The agreed price basis: the PO lines, a contract price schedule, or a quote; the contract's price-adjustment and payment clauses if a change is claimed.
- Any goods-receipt or delivery evidence the user holds beyond the records, labelled user-supplied.

Call `tool:invoice_reconciliation` FIRST with the supplier (and the invoice_id or po_id if given): one call returns every invoice line matched to its PO line with the differences by kind and summed per currency, the receipts recorded per PO and the open findings with observed and expected values; it is the reconciliation table. Never call get_corpus_facts in this skill: its counts are corpus-wide and would be read as this supplier's. Do not call list_calculations; call get_policy at most once for an overbilling rule and carry on if none governs. Call `tool:check_policy` only with an amount taken from the reconciliation's differences_by_currency, never with a figure of your own; if the reconciliation returns no lines, widen the period or ask for the invoice numbers rather than reaching for other tools. Use `tool:calc_supplier_scorecard` only for the supplier's overall price variance in context, `tool:search_contracts` and `tool:extract_clauses` (one `containing` word per call, at most two: "price" then "adjust") for the agreed basis and any adjustment right, `tool:obligation_tracker` for dated price-review or escalation clauses, `tool:similar_items` for what the same item cost from this and other suppliers as context, `tool:run_calculation` with `ppv.purchase_price_variance` on the matched lines for the amount above the PO price, with `deal.pct_change` for the size of the difference, and with `duplicate_invoice.pair_score` where a duplicate is suspected, and `tool:check_policy` for who may hold or dispute payment at the amount.

Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated. Use `tool:build_xlsx` when the user asks for a spreadsheet of the figures, with the tables as the tools returned them.

## Workflow

1. Scope. Identify the supplier, the invoices and the POs in question from the records; where the user names none, take the open findings against the supplier as the starting list. Say what the records hold and what they do not (no receipt recorded is not evidence of non-delivery).
2. Agreed basis. Establish the price and quantity basis per line: the PO line, or the contract price schedule, or the quote; quote the price-adjustment clause word for word if the supplier claims one; note a basis the records do not hold as "not on record".
3. Reconcile. Start from the result's `kinds` counts and `differences_by_currency` totals: they cover every line, including lines beyond the ones shown; state them first (how many lines matched, how many priced or quantified above the PO, how many had no PO line) and only then go line by line: ordered quantity and price, receipts recorded on the PO (quantities are never recorded), invoiced quantity and price, the difference and its kind (price above PO, quantity above ordered, no PO line, currency differs, ambiguous PO lines). The per-currency totals in the result are the amounts in question; use `deal.share_pct` for their size against the invoiced total; a line the result could not match is "unmatched", not a difference.
4. Duplicates. Where two invoices look alike, run `duplicate_invoice.pair_score` on them and report the score with the configured `config.thresholds.duplicate_score_min`; a candidate below it is not called a duplicate.
5. Findings on record. List the triage findings already raised for these documents with their kind and status; a finding is evidence that our process caught something, not a verdict against the supplier.
6. Amount in question. Sum only the differences the calculations produced, per currency, and state the share of the invoiced total; below `config.thresholds.materiality_min` in the invoice currency, say the amount is immaterial under this skill's default and leave the decision to the user.
7. Adjustment rights. If the contract allows a price change (index, escalation, review), say whether the claimed change follows the clause's own terms on the evidence (a loaded index via `tool:commodity_index_lookup` where the clause names one); otherwise say the clause does not provide for it or the terms are not on record.
8. Query and route. Draft the query to the supplier: the lines, the differences, the basis, the documents, and the request (credit note, corrected invoice, or evidence). Report the approval route for a payment hold or dispute at the amount from `tool:check_policy`; nothing is sent, held or disputed from the skill.

## Frameworks and heuristics

- Three-way match in words: ordered, received, invoiced; a difference needs two of the three on record to be called one.
- Price above basis and quantity above ordered are different disputes with different evidence; keep them apart in the table.
- A receipt we did not record is our gap, not the supplier's overbilling; say so and ask for the delivery evidence.
- A supplier's right to adjust price lives in the clause; the clause is quoted, never summarised.
- Immaterial differences are reported, not pursued by default; the threshold is a default, not policy.

## Policy hooks (SC-007)

Who may hold, dispute or approve payment at the amount comes from governed policy via `tool:check_policy`, quoted as replied. Materiality and duplicate-score defaults in config are not policy. The skill sends nothing to the supplier and changes no document.

## Output specification

1. Scope: supplier, documents, period, what the records hold and lack.
2. Agreed basis per line with its source (PO, contract schedule, quote) and any adjustment clause quoted.
3. Reconciliation table: line, ordered, received, invoiced, difference, kind, amount, calculation used.
4. Duplicate candidates with scores.
5. Findings on record.
6. Amount in question per currency, share of invoiced total, materiality note.
7. Query to the supplier (text) and the approval route for any hold or dispute; what to ask the user for.

## Quality bar

- Every amount in the reconciliation comes from a calculation result or a recorded document; no total is worked out by the model.
- No line is called a difference without the basis and the invoice both on record; unmatched lines are listed as unmatched.
- A missing receipt is never treated as proof of non-delivery.
- The adjustment clause, when relied on, is quoted with its reference.

## Escalation and approval

Escalate to the category lead when the amount in question is above the materiality default on a supplier with repeated findings, and to finance for any payment hold. The query letter is a draft for a person to send; the hold, dispute or credit note follows the approval route the policy reply states.

## Worked example

Asked whether Clearwater Logistics has been overbilling, the skill takes its ten open findings (quantity invoiced above PO on several lines, two invoices above PO total), reads the matched lines and the price-variance calculation (unit prices equal to the PO: no price component), reconciles quantities (invoiced above ordered on four lines; no receipts recorded, so received is not on record), finds no price-adjustment clause stored for the live contract, sums the four quantity differences from the calculation per currency, drafts the query asking for delivery evidence and a credit note for any quantity not delivered, and reports the approval route for holding the two invoices.

## Anti-patterns

- Calling the supplier's invoice an overcharge when the agreed basis is not on record.
- Adding up differences by hand or across currencies.
- Treating "no receipt recorded" as "not delivered".
- Summarising the price-adjustment clause instead of quoting it.
- Sending the query, holding payment or raising a dispute from the skill.
