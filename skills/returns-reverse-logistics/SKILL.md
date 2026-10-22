---
name: returns-reverse-logistics
description: "Handle returned goods from receipt to disposition - why they came back, what the contract and the supplier's return terms allow, the inspection and grading, the disposition per grade (restock, return to supplier, rework, resell, recycle, scrap), the credit or claim due, and the cost of the return flow - from the returns data the user supplies, the agreements on record and the records of what was bought. Use when asked how to process returns, what to do with returned stock, or how to recover value from it. Dormant until returns data can be read. Not for a single bad delivery (delivery-issue-handling) or dead stock that never moved (slob-management)."
---

# Returns and Reverse Logistics

## Purpose

Make returns a flow with rules: each return graded, each grade dispositioned by its value and the terms, the credit claimed where due, the cost known, and the causes fed back to whoever can stop them.

## When to use and when not

Use when goods come back (from customers, sites or to suppliers) and the question is process, disposition, credit or cost.

Not for a bad delivery on receipt (delivery-issue-handling), for the claim's legal side (dispute-and-claims), for stock that never moved (slob-management) or for the warehouse space it needs (warehouse-layout-design).

## Inputs and preconditions

- The returns: item, quantity, reason, condition, source (customer, site), date, from the user.
- Grading rules and disposition options the organisation uses, from policy or the user.

- Use `tool:search_contracts` for the supplier's agreement with its return, warranty and restocking terms
- Use `tool:extract_clauses` for the return, warranty, restocking-fee and credit clauses, quoted
- Use `tool:supplier_items` for what we bought of the item, at what price, from whom
- Use `tool:invoice_status` for the invoices the returned goods were billed on, for the credit note to match
- Use `tool:run_calculation` with `deal.scaled_total` for a return's value at cost and a disposition's recovery, `deal.share_pct` for a reason's share of returns and `finance.carrying_cost` for the cost of holding returns pending disposition
- Use `tool:get_policy` for the returns rules (grading, who authorises scrap, credit handling)
- Use `tool:check_policy` where a disposition commits or writes off value above a band
NOT AVAILABLE: query_wms. no warehouse connection; returns data is the user's
NOT AVAILABLE: record_goods_receipt. the return receipt is recorded by a person (goods-receipt-recording when it exists)

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.scrap_approval_value`.

## Workflow

1. Frame: the returns as supplied; the rules from policy; the agreements' return terms quoted for supplier returns.
2. Reasons: each return's reason and its share via the registry; reasons that point at a supplier (quality) versus at us (over-ordering, wrong item) versus the customer.
3. Grade: per the rules (A resaleable, B rework, C recycle or scrap); the grader is a person; the skill records the grade given.
4. Disposition per grade: restock (back to available stock), return to supplier under the quoted terms (credit, restocking fee), rework at the user's cost, resell, recycle, scrap; recovery per disposition via the registry; a disposition above `config.thresholds.scrap_approval_value` in value needs the approver.
5. Credit and claim: the credit due under the terms against the invoices the goods were billed on; a claim beyond credit goes to dispute-and-claims.
6. Cost of the flow: carrying cost pending disposition via the registry; the plan per return; the causes for supplier-development-plan or inventory-policy-setting.

## Frameworks and heuristics

- Grade first, disposition second; an ungraded return is a pile.
- The return terms were agreed when we bought; read them before we send anything back.
- Every week pending disposition costs money; the carrying cost is the clock.
- The reason code is the most valuable byproduct of a return.

## Policy hooks (SC-007)

Grading, scrap authority and credit handling come from `tool:get_policy` and are quoted; a write-off or disposal above the configured value goes through `tool:check_policy`. Nothing is scrapped, credited, sent back or restocked by the skill.

## Output specification

1. Returns table with reasons and shares.
2. Grades and dispositions per return with recovery via the registry.
3. Credits due under the quoted terms, matched to invoices; claims for dispute-and-claims.
4. Cost of the flow; approvals needed; owners.
5. Causes fed back with their owners.

## Quality bar

- Every disposition rests on a quoted term or the user's option.
- Values, recoveries, shares and carrying cost come from the registry.
- Grades are a person's; the skill records them.
- Nothing is scrapped, credited or sent.

## Escalation and approval

Escalate to the approver for disposals above the configured value, to finance for credits, to dispute-and-claims for claims beyond credit, and to supplier-development-plan for supplier-caused returns.

## Worked example

Given a month of returns from the sites, the skill tables the reasons (most are over-ordering, a fifth are quality), quotes the supplier's return clause for the quality returns (credit less a restocking fee), records the grades the site gave, dispositions the A grade to restock, the quality returns to the supplier for credit matched to the invoices, and the rest to scrap with the one line above the approval value flagged, computes recovery and carrying cost via the registry, and sends the over-ordering cause to inventory-policy-setting.

## Anti-patterns

- Sending goods back without the return terms.
- Scrapping a return a clause would credit.
- Grading from the skill.
- Issuing a credit from the skill.
