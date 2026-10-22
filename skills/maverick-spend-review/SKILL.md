---
name: maverick-spend-review
description: Find spend that bypassed the agreed route - invoices with no purchase order line, purchases from suppliers with no contract on record while a contracted supplier exists, spend split below approval limits, buying outside preferred suppliers - from our own invoices, orders, contracts and findings, size it per currency and say who should act. Use when asked about maverick, off-contract, non-compliant or unapproved spend, buying outside the process, invoices without a PO, or compliance to contract across the organisation. Not for checking one purchase against policy (procurement-policy-compliance) or for one invoice dispute (overbilling-investigation).
---

# Maverick Spend Review

## Purpose

Show, from the organisation's own records, where buying bypassed the agreed route and how much it was: invoices that arrived with no order behind them, spend with suppliers that have no contract on record in a category where a contracted supplier exists, purchases that look split to stay under an approval limit, and spend outside preferred suppliers. The review sizes each pattern per currency with the registered figures, names the suppliers as recorded and the buyer ids behind a pattern for the process owner only, and hands each case to the person or skill that acts; it changes nothing and accuses no one.

## When to use and when not

Use when the user asks about maverick, rogue, off-contract, off-process, non-compliant or unapproved spend, invoices with no PO, buying outside preferred suppliers or frameworks, split purchases, or compliance to contract across a period, a category's suppliers or a buyer.

Not for whether one purchase is allowed (procurement-policy-compliance), for one supplier's invoice dispute (overbilling-investigation), for a spend breakdown on its own (spend-analysis) or for contract coverage without the compliance angle (contract-portfolio-review, whose "no contract on record" list this review takes as one input).

## Inputs and preconditions

- The period (default the trailing `config.thresholds.default_period_months` months), the reporting currency, and the scope: the whole organisation, a category's suppliers, a buyer or a business unit the user names.
- The organisation's buying rules from `tool:get_policy` with the query `approval_threshold_policy`: the approval limit and the approving role; preferred-supplier and contract-before-order rules where that policy states them, otherwise "not governed" and the configured defaults as defaults.

Use `tool:build_spend_cube` by supplier, by buyer and by payment terms or status for the period's invoiced spend; `tool:invoice_status` per supplier of interest for the invoices with no order recorded (count and amount per currency) and `tool:invoice_reconciliation` for the open findings; `tool:open_orders` per supplier for its orders with date, total and buyer id; `tool:contract_pipeline` with a long horizon for the register's live contracts and `tool:search_contracts` by supplier to see whether a supplier is contracted; `tool:supplier_lookup` and `tool:supplier_discovery` for preferred flags and the suppliers already used for an item; `tool:similar_items` to find the same thing bought from several suppliers; `tool:check_policy` with a group of orders to test a suspected split; and `tool:run_calculation` with `deal.share_pct` for shares.

NOT AVAILABLE: build_xlsx. Give the registers as text tables.
NOT AVAILABLE: render_chart. Give the pattern totals as a table.

## Workflow

1. Frame. Fix period, currency and scope; read the approval policy once and state the limit, the approver and whether preferred-supplier and contract-before-order rules are governed. The round budget is small: the cube, the pipeline and a few reconciliations are the calls that matter.
2. Spend base. The cube by supplier for the period (share per supplier) and by buyer; shares via `deal.share_pct` only where both figures are read. Excluded currencies are shown as billed.
3. No order behind the invoice. For the top suppliers by spend (at most `config.thresholds.max_suppliers_reconciled`), run invoice_status and take its count and amount of invoices with no order recorded, per currency, as the result states them. An invoice with no order recorded may be a missing order or an unrecorded one: it is a pattern to examine, not a breach.
4. No contract where one exists. From the pipeline and the supplier searches: suppliers with spend above `config.thresholds.uncovered_spend_min` in the reporting currency and no live contract on record, in categories or items where another supplier is contracted (shown by similar_items and the contract search). Spend with them is "off contract on record".
5. Outside preferred. From the lookups: spend with suppliers not flagged preferred for items that a preferred supplier also supplies (discovery), per currency.
6. Suspected splits. Groups of orders from one buyer to one supplier within `config.thresholds.split_window_days` days whose sum exceeds the approval limit while each is below it, taken from open_orders for the supplier (order date, total and buyer id as recorded); test the group with `tool:check_policy` and report its reply as it is. A pattern is "suspected", never "deliberate".
7. Size and rank. Each pattern per currency with its share of the period's spend; rank by amount; name the suppliers as the records hold them; buyer ids go to the process owner, not into a general answer.
8. Actions. Per pattern: the owner (buyer's manager, category lead, finance, master-data) and the route: raise orders retrospectively where policy allows, add the supplier to the contract or re-source (category-strategy, rfx-design), confirm preferred status (supplier-onboarding-checklist), escalate suspected splits to the approver. Nothing is raised, blocked or reported to anyone by the skill.

## Frameworks and heuristics

- Patterns, not verdicts: the records show what bypassed the route; why is a question for the people involved.
- Our gaps first: an unrecorded order, an un-uploaded contract or an unmerged supplier record can look like maverick spend; each pattern says what record would clear it.
- Size per currency, never across; share of spend only from read figures.
- A split is suspected on timing and amounts alone and is tested with the policy tool; the finding is the policy's reply.
- Preferred-supplier and contract-before-order rules are applied only where policy states them; otherwise the pattern is reported without a compliance label.

## Policy hooks (SC-007)

The approval limit, the approving role and any preferred-supplier or contract-before-order rule come from governed policy via `tool:get_policy`; suspected splits are tested with `tool:check_policy` and its reply is quoted as it is. Where a rule is not governed the pattern is reported as a pattern, not a breach. Nothing here raises, blocks or reports.

## Output specification

1. Frame: period, currency, scope, policy limit and approver, which rules are governed.
2. Spend base: total, by supplier and buyer shares, exclusions.
3. Pattern tables: no order on record; off contract on record; outside preferred; suspected splits (with the policy reply); each with amount per currency, share, suppliers and buyers as recorded, and the record that would clear it.
4. Ranked list of the largest patterns.
5. Actions with owners and routes; what the user should check with the people involved.
6. Data gaps: suppliers not reconciled, contracts not on the register, currencies without rates.

## Quality bar

- Every amount comes from the cube or the reconciliation per currency; nothing is summed across currencies or estimated.
- No pattern is called a breach unless a governed rule states the requirement and the policy tool's reply says so.
- Buyers and suppliers are named as recorded, with the record that would clear the pattern.
- Nothing is raised, blocked or sent.

## Escalation and approval

Escalate suspected splits above the approval limit to the approving role named by the policy reply, and large off-contract spend to the category lead. The review is advice; actions run through the owning skills and approvals.

## Worked example

Asked where spend bypassed the process last year, the skill reads the approval policy (limit and Approver), takes the cube by supplier and buyer, reconciles the top suppliers (one with many lines carrying no PO line, summed per currency), finds two suppliers with material spend and no contract on record for items a contracted supplier also supplies, finds no preferred-supplier rule governed, tests one buyer's cluster of orders to a supplier within the window with the policy tool (reply: above the limit in aggregate, Approver needed), ranks the patterns, and lists the owners and the records that would clear each.

## Anti-patterns

- Calling "no order on record" a breach without a governed rule.
- Naming a buyer as deliberate for a split the policy tool has not tested.
- Summing currencies, or taking a corpus-wide count as a category's.
- Treating an un-uploaded contract as no contract.
- Raising orders, blocking suppliers or reporting people from the skill.
