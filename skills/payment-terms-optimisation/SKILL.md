---
name: payment-terms-optimisation
description: Find where payment terms can be improved and what a change is worth - the terms we actually pay on by supplier from invoices, the terms in the contracts and the supplier master, the organisation's standard terms from policy, the value of extending or shortening terms and of an early-payment discount at the governed cost of capital through the registered calculations, and the suppliers where terms should not be pushed. Use when asked about payment terms, days payable, early-payment or prompt-payment discounts, working capital from payables, or extending terms with suppliers. Not for negotiating the change (negotiation-planning) or for a supplier's clearance (supplier-risk-assessment).
---

# Payment Terms Optimisation

## Purpose

Give finance and procurement a defensible view of payment terms: what terms are recorded, where they fall short of the standard, what a change is worth in present value at the governed cost of capital, which early-payment discounts beat that rate, and which suppliers should not be pushed. Every value comes from a calculation on recorded spend and stated rates; the skill proposes, and the negotiation and contract changes go through their own skills.

## When to use and when not

Use when the user asks about payment terms across suppliers or for one supplier, days payable or working capital from payables, whether to take or offer an early-payment discount, what extending or harmonising terms is worth, or which suppliers' terms are below standard.

Not for planning the conversation with the supplier (negotiation-planning), for the contract wording (contract-clause-review), for a supplier's financial clearance (supplier-risk-assessment) or for a full TCO comparison (total-cost-of-ownership, which shares the calculations).

## Inputs and preconditions

- Scope: all suppliers, a supplier or a set; the period and the reporting currency (default the trailing `config.thresholds.default_period_months` months).
- The organisation's standard terms and any rule on prompt payment of small suppliers, from `tool:get_policy`; the cost of capital from the governed policy `negotiation_bounds` (rule `cost_of_capital_apr`) via `tool:get_policy`, or from finance if the policy does not state it; never assumed.
- For a discount question: the discount percentage, the discount window and the net days, from the quote, contract or user.

Use `tool:build_spend_cube` with `by` set to payment_terms for invoiced spend by the terms recorded on invoices, and by supplier for the spend per supplier; `tool:supplier_lookup` for the supplier's recorded currency and profile; `tool:search_contracts` for the payment terms stored on contracts and `tool:extract_clauses` (containing "payment") for the clause wording; `tool:financial_health_score` and `tool:screening_summary` before proposing to extend a supplier's terms; `tool:run_calculation` with `supplier.payment_terms_score` to place terms on a scale, with `finance.carrying_cost` to value a terms change or an early-payment discount (the supplier's spend in the period as the amount, the governed cost of capital as the rate, the change in days; a discount's worth is `deal.scaled_total` of the spend and the discount rate, set against the carrying cost of paying early), and with `deal.pct_change` for the difference between options; `tool:invoice_status` for the payment terms recorded on a supplier's invoices.

Use `tool:build_xlsx` when the user asks for a spreadsheet of the figures, with the tables as the tools returned them.

## Workflow

1. Frame. Fix scope, period and currency; read the standard terms, any small-supplier rule and the cost of capital; say what is not governed.
2. Terms as recorded. From the cube by payment_terms: spend per recorded term with invoice counts, and the share with no term recorded. For each supplier in scope: spend from the cube by supplier, contract terms from search_contracts, and the terms recorded on their invoices from invoice_status; a gap between contract and invoice terms is a finding. Suppliers below `config.thresholds.min_annual_spend_for_review` in the period are listed, not analysed.
3. Position. For each supplier in scope run `supplier.payment_terms_score` with the recorded days against the organisation's scale; terms at or below `config.thresholds.short_terms_days` days are short, above `config.thresholds.max_terms_days` days are long and need the supplier's financial health read before any further extension.
4. Value. For a proposed change (for example to the standard terms), run `finance.carrying_cost` on the supplier's spend in the period at the governed cost of capital for the change in days between current and proposed terms, and report that figure as the annual value of the change, with `deal.pct_change` for the movement between options; never work the value out by hand. For an early-payment discount, compare `deal.scaled_total` of the spend and the discount rate with `finance.carrying_cost` of the spend over the days paid early, and report whether taking it beats paying at net days, noting whether AP can reliably pay inside the discount window as the user states it.
5. Who not to push. A supplier with a flagged or stale financial health record, a small supplier under a prompt-payment rule, a sole source, or a supplier in a flagged screening is excluded from extension and listed with the reason.
6. Harmonisation. Where one supplier is paid on several terms, or terms differ from the contract, propose the single term with the value of aligning to it.
7. Plan. Rank opportunities by present value with the supplier, current and proposed terms, value, risk notes and the route (contract change at renewal via contract-clause-review and renewal-and-exit-strategy; negotiation via negotiation-planning; policy exception via `tool:check_policy`).

## Frameworks and heuristics

- Value is present value at one governed rate; a terms change on spend that is not recurring is worth once, not every year.
- An early-payment discount is compared with the cost of capital by the calculation, not by a rule of thumb; whether AP can reliably pay inside the window decides whether it is worth anything.
- Extending terms moves cost to the supplier; a supplier in difficulty or without alternatives is a risk, not an opportunity.
- Recorded terms on invoices are what was paid on; contract terms are what was agreed; the gap is the first fix.
- Shares and totals use the cube's currency rule; excluded currencies are shown as billed.

## Policy hooks (SC-007)

Standard terms, prompt-payment rules and the cost of capital come from governed policy via `tool:get_policy`; a proposed term outside the standard, or a discount offered to a supplier, is a policy exception checked with `tool:check_policy`. Nothing here changes a term; the negotiation and contract skills carry the change.

## Output specification

1. Frame: scope, period, currency, standard terms and cost of capital with their sources or "not governed".
2. Terms table: spend and invoices by recorded term; share with no term recorded.
3. Supplier table: spend, invoice terms, contract terms, gap, position on the scale, financial-health status.
4. Opportunities: supplier, current and proposed terms, present-value difference, discount verdicts, risk notes, route.
5. Exclusions with reasons.
6. Open items and hand-offs.

## Quality bar

- Every value comes from `finance.carrying_cost`, `deal.scaled_total` and `deal.pct_change` on recorded spend at a sourced cost of capital; no hand arithmetic.
- No extension is proposed for a supplier whose financial health is flagged, stale or not performed without saying so.
- Recorded terms and contract terms are both shown; neither is assumed from the other.
- Standard terms and the cost of capital carry their source.

## Escalation and approval

Escalate to finance when the cost of capital or standard terms are not governed, and to the category lead for any extension on a sole source or a supplier in difficulty. Terms change through negotiation and contract; the plan is advice.

## Worked example

Asked what moving suppliers to the standard terms would be worth, the skill reads the standard and the cost of capital from policy, takes spend by recorded term and by supplier from the cube, finds most suppliers on short terms with one large supplier paid on terms shorter than its contract, runs the ownership calculation on each supplier's spend with current and standard terms, ranks the present-value differences, excludes a supplier whose financial health is not performed, flags the contract gap as the first fix, and hands the two largest moves to negotiation-planning with the policy exception route for one supplier below the standard.

## Anti-patterns

- Valuing a terms change by hand or with an assumed cost of capital.
- Taking a discount's rate at face value without the calculation and the AP window.
- Pushing terms on a supplier whose financial health is unknown.
- Reading contract terms as what was paid, or invoice terms as what was agreed.
- Changing a term from this skill.
