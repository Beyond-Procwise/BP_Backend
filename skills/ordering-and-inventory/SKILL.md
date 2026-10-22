---
name: ordering-and-inventory
description: "Advise on how much and how often to order an item - the demand and order history on record, the supplier's minimum order quantity, lead time and price breaks from quotes and contracts, the stock and consumption the user holds, and the trade-off between order cost, holding cost and stock-out risk - and on arrangements such as blanket orders, scheduled deliveries, consignment or vendor-managed stock. Use when asked about order quantities, reorder points, safety stock, MOQs, or ordering arrangements. Not for a live stock-out (urgent-sourcing) or for pooling demand across sites (demand-aggregation)."
---

# Ordering and Inventory

## Purpose

Give a sensible ordering pattern from the facts available: what we actually consume per the records, what the supplier's terms allow, what the user knows about stock, and the costs the organisation has set for ordering and holding. Every parameter comes from a record, a clause, the user or policy; the skill never invents a lead time or a holding cost.

## When to use and when not

Use when the question is how much to order, how often, what reorder point or safety stock, whether to accept a minimum order quantity or a price break, or whether a blanket order, scheduled delivery, consignment or vendor-managed arrangement fits.

Not for the emergency when stock has run out (urgent-sourcing), for pooling demand across business units (demand-aggregation), for the price negotiation (negotiation-planning) or for warehouse operations.

## Inputs and preconditions

- The item and the supplier, or the category.
- Stock on hand, consumption and any seasonality the user knows, labelled user-supplied.
- Ordering and holding cost rates from policy or the user; service level wanted.

- Use `tool:supplier_items` for the item's order and invoice lines from this supplier over a period, with line counts, quantities, unit-price range and last date; read for two periods to see how demand moved
- Use `tool:similar_items` for whether the same item is bought elsewhere (counts per source only; it names no supplier)
- Use `tool:supplier_discovery` with the item description, for the other suppliers on our records that provide it
- Use `tool:search_contracts` for the agreement with the supplier
- Use `tool:extract_clauses` for minimum order quantity, lead time, price-break, call-off schedule, consignment and stock-holding clauses, quoted
- Use `tool:supplier_lookup` for the lead time on the supplier's master record, as stored
- Use `tool:run_calculation` with `finance.average_per_period` for the consumption rate from a period's quantity and its months or weeks, `deal.pct_change` for the movement between two periods and `deal.scaled_total` for lead-time demand (the rate over the lead time) or a price break's total; an order-quantity formula is not in the registry, so its result is labelled unverified with its inputs
- Use `tool:get_policy` for the ordering cost, holding cost rate, service-level target and any stock policy
NOT AVAILABLE: create_purchase_order. nothing is ordered; the pattern is advice for the buyer
NOT AVAILABLE: build_chart. figures are given as tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.history_months`.

## Workflow

1. Frame the item, supplier and the user's stock and consumption facts.
2. Demand from the records: the item's quantities from supplier_items over the last `config.thresholds.history_months` months, read as two halves and compared via `deal.pct_change`; the rate per month or week via `finance.average_per_period`; seasonality only if the user states it.
3. Supplier terms quoted: minimum order quantity, lead time, price breaks, delivery schedule, consignment or stock-holding clauses; where no contract exists, the quote on record or the user's statement, labelled.
4. Costs from policy: ordering cost per order, holding cost rate, service level; none means the pattern is given in units and frequency without a cost optimum, and the policy owner is asked.
5. Pattern: order quantity and frequency that respect the minimum order quantity and the price breaks, with every figure from the registry or labelled unverified; reorder point from lead-time demand (`deal.scaled_total` of the registry's rate and the lead time from the clause, the quote or the user) plus safety stock only where the user gives variability or a service level; the price-break trade-off (discount versus holding) shown both ways with inputs.
6. Arrangements: a blanket order with scheduled releases where demand is steady and the contract allows; consignment or vendor-managed stock where the clause exists or the supplier offers it, with the risks (ownership, obsolescence, audit) named; policy check for any commitment.

## Frameworks and heuristics

- Consumption is what the records show, not what the user remembers; the two are compared.
- A minimum order quantity above a year's demand is a sourcing problem, not an ordering one.
- A price break is worth taking only if the holding cost of the extra stock is less than the discount; show both.
- Lead time comes from the contract, the quote or the user; never from memory.
- Steady demand wants a blanket order and a schedule, not a monthly requisition.

## Policy hooks (SC-007)

Ordering cost, holding cost, service level and stock rules come from `tool:get_policy` and are quoted; without them no cost optimum is claimed. Any commitment (blanket order, consignment agreement) is checked with `tool:check_policy` and reported as replied. Nothing is ordered by the skill.

## Output specification

1. Frame: item, supplier, user's stock and consumption facts.
2. Demand history table per period with the rate and the movement via the registry.
3. Supplier terms quoted: minimum quantity, lead time, breaks, schedule, consignment.
4. Cost parameters from policy or absent.
5. Recommended order quantity and frequency with the arithmetic; reorder point and safety stock where inputs exist; price-break trade-off.
6. Arrangement recommendation with risks and approval route.

## Quality bar

- Demand figures are from the records; the user's figures are labelled.
- Lead time, minimum quantity and breaks are quoted from a clause, a quote or the user.
- Cost optimum only where policy gives the costs; arithmetic shown with inputs.
- Nothing is ordered or committed.

## Escalation and approval

Escalate to the policy owner where no ordering or holding cost is set, to the category lead where the minimum order quantity exceeds demand, and to finance for consignment or stock-holding commitments.

## Worked example

Asked how much safety gloves to order and how often, the skill reads two years of orders (a steady monthly rate with a small rise in the second year), quotes the contract's minimum order quantity, lead time and price break, reads the policy's ordering and holding costs, computes an order quantity with the arithmetic shown, finds the price break worth taking because the holding cost of the extra cartons is below the discount, sets a reorder point from lead-time demand with safety stock at the user's stated variability, and recommends a blanket order with monthly releases under the contract's call-off clause.

## Anti-patterns

- Inventing a lead time or a holding cost.
- Optimising on a formula with inputs nobody gave.
- Taking a price break without costing the stock it creates.
- Placing the order from the skill.
