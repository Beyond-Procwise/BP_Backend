---
name: constrained-supply-planning
description: "Balance the agreed demand against the supply constraints the user states - supplier capacity, lead times, material availability, minimum order quantities, agreements and their limits - into a feasible plan: what can be supplied when, what cannot, the options to close each shortfall with cost and risk, and the allocation rules when demand must be rationed. Use when asked whether demand can be met, to plan supply against capacity, or to allocate a shortage. Dormant until capacity and material data can be read; until then the constraints are the user's. Not for the forecast (consensus-forecasting) or the cycle (sop-ibp-process)."
---

# Constrained Supply Planning

## Purpose

Turn an unconstrained wish into a plan that supply can keep: constraints stated, shortfalls sized, options costed, allocations explained; the planner decides.

## When to use and when not

Use when demand per period is agreed and the question is whether and how supply can meet it.

Not for the demand number (consensus-forecasting), the cycle (sop-ibp-process), ordering patterns for one item (ordering-and-inventory), the emergency (urgent-sourcing) or inventory policy (inventory-policy-setting).

## Inputs and preconditions

- Agreed demand per family or item and period.
- Constraints: supplier capacity per period, lead times, minimum order quantities, material limits, agreement caps, from the user or the clauses.

- Use `tool:run_calculation` with `deal.share_pct` for a shortfall's share of demand, `deal.scaled_total` for a capacity over periods and `inventory.reorder_point` for the cover a lead time needs
- Use `tool:search_contracts` for the agreements' volume commitments and caps
- Use `tool:extract_clauses` for minimum volume, capacity reservation, lead time and allocation clauses, quoted
- Use `tool:supplier_discovery` for other suppliers on record for the items
- Use `tool:supplier_lookup` for lead times on record
- Use `tool:open_orders` for orders already placed that consume capacity
- Use `tool:get_policy` for allocation rules (who is served first) if governed
NOT AVAILABLE: run_mrp. no materials plan; the netting is done per period from the user's figures with the registry
NOT AVAILABLE: check_capacity. no capacity data; capacities are the user's or the clauses'
NOT AVAILABLE: simulate_scenario. no scenario model; options are costed one by one

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.allocation_trigger_pct`.

## Workflow

1. Frame: scope, periods, the demand as agreed, the constraints as stated with their sources (user, clause, record).
2. Net per period: demand against capacity plus orders already placed from open_orders; the shortfall per period and its share via the registry; lead-time cover via the registry where a switch is considered.
3. Constraints from the clauses: minimum volumes we must take, caps we cannot exceed, capacity reservations, allocation terms; quoted.
4. Options per shortfall: more from the incumbent (within the cap), another supplier on record (lead time, screening), pull-forward with inventory, demand shaping, accept the shortfall; cost and risk per option as stated or costed with the registry.
5. Allocation where a shortfall stays: the governed rule or the user's priority; who gets what and why; a shortfall above `config.thresholds.allocation_trigger_pct` percent of demand is allocated explicitly.
6. The feasible plan per period with the options chosen for the planner to decide; nothing committed.

## Frameworks and heuristics

- Capacity is a number per period with a source; 'they can usually cope' is not capacity.
- Orders already placed are capacity already spent.
- The lead time decides whether an option exists this period or next.
- Allocation is a rule written down before the shortage, not an argument during it.

## Policy hooks (SC-007)

Allocation rules come from `tool:get_policy` and are quoted; agreement limits come from the clauses. A commitment to a supplier (a new order, a reservation) runs through `tool:check_policy` in the carrying skill. The skill plans; it commits nothing.

## Output specification

1. Frame with constraints and sources.
2. Net table per period: demand, capacity, placed orders, shortfall with share.
3. Clause constraints quoted.
4. Options per shortfall with cost, risk, lead time.
5. Allocation with rule; the feasible plan for decision.

## Quality bar

- Every capacity figure has a source.
- Shortfalls and shares come from the registry.
- Every option names its lead time and risk.
- Nothing is committed.

## Escalation and approval

Escalate to the planning meeting for allocations, to the category lead for a new supplier, to urgent-sourcing for a shortfall inside the lead time, and to supply-continuity-planning where the constraint is a supplier's health.

## Worked example

Asked whether the pallet demand for the next quarter can be met, the skill nets each month's demand against the supplier's stated monthly capacity and the orders already placed, finds a shortfall in month two above the allocation trigger, quotes the agreement's cap and minimum, costs the options (the incumbent's overtime at the user's rate, a second supplier on record with a stated lead time that lands in month three, a pull-forward into inventory), proposes the allocation by the governed rule, and leaves the choice to the planner.

## Anti-patterns

- Assuming capacity.
- Forgetting the orders already placed.
- An option whose lead time misses the period.
- Committing an order from the skill.
