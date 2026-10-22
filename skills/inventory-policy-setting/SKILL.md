---
name: inventory-policy-setting
description: "Set the stocking policy per item class - the ABC/XYZ class from value and variability, the service level per class, the review period, the safety stock and reorder point per item from the registered calculations, and the stock value the policy implies - from the demand series, costs and lead times the user supplies. Use when asked what service level or stock policy an item or class should have, or to classify items for stocking. Dormant until stock and demand data can be read. Not for one item's order quantity (ordering-and-inventory) or slow stock (slob-management)."
---

# Inventory Policy Setting

## Purpose

Give each class of item a policy that matches its value and its variability, with the stock it costs stated, so the planner sets service levels knowingly.

## When to use and when not

Use when stocking rules are being set or reviewed for a range of items: classes, service levels, review periods, safety stocks.

Not for the order quantity of one item (ordering-and-inventory), for slow or obsolete stock (slob-management), for the network's stock placement (multi-echelon-optimisation) or for supply constraints (constrained-supply-planning).

## Inputs and preconditions

- Per item: annual value, the demand series per period, lead time, unit cost, from the user.
- Ordering and holding costs and the service-level targets, from policy or the user.

- Use `tool:run_calculation` with `inventory.abc_xyz` for the classes, `inventory.safety_stock` and `inventory.reorder_point` per item at the class's service level, `inventory.eoq` where the costs are given, `deal.scaled_total` for the stock value a policy implies and `deal.share_pct` for a class's share of value
- Use `tool:supplier_items` for our order quantities per period per supplier as a demand series we hold
- Use `tool:supplier_lookup` for lead times on record
- Use `tool:get_policy` for governed service levels, holding cost rate and stock rules
NOT AVAILABLE: detect_slob. no stock-on-hand data; slow and obsolete stock is slob-management's job on the user's stock list
NOT AVAILABLE: query_wms. no warehouse connection; stock figures are the user's

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.review_period_days`.

## Workflow

1. Frame: the range of items, the data supplied per item, the cost and service inputs with sources.
2. Classify via `inventory.abc_xyz` with the policy's or default cutoffs; show the class table and each class's share of value via the registry.
3. Service level per class from policy or the user's targets (defaults in this playbook are named as defaults): higher for A-X, lower for C-Z, Z items reviewed by hand.
4. Per item: safety stock and reorder point via the registry at the class's service level and the item's lead time and demand variability; order quantity via `inventory.eoq` where ordering and holding costs are given.
5. Stock value the policy implies per class via `deal.scaled_total` (safety stock times unit cost, summed by the registry per item); compared with the current stock value the user states.
6. Policy table per class and the review period of `config.thresholds.review_period_days` days where policy states none; exceptions (Z items, regulated items) listed for the planner.

## Frameworks and heuristics

- Value sets attention; variability sets stock.
- A service level is a cost decision; show the stock it buys.
- Z items get a person, not a formula.
- Review the policy on a calendar, not when something runs out.

## Policy hooks (SC-007)

Service levels, holding cost rate and stock rules come from `tool:get_policy` and are quoted; defaults are named as defaults. Nothing is set in any system by the skill; the planner applies the policy.

## Output specification

1. Class table with shares of value.
2. Service level per class with source.
3. Per-item safety stock, reorder point and order quantity from the registry.
4. Stock value implied per class against current.
5. Policy table, review period, exceptions.

## Quality bar

- Classes, safety stocks, reorder points and values come from the registry.
- Service levels have a source or are named defaults.
- Z items are flagged for a person.
- Nothing is set by the skill.

## Escalation and approval

Escalate to finance for the holding cost and the stock value, to the planner for service levels and Z items, and to supply-continuity-planning for critical items with no alternative.

## Worked example

Asked to set the stocking policy for two hundred spares, the skill classifies them via the registry from the user's values and demand series, finds most of the value in a handful of A-X items and a long tail of C-Z, takes the policy's service levels per class, computes safety stock and reorder point per item at those levels with the recorded lead times, sums the implied stock value per class via the registry against the user's current stock, and tables the policy with the Z items listed for the planner's judgement.

## Anti-patterns

- One service level for everything.
- Safety stock from a rule of thumb.
- A formula applied to a Z item.
- Setting the policy in a system from the skill.
