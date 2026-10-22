---
name: multi-echelon-optimisation
description: "Decide where in a supply network stock should sit - at the supplier, a central hub or the sites - from the network, lead times, demand per site and costs the user supplies: the cover each node needs, the pooling benefit of holding stock centrally, the service and cost trade-off per option, and a recommended position per item class for the planner. Use when asked where to hold stock, whether to centralise or decentralise inventory, or how much to hold at each node. Dormant until network and stock data can be read. Not for one site's policy (inventory-policy-setting)."
---

# Multi-Echelon Stock Positioning

## Purpose

Place stock where it serves the most sites for the least money, with the trade-off shown per option and nothing optimised in the dark: every input is the user's and every figure the registry's.

## When to use and when not

Use when the question spans more than one stocking location: central versus local, hub cover, pooling, which node holds what.

Not for a single location's parameters (inventory-policy-setting), for the physical network of warehouses (network-design), for supply constraints (constrained-supply-planning) or for ordering (ordering-and-inventory).

## Inputs and preconditions

- The network: nodes, the lead time between each pair, which sites each node serves.
- Per item class: demand per site per period with variability, unit cost, service level, holding cost rate; from the user or policy.

- Use `tool:run_calculation` with `inventory.safety_stock` per node at the class's service level and the node's lead time, `inventory.reorder_point` per node, `deal.scaled_total` for the stock value per option and `deal.share_pct` for the saving of one option against another; pooled variability is the user's figure (the skill names the square-root rule but does not compute a pooled standard deviation)
- Use `tool:supplier_lookup` for supplier lead times on record
- Use `tool:get_policy` for service levels, holding cost rate and any network rule
NOT AVAILABLE: simulate_scenario. no network model; options are compared one at a time with the registry on the user's inputs
NOT AVAILABLE: query_wms. no warehouse connection; stock positions are the user's

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_options`.

## Workflow

1. Frame: the network as supplied (nodes, lead times, who serves whom), the item classes and their demand per site, the costs and service levels with sources.
2. Options: local stock at each site; central stock at the hub with the sites' lead time from the hub; a mix (A items central, C items local); at most `config.thresholds.max_options` options.
3. Per option: safety stock and reorder point per node via the registry at the class's service level, using the user's demand variability per node (pooled variability for the hub as the user states it); stock value per option via the registry; the saving of each option against local-everywhere via the registry.
4. Service: the lead time each site sees under each option against the service it needs; a site whose lead time from the hub exceeds its tolerance keeps local stock.
5. Recommend per class: the option with the lowest stock value that keeps every site's service, with the transport effect the user states; the planner decides.

## Frameworks and heuristics

- Pooling cuts safety stock; it does not cut cycle stock or transport.
- A site that cannot wait the hub's lead time keeps its own stock.
- Centralise what is expensive and variable; localise what is cheap and steady.
- The network's lead times are the inputs that matter most; get them from the user, not from hope.

## Policy hooks (SC-007)

Service levels, the holding cost rate and any network rule come from `tool:get_policy` and are quoted. Nothing is moved, set or committed by the skill.

## Output specification

1. Network and inputs with sources.
2. Options table: per node safety stock, reorder point, stock value via the registry; service per site.
3. Savings of each option against local-everywhere.
4. Recommendation per class with the transport effect stated.

## Quality bar

- Every stock figure comes from the registry on the user's inputs.
- Pooled variability is the user's figure, labelled.
- Every site's service is checked per option.
- Nothing is set.

## Escalation and approval

Escalate to the planner for the decision, to logistics for the transport effect, and to finance for the holding cost rate.

## Worked example

Asked whether to hold filters centrally or at the four sites, the skill frames the network and lead times as supplied, computes safety stock per site at the policy's service level via the registry, then for the hub with the user's pooled variability, values both options via the registry, finds the central option lowers stock value by a stated share but leaves one site beyond its lead-time tolerance, and recommends central stock for three sites with local stock kept at the fourth, the transport effect as the user states it.

## Anti-patterns

- Computing a pooled standard deviation the user did not supply.
- Centralising past a site's lead-time tolerance.
- Ignoring transport.
- Moving stock from the skill.
