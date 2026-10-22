---
name: network-design
description: "Design or review the distribution network - how many warehouses or hubs, where, serving which customers or sites - from the demand by location, the lead times and the transport and facility costs the user supplies: the cost and service of each option, the stock each option needs, the risk of each, and a recommendation with what would change it. Use when asked how many warehouses to have, where to put a hub, or whether to consolidate or add sites. Dormant until network data can be read. Not for stock placement within a fixed network (multi-echelon-optimisation) or one warehouse's layout (warehouse-layout-design)."
---

# Distribution Network Design

## Purpose

Compare network options on the same costs and the same service, with every figure the user's or the registry's, so the number and place of sites is a decision with numbers rather than a map with pins.

## When to use and when not

Use when the question is the shape of the network: how many sites, where, serving whom, consolidate or add.

Not for stock positioning inside a given network (multi-echelon-optimisation), for one site's layout (warehouse-layout-design), for a lane's mode (mode-selection) or for buying the freight (carrier-procurement).

## Inputs and preconditions

- Demand by customer or site location per period; service requirement (lead time) per location.
- Per candidate site: facility cost, handling cost, inbound and outbound transport costs or rates, lead times to each location; from the user.

- Use `tool:run_calculation` with `deal.scaled_total` for a transport or facility cost over the horizon from rate and volume, `deal.share_pct` for the demand each site would serve, `deal.pct_change` for the cost gap between options, `inventory.safety_stock` for the stock each node needs at its lead time, and `finance.net_present_value` for an option's dated costs at the governed rate
- Use `tool:build_spend_cube` for our spend by country and region as a demand proxy
- Use `tool:supplier_lookup` for suppliers' countries and lead times on record
- Use `tool:get_policy` for the cost of capital and any service-level rule
- Use `tool:build_xlsx` for the option comparison workbook
NOT AVAILABLE: optimise_route. no routing engine; distances and transit times are the user's
NOT AVAILABLE: simulate_scenario. no network model; options are costed one at a time with the registry, at most the configured number

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_options`.

## Workflow

1. Frame: the locations and their demand and service needs (our spend by country and region as a proxy where the user has nothing better), the candidate sites and their costs with sources, the horizon.
2. Options: at most `config.thresholds.max_options` network shapes (one central site, two regional, three local, the current); each assigns locations to sites by the user's lead times.
3. Per option via the registry: facility cost over the horizon, inbound and outbound transport from rates and volumes, stock at each node from the lead times, the share of demand per site; the net present value of the dated costs at the governed rate.
4. Service per option: each location's lead time against its requirement; locations that miss it listed.
5. Risk per option: single-site exposure, supplier proximity, the user's stated risks; the cost gap between options via the registry.
6. Recommend the option with the lowest cost that meets every service requirement, with the risk stated and what would change it; the workbook built when asked.

## Frameworks and heuristics

- Fewer sites cut facility and stock cost and add transport and lead time; the trade-off is the whole question.
- Service requirements are constraints, not preferences; an option that misses one is out.
- One site is a risk; the cost of that risk is a line, not a feeling.
- The current network is an option too; it has no switching cost.

## Policy hooks (SC-007)

The cost of capital and service rules come from `tool:get_policy` and are quoted. The skill decides nothing; a site decision is a business case (business-case) and an investment approval.

## Output specification

1. Frame: locations, demand, service needs, candidate sites with costs and sources.
2. Options table: costs by element via the registry, stock per node, net present value, service misses, risk.
3. Cost gaps between options; the recommendation with what would change it.
4. Workbook export id when built; hand-off to business-case.

## Quality bar

- Every cost, stock figure and gap comes from the registry on the user's inputs.
- Every location's service is checked per option.
- The current network is costed as an option.
- Nothing is decided.

## Escalation and approval

Escalate to logistics for lead times and rates, to finance for facility costs and the rate, to supply-continuity-planning for single-site risk, and to business-case for the investment decision.

## Worked example

Asked whether two regional warehouses would beat the single central one, the skill takes demand by region from the user (with our spend by region as a check), costs both options via the registry from the user's facility, handling and transport figures, computes the stock each node needs at its lead time, finds the two-site option dearer on facilities but cheaper on outbound transport and stock with a net present value a stated share lower, checks that every region meets its lead time under both, and recommends two sites with the single-site risk stated and the switching cost as the user gives it.

## Anti-patterns

- A network with a site placed by instinct.
- Costing transport without the volumes.
- Ignoring the stock each extra node needs.
- Deciding the investment from the skill.
