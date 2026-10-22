---
name: warehouse-layout-design
description: "Design or review a warehouse layout - the zones (receiving, storage by velocity, picking, packing, dispatch, returns), the storage modes per item class, the flow between zones, the capacity in pallets or locations against the stock the user holds, and the pick-path and slotting principles - from the volumes, SKU profile, dimensions and equipment the user supplies. Use when asked how to lay out a warehouse, where to put fast movers, or whether the space fits the stock. Dormant until warehouse data can be read. Not for the network (network-design) or staffing (labour-planning)."
---

# Warehouse Layout Design

## Purpose

Give the site a layout that fits its flow and its stock: fast movers near dispatch, storage modes matched to item classes, capacity checked in numbers, and the principles a slotting tool would later apply.

## When to use and when not

Use when a warehouse is being set up, re-laid out or checked for capacity and flow.

Not for how many warehouses (network-design), for staffing the shifts (labour-planning), for stock policy (inventory-policy-setting) or for returns handling rules (returns-reverse-logistics).

## Inputs and preconditions

- The building: floor area, clear height, doors, columns, from the user.
- The SKU profile: items with dimensions, weight, velocity (picks and units per period), storage needs; stock on hand or the policy's stock; from the user.
- Equipment and handling constraints.

- Use `tool:run_calculation` with `inventory.abc_xyz` for velocity classes from the user's pick series, `deal.scaled_total` for locations or pallet positions needed per class from stock and footprint, and `deal.share_pct` for each class's share of picks and of space
- Use `tool:supplier_items` for our order quantities per item as a velocity proxy where the user has no pick data
- Use `tool:get_policy` for safety, segregation and storage rules if governed
- Use `tool:render_chart` for a share-of-picks or share-of-space chart when asked
NOT AVAILABLE: query_wms. no warehouse connection; stock and picks are the user's
NOT AVAILABLE: optimise_slotting. no slotting engine; slotting principles are stated per class, not computed per location
NOT AVAILABLE: optimise_pick_path. no pick-path engine; the path principle is stated

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_utilisation_pct`.

## Workflow

1. Frame: the building, the SKU profile, the stock, the flow (inbound to outbound), the constraints (safety, segregation, temperature) from policy or the user.
2. Classify velocity via `inventory.abc_xyz` from the pick series (or our order quantities as a proxy, labelled); share of picks per class via the registry.
3. Storage mode per class: pallet racking, shelving, flow rack, floor stack, by dimensions, weight and velocity; locations needed per class via the registry from stock and footprint; total against the building's capacity; a utilisation above `config.thresholds.max_utilisation_pct` percent is flagged as congested.
4. Zones and flow: receiving and dispatch placement by doors, picking near dispatch, fast movers closest, bulk storage behind, returns and quarantine segregated; the flow drawn as a sequence.
5. Slotting and pick-path principles per class (golden zone for A, ergonomics for heavy, families together); stated for the WMS to apply.
6. Output the layout description, the capacity table, the flagged constraints, and the chart when asked.

## Frameworks and heuristics

- Picks drive layout; pallets drive capacity.
- The golden zone (waist to shoulder) is for the fastest movers, not the biggest.
- Receiving and dispatch should not share a door unless the volume is tiny.
- Above the utilisation flag, every move costs two.

## Policy hooks (SC-007)

Safety, segregation and storage rules come from `tool:get_policy` and are quoted. The skill designs on paper; nothing is slotted, moved or built.

## Output specification

1. Frame with constraints.
2. Velocity classes and shares via the registry.
3. Storage mode and locations per class; capacity table against the building.
4. Zone plan and flow; slotting and pick-path principles.
5. Flags: congestion, segregation, safety; chart export id when drawn.

## Quality bar

- Classes, locations and shares come from the registry on the user's data.
- Capacity is a number against the building, not a feeling.
- Every rule cited is policy or the user's constraint.
- Nothing is slotted or moved.

## Escalation and approval

Escalate to health and safety for aisle widths and racking rules, to operations for the flow and equipment, and to labour-planning for the staffing the layout implies.

## Worked example

Asked to lay out the new spares warehouse, the skill classifies the two thousand items by picks via the registry, finds a small class carrying most picks, assigns shelving near dispatch for them and pallet racking behind for the bulk, computes the locations each class needs from the user's stock and footprints, finds the building at a utilisation above the flag at policy stock levels, draws the flow from the receiving door through quarantine, storage and picking to dispatch, states the golden-zone and family slotting principles, and flags the congestion for operations and inventory-policy-setting.

## Anti-patterns

- Slotting by item size instead of velocity.
- Capacity without the stock figures.
- A layout with no returns or quarantine space.
- Moving stock from the skill.
