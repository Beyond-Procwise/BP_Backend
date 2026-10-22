---
name: mode-selection
description: "Choose the transport mode and service for a shipment or a lane - air, sea, road, rail, courier or a combination - from the transit time the need allows, the cost per mode from the rates the user supplies or quotes on record, the shipment's weight, volume and value, the Incoterm, and the risk and emissions the user cares about, with a recommendation that says what would change it. Use when asked how to ship something, whether to use air or sea, or which service fits a deadline. Dormant until freight rates can be read; until then the rates are the user's. Not for the Incoterm itself (incoterms-selection) or for buying freight (carrier-procurement)."
---

# Transport Mode Selection

## Purpose

Match the mode to the need: fast enough, cheap enough, safe enough, with the trade-off in numbers the user supplied and the registry computed.

## When to use and when not

Use when a shipment or lane needs a mode decision: speed against cost, a deadline, a value or a risk that rules a mode out.

Not for the delivery term (incoterms-selection), for tendering carriers (carrier-procurement), for customs (trade-compliance), for the landed cost of a quote (total-cost-of-ownership) or for an urgent buy (urgent-sourcing).

## Inputs and preconditions

- The shipment or lane: origin, destination, weight, volume, value, the date it must arrive, special handling.
- Rates and transit times per mode from the user or quotes on record, labelled by source and date.

- Use `tool:run_calculation` with `deal.scaled_total` for a mode's cost from its rate and the chargeable weight or volume, `deal.pct_change` for the cost gap between modes, `finance.carrying_cost` for the value tied up in transit per mode, and `sustainability.scope3` for emissions per mode from the user's factors
- Use `tool:supplier_lookup` for the carriers on record with their Incoterms and lead times
- Use `tool:search_contracts` for a freight agreement that fixes the mode or the rate
- Use `tool:extract_clauses` for the delivery, transit and mode clauses, quoted
- Use `tool:commodity_index_lookup` for a fuel index on record behind a surcharge
NOT AVAILABLE: get_freight_quote. no carrier connection; rates are the user's or quotes on record
NOT AVAILABLE: track_shipment. no carrier connection
NOT AVAILABLE: optimise_route. no routing engine; the lane is as the user states it

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.indifference_gap_pct`.

## Workflow

1. Frame: the shipment facts, the arrival date, the Incoterm (who chooses the mode), special handling, and what the user weighs (cost, time, risk, emissions).
2. Rule out: modes that cannot meet the date, cannot carry the goods (dangerous, oversized, temperature) or that the agreement forbids; each exclusion with its reason.
3. Cost per remaining mode via the registry from the supplied rate and the chargeable weight or volume; the value tied up in transit per mode via `finance.carrying_cost` at the governed rate; the gap between modes via the registry.
4. Risk per mode as the user states or the records show (damage history from findings, transit variability); emissions per mode via the registry from the user's factors where asked.
5. Recommend the mode and service with the trade-off table; a cost gap under `config.thresholds.indifference_gap_pct` percent is called a tie decided on time or risk; state what would change the answer.

## Frameworks and heuristics

- The date rules first; cost chooses among what meets it.
- Chargeable weight, not actual weight, prices air and courier.
- High value in slow transit costs money the rate sheet does not show.
- A lane shipped every week deserves a tender, not a mode choice each time.

## Policy hooks (SC-007)

The cost of capital comes from the governed policy; freight agreements on the register govern the lanes they cover. Nothing is booked; carrier-procurement and a person buy the freight.

## Output specification

1. Shipment frame and exclusions with reasons.
2. Mode table: cost (registry), transit, value in transit, risk, emissions where asked, with sources.
3. Recommendation with the trade-off and what would change it.
4. Hand-off: carrier-procurement for a recurring lane; trade-compliance for cross-border.

## Quality bar

- Every rate has a source and a date.
- Costs, gaps, carrying cost and emissions come from the registry.
- Exclusions name their rule.
- Nothing is booked.

## Escalation and approval

Escalate to logistics for a mode the agreement does not cover, to trade-compliance for restricted goods, and to the budget owner when the date forces the dear mode.

## Worked example

Asked whether to fly or ship five pallets of spares needed in Rotterdam in three weeks, the skill rules out sea on the user's stated transit, costs air and road via the registry from the user's rates and the chargeable weights, adds the carrying cost of the value in transit, notes the damage history on the road lane from findings, finds road within the indifference gap of air and a week faster than the date, and recommends road with the condition that the date holds.

## Anti-patterns

- A rate from memory.
- Pricing air on actual weight.
- Ignoring the value in transit.
- Booking from the skill.
