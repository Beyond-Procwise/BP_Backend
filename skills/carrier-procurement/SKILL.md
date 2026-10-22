---
name: carrier-procurement
description: "Buy freight properly - define the lanes, volumes and service levels from the shipment history the user supplies and our records, design the freight tender (lane sheet, rate card structure, surcharges, service measures), compare carrier bids on a like-for-like basis per lane, and recommend the award per lane or the allocation across carriers with the service and risk stated. Use when asked to tender freight, compare carrier rates, or set up a rate card. Dormant until freight rates and shipment data can be read. Not for one shipment's mode (mode-selection) or the generic tender design (rfx-design)."
---

# Carrier Procurement

## Purpose

Turn freight spend into a lane-by-lane award with comparable rates, defined surcharges and service measures, so the cheapest headline rate does not win on paper and lose on the invoice.

## When to use and when not

Use when freight is to be tendered or re-tendered, carrier bids compared, or a rate card built or reviewed.

Not for a single shipment (mode-selection), for the general tender method (rfx-design, which this skill applies), for the award paper (award-recommendation) or for customs (trade-compliance).

## Inputs and preconditions

- The lane sheet: origin, destination, mode, volumes per period, service level needed, from the user's shipment history.
- Carrier bids as rate cards with surcharges, labelled by carrier and date.

- Use `tool:build_spend_cube` for our freight spend by supplier and month as the baseline
- Use `tool:supplier_items` for what we buy from each carrier on record (lanes as line descriptions) and the price range
- Use `tool:supplier_discovery` for carriers on record
- Use `tool:screening_summary` for each bidder's screening
- Use `tool:run_calculation` with `bids.normalise` to put bids on one basis per lane, `deal.scaled_total` for a lane's annual cost from the rate and the volume, `deal.pct_change` for the gap to the baseline and `scoring.weighted_total` for the service-weighted ranking
- Use `tool:search_contracts` for the current freight agreement and its rate card
- Use `tool:extract_clauses` for surcharge, fuel, minimum and service clauses, quoted
- Use `tool:commodity_index_lookup` for the fuel index the surcharge references
- Use `tool:build_xlsx` for the lane sheet and the comparison workbook
NOT AVAILABLE: get_freight_quote. no carrier connection; bids come in as rate cards the user supplies
NOT AVAILABLE: create_rfx. the tender is issued by a person (rfx-issuance when it exists)

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.min_carriers_per_lane`.

## Workflow

1. Baseline: freight spend by carrier from the cube; lanes and volumes from the user's shipment history (and our carrier line records); the current rate card and surcharge clauses quoted.
2. Tender design: the lane sheet with volumes and service levels; the rate card structure every bidder must fill (base rate per unit, fuel surcharge formula tied to a named index, accessorials listed, minimums); service measures and the award basis; built with the export tool; issued by a person.
3. Compare: each bid per lane on one basis via `bids.normalise` (same unit, same surcharge treatment, same currency), annual cost per lane via `deal.scaled_total` from the volume; a bid that leaves a surcharge open is incomplete and said so.
4. Rank per lane: service-weighted via `scoring.weighted_total` with the published weights; the gap to the baseline via the registry; at least `config.thresholds.min_carriers_per_lane` carriers per critical lane.
5. Recommend: award per lane or allocation across carriers with the reasons, the service and risk, the total against the baseline; hand to award-recommendation for the award paper.

## Frameworks and heuristics

- Freight is bought per lane; a total discount hides a dear lane.
- A surcharge without a formula is a price without a number.
- Normalise before you compare; the unit and the fuel treatment differ between bids.
- One carrier on a critical lane is a risk; two is a plan.

## Policy hooks (SC-007)

Tender rules come from `tool:get_policy` through rfx-design; the award's approval comes from `tool:check_policy` in award-recommendation. Nothing is issued, awarded or booked by the skill.

## Output specification

1. Baseline: spend, lanes, volumes, current rate card and surcharges.
2. Tender design: lane sheet, rate card structure, service measures, award basis (workbook export id).
3. Comparison per lane on one basis with annual cost and gap to baseline; incomplete bids named.
4. Ranking per lane and the recommended award or allocation.

## Quality bar

- Every bid comparison goes through the normalisation calculation.
- Every surcharge has a formula and an index, or the bid is incomplete.
- Annual costs and gaps come from the registry.
- Nothing is issued or awarded.

## Escalation and approval

Escalate to logistics for service levels, to the category lead for lane allocation, to supply-continuity-planning for single-carrier lanes, and to award-recommendation for the award.

## Worked example

Asked to tender the European road lanes, the skill baselines freight spend by carrier from the cube and the lanes from the user's history, quotes the current fuel-surcharge clause, builds the lane sheet and rate card structure as a workbook, and when the three bids arrive normalises them per lane, computes annual cost per lane from the volumes, ranks by the published weights, finds one carrier cheapest on most lanes but dearest on the two largest, and recommends a two-carrier allocation with the saving against the baseline via the registry.

## Anti-patterns

- Comparing headline rates with different surcharge treatments.
- Awarding on a total discount.
- A single carrier on every lane.
- Issuing or awarding from the skill.
