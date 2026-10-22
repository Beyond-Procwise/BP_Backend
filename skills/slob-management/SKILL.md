---
name: slob-management
description: "Work a list of slow-moving and obsolete stock - from the stock list and movements the user supplies, classify each item as slow, obsolete or excess by the configured rules, value it, find the dispositions on record (return to the supplier under a clause, transfer to another site that buys it, sell, rework, scrap) and recommend one per item with its recovery, the write-down for finance and the owner. Use when asked about slow stock, dead stock, excess or obsolete inventory, or what to do with it. Dormant until stock data can be read. Not for setting the policy that prevents it (inventory-policy-setting)."
---

# Slow and Obsolete Stock Management

## Purpose

Turn a dead-stock list into decisions with money attached: what each item is worth, who could take it, what it costs to keep, and who decides.

## When to use and when not

Use when stock is not moving and the question is what it is, what it is worth and what to do with it.

Not for the stocking policy (inventory-policy-setting), for ordering (ordering-and-inventory), for a return of a bad delivery (delivery-issue-handling) or for the supplier's claim (dispute-and-claims).

## Inputs and preconditions

- The stock list: item, quantity, unit cost, last movement date, location, from the user.
- Demand or consumption per item if known.

- Use `tool:run_calculation` with `inventory.turns` for an item's turns from the user's figures, `deal.scaled_total` for an item's stock value and a disposition's recovery, `deal.share_pct` for slow stock's share of total value and `finance.carrying_cost` for the cost of keeping it
- Use `tool:supplier_items` for whether we still buy the item and from whom
- Use `tool:supplier_discovery` for other sites or suppliers on record that deal in the item
- Use `tool:search_contracts` for the agreement with the item's supplier
- Use `tool:extract_clauses` for return, buy-back and consignment clauses, quoted
- Use `tool:get_policy` for the slow and obsolete rules (days without movement, write-down steps, who approves disposal)
NOT AVAILABLE: detect_slob. no stock-on-hand data; the classification runs on the user's list by the rules here
NOT AVAILABLE: query_wms. no warehouse connection

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.slow_days`, `config.thresholds.obsolete_days`, `config.thresholds.approval_value`.

## Workflow

1. Frame: the list as supplied, the period, the rules from policy (or the defaults: slow beyond `config.thresholds.slow_days` days without movement, obsolete beyond `config.thresholds.obsolete_days`).
2. Classify each item: slow, obsolete or excess (more than the cover the demand supports), with the days since movement and the turns via the registry where consumption is given.
3. Value: stock value per item and per class via the registry; slow stock's share of total value; the carrying cost of keeping it for a year at the governed rate via the registry.
4. Dispositions on record: a return or buy-back clause with the supplier (quoted), another site or supplier on record that still buys the item, whether we still buy it ourselves; sell, rework or scrap as the user's options with their stated recovery.
5. Recommend per item: the disposition with the best recovery net of effort, the write-down for finance, the approval it needs, the owner; a line above `config.thresholds.approval_value` in value is a decision for the approver.
6. Prevention: the policy gap that let the stock build, for inventory-policy-setting.

## Frameworks and heuristics

- Every month kept costs money; the carrying cost is the clock.
- A return clause is worth more than a scrap bin.
- Obsolete is a fact (no demand, no use); slow is a judgement (demand, but low).
- Write-downs are finance's; disposals are approved; the skill proposes.

## Policy hooks (SC-007)

Rules, write-down steps and disposal approval come from `tool:get_policy` and are quoted; defaults are named as defaults. A disposal above the configured value runs through `tool:check_policy` in the carrying skill. Nothing is written down, moved or scrapped by the skill.

## Output specification

1. Classified list with days since movement, turns, value.
2. Totals per class, share of value, carrying cost.
3. Dispositions available per item with clause or record evidence.
4. Recommendation per item: disposition, recovery, write-down, approval, owner.
5. Prevention note for inventory-policy-setting.

## Quality bar

- Values, shares, turns and carrying cost come from the registry.
- Every disposition names its evidence (clause, record, user's option).
- Write-downs and disposals are proposed, never done.

## Escalation and approval

Escalate to finance for write-downs, to the approver for disposals above the configured value, to the supplier's category lead for returns, and to inventory-policy-setting for the cause.

## Worked example

Given a list of sixty items with no movement for a year, the skill classifies them by the policy's days, values them via the registry (a small share of total stock value but a stated carrying cost), finds a buy-back clause for the largest line and another site that still buys three items, recommends return, transfer or scrap per item with recovery and write-down, lists the two lines above the approval value for the approver, and notes the over-ordering pattern behind most of the list.

## Anti-patterns

- Calling slow stock obsolete to clear it.
- Scrapping an item a clause lets us return.
- Valuing stock at a guess.
- Writing it down from the skill.
