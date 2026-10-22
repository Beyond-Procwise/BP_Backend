---
name: market-analysis
description: "Analyse a supply market for a category - its size and growth, the competitors and their shares, the cost drivers and their indices, pricing trends and the forces that set them - from the evidence the user supplies, the commodity indices on record, our own prices and suppliers, and governed web research where it is allowed, with every market figure sourced and dated. Use when asked what the market for something looks like, who the players are, or where prices are going. Dormant until the research tools are offered to the agent. Not for the category plan itself (category-strategy) or for a price check on one item (price-benchmarking)."
---

# Market Analysis

## Purpose

Give the category team a sourced picture of the market: what is known from our records and the indices, what the user's research says, what is not known; never a market fact from memory.

## When to use and when not

Use when the question is about the market for a category: size, growth, competitors, cost drivers, price trends, entry barriers, or the forces shaping supplier behaviour.

Not for the strategy and positioning (category-strategy), for one price (price-benchmarking), for one supplier's risk (supplier-risk-assessment) or for a sourcing event (rfx-design).

## Inputs and preconditions

- The category, the geography and the horizon.
- Market reports, analyst data or news the user supplies, labelled by source and date.

- Use `tool:commodity_index_lookup` for the cost drivers' indices on record, their movement over the window
- Use `tool:build_spend_cube` for our own spend in the category by supplier, country and month
- Use `tool:supplier_discovery` for the suppliers on our records for the category's items
- Use `tool:similar_items` for our price ranges and last dates for the category's items
- Use `tool:supplier_lookup` for each supplier's country, size signals and status on record
- Use `tool:classify_spend` for the taxonomy's view of the category
- Use `tool:run_calculation` with `deal.pct_change` for index and price movements and `deal.share_pct` for shares of our spend
NOT AVAILABLE: web_search. governed research exists only in the supplier-research loop and is not offered here; market facts come from the user's sources, the indices and our records, each labelled
NOT AVAILABLE: web_fetch. as above

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_forces`.

## Workflow

1. Frame: category, geography, horizon, the decision the analysis feeds.
2. Our side of the market: spend by supplier and country from the cube, suppliers on record, price ranges and movement via the registry; shares via the registry.
3. Cost drivers: the indices on record for the category's inputs, their movement over the window via the registry; drivers with no index on record named as unmeasured.
4. The wider market: size, growth, competitor shares and trends only from the user's sources, each labelled with source and date; nothing from memory; what is unknown listed.
5. Forces: supplier power, buyer power, substitutes, entry barriers, stated as evidence-backed judgements with their evidence, each at most `config.thresholds.max_forces` lines.
6. Hand to category-strategy with the gaps named.

## Frameworks and heuristics

- Our records are the one market fact we own; start there.
- An index measures a cost driver, not a price; say which.
- A market share without a source and a date is a rumour.
- Unknown is an answer; invented is not.

## Policy hooks (SC-007)

No market fact is stated without a source; governed web research is not offered here and is never simulated. The analysis commits nothing and names no supplier from memory.

## Output specification

1. Frame and the decision it feeds.
2. Our market: spend, suppliers, price ranges and movements with sources.
3. Cost drivers: indices, movements, unmeasured drivers.
4. Wider market from the user's sources, labelled and dated; the unknowns.
5. Forces with evidence; hand-off to category-strategy.

## Quality bar

- Every market figure has a source and a date or is marked unknown.
- Movements and shares come from the registry.
- No supplier or competitor is named from memory.
- Unknowns are listed, not filled.

## Escalation and approval

Escalate to the category lead where the user's sources conflict, and to the research owner where governed web research is needed.

## Worked example

Asked what the corrugated packaging market looks like, the skill reads our packaging spend by supplier and country, the paper and energy indices on record with their movement, our price ranges for the items, and the two analyst notes the user supplied (labelled and dated) for size and shares; states that competitor shares beyond those notes are unknown; writes the five forces with the evidence each rests on; and hands the picture to category-strategy with the unknowns named.

## Anti-patterns

- Quoting a market size from memory.
- Naming competitors not on our records or in the user's sources.
- Treating an index as a price.
- Simulating web research.
