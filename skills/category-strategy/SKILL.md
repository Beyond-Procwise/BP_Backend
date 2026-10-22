---
name: category-strategy
description: Build or review the sourcing strategy for a category - what we buy and from whom from our own spend and supplier records, the supply market as the user describes it, Kraljic positioning from stated signals, risks on record, contract coverage, the levers available and a sequenced roadmap. Use when asked for a category strategy, plan, review or roadmap, a sourcing strategy, or how we should approach a category over the next years. Not for one event (rfx-design), one negotiation (negotiation-planning) or pooling one item (demand-aggregation).
---

# Category Strategy

## Purpose

Give a category owner a strategy they can defend: the fact base from our own records, the market as stated by the user with its source, a positioning that follows from stated signals, the options and levers that fit that positioning, and a roadmap with owners and dates. Every market fact is the user's or a tool's; the skill brings the method, not the market.

## When to use and when not

Use when the user asks for a category or sourcing strategy, a category plan, review or roadmap, how to approach a category over a horizon, or whether to consolidate, dual-source, insource or re-source a category.

Not for designing one tender (rfx-design), planning one negotiation (negotiation-planning), pooling one item across sites (demand-aggregation), a supplier's risk clearance (supplier-risk-assessment) or spend analysis on its own (spend-analysis, whose output this skill uses).

## Inputs and preconditions

- The category and its scope (what is in and out), the horizon (default `config.thresholds.strategy_horizon_years` years) and the reporting currency.
- The business need over the horizon: volumes, sites, specifications, service levels, from the user.
- The supply market: number and strength of suppliers, switching costs, capacity, input-cost drivers, regulation, from the user with a source for each; the skill holds no market data.
- The organisation's policies for the category (preferred suppliers, dual-sourcing rules, approval limits) from `tool:get_policy`.

Use `tool:build_spend_cube` for spend by supplier, country, region, buyer and month (invoices carry no category: scope by the suppliers and items the user names), `tool:supplier_discovery` and `tool:supplier_lookup` for who already supplies the items and their recorded profiles, `tool:similar_items` for how the items are described and from how many suppliers and buyers they are bought, `tool:classify_spend` for taxonomy coverage, `tool:search_contracts` and `tool:obligation_tracker` for contract coverage and expiries, `tool:screening_summary` and `tool:calc_supplier_scorecard` for incumbent risk and performance, `tool:commodity_index_lookup` for a loaded input-cost index, `tool:list_opportunity_levers` and `tool:get_opportunity_bucket` for the levers the platform tracks, `tool:savings_tracker` for savings already recorded, and `tool:run_calculation` with `negotiation.kraljic_quadrant` for positioning from stated signals and `deal.pct_change` for shares and changes.

NOT AVAILABLE: build_docx. Give the strategy as text sections.
NOT AVAILABLE: build_pptx. Give the summary as a one-page text brief.
NOT AVAILABLE: render_chart. Give the positioning and spend views as tables.

## Workflow

1. Scope and fact base. Fix scope, horizon and currency. From the cube and the supplier tools: spend in scope by supplier and by site or buyer, number of suppliers, share of the largest (a dependence alert above `config.thresholds.concentration_alert_pct` percent), contract coverage and expiries, taxonomy coverage. A category below `config.thresholds.min_category_spend` in the reporting currency gets a light-touch plan, said as such.
2. Need over the horizon. Record the user's volumes, specifications and service levels; note what will change and when.
3. Market. Record the user's market facts with sources: supplier count and concentration, switching costs, capacity, input-cost drivers (with a loaded index where one exists), regulation, sustainability pressures. Anything the user cannot give is "not known", listed as a research item.
4. Position. Run `negotiation.kraljic_quadrant` with the stated profit-impact and supply-risk signals and the thresholds in config (`config.thresholds.kraljic_profit_impact_high`, `config.thresholds.kraljic_supply_risk_high`); report the quadrant and the signals it rests on. Without signals, say the position is not determined.
5. Incumbents. For each main supplier: performance from the scorecard, risk from the screening summary (not performed is not clear), contract status, dependence.
6. Options. For the quadrant, lay out the fitting options (consolidate, dual-source, long-term agreement, competitive tender, specification change, make or buy, re-source, exit) with pros, cons, prerequisites and the policy that governs each. Rule out options that policy forbids and say why.
7. Levers and value. List the platform's recorded levers for the suppliers and items, savings already recorded in the ledger, and the value range for each option only where a calculation or the user's figures support it; otherwise "not sized".
8. Roadmap. Sequence the chosen options over the horizon with owners, dates, dependencies (contract expiries, notice deadlines from the tracker) and the decision points that need approval (`tool:check_policy` for the ones that commit spend).

## Frameworks and heuristics

- Kraljic by signals, not by feel: the quadrant comes from the stated profit impact and supply risk against configured thresholds and is reported with them.
- Strategic quadrant favours partnership and risk sharing; leverage favours competition; bottleneck favours security of supply and specification work; non-critical favours efficiency and catalogue buying.
- Dependence is a fact before it is a problem: the alert says who, how much, and whether a second source exists on record.
- Contract expiries drive sequencing: a strategy that needs a change at renewal starts before the notice deadline with buffer.
- Market facts carry their source; a fact with no source is a research item, not an input.

## Policy hooks (SC-007)

Preferred-supplier, dual-sourcing, competitive-bidding and approval rules come from governed policy via `tool:get_policy`; an option that a policy forbids is listed as ruled out by that policy. Actions in the roadmap that commit spend are checked with `tool:check_policy` when they come due, by the skill that executes them.

## Output specification

1. Scope, horizon, currency; the fact base with record counts and what the invoices cannot show.
2. Need over the horizon and market facts with sources; research items.
3. Positioning: quadrant, signals, thresholds used.
4. Incumbent table: supplier, spend share, performance, risk status, contract and expiry.
5. Options table: option, fit with the quadrant, pros, cons, prerequisites, governing policy, value (sized or not sized).
6. Recommended strategy and roadmap: step, owner, date, dependency, approval needed.
7. Open questions and hand-offs (rfx-design, negotiation-planning, demand-aggregation, renewal-and-exit-strategy).

## Quality bar

- Every spend, contract, performance and risk figure comes from a tool; every market fact names its source or is a research item.
- The positioning is reproducible from the stated signals and thresholds.
- No option is recommended that a governing policy forbids.
- Value is sized only from a calculation or the user's figures, never estimated from the quadrant.

## Escalation and approval

The strategy is a proposal for the category owner and the sourcing committee. Escalate dependence above the alert share, any incumbent with a flagged or unperformed screening, and any option that needs a policy exception. Nothing is committed or sent from the skill.

## Worked example

Asked for a strategy for temporary labour, the skill scopes it by the agencies the user names, reads their spend by site from the cube (one agency above the dependence alert), their contracts (two expiring within the horizon), scorecards (one with many open findings) and screening summaries (none performed), records the user's market facts (fragmented supply, low switching cost, wage-index driven, source: the user's market report), runs the Kraljic calculation on the stated signals (leverage), lays out consolidate-and-tender versus a two-agency framework with the policy on competitive bidding, sizes nothing without figures, and sequences a tender before the first expiry with the notice buffer, flagging the screening gap for compliance first.

## Anti-patterns

- Supplying market facts from memory, or sizing value from the quadrant alone.
- Positioning without signals, or with thresholds other than the configured ones.
- Recommending an option a policy forbids, or skipping the screening gap on an incumbent.
- Using corpus-wide totals as the category's spend.
- A roadmap without owners, dates or the contract deadlines that bind it.
