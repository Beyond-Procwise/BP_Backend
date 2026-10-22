---
name: supply-continuity-planning
description: "Build the contingency plan for a supplier or an item failing - what we buy from them and how much, what else on our records could supply it, the screening status of those alternatives, the contract's force-majeure and business-continuity clauses, the stock and lead-time facts the user holds, the warning signs to watch and the actions in order if the supplier stops. Use when asked what happens if a supplier fails, goes into administration, is hit by a disruption, or when a backup or contingency plan is wanted. Not for the category's dual-sourcing strategy (category-strategy) or for the live emergency (urgent-sourcing)."
---

# Supply Continuity Planning

## Purpose

Answer 'what if they stop?' before it happens: the exposure in numbers from the records, the alternatives that already exist, the gaps, the triggers and the first actions, so the response on the day is a checklist rather than a scramble. The skill plans; it orders nothing.

## When to use and when not

Use when the question is the risk of losing a supplier or an item, what the fallback is, how to prepare for a disruption or an insolvency, or what to do when the warning signs appear.

Not for the sourcing strategy of the category (category-strategy), for the emergency itself (urgent-sourcing), for the due-diligence verdict on a supplier (supplier-risk-assessment) or for the sustainability dimension (supplier-sustainability-review).

## Inputs and preconditions

- The supplier or item, and the concern (insolvency, disruption, single-source, geography).
- Stock on hand, lead times and consumption, from the user, labelled user-supplied.
- The tolerance: how long the operation can run without supply, if known.

- Use `tool:build_spend_cube` for what we spend with the supplier and its share of total (the cube has no category)
- Use `tool:supplier_discovery` with each item description, for the other suppliers on our records that provide it, with evidence counts
- Use `tool:screening_summary` for the alternatives' screening status
- Use `tool:financial_health_score` for the supplier's recorded financial health check and its date
- Use `tool:monitor_risk_events` for recorded risk events against the supplier
- Use `tool:search_contracts` for the agreement
- Use `tool:extract_clauses` for force-majeure, business-continuity, step-in, stock-holding, exclusivity and termination-on-insolvency clauses, quoted
- Use `tool:get_policy` for any continuity or minimum-stock rule the organisation has set
- Use `tool:check_policy` where a backup agreement or a stock build commits spend
- Use `tool:supplier_items` for the items we buy from the supplier with quantities and last dates: the exposure is these items
- Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated
NOT AVAILABLE: create_purchase_order. nothing is ordered; the plan names what would be ordered and the route

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.min_cover_days`.

## Workflow

1. Frame: supplier or item, concern, tolerance, what the user knows about stock and lead times.
2. Exposure: spend and share of total from the cube; the items from supplier_items; items with no other supplier on record per supplier_discovery.
3. Alternatives: the suppliers on record for each item from supplier_discovery with their evidence counts (it holds no prices); screening status per candidate; days to first delivery from the user's lead times where given.
4. Signals: financial health record and date; risk events on record; the contract's insolvency, force-majeure and continuity clauses quoted; notice we would receive.
5. Gap: items where no screened alternative exists, or where the alternative's lead time exceeds the stock cover the user states. Cover below `config.thresholds.min_cover_days` days with no screened alternative is a red gap.
6. Plan: preparations now (screen a backup, agree a backup contract, stock build, drawings and tooling ownership, data escrow) with owners and approval routes; triggers to watch; the first actions on the day in order; who decides.

## Frameworks and heuristics

- Exposure is items, not spend: the cheapest item can stop the line.
- A backup that is not screened is not a backup.
- Tooling, drawings, data and stock held by the supplier are hostages; the plan names where they are.
- Triggers are facts you will see (late payments to their suppliers, a credit downgrade, a notice), not feelings.
- Prepare in proportion to the gap: a red gap gets a backup contract, an amber one gets a screened candidate.

## Policy hooks (SC-007)

Continuity and minimum-stock rules come from `tool:get_policy` and are quoted. A backup agreement, a stock build or a tooling transfer that commits spend is checked with `tool:check_policy` and reported as replied. Nothing is ordered, screened or signed by the skill; the plan names the acts and their owners.

## Output specification

1. Frame and exposure: spend and share of total, items from supplier_items, items with no other supplier on record.
2. Alternatives table: supplier, items, price and date, screening status, lead time (user-supplied).
3. Signals: financial health and date, risk events, clauses quoted.
4. Gaps by colour with the rule applied.
5. Plan: preparations with owners and approval routes; triggers; day-one actions in order; decision owner.

## Quality bar

- Every alternative is on our records with its screening status; none from memory.
- Stock and lead-time figures are the user's, labelled.
- Clauses are quoted; a missing clause is named as missing.
- Nothing is ordered or signed.

## Escalation and approval

Escalate to the category lead for red gaps, to compliance to screen backups, to legal for continuity and escrow clauses, and to urgent-sourcing when a trigger fires.

## Worked example

Asked what happens if a machined-parts supplier goes into administration, the skill reads the cube (a notable share of total spend) and the supplier's items (supplier_items), finds one alternative on record for most items (screened) but none for two parts, reads the financial health record (dated, lower rating), the contract's insolvency termination clause and the absence of a tooling clause, notes the user's stock cover is short for the two parts, marks them red, and plans: screen a second candidate found by discovery, confirm tooling ownership with legal, a stock build for the two parts via the approval route, triggers to watch, and the first-day actions in order.

## Anti-patterns

- Calling a supplier a backup without screening or records of the items.
- Estimating lead times or stock the user has not given.
- Planning on spend share while an unspent critical item has no alternative.
- Ordering stock or signing a backup from the skill.
