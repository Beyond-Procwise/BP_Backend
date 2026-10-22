---
name: supplier-segmentation
description: "Place each supplier in a segment (strategic, critical or bottleneck, leverage, transactional) from the spend and its share, the findings, the contract cover and the risk on record, and say what treatment each segment gets. Use when asked to segment or tier the supplier base, to name the strategic, critical or key suppliers, or to decide which suppliers deserve a relationship plan. Not for one supplier's performance (supplier-performance-review) or a category's sourcing strategy (category-strategy)."
---

# Supplier Segmentation

## Purpose

Give the supplier base a shape people can act on. Each supplier lands in one segment on evidence from the records, with the treatment the segment implies (relationship plan, scorecard, contract, consolidation or catalogue). The skill reads and proposes; it changes no record.

## When to use and when not

Use when the question is which suppliers matter most, how to tier or segment the base, who the strategic or critical suppliers are, or where to put relationship effort.

Not for one supplier's scorecard (supplier-performance-review), for the category plan (category-strategy), for cutting the tail (tail-spend-management) or for risk screening (supplier-risk-assessment).

## Inputs and preconditions

- The scope: the whole base, a category or a list of suppliers, and the period (default the last full year).
- The axes the user wants, if any; otherwise spend and criticality as described here.
- Any criticality the user knows that the records do not hold (sole-source items, safety-critical parts), labelled user-supplied.

- Use `tool:build_spend_cube` for spend by supplier over the period with the share each supplier holds (the cube has no category; a category scope comes from the user's list of suppliers or from `tool:classify_spend` on their items)
- Use `tool:calc_supplier_scorecard` for open findings, on-time delivery and price variance where a scorecard is wanted for a candidate strategic supplier
- Use `tool:search_contracts` for whether a supplier is under a live agreement
- Use `tool:screening_summary` for the screening status of suppliers proposed as strategic
- Use `tool:similar_items` for whether the items a supplier provides have alternatives on our records, which tells leverage from bottleneck
- Use `tool:get_policy` for any segmentation rule or treatment the organisation has set
- Use `tool:run_calculation` with `deal.share_pct` for a share the cube does not give directly
NOT AVAILABLE: build_docx. the segmentation is given as text and tables

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.strategic_share_pct`.

## Workflow

1. Frame the scope and period, and read the policy for any segmentation rule; where none exists, use the axes here and say so.
2. Read the spend cube by supplier for the period and take each supplier's share from the cube's own share figure; a share the cube does not give (a supplier's share of a user-named group) comes from `run_calculation` with `deal.share_pct`, never from arithmetic in the answer. Suppliers at or above `config.thresholds.strategic_share_pct` of total spend are candidates for strategic or leverage; a supplier the user names as holding a whole category is a candidate on that statement, labelled user-supplied.
3. Judge supply difficulty from the records only: an item with no alternative supplier on record (similar items) is bottleneck territory; the user's list of critical items is added, labelled user-supplied.
4. Place each supplier: high spend and hard to replace is strategic; low spend and hard to replace is bottleneck; high spend and easy to replace is leverage; the rest is transactional.
5. For strategic and bottleneck candidates read the contract cover and the screening status; a strategic supplier with no live agreement or an incomplete screening is a finding.
6. Name the treatment per segment and the first three actions, each with an owner.

## Frameworks and heuristics

- Kraljic's two axes (profit impact, supply risk) applied with the records' own evidence, not with a scored survey.
- Criticality is about the item, not the spend; a small supplier can hold the only source of a part, and the records show that through the absence of alternatives.
- A segment is a working hypothesis the category lead confirms; the skill never calls a supplier strategic on spend alone when the item has alternatives.
- Treatment follows segment: relationship plan and scorecard for strategic; alternatives and stock for bottleneck; competition and consolidation for leverage; catalogue and card for transactional.

## Policy hooks (SC-007)

Any segmentation rule or required treatment comes from `tool:get_policy` and is quoted. Spend commitments that follow from the plan (a new agreement with a strategic supplier) are checked with `tool:check_policy` and reported as replied. No record is changed and no supplier is told its segment.

## Output specification

1. Frame: scope, period, axes used and whether they come from policy or from this skill.
2. Segment table: supplier, spend, share of total (from the cube), replaceability evidence, segment, contract cover, screening status.
3. Findings: strategic suppliers without an agreement or with incomplete screening; bottleneck items with no alternative.
4. Treatment per segment and the first actions with owners.
5. Open questions for the category lead.

## Quality bar

- Every share comes from the spend cube; no spend figure from memory.
- Replaceability is evidenced by alternatives on record or the user's statement, labelled.
- Suppliers with no spend in the period are listed as such, not segmented.
- The segment is proposed, not recorded.

## Escalation and approval

Escalate to the category lead to confirm strategic and bottleneck placements, to compliance where a strategic supplier's screening is incomplete, and to legal where one has no live agreement.

## Worked example

Asked to segment the packaging base for last year, the skill reads the cube (nine suppliers, two holding most of the category), finds the largest has a live agreement and a complete screening (strategic), the second has alternatives on record for every item (leverage), one small supplier provides a film with no alternative on record (bottleneck, escalated), and six are transactional; it proposes a relationship plan for the first, a competitive event for the second and a search for a second film source.

## Anti-patterns

- Calling a supplier strategic because the spend is large when its items have alternatives.
- Scoring criticality from a survey the records do not hold.
- Changing a supplier's record or telling the supplier its segment.
- Treating a one-off project spend as a recurring strategic relationship.
