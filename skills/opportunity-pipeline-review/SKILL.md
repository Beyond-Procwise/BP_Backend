---
name: opportunity-pipeline-review
description: Review the pipeline of value opportunities and what has been realised - the levers the platform tracks per supplier and item, the opportunity buckets with their status, the value ledger's saved, recovered, avoided and still-claimed money for a period, and the candidates the other skills surfaced - and say what is recorded, what is only claimed, what is stalled and what to pursue next. Use when asked how the savings pipeline looks, what has been saved or delivered so far, where the value opportunities are, or which levers are open. Not for validating one saving (savings-methodology) or sizing one pooling case (demand-aggregation).
---

# Opportunity Pipeline Review

## Purpose

Show, from the platform's own records, the state of value delivery: what the ledger says has been saved and what is still only claimed, which opportunity levers are open per supplier and item and where they stand, and what to do next. The review reads the ledger and the buckets; it records no saving, moves no opportunity, and sizes nothing without a calculation or the user's figures.

## When to use and when not

Use when the user asks for the savings or value pipeline, what has been saved or delivered so far or this year, where the open opportunities or levers are, what is stalled, or what to pursue next.

Not for validating a single saving claim (savings-methodology), for sizing a pooling or harmonisation case (demand-aggregation), for the lever on one deal (the opportunity tools answer that directly), or for the negotiation that realises a lever (negotiation-planning).

## Inputs and preconditions

- The period for realised value (default the trailing `config.thresholds.default_period_months` months) and the reporting currency.
- The scope: the whole organisation, a supplier, an item or a category's suppliers.
- The organisation's rules on what counts as a saving and who may record one: read `tool:get_policy` once for a savings rule; where none governs, the ledger's own rule as `tool:savings_tracker` states it in its note is the rule to quote. Start with `tool:savings_tracker`, then `tool:list_opportunity_levers`, then the buckets; the round budget is small.

Use `tool:savings_tracker` for the period's saved (avoided, recovered, realised) against claimed money, month by month, as the ledger states them; `tool:list_opportunity_levers` for the levers the platform tracks and what each means; `tool:get_opportunity_bucket` with the supplier_id and item_id (the item_id from `tool:supplier_items` or `tool:similar_items`, never guessed) for the bucket's status, evidence and lever state; `tool:build_spend_cube` by supplier for the spend the pipeline sits on; `tool:similar_items` and `tool:classify_spend` to size the ground for a pooling candidate before handing it to demand-aggregation; `tool:calc_supplier_scorecard` for the price-variance and findings evidence behind a recovery lever; and `tool:run_calculation` with `deal.pct_change` for shares and changes and `deal.realised_savings` where the user gives baseline and new prices with volume.

NOT AVAILABLE: build_xlsx. Give the pipeline as a text table.
NOT AVAILABLE: render_chart. Describe the monthly profile as a table.

## Workflow

1. Frame. Fix the period, the currency and the scope; read the saving rules or say they are not governed; quote the ledger's definition of saved from the tracker's note.
2. Realised. From the tracker: saved by kind (avoided, recovered, realised) and the money still only claimed, month by month; the share claimed against saved via `deal.share_pct` only where both are read. Claimed money is never counted as saved.
3. Open levers. From the lever list and the buckets for the suppliers and items in scope: each open lever with its bucket status, evidence, when it was last evaluated, and whether it is stalled (its recorded detection or review-request date older than `config.thresholds.stalled_days` days with no later date on the bucket). A bucket with no recorded evaluation is "not evaluated", not "no opportunity".
4. Value at stake. For each open lever, the value the bucket records (as recorded) or "not sized"; the skill sizes nothing itself except through `deal.realised_savings` on figures the user gives, labelled as the user's.
5. Candidates. Note candidates other skills have surfaced (near-identical items from similar_items, price variance and open findings from the scorecard, index-driven increases) as "candidate, not yet a bucket", with the skill that would size them.
6. Ranking. Order open levers by recorded value, then by age; flag any above `config.thresholds.escalate_value` in the reporting currency that is stalled.
7. Next steps. For each of the top `config.thresholds.top_n` items: the lever, the owner to confirm, the skill that carries it (negotiation-planning, demand-aggregation, overbilling-investigation, renewal-and-exit-strategy), and what evidence is still needed. Recording a saving is a person's act with evidence through the ledger; the skill says so.

## Frameworks and heuristics

- Saved is what the ledger confirms; claimed is a promise; the two are never added.
- A lever is open until a bucket says otherwise; a missing evaluation is a gap in our work, not a closed opportunity.
- Value at stake is recorded or user-supplied; the quadrant, the category or the gut do not size it.
- Stalled beats small: an old, unmoved lever with material value goes to the top of the asks.
- Every next step names the skill that does the work and the person who owns the decision.

## Policy hooks (SC-007)

What counts as a saving and who may record one come from governed policy via `tool:get_policy` and from the ledger's rule as the tracker states it; realising a lever that commits spend or changes a supplier runs through the carrying skill and `tool:check_policy`. Nothing is recorded or moved here.

## Output specification

1. Frame: period, currency, scope, the ledger's definition quoted. Write for the procurement lead in plain words: suppliers, items, levers, amounts and dates; never name tools, tables or how the records are stored.
2. Realised table: saved by kind, claimed open, by month; share claimed where read.
3. Open levers table: supplier, item, lever, bucket status, evidence, last evaluated, stalled, value recorded or "not sized".
4. Candidates not yet in a bucket, with the sizing skill.
5. Ranked asks: lever, owner to confirm, carrying skill, evidence needed; escalations.
6. Data gaps: buckets not evaluated, spend without levers, ledger entries lacking evidence.

## Quality bar

- Every saved, claimed, lever and bucket figure comes from a tool result; nothing is sized by the model.
- Claimed money is shown apart from saved in every table and total.
- Stalled and not-evaluated are stated with their dates, never hidden.
- No saving is recorded, no bucket moved, no supplier contacted.

## Escalation and approval

Escalate stalled levers above the configured value to the category lead, and ledger entries without evidence to finance. Realisation and recording follow the carrying skills and the approval route.

## Worked example

Asked how the savings pipeline looks this year, the skill reads the ledger (saved by kind and month, a claimed amount still open and shown apart), lists the levers the platform tracks, reads the buckets for the suppliers in scope (two levers open and evaluated recently, one stalled beyond the configured age with recorded value above the escalation level, three buckets never evaluated), notes a laptop pooling candidate from similar_items not yet in a bucket, ranks the stalled lever first with negotiation-planning as the carrying skill and the category lead as owner, and lists the three unevaluated buckets as a gap.

## Anti-patterns

- Adding claimed money into saved.
- Calling an unevaluated bucket "no opportunity".
- Sizing a lever from the model's judgement.
- Recording a saving or moving a bucket from this skill.
- Reporting a candidate as a lever before a bucket exists.
