---
name: spend-analysis
description: Analyse the organisation's spend from its own invoices and lines - who it is spent with, how concentrated it is, the tail, how much is classified, where the same thing is bought under different descriptions, and what has been saved - with the same totals the dashboards show. Use when asked for a spend analysis, breakdown or profile, the top suppliers by spend, tail or maverick spend, category spend coverage, or a pareto view. Not for a simple one-figure lookup, for pooling demand (demand-aggregation) or for validating a saving (savings-methodology).
---

# Spend Analysis

## Purpose

Turn the invoices on record into an honest spend picture a buyer can act on: totals by the dimensions the invoices carry, concentration and tail, classification coverage against the organisation's taxonomy, near-duplicate buying, and savings already recorded. Every figure comes from a tool and carries its period and currency rule; gaps in the data are stated, not filled.

## When to use and when not

Use when the user wants an analysis rather than one number: a breakdown or profile of spend, top suppliers, concentration or pareto, tail or maverick spend, how much is classified, or where money goes by month, currency, country, region, buyer, payment terms or status.

Not for a single figure ("how much did we spend with X", which the Ask box answers from the spend cube directly), for sizing a pooling opportunity (demand-aggregation, which this skill hands candidates to), for validating a saving (savings-methodology), or for a supplier's performance (supplier-performance-review).

## Inputs and preconditions

- Period: default the trailing `config.thresholds.default_period_months` months, or the user's dates; reporting currency, default the dashboards' currency.
- Scope: the whole organisation, or a supplier, country, region, buyer or status to cut by.
- Any extract the user brings for dimensions the invoices do not carry (business unit, category code, cost centre), labelled as user-supplied.

Use `tool:build_spend_cube` for invoiced spend cut by supplier, month, currency, country, region, buyer, payment terms or status; its totals equal the dashboards and its note states the conversion rule and any currency left out. Invoices carry no category, no business unit and no price per item: say so. Use `tool:classify_spend` for classification coverage of item descriptions against the taxonomy, `tool:similar_items` to find a thing bought under different descriptions, `tool:supplier_lookup` and `tool:supplier_discovery` for who the suppliers are, `tool:savings_tracker` for what the value ledger says has been saved, `tool:get_corpus_facts` for corpus-wide counts (never for one item's spend), and `tool:run_calculation` with `deal.share_pct` for shares and `deal.pct_change` for changes between periods.

Use `tool:build_xlsx` when the user asks for a spreadsheet of the figures, with the tables as the tools returned them. Use `tool:render_chart` when the user asks for a chart, with the values the tools returned.

## Workflow

1. Frame. Confirm period, currency and scope; state the data boundary: invoices on record, converted once per currency at the stated rate, no category or business-unit dimension unless the user supplies one.
2. Totals. Run the spend cube by supplier and by month for the period; report the total, the invoice count and any excluded currencies. Compare with the same period a year earlier only when both are read from the tool, with the change from `deal.pct_change`.
3. Concentration. From the supplier cut: the top `config.thresholds.top_n_suppliers` suppliers with their shares, how many suppliers make up `config.thresholds.pareto_share_pct` percent of spend, and the tail: the suppliers whose combined share is the bottom `config.thresholds.tail_spend_pct` percent, with their count. Single-supplier dependence is noted where one supplier exceeds `config.thresholds.single_supplier_alert_pct` percent.
4. Dimensions. Cut by the dimensions asked for (country, region, buyer, payment terms, status), each from the cube, each with its total and any excluded currency.
5. Classification. Take a sample of item descriptions the user names or the top items by occurrence and run `tool:classify_spend`; report recorded coverage against `config.thresholds.coverage_floor_pct`, and list draft candidates as draft classification only.
6. Duplication. For items the user names, run `tool:similar_items` at `config.thresholds.similarity_min` and report near-identical descriptions with their supplier and buyer counts as candidates for demand-aggregation, never as confirmed duplicates.
7. Savings. Read `tool:savings_tracker` for the period and report saved against claimed, as the ledger states them.
8. Findings and next steps. List what the analysis shows, what it could not show and why, and which candidates go to demand-aggregation, supplier-performance-review or negotiation-planning.

## Frameworks and heuristics

- One rule for totals: the spend cube's, so an analysis figure always equals the dashboard's. A total with an excluded currency is reported with that currency shown as billed.
- Pareto and tail are computed from the cube's supplier shares in the display currency; cells with no converted amount are listed, not counted.
- A corpus-wide count is never an item's count: use tools that take the item as a parameter.
- Period comparisons use equal windows and the same currency rule; otherwise no comparison is made.
- Classification is the organisation's own: a name-match candidate is a draft, and uncovered spend is reported as uncovered.

## Policy hooks (SC-007)

Spend analysis changes nothing and needs no approval. When a finding leads to an action that commits spend or changes a supplier relationship, the receiving skill runs `tool:check_policy`; the only governed inputs read here are the reporting currency and period conventions via `tool:get_policy` where a policy states them.

## Output specification

1. Frame: period, currency, scope, data boundary and the conversion rule.
2. Totals: spend, invoices, by month; change against the prior period when read.
3. Concentration: top suppliers with shares, pareto count, tail count and share, dependence alerts.
4. Dimension cuts requested, each with total and exclusions.
5. Classification coverage and draft candidates; duplication candidates with evidence.
6. Savings recorded in the period.
7. What could not be shown and why; candidates handed to other skills.

## Quality bar

- Every figure traces to a tool result with period and currency; no total is summed across currencies without a rate.
- No category, business-unit or item-price figure is stated unless the user supplied the dimension or a tool returned it.
- Shares come from the cube's own share field or `deal.share_pct`, changes from `deal.pct_change`, never from the model's arithmetic across cuts.
- Gaps are listed in their own section.

## Escalation and approval

No approval is needed for the analysis. Escalate single-supplier dependence above the alert share to the category lead, and classification coverage below the floor to whoever owns the taxonomy. Actions arising go through the skill that owns them.

## Worked example

Asked for a spend analysis for the year to date, the skill runs the cube by supplier and by month (total, invoice count, two currencies without a rate shown as billed), finds that a small number of suppliers make up the pareto share and that the tail holds many suppliers with a small share, cuts by country as asked, classifies the top ten descriptions (none recorded; candidates listed as draft), finds a laptop description bought under several near-identical names from many suppliers and hands it to demand-aggregation, and reports the saved and claimed figures from the ledger. It closes with what the invoices cannot show: category and business-unit totals.

## Anti-patterns

- Quoting a corpus-wide count or ranking as the answer for one item or category.
- Adding amounts across currencies, or comparing unequal periods.
- Presenting a draft classification candidate as the category.
- Calling near-identical descriptions duplicates before a person confirms them.
- Inventing a category or business-unit split the invoices do not carry.
