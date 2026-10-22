---
name: csrd-esg-reporting
description: "Prepare the supply-chain part of an ESG or CSRD report - Scope 3 purchased-goods emissions from activity data and factors the user supplies, the coverage of suppliers by screening and rating from the register, the risk checks performed and not performed, the due-diligence steps taken, and the data gaps and estimates stated as such - as a sourced data pack for the reporting team. Use when asked for supply chain ESG figures, Scope 3 from procurement, or the procurement section of a sustainability report. Dormant pending review. Not for the supplier-level review (supplier-sustainability-review) or the sourcing criteria (sustainable-sourcing)."
---

# Supply Chain ESG Reporting

## Purpose

Give the reporting team numbers they can defend: every figure sourced, every factor named, every gap stated, nothing estimated by the skill; the report's assurance rests on the trail.

## When to use and when not

Use when procurement must contribute figures or narrative to an ESG, CSRD or sustainability report.

Not for reviewing suppliers' ESG status (supplier-sustainability-review), for tender criteria (sustainable-sourcing), for the policy (ethical-sourcing-policy) or for the company's own Scope 1 and 2.

## Inputs and preconditions

- The reporting period, the standard's requirements as the reporting team states them, the categories in scope.
- Activity data (spend or quantities by category or supplier) and emission factors with sources, from the user or the reporting team.

- Use `tool:build_spend_cube` for spend by supplier, country and month in the period as the activity base where spend-based factors are used
- Use `tool:classify_spend` for the taxonomy's categories and recorded ESG impact
- Use `tool:screening_summary` for coverage: suppliers with an ESG rating, forced-labour and conflict-minerals checks on record, per supplier in scope
- Use `tool:supplier_lookup` for certifications on file and countries
- Use `tool:run_calculation` with `sustainability.scope3` for emissions per category from activity data and stated factors, `deal.share_pct` for coverage shares and `deal.pct_change` for the movement against the prior period
- Use `tool:build_xlsx` for the data pack workbook with sources per figure
- Use `tool:get_policy` for the reporting rules (method, factor sources, assurance requirements)
NOT AVAILABLE: send_esg_survey. no survey is sent; data requests to suppliers go through the reporting team
NOT AVAILABLE: scope3_emissions_calc. the registered calculation sustainability.scope3 does this on supplied factors; nothing estimates factors

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.supplier_specific_target_pct`.

## Workflow

1. Frame: period, standard, categories and suppliers in scope, the method the reporting rules state (spend-based, activity-based, supplier-specific), the factor sources.
2. Activity base: spend by supplier and country from the cube for the period; quantities where the user has them; mapped to categories via the taxonomy; unmapped spend stated as a gap.
3. Emissions per category via `sustainability.scope3` from the activity base and the stated factors (source per factor); lines without a factor counted and reported as a lower bound; the movement against the prior period via the registry.
4. Coverage: suppliers in scope with an ESG rating, forced-labour and conflict-minerals checks on record, as shares via the registry; checks not performed listed as such.
5. Due diligence narrative: the steps taken (screening, audits, clauses) from the records, with counts; no step claimed that the records do not show.
6. Data pack: every figure with its source, factor and date in a workbook; gaps and estimates (the reporting team's) listed; the share of spend with a supplier-specific figure above `config.thresholds.supplier_specific_target_pct` percent is the quality target, stated as met or not.

## Frameworks and heuristics

- A factor without a source is not a factor.
- Spend-based is a start, not an answer; say which method each line uses.
- Coverage is a share of spend, not a count of suppliers.
- Gaps reported are assurance; gaps hidden are findings.

## Policy hooks (SC-007)

Reporting method, factor sources and assurance requirements come from `tool:get_policy` and are quoted. Screening comes from the register; nothing is estimated by the skill; the reporting team owns the report.

## Output specification

1. Frame: period, standard, scope, method, factor sources.
2. Emissions table per category via the registry with factors, sources, method and lower-bound flags; movement.
3. Coverage shares via the registry; checks not performed.
4. Due-diligence narrative from the records with counts.
5. Data pack workbook export id; gaps and estimates list; quality target status.

## Quality bar

- Every figure has a source, a factor and a date.
- Emissions, shares and movements come from the registry.
- Lines without factors are counted, never filled.
- Nothing is claimed that the records do not show.

## Escalation and approval

Escalate to the reporting team for factors and method, to compliance for screening coverage, and to the sustainability owner for the quality target.

## Worked example

Asked for the procurement section of the ESG report, the skill reads the reporting rules (spend-based for most categories, supplier-specific where available), takes spend by supplier and country from the cube, maps it through the taxonomy with the unmapped share stated, computes emissions per category via the registry with the reporting team's factors and sources (two categories lower bound for lack of factors), reports coverage of spend by rating and checks from the register via the registry, writes the due-diligence steps from the records with counts, and builds the data pack with a source per figure.

## Anti-patterns

- An emission factor from memory.
- Coverage by supplier count.
- A due-diligence step the records do not show.
- Filling a gap with an estimate of the skill's own.
