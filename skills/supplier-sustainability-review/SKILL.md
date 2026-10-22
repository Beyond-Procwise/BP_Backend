---
name: supplier-sustainability-review
description: Review the sustainability and ethical-sourcing position of a supplier, a category or the supply base from what is on record - ESG ratings, forced-labour and conflict-minerals checks compliance recorded, certifications on file, the taxonomy's ESG impact per category and spend coverage - and set out the gaps, the questionnaire or evidence to request and the priorities. Use when asked about sustainability, ESG performance or risk, carbon or emissions, modern slavery, ethical or responsible sourcing, or a supplier's environmental credentials. Not for clearance of one supplier (supplier-risk-assessment) or for an ESG rating on its own.
---

# Supplier Sustainability Review

## Purpose

Show where the organisation stands on supplier sustainability with evidence: which suppliers and how much spend have a recorded ESG rating or check, which categories the taxonomy marks as high impact, which certifications are on file, and what is missing. The skill assesses nothing itself and holds no emission factors: it reads what compliance and the supplier master record, and says what to ask for next.

## When to use and when not

Use when the user asks for a sustainability or ESG review of a supplier, a category or the supply base, for coverage of ESG ratings or ethical-sourcing checks, for carbon or scope 3 readiness, or for the evidence to request from suppliers.

Not for the risk clearance of one supplier before award or onboarding (supplier-risk-assessment, which this skill feeds), for a single ESG rating lookup (the esg_rating_lookup tool answers that directly), for the category strategy itself (category-strategy) or for the tender's sustainability criteria (rfx-design).

## Inputs and preconditions

- The scope: one supplier, the suppliers of a category, or the top suppliers by spend; the period and currency for spend.
- The organisation's sustainability policy and targets (minimum rating, required checks, reporting obligations) from `tool:get_policy`; where none is stated, the review reports coverage without a pass mark and says the target is not governed.
- Any supplier questionnaire results or emission data the user holds, labelled user-supplied.

Use `tool:build_spend_cube` by supplier for the spend in scope, `tool:supplier_lookup` for certifications on file (environmental, social, EcoVadis-type flags) and country, `tool:screening_summary` for each supplier's recorded ESG rating, forced-labour and conflict-minerals checks and risk events in one call, `tool:esg_rating_lookup`, `tool:forced_labour_screen` and `tool:conflict_minerals_check` when one check is asked about, `tool:classify_spend` with `query` for the taxonomy's recorded ESG impact of a category, and `tool:search_contracts` for sustainability or code-of-conduct clauses via `tool:extract_clauses` (one `containing` word per call, at most two: "sustainab" then "conduct").

NOT AVAILABLE: send_esg_survey. The questionnaire to send is drafted as text; a person sends it.
NOT AVAILABLE: build_xlsx. Give the coverage table as a text table.
NOT AVAILABLE: render_chart. Describe coverage and impact as tables.

## Workflow

1. Scope. Fix the supplier set (one, a category's, or the top `config.thresholds.top_n_suppliers` by spend), the period and the currency; read the policy for targets and required checks.
2. Impact. For the categories in scope, read the taxonomy's recorded ESG impact with `tool:classify_spend`; a category marked high impact gets the fuller review. The impact label is the organisation's own, as recorded.
3. Evidence per supplier. For each supplier: spend in scope, ESG rating on record (provider, rating, date; stale beyond `config.thresholds.rating_max_age_days` days), forced-labour and conflict-minerals checks where the category or geography makes them relevant, certifications on file, risk events, and sustainability or code-of-conduct clauses in the live contract. "Not performed", "no rating recorded" and "no clause stored" are findings, not blanks.
4. Coverage. Share of spend in scope with an in-date rating, with the relevant checks performed, and with a sustainability clause; against `config.thresholds.coverage_floor_pct` percent where no policy states a floor. Shares use the cube's amounts in one currency; excluded currencies are shown as billed.
5. Gaps and priorities. Rank suppliers by spend and impact: high-impact, high-spend suppliers with no rating or an unperformed check come first. For each, name the evidence to request (rating or questionnaire, specific check, certification, clause at renewal) and who acts (compliance records checks; the buyer requests evidence; legal adds clauses).
6. Targets. Where policy states a minimum rating or required checks, report each supplier against it as met, not met or not assessed; where it does not, report coverage only and say the target is not governed.
7. Questionnaire. Draft the evidence request to suppliers as text, limited to what the policy or the gaps call for; nothing is sent from the skill.
8. Output the review (section below) and hand single-supplier clearance to supplier-risk-assessment and tender criteria to rfx-design.

## Frameworks and heuristics

- Evidence has a provider and a date; a certificate on file is a certificate, not a rating; a claim in an email is neither.
- Impact first, then spend: a small high-impact supplier can outrank a large low-impact one for attention; both are shown.
- Coverage is of spend, not of supplier count, so the review reflects where the money and the exposure sit.
- Carbon and scope 3 figures are the user's or a provider's with their method stated; the skill computes none.
- Modern-slavery and conflict-minerals relevance follows the category and geography the user or the taxonomy states, not the model's assumption.

## Policy hooks (SC-007)

Minimum ratings, required checks and reporting obligations come from governed policy via `tool:get_policy`; without a policy the review reports coverage and says the target is not governed. Nothing here commits spend; actions arising (a clause at renewal, a supplier change) go through their own skills and `tool:check_policy`.

## Output specification

1. Scope, period, currency; policy targets or "not governed"; categories with their recorded ESG impact.
2. Supplier table: supplier, spend, share, impact, rating (provider, date, stale), checks performed / not performed, certifications on file, clause stored, risk events.
3. Coverage: share of spend with in-date rating, with required checks, with a clause; against floor or policy.
4. Gaps ranked by impact and spend, each with the evidence to request and the owner.
5. Target assessment per supplier where policy states targets.
6. Questionnaire text; hand-offs.

## Quality bar

- Every rating, check, certification and clause comes from a tool result with its date where one exists; nothing is inferred from a company's name, website or sector.
- Coverage shares come from the cube's amounts in one currency; nothing is summed across currencies.
- Not performed and not recorded are stated as such and never treated as compliant.
- No emission figure or score is produced by the skill.

## Escalation and approval

Escalate to compliance any forced-labour or conflict-minerals check not performed in a high-impact category, and to the category lead any high-spend supplier with no rating. The review and the questionnaire are advice; compliance records checks and a person sends requests.

## Worked example

Asked for a sustainability review of the packaging suppliers, the skill reads the taxonomy (packaging categories marked high impact), takes the top suppliers by spend from the cube, reads each screening summary (two have an in-date ESG rating, one has a stale one, five have none; forced-labour checks performed for none), the master record (three hold an environmental certification), and the live contracts (one stores a code-of-conduct clause). It reports coverage of spend with an in-date rating below the floor, ranks the two largest unrated suppliers first, drafts the evidence request, and lists the forced-labour checks for compliance to perform (a person sends the request).

## Anti-patterns

- Rating a supplier from its name, website, sector or a certificate.
- Treating "no check recorded" as compliant.
- Computing carbon or scope 3 figures from assumptions.
- Reporting coverage by supplier count when the question is exposure.
- Claiming a questionnaire was sent.
