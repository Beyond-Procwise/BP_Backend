---
name: sustainable-sourcing
description: "Build sustainability into a sourcing decision - the criteria that fit the category's recorded impact, their weight in the evaluation per policy, the evidence bidders must return (ratings, certifications, emissions data with factors), how the evidence is scored, and the clauses the contract will carry - from the taxonomy's recorded impact, the policy, the screening register and the registered emissions calculation. Use when asked to add sustainability criteria to a tender, score bidders on sustainability, or set sustainability requirements for a contract. Dormant pending review. Not for reviewing suppliers' current ESG status (supplier-sustainability-review) or the reporting (csrd-esg-reporting)."
---

# Sustainable Sourcing

## Purpose

Put sustainability into the decision where it changes the outcome: criteria tied to the category's impact, evidence that can be scored, and clauses that bind; never a box ticked with a logo.

## When to use and when not

Use when a sourcing event or a contract is being designed and sustainability must be part of the criteria, the evidence or the terms.

Not for the status review of existing suppliers (supplier-sustainability-review), for the organisation's ESG reporting (csrd-esg-reporting), for the policy itself (ethical-sourcing-policy) or for the general tender design (rfx-design, which this skill feeds).

## Inputs and preconditions

- The category and event; the policy's sustainability requirements.
- Bidders' submitted evidence when scoring, labelled by bidder and date.

- Use `tool:classify_spend` for the taxonomy's recorded ESG impact of the category
- Use `tool:get_policy` for the sustainability rules (mandatory criteria, minimum weight, required evidence, banned practices)
- Use `tool:screening_summary` for each bidder's recorded ESG rating, forced-labour and conflict-minerals checks
- Use `tool:esg_rating_lookup` when a bidder's rating alone is asked about
- Use `tool:search_contracts` for sustainability clauses in comparable agreements
- Use `tool:extract_clauses` for code-of-conduct, emissions-reporting and audit clauses, quoted
- Use `tool:run_calculation` with `sustainability.scope3` for a bidder's emissions from their activity data and stated factors, `scoring.weighted_total` for the sustainability score within the evaluation, and `deal.share_pct` for the criterion's weight share
NOT AVAILABLE: send_esg_survey. no survey is sent; the evidence request goes in the tender pack issued by a person

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.min_weight_high_impact_pct`.

## Workflow

1. Impact: the category's recorded ESG impact from the taxonomy; the policy's mandatory criteria and minimum weight quoted; where the impact is high the weight is at least `config.thresholds.min_weight_high_impact_pct` percent unless policy says otherwise.
2. Criteria: emissions (with factors stated), certifications relevant to the impact, labour practice evidence, materials and circularity, each with the evidence bidders must return and how it is scored; mandatory versus scored.
3. Evidence request text for the tender pack (rfx-design issues it): what, in what form, with what proof.
4. Scoring: each bidder's evidence against the criteria; screening on record per bidder (not performed is not a pass); emissions via the registry from their data and factors, labelled; the weighted score via the registry.
5. Clauses: code of conduct, reporting obligations, audit right, improvement commitments, drawn from comparable agreements quoted or named as gaps for legal.
6. Hand the criteria and weights to rfx-design and the scores to award-recommendation.

## Frameworks and heuristics

- Weight follows impact; a high-impact category with a token weight is theatre.
- Evidence is scored; a claim is not.
- A rating on record beats a brochure; a certificate on file beats a rating claim.
- The clause is what survives the award.

## Policy hooks (SC-007)

Mandatory criteria, minimum weights and required evidence come from `tool:get_policy` and are quoted; screening comes from the register. Nothing is sent, scored as final or awarded by the skill; the evaluation model and the award carry their own approvals.

## Output specification

1. Category impact and the policy's requirements.
2. Criteria table: criterion, mandatory or scored, evidence required, scoring rule, weight.
3. Evidence request text for the pack.
4. Scores per bidder with evidence and screening status; emissions via the registry where supplied.
5. Clauses quoted or named as gaps; hand-offs.

## Quality bar

- Weights and mandatory criteria cite the policy.
- Every score rests on evidence or a record; a claim scores nothing.
- Emissions and scores come from the registry with the factors stated.
- Nothing is sent or awarded.

## Escalation and approval

Escalate to the sustainability owner for weights above or below policy, to compliance for a bidder's screening, and to legal for the clauses.

## Worked example

Asked to add sustainability to the packaging tender, the skill reads the taxonomy's high impact for the category and the policy's minimum weight and mandatory evidence, sets the criteria (recycled content with certification, emissions per tonne from the bidder's data and stated factors, labour-practice evidence), writes the evidence request for the pack, and when the bids arrive scores each on evidence with screening read from the register and emissions via the registry, weights them via the registry, and quotes the code-of-conduct and reporting clauses from a comparable agreement for legal.

## Anti-patterns

- Scoring a claim without evidence.
- A weight below the policy's minimum for a high-impact category.
- Emissions from the skill's own factors.
- Treating an unperformed screening as a pass.
