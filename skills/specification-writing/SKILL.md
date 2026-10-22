---
name: specification-writing
description: "Help a stakeholder write a specification or statement of work that suppliers can price and we can hold them to - outcomes and acceptance criteria before features, measurable requirements, what is mandatory and what is desirable, deliverables, timescales and the information suppliers must return - drawing on what we bought before and the policy on specifications. Use when asked to write, review or tighten a spec, SOW or requirements for a purchase. Not for designing the tender around it (rfx-design) or for choosing suppliers (supplier-selection)."
---

# Specification Writing

## Purpose

Turn a need into a specification that is complete enough to price, neutral enough to compete and precise enough to accept against. The skill drafts and critiques; the stakeholder owns the content and the technical truth.

## When to use and when not

Use when a requirement needs writing up for suppliers, when a draft spec or statement of work needs a review for gaps or bias, or when acceptance criteria are missing.

Not for the sourcing event itself (rfx-design), for the evaluation model (rfx-design), for choosing who to invite (supplier-selection), for technical correctness of an engineering drawing (the engineer) or for contract wording (contract-clause-review).

## Inputs and preconditions

- The need in the stakeholder's words: what, why, for whom, by when, how much.
- Any existing draft, standard or drawing, labelled user-supplied.
- Constraints: site, safety, regulatory, compatibility with what we have.

- Use `tool:similar_items` for what we bought before that resembles the need, the descriptions used and the suppliers
- Use `tool:search_contracts` for an agreement whose schedule already specifies the item or service
- Use `tool:extract_clauses` for the specification or service description in that agreement, quoted
- Use `tool:get_policy` for the organisation's specification rules (standards, sustainability, accessibility, security)
- Use `tool:run_calculation` with `requirement.completeness_score` to score the draft's completeness against the required elements
NOT AVAILABLE: build_docx. the specification is given as structured text

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.min_completeness_pct`.

## Workflow

1. Frame the need: outcome, user, volume, timescale, budget band if given.
2. Look at what we bought before and any agreement schedule for the same thing; reuse language that worked and note the suppliers it came from.
3. Read the policy for mandatory content (standards, sustainability, security, accessibility) and quote it.
4. Draft in the order: purpose and outcome; scope and exclusions; mandatory requirements (each measurable and testable); desirable requirements with how they score; deliverables and milestones; acceptance criteria and tests; service levels where relevant (hand to sla-design for a regime); information suppliers must return (price basis, lead time, compliance evidence); constraints and assumptions; open questions.
5. Critique the draft: brand names or single-supplier features (bias), requirements without a measure, acceptance without a test, deliverables without a date, missing policy content. Score completeness with the registry calculation and name the gaps; a draft under `config.thresholds.min_completeness_pct` is not ready to issue.
6. Output the draft and the critique with the questions the stakeholder must answer.

## Frameworks and heuristics

- Specify the outcome and the test, then the features; a feature list invites the cheapest box that ticks it.
- Every mandatory requirement must be measurable and must be truly mandatory; everything else is desirable and scored.
- Neutral language: 'or equivalent' after any named standard or product, with the equivalence test stated.
- Acceptance criteria are the contract's teeth; write them as tests with a pass mark.
- Ask suppliers for the information you will evaluate on, nothing more.

## Policy hooks (SC-007)

Mandatory specification content comes from `tool:get_policy` and is quoted. Spend is not committed by a specification; where the specification leads to an event above a policy band, the band is noted and the event is handed to rfx-design. Nothing is issued to suppliers by the skill.

## Output specification

1. Frame: outcome, user, volume, timescale.
2. Prior purchases and agreement schedules found, with the language reused.
3. The draft specification in the stated order.
4. Critique: bias, unmeasurable requirements, missing tests, missing policy content; completeness score and gaps.
5. Questions for the stakeholder; next step (rfx-design or a requisition).

## Quality bar

- Every mandatory requirement has a measure and a test.
- Named products and standards carry 'or equivalent' with the test.
- Policy content is quoted, not assumed.
- The technical truth is the stakeholder's; the skill does not invent specifications for things it has no record of.

## Escalation and approval

Escalate to the engineer or service owner for technical content, to the policy owner where mandatory content is unclear, and to legal where the specification would become a contract schedule.

## Worked example

Asked to write a spec for a fleet of label printers, the skill finds the printers bought two years ago with their description and supplier, reads the policy (energy label and security requirements mandatory), drafts outcome (labels per hour at a stated quality), mandatory requirements with measures, desirables with scoring, acceptance tests (sample run at the site), information to return (price per unit, consumables price, lead time, energy rating), and critiques the stakeholder's draft for naming one brand without 'or equivalent' and for an acceptance clause with no test.

## Anti-patterns

- Copying a supplier's brochure as the specification.
- Mandatory requirements that are preferences.
- Acceptance criteria that say 'to our satisfaction'.
- Inventing technical values the stakeholder has not given.
