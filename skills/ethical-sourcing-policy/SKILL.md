---
name: ethical-sourcing-policy
description: "Apply the organisation's ethical sourcing policy to a supplier, a category or a case - human rights and labour standards, forced and child labour, conflict minerals, bribery, the code of conduct, the due-diligence steps and the consequences - by reading the governed policy, the screening register and the contract clauses, and saying what the policy requires, what the records show, and what is missing. Use when asked what our ethical sourcing rules are, whether a supplier meets them, or what to do about a concern. Dormant pending review. Not for the screening itself (supplier-risk-assessment) or the tender criteria (sustainable-sourcing)."
---

# Ethical Sourcing Policy

## Purpose

Make the policy the reference and the records the evidence: what we require, what is on file, what is not, and who acts; the skill never softens a requirement or clears a concern.

## When to use and when not

Use when the question is the ethical sourcing policy's content, a supplier's position against it, a concern raised, or the steps a case requires.

Not for performing or reading a screening alone (supplier-risk-assessment), for sourcing criteria (sustainable-sourcing), for the ESG report (csrd-esg-reporting) or for an audit (supplier-audit).

## Inputs and preconditions

- The supplier, category or concern, as the user states it.
- Any evidence the user holds (reports, allegations, documents), labelled by source and date.

- Use `tool:get_policy` for the ethical sourcing policy, code of conduct and due-diligence rules, quoted
- Use `tool:screening_summary` for the supplier's forced-labour, conflict-minerals, sanctions and risk-event checks on record
- Use `tool:forced_labour_screen` when the forced-labour check alone is asked about
- Use `tool:conflict_minerals_check` when the conflict-minerals check alone is asked about
- Use `tool:supplier_lookup` for the supplier's country and certifications on file
- Use `tool:search_contracts` for the agreement
- Use `tool:extract_clauses` for the code-of-conduct, audit, termination-for-breach and subcontracting clauses, quoted
- Use `tool:classify_spend` for the category's recorded ESG impact
- Use `tool:run_calculation` with `deal.share_pct` for the share of spend or suppliers covered by a check
NOT AVAILABLE: send_esg_survey. nothing is sent; requests to suppliers go through compliance
NOT AVAILABLE: log_non_conformance. no register; a concern is recorded by compliance in their system

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.concern_response_days`.

## Workflow

1. Read the policy once and quote the requirements that apply to the case (standards, prohibited practices, required checks, due-diligence steps, consequences); where no policy governs a point, say so.
2. The records: the supplier's checks on record (performed, not performed, expired, flagged), certifications on file, country, the category's recorded impact; coverage shares via the registry where the case is a category.
3. The contract: code-of-conduct, audit, subcontracting and breach clauses quoted; what they allow us to do.
4. The gap: each requirement met, not met or unknown, with the evidence; a concern raised is recorded as the user's report with its source and date, never as a finding of the skill.
5. Steps: what the policy requires next (request evidence, audit, suspend, terminate for breach, report), with the owner (compliance, legal, category lead) and the deadline the policy sets or `config.thresholds.concern_response_days` days where it sets none.
6. The answer states requirements, evidence, gaps and steps; it clears nothing.

## Frameworks and heuristics

- The policy is the standard; the records are the evidence; the skill is neither.
- Not performed is not clear; unknown is a gap, not a pass.
- A concern is reported, dated and routed; it is never judged here.
- Consequences follow the clause and the policy, not the mood.

## Policy hooks (SC-007)

Every requirement comes from `tool:get_policy` and is quoted; every check from the register. The skill suspends nobody, terminates nothing, sends nothing and clears nothing; compliance and legal act.

## Output specification

1. Requirements quoted for the case.
2. Records: checks, certifications, country, impact; coverage shares via the registry.
3. Clauses quoted.
4. Gap table: requirement, status, evidence.
5. Steps with owners and deadlines; the concern's record if any.

## Quality bar

- Every requirement is quoted, never paraphrased from memory.
- Every status rests on a record or is unknown.
- A concern carries its source and date and no judgement.
- Nothing is cleared, suspended or sent.

## Escalation and approval

Escalate every concern to compliance, every breach question to legal, every category-wide gap to the sustainability owner, and a supplier with a flagged check to the category lead for the supply consequence.

## Worked example

Asked whether a new mining supplier meets our ethical sourcing policy, the skill quotes the policy's requirements for high-impact categories (forced-labour check, conflict-minerals check, code of conduct signed, audit right), reads the register (conflict-minerals check performed and clear, forced-labour not performed), the contract (code of conduct clause present, audit clause absent), and the category's high impact, tables the gaps, and lists the steps: compliance to perform and record the forced-labour check, legal to add the audit right, within the policy's deadline; it clears nothing.

## Anti-patterns

- Paraphrasing the policy from memory.
- Treating an unperformed check as clear.
- Judging a concern.
- Suspending or clearing a supplier from the skill.
