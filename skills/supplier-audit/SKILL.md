---
name: supplier-audit
description: "Plan, run on paper and write up a supplier audit - the scope and standard from the contract and the risk, the checklist by area, the evidence to collect, the findings graded, the actions with owners and dates, and the report - from the agreements, the screening and performance on record, and the audit notes the user brings back. Use when asked to plan an audit, build the checklist, grade findings or write the audit report. Dormant pending review. Not for the risk screening (supplier-risk-assessment) or the actions' tracking (capa-management)."
---

# Supplier Audit

## Purpose

Make an audit a structured look at evidence: the right scope for the risk, a checklist tied to the standard, findings graded by a rule, and actions that someone owns; the report is what the records show, not what the visit felt like.

## When to use and when not

Use when a supplier is to be audited (quality, process, ethical, security), a checklist is needed, or audit notes are to become findings and a report.

Not for desk-based risk screening (supplier-risk-assessment), for tracking the actions afterwards (capa-management), for the supplier's 8D (eight-d-problem-solving) or for the sustainability review (supplier-sustainability-review).

## Inputs and preconditions

- The supplier, the reason for the audit and the standard or clause it is against.
- The audit notes and evidence collected, from the user, after the visit.

- Use `tool:supplier_lookup` for the supplier's record, certifications on file and sites
- Use `tool:screening_summary` for the screening on record (the audit does not replace it)
- Use `tool:calc_supplier_scorecard` for the delivery and findings pattern that may set the scope
- Use `tool:search_contracts` for the agreement's audit-rights, quality and compliance clauses
- Use `tool:extract_clauses` for the audit, quality, subcontracting and compliance clauses, quoted
- Use `tool:supplier_items` for what we buy from them, to scope the areas
- Use `tool:run_calculation` with `deal.share_pct` for the share of checklist items met and `quality.ppm` for a defect rate the user's inspection data gives
- Use `tool:get_policy` for the audit rules (frequency by risk, grading scale, who signs the report)
- Use `tool:build_docx` for the checklist and the audit report as documents
NOT AVAILABLE: log_inspection. no inspection register; the audit's findings are recorded in the report and the user's system

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.pass_share_pct`.

## Workflow

1. Scope: the reason (risk, pattern, new supplier, contract right), the standard or clauses it is against (quoted), the sites and processes, from the records and the user; the audit right in the contract quoted.
2. Checklist by area (management, process control, quality control, traceability, subcontracting, ethics, security as the scope needs) with the evidence to collect per item; built as a document.
3. Findings from the user's notes: each against a checklist item, with the evidence seen, graded by the policy's scale (critical, major, minor, observation); a finding with no evidence is an observation.
4. Summary: share of items met via the registry; the critical and major findings first; the overall grade by the policy's rule.
5. Actions: per finding, the action, owner (supplier or us), date, and the verification; a critical finding needs an action before the next delivery; hand to capa-management for tracking.
6. Report built as a document for the signatory the policy names; the supplier gets the findings and actions; the next audit date by the policy's frequency.

## Frameworks and heuristics

- Scope follows risk; a checklist that fits every supplier fits none.
- A finding is evidence against a requirement; a feeling is an observation.
- Grade by the rule, not by the mood of the visit.
- The report is for the next auditor as much as for this one.

## Policy hooks (SC-007)

Audit frequency, grading scale and the report's signatory come from `tool:get_policy` and are quoted; the audit right comes from the quoted clause. The skill plans and writes; the auditor audits and the signatory signs.

## Output specification

1. Scope with the standard and clauses quoted.
2. Checklist by area with evidence to collect (document export id).
3. Findings graded with evidence; share met via the registry; overall grade.
4. Actions with owners, dates, verification; hand-off.
5. Report export id; next audit date.

## Quality bar

- Every finding cites a checklist item and evidence.
- Grades follow the quoted scale.
- Shares and rates come from the registry.
- Nothing is signed or sent by the skill.

## Escalation and approval

Escalate critical findings to the category lead and quality before the next delivery, ethical or legal findings to compliance and legal, and the actions to capa-management.

## Worked example

Asked to audit the seal supplier after repeated defects, the skill scopes the audit to process and quality control at their main site under the agreement's audit clause, builds the checklist with the evidence to collect, and after the visit grades the user's notes (one major finding on incoming inspection with the evidence, three minors, two observations), computes the share met via the registry, sets the actions with owners and dates, writes the report for the quality owner to sign, and hands the actions to capa-management.

## Anti-patterns

- A generic checklist.
- A finding without evidence.
- Grading by impression.
- Signing or sending the report from the skill.
