---
name: eight-d-problem-solving
description: "Run the eight disciplines with a supplier on a problem they own - the team, the problem description, the containment, the root cause, the corrective action, its implementation, prevention and the close-out - as a structured report with each discipline's evidence and dates, from our findings and records and the supplier's submissions the user supplies. Use when asked for an 8D, to request one from a supplier, or to review one they sent. Dormant pending review. Not for our own internal root cause (root-cause-analysis) or our corrective actions (capa-management)."
---

# 8D Problem Solving

## Purpose

Hold a supplier to a disciplined response: a problem they can see in our evidence, containment first, a cause they proved, an action they implemented, and prevention we can check; the report is the record.

## When to use and when not

Use when a supplier-caused problem needs a formal response, when requesting an 8D, or when reviewing one submitted.

Not for our internal problems (root-cause-analysis, capa-management), for the claim (dispute-and-claims), for the performance plan over time (supplier-development-plan) or for sending the request (a person sends).

## Inputs and preconditions

- The problem with our evidence: findings, receipts, inspection results, dates.
- The supplier's 8D submission when reviewing, labelled by date.

- Use `tool:invoice_reconciliation` for the findings against the supplier's documents as evidence of the problem
- Use `tool:calc_supplier_scorecard` for the delivery and findings pattern
- Use `tool:supplier_items` for the items and deliveries involved
- Use `tool:search_contracts` for the agreement's quality and corrective-action clauses
- Use `tool:extract_clauses` for the quality, inspection, corrective-action and remedy clauses, quoted
- Use `tool:run_calculation` with `quality.ppm` for the defect rate from the user's counts and `deal.share_pct` for the affected share of deliveries
- Use `tool:build_docx` for the 8D request or the review report as a document
- Use `tool:get_policy` for the supplier-quality rules (when an 8D is required, response times)
NOT AVAILABLE: log_non_conformance. no register; the problem is evidenced from findings and the user's records
NOT AVAILABLE: send_message. the request is drafted; a person sends it

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.response_days`.

## Workflow

1. D0 and D1: the problem's evidence from our records and the user's inspection data (defect rate via the registry); the team on both sides as the user names them.
2. D2: the problem description in facts (what, where, when, how many, how found), with the affected share via the registry; the contract's quality clause quoted.
3. D3: containment: what protects us now (inspection, sorting, hold, alternative supply) with the date and who does it; a containment that depends on the supplier is requested, not assumed.
4. D4: root cause: the supplier's cause with their evidence (reviewing) or the request for it with the evidence we hold (requesting); tested against our timeline; a cause without evidence is sent back.
5. D5 to D7: corrective action, its implementation date and verification method, and the prevention across similar parts and processes; each with a date and an owner; a response later than `config.thresholds.response_days` days after the request is overdue.
6. D8: close-out only when the verification is evidenced (defect rate via the registry after the action); the report built as a document; hand to supplier-development-plan if the pattern persists.

## Frameworks and heuristics

- Containment before cause; cause before action; verification before closure.
- Our evidence defines the problem; the supplier's evidence defines the cause.
- A corrective action without a verification method is a promise.
- Prevention is the discipline most often skipped and the one that stops the next 8D.

## Policy hooks (SC-007)

Supplier-quality rules and response times come from `tool:get_policy` and are quoted; remedies come from the quoted clauses. The skill drafts and reviews; a person sends the request and quality closes the 8D.

## Output specification

1. The 8D report by discipline with evidence, dates and owners (document export id when built).
2. For a request: the problem evidence, the containment asked for, the response deadline.
3. For a review: each discipline accepted, sent back or open, with reasons.
4. Defect rate and affected share via the registry; hand-offs.

## Quality bar

- Every discipline has evidence, a date and an owner or is marked open.
- Rates and shares come from the registry.
- Containment is in place before the cause is debated.
- Nothing is sent or closed by the skill.

## Escalation and approval

Escalate to quality for close-out, to supplier-development-plan for a persisting pattern, to dispute-and-claims for the cost of the problem, and to supply-continuity-planning where containment needs alternative supply.

## Worked example

Asked to request an 8D from the seal supplier, the skill evidences the problem from the findings and the user's inspection counts (defect rate via the registry on three deliveries), quotes the agreement's corrective-action clause and the policy's response time, sets our containment (hundred-percent inspection on arrival, a hold on the open order) with owners and dates, drafts the request asking for containment on their side, the evidenced cause, the action with its verification and the prevention across the part family, builds it as a document and hands it to a person to send.

## Anti-patterns

- Accepting a cause without evidence.
- Closing an 8D on a promise.
- Skipping containment while arguing the cause.
- Sending the request from the skill.
