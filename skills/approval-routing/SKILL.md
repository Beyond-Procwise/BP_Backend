---
name: approval-routing
description: "Carry out the submission of a request for approval - work out from policy who must approve at the value and the action type, assemble the approval request with the facts and the attachments the approver needs, submit it through the platform's approval action to the right role, and confirm where it sits and what the approver was asked. Use when asked to send something for approval, route it, or get it signed off. Dormant until a routing tool exists; until then the skill prepares the request and names the approver for a person to submit. Not for whether approval is needed (procurement-policy-compliance), which this skill reads first."
---

# Approval Routing

## Purpose

Put a request in front of the right approver once, with everything they need to decide and nothing they do not: the facts, the policy basis, the attachments, and a plain question. The skill never approves and never names an approver the policy did not.

## When to use and when not

Use when the request is to submit, route, send for approval or get sign-off on a purchase, award, contract, exception or payment that is ready to decide.

Not for the ruling on whether approval is needed (procurement-policy-compliance), for the exception's justification (policy-exception-request), for approving anything, or for chasing an approver (supplier-communication-drafting is for suppliers; a person chases colleagues).

## Inputs and preconditions

- The request: what is to be approved, its value, the action type, and the facts that support it.
- The attachments the approver needs (the recommendation, the quote comparison, the case), by export id or document.

- Use `tool:check_policy` for the approver role the value and the action type require, as replied; the approver is never chosen by the skill
- Use `tool:get_policy` for the approval rules (what the request must contain, time limits, escalation when the approver is absent)
- Use `tool:build_docx` to assemble the approval note (facts, basis, recommendation, attachments list) as a file for the approver
- Use `tool:search_contracts` for the agreement an approval of a contract or amendment refers to
- Use `tool:savings_tracker` for value claims the request cites, as the ledger states them
NOT AVAILABLE: route_approval. the approval action does not exist yet; the skill prepares the request and names the role the policy requires, and a person submits it from the approvals screen
NOT AVAILABLE: approve_exception. the skill never approves; a decision is the approver's act

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.decision_sla_days`.

## Workflow

1. Read the approval route with `tool:check_policy` on the value and action type; the reply names the role (or says none is needed, or asks for facts). The role is the route; the skill adds no approver.
2. Quote the approval rules: required content, time limit of `config.thresholds.decision_sla_days` days where policy states none, escalation when the approver is absent.
3. Assemble the request: the facts (what, who, how much, when), the policy basis, the recommendation and its reasons, the attachments by reference, the question in one sentence; build it with `tool:build_docx` where the approver wants a note.
4. Check completeness against the rules; a request missing a required item is not submitted.
5. Submit once through the platform's approval action to the role named, under the agent-policy gate, with the user's name; the action returns the request reference and status.
6. Confirm where the request sits, who was asked, by when a decision is expected, and what the user does if it is late.

## Frameworks and heuristics

- The policy names the role; the person is whoever holds it.
- One question per request; an approver asked three things answers none.
- Attach the evidence, do not restate it.
- A request submitted twice is two requests.

## Policy hooks (SC-007)

The approver role comes from `tool:check_policy` as replied and is never chosen by the skill; the request content rules come from `tool:get_policy`. The approval action runs under the agent-policy gate with the user's name. The skill never approves, never marks approved and never reports a submission the action did not confirm.

## Output specification

1. The route as check_policy replied, with the facts it rested on.
2. The assembled request: facts, basis, recommendation, attachments, question; the note's export id where built.
3. Completeness check against the rules.
4. The submission record: reference, role asked, expected decision date (or the ready-to-submit state while the action does not exist).
5. What the user does next.

## Quality bar

- The approver role is the policy's reply, nothing else.
- The request contains every item the rules require.
- Nothing is approved or marked approved.
- A submission is reported only with the action's reference.

## Escalation and approval

Escalate to the policy owner where the route is unclear or the approver is absent beyond the escalation rule, and to the sponsor where the request is incomplete.

## Worked example

Asked to send the packaging award for approval, the skill reads check_policy on the award value and action, which names the role, quotes the rules on request content, assembles the note with the award recommendation's facts and the quote comparison by export id and the one question, checks completeness, and, while the approval action does not exist, reports the request as ready to submit to that role for a person to submit from the approvals screen, with the expected decision date.

## Anti-patterns

- Naming an approver the policy did not.
- Submitting an incomplete request.
- Marking something approved because the approver said so in a corridor.
- Submitting the same request twice.
