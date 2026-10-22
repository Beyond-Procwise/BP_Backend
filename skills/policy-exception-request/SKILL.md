---
name: policy-exception-request
description: "Build an exception or deviation request against a procurement rule - a retrospective order, a single-tender or non-competitive award, a limit or channel deviation - with the rule quoted, the facts from the records, the justification tested against the grounds the policy allows, the risks, the mitigations and the approver the policy names. Use when something has been bought or must be bought outside the rules and the question is how to request and justify the exception. Not for whether the rule applies (procurement-policy-compliance) or for sole-source awards as a strategy (category-strategy)."
---

# Policy Exception Request

## Purpose

Make exceptions honest and rare: the rule named, the breach or the need stated plainly, the justification tested against the grounds the policy allows, the risks and mitigations written down, and the right approver asked. The skill drafts the request; it grants nothing.

## When to use and when not

Use when a purchase or an award cannot or did not follow the rules and the user needs to request a waiver, deviation, dispensation or retrospective approval.

Not for the compliance ruling itself (procurement-policy-compliance), for a framework route (framework-call-off), for an urgent buy that can still follow the fast compliant route (urgent-sourcing) or for approving anything.

## Inputs and preconditions

- What was or will be bought, from whom, for how much, and which rule is in question.
- Why, in the user's words, with the evidence they hold, labelled user-supplied.
- Whether the purchase has already happened.

- Use `tool:get_policy` for the rule, the exception grounds the policy allows, the approver and the record the policy requires, quoted
- Use `tool:check_policy` for the route the value and the action take, as replied
- Use `tool:invoice_reconciliation` for the order and invoice facts where the purchase already happened
- Use `tool:supplier_lookup` for the supplier's record and status
- Use `tool:screening_summary` for the supplier's screening status
- Use `tool:similar_items` for whether an alternative supplier on record could have been used, which the justification must address
- Use `tool:search_contracts` for an agreement that covered the need, if one exists
NOT AVAILABLE: send_message. the request is drafted; the requester submits it to the approver
NOT AVAILABLE: approve_exception. no approval is given by the skill; the approver the policy names decides

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.senior_approval_gbp`.

## Workflow

1. Frame: the purchase, the rule, whether it is past or future, and the user's reasons.
2. Quote the rule and the exception grounds the policy allows (genuine urgency, sole capable supplier, compatibility, value below a band, and whatever else the policy lists); the approver and the record the policy requires. Where the policy allows no exception, say so and stop at the facts.
3. Facts from the records: the order and invoice if placed; the supplier's status and screening; alternatives on record for the items; any agreement that covered the need.
4. Test the justification against each allowed ground with the facts; name the grounds met and the ones not met; an exception above `config.thresholds.senior_approval_gbp` goes to the senior approver the policy names.
5. Risks (price not tested, supplier unscreened, pattern of repeat exceptions) and mitigations (benchmark now, screen now, plan the next purchase compliantly); the record to be kept.
6. Draft the request in the policy's form: rule, facts, ground, justification, risks, mitigations, approver; and the requester's own statement of what they will do differently next time where the purchase already happened.

## Frameworks and heuristics

- An exception is granted on a ground the policy names, not on inconvenience.
- Retrospective requests state the breach plainly; softening it is the fastest way to lose the approver.
- A third exception for the same supplier is a sourcing problem, not an exception.
- Alternatives on record must be addressed; silence about them reads as concealment.
- The mitigation is the price of the exception: benchmark, screen, plan.

## Policy hooks (SC-007)

The rule, the allowed grounds, the approver and the record requirement come from `tool:get_policy` and are quoted; the route for the value comes from `tool:check_policy` as replied. The skill grants nothing; the request goes to the approver the policy names. Where the policy allows no exception, the skill says so and does not draft around it.

## Output specification

1. Frame: purchase, rule, past or future, reasons.
2. Rule and allowed grounds quoted; approver and record requirement.
3. Facts from the records: order, invoice, supplier status, screening, alternatives, agreements.
4. Justification test: grounds met and not met with evidence.
5. Risks and mitigations; repeat-exception check.
6. The drafted request in the policy's form, with the approver.

## Quality bar

- The rule and the grounds are quoted, never paraphrased from memory.
- Alternatives on record are addressed in the justification.
- A retrospective breach is stated as a breach.
- Nothing is approved; the approver is named from policy.

## Escalation and approval

Escalate to the senior approver the policy names above the configured value, to compliance where the supplier is unscreened, and to the category lead where exceptions repeat for a supplier or category.

## Worked example

Told that a site manager ordered a repair from a local contractor without a purchase order and the invoice has now arrived, the skill quotes the rule and the retrospective-approval ground the policy allows, reads the invoice and finds no order and no supplier screening, finds two contractors on record for the same work, tests the justification (urgency claimed by the user; alternatives existed; supplier unscreened), lists the risks and mitigations (screen now, benchmark the invoice, a standing order for the site), and drafts the request stating the breach plainly for the approver the policy names.

## Anti-patterns

- Drafting a justification on a ground the policy does not allow.
- Omitting the alternatives on record.
- Calling a retrospective order a 'timing issue'.
- Approving the exception from the skill.
