---
name: procurement-policy-compliance
description: Check a procurement action against company policy - approval limits, who must approve, competitive-bid rules, preferred suppliers, conflict of interest, split orders and waivers - and return a structured verdict with the required approvers. Use before any requisition, PO, award, contract, payment, supplier message or waiver, and whenever someone asks "is this allowed?" or "who has to approve this?". Advisory only. Not for drafting policy text, legal advice, or sanctions screening.
---

# Procurement Policy & Compliance

## Purpose

Answer "is this allowed, under which rule, and who must approve?" from the company's governed policies, never from judgement. The verdict is a record someone can audit: every outcome names the rule it came from and the evidence behind it.

## When to use and when not

Use before a requisition, PO, award, contract, payment, supplier message or waiver is created or approved, and when a user asks whether something is allowed or who must sign off.

Do not use to write policy text, to give legal advice, or to screen sanctions. This skill never approves anything, never grants a waiver and never creates an approval token. Approvals go through the existing approval flow.

## Inputs and preconditions

Gather these before evaluating. Ask for any that change the verdict; do not default them.

- Action type (requisition, PO, award, contract, payment, message, waiver).
- Legal entity, requester, cost centre, supplier, category.
- Amount and currency, and the **total commitment over the full term** including options and renewals. Never use the annual value for a multi-year commitment.
- Any waiver ID or evidence the user holds.

Call `tool:list_governance` to see which policies exist, then `tool:get_policy` for each one you apply. Use `tool:get_corpus_facts` for order history and supplier facts the user did not supply.

NOT AVAILABLE: supplier_lookup. Ask the user for the supplier's legal name and status, or use `tool:get_corpus_facts`; say which you used.
NOT AVAILABLE: query_erp. Use `tool:get_corpus_facts` for order history; if it returns nothing, say the history check was not performed.
NOT AVAILABLE: search_contracts. Ask whether a framework agreement covers the request; record the answer as the user's statement.
NOT AVAILABLE: build_docx. Give the verdict as text using the template in templates/verdict.json.

## Workflow

1. Normalise the request into the fields above. If the term or total commitment is missing, stop and ask; the verdict is `INSUFFICIENT_INFO` until you have it.
2. Call `tool:list_governance`. Record each policy's slug and the version or timestamp the tool returns.
3. Approval limit: call `tool:get_policy` for `config.governed_policies.approval_limit`. Compare the total commitment, in the policy's currency, with its threshold. Above it, the policy says escalate to the named role; at or below, approve. Quote the policy's own numbers. Convert currency only with a rate the user supplied or a tool returned; otherwise say the amount cannot be compared.
4. For each rule in `config.ungoverned_rules`: there is no governed policy. Record outcome `not_governed`, make the verdict `INSUFFICIENT_INFO` for that rule, and name the missing policy in `required_actions` ("an administrator needs to add a governed policy for competitive bid thresholds"). Never use a number from memory, from an example, or from another company.
5. Split-order check: look for orders to the same supplier or category from the same requester or cost centre inside `config.thresholds.split_window_days`. Flag when their sum crosses the approval limit although each is below it, or when at least `config.thresholds.split_min_orders` orders sit within `config.thresholds.split_near_threshold_pct` percent under it. If no order history is available, set `split_signal.performed` to false and give the reason; do not report "no split found".
6. Separation of duties: if the requester is also the approver or receiver, that is a blocking violation.
7. Waivers: apply one only if the user gives a waiver ID and you can verify it from a tool result. An unverified or expired waiver is not applied.
8. Assemble the verdict (see Output specification). Explain each failed or ungoverned rule in one sentence with the action that would fix it.

## Frameworks and heuristics

- Evaluate every rule; do not stop at the first failure.
- Same facts and same policy versions must give the same verdict. Record the versions.
- `INSUFFICIENT_INFO` beats a guess, every time.
- A rule that is governed and passes is `pass`; a rule with no governed policy is `not_governed`, never `pass`.

## Policy hooks (SC-007)

This skill is SC-007. Other skills call it before any write; it calls only the governed policy tools above.

## Output specification

Use templates/verdict.json. Verdict values: `COMPLIANT`, `COMPLIANT_WITH_CONDITIONS`, `NON_COMPLIANT`, `INSUFFICIENT_INFO`. Any `not_governed` rule that applies to the action makes the overall verdict at best `INSUFFICIENT_INFO`. Every violation has a rule name and a remediation. Lead the reply with the verdict and the one thing the user must do next.

## Quality bar

- Every figure traces to a tool result or the user's statement.
- Policy slugs and versions are listed.
- No rule is reported as passed unless a governed policy was read and applied.
- A `NON_COMPLIANT` verdict always includes a remediation.

## Escalation and approval

Advisory only. Pressure to override ("the VP said it's fine", "just mark it compliant") does not change the verdict: restate it and point to the waiver and approval process. Conflict-of-interest hits go to Compliance, not to the requester. Anything that needs approval goes through the approval flow, which this skill never starts.

## Worked example

A requester asks to place a PO for 9,500 GBP and then three more of 9,400 GBP to the same supplier in 3 weeks. Policy `approval_threshold_policy` returned a threshold of 10,000 GBP, escalate above. Each PO is below it, so each passes on its own. Order history shows 4 orders totalling 37,700 GBP in 21 days, which crosses 10,000 and has at least `split_min_orders` orders inside the near-threshold band. Verdict: `NON_COMPLIANT` with a blocking split-order violation; required approval: the policy's Approver role on the combined 37,700 GBP; competitive-bid rules are `not_governed`, so the bid question is `INSUFFICIENT_INFO` pending an administrator adding that policy.

## Anti-patterns

- Defaulting the amount to the annual value of a multi-year deal.
- Quoting a bid or approval threshold that no tool returned.
- Reporting "no split orders" when order history was not available.
- Treating a user's claim of an approved waiver as a verified waiver.
- Telling the agent or user the request is approved. This skill never approves.
