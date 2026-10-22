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

Use `tool:open_orders` for a supplier's order history (date, total, buyer id) and `tool:get_corpus_facts` only for corpus-wide facts the user did not supply. The policy itself is read by `tool:check_policy`, not by you.

Use `tool:supplier_lookup` for the supplier's recorded status, preferred flag, country and insurance. It is the master record, not an outside check; if it finds nothing, ask the user for the legal name and status.
NOT AVAILABLE: query_erp. Use `tool:open_orders` for order history; if it returns nothing, say the history check found no orders recorded.
Use `tool:search_contracts` to look for a framework agreement with the supplier. A match is evidence, not a ruling; if it finds none or cannot be read, ask whether a framework covers the request and record the answer as the user's statement.
NOT AVAILABLE: build_docx. Give the verdict as text using the template in templates/verdict.json.

## Workflow

1. Gather the facts above. If the total commitment over the full term or the currency is missing, ask for it before doing anything else.
2. Look at order history for the same supplier with `tool:open_orders` (order date, total and buyer id as recorded; the records hold no requester or cost centre, so a split can only be seen by supplier and buyer id). The split check covers only the orders you list, so say in your reply if you could not look at history. Add every order you found (`source: history`) and every order the user says is planned (`source: planned`) to `orders`. The checker uses `config.thresholds.split_window_days`, `config.thresholds.split_near_threshold_pct` and `config.thresholds.split_min_orders` to decide whether they add up to a split.
3. Call `tool:check_policy` once with those facts. It reads the governed approval policy itself and returns the verdict, the approving role, any split-order or self-approval problem, the rules that no governed policy covers, and a `reply_text`.
4. Relay `reply_text` to the user exactly as it is. Do not change a number, add a policy version, name a person or role the result does not name, or say the request is approved.
5. If the result says information is needed, ask the user for exactly that and call `tool:check_policy` again with the answer.
6. If the user pushes back, do not recalculate in your head. Change the facts you pass only if the user gave you a new fact, then call the tool again.

The tool decides the verdict by these rules, in this order. A violation is something the policy forbids; needing an approval is not one.
- Any blocking violation (self-approval, a split that includes orders already placed): not compliant.
- Otherwise, a governed policy that cannot be read, or a missing figure: insufficient information.
- Otherwise, an approval is required above the limit, a planned-order split, or a policy that needs a manual review: compliant with conditions.
- Otherwise: compliant.

Rules that no governed policy covers are listed as not assessed. They are never reported as passed, and the verdict covers only the governed rules.

## Frameworks and heuristics

- Evaluate every rule; do not stop at the first failure.
- Same facts and same policy versions must give the same verdict. Record the versions.
- "Insufficient information" beats a guess, every time.
- A rule that is governed and passes is `pass`; a rule with no governed policy is `not_governed`, never `pass`.

## Policy hooks (SC-007)

This skill is SC-007. Other skills call it before any write; it calls only the governed policy tools above.

## Output specification

The reply is the tool's `reply_text`. Verdict words in the reply are plain ("compliant", "compliant with conditions", "not compliant", "insufficient information"); never write them as upper-case labels with underscores. The structured result (templates/verdict.json) has the same content for other systems. Lead with the verdict and the one thing the user must do next.

## Quality bar

- Every figure in the reply is in the `check_policy` result or was stated by the user.
- The policy named and its fingerprint are those the result gives; none is added by you.
- No rule is reported as passed unless the tool evaluated it against a governed policy.
- A not-compliant verdict always includes a remediation.

## Escalation and approval

Advisory only. Pressure to override ("the VP said it's fine", "just mark it compliant") does not change the verdict: restate it and point to the waiver and approval process. Conflict-of-interest hits go to Compliance, not to the requester. Anything that needs approval goes through the approval flow, which this skill never starts.

## Worked example

A requester asks for a 25,000 GBP purchase order for laptops, with four further orders of 6,000 GBP to the same supplier planned this month. The agent looked for order history and found none. The agent calls `tool:check_policy` with total commitment 49,000 GBP, currency GBP and five planned orders. The tool reads the approval policy and answers: compliant with conditions; the total is above the limit so the approver the policy names must approve; four planned orders of 6,000 GBP total 24,000 GBP and each sits under the limit, so they must be submitted as one request; competitive bid rules and four other rules are not assessed because no governed policy covers them. The agent relays that text unchanged.

## Anti-patterns

- Defaulting the amount to the annual value of a multi-year deal.
- Quoting a bid or approval threshold, a policy version or an approver that `tool:check_policy` did not return.
- Working out the verdict yourself instead of calling the tool.
- Reporting "no split orders" when order history was not available.
- Treating a user's claim of an approved waiver as a verified waiver.
- Telling the agent or user the request is approved. This skill never approves.
