---
name: urgent-sourcing
description: Source something that is needed fast without skipping the checks that cannot be skipped - who can supply it from our own records and existing contracts, which of them is already screened and set up, what the policy still requires at the value, what can be done in parallel, and the fastest compliant route with its risks - producing a same-day plan. Use when a request is urgent, needed this week, an emergency, a stock-out, a breakdown or a deadline-driven buy. Not for the routine supplier choice (supplier-selection) or for designing a tender (rfx-design).
---

# Urgent Sourcing

## Purpose

Give a buyer the fastest route that still passes: which suppliers can plausibly deliver in time from what the records show (contracts, past orders, recorded lead times, screening already done), what the policy requires at the value and who approves, which steps can run in parallel, and what an exception would need. Speed comes from using what is already on record and from parallel steps, never from skipping a check that policy requires.

## When to use and when not

Use when the user says a need is urgent, needed today or this week, an emergency, a breakdown, a stock-out, a missed delivery that must be covered, or a deadline that the normal route cannot meet.

Not for a routine supplier choice (supplier-selection), a formal tender (rfx-design), a price check (price-benchmarking) or a one-purchase policy question on its own (procurement-policy-compliance, which this skill calls for the approval route).

## Inputs and preconditions

- What is needed: item or service, quantity, specification or acceptable substitutes, the delivery point, and the deadline with the consequence of missing it.
- The expected value and currency, and whether an existing contract, framework or catalogue could cover it.
- Who can approve today, if the user knows; otherwise the policy's approver.

Use `tool:search_contracts` for a live contract or framework that already covers the item (the fastest route when it exists) and `tool:extract_clauses` with "lead time" or "delivery" for its delivery terms; `tool:supplier_discovery` for the suppliers that have supplied the item on our orders, invoices or quotes, with evidence counts; `tool:supplier_lookup` for each candidate's recorded lead time, Incoterms, currency, insurance and preferred flag; `tool:screening_summary` for which candidates are already screened (not performed is not clear, and an unscreened candidate costs the time to screen); `tool:similar_items` for substitutes bought under other descriptions; `tool:check_policy` with the value for the approval route and anything the policy still requires at that value; and `tool:calc_supplier_scorecard` for a candidate's on-time record where it is assessable.

NOT AVAILABLE: create_rfx. A quick quote request is drafted as text; a person sends it.
NOT AVAILABLE: get_freight_quote. Express delivery costs come from the supplier or forwarder; none is estimated.

Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated.

## Workflow

1. Frame. State the need, the deadline, the consequence, the value and the currency; call `tool:check_policy` with the value first so the approval route and any required steps are known before the route is chosen. The round budget is small: policy, contracts, discovery, screening summary for the shortlist, then the plan.
2. Existing route. Search the register for a live contract or framework covering the item; if one exists with a stored delivery or lead-time clause, quote it: calling off an existing agreement is the fastest compliant route and needs no new supplier.
3. Known suppliers. Run discovery on the item (and `tool:similar_items` for substitutes); shortlist at most `config.thresholds.max_candidates` by evidence, preferred flag and recorded lead time at or below the days available; read the screening summary for each: a screened supplier can be ordered from on approval; an unscreened one needs compliance first, which costs at least `config.thresholds.screening_lead_days` days by default.
4. Fit to the deadline. For each candidate: recorded lead time versus days available, delivery terms, currency and insurance status, screening status, on-time record where assessable; a candidate is "fits", "fits if expedited" (the user confirms with the supplier) or "does not fit". Nothing about lead times is assumed where the record is empty: it is a question for the supplier.
5. Parallel plan. Lay out the steps that can run at once: quick quote requests to the fitting candidates (drafted), compliance screening for an unscreened one, approval request at the value, and the fallback (substitute item, partial quantity, hire or loan) with its owner.
6. Exceptions. If no compliant route meets the deadline, say so plainly, name the exception the policy would need (who may grant it from the policy reply) and the risk of each shortcut; the skill grants nothing.
7. Output the plan with times, owners and the open questions for the supplier and the approver.

## Frameworks and heuristics

- Fastest compliant route first: an existing contract beats a new supplier; a screened supplier beats an unscreened one; a preferred supplier beats a stranger.
- Parallel, not sequential: screening, approval and quotes run at once; the plan shows the critical path.
- Lead times are the supplier's records or their word; an empty record is a question, not an estimate.
- An emergency changes the sequence, not the rules: the approval route and required checks come from policy and stand.
- The fallback is part of the plan, with an owner, from the first draft.

## Policy hooks (SC-007)

The approval route and anything required at the value come from `tool:check_policy`, reported as replied, and an exception needs the person the policy names; no check is waived here. Spend commits through the order and its approval, not through this skill.

## Output specification

1. Need, deadline, consequence, value; the policy reply (route, required steps, approver).
2. Existing contract or framework route, with the delivery clause quoted or "none on record".
3. Candidate table: supplier, evidence, preferred, recorded lead time, fit, delivery terms, currency, insurance, screening status, on-time record.
4. Parallel plan with critical path, times and owners; quick quote text; fallback with owner.
5. Exception needed (if any) and its risks; open questions.

## Quality bar

- Every supplier fact comes from the records or the user; no lead time, price or delivery promise is assumed.
- The policy reply is quoted and nothing it requires is dropped for speed.
- An unscreened candidate is never called ready; its screening time is in the plan.
- The plan names owners and the fallback.

## Escalation and approval

Escalate at once to the approver the policy names when the value needs it, to compliance for an unscreened candidate that is the only fit, and to the category lead when an exception is the only route. The order and any exception are a person's decision.

## Worked example

Told that a production line stops on Friday without a replacement drive motor, the skill reads the policy reply for the value (Approver route), finds no contract covering the item, finds three past suppliers of similar motors with evidence counts, reads their recorded lead times and screening (one preferred and screened with a lead time inside the window, one unscreened, one with no lead time on record), builds a parallel plan: quote request to the screened supplier now, approval request now, lead-time question to the third supplier, screening of the second started as fallback, and a hire unit as the contingency with the maintenance lead as owner; no check is waived and the critical path is the approval.

## Anti-patterns

- Assuming a lead time or an express cost because the need is urgent.
- Ordering from an unscreened supplier because there is no time, without naming the exception and who grants it.
- Dropping the approval route for speed.
- Sequencing steps that could run in parallel.
- Presenting a plan without a fallback or an owner.
