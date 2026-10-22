---
name: award-recommendation
description: Combine evaluation scores, total cost of ownership, risk checks and savings into an award memo with a clear recommendation, and say what approval it needs. Use when bids have been evaluated and someone asks who should win, whether to split an award, or to prepare the award decision. Advisory and approval-gated, so it never tells a supplier anything and never marks the award approved. Not for designing the event, negotiating, or signing the contract.
---

# Award Recommendation

## Purpose

Give the decision-maker one memo: who the evidence favours, why, what it costs, what could go wrong, and who must approve. The ranking is arithmetic from the published method; the award itself is a person's decision after approval.

## When to use and when not

Use after bids are evaluated and a decision is needed.

Not for designing the sourcing event, negotiating, or signing the contract. This skill never contacts a supplier, never sends a regret letter, never records an award and never says the award is approved.

## Inputs and preconditions

Ask for these; stop if the integrity items are missing.

- The published evaluation criteria, weights, scale and normalisation method, exactly as issued to bidders.
- Every evaluator's scores, and the panel's consensus scores if they exist.
- The bids themselves, or prices already on the same volumes, currency, delivery terms and payment terms. Never compare raw quotes: if they are not already like-for-like, normalise them with `bids.normalise` (step 2).
- Evaluation horizon, cost of capital and any ownership costs, from finance. Never supply a rate yourself.
- The current result and date of each risk check (sanctions, financial health, ESG, forced labour), supplied by compliance.
- The procurement regime, if public, and the baseline price for savings.

Use `tool:savings_tracker` only to show what the ledger already counts as saved; the saving on this award is worked out against the baseline and is not logged by this tool.
Use `tool:screen_sanctions` for the sanctions result compliance recorded, with provider and date. It performs no check: if it says not performed, or the record is expired, record "not performed" and do not recommend an award until compliance performs and records the check.
Use `tool:financial_health_score` for the recorded financial health assessment (provider, rating, score, date); if none is recorded, record "not performed".
Use `tool:esg_rating_lookup` for the recorded ESG rating (provider, rating, date); if none is recorded, record "not performed".
Use `tool:forced_labour_screen` where it applies, for the recorded result and date; if none is recorded, record "not performed".
NOT AVAILABLE: route_approval. Say which approval is needed; a person submits it.
NOT AVAILABLE: build_docx. Give the memo as text using templates/memo.md.

## Workflow

1. Screening gate, before anything else. For the proposed winner call `tool:screening_summary` first (one call covers sanctions, financial health, ESG, forced labour, conflict minerals and risk events): a result that is not performed, stale or flagged blocks the award, and when it does, say so and stop; no spend, PO or contract lookups are needed to reach that answer. Then the integrity check. Confirm that every bidder was evaluated against the published criteria and weights, no bid arrived after the deadline, evaluator conflict-of-interest declarations are on file, and clarifications did not change a bid's substance. If any answer is no or unknown, stop and say what must be fixed first.
2. Like-for-like prices. If the bids are not already on the same terms, run `bids.normalise` through `tool:run_calculation` with the line master, each bid, the base currency, the exchange rates and their date, the target Incoterm and payment days, and the cost of capital from finance. Every rate and every freight or insurance figure must come from the user or finance; never supply one. Report what it lists as not normalised, its exceptions, any missing lines and the policy applied, and use its normalised prices as the price values in the next step. If it comes back unassessed, say why and ask for the missing input.
3. Scores. Use `tool:run_calculation` with `scoring.weighted_total`, passing the published criteria, scale, evaluator scores, consensus scores, the price or metric values, and the published normalisation. If it comes back unassessed, say the scores cannot be combined and why (for example weights that do not sum to 1); do not total them yourself. Report its evaluator and criterion flags, and say that agreement statistics (Krippendorff alpha, ICC) were not computed.
4. Cost. For each option run `landed_cost.total_cost` and then `tco.total_cost` through `tool:run_calculation`, with the same horizon, cost of capital and volumes for every option. Use only figures the user or finance supplied. Anything a quote does not state is reported as not stated, not as zero.
5. Risk. For each risk check, record the result and its date. A result older than `config.thresholds.screening_max_age_days` days is stale. Any check not performed, stale, or with an unresolved potential match means the award is blocked pending compliance. Say so at the top of the memo.
6. Options. Compare a single award to rank 1, a split award where the user allows one (for example for a critical item with supply risk and volume above each supplier's minimum), and a cherry-pick by lot only if the published rules allowed it. For each option give score, total cost, risk status and the saving against the baseline as a percentage (`deal.pct_change`).
7. Decide. Recommend the option that ranks first under the published method. If the lowest-TCO option is not the top-scoring one, explain the difference in one paragraph. Run `scoring.weighted_total` again with `monte_carlo_n` and a seed the user gives, or read its one-at-a-time results: if a weight change of `config.thresholds.close_decision_shift_pp` percentage points flips rank 1, or removing one evaluator would, mark it a close decision and show the alternative.
8. Conditions. List conditions precedent (approvals, certificates, onboarding), points still to negotiate, and the transition plan for an incumbent. Standstill periods depend on the regime: ask the user for the rule; if not given, say it is not determined.
9. Approval. Call `tool:check_policy` with the total award value over the term and its currency. Report its reply as it is, including the approving role and anything it lists as not assessed. Ask the user for the requester and, if known, the approver.
10. Write the memo using templates/memo.md. The memo is a draft for approval. Notification of suppliers happens only after an authorised person approves, and never from this skill.

## Frameworks and heuristics

- The published method decides the ranking. Do not reweight after bids are in, and refuse a request to.
- Total cost over price, on equal terms.
- Split only when single-source risk is high and each supplier's share stays above its minimum order quantity.
- Separate fact (tool result), estimate (labelled) and judgement (yours) in the memo.

## Policy hooks (SC-007)

The approval limit, the approving role and any unassessed rules come from `tool:check_policy`, not from this skill. Do not state a limit, role or approver the tool did not return.

## Output specification

The memo in templates/memo.md. Lead with the recommendation, or with "blocked" and the reason. Every number is from a tool result or the user, and says which. Risk checks not performed are shown as not performed.

## Quality bar

- The published method was used unchanged.
- Every figure in the memo traces to a tool result or a named user input.
- Risk checks are current, or the award is shown as blocked.
- Close decisions are flagged, and a recommendation that is not the lowest cost is explained.

## Escalation and approval

Approval-gated. The memo is a draft; nothing goes to a supplier and nothing is recorded as awarded until an authorised person approves through the approval flow. Escalate: an integrity failure, an award to a bidder that is not top-ranked, a value over budget, an unresolved risk flag, or a sole bidder. Pressure to award to a lower-ranked bidder "because we know them", or to tell the winner before approval, is refused; restate the rule and the approval route.

## Worked example

Three packaging bidders. `scoring.weighted_total` ranks B first (0.84), A second (0.81), C excluded for failing the security gate. Total cost over three years at the finance-supplied 8 percent: A is lowest, B is 2.1 percent higher, so the memo explains that B wins on quality and delivery weight. Raising the price weight by 5 percentage points flips rank 1 to A, so it is a close decision and A is shown as the alternative. Sanctions screening for B was supplied by compliance dated 12 days ago and is clear; financial health was not supplied, so it is shown as not performed and the award is blocked until compliance provides it. `check_policy` on 410,000 GBP reports an Approver is required.

## Anti-patterns

- Recommending an award while a risk check is missing, stale or unresolved.
- Re-weighting criteria after bids are opened.
- Comparing quotes that are not on the same terms.
- Telling a supplier, or drafting a message to one, before approval.
- Stating an approver or limit that `tool:check_policy` did not return.
- Treating a cost a quote does not mention as zero.
