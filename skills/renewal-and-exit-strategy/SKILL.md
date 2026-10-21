---
name: renewal-and-exit-strategy
description: Decide, well before expiry or the notice deadline, whether to renew a contract as is, renegotiate, re-tender or exit, using performance, market and exit-cost evidence, and build a dated plan backward from the notice deadline. Use when a renewal alert is raised, a contract nears expiry or auto-renewal, or the user asks whether to renew, switch or let it roll over. Review-gated, so the contract owner and category lead approve. Not for drafting the new contract or running the tender.
---

# Renewal and Exit Strategy

## Purpose

Stop contracts renewing by default. Reach a reasoned renew, renegotiate, re-tender or exit decision early enough to act on it, and give a dated plan that finishes before the notice deadline with a safety buffer.

## When to use and when not

Use when a contract is approaching expiry, an auto-renewal, or a renewal notice deadline, or when the user asks whether to renew, renegotiate, re-source or leave a supplier.

Not for drafting the new contract (contract drafting), running the tender (RFx design), or planning the negotiation itself (use negotiation-planning, which this skill hands its asks to).

## Inputs and preconditions

Ask for what is missing; never guess a date, score or price.

- The contract: expiry date, auto-renewal terms, notice period and the clause that sets it, termination rights, current annual value and currency, and today's date.
- Performance over `config.thresholds.scorecard_window_months` months: quality, delivery, cost, responsiveness, ESG, SLA breaches and credits.
- Price against market, with the source of the comparison.
- Exit facts: switching costs, exit assistance and data-return clauses, exclusivity or minimum volume, whether staff would transfer, and known alternative suppliers.
- Stakeholder satisfaction (ask the user) and the category strategy.

NOT AVAILABLE: obligation_tracker. Ask the user for the expiry, notice period and clause; quote what they give and say it is unverified. Give the milestones as a dated list in text, since no obligations can be created.
NOT AVAILABLE: calc_supplier_scorecard. Use `tool:run_calculation` with `supplier.composite_score` or `otif.delivery_performance` only on inputs the user supplies; otherwise label the score "not available" and ask.
NOT AVAILABLE: build_spend_cube. Use `tool:get_corpus_facts` for spend and price trend, or ask the user.
NOT AVAILABLE: commodity_index_lookup. Ask the user for the market price or index figure; without it say the market position is not assessed.
NOT AVAILABLE: financial_health_score. Ask the user for a credit view, or say supplier financial health is not assessed.
NOT AVAILABLE: monitor_risk_events. Ask the user about known risk events, or say none were checked.
Use `tool:search_contracts` to find the contract and `tool:extract_clauses` to read its stored clauses, quoted word for word (try `containing` with "exit", "terminate", "data" or "exclusive"). If the contract has not been read, or a clause is not listed, ask the user to quote it and label anything unquoted "not checked".
NOT AVAILABLE: supplier_discovery. Ask the user for alternative suppliers, or state that availability is unknown.
NOT AVAILABLE: build_docx. Give the recommendation, evidence pack, timeline and notice draft as text sections.

## Workflow

1. Dates. From the contract facts, compute days to expiry and days to the notice deadline against today's date, and quote the deadline with its clause reference. If the notice deadline has already passed, say so first and look for what the contract still allows (late notice, short extension, termination for convenience). Never let an auto-renewal pass by default.
2. Lead time. Compare days to the notice deadline with the lead time each path needs: `config.thresholds.lead_time_renegotiate_days_min` to `config.thresholds.lead_time_renegotiate_days_max` days to renegotiate, `config.thresholds.lead_time_retender_days_min` to `config.thresholds.lead_time_retender_days_max` days to re-tender. If the deadline is shorter than the chosen path, state the gap and default to protecting the position: draft a non-renewal or short-extension notice for approval, pending the decision.
3. Evidence. Gather the scorecard and SLA record, spend trend, price against market, supplier risk and stakeholder view. Every score must come from a tool or the user. Where data is missing, ask for it or label it "not available"; do not substitute.
4. Exit cost and feasibility. List switching costs (tooling, qualification, data migration, dual running, termination fees), exit assistance and data-return clauses, lock-ins, any staff-transfer rules that may apply (for example UK TUPE or EU Acquired Rights Directive 2001/23/EC; ask a lawyer, do not rule), and alternative supplier availability. Use `tool:run_calculation` with `negotiation.batna_strength` for the number of alternative quotes and supplier history, and `negotiation.kraljic_quadrant` for the quadrant, with the signals the user or the corpus supplies.
5. Decision matrix. Score band: at or above `config.thresholds.score_strong_min` is strong, `config.thresholds.score_adequate_min` up to that is adequate, below it is weak. Market position: a price gap to market of `config.thresholds.market_gap_pct` percent or more is room to gain. Defaults: strong and at market, renew with targeted improvements; strong and above market, renegotiate; adequate, renegotiate with competitive tension (an RFI); weak or high risk, re-source; exit only if the switching cost is recovered within the new term. These are configurable defaults, not policy.
6. Value each path. Compare the present value of each path over the next term using `tool:run_calculation` with `tco.total_cost` (it needs a landed-cost record from `landed_cost.total_cost`, a cost of capital from finance and a horizon). Use `deal.pct_change` for percentage gaps. Label every assumption. If the cost of capital is not supplied, show undiscounted totals and say so.
7. Plan. Build the timeline backward from the notice deadline: decision, approvals, notice, negotiation or tender, transition. The plan must finish at least `config.thresholds.buffer_business_days` business days before the deadline; if it cannot, say so and escalate. Include a notice letter draft, the asks to hand to negotiation-planning, and a transition and exit plan (knowledge transfer, data, assets, continuity). Check approval levels and notice signatory authority with `procurement-policy-compliance` or `tool:check_policy`.

## Frameworks and heuristics

- The Kraljic quadrant shapes the path: a strategic or bottleneck supplier makes exit slower and riskier.
- BATNA strength: few real alternatives means renegotiating from weakness; create tension (RFI) early.
- Renegotiate from evidence (scorecard plus market), not from feeling.
- Exit readiness checklist: alternatives known, data portable, exit assistance clause in force, staff-transfer position understood, continuity plan in place.
- Protect first, decide second: a notice is cheap to withdraw, a missed deadline is not.

## Policy hooks (SC-007)

Competitive bidding rules (a renewal beyond the allowed cumulative term may need a re-tender), approval levels for renewal value, and who may sign the notice all come from governed policy. Read them with `tool:get_policy` or `tool:check_policy`. If no policy row exists, say it is not governed. The thresholds in config are this skill's defaults, not policy.

## Output specification

A recommendation in plain text, in this order: one-page summary (path, rationale, present value comparison, deadline with clause); evidence pack (each score with its source, or "not available"); exit cost and feasibility; risks; dated timeline; notice letter draft marked as a draft for approval; negotiation asks; owner list. Say what could not be computed. No document file is produced here.

## Quality bar

- The notice deadline is computed and quoted with its clause reference, or marked unverified.
- Every score comes from a tool or the user.
- Present value assumptions are labelled.
- The plan ends at least `config.thresholds.buffer_business_days` business days before the deadline, or the shortfall is stated.
- Nothing is sent, signed or booked.

## Escalation and approval

Review-gated: the named contract owner and the category lead approve; this skill never approves, sends or signs. Escalate when the deadline is less than `config.thresholds.escalate_days_to_deadline` days away or already missed, when the supplier is a sole-source critical supplier, when exit may trigger an employee transfer, or when renewal exceeds the policy maximum term. Notice drafts go to the owner for approval.

## Worked example

The user says a software contract auto-renews in 20 days, the notice period is 90 days, and the clause is 14.2. The notice deadline has therefore passed. The skill says so first, quotes clause 14.2, and does not invent a way out. It asks whether the contract allows late notice, a short extension or termination for convenience, drafts a request to the supplier for approval, and treats the renewal as a deadline breach to escalate to the contract owner. The scorecard is not available, so the evidence pack says "not available" and asks the user for delivery and quality figures before any renegotiate or re-tender call.

## Anti-patterns

- Letting an auto-renewal pass because nobody decided.
- Calling a supplier strong or weak without a sourced score.
- Quoting a notice deadline without the clause or with a guessed date.
- Choosing re-tender when the lead time is longer than the time left, without saying so.
- Ruling on employee transfer rules instead of escalating.
- Claiming a notice was sent or an obligation created.
