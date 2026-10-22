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

Use `tool:obligation_tracker` with the contract_id, document_id or title words (and the supplier to narrow) for the stored expiry date, status, auto-renewal flag, renewal term and the clauses that mention notice, termination, renewal or review, quoted word for word. It registers and schedules nothing, and the notice period is not a stored field: read the period from the quoted notice clause with the user and compute the deadline from it; if no notice clause is quoted, ask the user for it and label it unverified. If the contract is not in the register, ask the user for the expiry, notice period and clause and say they are unverified. Give the milestones as a dated list in text.
Use `tool:calc_supplier_scorecard` with the supplier and the evidence window for spend, purchase price variance, on-time delivery and open findings measured from this organisation's own records; state the record counts it gives and report any dimension it marks not assessed as "not assessed" with its reason. Quality, responsiveness, ESG and SLA credits are not recorded: ask the user. A composite needs the category owner's weights: with them, use `tool:run_calculation` with `supplier.composite_score`; without them, present the dimensions separately and label the overall score "not available".
Use `tool:build_spend_cube` to cut invoiced spend by supplier, month, currency, country, region, buyer, payment terms or status for a period; its totals equal the dashboards. Invoices carry no category, no business unit and no price-per-item, so it cannot give category totals or item prices: say so, and use `tool:similar_items` (price range and last date per source for an item description) or `tool:supplier_items` (one supplier's items with their price range) for item prices, or ask the user for an extract.
Use `tool:commodity_index_lookup` for a loaded index series and its move over the contract period as the market reference, quoting series, publisher and dates; if none is loaded, ask the user for the market price or index figure, and without it say the market position is not assessed.
Use `tool:financial_health_score` for the recorded financial health assessment; if none is recorded, say supplier financial health is not assessed.
Use `tool:monitor_risk_events` for the risk-event review compliance recorded and the risk signals in the platform's own feed, each with its date; if nothing is recorded, say risk events were not checked, not that there are none.
Use `tool:search_contracts` to find the contract and `tool:extract_clauses` to read its stored clauses, quoted word for word (try `containing` set to one word per call, at most two ("exit" then "terminate")). If the contract has not been read, or a clause is not listed, ask the user to quote it and label anything unquoted "not checked".
Use `tool:supplier_discovery` to list suppliers this organisation already deals with for the item. It is not a market scan: ask the user for other candidates and say availability beyond these is unknown.
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
