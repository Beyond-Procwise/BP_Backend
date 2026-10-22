---
name: supplier-risk-assessment
description: Assess one supplier's risk before onboarding, award, renewal or a material change, from what compliance has recorded (sanctions, financial health, ESG, forced labour, conflict minerals, risk events), the supplier master and our own trading record, and state what is cleared, flagged, stale or not performed. Use when asked whether a supplier is safe to use, to run due diligence or KYC, to screen a supplier, or what its risk status is. Not for choosing between suppliers (supplier-selection) or for recommending an award (award-recommendation).
---

# Supplier Risk Assessment

## Purpose

Give a buyer one honest page on a supplier's risk: which checks compliance has recorded and what they found, how old they are, what the platform's own records show, and which checks have not been performed. The skill performs no check itself and never clears a supplier on its own judgement.

## When to use and when not

Use when the user asks whether a supplier can be used, onboarded, awarded or renewed from a risk point of view; asks for due diligence, KYC, screening or a risk status; or when another skill needs the screening results for one supplier.

Not for comparing candidates (supplier-selection), for the award decision (award-recommendation), for a performance scorecard (supplier-performance-review), or for deciding whether a purchase is allowed under policy (procurement-policy-compliance).

## Inputs and preconditions

- The supplier: name or supplier_id. If several match, ask which; never assume from a partial name.
- What the supplier is for (category, country, spend, criticality) and the decision the assessment feeds (onboard, award, renew, continue), so the right checks are in scope.
- The organisation's own rule on which checks are required for that decision, from `tool:get_policy`; if none is stated, every check below is in scope and the output says the rule is not governed.

Use `tool:supplier_lookup` for the master record (preferred flag, risk score or "not scored", certifications on file, insurance and its expiry, country) and the number of invoices on file; confirm the supplier with the user when the match is on part of the name. Use `tool:screen_sanctions`, `tool:financial_health_score`, `tool:esg_rating_lookup`, `tool:forced_labour_screen` and `tool:conflict_minerals_check` for the results compliance recorded, each with provider and date, and `tool:monitor_risk_events` for the recorded risk-event review and the platform's own risk signals. Use `tool:calc_supplier_scorecard` for our own trading record (spend, price variance, on-time delivery, open findings) and `tool:search_contracts` for live contracts. Use `tool:get_policy` for the required checks and their maximum age; `tool:check_policy` when the decision commits spend.

NOT AVAILABLE: build_docx. Give the assessment as text sections.

## Workflow

1. Identify. Resolve the supplier and state the decision the assessment feeds. Read the master record and say what it does and does not hold (a risk score of "not scored" is not a low risk).
2. Scope the checks. From policy or, failing that, the full list: sanctions (entity, owners, directors), financial health, ESG, forced labour (where geography or material is high risk), conflict minerals (where tin, tantalum, tungsten or gold is in scope), risk events.
3. Read each check. For each tool, record: performed or not, result, provider, date, valid-until, and whether it is stale: sanctions older than `config.thresholds.sanctions_max_age_days` days, financial health older than `config.thresholds.financial_max_age_days`, ESG older than `config.thresholds.esg_max_age_days`. A check with no record is "not performed"; an expired record is "stale". Neither is clear.
4. Own record. From the scorecard: how much we trade with them, price variance, on-time delivery where assessable, open findings; from contracts: what is live and when it expires. Say what could not be assessed and why.
5. Risk events. List the recorded signals in the last `config.thresholds.risk_event_lookback_days` days with type, severity, source and date; none recorded means "none recorded", not "none".
6. Status. Clear only when every check in scope is performed, in date and not flagged, and the count of performed checks is at least `config.thresholds.min_checks_for_clearance`. Flagged if any check is flagged. Otherwise "incomplete", naming each missing or stale check and who must perform it (compliance records it in the screening register; the skill cannot).
7. Conditions. For an incomplete status, list the conditions precedent to use the supplier; for a flagged status, say the decision is blocked pending compliance and name the flag.
8. Approval. If the decision commits spend, call `tool:check_policy` with the value and report its reply as it is.

## Frameworks and heuristics

- A check is evidence only with a provider and a date. "The supplier says they are compliant" is a claim to record as not performed.
- Stale beats clear: a clear result past its maximum age is reported as stale and the status is incomplete.
- Subjects matter: a sanctions clear for the legal entity says nothing about an owner or director not screened; list the subjects covered.
- Our own record is a signal, not a check: many open findings or persistent price variance go into the narrative and the conditions, never into the clear/flagged status.
- Country and category risk come from the user or policy; the skill does not keep a country list.

## Policy hooks (SC-007)

Which checks are required for which decision, and their maximum age, come from governed policy via `tool:get_policy`; where no policy states them the skill applies the full list and its configured ages and says the rule is not governed. Committing spend to the supplier is checked with `tool:check_policy`.

## Output specification

1. Supplier and decision in scope; how the supplier was matched; master-record facts.
2. Check table: check, subject(s), performed, result, provider, date, valid-until, stale, flag.
3. Own trading record and live contracts, with what could not be assessed.
4. Risk events recorded in the lookback window.
5. Status: clear / flagged / incomplete, with the reason and the missing or stale checks.
6. Conditions precedent and who performs each; the approval reply when one was sought.

## Quality bar

- Every result quoted carries provider and date; nothing is inferred from a company name, website or the model's memory.
- No record is never clear; an expired record is never clear.
- The status rule is applied exactly as written; a flag anywhere means flagged.
- The supplier is confirmed with the user when the match was partial.

## Escalation and approval

Escalate to compliance for any flag, any check not performed on a supplier about to be awarded or onboarded, and any sanctions subject not screened. Escalate to the category lead when our own record shows persistent findings. The assessment is advice: compliance records checks, an authorised person decides, nothing is sent from the skill.

## Worked example

Asked whether Falconridge Systems can be awarded the pallets tender, the skill resolves the supplier (exact name), reads the master record (preferred, insurance valid, risk "not scored"), finds a sanctions screening recorded in the last year with result clear for the entity only, no financial-health, ESG or risk-event review recorded, and a scorecard showing low price variance and three open findings. Status: incomplete - owners and directors not screened, financial health and ESG not performed - with those as conditions precedent and a note that the award is blocked until compliance records them. The approval threshold reply from check_policy is reported as it is.

## Anti-patterns

- Clearing a supplier because nothing was flagged when nothing was checked.
- Treating a master-record risk score, a certification on file or a claim in an email as a performed check.
- Letting a stale clear stand because it was clear once.
- Naming country or sector risk from memory.
- Guessing which supplier was meant from a partial name.
