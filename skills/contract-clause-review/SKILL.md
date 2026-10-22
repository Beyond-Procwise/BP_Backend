---
name: contract-clause-review
description: Review what a contract on record says on a point (liability, indemnity, termination, notice, payment, price review, data, exit), compare two wordings of a clause or a whole document, and list the obligations each party carries, quoting the stored clauses word for word. Use when asked what a contract says, to redline or mark up a clause, to compare a supplier's wording with ours, or what we are obliged to do under an agreement. Not for deciding whether to renew (renewal-and-exit-strategy), for designing a price formula (pricing-mechanisms) or for legal advice.
---

# Contract Clause Review

## Purpose

Answer "what does the contract say, and what changed" from the clauses the platform has read and stored, quoted exactly, so a buyer can take a position on wording with evidence and knows which clauses were never read. The skill explains wording and its commercial effect; it does not give legal advice and never says a contract is enforceable.

## When to use and when not

Use when the user asks what a contract or clause says, wants a clause reviewed or marked up, wants two versions compared, asks what either party is obliged to do, or asks whether a proposed wording is better or worse for us.

Not for the renew-or-exit decision (renewal-and-exit-strategy), index-linked price formulas (pricing-mechanisms), negotiation tactics (negotiation-planning), or whether a purchase is allowed under policy (procurement-policy-compliance, which this skill calls for approval questions).

## Inputs and preconditions

- The contract: its contract_id, document_id or title words, and the supplier if several match.
- The point to review (liability, indemnity, termination, notice, payment terms, price review, confidentiality, data, exit assistance, exclusivity), or the two wordings to compare, pasted exactly.
- The user's position or concern, if any, and the contract's annual value and currency when a cap or exposure is at issue.

Use `tool:search_contracts` to find the contract and `tool:extract_clauses` to read its stored clauses word for word, filtering with `clause_type` or `containing` (for example "liab", "indemn", "terminat", "notice", "payment", "data"). A contract that has not been read returns no clauses: say so and ask the user to paste the clause, labelled unchecked. Use `tool:obligation_tracker` for the stored dates and the clauses that carry a notice, deadline or review, and `tool:redline_compare` for two wordings the user supplies. Use `tool:get_policy` for the organisation's own contracting rules (standard positions, who may accept a deviation) and `tool:check_policy` when accepting the wording commits money or needs an approver.

Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated.

## Workflow

1. Locate. Find the contract and confirm it is the one meant (several matches: ask). State whether its clauses were read and how many are stored; what is not stored is "not checked", never "absent".
2. Read the point. Pull the clauses on the point with `tool:extract_clauses`; quote each with its clause reference and type and name the parties it binds. If the user pasted a clause, quote theirs and label it unchecked against the document.
3. Explain the effect. For each clause say, in plain words, who must do what, when, and what happens if they do not; name any cap, carve-out, trigger or deadline in the wording itself. Do not infer a term the wording does not state; say "the clause is silent on X".
4. Compare, when asked. Run `tool:redline_compare` on the two wordings exactly as given and report every addition, removal and replacement in reading order, then say which party each change favours and why, in one sentence each. A comparison longer than `config.thresholds.redline_max_chars` characters is split by the user into clauses first.
5. Obligations. When asked what we or they must do, list the stored obligations as a register: party, obligation, trigger or deadline, clause, consequence, quoted wording. Flag any with a date or notice attached and hand renewal questions to renewal-and-exit-strategy.
6. Position. Against the organisation's standard position from `tool:get_policy` where one exists (otherwise against the user's stated concern), rate each clause as acceptable, negotiate or escalate, with the reason. A liability cap below the contract's annual value, an uncapped indemnity, or an exposure above `config.thresholds.escalate_liability_multiple` times annual value is always escalate. Read no more than `config.thresholds.max_clauses_per_pass` clauses in one pass; ask the user to narrow a longer contract.
7. Approval. If accepting the wording commits spend or changes a cap, call `tool:check_policy` with the value and term and report its reply as it is.
8. Output the review (section below). Say which clauses were quoted from the document, which were user-supplied, and which were not checked.

## Frameworks and heuristics

- Quote, then explain: the wording first, the plain-words effect second, the position third. Never paraphrase in place of the quote.
- Silence is a finding: a clause that does not state a cap, a cure period or a notice period is reported as silent, which is itself a risk.
- Mutuality check: for every obligation on us, look for the matching one on them (and the reverse); an imbalance is a negotiating point, not a verdict.
- Termination and notice are read together with `tool:obligation_tracker`: a right to terminate is worth little if the notice deadline has passed, and the notice period is never computed, only read from the clause. Any deadline the plan depends on keeps `config.thresholds.notice_buffer_days` business days of buffer.
- Wording that references a schedule, annex or definition not in the stored clauses is reported as depending on text not read.

## Policy hooks (SC-007)

The organisation's standard contract positions and who may accept a deviation come from governed policy via `tool:get_policy`; where no policy states a position, say the point is not governed and give the commercial reasoning only. Accepting wording that commits money or changes a liability cap is checked with `tool:check_policy`, whose reply is reported as it is.

## Output specification

1. Contract and scope: which contract, whether its clauses were read, how many are stored, what was asked.
2. Clause review table: clause reference, type, parties, quoted wording, effect in plain words, position (acceptable / negotiate / escalate) with reason.
3. Comparison (when two wordings were given): the change list in reading order and who each change favours.
4. Obligation register (when asked): party, obligation, trigger or deadline, clause, consequence.
5. Open points: clauses not stored or not checked, schedules referenced but not read, questions for legal.
6. Next step and who decides; the approval reply when one was sought.

## Quality bar

- Every quoted clause carries its clause reference and came from `tool:extract_clauses` or is labelled user-supplied.
- No legal conclusion: the skill says what the wording does and whether it meets our position, never whether it is enforceable or lawful.
- A missing clause is "not checked" or "silent", never "absent".
- Every "escalate" names the trigger: cap below annual value, uncapped exposure, missed notice, text not read.

## Escalation and approval

Escalate to legal for enforceability, governing-law and jurisdiction questions, indemnities for third-party claims, and any clause referencing text not on record. Escalate to the category lead when the position is "negotiate" on more than a handful of clauses. The review is advice; a named person accepts or rejects wording, and nothing is sent to the supplier from this skill.

## Worked example

Asked what the facilities contract says on termination, the skill finds one contract, reads its stored clauses filtered on "terminat" and "notice", and quotes clause 27.4: "Either party may terminate this Agreement on ninety days' written notice." It explains that either side can end the agreement without cause with the stated notice, notes the clause is silent on termination for cause and on what happens to prepaid fees, reads the expiry date from the register, and rates the clause "negotiate" because the organisation's standard position (from policy) asks for a cure period on breach. It lists the open point for legal and hands the renew-or-exit question to renewal-and-exit-strategy.

## Anti-patterns

- Paraphrasing a clause instead of quoting it.
- Saying a contract has no cap, no notice period or no exit clause when the clauses were simply not read.
- Computing a notice deadline from a period the clause does not state.
- Giving a legal opinion or telling the user a clause is unenforceable.
- Treating the supplier's proposed wording as the contract when the stored wording differs.
