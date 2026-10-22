---
name: contract-amendment-review
description: "Review a proposed change to a live contract - an amendment, variation, change order, addendum or side letter - against the agreement as stored: what the change alters, which clauses it touches, whether the contract's own change procedure is followed, the price and term effect, and the approval the change needs. Use when a supplier or a stakeholder proposes changing an existing agreement. Not for the first review of a contract (contract-clause-review), for renewal decisions (renewal-and-exit-strategy) or for drafting the new wording as a legal document."
---

# Contract Amendment Review

## Purpose

Make sure a change to a live agreement is understood before it is signed: what it changes, what it costs, what it touches that nobody mentioned, and who must approve it. The skill compares and reports; it drafts no legal wording and signs nothing.

## When to use and when not

Use when the request concerns an amendment, variation, change order, addendum, side letter, scope change or price change to an agreement already on the register.

Not for reviewing a new contract's clauses (contract-clause-review), for whether to renew or exit (renewal-and-exit-strategy), for negotiating the change (negotiation-planning) or for pricing clauses in a new agreement (pricing-mechanisms).

## Inputs and preconditions

- The contract (name, supplier or document) on the register.
- The proposed change: the text, or the user's description labelled user-supplied.
- Who proposed it and why, if known.

- Use `tool:search_contracts` to find the agreement and its stored summary
- Use `tool:extract_clauses` for the clauses the change touches and for the contract's own variation or change-control clause, quoted
- Use `tool:redline_compare` to compare the proposed wording with the stored wording when both are supplied
- Use `tool:obligation_tracker` for the dates and obligations the change would move
- Use `tool:run_calculation` with `deal.pct_change` for the price effect of a rate change and `deal.scaled_total` for the new total value over the term, with the contract's or the user's figures
- Use `tool:check_policy` for the approval the changed value or term needs
- Use `tool:build_docx` when the user asks for a Word file of the result, after the answer is written, with only what the tools returned or the user stated
NOT AVAILABLE: e_signature. nothing is signed; the amendment goes to the signatory after approval

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.material_change_pct`.

## Workflow

1. Find the agreement and read its stored summary, expiry, value and parties.
2. Quote the contract's variation or change-control clause: who may propose, in what form, who must sign. If none is stored, say the procedure is unknown and refer the change to legal before anything is treated as agreed.
3. Map the change to the clauses it touches: price, scope, term, service levels, liability, termination, notice. Quote each touched clause. Flag clauses the change affects by implication (a term extension moves the notice deadline; a scope increase may breach a liability cap or an insurance level).
4. Compare wording with redline compare when both versions are supplied; otherwise list the differences the user described, labelled.
5. Quantify: the price effect over the remaining term with `deal.pct_change` on the contract's figures; the new total contract value with `deal.scaled_total` from the new rate and the volume the contract or the user states; a change above `config.thresholds.material_change_pct` of the stored value is material and goes to the full approval route.
6. Approval: check policy on the new value and term; report as replied. List the signatories the clause requires.
7. Output the review with the open questions for legal and the proposer.

## Frameworks and heuristics

- A change to one clause moves others: term affects notice and renewal; scope affects price, liability and insurance; price affects any index or rebate clause.
- The contract's own change procedure governs; an email agreeing a change may not amend the contract.
- Material changes to value, scope or term are a new commercial decision and take the full approval route.
- A side letter that contradicts the main agreement is a finding for legal, not a convenience.

## Policy hooks (SC-007)

The new value and any term extension are checked with `tool:check_policy` and the route is reported as replied. Signing is a person's act under the signatory rule the contract or policy names; the skill never signs or sends the amendment.

## Output specification

1. Agreement: parties, expiry, stored value, change-control clause quoted.
2. Change table: what changes, clause touched (quoted), implied effects on other clauses.
3. Wording comparison or the described differences, labelled.
4. Financial effect: price change, new total value, materiality against the configured share.
5. Approval route and required signatories.
6. Open questions for legal and the proposer; recommendation.

## Quality bar

- Every clause cited is quoted from the stored contract; none paraphrased from memory.
- The price effect uses the contract's or the user's figures and the calculation registry.
- Implied effects are listed, not only the clauses the proposer named.
- Nothing is signed, sent or recorded as agreed.

## Escalation and approval

Escalate to legal where the change-control clause is missing or not followed, where a side letter contradicts the agreement, or where liability or termination terms move; to the approver the policy names for the value; to finance where the change alters the committed spend.

## Worked example

Asked to review a supplier's proposed amendment raising rates and extending the term by two years, the skill finds the agreement, quotes the variation clause (written, signed by both parties), the price review clause (index-linked, annual) and the notice clause, shows the proposed increase exceeds what the index clause allows, computes the effect over the remaining and extended term, finds it material against the configured share, reports the approval route the policy returns, and lists for legal that the extension moves the notice deadline and that the amendment is silent on the liability cap.

## Anti-patterns

- Reviewing only the clause the proposer mentioned.
- Accepting an email exchange as an amendment when the contract requires a signed variation.
- Computing the price effect from an assumed volume.
- Signing or sending the amendment.
