---
name: contract-drafting
description: "Draft a contract from the organisation's approved templates and clause library - the right template for the deal type, the commercial schedule filled from the award or the quote, the clauses chosen for the risks the deal carries, the gaps and deviations listed for legal - as a document for legal to finish, never as a signed instrument. Use when asked to draft a contract, an agreement, a schedule or an SOW from a template. Dormant until the template and clause library exist on the platform. Not for reviewing a supplier's paper (contract-clause-review) or amendments (contract-amendment-review)."
---

# Contract Drafting

## Purpose

Produce a first draft legal can work from fast: the right template, the commercial terms exactly as awarded, the clauses the risk calls for, and an honest list of what is missing. The skill drafts text; legal owns the contract and a person signs.

## When to use and when not

Use when a deal is awarded or agreed and the user asks for the contract, the agreement, the schedule or the statement of work drafted from the organisation's templates.

Not for reviewing the other side's draft (contract-clause-review), for an amendment (contract-amendment-review), for pricing clauses in depth (pricing-mechanisms), for service levels (sla-design) or for signing.

## Inputs and preconditions

- The deal: supplier, scope, prices, volumes, term, Incoterm, payment terms, service levels, from the award or the quote.
- The deal type and the risks the owner sees.
- The templates and clause library, when they exist.

- Use `tool:search_contracts` for comparable agreements on the register as the nearest thing to a template today
- Use `tool:extract_clauses` for clauses from those agreements, quoted, as a clause source until a library exists
- Use `tool:supplier_lookup` for the supplier's legal details on record
- Use `tool:get_policy` for the contracting rules (mandatory clauses, signatory, governing law, insurance levels)
- Use `tool:check_policy` for the approval the contract value needs
- Use `tool:build_docx` to produce the draft as a document for legal
NOT AVAILABLE: clause_library. no template or clause library exists yet; clauses are drawn from comparable agreements on the register, quoted, and the gaps named for legal
NOT AVAILABLE: e_signature. nothing is signed; the draft goes to legal and the signatory

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_gaps_before_legal_call`.

## Workflow

1. Frame: deal type, parties, scope, term, value; the approval route for the value.
2. Read the contracting rules: mandatory clauses, governing law, insurance levels, signatory.
3. Choose the template: the library's template for the deal type, or, until it exists, the comparable agreement on the register, with its clauses quoted.
4. Fill the commercial schedule from the award or quote exactly: prices, volumes, term, Incoterm, payment terms, service levels; nothing from memory.
5. Choose clauses for the risks: data, IP, liability, termination, exit assistance, audit; each quoted from the library or a comparable agreement; a risk with no clause on record is a gap for legal.
6. List deviations from the mandatory clauses and the gaps; build the draft with the export tool, marked DRAFT for legal; hand to approval-routing and legal.

## Frameworks and heuristics

- The commercial schedule is the award, word for word; the legal terms are the template, word for word.
- A clause invented by the skill is a gap for legal, not a clause.
- Every deviation from the mandatory set is a decision for legal, listed.
- The draft says DRAFT until legal says otherwise.

## Policy hooks (SC-007)

Contracting rules come from `tool:get_policy` and are quoted; the value's approval comes from `tool:check_policy` as replied. Nothing is signed or sent; the draft goes to legal and the signatory named by policy.

## Output specification

1. Frame and approval route.
2. Template or comparable agreement used, with its clauses quoted.
3. Commercial schedule as filled, with its source.
4. Clauses chosen for the risks; gaps and deviations for legal.
5. The draft's export id, marked DRAFT.

## Quality bar

- Every commercial term matches the award or quote exactly.
- Every clause is quoted from the library or a comparable agreement; none written from memory.
- Gaps and deviations are listed, not papered over.
- Nothing is signed or sent.

## Escalation and approval

Escalate to legal for every gap and deviation, to the approver for the value, and to the category lead where the award and the draft differ.

## Worked example

Asked to draft the pallet supply agreement after the award, the skill reads the contracting rules, finds the closest supply agreement on the register and quotes its clauses, fills the schedule from the award (prices, volumes, term, DAP, thirty days), chooses the liability, termination and exit-assistance clauses quoted from that agreement, lists that no data clause exists on record as a gap, builds the DRAFT with build_docx and hands it to legal and the approval route.

## Anti-patterns

- Writing a clause from memory.
- Changing a commercial term while drafting.
- Dropping a mandatory clause silently.
- Calling the draft final or sending it to the supplier.
