# Contract structures the maths can use — design

**Date:** 2026-10-02
**Status:** APPROVED by Nick 2026-10-02, including both stated assumptions (§3 sales order pipeline, §4 initial flag scope)
**Predecessor:** `specs/2026-10-01-document-relationship-layer-plan.md` (nine tasks, built and pushed)
**Rulings carried in:** `specs/2026-10-01-document-relationship-layer-rulings.md` — read it before changing any
classification behaviour.

---

## 1. What this is for

Nick's words: *"I want the various contract vocabulary types understood. So SOW, CCN, order form, sales order
are all part of the vocabulary, so when they are uploaded to contracts, THE MATH CAN START MAKING THE
CONNECTIONS."*

So the chain this design must complete is:

> a contract-family document is uploaded to Contracts → the system recognises **which structure** it is →
> that answer is **stored as a fact** → the relationship maths uses it to **propose the document's parent**
> (the SOW to its MSA, the call-off to its framework, the variation to the contract it changes).

**Success criteria.**

1. Every contract structure in the build spec is in the vocabulary, plus `sales order`, which Nick added.
2. A contract document's structure is readable from the database by something other than a log line.
3. The parent reference a structure declares can actually be extracted from the page.
4. For a contract document with a stored structure, the maths produces a **parent-link proposal** carrying
   its score and its reasons, for a person to confirm.
5. No document that classifies correctly today classifies differently afterwards. The one deliberate
   behaviour change (§6) affects only contract-category uploads, of which there have been **zero** — so it
   alters no existing document's outcome, and §4 is measured identical to today.

**Scope boundary.** Links are *proposed*, never written silently. This is not caution for its own sake: of
the 1,561 contracts that already carry a `parent_contract_id`, **0** resolve to a real contract (§2). Every
other relationship in this product proposes into a queue a buyer already works, and that is why.

---

## 2. The evidence this design rests on

Measured on 2026-10-02 against the configured database (`bp_testdb`, per
`reference_env_db_is_bp_testdb` — the seeded corpus, not `bp_sqldb`).

| Fact | Measurement |
|---|---|
| `proc.bp_contract_master` holds 3,051 contracts | `select count(*)` |
| `parent_contract_id` is populated on **1,561** of them | `where parent_contract_id is not null` |
| …and resolves to a real contract on **0** | the references are in another namespace: `C00002` points at `C1543`, which does not exist |
| `contract_type` is free text, 13 distinct values | `Consulting` 645, `NDA` 612, `SLA` 585, `Master Agreement` 581, `Service Agreement` 579, `NULL` 29, `Service Contract` 7, `Invoice` 4, `Policy` 3, `Service` 2, `Amendment` 2, `Indirect Procurement` 1, `Purchase Order` 1 |
| No SOW, CCN, order form or sales order anywhere in the corpus | none of the 13 values is one |
| `proc.process_monitor` holds 178 uploaded documents: 134 quote, 30 invoice, 14 po | `group by category` |
| **Zero contract-category uploads have ever happened** | same query — so nothing here regresses live contract behaviour, because there is none |
| 115 raw rows carry parsed text (`parser_snapshot.full_text`), covering 57 distinct `source_file` values, of which **53 are corpus documents** | those 53 are this design's measurement set. The same document has several raw rows from repeated extractions. The other 4 are another session's probe debris (`/tmp/.../scratchpad/zzprobe_*.csv`) and are excluded — they were the only `neither` and `evidence_only` outcomes |
| `proc.bp_contract_raw` carries `process_monitor_id` and `source_file` | the per-document anchor; `bp_contracts` and `bp_contract_master` carry neither |

Three code facts, each cited because a design decision rests on it:

- `src/services/extraction/dispatch.py:542` — the classification is *"Recorded, never acted on"*. It reaches
  a log line and, on disagreement, a review item. **It is never written to the document's row.**
- `extraction_schemas/contract.yaml` — has `parent_contract_id`, `amendment_ref`, `is_amendment`,
  `document_version`. Has **no** `framework_ref`, although the vocabulary declares that exact field as the
  call-off's pointer to its framework.
- `src/services/graph_resolution/pass_runner.py:273` — `run_all()` calls `supplier_identity` and
  `item_equivalence` only. `contract_succession` and `contract_coverage` are in `PASS_ORDER` and have
  scoring functions with unit tests, but **no runner calls them on real rows.**

---

## 3. The vocabulary carries every contract structure

Fifteen contract-family structures are already seeded in `proc.bp_document_type` and mirrored in
`src/services/concepts/seed.py`. They are listed here so the delta is unambiguous, not because they change:

| Structure | Matches on | Role |
|---|---|---|
| framework agreement | framework agreement, framework contract, framework | framework |
| master agreement | master agreement, msa, master service(s) agreement | master |
| SOW | sow, statement of work, work order, task order | master |
| call-off contract | call-off contract, call off contract, call-off | master |
| variation | variation, variation form, amendment, avenant, deed of variation | variation |
| schedule | schedule, annex, appendix, exhibit | attachment |
| addendum | addendum, supplemental agreement | variation |
| CCN | ccn, change control note, change note, change request | variation |
| termination notice | termination notice, notice of termination | termination |
| general notice | general notice, notice | notice |
| NDA | nda, non-disclosure agreement, confidentiality agreement | master |
| SLA | sla, service level agreement | attachment |
| service agreement | service agreement, service contract, services agreement | master |
| consulting agreement | consulting, consulting agreement, consultancy agreement | master |
| contract (unspecified) | contract, contracts, agreement | master |

**Two structures are added.** Both are rows in `proc.bp_document_type` plus the matching entry in `seed.py`,
applied by a migration in the established additive-and-idempotent shape, to **both** `bp_testdb` and
`bp_sqldb`. No code change makes a structure exist.

| New structure | Matches on | Role | Default parent | Pipeline |
|---|---|---|---|---|
| `doctype.order_form` | order form | master | `doctype.framework_agreement` | contract |
| `doctype.sales_order` | sales order, sales order acknowledgement, order acknowledgement | transaction | `doctype.order` | purchase_order |

Two notes on those rows.

- **`order form` is a structure in its own right, not an alias of the call-off contract.** It was dropped as
  a call-off alias on 2026-10-01 because it titled every quote-template workbook — 12 false disagreements out
  of 12 uses, no true positive. Re-adding it as an alias would repeat that. Adding it as its own structure,
  governed by §4, does not. This is the build spec's own sentence made operational:
  *"'order form' means one thing under a framework and another on its own"* (plan.md:703).
- **`sales_order`'s pipeline is `purchase_order`, not `contract`** — a sales order is the supplier's mirror of
  a purchase order and carries lines, quantities and a total, so the purchase-order extraction schema is the
  one that fits it. `pipeline_doc_type` only takes effect when an uploader types that category; a document
  dropped in the Contracts zone still routes on the category `contract`. **CONFIRMED by Nick, 2026-10-02:**
  a sales order extracts with the purchase-order schema, not the contract one.

`sales order` is longer than the existing `order` alias of `doctype.order`, and the resolver's longest-match
sweep (`type_resolver.py:442`) therefore gives it to `doctype.sales_order` without contest. Measured: adding
it changes **0** of the 53 documents.

---

## 4. A child structure must show its parent to claim the page

**The rule.** A structure whose purpose is to sit beneath a parent only wins the page when the page actually
names that parent or states an order of precedence. With no parent named, the structure stands down and the
remaining evidence decides.

**Where it lives.** A new boolean column `requires_parent_evidence` on `proc.bp_document_type`, so which
structures the rule governs is **data**, decided by a row, not by an edit to the resolver. The resolver reads
the flag when selecting tier-1 title concepts (`type_resolver.py:483-506`); a flagged concept is dropped from
`title_concepts` when the page shows no parent evidence.

**Which structures carry the flag initially: `doctype.order_form` alone.** `call_off_contract`, `sow` and
`schedule` are conceptually just as much children, but their behaviour today is measured — 47 agreed, 0
disagreed, 3 unresolved over 50 live documents (2026-10-01) — and flagging them would change outcomes with
no evidence calling for it. The column exists so flagging them later is an `UPDATE`.
**CONFIRMED by Nick, 2026-10-02:** the rule governs `doctype.order_form` alone to start.

**What counts as parent evidence.** Matchable phrases only, because the existing prose `structural_signals`
("lists incorporated documents") cannot match a page — open item 5 of the rulings. The initial set:
`framework`, `order of precedence`, `incorporat…`, `call-off`/`call off`, and
`(framework|master|parent|principal) (agreement|contract) (no|number|ref)`. These live beside the flag as
data, not as a regex buried in the resolver.

**Measured, both directions.** On the 53 corpus documents with stored parsed text:

| Run | Result |
|---|---|
| Today | 50 agreed, 3 declared_only |
| `order form` added, **no** rule | 40 agreed, **13 disagreed** |
| `order form` added, **with** the rule | 50 agreed, 3 declared_only — **identical to today** |

The 13 are the Aureus / Meridia / Orbis / Lattice / UITEST quote workbooks. Every document on this corpus
containing the words "order form" carries quote markers (`Quote Ref`, `Valid Until`, `Quotation`) and **not
one** carries `framework`, `order of precedence`, `incorporated`, `call-off` or a parent agreement number.

The rule also **fires**, which matters more than it passing: a page reading
`"ORDER FORM … made under Framework Agreement FW-2024-0012 … in the following order of precedence"`
classifies as `doctype.order_form`, while `"ORDER FORM … Quote Ref Q-1234 Valid Until 2026-12-01"` does not.

**Honest limit.** The corpus contains **zero real order forms**, so the true-positive half of the rule is
demonstrated on a constructed page and reasoned from the build spec, not measured on a real document. It
joins open item 1 of the rulings: the contract class still rests on one real PDF.

---

## 5. The answer gets stored

This is the blocker. A structure nothing records cannot feed any maths.

**Per-document.** Three columns on `proc.bp_contract_raw`, written where the classification already happens
(`dispatch.py` → `persistence.py`), and carried into `proc.bp_contracts` on promotion:

| Column | Holds |
|---|---|
| `resolved_doc_type` | the concept code the page resolved to, e.g. `doctype.sow` — `NULL` when the page said nothing |
| `resolved_role` | that structure's relationship role, e.g. `role.master` |
| `type_agreement` | `agreed` / `refined` / `disagreed` / `declared_only` / `evidence_only` / `neither` |

`bp_contract_raw` is the right home because it is the only contract table carrying `process_monitor_id` and
`source_file` — the per-document identity. `bp_contracts` and `bp_contract_master` carry neither.

Writing `NULL` when the page says nothing is deliberate and follows `feedback_no_fabrication_null_when_absent`:
an unrecognised page records no structure rather than a guessed one.

**The existing corpus.** The 13 free-text `contract_type` values map onto concept codes through the
vocabulary's own alias index. **Measured 2026-10-02: 9 of the 13 resolve, covering 3,016 of the 3,051 rows
(98.9%)** — `Consulting`→`doctype.consulting_agreement` (645), `NDA`→`doctype.nda` (612),
`SLA`→`doctype.sla` (585), `Master Agreement`→`doctype.master_agreement` (581),
`Service Agreement` (579) and `Service Contract` (7)→`doctype.service_agreement`,
`Invoice`→`doctype.invoice` (4), `Amendment`→`doctype.variation` (2),
`Purchase Order`→`doctype.order` (1).

The 35 rows that do not resolve stay `NULL`: `NULL` already (29), `Service` (2), `Indirect Procurement` (1),
and `Policy` (3). **`Policy` is the interesting one** — `doctype.policy_document` claims the alias `policy`,
but its concept is `status='proposed'`, and a proposed concept deliberately never resolves. So those 3 rows
are not a gap in the vocabulary; they are a row awaiting confirmation, which is the mechanism working. An
earlier draft of this section counted `Policy` as mapped; it does not. **The source column is not modified**
(`project_extraction_accuracy_priority`: never modify source data); the mapping is read through a view or a
resolved column written beside it.

---

## 6. A specific structure under a generic declaration is a refinement, not a disagreement

Uploading into the Contracts zone declares the category `contract`, which resolves to
`doctype.contract_unspecified`. Today any page naming its real structure — "Master Agreement", "Statement of
Work" — produces `evidence_concept != declared_concept`, which is `agreement = "disagreed"`
(`type_resolver.py:615`) and therefore a `document_type_disagreement` review item
(`type_resolver.py:694`).

That is wrong on its face: a document uploaded as "a contract" that turns out to be a master agreement has
not contradicted anybody. Left alone it would put a review item on **every contract Nick uploads**, which is
precisely the flood this layer exists to avoid.

**The rule.** When the declared structure is `doctype.contract_unspecified` and the page resolves to a
structure whose `pipeline_doc_type` is `contract`, the agreement is `refined`. The resolved structure is
stored (§5) and **no review item is raised**. Any other mismatch stays `disagreed` and still raises one.

`refined` is a new value for `agreement` and for the `type_agreement` column, so every check constraint and
consumer that enumerates the values must accept it — including
`tests/services/test_type_findings_lifecycle.py`.

**Regression risk: none today**, because zero contract-category documents have ever been uploaded (§2). The
change is invisible until contracts flow, which is the point of making it now.

---

## 7. The parent reference becomes extractable

The vocabulary declares that a call-off points at its framework through a field named `framework_ref`. **That
field does not exist** in `extraction_schemas/contract.yaml`, so the pointer is declared and never filled.

Added to the contract schema, in the shape the existing `parent_contract_id` field uses (anchored patterns, a
value regex that rejects `N/A`/`TBC`/`TBD`, `required: false`, a confidence threshold):

| Field | Anchors on | Points at |
|---|---|---|
| `framework_ref` | "Framework Agreement", "Framework Contract", "Framework No/Ref", "made under" | `doctype.framework_agreement` |
| `parent_agreement_ref` | "Master Agreement", "Master Services Agreement", "under the Agreement", "pursuant to" | `doctype.master_agreement` |

`parent_contract_id` already anchors on "Parent/Master/Principal Contract|Agreement" and
"amends|amendment to|supplements|varies", and keeps that job. `parent_agreement_ref` exists separately
because a SOW names its MSA without amending anything, and collapsing the two would make "the document this
one sits under" and "the document this one changes" the same fact.

Both are `required: false`. A contract with no parent named extracts and promotes exactly as it does now.

---

## 8. The maths runs, and proposes

A new profile, `contract_hierarchy`, registered through the existing `register_profile()` seam in
`src/services/linking_engine.py` — the seam the rulings already identify as the extension point (discovery
§4 row 10), so no new machinery.

**Input:** a contract document with a stored `resolved_doc_type` (§5), and the candidate parents — contract
rows whose `resolved_doc_type` matches the child structure's `default_parent_type`.

**Signals**, each scoring and reporting `OK` / `MISSING` / `CONFLICT` in the house style:

| Signal | What it compares |
|---|---|
| `declared_reference` | the child's extracted `framework_ref` / `parent_agreement_ref` / `parent_contract_id` against the parent's `contract_id`, after the declared normalisation |
| `expected_structure` | whether the parent's structure is the one the child's structure expects (`doctype.sow` → `doctype.master_agreement`) |
| `supplier` | same supplier, reusing the existing `supplier_identity` result rather than re-deriving it |
| `term_containment` | the child's dates falling inside the parent's term |
| `title_overlap` | shared distinctive words between the two titles |

**An exact reference match does not auto-link.** This contradicts the build spec's principle 3 and the
discovery already adjudicated it (§6.2): the only identifier available at scale resolves 0 times out of
1,561. An exact match after normalisation is the strongest *signal*; it is not a decision.

**Output:** a parent-link proposal, through `link_proposals.py`, into the queue a buyer already works —
carrying the score, the band, and the per-signal detail so a person sees why. `contract_hierarchy` joins
`edge_writer.UNCALIBRATED_PROFILES`, so it cannot auto-link at any score until a labelled sample exists —
the product's own standing rule, which applies here.

**It gets a runner**, and the runner is called. The failure mode this layer must not repeat is
`contract_succession`: a scored, unit-tested profile that nothing ever runs.

---

## 9. Testing

Every guard below must be **proven to fail** before it is accepted — the lesson of
`feedback_prove_the_guard_fails` and of this layer's own history, where fourteen guards were found green
while checking nothing. For each one, break the behaviour on purpose, watch the test go red, restore, watch it
go green. A test whose red state was never observed is not evidence.

| # | Guard | How it is broken to prove it |
|---|---|---|
| 1 | `order form` with no parent named does not claim the page | clear `requires_parent_evidence` on the row → the 13 quote workbooks go `disagreed` |
| 2 | `order form` naming its framework **does** classify as `doctype.order_form` | remove `framework` from the parent-evidence phrases → the real order form reads as unknown |
| 3 | The 53 corpus documents resolve identically before and after | an end-to-end comparison against a committed baseline of today's outcomes |
| 4 | `sales order` wins over the `order` alias | shorten the alias to `sales` → `doctype.order` takes the page |
| 5 | A resolved structure reaches `bp_contract_raw` and survives promotion into `bp_contracts` | drop the column write → the row holds `NULL` |
| 6 | A page that says nothing stores `NULL`, never a guess | make the writer fall back to the declared type → the test sees a structure where there is no evidence |
| 7 | `contract_unspecified` + a specific contract structure = `refined`, no review item | revert to the old comparison → a `document_type_disagreement` appears |
| 8 | A genuine mismatch (`quote` declared, `invoice` on the page) still raises a review item | widen `refined` to all mismatches → the finding vanishes |
| 9 | `framework_ref` extracts from a real framework reference and rejects `N/A` | remove the rejection branch from the value regex → `N/A` is stored as a reference |
| 10 | `contract_hierarchy` proposes and never links | remove it from `UNCALIBRATED_PROFILES` → a high score auto-links |
| 11 | An exact reference match alone does not reach the auto band | give `declared_reference` a weight that dominates → a single match auto-links |
| 12 | The runner is actually called | remove the call from the entry point → the test sees zero proposals for a corpus that should produce them |
| 13 | Seed and table do not drift | the existing full-column drift test, extended to the new column and the two new rows |

Live verification, per `feedback_demonstrate_on_local_server_live_data`: upload a real contract document to
the Contracts zone on the running local server, and show the stored structure, the extracted parent
reference, and the resulting proposal. Tests alone do not close this.

---

## 10. Out of scope

- **`contract_succession` and `contract_coverage` getting their missing runner.** Real, worth fixing,
  separate: they answer "which contract replaced which" and "what spend sits under a contract", not "what is
  this document's parent".
- **Repairing the 1,561 broken `parent_contract_id` values.** They are corpus artefacts in a dead namespace.
  This design reads them as a signal and never trusts them.
- **Flagging `call_off_contract`, `sow` and `schedule` as requiring parent evidence.** The column makes it an
  `UPDATE` once documents justify it.
- **New structures nobody has named** (DPA, licence agreement, supply agreement, LOI, MOU, novation, side
  letter, rate card). Nick confirmed the list is complete. A structure with no documents and no request is a
  row that proves nothing — the same empty-set mistake this layer has deleted repeatedly.
- **Scope resolution, sector glossary, conflict register.** Still blocked on a sector or normalised-region
  dimension that does not exist (discovery §4).

---

## 11. What is unproven, stated plainly

1. **Zero real order forms, frameworks, call-offs or SOWs exist in the corpus.** §4's rule is measured on its
   negative side (13/13 quote workbooks correctly unaffected) and constructed on its positive side. Ten to
   twenty real contract documents would change what can be proven here — open item 1 of the rulings, still open.
2. **`sales_order` is seeded on a reasoned mapping, not an observed document.** Nothing in the corpus contains
   the words.
3. **The parent-evidence phrase set is small by design.** It will miss a real order form that names its
   framework in words nobody listed. That failure is visible — the document reads as unknown and raises a
   review item — rather than silent.
4. **`bp_sqldb` needs the same migrations.** Both new rows, the new column, and the schema change. The
   governance tables were 8 migrations behind as recently as 2026-09-15
   (`project_bp_sqldb_governance_caught_up`), and the discrepancy index blackout cost 65 days of findings
   there. Deployment to both databases is a prerequisite, not a follow-up.
