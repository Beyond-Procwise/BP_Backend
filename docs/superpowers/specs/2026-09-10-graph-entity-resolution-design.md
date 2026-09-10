# Graph entity resolution — design

**Date:** 2026-09-10
**Status:** approved for planning
**Supersedes:** Tier 1 of the buy-side capability audit (2026-09-10), which proposed
SQL crosswalk extension and a `contract_id` backfill. Both were rejected: they made
connections by picking and comparing in SQL rather than by evidence in the graph.

---

## 1. Purpose

Three questions the buy-side product cannot answer today, all of them connection
questions rather than calculation questions:

1. **Are these two supplier records the same company?**
2. **Are these two line items the same thing?**
3. **Is this invoice covered by a contract?**

Each is answered here by a **registered relationship profile** on the existing
`linking_engine`, whose verdict lands as a **banded edge in Neo4j**. No new scoring
mathematics is written. The engine, its bands, its clustering and its dampening are
the framework; this design adds profiles, nodes, edges and one composition rule.

### What this unblocks

| Original gap | Today | After |
|---|---|---|
| Off-contract spend | impossible — `contract_id` 0/38,498, supplier keyspaces disjoint | answerable, with the missing evidence named |
| Price variance vs benchmark | SQL string-match on `item_description` | entity-based via shared `Item` |
| No-competing-quote | supplier count per deal | quotes for the same `Item` across suppliers |
| Maverick spend | detector flags 100% of POs | spend with no path to a contract |
| Renewal uplift | dead — 0 of 1,561 parent links resolve | contract succession as a resolution profile |

### Out of scope

- **Customer / buying-entity model.** A separate build. The tenant is the buyer and
  is a constant in a single-tenant deployment; a constant does not need resolving.
  Contract coverage is `supplier + date-in-term (+ category)` — every term is
  supplier-side or contract-side.
- **The category dimension.** `bp_category_product_map` holds 21 free-text rows and
  resolves 0 of 4,845 items. Category enters the profiles as a signal that will
  score `MISSING` until that dimension exists — contributing nothing, penalising
  nothing (§4.3).
- **Benchmark reference data, payment status, supplier performance history.**

---

## 2. The framework this builds on

Read before implementing. Nothing below is proposed; it is what exists.

### 2.1 Pairwise scorer — `src/services/linking_engine.py`

```
c_i    = weight_i × r_i × (2·s_i − 1)            per signal
total  = Σ_clusters  dampen(n_active) × Σ c_i    per cluster
L      = log(p0/(1−p0)) + alpha × total          log-odds
P_raw  = 1/(1+e^−L)
F      = min(F_cap, P_raw · C · S · Q) × 100
```

- `s ∈ [0,1]` is a comparator's match score. `(2s−1)` maps it to `[−1,+1]`, so a
  **disagreeing signal contributes negative evidence**, not merely zero.
- `q = 0` if the signal could not be evaluated (`MISSING`), else `1`. `r = q × appl`.
  **Unobservable evidence contributes nothing.** There is no middle value.
- `dampen(n_active)`: 1 → 1.0, 2 → 0.85, 3+ → 0.70. Applied **per cluster**. This is
  the correlation control: signals in one cluster are assumed to be correlated, and
  clusters are assumed independent of each other.
- `C` is coverage: `floor + (1−floor)·rho`. `S = 1.0` and `Q = 1.0` are pinned in the
  engine (see `docs/formula-registry-gap-report.md` D-6 for the `S` caveat).
- `F_cap` applies a Tier-1 conflicting signal's cap.

### 2.2 Bands — `_band()`

| F | Band | Meaning |
|---|---|---|
| ≥ 92.0 | `auto_link` | acted on |
| ≥ 80.0 | `auto_link_with_warning` | acted on, flagged |
| ≥ 65.0 | `review` | human queue |
| ≥ 45.0 | `weak_relation` | recorded, never acted on |
| < 45.0 | `block_or_exception` | nothing is said |

Policy is already settled and is not re-decided here
(`duplicate_invoice_detector.py:155-159`): *"The engine's own bands decide, rather
than a threshold invented here."*

### 2.3 Extension seam

`register_profile(name, profile)` and `register_signal(kind, fn)`. Used today by
`quote_rival` (`requirement_similarity.py:110`) and `invoice_duplicate`
(`duplicate_invoice_detector.py:150`). A profile is
`{p0, alpha, floor, signals, date_field}`.

### 2.4 Global resolution — `src/services/resolution/`

A HiGHS MILP turning pairwise verdicts into a globally coherent assignment, reporting
how forced it was: `RESOLVED` / `DEGENERATE` / `INFEASIBLE`, with a margin and a
`certify()` that explains infeasibility in procurement terms.

`CandidateEdge.log_odds` is documented as *"from the existing scorer, pre-sigmoid"*;
`confidence` is *"post-sigmoid, for reporting only"*. **Log-odds is already the
primitive that travels.** `consumes` already anticipates this work: *"a clause slot
consumes exactly one mapping."*

### 2.5 Formula registry — `src/services/formulas/`

Typed contracts, versions, source hashes, and golden vectors **checked at import
time**. Every profile added here gets a registered formula with vectors.

---

## 3. Graph model

### 3.1 Current state (measured 2026-09-10)

237,502 nodes, 270,271 relationships, mirroring `_trgt` exactly. `Contract` has a
live uniqueness constraint and **zero nodes**, because `procurement_kg_builder.py:63`
reads `proc.bp_contracts` (0 rows) while 3,051 contracts sit in `bp_contract_master`.
`Category` appears in `FK_RELATIONSHIPS` but has **no entity mapping**, so both
category edge rules are dead. **Every one of the 270,271 edges is bare** — no
properties at all.

### 3.2 New nodes

| Node | Key | Source |
|---|---|---|
| `Contract` | `contract_id` | §6 — source decision |
| `Item` | `item_key` | minted by `item_equivalence` (§5.2) |

`Item` is a **derived** node: it does not mirror a table. Its `item_key` is a digest
of the equivalence class's **canonical member, defined as the lexicographically
lowest `item_id` in the class** (falling back to the lowest normalised
`item_description` where no member carries an `item_id`). The rule is arbitrary but
it must be *deterministic*: a rebuild has to reproduce the same `item_key`, or every
`OF_ITEM` edge and every finding that cites one breaks on the next pass.

### 3.3 New edges

| Edge | From → To | Profile |
|---|---|---|
| `SAME_ENTITY` | `Supplier` → `Supplier` | `supplier_identity` |
| `OF_ITEM` | `InvoiceLine`/`POLine`/`QuoteLine` → `Item` | `item_equivalence` |
| `UNDER_CONTRACT` | `Invoice`/`PurchaseOrder` → `Contract` | `contract_coverage` |
| `SUCCEEDS` | `Contract` → `Contract` | `contract_succession` |

### 3.4 Edge properties — the end of bare edges

Every derived edge carries:

```
F               float    0–100, the banded score
band            string   auto_link | auto_link_with_warning | review | weak_relation
P_raw           float    this edge's own probability — what a later profile reads as `s` (§4.1)
L_evidence      float    alpha × total_cluster_score — the evidence term, NO prior (§4.2)
profile         string   e.g. "supplier_identity"
profile_version string   from the formula registry
signals         string   JSON of the per-signal breakdown score_link already returns
observations    string   sorted digest of base observations consumed (§4.4)
resolution      string   RESOLVED | DEGENERATE | INFEASIBLE, where §5.4 applies
margin          float    from the resolution certificate, where applicable
scored_at       datetime
```

`signals` is stored as JSON text rather than a map because Neo4j relationship
properties cannot hold nested structures. It is for explanation, not for querying.

### 3.5 New constraints

`Item.item_key` unique. `Contract.contract_id` already exists.

---

## 4. Composition — the one new rule

This is the only genuinely new thinking in the design, and it is a rule about
**how evidence is assembled**, not new arithmetic.

### 4.1 Composition happens inside a profile, never across scores

A derived edge participates in a later question as a **signal within that
question's profile**, whose match score `s` is the supporting edge's probability.

```
contract_coverage signal "supplier_same":
    s = the SAME_ENTITY edge's stored P_raw          (§3.4)
    q = 1 if such an edge exists at all, else 0
```

`s` reads the supporting edge's **stored `P_raw`** — the probability that edge's own
profile already computed, with that profile's own prior applied once, at the time it
was scored. It is never recomputed from `L_evidence` at read time. `L_evidence` is
persisted only so a future recomposition can rebuild an answer from prior-free
evidence terms without inheriting a prior it did not intend (§4.2).

`(2s−1)` then does the right thing with no further work:

| Supporting edge | `s` | `(2s−1)` | Contribution |
|---|---|---|---|
| strong (P=0.97) | 0.97 | +0.94 | strong positive |
| coin-flip (P=0.50) | 0.50 | 0.00 | **none** |
| actively disbelieved (P=0.05) | 0.05 | −0.90 | strong negative |
| no edge at all | — | `q=0` | **none** |

So four independently-resolved weak paths can compose past a band that none of them
reaches alone — which is the intent — while a genuinely uncertain edge contributes
nothing rather than half-support.

### 4.2 One prior, always

Every `L` embeds `log(p0/(1−p0))`. At `p0 = 0.02` that is **−3.89**. Summing the `L`
of four supporting edges would inject **−15.6** of evidence that does not exist.

**Rule: never sum `L` across edges.** Composition runs a *single* `score_link` call
for the question, with direct and derived signals in one profile. There is exactly
one prior per answer. Edges therefore persist `L_evidence` (the `alpha × total` term,
prior excluded) so a consumer cannot accidentally re-add a prior.

### 4.3 Missing evidence never inflates and never penalises

`q=0 → r=0 → c=0`. A category signal that cannot be evaluated because the category
dimension is empty contributes nothing at all. Contract coverage still scores on the
identity, temporal and commercial clusters. This is why the empty category dimension
degrades lift rather than blocking the feature.

### 4.4 Independence discipline is implemented by cluster assignment

The failure mode this design invites: item equivalence partly derived from supplier
co-occurrence, then reused as independent evidence about that supplier — one
observation counted twice.

**A base observation** is a `(document_id, field)` pair. Every derived edge records
the sorted digest of the observations its signals consumed, in `observations`.

**Composition rule:** when assembling a profile's signals, if two signals' observation
sets intersect, they are assigned to the **same cluster**. `dampen()` then discounts
them exactly as it discounts any correlated family. Signals with disjoint observation
sets go to separate clusters and compose at full weight.

This reuses the engine's existing correlation control rather than adding a second
one. No new mathematics; a routing rule.

### 4.5 The dependency DAG — no back-edges

```
supplier_identity ──┬──> item_equivalence ──┐
                    │                       ├──> contract_coverage
                    └──> contract_succession┘
```

`supplier_identity` **must not** read `UNDER_CONTRACT` or `OF_ITEM`.
`item_equivalence` **must not** read `UNDER_CONTRACT`.
Enforced by a declared `reads` allow-list per profile, asserted at registration; a
profile naming a downstream edge fails to register. A cycle here would be evidence
laundering, and it would not be visible in any single score.

---

## 5. The four profiles

### 5.1 `supplier_identity` → `SAME_ENTITY`

Resolves supplier records into equivalence classes.

> **CORRECTION, measured 2026-09-10.** This section originally claimed the profile
> resolves `SUP-*` (transactions), `S####` (contracts) and `SI######` (crosswalk)
> into one set of classes. **That is not achievable, and the claim was wrong.**
>
> The `S####` contract keyspace exists in exactly two places — `proc.bp_contract_master`
> and its FDW mirror `canonical.bp_contracts` — holding **2,527 distinct supplier ids
> and no attributes whatsoever**: no name, no VAT, no registration number, no address,
> no bank detail. No row anywhere in either schema describes a contract supplier. Every
> signal below is therefore `MISSING` for one, so **no `SAME_ENTITY` edge can ever reach
> a contract supplier.**
>
> What the profile *can* do is resolve transaction-side suppliers to each other:
> `bp_supplier_master` carries the attributes and bridges to the graph's `SUP-*` nodes
> through `bp_supplier_id_crosswalk` (`uicanvas_supplier_id` → `bp_supplier_id`,
> resolving 1,009/1,009).
>
> **Consequence for §5.3, and it is a safety requirement, not a caveat:** supplier
> identity is a **precondition** for contract coverage, not one signal among several.
> With `supplier_same` `MISSING`, coverage would rest on `date_in_term` and
> `amount_within_value` alone — which would report an invoice as covered by a contract
> belonging to an entirely unrelated supplier whenever the dates overlap and the value
> fits. That is a false finding, not a weak one. **When the identity cluster is entirely
> `MISSING`, `contract_coverage` must not report coverage at all.**

| Signal | Cluster | Tier | Notes |
|---|---|---|---|
| `name_exact` | identity | 1 | normalised legal name |
| `name_fuzzy` | identity | 2 | `cmp_supplier`'s shared-root rule (0.7/WEAK) already exists |
| `vat_number` | registration | 1 | `bp_supplier_master`, 1,009/1,009 populated |
| `registration_number` | registration | 1 | ditto |
| `duns_number` | registration | 2 | ditto |
| `address` | context | 3 | postal code + country |
| `bank_account` | financial | 1 | strong but sensitive — see §8 |

Registration signals share a cluster because a company that matches on VAT usually
matches on registration number; dampening prevents treating that as three
independent confirmations.

### 5.2 `item_equivalence` → `Item` + `OF_ITEM`

Connects 193,857 line nodes through shared products. Largest information gain in the
design: it turns price comparison from string-matching into a graph question.

| Signal | Cluster | Tier |
|---|---|---|
| `item_id_exact` | reference | 1 |
| `description_overlap` | description | 2 |
| `uom_compatible` | description | 3 |
| `price_proximity` | commercial | 3 |
| `supplier_same` (derived, reads `SAME_ENTITY`) | identity | 2 |

`item_id` is present on 55,421 of 55,483 invoice lines, so the reference signal
carries most cases; the rest is for descriptive drift across suppliers.

**Canonicalisation:** an `Item` node is minted per resolved equivalence class. UoM is
recorded, never converted — a pack-of-10 and 10 each are related, not equal, and
`bp_uom_canonical` (38 rows) is the only UoM authority.

### 5.3 `contract_coverage` → `UNDER_CONTRACT`

The headline capability. Contract side is ready: **958 Active contracts, 957 with a
full term window, 892 distinct suppliers, 958 with a spend category.**

| Signal | Cluster | Tier | Notes |
|---|---|---|---|
| `contract_ref` | reference | 1 | a contract number on the document — 0/38,498 today, non-zero once extraction populates it |
| `supplier_same` (derived) | identity | 1 | reads `SAME_ENTITY` |
| `date_in_term` | temporal | 1 | `contract_start_date ≤ doc_date ≤ contract_end_date` |
| `category_match` | category | 3 | `MISSING` until the category dimension exists |
| `amount_within_value` | commercial | 2 | against `total_contract_value` |
| `item_under_contract` (derived) | line | 2 | reads `OF_ITEM`; lines resolving to Items already under this contract |

**A document outside every active term window for a matched supplier is off-contract
spend** — and the per-signal breakdown says which evidence was absent, so the finding
can be defended rather than merely asserted.

**`supplier_same` is a precondition, not a contributor.** If the identity cluster is
entirely `MISSING` — no `SAME_ENTITY` edge and no contract reference on the document —
`contract_coverage` reports nothing, whatever `date_in_term` and `amount_within_value`
say. Dates and amounts alone would match an invoice to an unrelated supplier's contract.
On the current corpus this means contract coverage yields no findings at all, because
contract suppliers carry no attributes to resolve against (see the correction in §5.1).
That is the correct outcome: silence, rather than confident nonsense.

### 5.4 `contract_succession` → `SUCCEEDS`

Reconstructs the renewal chain that the ids cannot express: `parent_contract_id` is
populated on 1,561 contracts and **resolves on 0** (`C1543` against actual `C00002`).

| Signal | Cluster | Tier |
|---|---|---|
| `supplier_same` (derived) | identity | 1 |
| `term_adjacency` | temporal | 1 |
| `category_match` | category | 3 |
| `value_proximity` | commercial | 2 |
| `title_similarity` | description | 2 |

Renewal uplift is then `total_contract_value` across a resolved `SUCCEEDS` pair, in a
single currency, never across one. Where currencies differ the uplift is reported as
unavailable rather than converted — the FX honesty rule (`converted_amount_usd` is
populated on 10 of 12,408 invoices).

---

## 6. Contract source decision

`procurement_kg_builder.ENTITY_TABLE_MAP` reads `proc.bp_contracts` — 0 rows.
`bp_contract_master` has 3,051.

**Decision: both, with different roles.**

- `proc.bp_contracts` stays the **extraction destination** (`contract.yaml` already
  targets it) and becomes the graph's source for contracts the pipeline produced.
- `bp_contract_master` is loaded as **reference contracts**, marked
  `origin='reference'` on the node.

Rationale: they are different things and collapsing them would lose that. Extracted
contracts have provenance; the 3,051 seeded rows do not. Both are legitimate graph
citizens; a finding can then state which kind it rests on.

`contract.yaml` declares `db_lines_table: null` — **contracts have no line-items
table**, so no contracted unit price can be extracted. Contract coverage is therefore
a *scope* statement, not a *price* statement. Price-vs-contract variance stays
impossible and is not claimed anywhere in this design.

**`kg_sync._TRGT_TABLE` must gain a contract entry**, or contracts will never reach
the graph incrementally even once they exist.

---

## 7. Data flow

### 7.1 Batch — the resolution pass

Ordered by the DAG (§4.5), run by `backend_scheduler`:

```
1. supplier_identity   → score pairs → MILP → SAME_ENTITY edges
2. item_equivalence    → score pairs → MILP → Item nodes + OF_ITEM edges
3. contract_succession → score pairs → MILP → SUCCEEDS edges
4. contract_coverage   → score pairs → MILP → UNDER_CONTRACT edges
```

Each stage writes its edges before the next reads them. A stage that fails leaves
prior stages' edges intact; edges are `MERGE`d and idempotent.

### 7.2 Where the MILP earns its place

Pairwise scoring will happily assert A≈B, B≈C, A≠C. A supplier cannot be two
canonical entities at once; an invoice cannot be under two contracts at once. Both
are assignment problems:

- **`supplier_identity`** — each alias assigns to at most one canonical entity.
- **`contract_coverage`** — each document assigns to at most one contract, and
  `consumes` draws down `total_contract_value`, so a contract cannot silently cover
  more spend than it is worth. This is the `consumes` mechanism used as designed.

`DEGENERATE` is a first-class outcome: a near-tie identity is recorded with its
margin and **does not** get `auto_link`, regardless of `F`.

### 7.3 Incremental

`kg_sync.sync_row_to_kg` continues to mirror documents per promotion. Derived edges
are **not** recomputed per row — a new document is queued for the next resolution
pass. Rationale: composition is a property of the population, and rescoring one
document against the whole corpus per promotion is not a cheap read-model operation
(the same reasoning `value_summary_service.py:258-269` already applies to benchmark).

---

## 8. Error handling and safety

- **Neo4j unavailable.** `kg_sync` already never raises; the resolution pass adopts
  the same contract. `_trgt` remains source of truth for documents; the graph is
  rebuildable. A failed pass logs and leaves the previous edge set in place.
- **Governance reads are fail-open at every layer** — an outage looks exactly like
  "nothing governs this". The resolution pass must distinguish *no policy* from
  *policy unavailable*, and refuse to promote edges to `auto_link` in the latter case.
- **Bank account as an identity signal** (§5.1) is a fraud-adjacent field. It is
  Tier 1 evidence and must never appear in an edge's `signals` JSON in cleartext —
  store a digest. `query_engine` already excludes banking detail from the default
  supplier projection; that exclusion must hold here.
- **Never rewrite a document's literal value.** Resolution records that two records
  refer to one entity; it does not overwrite what the document said.
- **Circular evidence** is prevented structurally by the `reads` allow-list (§4.5),
  which fails at registration rather than at runtime.

---

## 9. Calibration — floors must be measured, not borrowed

The bands are fixed and inherited. What must be measured per profile is `alpha`,
`p0`, `floor` and the signal weights.

**Honest constraint:** the corpus cannot calibrate all four profiles. It has 0
Contract nodes, 1 deal with competing quotes, and a category dimension that resolves
nothing. Therefore:

- `supplier_identity` **can** be calibrated: 1,009 supplier-master rows with 100%
  populated VAT, registration and DUNS give a labelled sample by construction.
- `item_equivalence` **can** be calibrated against `item_id` as ground truth on the
  55,421 lines that carry it, holding it out and scoring on the other signals.
- `contract_coverage` and `contract_succession` **cannot** be calibrated until
  contracts are in the graph. Their parameters ship **declared unmeasured**, their
  edges capped at `review` — never `auto_link` — until a labelled sample exists.

A capped profile is stated in the spec, in the code, and on the edge. It is not a
borrowed 0.85 wearing a confident face.

---

## 10. Testing

1. **Golden vectors at import** for every registered formula, per §2.5. Vectors are
   snapshotted from the live implementation and are *today's numbers, not the numbers
   anyone believes the code produces*.
2. **Mutation testing is mandatory.** Each guard is broken on purpose and must go
   red: prior double-counting (§4.2), the `reads` allow-list (§4.5), cluster
   assignment for intersecting observation sets (§4.4), the `DEGENERATE` cap (§7.2),
   the unmeasured-profile cap (§9), bank-digest redaction (§8). A guard that stays
   green when broken is checking nothing.
3. **Independence property test:** compose N supporting edges derived from one shared
   observation; assert the composed `F` does not exceed the `F` of composing them as
   one cluster. This is the double-counting regression test.
4. **Live verification** against the running local server and the live database,
   reported with real figures — not tests alone.
5. **`tests/api/test_agent_workflows_router.py`** has 11 pre-existing failures
   unrelated to this work; deselect rather than "fix" them here.

---

## 11. Deliberately not built

- Customer / buying-entity model — separate build.
- Category dimension — signals score `MISSING` until it exists.
- Contracted unit prices — no contract line table exists (§6).
- External benchmark reference data.
- Any new confidence scheme, threshold or band.

---

## 12. Open question for the plan

`supplier_identity` will produce a `review` band population. `bp_supplier_review`
already holds **727 rows, all `pending`, none ever reviewed**. Adding to a queue
nobody works is not a resolution strategy. The plan must either route the review band
somewhere a person actually goes, or state plainly that `review` edges accumulate
unactioned — and size that population before shipping.
