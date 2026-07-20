# Deal Clustering & Confirmation — Design

**Date:** 2026-07-20
**Status:** Approved, ready for implementation plan
**Supersedes:** nothing. Extends `2026-06-06-linking-engine-and-promotion-design.md`
(the relationship scorer) and the deal-linking layer of 2026-06-11.

## Problem

An upload batch becomes exactly one deal, regardless of what is in it.

Upload `Analysis Set_190726` (2026-07-19) put **40 documents from 12 suppliers across 4
unrelated categories into a single `deal_id`** — `ANALYSISSET_19072620260719339`. The
batch contains freight lane pricing (`London (Heathrow) → Edinburgh - FTL`) and managed
IT services (`Service Desk (24x7, 3,000 users) — annual`) in the same "deal".

Observed state of that batch:

| | Count |
|---|---|
| Documents | 40 (28 quote, 5 PO, 7 invoice) |
| Distinct suppliers in quotes | 12 |
| Purchase orders | 5, each a **different** supplier |
| Quotes carrying a `po_id` | **0 / 28** |
| PO lines carrying a `quote_number` | **0 / 32** |
| Invoices carrying a `po_id` | 7 / 7 |

Consequences that reach the screen: `bp_deal_overview` aggregates all 40 documents into
one row, so `max(supplier_name)` picks an arbitrary supplier, `price_variance_pct`
compares unrelated totals (the UI renders `+1578.5%` bid spread), and `three_way_match`
reports `true` because it only tests *"this deal contains ≥1 quote AND ≥1 PO AND ≥1
invoice"* — trivially satisfied by a 40-document bucket and therefore meaningless.

## Root cause

The clustering intelligence **already exists and already runs**. It is gated behind a
condition the upload path guarantees will never be true.

`deal_assignment_service._run()` orders its passes:

```python
fwd = _look_forward(cur)    # stamps the user-typed batch name onto every document
back = _look_back(cur)      # then: "if (cur_deal or '').strip(): continue"
```

`_look_forward` reads `proc.process_monitor.deal_id` — which the Node gateway populated
for every file in the batch from the deal *name* the user typed
(`data-integration.service.ts:70-91` `generateDealId()`: `prefix + dateStr + sequence`).
`_look_back`, the pass that forms one deal per canonical PO using relationship scoring,
then finds every document already linked and skips all of them
(`deal_assignment_service.py:423-424`).

Two documentation defects found while tracing this, both worth correcting:

- `linking_engine.py:11-13` states `deal_id` "is assigned by a separate SQL trigger".
  **No such trigger or stored procedure exists.** `deploy/sql/2026-06-11_deal_linking.sql`
  is purely additive DDL. `deal_id` is owned by TypeScript in the gateway.
- `generateDealId()` builds its sequence with `String(count + 1).padStart(2, '0')` over
  the whole `analyse` table, which silently overflows past 99 (this batch got `339`).

## What a deal is

**A deal is a sourcing event**: one requirement, competitively bid by several suppliers,
optionally awarded via a PO and settled by invoices.

The uploaded batch decomposes cleanly under this definition:

| Sourcing event | Competing suppliers | Awarded PO |
|---|---|---|
| Managed IT / MSA | PrimeOps, Fortis, Synapse | PO-2024-0128 Synapse |
| Platform licence (3-yr) | ClearPath, NexusFlow, Orbis | PO-2024-0145 Orbis |
| Professional services | Meridian Consulting, Apex, Vantage | PO-2024-0114 Meridian |
| Freight | Swift, Condor, Meridian Freight | PO-2024-0091 Swift |
| Building works | *(none)* | PO-2024-0163 Caldwell |

Rejected alternatives: **one deal per PO** (leaves quote-only bidding groups orphaned and
cannot represent a sourcing event before award); **one deal per supplier** (destroys the
competitive comparison, which is the point of the Analyse screen).

### Quote versions are not separate bids

28 quotes collapse to **12 distinct base references**; 20 carry a version suffix:

```
CPS-Q-3380 · CPS-Q-3380 (V2) · CPS-Q-3380 (V3 (BAFO))   -> ONE bid, three rounds
```

Counting these as three competing bids inflates supplier counts and corrupts bid-spread
maths. Version collapse is a prerequisite for correct clustering, not a nicety.

## Approach

Add a clustering pass that **proposes** groupings, and a UI to **confirm** them.

### Two relations, not one

The existing engine models exactly one relation and it is **not** the one that groups a
sourcing event.

| Relation | Example | Supplier | Unit price | Modelled today |
|---|---|---|---|---|
| **Continuity** | Condor quote → Condor PO → Condor invoice | same | same | Yes (`quote_po`, `invoice_po`) |
| **Rivalry** | Condor bid ↔ Meridian Freight bid ↔ Swift bid | **different** | **different** | **No** |

`PROFILES` (`linking_engine.py:263-266`) contains only `invoice_po` and `quote_po`. There
is no quote↔quote profile. Worse, applying the existing one to rival bids actively
rejects them — `cmp_supplier` returns `(0.0, 'CONFLICT')` for two different suppliers on
the **highest-weighted signal in the model** (`supplier_id`, weight 5, tier 1), and
`amount` (weight 3) conflicts too because rival bids differ in total by design.

The engine is a *same-commercial-thread* detector. For a competitive event, "different
supplier, different price, same products, same quantities" is the defining signature —
so a second profile is required, in which supplier and price divergence are **expected
rather than penalised**.

### Precedence: declared linkage outranks inferred linkage

Correlation is only ever used to fill a gap a human has not already closed. In order:

1. **Human-declared connection.** If the uploader explicitly said these documents belong
   together — amend-mode targeting an existing deal, an explicit grouping at upload, or a
   previously *confirmed* proposal — that grouping is authoritative. Clustering must not
   re-group, split or second-guess it. Inference never overrides a human statement.
2. **Linked identifiers.** Explicit references carried in the documents: `po_id`
   (invoice→PO, 7/7 populated here) and `quote_number` on PO lines. When present these
   are decisive and no similarity scoring is needed.
3. **Inferred correlation.** Only for what remains. On this batch that is nearly
   everything — 0/28 quotes carry a `po_id`, 0/32 PO lines carry a `quote_number`.

### Correlation signals (new)

**No signal is a gate.** Each contributes evidence; confidence accumulates across them.
An earlier draft of this spec hard-capped price ratio at 3.0× and split a deal above it —
that was wrong, because it lets one imperfect measurement veto every other agreeing
signal. Signals accumulate; only the *total* decides, and it decides a routing band, not
a truth.

| Tier | Signal | Why it links rivals | Availability on this batch |
|---|---|---|---|
| 1 | `po_id`, `quote_number` | Explicit reference — decisive | **0/28**, 0/32 |
| 1 | Human-declared grouping | A person said so | none (folders are by doc type) |
| 2 | Product description overlap | Same requirement | 28/28 |
| 2 | Volume / quantity | Same requirement | 28/28 |
| 2 | PO line-set anchor | A quote whose lines match a PO's lines is a bid **for that requirement, whoever sent it** — links rivals *through* the award | 5 POs |
| 3 | Price proximity | Rivals cluster at the market rate | 28/28 |
| 4 | `buyer_id` | Same buying entity | 28/28 (all `Assurity Ltd`) |
| 4 | Currency | Same commercial basis | 28/28 (all GBP) |
| 4 | Country / region | Same delivery scope | partial |
| 4 | Bidding-round synchrony | Rivals answer one deadline | strong on 2 of 4 events |
| 4 | Validity-date window | Same tender timetable | 28/28 |

Tier 4 signals are **corroboration, never discrimination**. `buyer_id` is `Assurity Ltd`
and currency is GBP across all 28 quotes here, so they cannot separate the four events —
they can only raise or lower confidence in a grouping proposed on tier 2/3 evidence, and
a mismatch on them is meaningful even though a match is not.

Bidding-round synchrony illustrates why nothing may be a gate. Measured span between
rivals' submissions in the same round:

| Event | V1 | V2 | V3 |
|---|---|---|---|
| Freight | 1 day | 2 days | 1 day |
| IT-MSA | 1 day | 1 day | — |
| Consultancy | **39 days** | 5 days | 1 day |
| Platform | **28 days** | 1 day | — |

Excellent on two events, useless on two. As a gate it would destroy half the batch; as a
contributing signal it is genuinely informative where it holds.

**Pricing is compared by proximity, not equality.** Exact equality is the signature of
*continuity* (a supplier's own quote → PO). Rivals sit in a band around the market rate:

| Requirement | Bidders | Price range | Spread |
|---|---|---|---|
| Freight — Edinburgh lane | 9 | £872.34 – £945.00 | **1.08×** |
| IT — endpoint management | 5 | £272,000 – £291,000 | **1.07×** |
| Consultancy — Business Analyst | 8 | £680 – £780 | **1.15×** |

Within a requirement 1.07–1.15×; across requirements £872 vs £272,000 — **312×**. Ratio
maps to a graded score with **no cutoff**; a wide spread merely contributes little.

Measured pair scores on tier 2+3 alone:

| Pair | Existing `_line_pair_score` | Correlation score |
|---|---|---|
| Condor ↔ Meridian Freight (rival) | 0.75 | **1.000** |
| Condor ↔ Swift (rival, null supplier, description drift) | 0.667 | **0.933** |
| Freight ↔ consultancy (prices coincidentally within 1.37×) | — | **0.280** |
| Freight ↔ IT services | — | **0.000** |

The consultancy row is why several signals are needed: on price alone it looks related;
description and volume outvote the coincidence.

### Award exclusivity — the discriminator that separates rivalry from repeat buying

Product, volume and price correlation is **not sufficient** to establish rivalry. Tested
across the rest of the product, it produced two confident groupings that are wrong:

| Score | Quote A | Quote B |
|---|---|---|
| 0.744 | 128234 `SUP-DuncanLlc` | 136586 `SUP-PerryLtd` |
| 0.703 | 102494 `SUP-DixonReynoldsAndSolomon` | 104683 `SUP-GomezGoodAndCross` |

The line items look exactly like competing bids — same buyer (`Assurity Ltd`), same
products, identical quantities, prices a few percent apart:

```
128234  Staedtler Ballpoint Pen Black Ink, 10 per Pack  qty 100  £13.85
136586  Staedtler Ballpoint Pen                         qty 100  £12.49
128234  Faber-Castell A4 Ruled Notebook, White Cover    qty 100  £11.69
136586  Faber-Castell A4 Ruled                          qty 100  £10.10
```

They are not bids. **Each supplier has its own PO and its own invoice** — four suppliers,
four POs, four invoice chains. These are repeat commodity purchases of the same catalogue
items from different suppliers at different times, which are legitimately separate deals.

The structural difference is decisive:

| | Suppliers | POs |
|---|---|---|
| Rivalry (freight event) | 3 — Swift, Condor, Meridian Freight | **1** — only the winner |
| Repeat buying (Duncan/Perry/Dixon/Gomez) | 4 | **4** — one each |

**A competition has one award.** Two correlated quotes that *each* anchor their own PO
with their own invoices are not rivals — they are separate transactions, however alike
their contents. This is a structural fact about the documents, not a similarity measure,
so it acts as a **veto on rivalry** rather than as another graded signal: correlation
proposes the group, award structure can rule it out.

Stated precisely, two bids are **not** rivals when each anchors a distinct PO whose
invoices settle separately. A supplier winning one event and losing another is unaffected
— the test is pairwise over the specific quotes, not over the supplier's whole history.

### Universal rules

These hold for every quote in the product, awarded or not. Verified over all 45 quotes,
not only the analysis batch.

**R1 — Versions link regardless of award.** A supplier's proposal rounds group by base
reference whether or not that supplier won. Measured: 10 multi-version bids linked, **8
of them not awarded** (Condor 3 rounds, ClearPath 3, Apex 3, Meridian Freight 3, Swift 3
with a null supplier, NexusFlow 2, PrimeOps 2, Vantage 2). Award plays no part in version
linking and must never gate it.

**R2 — Losing bids still join their sourcing event.** With award exclusivity applied, all
four events still form with three bidders each, and 12 of 18 non-awarded bids are linked.
A loser is a full member of the deal — that is the whole point of capturing a competition.

**R3 — Same supplier is never rivalry.** Two quotes from one supplier are versions,
duplicates or unrelated purchases; they can never be competing bids. This is definitional
and must be enforced *before* correlation, not left to the score. Measured failure without
it: `WSG100024` and `WSG100025`, both `SUP-DellWorkspaceSolutionsLtd`, same date,
identical total £111,975.00, were grouped as "2 bidders". Version collapse on a `(V<n>)`
suffix alone does not catch sequentially-numbered quotes from one supplier.

**R4 — No connection found routes to a human.** Any bid that ends in a group of one goes
to review carrying the reason and its best rejected match — never silently dropped, never
force-fitted. Measured: 13 unconnected bids, each with a distinct reason:

```
128234    best_match 0.744 (136586)   -> correlated but award-vetoed
10253     best_match 0.680 (104680)   -> no correlated bid above threshold
DHA-2025-102  best_match 0.312        -> no correlated bid above threshold
```

The award-vetoed cases are the important ones: the veto **suppresses an automatic
grouping, it does not decide the documents are unrelated**. That judgement goes to a
person, which is what stops the veto becoming a silent splitter.

### Award detection must not rely on supplier-name matching

The veto is only as good as its notion of "awarded", and string-matching a PO's
`supplier_name` to a quote's `supplier_id` is not good enough. Measured failure:

```
quote  SUP-GomezGoodAndCross              -> "gomezgoodandcross"
PO     Gomez, Good and Cross Trading Ltd  -> "gomezgoodandcrosstradingltd"   MISMATCH
```

Gomez therefore read as un-awarded, the veto did not fire, and Dixon+Gomez merged — the
precise false merge the veto exists to prevent. The same weakness already loses Swift,
whose `supplier_id` is null.

**Award is established by continuity scoring, not by name comparison.** A PO is the award
for the bid whose lines match it under the existing `quote_po` profile — same supplier
*and* same price, which is what that profile is built to detect and where an exact
unit-price match is correct. Supplier name may corroborate; it may never be the test.
Where no award can be established with confidence, the pair goes to review under R4
rather than being silently merged or silently split.

### Validation: measured against the real batch

The approach was run read-only over the 28 quotes before being specified, with ground
truth taken from the corpus' `_WINNER` filename annotations and the 4 POs.

**Linkage strategy is load-bearing and was the one real failure found.** Single-linkage
clustering merged IT-MSA and Platform into one six-supplier blob: a single borderline pair
(Fortis↔NexusFlow, 0.626) was enough to chain two distinct events together. Both are
3-year annual IT contracts sharing tokens like *seats*, *licence*, *support*, *annual*.

**Complete linkage** — merge two groups only when *every* cross-pair clears the threshold
— fixes it. A single borderline pair can no longer drag two events together.

| Cluster | Outcome | Confidence |
|---|---|---|
| IT-MSA (Synapse, PrimeOps, Fortis) | exact | 99.7 |
| Freight (Swift, Condor, Meridian Freight) | exact | 93.8 |
| Consultancy (Meridian Consulting, Apex, Vantage) | exact | 84.8 |
| Platform (Orbis, ClearPath, NexusFlow) | exact | 75.6 |

4/4 events recovered, 12/12 bids correctly placed, zero cross-cluster leakage. Stable at
thresholds 0.65 and 0.70; **0.70** is specified as the midpoint of the separation gap.

Separation margin on this batch:

| | Score |
|---|---|
| Weakest **within**-cluster pair (ClearPath↔Orbis) | 0.756 |
| Strongest **cross**-cluster pair (Fortis↔NexusFlow) | 0.626 |
| **Margin** | **0.130** |

Notable: the Freight event recovers exactly **despite `SDP-Q-44120` having a null
supplier_id**, confirming that product/price/volume correlation carries a bid whose
supplier extraction failed.

**Honest limits of this validation.** One batch, four events, twelve bids. The 0.70
threshold sits in a gap measured *on this same batch*, so it is calibrated, not proven —
a second batch could place a real event below it or a false pair above it. The margin of
0.130 is narrow. Treat 0.70 as a starting value to be re-measured as batches accumulate,
which is precisely why nothing auto-applies and why the band below it routes to a human
rather than to a decision.

### Consistency across the rest of the product

Measured over every other deal, not only the analysis batch.

**Line-item coverage is universal** — 100% of quotes (avg 4.1 lines), POs (4.1) and
invoices (2.4) outside the analysis batch carry line items. Correlation has data to work
with product-wide, not just on this corpus.

**The analysis batch is the only competitive data that exists.** Every other deal is a
single supplier: 1 quote, 1 PO, 1–3 invoices. So the rivalry path has exactly one
worked example to calibrate against, and the 0.70 threshold cannot yet be validated on a
second competitive batch — there isn't one. This is a data limitation, not a design
choice, and it is the single biggest risk to this feature.

**The other deals are the negative control, and they caught the award-exclusivity gap
above.** With that veto applied, correlation-plus-structure produces no false merges
across the 91 scored cross-deal pairs.

**42 documents carry no `deal_id` at all** — 3 quotes, 20 POs, 19 invoices, roughly half
the POs and invoices in the system. They predate or fell outside batch assignment and
have no `bp_deal` row. Clustering as specified operates per upload batch and would never
see them. Bringing them in is deliberately **out of scope here** — they need their own
pass, and grouping them shares this design's machinery but not its entry point. Recorded
so it is a known gap rather than a silent one.

### Confidence from signal agreement

Accumulation reuses the machinery `linking_engine.score_link` already implements — per
signal `weight`/`tier`/`cap`, cluster subtotals, and a log-odds combination
(`linking_engine.py:344-370`) — with a new `quote_rival` profile supplying the signal set
above. Nothing new is invented for the maths; only the signal set and their treatment of
supplier/price divergence change.

Confidence routes a proposal to the existing bands (`_BAND_AUTO` 92, `_BAND_WARN` 80,
`_BAND_REVIEW` 65, `_BAND_WEAK` 45):

| Band | Meaning | Behaviour | On this batch |
|---|---|---|---|
| ≥ 92 | Many signals agree | Proposed, pre-selected | IT-MSA 99.7, Freight 93.8 |
| 80–92 | Solid, some signals absent | Proposed | Consultancy 84.8 |
| 65–80 | **Requirement matches, corroboration thin** | **Human review required** | **Platform 75.6** |
| 45–65 | Weak | Suggestion only, not pre-grouped | — |
| < 45 | No case | Left ungrouped | — |

Every proposal stores **which signals passed, which failed and which were unavailable**
in `match_evidence`, so the confirmation screen can say *why* — "same products, same
volumes, prices within 8%, same buyer; no shared PO reference" — rather than showing a
bare number. Nothing auto-applies at any band; confidence orders and explains the
proposals a human confirms.

### Human-in-the-loop

Review is **targeted, not blanket**. Every proposal is confirmable by a human, but only
those below 80 actively demand attention — on this batch, 1 of 4 groups. A design that
asked for review of all four would be ignored within a week; one that asked for none
would silently mis-group.

**What triggers review**

| Trigger | Rationale | This batch |
|---|---|---|
| Confidence < 80 | Correlation holds but corroboration is thin | Platform (75.6) |
| Any pair within 0.05 of the threshold | Sat near the decision boundary | ClearPath↔Orbis |
| A member with a null/unresolved supplier | Identity unverified | `SDP-Q-44120` |
| Line items missing on either side | Scored on partial evidence (`MISSING`) | — |
| A PO with no anchoring bid | Award with no visible competition | PO-2024-0163 Caldwell |
| Cross-cluster pair within 0.05 below threshold | A near-merge that was rejected | Fortis↔NexusFlow (0.626) |

**What the reviewer is shown** — the proposed group, its members, the per-signal evidence
for each pair (which passed, failed, were unavailable), the specific reason review was
triggered, and the nearest rejected alternative. For Platform that means surfacing that
Fortis↔NexusFlow scored 0.626 and was *not* merged, so the reviewer can see the call that
was made rather than only the outcome.

**What the reviewer can do** — confirm as proposed; move a document to another proposed
deal; split a proposal; merge two proposals; or leave it unresolved. Unresolved is a
first-class outcome: documents stay ungrouped rather than being forced into a deal.

**The decision is recorded and is authoritative.** A confirmed grouping becomes declared
linkage (precedence tier 1) and is never re-clustered — including a *rejection*, so the
same wrong pairing is not re-proposed on the next run. Every decision writes to
`bp_agent_actions` for audit, carrying the confidence and the evidence that was shown, so
a later reader can see what the human was looking at when they decided.

**Feeding calibration.** Confirmations and corrections are the only honest source of
threshold calibration. Recording where a human overrode the algorithm — a merge it missed
or a split it should have made — is what moves 0.70 off a single-batch guess. This is
observation only; no automatic retuning, since a threshold that drifts on its own would
silently change how deals form.

Continuity keeps using the existing `quote_po` / `invoice_po` profiles unchanged — that
maths is correct for what it does and is not touched.

### Grouping algorithm

Applied per upload batch:

0. **Honour declared linkage first.** Documents the uploader explicitly connected, or
   that belong to an already-confirmed deal, are set aside as fixed. They are reported in
   the proposal as declared (not inferred) and are never re-clustered. Everything below
   operates only on the remainder.
1. **Collapse quote versions.** Normalise `quote_id` by stripping a trailing
   `(V<n>...)` suffix. Members of one base reference are rounds of one bid; the highest
   version is the current offer. Two quotes from the *same* supplier are versions;
   two from *different* suppliers are rivals. 28 quotes → 12 bids.
2. **Cluster bids into requirement groups** by **complete linkage** at threshold 0.70 —
   two groups merge only when *every* cross-pair clears the bar. Single linkage was
   measured and rejected: it chained IT-MSA and Platform into one six-supplier blob on a
   single 0.626 pair. Each
   cluster is one sourcing event, holding **N rival suppliers' bids** — this is the step
   that makes a deal multi-supplier. A cluster of one is a single-bid event, not an error.
3. **Attach each PO to the requirement group it was awarded from**, scoring the PO
   against the group's bids with the existing `quote_po` profile (correct here: the
   awarded PO shares supplier *and* price with the winning bid — Condor V3 vs its PO
   scores 1.0 on the existing comparator).
4. **Attach invoices to POs** via the existing `po_id` (7/7 populated — deterministic,
   no scoring needed).
5. **Leave genuine orphans orphaned.** PO-2024-0163 (Caldwell) has no quotes in the
   batch. It must surface as awaiting-quote, never be forced into a neighbouring group.

The ordering matters: requirement clustering runs **before** PO attachment, so a sourcing
event exists in its own right and does not depend on an award having happened. A
quote-only event (no PO yet) is a first-class deal.

Supplier identity must not gate grouping. Normalised supplier-key matching resolves only
**3 of 5** POs on this batch; Swift (PO-2024-0091) fails because `SDP-Q-44120` has a
**null `supplier_id`** (3 of 28 quotes and 3 of 7 invoices are missing it). Note
`cmp_supplier(None, x)` returns `(0.5, 'MISSING')` — neutral, so a null supplier neither
helps nor blocks. Product-level matching is what recovers that group.

### Proposals are never auto-applied

Clustering writes to new tables and changes nothing that exists:

```sql
CREATE TABLE proc.bp_deal_proposal (
    proposal_id     BIGSERIAL PRIMARY KEY,
    batch_deal_id   VARCHAR NOT NULL,   -- the upload batch it came from
    session_id      TEXT,
    proposed_name   VARCHAR,
    confidence      NUMERIC(5,2),       -- min pairwise F across members
    status          VARCHAR NOT NULL DEFAULT 'proposed',
                    -- proposed | confirmed | rejected | superseded
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at    TIMESTAMPTZ,
    confirmed_by    VARCHAR
);

CREATE TABLE proc.bp_deal_proposal_member (
    proposal_id     BIGINT NOT NULL REFERENCES proc.bp_deal_proposal(proposal_id),
    doc_type        VARCHAR NOT NULL,   -- quote | po | invoice
    doc_pk          VARCHAR NOT NULL,
    base_reference  VARCHAR,            -- version-collapsed quote ref
    role            VARCHAR,            -- anchor_quote | competing_quote | po | invoice
    match_score     NUMERIC(5,2),
    match_evidence  JSONB,              -- score_link per-signal breakdown
    PRIMARY KEY (proposal_id, doc_type, doc_pk)
);
```

Table names follow the `bp_` prefix convention; indexes as `ix_bp_deal_proposal_*`.

`match_evidence` stores `score_link`'s per-signal output so the UI can show *why* two
documents were grouped, rather than an unexplained number. There are **no product codes
anywhere in the schema** — no SKU, part number or item code in any extraction schema — so
matching is fuzzy text over `item_description`. That is exactly why a human confirmation
step is required and why evidence must be visible.

### Confirmation assigns identity

`deal_id` is minted **at confirm time**, never at proposal time. This is deliberate:
`mint_document_id` embeds the deal (`f"{deal_id}::{doc_type}::{doc_pk}"`), so a document's
identity depends on its grouping. Deciding the group before minting avoids a primary-key
migration entirely for new uploads.

On confirm: mint `deal_id`, write `deal_id`/`deal_name`/`document_id` to `_stg`/`_trgt`
and line-item tables via the existing `_persist_deal`, upsert `bp_deal_document_map`, and
insert the `proc.bp_deal` row with `is_tracked = false` (drafts stay drafts; promotion
remains the separate existing gate).

### Upload path change

`_look_forward` must stop treating a batch name as an authoritative deal. The batch
identifier is retained on `process_monitor` as a **batch label** for traceability, but is
no longer stamped onto documents as their deal. Documents arriving from an upload batch
enter clustering unlinked, which is the condition `_look_back` and the new pass require.

Existing non-batch behaviour of `_look_forward` (amending an existing, already-confirmed
deal) is preserved: when the user explicitly targets a known deal, that assignment stays
authoritative and clustering does not run.

## Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `requirement_similarity.py` | `quote_rival` profile: the tier 1-4 signal set, graded not gated, accumulated by the existing log-odds combiner into a confidence + per-signal evidence. | `linking_engine.score_link` internals |
| `declared_linkage.py` | Identify human-declared connections that clustering must not touch | `process_monitor`, confirmed proposals |
| `deal_clustering.py` | Batch → proposed clusters. Pure; no writes. | rivalry + `linking_engine.score_link` |
| `version_collapse.py` | `quote_id` → base reference + round | none |
| `proposal_store.py` | Persist/read/confirm proposals | the two tables above |
| `POST /deals/proposals/generate` | Cluster a batch, store proposals | clustering + store |
| `GET /deals/proposals?batch=` | Proposals + members + evidence | store |
| `POST /deals/proposals/{id}/confirm` | Mint deal, assign documents | store + `_persist_deal` |
| `PATCH /deals/proposals/{id}/members` | Move/remove a document pre-confirm | store |
| Review screen (SpendIQ) | Show, adjust, confirm groupings | the endpoints |

`deal_clustering.py` is kept pure and write-free so it can be tested on fixtures without
a database, and so a bad clustering run can never corrupt assigned deals.

## Data flow

```
upload batch
   -> declared linkage set aside   human-stated connections are FIXED, never re-grouped
   -> linked identifiers           po_id / quote_number: decisive where present
   -> version collapse          28 quotes -> 12 bids   (same supplier = rounds)
   -> requirement clustering    12 bids -> 4 events    (RIVALRY: different suppliers,
                                                        correlated on description +
                                                        pricing + volume)
   -> PO attachment             award joins its event  (CONTINUITY: quote_po profile)
   -> invoice attachment        via po_id              (deterministic, 7/7)
   -> bp_deal_proposal[_member]          <-- nothing else written
   -> [ USER REVIEWS AND CONFIRMS ]
   -> mint deal_id -> _stg/_trgt/lines + bp_deal_document_map + bp_deal(is_tracked=false)
```

Worked example — the freight event:

```
Swift    SDP-Q-44120  V1 905.00  V2 884.00  V3 872.34  (null supplier_id)
Condor   CL-2024-0771 V1 930.00  V2 910.00  V3 898.00
MeridFr  MFS-Q-3391   V1 945.00  V2 928.00  V3 915.00
   9 quote documents -> 3 bids -> 1 sourcing event
   + PO-2024-0091 (Swift, awarded) + its invoices
```

## Error handling

- **Fetch/scoring failure** → proposal generation fails loudly; no partial proposals are
  written. A batch is clustered atomically or not at all.
- **Missing line items on either side** — `cmp_line_composite` returns `(0.5, "MISSING")`,
  contributing nothing to the score. Such pairs must **not** be grouped on a
  near-zero signal; they fall to review. There is currently no telemetry on line-item
  extraction rate; the clustering run will record members-with-lines so this stops being
  invisible.
- **Null supplier_id** (3/28 quotes here) — must not block grouping; product similarity
  carries it, and the proposal is flagged so the user sees the supplier is unidentified.
- **No confident cluster** → documents remain unassigned and are listed as ungrouped.
  Never invent a deal to absorb them.
- **Confirm on a stale proposal** (documents re-extracted since) → reject with a conflict
  and regenerate, rather than assigning against outdated members.

## Testing

- **Fixtures from this real batch** — the 4 sourcing events above are the golden case;
  clustering must reproduce them from the 40 documents.
- **Multi-supplier rivalry (the primary case)** — the freight event must group Swift,
  Condor and Meridian Freight into **one** deal with **three** rival bidders. Asserting
  the supplier count per event, not just the document count: a deal that ends up with one
  supplier where three bid is the exact failure this design exists to prevent.
- **Rivalry beats the old scorer** — regression-guard the measured figures: competing
  Condor↔Meridian Freight scores 1.0 under correlation vs 0.75 under `_line_pair_score`,
  and `cmp_supplier` on rival suppliers is asserted to be `(0.0, 'CONFLICT')` so the
  reason the old profile cannot be reused stays documented in a test.
- **Declared linkage is never overridden** — a document the user explicitly attached to a
  deal stays there even when correlation would score it into a different cluster. This is
  the test that protects the precedence rule; it must fail loudly if inference ever wins.
- **Price proximity, not equality** — rival bids at 1.08× group; the same description at
  312× (freight vs IT) does not. Guards against re-introducing exact-match price scoring.
- **No single signal decides** — freight↔consultancy, whose prices sit within 1.37×,
  must stay unclustered at ~0.28 because description and volume disagree.
- **No signal is a gate** — the consultancy event, whose V1 bids span 39 days, must still
  group. A test that fails if any single signal can veto a grouping the others support;
  this is the regression guard against re-introducing the price ceiling in any form.
- **Corroboration cannot discriminate** — `buyer_id` and currency are uniform across all
  28 quotes, so a grouping must never be justified by them alone; they may only adjust
  confidence in a grouping already supported by tier 2/3 evidence.
- **Evidence is legible** — every proposal names which signals passed, failed and were
  unavailable. A proposal carrying a bare score with no signal breakdown fails the test.
- **Complete linkage, not single linkage** — the measured regression: with single linkage
  IT-MSA and Platform merge into one six-supplier group on the Fortis↔NexusFlow pair at
  0.626. The test asserts 4 distinct clusters and pins the confidences (99.7 / 93.8 /
  84.8 / 75.6), so a linkage-strategy change cannot silently reintroduce chaining.
- **HITL routing** — Platform (75.6) must route to human review and the other three must
  not. Asserts review is targeted rather than blanket; a change making all four require
  review fails, as does one that lets 75.6 through unreviewed.
- **A confirmed grouping is never re-proposed** — including a rejected one, so the same
  wrong pairing does not resurface on the next clustering run.
- **Noise-row filter** — commentary lines (`OPPORTUNITY — …`, `Validity`, `WATCH …`) are
  excluded from scoring; a fixture asserts they never drive a match.
- **Award exclusivity vetoes rivalry (product-wide negative control)** — quotes 128234
  (Duncan) and 136586 (Perry) correlate at 0.744, and 102494 (Dixon) / 104683 (Gomez) at
  0.703, yet each anchors its own PO and invoice. All four must remain separate deals.
  This is the regression guard against merging repeat commodity buying into a fake
  competition, and it runs over the real non-analysis corpus, not fixtures.
- **Rivalry still forms where there is one award** — the freight event's 3 bidders share
  a single PO (Swift) and must still group. Asserts the veto is pairwise-structural and
  does not suppress genuine competition.
- **R1 versions link unawarded** — Condor's 3 rounds, ClearPath's 3, Apex's 3 all group
  despite none being awarded. Fails if award is ever allowed to gate version linking.
- **R2 losers are full members** — each of the 4 events keeps 3 bidders after the veto;
  12 of 18 non-awarded bids linked. A veto change that orphans losers fails here.
- **R3 same supplier never rivals** — `WSG100024` / `WSG100025` (both Dell Workspace,
  identical £111,975.00 total) must NOT form a 2-bidder group. Guards the definitional
  rule that version collapse by `(V<n>)` suffix alone does not cover.
- **R4 unconnected routes to human** — 13 bids end unconnected and each must carry a
  reason and its best rejected match. A bid dropped silently, or force-fitted into a
  group, fails.
- **Award detection is continuity-scored, not name-matched** — `SUP-GomezGoodAndCross`
  vs PO `Gomez, Good and Cross Trading Ltd` must still resolve as awarded. Pins the
  measured failure so string comparison cannot be reintroduced as the test.
- **Version collapse** — 28 quotes → 12 base references, `(V3 (BAFO))` recognised as the
  current round. Same supplier ⇒ rounds; different supplier ⇒ rivals, never merged.
- **Swift/null-supplier** — PO-2024-0091 groups via product similarity despite
  `SDP-Q-44120` having no supplier_id (rivalry score 0.888 against Condor).
- **Caldwell orphan** — PO-2024-0163 stays orphaned; asserting it is *not* absorbed.
- **Cross-category negative** — no freight quote ever joins the IT services group.
- **Idempotence** — regenerating proposals for an unchanged batch is stable.
- **Non-destructive** — proposal generation leaves every existing `deal_id` byte-identical.

## Migration

The existing `ANALYSISSET_19072620260719339` batch is backfilled through the same path:
generate proposals from its 40 documents, present them for confirmation. The current
single deal is marked superseded once its members are confirmed into real deals. No
existing row is rewritten before the user confirms.

## Out of scope

Product code / SKU extraction (no source field exists today); merging deals across
different upload batches; automatic award detection; changing the draft→tracked promotion
gate; fixing `bp_deal_overview.three_way_match` semantics (recorded as a follow-up —
correct grouping makes it meaningful again, but the presence-count definition remains
wrong on its own terms).
