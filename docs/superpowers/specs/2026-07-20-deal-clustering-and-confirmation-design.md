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

Where inference is required, correlate on **product description, pricing, volumes and
linked identifiers together**. No single signal decides.

```python
parts.append((0.40, jaccard(tokens(desc_a), tokens(desc_b))))   # product description
parts.append((0.30, 1.0 if abs(qa - qb) < 1e-9 else 0.0))       # volume / quantity
parts.append((0.30, price_proximity(pa, pb)))                   # pricing
```

**Pricing is compared by proximity, not equality.** Exact equality is the signature of
*continuity* (a supplier's own quote → PO). Rivals sit in a tight band around the market
rate, so a price ratio ≤1.25 scores 1.0, decaying to 0.0 by 3.0×. Measured on this batch:

| Requirement | Bidders | Price range | Spread |
|---|---|---|---|
| Freight — Edinburgh lane | 9 | £872.34 – £945.00 | **1.08×** |
| IT — endpoint management | 5 | £272,000 – £291,000 | **1.07×** |
| Consultancy — Business Analyst | 8 | £680 – £780 | **1.15×** |

Within a requirement, 1.07–1.15×. Across requirements, £872 vs £272,000 — **312×**.

Resulting pair scores:

| Pair | Existing `_line_pair_score` | Correlation score |
|---|---|---|
| Condor ↔ Meridian Freight (rival) | 0.75 | **1.000** |
| Condor ↔ Swift (rival, null supplier, description drift) | 0.667 | **0.933** |
| Freight ↔ consultancy (prices coincidentally within 1.37×) | — | **0.280** |
| Freight ↔ IT services | — | **0.000** |

The consultancy row is the reason all four signals are needed: on price alone it would
look related; description and volume outvote the coincidence.

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
2. **Cluster bids into requirement groups** using the correlation comparator above. Each
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
| `requirement_similarity.py` | Correlation scoring over description + pricing (proximity) + volume + linked ids. New `quote_rival` profile. | `linking_engine._tokens`, `_to_float` |
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
