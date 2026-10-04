# A real three-way match — design

**Date:** 2026-10-04
**Status:** DRAFT — awaiting Nick's ruling. §13 names the one fact that decides whether to start at all.
**Predecessor:** `5d1bfc4` renamed `three_way_match.py` → `two_way_match.py`, because it compares two
documents. This spec is the other half of that commit: building the third way.
**Reads with:** `specs/2026-10-01-document-relationship-layer-rulings.md` (document typing and parent
proposals), `project_services_deals_no_quantity` (lump-sum lines carry no quantity).

---

## 1. What this is for

Nick's words: *"Should it not just be a two way match and then 3 way match needs to be built?"*

The answer to the first half shipped. This is the second half.

A three-way match is the control that proves an invoice is payable: **what was ordered** (the purchase
order), **what actually arrived** (the goods receipt), and **what was billed** (the invoice) all agree.
Remove any one leg and the control does not hold. Today this product has two legs. An invoice for goods
that were never delivered passes every check it makes, and nothing in the system can see that.

**Success criteria.**

1. A goods receipt can be uploaded or emailed in, recognised as one, extracted, and linked to its PO —
   through the same path every other document already takes.
2. For a PO line, the system can state three numbers from three documents: ordered, received, billed.
3. Where billed exceeds received, a finding is raised naming the PO line, the quantity gap and the
   documents on both sides.
4. Where a line **cannot** be verified by receipt, the system says so rather than passing it. A line that
   was never checked must not read as a line that passed.
5. No document that classifies correctly today classifies differently afterwards.

**Scope boundary.** This raises findings. It does not block promotion and it does not stop a payment,
because nothing in this product pays anything. Blocking is the decision layer's job, and the existing
stance — line-level problems are non-blocking — is deliberate and unchanged here.

---

## 2. The evidence this design rests on

All measured against the live `proc` schema on 2026-10-04.

| Fact | Number | Why it matters |
|---|---|---|
| Receipt-like documents in the corpus | **0** of 184 | §13. Nothing positive can be proven on real data. |
| Invoices | 12,408 | |
| …citing a PO | 10,444 (84.2%) | The spine exists. |
| …whose cited PO resolves in `_trgt` | 10,251 | 82.6% of invoices could be three-way matched if receipts existed. |
| PO lines | 22,565 | |
| …carrying a quantity > 0 | 22,551 (**99.9%**) | A quantity-based match is feasible. This was the open question. |
| …carrying a unit of measure | 22,542 (99.9%) | |
| Invoice lines | 55,483 | |
| …carrying a quantity > 0 | 53,848 (97.1%) | |
| PO lines whose unit is time-based | 8,896 (**39.5%**) | Two fifths of lines can never have a goods receipt. §6. |
| `bp_uom_canonical` rows | 38, with a `dimension` column | The classifier is half-built already. §6. |

Two readings follow, and they point in opposite directions.

**The data supports this.** Quantity and unit are present on effectively every PO line and 97% of invoice
lines, and five sixths of invoices resolve to a real PO. The hard part of a quantity match — having
trustworthy quantities — is already done. That was not obvious before measuring it.

**The corpus cannot validate it.** There are no receipts. Every positive path in this design would be
exercised by fixtures and by nothing else, exactly as `specs/2026-10-02-contract-structures-design.md`
§11.1a had to say about contract children. That is a property of the data, not of the code, and it is the
subject of §13.

---

## 3. What today's checks are, so the new one is not confused with them

Three different things currently answer to similar names. After this design there are four, and each says
what it is.

| Thing | Compares | Today's name | What it becomes |
|---|---|---|---|
| Document check | invoice ↔ PO, by **value** | `two_way_match.py` | unchanged |
| Deal rollup | quote, PO, invoice **totals** within 10% | `bp_deal_overview.three_way_match` | `value_reconciled` (§9) |
| — | PO, receipt, invoice by **quantity** | does not exist | `three_way_matched` (§5, §9) |

The second is the one that has been publishing *"Three-way match rate for this deal"* into board papers.
It compares three document **types** but no receipt, so it never proved delivery. `5d1bfc4` fixed the
wording; §9 fixes the column.

---

## 4. The receipt is a document type, not a new pipeline

`proc.bp_document_type` is a data-driven registry: aliases, identity fields, parent type, distinguishing
phrases, and the `pipeline_doc_type` that routes extraction. `doctype.invoice` already declares parent
`doctype.order` and an identity field `po_id`. A goods receipt declares the same shape, so it needs a
**seed row**, not a new code path.

```
doctype.goods_receipt
  role             role.transaction
  parent_type      doctype.order
  execution        exec.unilateral
  aliases          goods receipt, goods received note, GRN, delivery note,
                   despatch note, dispatch note, advice note, packing list,
                   packing slip, proof of delivery, POD
  identity         grn_id ; po_id -> doctype.order
  phrases          quantities with no prices ; signed for on receipt ;
                   a carrier, vehicle or consignment reference
  pipeline_doc_type  goods_receipt          <- new value
  status           active
```

Storage follows the house `raw → _stg → _trgt` pattern exactly, keyed by `deal_id` at `_trgt` like every
other document:

```
bp_goods_receipt_raw / _stg / _trgt
    grn_id, po_id, supplier_id, receipt_date, delivery_note_ref,
    carrier_ref, received_by, source_file, deal_id
bp_goods_receipt_line_items_raw / _stg / _trgt
    line_no, description, quantity_received, unit_of_measure,
    quantity_rejected, po_line_ref
```

**The extraction schema carries no price fields, deliberately.** A goods receipt has no prices. Giving the
schema a `unit_price` or `line_total` invites the extractor to fill it from a referenced PO or from a
delivery note that happens to restate values, and a fabricated price on the document that is supposed to
prove delivery would be the worst possible place for one. `feedback_no_fabrication_null_when_absent`
applies with unusual force here: absent stays NULL, and the match reads quantity only.

`quantity_rejected` is separate from `quantity_received` on purpose. Goods delivered and refused were not
received, and a receipt that records both must not have them silently summed.

---

## 5. The match runs on quantity, over the set, with the PO line as the spine

The existing value check already learned the lesson this one must not relearn: comparing one document at a
time cannot see aggregate over-billing. `two_way_match._assign_lines` assigns an invoice's lines to PO
lines over the whole document at once, through the resolution layer, and `_check_po_consumed_as_a_set`
compares what a PO line has been billed **in total** against what it authorised.

The three-way match reuses that machinery rather than paralleling it. The PO line is the spine; both the
invoice lines and the receipt lines assign to it:

```
for each PO line L:
    ordered   = L.quantity
    received  = Σ quantity_received  over receipt lines assigned to L      (minus quantity_rejected)
    billed    = Σ quantity           over invoice lines assigned to L
```

`_assign_lines` is called a second time with the receipt's lines, under a new resolution profile
`receipt_line_po_line`, same `1:1` cardinality rule. Nothing about the invoice side changes.

The three findings, in the order they matter:

| Finding | Condition | Severity |
|---|---|---|
| `BILLED_NOT_RECEIVED` | `billed > received + tolerance` | critical — this is the money |
| `NOTHING_RECEIVED` | `billed > 0` and no receipt line assigned to L at all | critical |
| `OVER_DELIVERED` | `received > ordered + tolerance` | warning |

`received > billed` raises nothing. Goods delivered and not yet invoiced are normal, exactly as a partial
invoice is normal, and the existing module's reasoning applies unchanged: *"crying wolf on every one of
them is how a check gets ignored."*

Quantities are compared **in canonical units**. `bp_uom_canonical` already carries `dimension` and
`aliases`; a PO line in `box` and a receipt line in `each` are not comparable until one is converted, and
where no conversion exists the pair is `UNVERIFIABLE_UOM` rather than a silent pass or a false finding.

---

## 6. Not every line can be received, and the system must say which

**39.5% of PO lines carry a time-based unit** — `hour`, `day`, `month`, `year`. Consultancy days and
licence months are not delivered to a loading dock, and no goods receipt will ever exist for them. Running
a goods-receipt control over those lines and reporting a pass would be a lie of exactly the kind this whole
exercise is correcting.

`bp_uom_canonical.dimension` is close to the classifier needed but is not it. `dimension` describes the
physical measure — `count`, `time`, `mass`, `length` — and `licence`, `seat` and `module` all sit under
`count` while being no more deliverable than an hour is. So the classifier is a new column, seeded from
`dimension` and then corrected by hand for the intangible counts:

```
bp_uom_canonical.receipt_basis   goods_receipt | service_entry | none
    goods_receipt   each, box, case, pack, tonne, metre, roll, sheet, pen, set, shipment
    service_entry   hour, day, week, month, quarter, year, licence, seat, module
    none            everything else, including the 18 rows whose dimension is NULL
```

A line whose unit maps to `service_entry` is reported `UNVERIFIABLE_BY_RECEIPT`, with the reason. It does
not count as matched and it does not count as failed; it counts as not assessed. **Service entry sheets —
the services analogue of a goods receipt — are out of scope here (§12)** and are the natural phase 2 of
this work once the goods path is proven.

Those 18 `dimension IS NULL` rows deserve a note of their own: they hold values like
`"30 days from invoice"` and `"implementation (one-off, fixed) — £58,000.00"`. They are extraction noise
that reached a canonical reference table, and they should be cleaned as part of this work rather than
classified.

---

## 7. Tolerances are policy rows, not constants

`two_way_match` carries `_ABS_TOL = 0.01` and `_REL_TOL = 0.005` in the module. That was right for a value
comparison with a fixed rounding rationale. Quantity tolerances are a commercial decision — how much
over-delivery a buyer accepts differs by category and by supplier — so they belong where the other 33
governed limits live.

```
bp_policy  →  receipt_tolerances
    over_delivery_pct        default 0.00    (none accepted unless stated)
    billed_over_received_qty default 0        (absolute units)
    uom_conversion_required  default true
```

Per `project_governed_limits`, **a missing limit raises** rather than falling back to a number in the
code. That behaviour is inherited, not re-implemented.

---

## 8. What it writes

Findings go where the existing discrepancy findings go — the triage path and the Action Centre — keyed
per document, not per business key, following the rule the current module already states: *"A discrepancy
belongs to the piece of paper that contains it."*

Each finding carries the PO line, the three quantities, the canonical unit, and the `source_file` of every
document on both sides, so a buyer can open the pages rather than take the number on trust. The key
includes `source_file`, per `project_findings_key_per_document`, and must be normalised on write at every
site — that bug has been fixed twice and the clause has four normalise sites.

It does not block promotion. It does not mark an invoice unpayable. It raises a finding a person acts on.

---

## 9. The view column and the board paper

`bp_deal_overview.three_way_match` means "quote, PO and invoice totals agree within 10%". That is a useful
number and a wrong name. The migration is additive and runs in two steps so nothing breaks mid-deploy:

**Step 1 — add, do not rename.**
```sql
value_reconciled    boolean  -- today's three_way_match expression, verbatim
three_way_matched   boolean  -- NULL when the deal has no receipt at all
```
`three_way_match` stays, unchanged, until every reader has moved.

`three_way_matched` is **NULL, never false, when no receipt exists.** A deal nobody sent a GRN for has not
failed the three-way match; it has not been assessed. Reporting it as false would manufacture a failure
rate out of missing paperwork, and `feedback_no_fabrication_null_when_absent` is the rule being applied.

**Step 2 — move the readers, then drop.** Five in BP_Backend (`board_paper.py`,
`exec_procurement_summary.py`, `analysis_findings.py`, `opportunity_miner_agent.py`, the deal-overview SQL
tests) and one in the UI (`SpendIQ/engine.js`). Both databases: `bp_sqldb` has run migrations behind
before, and per `project_bp_sqldb_governance_caught_up` a deployment to both is a prerequisite, not a
follow-up.

The board paper then has two honest lines where it had one dishonest one: *"Quote, PO and invoice
reconciled"* (shipped in `5d1bfc4`) and *"Three-way matched"*, the second showing **not assessed** wherever
the receipt is absent.

---

## 10. Build sequence

Phase 0 is not engineering and gates everything after it.

| Phase | What | Gate to the next |
|---|---|---|
| **0** | Establish that customers receive goods receipts at all, and get 10–20 real ones | Receipts exist and can be obtained. **If not, stop here** (§13). |
| **1** | Receipt capture: doctype seed row, `pipeline_doc_type`, raw/`_stg`/`_trgt` tables, extraction schema, PO linking | A real GRN uploads, extracts and links to its PO |
| **2** | `receipt_basis` on `bp_uom_canonical`, seeded and hand-corrected; the 18 noise rows cleaned | Every unit in the corpus classifies |
| **3** | The match: second `_assign_lines` call, per-PO-line aggregation, three findings, tolerances in `bp_policy` | Findings raise correctly on fixtures and on the real receipts from phase 0 |
| **4** | Surface: the two view columns, both databases, the five BP_Backend readers, the UI, the board paper | Board paper shows **not assessed** where it should |

Phases 1–4 are each shippable on their own. Phase 1 alone is worth having even if 3 never lands: a GRN
that is captured, typed and linked to its PO is a document a buyer can find, which today they cannot.

---

## 11. Testing

Following the house pattern, each test names the production change that would make it fail.

- **The arithmetic**, on fixtures: billed > received raises; billed ≤ received is silent; received > ordered
  warns; two invoices that each pass alone and together exceed the received quantity raise once, over the
  set. This is the test that proves the set-level reasoning survived the port from the value check.
- **The absence cases**, which are where this design is most likely to be wrong: no receipt at all yields
  `three_way_matched IS NULL`, not false; a time-based unit yields `UNVERIFIABLE_BY_RECEIPT`, not a pass;
  an unconvertible unit pair yields `UNVERIFIABLE_UOM`, not a finding.
- **The guard fails on purpose.** Per `feedback_prove_the_guard_fails`: break the quantity comparison and
  watch the finding stop appearing. Three guards have shipped green while checking nothing in this
  codebase; a four-legged control that silently never fires is the same failure wearing a bigger badge.
- **No price field is ever populated** on a goods receipt row, asserted directly against the extracted
  record, because §4's reasoning is only as good as the thing that enforces it.
- **Live**, per `feedback_demonstrate_on_local_server_live_data`: the phase 0 receipts through the running
  server against the real database, not only fixtures.

---

## 12. Out of scope

- **Service entry sheets.** The services analogue of a goods receipt, covering the 39.5% of lines §6
  excludes. Phase 2 of the programme, not of this design.
- **Blocking anything.** No payment hold, no promotion block. Findings only (§1).
- **Receipt capture from an ERP feed.** This design reads documents, consistent with the rest of the
  product. If an ERP connection ever lands, receipts arrive as data and most of §4 becomes unnecessary —
  which is an argument for keeping §4 small.
- **Partial-delivery scheduling**, backorders and delivery windows. Quantities only.
- **Renaming `profile_registry_version="three_way_match/line_v1"`**, which is hashed into the resolution
  reproducibility fingerprint. Commented in place in `two_way_match.py`; changing it invalidates stored
  results for a cosmetic gain.

---

## 13. What is unproven, stated plainly

1. **There are zero receipt-like documents in the corpus — 0 of 184.** No delivery notes, no GRNs, no
   packing slips, under any spelling. Every positive path in this design would be exercised by fixtures
   and by nothing else. The negative paths (§11's absence cases) are measurable today; the thing the
   feature exists to do is not.

2. **Whether customers receive goods receipts at all is unknown, and it is the gate.** This is phase 0 and
   it is not an engineering task. A procurement function that never sees a GRN — because the ERP books
   receipts and no document is produced, or because the category is all services — cannot be given a
   three-way match by this product at any price. **Ask before building.** The cost of asking is one
   conversation; the cost of not asking is phases 1–4.

3. **39.5% of PO lines can never be covered**, by construction (§6). Even a complete, perfect
   implementation leaves two fifths of lines `UNVERIFIABLE_BY_RECEIPT` until service entry sheets exist.
   The headline rate this feature can ever report is bounded by that, and the board paper must show it as
   a denominator rather than quietly excluding it.

4. **UoM agreement between PO and receipt lines is unmeasurable today.** Receipts in `each` against POs in
   `box` is the single most likely practical failure, and with no receipts there is no way to size it. §5
   handles it by refusing rather than guessing, which is correct but untested against reality.

5. **The 1,964 invoices (15.8%) that cite no PO are out of reach**, as they already are for the two-way
   match. A three-way match cannot be built for a document with no spine.
