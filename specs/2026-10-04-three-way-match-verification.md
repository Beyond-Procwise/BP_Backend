# Three-way match — live verification record

**Date:** 2026-10-05 · **Branch:** `Development` · **Databases:** `bp_testdb` and `bp_sqldb`

The plan's Task 11 asks for this on a real document. The corpus holds **zero**
receipt-like documents (design §13), so the document below was **synthesised**
against a real purchase order and says so on its own face. Everything else —
the order, its lines, the two invoices, the deal — is live data in `bp_testdb`.

---

## What was deployed

Six migrations, applied to **both** databases, in this order:

| # | Migration | What it does |
|---|---|---|
| 1 | `2026-10-04_goods_receipt_tables.sql` | the six `bp_goods_receipt_*` tables |
| 2 | `2026-10-04_goods_receipt_doctype.sql` | `doctype.goods_receipt` + widens `ck_bp_document_type_pipeline` |
| 3 | `2026-10-04_uom_receipt_basis.sql` | `receipt_basis` on `bp_uom_canonical` |
| 4 | `2026-10-04_receipt_tolerances_policy.sql` | `ReceiptTolerancePolicy` (`policy_type='limit'`) |
| 5 | `2026-10-04_deal_overview_three_way.sql` | appends `value_reconciled` + `three_way_matched` |
| 6 | `2026-10-04_deal_overview_drop_three_way_match.sql` | drops `three_way_match`, rebuilds `bp_deal_kpis` |

`tests/sql/test_goods_receipt_both_databases.py` holds the two databases to the
same schema, the same vocabulary row, the same unit classification and the same
policy — 19 assertions, proven red by rolling migration 1 back on `bp_sqldb`
alone.

**Also deployed, outside BP_Backend:** the Node gateway
(`spendiq.service.ts`) was rebuilt and restarted. It queries
`bp_deal_overview` directly, so it returned HTTP 500 on every SpendIQ deal
request for as long as it ran old code against the new view. After the restart:
`GET /spendiq/deals` → 200, 5,037 deals, `valueReconciled` true/false and
`threeWayMatched` null throughout.

`procwise` restarted cleanly: **220 routes before, 220 after.**

---

## The document

`/home/muthu/Downloads/SpendIQDocs/three-way-match-2026-10-05/delivery_note_DN-100645.pdf`
— a landscape A4 delivery note, `DN-100645`, against **PO000645**
(deal `DEALV2-000645`, supplier `SUP-WindroseSupplies7`), carrying three
delivered lines **and the order's prices on its face**, because that is what a
real note does and it is Review Focus #4.

Uploaded by inserting a `proc.process_monitor` row with `category='delivery
note'` — one of the eleven seeded aliases — and letting the running watcher
pick it up. Record 2179.

## What the running server did with it

| Check | Result |
|---|---|
| Types as a goods receipt | `declared=doctype.goods_receipt evidence=doctype.goods_receipt agreement=agreed` |
| Routes at the fifth pipeline | `dispatch start doc_type=goods_receipt` |
| Reads the header | `grn_id=DN-100645`, `po_id=PO000645`, `receipt_date=2024-03-18`, `carrier_ref=CON-884215`, `received_by=J. Okafor` |
| Reads the lines | 3 lines: 5 each, 1 tonne, 8 pack, each with `po_line_ref` |
| **No price field populated** | the note printed `GBP 13.59` / `GBP 67.95`; **none of it is in any row** |
| Links to its PO | `goods receipt DN-100645 linked to PO PO000645 on deal DEALV2-000645` |
| Reaches `_trgt` with a deal | `document_id=DEALV2-000645::goods_receipt::DN-100645` |

## The match, on real data — and it is the set-level case

`PO000645` carries four lines. **Two** invoices (`INV000645-1`, `INV000645-2`)
each bill against it. Either alone would pass.

| PO line | Unit | Ordered | Received | Billed (both invoices) | Result |
|---|---|---|---|---|---|
| 1 | each | 5 | 5 | 5 | silent |
| 2 | tonne | 1 | 1 | **2** | `billed_not_received` |
| 3 | pack | 8 | 8 | **16** | `billed_not_received` |
| 4 | licence | 25 | — | 25 | not assessed — `service_entry`, no delivery note can prove it |

Both findings are in the discrepancy queue, `critical`, `open`, non-blocking,
and reach the Action Centre through the live gateway
(`GET /spendiq/discrepancies`):

> 'Heavy-Duty Compliance Subscription ITM004436' on purchase order PO000645
> line 2: 2 tonne billed (INV000645-1, INV000645-2) against 1 tonne received
> (receipts: DN-100645) — 1 tonne more than arrived

This is Review Focus #3 — *two invoices that each fit alone and together exceed
what was received* — demonstrated on live rows rather than a fixture.

## The deal, and the board paper

`proc.bp_deal_overview` for `DEALV2-000645`: `value_reconciled = false`,
`three_way_matched = false`. Two different readings, two different answers.

The board paper (`build_fact_pack("board_paper", …)`):

* `DEALV2-000645` → **"Goods billed were received: 0.0%"** — a verdict.
* `DEALV2-000394` (no receipt) → **value `None`, display `—`**, with a
  `MEASURE_UNAVAILABLE` finding: *"no goods receipt is recorded against this
  deal, so what was billed could not be compared with what arrived."*
  Not assessed, not nought per cent.

---

## What the live run FOUND, that no fixture test had reached

The first attempt (record 2146, `DN-000645`) extracted a header, linked to its
PO, reached `_trgt` — and produced **zero lines**. It raised no finding, and
the deal would have read `three_way_matched = true`. **A control that passes
because it looked at nothing.** Five defects, each now fixed with a test that
was watched red:

1. **`extract_line_items` discarded every row.** It requires a field named
   exactly `item_description` before it accepts a table row as a line item; the
   schema called the field `description`. Every row of a real delivery note was
   dropped in silence.
2. **A `Unit Price` column mapped onto `unit_of_measure`.** `unit_of_measure`
   carries the label "Unit", which is a substring of "Unit Price", and a goods
   receipt has no price field for the longest-match rule to prefer — so
   `GBP 13.59` was about to be stored as the unit a line arrived in. A doc type
   that declares no money field now refuses a money-named header outright.
3. **The note's number lost its prefix.** `DN-000645` was stored as `000645` —
   indistinguishable from the order it cites, and a `_stg` primary-key collision
   waiting between two notes against orders ending in the same digits.
4. **`received_by` read past the end of its line**, storing
   `"J. Okafor\n\nSignature"` as the person who signed for the goods.
5. **`receipt_date` was never read at all** — the receipt reached `_trgt` with
   no record of *when* anything arrived, which is most of what a delivery note
   is for.

And one in the view itself:

6. **A receipt with no lines made the deal read "goods received".** The
   `receipted` CTE counted the header. Since the match correctly skips a
   receipt with nothing to compare, that silence became a pass. It now counts
   only receipts that have lines.

Defects 1, 2 and 6 are the dangerous ones: each turns "nothing was checked"
into "nothing is wrong".

---

## Standing limits, unchanged by this run

* **39.5% of PO lines can never be covered** by a delivery note (design §6).
  Line 4 above is one of them. The rate this control can ever report is bounded
  by that, and the board paper shows it as a denominator.
* **The positive path is exercised by one synthesised document.** Phase 0 —
  whether customers produce goods receipts at all — has not been answered by a
  customer. If they do not, this is shelf-ware; nothing already live regresses,
  because every migration is additive and nothing blocks.
* **Tolerances are zero on both rules**, by opening position rather than
  measurement. There is no corpus to size them from. They are a `bp_policy`
  row somebody can widen.
* **`bp_deal_kpis.three_way_match_pct` still carries the old name** while
  computing the value reconciliation. Renaming it is a third cross-repo rename
  the design does not ask for.
* **The live receipt stays in `bp_testdb`.** It is the only readable receipt in
  the corpus and the only thing holding the positive path open; the document
  states on its face that it is synthesised.
