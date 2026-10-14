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


---

# Second pass — what the whole-branch review found, 2026-10-05

A fresh reviewer read the ten commits, the spec, the plan and the rulings, and
exercised the match against live `bp_testdb` with adversarial inputs. It proved
**three Critical** and **eleven Important** findings. Every one is fixed below,
each with a test that was watched red first.

Its verdict on the first pass stands as written: *"the control still has both
failure modes the review focus was written to catch."*

## The three that mattered most

Each of these turned **"this could not be checked"** into either a critical
accusation against a supplier who did nothing wrong, or a clean pass.

**1. An unreadable received quantity became a critical over-billing.**
`_f` returns `None` for an absent value and its own docstring says absence must
not read as "nothing arrived" — and the call site was `_f(...) or 0.0`. A
delivery note whose quantity column did not parse produced *"10 each billed
against 0 each received — 10 more than arrived"*, critical, against a supplier
who delivered everything. `quantity_received` is `required: true` in the schema,
but `build_line_items` does not enforce `required`, so a line with only a
description promotes with NULL. **Now:** `UNVERIFIABLE_QUANTITY`, not assessed.

**2. A receipt line the matcher could not place was discarded in silence, and
its PO line then read `NOTHING_RECEIVED`.** `po_line_ref` was extracted, stored
and selected — and never read. A note printing a shortened description or a
bare item code went unplaced, and the line reported *"10 billed and no delivery
recorded at all"* for a line whose delivery note is on file and names it.
**Now:** a receipt line is placed by the PO line it NAMES first, and description
matching only covers the rest; an unplaceable line is reported at order level
and suppresses `NOTHING_RECEIVED` for that order, because the delivery it is
missing may be the one that could not be placed.

**3. Every refusal reached a reader as a pass.** `check()` computed
`unverifiable` and `assessed` correctly and `check_against_receipts` threw them
away. The deal's verdict was "a receipt with lines exists AND no open gap", so a
note counting in `each` against an order in `box` — the design's own §13.4
"single most likely practical failure" — raised nothing, had lines, and
published as **"Goods billed were received: 100%"**. **Now:** the receipt
records `lines_assessed` / `lines_unverifiable`
(`deploy/sql/2026-10-05_goods_receipt_match_outcome.sql`), the view requires
`lines_assessed > 0` before giving a verdict at all, and each refusal is filed
as its own informational finding naming the reason.

## The rest

| # | Finding | Fix |
|---|---|---|
| 4 | 20 of 23 arithmetic guards were **red in the suite's default mode** — they needed a live database the default fake connection cannot provide. A guard red in the mode people run is a guard nobody reads. | the unit basis and the tolerances are injected, and a live-only test holds the injected copy to the real tables |
| 5 | A **unit-less** delivery note manufactured a critical over-billing (PO 40 `each`, note prints `5`, invoice bills 40 → a shortfall of 35). Trusting the order's unit is safe for silence, not for accusing. | a missing unit refuses, like a different one |
| 6 | **A gap that closed was never cleared.** Nothing in the codebase resolved an extraction discrepancy, so the Action Centre kept a stale critical finding and the deal stayed `false` for ever — while the module's docstring sold the receipt-side run as "the moment an over-billing stops being one". | `_clear_closed_gaps` resolves what this run no longer raises; resolved, not deleted |
| 7 | One over-billing was **filed against the wrong document**, and twice. The gap is contained in the invoice that over-billed, not in whichever paper was being read. | filed against each contributing invoice, never against the note; the note's return value carries nothing |
| 8 | An **invoice line with no quantity** billed 0, which is always ≤ what arrived, so the line counted as verified while the billed side was unreadable. Services and lump-sum lines have no quantity by design, so this was the common case. | `UNVERIFIABLE_QUANTITY` — and the live re-run now shows exactly this on PO000645 line 1 |
| 9 | A receipt that **could not link was never retried**. `goods_receipt` was added to `_DOC` but every sweep iterates a hard-coded `("invoice","quote","po")`, so a note arriving before its order stayed in `_stg` for ever. | `link_pending_receipts`, in the scheduler's downstream chain |
| 10 | The own-document substitution compared ids with a plain `strip()`, so `INV-1` and `inv-1` were **two invoices** — the document counted twice, manufacturing an over-billing: the exact failure the substitution exists to prevent. | normalised, like the PO side |
| 11 | The board paper printed **a boolean as a percentage**: a deal with one line passing, two failing and one unassessable published "0.0%". | a rate over the lines that could be checked, with the two counts beneath it as the denominator §13.3 asks for |
| 12 | The UI sentence this branch exists to delete **was still on the board paper** — only the `true` branch had moved — and the paper carried no delivery line at all. | both branches moved, the delivery line added, the analysis KPI tile relabelled |
| 13 | The synthesised note was **driving live KPIs and accusing real seeded invoices**, with nothing in the data marking it synthetic. | removed; `scripts/three_way_match_live_proof.py` re-runs or cleans it on demand |
| 14 | **Fail-silent loaders**: a missing table or a permission error was indistinguishable from "no receipts exist", with nothing in the log. | logged at warning |
| — | `_MONEY_HEADER_RE` omitted `value` and `rate`, so a "Unit Rate" column still landed money in `unit_of_measure` | both added |
| — | `rejected > received` produced "against -8 each received" | refuses |

## The live proof, re-run against the fixed code

`./.venv/bin/python scripts/three_way_match_live_proof.py`

```
status   Extracted
header   DN-100645 | PO000645 | 2024-03-18 | J. Okafor | CON-884215
         deal DEALV2-000645 | lines_assessed 2 | lines_unverifiable 2
lines    3   (5 each, 1 tonne, 8 pack — each with its po_line_ref)
priced columns on any goods-receipt table: none
gaps     INV000645-1 + INV000645-2, po_line[2]: 2 tonne billed against 1 received
         INV000645-1 + INV000645-2, po_line[3]: 16 pack billed against 8 received
deal     value_reconciled False | three_way_matched False
```

`lines_assessed 2, lines_unverifiable 2` is the second pass visible in one line.
Lines 2 and 3 were comparable and both failed. **Line 1 is now
`UNVERIFIABLE_QUANTITY`** — `INV000645-1` bills it as a lump sum with no
quantity, which the first pass read as "billed 0" and called verified. **Line 4
is `UNVERIFIABLE_BY_RECEIPT`** — a licence is not something a delivery note can
prove. The first pass reported two findings out of four lines and implied the
other two were fine; this one says which two it could not check.

A gap is filed against **each contributing invoice**, deliberately: two
invoices that together over-bill are both implicated, and an AP clerk looking at
either needs to see it. `_clear_closed_gaps` resolves them together when the
rest of the delivery arrives.

Everything the proof wrote is removed again by `--clean`, and the suite now
leaves the database exactly as it found it (measured: zero goods-receipt rows
and zero open quantity findings afterwards).
