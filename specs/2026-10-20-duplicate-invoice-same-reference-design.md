# Duplicate invoice: one invoice number on two different documents (backlog 1.16)

## Why
Invoice `01-2024-002` was uploaded twice as two *different* documents: `MASTER Invoice for PO1.pdf`
(supplier "Borcelle", £17,065.50) and `PO1_IT_Invoice_LowerCost.pdf` ("Borcelle Fauget", £15,877.50).
This is a deliberate scenario (the user, 2026-10-20): two invoices were sent by accident and could both have
been paid. The system must **highlight it as a duplicate**, and it is then actioned like any duplicate.

Today nothing is raised, for three independent reasons:
1. `find_duplicates` only compares invoices of the same supplier **and an identical total to the penny**
   ("a different amount is a different bill"), and its amount signal caps the pair far below the raise band.
2. It explicitly skips two documents that share an invoice number (`earlier.invoice_id == later.invoice_id`).
3. `_stg` / `_trgt` keep ONE row per invoice number (`ON CONFLICT (invoice_id) DO UPDATE`), so the detector,
   which reads `_trgt`, never even sees two invoices. Only `proc.bp_invoice_raw` (permanent) holds both.

## What changes
A second finding kind inside the existing detector: **same reference, two documents**.

* **Source:** `proc.bp_invoice_raw`. A *document* is one **source file**: reading it again, even with different
  extracted text after a reader fix, is the same document and its latest read speaks for it (revised after the
  independent review: hashing the extracted text made two reads of one file look like two invoices). Two file names
  whose stored text is identical and non-empty are one document sent twice by hand (upload-time dedup flags those);
  99.7% of raw invoices store no text, so an empty hash never merges anything.
* **Match (all required):** identical normalised invoice reference (case and spacing ignored); the suppliers agree (equal supplier id, or one
  normalised name's words are all contained in the other's, so "Borcelle" matches "Borcelle Fauget"; two suppliers
  with no word in common never match; names that merely contain one another also need invoice dates within 90 days, filler words like
  "the/group" are ignored); both totals present and positive (a credit note or an unreadable total is
  never compared: absent data is not agreement).
* **Two sub-kinds, one finding per invoice number:** *same amount* (an identical bill sent twice by two files) and
  *different amount* (conflicting versions of one number: only one can be right).
* **Which document is flagged:** the one first ingested LATER (it is the one that should not have been paid),
  against the earliest document with that number. The finding's `doc_pk` is the invoice number, so the existing
  idempotency (`_already_raised`, which also honours a human dismissal) works unchanged: a dismissed pair never returns.
* **Reference spelling:** the earliest document's, which can never change once set, so the finding key is stable.
* **Idempotency:** this kind of finding asks "already raised?" by its own wording, so a pair-scan finding on the same
  invoice id does not hide it (they say different things); a dismissed one still never returns.
* **Finding:** the existing `bp_extraction_discrepancy` row, `issue_type='duplicate_invoice'`, `blocks_promotion=false`,
  severity `critical` when the totals are identical, `warning` when they differ (a human confirms which is right).
  The note names both documents, both totals and the difference, and says to hold payment.
* **Recovery item:** the existing "Duplicate Invoice Recovery" opportunity, keyed `dupref:<invoice number>`. Impact
  (GBP) = the **lower** total: if both were paid that is the least that was overpaid whichever bill was the true one.
  The higher total is recorded in `calculation_details`. Same FX handling and "never claim a payment we cannot see"
  wording as today.
* **Nothing about promotion changes.** `_stg`/`_trgt` still keep the latest document per number; the finding is the
  highlight and the raw table keeps the other. (Keeping both in `_trgt` is a bigger change and is NOT proposed here.)

## Switch and safety
* Runs inside `run_detector`, so it obeys the same governed switch (`duplicate_invoice_detector_enabled`, OFF by
  default). Merging this code changes no behaviour on its own.
* `scripts/backfill_duplicate_invoices.py` (dry run by default) lists these too, so every case can be checked
  against the real documents before anything is written. Applying it, or enabling the switch, is a separate decision.

## Supplier agreement (my call)
Word-subset match on normalised names (lower-case, punctuation removed, company suffixes `ltd limited llp inc plc
co` ignored). "Borcelle" is contained in "Borcelle Fauget", so they match. Two unrelated suppliers that happen to
reuse "INV-001" share no word and never match. The smaller name must be fully contained in the larger and have at
least one non-suffix word. Accepted risk: "Acme" and "Acme Packaging" count as the same supplier; the same invoice
number from two names that share a word is overwhelmingly one supplier, and a human confirms every finding.

## Non-goals
Near-identical references (the existing engine already does re-issue markers across *different* ids); fuzzy amounts;
changing what `_trgt` stores; payment blocking itself (the Opportunity/finding is the action trigger, as today).

## Test plan (all offline, fake cursor / plain dicts)
Same ref + different totals flagged; same ref + same total flagged; same file re-read not flagged; different supplier
not flagged; credit note / missing total not flagged; "Borcelle" vs "Borcelle Fauget" flagged; later-ingested document flagged; three documents -> one finding; dismissed ->
not re-raised; opportunity uses the lower total; severity by sub-kind; backfill lists it; scheduler switch untouched.
