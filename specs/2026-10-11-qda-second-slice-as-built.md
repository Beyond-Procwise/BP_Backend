# QDA backend, second slice: B2, B3, B5, B8a — as built

**Date:** 2026-10-11
**Branch:** `qda-b2-quotepack` (worktree `BP_Backend-qda`), three commits on top of `origin/Development` `7897ceb0`. Not pushed, not deployed.
**Source:** `beyond_procwise_ui/docs/qda/backend-build-instructions.md` §2. First slice: `specs/2026-10-10-qda-first-slice-design.md`.

## What this slice does

A customer quote now has one path from the record to what the customer is sent:

    customer record -> pack -> document -> page -> gate -> (PDF)

The page a customer gets is drawn only from the pack, the pack is built only from the
customer record, and a page that fails its gate is not returned. No model is called.

No new tables, no migration, no database writes outside the test suite's own `LIVETEST` rows.

## B2. The pack and the document

- `rga/quote_models.py`. `QuotePack` extends `FactPack` with `identity` (who the quote is
  for, its dates, its status), `lines` (id `L<sales_quote_line_id>`), `terms` (`T0001`),
  `notes` (`N<justification_id>`, customer-safe justifications only) and `total_fact_id`.
  Every entry names the row it was read from. The total is a fact.
- `QuoteAST` is a flat list of `line_table`, `totals`, `terms`, `notes`, `payment` and
  `signature` blocks. Every field is an id that must resolve in the pack or one of a fixed
  set of words. There is no title, label, text or number field.
- `rga/builders/sales_quote.py` builds the pack, registered as report type `sales_quote`.
  It is handed the customer record only; a record carrying a cost or margin field is refused.

## B3. The gate — `postcheck.run_quote`

Runs on the finished page. Each check blocks.

| Finding | Caught |
|---|---|
| `QUOTE_LINE_OMITTED` | a line the document does not name, or a line not drawn with its quantity, price and total. One row cannot stand for two identical lines |
| `QUOTE_TOTAL_MISMATCH` | the total missing from the document or the page, or reading differently from the record; lines that do not add up to the total |
| `INTERNAL_FIELD_LEAK` | one of our cost or margin figures in what the customer reads (the B1b check, same rules); an internal field name anywhere in the file, attributes and comments included |
| `QUOTE_NUMBER_BEFORE_ISSUE` | a number on a quote that is not issued, or a number shown that the quote does not have |
| `QUOTE_NOT_APPROVED` | any status other than approved or issued |
| `REPORT_UNTRACED_FIGURE` | any figure on the page the pack does not hold |
| `UNKNOWN_FACT_REF`, `MISSING_HASH_RECORD` | as for reports |

A quote page carries no fact marker, confidence badge or provenance footnote, and its gate
asks for none (`MISSING_ORIGIN_BADGE` does not apply to quotes).

## B5. One renderer

- `rga/render/quote_html.py` draws the page from the document; `to_pdf` prints that page
  through `report_export.render_pdf`. The PDF is made from the page's bytes, so both carry
  the same AST hash and pack hash.
- `rga/quote_pipeline.produce(quote, internal, fmt="html"|"pdf")` is the one entry point.
- `GET /sales/quotes/{id}/customer.html` now takes this path. `GET /sales/quotes/{id}/customer.pdf`
  is new. Both ask `quote.read`. A failed gate is HTTP 409 with the finding codes; a leak
  raises `InternalFieldLeak` as before.
- `quote_render.render_html` is retired. `quote_render.customer_view` (the JSON view) and
  the leak guards stay.

## B8a. Action names

`sales_quote.create`, `sales_quote.price` (write), `sales_quote.send` (communicate),
`sales_quote.override_margin_floor` (transact). `POST /sales/quotes` now asks
`sales_quote.create` instead of `sales.write`. Only `create` has a caller; the other three
are named for B7a and B12.

## Differences from the build instructions

1. **Quote blocks are their own union, not added to `models.Block`.** That union is the
   grammar the report composer writes under; adding to it would offer a model a signature
   block. A test pins the composer's schema as unchanged.
2. **A `notes` block and pack notes were added** (not in the B2 list). The old page printed
   customer-safe justifications; without this the new path could not replace it.
3. **"Deny-not-default per action" holds for two of the four.** `send` and
   `override_margin_floor` are refused when no policy speaks. `create` and `price` are
   writes, which the platform allows a Buyer by default — the same as `sales.write` before.
   Making them refused-by-default needs policy rows on both databases (a decision, and
   B8b's territory), not a different class.
4. **`FactPack.hash` was restructured** so a subclass can add to it. A report pack hashes
   exactly as before; a test pins a value taken before the change.
5. **The page shows the draft reference (`SQ-…`) labelled "Reference"**, and "Quote number"
   above it once the quote has one (B6, below).
6. **The supplier's own name is not on the page.** No entity profile exists (G2). A supplier
   signature block reads "Signed for the supplier".
7. **Tax is not on the page.** No rate is configured (Q4). The only total is "Total excluding tax".
8. **Producing a page writes no audit row.** The route is a GET that may be called on every
   view; the pack-built and post-check events are off for it. B12's send is where the
   artefact hash should be recorded.

## Results

- New tests: 48 (pack and document), 60 (page, gate, PDF, path), 22 (actions). The route
  suite is now 28. Seven tests of the retired page were removed; what they covered is
  tested on the new path.
- 51 deliberate breaks, one guard at a time; every one turns at least one test red. Three
  did not on the first pass (a redundant branch, an untested column rule, note text missing
  from a hash test); tests were added and all three now go red.
- RGA, sell-side, sales-router and guardrail suites against bp_testdb: 925 passed, 4 failed.
  None of the four is from this change:
  - `test_board_paper_live::test_the_deal_is_measured` — fails the same way on untouched
    `origin/Development` (the corpus now holds 9 quotes where the test expects 3).
  - `test_policy_action_vocabulary` (2) — `ReceiptTolerancePolicy` (id 1893) on bp_testdb is
    written against `three_way_match`, which is not an action name.
  - `test_job_store_live::test_a_beat_keeps_this_process_s_jobs_alive` — failed once in the
    full run, passed three times alone; the files it tests are unchanged.
- The two issued quotes on bp_testdb (496, 529) produce a page and a PDF that pass the
  gate; neither file contains the quote's cost or margin. The one in review (554) has no page.

## Not built, and why

| Item | Waiting on |
|---|---|
| B7a price from cost, margin floor | a floor value, which needs the settings decision (Q4) |
| B12 send | a ruling: is sending the quote the act that issues it? The UI has one status (`sent`), the backend has `issue` as its own gated step. Also depends on the Email compose backend, which another stream owns |
| Dropping the seven old cost/margin columns | separate approval (first slice, step 2) |
| Storing or reloading a quote pack through the report job store | not wired; `job_store` loads `FactPack`, which refuses the quote fields. Needed only if quotes become jobs |

## B6. Numbering — added 2026-10-11, branch `qda-b6-numbering`

Approved by Nick 2026-10-11: the counter table, and a `quote_number` column on the existing
quote table to hold the number given.

- Migration `deploy/sql/2026-10-11_bp_doc_number_counter.sql` (+ rollback), applied to
  bp_testdb and bp_sqldb. On bp_testdb it was applied, rolled back, applied and re-run.
  - `proc.bp_doc_number_counter (doc_type, year)` holds the last number handed out.
  - `proc.bp_sales_quote.quote_number`, empty until issue, unique when set.
  - `tenant_id` set to `'default'` on the three bp_testdb quotes that had none (bp_sqldb
    holds no quotes). New quotes are written with it, so `UNIQUE (tenant_id, quote_ref)`
    is enforced for the first time.
- `sell_side/numbering.allocate` takes the next value on the caller's transaction.
  `quotes.issue` calls it in the same transaction as the status change and the audit event,
  so a failed issue gives its number back. The issue event carries the number.
- Format `Q-<year>-<4 digits>` (`Q-2026-0007`), one series per year. The format is two
  constants in `quotes.py`; review question Q12 (format and ownership) is still open and
  changing it later does not touch the counter.
- `quote_ref` (`SQ-…`) is unchanged and stays the internal draft reference.

Not done, on purpose:

- **The two quotes already issued on bp_testdb (496, 529) have no number.** A number is
  minted at issue, never afterwards. The gate does not refuse an issued quote without one,
  or those two could no longer be shown.
- **Deploy order:** the migration must be on a database before code that reads
  `quote_number` runs against it. It is on both now.

Tests (live database, `tests/sell_side/test_numbering_live.py`, 16): consecutive numbers one
after another; eight issues released together on eight connections take 1 to 8 with no gap
or repeat; a second allocation waits for the first; a rolled-back allocation is handed out
again; an issue whose audit row cannot be written gives its number back; an expired quote
takes none; a quote is numbered once; two quotes cannot hold one number. Tests draw from a
`LIVETEST` series that is deleted afterwards, so a run uses no real number. Eleven deliberate
breaks, including an allocator with no lock and one that commits early: all go red.
Suites against bp_testdb: 943 passed, the same 3 unrelated failures as above.

## B12. Send — added 2026-10-11, branch `qda-b12-send`

Ruling from Nick, 2026-10-11: **sending is the act that issues the quote.**

- `sell_side/quote_send.py`. `POST /sales/quotes/{id}/send` emails the quote's PDF to a
  contact on the quote's account. `GET /sales/quotes/{id}/send-draft` returns the cover
  draft and its checks and changes nothing.
- **Order.** An approved quote is issued and numbered; every check the send needs runs
  before that is kept (`quotes.issue(before_commit=)`), so a quote that could not be sent
  is not issued and its number goes back. The issue is then kept, the email goes, and
  `sales_quote.sent` is recorded. An email the relay does not take leaves the quote
  issued, numbered and unsent, and the same call sends it. The other order (email first)
  could put a number in a customer's hands that the next quote is then given.
- **One email at a time.** The quote's row is locked before its status is read and again
  from the last check until the record is written. Of two sends at once, one goes.
- **Recipient.** The quote's own contact, or another contact on the same account named by
  `contact_id`. No address is accepted from the caller. One plain address only.
- **Cover.** Fixed words around the quote's number: no price, total, date, term or draft
  reference. No model is called. Subject `Quote <number>`.
- **Checks** (all block; the send route returns them all on a 409): `quote_approved`,
  `quote_in_date`, `not_already_sent`, `quote_numbered`, `has_recipient`,
  `named_docs_attached` (the PDF passed `postcheck.run_quote`), `attachments_within_limit`,
  `no_internal_ids`, `figures_traced`, `no_bank_details`, `mailbox_can_send`,
  `sent_by_a_person`, `within_daily_limit`.
- **Permission.** The route asks `sales_quote.send`, and `sales_quote.issue` as well when
  the quote is still only approved. `sales_quote.send` is refused for everyone until
  `deploy/sql/2026-10-11_sales_quote_send_authority.sql` is applied. **Applied 2026-10-11 to bp_testdb
  (policy 2552) and bp_sqldb (policy 463) on Nick's yes**: Approver and Admin are allowed,
  Buyer, Viewer and an unsigned caller are refused, checked through the gate on both.
  `POST /sales/quotes/{id}/issue` stays (Nick's ruling): it issues a quote without emailing it.
- **Record.** `email.send` (allowed) is written on its own connection before the relay is
  called, so an email that cannot be recorded does not go and one that goes counts against
  the sender's daily limit (`email_dispatch_guard.check_daily_volume`, lifted out of
  `check_dispatch` unchanged). `sales_quote.sent` carries the message id, the recipient,
  the contact, and the PDF's SHA-256, size, pack hash, AST hash and renderer version.
  A refused send writes `sales_quote.send_blocked`; a failed one `sales_quote.send_failed`.

Differences from the contract (`EmailCompose/BACKEND_CONTRACT.md` §2.9), and limits:

1. **The event is `sales_quote.sent`, not `quote.sent`,** to sit with the other quote events.
2. **"Sent" is not a status.** The quote's status stays `issued`; that it went is the
   `sales_quote.sent` event, read back by `send-draft` as `sent`. A status or column would
   need a migration.
3. **The existing `POST /quotes/{id}/issue` is unchanged.** It still issues without
   sending. Under the ruling it is now a second way to issue; whether to remove it is Nick's.
4. **The PDF sent is not stored.** Its hash and the record needed to rebuild it are. §2.6
   asks for the exact artefact to be kept; there is nowhere to keep it yet.
5. **The compose envelope service (`/api/v1/negotiation/drafts/*`) is not built here.**
   The draft is returned in that envelope's shape for a quote, read-only. The cover cannot
   be edited, so there is no patch, regenerate or acknowledge.
6. **If the email goes and the record of it then cannot be written,** the call fails and
   the quote reads as not sent, so a retry sends it again (same number, same document).
   Rare, and reported as an error rather than hidden.
7. **Quotes 496 and 529 on bp_testdb** (issued before numbering) cannot be sent: a
   customer is not sent a quote with no number.
8. **Replies** go to the sending address (`ses_default_sender`). Nothing here reads them.
