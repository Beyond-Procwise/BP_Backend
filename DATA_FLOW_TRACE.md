# Document Upload — End-to-End Data Flow Trace

**Date:** 2026-07-13
**Traced with:** a real upload of `documents/quote/QUOTE_WSG100025_watermark_split tables.pdf`
through the live local stack (UI :3000 → gateway :3001 → BP_Backend :8000 → bp_sqldb).
Not a mock. Every shape below was read out of the running system.

**Headline:** the document extracted correctly, and then **stopped dead in `_stg`**. The
product reads `_trgt`, so an uploaded quote was invisible in the UI — permanently. One
gate was responsible. Fixed; the document now appears in the UI (Quotes 41 → 42).

---

## The stages

### Stage 1 — UI capture

`src/modules/SpendIQ/engine.js` → `siqPickFiles(documentType)` opens a hidden
`#siqFileInput`, and hands the files to the React bridge
`window.__SPENDIQ_UPLOAD__` (`src/modules/SpendIQ/index.jsx`).

Upload is offered from **dropzones inside detail views** (quote comparison, invoice
discrepancy, compliance), *not* from the "Analyse documents" button on Analyse.

**Out:** `FileList` + `documentType` (defaults `'quote'`).

### Stage 2 — Presign (gateway)

`POST /data-integration/presigned-url`

```
in : { fileNames: ["QUOTE_WSG100025….pdf"], documentType: "quote", totalFiles: 1 }
out: { sessionId: "ses-20260713-2NVK",
       files: [{ fileName, key: "documents/quote/<fileName>",
                 url: "<S3 presigned PUT>", processId: 1171,
                 contentType: "application/pdf" }] }
```

Side effect — **this is where the pipeline actually starts**: the gateway INSERTs a row
into `proc.process_monitor` (`status='Running'`, `file_path=<s3 key>`, `session_id`,
`category=documentType`).

### Stage 3 — S3 PUT (browser → S3 direct)

Bare `fetch` PUT to the presigned URL (no auth header — an `Authorization` header
invalidates the S3 signature). Observed: `200`.

### Stage 4 — Confirm (gateway)

`POST /data-integration/confirm-upload` → `{ processId, success, key }` → `201`.

Sets `process_monitor.status = 'Completed'`, `end_ts`, `file_path`.

**It does not call BP_Backend.** The UI's own comment says this step "triggers
extraction"; it does not. See Stage 5.

### Stage 5 — Trigger (Postgres → BP_Backend)

A DB trigger (`trg_process_monitor_ready`) fires on
`status IN ('Completed','Running') AND id IS NOT NULL` → `pg_notify`.
`src/services/process_monitor_watcher.py` listens, claims the row
(`status → 'Extracting'`, guarded `WHERE status IN ('Completed','Running')`, so only one
worker wins), downloads the object from S3 and dispatches extraction.

`status: Running → Extracting → Extracted | Extraction_Failed`

### Stage 6 — Extraction → `_raw`

Live path is `src/services/extraction/` (`.env EXTRACTION_RENOVATION_ENABLED=1`).

`proc.bp_quote_raw` row 51:

```
process_monitor_id = 1170
source_file        = documents/quote/QUOTE_WSG100025_watermark_split tables.pdf
quote_id           = WSG100025
total_amount       = 111975.00
promotion_status   = promoted        # this means raw → _stg only
```

### Stage 7 — `_raw` → `_stg`

Sanitise + recover + resolve supplier. Supplier resolved from the document text to the
master key `SUP-DellWorkspaceSolutionsLtd`. 14 line items carried through.

```
proc.bp_quote_stg: quote_id=WSG100025, supplier_id=SUP-DellWorkspaceSolutionsLtd,
                   total_amount=111975.00, confidence_score=100,
                   po_id=NULL, deal_id=NULL
```

### Stage 8 — `_stg` → `_trgt` (the promotion gate) ← **THE BREAK**

`src/services/linking_engine.py :: promote_ready` (run by `backend_scheduler`).
`_trgt` is the final destination and **the only tier the product reads**.

Observed before the fix:

```
{"promoted": 0, "held": 1, "by_reason": {"no_parent_reference": 1}}
  {doc_pk: WSG100025, action: held, reason: no_parent_reference}
```

### Stage 9 — Storage → API → UI

`proc.bp_quote_trgt` → gateway `GET /spendiq/quotes` → SpendIQ Quotes table.

---

## Issue found + fix applied

### ISSUE — every uploaded quote was held out of `_trgt` forever

**Where:** `src/services/linking_engine.py :: _evaluate`

```python
if row.get("po_id") is None:
    return None, None, "no_parent_reference"
```

The promotion gate required a staged document to reference a **parent purchase order**.
That is right for an invoice — the PO reference *is* the three-way match, and an invoice
without one is a real exception a human should see.

It is wrong for a quote. **A quote is raised before the PO exists.** Carrying no PO
reference is the *normal* state of a standalone quote, not a defect. So the gate held
every uploaded quote waiting for a parent that had not been raised yet — and since the
UI reads `_trgt`, the document a user just uploaded was **invisible in the product,
permanently**, with no error surfaced anywhere. `process_monitor` cheerfully said
`Extracted`.

**Why nobody noticed:** 37 of the 41 quotes already in `_trgt` *also* have no `po_id` —
they were loaded before this gate existed. So the corpus looked healthy while every new
upload silently fell on the floor.

**Fix (minimal, quotes only):**

1. `_evaluate` — a quote with no `po_id` is no longer held. There is nothing to link
   against, so it promotes on extraction confidence alone (`conf >= MIN_CONFIDENCE`);
   below that it holds as `low_extraction_confidence`. **Invoices are untouched** and
   still require their PO.
2. `_promote` — the promote branch dereferenced `link["F"]` and `po["po_id"]`
   unconditionally, so a parentless promotion would have crashed. It now records an
   unlinked promotion (`decision: "unlinked"`, `F: None`).

**Verified live:**

```
promote_ready(('quote',))  →  {"promoted": 1, "held": 0}
proc.bp_quote_trgt         →  WSG100025, SUP-DellWorkspaceSolutionsLtd, £111,975, GBP
                              14 line items in bp_quote_line_items_trgt
GET /spendiq/quotes        →  42 quotes (was 41), WSG100025 present
SpendIQ UI                 →  row "WSG100025 · Dell Workspace Solutions Ltd · £112k"
```

Regression checks: invoice promotion still holds the same 3 documents for the same
reasons (`parent_not_found` ×2, `no_parent_reference` ×1) — the invoice gate is
unchanged. 14 linking/promotion tests pass. Invoice spend untouched (still 14 invoices /
£323k) because the traced document is a quote.

---

## Observations (not fixed — flagged for Step 2 / your call)

1. **The S3 race is real but currently benign.** The trigger fires on INSERT
   (`status='Running'`), i.e. *before* the browser has PUT the file. In this trace the
   object already existed at the deterministic key `documents/quote/<fileName>`, so
   extraction succeeded anyway. With a genuinely new filename the watcher would reach for
   an object that is not there yet. The 3 historical `Extraction_Failed` rows
   (`test-quote.pdf`, `siq-pipeline-test.pdf`, `live_invoice.pdf`) are all prior UI
   uploads and are consistent with exactly this. *Not fixed — outside the "smallest fix"
   for the traced break, and it needs its own decision (retry-with-backoff vs. only
   notifying on `Completed`).*

2. **The S3 key has no uniqueness.** `documents/{documentType}/{fileName}` — two users
   uploading different files named `quote.pdf` overwrite each other.

3. **Content-hash dedup works.** My second upload of the same bytes was correctly marked
   `doc_action='duplicate'` (`process_monitor` 1171).

4. **`confirm-upload` is decorative for extraction.** It updates the row and notifies the
   client, but the pipeline was already running off the INSERT. The UI comment claiming
   it "triggers extraction" is wrong.

5. **The "Analyse documents" button shows demo numbers.** It renders a fixed panel
   (`QA-1042`, Contract £41,000 / Quote £48,200 / PO £48,200 / Invoice £49,100, "4
   uploaded documents") that does not match the real corpus. Hardcoded — Step 2 item.

6. **`bp_quote_raw.supplier_name` is NULL** for this document even though the supplier was
   resolved downstream in `_stg`. Cosmetic here, but it means `_raw` is not a faithful
   record of what the extractor read.
