# Email Attachment Ingestion Connector

**Date:** 2026-08-07
**Status:** Design approved, ready for planning
**Repos touched:** BP_Backend only

**Relationship to `docs/ses_inbound_pipeline.md`:** that document describes the SES → S3 → SQS →
Lambda path that exists today and continues to exist. This spec changes what happens *after* the raw
`.eml` lands in `s3://procwisemvp/emails/`: the reply-matching Lambda stops being the first reader
of that object and becomes a consumer of an event the new connector emits (§3).

---

## 1. The problem

Attachments arriving by email are silently dropped.

- `src/services/email_ingest_lambda.py` (727 lines) reads raw MIME from S3, parses it, extracts an
  RFQ id and upserts a supplier reply. It handles **no attachments whatsoever**.
- `src/services/imap_supplier_response_watcher.py:91` carries an `attachments` field and persists it
  to an `attachments JSONB` column — but it stores *metadata only*. The bytes are never written
  anywhere.
- `src/services/email_watcher.py:197,209` walks MIME parts looking exclusively for `text/plain` and
  `text/html`, explicitly skipping anything with an attachment disposition.

So a supplier who replies with a quote attached has their price list discarded at the door, and
nothing records that it happened. That is the gap this connector closes.

A second problem follows from the first. The moment attachment bytes from arbitrary senders reach an
LLM-backed extractor, the extraction stage becomes an attack surface reachable by anyone with an
email address. The connector therefore has to establish provenance *before* extraction runs, and the
boundary between the two has to be structural rather than conventional.

---

## 2. The two planes

**Ingestion Plane (this spec).** Deterministic. Holds mailbox credentials. Fetches, authenticates,
hashes, stores, records. Makes no interpretive judgement about document content. Contains no LLM
calls.

**Interpretation Plane (`src/services/extraction/`, exists).** LLM-backed. Receives an artifact
reference and raw bytes. Holds no credentials, has no network egress, no tool access.

|  | Ingestion Plane | Interpretation Plane |
|---|---|---|
| Lives in | `src/services/mail_intake/` | `src/services/extraction/` |
| Holds | mailbox credentials, object-store writes, DB writes | bytes and a schema |
| May import | mail adapters, boto3, secret store | neither adapters nor `tool_runtime` |
| Receives | a binding | an `artifact_id` + raw bytes |
| Emits | `DocumentIngested` only | a schema-conforming result only |

### 2.1 How the boundary is enforced

Four mechanisms, in descending order of strength:

1. **Extraction gets no tools.** Already true and to be kept true: `extraction/context_layer.py:803`
   calls Ollama with `format=<schema>` and never passes `tools=`. The tool-calling runtime
   (`src/services/tool_runtime.py`) is a separate path used by Ask, governance and supplier
   research. Extraction never touches it.
2. **An import-guard test.** Walks the transitive import graph of `services.extraction.*` and fails
   the build if it reaches a mail adapter, `tool_runtime`, `supplier_enrichment.web_tools`, or an
   HTTP client. A failing build, not a warning.
3. **Separate object-store prefixes.** Quarantined bytes are written under a prefix the extraction
   path holds no key for. Extraction is handed an `artifact_id`, never a path it can vary.
4. **Provenance written before extraction, never updated by it.** Sender, timestamps and
   authentication results are columns extraction has no write path to. A document claiming a
   different supplier than the email that carried it is a recorded mismatch, not a correction.

### 2.2 Stated limitation

Both planes run in one Python process under one DB user. An import guard is a **static** boundary.
Real capability enforcement — separate OS process, no network route, read-only DB role — requires a
deployment change and is explicitly out of scope for this pass. The code is structured so that split
needs no refactor. This spec does not claim the boundary is enforced by capability at runtime.

### 2.3 Existing boundary crossings this spec closes

| # | Crossing | Resolution |
|---|---|---|
| 1 | `POST /documents/extract-from-s3` (`src/api/routers/documents.py:228`) accepts an arbitrary S3 prefix, lists every key under it and extracts each — the safety gate is bypassable | Endpoint rejects any path resolving under the quarantine prefix, and rejects prefixes that expand to more than one key; ingested artifacts are addressed by `artifact_id`, never by caller-supplied path |
| 2 | `supplier_enrichment/research.py:44-52` gives an LLM `web_search` and `fetch_url` — network egress from a model context | Added to the import guard; ingested artifact text must not reach it |
| 3 | `email_ingest_lambda._move_to_unmatched` is best-effort shaped where §5 demands fail-closed | Superseded — the connector owns intake (§3) |

Noted and **not** a crossing: `supplier_enrichment/research.py:40` already refuses to research or
apply `bank_name`, `bank_account_number`, `bank_swift`, `bank_iban` and `credit_limit_amount`. That
is the "no privilege inference from content" principle already implemented, and it stays.

---

## 3. Ownership

The connector becomes the single front door. It parses, hashes, gates and stores every inbound
message, then emits `DocumentIngested`. Reply-matching moves to a subscriber of that event.

This is the only arrangement in which the audit spine holds. Two processes reading the same S3
object independently means an artifact can reach extraction without passing the safety gate, and
dedup and idempotency become guesswork.

**Migration:** `email_ingest_lambda.py`'s RFQ-resolution and thread-mapping logic is preserved intact
and re-homed behind a subscriber. Its S3 event wiring is replaced by the connector's. No matching
behaviour changes in this pass.

---

## 4. Adapters

`MailSource` is one method — `iter_messages(binding, since)` — yielding raw bytes plus provider ids.
Everything interpretive happens downstream in shared code, so all adapters get identical parsing,
gating and hashing. An adapter able to vary the safety gate would be a hole.

| Adapter | Status | Notes |
|---|---|---|
| Relay | **live** | Default and preferred. Per-binding address; raw MIME to S3; push-based; no mailbox-wide read scope |
| Graph | **live, wrapping existing** | Auth and scope probe already exist in `src/services/style/graph_source.py`. Must handle `fileAttachment`, `itemAttachment` (recurse to depth 3), `referenceAttachment` (resolve or quarantine), and the streaming endpoint above the inline size threshold |
| IMAP | **live, wrapping existing** | Wraps `ImapEmailFetcher` (`email_watcher.py:542`). Idempotent by UID + content hash |
| Gmail | **stub** | Same interface, raises `NotImplementedError`. No customer, no test mailbox today |

Graph and IMAP are implemented rather than stubbed because working code for both already exists;
validating the interface against three real providers is the point of §4's no-refactor requirement.

### 4.1 Sender authentication is read, not recomputed

SPF cannot be recomputed after forwarding — it checks the connecting IP, which is gone. Therefore:

- **Relay path:** our own boundary MTA performs the checks and we trust its verdict, because we
  control it. Another reason relay is preferred.
- **Graph/IMAP paths:** we read the `Authentication-Results` header written by the customer's mail
  system, and record **who asserted it** alongside the verdict.
- **Missing header:** recorded as `unknown` and raises a Finding. Never treated as a pass.

---

## 5. Data model

All tables `bp_` prefixed, in `proc`, indexes `ix_bp_*_*`.

### 5.1 Extended

`proc.bp_mailbox_binding` already exists (`src/services/style/mailbox.py`) with the correct
credential-reference shape — `credential_ref`, `scope_policy_ref`, `scope_verified_at`,
`scope_evidence_ref`, `health_state` — and a negative-control scope probe that deliberately tries to
read a mailbox it should not reach and treats success as a failure. One credential model for the
whole product.

Added: `folder_scope`, `poll_interval_seconds`, `last_sync_at`, and `role='intake'`.

### 5.2 New

| Table | Holds |
|---|---|
| `bp_ingest_bundle` | one inbound email: binding, provider message id, RFC-5322 `Message-ID`, thread id, sender address/display name/domain, recipients, subject, received-at, **body text and HTML**, raw-MIME key, state, quarantine reason |
| `bp_ingest_artifact` | one attachment: bundle, declared filename, declared and detected MIME, size, SHA-256, object key, nesting depth, parent artifact, disposition, content id, state, quarantine reason, `duplicate_of_artifact_id` |
| `bp_ingest_sender_auth` | SPF, DKIM, DMARC, envelope-from, header-from, alignment, `asserted_by`, supplier match state, matched supplier id |
| `bp_ingest_arrival` | every arrival of the same content — four forwards leave one object and four dated rows |
| `bp_artifact_derivation` | append-only artifact → extraction run, with pipeline version and timestamp. Reprocessing appends; never overwrites |
| `bp_check_definition` | check code, title, severity, `disposition`, `enabled`, disabled-reason, version |
| `bp_ingest_finding` | check code, bundle, artifact (nullable), supplier (nullable), deal (nullable), severity, disposition, detail, state, resolved-at, resolved-by |
| `bp_supplier_domain` | supplier, domain, first-seen, last-seen, source, confirmed-by — see §7.2 |

### 5.3 Three decisions

**Body lives on the bundle, not as an artifact.** Validity windows, exclusions and payment terms
routinely live in the covering note. Extraction receives bundle body and artifact bytes together as
one unit of meaning. This is acceptance test 17 and it is part of the contract.

**Tenancy: the binding is the scope.** The live `proc` schema has **zero** columns named `tenant_id`
(verified against bp_sqldb). Every bundle and artifact carries `binding_id`; a binding is one mailbox
with one credential, which is already a hard isolation boundary. "Within the tenant" for the
reference-attachment check (§7) means *resolvable using this binding's own credential*, which is
checkable today. A `tenant_id` added later to the binding alone propagates through the FK. Adding
nullable unenforced `tenant_id` columns to new tables while `bp_supplier` and the extraction tables
have none would be isolation theatre — enforced on the new tables, defeated one join away.

**Dedup scope is `bp_mailbox_binding.user_ref`.** §8 of the brief says "duplicate hash within
tenant"; there is no tenant. `user_ref` already groups bindings under one owner, so a document
arriving at two mailboxes belonging to the same customer still dedupes to one stored object.
Per-binding dedup would have missed that, and it is the more common real case.

### 5.4 States

State is explicit, stored, and *is* the work queue.

- **Bundle:** `received → parsed → authenticated → bound → complete`, or `quarantined`
- **Artifact:** `extracted → typed → gated → stored → ready_for_extraction → extracted`, or
  `quarantined`, or `duplicate`

Because `ready_for_extraction` is a durable row, no unit of work can be silently lost.

---

## 6. Pipeline

`src/services/mail_intake/`:

```
sources/          base.py · relay.py · graph.py · imap_source.py · gmail.py (stub)
mime_walk.py      parse + nested-message recursion to depth 3
sender_auth.py    SPF/DKIM/DMARC + alignment
supplier_bind.py  domain → supplier: matched | unmatched | ambiguous
typing_.py        magic-byte detection + permitted alias set
safety.py         the stage-7 gate
hidden_text.py    PDF via PyMuPDF + Office formats
store.py          content-addressed writes; separate quarantine prefix
dedup.py          content hash + logical duplicate
checks.py         the nine checks
pipeline.py       deterministic orchestrator, explicit state transitions
repo.py           all DB access
```

| Stage | Does | On failure |
|---|---|---|
| 1 Receive | raw MIME to object store **before parsing** | adapter error → retry with backoff; bytes never lost |
| 2 Parse | headers, bodies, parts; nested messages to depth 3 | unparseable → `quarantined`, raw retained, **no partial ingestion** |
| 3 Sender auth | read verdicts, record who asserted them | missing → `unknown` + Finding, never a pass |
| 4 Supplier bind | matched / unmatched / **ambiguous** | never guesses; ambiguous is a stored state |
| 5 Filter | drop CID-referenced inline parts only | non-CID inline parts retained |
| 6 Type detect | magic bytes (`python-magic`) | unrecognised → `quarantined` |
| 7 Safety gate | encrypted, macro, archive, executable, oversize, type mismatch | → `quarantined`, human-releasable, release audited |
| 8 Hash + store | SHA-256, content-addressed immutable key | duplicate → link + new `bp_ingest_arrival` row |
| 9 Dedup | logical duplicates across forwards | marked `duplicate_of`, never deleted |
| 10 Emit | state → `ready_for_extraction`, publish `DocumentIngested` | durable row is the queue; scheduler sweeps stragglers |

**Never execute, never auto-open, never auto-expand.** Archives are queued for a human decision, not
expanded. Quarantined items are retained with reason and are re-releasable by an authorised user,
which is itself an audited event.

### 6.1 Retry policy

Split by cause, deterministically:

- **Transient** (network, provider throttling, object-store timeout) → retry with backoff.
- **Deterministic** (parse failure, type mismatch, gate rejection) → **never retry**. The same bytes
  produce the same verdict; a retry is a slower quarantine.

Idempotency key is provider message id + content hash. This is what makes IMAP re-polling safe
(acceptance test 15).

### 6.2 Emit

The artifact's state column is the durable record — a row in `ready_for_extraction` *is* the pending
work. Publish on the existing in-process `EventBus` for prompt handoff, and register a sweep on
`backend_scheduler.register_job` to pick up anything an in-process publish missed. The `EventBus`
(`src/services/event_bus.py:13`) is synchronous and non-durable, so it cannot be the only record; a
lost event would mean a document that arrived, passed every gate, and quietly never got read — a
silent skip, which the non-negotiables forbid.

---

## 7. Checks

`bp_check_definition` carries `disposition` — `finding`, `quarantine`, or `human_review` — plus
`enabled` and a disabled-reason. Disposition is data, not code, so the fail-closed set is auditable
by reading a table.

### 7.1 The nine

| Check | Condition | Disposition | Ships |
|---|---|---|---|
| Sender authentication failure | SPF, DKIM or DMARC fails, or alignment fails | finding | **enabled** |
| Type mismatch | declared type/extension disagrees with detected | finding inside alias set; quarantine outside | **enabled** |
| Hidden text detected | text in content stream not visibly rendered | **human_review** (quarantines) | **enabled** |
| Reply-To divergence | Reply-To domain differs from From domain | finding | **enabled** |
| Unresolvable reference attachment | link-type attachment unresolvable within the binding | quarantine | **enabled** |
| Unknown sender domain | domain matches no supplier record | finding | disabled — §7.2 |
| Domain near-match | within edit distance of a known supplier domain | finding | disabled — §7.2 |
| First contact from new domain | established supplier writes from an unseen domain | finding | disabled — §7.2 |
| Bank detail change | payment instructions differ from supplier record | **human_review** | disabled — §7.2 |

Hidden text and bank-detail change fail closed to human review and may not be auto-cleared by any
automated path.

**Where the bank check sits.** It needs extracted payment details, so it runs *after* extraction,
gating the path from validated output to any action — not gating ingestion. Same fail-closed force,
different position in the pipeline.

### 7.2 Four checks ship disabled, with reasons

Verified against live bp_sqldb on 2026-08-06:

| Fact | Value |
|---|---|
| `bp_supplier` rows | 5,028 |
| `website_url` populated / **distinct** | 5,000 / **288** |
| Sample domains | `copperleafsystems.example`, `vantagesupplies.example` — reserved TLD, cannot appear in real mail |
| `bank_iban` populated / distinct | 5,000 / 5,000, all of form `IBAN<digits>` |
| Sample bank names | `Crossfell Bank`, `Ironbridge Bank` — invented |
| `draft_rfq_emails` / `workflow_email_tracking` rows | 1 / 8 |

The supplier master is seeded test data. A bank-detail check comparing a real invoice against
`IBAN1720767165` produces confident output from fiction, which is worse than no check. These four are
registered with full fixture coverage and `enabled = false`, each with its activation prerequisite
recorded in the row. They switch on when real reference data arrives — a config change against tested
code, no code change.

**`bp_supplier_domain` is how they switch on.** "First contact from a *new* domain" is meaningless
without a record of old ones, so that table is required regardless. It starts empty, accumulates from
observed traffic, and a human confirmation promotes an observed domain to a known one — the same
pattern already used for supplier alias resolution.

### 7.3 Supplier binding is ambiguous by default

288 distinct domains across 5,028 suppliers means domain-to-supplier binding is ambiguous constantly,
not occasionally. Stage 4's `ambiguous` outcome is the common path in this dataset and is designed as
a normal, stored, downstream-visible state — never resolved by picking the closest match.

### 7.4 Human review queue

`proc.bp_approval` already exists and already carries a `finding_id` column. Fail-closed findings
raise a row there. Bank-detail change presents both values — recorded and observed — side by side.

---

## 8. Injection containment

1. **No tool access in the extraction context** — §2.1.
2. **Structured output only.** Anything not conforming to schema is discarded whole; no repair
   attempted.
3. **Deterministic validation before influence.** No extracted value reaches a Finding without
   passing type, range, unit and cross-field consistency checks. Out-of-bounds values are
   quarantined, not clamped.
4. **No privilege inference from content.** Nothing in a document alters approval state, supplier
   standing, risk classification or scoring weights.
5. **Detect the attempt** — §8.1.
6. **Provenance isolated from content** — §2.1 item 4.

### 8.1 Hidden text, and the OCR false-positive trap

Scope: PDF gets invisible render mode (`Tr 3`), zero/low alpha, colour-on-background contrast,
off-page bounding box, sub-threshold font size, and hidden optional-content layers. Office formats
get hidden runs (`<w:vanish/>`), white fonts, and hidden rows/columns/sheets. True z-order occlusion
analysis is deferred — lowest yield, highest false-positive risk on documents that legitimately use
white boxes for layout.

**The trap.** Scanned documents carry an invisible OCR text layer rendered in `Tr 3` — exactly the
"text present in the content stream but not visibly rendered" signal. It is legitimate and, in a
procurement corpus full of scanned invoices, extremely common. A naive detector would quarantine
nearly every scanned document and the review queue would be useless within a day.

**The discriminator.** An OCR layer sits directly over a raster image covering the same region. The
rule is therefore *invisible text not backed by an underlying image region* — not *invisible text*.
This is the most likely way this feature fails in production, so it gets fixtures on both sides: a
real scanned invoice that must pass, and an injection that must quarantine.

No new dependencies required. `PyMuPDF`, `python-magic`, `olefile` and `openpyxl` are all present.
`python-magic` is importable but absent from `requirements.txt` and will be pinned.

---

## 9. Observability

Metrics: bundles received, artifacts ingested, quarantine count by reason, duplicate rate,
sender-auth failure rate, unmatched-domain rate, extraction validation failure rate, per-adapter sync
lag.

Logging at bundle granularity, correlation key = `Message-ID`. **Never** attachment content,
credentials or full body text; addresses redacted in non-production. Enforced in `repo.py` at the
write boundary rather than trusted to callers.

Alerts: sustained adapter failure, quarantine rate above baseline, any bank-detail-change firing.

---

## 10. Testing

All 17 acceptance cases get committed raw-MIME fixtures under `tests/fixtures/mail_intake/`:
single-attachment provenance; two-level forwarded nesting; CID logo filtered; non-CID inline document
ingested; four forwards → one object and four arrivals; `.pdf` over ZIP quarantined; password-protected
PDF; macro workbook; SPF-failing known supplier; near-match domain; zero-opacity injection; bank-detail
divergence; out-of-tenant reference attachment; malformed MIME; IMAP UID re-read; Graph streaming
threshold; body validity window plus attachment pricing in one bundle.

Plus the §2.1 import-guard test.

**Stated gap.** Tests 9, 10 and 12 exercise checks whose live reference data is fiction (§7.2). They
prove the logic is correct against fixtures. They do **not** prove the checks are useful in
production, and must not be reported as if they do.

---

## 11. Out of scope

Attachment content extraction (exists), supplier record management (exists), Finding presentation UI
(exists), outbound email, calendar and contact access. No mail scopes beyond read on the configured
folder. Process-level plane separation (§2.2). Z-order occlusion analysis (§8.1). Gmail adapter
beyond its stub (§4).

---

## 12. Deliverables

1. `MailSource` plus relay, Graph and IMAP adapters; Gmail stubbed.
2. Schema migrations for §5 under `scripts/migrations/`.
3. Deterministic pipeline implementing §6 with explicit state transitions.
4. Nine checks registered per §7, five enabled and four disabled with recorded reasons.
5. Test suite per §10 with committed MIME fixtures.
6. Configuration documentation covering required scopes per adapter, written for a customer's IT
   reviewer rather than for an engineer.
