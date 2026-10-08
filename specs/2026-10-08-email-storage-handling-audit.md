# Email storage and handling: coverage audit

Written 2026-10-08. **A report only: nothing in it was built.** It answers the audit you asked for ("check the codebase against each
item; for each, Covered / Partial / Missing, where it lives, what is needed"). Every claim below was read from the code or from a
read-only count on the databases, not recalled. Where I did not look, it says so.

Status words: **Covered** (does what the item asks), **Partial** (some of it, or only in one place), **Missing** (nothing does it).
"Built, not applied" means the code is on `origin/Development` but its migration has not been applied anywhere.

## The most important findings first

1. **The supplier's price is a claim that has become a fact.** The inbound analyser (`supplier_interaction_agent._parse_response`) takes the
   first number in the email body by regex as the price, replaces it with whatever a model returns if the model returns one, and
   writes it to `proc.supplier_response.price` with no record that it was extracted, by what, or with what confidence. The email
   assurance layer then reads `supplier_response.price` as the Postgres fact `supplier_current_offer`. So a figure extracted from
   an email is presented, and checked against, as if it were product data. This is acceptance criterion 2 of the build prompt, and
   it is **not met**; fixing it means changing a product table, so it needs your ruling.
2. **Nothing checks who an inbound email is from.** There is no SPF, DKIM, DMARC or sender-authentication check anywhere in `src`.
   An inbound email is matched to a dispatch by `unique_id`, thread headers and supplier, with a match score. A spoofed reply
   carrying a valid id can therefore write a price into `supplier_response`.
3. **No detection of bank-detail change requests or of injection attempts exists on the inbound side.** A search of `src` for either
   found nothing outside the new drafting layer. The outbound side now flags bank details in a draft (shadow mode); the inbound side,
   where the fraud pattern actually arrives, does nothing.
4. **Inbound bodies are stored raw, with no retention.** `proc.supplier_response` keeps `response_text`, `response_body` (HTML) and the
   raw headers; the IMAP watcher keeps a second copy in `proc.supplier_responses`; SES delivers the raw `.eml` to S3
   (`s3://procwisemvp/emails/...`). Only two local capture directories have an age-out (`capture_retention.py`).
5. **Volume is near zero.** Read-only counts: production (`bp_sqldb`) has **1** inbound reply ever (July 2026), **18** drafts, **0** sent,
   **0** tracked dispatches. `bp_testdb` has 7 inbound replies, 44 drafts (43 in the last 30 days, almost all test activity), 8 tracked
   dispatches. See "Scope check".

## Sources of truth

| item | status | where | what is needed |
|---|---|---|---|
| Business facts (PO, dates, amounts, terms, contacts) come from product Postgres tables | **Partial** | The wrapped drafting paths read facts from Postgres only (`draft_assurance/facts.py`), with table, column and row id recorded | Finding 1: the offer fact is itself extracted from an email. Everything else the layer reads (supplier contact, lead time, currency) is as good as its writer |
| The mailbox is the original record of email content | **Partial** | Raw `.eml` in S3 via the SES ingest Lambda (`email_ingest_lambda.py`); IMAP watcher and SES watcher copy bodies into `proc.supplier_responses` / `proc.supplier_response` | The bodies are COPIED into product tables rather than pointed to; there is no single "this is the original" pointer, and the S3 copy has no documented retention |
| `email_agent` tables are a derived index that can be rebuilt from the mailbox, with a rebuild script and a test | **Missing, and the premise does not match what exists** | `email_agent` holds the OUTBOUND side: the model's draft, the facts and checks each draft rested on, the outcome, the sent text and diff, the learning queues | That data is primary, not derived: the model's draft text, the assurance record and the reviewer's edits exist nowhere else and cannot be rebuilt from a mailbox. A rebuild script only makes sense for an INBOUND-derived index (extracted structure), which does not exist. Your ruling is needed on which of the two you mean |

## What to store

| # | item | status | where | what is needed |
|---|---|---|---|---|
| 1 | Message metadata (message id, thread id, sender, recipients, timestamp, subject, pointer to body) | **Partial** | `proc.supplier_response`: `response_message_id`, `original_message_id` (the thread anchor), `raw_headers` (carries In-Reply-To/References), `response_from`, `response_subject`, `received_time`, `match_confidence`/`match_score`/`matched_on`. `workflow_email_tracking` for dispatch side | Exists, but in a product table, mixed with the body copy rather than a pointer to it. Copy of the body in `email_agent` only where the learning job needs it: **done for SENT text** (writer-only, 90-day governed), not applicable to inbound |
| 2 | Extracted structure (intent, questions, requests, deadlines, entities) with source message id, model + prompt version, confidence, status `extracted_unverified` | **Missing**, with one partial piece | `email_intent.classify_reply` returns an intent for a reply; the inbound analyser extracts only price, lead time and a free-text summary (`context_summary`) | No provenance columns anywhere (no model, prompt version, confidence or status). No questions/requests/deadlines/entities extraction |
| 3 | Agent state (draft, approved/sent, reviewed_by/sent_by, status per open ask) | **Partial** | Draft: `proc.draft_rfq_emails`. Approval: `proc.bp_approval` (named approver, content hash). Reviewed-by/sent-by and outcome: `email_agent.bp_draft_outcome`. Abandoned: sweep (built) | **Status per open ask is Missing**: nothing models an ask (a question put to a supplier) as a thing with a state |
| 4 | Outcomes: link a reply to the open asks it resolves | **Missing** | What exists: a reply is matched to a DISPATCH (`workflow_email_tracking.responded_at`, `response_message_id`; `supplier_response.matched_on`) by hidden marker / `X-ProcWise-Unique-ID` / thread headers / supplier | See the design below. Not built, as you ordered |

**Design proposal for item 4 (not built).** Make an *ask* a row: `email_agent.bp_ask (ask_id, capture_id, workflow_id, supplier_id, kind
[price|lead_time|document|confirmation|other], text, status [open|answered|withdrawn|lapsed], opened_at, due_by)`, created from the
brief's `explicit_ask` when a draft is SENT (so it exists only for emails that actually went). A reply is linked to asks in two steps:
(1) to the dispatch, as today (marker / headers, which is deterministic); (2) to the asks of that dispatch by a proposed link with a
confidence, **never auto-closing**: a person confirms `answered` (the same confirm pattern as the assumption items). Reply-derived
values stay `extracted_unverified` claims on the link, not facts. Build after the provenance work below, because linking is only as
trustworthy as the extraction under it.

## Not storing bad information

| item | status | where | what is needed |
|---|---|---|---|
| Extracted values are claims, never facts; never write to or override a Postgres fact; differences recorded as a conflict and shown at drafting | **Missing for the inbound writer; Covered inside the drafting layer** | Drafting: a caller-supplied value that differs from Postgres is recorded as a conflict and Postgres wins (`assure.prepare_inputs`). Inbound: `_store_response` writes extracted price/lead time straight into `supplier_response` | Finding 1. Needs provenance on `supplier_response` (extracted_by, model + prompt version, confidence, status) and a fact-source rule: an `extracted_unverified` price is shown to the reviewer AS A CLAIM from the email, not as a verified fact. Product-table change: your ruling |
| Extracted fields never become exemplars, style rules or eval ground truth without human review | **Covered** (for what exists) | Every learning route ends in a queue a person decides; a reviewer's replacement for a Postgres-backed fact goes only to the data-quality queue, marked unverified, and is tested to appear in no eval/exemplar/style/classifier table; an exemplar needs a second person | Inbound extracted fields feed none of this today |
| Validate extraction output against a schema; reject and log malformed output rather than store partial fields | **Partial** | The new stages (classify, plan, judge) use strict parsers: malformed output is rejected, recorded `invalid`, never partly stored. The inbound analyser does not: `_coerce_float` on whatever comes back, and the regex first-number fallback | Put the inbound analyser behind the same strict-parse-or-reject pattern, with the rejection logged |
| Never store bank details | **Partial** | `email_agent`: bank details (IBAN, UK sort code, "account number ...") are masked before anything is stored: sent text, diff, the model's draft, and the changed-figure columns (testing found and fixed an account number being stored as a "figure"). Pattern-based, so an unusual format is missed | Inbound bodies in `supplier_response(s)` and the S3 `.eml` are stored raw, bank details included. Needs either masking on ingest of the DERIVED copies, or a deliberate ruling that the mailbox/S3 copy is the one place they may exist, under retention and access control |
| An email requesting new or changed bank/payment details is flagged as a suspected fraud pattern, routed to a human, never drafted against automatically | **Missing (inbound)**; **Partial (outbound)** | Outbound only: the `bank_details` pattern in every family fails a draft that contains bank details (shadow mode: recorded, not blocked) | Finding 3. An inbound detector (pattern-based, no model needed: change/new/update + bank/account/IBAN/sort code/payment details, plus a sender-mismatch signal) that stamps the reply `suspected_payment_fraud`, routes it to a person, and makes the drafting layer refuse to draft a reply against it. Cheap and deterministic |

## Untrusted content and prompt injection

| item | status | where | what is needed |
|---|---|---|---|
| Treat all email content as untrusted data; pass extracted fields as clearly delimited data, never raw bodies as instructions | **Partial** | The supplier's message reaches a model in two places: the inbound analyser (`message` inside a JSON payload, system prompt states the task) and the negotiation drafter (`supplier_message` and a summary inside the JSON `Context` block). Steering text, which I added, IS delimited and labelled as data | The inbound text is data inside JSON, which helps, but there is no "this is untrusted" framing, no length cap, no stripping of quoted history, and the drafter receives the supplier's own words rather than extracted fields |
| Instructions inside emails never trigger an action, change recipients or change agent behaviour | **Partial, by structure; not tested** | The LLM output on the inbound path is used only for fields (price, lead time, summary); it does not choose tools. The decision engine acts on a classified reply (`decide_email_reply`, `act_on_email_reply`); I did not trace what it can cause | A targeted test per inbound consumer: an email saying "ignore previous instructions and ..." changes nothing it must not. I did not examine the decision engine's reply actions in depth |
| Detect likely injection and flag it on the draft | **Missing** | | A heuristic detector on inbound text (instruction-like imperatives aimed at the assistant, "ignore previous", role-play markers, hidden/zero-width text, URLs asking to forward) that records a flag on the reply and on any draft written against it |
| Recipients come only from the Postgres supplier contact record, never from email content | **Covered for sends; one internal exception** | Draft recipients come from the supplier master (`_master_contact`); the send guard refuses any recipient not on the supplier master (check 2); the assurance layer records `recipient_not_on_master`. Tested | Human-written and manual drafts take caller recipients but are stopped at send by the same guard. The weekly value digest takes recipients from the `VALUE_DIGEST_RECIPIENTS` environment variable (internal mail, on the sending domain) |
| Injection, spoofed-sender and bank-change fixtures in the golden eval set | **Missing** | The golden set (56 cases) covers drafting only. The labelling set has 5 adversarial REQUESTS (outbound side) | Inbound fixtures need an inbound path under evaluation, which does not exist yet. Cheap first step: unit fixtures for the detectors above |
| (found) Sender authentication | **Missing** | | Finding 2. At least record the SPF/DKIM/DMARC result from the headers SES adds, and treat a failing or missing result as a reason to hold the reply for a person |

## Privacy and retention

| item | status | where | what is needed |
|---|---|---|---|
| Access controls for the `email_agent` schema (who reads raw bodies vs derived fields) | **Built, not applied** | Two NOLOGIN roles: the reader has no access to `email_agent`; the writer alone touches the raw-text table (`bp_draft_sent_text`) and may delete only from it; PUBLIC has nothing. 57+ tests connect AS the roles. API returns no raw text except one separately gated call (exemplar text, `configure` class) | Apply through the DBA change request. Until the dedicated logins exist the app's own login is used, so the restriction is not in force. Raw INBOUND bodies sit in product tables readable by the app login |
| Retention: one configurable policy for bodies, sent text and diffs (default 90 days raw, derived longer) | **Partial** | `EmailTextRetention.raw_text_days = 90` governs sent text, diffs and the model's draft text; a scheduled purge (ON by default) deletes/blanks them, keeping derived values; no period set = no raw text stored | Does not cover the inbound bodies (`supplier_response(s)`), the S3 `.eml`, or `data/conversations` beyond `capture_retention.py`'s two local directories. "Derived features longer" is true but unbounded: no derived retention period exists |
| Redact or exclude attachments, signatures and sensitive personal data before storage; list what and how | **Partial**; list below | Bank details only | See the redaction list |

**Redaction list (what I would redact, and how).** For the DERIVED stores (`email_agent`, any extraction index); the original stays only in
the mailbox/S3 under retention.

| what | how | status today |
|---|---|---|
| Bank details: IBAN, sort code, account number | Pattern mask before storage | Done in `email_agent` |
| Card numbers (Luhn-valid 13-19 digits), national insurance / tax / passport / ID numbers, dates of birth | Pattern mask | Not done |
| Attachments | Never stored in `email_agent`; store name, type, size and a hash only; the file stays in S3 | `email_agent` stores none; inbound `attachments` JSONB holds metadata (not re-read in this audit) |
| Signature blocks (names, titles, phones, addresses, disclaimers, tracking footers) | Cut at the signature delimiter / sign-off heuristic before extraction; not stored | Not done. Sent text is stored as sent, so our own signature is in it |
| Quoted thread history | Strip before extraction and before storing a body copy; the mailbox keeps it | Not done. Stored bodies include whatever the draft body carried |
| Third-party personal data in free text (names, mobile numbers, personal emails) | Mask phone/email patterns; names need a model, which I would not rely on | Not done. `style/redaction.py` exists but over-redacts figures and would destroy a diff |
| Hidden content (zero-width text, HTML comments other than our marker) | Strip before any model sees it; also an injection signal | Not done |

## Human approval: every outbound path, and whether a human approves it

| path | human approval? | notes |
|---|---|---|
| RFQ batch, and every supplier draft, via `EmailDispatchService.send_draft` | **Yes, per draft** | The guard requires a recorded approval by a named human, bound to the exact content by hash (an edit after approval is refused); the live policy denies on mismatch and `auto_reply_intents` is empty. **There is no single batch approval with a recipient preview**: each draft is approved on its own. A batch preview would be new |
| Agent-initiated negotiation drafts | **Yes** | Same guard. An agent with no approval is refused unless policy lists an intent as autonomous (none is); `hitl_auto_approve` in a payload is read, ignored and recorded as an attempted bypass |
| `POST /workflows/email` (the send route) | **Yes** | A signed-in person is required and the send goes through the same guard |
| `value_query_service.send_query` (a finding's query to a supplier) | **A human acts, but no approval row** | The caller is a named principal; the recipient allow-list, content sensitivity and the `email.send` policy are checked; it sends the reviewed draft. It does not use the `bp_approval` content-hash binding |
| `value_digest` (weekly internal summary) | **No** | Scheduled job; recipients from an environment variable; sends as a configuration-built identity gated only by the `email.send` policy. Internal mail (a domain check), but no human approves a given send |
| `support_agent` escalation | **No** | Automatic internal email to `SUPPORT_ADMIN_EMAIL` (default is a personal address; skipped if not on the sending domain) containing the user's message and the assistant's reply |
| Decision engine acting on a classified inbound reply | **Not examined** | I did not trace whether it can cause a send |

Paths where "nothing goes out without a human approving it" is **not** true: `value_digest`, `support_agent`, and (arguably) `send_query`.
The supplier-facing paths are all human-approved except `send_query`, which is human-initiated and gated but not content-bound.

## Scope check: is the full extraction pipeline justified now?

**Recommendation: no. For this phase, keep message ids, thread ids and statuses plus the mailbox's own thread history, and spend the effort on four small safety items instead.**

Reasons:

1. **There is almost nothing to extract from.** Production has one inbound reply ever and no tracked dispatch; even the test database has seven. A pipeline (model, prompt versions, confidence, a verification workflow, a rebuild script) built for that volume would be tested on fixtures and its quality could never be measured on real mail.
2. **The existing extraction is the weakest link, and it is cheaper to make safe than to replace.** Taking the first number in an email as the price, and storing it as a fact with no provenance, is a data-quality and fraud risk today, at any volume.
3. **The things that matter most do not need extraction:** sender authentication, a payment-detail-change detector, an injection flag, and provenance on what is already extracted are all deterministic or near-deterministic. They protect the one inbound reply that will matter.
4. **You cannot build an honest rebuild-from-mailbox test without a mailbox of real mail.**

What to do instead (in this order, each small): (1) provenance columns on `supplier_response` and the fact-source rule that an
unconfirmed extracted price is shown as a claim; (2) record sender-authentication results and hold failures; (3) the inbound
payment-detail-change detector, routing to a person and refusing to draft against it; (4) the injection flag; then (5) retention for
inbound bodies, and only then (6) revisit the structure pipeline and the ask/outcome linking above when real inbound volume exists.

## What I need from you

1. Finding 1: may I add provenance columns to `proc.supplier_response` and change how the offer fact is presented? (Product table; not mine to change unasked.)
2. Which did you mean by "derived index that can be rebuilt from the mailbox": an inbound index (which does not exist), or the outbound capture (which cannot be rebuilt)?
3. Do you accept the scope recommendation, including not building the extraction pipeline yet?
4. Should `value_digest` and `support_agent` stay as they are, be put behind a human approval, or be listed as accepted exceptions?
5. Raw inbound bodies and bank details: mask on ingest, or accept the mailbox/S3 as the one place they may live, under retention and access control?
