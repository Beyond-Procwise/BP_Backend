# Phase −1 — Existing Email Functionality Inventory and Reconciliation

**Date:** 2026-07-27
**Branch:** Development
**Status:** Stop-gate cleared 2026-07-27. Decisions below. Phase 0 may begin.

---

## Decisions taken (2026-07-27)

| # | Decision | Outcome |
|---|---|---|
| **E1** | Send capability | **Keep it.** The SES/RFQ dispatch path stays exactly as it is today. Invariant 2 is re-scoped to the style subsystem: *no send method on any `ExemplarSource` or mailbox adapter*, enforced by a test bounded to `src/services/style/`. The product claim becomes "we never send from your mailbox — drafts are written into it and nothing leaves it", not "the platform cannot send". |
| **E2** | Content-bearing style artifacts | No decision needed — nothing customer-derived exists to migrate. |
| **E3** | Vector store | **pgvector, on the row.** ~~Neither pgvector nor Qdrant~~ — *reversed 2026-07-27, see below.* `vector(1024)` on both `bp_style_exemplar` and `bp_style_ingest_staging`, HNSW cosine index on the former, using the platform's existing BGE-large model. |
| **E4(b)** | IMAP credentials in `.env` | **Deferred to Phase 5**, when `credential_ref` becomes real. Recorded as a known invariant-8 gap in existing code, not introduced by this work. New tables enforce the ARN shape at the DB level. |
| **E5** | Identity | **`user_ref TEXT` holding the Cognito `sub`.** No `persona_id`. The word "persona" is not used in any new table — it already means "report voice" here. |
| **§2** | Assumption corrections | **All seven accepted** — no `tenant_id`, `proc.bp_*` naming, `deploy/sql` migration pairs instead of Alembic, `bp_admin_config` for settings, identity PKs instead of `gen_random_uuid()`. |
| **§3.1** | Draft storage | **Extend `proc.draft_rfq_emails`** with `style_*` provenance columns rather than create a second draft table. |
| **§3.5** | Intent vocabulary | **Seed the merged 13 codes, plus an `_all` sentinel.** The **user-level profile (`intent='_all'`) is the primary artifact**; per-intent profiles compile only where an intent genuinely reaches `min_exemplars`. This keeps level 0 the normal case so the fallback banner stays meaningful. |
| **Scope** | Phasing | **Build Phases 0–4 and stop.** Mode C (Phases 5–7) is deferred: no Graph/OAuth foundation exists, and it is the majority of the work. `bp_mailbox_binding` is created now so the schema is coherent, and stays empty. |

E1 was the user's call. The remainder were taken on the recommendations in §1–§4 of this
document; all are reversible at this stage, since only the migration has been written.

### E3 reversed — embeddings are in, and in Postgres

The original recommendation to drop embeddings was wrong, and is superseded.

The argument for dropping was that retrieval "order by proximity, limit 3" over a set the
spec caps at 2–3 exemplars ranks three rows and returns three rows. That much still holds
— but it only covers *selecting among exemplars under Mode A*. It ignored every other job
the vector does:

* **intent classification** by nearest neighbour, instead of an LLM call per email;
* **Mode C selection**, where a bound mailbox offers dozens of candidates and three must
  be chosen — the case the spec's retrieval clause was actually written for;
* **near-duplicate detection**, so five pasted emails that are really one email five times
  do not compile into a profile claiming five exemplars;
* **Phase 7 divergence**, which is a distance between what was drafted and what was sent.

**Store: pgvector, not Qdrant** — deliberately against the codebase norm, and for one
reason: *lifecycle*. These vectors are derived from correspondence, so they must die
exactly when the row they came from dies. As a column that is automatic and transactional
— the staging purge in invariant 9 takes the vector with it in the same `DELETE`. In
Qdrant it is a second delete against a second service, and a delete that silently fails
leaves a lossy copy of a customer's email in a vector database after we told them it was
purged. For a subsystem whose entire premise is a provable data lifecycle, that is worth
one extension.

Dimension is **1024**, reusing `BAAI/bge-large-en-v1.5` (`config/settings.py:365-366`).
`embedding_model` is recorded per row: a profile embedded under one model and retrieved
under another returns confident nonsense, and the failure is silent.

Verified on live `bp_sqldb`: extension installs (0.7.3), HNSW index builds, and cosine
retrieval returns three distinct distances nearest-first (0.0000 / 0.2929 / 1.0000).

---

## 0. Executive summary

This repo is not a greenfield for email. It contains a large, live, orchestrated
email subsystem: an RFQ drafting agent (4,955 lines), an SES-backed sending
service with IAM credential rotation, an IMAP reply watcher, a supplier-response
matcher, and a negotiation-round loop. Email drafting and email *sending* are a
shipped product feature reachable from the public API.

Five escalations are raised. **Escalation E1 is blocking and is a product
decision, not a refactor**: the platform can already send mail, so invariant 2
("the platform holds no send capability") is false at the system level today.

Separately, **five of the six stated build assumptions are wrong for this repo**
(Alembic, SQLAlchemy, pgvector, multi-tenancy, and Secrets Manager usage). Those
change the Phase 0 deliverable materially and are listed in §2.

---

## 1. Escalations — resolve before Phase 0

### E1 — Existing send capability (BLOCKING, product decision)

The platform sends email today, in production code, on a live path.

| Layer | Location | What it does |
|---|---|---|
| Transport | `src/services/email_service.py:55` `EmailService.send_email` | Sends via SES SMTP (`smtplib`), with auth-failure fallback and retry |
| Credential rotation | `src/services/email_credentials_manager.py:41` `SESSMTPAccessManager.rotate_smtp_credentials` | Rotates the SES IAM access key, derives the SMTP password (`SendRawEmail` signature), writes it back to Secrets Manager |
| Orchestration service | `src/services/email_dispatch_service.py:375` | Calls `email_service.send_email` for each approved draft |
| Agent | `src/agents/email_dispatch_agent.py:22` `EmailDispatchAgent` | *"Send supplier drafts via SES and register dispatch metadata"* |
| Registration | `agent_definitions.json:402`, `src/api/main.py:47,125` | Registered as slug `email_dispatch` |
| Orchestrator | `src/orchestration/orchestrator.py:2000` | Executes `email_dispatch` as a workflow step |
| Public API | `src/api/routers/workflows.py:1257` `POST /workflows/email` | *"Send a previously drafted RFQ email using the dispatch service"* |
| Second caller | `src/services/support_agent.py:588` | Emails the support admin |

This is **not dead code**. It is the centre of the sourcing loop: draft RFQ →
send to suppliers → IMAP watcher reads replies → responses matched → negotiation
round → send again. Removing it removes the product.

**Consequence.** The claim *"the platform cannot send on your behalf"* cannot be
made about this system as it stands. The decision is one of:

1. **Drop the claim** and restate invariant 2 as a *scoped* invariant — e.g.
   "the style-learning subsystem holds no send capability; its adapters expose no
   send method" — and enforce it with a test bounded to the new module. The
   existing SES path stays, and the product position becomes "we send only what
   you approved, from our own verified sending identity — we never send from your
   mailbox." Note this is a genuinely different claim from the one in the brief.
2. **Remove the SES send path** and hand sending back to the customer's mail
   client (draft write-back only, Phase 6). This deletes the RFQ dispatch,
   negotiation-round automation and support-alert features, or requires them to
   be rebuilt as draft-only.

There is a useful precedent for option 1 already in the repo:
`tests/api/test_email_prepare_endpoint.py:166`
`test_prepare_email_draft_never_touches_the_send_path` monkeypatches both
`EmailDispatchService` and `EmailService.send_email` to raise, and asserts the
*prepare* handler never reaches them. That is exactly the shape a scoped
invariant-2 test would take.

**Recommendation:** option 1. Option 2 destroys shipped functionality to satisfy
a claim that a narrower, still-honest claim would satisfy.

---

### E2 — Existing style artifacts that contain content

No prior style-guide *extraction* exists — nothing in this repo reads a customer's
emails and derives a written style. So there is no customer-derived content to
migrate. But there are three content-bearing artifacts that the new design's
invariant 1 would forbid if they were treated as profiles:

| Artifact | Location | Content |
|---|---|---|
| Hardcoded exemplar sentences | `src/agents/email_drafting_agent.py:122,150,2713` and surrounding | Fully-worded negotiation lines, e.g. *"We've reached our maximum flexibility at £X with Net-15 payment terms… please confirm by close of business Thursday"* — baked into the module as prompt text |
| Literal RFQ email template | `prompts/EmailDraftingAgent_Prompt_Template.json` | A complete, fully-worded RFQ email body with greeting, T&C checkboxes and closing |
| Governed email prompts | `proc.bp_prompt` rows `email_compose_rfq`, `email_compose_response`, `email_polish`, `negotiation_playbook_system` (type `email_prompt`, agent `email_drafting`) | Instruction text authored by us |

**Assessment.** All three are *platform-authored fiction*, not lifted
correspondence. Under the new design's own vocabulary they are **exemplars**
(`origin='synthetic'`), not profile content. They are safe to keep — but they
must not be allowed to become the seed of `profile_json`, and the compile prompt
must not be pointed at them.

Correspondence that *is* stored, for completeness:

| Table | Rows (live `bp_sqldb`) | Content |
|---|---|---|
| `proc.draft_rfq_emails` | 18, all with non-empty `body` | Full generated outbound email bodies |
| `proc.supplier_response` | 1 | Inbound supplier reply, incl. `body_html` |
| `proc.workflow_email_tracking` | 0 | Metadata only (message ids, headers) |

These are outbound-generated and inbound-supplier, i.e. neither is "the
customer's own writing" that invariant 1 protects. No migration is required.
They are, however, a candidate exemplar source for Mode C and should be treated
as such deliberately rather than by accident.

---

### E3 — Embedding dimension and vector store

| Fact | Value | Source |
|---|---|---|
| Embedding model | `BAAI/bge-large-en-v1.5` | `config/settings.py:365` |
| Dimension | **1024** | `config/settings.py:366` `vector_size: int = 1024` |
| Vector store | **Qdrant**, not pgvector | `requirements.txt` (`qdrant-client`), `src/services/rag_service.py:62` |
| Installed PG extensions | `plpgsql` only | live query against `bp_sqldb` |
| `vector` extension | available 0.7.3, **not installed** | `pg_available_extensions` |

The proposed `vector(1024)` **dimension is correct**, but the **store is wrong**.
This repo has no pgvector column anywhere and no `pgvector` Python package. The
DDL as written would introduce a second vector store alongside Qdrant.

**Proposal:** keep 1024; store exemplar embeddings in Qdrant using the existing
`DocumentEmbeddingService` / `RAGService` collection pattern, and drop the
`embedding` column and the HNSW index from `style_exemplar`. If a Postgres-native
store is genuinely wanted, that is a platform decision (and needs
`rds_superuser` to `CREATE EXTENSION vector` on the Aurora cluster) — it should
not be smuggled in as a side effect of this subsystem.

---

### E4 — Existing mail credentials with broader scope than a single mailbox

Two separate problems.

**(a) SES send credentials are account-wide, not mailbox-scoped.** The secret
`ses/smtp/credentials` (`config/settings.py:162`) holds an SES SMTP username and
password derived from an IAM access key whose signature terminal is
**`SendRawEmail`** (`src/services/email_credentials_manager.py:145`). That grants
send-as any verified SES identity on the account — strictly broader than a single
mailbox. Verbatim scope: SES SMTP `SendRawEmail`, plus the rotation path needs
`iam:ListAccessKeys`, `iam:CreateAccessKey`, `iam:DeleteAccessKey`,
`iam:GetAccessKeyLastUsed` and `secretsmanager:GetSecretValue`/`PutSecretValue`.

**(b) IMAP credentials are in plain environment variables.** `config/settings.py:317-326`
defines `IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD`, `IMAP_MAILBOX` (default
`INBOX`) as `.env`-sourced settings. **Invariant 8 — "secrets are never stored in
application tables… never a token or password" — is already violated in spirit by
existing code**, which holds a mailbox password in process config rather than
behind a `credential_ref`.

There is **no Graph, no Gmail, no OAuth, and no Nango** anywhere in the repo.
Mail reading is IMAP-only (`imaplib`), against a single shared mailbox, via
`src/agents/email_watcher_agent.py`, `src/services/imap_supplier_response_watcher.py`
and `src/agents/supplier_interaction_agent.py:2866`. So there is no existing
OAuth scope grant to report — but there is also **no existing Graph client to
build Phase 5 on**; `GraphExemplarSource` is genuinely net-new.

**Consequence for the Mode C scope-restriction position:** the position survives
for *reading* (there is no broad read grant today), but the account-wide SES send
grant weakens any blanket "we cannot touch mail outside the mailbox you bound"
claim. Ties back to E1.

---

### E5 — No persona concept

There is no per-user writer identity in this platform.

`persona` exists, and means something entirely different: a **summary/answer
voice**, selected from `proc.bp_prompt`. Live rows:

```
analysis     summary_persona  summary_agent
compliance   summary_persona  summary_agent
negotiation  summary_persona  summary_agent
joshi        ask_persona      rag           (v5)
```

These are report/answer tones (`src/services/summary_agent.py:39` `resolve_persona`,
`src/api/routers/summary.py`), not people.

Sender identity for email is **derived from the email address local-part**:
`src/agents/email_drafting_agent.py:4119` `_derive_sender_identity` splits
`n.geelen@…` into `"N Geelen"` and hardcodes the title `"Procurement Lead"`.

There is likewise **no user table**. Authentication was added on 2026-07-24 via
Cognito ID tokens (`src/api/auth.py`), so a Cognito `sub` is the only durable
per-user identifier available.

**Proposed mapping — needs approval before the schema is written:**

- Drop `persona_id uuid`. Use `user_ref text` holding the **Cognito `sub`**,
  since that is the only identity the platform actually recognises.
- Keep the *concept* name "persona" out of the new tables entirely, to avoid
  colliding with the existing `summary_persona` / `ask_persona` meaning. Two
  things called persona in one codebase will be misread.
- The existing `_derive_sender_identity` becomes the display-name resolver that
  fills `profile_json.structural.sign_off` at compile time, not a second identity
  system.

---

## 2. Assumption corrections (blocking for Phase 0)

The brief invited these to be corrected. Five of six are wrong here.

| Assumption | Reality in this repo | Evidence |
|---|---|---|
| Python 3.11+, FastAPI | ✅ Correct — Python 3.12.3, FastAPI | `python3 --version`, `requirements.txt` |
| **SQLAlchemy + Alembic migrations** | ❌ **No Alembic at all** — not in `requirements.txt`, no `alembic.ini`, no `migrations/`. Migrations are hand-written idempotent SQL at `deploy/sql/YYYY-MM-DD_name.sql`, with a matching `_rollback.sql`. SQLAlchemy is a listed dep but data access is raw `psycopg2` via `services.db.get_conn`, with a sqlite fallback branch | `deploy/sql/`, `src/repositories/*.py` |
| **PostgreSQL 15 + pgvector** | ⚠️ Postgres on Aurora RDS ✅, but **pgvector is not installed**; vectors live in Qdrant | see E3 |
| Secrets in AWS Secrets Manager | ⚠️ Partly — SES creds yes, IMAP creds in `.env` | see E4(b) |
| **Multi-tenant, `tenant_id` on every row, RLS** | ❌ **No multi-tenancy whatsoever.** Zero `tenant_id` columns in the schema. No RLS. `src/api/auth.py:18` states outright: *"It does not scope retrieval. There is no tenant dimension in the corpus"* | live schema, `src/api/auth.py` |
| Table naming | ❌ Repo convention is **`proc.bp_*`** with `ix_bp_<table>_<col>` indexes. `style_profile`, `style_exemplar` etc. violate it | `deploy/sql/*`, project convention |

**Required amendments to the data model before Phase 0:**

1. **Drop `tenant_id` from all five tables** and every proposed index. There is no
   tenant dimension to scope by; adding one here creates a column that is always
   the same value and a false impression of isolation. If multi-tenancy is coming,
   it is a platform-wide change, not this subsystem's to invent.
2. **Rename** `style_intent` → `proc.bp_style_intent`, `style_ingest_staging` →
   `proc.bp_style_ingest_staging`, `style_profile` → `proc.bp_style_profile`,
   `style_exemplar` → `proc.bp_style_exemplar`, `mailbox_binding` →
   `proc.bp_mailbox_binding`, `style_draft` → `proc.bp_style_draft`. Indexes as
   `ix_bp_style_profile_active` etc.
3. **Replace `persona_id uuid` with `user_ref text`** throughout (E5).
4. **Remove `embedding vector(1024)` and the HNSW index** from `bp_style_exemplar`;
   embeddings go to Qdrant, and the row keeps a `qdrant_point_id` instead (E3).
5. **Deliver migrations as `deploy/sql/2026-XX-XX_bp_style_engine.sql` +
   `_rollback.sql`**, not Alembic revisions. "Applies and rolls back cleanly" is
   then tested by running the pair, which is the existing repo practice.
6. **Config**: `deployment_mode` and `min_exemplars` belong in
   `proc.bp_admin_config` (key/JSONB store, `sql/bp_admin_config.sql`), which is
   exactly this pattern already. Not a new table, and not per-tenant.
7. `gen_random_uuid()` requires `pgcrypto`, which is **not installed** either
   (only `plpgsql`). Either add it to the migration or use
   `BIGINT GENERATED ALWAYS AS IDENTITY`, which is what every existing `bp_` table
   does.

---

## 3. Full inventory with reconciliation decisions

### 3.1 Data model

| Finding | Location | In use? | Touches | Decision |
|---|---|---|---|---|
| `proc.draft_rfq_emails` — 21 cols incl. `subject`, `body`, `payload jsonb`, `sent`, `unique_id`, `workflow_id`, `mailbox`; 18 live rows | DDL inline at `src/agents/email_drafting_agent.py:4849`; repo `src/repositories/draft_rfq_emails_repo.py` | **Live** | Direct collision with `style_draft` | **Extend.** This *is* the draft store. Rather than a parallel `bp_style_draft`, add the provenance columns the design needs (`style_profile_id`, `style_profile_version`, `fallback_level`, `exemplar_ids`, `model_id`, `prompt_template_version`, `exemplar_set_hash`, `retrieved_at`, `external_draft_ref`) to `draft_rfq_emails`. A second draft table is invariant 10 by another name. |
| `proc.workflow_email_tracking` — dispatch/response correlation, 0 rows | `src/repositories/workflow_email_tracking_repo.py` | Live (empty) | Overlaps `style_draft.message_ids` | **Coexist.** Different purpose: it correlates a *sent* message to its *reply* for the negotiation loop. Not a drafting concern. |
| `proc.supplier_response` — inbound replies incl. `body_html`, 1 row | `src/services/imap_supplier_response_watcher.py` | Live | Potential Mode C exemplar source; E2 | **Coexist** for now. Flag: if it later feeds exemplar compilation, that is a Mode C read and must be governed by a `bp_mailbox_binding`, not read directly. |
| `proc.bp_prompt` / `proc.bp_policy` — governance store with `version`, `prompts_status`, `prompt_type`, `prompt_linked_agents` | `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql`; resolver `src/agents/base_agent.py:323` `resolve_prompt` | **Live** | `prompt_template_version` in `style_draft` | **Extend.** This is the template registry. `prompt_template_version` should be `bp_prompt.version` for the row that produced the draft — do not fork a second versioning scheme. |
| `proc.bp_admin_config` — key/JSONB config | `sql/bp_admin_config.sql` | Live | `deployment_mode`, `min_exemplars` | **Extend.** Add two keys. No new config table. |
| `proc.bp_extraction_template` | extraction pipeline | Live | Name collision only | **Coexist.** Unrelated (document extraction templates). |

### 3.2 Style logic

| Finding | Location | In use? | Touches | Decision |
|---|---|---|---|---|
| **No style extraction exists.** No tone analysis, no voice matching, no "write like the user" prompt anywhere in the repo | — | — | The whole design | **Net-new.** This subsystem has no predecessor to supersede. |
| `enforce_response_style()` — strips markdown headers/rules/blockquotes/decorative emoji from `/ask` answers | `src/services/response_style.py` | **Live** (`0697ab5`) | Superficially "style" | **Coexist.** It is a post-hoc *formatting* stripper for chat answers, not a writing-habit model, and it is deliberately content-preserving. Different subsystem, different output surface. Not invariant 10: it never produces an email. |
| `_interaction_tone_prefix()` — maps a tone word to a canned opening line (`"friendly"` → *"I hope you are well."*) | `src/agents/email_drafting_agent.py:3079` | Live | `profile_json.register`, `lexical.banned_phrases` | **Supersede.** This is a four-branch hardcoded tone model, and it emits precisely the kind of phrase the design lists as a *banned phrase* example. Replace with `profile_json.register` + `lexical`. Delete on Phase 4 cutover. |
| `_calculate_tone_guidance(round_no, gap_pct)` and the negotiation strategy tone table (`collaborative` / `assertive` / `firm` / `analytical` / `urgent`) | `src/agents/email_drafting_agent.py:403`; `src/engines/negotiation_strategy_engine.py:43-143` | Live | `behavioural.escalation_ladder` | **Extend.** The escalation ladder in `profile_json` should *consume* the strategy engine's round-based tone rather than duplicate it. The engine decides *how firm this round is*; the profile decides *what firm sounds like in this person's voice*. |
| `procurement_workflow.py:439-447` — per-round tone strings (`"Inquisitive, professional"`, `"Confident, data-driven"`, `"Decisive, respectful but firm"`) | `src/orchestration/procurement_workflow.py` | Demo path (see 3.6) | Duplicate of the above | **Supersede** with the same removal as 3.6. |

### 3.3 Mail integration

| Finding | Location | In use? | Touches | Decision |
|---|---|---|---|---|
| SES SMTP send | `src/services/email_service.py`, `email_credentials_manager.py` | **Live** | **Invariant 2 — E1** | **Escalated.** No decision taken. |
| IMAP reader (`imaplib`, IDLE loop, single shared `INBOX`) | `src/agents/email_watcher_agent.py`, `src/services/imap_supplier_response_watcher.py`, `src/agents/supplier_interaction_agent.py:2866` | **Live** | Phase 5 `ExemplarSource` | **Extend.** Implement `ImapExemplarSource` against the Phase 2 protocol reusing this connection code. Note `mailbox_binding.provider` already allows `'imap'` — good, but `'graph'` and `'gmail'` have no implementation and should not be seeded as if they do. |
| SES inbound → SQS → Lambda ingestion | `src/services/email_ingest_lambda.py`, `email_sqs_loader.py`, `docs/ses_inbound_pipeline.md` | Live | Mode C read path | **Coexist.** A second, push-based read route. If it ever feeds exemplars it must go through a binding. |
| No Graph / Gmail / OAuth / Nango | — | — | Phase 5 | **Net-new.** `GraphExemplarSource` has no foundation to build on. Budget accordingly. |
| Credentials: IMAP password in `.env` | `config/settings.py:317-326` | Live | **Invariant 8 — E4(b)** | **Supersede.** Move behind a `credential_ref` → Secrets Manager before any binding-based read is claimed to be governed. |

### 3.4 Prompt templates

| Finding | Location | In use? | Touches | Decision |
|---|---|---|---|---|
| Governed email system prompts, resolved live from `bp_prompt` with module-constant fallback: `email_compose_rfq`, `email_compose_response`, `email_polish`, `negotiation_playbook_system` (v2) | `src/agents/email_drafting_agent.py:1443-1453`; `src/agents/base_agent.py:323` | **Live** (`402b642`) | Phase 4 prompt assembly | **Extend.** The Phase 4 system prompt — including the profile-over-exemplars precedence instruction — must be a `bp_prompt` row, not a Python constant. Reuse `resolve_prompt` and the `version` column. |
| `prompts/EmailDraftingAgent_Prompt_Template.json` — literal RFQ email | file, loaded via `_extract_prompt_template` / `_load_prompt_template_from_db` | Live | E2, Phase 3 | **Supersede.** Becomes a seeded `origin='synthetic'` exemplar for intent `rfq_invite`, not a template the generator fills in. |
| Jinja2 rendering of drafts | `src/agents/email_drafting_agent.py` (`from jinja2 import Template`) | Live | Phase 4 | **Supersede.** Template-slot filling is the mechanism the style design replaces with profile-governed generation. Keep only for the fixed-format table blocks (`_build_rfq_table_html`), which are data, not voice. |

### 3.5 Intent vocabulary

| Finding | Location | In use? | Touches | Decision |
|---|---|---|---|---|
| `interaction_type` vocabulary with synonym normalisation: `rfq`, `negotiation`, `clarification`, `follow_up`, `reminder`, `update`, `award`, `thank_you` | `src/agents/email_drafting_agent.py:2883-2958` | **Live** | `style_intent` seed list | **Extend — and reconcile the two lists.** The proposed seed introduces a *second, competing* vocabulary. Mapping: `rfq`→`rfq_invite`, `clarification`→`clarification_request`, `negotiation`→`negotiation_counter`, `award`→`award_notification`, `update`→`internal_update`. The proposed set has no home for the live codes **`follow_up`, `reminder`, `thank_you`**; the live set has no home for **`supplier_rejection`, `escalation`, `contract_variation`, `exit_notification`**. Seed the union (13 codes) and make `_normalise_interaction_type` read `bp_style_intent` rather than its hardcoded dict. |
| Negotiation round → objective/tactics vocabulary | `src/agents/email_drafting_agent.py:252-402`, `src/engines/negotiation_strategy_engine.py` | Live | `behavioural` | **Coexist.** Strategy, not communication-type taxonomy. |

### 3.6 Drafting paths — invariant 10

There are currently **four** ways an email body comes into existence:

| # | Path | Location | Status |
|---|---|---|---|
| 1 | `EmailDraftingAgent` — LLM compose + polish + Jinja templates, 4,955 lines, registered as `email_drafting`, orchestrated | `src/agents/email_drafting_agent.py` | **Live, primary** |
| 2 | A **second class also named `EmailDraftingAgent`**, over `MockDatabaseConnection`, using `NegotiationEmailTemplateRenderer` | `src/orchestration/procurement_workflow.py:452` | Reference/demo workflow — name collision with (1) |
| 3 | `NegotiationEmailTemplateRenderer` — fixed template rendering from `src/resources/workflow/negotiation_email_context.json` | `src/services/negotiation_email_templates.py` | Live, used by (2) |
| 4 | `POST /workflows/email/prepare` — accepts a hand-written subject/body from the UI report panel and persists it via `EmailDraftingAgent._store_draft` | `src/api/routers/workflows.py:1181` | **Live** |

**Decision:**
- (1) **Extend** — it becomes the single generator, with the style profile
  injected ahead of its existing prompts.
- (2) and (3) **Supersede and remove.** Two classes with the same name in one
  codebase is a defect independent of this project; the demo workflow should call
  (1) or be deleted. Removal plan: confirm `procurement_workflow.py` is not on any
  live route (it is built on `MockDatabaseConnection`, so it should not be), then
  delete the duplicate class and fold `NegotiationEmailTemplateRenderer` output
  into seeded synthetic exemplars.
- (4) **Coexist** — it is a *pass-through persist* of user-authored text, not a
  generator. It writes no `generated_body` the model produced. Justification
  against invariant 10: invariant 10 is about *two provenance stories for
  generated text*; path 4 generates nothing, and its provenance is "the user typed
  it". It must record `fallback_level = NULL` / a distinct `origin`, never a
  profile reference it did not use.

**Note:** `EmailDraftingAgent` at 4,955 lines is a god class — the same structural
problem already recorded for this codebase. Adding the style subsystem *inside* it
will make it worse. Recommend the style profile be resolved and rendered by a
separate small service that the agent calls, so Phase 1–4 code lands outside the
god class.

### 3.7 Audit

| Finding | Location | Decision |
|---|---|---|
| `proc.bp_agent_actions` event log — best-effort writer with savepoint isolation | `src/services/agent_actions.py` | **Extend.** Profile compile, approve, recompile-suggest and TTL-sweep runs write here. No parallel audit table. |
| `process_routing_service.log_process` / `log_action` → `proc.bp_action` | `src/services/process_routing_service.py:935,1450` | **Extend.** The dispatch path already logs `process_name="email_dispatch"`; drafting should log its profile version the same way. |

### 3.8 Tenancy, secrets, redaction

| Finding | Decision |
|---|---|
| **No tenancy.** Zero `tenant_id`; `src/api/auth.py:18` confirms no tenant dimension | **Supersede the assumption.** Drop `tenant_id` (§2.1). Do not invent it. |
| **No RLS** | n/a |
| Secrets: SES via Secrets Manager ✅; IMAP via `.env` ❌ | **Supersede** the IMAP path (E4b) |
| **No redaction / masking / anonymisation / PII utility exists anywhere in the repo** | **Net-new, and larger than it looks.** Phase 2's redactor (`[NAME]`/`[ORG]`/`[AMOUNT]`/`[REF]`, signature-block and quoted-chain stripping) has nothing to reuse. Adjacent but not reusable: `src/services/output_safety.py` blocks backend vocabulary leaking into answers, and `utils/email_markers.py` splits hidden tracking markers out of bodies — the latter is directly useful for stripping our own marker before compilation. Budget Phase 2's redactor as real work, and note that invariant 1's n-gram test is the *only* thing standing behind it. |

---

## 4. Where the existing implementation is better

- **`bp_prompt` + `resolve_prompt` is a better template registry than the design
  implies.** It already has `version`, an active-status flag, agent scoping, a
  global fallback, and a live reload endpoint. `prompt_template_version` should be
  a `bp_prompt.version`, and the Phase 4 precedence instruction should be a row in
  it — editable without a deploy.
- **`deploy/sql` + `_rollback.sql` is a better fit than Alembic here.** The repo
  has 30 dated migration pairs and no Alembic history to graft onto. Introducing
  Alembic mid-life would need a baseline stamp against a schema Alembic never
  created.
- **`interaction_type` already has synonym normalisation.** The design's flat
  `style_intent` code list has no synonym handling; the existing mapper does, and
  should be kept as the ingestion-side normaliser over the seeded vocabulary.
- **`test_prepare_email_draft_never_touches_the_send_path` is the right shape for
  invariant 2's test** and already exists — extend rather than invent.

---

## 5. What Phase 0 needs before it can start

1. **E1 decision** — remove the SES send path, or restate invariant 2 as
   subsystem-scoped. *Blocking. Product decision.*
2. **E5 decision** — approve `user_ref text` (Cognito `sub`) in place of
   `persona_id uuid`. *Blocking — the schema cannot be written without it.*
3. **E3 decision** — Qdrant (recommended) vs installing pgvector. *Blocking for
   `bp_style_exemplar` DDL.*
4. **Confirm the §2 amendments**: drop `tenant_id`, `bp_` prefixes, `deploy/sql`
   migration pairs, `bp_admin_config` for mode/min-exemplars, identity columns
   instead of `gen_random_uuid()`.
5. **Approve the merged 13-code intent vocabulary** (§3.5).
6. **E4(b)** — agree that IMAP credentials move behind a `credential_ref` as part
   of this work, or explicitly defer it and accept invariant 8 is aspirational
   until then.

E2 needs no decision: there is no customer-derived style data to migrate.
