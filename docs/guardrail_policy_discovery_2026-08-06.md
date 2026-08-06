# Guardrail Policy Architecture — Discovery & Question Set (verified 2026-08-06)

**Scope:** BP_Backend, branch `Development`
**Status:** Step 1 (inventory) re-verified against the current code and the live database. Step 2 (questions) open — no policy drafted until answered.
**Method:** Re-derived from the current codebase and a live query against `proc` on the connected cluster. Supersedes `guardrail_policy_discovery_2026-08-04.md`; corrections since that pass are marked **[Δ]**.

---

## What changed since 2026-08-04

Two commits, both extraction-schema work (`918c161` drift guard, `3417d71` unit_of_measure). **No agent, policy, egress or enforcement surface changed.** The inventory therefore stands, with four verified corrections:

| **[Δ]** | Correction |
|---|---|
| Δ1 | **12** active policies, not 13. `bp_policy` has 12 rows, all `policy_status=1`. |
| Δ2 | The connected database is **`bp_testdb`** on the prod RDS cluster (`procwisemvpdb01…eu-west-1`), not `bp_sqldb`. 105 `bp_` tables. |
| Δ3 | **EmailDispatchAgent is less governed than previously recorded.** The earlier pass said "draft must be approved". Verified: `run()` takes a caller-supplied `drafts` list — including arbitrary `recipients` — and `_send_draft` → `EmailDispatchService.send_draft` validate only *presence* of a recipient (`email_dispatch_service.py:167`). **No approval status is checked anywhere on the send path.** |
| Δ4 | `HITL_ENABLED` still exists (`config/settings.py:347`, read at `negotiation_agent.py:2922`) and the caller-supplied `hitl_auto_approve` bypass is live at `negotiation_agent.py:2999–3001`. Both bypasses remain open. |

---

# PART 1 — CURRENT INVENTORY

## 1.1 Agents

15 agents in `agent_definitions.json` → registry. "Linked policy" = a row in `proc.bp_policy` whose free-text `policy_linked_agents` genuinely tokenises to that agent (live-verified).

| # | Agent | Purpose | Action class | Policies | Prompts |
|---|---|---|---|---|---|
| 1 | DataExtractionAgent | Extracts header + line data from PDF/xlsx/csv | read, compute, **write** (raw→_stg→_trgt) | **none** | none |
| 2 | SupplierRankingAgent | Ranks competing quotes within a deal | read, compute | 3 | 2 |
| 3 | QuoteComparisonAgent | Aggregates quotes into a comparison | read, compute | **none** | none |
| 4 | OpportunityMinerAgent | Finds savings opportunities across corpus | read, compute, **write** | 5 | none |
| 5 | EmailDraftingAgent | Composes RFQ / reply / negotiation email | compute, **write** (drafts) | 1 (#473) | 4 |
| 6 | NegotiationAgent | Multi-round negotiation incl. HITL checkpoints | compute, **write** | 1 (#473) | 1 |
| 7 | SupplierInteractionAgent | Supplier-facing correspondence | compute, **communicate** | 1 (#473) | none |
| 8 | ApprovalsAgent | Gates spend against governed threshold | read, compute, **write** | 1 (#10) | none |
| 9 | QuoteEvaluationAgent | Scores quotes | read, compute, write | **none** | none |
| 10 | EmailDispatchAgent | **Sends** drafts via Amazon SES | **communicate (egress)** | **none** | none |
| 11 | EmailWatcherAgent | Reads inbound supplier mail (IMAP) | **read external**, write | **none** | none |
| 12 | DiscrepancyDetectionAgent | Flags invoice/PO/quote mismatches | read, compute, write | **none** | none |
| 13 | RAGAgent | Answers questions over the corpus | read, compute | none | 1 |
| 14 | RequirementsAgent | Elicits a requirement from a user | read, compute, write | 1 (#9) | 2 |
| 15 | NegotiationAdvisorAgent | Negotiation advice per deal | read, compute, write | **none** (#605 orphaned) | none |

### Production agents NOT in the catalogue

No manifest entry, no declared capabilities, no governance hook — an agent-keyed policy cannot reach them.

| Agent | Location | What it does |
|---|---|---|
| SupportAgent | `services/support_agent.py` | Writes a ticket and **emails the admin autonomously** |
| SummaryAgent | `services/summary_agent.py` | Persona summaries over `_trgt` |
| AgentNick container | `agents/base_agent.py` | Holds DB / S3 / Qdrant / Ollama clients for every agent |
| Supplier research loop | `services/supplier_enrichment/{research,web_tools}.py` | **Searches the public internet and fetches arbitrary URLs** |
| Governed reasoning loop | `services/governance_tools/governed_reasoning.py` | Exposes policy/prompt lookups as model tool-calls |
| Tool runtime | `services/tool_runtime.py` | Dispatches model-requested tool calls |

> **Finding.** 7 of 15 catalogued agents are entirely ungoverned — including both agents that touch the outside world (EmailDispatch, EmailWatcher) and the one that writes the financial record (DataExtraction). The header comment in `orchestration/node_governance.py` ("Only 5 of the 14 agents") is stale.

## 1.2 Modules

| Module | Route prefix(es) |
|---|---|
| M1 Document Intake & Extraction | `/document`, `/promotion`, `/extraction`, `/training` |
| M2 Deals & Linking | `/deals` (summary, proposals) |
| M3 Analysis Events | `/analysis` |
| M4 Opportunities | `/opportunities` |
| M5 Negotiate & Advice | `/deals` (negotiate) |
| M6 Supplier Review & Research | `/suppliers` (×2), `/vendors` |
| M7 Requirements | `/requirements` |
| M8 Email & Mailbox | `/email` + style/mailbox subsystem |
| M9 Decisions / Action Centre | `/decisions` |
| M10 Contract Obligations | `/obligations` |
| M11 Benchmark Pricing | `/benchmark` |
| M12 SpendIQ Metrics & Reports | `/metrics`, `/spendiq`, `/summary`, `/fx` |
| M13 Ask / Conversational | `/workflows`, `/stream`, `/session` |
| M14 Agent Workspace & Orchestration | `/agents`, `/agent-groups`, `/agent-workflows`, `/run`, WebSocket |
| M15 Governance & Models | `/agents` (governed reasoning), `/models` |
| M16 Support | `/support` |
| M17 System & Health | `/system` |

## 1.3 Actions, tagged

| Verb | Class | Where | Human gate today |
|---|---|---|---|
| Extract / classify document | compute | DataExtractionAgent | none |
| Write `_raw` / `_stg` | write | extraction | none |
| Promote `_stg` → `_trgt` | write | promotion listener + scheduler | confidence gate only (F≥80, conf≥90) |
| Rank / compare / evaluate quotes | compute | agents 2, 3, 9 | none |
| Mine opportunities | compute + write | agent 4 | none |
| Draft email | write | agent 5 | none |
| **Send email (SES)** | **communicate** | agent 10 | **none — see Δ3** |
| **Read supplier mailbox (IMAP)** | read external | agent 11 | none |
| **Web search / fetch arbitrary URL** | share + read external | supplier research | none |
| Approve spend | transact-adjacent | ApprovalsAgent | £10,000 |
| Resolve / close a finding | write | `DecisionEngine.execute` | human-initiated; override needs typed reason |
| Retrain / fine-tune model | configure | `/training` | none |
| Edit policy / prompt rows | configure | `/models`, direct DB | none |
| Export report | share | `bp_reports` | none |
| Delegate to sub-agent | delegate | orchestrator, `tool_runtime` | none |

## 1.4 Data and sensitivity

| Dataset | Contents | Classification in code |
|---|---|---|
| `bp_invoice_*`, `bp_purchase_order_*`, `bp_quote_*` | Commercial terms, prices, totals, supplier & buyer identity | **none** |
| `bp_supplier`, `bp_tprm_supplier`, `bp_supplier_enrichment` | Supplier master + third-party risk | **none** |
| `bp_contracts`, `bp_contract_raw` | Contract prose, obligations | **none** |
| Email bodies (`bp_style_*`, dispatch chains) | Named individuals, addresses, direct dials | Redacted **only** on the tone-learning path (`style/redaction.py`) |
| `bp_decision`, `bp_agent_actions` | Who decided what | **none** |
| Qdrant vectors | Document text embeddings | **none**, no tenant field |

> **There is no data-classification scheme anywhere.** Live grep for `sensitivity` / `PII` / `confidential` across `src/` returns only a three-way-match comment and a redaction stop-word. A "sensitivity ceiling" cannot be enforced because no label exists to compare against.

## 1.5 Access model

| Control | Status |
|---|---|
| Cognito ID-token verification | Correctly implemented in `api/auth.py` — RS256 pinned, issuer / audience / expiry / `token_use` all checked |
| Endpoints requiring it | **2**, both in `workflows.py:843,891` (the Ask path). No other router imports `require_user`. |
| Current setting here | `ASK_AUTH_MODE="off"` — the Ask path answers unauthenticated callers |
| Roles / permission levels | **None.** No role concept in `auth.py`; no `%role%` column in `proc` beyond document fields (`contract_signatory_role`, `bp_deal_proposal_member.role`, `bp_mailbox_binding.role`) |
| Tenant scoping | **Impossible today** — zero `tenant%` or `customer%` columns across 105 `bp_` tables |
| Service-to-service key | `PROCWISE_API_KEY` via `verify_api_key` — not wired to any router |

> **Consequence.** *"An agent can never exceed the permissions of the human it acts for"* has nothing to inherit from.

## 1.6 Egress points

| # | Channel | Direction | What can leave | Gate today |
|---|---|---|---|---|
| E1 | Amazon SES | out | Full body + attachments | **none (Δ3)** |
| E2 | IMAP supplier mailbox | in | — (inbound untrusted content) | none |
| E3 | Microsoft Graph mailbox | in | Read-only; `verify_no_send_permission` **refuses** send-capable credentials | Genuinely enforced ✅ |
| E4 | DuckDuckGo search + `fetch_url` | out + in | Search query (may contain supplier names); fetches any http(s) URL | **none** — `web_tools.py:46` checks only the scheme |
| E5 | Local LLM (Ollama / AgentNick) | on-host | Everything in the prompt | Stays on host |
| E6 | Ollama Cloud (`api.ollama.com`) | out of tenant | Full prompt content | **Live credential in `.env:46`**; `ollama_cloud_generate` has no callers today |
| E7 | Report export (`bp_reports`) | out | Spend figures, supplier names | none |
| E8 | Support admin email | out | User question + ticket | none |
| E9 | WebSocket / SSE | out | Agent traces incl. tool arguments | Scrubbed by `OutputSafetyMiddleware` |
| E10 | S3 documents | both | Source documents | Presigned URLs |

**Nuance on output safety.** `services/output_safety.py` scrubs every HTTP and SSE response and is well built — grounded in the live `information_schema`, route table and real `.env` keys rather than a word list. But it protects **our internals** (table names, routes, env vars, stack traces, model names). **It does not protect customer data.** Supplier pricing passes through untouched, by design.

## 1.7 Value actions and thresholds (live values)

| Rule | Value | Source |
|---|---|---|
| Spend approval threshold | **£10,000 GBP**, `on_at_or_below: approve`, `on_above: escalate` | `bp_policy` #10 |
| Missing threshold | escalates; refuses to invent a limit ✅ | ApprovalsAgent |
| Email reply autonomy | `auto_reply_intents: []` — **nothing is auto-repliable today** | #473 |
| Escalate-always intents | price_change, terms_change, contract_variation, liability, dispute, new_commitment | #473 |
| Max auto-replies per thread | 2 · min intent confidence 0.80 | #473 |
| Email value limit | defers to `approval_threshold`; `on_missing_policy: escalate` (fail-closed ✅) | #473 |
| Negotiation advice | high_spend £98,175 / many_alternatives 93 | #605 — **linked to no agent** |

### The HITL gate has two bypasses **[Δ4]**

1. **`HITL_ENABLED=false`** (`config/settings.py:347` → `negotiation_agent.py:2922`) — a global off-switch for human approval.
2. **`hitl_auto_approve: true` in the request payload** (`negotiation_agent.py:2999–3001`) — a caller waives its own human checkpoint; the round returns `{"status": "approved", "source": "auto_approved"}`.

Both contradict *approval-class actions are never agent-executable*.

## 1.8 Policy engine — how it actually works

| Aspect | Reality |
|---|---|
| Storage | `proc.bp_policy`, 12 columns, unique index on active (type, name). **12 active rows [Δ1]** |
| Loading | `PolicyEngine` reads active rows at construction, caches, indexes by slug + aliases |
| Linking | Free-text `policy_linked_agents`, tokenised on non-alphanumerics |
| Enforcement — general | **None.** `validate_workflow()` (`policy_engine.py:359`) implements exactly one workflow, `supplier_ranking`; every other call falls through to `return {"allowed": True, "reason": "No policy checks"}` at line 383 |
| Enforcement — fail-closed islands | Two, both well built: `governance_tools/authority.py` (email autonomy) and `DecisionEngine` (spend). Both escalate rather than default-allow |
| Enforcement — read / write / egress | **No hook exists.** No code path consults a policy before reading data, writing a table, or sending an email |
| Reload | `reload_policies()` exists (line 328); no endpoint calls it |
| Audit | `proc.bp_agent_actions` — writer is explicitly **best-effort**, savepoint-wrapped, failures swallowed (`agent_actions.py:125,150`). `proc.bp_decision` records facts + evidence properly |

### The 12 active policies

| id | name | type | linked agent(s) |
|---|---|---|---|
| 1–3 | WeightAllocation / CategoricalScoring / NormalizationDirection | supplier_ranking | supplier_ranking_agent |
| 4–8 | ContractExpiry / PriceBenchmarkVariance / VolumeConsolidation / SupplierRiskAlert / MaverickSpend | opportunity | opportunity_miner_agent |
| 9 | requirement_required_fields | requirements | requirements_agent |
| 10 | ApprovalThresholdPolicy | approval | approvals_agent |
| 473 | EmailReplyAutonomyPolicy | email_autonomy | email_drafting, negotiation, supplier_interaction |
| 605 | negotiation_advice_thresholds | negotiation | **(none — orphaned)** |

---

# PART 2 — EXPLICIT GAPS

| Gap | Impact on policy design |
|---|---|
| **G-a** No data-classification scheme | Dimensions 1, 2, 3, 10 inexpressible until labels exist |
| **G-b** No roles/permissions for humans | "agent ≤ human" unimplementable |
| **G-c** No tenant dimension | "may not leave the tenant" not checkable |
| **G-d** No general policy enforcement point | Anything outside ranking / spend / email-autonomy would be documentation, not control |
| **G-e** Audit best-effort and droppable | Violates "auditing cannot be switched off" |
| **G-f** HITL waivable by config flag *and* caller payload | Violates "approval-class actions are never agent-executable" |
| **G-g** No rate limits or circuit-breakers anywhere | Live grep for `rate_limit` / `slowapi` / `circuit_breaker` across `src/`: **zero hits**. Dimension 11 has no mechanism |
| **G-h** No prompt-injection handling | Document text, supplier email bodies and fetched web pages reach the model as trusted text |
| **G-i** `ASK_AUTH_MODE=off` here | The only authenticated surface is unauthenticated in this environment |
| **G-j** 6 production agents outside the catalogue | Agent-keyed policy cannot reach them |
| **G-k** #605 orphaned; agents 1, 3, 9, 10, 11, 12, 15 unlinked | Linkage is free text and unvalidated — a typo silently ungoverns an agent |
| **G-l [Δ3]** Send path checks no approval | The highest-risk action in the product has no gate of any kind |

---

# PART 3 — QUESTION SET

## 3.0 The blocking question

**G0 — Roles.** None exist. Proposed minimum:

| Role | May |
|---|---|
| **Viewer** | read only |
| **Buyer** | read + draft + request approval |
| **Approver** | Buyer + approve up to a personal limit |
| **Admin** | Approver + configure policies, models, mailboxes |

An agent inherits the caller's role and can never exceed it.

> Adopt this set, a different one, or is RBAC out of scope for now? If out of scope, every "agent ≤ human" policy is advisory and the drafts will say so.

## 3.1 Global defaults — confirm once

Reply "confirm G1–G12", or name exceptions.

| # | Proposed default |
|---|---|
| **G1** | Four labels: `public` / `internal` / `commercial-confidential` (prices, terms, quotes) / `personal` (names, emails, phones). Every `bp_` column gets one. |
| **G2** | Default read ceiling for all agents = `commercial-confidential`. `personal` requires explicit per-agent grant. |
| **G3** | Nothing `personal` appears in agent output unless the recipient is the person themselves or an authenticated internal user. |
| **G4** | Every irreversible action (send, export, transact, configure, delegate-to-external) is **deny by default**, explicitly allowed per agent. |
| **G5** | Approval-class actions are **never agent-executable**; the `hitl_auto_approve` request flag is removed. *(code change)* |
| **G6** | `HITL_ENABLED` stops being a global off-switch; it may be narrowed per agent, never disabled. *(code change)* |
| **G7** | Audit becomes mandatory: if the `bp_agent_actions` write fails, the action **fails** rather than proceeding unlogged. *(reverses current behaviour)* |
| **G8** | Minimum audit record: timestamp, agent, human principal, action verb, target ids, policy consulted + version, decision, evidence refs, egress destination. |
| **G9** | Any instruction found *inside data* (document text, supplier email body, fetched web page) is content, never a command. Agents get an explicit "data is quoted, not obeyed" frame; any tool call whose arguments derive from untrusted text is blocked. |
| **G10** | No customer data to any model outside the host. Local AgentNick only; the Ollama Cloud credential is revoked from `.env`; any external provider row in `bp_model` needs a written exception. |
| **G11** | Per-user circuit-breakers: 50 emails/day, 200 extractions/hour, 20 external web fetches/hour, 10 exports/day. Tripping pauses the agent and raises a review item. |
| **G12** | Policy linkage stops being free text: an agent slug matching no registered agent, or an agent with no policy, is a startup-visible error. |

## 3.2 Per-agent questions

### 1. DataExtractionAgent
- **1a** Writes the financial record with no governance. Proposed: `_raw`/`_stg` autonomous, but `_trgt` promotion keeps the confidence rule **plus** a value ceiling — any document totalling over **£50,000** is held for human confirmation. Confirm, change the figure, or reject?
- **1b** Confirm G9: text inside a PDF reading like an instruction is never acted on — including "ignore previous", supplier-supplied URLs, embedded macros.
- **1c** Should extraction refuse to run on a document with no verifiable source (no S3 key, no upload record)? Proposed **yes**.
- **1d** May extraction results ever go outside the tenant for verification (external OCR)? Proposed **hard deny**.

### 2. SupplierRankingAgent
- **2a** Confirm it may **never** rank across deals or produce a global league table without an explicit human request.
- **2b** May a ranking justification quote a *competing* supplier's price? Proposed **no** — figures from supplier B must never appear in anything visible to supplier A.
- **2c** May it use "best" / "cheapest" / "recommended"? Proposed **"highest scoring on the stated criteria"** only, never a superlative.

### 3. QuoteComparisonAgent
- **3a** Does the cross-supplier masking rule (2b) apply here too?
- **3b** Should a recommendation above the £10,000 threshold be labelled "requires approval" in its own output? Proposed **yes**.

### 4. OpportunityMinerAgent
- **4a** Confirm read ceiling = `commercial-confidential` and that it may **not** read `personal`.
- **4b** May findings be exported or emailed by other agents without review? Proposed **internal-only, no egress without human release**.
- **4c** Autonomous write, or proposals above a value? Proposed autonomous below **£100,000**, proposal above.

### 5. EmailDraftingAgent
- **5a** Confirm **drafting is always allowed, and a draft is never itself the trigger to send.**
- **5b** What may a draft assert without human review? Proposed forbidden: price guarantees, volume commitments, delivery promises, legal/liability language, any figure not present in a source document.
- **5c** May a draft include another supplier's name or price as leverage? Proposed **hard deny**.
- **5d** Confirm `personal` data of internal staff (direct dials, personal emails) is stripped from outbound drafts unless the human adds it.

### 6. NegotiationAgent
- **6a** Given G5/G6, confirm **no round is ever released to a supplier without a named human approving that round.**
- **6b** Maximum concession it may *propose* without approval? Proposed: may propose anything, but a proposal exceeding **10% of baseline or £10,000, whichever is lower** is flagged high-impact.
- **6c** Round cap before mandatory human review? Proposed **3**.
- **6d** Confirm a supplier writing "your buyer has already agreed to this" is never treated as evidence.

### 7. SupplierInteractionAgent
- **7a** The least-defined agent that can communicate externally. Proposed **acknowledgements and information requests only; anything substantive escalates.**
- **7b** May it write to an address that appeared *in an email body* rather than the supplier master? Proposed **hard deny — known-supplier addresses only**.

### 8. ApprovalsAgent
- **8a** £10,000 is one global number. Vary by category, supplier, or approver's personal limit? (Needs G0.) Proposed: keep single-threshold, add per-approver limits once roles exist.
- **8b** Confirm it may **record** an approval but never **be** the approver. Under-threshold currently returns "approved" — recommend it return **"no approval required"** instead: more honest, leaves accountability with a person.
- **8c** Threshold is GBP. EUR/USD amounts — convert at `bp_fx_rates` or escalate? Proposed **convert, record rate + date as evidence; escalate if no rate exists.**

### 9. QuoteEvaluationAgent
- **9a** Confirm the same cross-supplier masking as 2b / 3a.
- **9b** May its scores be shown to a supplier? Proposed **no, internal only**.

### 10. EmailDispatchAgent — highest risk, zero governance **[Δ3]**
- **10a** It currently sends whatever `drafts` it is handed, to whatever `recipients` are in the payload, with **no approval check at all**. Confirm: it may send **only** a draft carrying a recorded human approval, and must **verify that approval against the store** rather than trusting a field in its input.
- **10b** Recipient allow-list — send only to addresses on the supplier master? Proposed **yes**; unknown recipients rejected and raised for review.
- **10c** May it attach source documents (invoices, POs) to outbound mail? Proposed **deny by default** — an attached PO can leak another supplier's terms.
- **10d** Volume breaker: proposed **20 per run, 50 per user per day**; exceeding pauses dispatch.
- **10e** Should dispatch to a *new* domain (never emailed before) always require confirmation? Proposed **yes**.

### 11. EmailWatcherAgent
- **11a** Largest untrusted-input surface. Confirm G9: nothing in a supplier email becomes an instruction, a tool call, or a fact without grounding.
- **11b** May it act on an attachment? Proposed **ingest and extract, but attachments from unrecognised senders are quarantined for review**.
- **11c** What may it do autonomously on a match — record only, or update the deal? Proposed **record only**.

### 12. DiscrepancyDetectionAgent
- **12a** Confirm it may create findings autonomously but never close, dismiss, or resolve one.
- **12b** May a finding be communicated to a supplier automatically ("we believe you overbilled us")? Proposed **hard deny**.
- **12c** Proposed precondition: raise findings only where **both** sides are `_trgt` (promoted, confidence-gated) records — never staged data.

### 13. RAGAgent
- **13a** Confirm it answers only from retrieved, cited corpus facts and refuses rather than generalising when there is no citation.
- **13b** With no tenant dimension (G-c), every authenticated user sees the whole corpus. Acceptable for now, or restrict Ask to a named allow-list until scoping exists?
- **13c** May it answer questions about *named individuals* (who approved what, who emailed whom)? Proposed **deny** until roles exist.
- **13d** Confirm figures must carry provenance, and a figure it cannot source is not stated at all.

### 14. RequirementsAgent
- **14a** Confirm it may write a requirement autonomously but not convert it into an RFQ or dispatch.
- **14b** May a requirement above a value be created without approval? Proposed flagged above **£25,000**.

### 15. NegotiationAdvisorAgent
- **15a** Policy #605 is linked to no agent — link it here, or is it dead? Its numbers (£98,175 / 93) look corpus-derived rather than chosen. **Are those the numbers you want?**
- **15b** Confirm advice is internal-only and never reaches a supplier verbatim.
- **15c** May advice name a specific competitor to leverage? Proposed **internal yes, external never**.

### 16. SupportAgent *(uncatalogued — emails the admin autonomously)*
- **16a** Confirm it may email the admin without approval as today. Proposed **yes — admin-only, fixed recipient, never a supplier or arbitrary address**.
- **16b** May a ticket body include the user's data (document contents, figures)? Proposed **summary and identifiers only, no document content**.

### 17. Supplier research loop *(uncatalogued — public internet)*
- **17a** May a **supplier's name** go to a public search engine? Proposed: **allowed for public-company verification only, never combined with a price, volume or contract term in the same query.**
- **17b** `fetch_url` checks only the URL scheme. Proposed **domain allow-list, deny everything else**; at minimum deny private/internal IP ranges.
- **17c** Confirm fetched web content is untrusted input under G9.
- **17d** May research findings ever *overwrite* a document-extracted field? Proposed **no — fills empty fields only**.

### 18. SummaryAgent, AgentNick container, governed-reasoning loop, tool runtime
- **18a** Add these to the agent catalogue so they can be governed at all? Proposed **yes**.
- **18b** Should the set of tools an agent may call be itself a policy (per agent, per tool)? Proposed **yes, default-deny**.

## 3.3 Per-module questions

### M1 Document Intake & Extraction
- **M1a** Who may upload, and may an agent ingest a document nobody uploaded (e.g. pulled from a mailbox)? Proposed **yes, but tagged `agent-ingested` and excluded from autonomous promotion.**
- **M1b** `/training` can trigger retraining — configure-class. Proposed **Admin only, never agent-executable**.
- **M1c** Source documents are never modified (your standing rule). Confirm this becomes an enforced policy, not a convention.
- **M1d** Retention: how long do raw documents and extracted rows live? Nothing expires anything today.

### M2 Deals & Linking
- **M2a** May an agent merge or split a deal autonomously, or only propose? Proposed **propose only**, given the known mis-grouping bug.
- **M2b** Confirm deal-level figures inherit the highest classification of any document in the deal.

### M3 Analysis Events
- **M3a** Confirm analysis events are **append-only**; no agent may delete or amend one.

### M4 Opportunities
- **M4a** What may be claimed externally? Proposed **"potential, based on data available" only**; "guarantee", "will save" and superlatives forbidden.

### M5 Negotiate & Advice
- **M5a** Confirm nothing in this module egresses without a named human release.
- **M5b** Should a negotiation position ever be visible to a Viewer-role user? (Needs G0.)

### M6 Supplier Review & Research
- **M6a** Confirm an agent may **never** auto-merge two supplier identities (82–96% fuzzy matches are human-confirmed today).
- **M6b** May public-web enrichment be written into the supplier master without review? Proposed **no — proposals only.**

### M7 Requirements
- **M7a** Should a requirement missing a governed required field (#9) be blocked from progressing? Proposed **yes**.

### M8 Email & Mailbox
- **M8a** Mailbox binding is a configure action holding customer credentials. Proposed **Admin only, never agent-executable, revocation honoured immediately**.
- **M8b** The MS Graph path actively refuses send-capable credentials. Confirm that pattern becomes the standard for every future integration.
- **M8c** Email bodies are redacted only on the tone-learning path. Apply the same redaction anywhere an email body reaches a model? Proposed **yes, everywhere**.

### M9 Decisions / Action Centre
- **M9a** `execute()` lets a human override the engine with a typed reason. Confirm that is right, and that the override is permanently recorded against their name.
- **M9b** May any agent call `execute()`? Proposed **hard deny — human-initiated only**.

### M10 Contract Obligations
- **M10a** Proposed `commercial-confidential`, read restricted to Buyer+, and **no contract prose ever leaves the tenant**.
- **M10b** May an agent assert an obligation was breached? Proposed **flag only, never assert**.

### M11 Benchmark Pricing
- **M11a** Benchmark figures are contractual (Excel penny-parity). Confirm no agent may alter a benchmark calculation, and any rounding change is configure-class requiring Admin.

### M12 SpendIQ Metrics & Reports
- **M12a** Who may export, and may an export be emailed? Proposed **Buyer+ may export; emailing an export requires approval.**
- **M12b** Should exports carry a provenance footer (source tables, date, FX basis)? Proposed **yes**.
- **M12c** May a report be shared to a link accessible without login? Proposed **hard deny**.

### M13 Ask / Conversational
- **M13a** `ASK_AUTH_MODE=off` here. Should it **never** be off outside a developer machine — service refuses to start with auth off when a production marker is present? Proposed **yes**.
- **M13b** Conversation history retention? Proposed **90 days**.
- **M13c** May a user's question text be used for model training? Proposed **no without explicit opt-in**.

### M14 Agent Workspace & Orchestration
- **M14a** Does a user-composed graph inherit the same policies? Proposed **identical policies; composition never grants authority.**
- **M14b** May a workspace graph include a communicate-class agent (EmailDispatch)? Proposed **only for users with an explicit send grant.**
- **M14c** The scheduler runs agents unattended. Which may run with **no human present**? Proposed: extraction, linking, promotion, opportunity mining, health sweeps — and **nothing that communicates or transacts.**

### M15 Governance & Models
- **M15a** Editing `bp_policy` / `bp_prompt` changes agent behaviour with no deploy. Proposed **Admin only, versioned, previous version retained, change logged to `bp_agent_actions`.**
- **M15b** Confirm no agent may read, write, or reason about the policy table in a way that changes its own limits.
- **M15c** Model switching (`bp_model`) — Admin only? Is AgentNick-only a policy or a preference? Proposed **hard policy**.

### M16 Support
- **M16a** Classification and retention for support tickets? Proposed `internal`, 12 months.

### M17 System & Health
- **M17a** Should `/system` require Admin? Proposed **yes**.

---

# PART 4 — COVERAGE CHECKLIST

**Dimensions:** 1 data access · 2 exposure · 3 external sharing · 4 autonomous actions · 5 communication · 6 mutations · 7 monetary · 8 recipients · 9 claims · 10 model/data use · 11 rate · 12 preconditions · 13 untrusted input · 14 audit

**Key:** **Q** = asked above · **G** = covered by a global default awaiting confirmation (G1–G12) · **–** = N/A, reason footnoted

## Agents

| Agent | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DataExtraction | G | G | Q | Q | –ᵃ | Q | Q | –ᵃ | –ᵇ | Q | G | Q | Q | G |
| SupplierRanking | G | Q | G | G | –ᵃ | G | G | –ᵃ | Q | G | G | G | G | G |
| QuoteComparison | G | Q | G | Q | –ᵃ | G | Q | –ᵃ | Q | G | G | G | G | G |
| OpportunityMiner | Q | G | Q | Q | –ᵃ | Q | Q | –ᵃ | G | G | G | G | G | G |
| EmailDrafting | G | Q | Q | Q | Q | G | G | Q | Q | G | G | G | G | G |
| Negotiation | G | G | Q | Q | Q | G | Q | G | Q | G | Q | G | Q | G |
| SupplierInteraction | G | G | Q | Q | Q | G | G | Q | Q | G | G | G | Q | G |
| Approvals | G | G | –ᶜ | Q | –ᵃ | Q | Q | –ᶜ | –ᵇ | G | G | Q | G | G |
| QuoteEvaluation | G | Q | Q | G | –ᵃ | G | G | –ᵃ | Q | G | G | G | G | G |
| **EmailDispatch** | G | G | Q | Q | Q | –ᵈ | Q | Q | Q | G | Q | Q | G | G |
| EmailWatcher | G | G | –ᵉ | Q | Q | Q | –ᵇ | –ᵉ | –ᵇ | G | G | Q | Q | G |
| DiscrepancyDetection | G | G | Q | Q | Q | Q | Q | –ᵃ | Q | G | G | Q | G | G |
| RAG | Q | Q | Q | G | –ᵃ | –ᵈ | –ᵇ | –ᵃ | Q | Q | G | Q | G | G |
| Requirements | G | G | G | Q | –ᵃ | Q | Q | –ᵃ | G | G | G | G | G | G |
| NegotiationAdvisor | G | G | Q | G | –ᵃ | G | Q | –ᵃ | Q | G | G | G | G | G |
| SupportAgent \* | G | Q | Q | Q | Q | G | –ᵇ | Q | G | G | G | G | G | G |
| Supplier research \* | G | G | Q | Q | –ᵃ | Q | –ᵇ | Q | G | Q | Q | G | Q | G |
| SummaryAgent \* | G | G | G | Q | –ᵃ | G | –ᵇ | –ᵃ | G | G | G | G | G | G |
| AgentNick container \* | G | G | G | Q | –ᵃ | G | –ᵇ | –ᵃ | –ᵇ | Q | G | G | G | G |
| Tool-call loop \* | Q | G | G | Q | –ᵃ | Q | –ᵇ | –ᵃ | –ᵇ | G | G | G | G | G |

\* uncatalogued today — see 18a.

**Footnotes:** **a** no communicate capability, so recipient control does not apply · **b** performs no value action / makes no external claim · **c** ApprovalsAgent never transmits, it records · **d** read/compose only, writes no business record · **e** inbound only.

## Modules

| Module | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 Intake & Extraction | G | G | G | Q | –ᵃ | Q | Q | –ᵃ | –ᵇ | G | G | Q | Q | Q |
| M2 Deals & Linking | Q | G | G | Q | –ᵃ | Q | G | –ᵃ | G | G | G | G | G | G |
| M3 Analysis Events | G | G | G | Q | –ᵃ | Q | –ᵇ | –ᵃ | –ᵇ | G | G | G | G | Q |
| M4 Opportunities | G | G | Q | G | –ᵃ | G | Q | –ᵃ | Q | G | G | G | G | G |
| M5 Negotiate & Advice | Q | G | Q | G | Q | G | G | G | G | G | G | G | G | G |
| M6 Supplier Review/Research | G | G | Q | Q | –ᵃ | Q | –ᵇ | Q | G | Q | Q | G | Q | G |
| M7 Requirements | G | G | G | Q | –ᵃ | Q | Q | –ᵃ | G | G | G | Q | G | G |
| M8 Email & Mailbox | G | Q | Q | Q | Q | Q | G | Q | G | Q | Q | G | Q | G |
| M9 Decisions/Action Centre | G | G | G | Q | –ᵃ | Q | G | –ᵃ | G | G | G | G | G | Q |
| M10 Obligations | Q | G | Q | G | –ᵃ | G | G | –ᵃ | Q | Q | G | G | G | G |
| M11 Benchmark | G | G | G | Q | –ᵃ | Q | Q | –ᵃ | G | G | G | G | G | G |
| M12 Metrics & Reports | G | G | Q | G | Q | G | G | Q | Q | G | Q | Q | G | G |
| M13 Ask | Q | Q | G | G | –ᵃ | –ᶜ | –ᵇ | –ᵃ | Q | Q | G | G | G | Q |
| M14 Workspace/Orchestration | Q | G | Q | Q | Q | Q | Q | Q | G | G | Q | G | G | G |
| M15 Governance & Models | Q | G | G | Q | –ᵃ | Q | –ᵇ | –ᵃ | –ᵇ | Q | G | G | Q | Q |
| M16 Support | Q | Q | Q | Q | Q | G | –ᵇ | Q | G | G | G | G | G | G |
| M17 System & Health | Q | G | G | –ᵈ | –ᵃ | –ᶜ | –ᵇ | –ᵃ | –ᵇ | G | G | G | G | G |

**Footnotes:** **a** module has no outbound channel · **b** no monetary value or external claim in scope · **c** read-only surface · **d** no autonomous action; request-driven only.

Every agent and every module is covered across all 14 dimensions — by a question, by a global default awaiting confirmation, or marked N/A with a stated reason.

---

# What to answer first

1. **G0 — roles.** Without them, "an agent can never exceed the human it acts for" is unenforceable.
2. **G5 / G6 — the two HITL bypasses.** Closing them is a code change.
3. **G7 — mandatory audit.** Today an action proceeds even when its audit write fails.
4. **10a–10e — EmailDispatchAgent.** **[Δ3]** It sends real mail to real suppliers, to caller-supplied recipient addresses, with no approval check, no allow-list and no volume cap. This is the single largest exposure in the inventory.

No policy will be drafted until these are answered.
