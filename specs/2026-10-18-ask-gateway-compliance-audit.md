# /ask gateway architecture: Step 1 compliance audit

Date: 2026-10-18. Scope: the Python backend (`src/`, `config/`) against the "Target architecture" (data gateway as the only
holder of store credentials; orchestrator calls the gateway; agents get data only through the orchestrator).
Method: read-only. Every count below was produced by a search of the tree on the commit `01680bc0` stack (plus the working tree);
database and graph facts were read from `bp_testdb` and the local Neo4j. Nothing was changed.

**Headline.** No data gateway exists. Every component reaches the stores directly, the credentials are global, there is no
caller-scoped row access on the ask path, and the knowledge graph duplicates record attributes. Several good seams do exist
(an egress choke point, an import-boundary freeze test, a governed tool layer with per-agent allowlists, a versioned
prompt engine, a deterministic approval-route table) and should be built on rather than replaced.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Credentials

There is one process-wide settings singleton, `config/settings.py::Settings`, holding **all** store credentials at once
(`db_host/db_user/db_password` :14-18, `qdrant_url/qdrant_api_key` :34-35, `neo4j_uri/neo4j_username/neo4j_password` :247-249,
`redis_url` :195, S3, Cognito). It is imported by 29 modules directly and reached through `agent_nick.settings` by the rest.

| Holder | Where | What it holds |
|---|---|---|
| `AgentNick` (`agents/base_agent.py`) | `get_db_connection` (DB), `qdrant_client` (:1868-1929 area), Neo4j driver `base_agent.py:2339` | DB + Qdrant + Neo4j; **handed to every agent** as `agent_nick` |
| `services/db.py::get_conn` | fresh psycopg2 connection per call, no pool | DB |
| `services/egress.py::vector_client` (:505) / `graph_client` (:536) | the audited constructors for Qdrant / Neo4j clients | Qdrant, Neo4j |
| Bypassing the egress constructors | `kg_ingestion_service.py:61`, `platform_kg.py:58`, `procurement_kg_builder.py:129`, `base_agent.py:2339` build `GraphDatabase.driver` directly | Neo4j |
| 34 files read `DB_PASSWORD` / `db_password` directly | e.g. `process_monitor_watcher.py`, `session_notify_listener.py`, `email_ingest_lambda.py`, `deal_analysis_service.py`, `extraction/promotion.py`, `uicanvas_bridge.py`, `output_safety.py` | DB |

**Security finding (act regardless of the architecture):** `config/settings.py:249` commits a default Neo4j password
(`"procwise2026"`) in source. A default for a credential means a missing environment variable silently falls back to a known secret.

Orchestration (`src/orchestration/`) does **not** hold credentials itself but uses `agent_nick.get_db_connection()` and
`services.db` (7 files; 27 textual `execute` sites in five of them; see section 5).

## 2. Direct data access outside a gateway

Static count over `src/` (files containing the pattern): **SQL 276** (services 201, api 30, repositories 18, agents 14,
orchestration 7, engines 4, training 2), **Qdrant 22**, **Neo4j 9**. There is no layer between callers and the stores.

The components you asked me to check:

| Component | Direct access | Reference |
|---|---|---|
| `rag_service.py` | Qdrant `query_points` / `upsert` / `delete_payload` / `create_collection`; the semantic cache | `rag_service.py:508, 671, 694, 943, 1116`; `_search_collections_parallel` ~693 |
| `RAGPipeline.answer_question` (`model_selector.py`) | prompt row by SQL (`_ask_persona`, `proc.bp_prompt`) ~2404; `corpus_facts.fetch_facts` (SQL + Neo4j); analytics layer SQL; audit writes `record_action`; Redis chat history `ChatHistoryManager` :56 | `model_selector.py` |
| `corpus_facts.py` | `agent_nick.get_db_connection()` + raw SQL strings per intent; `fetch_graph` (Cypher) | `corpus_facts.py:115, 122, 341`, `:316` |
| `/agents/reason` | `AgentNick.reason` → `orchestration/agentnick_control.build_tools`: agent runs, governance fetches, `get_corpus_facts` (= `corpus_facts`), opportunity and calculation tools. No free-form SQL tool, but the tools themselves read the DB directly | `agentnick_control.py:76-523` |
| Phase 4 exact-ID lookup (mine) | parameterised SQL on `bp_purchase_order_trgt`, `bp_po_line_items_trgt`, `bp_supplier`, `bp_contracts`, `bp_contract_master` | `ask_exact_id.py:50-` |
| Phase 4 inventory check (mine) | `information_schema` query | `ask_router.py:135` |
| Phase 6 retry / comparison (mine) | `rag.search` → Qdrant, via `RAGPipeline._timed_search/_search_graded/_search_comparison` | `model_selector.py` |
| Phase 7 nodes (mine) | four parallel SQL reads | `ask_po_status.py:58` |

Findings specific to the ask path:
* **No row-level access.** `corpus_facts.fetch_facts(agent_nick, query)` takes no caller; `_asker()` only keys the chat history.
  A buyer asking a corpus question sees the whole population. The only row scoping in the codebase is `services/report_data/scope.py`
  (role + `bp_user_buyer_scope` buyer ids), used by reports, not by /ask.
* **No provenance / as-of on results.** Rendered facts are "values printed exactly as stored" with no source ids or timestamp.
* **Results are bounded** only by per-query `LIMIT` and `_MAX_ROWS`; there is no common bound.
* **The database has no row-level security** (`pg_policies` is empty; zero tables with `relrowsecurity`).

## 3. The gateway

**It does not exist.** No Python module, class or function implements the gateway contract (the only "gateway" in the estate is the
separate Node API gateway for the UI, a different thing). The nearest relatives:

| Existing seam | What it gives | Gap against the gateway contract |
|---|---|---|
| `orchestration/agentnick_control.py` tools (`run_<agent>`, `list_governance`, `get_policy`, `get_prompt`, `get_corpus_facts`, `list_/get_/evaluate_opportunity_*`, `list_calculations`, `run_calculation`) | named, typed tools; `max_rounds` governed; per-agent `allowed_tools` stamped from the catalogue (`auto_registry.py:192, 344`, `base_agent.py:378`) | no caller context, no row access, no provenance/as-of, no audit of the call itself, results truncated at 6000 chars rather than bounded by contract |
| `repositories/*.py` (18 modules) | SQL kept in one place per concern | each opens its own connection; no caller context; not an interface agents are limited to |
| `services/egress.py` | one transport for outbound HTTP, SDK clients, vector and graph clients, with an audit row per call (`bp_egress_event`), an SSRF guard and a payload digest. **Its own docstring says policy evaluation is "honestly absent"** | records, does not refuse; `tenant_id` always `'default'` |
| `tests/test_egress_import_boundary.py` | a freeze-list test that fails on any new importer of an HTTP client | the right ratchet pattern, to be reused for "no SQL outside the gateway" |

Per-function record (parameters / caller context / row access / provenance + as-of / bounded / audit): **not applicable, there are
no gateway functions.** For the closest thing (the governed tools above): typed parameters yes; caller context no (`agent` and
`user_id` are passed for the policy gate only); row access no; provenance no; as-of no; bounded by character count only; audit via
`record_action` for some, not per call.

## 4. Embeddings

* **Store today:** Qdrant (local; collections `procwise_document_embeddings` 167 pts, `static_policy` 20, `learning` 45, plus
  `uploaded_documents`). **pgvector 0.7.3 is installed** and in use for two tables only (`bp_style_exemplar`,
  `bp_style_ingest_staging`), so the extension needs no new store, which matters for the "adding a data store requires sign-off" rule.
* **Model:** `BAAI/bge-large-en-v1.5` (`config/settings.py:398`), 1024-d. Points record `embedding_model` as a name only
  (`document_embedding_service.py:220`); no revision or hash, so a silent model change is not detectable.
* **Who generates them:** the API process itself. `AgentNick.embedding_model = SentenceTransformer(...)` (`base_agent.py:1929`)
  is shared by `RAGService`, `DocumentEmbeddingService.embed_document` (:140), `extraction/embed.py::embed_document` (:95) and the
  query path. Reranker `bge-reranker-large` is also in-process (CPU after a CUDA OOM at every restart; see the Phase 2 diagnosis).
* **Re-indexing on change:** documents are embedded when a row is promoted (`extraction/embed.py`); the policy corpus is synced at
  **service start** (`_initialise_static_policy_corpus` → `StaticPolicyLoader.sync_static_policy`), not on change; there is no
  model-version re-index. The vector store was empty for five weeks in Aug-Sep 2026 without a signal (memory: Qdrant outage).
* **Index-metadata defect found in Phase 5:** documents are typed inconsistently (the Ironbridge MSA is indexed as a `quote`).

## 5. Orchestrator

* **SQL in the orchestrator: yes.** `src/orchestration/` has 7 files with SQL access: `state_manager.py` (17 `execute` sites, workflow state),
  `orchestrator.py` (5), `agentnick_control.py` (3), `node_governance.py`, `worker.py`, `workflow_engine.py`, `prompt_engine.py`.
  Workflow state in Postgres is legitimate application state, but it is not behind a gateway, and the architecture says "no SQL".
* **How it calls a gateway:** it does not. Agents receive `agent_nick` and read the stores themselves; the orchestrator hands them
  context but is not the only caller of data.
* **Agent definitions:** `agent_definitions.json` at the repo root (canonical, `agents/definitions.py`) loaded by `AutoRegistry`
  (`auto_registry.py`), with DB-held definitions for created agents (`repositories/agent_definition_repo.py`).
* **Allowlists:** a tool allowlist exists per agent (`allowed_tools`, from `contract.tools`) plus `NODE_TOOL_EXCLUSIONS`.
  It lists *tools*, not data functions, and read and write are not separated.
* **Step and retry limits:** `max_rounds` for the tool loop (governed limit `agent_reach.tool_runtime_max_rounds`,
  `tool_runtime.py:44`); node `retry_count` (`workflow_engine.py:190, 568`); `MAX_DYNAMIC_AGENTS`; `MAX_NEGOTIATION_ROUNDS = 3`.
  **No limit on data requests per step**, because there is no data-request step.

## 6. Approvals

* **Rules today:** (a) `services/approval_route.py` is a deterministic table (`APPROVAL_ROUTES`, `APPROVAL_VALUE_RULES`,
  GBP only), hard-coded as module constants and mirrored in the UI; its docstring states no endpoint records a decision against a
  deal. (b) `bp_policy` holds `ApprovalThresholdPolicy` and several `*AuthorityPolicy` rows (spend gate; who may approve email, etc.),
  evaluated by `engines/policy_engine.py` and `guardrail.py`. (c) `bp_approval` + `approval_store.py` record approvals (emails, reports).
  (d) The **Delegation of Authority matrix exists only as document text** (policy PDF chunks in Qdrant `static_policy`), so
  "who must approve X above £Y" is answered today by retrieval + LLM, with no structured chain behind it.
* **Can an LLM decide approval?** Nothing found that does. `engines/decision_engine.py` calls the model only to classify a
  *supplier reply*. The gap is the reverse: no deterministic `get_approval_chain` exists for the ask path, and the DoA matrix is
  prose, so the LLM *explains* it from text with nothing to check it against.

## 7. Writes

* **Quotes:** `services/sell_side/quotes.py` (`create_draft` :87, `submit`, `approve` :364, `issue` :421) is a state machine with
  `record_event` (:62) writing to `bp_agent_actions` (must-record for the important events), actor passed in, margin-floor enforced
  at approval. **No idempotency key** on create/issue (`UNIQUE (tenant_id, quote_ref)` gives accidental protection only).
  Human confirmation is the approve/issue steps by role. `tenant_id` is the constant `'default'` ("There is one tenant").
* **POs / invoices / orders:** not created through any API or agent. They arrive only through the extraction pipeline
  (`INSERT` into `bp_purchase_order_*` from document intake). There is no "create a PO" write function.
* **Other write surface:** 209 POST/PUT/PATCH/DELETE route handlers, each with its own SQL.
* **Idempotency elsewhere:** present in ingestion paths (`catalog_import`, `deal_link_proposals`, `opportunity_store`, watchers), not as a caller-supplied key.
* **Write audit:** `agent_actions.record_action` / `record_action_or_fail`; guardrail decisions are attributed to a policy version.
  Provenance of the data a write relied on is recorded for quotes (justifications) but not generally.
* **Existing action path:** `guardrail.py` is "the single seam an irreversible action must pass through" (default-deny, principal required),
  to integrate with in Step 9.

## 8. Knowledge graph

* **Contents (Neo4j, read-only counts):** QuoteLine 115,814; InvoiceLine 55,483; POLine 22,560; Quote 21,054; Invoice 12,408;
  PurchaseOrder 5,041; Supplier 5,030; **Contract 3,051; Policy 43**; plus platform-ontology nodes (Stage, Endpoint, DataTable...).
  Relationships are record links (`LINE_OF_QUOTE`, `INVOICE_REFERENCES_PO`, `QUOTE_FOR_PO`...).
* **Conflicts with the architecture:** (1) the graph **duplicates record attributes** (a `Supplier` node carries `supplier_name`,
  `country`, `currency`, `clearance_level`... copied from Postgres) where the architecture wants IDs only; (2) it holds mostly *records*,
  while the architecture reserves it for relationships and documents (policies, contracts, programmes, approval chains);
  (3) no programme or approval-chain nodes exist.
* **Postgres → KG sync is one-way and already event-driven on promotion** (`extraction/kg_sync.py` per promoted `_trgt` row,
  `backend_scheduler.py:225` coalesced to one rebuild per 5 min by `KG_SYNC_THROTTLE_SECONDS`, plus a periodic backstop). The KG never
  writes back to Postgres (the graph-resolution passes write derived edges to the graph only).

## 9. Prompt engine and identity

* **Template model:** `services/prompt_engine/` is a versioned template system: blanks `{{name}}`, `{{deal.id}}`; fragments pinned
  to an exact version `{{> voice@3}}`; no conditions or loops; typed declared variables; tables `bp_prompt_registry`, `bp_prompt_version`,
  `bp_prompt_pointer_move`, `bp_prompt_render`, `bp_prompt_render_use` (a record of every render). `client.render(name, vars, allowed_tools, obligations)`.
  A legacy `proc.bp_prompt` table (36 rows) is still read directly by the ask path (`ask_persona`).
  **Inline prompts in the Phase 0-7 code that must move:** `ask_condense.SYSTEM`, `ask_input_guard.SYSTEM_BOUNDARIES`, the canned replies
  (`REPLIES`, `GREETING_REPLY`, `INVENTORY_CLARIFY`, `ask_grade.NOT_FOUND`, `ask_output_check.REPLY`), `ask_compare.instruction`, `_DIRECTIVE_SYSTEM`.
* **Identity per request:** `api.auth.Principal(subject, email, username, claims)` from a Cognito ID token (`ASK_AUTH_MODE=enforce`);
  `rbac.effective_role(principal)` (role assignments in `bp_role_assignment`, ranked, default-deny for irreversible classes);
  buyer scope via `bp_user_buyer_scope` (buyer ids). **There is no tenant, reseller or partner identity** (`egress_log` says so
  explicitly; `tenant_id` is `'default'` everywhere). A reseller-user to reseller-account mapping does not exist, so the cross-tenant
  requirement (Step 10) cannot be satisfied until that model is defined. Sell-side tables exist (`bp_catalog_*`, `bp_sales_quote*`)
  but are single-tenant.

---------------------------------------------------------------------------------------------------------------------------------

## 10. Phases 0-7: KEEP / REFACTOR / REPLACE

| Phase / module | Verdict | Reason tied to the architecture |
|---|---|---|
| 0 `ask_trace`, `scripts/ask_testsheet/*` | **KEEP** (extend) | The trace becomes the carrier for gateway calls, template versions and provenance; the runner becomes the evaluation harness |
| 1 `ask_input_guard` rules + `REPLIES` | **REFACTOR** | Categories and examples move to versioned config (50+ each); embedding comes from the gateway; wording moves to the prompt engine; add scanning of retrieved content |
| 1 `ask_output_check` | **KEEP** | Narrow, deterministic, logged; add the provenance check beside it |
| 1 `SYSTEM_BOUNDARIES` appended in `_generate_response` | **REPLACE** | Inline prompt; becomes a prompt-engine template with its version in the trace |
| 2 sub-stage spans / `ASK_FAST_RERANK` | **KEEP** the spans; **REPLACE** the in-process reranker | The reranker and embedder belong behind the gateway (it owns embedding generation); the OOM-to-CPU fallback goes away with it |
| 3 `output_safety` narrowing + rule logging | **KEEP** | Required by Step 8 |
| 4 `ask_exact_id`, `ask_po_status` | **REPLACE** | Direct SQL; becomes gateway functions (`resolve_entity`, `get_po_status`) with caller scope, provenance, as-of |
| 4 `ask_router` | **REPLACE** with the taxonomy (Step 6); keep its rules and the greeting/inventory behaviour as entries | Hard-coded routes vs config-defined domain x intent |
| 5 `ask_condense` | **REFACTOR** | Keep the behaviour and its safety checks; move the prompt to the engine; add entity-in-focus tracking |
| 6 `ask_compare` | **REFACTOR** | Becomes several gateway calls planned by the orchestrator |
| 6 `ask_grade` | **REFACTOR** | Keep the retry cap and the explicit not-found; replace the reranker-score judge (shown not to discriminate) with a sufficiency check over gateway results |
| 7 `ask_dates` | **REFACTOR** | Fold into the formula register (Step 8) |
| 7 type-filter fix, `hit.reranked`, condensed-query plumbing in `model_selector` | **KEEP** until Search moves behind `search_documents` | The `source_type` fix is a data-model correction that still applies |
| `ask_flags` | **KEEP** | Required by the rules; later moves to versioned config |

## 11. Remediation plan (ordered, with sign-offs)

0. **Immediate, independent of the architecture:** remove the committed Neo4j default password (`config/settings.py:249`) and require the
   variable. *Touches configuration; needs your OK.*
1. **Define the gateway skeleton in-process first** (Step 2): a typed function registry, a caller-context object built from the
   authenticated `Principal` by the orchestrator, a result envelope `{rows|aggregates, provenance, as_of, truncated}`, and an audit
   record per call. Wrap existing reads (corpus facts, exact-id, PO status, policy search) as the first functions. *No new store.*
2. **Enforce the boundary with a freeze-list test** modelled on `test_egress_import_boundary.py`: freeze today's 276/22/9 direct-access
   files, fail on any new one, and shrink the list each batch. This is what makes "no SQL outside the gateway" real without a big-bang move.
3. **Credentials:** the gateway process/module holds the store credentials; non-gateway components get none. Today's 34 direct readers and
   the global `settings` singleton are the work. *Giving any component credentials needs sign-off; moving them to one is the reverse and
   still changes deployment configuration.*
4. **Row access:** introduce reseller/tenant identity (a user-to-account mapping does not exist) before any claim of cross-tenant safety.
   *Auth/permission change: sign-off.*
5. **Embeddings:** moving from Qdrant to pgvector (already installed) is a change of store and needs your sign-off with a migration plan:
   re-embedding with a recorded model revision, re-index on document change, parity tests on the 21-test sheet plus the evaluation set,
   cut-over and rollback. *Schema migration: sign-off.*
6. **KG alignment:** stop copying record attributes onto nodes (IDs only), model programmes and approval chains, keep the one-way sync.
   *Writes to non-test data: sign-off.*
7. **Approvals as data:** turn the Delegation of Authority matrix into structured rules evaluated in the gateway
   (`get_approval_chain`); the existing `approval_route.py` and `bp_policy` rows are the starting point.
8. **Writes:** idempotency keys on quote/order functions; PO creation does not exist as a write and needs a product decision.

### Decisions I need from you before the steps that depend on them
* `[pgvector]`: confirm pgvector for embeddings (it is installed; Qdrant would be retired). Needed at Step 3.
* `[event-driven on change]`: the current record-to-KG sync is event-driven on promotion with a 5-minute coalescing window. Is that the
  intended meaning, or do you want per-row immediate sync? Needed at Step 3/6.
* `[above a set value, or of types X/Y]`: the human-confirmation rule for writes, and its values. Needed at Step 9.
* The tenant/reseller model (Step 4 and Step 10): what identifies a reseller user and the accounts they may see.
* Which personas and accounts you will provide for evaluation (Step 11); I will not create them.

### Not verified in this audit
* The ask path has never been run live as a non-admin user (no buyer token); row-level behaviour is from the code.
* Neo4j counts are from the local instance; the production graph may differ.
* The count of `execute` sites is textual; some are test doubles or comments.
* I did not exercise `/agents/reason` against D2/F1/F2.
