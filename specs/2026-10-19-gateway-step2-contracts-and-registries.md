# Step 2: gateway contracts, function catalogue, data-request contract, agent registry

Date: 2026-10-19. Status: **DESIGN FOR REVIEW. No code changed.**
Inputs: `specs/2026-10-19-gateway-target-architecture.md` (authoritative, verbatim) and the Step 1 audit
`specs/2026-10-18-ask-gateway-compliance-audit.md`. Branch `feature/pre-retrieval-phases-0-7` = `002826e5`.

Open decisions are marked `[decision]`. Nothing here adds a store, hands a credential to a new component, or adds a free-form
query function. Every function below is a fixed, typed, parameterised business question.

---------------------------------------------------------------------------------------------------------------------------------

## A. Gateway function standard

Every gateway function is declared once, in versioned config plus a typed Python signature, and must have all seven properties.

| # | Property | Rule |
|---|---|---|
| 1 | One named business question | Name is a verb phrase (`get_po_status`, `find_applicable_contracts`). One function answers one question. A cross-store question is still one function; the join on shared IDs happens inside it. No `run_sql`, `run_cypher`, `query(table, where)`. |
| 2 | Typed parameters | A schema (pydantic model) per function. Unknown parameters are rejected, not ignored. Entity parameters are IDs (`supplier_id`), never names; name to ID is `resolve_entity`. |
| 3 | Caller context | First argument, `CallerContext`, never part of the agent-visible schema (see A.1). |
| 4 | Bounded result | `limit`/`top_k` with a per-function hard maximum from config. Aggregates preferred over rows. `truncated: true` plus `total_available` when cut. |
| 5 | Provenance + as-of | Envelope fields below. A number with no provenance is not returned. |
| 6 | Access enforcement | The function applies row scope from `CallerContext` inside its own queries. There is no unscoped variant. |
| 7 | Audit | One `bp_gateway_call` row per call (A.3), written by the gateway wrapper, not by the function body. |

### A.1 CallerContext (built by the orchestrator only)

```
CallerContext(
  principal_subject,     # Cognito sub, from api.auth.Principal
  role,                  # rbac.effective_role(principal)
  buyer_scope,           # ids from bp_user_buyer_scope, or None = unrestricted (admin)
  tenant_id,             # 'default' today  [decision: see section F, item 1]
  reseller_account_ids,  # empty today: no mapping exists
  request_id, session_id,
  agent,                 # set by the orchestrator for the step that is running
)
```

It is a frozen dataclass. Agents and pipeline stages receive no handle that can construct or mutate one. A function signature that
accepts `CallerContext` from a request body, or an LLM-supplied `user_id`/`role`/`tenant` parameter, fails schema validation
(those names are on a deny-list in the registry loader).

### A.2 Result envelope

```
GatewayResult(
  function, version,
  ok,                       # False => `error` set, `data` empty
  data,                     # rows | aggregates | documents, shape fixed per function
  provenance=[ {store: "postgres"|"kg"|"vector", table_or_label, record_ids[], document_id?, clause?} ],
  as_of,                    # timestamp the data was read (UTC), plus per-source data_as_of when known
  truncated, total_available,
  latency_ms, call_id,      # call_id = bp_gateway_call primary key, cited by the composed answer
)
GatewayError(code, message, retryable)   # codes: NOT_ALLOWED, SCHEMA_INVALID, LIMIT_EXCEEDED, NOT_FOUND,
                                         # AMBIGUOUS(candidates), TIMEOUT, UNAVAILABLE, UNSUPPORTED
```

`NOT_FOUND` is a successful, explicit result (`ok=True`, `data` empty, provenance records what was searched). It is not an error,
so the grader can tell "nothing exists" from "the call failed".

### A.3 Audit record (`proc.bp_gateway_call`, **new table, needs sign-off at Step 3**: schema migration)

`call_id, ts, caller_subject, role, tenant_id, agent, request_id, function, function_version, params_json (after validation),
result_record_ids, row_count, truncated, ok, error_code, latency_ms`. Append-only (same pattern as the existing append-only audit
tables on both DBs). Writes: `bp_gateway_write` adds `idempotency_key, inputs_json, provenance_json, confirmation_id`.
Until the migration is approved, the in-process gateway writes the same fields to the trace and to `bp_agent_actions`
(existing table, no migration).

---------------------------------------------------------------------------------------------------------------------------------

## B. Function catalogue

Intent taxonomy used for the mapping is the Step 6 one: **domains** (procurement, supply chain, reseller/channel, sales,
quotes/CPQ, contracts & commercial terms, general) x **intent types** (policy/how-to, record lookup, analytics, calculation,
comparison, eligibility/approval, action request, clarification).

Status: **EXISTS** = a reusable implementation is there and only needs wrapping in the standard; **PARTIAL** = logic exists but
is spread across modules or lacks scope/provenance; **MISSING** = nothing to wrap.

### B.1 Batch 1: the in-process skeleton (proposed first; wraps what already exists)

| Function | Question it answers | Params | Status / source to wrap | Example questions |
|---|---|---|---|---|
| `resolve_entity` | What record does this text refer to? | `text`, `types[]` (supplier, customer, reseller, sku, quote, order, po, invoice, contract, rfq, programme), `limit` | PARTIAL: `ask_exact_id.extract` (ID patterns) + `supplier_resolution` (alias/fuzzy). Pgvector-over-names step is MISSING and waits on the pgvector decision | "Ironbridge MSA", "PO-000123", "what do we pay Acme?" |
| `get_record` | The attributes of one known record | `type`, `id`, `fields[]?` | PARTIAL: `ask_exact_id.fetch` | "Show PO-000123", "Who is the supplier on INV-77?" |
| `get_po_status` | Where is this PO (ordered, received, invoiced, paid)? | `po_id` | EXISTS: `ask_po_status.gather` | "Status of PO-000123?" |
| `get_corpus_facts` (to be split, see B.2) | Aggregates over the whole record population | intent-specific | EXISTS: `corpus_facts.fetch_facts`; to be decomposed into named functions below | "How many open POs?" |
| `get_spend_summary` | Spend over a period, grouped | `period`, `group_by`, `filters`, `currency` | EXISTS: `analytics/supplier_spend.py`, `analytics/ask.py` (canonical figure GBP 323,467.17) | "Total spend last quarter by supplier" |
| `search_documents` | Which document passages answer this? | `query`, `filters{document_type, supplier_id, contract_id,...}`, `top_k`, `variant_mode?` | PARTIAL: `rag_service` search + rerank + type-filter fix. Returns passages **with** `document_id` + clause; never returns numbers as facts (marked `from_document_text`) | "What does our policy say about three-way match?" |
| `get_policy` | The governed rule text/value for a named policy | `policy_name`, `as_of?` | EXISTS: governance fetch in `agentnick_control` / `bp_policy` | "What is the PO approval limit?" |
| `get_approval_chain` | Who must approve a document of this type/amount/unit, in what order | `doc_type`, `amount`, `currency`, `business_unit`, `as_of?` | **MISSING** as structured data. `approval_route.APPROVAL_ROUTES` + `bp_policy` threshold rows cover part; the Delegation of Authority matrix is prose only. Needs DoA captured as structured rules (sign-off: data entry by the business) | "Who approves a GBP 80k software PO?" |
| `evaluate_approval_status` | Is the approval requirement for this specific document satisfied? | `doc_type`, `doc_id` | **MISSING** (deterministic evaluation over recorded approvals + chain) | "Is PO-123 fully approved?" |
| `find_applicable_contracts` | Which contracts govern this supplier/customer/SKU on this date? | `supplier_id|customer_id`, `sku?`, `date` | PARTIAL: contract tables + KG `Contract` nodes; cross-store join to build | "Which contract covers licences from Ironbridge?" |

### B.2 Batch 2: the `corpus_facts` intents as named, row-scoped functions (BUILT 2026-10-19)

**Correction.** The first version of this section listed `get_open_pos`, `get_overdue_invoices`, `get_supplier_performance`,
`get_price_history`, `get_contract_expiry` and others. Those names were inferred and do not match `corpus_facts`, which has exactly
eight intents. Batch 2 follows the code, not the guess. The names above that do not exist as legacy intents (open POs, overdue
invoices, supplier performance, price history, contract expiry as a list, savings opportunities, demand summary) move to a later
batch and are still unbuilt.

| Legacy intent | Function | Scope | Notes |
|---|---|---|---|
| suppliers | `get_supplier_overview(limit)` | invoices by `buyer_id` | the count of every supplier on record is the whole population; withheld from a restricted caller |
| invoices | `get_invoice_overview(limit)` | `buyer_id` | |
| quotes | `get_quote_overview(limit)` | `buyer_id` | totals count each quote once at its latest version |
| purchase_orders | `get_purchase_order_overview(limit)` | `buyer_id` | |
| deals | `list_deals(limit)` | `buyer_id` (overview and documents) | |
| findings | `get_findings_overview(limit)` | through the finding's document, its deal, the deal's buyer (the report builder's rule) | findings with no deal: unrestricted callers only |
| policies | `list_policies(limit)` | none: the organisation's own governed rule set | |
| spend | `get_spend_summary` (Batch 1) | now scoped by invoice `buyer_id` | deliberate divergence: per supplier **and currency**, never one summed figure |

Each returns a dict keyed exactly as the legacy facts were. Verified on `bp_testdb`: for an unrestricted caller the rows and the
rendered prompt text are identical to the legacy queries for all seven non-spend intents.
`limit` is capped at 25 (legacy 10). Under `ASK_GATEWAY` the `/ask` corpus lookup calls these as the caller; with no caller it reads
no record rows.

Not covered: the knowledge-graph description (`corpus_facts.fetch_graph`) still reads Neo4j directly and is not caller-scoped.
RBAC action-class gating (`invoice.read`, `quote.read`, `deal.read`, `finding.read`, as the report builder applies) is not applied
here; scope is by buyer only. Both are open decisions for review.

### B.3 Batch 3: supply chain, reseller/channel, sales, CPQ (mostly MISSING)

| Function | Status | Dependency |
|---|---|---|
| `get_price(sku, customer_id, qty, date)` | PARTIAL (`sell_side` catalog + pricing policy `sales_quote_pricing`) | needs customer/reseller identity |
| `get_quote`, `list_quotes(filters)`, `get_quote_margin(quote_id)` | EXISTS in `sell_side/quotes.py`, no row scope | margin visible to roles with `margin.read` only (existing gating) |
| `get_inventory_position(sku, location?)` | **MISSING**: no inventory tables exist (audit finding D4) | product decision: where does stock data come from? |
| `get_shipment_status(po_id|shipment_id)` | **MISSING**: no ERP, no carrier tracking (F3) | product decision |
| `get_landed_cost(sku, supplier_id, qty, incoterm?)` | **MISSING**: no formula in the register (D2) | `[decision]` landed-cost definition |
| `get_reseller_programme_terms(reseller_id)`, `get_partner_tier(reseller_id)` | **MISSING**: no programme or reseller model | `[decision]` tenant/reseller model (section F, item 1) |
| `get_customer_pricing_agreement(customer_id, date)` | **MISSING** | same |

### B.4 Write functions (Step 9; listed so the registry can keep read and write apart)

`create_draft_quote`, `submit_quote`, `approve_quote`, `issue_quote` (EXISTS as `sell_side/quotes.py` state machine; needs
idempotency key and the confirmation rule), `create_purchase_order` (**MISSING**: POs never created by API today, product decision),
`record_approval_decision` (**MISSING** for deals; `approval_store` exists for emails/reports). All require an `idempotency_key`
parameter and write to `bp_gateway_write`.

### B.5 Intent to function map (what each domain x intent needs)

| Domain | policy/how-to | record lookup | analytics | calculation | comparison | eligibility/approval | action |
|---|---|---|---|---|---|---|---|
| Procurement | `search_documents`, `get_policy` | `resolve_entity`, `get_record`, `get_po_status` | `get_spend_summary`, `get_open_pos`, `get_supplier_performance` | formula register | `get_supplier_performance` x N, `get_price_history` x N | `get_approval_chain`, `evaluate_approval_status` | write agent (Batch 4) |
| Supply chain | `search_documents` | `get_po_status`, `get_shipment_status` (missing) | `get_demand_summary`, `get_inventory_position` (missing) | `get_landed_cost` (missing) | as procurement | n/a | n/a |
| Reseller / channel | `search_documents` | `get_partner_tier` (missing) | missing | formula register | missing | `get_reseller_programme_terms` (missing) | n/a |
| Sales | `search_documents` | `get_customer_pricing_agreement` (missing) | `list_quotes`, `get_quote_margin` | `get_price` | `get_price` x N | `evaluate_approval_status` | write agent |
| Quotes / CPQ | `get_policy` (`sales_quote_pricing`) | `get_quote` | `list_quotes` | `get_price` + margin formula | `get_quote` x N | `get_approval_chain` | `create_draft_quote`... |
| Contracts & terms | `search_documents`, `find_applicable_contracts` | `get_record(contract)` | `get_contract_expiry` | formula register (dates) | `find_applicable_contracts` x N | `get_approval_chain` | n/a |
| General | greeting: canned, **no gateway call** | | | | | | |

**Coverage today:** of the functions named above, roughly 9 can be built by wrapping existing code (Batches 1-2), about 8 are
blocked on a product or data decision, and 3 (`get_approval_chain`, `evaluate_approval_status`, pgvector name matching) need
structured data or a store decision before they can exist.

---------------------------------------------------------------------------------------------------------------------------------

## C. Agent data-request contract

An agent that needs data it was not prefetched returns, instead of text:

```
{"request_id": "<uuid, agent-generated>", "agent": "<registry name>", "function": "<name>",
 "params": { ... }, "reason": "<one sentence, logged, shown in the trace>"}
```

Orchestrator handling, in order (each failure returns a `GatewayError` **to the agent**, is logged, and never raises silently):

1. **Registry check:** `function` is in this agent's read allowlist (write functions need the agent's write allowlist). Else `NOT_ALLOWED`.
2. **Budget:** `data_requests_used < max_data_requests_per_step`. Else `LIMIT_EXCEEDED` (the agent is told to answer with what it has).
3. **Schema:** `params` validate against the function's schema; any caller-context-shaped key is stripped and counted as a
   violation (`SCHEMA_INVALID`, and a scope-widening attempt is logged to the review queue).
4. **Context:** orchestrator attaches its own `CallerContext` (agent name from the registry, not from the request).
5. **Call + audit:** gateway call, `bp_gateway_call` row, result envelope returned.
6. **Resume:** agent resumes with the envelope. `request_id` is echoed so the trace links ask to answer.

Duplicate `(function, params)` within a step returns the cached envelope without counting against the budget (idempotent read).

---------------------------------------------------------------------------------------------------------------------------------

## D. Agent registry (config)

Lives in versioned config (`config/gateway/agents.yaml`, version-stamped and written to the trace), loaded at start, validated
against the function catalogue (an allowlisted function that does not exist fails the load). The existing `agent_definitions.json`
keeps governing the workflow agents; the **ask-path agents** below are the new entries, and each existing workflow agent gets a
registry row with an empty allowlist until its data access is moved (Step 3), so nothing silently keeps a back door.

```yaml
version: 1
defaults: {max_steps: 4, max_retries: 1, max_data_requests_per_step: 3}
agents:
  ask_policy:        # policy / how-to, any domain
    purpose: Explain governed policy and process, citing the source.
    reads:  [search_documents, get_policy, find_applicable_contracts, get_approval_chain]
    writes: []
    prefetch: policy_default          # search_documents(top_k=6) + get_policy if a policy is named
    max_data_requests_per_step: 2
  ask_records:       # record lookup
    reads:  [resolve_entity, get_record, get_po_status, list_quotes, get_quote]
    writes: []
    prefetch: records_default         # resolve_entity on the question, then get_record / get_po_status
  ask_analytics:
    reads:  [get_spend_summary, get_open_pos, get_overdue_invoices, get_supplier_performance, get_price_history,
             get_savings_opportunities, get_discrepancies, get_contract_expiry, get_demand_summary, get_deal_summary]
    writes: []
    prefetch: analytics_default       # period resolved by the orchestrator (fiscal calendar), not the agent
    max_data_requests_per_step: 4
  ask_compare:
    reads:  [resolve_entity, get_supplier_performance, get_price_history, get_record, find_applicable_contracts]
    writes: []
    prefetch: compare_default         # one call per compared entity, planned by the orchestrator
    max_data_requests_per_step: 6
  ask_eligibility:
    reads:  [get_approval_chain, evaluate_approval_status, get_policy, search_documents, find_applicable_contracts]
    writes: []
    prefetch: eligibility_default     # case function + policy function together
    rule: explains the gateway result; never decides satisfaction
  ask_action:        # action requests (Step 9)
    reads:  [resolve_entity, get_record, get_quote, get_approval_chain]
    writes: [create_draft_quote, submit_quote]    # approve/issue need the human-confirmation rule [decision]
    prefetch: action_default
    max_data_requests_per_step: 3
```

Prefetch templates are separate config (`config/gateway/prefetch.yaml`): an ordered list of `{function, params_from, optional}`,
`params_from` drawing only on resolved entities, resolved time, and the user's literal values. A prefetch step never takes a
parameter from retrieved document text.

Limits are checked by the orchestrator, not the agent. `max_steps` and `max_retries` extend the existing governed limits
(`agent_reach.tool_runtime_max_rounds` etc.) rather than adding a parallel mechanism; the registry value is the ceiling and the
governed limit can only lower it.

---------------------------------------------------------------------------------------------------------------------------------

## E. How Phase 0-7 code maps into this (reuse, not rebuild)

- `ask_exact_id.extract/fetch` and `ask_po_status.gather` become the bodies of `resolve_entity`/`get_record` and `get_po_status`; their SQL
  moves into the gateway module, the callers (`model_selector`) call through the orchestrator stage instead.
- `ask_router` rules and thresholds become the first taxonomy entries in config (Step 6). Behaviour is unchanged behind the flag.
- `ask_trace` gains `gateway_calls[]`, `template_versions{}`, `registry_version`.
- `ask_flags` gets `ASK_GATEWAY` (off by default); with it off, today's path runs unchanged, which is also the shadow-mode baseline for Step 12.

---------------------------------------------------------------------------------------------------------------------------------

## F. Decisions and sign-offs needed before the next step

1. **Tenant / reseller identity `[decision]`:** no user-to-reseller mapping exists and `tenant_id` is the constant `'default'`.
   What identifies a reseller user, and which accounts may they see? Batch 3 and the whole of Step 10 depend on this.
2. **`[pgvector]`:** confirm pgvector (installed, no new store) for embeddings and entity-name matching; migration plan follows at Step 3.
3. **`[event-driven on change]`:** the Postgres to KG sync is already event-driven on promotion with a 5-minute coalescing window.
   Is that the intended meaning, or per-row immediate?
4. **`[above a set value, or of types X/Y]`:** the human-confirmation rule for writes, with values. Needed at Step 9 (and to decide whether
   `approve_quote`/`issue_quote` belong on `ask_action`).
5. **Approvals as data:** `get_approval_chain` needs the Delegation of Authority matrix as structured rules. Who supplies it
   (it currently exists only as policy-PDF text)?
6. **Schema migration approval:** `proc.bp_gateway_call` and `bp_gateway_write` (A.3). Until approved I log to the trace and `bp_agent_actions`.
7. **Missing-function product calls:** inventory, shipment tracking, landed cost, `create_purchase_order` (section B.3/B.4).
   I will list them as `UNSUPPORTED` in the registry and log asks to the function backlog until you decide.
8. **Review of this catalogue:** confirm Batch 1 as the first build (all wrapping existing code, no new store, no migration).

**Not verified:** Batch 1 and Batch 2 sections describe built and tested code. Batch 3 and the write functions are still design only.
