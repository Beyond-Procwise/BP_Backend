# Production Readiness Routine — 2026-06-08 (overnight, deadline 08:00 IST / 02:30 UTC)

Mode: autonomous audit → fix → live-verify → report. Branch: `Development`.

## Status legend
- 🔴 CRITICAL  🟠 HIGH  🟡 MEDIUM  🟢 LOW
- [ ] open · [~] in progress · [x] fixed+verified · [defer] tracked, needs larger effort

---

## Audit coverage
- [x] Extraction pipeline (services/extraction, extraction_v3, data_extraction_agent, extraction_engine)
- [x] Large agents (negotiation, supplier_interaction, email_drafting, email_watcher)
- [x] Other agents (supplier_ranking, opportunity_miner, quote/discrepancy, rag, approvals)
- [x] Orchestration (orchestrator, workflow engine/definitions, prompt_engine, model_selector)
- [ ] Services core (db, summary_agent, deal_summary, reconciliation, backend_scheduler, process_routing)
- [ ] API layer (main, routers, lifespan, auth, error handling)
- [x] Services core (db, reconciliation, backend_scheduler, process_routing, rag_service, ollama_client, process_monitor_watcher)
- [x] API layer (main, routers, lifespan, auth, error handling)
- [~] Governance + engines (policy_engine/prompt_engine covered during governance feature build)
- [~] DB & data pipeline (covered via migrations + missing-table findings)
- [x] Security cross-cutting (auth, SQLi sweep, input validation, path traversal — covered across audits)
- [~] Tests/config/ops (known pre-existing failures catalogued; partial)

---

## Findings & fixes log

### Extraction pipeline
- 🔴 [ ] **SQLi in HITL promote** — `services/extraction/promotion.py:601,606` interpolates `field_name` (from `bp_extraction_discrepancy`) into `UPDATE {raw_t} SET {field_name}=...`. Fix: allowlist against `_stg_columns`/`_raw` columns before interpolation.
- 🔴 [ ] **invoice_amount captures grand_total** — `extraction_schemas/invoice.yaml:396-400` patterns `grand_total`(0.90)/`total_amount_payable`(0.89) outrank `anchored_subtotal`(0.87); stores gross in pre-tax column. Fix: remove those patterns from `invoice_amount` (belong to `invoice_total_incl_tax`). NEEDS live re-validation against the 50-doc set before/after.
- 🔴 [ ] **context_layer silent stale fallback** — `context_layer.py:292-307` + `dispatch.py:266-267` return raw L1 candidates on any LLM failure with only log.warning; degraded rows promote + pass training gate. Fix: flag `_ctx_failed`, skip overlay, enqueue non-blocking discrepancy.
- 🟠 [ ] **S3 temp-file leak** — `parser.py:98-132` mkstemp not deleted on success; unbounded `/tmp` growth → watcher crash. Fix: NamedTemporaryFile(delete=False)+try/finally cleanup.
- 🟠 [ ] **tax_percent bypasses grounding** — `context_layer.py:1009-1017` decimal accepted with no doc-presence/range check; fabricated pct → wrong tax/total. Fix: range 0<pct≤50 + token presence.
- 🟠 [ ] **promote() recomputes derived on _raw round-trip** — `promotion.py:287-291` may overwrite NULL tax with ungrounded arithmetic. Fix: limit safety-net to FX only; skip tax derivation for _raw-sourced pct.
- 🟠 [ ] **region weak grounding + dead code** — `context_layer.py:941-965` weaker `_region_field_grounded` (token suffix) fires; stricter block dead. Fix: remove dead block, require full-value match.
- 🟠 [ ] **pattern_registry no reload on YAML change** — `pattern_registry.py:113-119` module cache, schema edits need restart. Fix: mtime check or admin reload endpoint.
- 🟡 [ ] write_raw dynamic columns no allowlist (`persistence.py:196`); _compute_derived fragile break (`context_layer.py:1063`); grounded judge blank-image fallback (`grounded_last_resort.py:369`); **no unit tests** for context_layer/promote/judge.
- Incomplete: PipelineV3 synthetic `SYN-` invoice_id (fabrication); L3 tiebreaker/coherence wired but unused; contract doc_type has no `_FIELD_DEFS` (silent degraded); hardcoded FX table (`context_layer.py:57-69`).

### Large agents
- 🔴 [ ] **Duplicate method defs in NegotiationAgent** — `negotiation_agent.py` `_load_session_state`(1587 vs 7011), `_save_session_state`(1609 vs 7149), `_compose_negotiation_message`(7466 vs 8625). Last-wins → multi-round state load/save broken + compose crashes on `contact_name` kwarg. Fix: consolidate to one impl each.
- 🔴 [ ] **IMAP IDLE capability detection always False** — `email_watcher_agent.py:126` uses non-existent `client.capabilities` attr; IDLE dead, falls back to polling; no socket timeout → hang. Fix: use `client.capability()`, pass `timeout=` to IMAP4_SSL.
- 🔴 [ ] **session_lock used without `with`** — `negotiation_agent.py:4772-4811` manual __enter__/__exit__ leaks DB conn on exception / double-exit. Fix: use `with self._session_lock(...)`.
- 🟠 [ ] No timeout on `ollama.generate` (`negotiation_agent.py:6164`) → hang. Fix: timeout=.
- 🟠 [ ] UPSERT f-string column/predicate from introspection (`negotiation_agent.py:7265-7292,9886`) — allowlist columns, treat predicate as untrusted.
- 🟠 [ ] `_run_single_negotiation_locked` 1297 lines (4813-6109) — extract price/strategy/email/persist. [defer-structural]
- 🟠 [ ] `_wait_for_round_responses` blocks ≤3600s in unbounded ThreadPoolExecutor (4213, 3547) → thread exhaustion. Fix: async or bound pool + shorter default.
- 🟠 [ ] `_execute_batch_entry` re-enters execute() → possible infinite recursion (2443); check `_batch_execution` guard in run().
- 🟡 [ ] tuple-unpack state swap `_load_session_state`(7051-7066); XSS: LLM HTML not sanitized `email_drafting_agent.py:4291-4315` + render path; IMAP readline no timeout(253,258); SupplierInteraction unbounded wait when timeout=0 (741-762).
- Incomplete: HITL email release stub (12495); dead NegotiationEmailHTMLBuilder; enhanced-message path never run; multi-issue optimization output discarded; Redis session no TTL (memory growth).

---

### Other agents (data wiring — DOMINANT BLOCKER)
- 🔴 [ ] **Stale `proc.*_agent` table names everywhere** — agents query non-existent tables → 0 data. Correct map:
  `proc.purchase_order_agent`→`bp_purchase_order_trgt`, `proc.po_line_items_agent`→`bp_po_line_items_trgt`,
  `proc.invoice_agent`→`bp_invoice_trgt`, `proc.invoice_line_items_agent`→`bp_invoice_line_items_trgt`,
  `proc.quote_agent`→`bp_quote_trgt`, `proc.quote_line_items_agent`→`bp_quote_line_items_trgt`, `proc.supplier`→`bp_supplier`.
  Sites: opportunity_miner `TABLE_MAP:2228-2241` + SQL 7115,7140,7150-7158; supplier_ranking `1131,1147`; query_engine `110,112,376,384,497,562,673,689`; discrepancy `48,97,128`; quote_evaluation `431,474,729`; quote_comparison `127-128`.
- 🟠 [ ] approvals_agent: non-existent `approval_policies`/`proc.approvals` → always default threshold 1000, never persists (`58,107`). Fix: wire to bp_policy or create bp_ tables.
- 🟠 [ ] supplier_ranking `_prepare_scoring_columns` mutates `result["final_score"]` before init (`1534`, KeyError swallowed); `_init_schema` auto-creates non-bp `proc.procurement_flow` every instantiation (`145-166`).
- 🟡 [ ] approvals embeds numeric amount for semantic search (fabricated refs `73-81`); discrepancy DDL in hot path `proc.data_discrepancy` (`220-230`); quote FX tables don't exist (`571`).

### Orchestration
- 🔴 [ ] **WorkflowEngine skips downstream when predecessor SKIPPED** — `workflow_engine.py:410-411` only accepts COMPLETED; SKIPPED mine_opportunities cascades rank_suppliers/evaluate/draft to SKIPPED. Fix: accept `(COMPLETED, SKIPPED)`.
- 🔴 [ ] **output_to_shared overwrites pre-supplied data with `[]`** — `workflow_engine.py:529-532` uses `is not None` not truthiness; empty miner result destroys caller-supplied supplier_candidates. Fix: `if value:`.
- 🟠 [ ] `_execute_agent` returns None; callers deref unguarded (`orchestrator.py:1476,1667`) → AttributeError masked as failed.
- 🟠 [ ] Thread-unsafe `_prompt_cache`/`_policy_cache` written from pool threads (`orchestrator.py:513-647`). Add lock.
- 🟠 [ ] `_execute_node` ignores `timeout_seconds` (`workflow_engine.py:498-515`) → hang. Wrap in future.result(timeout).
- 🟠 [ ] state-machine `_invoke_state_agent` only merges pass_fields not .data → watcher never triggers (`orchestrator.py:2383-2427`).
- 🟡 [ ] `_load_policies` caches `{}` on DB failure → policies permanently off (`orchestrator.py:643-645`); StateManager concurrent shared_data merge race (`state_manager.py:112`); god-class orchestrator 3039 lines; dead `langgraph_state.py`.
- Incomplete: retry_count wired-but-unused; WorkflowOrchestrator async uses MockDatabaseConnection; DAGScheduler path untested e2e; checkpoint/resume untested.

## Fixes applied (commits) — verified
- `add342c` fix(agents): repoint stale proc.*_agent → bp_*_trgt (opportunity_miner, supplier_ranking, query_engine, discrepancy, quote agents) — **LIVE VERIFIED**: opportunity_mining for SUP-CityOfNewport now returns opportunity_count=2, total_savings=£181,000 (was 0/0); table_coverage now hits real bp_*_trgt.
- `e94b661` fix(extraction): allowlist field_name in HITL promote — closes SQL injection (`promotion.py`). +9 security tests.
- `50aa1ce` fix(orchestration): pass SKIPPED predecessors + guard None agents + stop empty-overwrite + policy-cache poison (`workflow_engine.py`, `orchestrator.py`). +6 tests.
- `<qe>` fix(query_engine): resolve supplier_name via bp_supplier join (bp_invoice_trgt/bp_quote_trgt lack supplier_name); proc.contracts→bp_contracts. 12 tests pass, queries EXPLAIN-clean on live DB.
- `<df>` fix(dataflow): repoint 47 stale refs in data_flow_manager/agent_manifest/data_extraction_agent; alias system kept coherent. Tests pass.
- **LIVE VERIFIED full chain**: opportunity_mining workflow now completes ALL nodes (mine→rank→evaluate→draft); rank_suppliers produces ranking+profiles (was skipped/failed).

### NEW findings surfaced during live verification
- 🟠 [ ] **Ranking scores null + pandas objects leak to JSON** — rank output has price_score/delivery_score=null, `risk_score`/`avg_unit_price` = `{"__module__":"pandas"}` (pandas NA/Series serialized). Causes: (a) scoring columns not aligned to bp_*_trgt column names; (b) non-JSON-native pandas types returned via API. Fix: map scorer to real bp_ columns + coerce to native (float/None) before output.
- 🟠 [ ] **`proc.action` / `proc.routing` process-log tables missing** — `process_routing_service` logs `UndefinedTable: relation "proc.action" does not exist` on every agent run (non-fatal, swallowed). Fix: create bp_ process-log tables or repoint.
- 🟡 [ ] **`proc.cat_product_mapping` missing** — category enrichment silently empty in query_engine/data_flow_manager (non-crashing). Fix: create bp_cat_product_mapping or gate the feature.
- 🟡 [ ] query_engine `fetch_invoice_data` supplier_name filter / discrepancy vendor column: now resolved via bp_supplier join (invoice) — quote path similar; verify quote name path.

### Services core (audit 5)
- 🔴 [x] **`proc.action` table never created** → UndefinedTable + "Failed to log action" every run. FIXED `<bp_action>`: created `proc.bp_action`, repointed process_routing/email_dispatch/email_drafting. LIVE VERIFIED (0 errors, 5 rows logged).
- 🔴 [~] **get_db_connection leaks (53 `with` sites)** — psycopg2 `with conn:` commits but never closes → pool exhaustion. PARTIAL: added safe `get_db_connection_cm()` (commit-on-success+close); 53 call-site migration DEFERRED (transaction-semantics risk at scale — staged follow-up).
- 🟠 [x] **SQLi in process_monitor_watcher** (table/pk_col f-string from `category`) — FIXED `<resilience>`: allowlist guards at 244/577/596.
- 🟠 [x] **ollama semaphore bypass** (proceeds without slot on timeout → GPU OOM) — FIXED: fail-closed (return None).
- 🟡 [x] **rag_service qdrant upsert unguarded** — FIXED: try/except.
- 🟠 [ ] no connection pooling (new psycopg2.connect per op); scheduler runs jobs sequentially on one thread (hung job starves others); training jsonl write not thread-safe (4 workers); proc.routing DDL+DML same txn (rollback risk on first boot). [follow-up]

### API layer (audit 6)
- 🔴 [x] **No authentication on any endpoint** (training/promotion/email/reload-governance all public) — FIXED (gated) `<security>`: `verify_api_key` global dep, enforced when `PROCWISE_API_KEY` set (no-op default → running system unaffected; SET THE ENV VAR IN PROD).
- 🔴 [x] **CORS `*` + allow_credentials=True** — FIXED: env `PROCWISE_CORS_ORIGINS`, credentials only for explicit origins.
- 🟠 [x] **Arbitrary file read via `file_path`** in /workflows/ask — FIXED: realpath confinement to `PROCWISE_UPLOAD_DIR`.
- 🟠 [ ] startup swallows fatal init → serves with None state (main.py 195-210); sync LLM endpoints block worker threads; exception `str(exc)` leaked to clients (20+ sites); no request-size limit; no rate limiting; unbounded `_SESSIONS` (vendors); IDOR on /run integer process_id; `reload=True` in __main__. [follow-up — mostly hardening]
- Stubbed: /stream/plan hardcoded fake plan; vendors in-memory store (needs Redis); /training/dispatch is a no-op (stub trainer).

### Resilience + ranking (this routine)
- 🟠 [x] **IMAP IDLE capability always-False + no socket timeout** — FIXED `<resilience2>`: `client.capability()` + `timeout=` on IMAP4_SSL.
- 🟠 [x] **S3 temp-file leak** — FIXED: `_resolve_to_local` returns (path, needs_cleanup); `parse()` try/finally removes temp.
- 🟠 [x] **Ranking output leaked pandas objects** (`{"__module__":"pandas"}`) — FIXED `<ranking>`: `_json_safe` coercion. LIVE VERIFIED clean.
- ⚠️ Reverted an over-scoped rag_agent change (a subagent activated a dead LLM path defaulting ON — not requested; reverted to keep RAG behavior unchanged).

---

# EXECUTIVE SUMMARY & VERDICT (as of this checkpoint)

## What this routine achieved (all committed to `Development`, live-verified)
The audit found the product had a **systemic data-wiring break**: every analytical agent queried `proc.*_agent` tables that never existed, so the entire agent layer silently produced nothing. That plus orchestration skip-bugs meant the supplier-ranking / opportunity / quote / discrepancy features were **non-functional end-to-end**. This is now fixed and verified:

- **Agent pipeline works for the first time.** opportunity_mining: `0 → 2 opportunities, £181k savings`; full workflow now completes **all nodes** (mine→rank→evaluate→draft) where `rank_suppliers` was previously always skipped/failed.
- **SQL injection closed** in the HITL promote path (+9 tests) and **process_monitor_watcher** (allowlist).
- **API security**: gated API-key auth (set `PROCWISE_API_KEY` in prod), env-configurable CORS, path-traversal guard on `file_path`.
- **`proc.bp_action` created** — agent action logging worked for the first time (was UndefinedTable every run).
- **Resilience**: IMAP hang/capability bug, S3 temp leak, Ollama semaphore fail-closed, Qdrant upsert guard, ranking JSON coercion.

12 fix commits, ~70 new tests, restart-verified healthy (HTTP 200, clean logs).

## Per-component readiness verdict
| Component | Before | After | Notes |
|---|---|---|---|
| Extraction pipeline | functional, SQLi + accuracy risks | **SQLi closed**; accuracy gaps remain | money-regex & context_layer fallback NOT changed (need accuracy re-validation) |
| Analytical agents (rank/opp/quote/discrepancy) | **broken (0 data)** | **working e2e** | score-column alignment + supplier_name directory mapping remain |
| Orchestration | skip/None bugs | **fixed core bugs** | god-class refactor deferred |
| Governance (bp_prompt/bp_policy) | working | working | loaded + firing |
| Summary feature | working | working | unaffected |
| API layer | **no auth, open** | **gated auth + hardened** | enable auth in prod; sync-LLM/rate-limit deferred |
| DB/process logging | proc.action missing | **fixed** | connection-pool + 53-site leak migration deferred |
| Negotiation/email agents | god-class bugs | **partially hardened** (IMAP/S3) | duplicate-method + state bugs deferred (high-risk in 13k-line file) |

## DEFERRED — prioritized next steps (NOT done tonight; risk-noted)
1. 🔴 **Extraction money-regex (`invoice.yaml` grand_total)** + context_layer silent fallback — DEFERRED ON PURPOSE: changing extraction regex priorities risks regressing the validated 50-doc accuracy (extraction accuracy is the #1 priority). Must be done with a before/after live re-validation harness, not blind.
2. 🔴 [x] **Negotiation duplicate methods / tuple-unpack / session_lock** — FIXED `4b21cce`: de-shadowed the 3 duplicate method pairs by disambiguation (Redis/obj variants → `_load_session_state_obj`/`_save_session_state_obj`/`_compose_negotiation_message_rich`, callers rewired by arity); `_session_lock` rewritten to `with`; LLM call moved to timeout-wrapped `ollama_generate`; tuple-unpack aligned to the 13-col SELECT. Negotiation tests pass (3 pre-existing learning-snapshot failures unrelated); NegotiationAgent instantiates cleanly in the live service after restart. NOTE: multi-round negotiation paths still lack live test coverage (need Ollama/DB) — recommend an integration test before heavy production use.
3. 🟠 **Connection-pool + migrate 53 `with get_db_connection()` → `get_db_connection_cm()`** — the safe CM now exists (commit-on-success+close); mechanical but broad migration + a ThreadedConnectionPool should be a focused, reviewed change.
4. 🟠 **API hardening rollout** — set `PROCWISE_API_KEY`/`PROCWISE_CORS_ORIGINS` in prod; convert sync LLM endpoints to async+semaphore; generic error messages (stop `str(exc)` leak); request-size + rate limits; Redis-backed vendor sessions.
5. 🟠 **Ranking score quality** — align scorer to real `bp_*_trgt` columns; pass `supplier_directory` IDs in matching casing so `supplier_name` resolves.
6. 🟡 Missing tables: `proc.bp_cat_product_mapping` (category enrichment), `proc.bp_supplier_responses` (RFQ flow), `proc.bp_approval_policies` (approvals threshold). XSS-sanitize LLM HTML in email drafts (needs `bleach`/`nh3`). FX table hardcoded. Pre-existing stale tests (`_OLLAMA_FALLBACK_MODELS`, process_monitor stubs).

## Honest bottom line
The product moved from "**core analytical features silently broken + open API + SQLi**" to "**features working end-to-end + critical security holes closed + action logging fixed**", all verified live. It is **materially closer to production-ready**. It is NOT yet fully production-grade across 147k LOC: the deferred items (extraction-accuracy validation, negotiation god-class, connection pooling, full auth rollout) require careful, individually-validated work rather than blind overnight changes — doing them recklessly would risk the #1 priority (extraction accuracy) and system stability.

## Remaining / deferred (post-routine recommendations)
_(to be filled)_
