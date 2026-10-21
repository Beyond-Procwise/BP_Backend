# The 25 files that still call the model directly (step 4b: classification and recommendation)

Written 2026-10-21 on branch `platform-skills`. Step 4a is done: the two tool loops that had no policy gate
(`governance_tools/governed_reasoning.py`, `supplier_enrichment/research.py`) now ask `tool_runtime.gate_refusal`
before every tool call and reach the model through `agentnick_driver`. What remains is
`tests/agentnick_direct_calls_baseline.txt` (25 files; it may only shrink).

## Why the rest was not swapped mechanically

They use a second wire, `ollama_client.ollama_generate` (the `/api/generate` endpoint), which keeps process-wide
state: a count of foreground calls (`foreground_busy()`, which background work such as bulk translation yields
to), a request queue, retries, and a memory of GPU layout refusals. The callers import it through TWO paths:

- `services.ollama_client`: negotiation/analysis, agent_policy/extractor, structural_extractor (one branch),
  style/exemplars, style/compiler, style/drafting
- `src.services.ollama_client`: reasoning_engine, summary_agent, analytics/insight, obligations/agentnick,
  deal_summary, rga/compose, i18n/adapters, extraction_feedback/proposer, extraction_v3 judges,
  structural_extractor (other branch), negotiation_advice, sell_side x2, training/eval_gate, extraction/context_layer

Python loads those as two separate modules, so there are two foreground counters and two refusal memories. A
foreground call made through one copy is invisible to `foreground_busy()` on the other. Pointing every caller at a
driver that delegates to one copy would move about half of them onto different state. That needs its own decision
and a test that proves what changes; it is not part of "route through the driver".

## What each file does with the model's output

**Control** (the model's output decides what runs)
- `orchestration/reasoning_engine.py` `_llm_compose_plan`: composes a plan. The engine is not wired into the live
  path (see the 2026-06-18 diagnosis); leave it until it is used or removed.

**Proposals that are already propose-only or already gated** (output goes to a person, or through a gate)
- `sell_side/quote_agent.py` ("one instruction in, one PROPOSAL out"), `sell_side/template_reader.py` (PROPOSED template)
- `rga/compose.py` (report prose, with a gate on each place the model writes), `analytics/insight.py` (one sentence, with a gate)
- `style/compiler.py`, `style/drafting.py`, `style/exemplars.py` (style profile and email drafts)
- `negotiation_advice/advisor.py`, `summary_agent.py`, `deal_summary.py`, `i18n/adapters.py` (wording and translations)
- `opportunity_critic/llm.py` (a critique that adjusts a finding's value; no tools, no output gate)

**Output becomes stored data** (the model is a source of record)
- Extraction: `agents/extraction_engine.py` (NuExtract calls, supplier identification), `extraction/context_layer.py`,
  `structural_extractor/llm_fallback.py`, `extraction_v3/judge/grounded_last_resort.py`,
  `extraction_v3/judge/schema_coherence.py`
- Proposals that people review: `extraction_feedback/proposer.py` (hint drafts), `agent_policy/extractor.py`
  (policy drafts from documents)
- `obligations/agentnick.py` (Hyper-Extract reading contract prose into obligations)
- `agents/negotiation/analysis.py` `extract_negotiation_signals`: signals that feed a negotiation decision

**Not a caller of the model for a user's task**: `training/eval_gate.py` (evaluation harness), `ollama_client.py` (the wire).

## Recommendation, in order

1. **Leave extraction alone until you decide.** Memory and the standing rule are that extraction accuracy comes
   first and is never traded for anything. Any change here needs a golden-set before and after. The decision for
   you: should the model's extraction output be treated as a proposal that a verifier accepts (as the grounding
   guard already does), or stay as it is?
2. **DONE 2026-10-21: the two import paths of `ollama_client` are unified.** `src/services/__init__.py` lists
   `_SHARED_STATE_MODULES = ("ollama_client",)`; when the package is loaded as `services`, an import hook answers
   `services.ollama_client` with the one real `src.services.ollama_client`. No caller changed. Tests in
   `tests/test_import_unification.py` (fresh interpreters, both import orders) show identity and shared foreground
   count and refusal memory. The 139 test files touching these callers fail identically before and after (10 each).
3. **Then** add `agentnick_driver.generate()` delegating to the unified client, and move the wording/propose-only
   callers onto it (they change only an import). The baseline then shrinks file by file.
4. **Mark outputs as proposals** where they become stored data, one family at a time, starting with
   `agent_policy/extractor.py` and `extraction_feedback/proposer.py`, which already go to a human.

## Other modules loaded as two copies (found while doing this; NOT changed)

41 modules under `src/services` are imported both as `services.X` and `src.services.X`. Most hold no state of
consequence. The ones that do, and are split, in order of concern:

- `event_bus` (10 importers as `services`, 1 as `src.services`): a process-wide singleton, so two buses, plus a
  per-thread workflow context that was also split. The mail-intake sweep in `backend_scheduler.py` publishes
  `DOCUMENT_INGESTED` on the `src` copy; every subscriber was on the `services` copy. Latent, not active: nothing
  subscribes to `DOCUMENT_INGESTED` today. **UNIFIED 2026-10-21** (added to `_SHARED_STATE_MODULES`); 129 test files
  that touch it fail identically before and after (52 failures, 17 errors, all long-standing).
- `db` (98 / 206): holds a fake store for tests and connection helpers; check the pool before unifying.
- `egress`, `model_selector`, `backend_scheduler`, `email_watcher`: some module-level state each.

Add a module to `_SHARED_STATE_MODULES` only after checking what state it holds and what depends on it.
