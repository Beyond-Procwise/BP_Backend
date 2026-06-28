# Extraction pipelines — which code is live, and why several exist

There are **three** extraction code trees under `src/services/`. Only one runs in
production. This note records which, to save the next person the archaeology.

## Live: `extraction/` — the "renovation" pipeline  ✅ PRODUCTION

- Entry: `extraction/dispatch.py::dispatch_document(process_monitor_id, file_path, doc_type)`
- Selected when `EXTRACTION_RENOVATION_ENABLED=1` (it is, in `.env` and the
  running service). The watcher (`process_monitor_watcher.py`) routes here.
- Layering: L0 `parser.parse` → L1 `pattern_extractor` (regex) → L2
  `engineered/` (NER + table) → L3 `judge_gate` (grounded judge) →
  `context_layer.synthesize` (AgentNick, **authoritative**) → `persistence` + `promotion`.
- `pipeline_version` is stamped as `<gitsha>-renov` in every `dispatch end` log line.
- The AgentNick LLM call goes through the managed `ollama_client` (semaphore + retry).

## Legacy fallback: `extraction_v3/` (+ nested `extraction_v4/`)

- Entry: `extraction_v3/dispatch.py::dispatch_document(agent_nick, file_path, category, ...)`
- Runs **only** when `EXTRACTION_RENOVATION_ENABLED` is unset/false.
- `extraction_v4/engine.py` is the hybrid engine (NuExtract + `llm_extractor`
  LLM-fill). Still imported by the v3 dispatch and kept as a safety fallback.
- Not the production path today. Schema YAML loaders here ARE shared by the
  renovation pipeline (`extraction_v3/yaml_schema/loader.py`,
  `extraction_v3/judge/grounded_last_resort.py`), so v3 is not dead code.

## Library: `extraction_v2/`

- Template store, parsers, invariants. Imported widely (75+ sites) as a support
  library by both v3 and the binding/validation layers. Not a standalone pipeline.

## Do / don't

- **Don't** "merge" `extraction/dispatch.py` and `extraction_v3/dispatch.py` —
  they are different pipelines (renovation vs legacy), not duplicates.
- **Do** make production behaviour changes in `extraction/` (the renovation tree).
- The two `dispatch_document` functions have **different signatures** on purpose
  (renovation is keyword-only with `process_monitor_id`; v3 takes `agent_nick`/`category`).
