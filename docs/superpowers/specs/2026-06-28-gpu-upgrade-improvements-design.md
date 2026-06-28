# GPU-Upgrade Improvements — Design Spec

**Date:** 2026-06-28
**Branch:** Nick
**Status:** Approved design — pending implementation plan
**Hardware context:** GPU upgraded to **NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM** (previously a small, contended GPU that forced LLM timeouts and aggressive throttling).

## Goal

Capitalise on the GPU upgrade across three independent, sequenced workstreams:

1. **A — Extraction speed:** make extraction complete much sooner by lifting throttles the old GPU forced, and by routing all LLM traffic through one managed client.
2. **B — AgentNick intelligence:** improve procurement precision via prompt/pattern refinement (safe-first), plus a *gated* real finetune experiment that can only ship if it beats baseline.
3. **C — Code quality:** safe cleanups (no behaviour change) plus splitting the single largest god-class file behind its existing interface.

**Sequencing is A → B → C.** Each workstream is independent and ends with a **live-server robustness check on `bp_sqldb` and a written before/after report**. Implementation of a later workstream does not begin until the earlier one's report is produced and reviewed.

## Guiding constraints (from project memory / standing rules)

- **Extraction accuracy is priority #1.** Never modify source data; never fabricate; leave NULL when absent. No change may reduce extraction accuracy.
- **AgentNick is the ONLY base model.** Never repoint to a non-AgentNick serving model. Finetuning AgentNick's own qwen lineage is allowed; swapping to Gemma/other is not.
- **Prove on the running local server against live `bp_sqldb`**, not just tests/mocks.
- **Avoid over-engineering.** Match scope to requirement; renovate over rewrite; no speculative ML or multi-wave roadmaps.
- **Regex-primary extraction direction** stays intact; these changes are about speed, model quality, and structure — not re-architecting the extraction layering.

## Current-state ground truth (verified 2026-06-28)

| Lever | Current value | Set for | File / location |
|---|---|---|---|
| Ollama `OLLAMA_NUM_PARALLEL` | 2 | old GPU | systemd `ollama.service` env |
| Ollama `OLLAMA_MAX_LOADED_MODELS` | 3 | old GPU | systemd `ollama.service` env |
| Ollama `OLLAMA_KEEP_ALIVE` | 5m | old GPU | systemd `ollama.service` env |
| App `OLLAMA_MAX_CONCURRENT` | 2 | old GPU | `src/services/ollama_client.py:28` |
| Document worker pool | 4 | old GPU | `src/services/process_monitor_watcher.py:27` `DEFAULT_MAX_WORKERS` |
| Modelfile `num_gpu` | 25 (partial offload) | scarce VRAM | `Modelfile:6` |
| Modelfile base | `qwen3:30b` | — | `Modelfile:1` |

VRAM budget check: AgentNick:extract ≈ 8 GB + :unified/:latest ≈ 18 GB + nuextract ≈ 2 GB ≈ **28 GB of 96 GB** — all three can stay resident concurrently with large headroom.

Model variants in use:
- `BeyondProcwise/AgentNick:extract` (qwen2.5-7B, extraction specialist) — extraction path.
- `BeyondProcwise/AgentNick:unified` / `:latest` (qwen3-30B) — reasoning/summaries/agents.

Baseline accuracy: **doc_accuracy 0.8482** (`artifacts/e2e_audit/eval_ft_result.json`); a prior real finetune regressed candidate to **0.0** — this is the risk Workstream B must guard against.

---

## Workstream A — Extraction speed (lift the throttles)

### Scope
Config-level, reversible, env-driven changes plus plugging unmanaged LLM call sites into the shared client. **No extraction-engine algorithm changes; no in-document field/judge parallelism this pass** (explicitly deferred — see Non-Goals).

### Changes
1. **Ollama server env** (systemd drop-in): `OLLAMA_NUM_PARALLEL` 2→8; `OLLAMA_KEEP_ALIVE` 5m→30m; keep `OLLAMA_MAX_LOADED_MODELS=3`. Documented in repo (e.g. `resources/` deployment notes) so the change is reproducible, not just a live edit.
2. **App client** `src/services/ollama_client.py`: `OLLAMA_MAX_CONCURRENT` default 2→8 (still env-overridable).
3. **Document pool** `src/services/process_monitor_watcher.py`: `DEFAULT_MAX_WORKERS` 4→8 (env-overridable).
4. **Modelfile** `num_gpu` 25→-1 (full GPU offload). Rebuild `:latest` via the existing `ollama create` path. Leave `num_ctx`/`temperature` unchanged.
5. **Route unmanaged LLM calls through `ollama_client`:**
   - NuExtract chunk calls `src/services/extraction_v3/extraction_v4/engine.py` (`_call_nuextract_*`, ~549-694) — currently raw `requests.post`, serial per chunk, no semaphore.
   - LLM-fill `src/services/extraction_v3/extraction_v4/llm_extractor.py` (`_call_ollama`, ~362-400) — currently raw `requests.post`.
   - Both go behind `ollama_client.ollama_generate()` so all LLM traffic shares one concurrency budget + retry/backoff. This is correctness/observability as much as speed (prevents request storms).

### Robustness / proof (required deliverable)
- Pick a fixed sample of **live documents from `bp_sqldb`** (same set before and after).
- Measure: end-to-end wall-clock per doc and for the batch; LLM call count; **field-level extraction accuracy on the sample must be unchanged** (accuracy is priority #1 — speed must not cost accuracy).
- Output: `artifacts/gpu_upgrade/A_speed_before_after.md` with before/after table and the doc sample IDs.

### Acceptance criteria
- Measurable wall-clock reduction on the live batch.
- Extraction accuracy on the sample is **equal or better** than before (no regression).
- All LLM call sites route through `ollama_client` (no remaining raw `requests.post` to `/api/generate` in the extraction path).
- All new limits are env-overridable so they can be tuned without code edits.

---

## Workstream B — AgentNick intelligence (safe-first, gated experiment)

### Phase B1 — Prompt / schema / pattern refinement (ships first, zero regression risk)
- Refine the Modelfile system prompt, document schemas, and the auto-learned supplier-pattern section (`model_sync_service.py`) for sharper procurement precision and data-flow handling.
- **Gate:** every candidate Modelfile is scored by the existing `src/training/eval_gate.py` against the holdout set. A change ships **only if doc_accuracy ≥ 0.847** (no regression). Anything that regresses is rejected.

### Phase B2 — Gated real finetune (the daily cron made honest)
- Today `src/training/pipeline.py::_train_model` is a **stub that trains nothing**; the daily cron therefore does no real work.
- Implement real QLoRA training **on the AgentNick/qwen lineage** (reuse the working `scripts/finetune_native_agentnick.py` mechanics, retargeted to the qwen base — not Gemma), runnable on the new GPU.
- **Promotion gate:** the trained candidate is registered to a staging tag and scored by `eval_gate`. It is promoted to `:latest` **only if it beats 0.847**; otherwise it is discarded and `:latest` is untouched. The daily cron is rewired to this gated flow so it can never silently ship a regression.

### Robustness / proof (required deliverable)
- Run `eval_gate` before/after on the same holdout; additionally validate on a **live `bp_sqldb` document sample** end-to-end.
- Output: `artifacts/gpu_upgrade/B_agentnick_before_after.md` with eval_gate deltas, the promote/reject verdict, and live-sample accuracy.

### Acceptance criteria
- B1 prompt/pattern improvements shipped only where eval_gate ≥ 0.847.
- B2 produces a real trained candidate and a recorded promote-or-reject decision driven solely by eval_gate.
- `:latest` is never worse than 0.847 at any point.
- No serving model outside the AgentNick lineage is introduced.

---

## Workstream C — Code quality (safe wins + one god-class split)

### Safe wins (no behaviour change)
- Remove `Modelfile.prepollutionfix.bak` after confirming nothing references it.
- Consolidate the two duplicate dispatchers (`src/services/extraction/dispatch.py` vs `src/services/extraction_v3/dispatch.py`) into one canonical module; update imports.
- Move ad-hoc `main()` / debug scripts out of `src/` into `scripts/` (do not delete; relocate).
- Add a short `README` documenting why `extraction_v2` / `extraction_v3` / `extraction_v4` coexist and which is canonical.

### One god-class split
- Split **`src/agents/negotiation_agent.py` (13,296 lines — the single largest file)** into ~5 focused modules: email threading, supplier signals, negotiation strategy, position management, HTML building — **behind its existing public interface** (no caller changes).
- Chosen over the extraction engine deliberately: negotiation is **outside the accuracy-critical extraction path**, so the split cannot endanger extraction accuracy. The extraction engine split is staged as a separate future effort.

### Robustness / proof (required deliverable)
- Verify the negotiation agent produces **equivalent behaviour** before/after the split on a live scenario (same inputs → same outputs).
- Confirm safe-win cleanups changed no behaviour (imports resolve, app boots, extraction path unaffected).
- Output: `artifacts/gpu_upgrade/C_quality_before_after.md` listing files removed/moved/split, LOC before/after, and the behaviour-equivalence evidence.

### Acceptance criteria
- App imports and boots cleanly; no broken imports from moved/consolidated files.
- `negotiation_agent` behaviour is equivalent pre/post split on the live scenario.
- No change to the extraction path or its accuracy.

---

## Non-Goals (explicitly out of scope this pass)
- In-document parallelism (running one document's fields/judges concurrently) — deferred.
- Splitting the extraction engine or other god-classes beyond the one named — staged separately.
- Re-architecting the regex→engineered→AI-judge extraction layering.
- Introducing any non-AgentNick serving model.
- Unrelated refactoring not in service of the three workstreams.

## Overall deliverables
- Three before/after reports under `artifacts/gpu_upgrade/` (A, B, C).
- All limits env-overridable; all model promotions eval_gate-gated.
- A final short roll-up referencing the three reports for the robustness sign-off.

## Risks & mitigations
- **Speed at the cost of accuracy (A):** mitigated by requiring equal-or-better field accuracy on the live sample as an acceptance gate.
- **Finetune regression (B):** mitigated by eval_gate promotion gate; regressions auto-discarded.
- **Concurrency overload:** new limits are conservative (8) vs 96 GB headroom and remain env-tunable; can be dialled back live.
- **God-class split breaking callers (C):** split stays behind the existing interface; behaviour-equivalence check on live scenario before sign-off.
