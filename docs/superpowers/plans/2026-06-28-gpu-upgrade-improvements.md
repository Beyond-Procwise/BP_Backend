# GPU-Upgrade Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capitalise on the new 96 GB Blackwell GPU by speeding up extraction (A), sharpening AgentNick under an eval gate (B), and removing code debt + splitting one god-class (C) — each proven on live `bp_sqldb` with a before/after report.

**Architecture:** A is config + plumbing (lift old-GPU throttles, route all LLM calls through the managed `ollama_client`). B refines the Modelfile/patterns and turns the dead finetune cron into a real-but-eval-gated job. C does safe cleanups then splits `negotiation_agent.py` behind its existing interface. Workstreams run A→B→C; each ends with a robustness report.

**Tech Stack:** Python 3, Ollama (qwen3:30b / qwen2.5-7B AgentNick variants), `requests`, `concurrent.futures.ThreadPoolExecutor`, existing `src/training/eval_gate.py`, PostgreSQL `bp_sqldb`, systemd `ollama.service`.

## Global Constraints

- Extraction accuracy is priority #1 — no change may reduce field-level accuracy on the live sample. Never modify source data; never fabricate; NULL when absent.
- AgentNick is the ONLY base model — finetune the qwen lineage only; never introduce a non-AgentNick serving model.
- Prove every workstream on the running local server against live `bp_sqldb`, not mocks.
- All new limits must be env-overridable (no hard-coded magic that needs a code edit to tune).
- Model promotion to `:latest` is allowed only when `eval_gate` doc_accuracy ≥ 0.847 (current baseline).
- Frequent commits; no Co-Authored-By/Claude attribution lines in commit messages.

---

## Phase A — Extraction speed (lift the throttles)

### Task A1: Capture the live "before" baseline

**Files:**
- Create: `scripts/gpu_upgrade/bench_extraction.py`
- Create: `artifacts/gpu_upgrade/` (dir)

**Interfaces:**
- Produces: `bench_extraction.py` — `run_bench(doc_ids: list[str], label: str) -> dict` writing `artifacts/gpu_upgrade/bench_<label>.json` with per-doc wall-clock (s), batch total (s), LLM call count, and field-level accuracy vs each doc's persisted `_stg`/`_trgt` values.

- [ ] **Step 1:** Pick a fixed sample of ≥10 real documents already in `bp_sqldb` spanning invoice/PO/quote. Record their doc PKs/source keys in `artifacts/gpu_upgrade/sample_docs.txt`.
- [ ] **Step 2:** Write `bench_extraction.py` that re-runs `dispatch_document()` for each sample doc, times each end-to-end, counts LLM calls (instrument via a counter env or log parse), and compares extracted fields against the currently-persisted values to compute field-level accuracy.
- [ ] **Step 3:** Run it: `python scripts/gpu_upgrade/bench_extraction.py --label before`. Expected: `artifacts/gpu_upgrade/bench_before.json` exists with non-zero timings and an accuracy number.
- [ ] **Step 4:** Commit.

```bash
git add -f scripts/gpu_upgrade/bench_extraction.py artifacts/gpu_upgrade/sample_docs.txt artifacts/gpu_upgrade/bench_before.json
git commit -m "perf(extraction): capture live before-baseline for GPU-upgrade speed work"
```

### Task A2: Raise the Ollama server + app concurrency limits

**Files:**
- Modify: systemd drop-in for `ollama.service` (`/etc/systemd/system/ollama.service.d/override.conf`)
- Modify: `src/services/ollama_client.py:28`
- Modify: `src/services/process_monitor_watcher.py:27`
- Create: `resources/deployment/ollama_env.md` (documents the server env so it's reproducible)

**Interfaces:**
- Produces: env-tunable concurrency — `OLLAMA_NUM_PARALLEL=8`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_CONCURRENT` default 8, `DEFAULT_MAX_WORKERS` default 8.

- [ ] **Step 1:** In `ollama_client.py:28`, change default `"2"` → `"8"` and update the line-27 comment to reference the 96 GB GPU.

```python
# Max concurrent Ollama requests — 96 GB Blackwell GPU; match OLLAMA_NUM_PARALLEL (default 8)
_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "8"))
```

- [ ] **Step 2:** In `process_monitor_watcher.py:27`, change `DEFAULT_MAX_WORKERS = 4` → read env with default 8.

```python
DEFAULT_MAX_WORKERS = int(os.getenv("PROCWISE_DOC_WORKERS", "8"))
```

- [ ] **Step 3:** Update the systemd drop-in: `OLLAMA_NUM_PARALLEL=8`, `OLLAMA_KEEP_ALIVE=30m`, keep `OLLAMA_MAX_LOADED_MODELS=3`. Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
- [ ] **Step 4:** Verify: `systemctl show ollama | grep -i Environment` shows `OLLAMA_NUM_PARALLEL=8`; `printenv` check inside app context shows the new defaults. Record the final env in `resources/deployment/ollama_env.md`.
- [ ] **Step 5:** Commit.

```bash
git add -f src/services/ollama_client.py src/services/process_monitor_watcher.py resources/deployment/ollama_env.md
git commit -m "perf(extraction): raise Ollama + doc-pool concurrency to 8 for 96GB GPU"
```

### Task A3: Full-GPU model offload in the Modelfile

**Files:**
- Modify: `Modelfile:6`
- Modify: `src/services/ollama_client.py:67-71` (stale OOM comment)

- [ ] **Step 1:** Change `Modelfile:6` `PARAMETER num_gpu 25` → `PARAMETER num_gpu -1` (auto-fit all layers; 30B ≈ 18 GB fits in 96 GB with headroom).
- [ ] **Step 2:** Rebuild: `ollama create BeyondProcwise/AgentNick:latest -f Modelfile`. Verify `ollama ps` shows the model loaded with `PROCESSOR` = 100% GPU when invoked.
- [ ] **Step 3:** Update the stale comment in `ollama_client.py:67-71` (it warns num_gpu=99 OOM-kills on a *contended* card — no longer true on 96 GB); keep the auto-fit default behaviour.
- [ ] **Step 4:** Smoke test: one `ollama_generate` call returns text. Commit.

```bash
git add -f Modelfile src/services/ollama_client.py
git commit -m "perf(model): full-GPU offload (num_gpu=-1) now that VRAM is ample"
```

### Task A4: Route NuExtract through the managed client + parallelize chunks

**Files:**
- Modify: `src/services/extraction_v3/extraction_v4/engine.py` — `_call_nuextract_invoice/_po/_quote` (~549, ~640+, ~679) and `nuextract_extract_entities_invoice/_po/_quote` (~525, ~611, ~679)

**Interfaces:**
- Consumes: `ollama_client.ollama_generate(prompt, *, model, num_predict, timeout)`.
- Produces: NuExtract calls share the global semaphore; chunk loops run concurrently.

- [ ] **Step 1:** Replace each `_call_nuextract_*` body's raw `requests.post(.../api/generate, ...)` with a managed call, preserving the template prompt and parsing:

```python
from services.ollama_client import ollama_generate  # module-level import
...
def _call_nuextract_invoice(text: str) -> dict:
    template_str = json.dumps(INVOICE_TEMPLATE, indent=2)
    prompt = f"<|input|>\n{text}\n<|template|>\n{template_str}\n<|output|>"
    raw_output = ollama_generate(
        prompt, model=NUEXTRACT_MODEL, num_predict=2048, timeout=NUEXTRACT_TIMEOUT,
    ) or ""
    logger.debug("NuExtract raw output: %s", raw_output[:500])
    return _parse_response(raw_output.strip())
```

- [ ] **Step 2:** Parallelize the chunk loop in each `nuextract_extract_entities_*` using a bounded `ThreadPoolExecutor` (the `ollama_client` semaphore caps real concurrency, so this just feeds it):

```python
from concurrent.futures import ThreadPoolExecutor
...
def nuextract_extract_entities_invoice(text, labels=None):
    chunks = _split_text(text, max_chars=3000)
    entities: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(chunks)))) as ex:
        results = list(ex.map(_safe_call_nuextract_invoice, chunks))
    for extracted in results:
        for key, value in extracted.items():
            # ... existing _LABEL_MAP_INVOICE aggregation, unchanged ...
    return entities
```

with a `_safe_call_nuextract_invoice(chunk)` helper that wraps `_call_nuextract_invoice` in the existing try/except (returns `{}` on failure). Repeat for PO and quote.

- [ ] **Step 3:** Run the unit/smoke path on one sample doc; confirm extracted entities match the pre-change output (order-independent compare).
- [ ] **Step 4:** Commit.

```bash
git add -f src/services/extraction_v3/extraction_v4/engine.py
git commit -m "perf(extraction): route NuExtract via managed client + parallelize chunks"
```

### Task A5: Route the LLM-fill call through the managed client

**Files:**
- Modify: `src/services/extraction_v3/extraction_v4/llm_extractor.py:362-400` (`_call_ollama`)

- [ ] **Step 1:** Replace the raw `requests.post` in `_call_ollama` with `ollama_generate(prompt, model=LLM_MODEL, num_predict=..., timeout=LLM_TIMEOUT)`, preserving the existing prompt construction, `MAX_TEXT_CHARS` cap, and return parsing.
- [ ] **Step 2:** Smoke test `llm_fill_all_extractable` on one doc with missing required fields; confirm identical fill behaviour.
- [ ] **Step 3:** Grep guard: `grep -rn "requests.post" src/services/extraction_v3/extraction_v4/` returns no `/api/generate` call sites. Commit.

```bash
git add -f src/services/extraction_v3/extraction_v4/llm_extractor.py
git commit -m "perf(extraction): route LLM-fill via managed client"
```

### Task A6: Capture "after" + write the A report

**Files:**
- Create: `artifacts/gpu_upgrade/A_speed_before_after.md`

- [ ] **Step 1:** `python scripts/gpu_upgrade/bench_extraction.py --label after` on the SAME `sample_docs.txt`.
- [ ] **Step 2:** Write `A_speed_before_after.md`: before/after per-doc and batch wall-clock, LLM-call counts, and field-accuracy. **Gate:** after-accuracy ≥ before-accuracy AND batch wall-clock reduced. If accuracy dropped, stop and investigate (do not proceed to B).
- [ ] **Step 3:** Commit.

```bash
git add -f artifacts/gpu_upgrade/bench_after.json artifacts/gpu_upgrade/A_speed_before_after.md
git commit -m "perf(extraction): live before/after report — Workstream A"
```

---

## Phase B — AgentNick intelligence (safe-first, gated)

### Task B1: Establish the eval-gate harness as the promotion gate

**Files:**
- Modify/Verify: `src/training/eval_gate.py`
- Create: `scripts/gpu_upgrade/run_eval_gate.py` (thin CLI wrapper if one doesn't already exist)

**Interfaces:**
- Consumes: `eval_gate(candidate_fn, baseline_fn, examples, tolerance=0.0) -> {passed, delta_doc_accuracy, baseline, candidate, verdict}`.
- Produces: `run_eval_gate.py --candidate <modelfile_or_tag>` prints verdict + writes `artifacts/gpu_upgrade/eval_<tag>.json`.

- [ ] **Step 1:** Confirm `eval_gate.py` runs against the holdout and reproduces baseline ≈ 0.8482 for the current `:latest`/`:extract`. Record in `artifacts/gpu_upgrade/eval_baseline.json`.
- [ ] **Step 2:** Write the CLI wrapper that builds a candidate generate-fn from a given model tag and runs `eval_gate` vs the current baseline.
- [ ] **Step 3:** Commit.

```bash
git add -f scripts/gpu_upgrade/run_eval_gate.py artifacts/gpu_upgrade/eval_baseline.json
git commit -m "feat(agentnick): eval-gate CLI as the model-promotion gate"
```

### Task B2: Prompt / schema / pattern refinement (ships only if eval_gate ≥ 0.847)

**Files:**
- Modify: `Modelfile` (system prompt + schema sections)
- Verify: `src/services/model_sync_service.py` (pattern injection)

- [ ] **Step 1:** Make targeted system-prompt/schema improvements for procurement precision and data-flow handling (e.g. clearer supplier-vs-buyer rules, tax/total invariants, line-item stop tokens). Keep edits minimal and reversible.
- [ ] **Step 2:** Rebuild a candidate tag: `ollama create BeyondProcwise/AgentNick:cand-b2 -f Modelfile`.
- [ ] **Step 3:** `python scripts/gpu_upgrade/run_eval_gate.py --candidate BeyondProcwise/AgentNick:cand-b2`. **Gate:** promote to `:latest` ONLY if doc_accuracy ≥ 0.847; else revert the Modelfile edits.
- [ ] **Step 4:** If promoted, rebuild `:latest` and commit the Modelfile; else commit nothing for the prompt (record the rejected attempt in the report).

```bash
git add -f Modelfile && git commit -m "feat(agentnick): prompt/schema refinement (eval_gate-passed)"
```

### Task B3: Make the daily finetune real and eval-gated

**Files:**
- Modify: `src/training/pipeline.py:293-325` (`_train_model`, `_merge_adapters` stubs)
- Modify: `scripts/run_overnight_finetune.sh` (promotion step) and `scripts/agentnick_finetune_daily.sh`

**Interfaces:**
- Consumes: existing QLoRA mechanics from `scripts/finetune_native_agentnick.py`, retargeted to the qwen base.
- Produces: a trained candidate registered to a staging tag, promoted to `:latest` only on eval_gate pass.

- [ ] **Step 1:** Implement `_train_model` to run real QLoRA on the qwen base (NOT Gemma), using `data/training/*.jsonl`, writing the adapter to `cfg.output_dir`. Implement `_merge_adapters` to actually merge.
- [ ] **Step 2:** In `run_overnight_finetune.sh`, register the merged GGUF to a **staging** tag `BeyondProcwise/AgentNick:cand-nightly` (not `:latest`).
- [ ] **Step 3:** Add a promotion gate step: run `run_eval_gate.py --candidate BeyondProcwise/AgentNick:cand-nightly`; only `ollama cp` it to `:latest` if `passed`. Log the verdict.
- [ ] **Step 4:** Dry-run the pipeline once on the new GPU; capture the eval verdict (promote or reject). Commit.

```bash
git add -f src/training/pipeline.py scripts/run_overnight_finetune.sh scripts/agentnick_finetune_daily.sh
git commit -m "feat(agentnick): real eval-gated nightly finetune (no silent regression)"
```

### Task B4: Live validation + write the B report

**Files:**
- Create: `artifacts/gpu_upgrade/B_agentnick_before_after.md`

- [ ] **Step 1:** Run the bench from A6 again with whichever `:latest` is current after B; compare extraction accuracy on the live sample vs A's after-numbers.
- [ ] **Step 2:** Write `B_agentnick_before_after.md`: eval_gate deltas for B2 and B3, promote/reject verdicts, and live-sample accuracy. **Gate:** `:latest` accuracy ≥ 0.847 at all times. Commit.

```bash
git add -f artifacts/gpu_upgrade/B_agentnick_before_after.md
git commit -m "feat(agentnick): live before/after report — Workstream B"
```

---

## Phase C — Code quality (safe wins + one god-class split)

### Task C1: Safe cleanups (no behaviour change)

**Files:**
- Delete: `Modelfile.prepollutionfix.bak`
- Consolidate: `src/services/extraction/dispatch.py` + `src/services/extraction_v3/dispatch.py` → one canonical module
- Move: ad-hoc `main()`/debug scripts from `src/` → `scripts/`
- Create: `src/services/README_extraction_versions.md`

- [ ] **Step 1:** `grep -rn "prepollutionfix" .` → confirm zero references, then `git rm -f Modelfile.prepollutionfix.bak`.
- [ ] **Step 2:** Diff the two `dispatch.py` files; pick the canonical one (the v3 path used by `dispatch_document`), re-point imports of the other, delete the duplicate. Verify `python -c "import src..."` style smoke that the app imports.
- [ ] **Step 3:** Identify the `src/` files with `if __name__ == "__main__"` that are debug/demo (per the code-quality survey); `git mv` them to `scripts/`, fix any imports.
- [ ] **Step 4:** Write `README_extraction_versions.md` stating v3/v4 is canonical, v2 is load-bearing (75 imports), and the migration intent.
- [ ] **Step 5:** Boot the app (`uvicorn`/entrypoint) — confirm clean startup, no import errors. Commit.

```bash
git add -A && git commit -m "refactor(quality): remove .bak, consolidate dispatch, relocate debug scripts, document v2/v3/v4"
```

### Task C2: Characterize negotiation_agent before splitting

**Files:**
- Create: `tests/agents/test_negotiation_agent_characterization.py`

- [ ] **Step 1:** Write characterization tests capturing current public behaviour of `negotiation_agent.py` on a representative live scenario (same inputs → snapshot outputs). These lock behaviour before the split.
- [ ] **Step 2:** Run them green against the current monolith. Commit.

```bash
git add -f tests/agents/test_negotiation_agent_characterization.py
git commit -m "test(negotiation): characterization tests before god-class split"
```

### Task C3: Split negotiation_agent.py behind its existing interface

**Files:**
- Modify: `src/agents/negotiation_agent.py` (becomes a thin facade re-exporting the public class/API)
- Create: `src/agents/negotiation/email_threading.py`, `supplier_signals.py`, `strategy.py`, `position.py`, `html_builder.py`

- [ ] **Step 1:** Extract the email-threading helpers into `negotiation/email_threading.py`; import back into the facade. Run characterization tests — must stay green.
- [ ] **Step 2:** Repeat for supplier signals, strategy, position management, HTML building — one module per commit, characterization tests green after each.
- [ ] **Step 3:** Confirm `negotiation_agent.py`'s public imports are unchanged for all callers (`grep -rn "from agents.negotiation_agent import"` still resolves). Commit each extraction.

```bash
git add -f src/agents/negotiation_agent.py src/agents/negotiation/
git commit -m "refactor(negotiation): extract <module> from god-class (behaviour-equivalent)"
```

### Task C4: Live verification + write the C report

**Files:**
- Create: `artifacts/gpu_upgrade/C_quality_before_after.md`

- [ ] **Step 1:** Run the negotiation characterization tests + boot the full app; confirm extraction path untouched (re-run A6 bench accuracy unchanged).
- [ ] **Step 2:** Write `C_quality_before_after.md`: files removed/moved, LOC before/after per file, behaviour-equivalence evidence, app-boot confirmation. Commit.

```bash
git add -f artifacts/gpu_upgrade/C_quality_before_after.md
git commit -m "refactor(quality): live before/after report — Workstream C"
```

### Task C5: Final robustness roll-up

**Files:**
- Create: `artifacts/gpu_upgrade/ROLLUP.md`

- [ ] **Step 1:** Write `ROLLUP.md` linking the three reports with a one-line robustness sign-off each (speed gain, accuracy held, model gated, quality reduced with no behaviour change). Commit.

```bash
git add -f artifacts/gpu_upgrade/ROLLUP.md
git commit -m "docs(gpu-upgrade): final robustness roll-up across A/B/C"
```

---

## Self-Review notes
- **Spec coverage:** A (throttles + managed client + report) → A1-A6; B (prompt-gated + real gated finetune + report) → B1-B4; C (safe wins + one god-class split + report) → C1-C5. All spec acceptance criteria mapped.
- **Accuracy guard:** present as an explicit gate in A6, B4, C4.
- **Env-overridable:** A2 uses env defaults for all three limits.
- **AgentNick-only:** B3 Step 1 explicitly forbids Gemma; qwen lineage only.
- **No silent promotion:** B2/B3 promote only on eval_gate pass.
