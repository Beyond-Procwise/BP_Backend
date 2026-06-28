# Workstream A — Extraction Speed: Before/After Report (CORRECTED)

**Date:** 2026-06-28
**GPU:** NVIDIA RTX PRO 6000 Blackwell, 96 GB (replacing a 23 GB A10G)
**Pipeline:** the **live** renovation pipeline (`services/extraction/dispatch.dispatch_document`).

## ⚠️ Correction notice

An earlier version of this report claimed a **3× speed-up** from raising the Ollama
concurrency throttle (`OLLAMA_NUM_PARALLEL` 2→8). **That claim was wrong.** It was
based on a **single anomalous measurement** (one isolated NP=8 run at 14.2 s).
On repeated, controlled re-measurement — including a live production verification —
**that number did not reproduce**. The honest finding is below.

## What the data actually shows

Same 9 live documents, read-only (no DB writes), renovation pipeline, varying the
Ollama daemon's `NUM_PARALLEL` and the worker count:

| Config | batch (9 docs) | runs |
|---|---|---|
| NP=2, 4 workers | 33.2 s, 42.4 s | 2 |
| NP=4, 4 workers | 44.0 s | 1 |
| NP=8, 8 workers | 41.9, 42.8, 45.2, 46.0, 46.9, 48.6 s … **+ one 14.2 s outlier** | 7 |

- **NP=2 median ≈ 38 s; NP=8 median ≈ 45 s.** Within run-to-run noise (±25 %),
  raising concurrency gives **no reliable speed-up** — and is, if anything,
  marginally *worse*.
- The lone **14.2 s** NP=8 run could **not be reproduced** in 7 subsequent NP=8
  runs (fresh isolated instance and the live production daemon both ~42–48 s).

## Why: extraction is GPU-compute-bound

Per-document latency is dominated by the AgentNick LLM calls in `context_layer`.
A **single GPU** cannot run 8 concurrent 7B-model sequences meaningfully faster
than 2 — the compute is the bottleneck, not the concurrency cap. More parallel
requests just time-slice the same compute (and add scheduling overhead / queue
stragglers). The 96 GB card's benefit is **VRAM and faster per-call inference
(hardware)**, not extra parallel throughput from software.

## Production verification (what you asked for)

Activated `OLLAMA_NUM_PARALLEL=8` on the production daemon and benchmarked the
real `:11434` daemon: **45–49 s** across 3 runs (procwise running *and* stopped) —
**no improvement** over NP=2. The change was therefore **reverted**: the daemon is
back at `NUM_PARALLEL=2` (original), and the code defaults are restored
(`OLLAMA_MAX_CONCURRENT=2`, `PROCWISE_DOC_WORKERS=4`).

## What from Workstream A was kept (genuine, not speed-by-concurrency)

| Change | Status | Rationale |
|---|---|---|
| Route NuExtract + LLM-fill through the managed `ollama_client` (+ parallel chunks) | **Kept** | Robustness/observability: one concurrency budget + retry/backoff instead of raw unthrottled POSTs. (Legacy fallback path; the live renovation path already used the managed client.) |
| `Modelfile` `:latest` `num_gpu 25 → -1` | **Kept** | Full-GPU offload for the 30B agentic/summary model now that VRAM is ample. Sound, but its speed impact was **not** separately measured. |
| `OLLAMA_MAX_CONCURRENT` 2→8, `PROCWISE_DOC_WORKERS` 4→8, daemon `NUM_PARALLEL`=8 | **Reverted** | No measurable benefit; GPU-compute-bound. |

## The real speed levers (for a future, honest effort)

1. **Fix the PyTorch/Blackwell incompatibility** (see ROLLUP) so the torch-based L2
   extractors use the new GPU instead of falling back to CPU.
2. **Reduce per-doc LLM work** — `context_layer` makes ~2 sequential AgentNick
   calls per doc; a faster/smaller model or shorter prompts would help (accuracy
   trade-off, must be eval-gated).
3. Concurrency tuning is **not** a lever for this workload.

## Honesty note

Two process failures produced the wrong initial claim: (1) I trusted a single
measurement instead of requiring reproduction, and (2) early runs were confounded
by my own co-resident Ollama instances. Both are corrected here; the bench harness
itself (`scripts/gpu_upgrade/bench_renovation.py`) is sound and read-only.
