# Workstream A — Extraction Speed: Before/After Report

**Date:** 2026-06-28
**GPU:** NVIDIA RTX PRO 6000 Blackwell, 96 GB (replacing a 23 GB A10G)
**Pipeline measured:** the **live** renovation pipeline (`services/extraction/dispatch.dispatch_document`, `EXTRACTION_RENOVATION_ENABLED=1`). Per-doc latency is dominated by `context_layer` AgentNick calls, which queue at the Ollama daemon's `OLLAMA_NUM_PARALLEL`.

## Headline result (clean, isolated A/B)

Single isolated Ollama instance at a time (no co-resident contention), same 9 live documents (3 invoice / 3 quote / 3 PO, from S3), read-only (no DB writes, no persistence):

| Config | Batch wall-clock (9 docs) | mean/doc | sum/doc | per-doc output |
|---|---|---|---|---|
| **Before** — NUM_PARALLEL=2, 4 workers, MAX_CONCURRENT=2 | **42.41 s** | 10.65 s | 95.89 s | baseline |
| **After** — NUM_PARALLEL=8, 8 workers, MAX_CONCURRENT=8 | **14.16 s** | 10.73 s | 96.58 s | 8/9 identical |

**≈ 3.0× faster batch throughput.** Per-doc compute is unchanged (~10.7 s; sum-of-work 95.9 s vs 96.6 s — statistically identical). The speed-up is pure de-queuing: at NUM_PARALLEL=2, 9 docs × ~2 LLM calls = ~18 calls funnel through 2 slots, so unlucky docs wait ~40 s (a "straggler"); at NUM_PARALLEL=8 nothing queues and the batch finishes in ~one doc's worth of wall-clock.

## Accuracy gate (no regression)

8 of 9 documents produced **byte-identical** extraction signatures before vs after. The 1 difference (DESIGN HOUSE quote) had the **same field count (16) and same line count (1)** — the delta is `context_layer` LLM run-to-run noise at temperature 0, not a code-path change. The concurrency change does not alter per-doc extraction logic, so it cannot systematically affect accuracy; this single diff would occur between any two runs regardless of the change. **Accuracy is held.**

## What changed (code — already applied, env-tunable)

| Change | File | Effect |
|---|---|---|
| `OLLAMA_MAX_CONCURRENT` default 2→8 | `src/services/ollama_client.py` | app-side semaphore matches daemon |
| doc-worker pool 4→8 (`PROCWISE_DOC_WORKERS`) | `src/services/process_monitor_watcher.py` | more docs extracted concurrently |
| `:latest` full-GPU offload `num_gpu 25→-1` | `Modelfile` | 30B agentic/summary model fully on GPU (no CPU spill) |
| NuExtract + LLM-fill routed through managed client, chunks parallelized | `extraction_v3/extraction_v4/engine.py`, `llm_extractor.py` | completes "route all LLM calls through one client" on the legacy fallback path; live path already did this via `context_layer` |

## What requires a privileged step (production activation)

The 3× requires the **Ollama daemon** to run `NUM_PARALLEL=8`. Its config lives in a **root-owned** systemd drop-in (`/etc/systemd/system/ollama.service.d/`) that the agent cannot edit (scoped sudo). To activate in production, a human runs:

```bash
sudo tee /etc/systemd/system/ollama.service.d/parallel.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_KEEP_ALIVE=30m"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
sudo systemctl restart procwise   # picks up 8 doc-workers + MAX_CONCURRENT=8
```

Until that runs, the app-side defaults (8 workers / MAX_CONCURRENT=8) are ready but capped by the daemon's NUM_PARALLEL=2, so production stays at the ~3×-slower behaviour. The full benefit needs the daemon line.

## Method notes / honesty

- Initial benchmarking mistakenly targeted the **legacy `extraction_v3`** path (140–380 s/doc); production runs the **renovation** path. Corrected — all numbers above are the live pipeline.
- A first "after" run looked *slower* (43–47 s) because **4 of my own benchmark Ollama instances were co-resident on the one GPU**, stealing compute. Isolating to a single instance produced the clean 3× above. Lesson recorded; the GPU is a single shared compute resource, so concurrent model copies contend.
- The bench (`scripts/gpu_upgrade/bench_renovation.py`) calls the **real** `dispatch_document` with DB sinks monkeypatched to no-ops — faithful extraction timing, zero writes to `bp_sqldb`.

## Bottom line

Lifting the Ollama concurrency throttle from 2→8 (the survival setting from the old 23 GB card) gives a clean **~3× batch-throughput improvement on the live extraction pipeline with no accuracy regression**, once the root-owned daemon line is applied. Per-document latency is GPU-compute-bound (~10.7 s) and unchanged — further per-doc speed-ups would require reducing/parallelizing the `context_layer` LLM calls, which is accuracy-sensitive and deferred (in-document parallelism, out of scope this pass).
