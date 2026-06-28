# PyTorch / Blackwell GPU Upgrade — Scope

**Date:** 2026-06-28 · **Branch:** Nick · **Status:** Scope for review (not yet executed)

## Problem (plain English)

The new GPU (RTX PRO 6000 **Blackwell**, compute capability **sm_120**) is too new
for the installed PyTorch. `torch 2.5.1+cu121` was compiled with GPU kernels only
up to **sm_90** (the previous-generation Hopper). So every PyTorch GPU operation
throws *"no kernel image is available for execution on the device"* and PyTorch
**silently falls back to CPU**.

**Consequences today:**
- The production server logs a `CUDA error` at startup and runs in a **degraded
  CPU mode** for all torch-based ML: the L2 "engineered" extractors (LayoutLM,
  table-transformer), the SBERT embedding model, and the RAG/supplier rerankers.
- **Real AgentNick finetuning is impossible** — QLoRA needs `bitsandbytes` 4-bit
  on the GPU (`load_in_4bit`, bf16 compute). So Workstream B's `_train_model` can
  never actually train until this is fixed.
- Ollama is unaffected (its own CUDA runtime), which is why LLM extraction works.

## Goal

Install a Blackwell-capable PyTorch (sm_120) so the torch ML stack uses the new
GPU, and the real finetune becomes possible.

## Facts (verified 2026-06-28)

| | Value |
|---|---|
| GPU compute capability | **sm_120** (12.0) |
| NVIDIA driver | 610.43.02 (supports CUDA 12.8+) |
| Installed torch | `2.5.1+cu121` — arch list maxes at **sm_90** |
| torchvision | `0.20.1+cu121` (must move with torch) |
| Python | 3.12.3 |
| Torch-dependent libs | transformers 5.6.0.dev0, accelerate 1.14.0.dev0, peft 0.19.1.dev0, sentence-transformers 3.4.1, bitsandbytes 0.49.2, docling 2.93.0, easyocr 1.7.2, timm 1.0.26, onnxruntime 1.24.4, spacy 3.8.13, numpy 2.3.5 |
| PyTorch cu128 index | reachable (HTTP 200) — upgrade is feasible from this box |

## The change

Move the venv to a **cu128 (CUDA 12.8) PyTorch with sm_120 kernels**:
- `torch` 2.5.1+cu121 → **≥ 2.7 (cu128)** — first stable line with Blackwell sm_120.
  Target the latest stable cu128 build and its **matching** `torchvision`
  (per the pytorch.org version matrix), from `--index-url https://download.pytorch.org/whl/cu128`.
- Reinstall **`bitsandbytes`** against cu128 (needed for QLoRA on Blackwell).
- Leave transformers / accelerate / peft / sentence-transformers as-is unless a
  smoke test shows an incompatibility (they are already newer than torch 2.5 needs).

## Plan — validate in isolation FIRST, then production (with rollback)

**Phase 1 — Isolated validation (zero production risk):**
1. Create a throwaway venv (`/tmp/torch-blackwell-test`), `pip install torch torchvision --index-url .../cu128`.
2. GPU smoke: `torch.cuda.get_arch_list()` includes `sm_120`; a `randn(4096,4096).cuda()` matmul runs with **no** "no kernel image" error and a correct result.
3. Import-and-run smoke for the real stack: `sentence-transformers` encodes a sentence **on GPU**; `transformers`, `docling`, `easyocr` import cleanly.
4. `bitsandbytes` 4-bit load smoke (a tiny `load_in_4bit` model) to confirm QLoRA will work.
   → If any step fails, stop and report; production is untouched.

**Phase 2 — Production rollout (only if Phase 1 passes):**
5. `pip freeze > rollback-requirements.txt` (exact current versions for rollback).
6. `sudo systemctl stop procwise` (NOPASSWD available).
7. In the production venv: `pip install torch==<ver>+cu128 torchvision==<ver>+cu128 --index-url .../cu128` (+ bitsandbytes).
8. Repeat the GPU smoke in the production venv.
9. `sudo systemctl start procwise`; confirm startup log has **no** CUDA error and `configure_gpu()` returns `cuda`.
10. Re-run a few live extractions; confirm the L2 extractors run on GPU and extraction output is unchanged (accuracy gate).

**Rollback (if anything breaks):**
`pip install -r rollback-requirements.txt` (restores torch 2.5.1+cu121 exactly) → restart procwise. Back to today's working-but-degraded state in minutes.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dependency incompatibility (dev-version transformers/accelerate/peft vs torch 2.7+) | Medium | Phase-1 isolated smoke catches it before touching prod |
| `bitsandbytes` cu128 build mismatch | Medium | Validated in Phase 1; only needed for finetune, not serving |
| In-place upgrade corrupts the live venv | Low (mitigated) | Server stopped during install; exact-version rollback file |
| Disk space (cu128 torch + CUDA libs ≈ 3–5 GB) | Low | Check free space in Phase 2 prep |
| numpy 2.x ABI | Low | torch ≥2.7 supports numpy 2.x |

## Honest expected benefit

- **Fixes a real degradation** — the torch ML extractors + embeddings + rerankers
  stop running on CPU. This speeds up the **L2 engineered** extraction slice,
  embeddings/RAG, and supplier matching.
- **Unblocks real finetuning** (Workstream B) — the gated QLoRA can actually run.
- **Per-document extraction wall-time:** likely a **modest** improvement, not
  dramatic — the dominant cost per doc is the `context_layer` **LLM** calls, which
  already run on the GPU via Ollama. The torch L2 work is a smaller slice. Batch /
  embedding-heavy and RAG paths benefit more. (Setting expectations honestly after
  the concurrency lesson.)

## Recommendation

Proceed with **Phase 1 (isolated validation) now** — it carries zero production
risk and tells us definitively whether the upgrade is clean. Decide on Phase 2
(production rollout) based on Phase-1 results. I can run Phase 1 on request.
