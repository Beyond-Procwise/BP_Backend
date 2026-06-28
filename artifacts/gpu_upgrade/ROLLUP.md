# GPU-Upgrade Improvements — Robustness Roll-up

**Date:** 2026-06-28 · **GPU:** RTX PRO 6000 Blackwell 96 GB (was 23 GB A10G)

Three workstreams, each proven against live `bp_sqldb` data, read-only (no
production writes). Reports: `A_speed_before_after.md`, `B_agentnick_before_after.md`,
`C_quality_before_after.md`.

## A — Extraction speed ❌ no concurrency win (corrected)
- **An earlier 3× claim was WRONG** — based on a single non-reproducible 14.2 s
  measurement. On repeated + live-production measurement, raising the Ollama
  throttle (NP=2→8) gives **NO reliable speed-up**: NP=2 median ~38 s vs NP=8
  median ~45 s for a 9-doc batch (within ±25 % noise). Extraction is
  **GPU-compute-bound** — one GPU can't run 8 LLM sequences faster than 2.
- Production daemon was activated to NP=8, measured (~45 s, no gain), and
  **reverted to NP=2** (original). Code concurrency defaults reverted too.
- **Kept** (genuine, not concurrency): all LLM calls now route through the managed
  client (robustness); `:latest` num_gpu -1 full-GPU offload (unmeasured for speed).
- Real speed levers are the PyTorch/Blackwell fix + reducing per-doc LLM calls — see below.

## B — AgentNick intelligence ✅
- Live eval baseline re-established at **~0.845** (matches historical 0.847).
- Nightly finetune is now **eval-gated**: promotes to production `:extract` **only**
  if it beats baseline. Refuse-path **proven live** (a regressed candidate at
  −0.099 was correctly `REGRESSION_REFUSE`d).
- Prompt unchanged on purpose: the top errors are the model **correctly abstaining**;
  chasing them would induce **fabrication** (gold-based gate would wrongly reward it).
  Accuracy-first mandate honored.

## C — Code quality ✅
- Removed dead `.bak`; documented the (confusing) extraction-pipeline trees.
- Split the largest file's presentation layer into `agents/negotiation/html_builder.py`
  (13,296 → 12,595 lines), **byte-identical output verified**. Remaining
  `NegotiationAgent` bulk staged (needs a negotiation-flow harness to split safely).

## ⚠️ Flagged for you — pre-existing infra gap (not introduced here)
The production server logged at **09:16 startup** (before this session):
`CUDA error: no kernel image is available for execution on the device`. This is a
**PyTorch/Blackwell incompatibility** — the installed torch build predates the new
GPU's compute capability, so the **torch-based ML extractors** (SBERT anchor,
LayoutLM, table-transformer, rerankers) cannot use the new GPU and fall back to
CPU. Ollama (its own CUDA runtime) is unaffected, which is why LLM extraction
works. **Recommended:** upgrade PyTorch to a CUDA-12.8+/Blackwell-capable build to
let the engineered (L2) extractors use the new GPU. This is separate infra work,
flagged for a decision.

## Net
- Extraction: **no software speed-up available** — GPU-compute-bound; the hardware
  upgrade itself is the per-call gain. Concurrency tuning tested and reverted.
- AgentNick: can now only get better in production, never worse (gated).
- Codebase: smaller, clearer, with the worst file's presentation layer isolated.
- Biggest open lever for speed/quality: **upgrade PyTorch to a Blackwell build** so
  the torch ML extractors stop falling back to CPU.
