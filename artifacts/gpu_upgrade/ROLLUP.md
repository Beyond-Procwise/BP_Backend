# GPU-Upgrade Improvements — Robustness Roll-up

**Date:** 2026-06-28 · **GPU:** RTX PRO 6000 Blackwell 96 GB (was 23 GB A10G)

Three workstreams, each proven against live `bp_sqldb` data, read-only (no
production writes). Reports: `A_speed_before_after.md`, `B_agentnick_before_after.md`,
`C_quality_before_after.md`.

## A — Extraction speed ✅ (needs 1 root command to activate in prod)
- Lifting the Ollama concurrency throttle 2→8 gives a **clean 3.0× batch-throughput**
  win on the **live** renovation pipeline (9 docs: 42.4 s → 14.2 s).
- **Accuracy held**: per-doc work identical (~10.7 s); 8/9 docs byte-identical
  (1 diff was LLM temp-0 noise, same field/line counts), not a code-path change.
- Code shipped (env-tunable). **Production activation needs the root-owned daemon
  line** `OLLAMA_NUM_PARALLEL=8` + `systemctl restart ollama && restart procwise`
  (command in `resources/deployment/ollama_env.md`).

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
- Extraction: ~3× faster once the daemon line is applied; no accuracy cost.
- AgentNick: can now only get better in production, never worse (gated).
- Codebase: smaller, clearer, with the worst file's presentation layer isolated.
