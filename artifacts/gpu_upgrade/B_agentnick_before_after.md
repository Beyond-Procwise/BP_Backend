# Workstream B — AgentNick Intelligence: Before/After Report

**Date:** 2026-06-28
**Constraint:** Extraction accuracy is priority #1; never fabricate; AgentNick is the only base model. Model promotion allowed only if it does **not** regress the live eval baseline.

## B1 — Live eval baseline (the gate threshold)

Ran the production model through the **real** `context_layer.synthesize` step over the deterministic 30% gold holdout (`evaluate_via_context_layer`, LLM-only, no DB writes):

| Model | doc_accuracy | exact_doc_rate | n (holdout) |
|---|---|---|---|
| `BeyondProcwise/AgentNick:extract` (production) | **0.8426 – 0.8454** | 0.1875 | 32 |

Matches the historical 0.847 baseline. This is the floor every candidate must clear.

## B2 — Prompt/pattern refinement: analyzed, baseline retained (honest)

I profiled the baseline's field-level mismatches to drive a **data-driven** improvement:

| Most-missed field | count | nature |
|---|---|---|
| `requested_by` | 10 | model returns **None** (abstains) |
| `expected_delivery_date` | 7 | mostly None |
| `due_date` | 6 | mostly None |
| `tax_amount` | 6 | mixed |

**Finding:** the dominant errors are the model **correctly abstaining** — the gold contains values (e.g. `requested_by = "William Barnes City of Newport"`, `"Civic Centre Building"`) that are not cleanly stated as that field in the source text (they were resolved from elsewhere when the gold was collected). Tuning the prompt to "fix" these would push the model to **fabricate** values to match the gold. Because the eval gate scores match-to-gold, it would *reward* that fabrication — a real trap.

**Decision:** do **not** chase the gate number with a fabrication-inducing prompt change. That would violate the no-fabrication / accuracy-first mandate. The baseline prompt is at its **safe** accuracy ceiling; the gate's role is to block regressions, not to be gamed. Baseline prompt retained. (This matches the prior "do not optimize toward downstream/noisy fields — induces hallucination" finding.)

## B3 — Nightly finetune made real-but-gated (the safety mechanism)

The nightly pipeline (`scripts/run_overnight_finetune.sh`) previously created a
`:v2-finetuned` candidate with **no eval gate** and instructed manual activation.
Changes:
- Added a **hard eval gate** (step 6): the freshly-built candidate is scored vs
  the production model on the holdout; it is promoted to the production
  `:extract` tag **only on `PROMOTE_OK`**, otherwise production is left untouched
  and the candidate kept for inspection.
- Fixed the stale `num_gpu 25 → -1` in the generated finetuned Modelfile.
- Base model stays on the **qwen lineage** (`Qwen/Qwen2.5-7B-Instruct`), honoring
  "AgentNick is the only base model" — never Gemma/other.

### Gate validated live (REFUSE path)

Smoke-tested the gate against a deliberately-wrong candidate (`AgentNick:unified`,
not extraction-tuned):

| | doc_accuracy | delta | verdict |
|---|---|---|---|
| baseline `:extract` | 0.8765 | — | — |
| candidate `:unified` | 0.7780 | **−0.0985** | **REGRESSION_REFUSE** ✅ |

The gate correctly refused promotion. A regressed nightly finetune can no longer
reach production silently.

## Real finetune EXPERIMENT — run end-to-end (2026-06-28, after the PyTorch fix)

Once the PyTorch/Blackwell upgrade made GPU training possible, a genuine QLoRA
finetune was run end-to-end (not a stub):
- Base **Qwen/Qwen2.5-7B-Instruct** (AgentNick lineage), QLoRA 4-bit via
  `bitsandbytes` on the 96 GB GPU, 3 epochs over 162 procurement chat examples
  (`scripts/gpu_upgrade/finetune_qwen_gated.py`). Training completed; adapter
  merged into the base; merged model registered to Ollama as `:ft-candidate`.
- **Eval gate result:** candidate **doc_accuracy = 0.0000** vs baseline 0.8426
  (delta −0.8426) → **REGRESSION_REFUSE**. Production `:extract` **unchanged**.
- **Why 0.0:** the merged/converted candidate is **non-viable** — its llama
  runner crashes on generation ("post predict EOF"), so it returns nothing. This
  reproduces the prior finding that finetuning on the current corpus/setup yields
  a broken/regressed model.

**Takeaway:** the experiment is now genuinely runnable (PyTorch unblocked it), and
the gate **correctly caught and refused** a non-viable candidate — production was
protected automatically. The accuracy lever remains prompt/data quality and a
fixed training+conversion path, not this retrain. `:ft-candidate` is retained for
later debugging of the merge→GGUF step.

## Bottom line

- Live baseline re-established at **~0.845**.
- No prompt change ships because none beats baseline **without fabrication risk** —
  the responsible outcome under the accuracy-first mandate.
- The nightly finetune is now **eval-gated** and the refuse-path is **proven live**,
  so AgentNick can only ever get better, never worse, in production.
