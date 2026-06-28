# AgentNick — End-to-End Analysis & Intelligence/Knowledge Improvement Plan

**Date:** 2026-06-28 · Evidence: live runs against `:extract` (qwen2.5-7B) and
`:unified`/`:latest` (qwen3-30B). Logs under `artifacts/gpu_upgrade/`.

## Executive summary

AgentNick is **already strong** across the board. The headline finding for
"how to improve it" is counter-intuitive but well-evidenced: **the lever is
KNOWLEDGE (client/domain context via RAG) and DATA QUALITY — not retraining the
weights.** A real QLoRA finetune was run end-to-end this session and the gate
**refused** it (regressed to 0.0). The model's raw intelligence is not the
bottleneck; what it lacks is *this company's* specific context.

## Evidence scorecard

| Dimension | Result | Read |
|---|---|---|
| Extraction field-accuracy | **0.865–0.873** doc-accuracy | strong |
| Extraction whole-doc-exact | **0.08–0.17** exact_doc_rate | **weak** — rarely gets *every* field |
| Grounding / no-fabrication | **abstains** on absent fields (po_id/due_date → null) | strong (safe) |
| Procurement knowledge | **13/14** (Incoterms, 2/10-net-30 economics, 3-way match) | strong |
| Agentic (`:unified`) | planning 3/4, negotiation 3/3, summary 3/3, Q&A 3/3 | strong |
| Reasoning engine end-to-end | `planner=llm`, 11–12-step plans | works |
| Latency (warm) | extract **0.5 s**, short-QA **5.4 s**, plan **11.9 s** | good warm; slow cold |
| Consistency (temp 0) | structurally identical (5 vs 5 steps), not byte-identical | minor variance |
| Weight finetune (QLoRA) | candidate **0.0 → gate REFUSED** | not the lever |

**Diagnosis:** high field-accuracy + low whole-doc-exact + correct abstention means
the residual extraction "errors" are mostly the model *correctly declining* to
emit fields that aren't cleanly in the document (e.g. `requested_by`, some dates)
— while the gold has values resolved from elsewhere. So chasing the eval number
risks teaching fabrication.

## Best ways to improve — prioritized (value × feasibility × safety)

### 1. Inject CLIENT-SPECIFIC knowledge via RAG — **highest leverage** 🟢
The model knows *procurement*; it doesn't know *your* suppliers, contracts,
category taxonomy, price history, or past deals. A RAG pipeline + Qdrant already
exist. Feed AgentNick, at inference, the relevant: supplier profiles, contract
terms (Incoterms/payment terms per supplier), category/UOM taxonomies, and
recent-deal context. This makes negotiation, ranking, quote-comparison and
extraction-disambiguation materially smarter **without touching weights**.
*Effort: medium · Risk: low · Gate: agentic-task success before/after.*

### 2. Fix the GOLD/EVAL data quality — **truth + safety** 🟢
The eval is being held back (and could reward fabrication) by gold fields not
present in the source. Curate the gold so it scores only truly-in-document fields
(extend `DOWNSTREAM_FIELDS`), and separate "model should emit" from
"downstream-resolved". Gives a *true* accuracy number and a gate that can't be
gamed. *Effort: low–medium · Risk: none · Prereq for any weight work.*

### 3. Grow the learned-pattern / vendor-profile loop — **continuous knowledge** 🟢
`model_sync_service` already injects `bp_vendor_extraction_profiles` into the
Modelfile every 6 h. Broaden it: per-supplier currency/format/line-item hints,
known doc-id patterns, and the freight/rate-card quirks (e.g. the Condor
multi-section-total case). This is knowledge-as-prompt, eval-gated, zero training.
*Effort: low · Risk: low (eval-gated).*

### 4. Targeted few-shot prompts for the genuinely-hard fields — **precision** 🟡
For fields that ARE in the document but are hard (tax vs subtotal, supplier-vs-
buyer, multi-section quote totals, UK/EU date formats), add 2–3 worked examples to
the relevant Modelfile/context_layer prompt. Eval-gate each (≥ baseline).
*Effort: low · Risk: low (gated).*

### 5. Tighten output control on the 30B "thinking" model — **usability/latency** 🟡
`:unified` is verbose (chain-of-thought). For structured tasks use `format=json`,
stop tokens, and tight `num_predict`; reserve long reasoning for genuinely open
tasks. Cuts latency and parsing fragility. *Effort: low · Risk: low.*

### 6. Right-model routing hygiene — **already mostly done** 🟡
Keep `:extract` for extraction (fast 0.5 s, strong grounding) and `:unified` for
agentic. The dual-role `:latest` fix this session stopped the fallback from
refusing agentic tasks. Audit remaining call sites to ensure none route extraction
to a 30B agentic prompt or vice-versa. *Effort: low · Risk: low.*

### 7. Weight-level finetune — **only after #1–#4, and only if it passes the gate** 🔴
The current finetune is non-viable (broken GGUF, regresses). To make it work:
(a) clean, *in-document* training data matching the `context_layer` prompt format
(see #2); (b) fix the merge→GGUF conversion (the candidate's llama-runner crashed);
(c) promote only on `eval_gate ≥ baseline`. High effort, uncertain payoff —
pursue only if prompt+RAG+patterns plateau. *Effort: high · Risk: high (gated).*

## Recommended sequence
1. **#2 gold/eval cleanup** (unblocks honest measurement) →
2. **#1 RAG client-knowledge** + **#3 pattern loop** (the real intelligence/knowledge gains) →
3. **#4/#5/#6** prompt + routing polish (gated) →
4. Re-run this analysis + the eval gate to quantify the lift →
5. Only then reconsider **#7** weights.

Every model-facing change is gateable the same way already proven this session:
`run_eval_gate.py` for extraction (≥0.845) and the capability/improvement probes
(`agentnick_analysis.py`, `agentnick_improvement_probe.py`) for agentic tasks —
so nothing ships that regresses.
