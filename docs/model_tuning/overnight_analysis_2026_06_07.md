# Overnight AgentNick Auto-Tuning Analysis — 2026-06-07 → 06-08

**Window:** 2026-06-07 23:23 IST → 2026-06-08 09:00 IST (~9.5h). Living report; updated each work cycle.

**Mandate:** find efficient methods to auto-tune AgentNick. If a real finetune isn't safely possible in the window, do back-to-back analysis and ensure system stability + process-flow accuracy.

**Hard safety rule:** NEVER replace/regress the production `BeyondProcwise/AgentNick:extract` model. Current baseline is ~100% on the `_stg` set. Any tuning goes to a scratch adapter + candidate model and is evaluated against baseline; promotion is a human decision, not automated tonight.

---

## State of play (cycle 0 — orientation)

- **Time/GPU:** A10G 23 GiB, ~8.5 GiB free (procwise holds AgentNick:latest, 14.5 GiB). CUDA OK.
- **Training stack IS installed:** torch 2.5.1+cu121, transformers, peft, trl, bitsandbytes, datasets, accelerate. Only `unsloth` missing (handoff says skip it). → A real QLoRA is technically runnable.
- **`_train_model` is a STUB** (`src/training/pipeline.py:293`) — never trained. `:finetuned` is byte-identical to `:extract` (no tune ever happened).
- **Corpus:** `src/data/training/auto_collected_examples.jsonl` = 197 records, ~69 with usable `source_text` (up from 175/47 at handoff). `data/training/overnight_finetune.jsonl` = 162 (combined, mostly synthetic). The ~69 auto-collected are the only production-grounded gold examples. Still below the ~100–200 lower bound for stable 7B QLoRA.
- **Example format:** `{doc_type, pk, source_text, extracted:{header,lines}, confidence}`. Training/eval pair = `source_text → JSON(header+lines)`. These are confidence-gated production extractions = gold standard.
- **No eval gate** before promotion (handoff's highest-stakes gap).
- **Daily finetune cron:** DISABLED 2026-05-26 (correct).

## Efficient auto-tuning methods (analysis)

Ranked by efficiency × safety for THIS system (regex-primary extraction, ~100% baseline, thin corpus, GPU shared with prod):

1. **Eval-gated QLoRA (the "real" finetune, made safe).** Build dataset from auto_collected → QLoRA r=16 on a scratch adapter → **evaluate vs baseline on held-out gold** → promote only on non-regression. The eval gate is the missing prerequisite. *Best long-term; needs the gate + a real trainer.*
2. **Prompt / few-shot auto-tuning (cheapest, zero model risk).** Auto-mine the gold examples + recurring discrepancy patterns into improved few-shot exemplars / the context_layer prompt. No weights touched → cannot regress. Fast. *Best ROI given a ~100% baseline already.*
3. **Pattern-store auto-tuning.** Extraction is regex-primary; auto-tune the regex/pattern store from gold + discrepancies. Improves accuracy without the LLM. *High efficiency, deterministic.*
4. **Feedback-driven correction loop.** Use agent_actions discrepancies + linking/promotion verdicts as a signal to target tuning where the pipeline actually errs.

**Recommendation:** the corpus is too thin and the baseline too high for a 7B QLoRA to safely "move the needle" tonight — and without the eval gate it's reckless. So tonight: (a) BUILD the eval gate (unblocks all future tuning, zero risk), (b) implement a real `_train_model`, (c) run a *gated* QLoRA experiment to a scratch model and REPORT whether it helps — without touching prod, (d) verify system stability + process-flow accuracy continuously.

---

## Work log

### Cycle 1 (23:23 IST) — orientation + eval gate + key findings
- Orientation complete. Built `src/training/eval_gate.py` (model-vs-baseline non-regression harness).
- Stability sweep (background agent) → **system STABLE**: services up, GPU A10G 0 ECC/no throttle, no crashes/exceptions in logs, deal_id consistent, promotion hold-logic correct. WARN items: P1 a `tax_mismatch` discrepancy on `INV-1` raw_id=42 (this is leftover test pollution from my earlier Task-4 persistence test, pre-hermetic-fix — safe to clean); P2 the 6 known held docs (missing parent POs); P3 1 legacy orphan invoice 102938→PO 502110 (PO never loaded); P4 missing `proc.prompt`/`proc.policy` tables (non-fatal startup ERRORs); P5 36 NULL-quote_id seed rows + Q001 dup; P6 missing reference json configs.

#### CRITICAL FINDING — fine-tuning AgentNick tonight is NOT advisable (and would be unsafe)
1. **Two different models.** `:latest` = Qwen3-MoE **30.5B** (general/orchestration, 10GB VRAM now). Extraction uses `:extract` = **7B** (8.1GB). The thing to "tune for accuracy" is `:extract`.
2. **The corpus is mis-targeted for the extraction model.** Gold `extracted.header` mixes (a) model-output fields (ids, amounts, dates) with (b) **downstream-resolved fields the model never produces** — e.g. `supplier_id`=`SUP-CityOfNewport` is resolved by `promotion.py:323` from the NAME the model outputs (context_layer field doc line 134: "Output the company NAME — it will be resolved to an internal supplier ID downstream"). Training the model to emit `SUP-…` would teach it to **hallucinate** resolved IDs it cannot derive from text. My naive-prompt baseline scored 0.31 precisely because of this (supplier_id/buyer_id/date-format mismatches), NOT because the model is weak.
3. **No independent ground-truth eval set.** The "gold" is the pipeline's own confidence-gated output → circular for measuring accuracy. You can measure *consistency/non-regression*, not true accuracy gains, without human labels.
4. **Thin corpus** (~69 usable) for a 7B QLoRA; **GPU contention** (tuning needs procwise+ollama stopped → destabilizes the running system overnight); and the **#1 rule is not regressing the ~100% baseline**.

**Verdict:** per the mandate ("if finetuning is not possible … ensure stability + process-flow accuracy"), I am NOT running a production finetune tonight. Instead: refine the eval gate to use the REAL `context_layer.synthesize` step (faithful), deepen the method analysis, and verify/harden system stability + process-flow accuracy.

### Cycle 2 (00:03 IST) — faithful eval gate + baseline + opportunity sizing
- **Refined `eval_gate.py`**: added `evaluate_via_context_layer()` that runs the REAL `context_layer.synthesize` step with the model under test, and a `DOWNSTREAM_FIELDS` exclusion set (supplier_id, buyer_id, converted_amount_usd, region, status, audit/stamp cols) so the model is judged only on fields it actually produces.
- **TRUE FAITHFUL BASELINE (the non-regression yardstick):** `AgentNick:extract` via context_layer on the 20-doc holdout → **doc_accuracy = 0.847** (222/261 model-output fields), exact_doc_rate 0.20. (The earlier 0.31 was a naive-prompt artifact.) Any tuned candidate must score ≥ 0.847 to be promotable; `eval_gate()` enforces this.
- **Mismatch taxonomy (the prompt/pattern tuning targets, ~39 mismatched fields):**
  - *ID-prefix inconsistency* — `quote_id` `599390` vs `QUT599390` (same class as the PO `PO`-prefix bug; quotes carry a `QUT` prefix). Pure normalization fix.
  - *Date errors* — month/day confusion + off-by-one (`validity_date`, `due_date`, `expected_delivery_date`). Prompt + ISO-normalization fix.
  - *Address role-swap* — supplier_address ↔ buyer_address. Prompt disambiguation fix.
  - *Supplier-name variants* — "City of Newport" vs "City of Newport Council" (cosmetic; resolution handles it).
  - *Numeric total misreads* — quote `total_amount`, plus line-item `line_total`/`sum_mismatch` from the discrepancy table (6× `invoice_total_incl_tax`, multiple `line_items[*].line_total`). Line-item arithmetic is the densest error cluster.
- **Cleaned test pollution:** removed the bogus `INV-1`/raw_id=42 `tax_mismatch` discrepancy (my pre-hermetic Task-4 artifact, source_file `/tmp/x.pdf`). `blocks_promotion` count 1→0 — the false P1 is gone.
- **Stability:** re-verified at 00:03 IST — procwise active, health 200, GPU 12.8 GiB used / 0% util / 30 °C. Stable.

#### Quantified verdict on auto-tuning the model
A 7B QLoRA cannot fix dates/prefixes/address-roles/arithmetic — those are deterministic post-processing/prompt issues, and the gold's resolved fields would teach hallucination. **Estimated recoverable gain from prompt/pattern/normalization ≈ the bulk of the 15% gap (≈0.847→~0.95+), at zero model risk.** A model finetune's expected gain on this corpus is ≤0 (regression risk). Efficient-method ranking is now evidence-backed, not just a priori.

### Cycle 3 (00:54 IST) — measured the normalization ROI (corrects cycle-2 estimate)
Ran `scripts/eval_normalization_experiment.py` (offline; one cached model pass + re-scoring). Result on the 20-doc holdout:

| Level | doc_accuracy | Δ |
|---|---|---|
| L0 raw baseline | 0.842 | — |
| L1 + id-prefix norm (strip QUT/PO/INV) | 0.855 | +0.013 |
| L2 + date ISO norm | 0.855 | **+0.000** |
| L3 + address-swap tolerant | 0.868 | +0.013 |

**Residual mismatch categories after full normalization:** other 11, **date 11, numeric 8**, id_prefix 2, address 1, name 1.

**Correction to cycle-2's estimate:** I wrote that normalization could recover "the bulk of the gap (~0.95+)." **The measurement disproves that** — deterministic normalization recovers only **~+2.6 points (0.842→0.868)**. The +0.000 on date-ISO is decisive: the date mismatches are **value misreads** (e.g. `2025-04-12` vs `2025-01-12`, `2019-04-21` vs `-22`), not format issues. The dominant residual (dates + numerics + "other") is **genuine model read-error**, which normalization cannot touch.

**What this changes:**
- Normalization/prefix/address handling is still worth doing (clean, zero-risk) but is a **small** win, not a silver bullet.
- The real accuracy ceiling is set by the model's **date and numeric reading** — candidates to address it: (a) **prompt A/B** for date/number disambiguation, measured via the eval gate; (b) **try the 30B `:latest` model** for extraction (it may read dates/numbers better than the 7B — a model-SELECTION win needing no training); (c) lean on the existing discrepancy/arithmetic layer for numeric self-correction. NOT fine-tuning the 7B on the current (resolved-value) corpus.
- Net: the honest "efficient auto-tune" is **prompt iteration + possible model-tier swap, both gated by `eval_gate.py`** — measurable, reversible, no training run, no regression risk.

### Cycle 4 (01:03 IST) — 30B model-swap test FAILED + stability incident & recovery
Tested whether the 30B `:latest` reads dates/numbers better than the 7B `:extract` (a no-training "model selection" lever). **Result: not viable.**
- `:latest` returned **500 Server Error on every `/api/generate` call** (24+ failures) → doc_accuracy 0.000 (all failed). The 30B Qwen3-MoE cannot be served for extraction through context_layer on this A10G in the current config (serving error / memory pressure).
- **Stability incident:** the attempt **evicted the warm production model** — `/api/ps` went to `{"models":[]}`, GPU VRAM dropped 12.5 GB → 3.8 GB. An extraction request in that window would have cold-loaded (60s) or failed.
- **Recovery (per handoff):** re-primed `:extract` with `keep_alive:24h` → HTTP 200, reloaded in ~3s, GPU back to 12.5 GB, `/api/ps` shows `:extract` **resident** (vram==size). procwise health 200, deal summary 200. **Stable.**
- **Lessons:** (1) model-tier swap to the 30B is NOT an available lever (it errors). (2) Do NOT run alternate-model evals against the shared live Ollama overnight — they evict the warm production model. Any future model eval must use an isolated Ollama instance or a maintenance window. → **Halting GPU-heavy experiments for the rest of the night; remaining work is lightweight monitoring + the morning report.**

#### Efficient auto-tune methods — revised recommendation
- **Do NOT** QLoRA the 7B `:extract` on the current corpus — mis-targeted, would hurt the 100% baseline.
- **Highest ROI / zero model risk:** prompt & pattern auto-tuning inside the pipeline (regex-primary), driven by `agent_actions` discrepancies + held-doc gap reports. Improves accuracy without touching weights.
- **If/when fine-tuning the model:** first (a) build a **human-labeled held-out eval set** of RAW extraction targets (verbatim doc values, pre-resolution), (b) train ONLY on model-output fields (exclude resolved supplier_id etc.), (c) gate with `eval_gate.py` against that labeled set, refusing promotion on any regression. Prereqs that must exist before any safe auto-tune.

- MONITOR 02:06 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 03:07 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 04:08 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 05:09 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 06:10 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 07:11 IST — STABLE: procwise 200, GPU 12.5GB/0%/30C, :extract resident.
- MONITOR 08:12 IST — STABLE: procwise 200, GPU 12.5GB/0%/29C, :extract resident. Monitoring ended.

---

## MORNING REPORT (08:12 IST, 2026-06-08)

### Bottom line
**A safe production finetune of AgentNick was NOT possible tonight** — and attempting one would have *hurt* accuracy. Per the mandate's fallback, I instead (a) built the missing safety infrastructure, (b) measured the real tuning opportunity, and (c) kept the system stable all night. **System is stable; process flow is accurate.**

### Why no finetune (evidence-based)
1. **Corpus is mis-targeted.** The auto-collected "gold" mixes model-output fields with *downstream-resolved* values (e.g. `supplier_id=SUP-…` is resolved by `promotion.py` from the company NAME the model emits). Training on it would teach the model to **hallucinate** resolved IDs. 
2. **No independent ground-truth eval set** — the gold is the pipeline's own output (circular). Without human labels you can only measure non-regression, not true accuracy gain.
3. **Thin corpus** (~69 usable) and **GPU contention** (tuning needs procwise+Ollama stopped → destabilizes prod). The 30B `:latest` isn't even servable for extraction (500 errors; a test attempt evicted the warm model — recovered).
4. **#1 rule:** never regress the baseline.

### Measured facts (the new yardsticks)
- **Faithful non-regression baseline:** `AgentNick:extract` via the real `context_layer.synthesize` = **0.847 doc-accuracy** on a 20-doc holdout (model-output fields only). Any candidate must beat this.
- **Deterministic normalization ROI = only +2.6 pts** (0.842→0.868). Date-ISO normalization added **+0.000**, proving the date errors are genuine **value misreads**, not formatting. Dominant residual = **date + numeric misreads** (model-read quality), not field identification.

### Recommended efficient auto-tuning path (ranked, evidence-backed)
1. **Prompt iteration gated by `eval_gate.py`** — refine the context_layer prompt for date/number disambiguation; A/B each change against the 0.847 baseline; ship only on a measured non-regression. Zero training, reversible, no GPU contention. *Highest ROI.*
2. **Deterministic normalization** of id-prefixes (PO/QUT/INV) + address-role handling — small clean win (+2.6pts), zero model risk.
3. **Pattern-store / regex tuning** + the existing arithmetic discrepancy layer for the numeric-misread cluster.
4. **Only later, a real eval-gated QLoRA** — and only after the two prerequisites exist: (a) a **human-labeled raw-extraction eval set** (verbatim doc values, pre-resolution), and (b) training restricted to **model-output fields** (exclude resolved supplier_id etc.). The training stack is installed; `_train_model` still needs writing.

### Delivered tonight (all committed; nothing in production touched)
- `src/training/eval_gate.py` — faithful non-regression harness (commits 0ef066a, 106ee2f).
- `scripts/eval_normalization_experiment.py` — the ROI measurement (commit 2b05de5).
- Cleaned the bogus `INV-1`/raw_id=42 blocking discrepancy (my pre-hermetic test artifact) → `blocks_promotion` 1→0.
- Stability verified continuously 23:23→08:12 IST; one self-inflicted model eviction caught and recovered.

### Open items for the team (not blockers)
- Build the human-labeled raw-extraction eval set (the real prerequisite for model tuning).
- Implement `_train_model` (PEFT+TRL; deps ready) when the eval set exists.
- Housekeeping flagged by the stability sweep: missing `proc.prompt`/`proc.policy` tables (non-fatal startup ERRORs), 36 NULL-quote_id seed rows + Q001 dup in quote_trgt, 1 legacy orphan invoice (102938→PO 502110), missing reference json configs.
