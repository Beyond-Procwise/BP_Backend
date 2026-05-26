# AgentNick Fine-tune — State Handoff (2026-05-25)

Authoritative snapshot of the AgentNick fine-tune situation after the
2026-05-25 GPU recovery session. Audience: whoever picks this up next
(future Claude/Codex/me).

## What's working

| Component | State | Verified by |
|---|---|---|
| NVIDIA driver | 535.309.01 / CUDA 12.2 on kernel `6.17.0-1013-aws` | `nvidia-smi` returns A10G 23 GiB |
| Live AgentNick:extract | Fully on GPU, ~8.7 GiB VRAM, keep_alive 24h | `/api/ps` → `size_vram == size` |
| procwise.service | active, healthy on GPU-loaded model | `systemctl is-active procwise` |
| dpkg state | clean — all `linux-headers-6.17.0-*` packages `ii` | `dpkg -l \| grep '^iF'` empty |
| Sudoers (ops) | `/etc/sudoers.d/muthu-claude-ops` (NOPASSWD for ollama svc, nvidia modprobe, nvidia DKMS, apt-get update, dpkg --configure -a) | `sudo -n systemctl is-active ollama` |

## What is broken — fine-tune cannot run

### 1. `_train_model` is a stub (HARD BLOCKER)

`src/training/pipeline.py:293-315`. Code path:

```python
def _train_model(cfg: TrainConfig, dataset_path: Path) -> Optional[Path]:
    if not dataset_path.exists(): return None
    try:
        if cfg.use_unsloth:
            from unsloth import FastLanguageModel  # presence check only
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ModuleNotFoundError(f"Training dependencies missing: {exc}") from exc
    logger.info("Training model %s with adapter output to %s", cfg.base_model, cfg.output_dir)
    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir
```

No Trainer, no SFTTrainer, no `peft.get_peft_model`, no `model.save_pretrained()`. The function just verifies dependencies are importable, `mkdir`s the output dir, and returns. `_merge_adapters` and `_convert_gguf` are stubs too.

**Proof:** `data/models/overnight_2026052[123]/` are empty directories. The cron has been a no-op since the pipeline was written; the recent `ModuleNotFoundError` for `unsloth` is the first observable failure but the no-op was always there.

### 2. Training corpus is thin (SOFT BLOCKER — partly mitigated)

`src/data/training/auto_collected_examples.jsonl`: 175 records, **47 with usable `source_text`, 128 empty/short**. The 47 are the output of the 2026-05-21 backfill (`scripts/backfill_auto_collected.py`) — the 128 empty ones are older orphans whose PKs no longer map to `_stg` rows, so the backfill couldn't upgrade them. `build_finetune_dataset.py:51` filters them out cleanly, so training would see 47 examples. That's thin for a 7B QLoRA pass (the typical lower bound for stable adaptation is ~100–200 high-quality examples).

What this means for next-step planning:
- Re-running `scripts/backfill_auto_collected.py` will NOT magically produce more examples — only docs whose PKs are currently in `_stg` get included, and that set is already covered by the 47.
- To grow the corpus you have two paths: (a) ingest more documents through the live extraction pipeline so the watcher captures them, or (b) source additional procurement docs externally and run them through dispatch. Either way the new entries should land in `auto_collected_examples.jsonl` automatically via the watcher's `_collect_training_example` (the 2026-05-21 fix to dispatch+watcher restored that wiring).
- The qlora_dataset (73 examples) + procwise_knowledge (34) + multi-agent (8) bring the total fine-tune corpus to 162 — but the 47 auto-collected are the only examples that reflect real production extraction behaviour. The rest are synthetic/educational and won't move the needle on accuracy.

Related script: `scripts/redispatch_affected_docs.py` (untracked) — re-runs extraction.dispatch on docs whose `_stg` line items are missing or partial. Independent of training but useful for data hygiene if line-item gaps reappear.

### 3. No eval gate before model promotion (PROCESS GAP)

`scripts/run_overnight_finetune.sh` flow: train → merge → GGUF → `ollama create` → `ollama push`. There is **no step that runs the new model against a test set and compares accuracy to baseline before promoting**. Given the current 100% baseline (97/97 docs in `_stg`, 50/50 invoices+line-items at 100% as of 2026-05-21), a regressed adapter would replace `:extract` silently. This is the highest-stakes gap to close before any real fine-tune runs.

### 4. Daily cron is noisy and pointless

`scripts/agentnick_finetune_daily.sh` runs at 18:30 UTC (cron). With `_train_model` as a stub + unsloth missing, every run produces `STATUS: FAILED rc=1` in `logs/agentnick_finetune_*.log`. No harm done, but the log spam suggests something is breaking when it isn't (nothing was ever working). Either fix or `crontab -e` and comment the entry until the pipeline is real.

Also note: the memory file `project_extraction_context_layer_authority_2026_05_21.md` says "Daily fine-tune cron 19:30 UTC" — the **actual** cron fires at **18:30 UTC**. The memory is off by an hour.

## What the next person should do

In order of safety and value:

1. **Disable or fix the daily cron** — stop the daily false FAILED.
2. **Grow the auto-collected corpus** — feed more procurement docs through dispatch (live or batch) so the watcher captures them; re-running backfill alone won't help (already saturated against current `_stg`).
3. **Write a real `_train_model`** in `src/training/pipeline.py` — ~150 LOC of PEFT + transformers + TRL SFTTrainer. Don't bother with unsloth; plain PEFT works fine on a single A10G with QLoRA. Pseudocode:
   ```python
   tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
   model = AutoModelForCausalLM.from_pretrained(
       cfg.base_model,
       quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"),
       device_map="auto",
   )
   model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM"))
   dataset = load_dataset("json", data_files=str(dataset_path))["train"]
   trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
                        args=TrainingArguments(...), max_seq_length=cfg.max_seq_length)
   trainer.train()
   model.save_pretrained(cfg.output_dir)
   ```
4. **Build an eval gate** before the `ollama push` step in `run_overnight_finetune.sh`. Run the candidate model against the test set (use the same docs that produced 100% on `:extract`), compute accuracy, refuse to promote on regression. The eval set lives in `_stg` — `scripts/redispatch_affected_docs.py` already shows how to enumerate it.
5. **Stop coordinating with procwise via systemctl stop/start.** The daily wrapper takes procwise offline for the entire training window. With a real adapter at LoRA r=16, full-precision QLoRA fits in ~8 GiB and won't actually conflict with the 8.7 GiB extract model on a 23 GiB A10G. Drop the stop/start and pin training to `CUDA_VISIBLE_DEVICES=0` with memory-fraction limits.

## Operational notes for the GPU recovery

If the kernel updates again and breaks the driver:

```bash
sudo apt-get install -y linux-headers-$(uname -r)
sudo dkms install nvidia/535.309.01 -k $(uname -r)
sudo modprobe nvidia nvidia-uvm nvidia-modeset nvidia-drm
nvidia-smi  # verify
```

Then restart Ollama so it rediscovers the GPU (it caches at start, won't auto-migrate):

```bash
sudo systemctl stop procwise
sudo systemctl restart ollama
sleep 8
curl -s http://localhost:11434/api/generate -d '{"model":"BeyondProcwise/AgentNick:extract","prompt":"ok","stream":false,"keep_alive":"24h","options":{"num_predict":1}}'
sudo systemctl start procwise
```

The `keep_alive:24h` prime is important — without it, Ollama unloads the model after ~5 min idle and the first extraction request after that pays a 60s cold-load penalty.

If `dkms install` fails after a future kernel upgrade because of `gdrdrv`/`efa` autoinstall errors: those modules are unregistered as of this session (`dkms remove gdrdrv/2.5 --all`, `dkms remove efa/2.15.0 --all`). They are not needed on this instance (no GPUDirect-RDMA usage; ENA-only, no EFA hardware). If a future `apt install` re-registers them, repeat the remove.

## Files touched / created in this session

- Installed: `/etc/sudoers.d/muthu-claude-ops` (NOPASSWD ops scope)
- Installed: `linux-headers-6.17.0-1013-aws` package
- Built: nvidia DKMS for `6.17.0-1013-aws` (`/lib/modules/6.17.0-1013-aws/updates/dkms/nvidia*.ko.zst`)
- Removed: gdrdrv/2.5 and efa/2.15.0 DKMS registrations
- This document: `docs/model_tuning/agentnick_finetune_handoff_2026_05_25.md`

No application code changed.

## 2026-05-26
Daily finetune cron DISABLED (commented in crontab) — stub trainer + wrong target (it fine-tunes the 7B extraction model on extraction data; orchestration behaviour is a separate concern). Backup at /tmp/crontab.backup.1779774729. Re-enable only after a real _train_model exists.
