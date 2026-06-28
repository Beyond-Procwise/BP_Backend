#!/usr/bin/env python3
"""Real QLoRA finetune of the AgentNick base (Qwen2.5-7B-Instruct) — now that the
PyTorch/Blackwell upgrade makes GPU training possible.

Trains a LoRA adapter on the procurement chat dataset, merges it, and saves a
merged HF model dir ready for `ollama create`. The eval gate (run separately)
decides whether the candidate may replace production.

Honest expectation: per prior evidence + a prompt-format mismatch with the
context_layer eval, this may well be REFUSED by the gate. Running it is the
experiment; the gate is the safety net.
"""
import json
import logging
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
sys.path.insert(0, "src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ft")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET = "data/training/overnight_finetune.jsonl"
OUTPUT_DIR = "data/models/agentnick_qwen_ft"
MAX_STEPS = int(os.getenv("FT_MAX_STEPS", "0"))  # >0 for a smoke test


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    log.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    assert torch.cuda.is_available(), "CUDA not available — Blackwell torch upgrade required"

    rows = []
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if isinstance(msgs, list) and msgs:
                rows.append({"messages": msgs})
    log.info("Loaded %d conversational examples from %s", len(rows), DATASET)
    dataset = Dataset.from_list(rows)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("Loading base %s in 4-bit on GPU...", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )

    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_length=4096,
        report_to="none",
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=cfg,
                         peft_config=peft_cfg, processing_class=tok)
    log.info("Starting training (epochs=3, max_steps=%s)...", MAX_STEPS or "full")
    trainer.train()
    log.info("Training complete.")

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    log.info("Adapter saved to %s", final_dir)

    if MAX_STEPS > 0:
        log.info("SMOKE run (max_steps=%d) — skipping merge.", MAX_STEPS)
        print("SMOKE_TRAIN_OK")
        return

    # Merge adapter into fp16 base on CPU, save merged HF dir for ollama create.
    log.info("Merging adapter into base (fp16, CPU)...")
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, final_dir).merge_and_unload()
    merged_dir = os.path.join(OUTPUT_DIR, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    log.info("Merged model saved to %s", merged_dir)
    print("FT_MERGE_OK", merged_dir)


if __name__ == "__main__":
    main()
