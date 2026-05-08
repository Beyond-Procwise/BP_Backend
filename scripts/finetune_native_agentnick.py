#!/usr/bin/env python3
"""Fine-tune Gemma 4 26B-A4B as native AgentNick base model.

This creates a NATIVE base model — not a wrapper with a system prompt.
The model IS AgentNick with procurement expertise baked into weights.

Uses QLoRA (4-bit) on full GPU for overnight training.
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ["PYTHONPATH"] = "src:."
sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/finetune_native_agentnick.log"),
    ],
)
logger = logging.getLogger("finetune")

DATASET = "data/training/native_agentnick.jsonl"
OUTPUT_DIR = "data/models/native_agentnick"
BASE_MODEL = "google/gemma-2-27b-it"  # HuggingFace name for Gemma 26B


def main():
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    logger.info("=== Native AgentNick Fine-tuning on Gemma 4 26B ===")
    logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0)

    # Load dataset
    logger.info("Loading dataset: %s", DATASET)
    examples = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    logger.info("Loaded %d training examples", len(examples))

    # Format for Gemma chat template
    def format_gemma(example):
        msgs = example.get("messages", [])
        text = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_gemma)
    logger.info("Dataset formatted: %d examples", len(dataset))

    # QLoRA 4-bit config — use full GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    logger.info("Loading base model: %s (4-bit QLoRA, full GPU)", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Gemma compatibility
    )
    logger.info("Model loaded successfully")

    # LoRA configuration — target all attention + MLP layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Higher rank for deeper learning
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments — aggressive overnight training
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,  # More epochs for deeper learning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=20,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
    )

    # Train
    logger.info("Starting training: %d epochs, lr=%s, batch=%d, accum=%d",
                training_args.num_train_epochs, training_args.learning_rate,
                training_args.per_device_train_batch_size,
                training_args.gradient_accumulation_steps)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer.train()
    logger.info("Training complete!")

    # Save adapter + tokenizer
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Adapter saved to %s", final_dir)

    # Merge adapter into base model
    logger.info("Merging LoRA adapter into base model...")
    from peft import PeftModel

    # Reload base model in full precision for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, final_dir)
    merged_model = merged_model.merge_and_unload()

    merged_dir = os.path.join(OUTPUT_DIR, "merged")
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved to %s", merged_dir)

    logger.info("=== Native AgentNick Training Complete ===")
    logger.info("Next steps:")
    logger.info("  1. Convert to GGUF: python -m llama_cpp.convert %s", merged_dir)
    logger.info("  2. Quantize: llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M")
    logger.info("  3. Create Ollama: ollama create BeyondProcwise/AgentNick:v3-native -f Modelfile")


if __name__ == "__main__":
    main()
