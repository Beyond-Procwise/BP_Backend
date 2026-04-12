"""QLoRA fine-tuning script for AgentNick on qwen3:30b MoE.

Trains LoRA adapters on procurement-specific data, merges into base model,
and exports as GGUF for Ollama import.

Usage:
    cd /home/muthu/PycharmProjects/BP_Backend
    .venv/bin/python scripts/qlora_finetune.py
"""
import json
import logging
import os
import sys
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "qlora_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "agentnick-qlora"
MERGED_DIR = PROJECT_ROOT / "data" / "models" / "agentnick-merged"

# Model config — using the HuggingFace model ID for qwen3 MoE
BASE_MODEL = "Qwen/Qwen3-30B-A3B"  # MoE model, ~3B active params

# QLoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.1


def load_dataset():
    """Load training data in Alpaca format."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    # Convert to chat format for Qwen
    formatted = []
    for ex in data:
        text = (
            f"<|im_start|>system\n"
            f"You are AgentNick, an AI procurement agent for ProcWise. "
            f"Extract structured data from procurement documents accurately. "
            f"Return valid JSON. Never guess — extract only what's in the document.<|im_end|>\n"
            f"<|im_start|>user\n{ex['instruction']}\n\n{ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
        formatted.append({"text": text})

    return formatted


def main():
    logger.info("Starting QLoRA fine-tuning for AgentNick")
    logger.info(f"Dataset: {DATASET_PATH}")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Check if base model is available locally via Ollama
    # If not, we'll download from HuggingFace
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Load dataset
    logger.info("Loading training data...")
    train_data = load_dataset()
    dataset = Dataset.from_list(train_data)
    logger.info(f"Training examples: {len(dataset)}")

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    logger.info("Loading model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training config
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        max_seq_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("QLoRA fine-tuning complete!")
    logger.info(f"Adapter saved to: {OUTPUT_DIR}")
    logger.info("Next steps:")
    logger.info("  1. Merge adapter: python scripts/merge_and_export.py")
    logger.info("  2. Convert to GGUF: python scripts/convert_to_gguf.py")
    logger.info("  3. Import to Ollama: ollama create BeyondProcwise/AgentNick -f Modelfile.finetuned")


if __name__ == "__main__":
    main()
