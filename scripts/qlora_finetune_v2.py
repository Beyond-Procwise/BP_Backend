"""QLoRA fine-tuning v2 for AgentNick — chat format dataset.

Trains LoRA adapters on the extraction pipeline dataset (JSONL chat format),
merges into base model, and exports for Ollama import.

Usage:
    cd /home/muthu/PycharmProjects/BP_Backend
    .venv/bin/python scripts/qlora_finetune_v2.py \
        --dataset data/training/extraction_pipeline_dataset.jsonl \
        --output data/models/agentnick-extraction-v2 \
        --epochs 5 \
        --lr 2e-4
"""
import argparse
import json
import logging
import os
import sys
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model config
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# QLoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training defaults
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 2e-4
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_WARMUP_RATIO = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA finetuning for AgentNick extraction model")
    parser.add_argument("--dataset", type=str,
                        default=str(PROJECT_ROOT / "data" / "training" / "extraction_pipeline_dataset.jsonl"),
                        help="Path to JSONL dataset in chat format")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "models" / "agentnick-extraction-v2"),
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    return parser.parse_args()


def load_dataset(dataset_path: str):
    """Load JSONL chat-format dataset and convert to Qwen chat template."""
    examples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex.get("messages", [])
            if not messages:
                continue

            # Convert to Qwen chat template
            text_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(text_parts)
            examples.append({"text": text})

    return examples


def main():
    args = parse_args()

    logger.info("Starting QLoRA fine-tuning v2 for AgentNick")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("No GPU detected — training will be very slow")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Load dataset
    logger.info("Loading training data...")
    train_data = load_dataset(args.dataset)
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
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    logger.info("Loading model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training config
    os.makedirs(args.output, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )

    # Train
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter
    logger.info(f"Saving LoRA adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    logger.info("QLoRA fine-tuning v2 complete!")
    logger.info(f"Adapter saved to: {args.output}")
    logger.info("Next steps:")
    logger.info("  1. Merge adapter: python scripts/merge_and_export.py")
    logger.info("  2. Convert to GGUF and import to Ollama")


if __name__ == "__main__":
    main()
