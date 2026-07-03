"""Merge the QLoRA adapter into the base model and save a full HF model dir
that Ollama can import directly (modern Ollama converts safetensors -> GGUF).

Writes data/models/agentnick-merged. Run AFTER scripts/qlora_finetune.py.
"""
from __future__ import annotations
import logging, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path("/home/muthu/PycharmProjects/BP_Backend")
BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = ROOT / "data/models/agentnick-qlora"
MERGED = ROOT / "data/models/agentnick-merged"


def main():
    log.info("Loading base %s (fp16)...", BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    log.info("Loading adapter %s ...", ADAPTER)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    log.info("Merging adapter into base...")
    model = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model -> %s", MERGED)
    model.save_pretrained(str(MERGED), safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(str(MERGED))
    log.info("MERGE_DONE -> %s", MERGED)


if __name__ == "__main__":
    main()
