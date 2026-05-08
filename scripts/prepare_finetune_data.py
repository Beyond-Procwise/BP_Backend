"""Prepare fine-tuning dataset from auto-collected verified extractions.

Reads auto_collected_examples.jsonl (collected by ProcessMonitorWatcher
from high-confidence, zero-error extractions) and converts them into
the instruction-tuning format used by the QLoRA training pipeline.

Usage:
    python scripts/prepare_finetune_data.py [--min-confidence 0.90]

Output:
    data/training/auto_finetune_dataset.json
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTED_PATH = PROJECT_ROOT / "data" / "training" / "auto_collected_examples.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "auto_finetune_dataset.json"
MIN_CONFIDENCE = 0.90


def main():
    min_conf = float(sys.argv[1]) if len(sys.argv) > 1 else MIN_CONFIDENCE

    if not COLLECTED_PATH.exists():
        print(f"No collected examples at {COLLECTED_PATH}")
        return

    examples = []
    with open(COLLECTED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            if ex.get("confidence", 0) < min_conf:
                continue

            source_text = ex.get("source_text", "")
            extracted = ex.get("extracted", {})
            doc_type = ex.get("doc_type", "")

            if not source_text or not extracted.get("header"):
                continue

            # Build instruction-tuning example
            instruction = f"Extract all fields from this {doc_type} document. Return valid JSON."
            output_data = {
                "document_type": doc_type,
                "header": extracted.get("header", {}),
                "line_items": extracted.get("line_items", []),
            }

            examples.append({
                "instruction": instruction,
                "input": source_text[:6000],
                "output": json.dumps(output_data, default=str),
            })

    if not examples:
        print("No qualifying examples found")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(examples, f, indent=2, default=str)

    print(f"Prepared {len(examples)} fine-tuning examples → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
