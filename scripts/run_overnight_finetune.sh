#!/bin/bash
# Overnight QLoRA Fine-tuning for AgentNick
# Run: nohup bash scripts/run_overnight_finetune.sh > logs/finetune_$(date +%Y%m%d).log 2>&1 &

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="src:."

echo "=== AgentNick Overnight Fine-tuning ==="
echo "Started: $(date)"
echo ""

# Step 1: Build dataset
echo "[1/5] Building training dataset..."
.venv/bin/python scripts/build_finetune_dataset.py
echo ""

# Step 2: Run QLoRA fine-tuning
echo "[2/5] Starting QLoRA fine-tuning..."
DATASET="data/training/overnight_finetune.jsonl"
OUTPUT_DIR="data/models/overnight_$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

.venv/bin/python -c "
from training.pipeline import TrainConfig, _train_model
from pathlib import Path

config = TrainConfig(
    base_model='Qwen/Qwen2.5-7B-Instruct',
    train_file=Path('$DATASET'),
    output_dir=Path('$OUTPUT_DIR'),
    use_unsloth=True,
    epochs=3,
    learning_rate=2e-4,
    batch_size=2,
    gradient_accumulation_steps=8,
    lora_r=16,
    lora_alpha=32,
    max_seq_length=4096,
    system_prompt='You are AgentNick, the core AI engine for ProcWise.',
)

result = _train_model(config, Path('$DATASET'))
print(f'Training complete: adapter saved to {result}')
"

echo ""
echo "[3/5] Merging adapters..."
.venv/bin/python -c "
from training.pipeline import MergeConfig, _merge_adapters
from pathlib import Path

config = MergeConfig(
    base_model='Qwen/Qwen2.5-7B-Instruct',
    adapter_path=Path('$OUTPUT_DIR'),
    output_dir=Path('$OUTPUT_DIR/merged'),
)
result = _merge_adapters(config)
print(f'Merged model: {result}')
"

echo ""
echo "[4/5] Converting to GGUF..."
.venv/bin/python -c "
from training.pipeline import GGUFConfig, _convert_gguf
from pathlib import Path

config = GGUFConfig(
    hf_model_dir=Path('$OUTPUT_DIR/merged'),
    gguf_output=Path('$OUTPUT_DIR/model.gguf'),
    quantize='Q4_K_M',
    quantized_output=Path('$OUTPUT_DIR/model-Q4_K_M.gguf'),
)
result = _convert_gguf(config)
print(f'GGUF model: {result}')
"

echo ""
echo "[5/5] Creating Ollama model..."
cat > "$OUTPUT_DIR/Modelfile.finetuned" << 'MODELEOF'
FROM ./model-Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER num_predict 4096
PARAMETER top_p 0.9
PARAMETER num_gpu -1
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER top_k 20

SYSTEM """You are AgentNick, the core AI engine for ProcWise — an enterprise procurement intelligence platform. Extract structured data from procurement documents with absolute accuracy. Return only valid JSON."""
MODELEOF

cd "$OUTPUT_DIR"
ollama create BeyondProcwise/AgentNick:v2-finetuned -f Modelfile.finetuned
cd - >/dev/null

echo ""
echo "[6/6] EVAL GATE — candidate may replace production ONLY if it does not regress"
# Promotion gate: score the freshly-built candidate against the current
# production model on the held-out gold set. Promote to the production
# extraction tag (:extract) ONLY on PROMOTE_OK. Never ship a regression.
PROD_TAG="BeyondProcwise/AgentNick:extract"
CAND_TAG="BeyondProcwise/AgentNick:v2-finetuned"
.venv/bin/python scripts/gpu_upgrade/run_eval_gate.py \
    --label nightly --baseline "$PROD_TAG" --candidate "$CAND_TAG" \
    | tee "$OUTPUT_DIR/eval_gate.txt"

if grep -q "verdict=PROMOTE_OK" "$OUTPUT_DIR/eval_gate.txt"; then
    echo "EVAL GATE PASSED — promoting $CAND_TAG -> $PROD_TAG"
    ollama cp "$CAND_TAG" "$PROD_TAG"
    echo "PROMOTED: production extraction model updated to the new candidate."
else
    echo "EVAL GATE REFUSED (regression) — production $PROD_TAG left UNTOUCHED."
    echo "Candidate kept as $CAND_TAG for inspection; no production change."
fi

echo ""
echo "=== Fine-tuning Complete ==="
echo "Finished: $(date)"
echo "Candidate: $CAND_TAG (promoted to $PROD_TAG only if the eval gate passed above)"
