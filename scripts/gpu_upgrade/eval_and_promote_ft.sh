#!/bin/bash
# Post-training: register the merged finetune as an Ollama candidate, eval-gate it
# vs production, and promote ONLY if it does not regress. Never ships a worse model.
set +e
cd /home/muthu/PycharmProjects/BP_Backend
MERGED=data/models/agentnick_qwen_ft/merged
CAND=BeyondProcwise/AgentNick:ft-candidate
PROD=BeyondProcwise/AgentNick:extract
LOG=artifacts/gpu_upgrade/ft_eval.log
exec > >(tee "$LOG") 2>&1

echo "### EVAL+PROMOTE START $(date -u +%H:%M:%S)"
if [ ! -d "$MERGED" ]; then echo "FATAL: merged model dir missing: $MERGED"; exit 1; fi

# Modelfile importing the merged HF safetensors (ollama converts to GGUF internally).
cat > data/models/agentnick_qwen_ft/Modelfile.candidate <<EOF
FROM ./merged
PARAMETER temperature 0
PARAMETER num_predict 2048
PARAMETER num_ctx 8192
PARAMETER num_gpu -1
EOF

echo "[1] ollama create $CAND (imports safetensors -> GGUF; takes a few min)..."
( cd data/models/agentnick_qwen_ft && ollama create "$CAND" -f Modelfile.candidate ) 2>&1 | tail -6
ollama list 2>/dev/null | grep -q "ft-candidate" && echo "candidate registered" || { echo "FATAL: ollama create failed"; exit 1; }

echo "[2] EVAL GATE: candidate vs production baseline (real context_layer over gold holdout)"
set -a; . ./.env 2>/dev/null; set +a
env OLLAMA_BASE_URL=http://127.0.0.1:11434 PGCONNECT_TIMEOUT=3 \
  .venv/bin/python scripts/gpu_upgrade/run_eval_gate.py \
  --label ft-candidate --baseline "$PROD" --candidate "$CAND" | tee artifacts/gpu_upgrade/eval_ft-candidate.txt

if grep -q "verdict=PROMOTE_OK" artifacts/gpu_upgrade/eval_ft-candidate.txt; then
  echo "[3] PROMOTE_OK -> promoting $CAND to $PROD"
  ollama cp "$CAND" "$PROD" && echo "PROMOTED: production extraction model updated."
else
  echo "[3] REGRESSION_REFUSE -> production $PROD UNCHANGED. Candidate kept as $CAND for inspection."
fi
echo "### EVAL+PROMOTE DONE $(date -u +%H:%M:%S)"
