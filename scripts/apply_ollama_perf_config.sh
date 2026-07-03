#!/usr/bin/env bash
# Applies the Ollama performance config so the planner (:unified) and the
# extraction models (:extract, nuextract) can coexist/rotate on the shared GPU.
#   OLLAMA_KEEP_ALIVE   -1   -> 5m   (models unload after idle instead of never)
#   OLLAMA_MAX_LOADED_MODELS 2 -> 3  (planner + 2 extraction models fit)
# Run with: sudo bash scripts/apply_ollama_perf_config.sh
set -euo pipefail

CONF=/etc/systemd/system/ollama.service.d/vram.conf

echo "=== before ==="
grep -E "OLLAMA_KEEP_ALIVE|OLLAMA_MAX_LOADED" "$CONF"

cp "$CONF" "$CONF.bak.$(date +%s)"
sed -i \
  -e 's/OLLAMA_KEEP_ALIVE=-1/OLLAMA_KEEP_ALIVE=5m/' \
  -e 's/OLLAMA_MAX_LOADED_MODELS=2/OLLAMA_MAX_LOADED_MODELS=3/' \
  "$CONF"

echo "=== after (file) ==="
grep -E "OLLAMA_KEEP_ALIVE|OLLAMA_MAX_LOADED" "$CONF"

systemctl daemon-reload
systemctl restart ollama
sleep 4

echo "=== effective ollama Environment ==="
systemctl show ollama -p Environment | tr ' ' '\n' | grep OLLAMA || true
echo "=== ollama state ==="
systemctl is-active ollama
echo "DONE"
