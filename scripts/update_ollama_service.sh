#!/bin/bash
# Run with: sudo bash scripts/update_ollama_service.sh
# Updates Ollama systemd service with performance tuning for A10G GPU

set -e

SERVICE_FILE="/etc/systemd/system/ollama.service"
BACKUP="${SERVICE_FILE}.bak.$(date +%s)"

echo "Backing up ${SERVICE_FILE} to ${BACKUP}"
cp "$SERVICE_FILE" "$BACKUP"

cat > "$SERVICE_FILE" << 'SVCEOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/home/muthu/PycharmProjects/BP_Backend/.venv/bin:/home/muthu/.local/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/include:/home/muthu/.local/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/include:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/muthu/.lmstudio/bin:/home/muthu/.lmstudio/bin:/home/muthu/.lmstudio/bin"
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_GPU_OVERHEAD=512m"
SVCEOF

# Append the Install section
cat >> "$SERVICE_FILE" << 'INSTEOF'

[Install]
WantedBy=default.target
INSTEOF

echo "Reloading systemd and restarting Ollama..."
systemctl daemon-reload
systemctl restart ollama

echo "Waiting for Ollama to start..."
sleep 3

# Preload the primary model to avoid cold-start latency
echo "Preloading qwen2.5:32b model..."
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:32b","prompt":"","keep_alive":"24h"}' > /dev/null 2>&1 &

echo ""
echo "Done. Ollama service updated with:"
echo "  OLLAMA_NUM_PARALLEL=8      (up from 4 - handles 8 concurrent agent requests)"
echo "  OLLAMA_MAX_LOADED_MODELS=2 (keep 2 models hot in VRAM)"
echo "  OLLAMA_KEEP_ALIVE=24h      (no model unloading during the day)"
echo "  OLLAMA_FLASH_ATTENTION=1   (faster attention computation)"
echo "  OLLAMA_GPU_OVERHEAD=512m   (reserve 512MB for system/KV cache overhead)"
echo ""
echo "Verify with: systemctl status ollama"
