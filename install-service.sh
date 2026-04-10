#!/bin/bash
set -euo pipefail

SERVICE_NAME="procwise"
SERVICE_FILE="$(dirname "$(readlink -f "$0")")/procwise.service"
TARGET="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== ProcWise Systemd Service Installer ==="

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0"
    exit 1
fi

# Stop existing manual server
echo "[1/6] Stopping any running ProcWise instances..."
if lsof -ti :8000 >/dev/null 2>&1; then
    kill "$(lsof -ti :8000)" 2>/dev/null || true
    sleep 2
    echo "  Stopped."
else
    echo "  None running."
fi

# Ensure NVIDIA module is loaded
echo "[2/6] Ensuring NVIDIA GPU driver is loaded..."
if ! nvidia-smi >/dev/null 2>&1; then
    modprobe nvidia 2>/dev/null || echo "  WARNING: nvidia module not available"
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "  GPU not available"

# Add user to docker group if needed
echo "[3/6] Ensuring docker group membership..."
if ! groups muthu | grep -q docker; then
    usermod -aG docker muthu
    echo "  Added muthu to docker group (re-login needed for manual docker use)"
fi

# Install service file
echo "[4/6] Installing systemd service..."
cp "$SERVICE_FILE" "$TARGET"
chmod 644 "$TARGET"
systemctl daemon-reload
echo "  Installed: $TARGET"

# Enable and start
echo "[5/6] Enabling service for boot..."
systemctl enable "$SERVICE_NAME"

echo "[6/6] Starting ProcWise..."
systemctl start "$SERVICE_NAME"
sleep 5

# Verify
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo ""
    echo "=== ProcWise is RUNNING ==="
    echo "  Status:  systemctl status procwise"
    echo "  Logs:    journalctl -u procwise -f"
    echo "  Stop:    sudo systemctl stop procwise"
    echo "  Restart: sudo systemctl restart procwise"
    echo ""
    systemctl status "$SERVICE_NAME" --no-pager -l | head -15
else
    echo ""
    echo "=== FAILED to start ==="
    journalctl -u "$SERVICE_NAME" --no-pager -n 20
    exit 1
fi
