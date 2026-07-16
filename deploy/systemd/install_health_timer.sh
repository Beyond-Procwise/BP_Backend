#!/usr/bin/env bash
# Install the fixed bp-extraction-health timer (FINDINGS.md F12).
# Requires root: run with sudo. Idempotent.
set -euo pipefail

REPO=/home/muthu/PycharmProjects/BP_Backend/deploy/systemd
DEST=/etc/systemd/system

echo "Installing fixed timer (OnCalendar=*:0/5, self-healing)..."
install -m 0644 "$REPO/bp-extraction-health.timer"   "$DEST/bp-extraction-health.timer"
install -m 0644 "$REPO/bp-extraction-health.service" "$DEST/bp-extraction-health.service"

systemctl daemon-reload
systemctl enable --now bp-extraction-health.timer

echo
echo "Done. Current timer state:"
systemctl list-timers bp-extraction-health.timer --no-pager || true
echo
echo "NEXT elapse should now be a real time (not n/a). Tail logs with:"
echo "  journalctl -u bp-extraction-health.service -f"
