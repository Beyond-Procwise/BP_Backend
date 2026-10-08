#!/usr/bin/env python3
"""Check /health's output signals once and mail any change.

Run every five minutes by the user-level systemd timer
procwise-health-alerts.timer (unit files in deploy/systemd-user/). All the
logic is in src/services/signal_alerts.py; this prints one JSON line for the
journal and exits non-zero only when the alerting itself crashed.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, "ts": int(time.time()), **fields}, default=str), flush=True)


def main() -> int:
    try:
        from src.services import signal_alerts
        _emit("signal_alerts", **signal_alerts.run_from_env())
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit("signal_alerts_error", error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
