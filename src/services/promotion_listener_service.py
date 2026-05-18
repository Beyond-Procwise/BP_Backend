"""Daemon wrapper around extraction.promotion.run_listener.

Runs the NOTIFY listener for ``extraction_raw_ready_for_promotion`` in a
long-running thread started by backend_scheduler. The listener wakes when
the discrepancy table's resolution trigger fires, applies HITL fixes to
the corresponding _raw row, and promotes to _stg.

Mirrors the UicanvasBridge / EmailWatcherService start/stop contract.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)


class PromotionListenerService:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Import lazily so test environments without the extraction package
        # available can still construct the service.
        from src.services.extraction.promotion import run_listener
        self._thread = threading.Thread(
            target=self._run_forever,
            args=(run_listener,),
            name="promotion-listener",
            daemon=True,
        )
        self._thread.start()
        log.info("PromotionListenerService started")

    def stop(self) -> None:
        self._stop.set()

    def _run_forever(self, run_listener) -> None:
        while not self._stop.is_set():
            try:
                run_listener(self._stop)
            except Exception as exc:
                log.warning("promotion listener crashed; restarting in 10s: %s", exc)
                if self._stop.wait(10):
                    return
