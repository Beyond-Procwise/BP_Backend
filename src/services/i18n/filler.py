"""Fills the cache in the background, one batch at a time, what-is-on-screen first.

One daemon thread per process: the GPU runs one generation at a time anyway, and the
client's next /i18n/strings call picks up whatever has landed. BACKGROUND batches yield the
GPU: while the gate says foreground work (extraction, chat) wants it, they wait. SCREEN
batches do not -- a person is looking at that screen. Nothing here is durable. A
restart drops the queue, and the next client call re-enqueues what is still missing.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Optional

from src.services.i18n.store import source_hash

logger = logging.getLogger(__name__)

SCREEN, BACKGROUND = 0, 1


class Filler:
    def __init__(self, service, *, start_thread: bool = True, max_items: int = 20000,
                 gate=None, poll_seconds: float = 2.0):
        self.service = service
        self.max_items = max(1, max_items)
        self.gate, self.poll_seconds = gate, poll_seconds
        self._items: dict[tuple[str, str], list] = {}  # (lang, hash) -> [priority, seq, text, lang_name]
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._start = start_thread

    def enqueue(self, lang: str, texts: dict[str, str], priority: int, lang_name: Optional[str] = None) -> int:
        added = 0
        with self._cv:
            for text in texts.values():
                key = (lang, source_hash(text))
                cur = self._items.get(key)
                if cur is None:
                    if len(self._items) >= self.max_items and not self._evict_for(priority):
                        continue  # full of work at least as urgent; the next poll asks again
                    self._items[key] = [priority, next(self._seq), text, lang_name]
                    added += 1
                elif priority < cur[0]:
                    cur[0] = priority
            if texts:
                self._cv.notify()
            if self._start and self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="i18n-filler", daemon=True)
                self._thread.start()
        return added

    def _evict_for(self, priority: int) -> bool:
        """Make room for work of `priority` by dropping the newest less-urgent item."""
        victims = [(v[0], v[1], k) for k, v in self._items.items() if v[0] > priority]
        if not victims:
            return False
        del self._items[max(victims)[2]]
        return True

    def pending(self, lang: str) -> int:
        with self._cv:
            return sum(1 for (L, _h) in self._items if L == lang)

    def _take(self) -> tuple[str, dict[str, str], Optional[str]] | None:
        # Asked outside the lock: the card check can take seconds, and enqueue must not wait.
        yielding = self.gate is not None and bool(self._items) and self.gate.busy()
        with self._cv:
            if not self._items:
                return None
            order = sorted(self._items.items(), key=lambda kv: (kv[1][0], kv[1][1]))
            if yielding:
                # The GPU is wanted by foreground work: only on-screen items may go now.
                order = [kv for kv in order if kv[1][0] == SCREEN]
                if not order:
                    return None
            lang = order[0][0][0]
            size = getattr(self.service, "batch_size", 40)
            chosen = [(k, v) for k, v in order if k[0] == lang][:size]
            for k, _ in chosen:
                del self._items[k]
            return lang, {k[1]: v[2] for k, v in chosen}, chosen[0][1][3]

    def run_once(self) -> int:
        """Process one batch; 0 when there is nothing to do or background work is yielding."""
        job = self._take()
        if job is None:
            return 0
        lang, texts, lang_name = job
        try:
            self.service.translate(lang, texts, lang_name=lang_name)
        except Exception:
            logger.exception("i18n: background batch for %s failed; %d strings stay English until re-requested",
                             lang, len(texts))
        return len(texts)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._items:
                    self._cv.wait()
            if self.run_once() == 0:
                # Yielding to foreground work: look again shortly, or sooner if a SCREEN
                # item arrives (enqueue notifies).
                with self._cv:
                    self._cv.wait(timeout=self.poll_seconds)
