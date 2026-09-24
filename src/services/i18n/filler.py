"""Fills the cache in the background, one batch at a time, what-is-on-screen first.

One daemon thread per process: the GPU runs one generation at a time anyway, and the
client's next /i18n/strings call picks up whatever has landed. Nothing here is durable. A
restart drops the queue, and the next client call re-enqueues what is still missing.
"""
from __future__ import annotations

import itertools
import logging
import threading
from typing import Optional

from src.services.i18n.store import source_hash

logger = logging.getLogger(__name__)

SCREEN, BACKGROUND = 0, 1


class Filler:
    def __init__(self, service, *, start_thread: bool = True):
        self.service = service
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

    def pending(self, lang: str) -> int:
        with self._cv:
            return sum(1 for (L, _h) in self._items if L == lang)

    def _take(self) -> tuple[str, dict[str, str], Optional[str]] | None:
        with self._cv:
            if not self._items:
                return None
            order = sorted(self._items.items(), key=lambda kv: (kv[1][0], kv[1][1]))
            lang = order[0][0][0]
            size = getattr(self.service, "batch_size", 40)
            chosen = [(k, v) for k, v in order if k[0] == lang][:size]
            for k, _ in chosen:
                del self._items[k]
            return lang, {k[1]: v[2] for k, v in chosen}, chosen[0][1][3]

    def run_once(self) -> int:
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
            self.run_once()
