"""AI translation. English is the source; every other language is translated on demand.

get_service()/get_filler()/get_registry() build the process-wide instances lazily, from env
(see settings.py). Nothing here touches the model or the database at import time.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_service = None
_filler = None
_registry = None


def get_registry():
    global _registry
    with _lock:
        if _registry is None:
            from src.services.i18n.registry import build_registry
            from src.services.i18n.settings import load_settings
            _registry = build_registry(load_settings().model)
        return _registry


def get_service():
    global _service
    registry = get_registry()
    with _lock:
        if _service is None:
            from src.services.i18n import audit
            from src.services.i18n.adapters import build_provider
            from src.services.i18n.prompts import load_ui_prompt
            from src.services.i18n.service import TranslationService
            from src.services.i18n.settings import load_settings
            from src.services.i18n.store import MemoryLayer, PgTranslationStore
            s = load_settings()
            _service = TranslationService(
                provider=build_provider(s), store=PgTranslationStore(),
                memory=MemoryLayer(s.memory_cache_size, ttl=s.memory_ttl), registry=registry,
                system_prompt=load_ui_prompt(), batch_size=s.batch_size,
                batch_chars=s.batch_chars, failure_backoff=s.failure_backoff,
                on_generated=audit.record_generated,
            )
        return _service


def get_filler():
    global _filler
    service = get_service()
    with _lock:
        if _filler is None:
            from src.services.i18n.filler import Filler
            from src.services.i18n.settings import load_settings
            from src.services.i18n.gpu_gate import GpuGate
            s = load_settings()
            _filler = Filler(service, max_items=s.queue_limit,
                             gate=GpuGate(util_threshold=s.yield_gpu_util),
                             poll_seconds=s.yield_poll_seconds)
        return _filler
