"""Thread-safe lazy-loading singleton registry for NLU models.

Heavy models (BERT-NER, Table-Transformer, YOLO layout) are loaded on first
access and cached. Tests override `_loader` to avoid real model loads.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Optional


class ModelRegistry:
    _instances: dict[str, Any] = {}
    _locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    _loader: Optional[Callable[[str], Any]] = None  # overridable in tests

    @classmethod
    def _default_loader(cls, name: str) -> Any:
        # Subclassed / patched in actual NLU modules
        raise NotImplementedError(f"No loader registered for {name}")

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls._instances:
            return cls._instances[name]
        with cls._locks[name]:
            if name in cls._instances:
                return cls._instances[name]
            loader = cls._loader if cls._loader else cls._default_loader
            cls._instances[name] = loader(name)
            return cls._instances[name]

    @classmethod
    def warm(cls, names: list[str]) -> None:
        for n in names:
            cls.get(n)
