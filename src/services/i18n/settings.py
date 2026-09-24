"""Where the translator's model, endpoint and knobs come from: the environment, only."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationSettings:
    provider: str
    base_url: str
    model: str
    temperature: float
    timeout: int
    batch_size: int
    think: Optional[bool]
    memory_cache_size: int


def _clamp(name: str, value: float, lo: float, hi: float) -> float:
    if value < lo or value > hi:
        logger.warning("%s=%s is outside %s-%s; using %s", name, value, lo, hi, min(max(value, lo), hi))
    return min(max(value, lo), hi)


def load_settings(env: Mapping[str, str] | None = None) -> TranslationSettings:
    env = os.environ if env is None else env
    think_raw = env.get("TRANSLATION_THINK", "false").strip().lower()
    think = None if think_raw == "" else think_raw in ("1", "true", "yes", "on")
    return TranslationSettings(
        provider=env.get("TRANSLATION_PROVIDER", "ollama").strip().lower(),
        base_url=env.get("TRANSLATION_BASE_URL") or env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("TRANSLATION_MODEL", "BeyondProcwise/AgentNick:unified"),
        temperature=_clamp("TRANSLATION_TEMPERATURE", float(env.get("TRANSLATION_TEMPERATURE", "0.1")), 0.0, 0.2),
        timeout=int(env.get("TRANSLATION_TIMEOUT", "120")),
        batch_size=int(_clamp("TRANSLATION_BATCH_SIZE", int(env.get("TRANSLATION_BATCH_SIZE", "40")), 20, 50)),
        think=think,
        memory_cache_size=int(env.get("TRANSLATION_MEMORY_CACHE_SIZE", "20000")),
    )
