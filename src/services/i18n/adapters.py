"""One interface for any model provider; Ollama is the first adapter.

A provider takes a finished prompt and a JSON schema and returns the raw text, or None
when the call failed. Validation and retries live in the service, not here, so every
provider is held to the same rules.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from src.services.i18n.settings import TranslationSettings


class TranslationProvider(Protocol):
    model: str

    def complete_json(self, prompt: str, schema: dict) -> Optional[str]: ...


def batch_schema(keys: list[str]) -> dict:
    """The reply shape the prompt asks for. The flags come first, so the model states whether
    it knows the language before it writes a word in it. No unions: Ollama ignores them."""
    return {
        "type": "object",
        "properties": {
            "lang_recognized": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "strings": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in keys},
                "required": list(keys),
                "additionalProperties": False,
            },
        },
        "required": ["lang_recognized", "confidence", "strings"],
        "additionalProperties": False,
    }


class OllamaAdapter:
    def __init__(self, settings: TranslationSettings, generate: Callable[..., Optional[str]] | None = None):
        if generate is None:
            from src.services.ollama_client import ollama_generate as generate
        self._generate = generate
        self._s = settings
        self.model = settings.model

    def complete_json(self, prompt: str, schema: dict) -> Optional[str]:
        return self._generate(
            prompt,
            model=self._s.model,
            base_url=self._s.base_url,
            timeout=self._s.timeout,
            temperature=self._s.temperature,
            retries=1,  # the service owns the retry-once rule
            think=self._s.think,
            format=schema,
            background=True,  # never counted as foreground: extraction goes first
        )


def build_provider(settings: TranslationSettings) -> TranslationProvider:
    if settings.provider == "ollama":
        return OllamaAdapter(settings)
    raise ValueError(f"unknown TRANSLATION_PROVIDER {settings.provider!r}; supported: ollama")
