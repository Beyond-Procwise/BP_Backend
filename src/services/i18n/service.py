"""Translate English strings into any language, never calling the model for what is cached.

The unit of work is the English TEXT, not the key: two keys with the same English share one
translation and one cache row, and a key whose English changed is simply a new text (its old
translation is never shown against new English). Keys map back at the end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.services.i18n.adapters import TranslationProvider, batch_schema
from src.services.i18n.prompts import PROMPT_VERSION, build_ui_prompt
from src.services.i18n.registry import Language, LanguageRegistry, is_source_language
from src.services.i18n.store import MemoryLayer, source_hash
from src.services.i18n.validate import validate_batch

logger = logging.getLogger(__name__)


@dataclass
class TranslateResult:
    translations: dict[str, str]
    failed: list[str] = field(default_factory=list)
    model_calls: int = 0


class TranslationService:
    def __init__(self, *, provider: TranslationProvider, store, memory: MemoryLayer,
                 registry: LanguageRegistry, system_prompt: str, batch_size: int,
                 prompt_version: str = PROMPT_VERSION):
        self.provider, self.store, self.memory = provider, store, memory
        self.registry, self.system_prompt = registry, system_prompt
        self.batch_size, self.prompt_version = batch_size, prompt_version

    # -- language -----------------------------------------------------------------------
    def language(self, code: str, name: Optional[str] = None) -> Language:
        if code.lower().startswith("x-"):
            label = (name or code[2:].replace("-", " ").title()).strip()
            return Language(code=code.lower(), english=label, native=label, aliases=(),
                            dir="ltr", tier=3, custom=True)
        lang = self.registry.get(code)
        if lang is None:
            raise ValueError(f"unknown language code {code!r}")
        return lang

    def _mkey(self, lang: str, h: str) -> tuple:
        return (lang, h, self.prompt_version, self.provider.model)

    # -- cache --------------------------------------------------------------------------
    def _lookup(self, lang: str, by_hash: dict[str, str]) -> dict[str, str]:
        found: dict[str, str] = {}
        for h in by_hash:
            v = self.memory.get(self._mkey(lang, h))
            if v is not None:
                found[h] = v
        rest = [h for h in by_hash if h not in found]
        if rest:
            for h, v in self.store.lookup(lang, rest, self.prompt_version, self.provider.model).items():
                self.memory.put(self._mkey(lang, h), v)
                found[h] = v
        return found

    def cached(self, lang: str, texts: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        """Translations already cached, and the keys that are not. Never calls the model."""
        code = self.language(lang).code
        if is_source_language(code):
            return dict(texts), []
        hashes = {k: source_hash(v) for k, v in texts.items()}
        found = self._lookup(code, {h: texts[k] for k, h in hashes.items()})
        hits = {k: found[h] for k, h in hashes.items() if h in found}
        return hits, [k for k in texts if k not in hits]

    # -- model --------------------------------------------------------------------------
    def _call(self, lang: Language, batch: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        ids = {f"s{i + 1:02d}": h for i, h in enumerate(batch)}
        payload = {sid: batch[h] for sid, h in ids.items()}
        prompt = build_ui_prompt(self.system_prompt, lang.label(), lang.code, payload)
        raw = self.provider.complete_json(prompt, batch_schema(list(payload)))
        good, bad = validate_batch(payload, raw)
        return {ids[s]: t for s, t in good.items()}, {ids[s]: why for s, why in bad.items()}

    def translate(self, lang: str, texts: dict[str, str], *, lang_name: Optional[str] = None) -> TranslateResult:
        language = self.language(lang, lang_name)
        if is_source_language(language.code):
            return TranslateResult(translations=dict(texts))
        hashes = {k: source_hash(v) for k, v in texts.items()}
        by_hash = {h: texts[k] for k, h in hashes.items()}
        done = self._lookup(language.code, by_hash)
        todo = [h for h in by_hash if h not in done]
        calls, failed_hashes = 0, {}
        for i in range(0, len(todo), self.batch_size):
            chunk = {h: by_hash[h] for h in todo[i:i + self.batch_size]}
            good, bad = self._call(language, chunk)
            calls += 1
            if bad:  # retry once, only what failed
                good2, bad = self._call(language, {h: chunk[h] for h in bad})
                calls += 1
                good.update(good2)
            if good:
                self.store.save(language.code, self.prompt_version, self.provider.model,
                                {h: (chunk[h], t) for h, t in good.items()})
                for h, t in good.items():
                    self.memory.put(self._mkey(language.code, h), t)
            done.update(good)
            failed_hashes.update(bad)
        failed = [k for k, h in hashes.items() if h in failed_hashes]
        if failed:
            logger.warning("i18n: %d key(s) served in English for %s after one retry: %s",
                           len(failed), language.code,
                           {k: failed_hashes[hashes[k]] for k in failed[:20]})
        return TranslateResult(
            translations={k: done.get(h, texts[k]) for k, h in hashes.items()},
            failed=failed, model_calls=calls,
        )
