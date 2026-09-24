"""Translate English strings into any language, never calling the model for what is cached.

The unit of work is the English TEXT, not the key: two keys with the same English share one
translation and one cache row, and a key whose English changed is simply a new text (its old
translation is never shown against new English). Keys map back at the end.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.services.i18n.adapters import TranslationProvider, batch_schema
from src.services.i18n.prompts import PROMPT_VERSION, build_ui_prompt
from src.services.i18n.registry import Language, LanguageRegistry, is_source_language
from src.services.i18n.store import MemoryLayer, source_hash
from src.services.i18n.validate import validate_batch

logger = logging.getLogger(__name__)

# A private-use code as registry.resolve() mints it: x- plus 1-4 subtags of 1-8 [a-z0-9].
_CUSTOM_CODE = re.compile(r"^x(-[a-z0-9]{1,8}){1,4}$")


@dataclass
class TranslateResult:
    translations: dict[str, str]
    failed: list[str] = field(default_factory=list)
    model_calls: int = 0


class TranslationService:
    def __init__(self, *, provider: TranslationProvider, store, memory: MemoryLayer,
                 registry: LanguageRegistry, system_prompt: str, batch_size: int,
                 batch_chars: int = 6000, failure_backoff: float = 900.0,
                 prompt_version: str = PROMPT_VERSION, clock=time.monotonic, on_generated=None):
        self.provider, self.store, self.memory = provider, store, memory
        self.registry, self.system_prompt = registry, system_prompt
        self.batch_size, self.batch_chars, self.prompt_version = batch_size, batch_chars, prompt_version
        # A text that failed twice is not sent again until the back-off passes: without this,
        # every poll from an open screen would spend two model calls on it, forever.
        self.failure_backoff, self._clock = failure_backoff, clock
        self._failed: dict[tuple[str, str], float] = {}
        self._failed_lock = threading.Lock()
        # Called with lang/model/prompt_version/hashes whenever new translations are stored
        # (the audit trail's translation.generated). Best-effort: never breaks a translation.
        self._on_generated = on_generated

    # -- language -----------------------------------------------------------------------
    def language(self, code: str, name: Optional[str] = None) -> Language:
        """The language for a code. A custom x- code must be one resolve() could have minted,
        and a name sent with it must resolve to that same code: the name reaches the prompt
        but is not in the cache key, so it may not vary independently of the code."""
        if code.lower().startswith("x-"):
            code = code.lower()
            if not _CUSTOM_CODE.match(code):
                raise ValueError(f"malformed custom language code {code!r}")
            if name is not None and self.registry.resolve(name).code != code:
                raise ValueError(f"language name {name!r} does not match code {code!r}")
            label = (name or code[2:].replace("-", " ").title()).strip()
            return Language(code=code, english=label, native=label, aliases=(),
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

    def _recently_failed(self, lang: str, hashes) -> set[str]:
        now = self._clock()
        with self._failed_lock:
            return {h for h in hashes if self._failed.get((lang, h), 0) > now}

    def status(self, lang: str, texts: dict[str, str]) -> tuple[dict[str, str], list[str], list[str]]:
        """(cached translations, keys still to translate, keys that failed recently)."""
        hits, missing = self.cached(lang, texts)
        if not missing:
            return hits, [], []
        code = self.language(lang).code
        hashes = {k: source_hash(texts[k]) for k in missing}
        failed_h = self._recently_failed(code, hashes.values())
        return hits, [k for k in missing if hashes[k] not in failed_h], [k for k in missing if hashes[k] in failed_h]

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
    def _chunks(self, texts: dict[str, str]):
        """Batches of at most batch_size texts AND batch_chars characters: forty long texts
        would overrun the model's output budget and fail as one. An oversized text goes alone."""
        chunk: dict[str, str] = {}
        size = 0
        for h, text in texts.items():
            if chunk and (len(chunk) >= self.batch_size or size + len(text) > self.batch_chars):
                yield chunk
                chunk, size = {}, 0
            chunk[h] = text
            size += len(text)
        if chunk:
            yield chunk

    def _call(self, lang: Language, batch: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        ids = {f"s{i + 1:02d}": h for i, h in enumerate(batch)}
        payload = {sid: batch[h] for sid, h in ids.items()}
        prompt = build_ui_prompt(self.system_prompt, lang.label(), lang.code, payload)
        raw = self.provider.complete_json(prompt, batch_schema(list(payload)))
        good, bad = validate_batch(payload, raw, lang.code)
        return {ids[s]: t for s, t in good.items()}, {ids[s]: why for s, why in bad.items()}

    def translate(self, lang: str, texts: dict[str, str], *, lang_name: Optional[str] = None) -> TranslateResult:
        language = self.language(lang, lang_name)
        if is_source_language(language.code):
            return TranslateResult(translations=dict(texts))
        hashes = {k: source_hash(v) for k, v in texts.items()}
        by_hash = {h: texts[k] for k, h in hashes.items()}
        done = self._lookup(language.code, by_hash)
        pending = [h for h in by_hash if h not in done]
        skipped = self._recently_failed(language.code, pending)
        todo = [h for h in pending if h not in skipped]
        calls, failed_hashes = 0, {h: "failed recently; in back-off" for h in skipped}
        for chunk in self._chunks({h: by_hash[h] for h in todo}):
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
            if good and self._on_generated is not None:
                try:
                    self._on_generated(lang=language.code, model=self.provider.model,
                                       prompt_version=self.prompt_version, hashes=list(good))
                except Exception as exc:
                    logger.warning("i18n: auditing %d new translation(s) failed: %s", len(good), exc)
            if bad:
                until = self._clock() + self.failure_backoff
                with self._failed_lock:
                    for h in bad:
                        self._failed[(language.code, h)] = until
        failed = [k for k, h in hashes.items() if h in failed_hashes]
        if failed:
            logger.warning("i18n: %d key(s) served in English for %s after one retry: %s",
                           len(failed), language.code,
                           {k: failed_hashes[hashes[k]] for k in failed[:20]})
        return TranslateResult(
            translations={k: done.get(h, texts[k]) for k, h in hashes.items()},
            failed=failed, model_calls=calls,
        )
