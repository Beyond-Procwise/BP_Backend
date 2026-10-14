"""The translation prompt, its editable slots, and its version.

PROMPT_VERSION is part of every cache key: a translation made under one prompt is not
reused under another. PROMPT_SHA256 pins the file, so editing the prompt without bumping
the version fails a test instead of silently serving old-prompt translations.

The prompt's {{DO_NOT_TRANSLATE_LIST}}, {{GLOSSARY}} and {{TONE}} slots are filled from
config/i18n/translation.json (TRANSLATION_CONFIG_FILE overrides), re-read when it changes.
The config's digest is part of the effective version, so editing it refreshes the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPT_VERSION = "translate-ui/4"  # 3-4: the product owner's prompt (4: plural categories the language lacks are dropped); reply carries lang_recognized + confidence; slots from config/i18n/translation.json
PROMPT_SHA256 = "088c6453e9ca250215d85259d2a3a8f27234e060257f8d721d5b50ab8b228e8e"

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "i18n" / "translation.json"
_EMPTY: dict[str, Any] = {"do_not_translate": [], "glossary": {}, "tone": ""}
_config_cache: dict[str, Any] = {"key": None, "value": _EMPTY}
_config_lock = threading.Lock()


def load_translation_config() -> dict[str, Any]:
    """The slot values, re-read when the file changes. A missing or broken file is empty."""
    path = Path(os.environ.get("TRANSLATION_CONFIG_FILE") or _DEFAULT_CONFIG)
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return dict(_EMPTY)
    with _config_lock:
        if _config_cache["key"] == key:
            return _config_cache["value"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = {
            "do_not_translate": [str(w) for w in raw.get("do_not_translate") or [] if str(w).strip()],
            "glossary": {str(k): v for k, v in (raw.get("glossary") or {}).items()
                         if isinstance(v, (str, dict))},
            "tone": str(raw.get("tone") or "").strip(),
        }
    except (ValueError, AttributeError) as exc:
        logger.warning("i18n: %s is not valid; the prompt slots stay empty: %s", path, exc)
        value = dict(_EMPTY)
    with _config_lock:
        _config_cache.update(key=key, value=value)
    return value


def _glossary_for(glossary: dict[str, Any], lang: str) -> list[tuple[str, str]]:
    code = (lang or "").lower()
    base = code.split("-")[0]
    out = []
    for term, value in glossary.items():
        if isinstance(value, str):
            out.append((term, value))
        elif isinstance(value, dict):
            by_code = {str(k).lower(): v for k, v in value.items()}
            chosen = by_code.get(code) or by_code.get(base)
            if chosen:
                out.append((term, str(chosen)))
    return out


def fill_slots(template: str, config: dict[str, Any], lang: str) -> str:
    """Fill the three slots for one target language. Empty values read as 'none'."""
    words = config.get("do_not_translate") or []
    pairs = _glossary_for(config.get("glossary") or {}, lang)
    return (template
            .replace("{{DO_NOT_TRANSLATE_LIST}}", ", ".join(json.dumps(w, ensure_ascii=False) for w in words) or "none")
            .replace("{{GLOSSARY}}", "; ".join(f"{json.dumps(t, ensure_ascii=False)} -> {json.dumps(v, ensure_ascii=False)}"
                                               for t, v in pairs) or "none")
            .replace("{{TONE}}", config.get("tone") or "not specified"))


def effective_version(config: dict[str, Any], base: str = PROMPT_VERSION) -> str:
    """The prompt version plus a digest of the slot values: the cache key's prompt version."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    return f"{base}+cfg.{digest}"


def load_ui_prompt() -> str:
    return (_PROMPT_DIR / "translate-ui.txt").read_text(encoding="utf-8")


def build_ui_prompt(system: str, language_name: str, code: str, payload: dict[str, str]) -> str:
    return (
        f"{system.rstrip()}\n\n"
        f"Target language: {language_name} [{code}]\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=0)}"
    )
