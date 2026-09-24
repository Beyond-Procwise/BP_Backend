"""The translation prompt and its version.

PROMPT_VERSION is part of every cache key: a translation made under one prompt is not
reused under another. PROMPT_SHA256 pins the file, so editing the prompt without bumping
the version fails a test instead of silently serving old-prompt translations.
"""
from __future__ import annotations

import json
from pathlib import Path

PROMPT_VERSION = "translate-ui/1"
PROMPT_SHA256 = "2e43c51a6c50f557aec9abe672f97e188dd7366ec0407bc5067475d3e093955f"

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


def load_ui_prompt() -> str:
    return (_PROMPT_DIR / "translate-ui.txt").read_text(encoding="utf-8")


def build_ui_prompt(system: str, language_name: str, code: str, payload: dict[str, str]) -> str:
    return (
        f"{system.rstrip()}\n\n"
        f"Target language: {language_name} [{code}]\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=0)}"
    )
