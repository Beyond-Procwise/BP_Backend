"""The prompt's editable slots: filled from config/i18n/translation.json, empty by default.

Editing the config changes the effective prompt version, so translations made under the old
slot values are refreshed rather than served against a prompt that no longer asks for them.
"""
from __future__ import annotations

import json

from src.services.i18n import prompts

TEMPLATE = ("Never translate these words: {{DO_NOT_TRANSLATE_LIST}}\n"
            "Use these approved translations when they apply: {{GLOSSARY}}\n"
            "Tone: {{TONE}}.")


def test_the_shipped_config_is_empty():
    cfg = prompts.load_translation_config()
    assert cfg == {"do_not_translate": [], "glossary": {}, "tone": ""}


def test_empty_slots_read_as_none_not_blank():
    out = prompts.fill_slots(TEMPLATE, {"do_not_translate": [], "glossary": {}, "tone": ""}, "de")
    assert "{{" not in out
    assert "words: none" in out and "apply: none" in out and "Tone: not specified." in out


def test_slots_are_filled_for_the_target_language():
    cfg = {"do_not_translate": ["SpendIQ", "AgentNick"],
           "glossary": {"purchase order": {"de": "Bestellung", "fr": "bon de commande"},
                        "invoice": "Rechnung-für-alle"},
           "tone": "friendly and professional"}
    de = prompts.fill_slots(TEMPLATE, cfg, "de-AT")
    assert '"SpendIQ", "AgentNick"' in de
    assert '"purchase order" -> "Bestellung"' in de and '"invoice" -> "Rechnung-für-alle"' in de
    assert "Tone: friendly and professional." in de
    ja = prompts.fill_slots(TEMPLATE, cfg, "ja")
    assert "purchase order" not in ja and '"invoice"' in ja


def test_the_config_is_part_of_the_prompt_version():
    empty = {"do_not_translate": [], "glossary": {}, "tone": ""}
    toned = {**empty, "tone": "formal"}
    assert prompts.effective_version(empty) == prompts.effective_version(dict(empty))
    assert prompts.effective_version(empty) != prompts.effective_version(toned)
    assert prompts.effective_version(empty).startswith(prompts.PROMPT_VERSION + "+cfg.")


def test_the_config_file_is_reread_when_edited(tmp_path, monkeypatch):
    f = tmp_path / "translation.json"
    f.write_text(json.dumps({"do_not_translate": [], "glossary": {}, "tone": ""}))
    monkeypatch.setenv("TRANSLATION_CONFIG_FILE", str(f))
    assert prompts.load_translation_config()["tone"] == ""
    import os, time
    f.write_text(json.dumps({"do_not_translate": [], "glossary": {}, "tone": "formal"}))
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert prompts.load_translation_config()["tone"] == "formal"


def test_a_broken_config_falls_back_to_empty(tmp_path, monkeypatch):
    f = tmp_path / "translation.json"
    f.write_text("{not json")
    monkeypatch.setenv("TRANSLATION_CONFIG_FILE", str(f))
    assert prompts.load_translation_config() == {"do_not_translate": [], "glossary": {}, "tone": ""}


def test_the_prompt_describes_what_the_code_sends_and_reads():
    """The prompt adapts to the code: same input markers, same reply fields."""
    from src.services.i18n.adapters import batch_schema
    template = prompts.load_ui_prompt()
    sent = prompts.build_ui_prompt("", "日本語 — Japanese", "ja", {"s01": "Save"})
    for marker in ("Target language:", "Input JSON:"):
        assert marker in template and marker in sent
    for field in batch_schema(["s01"])["properties"]:
        assert f'"{field}"' in template


def test_both_prompt_files_carry_the_three_slots():
    from pathlib import Path
    root = Path(prompts.__file__).resolve().parents[3] / "prompts"
    for name in ("translate-ui.txt", "translate-audio.txt"):
        text = (root / name).read_text(encoding="utf-8")
        for slot in ("{{DO_NOT_TRANSLATE_LIST}}", "{{GLOSSARY}}", "{{TONE}}"):
            assert slot in text, (name, slot)
    assert "LANGUAGE_NOT_SUPPORTED" in (root / "translate-audio.txt").read_text(encoding="utf-8")
