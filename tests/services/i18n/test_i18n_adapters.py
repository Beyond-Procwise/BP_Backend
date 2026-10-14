"""Settings from env, the prompt's version guard, and the Ollama adapter's request."""
from __future__ import annotations

import hashlib
import json

import pytest

from src.services.i18n import prompts
from src.services.i18n.adapters import OllamaAdapter, batch_schema, build_provider
from src.services.i18n.settings import load_settings


def test_defaults():
    s = load_settings({"OLLAMA_BASE_URL": "http://gpu:11434"})
    assert s.provider == "ollama" and s.model == "BeyondProcwise/AgentNick:unified"
    assert s.base_url == "http://gpu:11434" and s.batch_size == 40 and s.think is False


def test_clamps_temperature_and_batch():
    s = load_settings({"TRANSLATION_TEMPERATURE": "0.9", "TRANSLATION_BATCH_SIZE": "500"})
    assert s.temperature == 0.2 and s.batch_size == 50
    s = load_settings({"TRANSLATION_TEMPERATURE": "-1", "TRANSLATION_BATCH_SIZE": "3"})
    assert s.temperature == 0.0 and s.batch_size == 20


def test_think_unset_means_not_sent():
    assert load_settings({"TRANSLATION_THINK": ""}).think is None


def test_prompt_version_tracks_the_prompt_file():
    text = prompts.load_ui_prompt()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == prompts.PROMPT_SHA256, (
        "prompts/translate-ui.txt changed: bump PROMPT_VERSION (cached translations were made "
        "with the old prompt) and set PROMPT_SHA256 to the new hash."
    )


def test_build_ui_prompt_carries_language_and_payload():
    p = prompts.build_ui_prompt("SYS", "Japanese (日本語)", "ja", {"s01": "Save"})
    assert p.startswith("SYS") and "Japanese (日本語)" in p and "ja" in p
    assert json.loads(p.split("Input JSON:\n", 1)[1]) == {"s01": "Save"}


def test_schema_requires_exactly_the_keys_and_has_no_unions():
    sch = batch_schema(["s01", "s02"])
    inner = sch["properties"]["strings"]  # prompt v3 wraps the strings beside the flags
    assert inner["required"] == ["s01", "s02"] and inner["additionalProperties"] is False
    assert "oneOf" not in json.dumps(sch) and "anyOf" not in json.dumps(sch)


def test_ollama_adapter_request():
    seen = {}

    def fake_generate(prompt, **kw):
        seen.update(kw, prompt=prompt)
        return '{"s01": "保存"}'

    s = load_settings({"TRANSLATION_BASE_URL": "http://tr:11434", "TRANSLATION_MODEL": "m:1"})
    out = OllamaAdapter(s, generate=fake_generate).complete_json("P", {"type": "object"})
    assert out == '{"s01": "保存"}'
    assert seen["model"] == "m:1" and seen["base_url"] == "http://tr:11434"
    assert seen["format"] == {"type": "object"} and seen["think"] is False
    assert seen["temperature"] == 0.1 and seen["retries"] == 1


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        build_provider(load_settings({"TRANSLATION_PROVIDER": "nope"}))


def test_memory_ttl_from_env():
    assert load_settings({}).memory_ttl == 600
    assert load_settings({"TRANSLATION_MEMORY_TTL_SECONDS": "60"}).memory_ttl == 60


def test_reply_schema_asks_for_the_flags_first():
    sch = batch_schema(["s01", "s02"])
    assert list(sch["properties"]) == ["lang_recognized", "confidence", "strings"]
    assert sch["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert sch["properties"]["strings"]["required"] == ["s01", "s02"]
    assert sch["properties"]["strings"]["additionalProperties"] is False
