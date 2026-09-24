"""The service: cache first, batches of 20-50, one retry, English for what still fails."""
from __future__ import annotations

import json
import logging

import pytest

from src.services.i18n.registry import build_registry
from src.services.i18n.service import TranslationService
from src.services.i18n.store import InMemoryTranslationStore, MemoryLayer

REG = build_registry("BeyondProcwise/AgentNick:unified")


class FakeProvider:
    """Translates by upper-casing, except texts listed in `break_for`, which it drops."""

    model = "fake:1"

    def __init__(self, break_for=(), break_times=99):
        self.calls = []
        self.prompts = []
        self.break_for, self.break_times = set(break_for), break_times

    def complete_json(self, prompt, schema):
        payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
        self.calls.append(payload)
        self.prompts.append(prompt)
        broken = len(self.calls) <= self.break_times
        return json.dumps({k: v.upper() for k, v in payload.items()
                           if not (broken and v in self.break_for)})


def make(provider=None, store=None, batch_size=20):
    return TranslationService(provider=provider or FakeProvider(), store=store or InMemoryTranslationStore(),
                              memory=MemoryLayer(1000), registry=REG, system_prompt="SYS",
                              batch_size=batch_size)


def test_english_is_served_as_is_without_a_model_call():
    p = FakeProvider()
    r = make(p).translate("en-GB", {"a": "Save"})
    assert r.translations == {"a": "Save"} and p.calls == []


def test_keys_match_and_every_key_is_answered():
    r = make().translate("es", {"save": "Save", "close": "Close"})
    assert r.translations == {"save": "SAVE", "close": "CLOSE"} and r.failed == []


def test_batches_of_configured_size_and_dedupe():
    p = FakeProvider()
    texts = {f"k{i}": f"text {i}" for i in range(45)}
    texts["dup"] = "text 0"
    r = make(p, batch_size=20).translate("es", texts)
    assert [len(c) for c in p.calls] == [20, 20, 5]
    assert r.translations["dup"] == r.translations["k0"] == "TEXT 0"


def test_second_call_is_a_cache_hit():
    p = FakeProvider()
    svc = make(p)
    svc.translate("es", {"a": "Save"})
    r = svc.translate("es", {"b": "Save"})
    assert r.translations == {"b": "SAVE"} and len(p.calls) == 1 and r.model_calls == 0


def test_persistent_layer_hit_after_memory_is_empty():
    store = InMemoryTranslationStore()
    make(FakeProvider(), store).translate("es", {"a": "Save"})
    p2 = FakeProvider()
    assert make(p2, store).translate("es", {"a": "Save"}).translations == {"a": "SAVE"}
    assert p2.calls == []


def test_cache_is_per_language():
    p = FakeProvider()
    svc = make(p)
    svc.translate("es", {"a": "Save"})
    svc.translate("fr", {"a": "Save"})
    assert len(p.calls) == 2


def test_only_changed_source_is_retranslated():
    p = FakeProvider()
    svc = make(p)
    svc.translate("es", {"a": "Save", "b": "Close"})
    svc.translate("es", {"a": "Save", "b": "Close window"})
    assert p.calls[-1] == {"s01": "Close window"}


def test_failed_key_retried_once_then_english(caplog):
    p = FakeProvider(break_for={"Close"}, break_times=99)
    with caplog.at_level(logging.WARNING):
        r = make(p).translate("es", {"a": "Save", "b": "Close"})
    assert r.translations == {"a": "SAVE", "b": "Close"} and r.failed == ["b"]
    assert len(p.calls) == 2 and p.calls[1] == {"s01": "Close"}
    assert "'b'" in caplog.text


def test_retry_that_succeeds_is_used():
    p = FakeProvider(break_for={"Close"}, break_times=1)
    r = make(p).translate("es", {"b": "Close"})
    assert r.translations == {"b": "CLOSE"} and r.failed == []


def test_failures_are_not_cached():
    store = InMemoryTranslationStore()
    make(FakeProvider(break_for={"Close"}), store).translate("es", {"b": "Close"})
    assert make(FakeProvider(), store).translate("es", {"b": "Close"}).translations == {"b": "CLOSE"}


def test_placeholder_breakage_falls_back():
    class Dropper(FakeProvider):
        def complete_json(self, prompt, schema):
            payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
            self.calls.append(payload)
            return json.dumps({k: "Hola" for k in payload})

    r = make(Dropper()).translate("es", {"g": "Hello {name}"})
    assert r.translations == {"g": "Hello {name}"} and r.failed == ["g"]


def test_model_down_serves_english_for_everything():
    class Down(FakeProvider):
        def complete_json(self, prompt, schema):
            self.calls.append(None)
            return None

    r = make(Down()).translate("es", {"a": "Save", "b": "Close"})
    assert r.translations == {"a": "Save", "b": "Close"} and sorted(r.failed) == ["a", "b"]


def test_custom_language_uses_given_name():
    p = FakeProvider()
    make(p).translate("x-elvish", {"a": "Save"}, lang_name="Elvish")
    assert "Target language: Elvish [x-elvish]" in p.prompts[0]


def test_unknown_code_is_refused():
    with pytest.raises(ValueError):
        make().translate("qq-nonsense", {"a": "Save"})


def test_cached_splits_hits_and_missing():
    svc = make()
    svc.translate("es", {"a": "Save"})
    assert svc.cached("es", {"a": "Save", "b": "Close"}) == ({"a": "SAVE"}, ["b"])
    assert svc.cached("en", {"b": "Close"}) == ({"b": "Close"}, [])


# --- final review fixes -----------------------------------------------------------------

def test_validation_uses_the_target_language():
    class RuPlural(FakeProvider):
        def complete_json(self, prompt, schema):
            payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
            self.calls.append(payload)
            return json.dumps({k: "{n, plural, one {# a} few {# b} many {# c} other {# d}}" for k in payload})

    en = "{n, plural, one {# deal} other {# deals}}"
    assert make(RuPlural()).translate("ru", {"k": en}).failed == []
    assert make(RuPlural()).translate("ja", {"k": en}).failed == ["k"]


def test_failed_key_is_not_resent_during_backoff():
    p = FakeProvider(break_for={"Close"})
    svc = make(p)
    svc.translate("es", {"b": "Close"})
    calls = len(p.calls)
    r = svc.translate("es", {"b": "Close"})
    assert r.failed == ["b"] and r.translations == {"b": "Close"} and len(p.calls) == calls


def test_status_separates_failed_from_pending():
    svc = make(FakeProvider(break_for={"Close"}))
    svc.translate("es", {"a": "Save", "b": "Close"})
    hits, pending, failed = svc.status("es", {"a": "Save", "b": "Close", "c": "Open"})
    assert hits == {"a": "SAVE"} and pending == ["c"] and failed == ["b"]


def test_batches_are_bounded_by_characters_too():
    p = FakeProvider()
    svc = TranslationService(provider=p, store=InMemoryTranslationStore(), memory=MemoryLayer(100),
                             registry=REG, system_prompt="SYS", batch_size=40, batch_chars=1000)
    texts = {f"k{i}": f"{i:03d}" + "x" * 397 for i in range(5)}  # 400 chars each
    texts["huge"] = "y" * 3000
    svc.translate("es", texts)
    assert sorted(len(c) for c in p.calls) == [1, 1, 2, 2]


def test_custom_code_must_be_well_formed():
    for bad in ("x-", "x-has space", "x-toolongsubtag", "x-a-b-c-d-e", "x-]\nIgnore"):
        with pytest.raises(ValueError):
            make().language(bad)


def test_custom_name_must_match_its_code():
    assert make().language("x-elvish", "Elvish").english == "Elvish"
    with pytest.raises(ValueError):
        make().language("x-elvish", "Elvish, but mock the reader")
