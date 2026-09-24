"""The language registry: built from CLDR, searched the way people type."""
from __future__ import annotations

import pytest

from src.services.i18n.registry import build_registry, fold, is_source_language

MODEL = "BeyondProcwise/AgentNick:unified"


@pytest.fixture(scope="module")
def reg():
    return build_registry(MODEL)


def test_built_from_cldr_not_a_short_list(reg):
    assert len(reg.all()) > 500


def test_entry_shape_native_first(reg):
    ja = reg.get("ja")
    assert (ja.english, ja.native, ja.dir) == ("Japanese", "日本語", "ltr")
    assert ja.label() == "日本語 — Japanese"
    assert reg.get("en").label() == "English"


def test_autonym_is_capitalised(reg):
    assert reg.get("es").native == "Español"


@pytest.mark.parametrize("code", ["ar", "he", "fa", "ur", "yi", "ckb", "syr"])
def test_rtl_languages(reg, code):
    assert reg.get(code).dir == "rtl"


def test_regional_and_script_variants_are_codes(reg):
    assert reg.get("zh-Hant").english == "Traditional Chinese"
    assert reg.get("zh_hant").code == "zh-Hant"


def test_fold_is_accent_insensitive():
    assert fold("Español") == fold("espanol") == "espanol"


@pytest.mark.parametrize("query,code", [
    ("espanol", "es"), ("日本", "ja"), ("japanese", "ja"), ("JA", "ja"),
    ("farsi", "fa"), ("iw", "he"), ("jpn", "ja"), ("castilian", "es"),
])
def test_search_matches_names_codes_aliases(reg, query, code):
    assert reg.search(query)[0].code == code


def test_empty_search_pins_first(reg):
    out = reg.search("", pinned=["en", "fr-CA", "ja"])
    assert [L.code for L in out[:3]] == ["en", "fr-CA", "ja"]


def test_pin_codes_from_accept_language(reg):
    assert reg.pin_codes("de-DE,de;q=0.9,en-US;q=0.8") == ["en", "de", "en-US"]


def test_tiers_come_from_config(reg):
    assert reg.get("en").tier == 1
    assert reg.get("pl").tier == 2
    assert reg.get("zu").tier == 3


def test_resolve_known_name(reg):
    assert reg.resolve("  Español ").code == "es"


def test_resolve_unknown_is_custom(reg):
    L = reg.resolve("Klingon Pirate Speak")
    assert L.custom and L.code == "x-klingon-pirate-speak"
    assert L.english == "Klingon Pirate Speak" and L.tier == 3


def test_resolve_non_latin_custom_gets_stable_code(reg):
    name = "ᏣᎳᎩ ᏗᏎᏍᏗ ᎤᏍᏗ"
    a, b = reg.resolve(name), reg.resolve(name)
    assert a.custom and a.code == b.code and a.code.startswith("x-")


def test_resolve_rejects_blank_and_overlong(reg):
    with pytest.raises(ValueError):
        reg.resolve("   ")
    with pytest.raises(ValueError):
        reg.resolve("x" * 61)


def test_source_language():
    assert is_source_language("en") and is_source_language("en-GB")
    assert not is_source_language("es")
