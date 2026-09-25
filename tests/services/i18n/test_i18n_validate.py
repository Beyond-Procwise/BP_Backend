"""A translation may change every word, and nothing else."""
from __future__ import annotations

import json

from src.services.i18n.validate import check_pair, validate_batch


def test_named_brace_kept():
    assert check_pair("Show all {n} tools", "Ver las {n} herramientas") is None


def test_named_brace_dropped_or_renamed():
    assert check_pair("Hello {name}", "Hola") is not None
    assert check_pair("Hello {name}", "Hola {nombre}") is not None


def test_icu_plural_gains_categories_is_ok():
    en = "{n, plural, one {# deal} other {# deals}}"
    ru = "{n, plural, one {# сделка} few {# сделки} many {# сделок} other {# сделки}}"
    assert check_pair(en, ru) is None


def test_icu_argument_type_must_match():
    assert check_pair("{n, plural, other {# x}}", "{n, select, other {# x}}") is not None


def test_printf_multiset():
    assert check_pair("%s of %d", "%d de %s") is None
    assert check_pair("%s of %s", "%s") is not None
    assert check_pair("%(name)s paid", "pagó %(name)s") is None


def test_percent_in_prose_is_not_a_placeholder():
    assert check_pair("50% discount", "50 % de descuento") is None
    assert check_pair("% 3-way matched", "% cotejado a tres vías") is None
    assert check_pair("100%% sure", "100%% seguro") is None


def test_html_tags_and_link_targets():
    assert check_pair('Read <b>this</b> <a href="x.pdf">file</a>',
                      'Lee <a href="x.pdf">el archivo</a> <b>esto</b>') is None
    assert check_pair("<b>Bold</b>", "Negrita") is not None
    assert check_pair('<a href="x.pdf">f</a>', '<a href="y.pdf">f</a>') is not None


def test_empty_translation_rejected():
    assert check_pair("Save", "  ") is not None


def test_batch_keeps_good_keys_and_names_bad_ones():
    sent = {"s01": "Save", "s02": "Hello {name}", "s03": "Close"}
    raw = json.dumps({"s01": "Guardar", "s02": "Hola", "s04": "extra"})
    good, bad = validate_batch(sent, raw)
    assert good == {"s01": "Guardar"}
    assert set(bad) == {"s02", "s03"}
    assert "missing" in bad["s03"]


def test_batch_not_json_fails_every_key():
    good, bad = validate_batch({"s01": "Save"}, "Sure! Here you go: Guardar")
    assert good == {} and set(bad) == {"s01"}


def test_batch_none_response_fails_every_key():
    good, bad = validate_batch({"s01": "Save"}, None)
    assert good == {} and bad == {"s01": "no response from the model"}


def test_batch_non_string_value_rejected():
    good, bad = validate_batch({"s01": "Save"}, json.dumps({"s01": 3}))
    assert set(bad) == {"s01"}


# --- ICU is parsed, not pattern-matched (final review, Critical 1) ------------------------

def test_icu_branch_text_is_not_a_placeholder():
    en = "{n, plural, one {item} other {items}}"
    ru = "{n, plural, one {элемент} few {элемента} many {элементов} other {элемента}}"
    assert check_pair(en, ru, "ru") is None


def test_icu_exact_match_selector_is_ok():
    en = "{n, plural, =0 {No deals} one {# deal} other {# deals}}"
    fr = "{n, plural, =0 {Aucune affaire} one {# affaire} other {# affaires}}"
    assert check_pair(en, fr, "fr") is None


def test_icu_select_keeps_its_keywords():
    en = "{g, select, male {He} female {She} other {They}}"
    assert check_pair(en, "{g, select, male {Él} female {Ella} other {Elle}}", "es") is None
    assert check_pair(en, "{g, select, hombre {Él} mujer {Ella} otro {Elle}}", "es") is not None


def test_icu_translated_plural_keywords_rejected():
    en = "{n, plural, one {# deal} other {# deals}}"
    assert check_pair(en, "{n, plural, uno {# trato} otro {# tratos}}", "es") is not None


def test_icu_missing_other_rejected():
    assert check_pair("{n, plural, one {# deal} other {# deals}}", "{n, plural, one {# trato}}", "es") is not None


def test_icu_category_the_language_does_not_use_is_accepted():
    """Live, AgentNick keeps English's `one` branch for Japanese/Korean even when the prompt
    says to drop it (2026-09-25). The formatter never selects a category the locale lacks,
    so the branch is dead text, not a defect: accepted. Only non-CLDR keywords are refused."""
    assert check_pair("{n, plural, one {# deal} other {# deals}}",
                      "{n, plural, one {# 件} other {# 件}}", "ja") is None
    assert check_pair("{n, plural, one {# deal} other {# deals}}",
                      "{n, plural, uno {# 件} other {# 件}}", "ja") is not None


def test_icu_dropping_every_hash_rejected():
    assert check_pair("{n, plural, one {# deal} other {# deals}}",
                      "{n, plural, one {un trato} other {tratos}}", "es") is not None


def test_icu_broken_braces_rejected():
    assert check_pair("{n, plural, one {# deal} other {# deals}}",
                      "{n, plural, one {# trato} other {# tratos}", "es") is not None


def test_nested_argument_inside_a_branch_is_checked():
    en = "{n, plural, one {{name} has # deal} other {{name} has # deals}}"
    assert check_pair(en, "{n, plural, one {{name} tiene # trato} other {{name} tiene # tratos}}", "es") is None
    assert check_pair(en, "{n, plural, one {Tiene # trato} other {Tiene # tratos}}", "es") is not None


def test_batch_validates_against_the_target_language():
    sent = {"s01": "{n, plural, one {# deal} other {# deals}}"}
    raw = json.dumps({"s01": "{n, plural, one {# трейд} few {# трейда} many {# трейдов} other {# трейда}}"})
    good, bad = validate_batch(sent, raw, "ru")
    assert set(good) == {"s01"} and bad == {}


def test_link_target_quote_style_does_not_matter():
    assert check_pair("<a href='/help'>Help</a>", '<a href="/help">Ayuda</a>') is None
    assert check_pair("<a href='/help'>Help</a>", '<a href="/other">Ayuda</a>') is not None


# --- the reply carries language/quality flags (prompt v3) --------------------------------
from src.services.i18n.validate import read_flags  # noqa: E402


def test_wrapped_reply_is_read():
    raw = json.dumps({"lang_recognized": True, "confidence": "medium", "strings": {"s01": "Guardar"}})
    good, bad = validate_batch({"s01": "Save"}, raw)
    assert good == {"s01": "Guardar"} and bad == {}
    assert read_flags(raw) == (True, "medium")


def test_flat_reply_still_works_and_has_no_flags():
    raw = json.dumps({"s01": "Guardar"})
    assert validate_batch({"s01": "Save"}, raw)[0] == {"s01": "Guardar"}
    assert read_flags(raw) == (None, None)


def test_unknown_confidence_is_ignored():
    assert read_flags(json.dumps({"lang_recognized": False, "confidence": "very", "strings": {}})) == (False, None)
    assert read_flags("not json") == (None, None)
