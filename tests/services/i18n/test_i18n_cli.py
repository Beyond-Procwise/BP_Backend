"""The admin CLIs: import pairs keys by English, pre-translate fills only what is missing."""
from __future__ import annotations

import json

from scripts import i18n_import_reviewed as imp
from scripts import i18n_pretranslate as pre
from src.services.i18n.service import TranslateResult
from src.services.i18n.store import source_hash


def test_pairs_only_keys_in_both_and_non_empty():
    out = imp.pairs({"a": "Save", "b": "Close", "c": "Open"}, {"a": "Guardar", "b": "", "z": "huérfano"})
    assert out == {source_hash("Save"): ("Save", "Guardar")}


def test_pretranslate_translates_only_missing(tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({"a": "Save", "b": "Close"}))
    seen = {}

    class Svc:
        batch_size = 40

        def cached(self, lang, texts):
            return {"a": "Guardar"}, ["b"]

        def translate(self, lang, texts, *, lang_name=None):
            seen.update(texts)
            return TranslateResult(translations={k: v.upper() for k, v in texts.items()})

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    assert pre.main(["--lang", "es", "--catalog", str(catalog)]) == 0
    assert seen == {"b": "Close"}
    assert "1 cached, 1 to translate" in capsys.readouterr().out


def test_pretranslate_reports_failures_in_exit_code(tmp_path, monkeypatch):
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({"a": "Save"}))

    class Svc:
        batch_size = 40

        def cached(self, lang, texts):
            return {}, ["a"]

        def translate(self, lang, texts, *, lang_name=None):
            return TranslateResult(translations=dict(texts), failed=["a"])

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    assert pre.main(["--lang", "es", "--catalog", str(catalog)]) == 1


def test_pretranslate_dry_run_calls_nothing(tmp_path, monkeypatch):
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({"a": "Save"}))

    class Svc:
        batch_size = 40

        def cached(self, lang, texts):
            return {}, ["a"]

        def translate(self, *a, **k):
            raise AssertionError("dry run must not translate")

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    assert pre.main(["--lang", "es", "--catalog", str(catalog), "--dry-run"]) == 0
