"""The admin CLIs: import pairs keys by English, pre-translate fills only what is missing."""
from __future__ import annotations

import json

import pytest

from scripts import i18n_import_reviewed as imp
from scripts import i18n_pretranslate as pre
from src.services.i18n.service import TranslateResult
from src.services.i18n.store import source_hash


@pytest.fixture(autouse=True)
def _gpu_is_free(monkeypatch):
    """The real gate reads the live card; these tests must not depend on how busy it is."""
    class Free:
        def busy(self):
            return False
    monkeypatch.setattr(pre, "_gate", lambda: Free())


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


# --- audit: publishing the signed-out key list, and reviewed imports ---------------------

class _Store:
    def __init__(self):
        self.public, self.reviewed = None, {}

    def set_public_keys(self, keys):
        self.public = dict(keys)

    def public_keys(self):
        return dict(self.public or {})

    def reviewed_changes(self, lang, rows):
        return [(h, self.reviewed[h], new) for h, (_s, new) in rows.items() if h in self.reviewed and self.reviewed[h] != new]

    def import_reviewed(self, lang, rows):
        self.reviewed.update({h: t for h, (_s, t) in rows.items()})
        return len(rows)


def test_publish_public_selects_by_prefix_and_is_audited(tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({"auth.signIn": "Sign in", "landing.hero": "Welcome", "nav.home": "Home",
                                   "tb:Key indicators": "Key indicators"}))
    store, events = _Store(), []

    class Svc:
        batch_size = 40

        def __init__(self):
            self.store = store

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    monkeypatch.setattr(pre.audit, "record_public_keys", lambda **kw: events.append(kw))
    assert pre.main(["--publish-public", "--catalog", str(catalog)]) == 0
    assert store.public == {"auth.signIn": "Sign in", "landing.hero": "Welcome"}
    assert events and events[0]["total"] == 2 and events[0]["added"] == ["auth.signIn", "landing.hero"]


def test_publish_public_is_refused_when_it_cannot_be_audited(tmp_path, monkeypatch):
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({"auth.signIn": "Sign in"}))
    store = _Store()

    class Svc:
        batch_size = 40

        def __init__(self):
            self.store = store

    def down(**_):
        raise pre.audit.AuditWriteError("db down")

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    monkeypatch.setattr(pre.audit, "record_public_keys", down)
    assert pre.main(["--publish-public", "--catalog", str(catalog)]) == 2
    assert store.public is None


def test_reviewed_import_audits_replaced_text_first(tmp_path, monkeypatch):
    en, tr = tmp_path / "en.json", tmp_path / "es.json"
    en.write_text(json.dumps({"a": "Save", "b": "Close"}))
    tr.write_text(json.dumps({"a": "Guardar", "b": "Cerrar"}))
    store, events = _Store(), []
    store.reviewed[source_hash("Save")] = "Salvar"
    monkeypatch.setattr(imp, "PgTranslationStore", lambda: store)
    monkeypatch.setattr(imp.audit, "record_reviewed_import", lambda **kw: events.append(kw))
    assert imp.main(["--lang", "es", "--catalog", str(en), "--translations", str(tr)]) == 0
    (e,) = events
    assert e["lang"] == "es" and e["added"] == 2 and e["changed"] == [(source_hash("Save"), "Salvar", "Guardar")]
    assert e["imported_by"].startswith("cli:")


def test_reviewed_import_is_refused_when_it_cannot_be_audited(tmp_path, monkeypatch):
    en, tr = tmp_path / "en.json", tmp_path / "es.json"
    en.write_text(json.dumps({"a": "Save"}))
    tr.write_text(json.dumps({"a": "Guardar"}))
    store = _Store()

    def down(**_):
        raise imp.audit.AuditWriteError("db down")

    monkeypatch.setattr(imp, "PgTranslationStore", lambda: store)
    monkeypatch.setattr(imp.audit, "record_reviewed_import", down)
    assert imp.main(["--lang", "es", "--catalog", str(en), "--translations", str(tr)]) == 2
    assert store.reviewed == {}


def test_reviewed_import_canonicalises_the_language_and_refuses_unknown(tmp_path, monkeypatch):
    en, tr = tmp_path / "en.json", tmp_path / "es.json"
    en.write_text(json.dumps({"a": "Save"}))
    tr.write_text(json.dumps({"a": "Guardar"}))
    store, events = _Store(), []
    monkeypatch.setattr(imp, "PgTranslationStore", lambda: store)
    monkeypatch.setattr(imp.audit, "record_reviewed_import", lambda **kw: events.append(kw))
    assert imp.main(["--lang", "ES", "--catalog", str(en), "--translations", str(tr)]) == 0
    assert events[0]["lang"] == "es"
    assert imp.main(["--lang", "qq-zz", "--catalog", str(en), "--translations", str(tr)]) == 2


def test_reviewed_import_skips_human_text_that_breaks_placeholders(tmp_path, monkeypatch, capsys):
    en, tr = tmp_path / "en.json", tmp_path / "es.json"
    en.write_text(json.dumps({"a": "Save", "b": "Show all {n} tools"}))
    tr.write_text(json.dumps({"a": "Guardar", "b": "Ver todas las herramientas"}))
    store, events = _Store(), []
    monkeypatch.setattr(imp, "PgTranslationStore", lambda: store)
    monkeypatch.setattr(imp.audit, "record_reviewed_import", lambda **kw: events.append(kw))
    assert imp.main(["--lang", "es", "--catalog", str(en), "--translations", str(tr)]) == 0
    assert list(store.reviewed.values()) == ["Guardar"] and events[0]["added"] == 1
    assert "b" in capsys.readouterr().out


def test_pretranslate_waits_while_extraction_wants_the_gpu(tmp_path, monkeypatch, capsys):
    """The CLI is bulk work too: between chunks it yields the GPU like the filler does."""
    catalog = tmp_path / "en.json"
    catalog.write_text(json.dumps({f"k{i}": f"text {i}" for i in range(3)}))
    order = []

    class Svc:
        batch_size = 1  # chunk = batch_size * 5 = 5 -> one chunk

        def cached(self, lang, texts):
            return {}, list(texts)

        def translate(self, lang, texts, *, lang_name=None):
            order.append("translate")
            return TranslateResult(translations=dict(texts))

    busy = iter([True, True, False])

    class Gate:
        def busy(self):
            b = next(busy)
            order.append("busy" if b else "free")
            return b

    monkeypatch.setattr(pre, "get_service", lambda: Svc())
    monkeypatch.setattr(pre, "_gate", lambda: Gate())
    monkeypatch.setattr(pre.time, "sleep", lambda s: order.append("sleep"))
    assert pre.main(["--lang", "es", "--catalog", str(catalog)]) == 0
    assert order == ["busy", "sleep", "busy", "sleep", "free", "translate"]
