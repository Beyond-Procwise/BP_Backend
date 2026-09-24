"""The two cache layers, and what an outage looks like (a miss, never an error)."""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from src.services.i18n.store import (
    InMemoryTranslationStore, MemoryLayer, PgTranslationStore, source_hash,
)


def test_hash_is_nfc_stable():
    assert source_hash("Café") == source_hash("Café")
    assert source_hash("Save") != source_hash("Save ")


def test_memory_layer_evicts_oldest():
    m = MemoryLayer(max_size=2)
    m.put("a", "1"); m.put("b", "2"); m.get("a"); m.put("c", "3")
    assert m.get("b") is None and m.get("a") == "1" and m.get("c") == "3"


def test_lookup_respects_version_and_model():
    s = InMemoryTranslationStore()
    h = source_hash("Save")
    s.save("es", "v1", "m1", {h: ("Save", "Guardar")})
    assert s.lookup("es", [h], "v1", "m1") == {h: "Guardar"}
    assert s.lookup("es", [h], "v2", "m1") == {}
    assert s.lookup("es", [h], "v1", "m2") == {}


def test_reviewed_wins_over_machine_for_any_version():
    s = InMemoryTranslationStore()
    h = source_hash("Save")
    s.save("es", "v1", "m1", {h: ("Save", "Salvar")})
    assert s.import_reviewed("es", {h: ("Save", "Guardar")}) == 1
    assert s.lookup("es", [h], "v1", "m1") == {h: "Guardar"}
    assert s.lookup("es", [h], "v9", "other") == {h: "Guardar"}


def test_store_outage_is_a_miss_not_an_error(monkeypatch):
    import src.services.i18n.store as mod

    @contextmanager
    def boom():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(mod, "get_conn", boom)
    s = PgTranslationStore()
    assert s.lookup("es", [source_hash("Save")], "v1", "m1") == {}
    s.save("es", "v1", "m1", {source_hash("Save"): ("Save", "Guardar")})  # logs, does not raise
    assert s.import_reviewed("es", {source_hash("Save"): ("Save", "Guardar")}) == 0


@pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB only")
def test_pg_round_trip():
    s = PgTranslationStore()
    h = source_hash("__i18n_live_test__")
    try:
        s.save("xx-test", "v1", "m1", {h: ("__i18n_live_test__", "machine")})
        assert s.lookup("xx-test", [h], "v1", "m1") == {h: "machine"}
        s.import_reviewed("xx-test", {h: ("__i18n_live_test__", "human")})
        assert s.lookup("xx-test", [h], "v1", "m1") == {h: "human"}
        assert s.lookup("xx-test", [h], "v2", "m2") == {h: "human"}
    finally:
        from src.services.db import get_conn
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_translation WHERE target_lang = 'xx-test'")


# --- audit: machine rows are immutable; reviewed changes are reported; public keys ------

def test_a_machine_translation_is_never_overwritten():
    s = InMemoryTranslationStore()
    h = source_hash("Save")
    s.save("es", "v1", "m1", {h: ("Save", "Guardar")})
    s.save("es", "v1", "m1", {h: ("Save", "Salvar")})
    assert s.lookup("es", [h], "v1", "m1") == {h: "Guardar"}


def test_reviewed_changes_lists_only_replaced_text():
    s = InMemoryTranslationStore()
    a, b, c = source_hash("Save"), source_hash("Close"), source_hash("Open")
    s.import_reviewed("es", {a: ("Save", "Salvar"), b: ("Close", "Cerrar")})
    changes = s.reviewed_changes("es", {a: ("Save", "Guardar"), b: ("Close", "Cerrar"), c: ("Open", "Abrir")})
    assert changes == [(a, "Salvar", "Guardar")]


def test_public_keys_and_languages_that_have_them():
    s = InMemoryTranslationStore()
    s.set_public_keys({"auth.signIn": "Sign in", "auth.password": "Password"})
    assert s.public_keys() == {"auth.signIn": "Sign in", "auth.password": "Password"}
    s.import_reviewed("es", {source_hash("Sign in"): ("Sign in", "Iniciar sesión")})
    s.save("ja", "v1", "m1", {source_hash("Sign in"): ("Sign in", "サインイン"),
                              source_hash("Password"): ("Password", "パスワード")})
    hashes = [source_hash("Sign in"), source_hash("Password")]
    assert s.languages_with(hashes, "v1", "m1") == {"es": 1, "ja": 2}
    s.set_public_keys({"auth.signIn": "Sign in"})
    assert s.public_keys() == {"auth.signIn": "Sign in"}


@pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB only")
def test_pg_machine_rows_are_immutable_and_changes_reported():
    s = PgTranslationStore()
    h = source_hash("__i18n_live_test2__")
    try:
        s.save("xx-test", "v1", "m1", {h: ("__i18n_live_test2__", "first")})
        s.save("xx-test", "v1", "m1", {h: ("__i18n_live_test2__", "second")})
        assert s.lookup("xx-test", [h], "v1", "m1") == {h: "first"}
        s.import_reviewed("xx-test", {h: ("__i18n_live_test2__", "human1")})
        assert s.reviewed_changes("xx-test", {h: ("__i18n_live_test2__", "human2")}) == [(h, "human1", "human2")]
        assert s.languages_with([h], "v1", "m1") == {"xx-test": 1}
    finally:
        from src.services.db import get_conn
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_translation WHERE target_lang = 'xx-test'")


def test_memory_entries_expire_so_a_reviewed_import_is_picked_up():
    now = [0.0]
    m = MemoryLayer(max_size=10, ttl=600, clock=lambda: now[0])
    m.put("k", "machine")
    now[0] = 599
    assert m.get("k") == "machine"
    now[0] = 601
    assert m.get("k") is None
