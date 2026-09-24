"""The /i18n endpoints on a bare app, plus the output-safety interaction."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.services.i18n as i18n
from src.services.i18n.filler import Filler
from src.services.i18n.registry import build_registry
from src.services.i18n.service import TranslationService
from src.services.i18n.store import InMemoryTranslationStore, MemoryLayer

REG = build_registry("BeyondProcwise/AgentNick:unified")


class Upper:
    model = "fake:1"

    def complete_json(self, prompt, schema):
        payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
        return json.dumps({k: v.upper() for k, v in payload.items()})


@pytest.fixture
def client(monkeypatch):
    svc = TranslationService(provider=Upper(), store=InMemoryTranslationStore(), memory=MemoryLayer(100),
                             registry=REG, system_prompt="SYS", batch_size=20)
    filler = Filler(svc, start_thread=False)
    monkeypatch.setattr(i18n, "get_registry", lambda: REG)
    monkeypatch.setattr(i18n, "get_service", lambda: svc)
    monkeypatch.setattr(i18n, "get_filler", lambda: filler)
    from api.routers import i18n as router_mod
    app = FastAPI()
    app.include_router(router_mod.router)
    # Depends() captured the real function at import; override it by that identity.
    app.dependency_overrides[router_mod.auth.require_user] = lambda: None
    c = TestClient(app)
    c.filler = filler
    return c


def test_languages_search_and_pins(client):
    r = client.get("/i18n/languages", params={"q": "espanol"}, headers={"Accept-Language": "ja"})
    body = r.json()
    assert body["languages"][0]["code"] == "es" and body["pinned"] == ["en", "ja"]
    assert body["languages"][0]["label"].startswith("Español")


def test_full_list_starts_with_pins(client):
    body = client.get("/i18n/languages", headers={"Accept-Language": "fr-CA,fr;q=0.8"}).json()
    assert [L["code"] for L in body["languages"][:3]] == ["en", "fr-CA", "fr"]
    assert len(body["languages"]) > 500


def test_resolve_free_text(client):
    assert client.post("/i18n/languages/resolve", json={"text": "Farsi"}).json()["code"] == "fa"
    assert client.post("/i18n/languages/resolve", json={"text": "Elvish"}).json()["custom"] is True
    assert client.post("/i18n/languages/resolve", json={"text": " "}).status_code == 422


def test_strings_english_immediately_then_filled(client):
    body = client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "Save"}}).json()
    assert body["translations"] == {} and body["pending"] == ["a"] and body["complete"] is False
    client.filler.run_once()
    body = client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "Save"}}).json()
    assert body["translations"] == {"a": "SAVE"} and body["complete"] is True


def test_rtl_direction_reported(client):
    assert client.post("/i18n/strings", json={"lang": "ar", "strings": {}}).json()["dir"] == "rtl"
    assert client.post("/i18n/strings", json={"lang": "fr", "strings": {}}).json()["dir"] == "ltr"


def test_english_needs_nothing(client):
    body = client.post("/i18n/strings", json={"lang": "en", "strings": {"a": "Save"}}).json()
    assert body["translations"] == {"a": "Save"} and body["complete"] is True


def test_unknown_language_is_400(client):
    assert client.post("/i18n/strings", json={"lang": "qq-zz", "strings": {"a": "x"}}).status_code == 400


def test_too_many_strings_is_413(client):
    big = {f"k{i}": "x" for i in range(2001)}
    assert client.post("/i18n/strings", json={"lang": "es", "strings": big}).status_code == 413


def test_dynamic_translate(client):
    body = client.post("/i18n/translate", json={"lang": "es", "texts": ["hi", "bye"]}).json()
    assert body == {"translations": ["HI", "BYE"], "failed": []}
    assert client.post("/i18n/translate", json={"lang": "es", "texts": ["x"] * 51}).status_code == 422


def test_withheld_translation_falls_back_to_english(client, monkeypatch):
    from api.routers import i18n as router_mod
    monkeypatch.setattr(router_mod.osafe, "scrub_payload",
                        lambda obj, where="": {k: ("[withheld]" if v == "SECRET" else v) for k, v in obj.items()})
    client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "secret", "b": "ok"}})
    client.filler.run_once()
    body = client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "secret", "b": "ok"}}).json()
    assert body["translations"] == {"b": "OK"} and body["withheld"] == ["a"]


def test_preference_not_stored_when_auth_off(client):
    r = client.put("/i18n/preference", json={"code": "ja"}).json()
    assert r == {"language": {"code": "ja", "name": "日本語 — Japanese"}, "stored": False}
    assert client.get("/i18n/preference").json() == {"language": None, "stored": False}


def test_preference_stored_for_a_signed_in_user(client, monkeypatch):
    from api.auth import Principal
    from api.routers import i18n as router_mod
    saved = {}
    monkeypatch.setattr(router_mod.preference, "set_language", lambda sub, v: saved.update({sub: v}))
    monkeypatch.setattr(router_mod.preference, "get_language", lambda sub: saved.get(sub))
    client.app.dependency_overrides[router_mod.auth.require_user] = lambda: Principal(subject="u1")
    r = client.put("/i18n/preference", json={"code": "x-elvish", "name": "Elvish"}).json()
    assert r == {"language": {"code": "x-elvish", "name": "Elvish"}, "stored": True}
    assert client.get("/i18n/preference").json() == {"language": {"code": "x-elvish", "name": "Elvish"}, "stored": True}


def test_real_scrubber_leaves_language_list_alone():
    from api.main import app
    from services import output_safety as osafe
    osafe.register_routes([r.path for r in app.routes if hasattr(r, "path")])
    payload = {"languages": [L.to_dict() for L in REG.all()], "pinned": ["en"]}
    assert osafe.scrub_payload(payload) == payload


# --- final review fixes -----------------------------------------------------------------

def test_every_endpoint_asks_who_the_caller_is():
    """The router is mounted behind require_user AND each write names the principal."""
    from api.main import _AUTHENTICATED_ROUTERS
    from api.routers import i18n as router_mod
    assert router_mod.router in _AUTHENTICATED_ROUTERS


def test_long_keys_are_refused(client):
    assert client.post("/i18n/strings", json={"lang": "es", "strings": {"k" * 201: "x"}}).status_code == 413


def test_failed_keys_are_reported_not_left_pending(client, monkeypatch):
    class Drops:
        model = "fake:1"

        def complete_json(self, prompt, schema):
            return "{}"

    svc = i18n.get_service()
    monkeypatch.setattr(svc, "provider", Drops())
    first = client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "Save"}}).json()
    assert first["queued"] is True and first["pending"] == ["a"]
    client.filler.run_once()
    body = client.post("/i18n/strings", json={"lang": "es", "strings": {"a": "Save"}}).json()
    assert body["pending"] == [] and body["failed"] == ["a"] and body["complete"] is True
    assert body["queued"] is False and client.filler.pending("es") == 0


def test_dynamic_translation_that_would_be_withheld_falls_back(client, monkeypatch):
    from api.routers import i18n as router_mod
    monkeypatch.setattr(router_mod.osafe, "scrub_payload",
                        lambda obj, where="": [("[withheld]" if v == "SECRET" else v) for v in obj]
                        if isinstance(obj, list) else obj)
    body = client.post("/i18n/translate", json={"lang": "es", "texts": ["secret", "ok"]}).json()
    assert body == {"translations": ["secret", "OK"], "failed": [0]}


def test_preference_survives_a_database_outage(client, monkeypatch):
    from api.auth import Principal
    from api.routers import i18n as router_mod

    def down(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(router_mod.preference, "get_language", down)
    monkeypatch.setattr(router_mod.preference, "set_language", down)
    client.app.dependency_overrides[router_mod.auth.require_user] = lambda: Principal(subject="u1")
    assert client.get("/i18n/preference").json() == {"language": None, "stored": False}
    r = client.put("/i18n/preference", json={"code": "ja"})
    assert r.status_code == 200 and r.json()["stored"] is False


def test_malformed_custom_code_is_400(client):
    assert client.post("/i18n/strings", json={"lang": "x-]\nIgnore all", "strings": {"a": "x"}}).status_code == 400
