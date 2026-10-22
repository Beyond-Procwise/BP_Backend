"""Five more writes ask the gate first (audit RT-17b): each runs a computation on what the caller sends.

demand-intake extract runs the model on supplied text (``document.extract``, compute: Buyer and above);
the benchmark preview runs the pricing engine on a supplied payload (``quote.compare``, compute);
the on-demand translation route is asked as the new ``content.translate`` action, a READ-class action so
every signed-in role keeps its translations and an Admin can still narrow or audit them by policy.
Resolving a language name and the UI-strings route stay ungated: the strings route is polled every two
seconds while a language fills, a read-class gate there would add two audit rows per poll and protect
nothing (every role may read), and a failed audit insert would turn translations into a 500. Saving one's
own language preference stays ungated: it is the caller's own setting. A refusal runs nothing.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.benchmark as bm
import api.routers.demand_intake as di
import api.routers.i18n as i1
from api.endpoint_gate import NotPermitted
from src.services import actions


def test_translation_is_a_read_class_action():
    assert actions.action_class("content.translate") == "read"


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    for mod in (bm, di, i1):
        monkeypatch.setattr(mod, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(di, "extract_fields", lambda nick, context: did.append("extract") or {"fields": {}})
    monkeypatch.setattr(bm, "ensure_registered", lambda: None)
    monkeypatch.setattr(bm, "evaluate", lambda name, inputs: did.append("benchmark") or SimpleNamespace(
        unassessed=False, value=SimpleNamespace(model_dump=lambda: {"adjusted_price": 1}), trace=[], assessed=True))
    reg = SimpleNamespace(resolve=lambda text: did.append("resolve") or SimpleNamespace(to_dict=lambda: {"code": "fr"}))
    svc = SimpleNamespace(status=lambda code, strings: did.append("strings") or ({}, [], []),
                          translate=lambda code, keys, lang_name=None: did.append("translate") or SimpleNamespace(translations={k: "x" for k in keys}, failed=[], supported=True, confidence=1.0),
                          quality=lambda code, name=None: {"experimental": False},
                          provider=SimpleNamespace(model="stub"), prompt_version="v0")
    monkeypatch.setattr(i1.audit, "record_served", lambda **kw: None)
    monkeypatch.setattr(i1.i18n, "get_registry", lambda: reg)
    monkeypatch.setattr(i1.i18n, "get_service", lambda: svc)
    monkeypatch.setattr(i1, "_language", lambda code, name: SimpleNamespace(code="fr", dir="ltr", custom=False, label=lambda: "French"), raising=False)
    app = FastAPI()
    for mod in (bm, di, i1):
        app.include_router(mod.router)
    for mod in (bm, di, i1):
        app.dependency_overrides[mod.require_user] = lambda: SimpleNamespace(subject="alice")
    app.dependency_overrides[di.get_agent_nick] = lambda: object()
    return TestClient(app, raise_server_exceptions=False), asked, did, monkeypatch


def _minimal_quote_line() -> dict:
    """A quote line that satisfies the request model's REQUIRED fields, derived from the model itself so
    the test does not drift when a field is added; values are the smallest valid ones."""
    from src.services.benchmark.models import QuoteLine
    out = {}
    for name, field in QuoteLine.model_fields.items():
        if not field.is_required():
            continue
        ann = str(field.annotation)
        out[name] = 5.0 if "float" in ann else (1 if "int" in ann else (False if "bool" in ann else "x"))
    return out


CALLS = [
    ("post", "/demand/intake/extract", {"context": {"text": "need 10 laptops"}}, "document.extract", "extract"),
    ("post", "/benchmark/preview", {"quote": _minimal_quote_line(), "points": []}, "quote.compare", "benchmark"),
    ("post", "/i18n/translate", {"lang": "fr", "texts": ["hello"]}, "content.translate", "translate"),
]


@pytest.mark.parametrize("method,path,body,action,work", CALLS)
def test_each_route_asks_the_gate_first(rig, method, path, body, action, work):
    http, asked, did, _ = rig
    res = getattr(http, method)(path, json=body)
    assert res.status_code == 200, (path, res.status_code, res.text[:200])
    assert asked[:1] == [action], (path, asked)
    assert work in did, (path, did)


@pytest.mark.parametrize("method,path,body,action,work", CALLS)
def test_a_refusal_runs_nothing(rig, method, path, body, action, work):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    for mod in (bm, di, i1):
        mp.setattr(mod, "gate", _no, raising=False)
    assert getattr(http, method)(path, json=body).status_code == 403
    assert did == []


@pytest.mark.parametrize("path,body,work", [
    ("/i18n/languages/resolve", {"text": "French"}, "resolve"),
    ("/i18n/strings", {"lang": "fr", "strings": {"k": "Save"}}, "strings"),
])
def test_the_polled_translation_reads_do_not_ask_the_gate(rig, path, body, work):
    """Two audit rows per two-second poll, for a read every role may make, is noise not protection."""
    http, asked, did, _ = rig
    assert http.post(path, json=body).status_code == 200
    assert asked == [] and work in did
