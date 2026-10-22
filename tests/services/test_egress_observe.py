"""1.20g: a call made by a library that owns its own HTTP client is still seen by the egress gate.

``web_search`` reaches DuckDuckGo through the ``ddgs`` library, which brings its own transport, so the
chokepoint never saw those calls: the query left unrecorded and the shadow could not judge it.
``egress.observe`` wraps such a call: the same decision, the same event row (purpose, destination,
agent, payload digest, shadow), outcome ``library_call`` or ``library_error``. Still shadow: it refuses
nothing until enforcement (step d) says so.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.services import egress, egress_log
from src.services import egress_policy as ep
from src.services.egress import Purpose


@pytest.fixture
def events(monkeypatch):
    out = []
    monkeypatch.setattr(egress_log, "record", lambda **kw: out.append(kw))
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(ep, "_load_row", lambda: None)
    ep.invalidate()
    yield out
    ep.invalidate()


def test_an_observed_library_call_is_recorded_like_a_request(events):
    with egress.observe(Purpose.SUPPLIER_RESEARCH, "duckduckgo.com", agent="supplier_research",
                        payload="Borcelle Ltd supplier profile"):
        pass
    e = events[-1]
    assert (e["purpose"], e["destination"], e["agent"], e["outcome"]) == (
        "supplier_research", "duckduckgo.com", "supplier_research", "library_call")
    assert e["shadow_decision"] == "deny" and e["policy_version"] == egress.SHADOW_RULES_VERSION
    assert e["payload_sha256"] and e["payload_bytes"] == len("Borcelle Ltd supplier profile")
    assert e["method"] == "LIBRARY"


def test_an_error_inside_is_recorded_and_re_raised(events):
    with pytest.raises(ValueError):
        with egress.observe(Purpose.SUPPLIER_RESEARCH, "duckduckgo.com", agent="x"):
            raise ValueError("provider down")
    assert events[-1]["outcome"] == "library_error" and events[-1]["detail"] == "ValueError"


def test_shadow_never_blocks_the_observed_call(events):
    ran = []
    with egress.observe(Purpose.MODEL_INFERENCE, "api.ollama.com", agent="neg", payload={"p": 1}):
        ran.append(1)
    assert ran == [1] and events[-1]["shadow_decision"] == "deny"


def test_web_search_is_observed(events, monkeypatch):
    seen = {}

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5, backend="auto"):
            seen["query"], seen["backend"] = query, backend
            return [{"title": "Borcelle", "href": "https://borcelle.example/", "body": "widgets"}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    from src.services.supplier_enrichment import web_tools
    out = web_tools.web_search("Borcelle Ltd", max_results=3)
    assert out and out[0]["url"] == "https://borcelle.example/" and seen["query"] == "Borcelle Ltd"
    assert seen["backend"] == "duckduckgo", "unpinned, the library fans out to eight providers and the label lies"
    assert web_tools.SEARCH_HOST == "html.duckduckgo.com"
    e = events[-1]
    assert (e["purpose"], e["destination"], e["agent"], e["outcome"]) == (
        "supplier_research", web_tools.SEARCH_HOST, "supplier_research", "library_call")
    assert e["payload_bytes"] == len("Borcelle Ltd")


def test_web_search_still_fails_open_and_records_the_error(events, monkeypatch):
    class Broken:
        def __enter__(self):
            raise RuntimeError("ratelimit")

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=Broken))
    from src.services.supplier_enrichment import web_tools
    assert web_tools.web_search("Borcelle") == []
    assert events[-1]["outcome"] == "library_error"


def test_an_enforced_refusal_is_no_results_not_a_crash(events, monkeypatch):
    from src.services.supplier_enrichment import web_tools
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=object))

    def refuse(*a, **k):
        raise egress.EgressDenied(reason="policy", destination="html.duckduckgo.com", purpose="supplier_research")
    monkeypatch.setattr(web_tools, "egress_observe", refuse)
    assert web_tools.web_search("Borcelle") == []


def test_the_boundary_lint_knows_the_search_library():
    import importlib
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    b = importlib.import_module("test_egress_import_boundary")
    assert "ddgs" in b._NETWORK_MODULES
    assert "src/services/supplier_enrichment/web_tools.py" in b._KNOWN
