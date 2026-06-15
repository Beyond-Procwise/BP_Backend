"""Regression: POST /summary sent as_of="" (empty string), which is not a valid
timestamp and crashed the historical-snapshot query. Empty/blank as_of must be
treated as "now" (current generation)."""
from __future__ import annotations

import pytest

import src.services.summary_agent as sa


class _Conn:
    def cursor(self):
        class _C:
            def execute(self, *a, **k):
                raise AssertionError("snapshot query must not run for blank as_of")
            def fetchone(self):
                return None
        return _C()


@pytest.mark.parametrize("blank", ["", "   ", None, "null", "undefined", "NULL", "none", "not-a-date"])
def test_blank_as_of_uses_current_generation(monkeypatch, blank):
    calls = {"gather": 0}
    monkeypatch.setattr(sa, "resolve_persona", lambda p, c: ("framing", "raw"))
    monkeypatch.setattr(sa, "gather_deal_context",
                        lambda deal_id, conn=None: calls.__setitem__("gather", calls["gather"] + 1) or {"sources": {}})
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "a summary")
    monkeypatch.setattr(sa, "_store_summary", lambda *a, **k: {"summary": "a summary"})

    out = sa.generate_summary("analysis", deal_id="D1", as_of=blank, conn=_Conn())
    assert out == {"summary": "a summary"}
    assert calls["gather"] == 1   # current-generation path, not the snapshot branch


def test_real_as_of_uses_snapshot_branch(monkeypatch):
    monkeypatch.setattr(sa, "resolve_persona", lambda p, c: ("framing", "raw"))
    monkeypatch.setattr(sa, "ollama_cloud_generate", lambda *a, **k: "a summary")
    monkeypatch.setattr(sa, "_store_summary", lambda *a, **k: {"summary": "a summary"})

    seen = {"snapshot_sql": False}

    class _SnapConn:
        def cursor(self):
            class _C:
                def execute(self, sql, params=()):
                    seen["snapshot_sql"] = "generated_at <=" in sql
                def fetchone(self):
                    return ({"sources": {}},)
            return _C()

    out = sa.generate_summary("analysis", deal_id="D1", as_of="2026-06-15T00:00:00Z", conn=_SnapConn())
    assert out == {"summary": "a summary"}
    assert seen["snapshot_sql"] is True   # historical snapshot path taken


def test_cloud_generate_retries_on_empty_response(monkeypatch):
    """An empty `response` (model glitch) must be retried, not returned as ''."""
    import src.services.ollama_client as oc

    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "x")
    monkeypatch.setattr(oc.time, "sleep", lambda *_: None)
    bodies = iter([{"response": ""}, {"response": "real summary"}])

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return next(bodies)

    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: _Resp())
    out = oc.ollama_cloud_generate("prompt", model="qwen3.5:397b", retries=3)
    assert out == "real summary"   # retried past the empty first response
