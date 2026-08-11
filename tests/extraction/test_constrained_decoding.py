"""Schema-constrained decoding: schema builder + format passthrough."""
import src.services.ollama_client as OC
from src.services import egress
from src.services.extraction.context_layer import _build_schema


def test_build_schema_structure_only():
    s = _build_schema([("quote_id", "str", "d"), ("total_amount", "money", "d")])
    assert s["type"] == "object"
    assert set(s["required"]) == {"quote_id", "total_amount"}
    assert s["additionalProperties"] is False
    # permissive value types (structure constrained, value format is not)
    assert "null" in s["properties"]["quote_id"]["type"]
    assert "number" in s["properties"]["total_amount"]["type"]


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "{}"}


def test_ollama_generate_forwards_format(monkeypatch):
    captured = {}

    def fake_post(url, *, purpose, json=None, timeout=None, **kw):
        captured.update(json or {})
        return _FakeResp()

    # Patched at the egress layer: ollama_client no longer holds an HTTP client
    # of its own (see tests/test_egress_import_boundary.py). What this test is
    # about — that `format` reaches the wire — is unchanged by where the
    # transport lives, so the assertion is the same one on the same payload.
    monkeypatch.setattr(egress, "post", fake_post)
    OC.ollama_generate("hi", format={"type": "object"})
    assert captured.get("format") == {"type": "object"}


def test_ollama_generate_omits_format_when_none(monkeypatch):
    captured = {}

    def fake_post(url, *, purpose, json=None, timeout=None, **kw):
        captured.update(json or {})
        return _FakeResp()

    # Patched at the egress layer: ollama_client no longer holds an HTTP client
    # of its own (see tests/test_egress_import_boundary.py). What this test is
    # about — that `format` reaches the wire — is unchanged by where the
    # transport lives, so the assertion is the same one on the same payload.
    monkeypatch.setattr(egress, "post", fake_post)
    OC.ollama_generate("hi")
    assert "format" not in captured
