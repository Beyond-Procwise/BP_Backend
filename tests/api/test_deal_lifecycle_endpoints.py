import importlib
mod = importlib.import_module("src.api.routers.deal_summary")

def test_promote_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(mod, "promote_deal", lambda deal_id: called.setdefault("promote", deal_id))
    res = mod.post_promote_deal("ACME2026071501")
    assert res["status"] == "ok"
    assert res["is_tracked"] is True
    assert called["promote"] == "ACME2026071501"

def test_save_reference_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(mod, "save_reference", lambda deal_id: called.setdefault("save", deal_id))
    res = mod.post_save_reference("ACME2026071501")
    assert res["status"] == "ok"
    assert res["is_saved_reference"] is True
    assert called["save"] == "ACME2026071501"
