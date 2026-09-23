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


class _Principal:
    subject = "user-123"


def test_save_endpoint_names_and_confirms_as_the_signed_in_user(monkeypatch):
    called = {}
    monkeypatch.setattr(mod, "save_deal", lambda deal_id, name, *, actor: called.update(
        deal_id=deal_id, name=name, actor=actor))
    res = mod.post_save_deal("ACME2026071501", mod.DealSaveIn(name="Laptops Q3"), _Principal())
    assert res == {"status": "ok", "deal_id": "ACME2026071501", "name": "Laptops Q3",
                   "is_tracked": True}
    assert called == {"deal_id": "ACME2026071501", "name": "Laptops Q3", "actor": "user-123"}


def test_save_endpoint_refuses_a_blank_name_with_400(monkeypatch):
    import pytest
    from fastapi import HTTPException

    def boom(*a, **k):
        raise ValueError("a deal needs a name")
    monkeypatch.setattr(mod, "save_deal", boom)
    with pytest.raises(HTTPException) as e:
        mod.post_save_deal("ACME2026071501", mod.DealSaveIn(name=" "), _Principal())
    assert e.value.status_code == 400


def test_save_endpoint_404s_a_deal_with_no_documents(monkeypatch):
    import pytest
    from fastapi import HTTPException

    def boom(*a, **k):
        raise LookupError("no documents")
    monkeypatch.setattr(mod, "save_deal", boom)
    with pytest.raises(HTTPException) as e:
        mod.post_save_deal("NOPE", mod.DealSaveIn(name="x"), _Principal())
    assert e.value.status_code == 404
