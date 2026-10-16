from services import mail_providers
from services.mail_m365 import sending
from services.mail_providers.m365 import M365Provider


def test_get_provider_none_when_never_connected(monkeypatch):
    monkeypatch.setattr(mail_providers, "read_integration", lambda conn: {})
    assert mail_providers.get_provider(object()) is None


def test_get_provider_m365_when_connected(monkeypatch):
    monkeypatch.setattr(mail_providers, "read_integration", lambda conn: {"provider": "m365"})
    p = mail_providers.get_provider(object())
    assert isinstance(p, M365Provider)
    assert (p.name, p.label) == ("m365", "Microsoft 365")


def test_m365_provider_wraps_existing_code(monkeypatch):
    p = M365Provider()
    assert p.size_limit() == sending.MAX_MIME_BYTES == 3_000_000
    seen = []
    monkeypatch.setattr(sending, "send_via_graph", lambda b, m: seen.append((b, m)))
    p.send("binding", b"mime")
    assert seen == [("binding", b"mime")]

    # never connected: one row that proves nothing, never an exception
    from services.mail_providers import m365 as m365_mod
    monkeypatch.setattr(m365_mod, "read_integration", lambda conn: {})
    rows = p.prove(object())
    assert [(x["restriction"], x["result"]) for x in rows] == [("no_mail_permissions", "unknown")]
