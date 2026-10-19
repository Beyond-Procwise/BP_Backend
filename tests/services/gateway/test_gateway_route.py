"""The /ask routes hand the pipeline a session-built caller only when ASK_GATEWAY is on."""
import types

from src.api.routers import workflows
from src.services import ask_flags


def test_no_caller_is_passed_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "0")
    assert workflows._gateway_caller(types.SimpleNamespace(subject="s"), "sess") == {}


def test_the_caller_is_built_from_the_principal_when_the_flag_is_on(monkeypatch):
    from src.services import rbac
    from src.services.report_data import scope
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Buyer")
    monkeypatch.setattr(scope, "_read_buyers", lambda s: ["B7"])
    out = workflows._gateway_caller(types.SimpleNamespace(subject="sub-7", claims={}), "sess-1")
    c = out["caller"]
    assert c.principal_subject == "sub-7" and c.buyer_scope == ("B7",) and c.session_id == "sess-1"
