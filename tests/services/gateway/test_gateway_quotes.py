"""Sales quotes through the gateway: Sales/Admin only, the existing permits decide, and cost/margin stay behind margin.read."""
import types

import pytest

from src.services.gateway import Backends, CallerContext, build_default_gateway
from src.services.gateway.gateway import FunctionSpec, Gateway
from src.services.sell_side import margin_permit


class NotPermitted(Exception):
    def __init__(self, detail="refused"):
        super().__init__(detail)
        self.detail = detail


class DictDB:
    """A transactional connection whose cursor returns dict rows; rules are (substring, rows)."""

    autocommit = False

    def __init__(self, *rules):
        self.rules, self.executed = list(rules), []

    def cursor(self, cursor_factory=None):
        return _DCur(self)

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def rollback(self):
        pass


class _DCur:
    def __init__(self, db):
        self.db, self._rows = db, []

    def execute(self, sql, params=()):
        self.db.executed.append((sql, params))
        low = " ".join(sql.lower().split())
        self._rows = next((list(r) for needle, r in self.db.rules if needle in low), [])

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


PRINCIPAL = types.SimpleNamespace(subject="sub-1", claims={})
QUOTE = {"sales_quote_id": 7, "quote_ref": "SQ-1", "status": "draft", "total_ex_tax": 100}


def ctx(role="Admin", principal=PRINCIPAL):
    return CallerContext("sub-1", role, None, agent="ask_sales", principal=principal)


@pytest.fixture
def allowed(monkeypatch):
    """Both existing permit checks pass; record what they were asked."""
    seen = {"gate": [], "margin": []}
    import api.endpoint_gate as eg
    monkeypatch.setattr(eg, "require", lambda action, principal, **kw: seen["gate"].append((action, principal, kw)) or types.SimpleNamespace(policy_id=1))
    monkeypatch.setattr(margin_permit, "check", lambda principal, context, **kw: seen["margin"].append((principal, context, kw)))
    return seen


def run(c, name, params, db):
    g = build_default_gateway(Backends(connect=None, extra={"sales_connect": db}))
    return g, g.call(c, name, params)


QUOTE_FNS = {"get_quote": {"quote_id": 7}, "list_quotes": {}, "get_quote_margin": {"quote_id": 7}}


# --- the role rule (versioned config), enforced before anything runs ---------------------------------------------------------------

@pytest.mark.parametrize("name", QUOTE_FNS)
@pytest.mark.parametrize("role", ["Buyer", "Viewer", "Approver", "SecurityReviewer"])
def test_only_sales_and_admin_may_use_the_quote_functions(name, role, allowed):
    db = DictDB()
    _, r = run(ctx(role), name, QUOTE_FNS[name], db)
    assert not r.ok and r.error.code == "NOT_ALLOWED" and db.executed == [] and allowed["gate"] == [] and allowed["margin"] == []


@pytest.mark.parametrize("name", QUOTE_FNS)
@pytest.mark.parametrize("role", ["Sales", "Admin"])
def test_sales_and_admin_pass_the_role_rule(name, role, allowed):
    _, r = run(ctx(role), name, QUOTE_FNS[name], DictDB(("from proc.bp_sales_quote q", [QUOTE]), ("select 1 from proc.bp_sales_quote", [{"?column?": 1}])))
    assert r.error is None or r.error.code != "NOT_ALLOWED"


@pytest.mark.parametrize("name", QUOTE_FNS)
def test_with_no_signed_in_user_nothing_is_read(name, allowed):
    db = DictDB()
    _, r = run(ctx("Admin", principal=None), name, QUOTE_FNS[name], db)
    assert not r.ok and r.error.code == "NOT_ALLOWED" and db.executed == []


def test_the_role_rule_comes_from_config():
    g = build_default_gateway()
    assert all(tuple(g.functions[n].allowed_roles) == ("Sales", "Admin") for n in QUOTE_FNS)
    assert g.functions["get_record"].allowed_roles is None


def test_the_generic_role_check_blocks_before_the_function_body_runs():
    ran = []
    spec = FunctionSpec(name="x", params=type("P", (__import__("pydantic").BaseModel,), {}), fn=lambda c, p, b: ran.append(1), allowed_roles=("Admin",))
    r = Gateway({"x": spec}).call(CallerContext("s", "Buyer", None), "x", {})
    assert r.error.code == "NOT_ALLOWED" and not ran


# --- get_quote: the customer record ------------------------------------------------------------------------------------------------

def test_get_quote_asks_the_existing_quote_read_permit_with_the_signed_in_user(allowed):
    db = DictDB(("from proc.bp_sales_quote q", [QUOTE]), ("from proc.bp_sales_quote_line", []))
    _, r = run(ctx(), "get_quote", {"quote_id": 7}, db)
    assert r.ok and r.data["sales_quote_id"] == 7
    action, principal, kw = allowed["gate"][0]
    assert action == "quote.read" and principal is PRINCIPAL and kw["context"] == {"sales_quote_id": 7}
    assert r.provenance[0]["table"] == "bp_sales_quote" and r.provenance[0]["record_ids"] == ["7"]


def test_a_refusal_by_the_permit_check_is_not_allowed_and_does_not_leak_the_policy(monkeypatch):
    import api.endpoint_gate as eg
    def refuse(*a, **k):
        raise NotPermitted("quote.read refused by rule 12")
    monkeypatch.setattr(eg, "require", refuse)
    monkeypatch.setattr(eg, "NotPermitted", NotPermitted)
    db = DictDB()
    _, r = run(ctx(), "get_quote", {"quote_id": 7}, db)
    assert not r.ok and r.error.code == "NOT_ALLOWED" and db.executed == []
    assert "rule 12" not in r.error.message


def test_an_unknown_quote_is_a_successful_empty_result(allowed):
    _, r = run(ctx(), "get_quote", {"quote_id": 999}, DictDB())
    assert r.ok and r.data == {} and r.provenance[0]["record_ids"] == []


def test_the_customer_record_carries_no_cost_or_margin(allowed):
    db = DictDB(("from proc.bp_sales_quote q", [QUOTE]), ("from proc.bp_sales_quote_line", []))
    _, r = run(ctx(), "get_quote", {"quote_id": 7}, db)
    assert not ({"total_cost", "total_margin", "margin_pct"} & set(r.data))
    assert not any("_internal" in s.lower() or "total_cost" in s.lower() for s, _ in db.executed)


# --- get_quote_margin: only through the one existing permit ------------------------------------------------------------------------

def test_margin_goes_through_the_shared_permit_check_with_the_signed_in_user(allowed):
    db = DictDB(("select 1 from proc.bp_sales_quote", [{"?column?": 1}]),
                ("from proc.bp_sales_quote_internal", [{"total_cost": 40, "total_margin": 16, "margin_pct": 0.28}]))
    _, r = run(ctx(), "get_quote_margin", {"quote_id": 7}, db)
    principal, context, kw = allowed["margin"][0]
    assert principal is PRINCIPAL and context == {"sales_quote_id": 7}
    assert r.ok and "front-end margin only" in r.data["margin_note"] and r.data["total_margin"] == 16


def test_margin_is_withheld_and_never_read_when_the_permit_check_refuses(monkeypatch):
    def refuse(*a, **k):
        raise margin_permit.NotPermitted("no stated permit")
    monkeypatch.setattr(margin_permit, "check", refuse)
    db = DictDB(("select 1 from proc.bp_sales_quote", [{"?column?": 1}]))
    _, r = run(ctx(), "get_quote_margin", {"quote_id": 7}, db)
    assert not r.ok and r.error.code == "NOT_ALLOWED" and db.executed == []


def test_the_router_uses_the_same_shared_check_not_a_copy(monkeypatch):
    from api.routers import sales
    seen = []
    monkeypatch.setattr(margin_permit, "check", lambda principal, context, **kw: seen.append((principal, context)))
    sales._require_margin_permit(PRINCIPAL, {"sales_quote_id": 3})
    assert seen == [(PRINCIPAL, {"sales_quote_id": 3})]
    assert sales.MARGIN_NOTE == margin_permit.MARGIN_NOTE


# --- list_quotes ---------------------------------------------------------------------------------------------------------------------

def test_list_quotes_selects_only_customer_record_columns_and_is_bounded(allowed):
    db = DictDB(("from proc.bp_sales_quote", [QUOTE]))
    _, r = run(ctx(), "list_quotes", {"limit": 5}, db)
    sql, params = db.executed[0]
    assert r.ok and r.data == [QUOTE] and params[-1] == 5
    assert "total_cost" not in sql.lower() and "margin" not in sql.lower()
    assert run(ctx(), "list_quotes", {"limit": 500}, DictDB())[1].error.code == "SCHEMA_INVALID"


def test_list_quotes_rejects_an_unknown_status(allowed):
    assert run(ctx(), "list_quotes", {"status": "'; drop table x"}, DictDB())[1].error.code == "SCHEMA_INVALID"


# --- the principal rides on the context without leaking into logs ------------------------------------------------------------------

def test_the_principal_is_kept_for_the_agent_step_and_is_not_in_the_audit_row_or_repr(allowed):
    c = ctx().for_agent("ask_sales")
    assert c.principal is PRINCIPAL and "sub-1" not in repr(c).replace("principal_subject='sub-1'", "")
    g, r = run(c, "get_quote", {"quote_id": 7}, DictDB(("from proc.bp_sales_quote q", [QUOTE]), ("from proc.bp_sales_quote_line", [])))
    assert "principal" not in g.audit.rows[-1]


def test_the_session_builder_attaches_the_principal(monkeypatch):
    from src.services import rbac
    from src.services.gateway.session import caller_from_principal
    from src.services.report_data import scope
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Admin")
    monkeypatch.setattr(scope, "_read_buyers", lambda s: [])
    assert caller_from_principal(PRINCIPAL).principal is PRINCIPAL


def test_a_quote_with_no_internal_cost_record_is_refused_not_read_as_cost_unknown(allowed):
    db = DictDB(("select 1 from proc.bp_sales_quote", [{"?column?": 1}]))
    _, r = run(ctx(), "get_quote_margin", {"quote_id": 7}, db)
    assert not r.ok and r.error.code == "UNAVAILABLE" and "no internal cost record" in r.error.message
