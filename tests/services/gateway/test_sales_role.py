"""The Sales role (approved 2026-10-19): rank 1, read and compute only, mapped from the bp-sales group.

Least privilege on purpose. Policies that name a required role deny anyone BELOW that rank, so a rank-2 Sales role would pass the
Buyer-level permits (supplier master, document upload, email send, web research); and any class beyond read/compute would let a
sales user write contracts, deals and findings, which no per-action policy blocks. Margin therefore stays withheld from Sales
until a deliberate decision changes SalesMarginReadAuthorityPolicy.
"""
import os
import re
import types

import pytest

SQL = open("deploy/sql/2026-10-19_sales_role.sql").read()
ROLLBACK = open("deploy/sql/2026-10-19_sales_role_rollback.sql").read()


def test_the_migration_adds_one_role_and_one_group_idempotently():
    low = SQL.lower()
    assert "'{rules,roles}'" in low and '"sales"' in low and '"rank": 1' in low and '"allow": ["read", "compute"]' in low
    assert '"bp-sales": "sales"' in low
    assert "policy_identifier' = 'role_definition'" in low and "policy_identifier' = 'role_assignment'" in low
    assert low.count("version = version + 1") == 2
    assert "not (policy_details #> '{rules,roles}') ? 'sales'" in low.replace("\n", " ") or "? 'sales'" in low   # guarded, so a re-run changes nothing


def test_the_migration_grants_nothing_beyond_read_and_compute():
    allow = re.search(r'"allow":\s*\[([^\]]*)\]', SQL).group(1)
    assert {w.strip(' "') for w in allow.split(",")} == {"read", "compute"}


def test_the_rollback_removes_exactly_what_was_added():
    low = ROLLBACK.lower()
    assert "#- '{rules,roles,sales}'" in low and "#- '{rules,group_to_role,bp-sales}'" in low


_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1 and the migration applied to bp_testdb")


def P(group):
    return types.SimpleNamespace(subject="sales-role-probe", email=None, username="x", claims={"cognito:groups": [group]})


@live
def test_a_bp_sales_user_resolves_to_sales_with_rank_one_and_only_read_and_compute():
    from src.services import rbac
    rbac.reset_caches() if hasattr(rbac, "reset_caches") else None
    assert rbac.effective_role(P("bp-sales")) == "Sales"
    assert rbac.role_rank("Sales") == 1
    assert rbac.may("Sales", "read") and rbac.may("Sales", "compute")
    for denied in ("write", "communicate", "transact", "share", "configure", "delegate", "approve_email", "inspect"):
        assert not rbac.may("Sales", denied), denied


@live
def test_existing_roles_are_unchanged_by_the_new_one():
    from src.services import rbac
    assert {g: rbac.effective_role(P(g)) for g in ("bp-admins", "bp-buyers", "bp-viewers", "bp-approvers")} == {
        "bp-admins": "Admin", "bp-buyers": "Buyer", "bp-viewers": "Viewer", "bp-approvers": "Approver"}
    assert [rbac.role_rank(r) for r in ("Viewer", "Buyer", "Approver", "Admin")] == [1, 2, 3, 4]


@live
def test_a_user_in_sales_and_buyers_resolves_to_the_higher_rank_not_to_sales():
    from src.services import rbac
    both = types.SimpleNamespace(subject="x", email=None, username="x", claims={"cognito:groups": ["bp-sales", "bp-buyers"]})
    assert rbac.effective_role(both) == "Buyer"


@live
def test_sales_passes_the_gateway_role_rule_for_quotes_but_margin_stays_withheld():
    from src.services.db import get_conn
    from src.services.gateway import Backends, build_default_gateway
    from src.services.gateway.session import caller_from_principal
    g = build_default_gateway(Backends(connect=get_conn))
    c = caller_from_principal(P("bp-sales")).for_agent("ask_sales")
    assert c.role == "Sales"
    r = g.call(c, "list_quotes", {"limit": 2})
    assert r.ok, r.error
    m = g.call(c, "get_quote_margin", {"quote_id": r.data[0]["sales_quote_id"]})
    assert not m.ok and m.error.code == "NOT_ALLOWED"                   # margin.read requires Buyer rank; Sales is rank 1


@live
def test_sales_cannot_use_the_procurement_only_write_paths():
    """The Buyer-level authority permits still refuse Sales (rank 1 < Buyer's 2), proving the role does not widen them."""
    from src.services import guardrail, rbac
    for action, cls in (("supplier.write", "write"), ("document.upload", "write"), ("research.web", "share")):
        d = guardrail.authorize(action, cls, P("bp-sales"), {}, policy_engine=rbac.policy_engine())
        assert not d.allowed, action
