"""The caller's reseller accounts come from the user->reseller link table, by the same rules as buyer scope."""
import types

import pytest

from src.services.gateway import CallerContext
from src.services.gateway import session


def principal(sub="sub-5"):
    return types.SimpleNamespace(subject=sub, email=None, username=sub, claims={})


@pytest.fixture
def roles(monkeypatch):
    from src.services import rbac
    from src.services.report_data import scope

    def set_role(role, buyers=()):
        monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: role)
        monkeypatch.setattr(scope, "_read_buyers", lambda s: list(buyers))
    return set_role


def test_an_admin_is_unrestricted_for_resellers_too(roles, monkeypatch):
    roles("Admin")
    monkeypatch.setattr(session, "_read_reseller_accounts", lambda s: (_ for _ in ()).throw(AssertionError("admin needs no link rows")))
    assert session.caller_from_principal(principal()).reseller_account_ids is None


def test_a_user_gets_exactly_their_linked_reseller_accounts(roles, monkeypatch):
    roles("Buyer")
    monkeypatch.setattr(session, "_read_reseller_accounts", lambda s: ["R1", "R2"] if s == "sub-5" else [])
    assert session.caller_from_principal(principal()).reseller_account_ids == ("R1", "R2")


def test_no_link_rows_means_no_reseller_access(roles, monkeypatch):
    roles("Buyer")
    monkeypatch.setattr(session, "_read_reseller_accounts", lambda s: [])
    assert session.caller_from_principal(principal()).reseller_account_ids == ()


def test_an_unreadable_link_table_sees_nothing_it_never_widens(roles, monkeypatch):
    roles("Buyer")
    monkeypatch.setattr(session, "_read_reseller_accounts", lambda s: (_ for _ in ()).throw(RuntimeError("relation does not exist")))
    assert session.caller_from_principal(principal()).reseller_account_ids == ()


def test_no_principal_has_no_reseller_accounts(roles):
    roles("Buyer")
    assert session.caller_from_principal(None).reseller_account_ids == ()


def test_the_context_carries_reseller_scope_through_for_agent():
    c = CallerContext("s", "Buyer", ("B1",), reseller_account_ids=("R1",)).for_agent("ask_sales")
    assert c.reseller_account_ids == ("R1",)
    assert CallerContext("s", "Admin", None, reseller_account_ids=None).for_agent("x").reseller_account_ids is None


def test_the_migration_follows_the_existing_scope_table_pattern_and_has_a_rollback():
    sql = open("deploy/sql/2026-10-19_user_reseller_link.sql").read().lower()
    for needle in ("create table if not exists proc.bp_user_reseller_link", "subject", "reseller_account_id", "granted_by",
                   "revoked_at", "create unique index if not exists ix_bp_user_reseller_link_active", "where revoked_at is null"):
        assert needle in sql, needle
    assert "drop table if exists proc.bp_user_reseller_link" in open("deploy/sql/2026-10-19_user_reseller_link_rollback.sql").read().lower()
