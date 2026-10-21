"""Deal visibility: an administrator's setting, narrow by default, stored as a policy row.

D-36: reading and writing are separate; the default is only the deals a person created or approved;
anything unknown or unreadable is the narrow scope, never the wide one.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services import deal_visibility as dv
from services import permission_settings as ps
from tests.services.test_permission_settings import FakePolicyTable

ROLES = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}
ROW = dv.POLICY_NAME


@pytest.fixture
def table(monkeypatch):
    t = FakePolicyTable()
    monkeypatch.setattr(dv, "_record", lambda conn, **kw: t.audit.append(kw))
    return t


def roles_of(t):
    return t.rows[ROW]["details"]["rules"]["roles"]


# --- defaults -------------------------------------------------------------------
def test_seed_makes_every_non_admin_role_see_only_what_they_created_or_approved(table):
    assert dv.seed_defaults("onboarding", conn=table, roles=ROLES) == {"created": True}
    r = roles_of(table)
    for role in ("Viewer", "Buyer", "Approver"):
        assert r[role] == {"read": "mine", "write": "mine"}, role
    assert r["Admin"] == {"read": "all", "write": "all"}


def test_seed_never_overwrites_a_choice(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    dv.set_scope("Buyer", "alice", read="all", conn=table, roles=ROLES)
    assert dv.seed_defaults("onboarding", conn=table, roles=ROLES) == {"created": False}
    assert roles_of(table)["Buyer"]["read"] == "all"


def test_current_with_no_row_reports_the_narrow_default_and_not_configured(table):
    got = {r["role"]: r for r in dv.current(conn=table, roles=ROLES)}
    assert got["Buyer"]["read"] == "mine" and got["Buyer"]["configured"] is False
    assert got["Admin"]["read"] == "all"


# --- changing it ----------------------------------------------------------------
def test_read_and_write_are_separate(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    out = dv.set_scope("Approver", "alice", read="all", conn=table, roles=ROLES)
    assert out["changed"] and (out["read"], out["write"]) == ("all", "mine")
    assert roles_of(table)["Approver"] == {"read": "all", "write": "mine"}


def test_a_change_records_who_what_and_the_history(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    dv.set_scope("Buyer", "alice", write="none", conn=table, roles=ROLES)
    row = table.rows[ROW]
    h = row["details"]["history"][-1]
    assert (h["changed_by"], h["field"], h["before"], h["after"], h["version"]) == \
        ("alice", "Buyer.write", "mine", "none", 2)
    a = table.audit[-1]
    assert a["action_type"] == "deal_visibility_changed" and a["details"]["after"]["write"] == "none"


def test_saving_the_same_value_writes_and_audits_nothing(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    table.audit.clear()
    out = dv.set_scope("Buyer", "alice", read="mine", conn=table, roles=ROLES)
    assert out["changed"] is False and table.rows[ROW]["version"] == 1 and table.audit == []


def test_unknown_role_and_unknown_scope_are_refused(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    with pytest.raises(ps.InvalidRole):
        dv.set_scope("Wizard", "alice", read="all", conn=table, roles=ROLES)
    with pytest.raises(dv.InvalidScope):
        dv.set_scope("Buyer", "alice", read="everyone", conn=table, roles=ROLES)
    with pytest.raises(dv.InvalidScope):
        dv.set_scope("Buyer", "alice", read="none", conn=table, roles=ROLES)   # none is write-only
    with pytest.raises(dv.InvalidScope):
        dv.set_scope("Buyer", "alice", conn=table, roles=ROLES)
    assert table.rows[ROW]["version"] == 1


def test_admin_cannot_be_narrowed(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    with pytest.raises(dv.AdminCannotBeNarrowed):
        dv.set_scope("Admin", "alice", read="mine", conn=table, roles=ROLES)
    assert roles_of(table)["Admin"] == {"read": "all", "write": "all"}


def test_a_stale_save_changes_nothing(table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    real = table.cursor

    def racing():
        c = real()
        orig = c.execute

        def execute(sql, params=None):
            if sql.strip().lower().startswith("update"):
                table.rows[ROW]["version"] += 1
            return orig(sql, params)
        c.execute = execute
        return c
    table.cursor = racing
    with pytest.raises(ps.SettingConflict):
        dv.set_scope("Buyer", "alice", read="all", conn=table, roles=ROLES)


def test_changing_before_seeding_creates_the_row_with_the_other_roles_at_default(table):
    dv.set_scope("Buyer", "alice", read="all", conn=table, roles=ROLES)
    r = roles_of(table)
    assert r["Buyer"]["read"] == "all" and r["Viewer"] == {"read": "mine", "write": "mine"}


# --- which scope a person acts with ---------------------------------------------
def _resolve(monkeypatch, table, role):
    from services import rbac
    monkeypatch.setattr(rbac, "effective_role", lambda p: role)
    return dv.resolve(SimpleNamespace(subject="u1"), conn=table)


def test_resolve_reads_the_stored_scope(monkeypatch, table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    dv.set_scope("Approver", "alice", read="all", conn=table, roles=ROLES)
    v = _resolve(monkeypatch, table, "Approver")
    assert (v.read, v.write) == ("all", "mine")


def test_resolve_with_no_row_is_narrow_for_everyone_but_admin(monkeypatch, table):
    assert (_resolve(monkeypatch, table, "Buyer").read, _resolve(monkeypatch, table, "Buyer").write) == ("mine", "mine")
    assert _resolve(monkeypatch, table, "Admin").read == "all"


def test_resolve_with_a_garbled_row_is_narrow(monkeypatch, table):
    dv.seed_defaults("onboarding", conn=table, roles=ROLES)
    table.rows[ROW]["details"]["rules"]["roles"]["Buyer"] = {"read": "EVERYTHING", "write": 7}
    v = _resolve(monkeypatch, table, "Buyer")
    assert (v.read, v.write) == ("mine", "mine")


def test_resolve_when_the_database_fails_is_narrow(monkeypatch):
    from services import rbac
    monkeypatch.setattr(rbac, "effective_role", lambda p: "Buyer")

    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(dv, "get_conn", boom)
    v = dv.resolve(SimpleNamespace(subject="u1"))
    assert (v.read, v.write) == ("mine", "mine")


# --- checking one deal ----------------------------------------------------------
def test_all_allows_any_deal():
    assert dv.allows("all", "u1") is True


def test_mine_allows_a_creator_or_an_approver_only():
    assert dv.allows("mine", "u1", created_by=["u1"])
    assert dv.allows("mine", "u1", created_by=["u2"], approved_by=["u1"])
    assert not dv.allows("mine", "u1", created_by=["u2"], approved_by=["u3"])


def test_nobody_named_or_an_unknown_scope_is_refused():
    assert not dv.allows("mine", None, created_by=["u1"])
    assert not dv.allows("none", "u1", created_by=["u1"])
    assert not dv.allows("", "u1", created_by=["u1"])
