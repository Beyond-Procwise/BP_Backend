"""Onboarding permission settings: an administrator's choice that is stored as a policy row.

The rule these tests hold: who may do a sensitive action is NOT compiled into the code. It is a
named setting (catalogue), chosen once at onboarding, stored as a ``proc.bp_policy`` row that the
gate already reads, changed only with history, and refused if the role named does not exist.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services import actions
from services import permission_settings as ps

ROLES = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}


class FakePolicyTable:
    """Just enough of proc.bp_policy for the store: rows keyed by policy_name, version-checked."""

    def __init__(self):
        self.rows = {}   # name -> dict(policy_id, version, details)
        self.audit = []
        self.next_id = 1
        self.committed = 0

    # connection protocol ---------------------------------------------------
    def cursor(self):
        return _Cur(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass

    autocommit = False


class _Cur:
    def __init__(self, t):
        self.t, self.rowcount, self._one, self._all = t, 0, None, []

    def execute(self, sql, params=None):
        s = " ".join(sql.split()).lower()
        p = params or ()
        if s.startswith("select") and "from proc.bp_policy" in s:
            names = list(p[0]) if p and isinstance(p[0], (list, tuple)) else list(p)
            self._all = [(r["policy_id"], n, r["version"], r["details"])
                         for n, r in self.t.rows.items() if not names or n in names]
        elif s.startswith("insert into proc.bp_policy"):
            name, _typ, desc, details = p[0], p[1], p[2], p[3]
            self.t.rows[name] = dict(policy_id=self.t.next_id, version=1, details=json.loads(details))
            self.t.next_id += 1
            self.rowcount = 1
        elif s.startswith("update proc.bp_policy"):
            details, _by, pid, ver = p
            for r in self.t.rows.values():
                if r["policy_id"] == pid and r["version"] == ver:
                    r["details"], r["version"] = json.loads(details), ver + 1
                    self.rowcount = 1
                    return
            self.rowcount = 0

    def fetchone(self):
        return self._all[0] if self._all else None

    def fetchall(self):
        return list(self._all)


@pytest.fixture
def table(monkeypatch):
    t = FakePolicyTable()
    monkeypatch.setattr(ps, "_record", lambda conn, **kw: t.audit.append(kw))
    return t


# --- the catalogue ---------------------------------------------------------

def test_every_catalogued_action_is_a_real_action():
    for s in ps.CATALOGUE:
        for a in s.applies_to:
            actions.action_class(a)   # raises on a typo


def test_no_action_is_governed_by_two_settings():
    seen = {}
    for s in ps.CATALOGUE:
        for a in s.applies_to:
            assert a not in seen, f"{a} is in both {seen[a]} and {s.key}"
            seen[a] = s.key


def test_dangerous_actions_default_to_admin_only():
    """D-32/D-33/D-34: bulk renumber, supplier master, activating agents/prompts/policies."""
    for key in ("po_renumber", "supplier_master", "agent_use", "prompt_use", "agent_policy_use"):
        assert ps.setting(key).default_role == "Admin", key


def test_creating_drafts_stays_open_to_buyers():
    """D-34: any user may create; the approval is on USE, not on creation."""
    assert ps.setting("draft_authoring").default_role == "Buyer"
    assert {"prompt.author", "agent_policy.write"} <= set(ps.setting("draft_authoring").applies_to)


def test_unknown_key_is_refused():
    with pytest.raises(ps.UnknownSetting):
        ps.setting("make_everyone_admin")


# --- seeding ---------------------------------------------------------------

def test_seed_writes_one_row_per_setting_with_the_default_role(table):
    out = ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    assert set(out["created"]) == {s.key for s in ps.CATALOGUE} | {c.key for c in ps.CHOICES}
    for s in ps.CATALOGUE:
        d = table.rows[s.policy_name]["details"]
        assert d["required_role"] == s.default_role
        assert d["applies_to"] == list(s.applies_to)
        assert d["rules"] == {"effect": "allow"}
        assert d["policy_identifier"] == s.policy_identifier


def test_seed_is_idempotent_and_never_overwrites_a_choice(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    ps.set_role("supplier_master", "Approver", by="alice", conn=table, roles=ROLES)
    again = ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    assert again["created"] == []
    assert table.rows["SupplierMasterAuthorityPolicy"]["details"]["required_role"] == "Approver"


# --- changing a setting ----------------------------------------------------

def test_set_role_changes_the_row_and_records_who_and_what(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    out = ps.set_role("supplier_master", "Approver", by="alice", conn=table, roles=ROLES)
    row = table.rows["SupplierMasterAuthorityPolicy"]
    assert row["details"]["required_role"] == "Approver" and row["version"] == 2
    assert out["role"] == "Approver" and out["changed"] is True
    h = row["details"]["history"][-1]
    assert (h["changed_by"], h["before"], h["after"], h["version"]) == ("alice", "Admin", "Approver", 2)
    assert any(a["action_type"] == "permission_setting_changed" for a in table.audit)


def test_setting_the_same_role_changes_nothing(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    table.audit.clear()
    out = ps.set_role("supplier_master", "Admin", by="alice", conn=table, roles=ROLES)
    assert out["changed"] is False
    assert table.rows["SupplierMasterAuthorityPolicy"]["version"] == 1
    assert table.audit == []


def test_a_role_that_does_not_exist_is_refused_and_nothing_is_written(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    with pytest.raises(ps.InvalidRole):
        ps.set_role("supplier_master", "Wizard", by="alice", conn=table, roles=ROLES)
    assert table.rows["SupplierMasterAuthorityPolicy"]["version"] == 1


def test_a_stale_save_changes_nothing(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    real = table.cursor

    def racing():
        c = real()
        orig = c.execute

        def execute(sql, params=None):
            if sql.strip().lower().startswith("update"):
                table.rows["SupplierMasterAuthorityPolicy"]["version"] += 1   # somebody else saved first
            return orig(sql, params)
        c.execute = execute
        return c
    table.cursor = racing
    with pytest.raises(ps.SettingConflict):
        ps.set_role("supplier_master", "Buyer", by="alice", conn=table, roles=ROLES)


def test_changing_a_setting_before_it_is_seeded_creates_it(table):
    out = ps.set_role("po_renumber", "Approver", by="alice", conn=table, roles=ROLES)
    assert table.rows["PoRenumberAuthorityPolicy"]["details"]["required_role"] == "Approver"
    assert out["changed"] is True


# --- reading ---------------------------------------------------------------

def test_current_reports_each_setting_and_flags_the_unseeded_ones(table):
    ps.set_role("supplier_master", "Buyer", by="alice", conn=table, roles=ROLES)
    got = {r["key"]: r for r in ps.current(conn=table)}
    assert got["supplier_master"]["role"] == "Buyer" and got["supplier_master"]["configured"] is True
    assert got["po_renumber"]["configured"] is False
    assert got["po_renumber"]["role"] == ps.setting("po_renumber").default_role
    assert got["po_renumber"]["label"] and got["po_renumber"]["description"]


# --- choice settings (workflow_question_answer, D-60) ----------------------

def test_choice_listed_with_default_when_no_row(table):
    row = next(r for r in ps.current(table) if r["key"] == "workflow_question_answer")
    assert row["kind"] == "choice" and row["value"] == "approver_only" and not row["configured"]
    assert row["choices"] == ["approver_only", "starter_or_approver", "any_signed_in"]
    assert ps.choice_value("workflow_question_answer", table) == "approver_only"


def test_seed_writes_the_choice_row_with_the_default(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    d = table.rows["WorkflowQuestionAnswerPolicy"]["details"]
    assert d["rules"] == {"mode": "approver_only"} and "applies_to" not in d


def test_set_choice_changes_records_history_and_refuses_unknown_values(table):
    ps.seed_defaults(by="onboarding", conn=table, roles=ROLES)
    out = ps.set_choice("workflow_question_answer", "any_signed_in", by="alice", conn=table)
    assert out["changed"] and out["version"] == 2
    d = table.rows["WorkflowQuestionAnswerPolicy"]["details"]
    assert d["rules"]["mode"] == "any_signed_in"
    assert d["history"][-1]["before"] == "approver_only" and d["history"][-1]["changed_by"] == "alice"
    assert ps.set_choice("workflow_question_answer", "any_signed_in", by="alice", conn=table)["changed"] is False
    with pytest.raises(ps.InvalidChoice):
        ps.set_choice("workflow_question_answer", "everyone", by="alice", conn=table)
    with pytest.raises(ps.UnknownSetting):
        ps.set_choice("agent_use", "approver_only", by="alice", conn=table)
    assert ps.choice_value("workflow_question_answer", table) == "any_signed_in"
