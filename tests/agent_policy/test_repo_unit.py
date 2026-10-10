"""Repository paths that need no database: refusals that happen before anything is read."""
import pytest

from repositories import agent_policy_repo as repo
from services import output_safety as osafe


class _Cur:
    def __init__(self, rows):
        self.rows = list(rows)
        self.connection = None
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchone(self):
        return self.rows.pop(0)


class _Conn:
    def __init__(self, rows=()):
        self.cur = _Cur(rows)
        self.autocommit = True
        self.rolled_back = self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


WITHHELD = [{"field": "form", "code": "withheld_text",
             "message": "Some text could not be shown, so nothing was saved. Reload the policy and try again."}]


@pytest.mark.parametrize("form", [
    {"name": osafe.SAFE_FIELD},
    {"name": "ok", "source": {"excerpt": osafe.SAFE_FIELD}},
    {"name": "ok", "deciders": ["CFO", osafe.SAFE_REPLY]},
    {"name": "ok", "hidden": {"condition": {"all": [{"field": "agent.reason", "op": "eq", "value": osafe.SAFE_FIELD}]}}},
])
def test_create_and_save_refuse_a_form_holding_withheld_text(form):
    conn = _Conn()
    with pytest.raises(repo.NotReady) as err:
        repo.create_draft(conn, form, actor="t")
    assert err.value.problems == WITHHELD and conn.cur.sql == []   # nothing read, nothing written
    with pytest.raises(repo.NotReady) as err:
        repo.save_version(conn, "GEN-0001", form, base_version=1, intent="draft", actor="t", change_note="")
    assert err.value.problems == WITHHELD and conn.cur.sql == []


def test_text_merely_containing_the_marker_is_not_refused():
    assert not repo._holds_withheld_text({"name": "A note about [withheld] fields", "n": 3, "x": None})


def test_save_version_is_not_found_when_the_base_row_is_missing():
    # the lock row says latest=1, but the select for version 1 finds nothing
    conn = _Conn(rows=[("draft", None, 1), None])
    with pytest.raises(repo.NotFound):
        repo.save_version(conn, "GEN-0001", {"name": "x"}, base_version=1, intent="draft", actor="t", change_note="")
    assert conn.rolled_back and not conn.committed and conn.autocommit is True
    assert "form_state" in conn.cur.sql[-1]


# ------------------------------------------------------------------ the policy page and learning
class _PageCur(_Cur):
    def fetchall(self):
        return []                # no versions: the page's own parts are not what is tested here


def _page(monkeypatch, for_policy):
    conn = _Conn()
    conn.cur = _PageCur([("live", 2, 2, "Operations")])
    monkeypatch.setattr(repo.conflict_cases, "history_for",
                        lambda cur, key, latest, status, viewer=None: {"conflicts": [], "pendingAction": None})
    monkeypatch.setattr(repo.second_review, "status", lambda cur, key, head, versions: {"required": False})
    monkeypatch.setattr(repo.learning, "for_policy", for_policy)
    return repo.get_policy(conn, "OPS-0042")


def test_the_policy_page_carries_what_learning_returns(monkeypatch):
    assert _page(monkeypatch, lambda cur, key, caller: [{"id": 7, "policyKey": key}])["suggestions"] == [
        {"id": 7, "policyKey": "OPS-0042"}]


def test_the_policy_page_reads_its_cards_as_the_caller(monkeypatch):
    who, seen = object(), []
    monkeypatch.setattr(repo.conflict_cases, "history_for",
                        lambda cur, key, latest, status, viewer=None: {"conflicts": [], "pendingAction": None})
    monkeypatch.setattr(repo.second_review, "status", lambda cur, key, head, versions: {"required": False})
    monkeypatch.setattr(repo.learning, "for_policy", lambda cur, key, caller: seen.append(caller) or [])
    for given in ({"caller": who}, {}):
        conn = _Conn()
        conn.cur = _PageCur([("live", 2, 2, "Operations")])
        repo.get_policy(conn, "OPS-0042", **given)
    assert seen == [who, None], "whose cards, and who may hide them, is the caller's; nobody when none is given"


def test_the_policy_page_opens_without_suggestions_when_learning_cannot_be_read(monkeypatch, caplog):
    def boom(cur, key, caller):
        raise RuntimeError("relation proc.bp_agent_policy_suggestion does not exist")
    got = _page(monkeypatch, boom)
    assert got["suggestions"] == [] and got["policyKey"] == "OPS-0042" and got["status"] == "live"
    assert got["secondReview"] == {"required": False}, "the rest of the page is still built"
    [line] = caplog.records
    assert line.levelname == "WARNING" and line.exc_info is None and "\n" not in line.getMessage()
    assert "OPS-0042" in line.getMessage() and "RuntimeError" in line.getMessage()
