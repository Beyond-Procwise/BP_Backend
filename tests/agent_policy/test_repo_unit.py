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
