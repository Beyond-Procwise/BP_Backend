"""The target and walk-away a negotiation is working to survive a restart.

They were kept only in the agent's in-memory state cache: the saved row had no
place for them. A supplier's reply usually arrives hours or days after our
email, by which time the server has restarted, so the reply was judged against
no target at all -- and a final offer could only ever be escalated as
"no target on record".
"""
import json

import pytest

from agents.negotiation_agent import NegotiationAgent
from tests.test_negotiation_agent import DummyNick

POSITIONS = {"start": 490.0, "desired": 455.0, "no_deal": None, "supplier_offer": 490.0, "history": []}


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if self.conn.fail:
            raise RuntimeError("column positions does not exist")

    def fetchone(self):
        return self.conn.row


class _Conn:
    def __init__(self, row=None, fail=False):
        self.row, self.fail, self.executed = row, fail, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _Cursor(self)


def _agent(conn):
    nick = DummyNick()
    nick.get_db_connection = lambda: conn
    return NegotiationAgent(nick)


class TestSaving:
    def test_positions_are_written_when_the_table_has_the_column(self):
        value = NegotiationAgent._positions_for_record({"positions": POSITIONS}, {"positions": {"nullable": True}})
        assert json.loads(value) == POSITIONS

    def test_a_table_without_the_column_is_written_as_before(self):
        assert NegotiationAgent._positions_for_record({"positions": POSITIONS}, {"status": {}}) is None

    @pytest.mark.parametrize("state", [{}, {"positions": None}, {"positions": "455"}, {"positions": {}}])
    def test_nothing_to_save_writes_nothing(self, state):
        assert NegotiationAgent._positions_for_record(state, {"positions": {}}) is None


class TestLoading:
    def test_saved_positions_come_back(self):
        conn = _Conn(row=(POSITIONS,))
        assert _agent(conn)._load_saved_positions("wf-1", "SUP-1") == POSITIONS
        sql, params = conn.executed[-1]
        assert "positions" in sql and params == ("wf-1", "SUP-1")

    def test_positions_stored_as_text_are_parsed(self):
        assert _agent(_Conn(row=(json.dumps(POSITIONS),)))._load_saved_positions("wf-1", "SUP-1") == POSITIONS

    @pytest.mark.parametrize("row", [None, (None,), ("not json",), ([1, 2],)])
    def test_nothing_usable_saved_reads_as_none(self, row):
        assert _agent(_Conn(row=row))._load_saved_positions("wf-1", "SUP-1") is None

    def test_a_database_without_the_column_reads_as_none(self):
        assert _agent(_Conn(fail=True))._load_saved_positions("wf-1", "SUP-1") is None


class _StateCursor(_Cursor):
    """Answers the three reads `_load_session_state` makes, by what each asks for."""

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self.last = sql

    def fetchall(self):
        if "information_schema.columns" in self.last:
            return [(name, "YES") for name in (
                "workflow_id", "supplier_id", "supplier_reply_count", "current_round", "status",
                "awaiting_response", "base_subject", "initial_body", "thread_state", "email_history",
                "session_reference", "positions")]
        return []

    def fetchone(self):
        if "SELECT positions" in self.last:
            return (self.conn.positions,)
        if "FROM proc.negotiation_session_state" in self.last:
            #      replies round status   awaiting sup_msg agent_msg sent_at subject initial thread workflow ref   history
            return (0, 1, "ACTIVE", True, None, None, None, "Pricing Discussion", None, None, "wf-1", None, None)
        return None


class _StateConn(_Conn):
    def __init__(self, positions):
        super().__init__()
        self.positions = positions

    def cursor(self):
        return _StateCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


class TestTheWholeLoad:
    """The path a supplier's reply takes after a restart: nothing in memory, a row in the database."""

    def _load(self, positions):
        agent = _agent(_StateConn(positions))
        agent._state_schema_checked = True
        agent._rehydrate_email_history = lambda **kw: None
        return agent._load_session_state("wf-1", "SUP-1")

    def test_a_reloaded_session_has_its_target_back(self):
        state, exists = self._load(POSITIONS)
        assert exists is True
        assert state["positions"]["desired"] == 455.0

    def test_a_session_saved_without_positions_loads_as_before(self):
        state, exists = self._load(None)
        assert exists is True
        assert not state.get("positions")
