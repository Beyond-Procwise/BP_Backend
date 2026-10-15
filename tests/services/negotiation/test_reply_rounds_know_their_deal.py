"""A supplier's emailed reply is negotiated with its deal's plays.

The reply reaches the agent with no deal or quote on it, and nothing saved the
deal with the session, so a reply round had no Advisor posture and no plays.
The deal named on a round is now saved with the session and read back on the
rounds that name none. A deal is never guessed, never blanked, never switched.
"""
import logging

import pytest

from agents.negotiation_agent import NegotiationAgent
from agents.base_agent import AgentContext
from src.agents.negotiation.models import UniqueConstraintInfo
from tests.test_negotiation_agent import DummyNick


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
            raise RuntimeError("column deal_id does not exist")

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

    def commit(self):
        pass

    def rollback(self):
        pass


def _agent(conn=None):
    nick = DummyNick()
    nick.get_db_connection = lambda: conn or _Conn()
    return NegotiationAgent(nick)


def _context(**data):
    return AgentContext(workflow_id="wf-1", agent_id="NegotiationAgent",
                        user_id=None, input_data=dict(data))


class TestSaving:
    def test_the_deal_is_written_when_the_table_has_the_column(self):
        assert _agent()._deal_for_record({"deal_id": "DEAL-1"}, {"deal_id": {}}) == "DEAL-1"

    def test_a_table_without_the_column_is_written_as_before(self):
        assert _agent()._deal_for_record({"deal_id": "DEAL-1"}, {"status": {}}) is None

    @pytest.mark.parametrize("state", [{}, {"deal_id": None}, {"deal_id": "  "}])
    def test_no_deal_writes_nothing(self, state):
        assert _agent()._deal_for_record(state, {"deal_id": {}}) is None

    def test_a_round_with_no_deal_cannot_blank_the_saved_one(self):
        """The upsert updates only the columns it writes; deal_id must not be one of them."""
        conn = _Conn()
        agent = _agent(conn)
        agent._state_schema_checked = True
        columns = {name: {"nullable": True} for name in (
            "workflow_id", "supplier_id", "status", "deal_id", "unique_id")}
        agent._get_column_metadata = lambda table: columns
        agent._get_unique_constraint = lambda *a, **kw: UniqueConstraintInfo(
            columns=("workflow_id", "supplier_id"))

        agent._save_session_state("wf-1", "SUP-1", {"status": "ACTIVE"})
        sql, _ = conn.executed[-1]
        assert "INSERT INTO proc.negotiation_session_state" in sql
        assert "deal_id" not in sql

        agent._save_session_state("wf-1", "SUP-1", {"status": "ACTIVE", "deal_id": "DEAL-1"})
        sql, params = conn.executed[-1]
        assert "deal_id = EXCLUDED.deal_id" in sql and "DEAL-1" in params


class TestLoading:
    def test_the_saved_deal_comes_back(self):
        conn = _Conn(row=("DEAL-1",))
        assert _agent(conn)._load_saved_deal("wf-1", "SUP-1") == "DEAL-1"
        sql, params = conn.executed[-1]
        assert "SELECT deal_id" in sql and params == ("wf-1", "SUP-1")

    @pytest.mark.parametrize("row", [None, (None,), ("",)])
    def test_nothing_saved_reads_as_none(self, row):
        assert _agent(_Conn(row=row))._load_saved_deal("wf-1", "SUP-1") is None

    def test_a_database_without_the_column_reads_as_none(self):
        assert _agent(_Conn(fail=True))._load_saved_deal("wf-1", "SUP-1") is None


class _StateCursor(_Cursor):
    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self.last = sql

    def fetchall(self):
        if "information_schema.columns" in self.last:
            return [(name, "YES") for name in (
                "workflow_id", "supplier_id", "supplier_reply_count", "current_round", "status",
                "awaiting_response", "base_subject", "initial_body", "thread_state", "email_history",
                "session_reference", "deal_id")]
        return []

    def fetchone(self):
        if "SELECT deal_id" in self.last:
            return (self.conn.deal,)
        if "SELECT positions" in self.last:
            return (None,)
        if "FROM proc.negotiation_session_state" in self.last:
            return (1, 2, "ACTIVE", True, None, None, None, "Pricing Discussion", None, None, "wf-1", None, None)
        return None


class _StateConn(_Conn):
    def __init__(self, deal):
        super().__init__()
        self.deal = deal

    def cursor(self):
        return _StateCursor(self)


class TestTheReplyRound:
    """What a supplier's reply goes through: a saved session, an input naming no deal."""

    def _load(self, deal):
        agent = _agent(_StateConn(deal))
        agent._state_schema_checked = True
        agent._rehydrate_email_history = lambda **kw: None
        return agent, agent._load_session_state("wf-1", "SUP-1")

    def test_a_reloaded_session_has_its_deal_back(self):
        _, (state, exists) = self._load("DEAL-1")
        assert exists is True and state["deal_id"] == "DEAL-1"

    def test_a_session_saved_without_a_deal_loads_as_before(self):
        _, (state, exists) = self._load(None)
        assert exists is True and not state.get("deal_id")

    def test_the_reply_is_judged_against_the_saved_deals_posture(self, monkeypatch):
        from src.services.negotiation_advice import advisor
        asked = []
        monkeypatch.setattr(advisor, "read_posture",
                            lambda **kw: asked.append(kw) or {"quadrant": "Leverage"})
        agent, (state, _) = self._load("DEAL-1")
        context = _context(supplier_message="We can do 480.")
        agent._bind_session_deal(context, state)
        assert agent._deal_posture(context) == {"quadrant": "Leverage"}
        assert asked == [{"deal_id": "DEAL-1", "quote_id": None}]


class TestBinding:
    def test_the_deal_named_on_the_first_round_is_kept_with_the_session(self):
        state = {}
        _agent()._bind_session_deal(_context(deal_id="DEAL-1"), state)
        assert state["deal_id"] == "DEAL-1"

    def test_a_round_naming_no_deal_uses_the_saved_one(self):
        context = _context()
        _agent()._bind_session_deal(context, {"deal_id": "DEAL-1"})
        assert context.input_data["deal_id"] == "DEAL-1"

    def test_a_negotiation_does_not_change_deal(self, caplog):
        state, context = {"deal_id": "DEAL-1", "workflow_id": "wf-1"}, _context(deal_id="DEAL-2")
        with caplog.at_level(logging.WARNING):
            _agent()._bind_session_deal(context, state)
        assert state["deal_id"] == "DEAL-1" and context.input_data["deal_id"] == "DEAL-1"
        assert "ignoring deal DEAL-2" in caplog.text

    def test_a_quote_names_its_deal(self, monkeypatch):
        from src.services.negotiation_advice import advisor
        monkeypatch.setattr(advisor, "_deal_for_quote",
                            lambda conn, quote_id: {"Q-1": "DEAL-9"}.get(quote_id))
        state = {}
        _agent()._bind_session_deal(_context(quote_id="Q-1"), state)
        assert state["deal_id"] == "DEAL-9"

    def test_nothing_named_and_nothing_saved_assumes_nothing(self, monkeypatch):
        from src.services.negotiation_advice import advisor
        monkeypatch.setattr(advisor, "_deal_for_quote", lambda conn, quote_id: "DEAL-X")
        state, context = {}, _context(supplier_message="hello")
        _agent()._bind_session_deal(context, state)
        assert "deal_id" not in state and "deal_id" not in context.input_data
