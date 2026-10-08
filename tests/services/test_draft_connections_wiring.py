"""The assurance layer reads through the reader door and writes through the writer door - no database needed."""

from contextlib import contextmanager
from types import SimpleNamespace

from src.services.draft_assurance import connections
from tests.services.test_draft_agent_stages import DECISION, _decision_agent
from tests.services.test_draft_assurance import FakeConn, TABLES


class RealishConn(FakeConn):
    """A fake that, like a real connection, can be made read-only and reports it."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sessions, self.readonly, self.autocommit, self.closed = [], None, True, False

    def set_session(self, readonly=None, autocommit=None):
        self.sessions.append((readonly, autocommit))
        if readonly is not None:
            self.readonly = None if readonly == "default" else readonly
        if autocommit is not None:
            self.autocommit = autocommit

    def close(self):
        self.closed = True


def test_every_draft_records_which_control_its_reads_were_made_under(monkeypatch):
    agent = _decision_agent(monkeypatch)
    conn = RealishConn(TABLES)
    agent.agent_nick.get_db_connection = lambda: conn
    rec = agent.from_decision(dict(DECISION))["assurance"]
    assert rec["read_control"] == connections.INTERIM
    assert conn.sessions[0] == (True, True) and conn.sessions[-1][0] == "default"      # set read-only, then handed back


def test_a_connection_that_cannot_be_set_read_only_is_recorded_as_unenforced(monkeypatch):
    agent = _decision_agent(monkeypatch)                  # the default fake has no set_session at all
    assert agent.from_decision(dict(DECISION))["assurance"]["read_control"] == connections.UNENFORCED


def test_with_dedicated_credentials_the_reads_go_through_the_dedicated_role(monkeypatch):
    agent = _decision_agent(monkeypatch)
    ro, used = RealishConn(TABLES), []
    monkeypatch.setenv("EMAIL_AGENT_RO_USER", "email_agent_ro_svc")
    monkeypatch.setenv("EMAIL_AGENT_RO_PASSWORD", "x")
    monkeypatch.setattr(connections, "_connect_as", lambda settings, user, pw: used.append(user) or ro)
    agent.agent_nick.settings = SimpleNamespace(**{**vars(agent.agent_nick.settings), "db_host": "h", "db_name": "n", "db_port": 5432})
    rec = agent.from_decision(dict(DECISION))["assurance"]
    assert used == ["email_agent_ro_svc"] and rec["read_control"] == connections.DEDICATED
    assert ro.closed and ro.sessions[0] == (True, True)
    assert rec["facts"]["supplier_current_offer"]["row_id"] == "2"                       # and the facts still resolved


def test_capture_is_written_through_the_writer_door(monkeypatch):
    from src.services.draft_assurance import capture
    agent = _decision_agent(monkeypatch)
    draft = agent.from_decision(dict(DECISION))        # built first: drafting itself also reads the flag table through the writer door
    seen = []

    @contextmanager
    def fake_writer(nick=None):
        seen.append("writer")
        yield "WRITER-CONN"
    monkeypatch.setattr(connections, "writer", fake_writer)
    monkeypatch.setattr(capture, "record_draft", lambda conn, d: seen.append(conn))
    agent._capture_draft(draft)
    assert seen == ["writer", "WRITER-CONN"]


def test_the_learning_job_writes_through_the_writer_door(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    from src.services.draft_assurance import learning
    seen = []

    @contextmanager
    def fake_writer(nick=None):
        seen.append("writer")
        yield "WRITER-CONN"
    monkeypatch.setattr(connections, "writer", fake_writer)
    monkeypatch.setattr(learning, "run_learning", lambda conn, engine: seen.append(conn) or {})
    sched = BackendScheduler.__new__(BackendScheduler)
    sched.agent_nick = SimpleNamespace()
    sched._run_email_learning()
    assert seen == ["writer", "WRITER-CONN"]


def test_the_reads_never_use_the_writers_door_or_the_other_way_round(monkeypatch):
    agent = _decision_agent(monkeypatch)
    monkeypatch.setattr(connections, "writer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a read opened the writer")))
    agent.from_decision(dict(DECISION))                    # reading facts and tone must not touch the writer
