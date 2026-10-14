"""A round that holds its own lock must still be able to run.

`_execute_round_entry_with_lock` takes the (workflow, supplier, round) advisory
lock and then runs the supplier's negotiation, which arrives at
`_run_single_negotiation` -- and that takes the same lock again, on a second
database connection. Postgres advisory locks belong to the connection, so the
second attempt was refused every time: the round reported
`negotiation_session_locked` against itself and no counter-offer was ever
drafted on the multi-round path.

The lock still has to refuse a genuinely different caller. Both halves are
tested here against a stand-in that keeps Postgres's rule: a lock held on one
connection cannot be taken on another.
"""
import threading

import pytest

from agents.negotiation_agent import NegotiationAgent
from tests.test_negotiation_agent import DummyNick


class _AdvisoryLocks:
    """Lock ids held, by the connection that holds them."""

    def __init__(self):
        self.held = {}
        self.attempts = 0

    def connection(self):
        return _Conn(self)


class _Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        locks = self.conn.locks
        lock_id = params[0]
        if "pg_try_advisory_lock" in sql:
            locks.attempts += 1
            owner = locks.held.get(lock_id)
            if owner is None:
                locks.held[lock_id] = self.conn
            self.result = (owner is None or owner is self.conn,)
        elif "pg_advisory_unlock" in sql:
            if locks.held.get(lock_id) is self.conn:
                del locks.held[lock_id]
            self.result = (True,)

    def fetchone(self):
        return self.result


class _Conn:
    def __init__(self, locks):
        self.locks = locks

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def locks():
    return _AdvisoryLocks()


@pytest.fixture
def agent(locks):
    nick = DummyNick()
    nick.get_db_connection = locks.connection
    return NegotiationAgent(nick)


def test_the_stand_in_refuses_a_second_connection(locks):
    """Guards the premise: without this rule the tests below prove nothing."""
    first, second = locks.connection(), locks.connection()
    with first.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (7,))
        assert cur.fetchone() == (True,)
    with second.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (7,))
        assert cur.fetchone() == (False,)


def test_a_round_can_take_the_lock_it_already_holds(agent):
    with agent._session_lock("wf-1", "SUP-1", 1) as outer:
        assert outer is True
        with agent._session_lock("wf-1", "SUP-1", 1) as inner:
            assert inner is True


def test_the_lock_is_still_held_after_the_inner_use_ends(agent, locks):
    """The inner use must not release what the outer one is relying on."""
    with agent._session_lock("wf-1", "SUP-1", 1):
        with agent._session_lock("wf-1", "SUP-1", 1):
            pass
        assert locks.held, "inner exit released the outer round's lock"
    assert not locks.held, "lock leaked after the round finished"


def test_another_thread_is_still_refused(agent):
    seen = {}

    def rival():
        with agent._session_lock("wf-1", "SUP-1", 1) as acquired:
            seen["acquired"] = acquired

    with agent._session_lock("wf-1", "SUP-1", 1):
        worker = threading.Thread(target=rival)
        worker.start()
        worker.join()

    assert seen["acquired"] is False


def test_a_different_round_is_not_covered_by_this_one(agent, locks):
    with agent._session_lock("wf-1", "SUP-1", 1):
        with agent._session_lock("wf-1", "SUP-1", 2) as other:
            assert other is True
        assert len(locks.held) == 1


def test_the_lock_can_be_taken_again_once_released(agent):
    with agent._session_lock("wf-1", "SUP-1", 1) as first:
        assert first is True
    with agent._session_lock("wf-1", "SUP-1", 1) as second:
        assert second is True
