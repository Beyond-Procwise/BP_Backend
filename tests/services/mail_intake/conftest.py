"""Shared test doubles for the mail_intake suite.

A fake cursor and a fake object store, defined once. Every mail_intake test
drives these rather than a real database or a real bucket, following the
pattern in tests/services/test_analysis_store.py — the suite never needs
either to be running.
"""
import pytest


class FakeCursor:
    """Records every statement and replays queued results in order.

    Results can be supplied at construction or queued afterwards, because some
    tests know their fixtures up front and others set them up per-case.
    """

    def __init__(self, results=()):
        self._results = list(results)
        self.calls = []
        # Mirrors psycopg2: rows affected by the last statement. Defaults to 1
        # so ordinary tests read as "the update applied"; set it to 0 to model
        # a guarded UPDATE that lost a race.
        self.rowcount = 1

    # -- cursor API --
    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._results.pop(0) if self._results else None

    def fetchall(self):
        """Rows for the CURRENT query, matching psycopg2 — not the whole queue.

        An earlier version drained everything queued, which is not what a real
        cursor does. Across a function issuing several queries, the first
        fetchall swallowed every later query's rows and handed them back nested.
        That pushed a workaround into production code — `metrics.collect` grew a
        branch unwrapping a nested list, with a comment explaining it existed
        for the test double. Shaping shipped code around a fake is backwards:
        the fake is the thing that was wrong.

        Semantics: pop one queued entry. A list is a row set and is returned as
        is; anything else is a single row and is wrapped.
        """
        if not self._results:
            return []
        entry = self._results.pop(0)
        return list(entry) if isinstance(entry, list) else [entry]

    # -- test helpers --
    def queue(self, *rows):
        """Append rows for subsequent fetches. Returns self for chaining."""
        self._results.extend(rows)
        return self

    def statements(self, needle):
        """Every recorded (sql, params) whose SQL contains `needle`."""
        return [call for call in self.calls if needle in call[0]]

    def states_written(self):
        """Params of every statement that set a state column."""
        return [params for sql, params in self.calls if "SET state" in sql]


class FakeStore:
    """In-memory stand-in for store.ObjectStore."""

    def __init__(self):
        self.objects = {}

    def put_immutable(self, key, content):
        self.objects[key] = content
        return key

    def exists(self, key):
        return key in self.objects

    def get(self, key):
        return self.objects.get(key)

    def keys_under(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def fake_cursor():
    return FakeCursor()


@pytest.fixture
def fake_store():
    return FakeStore()
