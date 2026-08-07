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

    # -- cursor API --
    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._results.pop(0) if self._results else None

    def fetchall(self):
        out, self._results = list(self._results), []
        return out

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
