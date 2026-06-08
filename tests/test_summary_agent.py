import json
import src.services.summary_agent as sa


class _FakeCursor:
    """Matches a SQL substring -> (columns, rows). Supports fetchone/fetchall."""

    def __init__(self, table_data, recorder=None):
        self._table_data = table_data
        self._recorder = recorder
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        if self._recorder is not None:
            self._recorder.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data, recorder=None):
        self._cur = _FakeCursor(table_data, recorder)
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def test_resolve_persona_hits_bp_prompt():
    conn = _FakeConn({
        "FROM proc.bp_prompt": (
            ["prompts_desc"],
            [({"prompt_template": "You are a compliance auditor."},)],
        ),
    })
    framing, source = sa.resolve_persona("compliance", conn)
    assert framing == "You are a compliance auditor."
    assert source == "bp_prompt"


def test_resolve_persona_falls_back_to_raw():
    conn = _FakeConn({"FROM proc.bp_prompt": (["prompts_desc"], [])})
    framing, source = sa.resolve_persona("some ad-hoc persona", conn)
    assert framing == "some ad-hoc persona"
    assert source == "raw"
