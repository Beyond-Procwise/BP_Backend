"""Style subsystem configuration.

The behaviour that matters here is the failure mode: a missing row, an unreachable
database or a malformed value must never promote the system into a mailbox-reading
mode. Mode A is the safe default and every degraded path lands on it.
"""

from __future__ import annotations

import json

import pytest

from services.style.config import (
    DEFAULT_MIN_EXEMPLARS,
    DEFAULT_MODE,
    StyleConfig,
    load_style_config,
)


class _Cursor:
    def __init__(self, row, *, raises=False):
        self._row = row
        self._raises = raises
        self.closed = False

    def execute(self, query, params=None):
        if self._raises:
            raise RuntimeError("database is unreachable")
        assert "proc.bp_admin_config" in query
        assert params == ("style_engine",)

    def fetchone(self):
        return self._row

    def close(self):
        self.closed = True


class _Conn:
    def __init__(self, row, *, raises=False):
        self._row = row
        self._raises = raises

    def cursor(self):
        return _Cursor(self._row, raises=self._raises)


def test_reads_the_seeded_row():
    conn = _Conn(({"deployment_mode": "C1", "min_exemplars": 5, "staging_ttl_hours": 12},))
    cfg = load_style_config(conn)
    assert cfg.deployment_mode == "C1"
    assert cfg.min_exemplars == 5
    assert cfg.staging_ttl_hours == 12
    assert cfg.loaded_from_db is True


def test_accepts_jsonb_returned_as_a_string():
    conn = _Conn((json.dumps({"deployment_mode": "B", "min_exemplars": 4}),))
    cfg = load_style_config(conn)
    assert cfg.deployment_mode == "B"
    assert cfg.min_exemplars == 4


def test_defaults_apply_when_the_row_is_missing():
    cfg = load_style_config(_Conn(None))
    assert cfg.deployment_mode == DEFAULT_MODE
    assert cfg.min_exemplars == DEFAULT_MIN_EXEMPLARS
    assert cfg.loaded_from_db is False


@pytest.mark.parametrize(
    "payload",
    [
        ({"deployment_mode": "C3"},),          # not a real mode
        ({"deployment_mode": None},),          # absent
        ("this is not json",),                 # unparseable
        (["not", "an", "object"],),            # wrong shape
    ],
)
def test_every_malformed_value_falls_back_to_mode_a(payload):
    """A configuration failure must never silently promote the system into reading
    someone's mailbox."""
    cfg = load_style_config(_Conn(payload))
    assert cfg.deployment_mode == "A"
    assert cfg.reads_mailbox_at_compile_time is False
    assert cfg.reads_mailbox_at_draft_time is False


def test_unreachable_database_falls_back_to_mode_a():
    cfg = load_style_config(_Conn(None, raises=True))
    assert cfg.deployment_mode == "A"
    assert cfg.loaded_from_db is False


def test_min_exemplars_cannot_be_lowered_below_three():
    """Invariant 7 — below three exemplars a profile would be describing noise, so the
    floor is not configurable away."""
    cfg = load_style_config(_Conn(({"min_exemplars": 1},)))
    assert cfg.min_exemplars == 3

    cfg = load_style_config(_Conn(({"min_exemplars": "not a number"},)))
    assert cfg.min_exemplars == DEFAULT_MIN_EXEMPLARS


def test_mode_is_case_insensitive():
    assert load_style_config(_Conn(({"deployment_mode": "c2"},))).deployment_mode == "C2"


def test_only_c2_reads_the_mailbox_while_drafting():
    """Under every other mode drafting must complete with the provider unreachable."""
    assert StyleConfig(deployment_mode="C2").reads_mailbox_at_draft_time is True
    for mode in ("A", "B", "C1"):
        assert StyleConfig(deployment_mode=mode).reads_mailbox_at_draft_time is False, mode


def test_c1_and_c2_read_at_compile_time_but_a_and_b_never_do():
    for mode in ("C1", "C2"):
        assert StyleConfig(deployment_mode=mode).reads_mailbox_at_compile_time is True, mode
    for mode in ("A", "B"):
        assert StyleConfig(deployment_mode=mode).reads_mailbox_at_compile_time is False, mode
