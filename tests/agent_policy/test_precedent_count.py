"""N, the precedent count, is one governed value read fresh (design §3.3, rulings R2 and R4)."""
import inspect
from datetime import datetime, timezone

import pytest

from services.agent_policy import conflict_live as CL
from services.agent_policy import settings as S
from src.services import governed_limits as GL
from tests.agent_policy.fixtures import precedent_n


def test_precedent_count_is_the_governed_value_read_fresh(monkeypatch):
    def shared():
        raise AssertionError("the cached engine was used")
    monkeypatch.setattr(GL, "_engine", shared)
    precedent_n(monkeypatch, 3)
    assert S.precedent_count() == 3


def test_the_seed_holds_five():
    assert S.precedent_count() == 5


def test_a_missing_row_raises(monkeypatch):
    precedent_n(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        S.precedent_count()


def test_threshold_is_none_and_warns_when_the_row_is_missing(monkeypatch, caplog):
    precedent_n(monkeypatch, None, missing=True)
    with caplog.at_level("WARNING", logger=CL.__name__):
        assert CL.threshold() is None
    assert "repeat proposal skipped: the precedent count cannot be read (LimitUnavailable)" in caplog.text


def test_threshold_is_none_for_a_value_that_is_not_a_number(monkeypatch, caplog):
    precedent_n(monkeypatch, "five")
    with caplog.at_level("WARNING", logger=CL.__name__):
        assert CL.threshold() is None
    assert "(ValueError)" in caplog.text


class _NoSql:
    def execute(self, sql, params=None):
        raise AssertionError(f"nothing may run when the proposal is off: {sql}")


@pytest.mark.parametrize("n", [None, 0, -1])
def test_no_proposal_when_off_or_unreadable(monkeypatch, n):
    monkeypatch.setattr(CL, "threshold", lambda: n)
    CL._propose_safely(_NoSql(), 1, now=datetime.now(timezone.utc))


def test_live_conflict_repeat_is_no_longer_a_setting():
    assert "live_conflict_repeat" not in S.DEFAULTS
    assert "live_conflict_repeat" not in inspect.getsource(CL)
