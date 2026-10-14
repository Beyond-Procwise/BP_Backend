from decimal import Decimal as D

import pytest

from src.services.sell_side import calibration as cal


def test_a_type_with_enough_history_gets_its_win_rate():
    (c,) = cal.rates({"upsell": (9, 21)}, min_closed=30)
    assert (c.rate, c.applied) == (D("0.3000"), True)


def test_one_short_of_the_threshold_gets_nothing():
    (c,) = cal.rates({"upsell": (9, 20)}, min_closed=30)
    assert (c.rate, c.applied) == (None, False)


def test_no_history_at_all_gets_nothing():
    assert cal.rates({}, min_closed=30) == []


def test_the_scheduler_registers_the_daily_calibration():
    from src.services import backend_scheduler as bs
    assert "_register_sales_calibration_job()" in open(bs.__file__).read()
    assert bs.BackendScheduler.SALES_CALIBRATION_JOB_NAME == "sales_win_probability_calibration"


def test_calibrate_rolls_back_when_an_update_fails(monkeypatch):
    """calibrate() reads one applied group, then the UPDATE for it blows up: the whole
    transaction must roll back rather than leave a partial write committed."""
    monkeypatch.setattr(cal, "_MIN_CLOSED", lambda: 1)

    class FakeCursor:
        def __init__(self):
            self._rows = []

        def execute(self, sql, params=None):
            flat = " ".join(sql.split())
            if flat.startswith("UPDATE"):
                raise RuntimeError("boom: simulated database failure")
            self._rows = [{"opportunity_type": "upsell", "won": 3, "lost": 1}]

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self):
            self.rolled_back = False
            self.committed = False

        def cursor(self, cursor_factory=None):
            return FakeCursor()

        def rollback(self):
            self.rolled_back = True

        def commit(self):
            self.committed = True

    conn = FakeConn()
    with pytest.raises(RuntimeError):
        cal.calibrate(conn)
    assert conn.rolled_back is True
    assert conn.committed is False
