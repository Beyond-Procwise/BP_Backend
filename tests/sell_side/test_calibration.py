from decimal import Decimal as D

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
