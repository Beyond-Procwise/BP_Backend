import pytest

from services.price_outlier.rule import OutlierSettings, assess

S = OutlierSettings()
TIGHT = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2]


def test_ten_times_the_median_is_critical():
    v = assess(1000.0, TIGHT, S)
    assert v.flagged and v.severity == "critical"
    assert v.ratio == pytest.approx(10.0, rel=0.02)


def test_a_tenth_of_the_median_is_critical():
    assert assess(10.0, TIGHT, S).severity == "critical"


def test_three_times_the_median_is_a_warning():
    v = assess(300.0, TIGHT, S)
    assert v.flagged and v.severity == "warning"


def test_a_statistically_huge_but_trivial_difference_does_not_flag():
    """2% away from identical peers is arithmetically enormous and
    commercially meaningless. Both conditions must hold."""
    identical = [100.0] * 8
    v = assess(102.0, identical, S)
    assert not v.flagged


def test_identical_peers_fall_back_to_the_ratio_test():
    identical = [100.0] * 8
    assert assess(1000.0, identical, S).flagged


def test_genuine_price_spread_does_not_flag():
    spread = [50.0, 80.0, 100.0, 130.0, 160.0, 200.0]
    assert not assess(210.0, spread, S).flagged


def test_too_few_peers_never_flags():
    v = assess(1000.0, [100.0, 100.0, 100.0, 100.0], S)
    assert not v.flagged
    assert v.peer_count == 4


def test_a_zero_or_negative_median_cannot_form_a_ratio():
    assert not assess(100.0, [0.0] * 8, S).flagged


def test_a_normal_price_does_not_flag():
    assert not assess(100.3, TIGHT, S).flagged
