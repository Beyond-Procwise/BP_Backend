from decimal import Decimal as D

import pytest

from src.services.governed_limits import LimitUnavailable
from src.services.triage.tolerance import load_config, resolve_tolerance
from tests.conftest import GOVERNED_LIMIT_SEED
from tests.triage.helpers import make_cfg

SEED = GOVERNED_LIMIT_SEED["triage_tolerances"]


def test_missing_key_refuses():
    rules = dict(SEED)
    rules.pop("band_s1")

    def read(policy, key, cast):
        if key not in rules:
            raise LimitUnavailable(key)
        return cast(rules[key])

    with pytest.raises(LimitUnavailable):
        load_config(read=read)


def test_null_value_refuses():
    rules = dict(SEED, band_s1=None)
    with pytest.raises(LimitUnavailable):
        load_config(read=lambda p, k, cast: None if rules[k] is None else cast(rules[k]))


def test_unit_price_over_takes_the_stricter_of_pct_and_abs():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    assert tol.allowance(D("12.00"), D("1")) == D("0.12")   # 1% < £5
    assert tol.allowance(D("2300"), D("1")) == D("5")       # £5 < 1%


def test_absolute_part_converts_to_document_currency():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    # 1 unit of document currency = £0.50, so £5 = 10 units; 1% of 2,300 = 23 -> min is 10
    assert tol.allowance(D("2300"), D("0.5")) == D("10")


def test_without_fx_the_percentage_applies():
    tol = resolve_tolerance("unit_price_over", make_cfg())
    assert tol.allowance(D("2300"), None) == D("23")


def test_under_and_quantity_are_percentage_only():
    cfg = make_cfg()
    assert resolve_tolerance("unit_price_under", cfg).allowance(D("12"), D("1")) == D("0.6")
    assert resolve_tolerance("quantity_over", cfg).allowance(D("10"), None) == D("0.5")


def test_cumulative_total():
    tol = resolve_tolerance("cumulative_total", make_cfg())
    assert tol.allowance(D("120"), D("1")) == D("0.6")
    assert tol.allowance(D("100000"), D("1")) == D("50")
    assert tol.as_dict()["source"].startswith("bp_policy:triage_tolerances")


def test_unknown_check_raises():
    with pytest.raises(KeyError):
        resolve_tolerance("nonsense", make_cfg())


def test_fingerprint_tracks_values():
    assert make_cfg().fingerprint == make_cfg().fingerprint
    assert make_cfg().fingerprint != make_cfg(band_s1=75).fingerprint
