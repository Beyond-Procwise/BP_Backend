"""The one ISO 8601 duration reader shared by the enforcement check and the approval timer."""
import pytest

from services.agent_policy import durations as D


@pytest.mark.parametrize("raw,secs", [
    ("PT4H", 14400), ("P1D", 86400), ("P1DT2H30M", 95400), ("PT90S", 90), ("pt15m", 900),
    ("P1W", 604800), ("P2W1D", 1296000), ("PT1.5H", 5400), ("P0,5D", 43200), (" PT4H ", 14400),
])
def test_parse_valid(raw, secs):
    assert D.parse(raw) == secs


@pytest.mark.parametrize("raw", [None, "", "P", "PT", "4H", "PT0S", "P0D", "P1DT", "PT4X", "PT-1H",
                                 "soon", 3600, "P1H", "PT1H2D"])
def test_unreadable_or_zero_is_none_and_resolves_to_the_default(raw):
    assert D.parse(raw) is None
    assert D.resolve(raw) == "PT4H"
    assert D.resolve(raw, "PT6H") == "PT6H"


def test_resolve_keeps_a_readable_value_and_checks_the_default():
    assert D.resolve("p1w", "PT6H") == "P1W"
    assert D.resolve(None, "nonsense") == "PT4H"
