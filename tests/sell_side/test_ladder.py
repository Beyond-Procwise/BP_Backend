import pytest

from src.services.sell_side import ladder


def test_the_ids_are_the_ui_seed_verbatim():
    # beyond_procwise_ui/src/lib/processTaxonomy/salesLifecycle.js, v1.0.0
    assert ladder.PHASES == ("sales.opportunity", "sales.margin", "sales.approval")
    assert set(ladder.SUBPROCESSES) == {
        "sales.opportunity.qualified", "sales.opportunity.quote-drafted",
        "sales.opportunity.quote-reviewed", "sales.margin.cost-to-serve-modelled",
        "sales.margin.discount-checked", "sales.margin.margin-floor-tested",
        "sales.approval.deal-desk-review", "sales.approval.pricing-approval",
        "sales.approval.conditions-attached"}


def test_a_matching_pair_and_an_empty_pair_pass():
    ladder.check("sales.margin", "sales.margin.discount-checked")
    ladder.check(None, None)
    ladder.check("sales.margin", None)


@pytest.mark.parametrize("phase,sub", [
    ("sales.nope", None),
    ("sales.margin", "sales.approval.pricing-approval"),
    (None, "sales.margin.discount-checked"),
])
def test_an_unknown_or_mismatched_pair_is_refused(phase, sub):
    with pytest.raises(ValueError):
        ladder.check(phase, sub)


def test_every_quote_rung_is_on_the_ladder():
    for phase, sub in ladder.QUOTE_RUNG.values():
        ladder.check(phase, sub)
