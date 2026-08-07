"""Domain-to-supplier binding, with ambiguity as a first-class outcome.

In this corpus ambiguity is the COMMON case, not an edge: bp_supplier holds
5,028 rows sharing only 288 distinct domains. Any implementation that quietly
picks the first candidate would attribute most inbound mail to the wrong
supplier, confidently.
"""
from src.services.mail_intake.supplier_bind import bind


def _lookup(rows):
    return lambda domain: [r for r in rows if r[1] == domain]


def test_single_candidate_matches():
    result = bind("copperleaf.test", _lookup([("SUP-001", "copperleaf.test")]))
    assert result.state == "matched"
    assert result.supplier_id == "SUP-001"


def test_no_candidate_is_unmatched():
    result = bind("stranger.test", _lookup([("SUP-001", "copperleaf.test")]))
    assert result.state == "unmatched"
    assert result.supplier_id is None


def test_several_candidates_are_ambiguous_and_never_guessed():
    rows = [("SUP-001", "shared.test"), ("SUP-002", "shared.test")]
    result = bind("shared.test", _lookup(rows))
    assert result.state == "ambiguous"
    assert result.supplier_id is None
    assert set(result.candidates) == {"SUP-001", "SUP-002"}


def test_domain_comparison_is_case_insensitive():
    result = bind("CopperLeaf.TEST", _lookup([("SUP-001", "copperleaf.test")]))
    assert result.state == "matched"


def test_missing_domain_is_unmatched():
    result = bind(None, _lookup([("SUP-001", "copperleaf.test")]))
    assert result.state == "unmatched"


def test_lookup_failure_is_unmatched_but_is_logged(caplog):
    """An outage and a genuinely unknown domain produce identical results.

    That is correct for the verdict — a failed lookup is not evidence of a
    match — but without a log the two are indistinguishable, and a database
    outage would masquerade as a run of unmatched senders. The unmatched-domain
    rate is watched as a security signal, so an infrastructure failure quietly
    inflating it is worse than a visible error.
    """
    def broken(_domain):
        raise RuntimeError("db down")

    with caplog.at_level("ERROR"):
        result = bind("copperleaf.test", broken)

    assert result.state == "unmatched"
    assert any("supplier lookup failed" in r.getMessage() for r in caplog.records)
