"""Selection is deterministic, and it refuses rather than guesses."""

import logging
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.finding_source import Finding  # noqa: E402
from src.services.playbooks.selector import select, tied_candidates  # noqa: E402
from src.services.playbooks.store import Playbook  # noqa: E402


def pb(pid, match, source="detection_finding", name=None, version=1):
    return Playbook(
        playbook_id=pid,
        playbook_name=name or f"playbook {pid}",
        trigger_source=source,
        trigger_match=match,
        agent_workflow_id=958,
        params={},
        version=version,
    )


def finding(**attrs):
    return Finding(
        source=attrs.pop("source", "detection_finding"),
        finding_id=attrs.pop("finding_id", "1"),
        deal_id=attrs.pop("deal_id", "D-1"),
        attrs=attrs,
    )


def test_exact_match_selects():
    chosen = select(
        finding(rule_id="duplicate", severity="critical"),
        [pb(7, {"rule_id": "duplicate"})],
    )
    assert chosen is not None
    assert chosen.playbook.playbook_id == 7
    assert chosen.evidence["matched"] == {"rule_id": "duplicate"}
    assert chosen.evidence["key_count"] == 1
    assert chosen.evidence["playbook_version"] == 1


def test_a_non_matching_value_does_not_select():
    assert select(
        finding(rule_id="quantity"),
        [pb(7, {"rule_id": "duplicate"})],
    ) is None


def test_most_specific_wins():
    chosen = select(
        finding(rule_id="duplicate", severity="critical", category="duplicate"),
        [
            pb(1, {"rule_id": "duplicate"}),
            pb(2, {"rule_id": "duplicate", "severity": "critical"}),
        ],
    )
    assert chosen.playbook.playbook_id == 2
    assert chosen.evidence["key_count"] == 2


def test_a_tie_proposes_nothing_and_names_both(caplog):
    """A configuration error that resolves itself quietly is how four of five
    detector bindings were wrong for months. A tie is visible or it is nothing."""
    two = [
        pb(1, {"rule_id": "duplicate"}, name="Recover it"),
        pb(2, {"severity": "critical"}, name="Escalate it"),
    ]
    f = finding(rule_id="duplicate", severity="critical")
    with caplog.at_level(logging.ERROR):
        assert select(f, two) is None
    logged = caplog.text
    assert "Recover it" in logged and "Escalate it" in logged
    assert "1" in logged and "2" in logged
    assert [p.playbook_id for p in tied_candidates(f, two)] == [1, 2]


def test_tied_candidates_is_empty_when_there_is_a_winner():
    f = finding(rule_id="duplicate", severity="critical")
    one = [pb(1, {"rule_id": "duplicate"}), pb(2, {"rule_id": "duplicate", "severity": "critical"})]
    assert select(f, one) is not None
    assert tied_candidates(f, one) == []


def test_tied_candidates_is_empty_when_nothing_matched():
    assert tied_candidates(finding(rule_id="quantity"), [pb(1, {"rule_id": "duplicate"})]) == []


def test_empty_trigger_match_is_a_catch_all():
    chosen = select(finding(rule_id="anything"), [pb(9, {})])
    assert chosen.playbook.playbook_id == 9
    assert chosen.evidence["matched"] == {}
    assert chosen.evidence["key_count"] == 0


def test_a_catch_all_loses_to_a_specific_playbook():
    chosen = select(
        finding(rule_id="duplicate"),
        [pb(9, {}), pb(3, {"rule_id": "duplicate"})],
    )
    assert chosen.playbook.playbook_id == 3


def test_two_catch_alls_for_one_source_are_a_tie():
    assert select(finding(rule_id="duplicate"), [pb(9, {}), pb(10, {})]) is None


def test_the_wrong_source_never_matches():
    """An opportunity playbook must not be reachable from a detection finding
    even if a key name happened to coincide."""
    assert select(
        finding(source="detection_finding", rule_id="duplicate"),
        [pb(5, {}, source="opportunity")],
    ) is None


def test_matching_is_case_and_type_insensitive_on_both_sides():
    """A boolean column authored in JSON as a string, and a capitalised
    severity, must still match -- otherwise the playbook silently never fires."""
    chosen = select(
        finding(severity="critical", blocks_promotion=True),
        [pb(4, {"severity": "Critical", "blocks_promotion": "true"})],
    )
    assert chosen is not None and chosen.playbook.playbook_id == 4


def test_a_null_finding_attribute_does_not_match_a_present_key():
    """doc_type IS NULL is not doc_type = 'invoice', and must never be treated
    as a wildcard."""
    assert select(
        finding(rule_id="duplicate", doc_type=None),
        [pb(6, {"rule_id": "duplicate", "doc_type": "invoice"})],
    ) is None


def test_a_missing_finding_attribute_does_not_match():
    assert select(
        finding(rule_id="duplicate"),
        [pb(6, {"rule_id": "duplicate", "doc_type": "invoice"})],
    ) is None


def test_evidence_records_the_findings_values_not_the_playbooks():
    """A proposal must be re-derivable from source, so the evidence has to say
    what the finding held, canonically, at the moment it matched."""
    chosen = select(
        finding(severity="CRITICAL", blocks_promotion=True),
        [pb(4, {"severity": "critical", "blocks_promotion": True})],
    )
    assert chosen.evidence["matched"] == {"severity": "critical", "blocks_promotion": "true"}


def test_no_playbooks_selects_nothing():
    assert select(finding(rule_id="duplicate"), []) is None
