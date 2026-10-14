# tests/services/graph_resolution/test_contract_renewal.py
"""A renewal begins where the agreement it renews ends.

Found live 2026-10-11: RN-2026-0313 named SA-2026-0310, same supplier, same buyer,
same law -- and scored 47.2, below the floor, because the amendment profile reads
"starts after the parent ended" as a CONFLICT. For a variation that is right. For
a renewal it is the definition.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_renewal.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                   # noqa: E402
from src.services.graph_resolution import edge_writer                           # noqa: E402
from src.services.graph_resolution.profiles import contract_amendment as am     # noqa: E402
from src.services.graph_resolution.profiles import contract_renewal as rn       # noqa: E402


def _parent(**o):
    r = {"contract_id": "SA-1", "contract_title": "Service Agreement", "supplier_id": "S-1",
         "resolved_doc_type": "doctype.service_agreement",
         "contract_start_date": "2026-03-01", "contract_end_date": "2027-02-28"}
    r.update(o)
    return r


def _renewal(**o):
    r = {"contract_id": "RN-1", "contract_title": "Renewal Agreement", "supplier_id": "S-1",
         "resolved_doc_type": "doctype.renewal", "parent_agreement_ref": "SA-1",
         "parent_contract_id": None, "framework_ref": None, "_ref_resolves": True,
         "contract_start_date": "2027-03-01", "contract_end_date": "2028-02-29"}
    r.update(o)
    return r


def _term(got):
    return next(s for s in got["signals"] if s["id"] == "term_succession")


def test_the_profile_is_registered_and_never_reaches_auto_link():
    assert rn.PROFILE in le.PROFILES
    assert rn.PROFILE in edge_writer.UNCALIBRATED_PROFILES


def test_the_live_case_a_renewal_starting_the_day_after_is_proposed():
    got = rn.score(_renewal(), _parent())
    assert got["F"] >= 65.0, got
    assert got["profile"] == "contract_renewal"
    assert _term(got)["status"] == "OK"


def test_the_amendment_profile_refuses_the_same_pair():
    """Why a separate profile exists: this is the 47.2 measured live."""
    got = am.score(_renewal(), _parent())
    assert got["F"] < 65.0, got


def test_it_carries_no_title_or_structure_signal_like_an_amendment():
    ids = [s["id"] for s in rn.score(_renewal(), _parent())["signals"]]
    assert "title_overlap" not in ids and "expected_structure" not in ids
    assert "term_containment" not in ids


def test_a_renewal_signed_and_dated_inside_the_parents_term_is_still_ok():
    got = rn.score(_renewal(contract_start_date="2026-12-01",
                            contract_end_date="2028-02-29"), _parent())
    assert _term(got)["status"] == "OK"


def test_a_renewal_that_ended_before_its_parent_began_conflicts():
    """A renewal cannot predate what it renews."""
    got = rn.score(_renewal(contract_start_date="2024-03-01",
                            contract_end_date="2025-02-28"), _parent())
    assert _term(got)["status"] == "CONFLICT"
    assert got["F"] < 65.0, got


def test_a_renewal_starting_before_its_parent_started_is_weak_not_ok():
    got = rn.score(_renewal(contract_start_date="2026-01-01",
                            contract_end_date="2028-02-29"), _parent())
    assert _term(got)["status"] == "WEAK"


def test_missing_dates_are_missing_not_a_verdict():
    got = rn.score(_renewal(contract_start_date=None), _parent())
    assert _term(got)["status"] == "MISSING"


def test_no_reference_is_not_proposed():
    got = rn.score(_renewal(parent_agreement_ref=None, _ref_resolves=False), _parent())
    assert got["F"] < 65.0


def test_another_suppliers_agreement_is_not_proposed():
    got = rn.score(_renewal(), _parent(supplier_id="S-OTHER"))
    assert got["F"] < 65.0
