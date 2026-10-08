# tests/services/graph_resolution/test_contract_amendment.py
"""An amendment is identified by what it amends, not by repeating its title.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_amendment.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                   # noqa: E402
from src.services.graph_resolution import edge_writer                           # noqa: E402
from src.services.graph_resolution.profiles import contract_amendment as am     # noqa: E402


def _parent(**o):
    r = {"contract_id": "SOW-1", "contract_title": "Statement of Work Helix Migration",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.sow",
         "contract_start_date": "2026-01-01", "contract_end_date": "2027-12-31"}
    r.update(o)
    return r


def _amend(**o):
    r = {"contract_id": "ADD-1", "contract_title": "Addendum No. 1", "supplier_id": "S-1",
         "resolved_doc_type": "doctype.addendum", "parent_agreement_ref": "SOW-1",
         "parent_contract_id": None, "framework_ref": None, "_ref_resolves": True,
         "contract_start_date": "2026-03-01", "contract_end_date": "2026-09-30"}
    r.update(o)
    return r


def test_the_profile_is_registered_and_never_reaches_auto_link():
    assert am.PROFILE in le.PROFILES
    assert am.PROFILE in edge_writer.UNCALIBRATED_PROFILES


def test_a_resolving_reference_with_a_generic_title_is_proposed():
    """The defect found 2026-10-08: 'Addendum No. 1' scored 36 under the hierarchy profile."""
    got = am.score(_amend(), _parent())
    assert got["F"] >= 65.0, got
    assert got["profile"] == "contract_amendment"
    ids = [s["id"] for s in got["signals"]]
    assert "title_overlap" not in ids and "expected_structure" not in ids


def test_a_matching_buyer_strengthens_it():
    plain = am.score(_amend(), _parent())["F"]
    with_buyer = am.score(_amend(buyer_org_id="B-1"), _parent(buyer_org_id="B-1"))["F"]
    assert with_buyer > plain


def test_no_reference_is_not_proposed():
    """Supplier alone cannot say WHICH contract is being amended."""
    got = am.score(_amend(parent_agreement_ref=None, _ref_resolves=False), _parent())
    assert got["F"] < 65.0


def test_a_reference_to_a_different_real_contract_is_a_conflict():
    got = am.score(_amend(parent_agreement_ref="SOW-9", _ref_resolves=True), _parent())
    assert got["F"] < 50.0


def test_payment_terms_are_never_read_for_an_amendment():
    got = am.score(_amend(payment_terms="Net 60"), _parent(payment_terms="Net 30"))
    assert "payment_terms" not in [s["id"] for s in got["signals"]]
