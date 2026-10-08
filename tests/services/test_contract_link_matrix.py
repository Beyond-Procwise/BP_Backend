# tests/services/test_contract_link_matrix.py
"""Every contract child type x six situations, through the real runner.

    PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/test_contract_link_matrix.py -v
"""
from __future__ import annotations

import os
import re
import uuid

import pytest

from src.services import contract_links as CL
from src.services.db import get_conn

pytestmark = pytest.mark.skipif(
    os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() not in ("1", "true", "yes", "on"),
    reason="needs PROCWISE_TEST_LIVE_DB=1")

# child type, parent type, a parent type that must NOT be accepted, child title, role
CASES = {
    "sow":       ("doctype.sow", "doctype.master_agreement", "doctype.framework_agreement",
                  "Statement of Work Helix", "role.master", "Master Services Agreement Helix"),
    "call_off":  ("doctype.call_off_contract", "doctype.framework_agreement", "doctype.master_agreement",
                  "Call-Off Contract Helix", "role.master", "Framework Agreement Helix"),
    "order_form": ("doctype.order_form", "doctype.framework_agreement", "doctype.master_agreement",
                   "Order Form Helix", "role.master", "Framework Agreement Helix"),
    "variation": ("doctype.variation", "doctype.sow", None, "Variation 2", "role.variation",
                  "Statement of Work Helix Migration"),
    "addendum":  ("doctype.addendum", "doctype.master_agreement", None, "Addendum No. 1",
                  "role.variation", "Master Services Agreement Helix"),
    "ccn":       ("doctype.ccn", "doctype.sow", None, "Change Control Note 4", "role.variation",
                  "Statement of Work Helix Migration"),
    "sla":       ("doctype.sla", "doctype.master_agreement", None, "Service Level Agreement Helix",
                  "role.attachment", "Master Services Agreement Helix"),
    "schedule":  ("doctype.schedule", "doctype.master_agreement", None, "Schedule 2 Helix Services",
                  "role.attachment", "Master Services Agreement Helix"),
}
AMENDMENT_TYPES = {"variation", "addendum", "ccn"}


def _insert(made, cid, dtype, title, sup, start, end, role, ref=None):
    made.append(cid)
    with get_conn() as c:
        c.cursor().execute(
            """INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                   contract_start_date, contract_end_date, resolved_doc_type, resolved_role,
                   type_agreement, parent_agreement_ref)
               VALUES (%s,%s,%s,%s,%s,%s,%s,'refined',%s)""",
            (cid, title, sup, start, end, dtype, role, ref))


def _proposal(cid):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT expected_value, notes FROM proc.bp_extraction_discrepancy
                        WHERE doc_pk_candidate=%s AND issue_type='contract_parent_proposed'
                          AND status='open'""", (cid,))
        return cur.fetchone()


@pytest.fixture()
def world():
    made = []
    yield made
    try:
        with get_conn() as c:
            c.cursor().execute(
                "DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", (made,))
    finally:
        with get_conn() as c:
            c.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", (made,))


def _score(notes):
    return float(re.search(r"score ([\d.]+)", notes).group(1))


@pytest.mark.parametrize("name", sorted(CASES))
def test_declared_reference_and_same_supplier_finds_the_right_parent(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role, ref=P)
    r = CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    assert row and row[0] == P, (name, r)
    assert _score(row[1]) >= 65.0, row
    if name not in AMENDMENT_TYPES and name not in ("sla", "schedule"):
        assert _score(row[1]) >= 92.0, "an exact reference on a SOW/call-off/order form is auto_link band"


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_reference_is_proposed_only_for_the_types_identified_by_supplier_and_structure(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    r = CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    if name in ("sow", "call_off", "order_form"):
        assert row and row[0] == P
    else:
        assert row is None, "an amendment or attachment with no reference is not guessed a parent"
        print(name, "no_ref counters", r["considered"], r["no_candidate"], r["below_threshold"])
        assert r["considered"]["with_structure"] == 1, r
        assert r["below_threshold"] == 1, r


@pytest.mark.parametrize("name", ["sow", "call_off", "order_form"])
def test_only_the_wrong_parent_type_proposes_nothing(world, name):
    ct, _pt, wrong, title, role, _ = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, wrong, "Wrong Type Helix", sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    r = CL.propose_parent_links(contract_id=C)
    assert _proposal(C) is None and r["no_candidate"] == 1


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_reference_naming_a_parent_of_another_supplier_is_not_proposed(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    P, C = f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, f"S-{k}1", "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, f"S-{k}2", "2026-03-01", "2026-09-30", role, ref=P)
    r = CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    # The reference resolves a candidate (step 1 of candidate_parents), but the supplier
    # conflict (tier 1) caps the score below the proposal floor.
    assert row is None
    assert r["considered"]["with_candidates"] == 1, r
    assert r["below_threshold"] == 1, r


@pytest.mark.parametrize("name", ["sow", "call_off", "order_form"])
def test_two_equal_parents_are_contested(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, C = f"S-{k}", f"C-{k}"
    _insert(world, f"P1-{k}", pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, f"P2-{k}", pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    r = CL.propose_parent_links(contract_id=C)
    assert r["contested"] == 1 and r["proposed"] == 0
