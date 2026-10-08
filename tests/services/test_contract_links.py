"""Proposing a contract document's parent, and never linking it.

The failure this file exists to prevent is not a wrong score. It is
contract_succession.py: a scored, unit-tested profile that nothing ever calls.
test_the_runner_is_actually_called is therefore the load-bearing test here.

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/test_contract_links.py -v
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services import contract_links as CL                       # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

ISSUE = "contract_parent_proposed"


def _issue_ids(cur):
    cur.execute("SELECT discrepancy_id FROM proc.bp_extraction_discrepancy WHERE issue_type = %s", (ISSUE,))
    return {r[0] for r in cur.fetchall()}


@pytest.fixture()
def fixture_contracts():
    """A master agreement and a SOW naming it, removed again afterwards.

    propose_parent_links() scores the WHOLE parentless corpus, so a run also
    writes proposals for real contracts this test never created. The snapshot of
    existing proposal ids taken here lets teardown delete exactly the rows the
    test runs added, by id, and nothing else.
    """
    tag = uuid.uuid4().hex[:8].upper()
    msa = f"MSA-{tag}"
    sow = f"SOW-{tag}"
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        before = _issue_ids(cur)
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, contract_start_date,
                    contract_end_date, resolved_doc_type, resolved_role, type_agreement)
               VALUES (%s, 'Master Services Agreement Helix Migration', %s,
                       '2026-01-01', '2027-12-31',
                       'doctype.master_agreement', 'role.master', 'refined')""",
            (msa, f"S-{tag}"),
        )
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, contract_start_date,
                    contract_end_date, resolved_doc_type, resolved_role, type_agreement,
                    parent_agreement_ref)
               VALUES (%s, 'Statement of Work Helix Migration', %s,
                       '2026-03-01', '2026-09-30',
                       'doctype.sow', 'role.master', 'refined', %s)""",
            (sow, f"S-{tag}", msa),
        )
    yield {"msa": msa, "sow": sow, "supplier": f"S-{tag}", "tag": tag}
    with get_conn() as conn:
        cur = conn.cursor()
        added = _issue_ids(cur) - before
        if added:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id = ANY(%s)",
                        (sorted(added),))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)",
                    ([msa, sow],))
        cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", ([msa, sow],))


def _open_proposals(doc_pk):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT expected_value, computed_value, severity, blocks_promotion,
                      source_file, field_name, notes
                 FROM proc.bp_extraction_discrepancy
                WHERE doc_pk_candidate = %s AND issue_type = %s AND status = 'open'""",
            (doc_pk, ISSUE),
        )
        cols = ("expected_value", "computed_value", "severity", "blocks_promotion",
                "source_file", "field_name", "notes")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_the_runner_is_actually_called(fixture_contracts):
    """THE test. contract_succession has a scorer and no runner; this must not.

    A profile nothing calls is indistinguishable from a profile that found
    nothing, and both report zero.
    """
    result = CL.propose_parent_links()
    assert result["considered"]["children"] > 0, (
        "the runner looked at no children at all, so it cannot have scored anything"
    )
    assert result["proposed"] + result["contested"] >= 1, result


def test_the_sow_is_proposed_under_its_master_agreement(fixture_contracts):
    CL.propose_parent_links()
    rows = _open_proposals(fixture_contracts["sow"])
    assert len(rows) == 1, rows
    assert rows[0]["expected_value"] == fixture_contracts["msa"]


def test_a_proposal_never_blocks_promotion_and_is_informational(fixture_contracts):
    CL.propose_parent_links()
    row = _open_proposals(fixture_contracts["sow"])[0]
    assert row["blocks_promotion"] is False
    assert row["severity"] == "info"
    assert row["field_name"] == "parent_contract_id"


def test_nothing_is_linked_without_a_person(fixture_contracts):
    """A proposal is a proposal. parent_contract_id stays untouched."""
    CL.propose_parent_links()
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_contract_id FROM proc.bp_contracts WHERE contract_id = %s",
                    (fixture_contracts["sow"],))
        assert cur.fetchone()[0] is None


def test_running_twice_does_not_stack_proposals(fixture_contracts):
    CL.propose_parent_links()
    CL.propose_parent_links()
    CL.propose_parent_links()
    assert len(_open_proposals(fixture_contracts["sow"])) == 1


def test_two_documents_sharing_a_contract_id_each_keep_their_proposal(fixture_contracts):
    """Review Focus 4. doc_pk_candidate is a value read OUT of a document, so two
    documents can carry the same one; source_file is what separates them. Exactly
    the collision that cost 65 days of findings on bp_sqldb.

    The OTHER document's open proposal is planted BEFORE the runner runs. That
    order is the point: the runner's idempotency SELECT is what must not mistake
    the other document's row for its own. (Planting it afterwards, as an earlier
    draft did, never exercised the SELECT and passed with its source_file clause
    deleted.)
    """
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_extraction_discrepancy
                   (doc_type, source_file, doc_pk_candidate, field_name, issue_type,
                    severity, expected_value, blocks_promotion, notes, status)
               VALUES ('contract', %s, %s, 'parent_contract_id', %s, 'info', %s,
                       false, 'a second document carrying the same contract_id', 'open')""",
            (f"documents/contract/other-{fixture_contracts['tag']}.pdf", sow, ISSUE,
             fixture_contracts["msa"]),
        )
    CL.propose_parent_links()
    rows = _open_proposals(sow)
    assert len(rows) == 2, "this document's proposal was swallowed by the other document's row"
    assert len({r["source_file"] for r in rows}) == 2


def test_the_source_file_is_stored_normalised(fixture_contracts):
    """Needs a raw row whose source_file is spelled untidily: the 'contract:<id>'
    fallback is already canonical, so without one this test cannot fail."""
    from src.services.db import get_conn
    from src.services.extraction.persistence import normalise_source_file
    sow = fixture_contracts["sow"]
    untidy = f" ./documents//contract/{fixture_contracts['tag']}.pdf "
    assert normalise_source_file(untidy) != untidy, "fixture is not actually untidy"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_contract_raw
                   (contract_id, doc_pk_candidate, source_file, raw_payload, pipeline_version)
               VALUES (%s, %s, %s, '{}'::jsonb, 'test-task-10') RETURNING raw_id""",
            (sow, sow, untidy))
        raw_id = cur.fetchone()[0]
    try:
        CL.propose_parent_links()
        row = _open_proposals(sow)[0]
        assert row["source_file"] == normalise_source_file(untidy)
        assert row["source_file"] != untidy
        assert "/" in row["source_file"], "never reduce source_file to a basename"
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contract_raw WHERE raw_id = %s", (raw_id,))


def test_nothing_proposed_is_distinguishable_from_everything_parented():
    """Review Focus 5. An empty screen must not read as success.

    'proposals: 0' invites the reading that every contract has a parent. The
    considered counts are what let a screen tell that apart from 'no contract
    resembled a parent its supplier holds'.
    """
    result = CL.propose_parent_links()
    assert set(result["considered"]) >= {"children", "with_structure", "with_candidates"}
    assert all(isinstance(v, int) for v in result["considered"].values())


def test_a_child_with_no_candidate_parent_proposes_nothing(fixture_contracts):
    from src.services.db import get_conn
    orphan = f"SOW-ORPH-{fixture_contracts['tag']}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, resolved_doc_type,
                    resolved_role, type_agreement)
               VALUES (%s, 'Statement of Work Nothing Above It', 'S-NOBODY',
                       'doctype.sow', 'role.master', 'refined')""",
            (orphan,),
        )
    try:
        result = CL.propose_parent_links()
        assert _open_proposals(orphan) == []
        assert result["no_candidate"] >= 1
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s",
                                  (orphan,))


def test_the_result_counts_scored_but_too_weak_apart_from_never_found():
    """'no_candidate' is "nothing to score"; 'below_threshold' is "scored, too weak"."""
    result = CL.propose_parent_links()
    assert isinstance(result["below_threshold"], int)
    assert "below_threshold" in result and "no_candidate" in result


def test_a_proposal_names_the_decision_band(fixture_contracts):
    """Design section 8 promises the band; a reader needs it to weigh the score."""
    CL.propose_parent_links()
    rows = [r for c in fixture_contracts.values() if isinstance(c, str)
            for r in _open_proposals(c)]
    assert rows, "the fixtures produced no proposal to read"
    assert all(" band " in str(r) for r in rows)


def test_a_structure_with_no_declared_parent_is_not_a_child(fixture_contracts):
    """A master agreement sits under nothing, so it is never proposed a parent."""
    CL.propose_parent_links()
    assert _open_proposals(fixture_contracts["msa"]) == []


def test_the_existing_corpus_supplies_candidate_parents():
    """Task 7 pays off here: 3,016 bp_contract_master rows read as structures.

    Without them the candidate set would be empty until enough contracts had
    been uploaded to form a hierarchy.
    """
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        child = {"contract_id": "X", "resolved_doc_type": "doctype.sow",
                 "supplier_id": None, "contract_start_date": None}
        cur.execute("SELECT supplier_id FROM proc.bp_contract_master "
                    "WHERE contract_type = 'Master Agreement' LIMIT 1")
        row = cur.fetchone()
        assert row, "the corpus has no Master Agreement to be a candidate parent"
        child["supplier_id"] = row[0]
        candidates = CL.candidate_parents(cur, child)
    assert candidates, "no candidate parent came from the 3,051-row corpus"
    assert all(c["resolved_doc_type"] == "doctype.master_agreement" for c in candidates)


def test_confirming_sets_the_parent_and_closes_the_proposal(fixture_contracts):
    CL.propose_parent_links()
    assert CL.confirm(
        fixture_contracts["sow"], fixture_contracts["msa"],
        _open_proposals(fixture_contracts["sow"])[0]["source_file"],
        reviewer="test",
    ) is True
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_contract_id FROM proc.bp_contracts WHERE contract_id = %s",
                    (fixture_contracts["sow"],))
        assert cur.fetchone()[0] == fixture_contracts["msa"]
    assert _open_proposals(fixture_contracts["sow"]) == []


def test_confirming_leaves_the_proposal_resolved_by_the_reviewer(fixture_contracts):
    """The other half of what confirm() promises: the finding is CLOSED, not deleted."""
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    sf = _open_proposals(sow)[0]["source_file"]
    assert CL.confirm(sow, fixture_contracts["msa"], sf, reviewer="rev-1") is True
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT status, resolved_by, resolved_at IS NOT NULL
                         FROM proc.bp_extraction_discrepancy
                        WHERE doc_pk_candidate = %s AND issue_type = %s""", (sow, ISSUE))
        assert cur.fetchall() == [("resolved", "rev-1", True)]


def test_the_notes_say_why_in_words_a_person_can_check(fixture_contracts):
    CL.propose_parent_links()
    notes = _open_proposals(fixture_contracts["sow"])[0]["notes"]
    assert fixture_contracts["msa"] in notes
    assert "confirm" in notes.lower()
    for word in ("reference", "supplier", "term"):
        assert word in notes.lower(), f"the reason omits {word}: {notes}"


# --------------------------------------------------------------------------
# _ref_resolves: the flag that decides whether a non-matching reference is a
# contradiction (True) or a dangling pointer (False). Task 9 reads it; if this
# module never sets it, the CONFLICT path is dead in production.
# --------------------------------------------------------------------------

def _insert_contract(cur, cid, title, supplier, doc_type, **extra):
    cols = {"contract_id": cid, "contract_title": title, "supplier_id": supplier,
            "resolved_doc_type": doc_type, "resolved_role": "role.master",
            "type_agreement": "refined", **extra}
    cur.execute(
        f"INSERT INTO proc.bp_contracts ({', '.join(cols)}) "
        f"VALUES ({', '.join(['%s'] * len(cols))})", tuple(cols.values()))


def _delete_contracts(*ids):
    from src.services.db import get_conn
    with get_conn() as conn:
        conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)",
                              (list(ids),))


def _capture_flags(monkeypatch, wanted):
    """Record the _ref_resolves value the runner hands the scorer, per child.

    Wraps every profile the runner can choose (hierarchy, amendment, attachment),
    since a variation or a schedule is no longer scored by contract_hierarchy.
    """
    from src.services.graph_resolution.profiles import (
        contract_amendment, contract_attachment, contract_hierarchy)
    seen = {}

    def wrap(mod):
        real = mod.score

        def spy(child, parent):
            if child.get("contract_id") in wanted:
                seen[child["contract_id"]] = child.get("_ref_resolves", "ABSENT")
            return real(child, parent)

        monkeypatch.setattr(mod, "score", spy)

    for mod in (contract_hierarchy, contract_amendment, contract_attachment):
        wrap(mod)
    return seen


def test_ref_resolves_is_true_when_the_reference_names_a_real_contract(
        fixture_contracts, monkeypatch):
    """A child naming a contract that exists, but not this candidate: CONFLICT territory."""
    from src.services.db import get_conn
    tag = fixture_contracts["tag"]
    other_real = f"MSA-REAL-{tag}"
    child = f"SOW-REALREF-{tag}"
    with get_conn() as conn:
        cur = conn.cursor()
        _insert_contract(cur, other_real, "Some Other Agreement", f"S-OTHER-{tag}",
                         "doctype.master_agreement")
        _insert_contract(cur, child, "Statement of Work Helix Migration",
                         fixture_contracts["supplier"], "doctype.sow",
                         parent_agreement_ref=other_real,
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30")
    try:
        seen = _capture_flags(monkeypatch, {child})
        CL.propose_parent_links()
        assert seen.get(child) is True, seen
    finally:
        _delete_contracts(other_real, child)


def test_ref_resolves_is_false_when_the_reference_names_nothing(
        fixture_contracts, monkeypatch):
    from src.services.db import get_conn
    tag = fixture_contracts["tag"]
    child = f"SOW-DANGLE-{tag}"
    with get_conn() as conn:
        _insert_contract(conn.cursor(), child, "Statement of Work Helix Migration",
                         fixture_contracts["supplier"], "doctype.sow",
                         parent_agreement_ref=f"MSA-9999-{tag}",
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30")
    try:
        seen = _capture_flags(monkeypatch, {child})
        CL.propose_parent_links()
        assert seen.get(child) is False, seen
    finally:
        _delete_contracts(child)


def test_ref_resolves_is_false_when_the_child_claims_no_reference(
        fixture_contracts, monkeypatch):
    """No claim at all. The flag is moot (the signal reads MISSING either way) but it
    must still be set, never left absent, so the answer never depends on a default."""
    from src.services.db import get_conn
    tag = fixture_contracts["tag"]
    child = f"SOW-NOREF-{tag}"
    with get_conn() as conn:
        _insert_contract(conn.cursor(), child, "Statement of Work Helix Migration",
                         fixture_contracts["supplier"], "doctype.sow",
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30")
    try:
        seen = _capture_flags(monkeypatch, {child})
        CL.propose_parent_links()
        assert seen.get(child) is False, seen
    finally:
        _delete_contracts(child)


def test_a_real_but_wrong_reference_is_a_conflict_and_is_not_proposed(fixture_contracts):
    """End to end through the runner: the CONFLICT path is alive, not just the flag."""
    from src.services.db import get_conn
    tag = fixture_contracts["tag"]
    other_real = f"MSA-REAL2-{tag}"
    child = f"SOW-CONFLICT-{tag}"
    with get_conn() as conn:
        cur = conn.cursor()
        _insert_contract(cur, other_real, "Some Other Agreement", f"S-OTHER2-{tag}",
                         "doctype.master_agreement")
        # Dates and title that fit the candidate: read as a dangling reference this
        # child scores 75.59 and WOULD be proposed; read as a real CONFLICT it scores
        # 45.00 and is not. Only the _ref_resolves flag separates the two outcomes.
        _insert_contract(cur, child, "Statement of Work Helix Migration",
                         fixture_contracts["supplier"], "doctype.sow",
                         parent_agreement_ref=other_real,
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30")
    try:
        CL.propose_parent_links()
        assert _open_proposals(child) == [], (
            "a child naming a different REAL contract was proposed the candidate anyway")
    finally:
        _delete_contracts(other_real, child)


def test_a_dangling_reference_is_still_proposed_and_named_in_the_notes(fixture_contracts):
    """A pointer to nothing is no information, not a contradiction -- and the notes
    must say it was ignored, because the scored output reads MISSING for both a
    dangling reference and an absent one."""
    from src.services.db import get_conn
    tag = fixture_contracts["tag"]
    child = f"SOW-DANGLE2-{tag}"
    dangling = f"MSA-9999-{tag}"
    with get_conn() as conn:
        _insert_contract(conn.cursor(), child, "Statement of Work Helix Migration",
                         fixture_contracts["supplier"], "doctype.sow",
                         parent_agreement_ref=dangling,
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30")
    try:
        CL.propose_parent_links()
        rows = _open_proposals(child)
        assert len(rows) == 1, rows
        assert dangling in rows[0]["notes"], rows[0]["notes"]
        assert "no contract matches" in rows[0]["notes"]
    finally:
        _delete_contracts(child)


def test_an_absent_reference_is_not_described_as_dangling(fixture_contracts):
    CL.propose_parent_links()
    # the fixture SOW names a REAL contract (the msa), so nothing is dangling
    notes = _open_proposals(fixture_contracts["sow"])[0]["notes"]
    assert "no contract matches" not in notes


def test_a_dismissed_proposal_still_occupies_the_slot(fixture_contracts):
    """The index's partial clause is coalesce(status,'open') <> 'resolved', so a
    dismissed row blocks the slot. Re-running must neither raise a unique violation
    nor resurrect it."""
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    sf = _open_proposals(sow)[0]["source_file"]
    with get_conn() as conn:
        conn.cursor().execute(
            """UPDATE proc.bp_extraction_discrepancy SET status = 'ignored'
                WHERE doc_pk_candidate = %s AND issue_type = %s""", (sow, ISSUE))
    CL.propose_parent_links()          # would raise on a status='open' idempotency test
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT status FROM proc.bp_extraction_discrepancy
                        WHERE doc_pk_candidate = %s AND issue_type = %s""", (sow, ISSUE))
        assert [r[0] for r in cur.fetchall()] == ["ignored"]


# --------------------------------------------------------------------------
# The third headline relationship: a variation against the contract it changes.
# Its parent cannot be identified by TYPE (default_parent_type is NULL), only by
# the reference it carries, so candidate_parents takes a different path for it.
# --------------------------------------------------------------------------

def _candidates_for(child):
    from src.services.db import get_conn
    with get_conn() as conn:
        return CL.candidate_parents(conn.cursor(), child)


def _variation_child(fx, doc_type="doctype.variation", suffix="VAR", **extra):
    cid = f"{suffix}-{fx['tag']}"
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cols = {"contract_start_date": "2026-03-01", "contract_end_date": "2026-09-30",
                "parent_agreement_ref": fx["msa"], **extra}
        _insert_contract(cur, cid, "Variation Helix Migration", fx["supplier"], doc_type,
                         **cols)
        cur.execute("UPDATE proc.bp_contracts SET resolved_role = 'role.variation' "
                    "WHERE contract_id = %s", (cid,))
    return cid


@pytest.mark.parametrize("doc_type,suffix", [
    ("doctype.variation", "VAR"), ("doctype.addendum", "ADD"), ("doctype.ccn", "CCN")])
def test_a_variation_is_proposed_the_contract_it_names(fixture_contracts, doc_type, suffix):
    """variation, addendum and CCN are all role.variation and take the same path."""
    cid = _variation_child(fixture_contracts, doc_type, suffix)
    try:
        CL.propose_parent_links()
        rows = _open_proposals(cid)
        assert len(rows) == 1, "a variation was never proposed a parent"
        assert rows[0]["expected_value"] == fixture_contracts["msa"]
    finally:
        _delete_contracts(cid)


def test_a_variation_is_not_proposed_a_variation_as_its_parent(fixture_contracts):
    """A variation amends a contract, not another variation."""
    other = f"VAR-OTHER-{fixture_contracts['tag']}"
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        _insert_contract(cur, other, "Variation Helix Migration", fixture_contracts["supplier"],
                         "doctype.variation", contract_start_date="2026-01-01",
                         contract_end_date="2027-12-31")
        cur.execute("UPDATE proc.bp_contracts SET resolved_role = 'role.variation' "
                    "WHERE contract_id = %s", (other,))
    cid = _variation_child(fixture_contracts)
    try:
        ids = {c["contract_id"] for c in _candidates_for(
            {"contract_id": cid, "resolved_doc_type": "doctype.variation",
             "resolved_role": "role.variation", "supplier_id": fixture_contracts["supplier"]})}
        assert other not in ids, ids
        assert fixture_contracts["msa"] in ids, ids
    finally:
        _delete_contracts(cid, other)


def test_a_sow_still_considers_only_master_agreements(fixture_contracts):
    """The variation path must not have widened the exact path for everyone."""
    fw = f"FW-{fixture_contracts['tag']}"
    from src.services.db import get_conn
    with get_conn() as conn:
        _insert_contract(conn.cursor(), fw, "Framework Helix Migration",
                         fixture_contracts["supplier"], "doctype.framework_agreement")
    try:
        ids = {c["contract_id"] for c in _candidates_for(
            {"contract_id": fixture_contracts["sow"], "resolved_doc_type": "doctype.sow",
             "resolved_role": "role.master", "supplier_id": fixture_contracts["supplier"]})}
        assert fixture_contracts["msa"] in ids
        assert fw not in ids, "a SOW was offered a framework agreement as a parent"
    finally:
        _delete_contracts(fw)


# --------------------------------------------------------------------------
# "Unparented" means the pointer does not RESOLVE, not that it is absent: a dangling
# pointer is not a parent. (Parent-side counts from bp_contract_master, which supplies
# candidate parents, not children: 1,490 NULL / 0 resolving / 1,561 dangling. Children
# come from bp_contracts, which has 0 rows, so the impact today is zero.) Each case
# below is independently provable.
# --------------------------------------------------------------------------

def _sow_with_pointer(fx, label, pointer):
    cid = f"SOW-{label}-{fx['tag']}"
    from src.services.db import get_conn
    with get_conn() as conn:
        _insert_contract(conn.cursor(), cid, "Statement of Work Helix Migration",
                         fx["supplier"], "doctype.sow",
                         contract_start_date="2026-03-01", contract_end_date="2026-09-30",
                         parent_agreement_ref=fx["msa"], parent_contract_id=pointer)
    return cid


def test_a_child_with_a_dangling_parent_pointer_is_considered(fixture_contracts):
    cid = _sow_with_pointer(fixture_contracts, "DANG", f"GHOST-{fixture_contracts['tag']}")
    try:
        CL.propose_parent_links()
        rows = _open_proposals(cid)
        assert len(rows) == 1, "a dangling pointer excluded the child from scoring"
        assert rows[0]["expected_value"] == fixture_contracts["msa"]
    finally:
        _delete_contracts(cid)


def test_a_child_whose_pointer_resolves_is_skipped(fixture_contracts):
    cid = _sow_with_pointer(fixture_contracts, "RES", fixture_contracts["msa"])
    try:
        CL.propose_parent_links()
        assert _open_proposals(cid) == []
    finally:
        _delete_contracts(cid)


def test_a_child_whose_pointer_resolves_in_the_corpus_table_is_skipped(fixture_contracts):
    """A parent may live in bp_contract_master rather than bp_contracts."""
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT contract_id FROM proc.bp_contract_master LIMIT 1")
        row = cur.fetchone()
    assert row, "bp_contract_master is empty"
    cid = _sow_with_pointer(fixture_contracts, "MRES", row[0])
    try:
        CL.propose_parent_links()
        assert _open_proposals(cid) == []
    finally:
        _delete_contracts(cid)


def test_a_child_with_a_null_pointer_is_still_considered(fixture_contracts):
    cid = _sow_with_pointer(fixture_contracts, "NULLP", None)
    try:
        CL.propose_parent_links()
        assert len(_open_proposals(cid)) == 1
    finally:
        _delete_contracts(cid)


def test_after_confirm_the_child_resolves_and_drops_out(fixture_contracts):
    """The exit condition, end to end: confirm writes a resolving id, so the next
    pass proposes nothing further for that child."""
    cid = _sow_with_pointer(fixture_contracts, "EXIT", f"GHOST-{fixture_contracts['tag']}")
    try:
        CL.propose_parent_links()
        sf = _open_proposals(cid)[0]["source_file"]
        assert CL.confirm(cid, fixture_contracts["msa"], sf, reviewer="test") is True
        before = CL.propose_parent_links()
        assert cid not in {d["contract_id"] for d in before["details"]}
        assert _open_proposals(cid) == []
    finally:
        _delete_contracts(cid)


def test_a_variation_with_a_dangling_parent_contract_id_and_a_resolvable_reference_is_proposed(
        fixture_contracts):
    """parent_contract_id is the realistic field for an amendment. A dangling value
    there used to exclude the variation before it reached candidate_parents, even
    when another pointer on the same document named a real contract. Proposed now."""
    cid = _variation_child(fixture_contracts, "doctype.variation", "VARPC",
                           parent_contract_id=f"{fixture_contracts['msa']}-OLDREV")
    try:
        CL.propose_parent_links()
        rows = _open_proposals(cid)
        assert len(rows) == 1, rows
        assert rows[0]["expected_value"] == fixture_contracts["msa"]
    finally:
        _delete_contracts(cid)


def test_a_variation_whose_only_pointer_dangles_is_considered_but_not_proposed(
        fixture_contracts, monkeypatch):
    """Considered, then honestly below the gate: with no resolvable reference and no
    parent type to lean on, supplier + term + title score 38.46 against 65. The
    filter lets it reach the scorer; the scorer declines. The spy proves THIS child
    was scored (the fixture SOW alone satisfies any global counter)."""
    cid = _variation_child(fixture_contracts, "doctype.variation", "VARONLY",
                           parent_agreement_ref=None,
                           parent_contract_id=f"GHOST-{fixture_contracts['tag']}")
    try:
        seen = _capture_flags(monkeypatch, {cid})
        CL.propose_parent_links()
        assert cid in seen, "this variation was never handed to the scorer"
        assert seen[cid] is False
        assert _open_proposals(cid) == []
    finally:
        _delete_contracts(cid)


def test_a_pointer_differing_only_in_case_and_spacing_resolves(fixture_contracts):
    """Filter and scorer agree: 'msa 4417' IS 'MSA-4417', so the child is parented."""
    ptr = fixture_contracts["msa"].lower().replace("-", " ")
    cid = _sow_with_pointer(fixture_contracts, "CASE", ptr)
    try:
        CL.propose_parent_links()
        assert _open_proposals(cid) == []
    finally:
        _delete_contracts(cid)


# --------------------------------------------------------------------------
# confirm() links only a parent that was actually proposed.
# --------------------------------------------------------------------------

def _parent_of(cid):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_contract_id FROM proc.bp_contracts WHERE contract_id = %s",
                    (cid,))
        return cur.fetchone()[0]


def test_confirm_of_a_proposed_parent_links_it(fixture_contracts):
    CL.propose_parent_links()
    sf = _open_proposals(fixture_contracts["sow"])[0]["source_file"]
    assert CL.confirm(fixture_contracts["sow"], fixture_contracts["msa"], sf) is True
    assert _parent_of(fixture_contracts["sow"]) == fixture_contracts["msa"]


def test_confirm_of_a_parent_never_proposed_links_nothing(fixture_contracts):
    CL.propose_parent_links()
    sf = _open_proposals(fixture_contracts["sow"])[0]["source_file"]
    assert CL.confirm(fixture_contracts["sow"], "MSA-NEVER-PROPOSED", sf) is False
    assert _parent_of(fixture_contracts["sow"]) is None
    assert len(_open_proposals(fixture_contracts["sow"])) == 1, "proposal must stay open"


def test_confirm_with_no_proposal_at_all_links_nothing(fixture_contracts):
    assert CL.confirm(fixture_contracts["sow"], fixture_contracts["msa"],
                      "contract:nothing-here") is False
    assert _parent_of(fixture_contracts["sow"]) is None


def test_confirm_of_a_dismissed_proposal_links_nothing(fixture_contracts):
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    sf = _open_proposals(sow)[0]["source_file"]
    with get_conn() as conn:
        conn.cursor().execute(
            """UPDATE proc.bp_extraction_discrepancy SET status = 'ignored'
                WHERE doc_pk_candidate = %s AND issue_type = %s""", (sow, ISSUE))
    assert CL.confirm(sow, fixture_contracts["msa"], sf) is False
    assert _parent_of(sow) is None


def test_confirm_claims_the_proposal_before_linking(fixture_contracts):
    """With the proposal already resolved (someone else's confirm or a close got
    there first) the claim matches zero rows, so confirm() returns False and links
    nothing. The claim is the atomic step; the link must come after it."""
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    sf = _open_proposals(sow)[0]["source_file"]
    with get_conn() as conn:
        conn.cursor().execute(
            """UPDATE proc.bp_extraction_discrepancy SET status = 'resolved',
                      resolved_by = 'someone-else', resolved_at = now()
                WHERE doc_pk_candidate = %s AND issue_type = %s""", (sow, ISSUE))
    assert CL.confirm(sow, fixture_contracts["msa"], sf) is False
    assert _parent_of(sow) is None


# ---------------------------------------------------------------------------
# Final review, Important 1: a re-read must REFRESH the proposal, not skip it.
# ---------------------------------------------------------------------------

def test_a_better_parent_refreshes_the_open_proposal(fixture_contracts):
    """The queue must not keep naming the parent the FIRST read chose.

    A contract document is re-read (routine in this product) and the second read
    extracts a reference that points somewhere else. The open finding is what a
    person reads and what confirm() validates against, so if it still names the
    superseded parent, confirming it links the wrong contract.
    """
    from src.services.db import get_conn
    sow, msa, tag = fixture_contracts["sow"], fixture_contracts["msa"], fixture_contracts["tag"]
    msa2 = f"MSA2-{tag}"
    try:
        # First read: the SOW names msa, so msa is proposed.
        CL.propose_parent_links()
        first = _open_proposals(sow)
        assert len(first) == 1 and first[0]["expected_value"] == msa, first

        # Second read of the same document: it now names msa2 instead.
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role, type_agreement)
                   VALUES (%s, 'Master Services Agreement Helix Migration Phase Two', %s,
                           '2026-01-01', '2027-12-31',
                           'doctype.master_agreement', 'role.master', 'refined')""",
                (msa2, fixture_contracts["supplier"]),
            )
            cur.execute("UPDATE proc.bp_contracts SET parent_agreement_ref = %s "
                        "WHERE contract_id = %s", (msa2, sow))

        CL.propose_parent_links()
        after = _open_proposals(sow)
        assert len(after) == 1, f"the re-read stacked or lost the proposal: {after}"
        assert after[0]["expected_value"] == msa2, (
            "the open proposal still names the parent the first read chose; a person "
            "confirming it would link the superseded contract")
    finally:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = %s", (msa2,))
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (msa2,))


def test_a_refreshed_proposal_keeps_one_row_per_document(fixture_contracts):
    """Refreshing must go through the open key, so a second pass cannot raise
    UniqueViolation out of the middle of a run (get_conn is AUTOCOMMIT: the rows
    written before the raise would already be committed)."""
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    CL.propose_parent_links()        # must not raise
    assert len(_open_proposals(sow)) == 1


# ---------------------------------------------------------------------------
# Final review, Important 3: one contract in both tables is ONE candidate.
# ---------------------------------------------------------------------------

def test_a_parent_in_both_tables_is_not_contested_with_itself():
    """A registered contract whose PDF is later uploaded exists in BOTH
    proc.bp_contract_master (the 3,051-row register) and proc.bp_contracts (the
    uploaded document). It is one contract. Counted twice it ties with itself, so
    the person reads "contested ... Other candidates: C02397" about a single
    unambiguous parent, and computed_value lists the proposed parent as its own
    alternative.

    The register is a FOREIGN table (no inserts), so the duplicate is made the
    way it happens in production: an existing corpus row, C02397 -- the only
    contract its supplier S4702 has -- is uploaded as a document.
    """
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:8].upper()
    parent, supplier, child = "C02397", "S4702", f"SOW-DUP-{tag}"
    with get_conn() as conn:
        cur = conn.cursor()
        before = _issue_ids(cur)
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role, type_agreement)
                   VALUES (%s, 'Drive 24/365 Bandwidth', %s, '2021-02-05', '2024-02-05',
                           'doctype.master_agreement', 'role.master', 'refined')""",
                (parent, supplier),
            )
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role,
                        type_agreement, parent_agreement_ref)
                   VALUES (%s, 'Statement of Work Drive 24/365 Bandwidth', %s,
                           '2022-01-01', '2023-01-01',
                           'doctype.sow', 'role.master', 'refined', %s)""",
                (child, supplier, parent),
            )
        CL.propose_parent_links()
        rows = _open_proposals(child)
        assert len(rows) == 1, rows
        assert rows[0]["expected_value"] == parent, rows[0]
        assert parent not in (rows[0]["computed_value"] or ""), (
            f"the proposed parent is listed as its own alternative: "
            f"computed_value={rows[0]['computed_value']!r}")
        assert "contested" not in (rows[0]["notes"] or ""), (
            f"one candidate, counted twice, was presented as a contested choice: "
            f"{rows[0]['notes']!r}")
    finally:
        with get_conn() as conn:
            cur = conn.cursor()
            added = _issue_ids(cur) - before
            if added:
                cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                            "WHERE discrepancy_id = ANY(%s)", (sorted(added),))
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate = ANY(%s)", ([parent, child],))
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)",
                        ([parent, child],))


# ---------------------------------------------------------------------------
# Found by Task 11's LIVE upload, 2026-10-03. An order form that printed
# "Framework Agreement No. FA-2026-0042", whose framework was sitting in
# proc.bp_contracts under exactly that id, got NO proposal: candidate_parents
# returns [] the moment supplier_id is empty, and on that document the supplier
# extractor had read the words "Framework Agreement No" as the supplier name.
#
# So the strongest signal this layer has -- an exact reference that resolves to a
# real contract -- was never looked at, because a different field was wrong. The
# reference is evidence in its own right; the supplier is a way of NARROWING a
# search, not a precondition for searching.
# ---------------------------------------------------------------------------

def test_a_named_contract_is_a_candidate_even_when_the_supplier_does_not_match():
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:8].upper()
    fa, of = f"FA-{tag}", f"OF-{tag}"
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role, type_agreement)
                   VALUES (%s, 'Framework Agreement Marketing Services', 'NexaSpark Marketing Ltd.',
                           '2026-01-05', '2029-01-05',
                           'doctype.framework_agreement', 'role.framework', 'refined')""",
                (fa,),
            )
            child = {"contract_id": of, "resolved_doc_type": "doctype.order_form",
                     "supplier_id": "Framework Agreement No",   # what the live run read
                     "framework_ref": fa, "parent_agreement_ref": None,
                     "contract_start_date": None, "contract_end_date": None,
                     "contract_title": "Order Form SEO Retainer"}
            got = CL.candidate_parents(cur, child)
        assert [c["contract_id"] for c in got] == [fa], (
            "the framework the document names by number was not even considered")
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (fa,))


def test_a_named_contract_is_not_duplicated_when_the_supplier_matches_too():
    """Reached by both paths, counted once."""
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:8].upper()
    fa, of, sup = f"FA-{tag}", f"OF-{tag}", f"S-{tag}"
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role, type_agreement)
                   VALUES (%s, 'Framework Agreement Marketing Services', %s,
                           '2026-01-05', '2029-01-05',
                           'doctype.framework_agreement', 'role.framework', 'refined')""",
                (fa, sup),
            )
            child = {"contract_id": of, "resolved_doc_type": "doctype.order_form",
                     "supplier_id": sup, "framework_ref": fa,
                     "parent_agreement_ref": None, "contract_start_date": None,
                     "contract_end_date": None, "contract_title": "Order Form"}
            got = CL.candidate_parents(cur, child)
        assert [c["contract_id"] for c in got] == [fa], got
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (fa,))


def test_a_named_contract_of_the_wrong_structure_is_still_not_a_candidate():
    """A reference is evidence, not an override: a SOW does not sit under an
    order form, however explicitly it names one."""
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:8].upper()
    wrong = f"OF-{tag}"
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role, type_agreement)
                   VALUES (%s, 'Order Form', 'S-X', '2026-01-05', '2029-01-05',
                           'doctype.order_form', 'role.master', 'agreed')""",
                (wrong,),
            )
            child = {"contract_id": f"SOW-{tag}", "resolved_doc_type": "doctype.sow",
                     "supplier_id": None, "parent_agreement_ref": wrong,
                     "framework_ref": None, "contract_start_date": None,
                     "contract_end_date": None, "contract_title": "SOW"}
            got = CL.candidate_parents(cur, child)
        assert got == [], got
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (wrong,))


_CORROBORATING_KEYS = {"buyer_org_id", "currency", "payment_terms", "governing_law",
                       "contract_signatory_name", "buyer_signatory_name", "cost_centre_id"}


def test_candidate_rows_carry_the_corroborating_fields_from_both_tables(fixture_contracts):
    """Review Focus 4: bp_contract_master has no buyer_signatory_name; it must not crash."""
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        child = {"contract_id": "X-NONE", "resolved_doc_type": "doctype.sow",
                 "supplier_id": fixture_contracts["supplier"]}
        rows = CL.candidate_parents(cur, child)
        master = CL._fetch(cur, "proc.bp_contract_master", "supplier_id = %s", ("nobody",))
        # A real bp_contract_master candidate, so the master branch of _fetch is exercised.
        cur.execute("SELECT supplier_id FROM proc.bp_contract_master "
                    "WHERE contract_type = 'Master Agreement' AND supplier_id IS NOT NULL LIMIT 1")
        found = cur.fetchone()
        assert found, "bp_testdb has no Master Agreement row in bp_contract_master"
        master_rows = CL.candidate_parents(cur, {
            "contract_id": "X-NONE", "resolved_doc_type": "doctype.sow",
            "supplier_id": found[0]})
    assert any(r["contract_id"] == fixture_contracts["msa"] for r in rows)
    for r in rows:
        assert _CORROBORATING_KEYS <= set(r)
    assert master == []
    assert master_rows, "expected at least one candidate from bp_contract_master"
    for r in master_rows:
        assert _CORROBORATING_KEYS <= set(r)


def test_a_proposal_for_an_addendum_says_amend_and_reports_its_link_type():
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:6].upper()
    sup, sow, add = f"S-{tag}", f"SOW-{tag}", f"ADD-{tag}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                contract_start_date, contract_end_date, resolved_doc_type, resolved_role, type_agreement)
            VALUES (%s,'Statement of Work Helix',%s,'2026-01-01','2027-12-31','doctype.sow','role.master','refined')""",
                    (sow, sup))
        cur.execute("""INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                contract_start_date, contract_end_date, resolved_doc_type, resolved_role, type_agreement,
                parent_agreement_ref)
            VALUES (%s,'Addendum No. 1',%s,'2026-03-01','2026-09-30','doctype.addendum','role.variation','refined',%s)""",
                    (add, sup, sow))
    try:
        result = CL.propose_parent_links(contract_id=add)
        assert result["details"][0]["link_type"] == "amends"
        assert result["details"][0]["profile"] == "contract_amendment"
        row = _open_proposals(add)[0]
        assert row["expected_value"] == sow and "appears to amend contract" in row["notes"]
    finally:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", ([add, sow],))
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", ([add, sow],))


def test_link_type_of_reads_the_stored_kind_of_a_contract():
    from src.services.db import get_conn
    add = f"ADD-{uuid.uuid4().hex[:6].upper()}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                contract_start_date, contract_end_date, resolved_doc_type, resolved_role, type_agreement)
            VALUES (%s,'Addendum No. 1','S-LT','2026-03-01','2026-09-30','doctype.addendum','role.variation','refined')""",
                    (add,))
    try:
        assert CL.link_type_of(add) == "amends"
        assert CL.link_type_of("NO-SUCH-ID") is None
    finally:
        _delete_contracts(add)
