"""The extraction run against bp_testdb, with the model stubbed by canned ChunkResults.

Needs PROCWISE_TEST_LIVE_DB=1. Each test registers its own document (text stored, so no S3)
and its own run; policies land under the Unassigned (GEN) prefix. Versions are immutable,
so nothing is cleaned up -- the same convention as test_repo_live.
"""
import hashlib
import os
import uuid

import pytest

from repositories import agent_policy_repo as repo
from services.agent_policy import extraction_run as X
from services.agent_policy import extractor, matching
from services.agent_policy import run_store as store
from services.agent_policy.extraction_schema import (ChunkResult, Example, ExampleValue, InputSpec,
                                                     NotEnforceable, ProposedPolicy, Rule)

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

V1 = """1. Refunds
1.1 Refunds above 500 dollars need approval from the Finance Manager before they are issued.
1.2 Credits to a customer account above 200 dollars must be reported to the Finance team.
2. Large refunds
2.1 Refunds above 500 dollars need Finance Manager approval, and refunds above 10000 dollars are not allowed at all.
3. Conduct
3.1 Staff should always be courteous and helpful to customers.
"""

# 1.1 changed (500 -> 750), 1.2 removed, 1.3 added; 2.1 and 3.1 as before.
V2 = """1. Refunds
1.1 Refunds above 750 dollars need approval from the Finance Manager before they are issued.
1.3 Payments to a new supplier above 1000 dollars need approval from the CFO.
2. Large refunds
2.1 Refunds above 500 dollars need Finance Manager approval, and refunds above 10000 dollars are not allowed at all.
3. Conduct
3.1 Staff should always be courteous and helpful to customers.
"""


def _p(ref, excerpt, outcome, threshold, name, deciders=("Finance Manager",), notify=()):
    return ProposedPolicy(
        name=name, category="Approval", business_area="Unassigned", sub_area="General",
        situation=f"The agent is about to {name.lower()}.", match="all",
        rules=[Rule(field="args.amount", op="gt", value_number=threshold)],
        outcome=outcome, outcome_phrase="", deciders=list(deciders) if outcome == "approve" else [],
        notify=list(notify), reference=ref, excerpt=excerpt,
        examples=[Example(values=[ExampleValue(field="args.amount", value_number=threshold + 1)],
                          expected=outcome),
                  Example(values=[ExampleValue(field="args.amount", value_number=threshold)], expected="none")],
        checkpoint="tool.call.before", action_tools=["refund.issue"], action_plain="issuing a refund",
        inputs=[InputSpec(name="Refund amount", field="args.amount", type="number", is_amount=True)],
        reason_code="over_limit", message_for_agent=f"{name} needs a check.", owner="Finance Director")


P11_V1 = _p("1.1", "Refunds above 500 dollars need approval from the Finance Manager", "approve", 500,
            "Issue a refund over 500")
P12 = _p("1.2", "Credits to a customer account above 200 dollars must be reported to the Finance team",
         "notify", 200, "Credit an account over 200", notify=("Finance team",))
P21_APPROVE = _p("2.1", "Refunds above 500 dollars need Finance Manager approval", "approve", 500,
                 "Issue a large refund")
P21_BLOCK = _p("2.1", "refunds above 10000 dollars are not allowed at all", "block", 10000,
               "Issue a very large refund")
NE31 = NotEnforceable(reference="3.1", excerpt="Staff should always be courteous and helpful to customers.",
                      reason="Courtesy is not something an orchestrator can check before an action.")
P11_V2 = _p("1.1", "Refunds above 750 dollars need approval from the Finance Manager", "approve", 750,
            "Issue a refund over 500")
P13 = _p("1.3", "Payments to a new supplier above 1000 dollars need approval from the CFO", "approve", 1000,
         "Pay a new supplier over 1000", deciders=("CFO",))

CANNED_V1 = [P11_V1, P12, P21_APPROVE, P21_BLOCK]
CANNED_V2 = [P11_V2, P13, P21_APPROVE, P21_BLOCK]


@pytest.fixture
def conn():
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def actor():
    return f"extraction-test-{uuid.uuid4().hex[:8]}"


def _document(conn, *texts):
    """A document whose versions' text is already parsed (no S3 read)."""
    tag = uuid.uuid4().hex[:10]
    cur = conn.cursor()
    cur.execute("INSERT INTO proc.bp_policy_document (title, match_name, latest_version, created_by)"
                " VALUES (%s, %s, %s, 'test') RETURNING document_id",
                (f"Run test {tag}", f"run test {tag}", len(texts)))
    doc_id = cur.fetchone()[0]
    for n, text in enumerate(texts, start=1):
        cur.execute("INSERT INTO proc.bp_policy_document_version (document_id, version, filename, s3_key,"
                    " byte_size, content_hash, parsed_text, parsed_at, uploaded_by)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,now(),'test')",
                    (doc_id, n, f"run-test-{tag}.txt", f"agent-policy-documents/uploads/test/{tag}-{n}.txt",
                     len(text), hashlib.sha256(f"{tag}{n}{text}".encode()).hexdigest(), text))
    return doc_id


def _stub(monkeypatch, canned, not_enforceable=(NE31,), fail_refs=()):
    seen = []

    def fake(sections, *, document, taxonomy, registry, call=None):
        refs = [s.get("reference") for s in sections]
        seen.append(refs)
        if any(r in fail_refs for r in refs):
            raise extractor.ExtractionError("the model did not return a usable answer")
        return ChunkResult(policies=[p for p in canned if p.reference in refs],
                           not_enforceable=[n for n in not_enforceable if n.reference in refs])

    monkeypatch.setattr(extractor, "extract_chunk", fake)
    monkeypatch.setattr(extractor, "fix_policy", _no_model)
    return seen


def _no_model(*a, **k):
    raise AssertionError("the model must not be called")


def _run(conn, doc_id, version, actor):
    run = store.create(conn, kind="extract", request={"documents": [{"documentId": doc_id, "version": version}]},
                       actor=actor)

    def emit(kind, payload, **fields):
        return store.append_item(conn, run["run_id"], kind=kind, payload=payload, **fields)

    counts = X.run_extract(conn, run, emit)
    return counts, store.get(conn, run["run_id"])["items"]


def _policies(conn, doc_id):
    cur = conn.cursor()
    cur.execute("SELECT policy_key, status, live_version, latest_version, source_reference, source_split"
                " FROM proc.bp_agent_policy WHERE source_document_id = %s ORDER BY policy_key", (doc_id,))
    cols = ("key", "status", "live", "latest", "ref", "split")
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _version_count(conn, doc_id):
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_agent_policy_version v JOIN proc.bp_agent_policy p"
                " ON p.policy_key = v.policy_key WHERE p.source_document_id = %s", (doc_id,))
    return cur.fetchone()[0]


def _by_decision(items):
    out = {}
    for i in items:
        out.setdefault(i["decision"] or i["kind"], []).append(i)
    return out


def test_first_extraction_creates_drafts_and_streams_items(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    counts, items = _run(conn, doc, 1, actor)
    assert counts == {"documents": 1, "chunks": 1, "policies": 4, "new": 4, "changed": 0, "unchanged": 0,
                      "proposedRetire": 0, "notEnforceable": 1, "errors": 0}
    assert [i["seq"] for i in items] == list(range(1, len(items) + 1))
    ne = [i for i in items if i["kind"] == "not_enforceable"]
    assert len(ne) == 1 and ne[0]["reference"] == "3.1" and "orchestrator" in ne[0]["payload"]["reason"]
    pols = [i for i in items if i["kind"] == "policy"]
    assert [i["decision"] for i in pols] == ["new"] * 4 and all(i["saved_version"] == 1 for i in pols)
    assert {i["policy_key"] for i in pols} == {p["key"] for p in _policies(conn, doc)}


def test_tiered_clause_is_two_new_policies_with_one_reference_and_different_splits(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    tiers = [p for p in _policies(conn, doc) if p["ref"] == "2.1"]
    assert len(tiers) == 2
    assert {p["split"] for p in tiers} == {"approve|gt:500", "block|gt:10000"}


def test_reextracting_unchanged_document_writes_nothing(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    before, versions_before = _policies(conn, doc), _version_count(conn, doc)
    counts, items = _run(conn, doc, 1, actor)
    assert _policies(conn, doc) == before                     # no new policy, no pointer moved
    assert _version_count(conn, doc) == versions_before       # no new version
    assert counts["new"] == counts["changed"] == counts["proposedRetire"] == 0 and counts["unchanged"] == 4
    pols = [i for i in items if i["kind"] == "policy"]
    assert {i["decision"] for i in pols} == {"unchanged"} and all(i["saved_version"] is None for i in pols)


def test_bad_chunk_is_an_error_item_and_run_continues(conn, actor, monkeypatch):
    seen = _stub(monkeypatch, CANNED_V1, fail_refs=("1.1",))
    monkeypatch.setattr(X, "chunk_sections", lambda sections: [[s] for s in sections])  # one section a chunk
    doc = _document(conn, V1)
    counts, items = _run(conn, doc, 1, actor)
    assert len(seen) == len(X.split_sections(V1))              # every chunk was still tried
    errors = [i for i in items if i["kind"] == "error"]
    assert len(errors) == 1 and errors[0]["payload"]["references"] == ["1.1"] and "1.1" in errors[0]["payload"]["message"]
    assert counts["errors"] == 1 and counts["new"] == 3        # 1.2 and both tiers of 2.1 still saved
    assert {p["ref"] for p in _policies(conn, doc)} == {"1.2", "2.1"}


def test_unreadable_section_holds_back_proposed_retire(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V1)
    _run(conn, doc, 1, actor)
    _stub(monkeypatch, CANNED_V1, fail_refs=("1.1",))
    monkeypatch.setattr(X, "chunk_sections", lambda sections: [[s] for s in sections])
    counts, items = _run(conn, doc, 2, actor)
    assert counts["proposedRetire"] == 0 and not [i for i in items if i["kind"] == "proposed_retire"]
    note = [i for i in items if i["kind"] == "note"]
    assert len(note) == 1 and len(note[0]["payload"]["policyKeys"]) == 1


def test_revised_document_changed_new_and_proposed_retire_and_nothing_retired(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V2)
    _run(conn, doc, 1, actor)
    first = {p["ref"] + "|" + p["split"]: p for p in _policies(conn, doc)}
    k11, k12 = first["1.1|approve|gt:500"]["key"], first["1.2|notify|gt:200"]["key"]

    # A person activates 1.1 (readiness patched: the canned policy names test-only fields).
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    live_form = repo.get_policy(conn, k11)["versions"][0]["form"]
    repo.save_version(conn, k11, live_form, base_version=1, intent="activate", actor="person", change_note="go")

    _stub(monkeypatch, CANNED_V2)
    counts, items = _run(conn, doc, 2, actor)
    assert counts["changed"] == 1 and counts["new"] == 1 and counts["proposedRetire"] == 1
    assert counts["unchanged"] == 2
    by = _by_decision(items)
    assert [i["policy_key"] for i in by["changed"]] == [k11] and by["changed"][0]["saved_version"] == 3
    assert by["new"][0]["reference"] == "1.3"
    retire = by["proposed_retire"]
    assert len(retire) == 1 and retire[0]["policy_key"] == k12 and retire[0]["kind"] == "proposed_retire"
    assert retire[0]["payload"]["excerpt"] == P12.excerpt

    p11 = repo.get_policy(conn, k11)
    assert p11["status"] == "live" and p11["liveVersion"] == 2 and p11["latestVersion"] == 3
    assert p11["versions"][2]["savedAs"] == "draft"
    assert p11["versions"][2]["changeNote"].startswith("Re-extracted from Run test ")
    assert p11["versions"][2]["changeNote"].endswith(" v2, section 1.1.")
    assert repo.get_policy(conn, k12)["status"] == "draft"          # proposed, never retired
    assert not [p for p in _policies(conn, doc) if p["status"] == "retired"]


def test_every_version_the_agent_saved_is_an_unconfirmed_draft(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V2)
    _run(conn, doc, 1, actor)
    _stub(monkeypatch, CANNED_V2)
    _run(conn, doc, 2, actor)
    cur = conn.cursor()
    cur.execute("SELECT v.saved_as, v.form_state, v.confidence FROM proc.bp_agent_policy_version v"
                " JOIN proc.bp_agent_policy p ON p.policy_key = v.policy_key"
                " WHERE p.source_document_id = %s AND v.saved_by = %s", (doc, actor))
    rows = cur.fetchall()
    assert len(rows) == 6                                           # 4 new, then 1 changed + 1 new
    for saved_as, form, confidence in rows:
        assert saved_as == "draft"
        assert form["checked"] is None and form["hidden"]["setBy"] == "extraction_agent"
        assert confidence is not None                               # the document text reached the save


def test_ungrounded_excerpt_keeps_the_policy_with_a_note(conn, actor, monkeypatch):
    made_up = _p("1.1", "Refunds of any size need two signatures from the board", "approve", 500,
                 "Issue a refund over 500")
    _stub(monkeypatch, [made_up])
    doc = _document(conn, V1)
    counts, items = _run(conn, doc, 1, actor)
    assert counts["new"] == 1
    key = [i for i in items if i["kind"] == "policy"][0]["policy_key"]
    v1 = repo.get_policy(conn, key)["versions"][0]
    assert X.UNGROUNDED_NOTE in v1["form"]["hidden"]["agentNotes"]
    assert "The excerpt does not appear word for word in the document" in v1["confidence"]["failed"]


def test_stale_save_is_an_error_item_and_run_continues(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V2)
    _run(conn, doc, 1, actor)
    real_save = repo.save_version

    def racing_save(*a, **k):
        raise repo.StaleVersion("latest is 2, edit was based on 1")

    monkeypatch.setattr(repo, "save_version", racing_save)
    _stub(monkeypatch, CANNED_V2)
    counts, items = _run(conn, doc, 2, actor)
    monkeypatch.setattr(repo, "save_version", real_save)
    errors = [i for i in items if i["kind"] == "error"]
    assert len(errors) == 1 and "edited while the agent was reading" in errors[0]["payload"]["message"]
    assert counts["errors"] == 1 and counts["new"] == 1             # 1.3 still saved after the refusal


def test_fix_run_emits_one_fix_item_and_writes_no_version(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    key = [p for p in _policies(conn, doc) if p["ref"] == "1.1"][0]["key"]
    asked = {}

    def fake_fix(form, flipped, *, registry, taxonomy, settings, call=None):
        asked.update(form=form, flipped=flipped)
        return P11_V2

    monkeypatch.setattr(extractor, "fix_policy", fake_fix)
    flipped = [{"input": {"tool.name": "refund.issue", "args.amount": 600}, "expects": "none"}]
    run = store.create(conn, kind="fix", request={"policyKey": key, "baseVersion": 1, "flipped": flipped},
                       actor=actor)
    emitted = []

    def emit(kind, payload, **fields):
        emitted.append((kind, payload, fields))
        return store.append_item(conn, run["run_id"], kind=kind, payload=payload, **fields)

    versions = _version_count(conn, doc)
    X.run_fix(conn, run, emit)
    assert _version_count(conn, doc) == versions                    # never saves
    assert asked["flipped"] == flipped and asked["form"]["source"]["reference"] == "1.1"
    assert len(emitted) == 1
    kind, payload, fields = emitted[0]
    assert kind == "fix" and payload["basedOn"] == 1 and fields["policy_key"] == key
    prop = payload["proposed"]
    assert set(prop) == {"situation", "hidden", "examples", "outcome"}
    assert set(prop["hidden"]) == set(X._FIX_HIDDEN)
    assert matching.split_key({"outcome": prop["outcome"], "hidden": prop["hidden"]}) == "approve|gt:750"


def test_a_fix_never_proposes_what_a_person_sets(conn, actor, monkeypatch):
    """I2: accepting a fix must not wipe deciders, notify, owner, messages or the reviewer's
    other hidden settings -- the proposal never carries them."""
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    key = [p for p in _policies(conn, doc) if p["ref"] == "1.1"][0]["key"]
    got = repo.get_policy(conn, key)
    form = got["versions"][-1]["form"]
    hidden = dict(form["hidden"], onMissingData="fail_closed", whilePaused="retry",
                  timeWindow={"from": "09:00", "to": "17:00", "timeZone": "Europe/London"},
                  units={"currency": "EUR", "convertOther": "rate_on_action_date", "amountsIncludeTax": True})
    _person_saves(conn, key, deciders=["Head of Refunds"], notify=["Audit"], owner="Refunds Lead",
                  messageForAgent="Ask the refunds lead.", messageForPerson="Please check.", hidden=hidden)
    base = repo.get_policy(conn, key)["latestVersion"]
    monkeypatch.setattr(extractor, "fix_policy", lambda *a, **k: P11_V2)   # its deciders etc. differ
    run = store.create(conn, kind="fix", request={"policyKey": key, "baseVersion": base,
                                                  "flipped": [{"input": {"args.amount": 600}}]}, actor=actor)
    emitted = []
    X.run_fix(conn, run, lambda kind, payload, **f: emitted.append(payload) or 1)
    prop = emitted[0]["proposed"]
    for person_field in ("deciders", "notify", "owner", "messages", "messageForAgent", "messageForPerson"):
        assert person_field not in prop
    for kept in ("onMissingData", "whilePaused", "timeWindow", "units", "reasonCode", "setBy"):
        assert kept not in prop["hidden"]
    assert prop["hidden"]["condition"] == {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                                                   {"field": "args.amount", "op": "gt", "value": 750}]}


# ---------------------------------------------------------------- fix round 1

def _run_many(conn, docs, actor):
    run = store.create(conn, kind="extract",
                       request={"documents": [{"documentId": d, "version": v} for d, v in docs]}, actor=actor)

    def emit(kind, payload, **fields):
        return store.append_item(conn, run["run_id"], kind=kind, payload=payload, **fields)

    counts = X.run_extract(conn, run, emit)
    return counts, store.get(conn, run["run_id"])["items"]


def _person_saves(conn, key, **changes):
    got = repo.get_policy(conn, key)
    form = dict(got["versions"][-1]["form"], **changes)
    return repo.save_version(conn, key, form, base_version=got["latestVersion"], intent="draft",
                             actor="person", change_note="a person's edit")


def test_a_persons_edit_is_not_mistaken_for_a_document_change(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    keys = {p["ref"] + "|" + p["split"]: p["key"] for p in _policies(conn, doc)}
    k11, k12 = keys["1.1|approve|gt:500"], keys["1.2|notify|gt:200"]
    _person_saves(conn, k11, deciders=["CFO"])
    _person_saves(conn, k12, outcome="block")
    v2_before = {k: repo.get_policy(conn, k)["versions"][1] for k in (k11, k12)}

    counts, items = _run(conn, doc, 1, actor)          # the same text again
    assert counts["unchanged"] == 4 and counts["changed"] == counts["new"] == counts["proposedRetire"] == 0
    assert not [i for i in items if i["kind"] == "proposed_retire"]
    for k in (k11, k12):
        got = repo.get_policy(conn, k)
        assert got["latestVersion"] == 2 and got["versions"][1] == v2_before[k]   # v2 untouched, no v3


def test_changed_clause_keeps_the_persons_owner_and_dates_and_says_it_replaces_their_edits(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V2)
    _run(conn, doc, 1, actor)
    k11 = [p for p in _policies(conn, doc) if p["ref"] == "1.1"][0]["key"]
    _person_saves(conn, k11, deciders=["CFO"], owner="Head of Refunds", effectiveFrom="2026-11-01",
                  reviewBy="2027-11-01")
    _stub(monkeypatch, CANNED_V2)
    _run(conn, doc, 2, actor)
    got = repo.get_policy(conn, k11)
    assert got["latestVersion"] == 3
    v3 = got["versions"][2]
    assert (v3["form"]["owner"], v3["form"]["effectiveFrom"], v3["form"]["reviewBy"]) == \
        ("Head of Refunds", "2026-11-01", "2027-11-01")
    assert v3["form"]["deciders"] == ["Finance Manager"]                 # the document's reading
    assert v3["changeNote"].endswith(" v2, section 1.1. Replaces the edits in v2.")
    assert got["versions"][1]["form"]["deciders"] == ["CFO"]              # history keeps the edit


def test_failed_conversion_holds_back_proposed_retire(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    real = X.converter.to_form

    def flaky(p, **k):
        if p.reference == "1.2":
            raise ValueError("cannot convert")
        return real(p, **k)

    monkeypatch.setattr(X.converter, "to_form", flaky)
    counts, items = _run(conn, doc, 1, actor)
    assert counts["errors"] == 1 and counts["proposedRetire"] == 0
    assert not [i for i in items if i["kind"] == "proposed_retire"]
    assert [i["kind"] for i in items if i["kind"] == "note"] == ["note"]


def test_one_documents_crash_is_an_error_item_and_the_next_document_runs(conn, actor, monkeypatch):
    _stub(monkeypatch, CANNED_V1)
    first, second = _document(conn, V1), _document(conn, V1)
    real, calls = matching.match, []

    def crash_once(existing, proposed):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("matching blew up")
        return real(existing, proposed)

    monkeypatch.setattr(X.matching, "match", crash_once)
    counts, items = _run_many(conn, [(first, 1), (second, 1)], actor)
    errors = [i for i in items if i["kind"] == "error"]
    assert len(errors) == 1 and errors[0]["document_id"] == first and "matching blew up" in errors[0]["payload"]["message"]
    assert counts["documents"] == 2 and counts["new"] == 4
    assert len(_policies(conn, second)) == 4 and _policies(conn, first) == []


# ---------------------------------------------------------------- final fix wave

def test_a_persons_save_keeps_the_agents_confidence(conn, actor, monkeypatch):
    """I3: a person's save passes no document text; the save reads the source version's stored
    text, so the excerpt is still found and confidence does not drop."""
    from services.agent_policy import readiness
    monkeypatch.setattr(readiness, "activation_problems", lambda f, r, s, deciders=None: [])   # test-only names
    monkeypatch.setattr(readiness, "_cant_enforce", lambda f, r: False)
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    k11 = [p for p in _policies(conn, doc) if p["ref"] == "1.1"][0]["key"]
    assert repo.get_policy(conn, k11)["versions"][0]["confidence"] == {"level": "High", "failed": []}
    _person_saves(conn, k11, situation="The agent is about to issue a refund over 500 dollars.")
    assert repo.get_policy(conn, k11)["versions"][1]["confidence"] == {"level": "High", "failed": []}


def test_a_persons_first_save_of_an_extracted_form_finds_its_document_by_title(conn, actor, monkeypatch):
    """I3, create_draft: no source block, so the document is found by the form's source title."""
    from services.agent_policy import readiness
    monkeypatch.setattr(readiness, "activation_problems", lambda f, r, s, deciders=None: [])
    monkeypatch.setattr(readiness, "_cant_enforce", lambda f, r: False)
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1)
    _run(conn, doc, 1, actor)
    k11 = [p for p in _policies(conn, doc) if p["ref"] == "1.1"][0]["key"]
    form = repo.get_policy(conn, k11)["versions"][0]["form"]
    saved = repo.create_draft(conn, dict(form, changeNote="a copy"), actor="person")
    assert repo.get_policy(conn, saved["policyKey"])["versions"][0]["confidence"] == {"level": "High", "failed": []}


V3_RENUMBERED = """1. Refunds
1.4 Credits to a customer account above 200 dollars must be reported to the Finance team.
1.5 Refunds above 750 dollars need approval from the Finance Manager before they are issued.
2. Large refunds
2.1 Refunds above 500 dollars need Finance Manager approval, and refunds above 10000 dollars are not allowed at all.
3. Conduct
3.1 Staff should always be courteous and helpful to customers.
"""


def test_a_renumbered_clause_moves_the_policys_reference(conn, actor, monkeypatch):
    """Minor 4: 1.1 -> 1.5 (and changed), 1.2 -> 1.4 (unchanged). Both policies follow their
    clause, so the next run matches them exactly instead of by excerpt."""
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V3_RENUMBERED)
    _run(conn, doc, 1, actor)
    first = {p["ref"] + "|" + p["split"]: p["key"] for p in _policies(conn, doc)}
    k11, k12 = first["1.1|approve|gt:500"], first["1.2|notify|gt:200"]
    p15 = _p("1.5", "Refunds above 750 dollars need approval from the Finance Manager", "approve", 750,
             "Issue a refund over 500")
    p14 = _p("1.4", P12.excerpt, "notify", 200, "Credit an account over 200", notify=("Finance team",))
    _stub(monkeypatch, [p14, p15, P21_APPROVE, P21_BLOCK])
    counts, items = _run(conn, doc, 2, actor)
    assert counts["changed"] == 1 and counts["unchanged"] == 3 and counts["new"] == counts["proposedRetire"] == 0
    by = {p["key"]: p for p in _policies(conn, doc)}
    assert (by[k11]["ref"], by[k11]["split"], by[k11]["latest"]) == ("1.5", "approve|gt:750", 2)
    assert (by[k12]["ref"], by[k12]["split"], by[k12]["latest"]) == ("1.4", "notify|gt:200", 1)


def test_a_busy_model_is_an_error_item_per_chunk_and_holds_back_retires(conn, actor, monkeypatch):
    """Fast failure: a chunk whose call would reload the shared model is not called; it is an
    error item with the plain message, the run goes on, and Proposed retire waits."""
    _stub(monkeypatch, CANNED_V1)
    doc = _document(conn, V1, V1)
    _run(conn, doc, 1, actor)
    seen = _stub(monkeypatch, CANNED_V1)
    real = X.extractor.extract_chunk

    def busy_on_1_2(sections, **k):
        if any(s.get("reference") == "1.2" for s in sections):
            raise extractor.ModelBusy()
        return real(sections, **k)

    monkeypatch.setattr(X.extractor, "extract_chunk", busy_on_1_2)
    monkeypatch.setattr(X, "chunk_sections", lambda sections: [[s] for s in sections])
    counts, items = _run(conn, doc, 2, actor)
    errors = [i for i in items if i["kind"] == "error"]
    assert [e["payload"]["message"] for e in errors] == [extractor.MODEL_BUSY]
    assert errors[0]["payload"]["references"] == ["1.2"] and counts["errors"] == 1
    assert counts["unchanged"] == 3 and counts["proposedRetire"] == 0
    assert [i["kind"] for i in items if i["kind"] == "note"] == ["note"]
    assert len(seen) == len(X.split_sections(V1)) - 1           # every other chunk still ran
