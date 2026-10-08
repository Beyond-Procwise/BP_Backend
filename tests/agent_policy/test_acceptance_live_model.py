"""Stage 2 acceptance on the real AgentNick model (brief section 8, tests 1 and 2, plus the plan's extras).

Three short policy documents from three business areas go through ONE extraction run, end to
end without the gateway: issue uploads, PUT each file's bytes to S3 at the key register rebuilds,
register, create a run with the real run store and execute it with the real runner (the model
call is AgentNick, as in production). Then:

1. every policy item's saved form has a business area (or an agent note saying why not), a
   sub-area, a source (document, reference, excerpt) and a computed extraction confidence;
2. the tiered Finance clause 1.1 gives two policies under one source: same source_reference,
   different source_split;
3. at least one not_enforceable item, with a reason;
4. re-running the same versions writes nothing (every policy item is ``unchanged``);
5. a revised Finance document (1.1 amount changed, 1.2 removed), with one Finance policy
   activated first, gives ``changed`` (a new draft; the live version stays live) and
   ``proposed_retire`` (nothing is retired). Activation is test setup by a named test actor:
   the first Finance policy that is ready as proposed, else the clause-1.2 policy after a
   reviewer's correction (the registry has no refund or amount field, so clause 1.1 can
   never be activated; see _reviewer_correction).

The model's wording varies, so what is asserted hard is what the CODE guarantees (one run,
drafts only, checked None, setBy, grouping by source, decisions); model-dependent facts are
asserted loosely and written to the record (AGENT_POLICY_ACCEPTANCE_OUT, a JSON path).

Gated: AGENT_POLICY_LIVE_MODEL=1, PROCWISE_TEST_LIVE_DB=1, DB_NAME=bp_testdb and a reachable
Ollama. Every S3 object created is deleted at the end, also on failure. The documents, runs and
policies stay in bp_testdb as the record; the document names carry a unique tag so a later
upload of the same files never matches them, and the policy activated as setup is retired at
the end so it never governs anyone else's agents.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest

ACTOR = "acceptance-test"
FIXTURES = Path(__file__).parent / "fixtures_docs"
FILES = ("finance_payments_policy.md", "customer_refund_standard.md", "security_data_handling.md")


def _ollama_answers() -> bool:
    try:
        import requests

        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return requests.get(f"{base}/api/tags", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001 - unreachable is a skip, not an error
        return False


_GATED = os.getenv("AGENT_POLICY_LIVE_MODEL") == "1" and os.getenv("PROCWISE_TEST_LIVE_DB") == "1"
pytestmark = pytest.mark.skipif(not (_GATED and _ollama_answers()),
                                reason="needs AGENT_POLICY_LIVE_MODEL=1, PROCWISE_TEST_LIVE_DB=1 and Ollama")

RECORD: dict = {"chunks": [], "runs": {}, "activation": [], "notes": []}


def _revised_finance(text: str) -> str:
    """1.1's lower threshold $500 -> $750; clause 1.2 removed. Everything else byte-identical."""
    assert text.count("above $500 ") == 1
    out = text.replace("above $500 ", "above $750 ")
    start = out.index("1.2 ")
    end = out.index("2. Good practice")
    out = out[:start] + out[end:]
    assert "1.2" not in out and "$750" in out and "1.1 " in out
    return out


@pytest.fixture(scope="module")
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "the acceptance test runs on bp_testdb only"
    from services.db import get_conn

    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT current_database()")
        assert cur.fetchone()[0] == "bp_testdb"
        yield c


@pytest.fixture(scope="module")
def world(conn):
    """Upload + register the three documents, then run ONE extraction over all three."""
    from services.agent_policy import documents as d
    from services.agent_policy import extractor

    tag = uuid.uuid4().hex[:8]
    client, bucket = d._s3(), d._bucket()
    keys: list = []
    activated: list = []
    mp = pytest.MonkeyPatch()

    real_extract = extractor.extract_chunk

    def timed(sections, **kw):  # same call, timed; behaviour unchanged
        t0 = time.monotonic()
        ok = False
        try:
            out = real_extract(sections, **kw)
            ok = True
            return out
        finally:
            RECORD["chunks"].append({"document": (kw.get("document") or {}).get("title"),
                                     "version": (kw.get("document") or {}).get("version"),
                                     "references": [s.get("reference") for s in sections],
                                     "seconds": round(time.monotonic() - t0, 2), "ok": ok})

    mp.setattr(extractor, "extract_chunk", timed)

    def upload(name: str, body: bytes, **extra) -> dict:
        issued = d.issue_uploads([{"name": name, "size": len(body)}], actor=ACTOR)[0]
        key = f"agent-policy-documents/uploads/{issued['uploadId']}/{issued['safeName']}"
        assert key == d.upload_key(issued["uploadId"], name)
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=issued["contentType"])
        keys.append(key)
        reg = d.register_uploads(conn, [{"uploadId": issued["uploadId"], "name": name, **extra}], actor=ACTOR)[0]
        assert reg["duplicate"] is False
        return reg

    state = {"tag": tag, "upload": upload, "activated": activated, "docs": {}}
    try:
        for fname in FILES:
            body = (FIXTURES / fname).read_bytes()
            # The tag makes each test run its own document (never a revision of an earlier run's,
            # and never matched by a later upload of the same file through the gateway).
            reg = upload(f"{Path(fname).stem} acceptance {tag}.md", body)
            assert reg["version"] == 1
            state["docs"][fname] = {"documentId": reg["documentId"], "version": 1, "text": body.decode()}
        state["run1"] = _run(conn, [{"documentId": v["documentId"], "version": 1}
                                    for v in state["docs"].values()], "run1")
        yield state
    finally:
        mp.undo()
        _cleanup(conn, client, bucket, keys, activated)
        out = os.getenv("AGENT_POLICY_ACCEPTANCE_OUT")
        if out:
            Path(out).write_text(json.dumps(RECORD, indent=2, default=str))


def _cleanup(conn, client, bucket, keys, activated):
    from repositories import agent_policy_repo as repo

    for key in activated:  # never leave a test policy live in a shared database
        try:
            p = repo.get_policy(conn, key)
            if p["status"] != "retired":
                repo.retire(conn, key, base_version=p["latestVersion"], actor=ACTOR,
                            change_note="Acceptance test clean-up: retired after the test.")
                RECORD["notes"].append(f"{key} retired at clean-up")
        except Exception as exc:  # noqa: BLE001 - keep cleaning up
            RECORD["notes"].append(f"could not retire {key}: {exc}")
    for key in keys:
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            RECORD["notes"].append(f"could not delete s3 object {key}: {exc}")
    from botocore.exceptions import ClientError

    left = []
    for key in keys:
        try:
            client.head_object(Bucket=bucket, Key=key)
            left.append(key)
        except ClientError:
            pass
    RECORD["s3"] = {"created": len(keys), "left": left}
    assert not left, f"S3 objects left behind: {left}"


def _run(conn, docs, label):
    from services.agent_policy import extraction_run, run_runner, run_store

    run = run_store.create(conn, kind="extract", request={"documents": docs}, actor=ACTOR)
    t0 = time.monotonic()
    run_runner.run(run["run_id"], extraction_run.run_extract)  # real store, real connection, real model
    seconds = round(time.monotonic() - t0, 2)
    got = run_store.get(conn, run["run_id"])
    RECORD["runs"][label] = {
        "runId": got["run_id"], "status": got["status"], "error": got["error"], "counts": got["counts"],
        "seconds": seconds,
        "items": [{k: it[k] for k in ("seq", "kind", "document_id", "document_version", "reference",
                                      "policy_key", "decision", "saved_version", "payload")}
                  for it in got["items"]]}
    assert got["status"] == "done", f"run {label} ended {got['status']}: {got['error']}"
    return got


def _items(run, kind=None, document_id=None):
    return [i for i in run["items"] if (kind is None or i["kind"] == kind)
            and (document_id is None or i["document_id"] == document_id)]


def _source(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT source_document_id, source_reference, source_split, status, live_version, latest_version"
                " FROM proc.bp_agent_policy WHERE policy_key = %s", (key,))
    return dict(zip(("documentId", "reference", "split", "status", "live", "latest"), cur.fetchone()))


# ------------------------------------------------------------------ 1. one run, three areas

def test_1_one_run_three_documents_every_policy_complete(conn, world):
    from repositories import agent_policy_repo as repo

    run = world["run1"]
    errors = _items(run, "error")
    assert not errors, f"the run had error items: {json.dumps([e['payload'] for e in errors], indent=1)}"
    doc_ids = {v["documentId"] for v in world["docs"].values()}
    assert {i["document_id"] for i in _items(run, "policy")} == doc_ids, "every document gave a policy in ONE run"
    areas = {}
    for it in _items(run, "policy"):
        assert it["decision"] == "new" and it["saved_version"] == 1, it
        version = repo.get_policy(conn, it["policy_key"])["versions"][0]
        form = version["form"]
        assert version["savedAs"] == "draft" and form["checked"] is None
        assert form["hidden"]["setBy"] == "extraction_agent"
        notes = " ".join(form["hidden"].get("agentNotes") or [])
        assert (form.get("businessArea") and form.get("subArea")) or "business area" in notes, (it, form)
        src = form.get("source") or {}
        assert src.get("document") and src.get("reference") and src.get("excerpt"), src
        assert version["confidence"] and version["confidence"].get("level") in ("High", "Medium", "Low"), version
        areas.setdefault(it["document_id"], set()).add(form.get("businessArea"))
        RECORD.setdefault("policies", []).append({
            "policyKey": it["policy_key"], "documentId": it["document_id"], "reference": src.get("reference"),
            "businessArea": form.get("businessArea"), "subArea": form.get("subArea"), "outcome": form.get("outcome"),
            "split": it["payload"].get("split"), "confidence": version["confidence"],
            "unknownNames": form["hidden"].get("unknownNames"), "agentNotes": form["hidden"].get("agentNotes")})
    RECORD["areasByDocument"] = {k: sorted(map(str, v)) for k, v in areas.items()}
    # Model-dependent (loose): the three documents are not all filed under one area.
    assert len(set().union(*areas.values())) >= 2, RECORD["areasByDocument"]


# ------------------------------------------------------------------ 2. tiered clause

def test_2_tiered_clause_two_policies_one_source(conn, world):
    fin = world["docs"]["finance_payments_policy.md"]["documentId"]
    tier = [i for i in _items(world["run1"], "policy", fin) if i["reference"] == "1.1"]
    raw = json.dumps([i["payload"] for i in tier], indent=1)
    assert len(tier) >= 2, f"clause 1.1 did not split into tiers; raw items: {raw}"
    rows = [_source(conn, i["policy_key"]) for i in tier]
    assert {r["documentId"] for r in rows} == {fin}
    assert {r["reference"] for r in rows} == {"1.1"}, rows
    assert len({r["split"] for r in rows}) == len(rows), f"tiers share a split: {rows}; raw: {raw}"
    RECORD["tiers"] = [{"policyKey": i["policy_key"], "split": r["split"]} for i, r in zip(tier, rows)]


# ------------------------------------------------------------------ 3. not enforceable

def test_3_at_least_one_not_enforceable_with_reason(world):
    ne = _items(world["run1"], "not_enforceable")
    RECORD["notEnforceable"] = [{"document_id": i["document_id"], **i["payload"]} for i in ne]
    assert ne, "no not_enforceable item"
    assert all(str(i["payload"].get("reason") or "").strip() for i in ne), ne


# ------------------------------------------------------------------ 4. re-run: nothing new

def test_4_rerun_same_versions_writes_nothing(conn, world):
    from repositories import agent_policy_repo as repo

    keys = [i["policy_key"] for i in _items(world["run1"], "policy")]
    before = {k: repo.get_policy(conn, k)["latestVersion"] for k in keys}
    run = _run(conn, [{"documentId": v["documentId"], "version": 1} for v in world["docs"].values()], "run2_rerun")
    pol = _items(run, "policy")
    decisions = [(i["policy_key"], i["decision"], i["saved_version"]) for i in pol]
    assert pol and all(d == "unchanged" and sv is None for _, d, sv in decisions), \
        f"re-run wrote something: {decisions}; raw: {json.dumps([i['payload'] for i in pol], indent=1)}"
    assert sorted(k for k, _, _ in decisions) == sorted(keys), "re-run matched a different set of policies"
    assert not _items(run, "proposed_retire") and not _items(run, "error")
    assert {k: repo.get_policy(conn, k)["latestVersion"] for k in keys} == before


# ------------------------------------------------------------------ 5. revised Finance document

def _reviewer_correction(form):
    """What a reviewer does to the clause-1.2 proposal before activating it (test setup only).

    The registry has no field for "payment emails", so the model's extra rules on the whole
    payload (args.payload_json) are flagged and the policy cannot go live as proposed; and the
    model filed the Finance Manager as owner but not as decider. The person keeps the part the
    registry can check (the tool) and names the decider the clause names.
    """
    h = dict(form["hidden"])
    tool = (h.get("actions") or {}).get("tools", [None])[0]
    h["condition"] = {"all": [{"op": "eq", "field": "tool.name", "value": tool}]}
    h["inputs"] = [i for i in h.get("inputs") or [] if i.get("field") in ("tool.name", "agent.reason")]
    h["unknownNames"] = []
    examples = [{"input": {"tool.name": tool}, "agentExpected": "approve", "flipped": False},
                {"input": {"tool.name": "run_rag"}, "agentExpected": "none", "flipped": False}]
    return dict(form, hidden=h, examples=examples, deciders=list(form.get("deciders") or []) or ["Finance Manager"])


def _activate_one(conn, items):
    """Test setup: a named test actor confirms and activates a Finance policy.

    First as the agent proposed it (the first one that is ready); if none is, the clause-1.2
    policy after a reviewer's correction (recorded).
    """
    from repositories import agent_policy_repo as repo
    from services.agent_policy import readiness

    # Test setup: the named deciders are linked (in-memory), as an administrator would have done.
    linked = {"groups": ["ACCEPTANCE_TEST"], "emails": []}
    names = {"Finance Manager"}
    for i in items:
        names.update(repo.get_policy(conn, i["policy_key"])["versions"][-1]["form"].get("deciders") or [])
    readiness_load, readiness._load_deciders = readiness._load_deciders, lambda: {n: linked for n in names}
    try:
        return _activate_one_linked(conn, items)
    finally:
        readiness._load_deciders = readiness_load


def _activate_one_linked(conn, items):
    from repositories import agent_policy_repo as repo

    attempts = [(i["policy_key"], None) for i in items]
    attempts += [(i["policy_key"], _reviewer_correction) for i in items if i["reference"] == "1.2"]
    for key, correct in attempts:
        p = repo.get_policy(conn, key)
        form = p["versions"][-1]["form"]
        if correct:
            form = correct(form)
        form = dict(form, checked={"by": ACTOR, "at": "2026-10-08T00:00:00Z"})
        try:
            saved = repo.save_version(conn, key, form, base_version=p["latestVersion"], intent="activate",
                                      actor=ACTOR, change_note="Acceptance test setup: activated.")
        except repo.NotReady as exc:
            RECORD["activation"].append({"policyKey": key, "reviewerCorrected": bool(correct), "activated": False,
                                         "problems": [{k: q.get(k) for k in ("code", "names", "missing")}
                                                      for q in exc.problems]})
            continue
        RECORD["activation"].append({"policyKey": key, "reviewerCorrected": bool(correct), "activated": True,
                                     "liveVersion": saved["version"]})
        return key, saved["version"]
    return None, None


def test_5_revised_finance_changed_and_proposed_retire(conn, world):
    from repositories import agent_policy_repo as repo

    fin = world["docs"]["finance_payments_policy.md"]
    fin_items = _items(world["run1"], "policy", fin["documentId"])
    # Prefer a clause-1.1 policy (it changes), then 1.2 (it is removed), then anything else.
    order = sorted(fin_items, key=lambda i: {"1.1": 0, "1.2": 1}.get(i["reference"], 2))
    live_key, live_version = _activate_one(conn, order)
    assert live_key, f"no Finance policy could be activated: {RECORD['activation']}"
    world["activated"].append(live_key)
    live_ref = _source(conn, live_key)["reference"]

    revised = _revised_finance(fin["text"]).encode()
    reg = world["upload"](f"finance_payments_policy acceptance {world['tag']} revised.md", revised,
                          revisionOf=fin["documentId"])
    assert reg["documentId"] == fin["documentId"] and reg["version"] == 2 and reg["isRevision"]
    run = _run(conn, [{"documentId": fin["documentId"], "version": 2}], "run3_revised")
    assert not _items(run, "error"), [i["payload"] for i in _items(run, "error")]

    pol, retire = _items(run, "policy"), _items(run, "proposed_retire")
    decisions = {i["policy_key"]: (i["reference"], i["decision"]) for i in pol}
    RECORD["revision"] = {"liveKey": live_key, "liveReference": live_ref, "decisions": decisions,
                          "proposedRetire": [i["policy_key"] for i in retire]}

    changed = [i for i in pol if i["decision"] == "changed"]
    assert changed, f"no changed item: {decisions}"
    for i in changed:  # a new DRAFT version on the same policy key
        p = repo.get_policy(conn, i["policy_key"])
        v = next(x for x in p["versions"] if x["version"] == i["saved_version"])
        assert v["savedAs"] == "draft" and v["form"]["checked"] is None and p["latestVersion"] == i["saved_version"]
    # The approval tier carries the changed figure; a block tier whose excerpt is only the
    # unchanged second sentence may legitimately stay unchanged.
    assert any("$750" in (i["payload"].get("excerpt") or "") for i in changed), [i["payload"] for i in changed]
    assert {i["reference"] for i in changed} == {"1.1"}, decisions

    assert retire, f"no proposed_retire item: {decisions}"
    run1_12 = {i["policy_key"] for i in fin_items if i["reference"] == "1.2"}
    assert {i["policy_key"] for i in retire} == run1_12, (retire, run1_12)

    # Nothing is retired; the live version stays live whether its clause changed or was removed.
    for key in {i["policy_key"] for i in fin_items}:
        assert _source(conn, key)["status"] != "retired", key
    after = _source(conn, live_key)
    assert after["status"] == "live" and after["live"] == live_version, after
    if live_key in decisions and decisions[live_key][1] == "changed":
        assert after["latest"] > after["live"], after  # the new draft sits beside the live version
