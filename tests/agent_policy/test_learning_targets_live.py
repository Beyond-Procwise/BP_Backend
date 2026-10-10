"""Which live policies learning may suggest on, read from REAL policies on bp_testdb (brief §8 test 19).

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Four real "TST …" policies are created and
activated, each on its own tool tst_<tag> (it can never apply to anyone's real action or contradict
anyone's policy), and every one is RETIRED at teardown, one step at a time. Only `learning.targets`
is called: it reads, so nothing here writes a suggestion row, and the scan itself is never run.
No business area's setting is changed: the test uses "Legal and compliance" (never suggest ON as
seeded) and "Operations" (OFF), and is skipped if the seed differs.

`_TARGETS_SQL` leaves a live policy out for three reasons. A policy saved the ordinary way is
usually caught by two of them at once (a block policy is not `approve` AND its compiled document
says not eligible), so a missing clause would hide behind the other. Here the compiled document of
each policy is made to say what is written beside it below, so that exactly ONE clause stands
between each left-out policy and a suggestion:

  policy               area                   outcome   compiled eligible   left out by
  control              Operations             approve   true                (nothing: it is a target)
  block                Operations             block     true                outcome = 'approve'
  never-suggest area   Legal and compliance   approve   true                the area's never_suggest, NOW
  not eligible         Operations             approve   "maybe"             eligible = 'true'

"true" for the never-suggest policy is what a version saved BEFORE its area was marked holds: the
setting is read at scan time for exactly that case. "maybe" is a stored value that is not a
boolean: it must simply not match, where a cast to boolean fails the whole statement, and with it
the scan of every other policy.
"""
import os
import uuid
from types import SimpleNamespace

import pytest

from repositories import agent_policy_repo as repo
from services.agent_policy import learning as L
from tests.agent_policy.test_learning_endpoints_live import _form

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NEVER, FREE = "Legal and compliance", "Operations"
#: what each policy is: (business area, outcome, what its compiled document says of `eligible`)
KINDS = {"control": (FREE, "approve", True), "block": (FREE, "block", True),
         "never": (NEVER, "approve", True), "ineligible": (FREE, "approve", "maybe")}


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "learning target tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def _retire(conn, key):
    got = repo.get_policy(conn, key)
    if got["status"] != "retired":
        repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


@pytest.fixture
def policies(conn, monkeypatch):
    """The four live policies of the table above, by kind."""
    cur = conn.cursor()
    cur.execute("SELECT area_name, never_suggest FROM proc.bp_business_area WHERE area_name IN (%s, %s)", (NEVER, FREE))
    flags = dict(cur.fetchall())
    if flags != {NEVER: True, FREE: False}:
        pytest.skip(f"needs the seeded settings (never suggest ON for {NEVER!r}, OFF for {FREE!r}); found {flags}. "
                    "This test never changes a business area's setting.")
    # tst_ tools are unknown to the real registry: readiness and the final check are bypassed
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    says = {}                                   # policy name -> what its compiled document is made to say
    compile_policy = repo.compile_policy

    def _compiled(form, **kw):
        doc = compile_policy(form, **kw)
        doc["learning"] = {"eligible": says[form["name"]]}
        return doc
    monkeypatch.setattr(repo, "compile_policy", _compiled)
    made = {}
    try:
        for kind, (area, outcome, eligible) in KINDS.items():
            tag = uuid.uuid4().hex[:8]
            form = _form(tag)
            form["name"] = f"TST learning target {kind} {tag}"
            form["businessArea"], form["outcome"] = area, outcome
            if outcome == "block":
                form["deciders"] = []
                for example in form["examples"]:
                    example["agentExpected"] = "block" if example["agentExpected"] == "approve" else "none"
            says[form["name"]] = eligible
            w = made[kind] = SimpleNamespace(kind=kind, key=None, area=area, outcome=outcome, eligible=eligible)
            w.key = repo.create_draft(conn, form, actor="test")["policyKey"]
            repo.save_version(conn, w.key, form, base_version=1, intent="activate", actor="test", change_note="go")
        yield made
    finally:
        failed = []
        for w in made.values():
            if w.key:
                try:
                    _retire(conn, w.key)
                except Exception as exc:  # noqa: BLE001 - one failure must not leave the others live
                    failed.append(f"{w.key}: {exc!r}")
        assert not failed, f"teardown left live test policies behind: {failed}"


def _stored(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT p.status, p.area_name, v.form_state ->> 'outcome', v.compiled -> 'learning' -> 'eligible' "
                "FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v "
                "ON v.policy_key = p.policy_key AND v.version = p.live_version WHERE p.policy_key = %s", (key,))
    return cur.fetchone()


def test_no_suggestion_for_a_block_policy_a_never_suggest_area_or_a_policy_not_marked_eligible(conn, policies):
    # the world is what the table in the module docstring says it is, read back from the database
    for w in policies.values():
        assert _stored(conn, w.key) == ("live", w.area, w.outcome, w.eligible), w.kind
    targets = {t.key: t for t in L.targets(conn.cursor())}
    assert policies["control"].key in targets, "the control proves the read reaches a policy made this way"
    assert targets[policies["control"].key].live_version == 2
    assert policies["block"].key not in targets, "a block policy is never loosened"
    assert policies["never"].key not in targets, "its business area is marked 'Never suggest changes'"
    assert policies["ineligible"].key not in targets, "its compiled document does not say it is eligible"

