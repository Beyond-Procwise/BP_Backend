# tests/api/test_agent_workflows_router.py
import pytest

pytestmark = pytest.mark.integration   # touches the live proc schema

GRAPH = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0},
        {"id": "n2", "agent_slug": "supplier_ranking", "x": 200, "y": 0},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}

# Every workflow this test suite creates uses one of these names. Sweeping by
# name lets teardown find and remove rows even if a run crashed before its
# id was captured.
_TEST_NAMES = ("router-test", "hitl-test", "gov-test")


def _sweep_namespace():
    """Belt-and-braces: remove anything left behind by this suite's namespace,
    including rows orphaned by an earlier crashed/interrupted run.

    proc.bp_agent_workflow / proc.bp_workflow_input_request live in the LIVE
    production database (bp_sqldb). The router's DELETE endpoint only soft-
    deletes (is_active=false) — the row stays forever — so teardown here does
    a real DELETE unconditionally, matching the pattern in
    tests/orchestration/test_agent_workflow_repo.py.
    """
    from services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT workflow_id FROM proc.bp_agent_workflow WHERE name = ANY(%s)",
            (list(_TEST_NAMES),),
        )
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            # Request rows and run rows are linked back to the numeric
            # workflow via the agent_workflow_id column (see
            # workflow_input_request_repo) — not by parsing the run_id string.
            cur.execute(
                "DELETE FROM proc.bp_workflow_input_request WHERE agent_workflow_id = ANY(%s)",
                (ids,),
            )
            # bp_workflow_run is created by workflow_input_request_repo.ensure_schema();
            # tolerate its absence (e.g. against a pre-fix checkout) via to_regclass
            # instead of a bare DELETE, which would raise UndefinedTable.
            cur.execute("SELECT to_regclass('proc.bp_workflow_run')")
            if cur.fetchone()[0] is not None:
                cur.execute(
                    "DELETE FROM proc.bp_workflow_run WHERE agent_workflow_id = ANY(%s)",
                    (ids,),
                )
            cur.execute(
                "DELETE FROM proc.bp_agent_workflow WHERE workflow_id = ANY(%s)", (ids,)
            )
        cur.close()


@pytest.fixture(scope="module", autouse=True)
def _schema():
    from repositories import agent_workflow_repo as repo
    from repositories import workflow_input_request_repo as reqrepo

    repo.ensure_schema()
    reqrepo.ensure_schema()
    # Sweep before the module runs too, so a prior crashed run never masks a
    # false-positive pass.
    _sweep_namespace()


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    """Hard-delete every row this test creates, on success AND on failure.

    The post-yield block runs even when the test body raises, so a live-DB
    row can never survive a failed assertion.
    """
    created_workflow_ids: list = []
    created_run_ids: list = []

    yield created_workflow_ids, created_run_ids

    try:
        from services.db import get_conn

        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT to_regclass('proc.bp_workflow_run')")
            has_run_table = cur.fetchone()[0] is not None
            for run_id in created_run_ids:
                cur.execute(
                    "DELETE FROM proc.bp_workflow_input_request WHERE workflow_id = %s",
                    (run_id,),
                )
                if has_run_table:
                    cur.execute(
                        "DELETE FROM proc.bp_workflow_run WHERE run_id = %s", (run_id,)
                    )
            for wid in created_workflow_ids:
                cur.execute(
                    "DELETE FROM proc.bp_agent_workflow WHERE workflow_id = %s", (wid,)
                )
            cur.close()
    finally:
        _sweep_namespace()


@pytest.fixture(scope="module")
def client():
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_create_list_get_delete(client, _cleanup_test_rows):
    created_workflow_ids, _ = _cleanup_test_rows

    r = client.post("/agent-workflows", json={"name": "router-test", "graph": GRAPH})
    assert r.status_code == 200, r.text
    wid = r.json()["workflow_id"]
    created_workflow_ids.append(wid)

    assert any(w["workflow_id"] == wid for w in client.get("/agent-workflows").json()["workflows"])

    got = client.get(f"/agent-workflows/{wid}").json()
    assert got["entry_node"] == "n1"          # derived, not supplied
    assert got["graph"]["nodes"][0]["agent_slug"] == "data_extraction"

    assert client.delete(f"/agent-workflows/{wid}").status_code == 200


def test_a_cyclic_graph_is_rejected_with_a_reason(client, _cleanup_test_rows):
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    r = client.post("/agent-workflows", json={"name": "bad", "graph": bad})
    assert r.status_code == 400
    assert "cycle" in r.json()["detail"].lower()


def test_run_halts_and_asks_for_documents(client, _cleanup_test_rows):
    """The HITL contract, end to end at the HTTP layer."""
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    created_run_ids.append(run["run_id"])
    assert run["status"] == "awaiting_input"
    fields = {q["required_field"] for q in run["pending"]}
    assert "s3_object_keys" in fields     # data_extraction wants documents (exact keys)
    assert "query" in fields              # supplier_ranking wants a query

    client.delete(f"/agent-workflows/{wid}")


def test_each_node_reports_whether_it_is_governed(client, _cleanup_test_rows):
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "gov-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    created_run_ids.append(run["run_id"])
    gov = {n["node_id"]: n["governance"]["governed"] for n in run["nodes"]}
    assert gov["n1"] is False    # data_extraction: ungoverned, built-in default
    assert gov["n2"] is True     # supplier_ranking: governed
    client.delete(f"/agent-workflows/{wid}")


class _FakeState:
    """Stand-in for orchestration.workflow_engine.WorkflowState — just enough
    surface for agent_workflows._execute to build its response from."""
    status = "completed"
    node_statuses: dict = {}
    errors: list = []


def _spy_on_engine(client, monkeypatch):
    """Replace the real WorkflowEngine.execute with a spy that records the
    input_data it was handed and returns a fake completed state, instead of
    performing a real (LLM/GPU/S3-touching) agent run."""
    from api.main import app

    engine = app.state.orchestrator._workflow_engine
    calls = []

    def fake_execute(graph, *, input_data=None, user_id="system", workflow_id=None,
                      resume_state=None):
        calls.append(dict(input_data or {}))
        return _FakeState()

    monkeypatch.setattr(engine, "execute", fake_execute)
    return calls


def test_resume_merges_original_payload_with_answers(client, _cleanup_test_rows, monkeypatch):
    """FINDING 1 (critical): the original run payload must survive a resume.

    A run started with a payload key that is NOT one of the elicited
    questions must still be visible to the engine after the human answers
    the questions that WERE elicited — merged together, answers winning on
    conflict. Spies on the engine's execute() boundary rather than
    performing a real agent run (LLM/GPU/S3), per the task instructions.
    """
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    calls = _spy_on_engine(client, monkeypatch)

    payload = {"not_elicited_key": "keep-me"}
    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": payload}).json()
    created_run_ids.append(run["run_id"])
    assert run["status"] == "awaiting_input"
    pending = run["pending"]
    assert pending, "expected the graph to elicit at least one question"

    for q in pending:
        r = client.post(
            f"/agent-workflows/runs/{run['run_id']}/input",
            json={"request_id": q["request_id"], "answer": f"answer-for-{q['required_field']}"},
        )
        assert r.status_code == 200, r.text

    # The engine must have been called exactly once, on resume, with BOTH the
    # original payload key AND every elicited answer.
    assert len(calls) == 1
    input_data = calls[0]
    assert input_data["not_elicited_key"] == "keep-me"
    for q in pending:
        assert input_data[q["required_field"]] == f"answer-for-{q['required_field']}"

    client.delete(f"/agent-workflows/{wid}")


def test_final_answer_replay_executes_workflow_only_once(client, _cleanup_test_rows, monkeypatch):
    """FINDING 2 (critical): a replayed/duplicated final answer must not run
    the workflow twice — the agents this can trigger include email_dispatch,
    negotiation and supplier_interaction, so a double-execution means
    sending real emails twice. Spies on the engine's execute() boundary and
    counts calls instead of performing a real agent run.
    """
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    calls = _spy_on_engine(client, monkeypatch)

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    created_run_ids.append(run["run_id"])
    pending = run["pending"]
    assert len(pending) >= 1

    # Answer every question but the last one.
    for q in pending[:-1]:
        r = client.post(
            f"/agent-workflows/runs/{run['run_id']}/input",
            json={"request_id": q["request_id"], "answer": "x"},
        )
        assert r.status_code == 200, r.text

    last = pending[-1]
    body = {"request_id": last["request_id"], "answer": "y"}

    # The FINAL answer, submitted twice — e.g. a client/proxy retry.
    r1 = client.post(f"/agent-workflows/runs/{run['run_id']}/input", json=body)
    r2 = client.post(f"/agent-workflows/runs/{run['run_id']}/input", json=body)

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert len(calls) == 1, f"workflow executed {len(calls)} times, expected exactly 1"
    assert r1.json()["status"] == "completed"
    assert r2.json()["status"] == "completed"

    client.delete(f"/agent-workflows/{wid}")


def test_stale_executing_claim_can_be_reclaimed(client, _cleanup_test_rows):
    """FINDING 1(a): a run wedged in 'executing' by a killed worker (OOM,
    segfault, deploy restart, a hung LLM/GPU call that got reaped) must not
    be wedged forever with no recovery path — past STALE_EXECUTING_WINDOW it
    becomes reclaimable again. Equally important: a FRESH 'executing' run
    must NOT be reclaimable — that is the original double-execution guard
    still doing its job, and this test must prove both halves.
    """
    from datetime import timedelta

    from repositories import workflow_input_request_repo as reqrepo
    from services.db import get_conn

    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    # --- Half 1: a STALE 'executing' run CAN be reclaimed -------------------
    stale_run_id = "awf-stale-claim-test"
    created_run_ids.append(stale_run_id)
    reqrepo.create_run(stale_run_id, agent_workflow_id=wid, payload={}, status="pending")

    # Simulate a worker that claimed the run and was then killed mid-run,
    # never reaching finish_run(): force status='executing' with an
    # updated_at OLDER than the staleness window, via direct SQL (not
    # through claim_for_execution, since we need to backdate the timestamp).
    stale_age = reqrepo.STALE_EXECUTING_WINDOW + timedelta(minutes=5)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE proc.bp_workflow_run
                  SET status = 'executing', updated_at = now() - %s::interval
                WHERE run_id = %s""",
            (f"{stale_age.total_seconds()} seconds", stale_run_id),
        )
        cur.close()

    assert reqrepo.claim_for_execution(stale_run_id) is True, (
        "a run stuck 'executing' past STALE_EXECUTING_WINDOW must be reclaimable"
    )

    # --- Half 2: a FRESH 'executing' run CANNOT be reclaimed ----------------
    fresh_run_id = "awf-fresh-claim-test"
    created_run_ids.append(fresh_run_id)
    reqrepo.create_run(fresh_run_id, agent_workflow_id=wid, payload={}, status="pending")

    assert reqrepo.claim_for_execution(fresh_run_id) is True   # first claim wins, sets updated_at = now()
    assert reqrepo.claim_for_execution(fresh_run_id) is False, (
        "a run that is genuinely still executing (fresh updated_at) must NOT be reclaimable"
    )

    client.delete(f"/agent-workflows/{wid}")


def test_answering_one_run_cannot_mutate_another_runs_request(client, _cleanup_test_rows):
    """CRITICAL: proc.bp_workflow_input_request is the human-in-the-loop audit
    trail. A request_id must be scoped to the run it was raised for --
    answering run B's endpoint with a request_id that belongs to run A must
    be rejected outright, and run A's row must be left COMPLETELY untouched:
    same status, same (absent) answer, same (null) answered_at. Silently
    rewriting another run's audit trail -- including one that has already
    completed -- destroys the one guarantee this table exists to provide.
    """
    from services.db import get_conn

    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid_a = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid_a)
    wid_b = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid_b)

    run_a = client.post(f"/agent-workflows/{wid_a}/run", json={"payload": {}}).json()
    created_run_ids.append(run_a["run_id"])
    run_b = client.post(f"/agent-workflows/{wid_b}/run", json={"payload": {}}).json()
    created_run_ids.append(run_b["run_id"])

    assert run_a["status"] == "awaiting_input"
    assert run_b["status"] == "awaiting_input"
    assert run_a["pending"], "expected run A to have at least one open question"

    request_id_a = run_a["pending"][0]["request_id"]

    def _row(request_id):
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT status, answer, answered_by, answered_at
                     FROM proc.bp_workflow_input_request WHERE request_id = %s""",
                (request_id,),
            )
            r = cur.fetchone()
            cur.close()
            return r

    before = _row(request_id_a)
    assert before[0] == "pending"
    assert before[3] is None          # answered_at

    # Answer run B's endpoint using run A's request_id -- a caller mixing up
    # (or forging) a run_id/request_id pair.
    r = client.post(
        f"/agent-workflows/runs/{run_b['run_id']}/input",
        json={"request_id": request_id_a, "answer": "hijacked"},
    )
    assert r.status_code in (400, 404), r.text

    after = _row(request_id_a)
    assert after == before, "run A's request row must be left COMPLETELY untouched"

    # Run A's own question is still pending, unaffected.
    got_a = client.get(f"/agent-workflows/runs/{run_a['run_id']}")
    assert got_a.json()["status"] == "awaiting_input"

    client.delete(f"/agent-workflows/{wid_a}")
    client.delete(f"/agent-workflows/{wid_b}")


def test_get_run_surfaces_status_and_404s_for_unknown_run(client, _cleanup_test_rows):
    """FINDING 1(b): GET /agent-workflows/runs/{run_id} must return the run's
    real persisted status — not just its pending questions, which made an
    unknown run and a completed run indistinguishable — and must 404 for a
    run_id that was never created.
    """
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    created_run_ids.append(run["run_id"])
    assert run["status"] == "awaiting_input"

    got = client.get(f"/agent-workflows/runs/{run['run_id']}")
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["status"] == "awaiting_input"
    assert body["pending"]

    r = client.get("/agent-workflows/runs/no-such-run-id-at-all")
    assert r.status_code == 404

    client.delete(f"/agent-workflows/{wid}")


def test_a_blank_payload_value_still_elicits_not_executes(client, _cleanup_test_rows, monkeypatch):
    """CRITICAL 2, THE regression guard, at the HTTP layer.

    POST /agent-workflows/{id}/run with a payload of {"s3_prefix": "", "query": None}
    used to satisfy BOTH nodes' elicit groups (key presence, values never inspected)
    and execute the workflow immediately, silently sweeping the entire default
    corpus data_extraction would fall back to. It must now ask the same questions
    as an empty payload would.
    """
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    calls = _spy_on_engine(client, monkeypatch)

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    run = client.post(
        f"/agent-workflows/{wid}/run",
        json={"payload": {"s3_prefix": "", "query": None}},
    ).json()
    created_run_ids.append(run["run_id"])

    assert run["status"] == "awaiting_input"
    fields = {q["required_field"] for q in run["pending"]}
    assert "s3_object_keys" in fields
    assert "query" in fields
    assert calls == [], "the workflow must not have executed against a blank source"

    client.delete(f"/agent-workflows/{wid}")


def test_submitting_a_blank_answer_is_rejected(client, _cleanup_test_rows):
    """CRITICAL 2: POST /runs/{id}/input {"answer": ""} used to mark the question
    answered and (once it was the last open question) execute the workflow on a
    blank. It must be rejected outright, and the question must remain open.
    """
    created_workflow_ids, created_run_ids = _cleanup_test_rows

    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]
    created_workflow_ids.append(wid)

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    created_run_ids.append(run["run_id"])
    assert run["status"] == "awaiting_input"
    request_id = run["pending"][0]["request_id"]

    r = client.post(
        f"/agent-workflows/runs/{run['run_id']}/input",
        json={"request_id": request_id, "answer": ""},
    )
    assert r.status_code == 400, r.text

    got = client.get(f"/agent-workflows/runs/{run['run_id']}")
    still_open_ids = {q["request_id"] for q in got.json()["pending"]}
    assert request_id in still_open_ids, "a rejected blank answer must leave the question open"

    client.delete(f"/agent-workflows/{wid}")
