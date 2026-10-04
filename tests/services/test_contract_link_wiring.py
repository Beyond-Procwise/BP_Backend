"""Something in the product asks for a contract's parent, unprompted.

THE failure this file exists to prevent, stated once: for two days
`contract_links.propose_parent_links()` was a proven, 44-test, live-verified
runner that **nothing called**. Its own docstring said so. A runner nothing
calls reports exactly what a runner that found nothing reports, which is the
same trap `contract_succession.py` fell into and the reason
`test_the_runner_is_actually_called` exists one file over — that test proves the
function works WHEN CALLED, not that anything calls it.

`test_the_product_itself_calls_the_proposer` is the load-bearing test here. It
reads source, deliberately: a mocked call proves a seam, and the seam was never
what was missing.

Most of this file needs no database. The two scoped-pass tests do, and are
skipped without PROCWISE_TEST_LIVE_DB=1:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/test_contract_link_wiring.py -v
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.services import contract_links as CL                        # noqa: E402
from src.services.extraction import promotion                        # noqa: E402
from src.services.governed_limits import LimitUnavailable            # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live_only = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

ISSUE = "contract_parent_proposed"


# ---------------------------------------------------------------- the guard

def _calls_within(func_node) -> set[str]:
    """Every function name called inside one function body."""
    import ast
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _function_named(path: Path, name: str):
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_promotion_reaches_the_proposer():
    """Link 2 of the chain: the hook actually calls the runner.

    THE WEAKNESS THIS REPLACED, found by breaking it on purpose: the first
    version grepped all of src/ for the NAME `propose_parent_links`. With the
    call deleted from promote(), the words still sat inside the now-orphaned
    hook body, so the guard stayed green on exactly the unwired state it exists
    to catch. A corpus-wide name scan cannot tell a caller from a mention, so
    the chain is asserted link by link instead — each link red on its own break.
    """
    hook = _function_named(_ROOT / "src/services/extraction/promotion.py",
                           "propose_contract_parent")
    assert hook is not None, "the promotion hook is gone"
    assert "propose_parent_links" in _calls_within(hook), (
        "the promotion hook does not reach contract_links.propose_parent_links"
    )


def test_promotion_itself_is_the_caller():
    """And it is promote() specifically, the one funnel all three promotion
    paths go through. A hook defined beside promote() but called from nowhere is
    the same unwired state wearing a function definition."""
    promote = _function_named(_ROOT / "src/services/extraction/promotion.py", "promote")
    assert promote is not None, "promote() is gone — this test is out of date"
    assert "propose_contract_parent" in _calls_within(promote), (
        "promote() does not ask for a promoted contract's parent, so nothing on "
        "the live watcher path proposes one"
    )


# ------------------------------------------------- the promotion-time hook

class _Spy:
    def __init__(self, raises: Exception | None = None):
        self.calls: list[dict] = []
        self.raises = raises

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return {"proposed": 1, "contested": 0, "no_candidate": 0,
                "considered": {"children": 1}, "details": []}


@pytest.fixture()
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setattr(CL, "propose_parent_links", s)
    monkeypatch.setattr(promotion, "_proposals_enabled", lambda: True)
    return s


def test_a_promoted_contract_is_asked_for_its_parent(spy):
    promotion.propose_contract_parent("contract", "FA-2026-0042")
    assert spy.calls == [{"contract_id": "FA-2026-0042"}], (
        "a promoted contract must ask for its parent, scoped to itself"
    )


def test_a_promoted_invoice_is_not(spy):
    for doc_type in ("invoice", "quote", "purchase_order"):
        promotion.propose_contract_parent(doc_type, "INV-1")
    assert spy.calls == [], "only a contract sits under a contract"


def test_a_promotion_with_no_doc_pk_proposes_nothing(spy):
    """There is nothing to scope the pass to, and an unscoped pass here would
    score the whole corpus on a document that did not even get a key."""
    promotion.propose_contract_parent("contract", None)
    promotion.propose_contract_parent("contract", "")
    assert spy.calls == []


def test_a_proposer_that_fails_never_fails_the_promotion(monkeypatch):
    """The _stg write is already committed by the time this runs. A proposal is
    evidence ABOUT the document, like provenance, and must not undo it."""
    monkeypatch.setattr(CL, "propose_parent_links", _Spy(raises=RuntimeError("boom")))
    monkeypatch.setattr(promotion, "_proposals_enabled", lambda: True)
    promotion.propose_contract_parent("contract", "FA-2026-0042")   # must not raise


# ------------------------------------------------------------- governance

def test_a_policy_that_says_no_proposes_nothing(monkeypatch):
    s = _Spy()
    monkeypatch.setattr(CL, "propose_parent_links", s)
    monkeypatch.setattr(promotion, "_proposals_enabled", lambda: False)
    promotion.propose_contract_parent("contract", "FA-2026-0042")
    assert s.calls == []


def test_an_unreadable_policy_proposes_nothing(monkeypatch):
    """Fails CLOSED, unlike the governance read paths.

    This writes rows into a queue a person works. 'the policy could not be read'
    must not become 'so go ahead' — that is the fail-open hole P6 found, and it
    is the wrong default for a writer.
    """
    s = _Spy()
    monkeypatch.setattr(CL, "propose_parent_links", s)

    def _raise():
        raise LimitUnavailable("no autonomous_operation policy")

    monkeypatch.setattr(promotion, "_proposals_enabled", _raise)
    promotion.propose_contract_parent("contract", "FA-2026-0042")
    assert s.calls == []


def test_the_governed_flag_is_read_from_policy_not_only_the_environment():
    """The env var is the deprecated override, not the source of truth.

    Read out of the AST rather than through `promotion._proposals_enabled`,
    because tests/governance/test_governed_limit_callers.py rightly refuses a
    governed-limit accessor referenced as a VALUE anywhere in src/, tests/ or
    scripts/ — an accessor handed around unqualified is how a limit gets read
    once and cached stale.
    """
    import ast
    fn = _function_named(_ROOT / "src/services/extraction/promotion.py",
                         "_proposals_enabled")
    assert fn is not None, "the governed flag has no reader"
    strings = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "autonomous_operation" in strings
    assert "contract_parent_proposals_enabled" in strings
    assert "CONTRACT_PARENT_PROPOSALS_ENABLED" in strings, (
        "the deprecated env override is not honoured, so a deployment that set "
        "it would silently lose its setting"
    )


# ---------------------------------------------------- the scheduler backstop

def test_the_backstop_job_is_registered(monkeypatch):
    from src.services import backend_scheduler as BS
    monkeypatch.setattr(BS, "_governed_limit", lambda *a, **k: True)
    sched = BS.BackendScheduler.__new__(BS.BackendScheduler)
    sched._jobs = {}
    import threading
    sched._lock = threading.RLock()
    sched._register_contract_link_job()
    job = sched._jobs.get(BS.BackendScheduler.CONTRACT_LINK_JOB_NAME)
    assert job is not None, "the backstop pass is not scheduled"
    assert job.interval >= timedelta(hours=1), (
        "a corpus-wide pass writing to a person's queue does not belong on a "
        "minutes-scale tick"
    )


def test_the_backstop_job_is_not_registered_when_policy_says_no(monkeypatch):
    from src.services import backend_scheduler as BS
    monkeypatch.setattr(BS, "_governed_limit", lambda *a, **k: False)
    sched = BS.BackendScheduler.__new__(BS.BackendScheduler)
    sched._jobs = {}
    import threading
    sched._lock = threading.RLock()
    sched._register_contract_link_job()
    assert BS.BackendScheduler.CONTRACT_LINK_JOB_NAME not in sched._jobs


def test_the_backstop_job_reaches_the_proposer():
    """Link 3: the sweep's runner calls the runner. A registered job whose body
    does not reach the proposer is a tick that does nothing."""
    from src.services import backend_scheduler as BS
    runner = _function_named(_ROOT / "src/services/backend_scheduler.py",
                             "_run_contract_link_job")
    assert runner is not None, "the backstop job has no runner"
    assert "propose_parent_links" in _calls_within(runner)
    assert BS.BackendScheduler.CONTRACT_LINK_JOB_NAME == "contract-parent-links"


def test_the_backstop_job_is_registered_at_startup():
    """Registered by _register_default_jobs, not by something a reader must find."""
    import inspect
    from src.services import backend_scheduler as BS
    src = inspect.getsource(BS.BackendScheduler._register_default_jobs)
    assert "_register_contract_link_job" in src


# ------------------------------------------------------- the scoped pass

@pytest.fixture()
def two_children():
    """One master agreement and TWO children naming it, cleaned up afterwards."""
    tag = uuid.uuid4().hex[:8].upper()
    msa, sow, var = f"MSA-{tag}", f"SOW-{tag}", f"VAR-{tag}"
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT discrepancy_id FROM proc.bp_extraction_discrepancy "
                    "WHERE issue_type = %s", (ISSUE,))
        before = {r[0] for r in cur.fetchall()}
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, contract_start_date,
                    contract_end_date, resolved_doc_type, resolved_role, type_agreement)
               VALUES (%s, 'Master Services Agreement Helix Migration', %s,
                       '2026-01-01', '2027-12-31',
                       'doctype.master_agreement', 'role.master', 'refined')""",
            (msa, f"S-{tag}"),
        )
        for cid, title, dt in ((sow, "Statement of Work Helix Migration", "doctype.sow"),
                               (var, "Variation Helix Migration", "doctype.variation")):
            cur.execute(
                """INSERT INTO proc.bp_contracts
                       (contract_id, contract_title, supplier_id, contract_start_date,
                        contract_end_date, resolved_doc_type, resolved_role,
                        type_agreement, parent_agreement_ref)
                   VALUES (%s, %s, %s, '2026-03-01', '2026-09-30', %s, 'role.master',
                           'refined', %s)""",
                (cid, title, f"S-{tag}", dt, msa),
            )
    yield {"msa": msa, "sow": sow, "var": var}
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT discrepancy_id FROM proc.bp_extraction_discrepancy "
                    "WHERE issue_type = %s", (ISSUE,))
        added = {r[0] for r in cur.fetchall()} - before
        if added:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE discrepancy_id = ANY(%s)", (sorted(added),))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                    "WHERE doc_pk_candidate = ANY(%s)", ([msa, sow, var],))
        cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)",
                    ([msa, sow, var],))


def _open(contract_id):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT expected_value FROM proc.bp_extraction_discrepancy
                WHERE issue_type = %s AND doc_pk_candidate = %s
                  AND coalesce(status,'open') <> 'resolved'""", (ISSUE, contract_id))
        return [r[0] for r in cur.fetchall()]


@live_only
def test_a_scoped_pass_proposes_for_that_child_alone(two_children):
    result = CL.propose_parent_links(contract_id=two_children["sow"])
    assert result["considered"]["children"] == 1, result
    assert _open(two_children["sow"]) == [two_children["msa"]]
    assert _open(two_children["var"]) == [], (
        "a pass scoped to one document touched another document's queue"
    )


@live_only
def test_a_scoped_pass_for_an_unknown_contract_does_not_become_a_corpus_pass(two_children):
    result = CL.propose_parent_links(contract_id=f"NOSUCH-{uuid.uuid4().hex[:6]}")
    assert result["considered"]["children"] == 0, result
    assert _open(two_children["sow"]) == []
    assert _open(two_children["var"]) == []
