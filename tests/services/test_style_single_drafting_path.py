"""Invariant 10: one email drafting path.

"Two drafting paths means two provenance stories and only one of them will be correct."

The subsystem this suite belongs to created that problem before it solved it: a style
engine that could generate email, sitting beside an EmailDraftingAgent that already did.
These tests hold the resolution in place — the agent is the single generator, the style
engine governs it rather than competing with it, and nothing else in the codebase produces
a real draft.
"""

from __future__ import annotations

import ast
import inspect
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.integration import (
    AppliedStyle,
    augment_system_prompt,
    resolve_intent,
    resolve_user_ref,
    style_for_draft,
)
from services.style.profile import parse_profile
from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
from tests.services.test_style_compiler import STUB_PROFILE

PROFILE = parse_profile(STUB_PROFILE)


def _agent_class():
    """Import EmailDraftingAgent, stubbing a declared dependency this environment lacks.

    sqlalchemy is in requirements.txt but absent here (PEP 668 blocks installing into the
    system Python), and importing the agent pulls it in transitively via query_engine.
    Stubbing it lets these tests exercise the real methods rather than pattern-match the
    source, which is what makes them worth having.
    """

    import importlib.machinery as machinery
    import sys
    import types

    if "sqlalchemy" not in sys.modules:
        stub = types.ModuleType("sqlalchemy")
        exc = types.ModuleType("sqlalchemy.exc")

        class SQLAlchemyError(Exception):
            pass

        exc.SQLAlchemyError = SQLAlchemyError
        exc.OperationalError = SQLAlchemyError
        stub.exc = exc
        stub.__version__ = "0-stub"
        # `datasets` calls find_spec("sqlalchemy") at import time and rejects a stub
        # without one.
        stub.__spec__ = machinery.ModuleSpec("sqlalchemy", None)
        exc.__spec__ = machinery.ModuleSpec("sqlalchemy.exc", None)
        sys.modules["sqlalchemy"] = stub
        sys.modules["sqlalchemy.exc"] = exc

    from agents.email_drafting_agent import EmailDraftingAgent

    return EmailDraftingAgent


class _Agent:
    """An EmailDraftingAgent with its heavy __init__ bypassed.

    Only the style seam is under test here; constructing the real agent would need a
    live AgentNick, a database and a model.
    """

    def __new__(cls):
        agent = object.__new__(_agent_class())
        agent._style_for_current_draft = None
        agent._style_resolved = False
        agent._style_context = {}
        agent._style_sender = None
        return agent


def _with_resolved_style(agent, style):
    """Inject a settled style. Both fields, because the sentinel is what stops the agent
    re-resolving — setting only the value would have it looked up again and discarded."""

    agent._style_for_current_draft = style
    agent._style_resolved = True
    return agent


def _connect():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except Exception:  # pragma: no cover
        pass
    if not os.getenv("DB_HOST"):
        pytest.skip("no database configured")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=10,
        )
    except psycopg2.Error as exc:
        pytest.skip(f"database unreachable: {exc}")
    conn.autocommit = True
    return conn


@pytest.fixture
def bench():
    conn = _connect()
    user_ref = f"test-{uuid.uuid4().hex[:12]}"
    repo = StyleProfileRepository(conn)
    try:
        yield conn, user_ref, repo
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM proc.draft_rfq_emails WHERE style_user_ref = %s",
                        (user_ref,))
            cur.execute("DELETE FROM proc.bp_style_exemplar WHERE user_ref = %s", (user_ref,))
            cur.execute("DELETE FROM proc.bp_style_profile WHERE user_ref = %s", (user_ref,))
        conn.close()


# --- exactly one class of that name ---------------------------------------------------

def test_only_one_class_is_called_emaildraftingagent():
    """Two classes sharing the name, one of them inside the live orchestration module, is
    how the wrong one eventually gets wired up."""
    found = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "EmailDraftingAgent":
                found.append(str(path))
    assert found == ["src/agents/email_drafting_agent.py"], found


def test_the_simulation_cannot_produce_a_real_draft():
    """It is a workflow-shape harness. It reaches no database and no mail provider, so it
    has no provenance story to compete with."""
    source = Path("src/orchestration/procurement_workflow.py").read_text()
    for forbidden in ("draft_rfq_emails", "get_conn", "psycopg2",
                      "EmailService", "send_email", "smtplib"):
        assert forbidden not in source, forbidden


def test_the_simulations_exports_are_not_broken_by_the_rename():
    import orchestration.procurement_workflow as mod

    assert [name for name in mod.__all__ if not hasattr(mod, name)] == []
    assert "SimulatedRFQDrafter" in mod.__all__


# --- the style engine governs rather than competes --------------------------------------

def test_the_style_engine_exposes_no_second_generator_to_the_agent():
    """integration.py hands the agent RULES. If it handed back a finished email, that
    would be the second path arriving through the back door."""
    import services.style.integration as mod

    source = inspect.getsource(mod)
    for forbidden in ("ollama_generate", "StyleDraftingService", "_call_model"):
        assert forbidden not in source, forbidden


def test_the_agent_applies_style_at_the_prompt_seam():
    """Exercised, not pattern-matched: the real method must fold the rules in."""
    agent = _Agent()
    agent.resolve_prompt = lambda name: "BASE PROMPT for " + name
    _with_resolved_style(agent, AppliedStyle(
        rules="1. Open with: Hi {first_name},", user_ref="u", intent="_all",
        fallback_level=0, fallback_reason="", profile_id=1, profile_version=1,
        exemplar_ids=[], exemplar_set_hash=None,
    ))
    for method in ("_sys_compose_rfq", "_sys_compose_response", "_sys_negotiation_playbook"):
        prompt = getattr(agent, method)()
        assert "BASE PROMPT" in prompt, method
        assert "WRITING STYLE" in prompt, method
        assert "Hi {first_name}," in prompt, method


def test_the_agent_prompts_are_untouched_without_a_profile():
    """The no-regression guarantee, exercised on the real methods."""
    agent = _Agent()
    agent.resolve_prompt = lambda name: "BASE PROMPT for " + name
    _with_resolved_style(agent, None)
    for method in ("_sys_compose_rfq", "_sys_compose_response", "_sys_negotiation_playbook"):
        assert getattr(agent, method)() == "BASE PROMPT for " + {
            "_sys_compose_rfq": "email_compose_rfq",
            "_sys_compose_response": "email_compose_response",
            "_sys_negotiation_playbook": "negotiation_playbook_system",
        }[method]


def test_polishing_is_deliberately_not_styled():
    """Polish rewrites an already-composed email. Applying the voice twice compounds it."""
    agent = _Agent()
    agent.resolve_prompt = lambda name: "POLISH BASE"
    _with_resolved_style(agent, AppliedStyle(
        rules="1. Open with: Hi {first_name},", user_ref="u", intent="_all",
        fallback_level=0, fallback_reason="", profile_id=1, profile_version=1,
        exemplar_ids=[], exemplar_set_hash=None,
    ))
    assert agent._sys_polish() == "POLISH BASE"


# --- inert without an approved profile ---------------------------------------------------

def test_no_profile_means_no_style_and_no_behaviour_change(bench):
    """An organisation that never asked for style learning must draft exactly as before.
    The resolver would return a platform baseline here; applying it would rewrite the
    voice of every RFQ they send."""
    conn, user_ref, _ = bench
    assert style_for_draft(user_ref=user_ref, conn=conn) is None


def test_an_approved_profile_is_applied(bench):
    conn, user_ref, repo = bench
    rec = repo.insert_version(user_ref=user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    approved = repo.approve(rec.profile_id, "tester")

    style = style_for_draft(user_ref=user_ref, conn=conn)
    assert style is not None
    assert style.profile_id == approved.profile_id
    assert style.fallback_level == 0
    assert "Hi {first_name}," in style.rules


def test_a_missing_user_ref_yields_no_style():
    assert style_for_draft(user_ref=None) is None
    assert style_for_draft(user_ref="") is None


def test_a_failing_lookup_never_breaks_drafting():
    """Losing the voice is a degraded email; losing the email is a stalled sourcing
    round."""

    class _Exploding:
        def cursor(self):
            raise RuntimeError("database on fire")

    assert style_for_draft(user_ref="someone", conn=_Exploding()) is None


def test_the_base_prompt_passes_through_untouched_without_style():
    base = "You are drafting an RFQ."
    assert augment_system_prompt(base, None) == base


def test_style_is_appended_not_substituted():
    """The agent's own prompts carry the procurement substance; the profile carries only
    the voice. Replacing one with the other trades correct content for correct tone."""
    base = "You are drafting an RFQ. It must contain a pricing table."
    style = AppliedStyle(
        rules="1. Open with: Hi {first_name},", user_ref="u", intent="_all",
        fallback_level=0, fallback_reason="", profile_id=1, profile_version=1,
        exemplar_ids=[], exemplar_set_hash=None,
    )
    merged = augment_system_prompt(base, style)
    assert merged.startswith(base)
    assert "pricing table" in merged
    assert "Hi {first_name}," in merged
    assert "the instructions above win" in merged


# --- identity and intent -----------------------------------------------------------------

def test_an_explicit_user_ref_wins_over_the_sender():
    assert resolve_user_ref({"user_ref": "cognito-sub-1"}, "nick@acme.example") == "cognito-sub-1"


def test_the_sending_mailbox_is_the_fallback_identity():
    """An agent-initiated RFQ has no signed-in user but is still written in somebody's
    voice."""
    assert resolve_user_ref({}, "Nick@Acme.Example") == "nick@acme.example"
    assert resolve_user_ref({}, None) is None
    assert resolve_user_ref({}, "not-an-address") is None


@pytest.mark.parametrize("interaction,intent", [
    ("rfq", "rfq_invite"),
    ("negotiation", "negotiation_counter"),
    ("clarification", "clarification_request"),
    ("award", "award_notification"),
    ("reminder", "reminder"),
])
def test_the_agents_interaction_types_map_onto_the_style_vocabulary(interaction, intent):
    """One vocabulary, not two — the style codes were seeded as the union of both sets."""
    assert resolve_intent(interaction) == intent


@pytest.mark.parametrize("unknown", ["something_new", "", None])
def test_an_unrecognised_interaction_falls_back_rather_than_guessing(unknown):
    """A wrong intent selects the wrong profile, which is worse than the general one."""
    assert resolve_intent(unknown) == USER_LEVEL_INTENT


# --- provenance -------------------------------------------------------------------------

def test_applied_style_maps_onto_the_draft_columns():
    style = AppliedStyle(
        rules="x", user_ref="u", intent="rfq_invite", fallback_level=1,
        fallback_reason="r", profile_id=7, profile_version=3,
        exemplar_ids=[1, 2], exemplar_set_hash="abc",
    )
    columns = style.as_draft_columns()
    assert columns["style_user_ref"] == "u"
    assert columns["style_intent"] == "rfq_invite"
    assert columns["style_profile_id"] == 7
    assert columns["style_profile_version"] == 3
    assert columns["style_fallback_level"] == 1
    assert columns["style_exemplar_ids"] == [1, 2]


def test_an_empty_exemplar_list_is_stored_as_null_not_an_empty_array():
    style = AppliedStyle(
        rules="x", user_ref="u", intent="_all", fallback_level=0, fallback_reason="",
        profile_id=1, profile_version=1, exemplar_ids=[], exemplar_set_hash=None,
    )
    assert style.as_draft_columns()["style_exemplar_ids"] is None


def test_the_agent_writes_style_provenance_columns():
    source = inspect.getsource(_agent_class()._store_draft)
    for column in ("style_user_ref", "style_intent", "style_profile_id",
                   "style_profile_version", "style_fallback_level",
                   "style_exemplar_ids", "style_exemplar_set_hash"):
        assert column in source, column


def test_provenance_is_null_when_no_style_governed():
    """A row citing a profile it never used would be worse than one citing nothing,
    because it would look explicable."""
    source = inspect.getsource(_agent_class()._store_draft)
    assert '"style_user_ref": None' in source
    assert "applied_style is not None" in source


# --- the per-run cache -------------------------------------------------------------------

def test_style_state_exists_before_any_entry_point_runs():
    """from_prompt and from_decision bypass run(); none of them may depend on attribute
    lookup order."""
    assert "_style_for_current_draft = None" in inspect.getsource(_agent_class().__init__)


def test_every_entry_point_resets_the_style_cache():
    """The agent is long-lived and loops over suppliers. A style left over from a previous
    run would put one sender's voice on another sender's email — a plausible, wrong email
    nobody notices."""
    agent_cls = _agent_class()
    for entry in ("run", "from_prompt", "from_decision"):
        assert "_reset_style" in inspect.getsource(getattr(agent_cls, entry)), entry


def test_resetting_actually_clears_a_previous_senders_style():
    """Exercised: the cache from supplier A must not survive into supplier B's draft."""
    agent = _Agent()
    _with_resolved_style(agent, AppliedStyle(
        rules="stale", user_ref="previous-sender", intent="_all", fallback_level=0,
        fallback_reason="", profile_id=1, profile_version=1,
        exemplar_ids=[], exemplar_set_hash=None,
    ))
    agent.agent_nick = None
    agent._reset_style(None, {"sender": "someone-else@acme.example"})

    assert agent._style_for_current_draft is None
    assert agent._style_resolved is False, "the sentinel must clear too, or the stale None sticks"
    assert agent._style_sender == "someone-else@acme.example"


def test_the_style_lookup_runs_once_per_run_not_once_per_prompt():
    """None is a legitimate result — no approved profile — so conflating it with 'not
    looked up yet' would re-run the resolver for every supplier in a drafting loop."""
    agent = _Agent()
    agent.resolve_prompt = lambda name: "BASE"
    agent.agent_nick = None
    agent._reset_style(None, {})

    calls = []
    import services.style.integration as integration

    original = integration.style_for_draft
    integration.style_for_draft = lambda **kw: calls.append(kw) or None
    try:
        agent._sys_compose_rfq()
        agent._sys_compose_response()
        agent._sys_negotiation_playbook()
    finally:
        integration.style_for_draft = original

    assert len(calls) == 1, f"resolver ran {len(calls)} times for one draft"
