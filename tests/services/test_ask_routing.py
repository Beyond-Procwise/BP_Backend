"""Who answers a question typed into an Ask box.

The person's choice wins, then the screen's agent, then General. A choice or a
map entry that names an agent that is not there is answered by General, with a
note, never by an error.
"""

from src.services.ask_routing import (
    GENERAL, Responder, agent_label, askable_agents, choose_responder,
)

AVAILABLE = ["negotiation", "quote_comparison", "supplier_ranking"]
SCREENS = {"negotiations": "negotiation", "quotes": "quote_comparison",
           "invoices": "discrepancy_detection", "outbox": "email_dispatch"}


def test_a_chosen_agent_answers():
    r = choose_responder("quote_comparison", "negotiations", AVAILABLE, SCREENS)
    assert r == Responder("quote_comparison", "chosen")


def test_general_chosen_on_a_mapped_screen_is_general():
    r = choose_responder(GENERAL, "negotiations", AVAILABLE, SCREENS)
    assert r == Responder(None, "chosen")


def test_unknown_agent_falls_back_with_a_note():
    r = choose_responder("deleted_agent", "negotiations", AVAILABLE, SCREENS)
    assert r.slug is None and r.source == "fallback"
    assert r.note == "Deleted agent agent is not available, so this was answered generally."


def test_the_screen_agent_answers_when_nobody_chose():
    assert choose_responder(None, "negotiations", AVAILABLE, SCREENS) == Responder("negotiation", "screen")


def test_a_mapped_agent_that_is_not_loaded_falls_back():
    r = choose_responder(None, "invoices", AVAILABLE, SCREENS)
    assert r.slug is None and r.source == "fallback"
    assert "Discrepancy detection agent" in r.note


def test_a_mapped_sending_agent_is_never_offered():
    # email_dispatch can be loaded and still must not answer a question.
    registry = {s: object() for s in AVAILABLE + ["email_dispatch"]}
    r = choose_responder(None, "outbox", askable_agents(registry), SCREENS)
    assert r.slug is None and r.source == "fallback"


def test_an_unmapped_screen_and_no_choice_is_general():
    assert choose_responder(None, "home", AVAILABLE, SCREENS) == Responder(None, "default")
    assert choose_responder(None, None, AVAILABLE, {}) == Responder(None, "default")


def test_blank_and_cased_inputs_are_normalised():
    assert choose_responder("  Negotiation ", None, AVAILABLE, {}) == Responder("negotiation", "chosen")
    assert choose_responder("", "NEGOTIATIONS", AVAILABLE, SCREENS) == Responder("negotiation", "screen")


def test_askable_agents_drops_the_ones_that_send_or_act():
    registry = {s: object() for s in ["negotiation", "email_dispatch", "email_drafting",
                                     "email_watcher", "data_extraction", "supplier_interaction"]}
    assert askable_agents(registry) == ["negotiation"]


def test_labels_read_as_names():
    assert agent_label("quote_comparison") == "Quote comparison agent"
    assert agent_label(None) == "General"


def test_answered_by_shape():
    assert Responder("negotiation", "screen").answered_by() == {
        "agent": "negotiation", "label": "Negotiation agent", "source": "screen", "note": None}
    assert Responder(None, "default").answered_by()["agent"] is None


from src.services import ask_routing
from src.services.governed_limits import LimitUnavailable


def test_the_map_is_read_from_policy_and_lower_cased():
    read = lambda policy: {"screens": {"Negotiations": "Negotiation", "quotes": "quote_comparison"}}
    assert ask_routing.screen_agent_map(read) == {
        "negotiations": "negotiation", "quotes": "quote_comparison"}


def test_an_unreadable_map_routes_nothing():
    def read(policy):
        raise LimitUnavailable("no active ask_screen_agents policy")
    assert ask_routing.screen_agent_map(read) == {}


def test_a_malformed_map_routes_nothing():
    assert ask_routing.screen_agent_map(lambda p: {"screens": ["negotiations"]}) == {}
    assert ask_routing.screen_agent_map(lambda p: {}) == {}


def test_the_policy_name_read_is_ask_screen_agents():
    seen = []
    ask_routing.screen_agent_map(lambda p: seen.append(p) or {"screens": {}})
    assert seen == ["ask_screen_agents"]


from src.services.tool_runtime import ToolRunResult


class _Agent:
    def __init__(self, result=None, raises=None):
        self.result, self.raises, self.seen = result, raises, {}

    def reason(self, task, **kwargs):
        self.seen = {"task": task, **kwargs}
        if self.raises:
            raise self.raises
        return self.result


def test_the_agent_answers_with_the_person_and_the_screen():
    agent = _Agent(ToolRunResult(answer="Counter at £455."))
    text = ask_routing.agent_answer(agent, "what should I counter at?",
                                    user_id="nick", context="Screen: Negotiations")
    assert text == "Counter at £455."
    assert agent.seen["task"] == "what should I counter at?"
    assert agent.seen["user_id"] == "nick"
    assert "Screen: Negotiations" in agent.seen["extra_system"]


def test_an_errored_run_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(ToolRunResult(answer="", error="model down")),
                                    "q", user_id="u", context=None) is None


def test_a_blank_answer_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(ToolRunResult(answer="   ")),
                                    "q", user_id="u", context=None) is None


def test_a_raising_agent_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(raises=RuntimeError("boom")),
                                    "q", user_id="u", context=None) is None


def test_no_context_sends_no_extra_system():
    agent = _Agent(ToolRunResult(answer="ok"))
    ask_routing.agent_answer(agent, "q", user_id="u", context=None)
    assert "extra_system" not in agent.seen
