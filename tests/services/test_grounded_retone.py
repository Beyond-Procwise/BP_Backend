"""The tone control may change the wording. It may not change the money.

A supplier-query email accuses a named company of over-billing, and every figure in it is
filled from a stored row rather than written by a model. Offering a tone control puts a
model back in that path, so the model's output is checked against the draft it was given:
nothing new may appear, and nothing that matters may disappear. A rewrite that fails goes
in the bin and the grounded draft is sent instead -- the buyer is told, never silently
served the template as if the tone had been applied.
"""

import pytest

from src.services import grounded_retone as gr


class FakeAgent:
    """Anything with call_ollama. Records what it was asked, returns what it was told to."""

    def __init__(self, reply="", raises=False):
        self.reply, self.raises, self.calls = reply, raises, []

    def call_ollama(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("ollama is not reachable")
        return {"message": {"content": self.reply}}


ORIGINAL = (
    "Hello Northfield Laboratory Supplies Ltd,\n\n"
    "Invoice INV000047-1A appears to duplicate invoice INV000047-1: both are for "
    "147,783.11 USD against purchase order PO-9912.\n\n"
    "Many thanks,\nAccounts Payable"
)
NAME = "Northfield Laboratory Supplies Ltd"
# The money and the references only. The supplier name goes to must_name -- see the
# salutation tests below for why it is not held byte-exact.
MUST_KEEP = ["147,783.11 USD", "INV000047-1A", "INV000047-1", "PO-9912"]


# --------------------------------------------------------------------------- numbers

def test_numbers_ignores_thousands_separators():
    # "147,783.11" and "147783.11" are the same money. A guard that thought otherwise
    # would reject every rewrite that reformatted a figure it did not actually change.
    assert gr.numbers("147,783.11") == gr.numbers("147783.11")


def test_numbers_finds_digits_inside_references():
    assert "000047" in gr.numbers("INV000047-1A")


# --------------------------------------------------------------------------- violations

def test_a_faithful_rewrite_passes():
    rewritten = (
        "Hello Northfield Laboratory Supplies Ltd,\n\n"
        "We think invoice INV000047-1A may be a duplicate of INV000047-1 — both come to "
        "147,783.11 USD against purchase order PO-9912.\n\n"
        "Many thanks,\nAccounts Payable"
    )
    assert gr.violations(ORIGINAL, rewritten, must_keep=MUST_KEEP) == []


def test_an_invented_figure_is_a_violation():
    # The single most dangerous failure on this path: a plausible number nobody stored.
    rewritten = ORIGINAL.replace("147,783.11 USD", "147,783.11 USD (plus 12,500.00 USD VAT)")
    problems = gr.violations(ORIGINAL, rewritten, must_keep=MUST_KEEP)
    assert problems and any("12500.00" in p for p in problems)


def test_a_changed_amount_is_a_violation():
    rewritten = ORIGINAL.replace("147,783.11", "174,783.11")   # two digits swapped
    assert gr.violations(ORIGINAL, rewritten, must_keep=MUST_KEEP)


def test_a_dropped_reference_is_a_violation():
    rewritten = ORIGINAL.replace(" against purchase order PO-9912", "")
    problems = gr.violations(ORIGINAL, rewritten, must_keep=MUST_KEEP)
    assert problems and any("PO-9912" in p for p in problems)


def test_a_dropped_supplier_name_is_a_violation():
    rewritten = ORIGINAL.replace("Northfield Laboratory Supplies Ltd", "there")
    assert gr.violations(ORIGINAL, rewritten, must_keep=[], must_name=NAME)


# The supplier name is a salutation, not a figure, so it is held to a looser rule than the
# money. Found live: this corpus names suppliers with a trailing counter, and the byte-exact
# rule binned a flawless rewrite on nearly every one of them.

def test_a_trailing_counter_may_be_dropped_from_the_name():
    original = "Hello Ashcroft Logistics 14,\n\nInvoice INV-1 is 10.00 GBP over.\n\nThanks"
    rewritten = "Hi Ashcroft Logistics - INV-1 looks 10.00 GBP over. Thanks"
    assert gr.violations(original, rewritten, must_keep=["10.00 GBP", "INV-1"],
                         must_name="Ashcroft Logistics 14") == []


def test_renaming_the_company_is_still_a_violation():
    original = "Hello Ashcroft Logistics 14,\n\nInvoice INV-1 is 10.00 GBP over.\n\nThanks"
    rewritten = "Hi Ashcroft Freight - INV-1 looks 10.00 GBP over. Thanks"
    problems = gr.violations(original, rewritten, must_keep=[],
                             must_name="Ashcroft Logistics 14")
    assert problems and "no longer names" in problems[0]


def test_the_looser_name_rule_does_not_reach_the_figures():
    # A rewrite that drops the trailing digits of an INVOICE number is not a salutation
    # tidy-up. must_keep stays byte-exact.
    rewritten = ORIGINAL.replace("INV000047-1A", "INV000047")
    assert gr.violations(ORIGINAL, rewritten, must_keep=MUST_KEEP, must_name=NAME)


def test_must_keep_only_covers_what_the_draft_actually_said():
    # A finding with no PO reference must not have the rewrite rejected for "dropping" a
    # string that was never in the draft to begin with.
    original = "Hello Supplier,\n\nInvoice INV-1 is 10.00 GBP over.\n\nThanks"
    assert gr.violations(original, "Hi Supplier — INV-1 looks 10.00 GBP over. Thanks",
                         must_keep=["10.00 GBP", "INV-1", "PO-9912"]) == []


# --------------------------------------------------------------------------- retone

def test_formal_is_the_template_and_never_calls_a_model():
    agent = FakeAgent(reply="something else entirely")
    out = gr.retone(ORIGINAL, tone="formal", must_keep=MUST_KEEP, must_name=NAME, agent_nick=agent)
    assert out.body == ORIGINAL
    assert out.applied is False
    assert agent.calls == [], "the template IS the formal voice; there is nothing to ask"


def test_an_unknown_tone_is_refused_without_calling_a_model():
    agent = FakeAgent(reply="whatever")
    out = gr.retone(ORIGINAL, tone="shouty", must_keep=MUST_KEEP, must_name=NAME, agent_nick=agent)
    assert out.body == ORIGINAL and out.applied is False and agent.calls == []


def test_a_clean_rewrite_is_applied():
    rewritten = ("Hi Northfield Laboratory Supplies Ltd,\n\nQuick one — INV000047-1A looks "
                 "like a duplicate of INV000047-1, both 147,783.11 USD on PO-9912.\n\nThanks")
    out = gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME,
                    agent_nick=FakeAgent(reply=rewritten))
    assert out.applied is True
    assert out.body == rewritten
    assert out.note is None


def test_a_rewrite_that_moves_the_money_is_thrown_away():
    poisoned = ORIGINAL.replace("147,783.11", "247,783.11")
    out = gr.retone(ORIGINAL, tone="warm", must_keep=MUST_KEEP, must_name=NAME,
                    agent_nick=FakeAgent(reply=poisoned))
    assert out.applied is False
    assert out.body == ORIGINAL, "the grounded draft is what survives, always"
    assert out.note and "247783.11" in out.note


def test_a_failed_rewrite_says_so_rather_than_pretending():
    out = gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME,
                    agent_nick=FakeAgent(raises=True))
    assert out.applied is False and out.body == ORIGINAL
    assert out.note, "silence would read as 'the tone was applied'"


def test_an_empty_answer_is_not_a_rewrite():
    out = gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME,
                    agent_nick=FakeAgent(reply="   "))
    assert out.applied is False and out.body == ORIGINAL and out.note


def test_no_agent_at_all_degrades_to_the_template():
    out = gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME, agent_nick=None)
    assert out.applied is False and out.body == ORIGINAL and out.note


# --------------------------------------------------------------------------- resolve_caller

def test_a_bare_agent_nick_resolves_to_one_of_its_registered_agents():
    """The live failure this exists to stop.

    Routers carry `app.state.agent_nick`, an AgentNick — which does NOT inherit BaseAgent
    and has no call_ollama. Passed straight through, every re-tone degraded to "the
    drafting agent is not available" on a completely healthy server.
    """
    registered = FakeAgent(reply="ok")

    class BareAgentNick:                 # no call_ollama, exactly like the real one
        agents = {"email_drafting_agent": registered}

    assert gr.resolve_caller(BareAgentNick()) is registered


def test_a_caller_that_already_calls_is_used_as_is():
    agent = FakeAgent(reply="ok")
    assert gr.resolve_caller(agent) is agent


def test_nothing_at_all_resolves_to_nothing():
    assert gr.resolve_caller(None) is None


def test_the_resolved_caller_is_the_one_actually_asked():
    registered = FakeAgent(reply=ORIGINAL)

    class BareAgentNick:
        agents = {"email_drafting_agent": registered}

    gr.retone(ORIGINAL, tone="warm", must_keep=MUST_KEEP, must_name=NAME, agent_nick=BareAgentNick())
    assert len(registered.calls) == 1


def test_an_ollama_response_OBJECT_is_read_the_same_as_a_dict():
    """The live client returns objects, not dicts.

    Caught on the first real run: AgentNick returned a flawless rewrite that kept every
    figure, a dict-only reader fell through to str(response), and the guard compared the
    finding against the repr — `total_duration=4723328056`, `created_at='2026-07-31T...'`
    — and rejected a perfectly good rewrite for "introducing figures". The tone control
    would have looked implemented while never once working.
    """
    clean = ("Hi Northfield Laboratory Supplies Ltd, INV000047-1A looks like a duplicate "
             "of INV000047-1 — both 147,783.11 USD on PO-9912.")

    class Message:
        content = clean

    class ChatResponse:          # shaped like ollama's, repr full of big numbers
        message = Message()
        total_duration = 4723328056
        created_at = "2026-07-31T16:14:21.386996928Z"

    class ObjectAgent:
        def call_ollama(self, **kwargs):
            return ChatResponse()

    out = gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME, agent_nick=ObjectAgent())
    assert out.applied is True, out.note
    assert out.body == clean


def test_the_generate_shape_is_read_too():
    class GenerateResponse:
        response = "Hi Northfield Laboratory Supplies Ltd — INV000047-1A duplicates " \
                   "INV000047-1, both 147,783.11 USD on PO-9912."
        message = None

    class ObjectAgent:
        def call_ollama(self, **kwargs):
            return GenerateResponse()

    out = gr.retone(ORIGINAL, tone="warm", must_keep=MUST_KEEP, must_name=NAME, agent_nick=ObjectAgent())
    assert out.applied is True, out.note


def test_the_model_is_asked_at_temperature_zero_with_thinking_off():
    # Same call convention as every other AgentNick path: a reasoning model returns an
    # empty response with think on, and temperature must sit inside options or the client
    # raises TypeError.
    agent = FakeAgent(reply=ORIGINAL)
    gr.retone(ORIGINAL, tone="direct", must_keep=MUST_KEEP, must_name=NAME, agent_nick=agent)
    kwargs = agent.calls[0]
    assert kwargs["think"] is False
    assert kwargs["options"]["temperature"] == 0
    assert "direct" in kwargs["messages"][-1]["content"]
