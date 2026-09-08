"""The ask endpoint carries the reader's currency, and hands back the steps.

Two contract changes come with the analytic answer. The display currency and
the persona are chosen on screen — the currency by the control on Procurement
Home's top bar — and neither travelled with an ask request before, so the
answer could not be stated in the currency the dashboard beside it was using.
And the next steps must reach the client as data: a step carries an action and
the ids it applies to, and a chip that could only re-ask a sentence is the
round trip this work exists to remove.

The endpoint is called directly with a stub pipeline: this is about what the
route passes on and passes back, not about retrieval or auth.
"""

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.routers.workflows import AskRequest, ask_question  # noqa: E402

PAYLOAD = {
    "answer": '<section class="agent-answer">…</section>',
    "follow_ups": [],
    "retrieved_documents": [],
    "next_steps": [{"action_id": "analytic.supplier_concentration",
                    "label": "Review supplier concentration (35.4%)",
                    "rung": "concentration", "reason": "fact:CONCENTRATION_THRESHOLD_BREACHED",
                    "entity_refs": ["s-1", "s-2"]}],
}


class _Pipeline:
    def __init__(self):
        self.seen = {}

    def answer_question(self, **kwargs):
        self.seen = kwargs
        return PAYLOAD


class _Request:
    headers: dict = {}


def _ask(**fields):
    pipeline = _Pipeline()
    request = AskRequest(query="top 10 suppliers by spend", user_id="u1", **fields)
    result = asyncio.run(ask_question(request, _Request(), pipeline, principal=object()))
    return pipeline, result


def test_the_currency_chosen_on_screen_reaches_the_pipeline():
    pipeline, _ = _ask(display_currency="USD")
    assert pipeline.seen["display_currency"] == "USD"


def test_the_persona_reaches_the_pipeline():
    pipeline, _ = _ask(persona="cpo")
    assert pipeline.seen["persona"] == "cpo"


def test_a_request_that_says_nothing_about_either_still_works():
    pipeline, _ = _ask()
    assert pipeline.seen["display_currency"] is None
    assert pipeline.seen["persona"] is None


def test_a_dispatched_step_travels_as_an_action_not_as_a_sentence():
    # The chip's label is not a question. Sent back as text it would have to be
    # guessed at; sent back as an action id it is exactly what was offered.
    pipeline, _ = _ask(action_id="analytic.supplier_concentration")
    assert pipeline.seen["action_id"] == "analytic.supplier_concentration"


def test_the_next_steps_are_returned_to_the_client():
    _, result = _ask()
    assert result["next_steps"][0]["action_id"] == "analytic.supplier_concentration"
    assert result["next_steps"][0]["entity_refs"] == ["s-1", "s-2"]
