"""The critic's model call offers no tools and gates no output.

Both were measured live on 2026-09-12: with AgentNick's controller prompt and
its tools, the model spent all six rounds calling tools and never answered; and
output_safety flags a realistic critique (SUPPLIER_NOT_IN_CONTRACT_MASTER reads
as an env var), which would replace a valid critique with a canned reply.

What the call keeps is the egress audit every other inference carries.
"""
import json
from unittest.mock import MagicMock, patch

from src.services import egress
from src.services.opportunity_critic.llm import ask_for_critique

_CRITIQUE = json.dumps({
    "verdict": "INVALID",
    "tests": [{"test": "evidence_quality", "result": "UNASSESSED",
               "reason": "contract_resolution is SUPPLIER_NOT_IN_CONTRACT_MASTER"}],
    "gaps": [{"gap_id": "G1", "likely_source": "contract repository"}],
})


def _response(content):
    response = MagicMock()
    response.json.return_value = {"message": {"content": content}}
    response.raise_for_status.return_value = None
    return response


def test_no_tools_are_offered_so_there_is_nothing_to_call_instead():
    with patch.object(egress, "post", return_value=_response(_CRITIQUE)) as post:
        ask_for_critique("system", "task")
    payload = post.call_args.kwargs["json"]
    assert "tools" not in payload, "offering tools is what stalled the live run"
    assert payload["messages"][0]["content"] == "system"
    assert payload["messages"][1]["content"] == "task"


def test_the_answer_is_returned_verbatim_not_through_output_safety():
    # output_safety.is_safe() is False for this text. If the critic ever routed
    # its answer through that gate, a correct critique would come back as a
    # canned reply and parse as nothing.
    with patch.object(egress, "post", return_value=_response(_CRITIQUE)):
        answer, error = ask_for_critique("system", "task")
    assert error is None
    assert answer == _CRITIQUE
    assert "SUPPLIER_NOT_IN_CONTRACT_MASTER" in answer


def test_the_model_call_stays_audited():
    with patch.object(egress, "post", return_value=_response(_CRITIQUE)) as post:
        ask_for_critique("system", "task")
    assert post.call_args.kwargs["purpose"] is egress.Purpose.MODEL_INFERENCE


def test_a_dead_model_comes_back_as_an_error_never_a_raise():
    with patch.object(egress, "post", side_effect=RuntimeError("connection refused")):
        answer, error = ask_for_critique("system", "task")
    assert answer is None
    assert "connection refused" in error


def test_an_empty_answer_is_an_error_not_an_empty_critique():
    with patch.object(egress, "post", return_value=_response("")):
        answer, error = ask_for_critique("system", "task")
    assert answer is None
    assert "empty" in error
