"""A full card must cost speed, not every answer in the product.

Every request this client sends pins ``num_gpu`` to a number above the model's
layer count, because the measurement on this box is stark: forced onto the GPU
AgentNick runs at 189 tok/s, and left to the server's own scheduler at 19.7.
The pin is worth having.

But it is a demand, not a preference. On 2026-09-08, with the API holding 3.8GB
of a 23GB card and the 30B model needing 19.1GB whole, every single call came
back ``500 {"error":"memory layout cannot be allocated with num_gpu = 999"}`` —
extraction, /ask, agents, all of it, with three retries each and no fallback,
because there was no path in the client that could say "then let the server
decide". Ten times slower is a bad answer; no answer at all is not an answer.

So: pin by default, and when the card refuses the layout, drop the pin, say so,
and keep sending answers. The refusal sticks for a short window rather than
being rediscovered on every call, because Ollama keys a loaded model on its
load-affecting options — flip-flopping between pinned and unpinned would load
the model twice and block on it.
"""

import pytest

from src.services import egress, ollama_client


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise egress.HTTPError(f"{self.status_code} Server Error")


REJECTION = '{"error":"memory layout cannot be allocated with num_gpu = 999"}'


@pytest.fixture(autouse=True)
def _clear_state():
    ollama_client.clear_layout_rejection()
    yield
    ollama_client.clear_layout_rejection()


class TestWhatTheClientAsksFor:
    def test_the_whole_model_goes_on_the_card_by_default(self):
        assert ollama_client.gpu_options()["num_gpu"] == ollama_client.ALL_GPU_LAYERS

    def test_after_a_refusal_the_server_is_left_to_decide(self):
        ollama_client.note_layout_rejection("num_gpu = 999")
        assert "num_gpu" not in ollama_client.gpu_options()

    def test_the_pin_comes_back_once_the_window_has_passed(self):
        # Memory frees up — another process exits, a model is evicted — and the
        # fast path is worth retrying rather than being given up on for the
        # life of the process.
        clock = [1000.0]
        ollama_client.note_layout_rejection("full", now=clock[0])
        clock[0] += ollama_client.LAYOUT_RETRY_SECONDS + 1
        assert ollama_client.gpu_options(now=clock[0])["num_gpu"] == ollama_client.ALL_GPU_LAYERS


class TestRecognisingTheRefusal:
    def test_the_servers_own_words_are_recognised(self):
        assert ollama_client.is_layout_rejection(REJECTION) is True

    def test_another_failure_is_not_mistaken_for_it(self):
        assert ollama_client.is_layout_rejection('{"error":"model not found"}') is False
        assert ollama_client.is_layout_rejection("") is False


class TestTheCallThatWasRefused:
    def _transport(self, responses, seen):
        import copy

        def _post(url, **kwargs):
            # A copy: what was sent, not what the payload became afterwards.
            seen.append(copy.deepcopy(kwargs.get("json")))
            return responses[len(seen) - 1]
        return _post

    def test_the_answer_still_arrives_without_the_pin(self, monkeypatch):
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text=REJECTION), _Response(200, {"response": "  hello  "})], seen))

        assert ollama_client.ollama_generate("hi", retries=1) == "hello"
        assert seen[0]["options"]["num_gpu"] == ollama_client.ALL_GPU_LAYERS
        assert "num_gpu" not in seen[1]["options"]

    def test_the_second_try_is_not_taken_out_of_the_callers_retries(self, monkeypatch):
        # The insight writer asks for one attempt because it sits on a request
        # path. One attempt must still mean one real attempt, not "your attempt
        # was spent discovering the card is full".
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text=REJECTION), _Response(200, {"response": "hello"})], seen))

        assert ollama_client.ollama_generate("hi", retries=1) == "hello"
        assert len(seen) == 2

    def test_the_next_call_does_not_pay_for_the_same_discovery(self, monkeypatch):
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text=REJECTION), _Response(200, {"response": "one"}),
             _Response(200, {"response": "two"})], seen))

        ollama_client.ollama_generate("hi", retries=1)
        assert ollama_client.ollama_generate("again", retries=1) == "two"
        assert "num_gpu" not in seen[2]["options"]

    def test_a_failure_that_is_not_about_the_card_is_left_alone(self, monkeypatch):
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text='{"error":"model not found"}'),
             _Response(500, text='{"error":"model not found"}')], seen))

        assert ollama_client.ollama_generate("hi", retries=1) is None
        assert len(seen) == 1
        assert "num_gpu" in seen[0]["options"]


class TestEveryCallerAgrees:
    """One value at a time, across every path that talks to this server.

    Ollama keys a loaded model on its load-affecting options, so a caller still
    pinning 999 while the rest have fallen back would load a second copy of a
    20GB model and block on it.
    """

    def test_the_agent_path_reads_the_same_state(self):
        from src.agents.base_agent import AgentNick

        agent = AgentNick.__new__(AgentNick)
        agent.device = "cuda"
        assert agent.ollama_options()["num_gpu"] == ollama_client.ALL_GPU_LAYERS
        ollama_client.note_layout_rejection("full")
        assert "num_gpu" not in agent.ollama_options()

    def test_extraction_pins_nothing_of_its_own(self):
        # Three NuExtract calls and one AgentNick call carried a hard-coded
        # num_gpu of their own, so they could neither be told about a refusal
        # nor recover from one.
        source = open("src/agents/extraction_engine.py").read()
        assert '"num_gpu": 99' not in source
        assert "'num_gpu': 99" not in source
