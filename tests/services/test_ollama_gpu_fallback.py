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


class TestTheValueSystemdActuallyHandsUs:
    """`keep_alive` arrives mangled from the unit file, and it 400s every call.

    .env carries `OLLAMA_KEEP_ALIVE="-1"          # sent per-request; ...`.
    python-dotenv strips that trailing comment, so every script and every test
    sees a clean "-1" — and systemd's EnvironmentFile does not, so the service
    was started with

        OLLAMA_KEEP_ALIVE=-1# sent per-request; overrides the server's 5m

    which went onto the wire verbatim and came back `400 Bad Request` from
    Ollama. Every ollama_generate call the API made failed instantly, and the
    log said only "400 Client Error" with no reason. The comment is off its own
    line now; this is the belt to that pair of braces, because the next value
    with a comment after it must not take the model layer down again.
    """

    def test_a_plain_number_is_a_number(self):
        assert ollama_client._coerce_keep_alive("-1") == -1

    def test_a_duration_is_left_as_it_is(self):
        assert ollama_client._coerce_keep_alive("24h") == "24h"

    def test_a_comment_that_came_along_for_the_ride_is_dropped(self):
        assert ollama_client._coerce_keep_alive(
            "-1# sent per-request; overrides the server's 5m") == -1

    def test_a_value_this_server_would_reject_is_not_sent_at_all(self):
        # Sending it costs every call in the product; the default costs a model
        # eviction at worst.
        assert ollama_client._coerce_keep_alive("what even is this") == \
            ollama_client.DEFAULT_KEEP_ALIVE

    def test_no_stray_comment_reaches_the_wire(self, monkeypatch):
        seen = []

        def _post(url, **kwargs):
            seen.append(kwargs.get("json"))
            class _R:
                status_code = 200
                text = ""
                def raise_for_status(self): pass
                def json(self): return {"response": "ok"}
            return _R()

        monkeypatch.setattr(egress, "post", _post)
        monkeypatch.setattr(ollama_client, "KEEP_ALIVE",
                            ollama_client._coerce_keep_alive("-1# a comment"))
        ollama_client.ollama_generate("hi", retries=1)
        assert seen[0]["keep_alive"] == -1


class TestTheStartupPreload:
    """The discovery belongs at startup, not in the first reader's answer.

    The API preloads the model when it boots. That preload was refused for the
    same reason as everything else and only warned about it, so the shared state
    was still "pinned" when the first question arrived — and that reader waited
    47 seconds while the client found out what the preload already knew.
    """

    def _transport(self, responses, seen):
        import copy

        def _post(url, **kwargs):
            seen.append(copy.deepcopy(kwargs.get("json")))
            return responses[len(seen) - 1]
        return _post

    def test_a_refused_preload_tells_everyone_else(self, monkeypatch):
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text=REJECTION), _Response(200, {})], seen))

        assert ollama_client.preload_model("m") is True
        assert "num_gpu" not in ollama_client.gpu_options()
        assert "num_gpu" not in seen[1]["options"]

    def test_the_model_is_still_warmed_in_the_configuration_that_works(self, monkeypatch):
        # Giving up on the preload would leave the first real request paying a
        # two-minute cold load on top of everything else.
        seen = []
        monkeypatch.setattr(egress, "post", self._transport(
            [_Response(500, text=REJECTION), _Response(200, {})], seen))
        ollama_client.preload_model("m")
        assert len(seen) == 2


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
