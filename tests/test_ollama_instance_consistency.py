"""One Ollama model instance, not three.

Ollama keys a loaded model on its load-affecting options, so each distinct num_gpu
value spawns its OWN 20GB runner. This codebase asked for three different values —
preload_model() sent none (falling back to the Modelfile's num_gpu 25),
ollama_generate() defaulted to 99, and BaseAgent.ollama_options() sent 999 — so a
request naming a not-yet-loaded configuration blocked for minutes while a new runner
loaded, at 0% GPU with 81GB free. Live, that timed out every summary generation and
hung /workflows/ask.

These lock the three call sites to one value.
"""
import src.services.ollama_client as oc


def test_all_gpu_layers_constant_is_exported():
    assert isinstance(oc.ALL_GPU_LAYERS, int)
    assert oc.ALL_GPU_LAYERS >= 99  # any value at/above the layer count means "all"


# The transport moved from `requests` to `services.egress` (the audited egress
# path); these patched oc.requests, which stopped existing, and had been erroring
# rather than asserting ever since.


class _Resp:
    status_code = 200
    text = ""
    def raise_for_status(self): pass
    def json(self): return {"response": "ok"}


def _capture(seen):
    def fake_post(url, json=None, timeout=None, **kw):
        seen["url"] = url
        seen["payload"] = json
        return _Resp()
    return fake_post


def test_generate_defaults_to_the_shared_layer_count(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model="m", retries=1)
    assert seen["payload"]["options"]["num_gpu"] == oc.ALL_GPU_LAYERS


def test_preload_pins_the_same_configuration(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    # Nothing resident, so there is a cold load to pay for. A preload against an
    # ALREADY-loaded model is skipped entirely (test_ollama_preload_does_not_reload).
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.preload_model("m")
    # Preloading WITHOUT the layer count pins a Modelfile-default instance that no
    # later request matches — the request then blocks loading a second runner.
    assert seen["payload"].get("options", {}).get("num_gpu") == oc.ALL_GPU_LAYERS


def test_a_card_that_refused_moves_the_preload_too(monkeypatch):
    # The fallback is only "one instance" if everything falls back together:
    # a preload still pinning the whole model would load a second copy of it.
    seen = {}
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.note_layout_rejection("full")
    try:
        oc.preload_model("m")
        assert "num_gpu" not in seen["payload"].get("options", {})
    finally:
        oc.clear_layout_rejection()


def test_agent_options_use_the_same_constant():
    # AgentNick.ollama_options() is the other place a num_gpu reaches Ollama.
    from src.agents.base_agent import AgentNick
    assert AgentNick._ALL_GPU_LAYERS == oc.ALL_GPU_LAYERS
