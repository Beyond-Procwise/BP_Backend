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


# ---------------------------------------------------------------------------
# THE WHOLE LOAD-AFFECTING SET, NOT JUST num_gpu (2026-10-05).
#
# The tests above locked num_gpu to one value and stopped there. num_ctx, num_batch and
# num_thread are keyed by Ollama in exactly the same way, and they were NOT locked, so the
# same bug came back wearing a different option:
#
#   preload_model()          {num_gpu: 999}                        -> auto-sized, ctx 12288
#   BaseAgent.call_ollama    + num_ctx 8192, num_batch, num_thread -> ctx 8192
#   model_selector           ollama_options() + sampling            -> back to ctx 12288
#
# Measured on the live box: FOUR loads of the 18GB model in five minutes, one /api/chat at
# 7m44s, and a demand-intake turn that took 159s of which 153s was "llama runner started".
# Generation was about six seconds.
# ---------------------------------------------------------------------------

# Every key Ollama keys a runner on. A new one belongs here AND in load_options().
LOAD_AFFECTING = ("num_gpu", "num_ctx", "num_batch", "num_thread")


def test_load_options_carries_every_load_affecting_key():
    oc.clear_layout_rejection()
    opts = oc.load_options()
    for key in LOAD_AFFECTING:
        assert key in opts, f"{key} is keyed by Ollama and must travel with every request"


def test_the_preload_pins_exactly_what_callers_ask_for(monkeypatch):
    # THE BUG, pinned. The preload used to send num_gpu alone; Ollama then auto-sized the
    # context and every caller naming a num_ctx reloaded the whole model.
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.preload_model("m")
    pinned = seen["payload"].get("options", {})
    assert pinned == oc.load_options(), (
        "the preload must pin the SAME runner callers will ask for, or the first real "
        "request loads a second copy of the model")


def test_agent_options_are_the_shared_set(monkeypatch):
    # AgentNick.ollama_options() is what call_ollama and model_selector both build on.
    from src.agents.base_agent import AgentNick
    oc.clear_layout_rejection()
    nick = AgentNick.__new__(AgentNick)          # no __init__: this reads one method
    nick.device = "cuda"
    assert nick.ollama_options() == oc.load_options()


def test_keep_alive_is_not_smuggled_inside_options(monkeypatch):
    # Inside `options` Ollama logs `invalid option provided option=keep_alive` and ignores
    # it, so the model expired on the daemon's 5m default and the next caller after any
    # lull paid a cold load. It is a top-level request field.
    from src.agents.base_agent import AgentNick
    oc.clear_layout_rejection()
    nick = AgentNick.__new__(AgentNick)
    nick.device = "cuda"
    assert "keep_alive" not in nick.ollama_options()


def test_the_preload_sends_keep_alive_where_ollama_reads_it(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.preload_model("m")
    assert seen["payload"].get("keep_alive") == oc.KEEP_ALIVE
    assert "keep_alive" not in seen["payload"].get("options", {})


def test_a_refused_card_still_leaves_the_rest_of_the_set_matching(monkeypatch):
    # The fallback only yields one instance if the non-GPU keys still agree.
    oc.note_layout_rejection("full")
    try:
        opts = oc.load_options()
        assert "num_gpu" not in opts
        for key in ("num_ctx", "num_batch", "num_thread"):
            assert key in opts
    finally:
        oc.clear_layout_rejection()


def test_the_context_window_is_not_smaller_than_the_auto_sized_runner():
    # Unifying DOWN to 8192 would shrink the context the RAG and chat paths were already
    # getting from the auto-sized runner (12288) and start truncating retrieved context.
    # That is an accuracy regression, so the floor is pinned.
    assert oc.CONTEXT_WINDOW >= 12288


# ---------------------------------------------------------------------------
# NOBODY ELSE NAMES A LOAD-AFFECTING OPTION.
#
# Every instance of this bug has been the same shape: some module sets num_ctx (or the
# ignored llama.cpp spelling num_gpu_layers) in its own little options dict, asks Ollama for
# a runner nothing else is using, and reloads 18GB. Locking the values in load_options()
# does not prevent that — only noticing the new call site does.
#
# Six call sites were in play on 2026-10-05: base_agent, model_selector, preload_model,
# email_drafting_agent, rag_qwen30b and prompt_engine. This is the test that fails when a
# seventh appears.
# ---------------------------------------------------------------------------
import pathlib
import re

# Keys that make Ollama load a separate runner, plus the llama.cpp spellings it silently
# ignores (which are their own bug: accepted, ignored, and the Modelfile wins).
_FORBIDDEN_ELSEWHERE = ("num_ctx", "num_batch", "num_thread", "num_gpu", "num_gpu_layers",
                        "gpu_layers")
# ollama_client owns them. base_agent names them only in prose and in its own guard.
_OWNER = "src/services/ollama_client.py"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def test_no_module_outside_ollama_client_sets_a_load_affecting_option():
    root = _repo_root()
    offenders = []
    # Only assignments/dict-literals, so prose and comments do not trip it.
    pattern = re.compile(
        r'''(?:"(?:%s)"|'(?:%s)')\s*:''' % ("|".join(_FORBIDDEN_ELSEWHERE),
                                            "|".join(_FORBIDDEN_ELSEWHERE))
        + r'''|\[\s*["'](?:%s)["']\s*\]\s*=''' % "|".join(_FORBIDDEN_ELSEWHERE))
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == _OWNER:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(f"{rel}:{n}: {line.strip()[:95]}")
    assert not offenders, (
        "these set an option Ollama keys a loaded runner on, outside the one module that "
        "owns them — each one costs a full reload of the model:\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# THE DEFAULT PATH ASKS FOR THE SHARED RUNNER (user ruling 2026-10-08).
#
# Until then load_options() was opt-in, so every caller that did not pass it sent num_gpu
# alone and Ollama fell back to the Modelfile's num_ctx 8192 -- a different runner from the
# preload's 12288. Now the shared model always asks for load_options(); a DIFFERENT model
# (the extraction specialist, a model under evaluation) keeps exactly its old body.
# ---------------------------------------------------------------------------


def test_default_model_sends_the_shared_runner_set_by_default(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", retries=1)
    assert seen["payload"] == {
        "model": oc.DEFAULT_MODEL,
        "prompt": "hello",
        "stream": False,
        "keep_alive": oc.KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 8192, **oc.load_options()},
    }


def test_naming_the_default_model_explicitly_is_the_same(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model=oc.DEFAULT_MODEL, retries=1)
    for key, value in oc.load_options().items():
        assert seen["payload"]["options"][key] == value


def test_a_pinned_num_gpu_still_wins_on_the_default_path(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", retries=1, num_gpu=7)
    opts = seen["payload"]["options"]
    assert opts["num_gpu"] == 7 and opts["num_ctx"] == oc.load_options()["num_ctx"]


def test_a_refused_card_on_the_default_path_drops_num_gpu_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.note_layout_rejection("full")
    try:
        oc.ollama_generate("hello", retries=1)
        opts = seen["payload"]["options"]
        assert "num_gpu" not in opts and opts["num_ctx"] == oc.CONTEXT_WINDOW
    finally:
        oc.clear_layout_rejection()


def test_the_extraction_specialist_is_untouched(monkeypatch):
    """AgentNick:extract's accuracy was measured on its own Modelfile options."""
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model="BeyondProcwise/AgentNick:extract", retries=1)
    assert seen["payload"]["options"] == {"temperature": 0, "num_predict": 8192, **oc.gpu_options()}


def test_another_server_is_untouched(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", base_url="http://translator.example:11434", retries=1)
    assert "num_ctx" not in seen["payload"]["options"]
    assert seen["url"].startswith("http://translator.example:11434")


def test_use_load_options_false_keeps_the_old_body(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", retries=1, use_load_options=False)
    assert seen["payload"]["options"] == {"temperature": 0, "num_predict": 8192, **oc.gpu_options()}


def test_a_different_model_body_is_unchanged(monkeypatch):
    """Not the shared model: a caller that does not pass use_load_options sends exactly this."""
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model="m", retries=1)
    assert seen["payload"] == {
        "model": "m",
        "prompt": "hello",
        "stream": False,
        "keep_alive": oc.KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 8192, **oc.gpu_options()},
    }
    assert "num_ctx" not in seen["payload"]["options"]


def test_use_load_options_sends_the_shared_runner_set(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model="m", retries=1, use_load_options=True)
    opts = seen["payload"]["options"]
    shared = oc.load_options()
    assert opts["num_ctx"] == shared["num_ctx"]
    for key, value in shared.items():
        assert opts[key] == value
    assert opts["temperature"] == 0 and opts["num_predict"] == 8192


def test_pinned_num_gpu_is_kept_with_use_load_options(monkeypatch):
    seen = {}
    oc.clear_layout_rejection()
    monkeypatch.setattr(oc.egress, "post", _capture(seen))
    oc.ollama_generate("hello", model="m", retries=1, num_gpu=7, use_load_options=True)
    opts = seen["payload"]["options"]
    assert opts["num_gpu"] == 7
    assert opts["num_ctx"] == oc.load_options()["num_ctx"]
