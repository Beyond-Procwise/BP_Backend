"""The GPU option has to be the one Ollama actually reads.

``ollama_options`` asked for full GPU offload with ``num_gpu_layers: -1``. That
is llama.cpp's spelling; Ollama's API option is ``num_gpu``, so the request was
accepted and ignored, and the model ran under whatever its Modelfile pinned —
``num_gpu 25`` for AgentNick, i.e. roughly half the layers on the CPU.

Measured on the live server against AgentNick:unified, same prompt:

    num_gpu_layers: -1   ->   19.78 tok/s   (48% CPU / 52% GPU)
    num_gpu: -1          ->   19.71 tok/s   (-1 is not "all layers")
    num_gpu: 999         ->  189.19 tok/s   (100% GPU)

Both halves matter: the key must be ``num_gpu``, and the value must be an
explicit layer count. ``-1`` reads as "decide for me" and lands back on the
Modelfile's cap, so a test that only checked the key name would still pass while
the model crawled.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentNick


def _options_for(device: str) -> dict:
    agent = AgentNick.__new__(AgentNick)
    agent.device = device
    return AgentNick.ollama_options(agent)


def test_cuda_offloads_every_layer_with_the_key_ollama_reads():
    options = _options_for("cuda")

    assert "num_gpu_layers" not in options, (
        "num_gpu_layers is llama.cpp's name — Ollama ignores it silently"
    )
    assert "num_gpu" in options
    layers = options["num_gpu"]
    assert isinstance(layers, int)
    # -1/0 hand the decision back to the Modelfile, which is the cap we are lifting.
    assert layers > 0
    # Comfortably above any layer count AgentNick has; llama.cpp clamps the excess.
    assert layers >= 100


def test_cpu_does_not_ask_for_gpu_layers():
    options = _options_for("cpu")

    assert "num_gpu" not in options
    assert "num_gpu_layers" not in options


def test_keep_alive_does_not_travel_inside_options():
    """Reloading a 20GB model per request is its own latency bug — but `keep_alive` inside
    `options` was never preventing it.

    This used to assert `options["keep_alive"] == "10m"`, which is the SAME class of
    mistake the rest of this file is about: a key Ollama accepts and ignores. The live
    server says so in as many words, on every request —

        level=WARN source=types.go:977 msg="invalid option provided" option=keep_alive

    — so the model expired on the daemon's 5m default and the first caller after any lull
    paid a cold load of about 150 seconds. `keep_alive` is a TOP-LEVEL request field.
    model_selector already popped it out to the top level and said why; this path did not.
    """
    for device in ("cuda", "cpu"):
        assert "keep_alive" not in _options_for(device), (
            f"{device}: Ollama ignores keep_alive inside options — it belongs beside the "
            "request, not in it")


def test_the_shared_pin_is_what_callers_send_alongside_the_request():
    # The value itself still has to exist somewhere, and in one place, or the two paths
    # disagree about how long the model stays resident.
    from src.services.ollama_client import KEEP_ALIVE
    assert KEEP_ALIVE is not None
