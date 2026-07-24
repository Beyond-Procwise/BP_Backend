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


def test_keep_alive_is_preserved_on_both_paths():
    """Unrelated to placement, but reloading a 20GB model per request is its own
    latency bug — the existing keep_alive must survive the fix."""

    assert _options_for("cuda").get("keep_alive") == "10m"
    assert _options_for("cpu").get("keep_alive") == "10m"
