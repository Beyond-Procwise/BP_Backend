"""What a critic batch must not do to the shared model server.

AgentNick() preloads its model at construction, asking Ollama for
num_gpu=ALL_GPU_LAYERS (999 unless OLLAMA_NUM_GPU_LAYERS says otherwise).
AgentNick:unified's Modelfile pins num_gpu 25 because the whole model does not
fit on the card, so every preload asks for a runner that cannot exist: Ollama
tries it, fails ("Load failed"), and the next request reloads at 25. On
2026-09-11 that alternation timed out two live critic runs and took the local
server's model away with them.

A batch critiques; it does not warm or pin models. Its own model calls go
through tool_runtime, which sends no num_gpu and so reuses whatever runner is
already resident.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def skip_model_preload() -> None:
    """Make AgentNick's startup preload a no-op for this process.

    AgentNick._preload_ollama_model imports preload_model from
    src.services.ollama_client at call time, so replacing that attribute before
    AgentNick() is constructed is enough.
    """
    from src.services import ollama_client

    def _skipped(*_args, **_kwargs) -> bool:
        logger.info("critic batch: AgentNick model preload skipped; the resident "
                    "runner is reused")
        return False

    ollama_client.preload_model = _skipped


def skip_embedding_model() -> None:
    """Stop AgentNick loading a sentence-transformer into this process.

    AgentNick() loads one at construction. With the GPU hidden -- which the
    batch must do, because the card is full -- it lands in RAM, and on
    2026-09-12 that cost 2.6 GiB on a host with 361 MiB free and no swap left,
    so the run was killed before it critiqued anything.

    The critic never embeds: it reasons over an evidence envelope assembled
    from SQL. Anything that does reach for embeddings gets a loud failure
    rather than a quiet wrong answer.
    """
    from agents import base_agent

    class _AbsentEmbeddingModel:
        def __getattr__(self, name):
            raise RuntimeError(
                "the critic batch runs without an embedding model; something "
                f"asked for embedding_model.{name}"
            )

    base_agent.SentenceTransformer = lambda *_a, **_k: _AbsentEmbeddingModel()
