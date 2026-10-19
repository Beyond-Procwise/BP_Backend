"""The embedding model, behind a tiny interface so the worker is testable and the model is swappable.

``encode(texts)`` returns one unit-length vector per text. Embedding runs in the worker, never in a request handler.
"""
from __future__ import annotations

from typing import Any, List, Protocol, Sequence


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> List[List[float]]: ...


class SentenceTransformerEmbedder:
    """Loads the sentence-transformers model on first use (it is heavy), then encodes normalised vectors."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name, self._device, self._model = model_name, device, None

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


class AgentNickEmbedder:
    """Reuses the model the API process already loaded (``agent_nick.embedding_model``) instead of loading a second copy."""

    def __init__(self, agent_nick: Any) -> None:
        self._nick = agent_nick

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._nick.embedding_model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
