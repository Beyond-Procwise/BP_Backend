"""LangChain bindings for AgentNick — the bridge that lets Hyper-Extract run on our model.

Hyper-Extract asks for structured output via ``with_structured_output(schema,
method="function_calling")``. We do not use function calling: AgentNick produces structured
output through Ollama's native ``format=`` JSON-schema grammar, which masks invalid tokens so
the model *cannot* emit malformed JSON. That is strictly stronger, and it is the mechanism the
rest of this codebase already relies on.

Everything goes through ``services.ollama_client``, which owns a semaphore capped at 2
concurrent requests because GPU contention causes timeouts on the live extraction path.
Hyper-Extract otherwise fans out 10 concurrent calls and would starve extraction of GPU.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.services.ollama_client import DEFAULT_MODEL, ollama_generate

# Hyper-Extract's default is 10. ollama_client's semaphore is 2.
MAX_WORKERS = 2


def _messages_to_prompt(messages: Sequence[BaseMessage]) -> str:
    parts = [
        m.content if isinstance(m.content, str) else json.dumps(m.content) for m in messages
    ]
    return "\n\n".join(p for p in parts if p)


class AgentNickChat(BaseChatModel):
    """A LangChain chat model backed by AgentNick via BP_Backend's Ollama client."""

    model_name: str = DEFAULT_MODEL
    temperature: float = 0.0
    # AgentNick:unified is a hybrid reasoner: without think=False the answer lands in a
    # separate `thinking` field and `response` comes back empty.
    think: Optional[bool] = False
    num_predict: int = 8192
    response_schema: Optional[Type[BaseModel]] = None

    @property
    def _llm_type(self) -> str:
        return "agentnick-ollama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = ollama_generate(
            _messages_to_prompt(messages),
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            think=self.think,
            stop=stop,
            format=(
                self.response_schema.model_json_schema() if self.response_schema else None
            ),
        )
        if text is None:
            # A dead model is a failure, not an empty result. Returning "" here would let
            # the caller report "0 obligations found" for a contract it never read.
            raise RuntimeError(f"AgentNick returned no response (model={self.model_name})")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(
        self, schema: Type[BaseModel], *, method: str = "function_calling", **kwargs: Any
    ) -> Runnable:
        """``method`` is accepted and ignored — we always use grammar-constrained decoding."""
        bound = self.model_copy(update={"response_schema": schema})
        return bound | RunnableLambda(lambda msg: schema.model_validate_json(msg.content))


class AgentNickEmbeddings(Embeddings):
    """LangChain Embeddings over the GPU-resident SentenceTransformer BaseAgent already holds.

    A stub or zero-vector embedder silently collapses Hyper-Extract's dedup layer, yielding
    zero results with no error — so this must be the real model.
    """

    def __init__(self, model: Any):
        if model is None:
            raise ValueError("AgentNickEmbeddings requires a real embedding model")
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
