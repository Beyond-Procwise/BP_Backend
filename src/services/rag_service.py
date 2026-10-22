import importlib
import importlib.util
import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set
from types import SimpleNamespace

import hashlib

import numpy as np
try:  # pragma: no cover - optional sparse retriever dependency
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    BM25Okapi = None  # type: ignore
from src.services.vector_store.filter_models import models
from utils.gpu import configure_gpu, load_cross_encoder
from services.semantic_cache import SemanticCacheManager
from . import ask_flags, ask_trace
from services.document_extractor import LayoutAwareParser
from services.semantic_chunker import SemanticChunker, SemanticChunk

try:  # pragma: no cover - FAISS optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    faiss = None  # type: ignore

# Imported, never redeclared. This module and document_embedding_service both
# write to the same vector store collections; a second local set would drift and one
# of the two write paths would stop redacting without anything failing.
from services.document_embedding_service import (  # noqa: E402
    DISALLOWED_METADATA_KEYS,
)

_TRAINING_ROOT = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "training"
    / "rag"
)
_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
_PREFERENCE_WEIGHTS_PATH = _TRAINING_ROOT / "preference_weights.json"

configure_gpu()


logger = logging.getLogger(__name__)


# Question words that used to become a HARD filter (``source_type == 'Contract'`` and so on).
# The main collection holds only a lowercase ``document_type`` (quote / purchase_order / invoice /
# contract) and no ``source_type``, so for every contract, PO, invoice or quote question that filter
# matched nothing and retrieval returned no documents. Worse, the typing in the index is not reliable:
# the Ironbridge master services agreement is indexed as a 'quote'. So these four are a PREFERENCE
# (the focus-term bonus in ``search``) rather than an exclusion. "Policy" stays a hard filter: the
# policy collection does carry ``source_type == 'Policy'`` and relies on it.
SOFT_TYPE_WORDS = frozenset({"PO", "Invoice", "Quote", "Contract"})

FAST_RERANK_MODEL = "BAAI/bge-reranker-base"
FAST_RERANK_LIMIT = 6          # candidates scored by the cross-encoder
FAST_RERANK_MAX_CHARS = 600    # characters of each candidate shown to it


class RAGService:
    """Handles embedding upserts and retrieval for RAG workflows."""

    def __init__(self, agent_nick, cross_encoder_cls=None):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.embedder = agent_nick.embedding_model
        self.primary_collection = getattr(
            self.settings,
            "document_collection_name",
            "procwise_document_embeddings",
        )
        self.uploaded_collection = getattr(
            self.settings,
            "uploaded_documents_collection_name",
            "uploaded_documents",
        )
        self.static_policy_collection = getattr(
            self.settings,
            "static_policy_collection_name",
            "static_policy",
        )
        self.learning_collection = getattr(
            self.settings,
            "learning_collection_name",
            "learning",
        )
        # Local FAISS and BM25 indexes to complement the vector store
        self._faiss_available = faiss is not None
        self._faiss_warned = False
        self._faiss_index = None
        self._bm25_available = BM25Okapi is not None
        self._bm25_warned = False
        self._doc_vectors: List[np.ndarray] = []
        self._documents: List[Dict] = []  # payloads with content
        self._bm25 = None
        self._bm25_corpus: List[List[str]] = []
        if cross_encoder_cls is None:
            from sentence_transformers import CrossEncoder
            cross_encoder_cls = CrossEncoder
        model_name = getattr(
            self.settings,
            "reranker_model",
            "BAAI/bge-reranker-large",
        )
        # The large reranker does not fit the GPU next to the live LLM and runs on the CPU
        # (~21-27 s for 12 candidates on this host). The fast setting trades it for the base
        # model on fewer, shorter candidates; see FAST_RERANK_* below. Read once, at start-up.
        self._fast_rerank = ask_flags.enabled(ask_flags.FAST_RERANK)
        if self._fast_rerank:
            model_name = FAST_RERANK_MODEL
        self.reranker_model_name = model_name      # which scale the rerank scores are on (ask_grade thresholds are per model)
        self._reranker = load_cross_encoder(
            model_name, cross_encoder_cls, getattr(self.agent_nick, "device", None)
        )
        self._semantic_cache = SemanticCacheManager(
            self.settings, namespace="rag_service"
        )
        self._preference_weights_path = _PREFERENCE_WEIGHTS_PATH
        self._preference_weights = self._load_preference_weights()
        self._rrf_k = 60
        self._layout_parser = LayoutAwareParser()
        self._chunker = SemanticChunker(settings=self.settings)
        self._payload_index_cache: Dict[str, Set[str]] = {}
        self._prefetch_limit = max(
            int(getattr(self.settings, "rag_prefetch_limit", 48) or 1), 1
        )
        ratio = float(getattr(self.settings, "rag_prefetch_ratio", 2.2) or 2.0)
        self._prefetch_ratio = max(1.2, ratio)
        self._prefetch_cap = max(
            int(getattr(self.settings, "rag_prefetch_cap", 32) or 1), 8
        )
        self._reranker_batch_size = max(
            int(getattr(self.settings, "rag_reranker_batch_size", 32) or 1), 1
        )
        self._reranker_max_chars = max(
            int(getattr(self.settings, "rag_reranker_max_chars", 1400) or 256), 256
        )
        self._reranker_limit = max(
            int(getattr(self.settings, "rag_reranker_limit", 12) or 1), 1
        )
        self._reranker_score_floor = float(
            getattr(self.settings, "rag_reranker_score_floor", 0.35) or 0.0
        )
        if self._fast_rerank:
            self._reranker_limit = min(self._reranker_limit, FAST_RERANK_LIMIT)
            self._reranker_max_chars = min(self._reranker_max_chars, FAST_RERANK_MAX_CHARS)
        self._rerank_cache_size = max(
            int(getattr(self.settings, "rag_reranker_cache_size", 384) or 0), 0
        )
        self._rerank_cache: "OrderedDict[str, float]" = OrderedDict()
        self._rerank_cache_lock = threading.Lock()
        self._search_workers = max(
            int(getattr(self.settings, "rag_search_workers", 4) or 1), 1
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def reload_preference_weights(self) -> None:
        """Reload retrieval preference weights from disk."""

        self._preference_weights = self._load_preference_weights()

    def _load_preference_weights(self) -> Dict[str, Dict[str, float]]:
        defaults: Dict[str, Dict[str, float]] = {
            "collection": {},
            "document_type": {},
        }
        defaults["collection"][self.primary_collection] = 0.12
        defaults["collection"][self.uploaded_collection] = 0.08
        if self.static_policy_collection:
            defaults["collection"].setdefault(self.static_policy_collection, 0.14)
        defaults["document_type"].setdefault("policy", 0.1)

        path_obj = Path(self._preference_weights_path)
        if not path_obj.exists():
            return defaults

        try:
            loaded = json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "Falling back to default preference weights due to parse failure",
                exc_info=True,
            )
            return defaults

        if not isinstance(loaded, dict):
            return defaults

        result = {
            "collection": dict(defaults["collection"]),
            "document_type": dict(defaults["document_type"]),
        }

        for section_key in ("collection", "document_type"):
            section = loaded.get(section_key)
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    continue
                result[section_key][key] = float(value)

        return result

    def _chunk_text(
        self, text: str, max_chars: int = 1000, overlap: int = 200
    ) -> List[SemanticChunk]:
        """Chunk text using the structure-aware chunker for previews.

        The ``max_chars`` and ``overlap`` parameters are retained for backward
        compatibility with older call sites. They now map to the runtime
        configuration derived from :class:`SemanticChunker`.
        """

        del max_chars, overlap  # legacy parameters retained for compatibility
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        try:
            structured = self._layout_parser.from_text(cleaned, scanned=False)
        except Exception:
            logger.debug("Layout parsing failed; using fallback chunking", exc_info=True)
            structured = None

        chunks: List[SemanticChunk] = []
        title_hint = None
        if structured and structured.elements:
            first = structured.elements[0]
            candidate = getattr(first, "text", None)
            if candidate:
                title_hint = candidate.strip()
        chunk_seed = {
            "document_type": "General",
            "source_type": "Upload",
            "title": title_hint or "Preview",
        }

        if structured:
            try:
                chunks = self._chunker.build_from_structured(
                    structured,
                    document_type="General",
                    base_metadata=chunk_seed,
                    title_hint=chunk_seed["title"],
                    default_section="document_overview",
                )
            except Exception:
                logger.debug(
                    "Semantic chunking failed; falling back to heuristic chunker",
                    exc_info=True,
                )
                chunks = []

        if not chunks or self._requires_fallback(chunks):
            return self._fallback_chunk_text(cleaned, seed_metadata=chunk_seed)

        return chunks

    def _requires_fallback(self, chunks: List[SemanticChunk]) -> bool:
        """Determine whether the semantic chunker output needs refinement."""

        if not chunks:
            return True

        if len(chunks) > 1:
            return False

        char_limit = self._chunk_char_limit()
        return len(chunks[0].content or "") > int(char_limit * 1.2)

    def _chunk_char_limit(self) -> int:
        """Return the configured character limit for fallback chunking."""

        approx_chars = getattr(self.settings, "rag_chunk_chars", None)
        try:
            approx_value = int(approx_chars) if approx_chars else 0
        except (TypeError, ValueError):
            approx_value = 0
        if approx_value <= 0:
            approx_value = 1800
        return max(900, min(approx_value, 2400))

    def _chunk_overlap_chars(self) -> int:
        """Return the configured overlap (in characters) for fallback chunks."""

        overlap_value = getattr(self.settings, "rag_chunk_overlap", None)
        try:
            overlap = int(overlap_value) if overlap_value else 0
        except (TypeError, ValueError):
            overlap = 0
        if overlap <= 0:
            overlap = int(self._chunk_char_limit() * 0.1)
        return max(120, min(overlap, int(self._chunk_char_limit() * 0.4)))

    def _fallback_chunk_text(
        self, text: str, *, seed_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SemanticChunk]:
        """Fallback heuristic chunker when structural parsing fails.

        This splitter groups paragraphs and long sentences into overlapping
        windows so that downstream retrieval receives semantically coherent
        spans rather than single large blocks of text.
        """

        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []

        char_limit = self._chunk_char_limit()
        overlap_chars = self._chunk_overlap_chars()

        paragraphs = [
            part.strip()
            for part in re.split(r"\n{2,}", text)
            if part and part.strip()
        ]
        if not paragraphs:
            paragraphs = [cleaned]

        expanded_parts: List[str] = []
        for paragraph in paragraphs:
            expanded_parts.extend(self._split_paragraph(paragraph, char_limit))

        chunks: List[SemanticChunk] = []
        buffer: List[str] = []
        buffer_len = 0
        idx = 0
        while idx < len(expanded_parts):
            part = expanded_parts[idx]
            part_len = len(part)
            if buffer and buffer_len + part_len + 2 > char_limit:
                chunk_text = self._normalise_chunk("\n\n".join(buffer))
                if chunk_text:
                    chunks.append(
                        SemanticChunk(
                            content=chunk_text,
                            metadata=self._fallback_metadata(
                                seed_metadata, len(chunks)
                            ),
                        )
                    )
                buffer, buffer_len = self._build_overlap_buffer(
                    buffer, overlap_chars
                )
                continue

            buffer.append(part)
            buffer_len += part_len + 2
            idx += 1

        if buffer:
            chunk_text = self._normalise_chunk("\n\n".join(buffer))
            if chunk_text:
                chunks.append(
                    SemanticChunk(
                        content=chunk_text,
                        metadata=self._fallback_metadata(seed_metadata, len(chunks)),
                    )
                )

        return chunks

    def _split_paragraph(self, paragraph: str, limit: int) -> List[str]:
        """Split long paragraphs into sentence-based windows."""

        paragraph = paragraph.strip()
        if len(paragraph) <= limit:
            return [paragraph]

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return [paragraph[i : i + limit] for i in range(0, len(paragraph), limit)]

        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + sentence_len + 1 > limit:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sentence)
            current_len += sentence_len + 1
        if current:
            parts.append(" ".join(current))
        return parts

    def _build_overlap_buffer(
        self, buffer: List[str], overlap_chars: int
    ) -> Tuple[List[str], int]:
        if not buffer or overlap_chars <= 0:
            return [], 0

        tail = buffer[-1]
        overlap_text = self._tail_sentences(tail, overlap_chars)
        if not overlap_text:
            return [], 0
        return [overlap_text], len(overlap_text)

    def _tail_sentences(self, text: str, limit: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return text[-limit:].strip()

        collected: List[str] = []
        total = 0
        for sentence in reversed(sentences):
            collected.insert(0, sentence)
            total += len(sentence) + 1
            if total >= limit:
                break
        return " ".join(collected).strip()

    def _normalise_chunk(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _fallback_metadata(
        self, seed_metadata: Optional[Dict[str, Any]], index: int
    ) -> Dict[str, Any]:
        metadata = dict(seed_metadata or {})
        metadata.setdefault("section", "document_overview")
        metadata.setdefault("section_path", "document_overview")
        metadata.setdefault("content_type", "paragraph")
        metadata.setdefault("chunk_strategy", "fallback")
        metadata.setdefault("fallback_index", index)
        return metadata

    def _embedding_dimension(self) -> int:
        """Determine the embedding dimensionality of the active encoder."""

        getter = getattr(self.embedder, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                dimension = int(getter())
                if dimension > 0:
                    return dimension
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug("Failed to query embedding dimension from model", exc_info=True)

        probe = self.embedder.encode("dimension probe", normalize_embeddings=True)
        if isinstance(probe, np.ndarray):
            if probe.ndim == 1:
                return int(probe.shape[0])
            if probe.ndim == 2:
                return int(probe.shape[1])
        if isinstance(probe, list):
            if probe and isinstance(probe[0], (float, int)):
                return len(probe)
            if probe and isinstance(probe[0], (list, tuple, np.ndarray)):
                first = probe[0]
                return len(first)
        raise RuntimeError("Unable to determine embedding dimension")

    def ensure_collection(self, collection_name: Optional[str] = None) -> None:
        """Check ``collection_name`` is one of the vector store's collections. Nothing is created: the tables come from migrations."""

        target = collection_name or self.primary_collection
        if self._logical_collection(target) is None:
            raise ValueError(f"{target!r} is not a vector store collection")

    def _logical_collection(self, name: Optional[str]) -> Optional[str]:
        """The vector store's collection for one of the configured collection names (None if it is not one of ours)."""
        return {self.primary_collection: "documents", self.uploaded_collection: "uploaded",
                self.static_policy_collection: "policy", self.learning_collection: "learning"}.get(name)

    def _query_collection(
        self, collection_name: str, vector: Sequence[float], *, query_filter: Optional[models.Filter], limit: int
    ) -> List[SimpleNamespace]:
        """One collection through the VectorStore, as the one tenant there is today. A failure is no hits."""
        logical = self._logical_collection(collection_name)
        if logical is None:
            return []
        try:
            from src.services.tenant_context import company_label
            from src.services.vector_store import get_vector_store
            from src.services.vector_store.filters import to_store_filter
            store = getattr(self, "_vector_store", None) or get_vector_store()
            hits = store.search(company_label(), list(vector), to_store_filter(query_filter), int(limit), collection=logical)
        except Exception:
            logger.warning("Vector search failed for collection %s", collection_name, exc_info=True)
            return []
        wrapped: List[SimpleNamespace] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload.setdefault("collection_name", collection_name)
            wrapped.append(SimpleNamespace(id=str(hit.id), payload=payload, score=float(hit.score or 0.0)))
        return wrapped

    def _search_collections_parallel(
        self,
        requests: Sequence[Tuple[str, Optional[models.Filter]]],
        vector: Sequence[float],
        *,
        limit: int,
    ) -> Dict[str, List[SimpleNamespace]]:
        if not requests:
            return {}

        tasks: List[Tuple[str, Optional[models.Filter]]] = [
            (name, query_filter)
            for name, query_filter in requests
            if isinstance(name, str) and name
        ]
        if not tasks:
            return {}

        max_workers = min(len(tasks), max(self._search_workers, 1))
        results: Dict[str, List[SimpleNamespace]] = {}

        if max_workers <= 1:
            for name, query_filter in tasks:
                results[name] = self._query_collection(
                    name,
                    vector,
                    query_filter=query_filter,
                    limit=limit,
                )
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._query_collection,
                    name,
                    vector,
                    query_filter=query_filter,
                    limit=limit,
                ): name
                for name, query_filter in tasks
            }

            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                except Exception:
                    logger.warning(
                        "Parallel vector search failed for collection %s",
                        name,
                        exc_info=True,
                    )
                    results[name] = []

        return results

    def _prepare_reranker_text(self, payload: Dict[str, Any]) -> str:
        """Extract a concise text snippet for cross-encoder reranking."""

        if not isinstance(payload, dict):
            return ""

        parts: List[str] = []

        highlights = payload.get("highlights")
        if isinstance(highlights, str) and highlights.strip():
            parts.append(highlights.strip())
        elif isinstance(highlights, (list, tuple, set)):
            highlight_bits = [
                str(item).strip()
                for item in highlights
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
            if highlight_bits:
                parts.append(" ".join(highlight_bits))

        text_keys = (
            "text_summary",
            "summary",
            "chunk_text",
            "content",
            "text",
            "description",
        )
        for key in text_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
            if isinstance(value, (list, tuple)):
                merged = " ".join(
                    str(item).strip()
                    for item in value
                    if isinstance(item, (str, int, float)) and str(item).strip()
                ).strip()
                if merged:
                    parts.append(merged)
                    break

        if not parts:
            fallback_bits: List[str] = []
            for key in ("title", "document_type", "source_type", "record_id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    fallback_bits.append(value.strip())
            if fallback_bits:
                parts.append(" - ".join(fallback_bits))

        combined = " ".join(parts).strip()
        if not combined:
            return ""

        max_chars = self._reranker_max_chars
        if len(combined) <= max_chars:
            return combined

        truncated = combined[:max_chars].rsplit(" ", 1)[0].strip()
        if not truncated:
            truncated = combined[:max_chars].strip()
        return truncated + "…"

    def _build_rerank_cache_key(
        self, query: Optional[str], document: Optional[str]
    ) -> Optional[str]:
        if self._rerank_cache_size <= 0:
            return None
        if not query or not document:
            return None
        try:
            query_text = str(query).strip()
            document_text = str(document).strip()
        except Exception:
            return None
        if not query_text or not document_text:
            return None
        trimmed_document = document_text[: self._reranker_max_chars]
        payload = f"{query_text}\u241f{trimmed_document}"
        try:
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _batched_reranker_predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Score ``pairs`` using the cross-encoder in small batches.

        Results are memoised in a bounded LRU cache so repeated or highly
        similar queries avoid recomputing expensive cross-encoder scores.
        """

        if not pairs:
            return []

        cache_keys: List[Optional[str]] = [
            self._build_rerank_cache_key(query, document) for query, document in pairs
        ]
        scores: List[Optional[float]] = [None] * len(pairs)

        if self._rerank_cache_size > 0:
            with self._rerank_cache_lock:
                for idx, key in enumerate(cache_keys):
                    if key is None:
                        continue
                    cached = self._rerank_cache.get(key)
                    if cached is None:
                        continue
                    # Refresh the LRU position
                    self._rerank_cache.move_to_end(key)
                    scores[idx] = float(cached)

        missing_pairs: List[Tuple[str, str]] = []
        missing_indices: List[int] = []
        for idx, (pair, cached_score) in enumerate(zip(pairs, scores)):
            if cached_score is None:
                missing_pairs.append(pair)
                missing_indices.append(idx)

        if missing_pairs:
            batch_size = max(self._reranker_batch_size, 1)
            computed_scores: List[float] = []
            for start in range(0, len(missing_pairs), batch_size):
                chunk = list(missing_pairs[start : start + batch_size])
                if not chunk:
                    continue
                chunk_scores = self._reranker.predict(chunk)
                if hasattr(chunk_scores, "tolist"):
                    chunk_scores = chunk_scores.tolist()
                computed_scores.extend(float(score) for score in chunk_scores)

            for idx, score in zip(missing_indices, computed_scores):
                scores[idx] = float(score)

            if self._rerank_cache_size > 0 and computed_scores:
                with self._rerank_cache_lock:
                    for idx, score in zip(missing_indices, computed_scores):
                        key = cache_keys[idx]
                        if key is None:
                            continue
                        self._rerank_cache[key] = float(score)
                        while len(self._rerank_cache) > self._rerank_cache_size:
                            self._rerank_cache.popitem(last=False)

        return [float(score) if score is not None else 0.0 for score in scores]

    def upsert_payloads(
        self,
        payloads: List[Dict[str, Any]],
        text_representation_key: str = "content",
        collection_name: Optional[str] = None,
    ) -> None:
        """Write each payload as a SOURCE row in Postgres; the worker chunks, embeds and indexes it.

        No inline embedding and no vector write here (and no local FAISS/BM25 update: those were in-process caches of what this process had
        upserted since it started, and an asynchronous write cannot feed them). A failure is logged and the caller carries on, as before.
        """
        from src.services.tenant_context import company_label
        from src.services.vector_store import sources

        default_collection = collection_name or self.primary_collection
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            structured = payload.get("payload")
            base_payload = dict(structured) if isinstance(structured, dict) else dict(payload)
            base_payload.pop("payload", None)
            for key in DISALLOWED_METADATA_KEYS:
                base_payload.pop(key, None)
            record_id = base_payload.get("record_id") or payload.get("record_id") or str(uuid.uuid4())
            base_payload["record_id"] = record_id
            target = self._extract_source_collection(base_payload) or default_collection
            base_payload.pop(text_representation_key, None)
            text = payload.get(text_representation_key)
            if text is None and text_representation_key != "content":
                text = base_payload.get(text_representation_key)
            if not isinstance(text, str) or not text.strip():
                fallback = payload.get("text_summary")
                text = fallback if isinstance(fallback, str) and fallback.strip() else None
            if not isinstance(text, str) or not text.strip():
                try:
                    text = json.dumps(base_payload, ensure_ascii=False)
                except (TypeError, ValueError):
                    text = ""
            if not text.strip():
                continue
            base_payload.pop("content", None)
            base_payload.pop("text_summary", None)
            # Several payloads can share a record_id (the chunks of one document, told apart by chunk_id): each is its own source, as each was
            # its own point before, or they would overwrite one another.
            chunk_id = base_payload.get("chunk_id")
            source_id = f"{record_id}:{chunk_id}" if chunk_id is not None else str(record_id)
            try:
                sources.put_source(company_label(), self._logical_collection(target) or str(target), source_id, text, base_payload)
            except Exception:
                logger.exception("Vector source write failed for collection %r record %r — document processing continues", target, record_id)

    def upsert_texts(self, texts: List[str], metadata: Optional[Dict] = None):
        """Backward compatible wrapper around :meth:`upsert_payloads`."""

        metadata = metadata or {}
        payloads: List[Dict[str, Any]] = []
        for text in texts:
            payloads.append({**metadata, "content": text})

        self.upsert_payloads(payloads, text_representation_key="content")

    @staticmethod
    def _extract_source_collection(payload: Dict[str, Any]) -> Optional[str]:
        """Return a collection override specified by the payload, if any."""

        candidate = payload.get("source_collection")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        nested = payload.get("payload")
        if isinstance(nested, dict):
            nested_candidate = nested.get("source_collection")
            if isinstance(nested_candidate, str) and nested_candidate.strip():
                return nested_candidate.strip()
        return None

    @staticmethod
    def _normalise_uuid(value: str) -> uuid.UUID:
        """Return a UUID, deriving one deterministically when needed."""

        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError):
            return uuid.uuid5(uuid.NAMESPACE_URL, str(value))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[models.Filter] = None,
        *,
        session_hint: Optional[str] = None,
        memory_fragments: Optional[List[str]] = None,
        policy_mode: bool = False,
        nltk_features: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
    ):
        """Retrieve and rerank documents for the given query."""
        if not query or not query.strip():
            return []

        base_query = re.sub(r"\s+", " ", query or "").strip()
        if not base_query:
            return []

        hint_text = self._compose_hint_text(session_hint, memory_fragments)
        feature_keywords: List[str] = []
        feature_phrases: List[str] = []
        nltk_features: Optional[Dict[str, Any]] = None
        if isinstance(nltk_features, dict):
            raw_keywords = nltk_features.get("keywords", [])
            raw_phrases = nltk_features.get("key_phrases", [])
            feature_keywords = [
                str(item).strip()
                for item in raw_keywords
                if isinstance(item, str) and str(item).strip()
            ]
            feature_phrases = [
                str(item).strip()
                for item in raw_phrases
                if isinstance(item, str) and str(item).strip()
            ]
            if feature_keywords or feature_phrases:
                logger.debug(
                    "Applying NLTK feature hints: keywords=%s phrases=%s",
                    feature_keywords[:8],
                    feature_phrases[:5],
                )

        (
            rewritten_query,
            auto_conditions,
            focus_document_types,
        ) = self._rewrite_query(base_query)
        ask_trace.current().set_queries(rewritten=rewritten_query)
        combined_filter = self._merge_filters(filters, auto_conditions)
        combined_filter, session_key = self._separate_session_scope(
            combined_filter, session_id
        )
        # `auto_conditions` are keyword heuristics: the word "invoice" in the query
        # pins source_type == 'Invoice'. That must not narrow the uploaded-documents
        # collection. A user who attaches a file has already named the document, and
        # the caller pins it by document_id; applying the heuristic on top silently
        # excluded the attachment whenever the classifier disagreed with the wording
        # (e.g. a file named *_Invoice_* classified source_type 'PO' answered
        # "I couldn't find that information"). Keep the caller's explicit filter only.
        explicit_filter, _ = self._separate_session_scope(filters, session_id)
        policy_mode = bool(policy_mode or self._looks_like_policy_query(base_query, hint_text))
        search_text = rewritten_query or base_query
        focus_document_types = {
            token.strip().lower()
            for token in (focus_document_types or [])
            if isinstance(token, str) and token.strip()
        }
        feature_hint_parts = feature_keywords[:6] + feature_phrases[:4]
        if feature_hint_parts:
            features_line = ", ".join(dict.fromkeys(feature_hint_parts))
            search_text = f"{search_text}\n\nFocus terms: {features_line}".strip()
            if hint_text:
                hint_text = f"{hint_text} | focus terms: {features_line}"
        if hint_text:
            search_text = f"{search_text}\n\nConversation context: {hint_text}".strip()
        candidates: Dict[str, SimpleNamespace] = {}

        query_matrix: Optional[np.ndarray] = None
        query_vector: Optional[List[float]] = None

        def _ensure_query_vectors() -> Tuple[np.ndarray, List[float]]:
            nonlocal query_matrix, query_vector
            if query_matrix is not None and query_vector is not None:
                return query_matrix, query_vector

            with ask_trace.current().span("embed_query"):
                encoded = self.embedder.encode(search_text, normalize_embeddings=True)
            arr = (
                encoded
                if isinstance(encoded, np.ndarray)
                else np.array(encoded, dtype="float32")
            )
            if arr.ndim == 1:
                matrix = arr.reshape(1, -1).astype("float32")
            else:
                matrix = np.array(arr, dtype="float32")
            vector = matrix[0].astype("float32").tolist()
            query_matrix, query_vector = matrix, vector
            return query_matrix, query_vector

        use_cache = combined_filter is None and not auto_conditions and not hint_text
        cached_entries: List[Dict[str, Any]] = []
        if use_cache:
            cached_entries = self._semantic_cache.get_cached_queries(search_text)
            direct_cache = self._rebuild_cached_hits(
                cached_entries, search_text, top_k
            )
            if direct_cache is not None:
                return direct_cache

        effective_top_k = max(1, int(top_k or 1))
        prefetch_ceiling = max(
            effective_top_k,
            min(self._prefetch_limit, self._prefetch_cap),
        )
        ratio_target = max(
            effective_top_k,
            int(math.ceil(effective_top_k * self._prefetch_ratio)),
        )
        prefetch = min(ratio_target, prefetch_ceiling)
        prefetch = max(effective_top_k, prefetch)
        weights = {"dense": 1.0, "sparse": 0.75, "vector": 1.0, "cache": 0.6}

        def _register_candidate(
            source: str,
            payload: Dict[str, Any],
            doc_id: Any,
            raw_score: float,
            rank: int,
            weight: float,
        ) -> None:
            if not isinstance(payload, dict) or doc_id is None:
                return
            normalised_payload = {
                key: value
                for key, value in payload.items()
                if key not in DISALLOWED_METADATA_KEYS
            }
            doc_id_str = str(doc_id)
            collection = normalised_payload.get(
                "collection_name", self.primary_collection
            )
            chunk_marker = normalised_payload.get("chunk_id")
            if chunk_marker is None:
                chunk_marker = normalised_payload.get("chunk_index")
            if chunk_marker is None:
                chunk_marker = normalised_payload.get("chunk_hash")
            key = (
                f"{collection}:{doc_id_str}:{chunk_marker}"
                if chunk_marker is not None
                else f"{collection}:{doc_id_str}"
            )
            entry = candidates.get(key)
            if entry is None:
                entry = SimpleNamespace(
                    id=(
                        f"{doc_id_str}:{chunk_marker}"
                        if chunk_marker is not None
                        else doc_id_str
                    ),
                    payload=dict(normalised_payload),
                    score_components={},
                    aggregated=0.0,
                    chunk_id=chunk_marker,
                )
                candidates[key] = entry
            rrf = weight * (1.0 / (self._rrf_k + rank))
            entry.score_components[source] = {
                "raw": float(raw_score),
                "rank": int(rank),
                "rrf": float(rrf),
            }
            entry.aggregated += float(rrf)
            entry.payload.setdefault("_query_rewrite", rewritten_query or base_query)
            if hint_text:
                entry.payload.setdefault("_query_context", hint_text)

        if cached_entries:
            self._augment_candidates_from_cache(
                cached_entries, _register_candidate, weights["cache"]
            )

        if self._doc_vectors:
            matrix, _ = _ensure_query_vectors()
            dense_candidates: List[Tuple[float, int]] = []
            if self._faiss_index is not None:
                scores, ids = self._faiss_index.search(matrix, prefetch)
                dense_candidates = list(zip(scores[0], ids[0]))
            else:
                doc_matrix = np.vstack(self._doc_vectors)
                query_vec = matrix[0]
                sims = doc_matrix @ query_vec
                order = np.argsort(sims)[::-1][:prefetch]
                dense_candidates = [(float(sims[idx]), int(idx)) for idx in order]

            for rank, (score, idx) in enumerate(dense_candidates, 1):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                payload = dict(doc)
                payload.setdefault(
                    "collection_name", doc.get("collection_name", self.primary_collection)
                )
                _register_candidate(
                    "dense", payload, doc.get("id"), float(score), rank, weights["dense"]
                )

        if self._bm25 is not None and self._documents:
            tokens = rewritten_query.lower().split()
            if feature_keywords:
                tokens.extend(
                    keyword.lower()
                    for keyword in feature_keywords
                    if keyword.lower() not in tokens
                )
            if feature_phrases:
                for phrase in feature_phrases:
                    for part in phrase.lower().split():
                        if part not in tokens:
                            tokens.append(part)
            bm25_scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                enumerate(bm25_scores), key=lambda item: item[1], reverse=True
            )[:prefetch]
            for rank, (idx, score) in enumerate(ranked, 1):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                payload = dict(doc)
                payload.setdefault(
                    "collection_name", doc.get("collection_name", self.primary_collection)
                )
                _register_candidate(
                    "sparse", payload, doc.get("id"), float(score), rank, weights["sparse"]
                )

        collections_to_query: Tuple[Tuple[str, Optional[models.Filter]], ...] = tuple()
        base_collections: List[Tuple[str, Optional[models.Filter]]] = []

        def _filter_signature(value: Optional[models.Filter]) -> str:
            if value is None:
                return "∅"
            try:
                return value.model_dump_json(sort_keys=True)
            except Exception:
                try:
                    return json.dumps(value.model_dump(), sort_keys=True)
                except Exception:
                    return repr(value)

        seen_pairs: Set[Tuple[str, str]] = set()

        def _append_collection(
            name: Optional[str], filt: Optional[models.Filter]
        ) -> None:
            if not name:
                return
            signature = _filter_signature(filt)
            key = (name, signature)
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            base_collections.append((name, filt))

        forced_names: List[str] = []
        for candidate in collections or []:
            try:
                cleaned_name = str(candidate).strip()
            except Exception:
                cleaned_name = ""
            if cleaned_name:
                forced_names.append(cleaned_name)
        forced_collections: Tuple[str, ...] = tuple(forced_names)

        session_specific_filter: Optional[models.Filter] = None
        if session_key:
            # Built from `explicit_filter`, not `combined_filter`: see the note above
            # on keyword auto_conditions excluding the user's own attachment.
            session_specific_filter = self._merge_filters(
                explicit_filter,
                [
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_key),
                    )
                ],
            )
        if forced_collections:
            for name in forced_collections:
                if (
                    session_specific_filter is not None
                    and name == self.uploaded_collection
                ):
                    _append_collection(name, session_specific_filter)
                else:
                    _append_collection(name, combined_filter)
        else:
            if session_specific_filter is not None:
                _append_collection(self.uploaded_collection, session_specific_filter)

            if policy_mode:
                ordered_candidates: List[Tuple[Optional[str], Optional[models.Filter]]] = [
                    (self.static_policy_collection, combined_filter),
                    (self.uploaded_collection, combined_filter),
                    (self.primary_collection, combined_filter),
                    (self.learning_collection, combined_filter),
                ]
            else:
                ordered_candidates = [
                    (self.uploaded_collection, combined_filter),
                    (self.primary_collection, combined_filter),
                    (self.static_policy_collection, combined_filter),
                    (self.learning_collection, combined_filter),
                ]

            for name, filt in ordered_candidates:
                if session_specific_filter is not None and name == self.uploaded_collection:
                    continue
                _append_collection(name, filt)

        collections_to_query = tuple(base_collections)

        if collections_to_query:
            _, vector = _ensure_query_vectors()
            with ask_trace.current().span("vector_search"):
                batched_hits = self._search_collections_parallel(
                    collections_to_query,
                    vector,
                    limit=prefetch,
                )
            for name, _ in collections_to_query:
                hits = batched_hits.get(name, [])
                for rank, hit in enumerate(hits[:prefetch], 1):
                    payload = dict(getattr(hit, "payload", {}) or {})
                    payload.setdefault("collection_name", name)
                    _register_candidate(
                        "vector",
                        payload,
                        getattr(hit, "id", payload.get("record_id")),
                        float(getattr(hit, "score", 0.0)),
                        rank,
                        weights["vector"],
                    )

        if not candidates:
            return []

        ranked_candidates = sorted(
            candidates.values(), key=lambda item: item.aggregated, reverse=True
        )
        hits = ranked_candidates[:prefetch]
        if not hits:
            return []

        rerank_scores: Dict[int, float] = {}
        rerank_span = min(len(hits), self._reranker_limit)
        confident_targets: List[Tuple[int, SimpleNamespace]] = [
            (idx, hits[idx])
            for idx in range(rerank_span)
            if float(getattr(hits[idx], "aggregated", 0.0)) >= self._reranker_score_floor
        ]
        rerank_targets = confident_targets or [
            (idx, hits[idx]) for idx in range(rerank_span)
        ]

        if rerank_targets:
            pair_query = search_text if search_text else rewritten_query
            pairs: List[Tuple[str, str]] = []
            for _, candidate in rerank_targets:
                payload = getattr(candidate, "payload", {}) or {}
                doc_text = self._prepare_reranker_text(payload)
                pairs.append((pair_query, doc_text))

            try:
                with ask_trace.current().span("rerank"):
                    target_scores = self._batched_reranker_predict(pairs)
            except Exception:  # pragma: no cover - defensive logging only
                logger.warning(
                    "Cross-encoder reranking failed; using aggregated scores for fallback",
                    exc_info=True,
                )
                target_scores = [0.0 for _ in pairs]

            for (idx, _), score in zip(rerank_targets, target_scores):
                rerank_scores[idx] = float(score)

        def _matches_focus_payload(payload: Dict[str, Any]) -> bool:
            if not focus_document_types:
                return False
            if not isinstance(payload, dict):
                return False
            text_bits: List[str] = []
            for key in ("document_type", "source_type", "source_category"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text_bits.append(value.lower())
            label_fields = payload.get("tags") or payload.get("labels")
            if isinstance(label_fields, (list, tuple, set)):
                for item in label_fields:
                    if isinstance(item, str) and item.strip():
                        text_bits.append(item.lower())
            if not text_bits:
                return False
            merged = " ".join(text_bits)
            return any(token in merged for token in focus_document_types)

        def _priority_bonus(hit: SimpleNamespace) -> float:
            payload = getattr(hit, "payload", {}) or {}
            collection = payload.get("collection_name", self.primary_collection)
            document_type = str(payload.get("document_type", "")).lower()
            source_type = str(payload.get("source_type", "")).lower()
            weights_map = getattr(self, "_preference_weights", {}) or {}
            collection_weights = weights_map.get("collection", {}) or {}
            document_weights = weights_map.get("document_type", {}) or {}

            collection_bonus = float(collection_weights.get(collection, 0.0))
            doc_bonus = float(document_weights.get(document_type, 0.0))

            is_policy_doc = (
                document_type == "policy"
                or "policy" in document_type
                or source_type == "policy"
                or "policy" in source_type
            )
            matches_focus = _matches_focus_payload(payload)

            if collection == self.static_policy_collection and not policy_mode:
                collection_bonus = min(collection_bonus, 0.08)
                doc_bonus = min(doc_bonus, 0.05)

            if policy_mode and (
                collection == self.static_policy_collection or is_policy_doc
            ):
                collection_bonus = max(collection_bonus, 0.14)
                doc_bonus = max(doc_bonus, 0.1)

            if focus_document_types:
                if matches_focus:
                    collection_bonus = max(collection_bonus, 0.12) + 0.12
                    if document_type:
                        doc_bonus = max(doc_bonus, 0.1)
                elif not policy_mode and (
                    collection == self.static_policy_collection or is_policy_doc
                ):
                    collection_bonus = min(collection_bonus, 0.02)
                    doc_bonus = 0.0

            return float(collection_bonus + doc_bonus)

        scored_hits: List[Tuple[SimpleNamespace, float, float]] = []
        rerank_weight = 1.35
        context_weight = 0.65
        for idx, hit in enumerate(hits):
            aggregated_score = float(getattr(hit, "aggregated", 0.0))
            base_score = float(rerank_scores.get(idx, aggregated_score))
            combined = (
                base_score * rerank_weight
                + aggregated_score * context_weight
                + _priority_bonus(hit)
            )
            hit.rerank_score = base_score
            # True only when the cross-encoder actually scored this hit. Otherwise ``rerank_score`` holds the
            # tiny retrieval-fusion score, which is on a different scale and must not be read as relevance.
            hit.reranked = idx in rerank_scores
            hit.combined_score = combined
            hit.context_score = aggregated_score
            scored_hits.append((hit, combined, base_score))

        if not scored_hits:
            return []

        scored_hits.sort(key=lambda item: item[1], reverse=True)

        if top_k <= 0:
            return []

        selected = scored_hits[:top_k]

        def _collection_name(hit_obj: SimpleNamespace) -> str:
            payload = getattr(hit_obj, "payload", {}) or {}
            return payload.get("collection_name", self.primary_collection)

        required_collections: Tuple[str, ...] = tuple(
            name for name, _ in collections_to_query if name
        )

        available_by_collection: Dict[str, List[Tuple[SimpleNamespace, float, float]]] = {}
        for record in scored_hits:
            collection = _collection_name(record[0])
            available_by_collection.setdefault(collection, []).append(record)

        def _has_collection(collection: str) -> bool:
            return any(_collection_name(hit) == collection for hit, _, _ in selected)

        for collection in required_collections:
            candidates_for_collection = available_by_collection.get(collection)
            if not candidates_for_collection or _has_collection(collection):
                continue
            replacement_candidate = candidates_for_collection[0]
            if len(selected) < top_k:
                selected.append(replacement_candidate)
            else:
                lowest_idx = min(range(len(selected)), key=lambda idx: selected[idx][1])
                selected[lowest_idx] = replacement_candidate

        if policy_mode and self.static_policy_collection:
            static_hits = [
                record
                for record in scored_hits
                if _collection_name(record[0]) == self.static_policy_collection
            ]
            if static_hits:
                max_static = min(len(static_hits), max(1, top_k // 2))
                for idx in range(max_static):
                    candidate = static_hits[idx]
                    if candidate not in selected:
                        if len(selected) < top_k:
                            selected.append(candidate)
                        else:
                            lowest_idx = min(
                                range(len(selected)), key=lambda i: selected[i][1]
                            )
                            selected[lowest_idx] = candidate

        def _matches_focus(hit_obj: SimpleNamespace) -> bool:
            return _matches_focus_payload(getattr(hit_obj, "payload", {}) or {})

        if focus_document_types and not any(
            _matches_focus(hit) for hit, _, _ in selected
        ):
            for candidate in scored_hits:
                if _matches_focus(candidate[0]):
                    if len(selected) < top_k:
                        selected.append(candidate)
                    else:
                        lowest_idx = min(
                            range(len(selected)), key=lambda i: selected[i][1]
                        )
                        selected[lowest_idx] = candidate
                    break

        deduped: List[Tuple[SimpleNamespace, float, float]] = []
        seen_keys: Set[Tuple[str, str, Any]] = set()
        for hit, combined, base in sorted(selected, key=lambda item: item[1], reverse=True):
            payload = getattr(hit, "payload", {}) or {}
            chunk_marker = payload.get("chunk_id")
            if chunk_marker is None:
                chunk_marker = payload.get("chunk_index")
            if chunk_marker is None:
                chunk_marker = payload.get("chunk_hash")
            key = (
                str(getattr(hit, "id", payload.get("record_id"))),
                _collection_name(hit),
                chunk_marker,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append((hit, combined, base))

        results = [hit for hit, _, _ in deduped[:top_k]]

        if use_cache:
            cache_payload: List[Dict[str, Any]] = []
            for hit, combined, base in deduped[:top_k]:
                payload = self._normalise_payload_for_cache(
                    getattr(hit, "payload", {}) or {}
                )
                cache_payload.append(
                    {
                        "id": getattr(hit, "id", payload.get("record_id")),
                        "collection_name": payload.get(
                            "collection_name", self.primary_collection
                        ),
                        "payload": payload,
                        "score": base,
                        "combined_score": combined,
                        "aggregated_score": getattr(hit, "aggregated", 0.0),
                    }
                )
            self._semantic_cache.set_query_results(search_text, cache_payload)

        return results

    def _rewrite_query(
        self, query: str
    ) -> Tuple[str, List[models.FieldCondition], Set[str]]:
        normalised = re.sub(r"\s+", " ", query or "").strip()
        if not normalised:
            return "", [], set()

        normalised = re.sub(
            r"(?i)\bpo[-\s]?(\d{3,})\b",
            lambda match: f"purchase order {match.group(1)}",
            normalised,
        )

        conditions: List[models.FieldCondition] = []
        focus_tokens: Set[str] = set()

        def _ensure_condition(field: str, value: str) -> None:
            for condition in conditions:
                if getattr(condition, "key", None) == field:
                    match_obj = getattr(condition, "match", None)
                    if isinstance(match_obj, models.MatchValue) and match_obj.value == value:
                        return
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            )

        def _mark_focus(*values: str) -> None:
            for value in values:
                if isinstance(value, str) and value.strip():
                    focus_tokens.add(value.strip().lower())

        lowered = normalised.lower()
        if "policy" in lowered:
            _ensure_condition("source_type", "Policy")
            _mark_focus("policy")
        if "invoice" in lowered:
            _ensure_condition("source_type", "Invoice")
            _mark_focus("invoice")
        if "purchase order" in lowered:
            _ensure_condition("source_type", "PO")
            _mark_focus("purchase order", "po")
        elif re.search(r"\bpo\b", lowered):
            _ensure_condition("source_type", "PO")
            _mark_focus("purchase order", "po")
        if "quote" in lowered:
            _ensure_condition("source_type", "Quote")
            _mark_focus("quote", "quotation")
        if "quotation" in lowered:
            _ensure_condition("source_type", "Quote")
            _mark_focus("quote", "quotation")
        if "contract" in lowered:
            _ensure_condition("source_type", "Contract")
            _mark_focus("contract", "agreement")
        if "agreement" in lowered:
            _ensure_condition("source_type", "Contract")
            _mark_focus("contract", "agreement")

        synonyms: Dict[str, List[str]] = {
            "payment terms": ["net terms", "discount period"],
            "lead time": ["delivery time", "turnaround"],
            "escalation": ["escalation clause"],
        }
        expansions: List[str] = []
        for phrase, alternatives in synonyms.items():
            if phrase in lowered:
                expansions.extend(alternatives)

        if expansions:
            unique = " ".join(sorted(set(expansions)))
            normalised = f"{normalised} {unique}".strip()

        if ask_flags.enabled(ask_flags.TYPE_FILTER_FIX):
            conditions = self._fold_type_conditions(conditions)
        return normalised, conditions, focus_tokens

    @staticmethod
    def _fold_type_conditions(conditions: List[Any]) -> List[Any]:
        """Drop the hard document-type conditions (see SOFT_TYPE_WORDS); the focus tokens still rank them first."""
        return [
            c for c in conditions
            if not (getattr(c, "key", None) == "source_type"
                    and isinstance(getattr(c, "match", None), models.MatchValue)
                    and c.match.value in SOFT_TYPE_WORDS)
        ]

    def _compose_hint_text(
        self,
        session_hint: Optional[str],
        memory_fragments: Optional[List[str]],
    ) -> str:
        hints: List[str] = []
        if session_hint:
            snippet = re.sub(r"\s+", " ", session_hint).strip()
            if snippet:
                hints.append(snippet)
        if memory_fragments:
            for fragment in memory_fragments:
                cleaned = re.sub(r"\s+", " ", str(fragment or "")).strip()
                if cleaned:
                    hints.append(cleaned)
        return " ".join(hints)

    def _looks_like_policy_query(
        self, query: str, hint_text: Optional[str]
    ) -> bool:
        """Detect policy-focused prompts without over-weighting conversation memory."""

        query_text = " ".join(
            part.strip().lower() for part in (query or "",) if part and part.strip()
        )
        if not query_text:
            return False

        policy_tokens = (
            "policy",
            "policies",
            "approved supplier",
            "supplier approval",
            "compliance",
            "procurement rule",
            "sourcing rule",
            "maverick",
            "non-approved",
            "delegation of authority",
        )
        if any(token in query_text for token in policy_tokens):
            return True

        if not hint_text:
            return False

        history_text = hint_text.strip().lower()
        if not history_text:
            return False

        reinforcing_tokens = (
            "policy exception",
            "policy breach",
            "approval matrix",
            "delegation of authority",
            "compliance escalation",
        )
        soft_cues = ("approval", "compliance", "policy")
        return any(token in history_text for token in reinforcing_tokens) and any(
            cue in query_text for cue in soft_cues
        )

    def _merge_filters(
        self,
        base_filter: Optional[models.Filter],
        extra_conditions: List[models.FieldCondition],
    ) -> Optional[models.Filter]:
        if not extra_conditions:
            return base_filter
        if base_filter is None:
            return models.Filter(must=list(extra_conditions))

        merged_must = list(getattr(base_filter, "must", []) or [])
        merged_must.extend(extra_conditions)
        merged_must_not = list(getattr(base_filter, "must_not", []) or [])
        merged_should = list(getattr(base_filter, "should", []) or [])
        return models.Filter(
            must=merged_must,
            must_not=merged_must_not if merged_must_not else None,
            should=merged_should if merged_should else None,
        )

    def _separate_session_scope(
        self,
        active_filter: Optional[models.Filter],
        session_id: Optional[str],
    ) -> Tuple[Optional[models.Filter], Optional[str]]:
        """Detach session-specific constraints from the shared filter space."""

        session_token = self._normalise_session_token(session_id)
        if active_filter is None:
            return None, session_token

        def _extract_from_condition(condition: Any) -> Tuple[bool, Optional[str]]:
            if isinstance(condition, models.FieldCondition):
                key = getattr(condition, "key", None)
                if key and str(key).strip().lower() == "session_id":
                    match_obj = getattr(condition, "match", None)
                    candidate: Optional[str] = None
                    if isinstance(match_obj, models.MatchValue):
                        candidate = self._normalise_session_token(match_obj.value)
                    elif isinstance(match_obj, models.MatchAny):
                        for item in getattr(match_obj, "any", []) or []:
                            candidate = self._normalise_session_token(item)
                            if candidate:
                                break
                    return True, candidate
            if isinstance(condition, dict):
                key = str(condition.get("key") or condition.get("field") or "").strip().lower()
                if key == "session_id":
                    value = condition.get("value")
                    if value is None:
                        match_payload = condition.get("match")
                        if isinstance(match_payload, dict):
                            value = match_payload.get("value") or match_payload.get("any")
                    candidate = None
                    if isinstance(value, list):
                        for item in value:
                            candidate = self._normalise_session_token(item)
                            if candidate:
                                break
                    else:
                        candidate = self._normalise_session_token(value)
                    return True, candidate
            return False, None

        def _process_conditions(items: Optional[Sequence[Any]]) -> List[Any]:
            nonlocal session_token
            processed: List[Any] = []
            for condition in list(items or []):
                should_remove, found = _extract_from_condition(condition)
                if should_remove:
                    if not session_token and found:
                        session_token = found
                    elif found:
                        session_token = session_token or found
                    continue
                processed.append(condition)
            return processed

        must_conditions = _process_conditions(getattr(active_filter, "must", None))
        must_not_conditions = _process_conditions(getattr(active_filter, "must_not", None))
        should_conditions = _process_conditions(getattr(active_filter, "should", None))

        if not (must_conditions or must_not_conditions or should_conditions):
            return None, session_token

        cleaned_filter = models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
            should=should_conditions or None,
        )
        return cleaned_filter, session_token

    @staticmethod
    def _normalise_session_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    def create_langchain_retriever_tool(
        self,
        name: str = "procwise_rag_retriever",
        description: Optional[str] = None,
        top_k: int = 5,
    ):
        """Expose this service as a LangChain retriever tool."""

        if importlib.util.find_spec("langchain_core") is None:
            raise RuntimeError(
                "LangChain integrations requested but 'langchain-core' is missing."
            )
        if importlib.util.find_spec("langchain.tools") is None:
            raise RuntimeError(
                "LangChain tool support requires the 'langchain' extra dependencies."
            )

        retriever_module = importlib.import_module("langchain_core.retrievers")
        documents_module = importlib.import_module("langchain_core.documents")
        tools_module = importlib.import_module("langchain.tools")

        base_retriever_cls = getattr(retriever_module, "BaseRetriever")
        document_cls = getattr(documents_module, "Document")
        create_retriever_tool = getattr(tools_module, "create_retriever_tool")

        rag_service = self

        class _RAGServiceRetriever(base_retriever_cls):
            def __init__(self, limit: int):
                super().__init__()
                self._limit = max(1, int(limit))

            def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs):
                hits = rag_service.search(query, top_k=self._limit)
                documents: List[Any] = []
                for hit in hits:
                    payload: Dict[str, Any]
                    if hasattr(hit, "payload") and isinstance(hit.payload, dict):
                        payload = dict(hit.payload)
                    elif isinstance(hit, dict):
                        payload = dict(hit)
                    else:
                        payload = {"content": str(hit)}
                    text = (
                        payload.get("content")
                        or payload.get("text_summary")
                        or payload.get("summary")
                        or ""
                    )
                    documents.append(
                        document_cls(page_content=str(text), metadata=payload)
                    )
                return documents

            async def _aget_relevant_documents(  # type: ignore[override]
                self,
                query: str,
                *,
                run_manager=None,
                **kwargs,
            ):
                return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)

        tool_description = description or (
            "Retrieve procurement knowledge base documents from the vector store to "
            "ground supplier negotiations and policy checks."
        )
        retriever = _RAGServiceRetriever(limit=top_k)
        return create_retriever_tool(retriever, name=name, description=tool_description)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_init_gpu_index(self, index: "faiss.Index"):
        """Attempt to move ``index`` to GPU when available.

        FAISS exposes GPU helpers only in GPU-enabled builds. Some
        environments (including CI or developer laptops) only have the CPU
        variant installed; attempting to access ``StandardGpuResources`` in
        those cases raises :class:`AttributeError`. To keep the agentic
        workflow robust we detect this condition and gracefully fall back to
        the CPU index while logging the reason.
        """

        if faiss is None:
            return index

        if getattr(self.agent_nick, "device", "cpu") != "cuda":
            return index

        if not hasattr(faiss, "StandardGpuResources"):
            logger.warning(
                "FAISS GPU resources are unavailable in this environment; "
                "falling back to CPU index."
            )
            return index

    # ------------------------------------------------------------------
    # LangCache integration helpers
    # ------------------------------------------------------------------
    def _rebuild_cached_hits(
        self,
        cached_entries: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> Optional[List[SimpleNamespace]]:
        rebuilt: List[SimpleNamespace] = []
        for entry in cached_entries:
            response = entry.get("response")
            if not isinstance(response, str):
                continue
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                continue
            hits = payload.get("hits")
            if not isinstance(hits, list):
                continue
            similarity = float(entry.get("similarity") or 0.0)
            if entry.get("prompt") == query and similarity >= 0.995:
                for hit in hits[:top_k]:
                    rebuilt_hit = self._convert_cache_hit(hit)
                    if rebuilt_hit is not None:
                        collection_name = (
                            getattr(rebuilt_hit, "payload", {}) or {}
                        ).get("collection_name")
                        if (
                            collection_name
                            and collection_name == self.learning_collection
                        ):
                            continue
                        rebuilt.append(rebuilt_hit)
                if rebuilt:
                    return rebuilt[:top_k]
        return None

    def _augment_candidates_from_cache(
        self,
        cached_entries: List[Dict[str, Any]],
        register: Callable[[str, Dict[str, Any], Any, float, int, float], None],
        weight: float,
    ) -> None:
        for entry in cached_entries:
            response = entry.get("response")
            if not isinstance(response, str):
                continue
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                continue
            hits = payload.get("hits")
            if not isinstance(hits, list):
                continue
            similarity = float(entry.get("similarity") or 0.0)
            similarity_weight = max(0.45, min(1.0, similarity))
            effective_weight = weight * similarity_weight
            for rank, cached_hit in enumerate(hits, 1):
                rebuilt = self._convert_cache_hit(cached_hit)
                if rebuilt is None:
                    continue
                score = float(
                    cached_hit.get("combined_score", cached_hit.get("score", 0.0))
                )
                register(
                    "cache",
                    rebuilt.payload,
                    rebuilt.id,
                    score,
                    rank,
                    effective_weight,
                )

    def _convert_cache_hit(self, cached_hit: Dict[str, Any]) -> Optional[SimpleNamespace]:
        if not isinstance(cached_hit, dict):
            return None
        payload = cached_hit.get("payload")
        if not isinstance(payload, dict):
            return None
        payload = dict(payload)
        collection = cached_hit.get(
            "collection_name", payload.get("collection_name", self.primary_collection)
        )
        payload.setdefault("collection_name", collection)
        hit_id = cached_hit.get("id") or payload.get("record_id")
        if hit_id is None:
            return None
        score = float(cached_hit.get("combined_score", cached_hit.get("score", 0.0)))
        return SimpleNamespace(id=str(hit_id), payload=payload, score=score)

    def _normalise_payload_for_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _normalise(value: Any):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [_normalise(item) for item in value]
            if isinstance(value, tuple):
                return [_normalise(item) for item in value]
            if isinstance(value, dict):
                return {str(key): _normalise(val) for key, val in value.items()}
            return str(value)

        filtered = {
            key: value
            for key, value in (payload or {}).items()
            if key not in DISALLOWED_METADATA_KEYS
        }
        try:
            json.dumps(filtered)
            return filtered
        except (TypeError, ValueError):
            return _normalise(filtered)

        try:  # pragma: no cover - depends on GPU libraries
            resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:  # pragma: no cover - hardware specific
            logger.warning(
                "Failed to initialise FAISS GPU resources, falling back to CPU "
                "index: %s",
                exc,
            )
            return index
