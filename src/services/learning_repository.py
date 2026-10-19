"""Centralised repository for vectorised workflow learnings.

The repository maintains a dedicated vector store collection that stores metadata
based learnings produced across workflows.  Only structured metadata and
high-level summaries are embedded – raw email bodies, internal scoring logic
and any sensitive calculations are explicitly filtered out.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional



logger = logging.getLogger(__name__)


_EXCLUDED_METADATA_KEYS: tuple[str, ...] = (
    "body",
    "raw_body",
    "raw_email",
    "analysis",
    "score",
    "scores",
    "scoring",
    "calculation",
    "calculations",
    "formula",
    "explanation",
    "debug",
)


class LearningRepository:
    """Helper that coordinates the ``learning`` collection of the vector store."""

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        self.collection_name: str = getattr(
            self.settings, "learning_collection_name", "learning"
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _put(self, content: str, payload: Dict[str, Any], failure: str, *args: Any) -> Optional[str]:
        """A learning record is a SOURCE row (its text and payload); the vector worker embeds it. Returns its id, or None on failure."""
        from src.services.vector_store import DEFAULT_TENANT, sources
        source_id = str(uuid.uuid4())
        try:
            sources.put_source(DEFAULT_TENANT, "learning", source_id, content, payload)
        except Exception:
            logger.exception(failure, *args)
            return None
        return source_id

    def _list(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Stored records, newest first, as the same payload dicts that were stored (the store's bookkeeping keys removed)."""
        from src.services.vector_store import DEFAULT_TENANT, get_vector_store
        try:
            hits = get_vector_store().list(DEFAULT_TENANT, collection="learning", filters=filters, limit=max(1, min(int(limit), 1000)),
                                           newest_first=True)
        except Exception:
            logger.exception("Failed to list learning records")
            return []
        out: List[Dict[str, Any]] = []
        for hit in hits:
            payload = {k: v for k, v in (hit.payload or {}).items() if k not in ("content", "chunk_id", "chunk_index")}
            if payload.get("document_type") == "model_plan":
                payload.pop("summary", None)                      # a model plan never had one; the worker adds a default
            out.append(payload)
        return out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def record_learning(
        self,
        *,
        workflow_id: Optional[str],
        source: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Persist a metadata learning point into the vector store."""

        safe_metadata = self._normalise_metadata(metadata)
        if workflow_id:
            safe_metadata.setdefault("workflow_id", workflow_id)

        summary = self._build_summary(source, event_type, safe_metadata)
        if not summary:
            logger.debug("Skipping learning record for %s – empty summary", event_type)
            return None

        payload: Dict[str, Any] = {
            "document_type": "learning",
            "source": source,
            "event_type": event_type,
            "workflow_id": workflow_id,
            "summary": summary,
            "metadata": safe_metadata,
        }

        if tags:
            cleaned_tags = [
                str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()
            ]
            if cleaned_tags:
                payload["tags"] = sorted(set(cleaned_tags))

        return self._put(summary, payload, "Failed to persist learning record for %s", event_type)

    def record_model_plan(
        self,
        *,
        model_name: str,
        plan_text: str,
        plan_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Persist a lightweight fine-tuning plan for a model.

        The record is stored in the learning collection so that agents can
        retrieve the latest humanisation playbook directly from the vector store.
        """

        cleaned_plan = str(plan_text or "").strip()
        if not cleaned_plan:
            logger.debug("Skipping model plan for %s – empty body", model_name)
            return None

        metadata = self._normalise_metadata(plan_metadata)
        metadata["model_name"] = model_name

        payload: Dict[str, Any] = {
            "document_type": "model_plan",
            "model_name": model_name,
            "plan_text": cleaned_plan,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if tags:
            cleaned_tags = [
                str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()
            ]
            if cleaned_tags:
                payload["tags"] = sorted(set(cleaned_tags))

        return self._put(cleaned_plan, payload, "Failed to persist model plan for %s", model_name)

    def record_email_learning(
        self,
        *,
        workflow_id: Optional[str],
        draft: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[str]:
        metadata = {
            "rfq_id": draft.get("rfq_id"),
            "supplier_id": draft.get("supplier_id"),
            "supplier_name": draft.get("supplier_name"),
            "subject": draft.get("subject"),
            "contact_level": draft.get("contact_level"),
            "recipients": draft.get("recipients"),
            "intent": context.get("intent") or draft.get("metadata", {}).get("intent"),
            "document_origin": context.get("document_origin"),
            "negotiation_round": draft.get("thread_index")
            or draft.get("metadata", {}).get("round"),
            "target_price": context.get("target_price"),
            "current_offer": context.get("current_offer"),
        }
        tags = [context.get("document_origin") or "digital"]
        return self.record_learning(
            workflow_id=workflow_id,
            source="email_drafting_agent",
            event_type="email_draft_generated",
            metadata=metadata,
            tags=tags,
        )

    def record_negotiation_learning(
        self,
        *,
        workflow_id: Optional[str],
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        awaiting_response: bool,
        supplier_reply_registered: bool,
    ) -> Optional[str]:
        metadata = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "strategy": decision.get("strategy"),
            "counter_price": decision.get("counter_price"),
            "target_price": decision.get("target_price"),
            "asks": decision.get("asks"),
            "lead_time_request": decision.get("lead_time_request"),
            "round": decision.get("round"),
            "awaiting_response": awaiting_response,
            "supplier_reply_count": state.get("supplier_reply_count"),
            "supplier_reply_registered": supplier_reply_registered,
        }
        return self.record_learning(
            workflow_id=workflow_id,
            source="negotiation_agent",
            event_type="negotiation_round",
            metadata=metadata,
            tags=["negotiation"],
        )

    def record_workflow_learning(
        self,
        *,
        workflow_id: str,
        workflow_name: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        opportunities = result.get("opportunities") or {}
        metadata = {
            "workflow_name": workflow_name,
            "status": result.get("status"),
            "total_suppliers": result.get("supplier_count"),
            "opportunity_categories": list(opportunities.keys()) if isinstance(opportunities, dict) else [],
        }
        return self.record_learning(
            workflow_id=workflow_id,
            source="workflow",
            event_type="workflow_complete",
            metadata=metadata,
            tags=[workflow_name or "workflow"],
        )

    def get_recent_learnings(
        self,
        *,
        workflow_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"document_type": "learning"}
        if workflow_id:
            filters["metadata"] = {"workflow_id": workflow_id}
        return self._list(filters, limit)

    def build_context_snapshot(
        self, *, workflow_id: Optional[str], limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        records = self.get_recent_learnings(workflow_id=workflow_id, limit=limit)
        if not records:
            return None
        return {
            "workflow_id": workflow_id,
            "count": len(records),
            "events": records,
        }

    def fetch_model_plans(
        self,
        *,
        model_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored model plans for a given model name, newest first."""

        plans = self._list({"document_type": "model_plan", "metadata": {"model_name": model_name}}, limit)
        plans.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return plans[:limit]

    def _normalise_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        if not isinstance(metadata, dict):
            return cleaned
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if any(excluded in lowered for excluded in _EXCLUDED_METADATA_KEYS):
                continue
            serialised = self._serialise_value(value)
            if serialised is None:
                continue
            cleaned[key] = serialised
        return cleaned

    def _serialise_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            collapsed = [self._serialise_value(item) for item in value]
            return [item for item in collapsed if item is not None]
        if isinstance(value, dict):
            return {
                str(key): self._serialise_value(val)
                for key, val in value.items()
                if self._serialise_value(val) is not None
            }
        text = str(value).strip()
        if not text:
            return None
        if len(text) > 512:
            text = text[:509] + "..."
        return text

    def _build_summary(self, source: str, event_type: str, metadata: Dict[str, Any]) -> str:
        parts = [f"Source: {source}", f"Event: {event_type}"]
        for key in sorted(metadata.keys()):
            value = metadata[key]
            if isinstance(value, (list, tuple)):
                display = ", ".join(str(item) for item in value[:5])
            elif isinstance(value, dict):
                display = json.dumps(value, ensure_ascii=False)[:200]
            else:
                display = str(value)
            parts.append(f"{key}: {display}")
        return "; ".join(parts)

