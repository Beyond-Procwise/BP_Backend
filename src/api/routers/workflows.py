# ProcWise/api/routers/workflows.py
"""API routes exposing the agent workflows."""
from __future__ import annotations

import json
import os
import time
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator, ConfigDict

from orchestration.orchestrator import Orchestrator
from api.auth import require_user
from services.model_selector import RAGPipeline
from services.opportunity_service import record_opportunity_feedback
from services.email_dispatch_service import EmailDispatchService
from services.backend_scheduler import BackendScheduler
from repositories import draft_rfq_emails_repo, workflow_email_tracking_repo
from agents.email_drafting_agent import EmailDraftingAgent

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")


logger = logging.getLogger(__name__)
def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return orchestrator


def get_rag_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline service is not available.")
    return pipeline


def get_agent_nick(request: Request):
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if not agent_nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return agent_nick


class AskRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    model_name: Optional[str] = None
    doc_type: Optional[str] = None
    product_type: Optional[str] = None
    file_path: Optional[str] = Field(default=None, description="Optional local file path", json_schema_extra={"example": None})
    context: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional caller-supplied screen context (e.g. a summary of what the "
        "user is currently viewing). Folded into the same redacted ad-hoc context slot as "
        "uploaded-file notes.",
    )
    display_currency: Optional[str] = Field(
        default=None,
        max_length=12,
        description="The currency the reader has selected on screen (the control on "
        "Procurement Home's top bar), or 'native' for as-billed. An analytic answer is "
        "stated in it, converted from the same rate batch GET /fx/rates serves the client, "
        "so a figure in the answer and the same figure on the tile beside it agree.",
    )
    persona: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Who is reading — cpo, category manager, finance. Orders the next "
        "steps offered under an analytic answer; ignored elsewhere.",
    )
    action_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="A next step the reader clicked, dispatched by id (e.g. "
        "analytic.supplier_concentration). The chip's label is not a question and must "
        "not be re-parsed as one; the id names the answer that was offered.",
    )

    @field_validator("doc_type", "product_type", "file_path", "session_id", "context",
                     "display_currency", "persona", "action_id", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == "string":
                return None
        return value

    @field_validator("doc_type", "product_type")
    @classmethod
    def _normalize_case(cls, value: Optional[str]) -> Optional[str]:
        return value.lower() if isinstance(value, str) else value


class RankingRequest(BaseModel):
    query: str


class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


class OpportunityMiningRequest(BaseModel):
    """Parameters for opportunity mining workflow."""

    workflow: str
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy specific parameters keyed by requirement name.",
    )
    min_financial_impact: float = Field(
        default=100.0,
        ge=0,
        description="Minimum savings required for an opportunity to be returned.",
    )

    @field_validator("workflow", mode="before")
    @classmethod
    def _normalise_workflow(cls, value: Any) -> str:
        if value is None:
            raise ValueError("workflow is required")
        if not isinstance(value, str):
            value = str(value)
        workflow = value.strip()
        if not workflow:
            raise ValueError("workflow must not be empty")
        return workflow

    @field_validator("conditions", mode="before")
    @classmethod
    def _default_conditions(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("conditions must be a mapping of field names to values")


class QuoteEvaluationRequest(BaseModel):
    """Input parameters for quote evaluation workflow."""

    supplier_names: Optional[List[str]] = None
    product_category: Optional[str] = None


class NegotiationRequest(BaseModel):
    supplier: str
    current_offer: float
    target_price: float
    user_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    amount: float
    currency: Optional[str] = None
    supplier_id: Optional[str] = None
    threshold: Optional[float] = None
    user_id: Optional[str] = None
    # Link the approval back to what it is about, so a decision in the Action
    # Centre can be traced to the finding/deal that raised it. Without these the
    # approval row is an orphan: an amount with no subject.
    deal_id: Optional[str] = None
    rfq_id: Optional[str] = None
    finding_id: Optional[str] = None


class SupplierInteractionRequest(BaseModel):
    message: str
    supplier_id: Optional[str] = None
    user_id: Optional[str] = None


class DiscrepancyRequest(BaseModel):
    extracted_docs: List[Dict[str, Any]]
    user_id: Optional[str] = None


class AgentType(BaseModel):
    """What the catalogue tells a client about an agent.

    `agentType` used to be here too, and it carried the Python class name —
    "QuoteComparisonAgent", "DataExtractionAgent". Nothing consumed it: the UI keys entirely
    off `slug` (`wfAgentBySlug`), the gateway never reads it, and the internal callers that
    do want the class name read it from `agent_definitions.json`, not from this response. So
    it was shipping our class names to the browser for no one. It is gone; `slug` is the
    contract, as it already was in practice.
    """

    agentId: int
    slug: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    required_inputs: List[str] = Field(default_factory=list)
    dependencies: List[str]
    # Additive: set only for DERIVED agents (catalogue entries created via
    # POST /agents), carrying the backing_slug they were derived from. None
    # for the 14 built-in agents. The workspace UI uses its presence to
    # decide which agent cards get a delete control -- built-ins can't be
    # deleted via DELETE /agents/{slug} and shouldn't invite the attempt.
    derived_from: Optional[str] = None


class OpportunityRejectionRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        description="Optional feedback describing why the opportunity was rejected.",
        max_length=2000,
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Identifier for the user submitting the feedback.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to persist alongside the feedback.",
    )
    opportunity_ref_id: Optional[str] = Field(
        default=None,
        description="Original opportunity reference identifier from the mining agent.",
    )


router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])


class EmailDraftPayload(BaseModel):
    """Structured representation of a stored RFQ draft."""

    unique_id: Optional[str] = Field(
        default=None,
        description="Unique identifier assigned to the draft (PROC-WF-XXXXXX).",
    )
    rfq_id: Optional[str] = Field(
        default=None,
        description="Legacy RFQ identifier used as a fallback when unique_id is absent.",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        description="Supplier reference associated with the draft.",
    )
    recipients: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of recipient email addresses to override the stored draft values.",
    )
    sender: Optional[EmailStr] = Field(
        default=None,
        description="Sender email address override.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject override to apply when dispatching the draft.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Body override to apply when dispatching the draft.",
    )
    action_id: Optional[str] = Field(
        default=None,
        description="Action identifier emitted by upstream orchestration steps.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary metadata captured alongside the draft for auditing.",
    )
    workflow_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Workflow context metadata propagated from the calling agent.",
    )
    workflow_email: Optional[bool] = Field(
        default=None,
        description="Indicates whether the draft participates in workflow tracking.",
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("unique_id", "rfq_id", "supplier_id", "action_id", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("recipients", mode="before")
    @classmethod
    def _normalise_recipients(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            emails = [part.strip() for part in value.split(",") if part.strip()]
            return emails or None
        if isinstance(value, (list, tuple)):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned or None
        raise TypeError("recipients must be a string, list, or tuple of email addresses")

    def resolved_identifier(self) -> Optional[str]:
        for candidate in (self.unique_id, self.rfq_id):
            if candidate:
                identifier = candidate.strip()
                if identifier:
                    return identifier
        return None

    def resolved_recipients(self) -> Optional[List[str]]:
        if not self.recipients:
            return None
        return [str(email).strip() for email in self.recipients if str(email).strip()]

    def resolved_sender(self) -> Optional[str]:
        if self.sender is None:
            return None
        sender = str(self.sender).strip()
        return sender or None

    def resolved_subject(self) -> Optional[str]:
        if self.subject is None:
            return None
        return str(self.subject).strip()

    def resolved_body(self) -> Optional[str]:
        if self.body is None:
            return None
        return str(self.body)

    @staticmethod
    def _coerce_bool_flag(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return None

    def resolved_workflow_context(self) -> Optional[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if isinstance(self.workflow_context, dict):
            candidates.append(self.workflow_context)
        if isinstance(self.metadata, dict):
            meta_ctx = self.metadata.get("workflow_context") or self.metadata.get(
                "workflow_dispatch_context"
            )
            if isinstance(meta_ctx, dict):
                candidates.append(meta_ctx)
        for candidate in candidates:
            cleaned = {
                str(key): value
                for key, value in candidate.items()
                if value is not None and value != ""
            }
            if cleaned:
                return cleaned
        return None

    def resolved_workflow_email(self) -> Optional[bool]:
        candidates: List[Any] = []
        if self.workflow_email is not None:
            candidates.append(self.workflow_email)
        if isinstance(self.metadata, dict):
            for key in ("workflow_email", "is_workflow_email"):
                if key in self.metadata:
                    candidates.append(self.metadata.get(key))
        for candidate in candidates:
            flag = self._coerce_bool_flag(candidate)
            if flag is not None:
                return flag
        return None


class EmailDispatchRequest(BaseModel):
    """Request payload for dispatching stored RFQ drafts."""

    unique_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the draft to dispatch (PROC-WF-XXXXXX).",
    )
    rfq_id: Optional[str] = Field(
        default=None,
        description="Fallback RFQ identifier when unique_id is unavailable.",
    )
    recipients: Optional[List[str]] = Field(
        default=None,
        description="Override recipient list for the dispatched email.",
    )
    sender: Optional[EmailStr] = Field(
        default=None,
        description="Override sender email address.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional subject override.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Optional body override.",
    )
    action_id: Optional[str] = Field(
        default=None,
        description="Identifier linking this dispatch request to upstream workflow actions.",
    )
    draft: Optional[EmailDraftPayload] = Field(
        default=None,
        description="Single draft payload when the request originates from the drafting agent.",
    )
    drafts: Optional[List[EmailDraftPayload]] = Field(
        default=None,
        description="Collection of draft payloads for batch dispatch scenarios.",
    )
    workflow_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explicit workflow context for the dispatch request.",
    )
    is_workflow_email: Optional[bool] = Field(
        default=None,
        description="Flag indicating whether this dispatch participates in workflow tracking.",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _extract_identifier(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if values.get("unique_id") or values.get("rfq_id"):
            return values

        draft_obj = values.get("draft")
        if isinstance(draft_obj, dict):
            unique_id = draft_obj.get("unique_id") or draft_obj.get("rfq_id")
            if unique_id:
                values["unique_id"] = unique_id
                return values

        drafts_array = values.get("drafts")
        if isinstance(drafts_array, list) and drafts_array:
            first_draft = drafts_array[0]
            if isinstance(first_draft, dict):
                unique_id = first_draft.get("unique_id") or first_draft.get("rfq_id")
                if unique_id:
                    values["unique_id"] = unique_id
                    return values

        if "unique_id" in values or "supplier_id" in values:
            return values

        available_fields = list(values.keys())
        raise ValueError(
            "No identifier found in request. Provide 'unique_id' or 'rfq_id'. "
            f"Available fields: {available_fields}"
        )

    @field_validator("unique_id", "rfq_id", "action_id", mode="before")
    @classmethod
    def _strip_identifiers(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("recipients", mode="before")
    @classmethod
    def _normalise_recipients(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            emails = [part.strip() for part in value.split(",") if part.strip()]
            return emails or None
        if isinstance(value, (list, tuple)):
            cleaned: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned or None
        raise TypeError("recipients must be provided as a string, list, or tuple of emails")

    def get_identifier(self) -> str:
        for candidate in (self.unique_id, self.rfq_id):
            if candidate:
                identifier = candidate.strip()
                if identifier:
                    return identifier

        if self.draft:
            identifier = self.draft.resolved_identifier()
            if identifier:
                return identifier

        if self.drafts:
            for draft in self.drafts:
                identifier = draft.resolved_identifier()
                if identifier:
                    return identifier

        return ""

    def resolve_recipients(self) -> Optional[List[str]]:
        if self.recipients:
            return [str(email).strip() for email in self.recipients if str(email).strip()]
        if self.draft:
            recipients = self.draft.resolved_recipients()
            if recipients:
                return recipients
        return None

    def resolve_sender(self) -> Optional[str]:
        if self.sender is not None:
            sender = str(self.sender).strip()
            return sender or None
        if self.draft:
            return self.draft.resolved_sender()
        return None

    def resolve_workflow_context(self) -> Optional[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if isinstance(self.workflow_context, dict):
            candidates.append(self.workflow_context)
        if self.draft:
            ctx = self.draft.resolved_workflow_context()
            if ctx:
                candidates.append(ctx)
        if self.drafts:
            for draft in self.drafts:
                ctx = draft.resolved_workflow_context()
                if ctx:
                    candidates.append(ctx)
                    break
        for candidate in candidates:
            cleaned = {
                str(key): value
                for key, value in candidate.items()
                if value is not None and value != ""
            }
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return None

    def resolve_is_workflow_email(self) -> Optional[bool]:
        candidates: List[Any] = [self.is_workflow_email]
        if self.draft:
            candidates.append(self.draft.resolved_workflow_email())
        if self.drafts:
            for draft in self.drafts:
                flag = draft.resolved_workflow_email()
                if flag is not None:
                    candidates.append(flag)
                    break
        for candidate in candidates:
            flag = self._coerce_bool(candidate)
            if flag is not None:
                return flag
        return None

    def resolve_subject(self) -> Optional[str]:
        if self.subject is not None:
            return str(self.subject).strip()
        if self.draft:
            return self.draft.resolved_subject()
        return None

    def resolve_body(self) -> Optional[str]:
        if self.body is not None:
            return str(self.body)
        if self.draft:
            return self.draft.resolved_body()
        return None

    def resolve_action_id(self) -> Optional[str]:
        if self.action_id:
            return self.action_id
        if self.draft and self.draft.action_id:
            return self.draft.action_id
        return None


class EmailBatchDispatchRequest(BaseModel):
    """Request payload for batch email dispatch operations."""

    drafts: List[EmailDraftPayload] = Field(
        ..., description="List of drafts to dispatch in a single batch operation."
    )

    model_config = ConfigDict(extra="allow")


class EmailDispatchResponse(BaseModel):
    success: bool
    unique_id: str
    sent: bool
    # `sent` (and `success`) mean "this draft has been sent", which is TRUE for a draft
    # that went out earlier: the dispatch service short-circuits a re-send and returns
    # the original message_id without putting anything on the wire. These two say what
    # THIS call did, so a caller reporting an outcome to a person cannot fabricate a
    # success -- declared rather than left to `extra="allow"` so the contract is visible
    # in the schema the gateway and Swagger read.
    dispatched_now: Optional[bool] = None
    duplicate: Optional[bool] = None
    duplicate_note: Optional[str] = None
    message_id: Optional[str] = None
    recipients: List[str]
    sender: str
    subject: str
    body: Optional[str] = None
    draft: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="allow")


def _merge_form_values(form_data) -> Dict[str, Any]:
    """Flatten ``FormData`` into a standard dictionary preserving multi-values."""

    flattened: Dict[str, Any] = {}
    for key, value in getattr(form_data, "multi_items", lambda: [])():
        if key in flattened:
            existing = flattened[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                flattened[key] = [existing, value]
        else:
            flattened[key] = value

    for key, value in list(flattened.items()):
        if isinstance(value, list) and len(value) == 1:
            flattened[key] = value[0]

    return flattened


def _maybe_parse_embedded_json(value: Any) -> Any:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate[0] in ("{", "["):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return value
    return value


def _deserialise_structured_fields(payload: Dict[str, Any]) -> None:
    for key in ("draft", "drafts"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, list):
            payload[key] = [_maybe_parse_embedded_json(item) for item in value]
        else:
            payload[key] = _maybe_parse_embedded_json(value)


def _coerce_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _extract_workflow_token(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        try:
            payload = dict(payload)  # type: ignore[arg-type]
        except Exception:
            return None

    for key in ("workflow_id", "workflowId", "workflowID", "process_workflow_id"):
        token = _coerce_identifier(payload.get(key))
        if token:
            return token
    return None


def _resolve_workflow_from_dispatch_result(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None

    workflow_identifier = _coerce_identifier(result.get("workflow_id"))
    if workflow_identifier:
        return workflow_identifier

    draft_payload = result.get("draft")
    if isinstance(draft_payload, dict):
        candidate = _extract_workflow_token(draft_payload)
        if candidate:
            return candidate

    context_payload = result.get("workflow_context")
    if isinstance(context_payload, dict):
        candidate = _coerce_identifier(context_payload.get("workflow_id"))
        if candidate:
            return candidate

    dispatch_metadata = result.get("dispatch_metadata")
    if isinstance(dispatch_metadata, dict):
        candidate = _coerce_identifier(dispatch_metadata.get("workflow_id"))
        if candidate:
            return candidate

    return None


def _resolve_dispatch_workflow_id(
    request_model: "EmailDispatchRequest", identifier: Optional[str]
) -> Optional[str]:
    candidate = None

    extras = getattr(request_model, "model_extra", None)
    if isinstance(extras, dict):
        candidate = _extract_workflow_token(extras)

    if not candidate:
        draft = getattr(request_model, "draft", None)
        if draft is not None:
            try:
                candidate = _extract_workflow_token(draft.model_dump(exclude_none=True))
            except Exception:
                candidate = None
            if not candidate:
                draft_extras = getattr(draft, "model_extra", None)
                if isinstance(draft_extras, dict):
                    candidate = _extract_workflow_token(draft_extras)
            if not candidate:
                metadata = getattr(draft, "metadata", None)
                if isinstance(metadata, dict):
                    candidate = _extract_workflow_token(metadata)

    if not candidate:
        drafts = getattr(request_model, "drafts", None)
        if isinstance(drafts, list):
            for entry in drafts:
                if entry is None:
                    continue
                try:
                    entry_dict = entry.model_dump(exclude_none=True)  # type: ignore[call-arg]
                except AttributeError:
                    entry_dict = entry if isinstance(entry, dict) else None
                if not isinstance(entry_dict, dict):
                    continue
                candidate = _extract_workflow_token(entry_dict)
                if candidate:
                    break
                metadata = entry_dict.get("metadata")
                candidate = _extract_workflow_token(metadata)
                if candidate:
                    break

    if candidate:
        return candidate

    identifier_token = _coerce_identifier(identifier)
    if not identifier_token:
        return None

    try:
        draft_record = draft_rfq_emails_repo.load_by_unique_id(identifier_token)
    except Exception:
        logger.exception(
            "Failed to resolve workflow_id from draft repository for identifier=%s",
            identifier_token,
        )
        return None

    if isinstance(draft_record, dict):
        return _coerce_identifier(draft_record.get("workflow_id"))
    return None


async def build_email_dispatch_request(request: Request) -> EmailDispatchRequest:
    """Load the dispatch request supporting both JSON and form submissions."""

    body_bytes = await request.body()
    raw_payload: Any

    if not body_bytes:
        raw_payload = {}
    else:
        try:
            raw_payload = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                form = await request.form()
            except Exception:
                raw_payload = {}
            else:
                raw_payload = _merge_form_values(form)

    if not isinstance(raw_payload, dict):
        raw_payload = {}

    _deserialise_structured_fields(raw_payload)

    try:
        return EmailDispatchRequest.model_validate(raw_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask")
async def ask_question(
    req: AskRequest,
    request: Request,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    principal: object = Depends(require_user),
):
    def _clean(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value or None

    file_data: List[tuple[bytes, str]] = []
    if req.file_path:
        _upload_base = os.path.realpath(
            os.getenv("PROCWISE_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "uploads"))
        )
        _resolved = os.path.realpath(req.file_path)
        if not _resolved.startswith(_upload_base + os.sep) and _resolved != _upload_base:
            raise HTTPException(status_code=400, detail="file_path outside allowed directory")
        if not os.path.isfile(_resolved):
            raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")
        try:
            with open(_resolved, "rb") as f:
                file_data.append((f.read(), os.path.basename(_resolved)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    header_session = _clean(request.headers.get("x-session-id"))
    resolved_session = _clean(req.session_id) or header_session or _clean(req.user_id)

    result = await run_in_threadpool(
        pipeline.answer_question,
        query=req.query,
        user_id=req.user_id,
        session_id=resolved_session,
        model_name=req.model_name,
        files=file_data or None,
        doc_type=req.doc_type,
        product_type=req.product_type,
        screen_context=req.context,
        display_currency=req.display_currency,
        persona=req.persona,
        action_id=req.action_id,
    )
    return result


@router.post("/ask/stream")
async def ask_question_stream(
    req: AskRequest,
    request: Request,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    principal: object = Depends(require_user),
):
    """Stream an answer as Server-Sent Events.

    The same grounded pipeline as POST /ask — same retrieval, same corpus facts,
    same session continuity — but the answer arrives progressively instead of
    after a 10-30s stare at a spinner.

    Events (each `data:` is one JSON object with a `type`):
      stage  - retrieving | grounded (with the sources) | generating
      delta  - a piece of ANSWER PROSE. Not raw model output: the model replies in
               JSON, so the tokens are decoded out of the `answer` field before
               they are sent (see JsonFieldStreamer). The client can append these
               straight to the screen.
      done   - the final answer, follow-ups and retrieved documents
      error  - something failed; the message is included

    The pipeline is blocking and CPU/GPU-bound, so it runs in a worker thread and
    pushes events onto a queue that this coroutine drains. Doing it inline would
    block the event loop and stall every other request on the server.
    """
    import queue as _queue

    events: _queue.Queue = _queue.Queue()
    _DONE = object()

    header_session = (request.headers.get("x-session-id") or "").strip() or None
    resolved_session = req.session_id or header_session or req.user_id

    def _on_event(kind: str, payload: Dict[str, Any]) -> None:
        events.put({"type": kind, **payload})

    def _work() -> None:
        try:
            result = pipeline.answer_question(
                query=req.query,
                user_id=req.user_id,
                session_id=resolved_session,
                model_name=req.model_name,
                doc_type=req.doc_type,
                product_type=req.product_type,
                screen_context=req.context,
                display_currency=req.display_currency,
                persona=req.persona,
                action_id=req.action_id,
                on_event=_on_event,
            )
            events.put(
                {
                    "type": "done",
                    "answer": result.get("answer") or "",
                    "follow_ups": result.get("follow_ups") or [],
                    "retrieved_documents": result.get("retrieved_documents") or [],
                    # An analytic answer's steps: an action and the ids it
                    # applies to, so a chip dispatches rather than re-asks.
                    "next_steps": result.get("next_steps") or [],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask stream failed")
            events.put({"type": "error", "message": str(exc)[:300]})
        finally:
            events.put(_DONE)

    async def _publish():
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _work)
        try:
            while True:
                event = await loop.run_in_executor(None, events.get)
                if event is _DONE:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(
        _publish(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would hold the
            # whole stream back and deliver it in one lump — defeating the point.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/rank")
def rank_suppliers(
    req: RankingRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    # Use execute_ranking_flow which derives criteria from the query text
    # before calling execute_workflow. This satisfies the policy engine's
    # requirement for ranking criteria to be present.
    return orchestrator.execute_ranking_flow(req.query)


@router.post("/quotes/evaluate")
def evaluate_quotes(
    req: QuoteEvaluationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Evaluate and compare supplier quotes."""
    return orchestrator.execute_workflow("quote_evaluation", req.model_dump())


@router.post("/opportunities")
def mine_opportunities(
    req: OpportunityMiningRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    prs = orchestrator.agent_nick.process_routing_service
    process_id = prs.log_process(
        process_name="opportunity_mining",
        process_details=req.model_dump(),
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="opportunity_miner",
        action_desc=req.model_dump(),
        status="started",
    )
    try:
        result = orchestrator.execute_workflow(
            "opportunity_mining", req.model_dump()
        )
        prs.log_action(
            process_id=process_id,
            agent_type="opportunity_miner",
            action_desc=req.model_dump(),
            process_output=result,
            status="completed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, 1)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        prs.log_action(
            process_id=process_id,
            agent_type="opportunity_miner",
            action_desc=str(exc),
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/opportunities/{opportunity_id}/reject")
def reject_opportunity(
    opportunity_id: str,
    req: OpportunityRejectionRequest,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    if not opportunity_id or not opportunity_id.strip():
        raise HTTPException(status_code=400, detail="opportunity_id must be provided")

    try:
        # Who rejected it is the token. `req.user_id` is kept -- clients send
        # it and it sometimes carries something a person meant -- but as a
        # label inside metadata, where nothing reads it as identity.
        metadata = dict(req.metadata or {})
        if req.user_id:
            metadata["user_id_label"] = req.user_id
        record = record_opportunity_feedback(
            agent_nick,
            opportunity_id.strip(),
            opportunity_ref_id=req.opportunity_ref_id,
            status="rejected",
            reason=req.reason,
            user_id=getattr(principal, "subject", None) or None,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - database/network
        logger.exception("Failed to record rejection for opportunity %s", opportunity_id)
        raise HTTPException(status_code=500, detail="Failed to record opportunity feedback") from exc

    updated_on = record.get("updated_on")
    if isinstance(updated_on, (bytes, str)):
        updated_iso = str(updated_on)
    elif updated_on is not None:
        updated_iso = updated_on.isoformat()
    else:
        updated_iso = None

    payload = {
        "opportunity_id": record.get("opportunity_id"),
        "opportunity_ref_id": record.get("opportunity_ref_id"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "user_id": record.get("user_id"),
        "metadata": record.get("metadata"),
        "updated_on": updated_iso,
    }
    return {"status": "success", "feedback": payload}


@router.post("/extract")
async def extract_documents(
    req: ExtractRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    prs = orchestrator.agent_nick.process_routing_service
    process_id = prs.log_process(
        process_name="document_extraction",
        process_details={
            "s3_prefix": req.s3_prefix,
            "s3_object_key": req.s3_object_key,
        },
        # process_status=1,
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="document_extraction",
        action_desc={
            "s3_prefix": req.s3_prefix,
            "s3_object_key": req.s3_object_key,
        },
        status="started",
    )

    async def run_flow() -> None:
        try:
            result = await run_in_threadpool(
                orchestrator.execute_extraction_flow,
                req.s3_prefix,
                req.s3_object_key,
            )
            prs.log_action(
                process_id=process_id,
                agent_type="document_extraction",
                action_desc={
                    "s3_prefix": req.s3_prefix,
                    "s3_object_key": req.s3_object_key,
                },
                process_output=result,
                status="completed",
                action_id=action_id,
            )
            prs.update_process_status(process_id, 1)
        except Exception as exc:  # pragma: no cover - network/runtime
            prs.log_action(
                process_id=process_id,
                agent_type="document_extraction",
                action_desc=str(exc),
                status="failed",
                action_id=action_id,
            )
            prs.update_process_status(process_id, -1)

    asyncio.create_task(run_flow())
    return {"status": "process started", "process_id": process_id}


# ---------------------------------------------------------------------------
# Email draft persistence endpoint (report panel "Save"/pre-send step)
# ---------------------------------------------------------------------------
class EmailPrepareRequest(BaseModel):
    """Payload for persisting a report-panel email edit ahead of dispatch.

    The report's email panel edits recipients/subject/body but has no
    ``unique_id``/``rfq_id`` of its own, so ``POST /workflows/email`` 400s
    (it requires an identifier that resolves to a stored draft). This model
    backs ``POST /workflows/email/prepare``, which persists the panel's
    edited content as a draft and hands back an identifier the panel can
    then pass to ``/workflows/email``.
    """

    deal_id: Optional[str] = Field(
        default=None,
        description="Deal this draft is associated with (used to scope/label the draft).",
    )
    to: List[EmailStr] = Field(
        ..., min_length=1, description="Recipient email address(es)."
    )
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body (HTML or plain text).")
    reply_to_unique_id: Optional[str] = Field(
        default=None,
        description=(
            "The draft this is a REPLY to (its unique_id). When given, the new draft "
            "inherits that thread's supplier, workflow, In-Reply-To/References headers "
            "and attachment records, so the reply lands in the supplier's existing "
            "conversation instead of starting a new one."
        ),
    )

    @field_validator("subject", "body", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("subject", "body")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class EmailPrepareResponse(BaseModel):
    """Identifier the panel should send to ``POST /workflows/email``.

    The three ``reply_to_unique_id`` fields are reported rather than assumed: a caller
    that asked for a threaded reply must be able to see whether the thread headers and
    the attachments actually made it onto the new draft, and refuse to send if they did
    not. Silence would let a reply arrive as a new conversation, or without the files the
    human attached, with nothing on screen saying so.
    """

    unique_id: str
    workflow_id: str
    status: str = "prepared"
    # None when no reply_to_unique_id was given, or when the thread carried no
    # message id to reply to (in which case threading is genuinely unavailable).
    in_reply_to: Optional[str] = None
    thread_headers_carried: bool = False
    attachments_carried: int = 0


def _draft_payload(draft: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The draft row's ``payload`` column as a dict, tolerating a JSON string."""
    raw = (draft or {}).get("payload")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    return raw if isinstance(raw, dict) else {}


def _reply_thread_headers(
    *, source_draft: Dict[str, Any], source_payload: Dict[str, Any]
) -> Dict[str, List[str]]:
    """``In-Reply-To``/``References`` that put a reply in the supplier's own thread.

    Where the values come from, in the order they matter:

      * ``proc.workflow_email_tracking.response_message_id`` -- the Message-ID of the
        SUPPLIER'S reply, which is the message we are answering, so it is the correct
        ``In-Reply-To``. Written by the watcher when it matched the reply.
      * ``...message_id`` -- the Message-ID of OUR outbound RFQ. Used as ``In-Reply-To``
        only when the supplier's own is not recorded (an older row), and always folded
        into ``References`` so the chain is complete.
      * any ``References`` already recorded against the thread, kept in front.

    Nothing is fabricated. A thread with no recorded message ids returns ``{}`` and the
    caller reports that, rather than sending a reply that looks like a new conversation
    while claiming otherwise.
    """
    tracked = None
    workflow_id = source_draft.get("workflow_id")
    unique_id = source_draft.get("unique_id")
    if workflow_id and unique_id:
        try:
            tracked = workflow_email_tracking_repo.lookup_dispatch_row(
                workflow_id=str(workflow_id), unique_id=str(unique_id)
            )
        except Exception:  # noqa: BLE001 - absence of threading is reported, not raised
            logger.exception(
                "could not read the dispatch row for unique_id=%s while preparing a reply",
                unique_id,
            )

    # Whatever the thread already recorded, from the tracking row or the draft itself.
    prior: Dict[str, Any] = {}
    for candidate in (
        getattr(tracked, "thread_headers", None),
        source_draft.get("thread_headers"),
        source_payload.get("thread_headers"),
        (source_payload.get("metadata") or {}).get("thread_headers")
        if isinstance(source_payload.get("metadata"), dict) else None,
    ):
        if isinstance(candidate, dict) and candidate:
            prior = candidate
            break

    def _as_list(value: Any) -> List[str]:
        """Message ids in one canonical ``<id>`` form.

        The two sources disagree on form and both end up in ``References``:
        ``workflow_email_tracking._parse_thread_headers`` strips ``<>`` off everything it
        reads back, while the ``message_id``/``response_message_id`` columns come back
        verbatim with their brackets. RFC 5322 requires every msg-id in ``In-Reply-To``/
        ``References`` to be bracketed, so a bare one makes the header malformed; and
        without a single form, the same message listed both ways survives the dedupe
        twice. A ``References`` string may also carry several space-separated ids.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            tokens = [str(v) for v in value]
        else:
            tokens = [str(value)]

        ids: List[str] = []
        for token in tokens:
            for part in token.split():
                bare = part.strip().strip("<>").strip()
                if bare:
                    ids.append(f"<{bare}>")
        return ids

    our_outbound = _as_list(getattr(tracked, "message_id", None) or prior.get("Message-ID"))
    their_reply = _as_list(getattr(tracked, "response_message_id", None))

    # The message we are actually answering. Their reply if we recorded it; our own
    # outbound otherwise -- which still lands in the right conversation, one hop up.
    in_reply_to = (their_reply or our_outbound)[:1]
    if not in_reply_to:
        return {}

    references: List[str] = []
    for item in _as_list(prior.get("References")) + our_outbound + their_reply:
        if item not in references:
            references.append(item)

    headers: Dict[str, List[str]] = {"In-Reply-To": in_reply_to}
    if references:
        headers["References"] = references
    return headers


@router.post(
    "/email/prepare",
    response_model=EmailPrepareResponse,
    summary="Persist an edited report-panel email as a draft (no send)",
)
def prepare_email_draft(
    payload: EmailPrepareRequest,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> EmailPrepareResponse:
    """Persist ``payload`` into ``proc.draft_rfq_emails`` and return its identifier.

    This handler ONLY persists a draft -- it never calls the dispatch/send
    path. It reuses :meth:`EmailDraftingAgent._store_draft`, the same
    id-generation and persistence helper the drafting agent itself uses, so
    the resulting row lives in the exact table/shape that
    ``EmailDispatchService.resolve_workflow_id``/``send_draft`` read. The
    caller (report panel) should follow up with
    ``POST /workflows/email {"unique_id": <returned unique_id>, ...}`` to
    actually send.

    ``reply_to_unique_id`` makes this the REPLY path used by the Action Centre's
    supplier-reply panel. A reply is a new message and needs its own draft row: the
    escalation's own ``subject_id`` is the outbound RFQ the supplier already answered,
    and dispatching THAT id short-circuits in
    ``EmailDispatchService._maybe_return_existing_dispatch`` (already sent) and puts
    nothing on the wire. Preparing a fresh draft is what makes the send real -- and the
    two things a fresh draft would otherwise LOSE are carried explicitly here:

      * the thread (``In-Reply-To``/``References``, from the tracking row), so the reply
        arrives inside the supplier's conversation rather than starting a new one;
      * the attachments, whose records live on the ORIGINAL draft row because that is the
        id the panel uploaded them against.

    A new ``unique_id`` is minted per draft (``generate_unique_email_id`` mixes in
    ``secrets.token_hex``), so reusing the source thread's ``workflow_id`` cannot collide
    with the source row on ``ON CONFLICT (workflow_id, unique_id)`` -- the reply is a new
    row in the same workflow, and the original sent record is left untouched.
    """

    recipients = [str(addr).strip() for addr in payload.to if str(addr).strip()]
    if not recipients:
        raise HTTPException(
            status_code=400, detail="At least one recipient email is required"
        )

    source_draft: Optional[Dict[str, Any]] = None
    if payload.reply_to_unique_id:
        source_draft = draft_rfq_emails_repo.load_by_unique_id(
            str(payload.reply_to_unique_id).strip()
        )
        if not source_draft:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The thread this reply belongs to could not be found, so nothing "
                    "was prepared and nothing was sent."
                ),
            )

    # ``_store_draft`` requires a truthy supplier_id. On the reply path it is the real
    # supplier off the thread, so the reply is attributable; otherwise this draft isn't
    # tied to a supplier and a stable, harmless scoping value is derived from the deal.
    supplier_id = (source_draft or {}).get("supplier_id") or (
        f"report-panel:{payload.deal_id}"
        if payload.deal_id
        else f"report-panel:{uuid.uuid4().hex[:12]}"
    )

    # This is the human path into proc.draft_rfq_emails, so the row records who
    # asked. The approvals surface reads it back to refuse an approval signed by
    # the person who requested it (P3) -- which it could not do while no draft
    # named a requester at all. Taken from the principal and never from
    # ``payload``: a caller who could name the requester could name someone else
    # and approve their own draft freely.
    requested_by = str(getattr(principal, "subject", "") or "").strip() or None

    draft: Dict[str, Any] = {
        "supplier_id": supplier_id,
        "subject": payload.subject,
        "body": payload.body,
        "recipients": recipients,
        "receiver": recipients[0],
        "requested_by": requested_by,
        "metadata": {
            "source": "report_email_panel",
            "deal_id": payload.deal_id,
        },
    }

    thread_headers: Dict[str, List[str]] = {}
    if source_draft:
        source_payload = _draft_payload(source_draft)
        draft["metadata"]["source"] = "action_centre_reply_panel"
        draft["metadata"]["reply_to_unique_id"] = source_draft.get("unique_id")
        draft["supplier_name"] = source_draft.get("supplier_name")
        # Same workflow as the thread being answered, so the reply is tracked with it
        # rather than as an orphan run.
        if source_draft.get("workflow_id"):
            draft["workflow_id"] = str(source_draft["workflow_id"])
        thread_headers = _reply_thread_headers(
            source_draft=source_draft, source_payload=source_payload
        )
        if thread_headers:
            # TOP-LEVEL on purpose. ``_resolve_initial_thread_headers`` reads
            # ``draft["thread_headers"]`` FIRST and ``draft["headers"]`` second, and
            # ``_store_draft`` always fills ``headers`` with the X-ProcWise tracking
            # headers -- so a copy under ``metadata`` alone would lose to those and never
            # reach the wire. Recorded under metadata too, for the audit trail.
            draft["thread_headers"] = thread_headers
            draft["metadata"]["thread_headers"] = thread_headers

    drafting_agent = EmailDraftingAgent(agent_nick)
    try:
        drafting_agent._store_draft(draft)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to persist report-panel email draft for deal_id=%s",
            payload.deal_id,
        )
        raise HTTPException(status_code=500, detail="Failed to persist email draft")

    # ``_store_draft`` swallows DB errors internally (best-effort logging) and
    # only stamps these fields on the draft dict once the row is actually
    # committed -- so their absence is our fail-closed signal, not a warning.
    unique_id = draft.get("unique_id")
    workflow_id = draft.get("workflow_id")
    record_id = draft.get("draft_record_id")
    if not unique_id or not workflow_id or record_id is None:
        raise HTTPException(status_code=500, detail="Draft was not persisted")

    # The human's attachments were uploaded against the ORIGINAL draft's unique_id --
    # that is the only id the panel had. Their records are COPIED onto the new draft
    # (the S3 objects are not moved, so the source thread's record stays intact and
    # both rows point at the same bytes); ``_load_attachments`` at dispatch time reads
    # them by ``s3_key``, so the files reach the sent message. A failure here is raised,
    # not logged: sending a reply the user believes carries their contract, without it,
    # is worse than not sending.
    attachments_carried = 0
    if source_draft:
        source_attachments = _load_draft_attachments(source_draft)
        if source_attachments:
            try:
                _persist_draft_attachments(agent_nick, str(unique_id), source_attachments)
            except Exception:
                logger.exception(
                    "could not carry attachments from %s onto reply draft %s",
                    source_draft.get("unique_id"), unique_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The reply was prepared but its attachments could not be carried "
                        "onto it, so nothing was sent. Nothing has gone to the supplier."
                    ),
                )
            attachments_carried = len(source_attachments)

    return EmailPrepareResponse(
        unique_id=str(unique_id),
        workflow_id=str(workflow_id),
        in_reply_to=(thread_headers.get("In-Reply-To") or [None])[0],
        thread_headers_carried=bool(thread_headers),
        attachments_carried=attachments_carried,
    )


# ---------------------------------------------------------------------------
# Email dispatch endpoint
# ---------------------------------------------------------------------------
@router.post("/email")
async def send_email(
    dispatch_request: EmailDispatchRequest = Depends(build_email_dispatch_request),
    orchestrator: Orchestrator = Depends(get_orchestrator),
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    """Send a previously drafted RFQ email using the dispatch service."""

    identifier = dispatch_request.get_identifier()
    if not identifier:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing Identifier",
                "message": "No unique_id or rfq_id provided",
                "hint": "Send: {\"unique_id\": \"PROC-WF-xxx\"}",
            },
        )

    input_data = dispatch_request.model_dump(exclude_none=True)
    dispatch_service = EmailDispatchService(agent_nick)

    workflow_id_hint = _resolve_dispatch_workflow_id(dispatch_request, identifier)
    repository_workflow_id = dispatch_service.resolve_workflow_id(identifier)

    if workflow_id_hint and repository_workflow_id and workflow_id_hint != repository_workflow_id:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "WorkflowMismatch",
                "message": "Dispatch identifier is associated with a different workflow",
                "request_workflow_id": workflow_id_hint,
                "stored_workflow_id": repository_workflow_id,
            },
        )

    workflow_id_hint = workflow_id_hint or repository_workflow_id

    if not workflow_id_hint:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "WorkflowUnavailable",
                "message": "Unable to resolve workflow_id for dispatch identifier",
                "identifier": identifier,
            },
        )

    prs = orchestrator.agent_nick.process_routing_service

    initial_details = {
        "input": input_data,
        "agents": [],
        "output": {},
        "status": "saved",
        "workflow_id": workflow_id_hint,
    }
    process_id = prs.log_process(
        process_name="email_dispatch",
        process_details=initial_details,
        workflow_id=workflow_id_hint,
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_reference = dispatch_request.resolve_action_id()

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="email_dispatch",
        action_desc=input_data,
        status="started",
        action_id=action_reference,
    )

    workflow_id_final = workflow_id_hint

    try:
        result = await run_in_threadpool(
            dispatch_service.send_draft,
            identifier=identifier,
            recipients=dispatch_request.resolve_recipients(),
            sender=dispatch_request.resolve_sender(),
            subject_override=dispatch_request.resolve_subject(),
            body_override=dispatch_request.resolve_body(),
            is_workflow_email=dispatch_request.resolve_is_workflow_email(),
            workflow_dispatch_context=dispatch_request.resolve_workflow_context(),
            principal=principal,
        )

        dispatch_timestamp = time.time()
        setattr(agent_nick, "dispatch_service_started", True)
        setattr(agent_nick, "email_dispatch_last_sent_at", dispatch_timestamp)

        raw_recipients = result.get("recipients")
        if isinstance(raw_recipients, str):
            response_recipients = [raw_recipients]
        elif isinstance(raw_recipients, list):
            response_recipients = raw_recipients
        elif raw_recipients:
            response_recipients = list(raw_recipients)
        else:
            response_recipients = dispatch_request.resolve_recipients() or []

        result_workflow_id = _resolve_workflow_from_dispatch_result(result)

        if workflow_id_hint and result_workflow_id and workflow_id_hint != result_workflow_id:
            logger.error(
                "Dispatch workflow mismatch for identifier=%s request_workflow=%s result_workflow=%s",
                identifier,
                workflow_id_hint,
                result_workflow_id,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "WorkflowMismatch",
                    "message": "Dispatch completed under different workflow identifier",
                    "request_workflow_id": workflow_id_hint,
                    "result_workflow_id": result_workflow_id,
                },
            )

        workflow_id_final = result_workflow_id or workflow_id_hint

        response_data: Dict[str, Any] = dict(result)
        response_data.update(
            success=bool(result.get("sent", False)),
            unique_id=result.get("unique_id", identifier),
            sent=bool(result.get("sent", False)),
            message_id=result.get("message_id"),
            recipients=[str(r) for r in response_recipients],
            sender=str(result.get("sender") or dispatch_request.resolve_sender() or ""),
            subject=str(result.get("subject") or dispatch_request.resolve_subject() or ""),
            error=result.get("error"),
            workflow_id=workflow_id_final,
        )

        if response_data.get("body") is None:
            response_data["body"] = dispatch_request.resolve_body()

        response = EmailDispatchResponse(**response_data)

        response_payload = response.model_dump()
        status_label = "completed" if response.sent else "failed"

        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            process_output=response_payload,
            status=status_label,
            action_id=action_id,
        )

        final_details = {
            "input": input_data,
            "agents": [],
            "output": response_payload,
            "status": status_label,
            "workflow_id": workflow_id_final,
        }
        prs.update_process_details(process_id, final_details)
        prs.update_process_status(process_id, 1 if response.sent else -1)

    except ValueError as exc:
        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)

        error_message = str(exc)
        status_code = 404 if "No stored draft" in error_message else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": "Draft Dispatch Failed",
                "message": error_message,
                "identifier": identifier,
            },
        )

    except Exception as exc:  # pragma: no cover - network/runtime
        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        **response_payload,
        "status": status_label,
        "result": response_payload,
        "action_id": action_id,
        "workflow_id": workflow_id_final,
    }


@router.post("/email/batch")
async def dispatch_batch_emails(
    request: EmailBatchDispatchRequest,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    if not request.drafts:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No Drafts",
                "message": "No drafts found in request body",
                "hint": "Send EmailDraftingAgent output with 'drafts' array",
            },
        )

    dispatch_service = EmailDispatchService(agent_nick)

    results: List[Dict[str, Any]] = []
    workflows_to_notify: Set[str] = set()
    # Sends actually attempted in this batch so far -- what the guard's
    # volume cap (check 5) counts against. A draft skipped below for having
    # no identifier never reaches send_draft, so it must not advance this.
    send_attempts = 0
    for draft in request.drafts:
        identifier = draft.resolved_identifier()
        if not identifier:
            logger.warning(
                "Skipping draft without identifier: supplier_id=%s", draft.supplier_id
            )
            results.append(
                {
                    "unique_id": None,
                    "sent": False,
                    "error": "Draft missing unique identifier",
                    "supplier_id": draft.supplier_id,
                    "subject": draft.resolved_subject(),
                }
            )
            continue

        try:
            result = await run_in_threadpool(
                dispatch_service.send_draft,
                identifier=identifier,
                recipients=draft.resolved_recipients(),
                sender=draft.resolved_sender(),
                subject_override=draft.resolved_subject(),
                body_override=draft.resolved_body(),
                notify_watcher=False,
                principal=principal,
                run_count=send_attempts,
            )
            send_attempts += 1
            results.append(
                {
                    "unique_id": identifier,
                    "sent": bool(result.get("sent")),
                    "message_id": result.get("message_id"),
                    "supplier_id": draft.supplier_id,
                    "subject": result.get("subject") or draft.resolved_subject(),
                }
            )
            if result.get("sent"):
                workflow_identifier = _resolve_workflow_from_dispatch_result(result)
                if workflow_identifier:
                    workflows_to_notify.add(workflow_identifier)
        except Exception as exc:  # pragma: no cover - runtime dependent
            send_attempts += 1
            logger.error("Failed to dispatch %s: %s", identifier, str(exc))
            results.append(
                {
                    "unique_id": identifier,
                    "sent": False,
                    "error": str(exc),
                    "supplier_id": draft.supplier_id,
                    "subject": draft.resolved_subject(),
                }
            )

    success_count = sum(1 for r in results if r.get("sent"))

    if workflows_to_notify:
        try:
            scheduler = BackendScheduler.ensure(agent_nick)
            for workflow_identifier in workflows_to_notify:
                try:
                    scheduler.notify_email_dispatch(workflow_identifier)
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to trigger email watcher for workflow %s",
                        workflow_identifier,
                    )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialise backend scheduler for batch dispatch")

    return {
        "success": success_count == len(results) if results else False,
        "total": len(results),
        "sent": success_count,
        "failed": len(results) - success_count,
        "results": results,
    }


_ATTACHMENT_ALLOWED_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".txt",
}


def _load_draft_attachments(draft: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Existing attachment records for a draft, tolerating a JSON-string column."""
    existing = draft.get("attachments") if draft else None
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except (TypeError, ValueError):
            existing = []
    return list(existing or [])


def _unique_attachment_key(unique_id: str, name: str, existing_names: Set[str]) -> str:
    """Build the S3 key for one attachment, so storage identity never disagrees
    with the record that names it.

    ``email-attachments/{unique_id}/{name}`` collides whenever a second file
    with the same display name is stored against the same draft -- either a
    re-upload of a corrected version, or two identically-named files in one
    request. An unguarded second write would silently replace the first
    object's bytes in S3 while the FIRST record stayed in the attachments
    array pointing at that same (now different) key: the record would then
    say one thing, S3 would hold another, and ``_load_attachments`` would send
    the wrong bytes for one of the two entries at dispatch time.

    We keep the re-upload (rather than rejecting it -- the user's intent in
    re-attaching "terms.pdf" is obvious and rejecting it would just be
    annoying) by disambiguating the STORAGE key with a short random suffix.
    The record's own ``filename`` stays exactly what the user typed; only the
    key changes, so two "terms.pdf" entries can coexist, each pointing at its
    own untouched bytes.
    """
    if name not in existing_names:
        return f"email-attachments/{unique_id}/{name}"
    stem, ext = os.path.splitext(name)
    disambiguator = uuid.uuid4().hex[:8]
    return f"email-attachments/{unique_id}/{stem}__{disambiguator}{ext}"


def _persist_draft_attachments(agent_nick, unique_id: str, records: List[Dict[str, Any]]) -> None:
    """Write the attachment list onto the draft row. Raises on failure — a stored
    file with no record is invisible, and silence here would produce exactly that."""
    with agent_nick.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE proc.draft_rfq_emails SET attachments = %s::jsonb, updated_on = now()"
                " WHERE unique_id = %s",
                (json.dumps(records), unique_id),
            )
        conn.commit()


@router.post(
    "/email/{unique_id}/attachments",
    summary="Attach files to a persisted draft (no send)",
)
async def add_email_attachments(
    unique_id: str,
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="api"),
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Store attachments against a draft, in S3 plus a record on the draft row.

    The bytes go to an email-attachments/ prefix, NOT through
    /data-integration/presigned-url: that path feeds document extraction, and a
    supplier's countersigned contract arriving as an "uploaded document" would
    raise discrepancies against itself.

    A previous upload's attachments are loaded first and appended to — never
    overwritten — so successive uploads to the same draft accumulate. A file
    whose display name collides with one already stored (or with another file
    in the same request) is kept under a disambiguated storage key (see
    ``_unique_attachment_key``) so the stored bytes, the record, and what gets
    sent can never disagree about which file is which.
    """
    from datetime import datetime, timezone

    draft = draft_rfq_emails_repo.load_by_unique_id(unique_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"No draft for unique_id {unique_id}")

    records: List[Dict[str, Any]] = _load_draft_attachments(draft)
    total = sum(int(r.get("bytes") or 0) for r in records)
    existing_names: Set[str] = {r.get("filename") for r in records if r.get("filename")}

    service = EmailDispatchService(agent_nick)
    rejected: List[Dict[str, str]] = []
    for upload in files:
        name = os.path.basename(upload.filename or "")
        suffix = os.path.splitext(name)[1].lower()
        if not name or suffix not in _ATTACHMENT_ALLOWED_SUFFIXES:
            rejected.append({"filename": name, "reason": f"file type {suffix or 'unknown'} not allowed"})
            continue
        data = await upload.read()
        if len(data) > service._ATTACHMENT_MAX_BYTES:
            rejected.append({"filename": name, "reason": "file exceeds the per-file limit"})
            continue
        if total + len(data) > service._ATTACHMENT_MAX_TOTAL_BYTES:
            rejected.append({"filename": name, "reason": "message would exceed the total attachment limit"})
            continue
        key = _unique_attachment_key(unique_id, name, existing_names)
        try:
            await run_in_threadpool(service._write_s3_bytes, key, data, upload.content_type)
        except Exception as exc:  # noqa: BLE001
            # `rejected[].reason` is rendered VERBATIM in the review panel's "Not
            # attached" box. A boto3/ClientError message names the bucket and the object
            # key, so interpolating it here put storage internals on a buyer's screen --
            # the same defect already fixed on the fail-closed rationale and GET
            # /decisions. The type, the message and the traceback go to the LOG under a
            # short reference that also appears in the reason, which is what connects the
            # two without the user reading either.
            ref = uuid.uuid4().hex[:8]
            logger.exception(
                "attachment upload failed for unique_id=%s key=%s [ref %s]: %s: %s",
                unique_id, key, ref, type(exc).__name__, exc,
            )
            rejected.append({
                "filename": name,
                "reason": (
                    "it could not be stored, so it was not attached. The technical "
                    f"details were recorded for support under reference {ref}."
                ),
            })
            continue
        total += len(data)
        existing_names.add(name)
        records.append(
            {
                "filename": name,
                "content_type": upload.content_type or "application/octet-stream",
                "bytes": len(data),
                "s3_key": key,
                # These attachments ride out to a supplier on a real email, so
                # who added one is the token. The Form field stays as a label:
                # it defaults to the literal string "api", which was being
                # written as though it named somebody.
                "added_by": getattr(principal, "subject", None) or None,
                "added_by_label": user_id or None,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    _persist_draft_attachments(agent_nick, unique_id, records)
    return {"unique_id": unique_id, "attachments": records, "rejected": rejected}


@router.delete(
    "/email/{unique_id}/attachments/{index}",
    summary="Remove one attachment from a draft before sending",
)
def remove_email_attachment(
    unique_id: str,
    index: int,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    draft = draft_rfq_emails_repo.load_by_unique_id(unique_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"No draft for unique_id {unique_id}")
    records = _load_draft_attachments(draft)
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail=f"No attachment at index {index}")
    records.pop(index)
    _persist_draft_attachments(agent_nick, unique_id, records)
    return {"unique_id": unique_id, "attachments": records}


@router.post("/{workflow_id}/email/dispatch-all")
async def dispatch_workflow_drafts(
    workflow_id: str,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT unique_id, supplier_id, subject, sent
                    FROM proc.draft_rfq_emails
                    WHERE workflow_id = %s
                      AND sent = FALSE
                    ORDER BY id ASC
                    """,
                    (workflow_id,),
                )
                draft_rows = cur.fetchall()

        if not draft_rows:
            return {
                "success": True,
                "workflow_id": workflow_id,
                "total": 0,
                "sent": 0,
                "failed": 0,
                "message": "No unsent drafts found for workflow",
            }

        dispatch_service = EmailDispatchService(agent_nick)

        results: List[Dict[str, Any]] = []
        workflows_to_notify: Set[str] = set()
        # Sends actually attempted in this run so far -- what the guard's
        # volume cap (check 5) counts against.
        send_attempts = 0
        for unique_id, supplier_id, subject, sent in draft_rows:
            try:
                result = await run_in_threadpool(
                    dispatch_service.send_draft,
                    identifier=unique_id,
                    subject_override=subject,
                    notify_watcher=False,
                    principal=principal,
                    run_count=send_attempts,
                )
                send_attempts += 1
                results.append(
                    {
                        "unique_id": unique_id,
                        "sent": bool(result.get("sent")),
                        "message_id": result.get("message_id"),
                        "supplier_id": supplier_id,
                        "subject": subject,
                    }
                )
                if result.get("sent"):
                    workflow_identifier = _resolve_workflow_from_dispatch_result(result)
                    if workflow_identifier:
                        workflows_to_notify.add(workflow_identifier)
            except Exception as exc:  # pragma: no cover - runtime dependent
                send_attempts += 1
                logger.error("Failed to dispatch %s: %s", unique_id, str(exc))
                results.append(
                    {
                        "unique_id": unique_id,
                        "sent": False,
                        "error": str(exc),
                        "supplier_id": supplier_id,
                        "subject": subject,
                    }
                )

        success_count = sum(1 for r in results if r.get("sent"))

        if workflows_to_notify:
            try:
                scheduler = BackendScheduler.ensure(agent_nick)
                for workflow_identifier in workflows_to_notify:
                    try:
                        scheduler.notify_email_dispatch(workflow_identifier)
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "Failed to trigger email watcher for workflow %s",
                            workflow_identifier,
                        )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to initialise backend scheduler for workflow dispatch"
                )

        return {
            "success": success_count == len(results),
            "workflow_id": workflow_id,
            "total": len(results),
            "sent": success_count,
            "failed": len(results) - success_count,
            "results": results,
        }

    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("Workflow dispatch failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Workflow Dispatch Failed",
                "message": str(exc),
            },
        )



@router.post("/negotiate")
def negotiate(
    req: NegotiationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute the negotiation agent."""
    return orchestrator.execute_workflow("negotiation", req.model_dump())


@router.post("/approvals")
def approvals(
    req: ApprovalRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Run the approvals agent."""
    return orchestrator.execute_workflow("approvals", req.model_dump())


@router.post("/supplier-interaction")
def supplier_interaction(
    req: SupplierInteractionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Trigger the supplier interaction agent."""
    return orchestrator.execute_workflow("supplier_interaction", req.model_dump())


@router.post("/discrepancy")
def detect_discrepancy(
    req: DiscrepancyRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Expose the discrepancy detection agent."""
    return orchestrator.execute_workflow("discrepancy_detection", req.model_dump())


@router.get(
    "/types",
    response_model=List[AgentType],
    summary="Get agent types and their resource dependencies",
)
def get_agent_types():
    """Return the agent catalogue defined in ``agent_definitions.json``."""
    from agents.definitions import load_agent_definitions

    try:
        return [AgentType(**agent) for agent in load_agent_definitions()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="agent_definitions.json not found")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid agent definitions: {exc}")


# ── Observability Endpoints ─────────────────────────────────────

@router.get("/workflows/{workflow_id}/status")
def get_workflow_status(workflow_id: str, request: Request):
    """Get current workflow state and all node statuses."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_db_connection)
    conn = orchestrator.agent_nick.get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT execution_id, workflow_name, status, shared_data,
                      current_round, created_at, updated_at, completed_at
               FROM proc.workflow_execution
               WHERE workflow_id = %s
               ORDER BY created_at DESC LIMIT 1""",
            (workflow_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Workflow not found")
        execution_id = row[0]
        node_statuses = sm.get_node_statuses(execution_id, row[4])
        return {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "workflow_name": row[1],
            "status": row[2],
            "current_round": row[4],
            "created_at": str(row[5]),
            "updated_at": str(row[6]),
            "completed_at": str(row[7]) if row[7] else None,
            "nodes": node_statuses,
        }


@router.get("/workflows/{workflow_id}/events")
def get_workflow_events(workflow_id: str, request: Request, limit: int = 100):
    """Get event timeline for a workflow."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_db_connection)
    events = sm.get_events(workflow_id, limit=limit)
    return {"workflow_id": workflow_id, "events": events}


@router.get("/system/workflows/active")
def get_active_workflows(request: Request):
    """List all running/paused workflows."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_db_connection)
    return {"workflows": sm.get_active_workflows()}
