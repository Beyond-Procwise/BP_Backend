import asyncio
import json
import sys, os, uvicorn, logging
from contextlib import asynccontextmanager
from typing import Any, Optional, Protocol, cast

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

# Ensure GPU utilisation by default on compatible hardware
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
# Reduce peak CUDA-alloc fragmentation so a fresh start succeeds even when
# the prior process hasn't finished releasing memory yet (per PyTorch's own
# recommendation in OutOfMemoryError messages).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Force HuggingFace libraries to use local cached models only - no HTTP calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Suppress progress bars that journald renders as `[197B blob data]` lines.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline
from services.model_training_endpoint import ModelTrainingEndpoint
from services.email_watcher import run_email_watcher_for_workflow
from agents.base_agent import AgentNick
from agents.registry import AgentRegistry
from agents.data_extraction_agent import DataExtractionAgent
from agents.supplier_ranking_agent import SupplierRankingAgent
from agents.quote_evaluation_agent import QuoteEvaluationAgent
from agents.quote_comparison_agent import QuoteComparisonAgent
from agents.opportunity_miner_agent import OpportunityMinerAgent
from agents.discrepancy_detection_agent import DiscrepancyDetectionAgent
from agents.email_drafting_agent import EmailDraftingAgent
from agents.email_dispatch_agent import EmailDispatchAgent
from agents.negotiation_agent import NegotiationAgent
from agents.approvals_agent import ApprovalsAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from api.routers import agents as agents_router_mod, documents, email, metrics, run, stream, system, training, vendors, workflows, deal_summary, deal_proposals, promotion, summary, negotiate, opportunities,session
from api.routers import ws as ws_router_mod
# Imported here, not further down, because every router below is mounted with
# require_user as a dependency and that reference is resolved at mount time.
from api import auth as _ask_auth
from api.routers import agent_workflows as agent_workflows_router
from api.routers import agent_groups as agent_groups_router
from api.routers import models as models_router
from api.routers import decisions as decisions_router
from api.routers import support as support_router
from api.routers import extraction_feedback as extraction_feedback_router
from api.routers import supplier_review as supplier_review_router
from api.routers import supplier_research as supplier_research_router
from api.routers import governance as governance_router
from api.routers import obligations as obligations_router
from api.routers import benchmark as benchmark_router
# The requirements router existed but was never mounted, so POST
# /requirements/message 404'd and no UI could reach the agent at all.
from api.routers import requirements as requirements_router
from api.routers import fx as fx_router
from api.routers import value_summary as value_summary_router
from api.routers import analysis as analysis_router
from api.routers import approvals as approvals_router

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "procwise.log"))])
logger = logging.getLogger(__name__)

# Quiet noisy upstream loggers that emit at INFO/WARNING for harmless events.
# - neo4j.notifications: schema "index already exists" notices on every KG sync.
# - absl: LangExtract few-shot prompt-template self-alignment warnings.
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.ERROR)


class ProcwiseAppState(Protocol):
    agent_nick: Optional["AgentNick"]
    model_training_endpoint: Optional["ModelTrainingEndpoint"]
    orchestrator: Optional["Orchestrator"]
    rag_pipeline: Optional["RAGPipeline"]
    agent_registry: Optional["AgentRegistry"]
    supplier_interaction_agent: Optional["SupplierInteractionAgent"]
    negotiation_agent: Optional["NegotiationAgent"]
    email_watcher_runner: Optional[Any]
    backend_scheduler: Any
    email_watcher_service: Optional[Any]
    email_watcher_owned: bool
    process_monitor_watcher: Optional[Any]
    extraction_v3_schemas: dict
    session_notify_listener: Optional[Any]

def _optional(name: str, build, *, default=None):
    """Build a startup subsystem whose absence the API can serve without.

    Returns ``default`` and logs, rather than propagating, so one subsystem's
    failure costs that subsystem alone.

    The distinction this draws is the point. ``lifespan`` catches everything
    around its whole initialisation and responds by nulling fourteen pieces of
    app state -- which is the right answer when the core is gone and there is
    nothing left to serve with, and the wrong answer to a reranker that would
    not fit on the GPU. That is not a hypothetical: it happened on every boot of
    this host, and because the handler is at the end, everything after the
    failure point was skipped too -- the session-status bridge, the
    extraction-hint cache and the formula audit sink all silently did not start.

    A subsystem belongs here when its consumers already tolerate its absence.
    ``state.rag_pipeline`` is the model: both routers that use it raise a clean
    503 when it is missing, so losing it was always meant to be survivable. The
    core -- AgentNick, the registries, the orchestrator -- stays outside, and a
    failure there is still fatal and still loud.
    """
    try:
        return build()
    except Exception:
        logger.exception(
            "startup: %s is unavailable; continuing without it. The API will "
            "serve, and any endpoint that needs it will say so.", name,
        )
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    state = cast(ProcwiseAppState, app.state)

    # Make formula evaluation records durable, BEFORE the initialisation block
    # below. Every `evaluate` call already built a full EvaluationRecord --
    # inputs, output, version hash, findings, confidence -- and emitted it to a
    # 5,000-entry in-process ring buffer, so the records were correct and then
    # died with the process. Nothing reached proc.bp_agent_actions.
    #
    # Placement is the point. This sits outside that `try` because the whole of
    # it is skipped by one `except Exception` when anything in it fails, and it
    # does fail: a cross-encoder load OOMs whenever the GPU is busy, and the
    # handler nulls out the app state and yields. An audit sink depends on no
    # agent, no model and no GPU, so making its durability contingent on a warm
    # GPU is exactly backwards -- a degraded boot is when you most want the
    # record of what was computed.
    #
    # The database is deliberately not the *default* sink (ADR 0002 D6): a
    # formula run over pairs would flood the audit table, which is why the batch
    # sweeps go through `evaluate_many` and write one record each. That reasoning
    # is about sweeps, not this process -- the API's formulas run a handful of
    # times per request (benchmark's per-deal loop averages 4.5 quote lines, p95
    # of 9, worst observed 49). Idempotent, because lifespan can run more than
    # once in a process.
    try:
        from src.services.formulas import install_db_audit_sink

        install_db_audit_sink()
    except Exception:
        logger.exception("formula audit sink install failed (non-critical)")

    try:
        agent_nick = AgentNick()

        # Auto-discover agents from agent_definitions.json (replaces manual registration)
        from agents.auto_registry import AutoRegistry
        auto_registry = AutoRegistry.from_json()
        auto_registry.set_agent_nick(agent_nick)
        agent_nick.auto_registry = auto_registry

        # Eagerly instantiate core agents for backward compatibility
        # (AutoRegistry lazy-loads, but some code accesses agents dict directly)
        agents_dict = {}
        for agent_id in auto_registry.agent_ids:
            contract = auto_registry.get_contract(agent_id)
            if contract.class_path:
                try:
                    agents_dict[agent_id] = auto_registry.get_agent(agent_id)
                except Exception:
                    logger.exception("Failed to instantiate agent: %s", agent_id)
        agent_nick.agents = AgentRegistry(agents_dict)
        agent_nick.agents.add_aliases({
            "DataExtractionAgent": "data_extraction",
            "SupplierRankingAgent": "supplier_ranking",
            "QuoteEvaluationAgent": "quote_evaluation",
            "QuoteComparisonAgent": "quote_comparison",
            "OpportunityMinerAgent": "opportunity_miner",
            "DiscrepancyDetectionAgent": "discrepancy_detection",
            "EmailDraftingAgent": "email_drafting",
            "EmailDispatchAgent": "email_dispatch",
            "NegotiationAgent": "negotiation",
            "ApprovalsAgent": "approvals",
            "SupplierInteractionAgent": "supplier_interaction",
        })

        # Initialize reasoning engine. Optional: the pattern store is a DDL
        # round-trip and the reasoning engine is not reachable from any request
        # path today (`create_plan` has no callers), so neither is worth the
        # whole system. Nothing outside this module reads
        # `agent_nick.reasoning_engine`.
        from services.pattern_service import PatternService
        from services.procurement_context_service import ProcurementContextService
        from orchestration.reasoning_engine import ReasoningEngine

        def _build_pattern_service():
            service = PatternService(agent_nick)
            service.ensure_table()
            return service

        pattern_service = _optional("the pattern store", _build_pattern_service)
        context_service = _optional(
            "the procurement context service",
            lambda: ProcurementContextService(agent_nick),
        )
        reasoning_engine = _optional(
            "the reasoning engine",
            lambda: ReasoningEngine(
                agent_nick, auto_registry, pattern_service, context_service
            ),
        )
        agent_nick.reasoning_engine = reasoning_engine
        agent_nick.pattern_service = pattern_service

        # Seed initial patterns if table is empty
        if pattern_service is not None:
            def _seed_patterns_if_empty():
                from services.seed_patterns import seed_patterns
                if not pattern_service.get_patterns():
                    seed_patterns(pattern_service)
                    logger.info("Seeded initial procurement patterns")

            _optional("the initial procurement patterns", _seed_patterns_if_empty)

        # === Extraction V3: schema validation (fail-loud on drift) ===
        try:
            from src.services.extraction_v3.yaml_schema.loader import load_all_schemas, SchemaDriftError
            extraction_v3_schemas = load_all_schemas()
            state.extraction_v3_schemas = extraction_v3_schemas
            logger.info(
                "extraction_v3: loaded %d doc-type schemas: %s",
                len(extraction_v3_schemas), list(extraction_v3_schemas.keys()),
            )
        except SchemaDriftError as exc:
            logger.error("extraction_v3 schema drift detected; refusing to start: %s", exc)
            raise  # crash startup loud
        except Exception:
            logger.exception("extraction_v3 schema load failed; continuing without v3 schemas")
            state.extraction_v3_schemas = {}

        # Ensure the agent-workflows schema exists ONCE, at startup — not on
        # every request (that was a DDL round-trip on the hot path of all 8
        # handlers in api/routers/agent_workflows.py).
        try:
            from repositories import agent_workflow_repo as _agent_workflow_repo
            from repositories import agent_group_repo as _agent_group_repo
            from repositories import model_catalogue_repo as _model_catalogue_repo
            from repositories import workflow_input_request_repo as _workflow_input_request_repo
            _agent_workflow_repo.ensure_schema()
            _agent_group_repo.ensure_schema()
            _workflow_input_request_repo.ensure_schema()
            # Seeded once, from what this machine already runs, so the offered-model
            # list agrees with reality rather than announcing a model nothing uses.
            _model_catalogue_repo.ensure_schema()
            logger.info("Agent-workflows schema ensured")
        except Exception:
            logger.exception("Agent-workflows schema init failed (non-critical)")

        # Ensure the supplier-response schema exists. supplier_response_repo has the DDL and
        # calls init_schema() itself, but only from paths reached AFTER the negotiation agent
        # has already queried the table: wait_for_response -> lookup_workflow_for_unique hits
        # proc.supplier_response first and died with UndefinedTable. The exception was caught
        # and logged, so the agent then sat in its "awaiting supplier responses" loop for a
        # reply it could never observe — POST /workflows/negotiate simply never returned.
        try:
            from repositories import supplier_response_repo as _supplier_response_repo
            _supplier_response_repo.init_schema()
            logger.info("Supplier-response schema ensured")
        except Exception:
            logger.exception("Supplier-response schema init failed (non-critical)")

        # Ensure the FX-rates cache schema exists (GET /fx/rates).
        try:
            from repositories import fx_rate_repo as _fx_rate_repo
            _fx_rate_repo.ensure_schema()
            logger.info("FX-rates schema ensured")
        except Exception:
            logger.exception("FX-rates schema init failed (non-critical)")

        # Ensure the egress audit table exists. Non-critical: services.egress_log
        # swallows its own write failures by design, so a missing table costs the
        # audit trail and not the requests it observes — but that is a gap worth
        # seeing in the startup log rather than discovering when someone asks
        # what left the boundary last month.
        try:
            from src.services.egress_log import ensure_schema as _ensure_egress_schema
            _ensure_egress_schema()
            logger.info("Egress audit schema ensured")
        except Exception:
            logger.exception("Egress audit schema init failed (non-critical)")

        # Ensure provenance sidecar schema exists.
        try:
            from services.db import get_conn as _prov_db_get_conn
            from src.services.extraction_v2.provenance import DDL as _PROV_DDL
            with _prov_db_get_conn() as _pconn:
                with _pconn.cursor() as _pcur:
                    _pcur.execute(_PROV_DDL)
                _pconn.commit()
            logger.info("Extraction provenance schema ensured")
        except Exception:
            logger.exception(
                "Extraction provenance schema init failed — "
                "per-field provenance writes will silently no-op"
            )

        state.agent_nick = agent_nick

        # Optional: Orchestrator's own signature defaults training_endpoint to
        # None, so it is built to run without this.
        state.model_training_endpoint = _optional(
            "the model training endpoint", lambda: ModelTrainingEndpoint(agent_nick)
        )

        # Core. Every request routes through the orchestrator; without it there
        # is nothing to degrade to, so a failure here is still fatal.
        orchestrator = Orchestrator(
            agent_nick,
            training_endpoint=state.model_training_endpoint,
        )
        state.orchestrator = orchestrator

        # Optional, and the one that actually fired: RAGPipeline builds a
        # cross-encoder, which OOMs when the GPU is busy. Both routers that use
        # it already raise 503 when it is missing.
        state.rag_pipeline = _optional(
            "the RAG pipeline", lambda: RAGPipeline(agent_nick)
        )

        state.agent_registry = agent_nick.agents
        state.supplier_interaction_agent = agents_dict.get("supplier_interaction")
        state.negotiation_agent = agents_dict.get("negotiation")
        state.email_watcher_runner = run_email_watcher_for_workflow
        backend_scheduler = _optional(
            "the backend scheduler", lambda: orchestrator.backend_scheduler
        )
        state.backend_scheduler = backend_scheduler
        try:
            email_watcher_service = backend_scheduler.get_email_watcher_service()
        except Exception:
            logger.exception("Failed to obtain email watcher service from backend scheduler")
            email_watcher_service = None
        state.email_watcher_service = email_watcher_service
        state.email_watcher_owned = False
        try:
            process_monitor_watcher = backend_scheduler.get_process_monitor_watcher()
        except Exception:
            logger.exception("Failed to obtain process monitor watcher from backend scheduler")
            process_monitor_watcher = None
        state.process_monitor_watcher = process_monitor_watcher

        # Start the session-status LISTEN → WebSocket bridge.
        # Uses the current asyncio event loop so the background thread can
        # schedule ws_manager coroutines safely.
        try:
            from services.session_notify_listener import SessionNotifyListener
            _event_loop = asyncio.get_event_loop()
            session_notify_listener = SessionNotifyListener(_event_loop)
            session_notify_listener.start()
            state.session_notify_listener = session_notify_listener
            logger.info("SessionNotifyListener started")
        except Exception:
            logger.exception("Failed to start SessionNotifyListener")
            state.session_notify_listener = None

        # Warm the extraction-hint cache so approved per-vendor hints are live
        # from the first document (extraction feedback loop).
        try:
            from src.services.extraction_feedback.hint_store import HINT_STORE
            logger.info("ExtractionHintStore: %d active hints loaded", HINT_STORE.refresh())
        except Exception:
            logger.exception("ExtractionHintStore init failed (non-critical)")

        # ------------------------------------------------------------------
        # These two MUST live here, inside lifespan.
        #
        # They were written as @app.on_event("startup") handlers, and this app is constructed
        # with lifespan=... — which makes FastAPI ignore on_event entirely. So both ran
        # exactly never, in silence, and the only way to notice was that a log line you
        # expected was not there.
        # ------------------------------------------------------------------

        # Teach the output-safety gate our real route table, so it can recognise one of our
        # own endpoints being quoted back at a user. Without this it falls back to pattern
        # matching alone — which does still catch leaks, but it is the weaker half.
        try:
            from services import output_safety as _osafe

            _osafe.register_routes(
                [getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")]
            )
        except Exception:  # noqa: BLE001
            logger.warning("output-safety route registration failed", exc_info=True)

        # Push the platform's model of itself into the graph. This had NO programmatic caller
        # at all: sync() ran only from its own __main__, so the ontology could be edited and
        # nothing would propagate — the assistant went on answering from whatever had last
        # been synced by hand, and nothing said so. MERGE-on-id, so it is idempotent per boot.
        # A missing platform description degrades answers; it must not stop documents being
        # processed, so this never raises.
        try:
            from services.platform_kg import sync as _sync_ontology

            counts = await run_in_threadpool(_sync_ontology)
            logger.info("platform ontology synced to the knowledge graph: %s", counts)
        except Exception:  # noqa: BLE001
            logger.warning(
                "could not sync the platform ontology; the assistant will answer from "
                "whatever was last synced", exc_info=True,
            )

        logger.info("System initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: System initialization failed: {e}", exc_info=True)
        state.agent_nick = None
        state.model_training_endpoint = None
        state.orchestrator = None
        state.rag_pipeline = None
        state.agent_registry = None
        state.supplier_interaction_agent = None
        state.negotiation_agent = None
        state.email_watcher_runner = None
        state.email_watcher_service = None
        state.email_watcher_owned = False
        state.backend_scheduler = None
        state.process_monitor_watcher = None
        state.extraction_v3_schemas = {}
        state.session_notify_listener = None
    yield
    if hasattr(state, "agent_nick"):
        state.agent_nick = None
    if hasattr(state, "email_watcher_runner"):
        state.email_watcher_runner = None
    if hasattr(state, "email_watcher_service"):
        service = state.email_watcher_service
        owned = getattr(state, "email_watcher_owned", True)
        if service and owned:
            try:
                service.stop()
            except Exception:  # pragma: no cover - defensive shutdown
                logger.exception("Failed to stop EmailWatcherService during shutdown")
        state.email_watcher_service = None
    if hasattr(state, "process_monitor_watcher"):
        state.process_monitor_watcher = None
    if hasattr(state, "session_notify_listener"):
        listener = state.session_notify_listener
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                logger.exception("Failed to stop SessionNotifyListener during shutdown")
        state.session_notify_listener = None
    if hasattr(state, "email_watcher_owned"):
        state.email_watcher_owned = False
    if hasattr(state, "backend_scheduler"):
        state.backend_scheduler = None
    if hasattr(state, "supplier_interaction_agent"):
        state.supplier_interaction_agent = None
    if hasattr(state, "negotiation_agent"):
        state.negotiation_agent = None
    if hasattr(state, "agent_registry"):
        state.agent_registry = None
    if hasattr(state, "model_training_endpoint"):
        state.model_training_endpoint = None

    # Release CUDA memory before exiting so the next systemd-restarted
    # uvicorn process can claim the VRAM without an OutOfMemory at boot.
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(
                "CUDA cache released on shutdown (allocated=%.2fGiB reserved=%.2fGiB)",
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
            )
    except Exception:
        logger.debug("CUDA cleanup on shutdown skipped", exc_info=True)

    logger.info("API shutting down.")

app = FastAPI(title="ProcWise API v4 (Definitive)", version="4.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("PROCWISE_CORS_ORIGINS", "*").split(",") if o.strip()]
_allow_creds = _origins != ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=_allow_creds, allow_methods=["*"], allow_headers=["*"])

# WebSocket router: no auth DEPENDENCY, because a dependency that raises
# HTTPException cannot answer a WebSocket upgrade. Browsers also cannot set
# headers on that upgrade, so the token arrives as ?token= and ws.py verifies it
# in the handler, through the same api.auth verifier and the same ASK_AUTH_MODE
# semantics as require_user, before accepting the handshake.
#
# This comment previously described that arrangement as already existing. It did
# not: the handler took session_id and websocket, and nothing else. Anyone who
# could reach the port and knew a session id was served its outcomes.
# tests/api/test_ws_authentication.py is what now holds this to be true.
app.include_router(ws_router_mod.router)

# --------------------------------------------------------------------------
# Every HTTP router below is mounted WITH authentication.
#
# Four of thirty-four carried a principal before this: approvals, decisions,
# value_summary and workflows. The other thirty answered anyone who could reach
# the port — /benchmark, /opportunities, /documents, /analysis, /summary, and
# /agents/reason, which hands AgentNick a task and exposes every registered
# agent to it as a callable tool.
#
# The dependency is applied HERE rather than in each router on purpose. A
# per-router decoration is a control a new router forgets; a list every router
# must appear in to be served is one a new router cannot be added without
# touching. tests/api/test_every_router_is_authenticated.py asserts the two
# stay in step, so adding a router without a deliberate decision fails a test
# rather than quietly opening an endpoint.
#
# require_user honours ASK_AUTH_MODE: `enforce` (the default, and what .env
# sets) verifies a Cognito ID token; `off` returns None and admits everyone,
# logging a warning at startup; a misconfigured `enforce` refuses with 503
# rather than falling open.
#
# The UI already sends the token: services/api.js sets
# axios.defaults.headers.common["Authorization"] at login, and the SpendIQ
# views that call this service directly on VITE_AI_API_URL rely on that same
# default (see the comment at useSpendData.js:8). So this is enforcement of a
# credential that is already on the wire, not a new requirement on callers.
# --------------------------------------------------------------------------
_AUTHENTICATED_ROUTERS = [
    agents_router_mod.router,
    documents.router,
    email.router,
    workflows.router,
    system.router,
    run.router,
    stream.router,
    training.router,
    vendors.build_router(),
    metrics.router,
    extraction_feedback_router.router,
    decisions_router.router,
    agent_workflows_router.router,
    agent_groups_router.router,
    models_router.router,
    support_router.router,
    supplier_review_router.router,
    supplier_research_router.router,
    governance_router.router,
    deal_summary.router,
    deal_proposals.router,
    negotiate.router,
    opportunities.router,
    promotion.router,
    summary.router,
    session.router,
    obligations_router.router,
    benchmark_router.router,
    requirements_router.router,
    fx_router.router,
    value_summary_router.router,
    analysis_router.router,
    approvals_router.router,
]

for _router in _AUTHENTICATED_ROUTERS:
    app.include_router(_router, dependencies=[Depends(_ask_auth.require_user)])
import src.services.requirement_similarity  # noqa: F401 — registers the quote_rival profile at startup


# ======================================================================================
# The output-safety boundary.
#
# Layer A (services/tool_runtime._gate) gives the agent a chance to re-frame an answer that
# described the machine instead of the product. This is Layer B, and it does not negotiate:
# it is the last thing every response passes through before it becomes bytes on a socket.
#
# It exists because Layer A can be bypassed. Forty-six call sites raise `HTTPException(...,
# detail=str(exc))`, and a psycopg2 error message *is* a schema disclosure — it names the
# table and the column. None of those go anywhere near the agent loop. A guarantee that only
# holds on the paths someone remembered to route through it is not a guarantee.
# ======================================================================================

from fastapi.exception_handlers import http_exception_handler  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402

from services import output_safety as osafe  # noqa: E402

# Endpoints whose whole job is to describe the machine to an operator. They are not user
# surfaces, and scrubbing them would leave nothing behind. They must not be reachable by an
# end user — see the note in the security spec.
# /models names models on purpose: it is the list an operator picks from, and a
# scrubbed list reads "[withheld], [withheld], [withheld]" and cannot be chosen
# from at all. It returns display names and keys only — never the provider-facing
# reference — so exempting it leaks nothing the operator did not already decide.
_OPERATOR_PATHS = ("/docs", "/redoc", "/openapi.json", "/models")


@app.exception_handler(StarletteHTTPException)
async def _safe_http_exception(request: Request, exc: StarletteHTTPException):
    """`detail=str(exc)` is the single commonest leak in this codebase.

    The operator still gets the real thing — in the log, where they would actually read it.
    The user gets a sentence.
    """
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    safe = osafe.enforce(detail, where=f"HTTP {exc.status_code} {request.url.path}")
    if safe != detail:
        logger.warning(
            "output_safety: %s %s detail withheld from client: %s",
            exc.status_code, request.url.path, detail[:300],
        )
    return await http_exception_handler(
        request, StarletteHTTPException(exc.status_code, safe, headers=exc.headers)
    )


@app.exception_handler(Exception)
async def _safe_unhandled(request: Request, exc: Exception):
    """An unhandled exception must never become a description of the internals."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": osafe.SAFE_REPLY})


class OutputSafetyMiddleware(BaseHTTPMiddleware):
    """Every JSON body and every SSE frame, scrubbed on the way out."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        path = request.url.path
        if any(path.startswith(p) for p in _OPERATOR_PATHS):
            return response
        ctype = response.headers.get("content-type", "")

        if ctype.startswith("text/event-stream"):
            return self._guard_stream(response, path)
        if not ctype.startswith("application/json"):
            return response

        raw = b"".join([chunk async for chunk in response.body_iterator])
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            return Response(
                content=raw, status_code=response.status_code,
                headers=dict(response.headers), media_type=ctype,
            )

        safe = osafe.scrub_payload(body, where=path)
        out = json.dumps(safe, default=str).encode()
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=out, status_code=response.status_code,
            headers=headers, media_type=ctype,
        )

    @staticmethod
    def _guard_stream(response, path: str):
        """SSE frames are already whole events by the time they reach here.

        The agent loop buffers its answer before emitting it (a token cannot be un-sent), so
        a frame arriving here is a complete thought, not half an identifier. That makes it
        safe to scan one frame at a time without holding the whole stream.
        """
        async def _gen():
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                out = []
                for line in text.split("\n"):
                    if line.startswith("data: "):
                        try:
                            evt = json.loads(line[6:])
                        except Exception:  # noqa: BLE001
                            out.append(
                                "data: " + json.dumps(
                                    {"type": "error", "message": osafe.SAFE_REPLY}
                                )
                            )
                            continue
                        if isinstance(evt, dict):
                            evt = osafe.scrub_payload(evt, where=f"sse {path}")
                        out.append("data: " + json.dumps(evt, default=str))
                    else:
                        out.append(line)
                yield "\n".join(out).encode()

        headers = dict(response.headers)
        headers.pop("content-length", None)
        return StreamingResponse(
            _gen(), status_code=response.status_code,
            headers=headers, media_type="text/event-stream",
        )


# Authentication mode. `_ask_auth` itself is imported near the top, because the
# router mounting above depends on require_user; only configure() runs here, so
# the mode is decided once at import, before the first request, and is surfaced
# on /health — auth being off must be visible, not something you discover by
# testing an endpoint.
from config.settings import settings as _settings  # noqa: E402

_ASK_AUTH_MODE = _ask_auth.configure(_settings)

app.add_middleware(OutputSafetyMiddleware)


# NOTE: do not add @app.on_event("startup") handlers to this app. It is constructed with
# lifespan=..., which makes FastAPI ignore on_event entirely — a handler added here runs
# never, and says nothing about it. Two of them did exactly that. Startup work belongs in
# `lifespan` above.


@app.get("/", tags=["General"])
def read_root(): return {"message": "Welcome to the ProcWise Agentic System API"}


def _shadow_status():
    """Shadow-mode enrolments, or why they could not be read.

    Never raises: /health must answer even when the policy store cannot.
    """

    try:
        from services import guardrail

        return guardrail.shadow_status()
    except Exception as exc:  # noqa: BLE001
        logger.error("shadow status unavailable: %s", exc)
        return {"error": "unavailable"}


def _critic_shadow_status():
    """The Opportunity Critic's shadow enrolments, or why they could not be read.

    Never raises. health_status reports "unavailable" rather than an empty list
    when the policy cannot be resolved -- an outage must not read as "nothing
    is enrolled".
    """

    try:
        from src.services.opportunity_critic.shadow import health_status

        engine = getattr(getattr(app.state, "agent_nick", None), "policy_engine", None)
        return health_status(engine)
    except Exception as exc:  # noqa: BLE001
        logger.error("critic shadow status unavailable: %s", exc)
        return {"error": "unavailable"}


@app.get("/health", tags=["General"])
def health():
    state = app.state
    initialized = bool(getattr(state, "agent_nick", None))
    schemas = getattr(state, "extraction_v3_schemas", {})
    from services.capability_status import get_degraded

    return {
        "status": "ok" if initialized else "starting",
        "agent_nick": initialized,
        "orchestrator": bool(getattr(state, "orchestrator", None)),
        "email_watcher_service": bool(getattr(state, "email_watcher_service", None)),
        "process_monitor_watcher": bool(getattr(state, "process_monitor_watcher", None)),
        "extraction_v3": {
            "schemas_loaded": len(schemas),
            "doc_types": sorted(schemas.keys()) if schemas else [],
        },
        # Whether the ask endpoints require a token. Reported because "off" is a
        # deployment state someone must be able to see without probing the API.
        # No pool ids or issuer here — this endpoint needs no auth itself.
        "ask_auth": _ask_auth.auth_mode(),
        # Which actions are being observed rather than enforced, and until when.
        # Reported for the same reason ask_auth is: a control that is off is a
        # deployment state someone must be able to see without reading code.
        # Action names only — no policy content, no principals.
        "shadow_mode": _shadow_status(),
        # The Opportunity Critic's per-detector shadow enrolments, with expiries.
        # "unavailable" when its policy cannot be read -- never an empty list.
        "critic_shadow": _critic_shadow_status(),
        # Honest surface for features that lost a dependency they can never have. It stays
        # honest — the capability is still named, and it still says it is degraded — but the
        # *reason* no longer ships. It used to read "proc.agent table does not exist; …
        # falling back to agent_definitions.json", which is a schema disclosure and an
        # internal filename on an endpoint that needs no auth at all. The reason is logged
        # by capability_status.mark_degraded the moment it happens, which is where an
        # operator would look for it anyway.
        "degraded": [
            {"capability": d["capability"], "status": "degraded"}
            for d in get_degraded()
        ],
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
