import sys, os, uvicorn, logging
from contextlib import asynccontextmanager
from typing import Any, Optional, Protocol, cast

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.auth import verify_api_key

# Ensure GPU utilisation by default on compatible hardware
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
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
from api.routers import agents as agents_router_mod, documents, email, metrics, run, stream, system, training, vendors, workflows, deal_summary, promotion, summary, negotiate, opportunities, requirements

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    state = cast(ProcwiseAppState, app.state)
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

        # Initialize reasoning engine
        from services.pattern_service import PatternService
        from services.procurement_context_service import ProcurementContextService
        from orchestration.reasoning_engine import ReasoningEngine

        pattern_service = PatternService(agent_nick)
        try:
            pattern_service.ensure_table()
        except Exception:
            logger.exception("pattern_service.ensure_table failed (DB unreachable?) "
                             "— continuing; pattern features degraded")
        context_service = ProcurementContextService(agent_nick)
        reasoning_engine = ReasoningEngine(
            agent_nick, auto_registry, pattern_service, context_service
        )
        agent_nick.reasoning_engine = reasoning_engine
        agent_nick.pattern_service = pattern_service
        # Make AgentNick available to DB-independent endpoints (e.g. /agents/instruct
        # reasoning, which uses Ollama, not the DB) as early as possible — BEFORE the
        # DB-dependent init below, which degrades gracefully when the DB is unreachable.
        state.agent_nick = agent_nick

        # One-shot DB reachability probe (fast-fail). When the DB is DOWN we SKIP the
        # DB-coupled subsystems below (pattern seeding, provenance, orchestrator,
        # scheduler, email/process watchers) — several of them RETRY-LOOP on the DB
        # and would otherwise BLOCK startup indefinitely (the server never binds its
        # port). AgentNick + reasoning (/agents/instruct) still serve. Full features
        # auto-recover on the next restart once the database is back.
        db_reachable = False
        try:
            with agent_nick.get_db_connection() as _probe:
                db_reachable = True
            logger.info("DB reachability probe: OK")
        except Exception:
            logger.error(
                "DB REACHABILITY PROBE FAILED — starting in REDUCED mode: AgentNick + "
                "reasoning (/agents/instruct) available; extraction, scheduling, watchers "
                "and persistence DISABLED until the database is restored.")

        # Pin the planning model (AgentNick:unified) resident on a background
        # thread so the first /agents/instruct call doesn't pay a ~2 min
        # cold-load. Non-blocking: boot proceeds while the model warms.
        import threading
        threading.Thread(
            target=reasoning_engine.warm_up, name="planner-warmup", daemon=True
        ).start()

        # Seed initial patterns if table is empty (DB-dependent — skip when down)
        if db_reachable:
            try:
                from services.seed_patterns import seed_patterns
                existing = pattern_service.get_patterns()
                if not existing:
                    seed_patterns(pattern_service)
                    logger.info("Seeded initial procurement patterns")
            except Exception:
                logger.exception("pattern seeding failed — continuing")

        # === Extraction V3: schema validation (fail-loud on drift) ===
        # The validator verifies each schema against the DB; skip when the DB is
        # down (extraction is disabled in reduced mode anyway) — otherwise it
        # hangs on its own un-timed DB connect.
        if db_reachable:
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
        else:
            state.extraction_v3_schemas = {}

        # Ensure provenance sidecar schema exists (DB-dependent — skip when down).
        if db_reachable:
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

        state.agent_registry = agent_nick.agents
        state.supplier_interaction_agent = agents_dict.get("supplier_interaction")
        state.negotiation_agent = agents_dict.get("negotiation")
        state.email_watcher_runner = run_email_watcher_for_workflow
        # Defaults for DB-dependent components (overwritten on success below; they
        # remain None in REDUCED mode when the database is unreachable).
        state.model_training_endpoint = None
        state.orchestrator = None
        state.rag_pipeline = None
        state.backend_scheduler = None
        state.email_watcher_service = None
        state.email_watcher_owned = False
        state.process_monitor_watcher = None
        # Orchestrator + scheduler + watchers are DB-dependent; when the DB is
        # unreachable they degrade (no extraction/scheduling) but must NOT abort
        # startup — AgentNick + reasoning are already available above.
        try:
            if not db_reachable:
                # Skip the DB-coupled subsystems entirely: the scheduler/email-watcher
                # retry-loop on the DB and would block startup forever.
                raise RuntimeError("DB unreachable — skipping orchestrator/scheduler/watchers")
            state.model_training_endpoint = ModelTrainingEndpoint(agent_nick)
            orchestrator = Orchestrator(
                agent_nick,
                training_endpoint=state.model_training_endpoint,
            )
            state.orchestrator = orchestrator
            state.rag_pipeline = RAGPipeline(agent_nick)
            backend_scheduler = orchestrator.backend_scheduler
            state.backend_scheduler = backend_scheduler
            try:
                state.email_watcher_service = backend_scheduler.get_email_watcher_service()
            except Exception:
                logger.exception("Failed to obtain email watcher service from backend scheduler")
                state.email_watcher_service = None
            try:
                state.process_monitor_watcher = backend_scheduler.get_process_monitor_watcher()
            except Exception:
                logger.exception("Failed to obtain process monitor watcher from backend scheduler")
                state.process_monitor_watcher = None
            logger.info("System initialized successfully.")
        except Exception:
            logger.exception(
                "Orchestrator/scheduler init failed (DATABASE UNREACHABLE?) — server "
                "running in REDUCED mode: AgentNick + reasoning (/agents/instruct) "
                "available; extraction, scheduling and persistence degraded until the "
                "database is restored.")
    except Exception as e:
        logger.critical(f"FATAL: System initialization failed: {e}", exc_info=True)
        # Preserve AgentNick/reasoning if already attached — a late DB-dependent
        # failure must not take down DB-independent endpoints (/agents/instruct).
        if not getattr(state, "agent_nick", None):
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

app = FastAPI(title="ProcWise API v4 (Definitive)", version="4.0", lifespan=lifespan, dependencies=[Depends(verify_api_key)])

_origins = [o.strip() for o in os.getenv("PROCWISE_CORS_ORIGINS", "*").split(",") if o.strip()]
_allow_creds = _origins != ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=_allow_creds, allow_methods=["*"], allow_headers=["*"])

app.include_router(agents_router_mod.router)
app.include_router(documents.router)
app.include_router(email.router)
app.include_router(workflows.router)
app.include_router(system.router)
app.include_router(run.router)
app.include_router(stream.router)
app.include_router(training.router)
app.include_router(vendors.build_router())
app.include_router(metrics.router)
app.include_router(deal_summary.router)
app.include_router(negotiate.router)
app.include_router(opportunities.router)
app.include_router(promotion.router)
app.include_router(summary.router)
app.include_router(requirements.router)

@app.get("/", tags=["General"])
def read_root(): return {"message": "Welcome to the ProcWise Agentic System API"}


@app.get("/health", tags=["General"])
def health():
    state = app.state
    initialized = bool(getattr(state, "agent_nick", None))
    schemas = getattr(state, "extraction_v3_schemas", {})
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
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
