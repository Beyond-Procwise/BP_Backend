import sys, os, uvicorn, logging
from contextlib import asynccontextmanager
from typing import Any, Optional, Protocol, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure GPU utilisation by default on compatible hardware
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

# Force HuggingFace libraries to use local cached models only - no HTTP calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
from api.routers import agents as agents_router_mod, documents, email, run, stream, system, training, workflows

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "procwise.log"))])
logger = logging.getLogger(__name__)


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
                    logger.warning("Failed to instantiate agent: %s", agent_id)
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
        pattern_service.ensure_table()
        context_service = ProcurementContextService(agent_nick)
        reasoning_engine = ReasoningEngine(
            agent_nick, auto_registry, pattern_service, context_service
        )
        agent_nick.reasoning_engine = reasoning_engine
        agent_nick.pattern_service = pattern_service

        # Seed initial patterns if table is empty
        from services.seed_patterns import seed_patterns
        existing = pattern_service.get_patterns(limit=1)
        if not existing:
            seed_patterns(pattern_service)
            logger.info("Seeded initial procurement patterns")
        state.agent_nick = agent_nick
        state.model_training_endpoint = ModelTrainingEndpoint(agent_nick)
        orchestrator = Orchestrator(
            agent_nick,
            training_endpoint=state.model_training_endpoint,
        )
        state.orchestrator = orchestrator
        state.rag_pipeline = RAGPipeline(agent_nick)
        state.agent_registry = agent_nick.agents
        state.supplier_interaction_agent = supplier_interaction_agent
        state.negotiation_agent = negotiation_agent
        state.email_watcher_runner = run_email_watcher_for_workflow
        backend_scheduler = orchestrator.backend_scheduler
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
    logger.info("API shutting down.")

app = FastAPI(title="ProcWise API v4 (Definitive)", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(agents_router_mod.router)
app.include_router(documents.router)
app.include_router(email.router)
app.include_router(workflows.router)
app.include_router(system.router)
app.include_router(run.router)
app.include_router(stream.router)
app.include_router(training.router)

@app.get("/", tags=["General"])
def read_root(): return {"message": "Welcome to the ProcWise Agentic System API"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
