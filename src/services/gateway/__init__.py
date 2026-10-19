"""The in-process data gateway (specs/2026-10-19-gateway-step2-contracts-and-registries.md)."""
from .audit import AuditSink
from .broker import AgentRegistry, AgentSpec, DataRequestBroker, load_agent_registry
from .context import CONTEXT_KEYS, CallerContext
from .envelope import GatewayError, GatewayFault, GatewayResult, Raw
from .gateway import Backends, FunctionSpec, Gateway


def build_default_gateway(backends: "Backends | None" = None, audit: "AuditSink | None" = None) -> Gateway:
    """The shipped catalogue, with its row limits read from config/gateway/functions.yaml."""
    from .functions import build_catalogue
    return Gateway(build_catalogue(), audit=audit, backends=backends)


__all__ = ["AgentRegistry", "AgentSpec", "AuditSink", "Backends", "CONTEXT_KEYS", "CallerContext", "DataRequestBroker", "FunctionSpec",
           "Gateway", "GatewayError", "GatewayFault", "GatewayResult", "Raw", "build_default_gateway", "load_agent_registry"]
