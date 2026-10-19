"""The orchestrator's side of the data-request contract (spec section C) and the agent registry (section D)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

from .context import CallerContext
from .envelope import GatewayError, GatewayResult
from .gateway import Gateway

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config", "gateway")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    max_data_requests_per_step: int
    max_steps: int
    max_retries: int
    prefetch: str = ""
    purpose: str = ""


class AgentRegistry:
    def __init__(self, cfg: Dict[str, Any], *, known_functions: Iterable[str]) -> None:
        known: Set[str] = set(known_functions)
        defaults = cfg.get("defaults", {}) or {}
        self.version = int(cfg.get("version", 0))
        self.agents: Dict[str, AgentSpec] = {}
        for name, a in (cfg.get("agents") or {}).items():
            reads, writes = tuple(a.get("reads") or ()), tuple(a.get("writes") or ())
            for fn in reads + writes:
                if fn not in known:
                    raise ValueError(f"agent {name!r} allowlists {fn!r}, which is not in the function catalogue")
            both = set(reads) & set(writes)
            if both:
                raise ValueError(f"agent {name!r} lists {sorted(both)} as both read and write")
            self.agents[name] = AgentSpec(
                name=name, reads=reads, writes=writes, prefetch=a.get("prefetch", ""), purpose=a.get("purpose", ""),
                max_data_requests_per_step=int(a.get("max_data_requests_per_step", defaults.get("max_data_requests_per_step", 3))),
                max_steps=int(a.get("max_steps", defaults.get("max_steps", 4))),
                max_retries=int(a.get("max_retries", defaults.get("max_retries", 1))))


def load_agent_registry(path: Optional[str] = None, *, known_functions: Iterable[str]) -> AgentRegistry:
    with open(path or os.path.join(CONFIG_DIR, "agents.yaml"), encoding="utf-8") as fh:
        return AgentRegistry(yaml.safe_load(fh), known_functions=known_functions)


class DataRequestBroker:
    """Validates an agent's data request, attaches the orchestrator's own caller context, calls the gateway, logs it.

    One broker per agent step. A rejected or over-limit request comes back to the agent as a structured error.
    """

    def __init__(self, gateway: Gateway, registry: AgentRegistry) -> None:
        self.gateway, self.registry = gateway, registry
        self._used = 0
        self._cache: Dict[Tuple[str, str], GatewayResult] = {}

    def new_step(self) -> None:
        self._used, self._cache = 0, {}

    def handle(self, ctx: CallerContext, request: Dict[str, Any]) -> GatewayResult:
        fn, params = str(request.get("function", "")), dict(request.get("params") or {})

        def refuse(code: str, message: str) -> GatewayResult:
            return self.gateway.reject(ctx, fn, params, GatewayError(code, message))

        agent = self.registry.agents.get(ctx.agent)
        if agent is None:
            return refuse("NOT_ALLOWED", f"agent {ctx.agent!r} is not in the registry")
        if request.get("agent") != ctx.agent:
            return refuse("NOT_ALLOWED", "the request names a different agent than the one running this step")
        if fn not in agent.reads:
            return refuse("NOT_ALLOWED", f"{fn!r} is not on {ctx.agent}'s read allowlist")
        key = (fn, json.dumps(params, sort_keys=True, default=str))
        if key in self._cache:
            return self._cache[key]
        if self._used >= agent.max_data_requests_per_step:
            return refuse("LIMIT_EXCEEDED", f"{agent.max_data_requests_per_step} data requests per step already used; answer with what you have")
        self._used += 1
        result = self.gateway.call(ctx, fn, params)
        if result.ok:
            self._cache[key] = result
        return result
