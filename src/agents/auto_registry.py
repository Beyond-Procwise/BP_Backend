"""Declarative agent discovery and lazy instantiation from agent_definitions.json.

AutoRegistry reads the canonical agent_definitions.json at the project root and
provides a single source of truth for agent metadata, capability lookup, and
on-demand instantiation.  Agents are never imported until first requested, keeping
startup cost near zero.

Usage::

    registry = AutoRegistry.from_json()
    contract = registry.get_contract("supplier_ranking")
    agents_with_email = registry.find_by_capability("email_drafting")
    agent = registry.get_agent("data_extraction")   # lazily instantiated
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default location: two levels up from this file (project root)
_DEFAULT_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "agent_definitions.json"


@dataclass
class AgentContract:
    """Lightweight contract derived from the JSON definition.

    Unlike the heavy ``AgentContract`` in ``agent_interface.py`` (which uses
    frozen sets and enum types), this dataclass uses plain Python lists and
    strings so that it can be populated directly from JSON without any agent
    imports.
    """

    id: str
    class_path: Optional[str]
    capabilities: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    output_fields: List[str] = field(default_factory=list)
    description: str = ""


class AutoRegistry:
    """Declarative agent registry backed by agent_definitions.json.

    Agents are discovered from JSON at construction time.  Class imports and
    instance creation are deferred until :meth:`get_agent` is called for the
    first time for a given agent ID.

    Parameters
    ----------
    contracts:
        Mapping of agent slug -> :class:`AgentContract`.
    agent_nick:
        Optional dependency object forwarded to agent constructors.  Must be
        set via :meth:`set_agent_nick` before calling :meth:`get_agent`.
    """

    def __init__(
        self,
        contracts: Dict[str, AgentContract],
        agent_nick: Any = None,
    ) -> None:
        self._contracts: Dict[str, AgentContract] = contracts
        self._instances: Dict[str, Any] = {}
        self._agent_nick: Any = agent_nick

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Optional[str] = None) -> "AutoRegistry":
        """Load registry from a JSON file.

        Parameters
        ----------
        path:
            Absolute or relative path to the definitions file.  Defaults to
            the ``agent_definitions.json`` at the project root.
        """
        json_path = Path(path) if path else _DEFAULT_JSON_PATH
        if not json_path.exists():
            raise FileNotFoundError(
                f"agent_definitions.json not found at {json_path}. "
                "Ensure the project root contains the file."
            )

        with json_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)

        # Support both the legacy flat-array format and the new object format
        if isinstance(raw, list):
            agent_list = raw
        elif isinstance(raw, dict) and "agents" in raw:
            agent_list = raw["agents"]
        else:
            raise ValueError(
                "agent_definitions.json must be either a JSON array or an object "
                "with an 'agents' key."
            )

        contracts: Dict[str, AgentContract] = {}
        for entry in agent_list:
            slug = entry.get("slug") or entry.get("agentType", "")
            if not slug:
                logger.warning("Skipping agent definition without a slug: %s", entry)
                continue

            contract = AgentContract(
                id=slug,
                class_path=entry.get("class_path"),
                capabilities=list(entry.get("capabilities", [])),
                required_inputs=list(entry.get("required_inputs", entry.get("inputs", {}).get("required", []))),
                output_fields=list(entry.get("output_fields", entry.get("outputs", []))),
                description=entry.get("description", ""),
            )
            contracts[slug] = contract
            logger.debug("Registered agent contract: %s", slug)

        logger.info("AutoRegistry loaded %d agent definitions from %s", len(contracts), json_path)
        return cls(contracts)

    # ------------------------------------------------------------------
    # Dependency injection
    # ------------------------------------------------------------------

    def set_agent_nick(self, agent_nick: Any) -> None:
        """Set the dependency object forwarded to agent constructors.

        Must be called before :meth:`get_agent` when agents require
        constructor arguments.
        """
        self._agent_nick = agent_nick

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @property
    def agent_ids(self) -> List[str]:
        """Return a sorted list of all registered agent IDs (slugs)."""
        return sorted(self._contracts.keys())

    def get_contract(self, agent_id: str) -> AgentContract:
        """Return the :class:`AgentContract` for the given agent ID.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        """
        try:
            return self._contracts[agent_id]
        except KeyError:
            raise KeyError(
                f"No agent registered with id '{agent_id}'. "
                f"Available: {self.agent_ids}"
            )

    def find_by_capability(self, capability: str) -> List[str]:
        """Return agent IDs whose capability list contains *capability*.

        Parameters
        ----------
        capability:
            Capability string to search for (e.g. ``"email_drafting"``).
        """
        return [
            slug
            for slug, contract in self._contracts.items()
            if capability in contract.capabilities
        ]

    def all_capabilities(self) -> Set[str]:
        """Return the union of all capabilities declared across all agents."""
        caps: Set[str] = set()
        for contract in self._contracts.values():
            caps.update(contract.capabilities)
        return caps

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def is_instantiated(self, agent_id: str) -> bool:
        """Return True if *agent_id* has already been instantiated and cached."""
        return agent_id in self._instances

    def get_agent(self, agent_id: str) -> Any:
        """Return the agent instance for *agent_id*, creating it if needed.

        Instantiation is lazy: the agent class is imported and constructed
        only on the first call.  Subsequent calls return the cached instance.

        Parameters
        ----------
        agent_id:
            The slug identifying the agent (e.g. ``"data_extraction"``).

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        ValueError
            If the contract has no ``class_path`` (e.g. ``email_watcher``
            which is defined inline in services).
        ImportError
            If the module or class cannot be imported.
        """
        if agent_id in self._instances:
            return self._instances[agent_id]

        contract = self.get_contract(agent_id)  # raises KeyError if unknown

        if not contract.class_path:
            raise ValueError(
                f"Agent '{agent_id}' has no class_path defined and cannot be "
                "instantiated via AutoRegistry.  It is defined inline in services."
            )

        module_path, class_name = self._split_class_path(contract.class_path)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import module '{module_path}' for agent '{agent_id}': {exc}"
            ) from exc

        try:
            agent_class = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Module '{module_path}' has no class '{class_name}' "
                f"for agent '{agent_id}': {exc}"
            ) from exc

        try:
            instance = agent_class(self._agent_nick)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate {class_name} for agent '{agent_id}': {exc}"
            ) from exc

        self._instances[agent_id] = instance
        logger.info("Instantiated agent '%s' from %s", agent_id, contract.class_path)
        return instance

    # ------------------------------------------------------------------
    # LLM description
    # ------------------------------------------------------------------

    def describe_for_llm(self) -> str:
        """Return a formatted string describing all agents for the reasoning LLM.

        The output is designed to be injected directly into a system prompt so
        the LLM can reason about which agents to invoke for a given task.
        """
        lines: List[str] = [
            "# Available Procurement Agents",
            "",
            "The following agents are available in the ProcWise system.  "
            "Each entry lists the agent's ID, description, capabilities, "
            "required inputs, and produced outputs.",
            "",
        ]

        for slug in self.agent_ids:
            contract = self._contracts[slug]
            caps = ", ".join(contract.capabilities) if contract.capabilities else "none"
            req = ", ".join(contract.required_inputs) if contract.required_inputs else "none"
            out = ", ".join(contract.output_fields) if contract.output_fields else "none"
            instantiable = "yes" if contract.class_path else "no (inline service)"

            lines += [
                f"## {slug}",
                f"  Description : {contract.description}",
                f"  Capabilities: {caps}",
                f"  Inputs (req) : {req}",
                f"  Outputs     : {out}",
                f"  Instantiable: {instantiable}",
                "",
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_class_path(class_path: str):
        """Split ``'module.path.ClassName'`` into ``('module.path', 'ClassName')``."""
        if "." not in class_path:
            raise ValueError(
                f"class_path '{class_path}' must be in the form "
                "'module.path.ClassName'."
            )
        module_path, class_name = class_path.rsplit(".", 1)
        return module_path, class_name

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AutoRegistry(agents={len(self._contracts)}, "
            f"instantiated={len(self._instances)})"
        )
