"""Canonical access to agent_definitions.json.

There were two copies of this file on disk — one at the repo root and one in
``src/`` — and different modules read different ones. ``AutoRegistry`` (which
startup uses to build the live registry) resolved the root copy; the
orchestrator, the manifest service, the routing service and the workflows
router all resolved the ``src/`` copy. The two were byte-identical, so nothing
had broken yet, but an edit to either would have been invisible to half the
system: registering an agent in one copy would leave the other half of the
process believing it did not exist.

One file, one loader. Everything that needs the catalogue comes through here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# src/agents/definitions.py -> src/agents -> src -> <repo root>
DEFINITIONS_PATH: Path = Path(__file__).resolve().parents[2] / "agent_definitions.json"


def load_agent_definitions(
    path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Return the agent catalogue as a list of definition dicts.

    The file is an ``{"agents": [...]}`` envelope. A bare top-level list is also
    accepted: older copies of the catalogue used that shape and some callers
    still pass hand-built fixtures in it.
    """
    target = Path(path) if path else DEFINITIONS_PATH
    if not target.exists():
        raise FileNotFoundError(f"agent_definitions.json not found at {target}")

    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)

    agents = data.get("agents", []) if isinstance(data, dict) else data
    if not isinstance(agents, list):
        raise ValueError(
            "agent_definitions.json must be either a JSON array or an object "
            f"with an 'agents' array; got {type(agents).__name__}"
        )
    return [entry for entry in agents if isinstance(entry, dict)]


def get_elicit(slug: str) -> List[Dict[str, Any]]:
    """Input groups this agent must have satisfied before it can run.

    Each group is {"any_of": [...], "type": str, "prompt": str} and is satisfied
    when ANY member key is available. Deliberately separate from `required_inputs`,
    which under-declares: data_extraction lists its document inputs as *optional*,
    so a required-inputs rule would ask for nothing and the agent would run against
    no documents at all.
    """
    for agent in load_agent_definitions():
        if agent.get("slug") == slug:
            return list(agent.get("elicit") or [])
    return []
