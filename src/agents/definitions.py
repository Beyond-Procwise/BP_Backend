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
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# src/agents/definitions.py -> src/agents -> src -> <repo root>
DEFINITIONS_PATH: Path = Path(__file__).resolve().parents[2] / "agent_definitions.json"


# Where created agents come from: ``() -> (live entries, deleted slugs)``. The
# file ships the built-in agents; the agents a person made live in the database
# (repositories.agent_definition_repo) and are joined on here, so every reader
# of the catalogue sees one list. Not set — a test, a script, a server that
# started without its database — the catalogue is the file alone.
CreatedAgentSource = Callable[[], Tuple[List[Dict[str, Any]], Set[str]]]
_created_source: Optional[CreatedAgentSource] = None
_FAILURE_LOG_EVERY_SECONDS = 60.0
_last_failure_logged = float("-inf")


def set_created_agent_source(source: Optional[CreatedAgentSource]) -> None:
    global _created_source
    _created_source = source


def created_agent_source() -> Optional[CreatedAgentSource]:
    return _created_source


def load_agent_definitions(
    path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Return the agent catalogue as a list of definition dicts.

    With no ``path`` this is the whole catalogue: the built-in agents from the
    file and the created agents from their source. A created agent wins over a
    file entry with the same slug, and a deleted one hides it — an older
    version kept created agents in the file, and a copy left there must not
    come back to life. With a ``path`` it is that file and nothing else.
    """
    entries = _load_file(path)
    return entries if path is not None else with_created_agents(entries)


def with_created_agents(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``entries`` (the file's) with the created agents joined on."""
    if _created_source is None:
        return entries
    try:
        created, deleted = _created_source()
    except Exception:
        # Loud, and the built-ins still load: refusing to start the whole
        # catalogue because the created agents cannot be read would take the
        # shipped agents down with them.
        global _last_failure_logged
        now = time.monotonic()
        if now - _last_failure_logged >= _FAILURE_LOG_EVERY_SECONDS:
            # The catalogue is read many times a minute; one line per read
            # would bury the log an operator needs during the outage itself.
            _last_failure_logged = now
            logger.error(
                "created agents could not be read; serving the built-in agents alone",
                exc_info=True,
            )
        return entries
    taken = {entry.get("slug") for entry in created} | set(deleted)
    return [entry for entry in entries if entry.get("slug") not in taken] + list(created)


def _load_file(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """The catalogue FILE as a list of definition dicts.

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
