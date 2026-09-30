"""Active playbooks, loaded from ``proc.bp_playbook``.

Mirrors ``engines.rule_book.RuleBook``: load once, cache in memory, reload on
demand, take rows by injection so the selector can be tested without a database.

IT DIFFERS FROM THE RULE BOOK IN ONE PLACE, ON PURPOSE.

``RuleBook`` raises when it holds zero rules, because a sweep that runs no
detectors reports no findings and looks exactly like a clean scan. Zero
*playbooks* is not that. It is the honest state on the day this ships and for
as long as nobody has authored a strategy; failing closed there would make the
service unbootable until an expert wrote one. An empty rule book hides work
that should have happened -- an empty playbook table simply means no strategy
is on file. The sweep logs the count it proposed on every run, so zero stays
visible rather than becoming silence.

An unreadable store still raises. That is an outage, and it is not the same
thing as being empty.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .finding_source import MATCH_FIELDS

logger = logging.getLogger(__name__)

_SELECT = """
    SELECT p.playbook_id, p.playbook_name, p.trigger_source, p.trigger_match,
           p.agent_workflow_id, p.params, p.playbook_status, p.version,
           COALESCE(w.is_active, FALSE) AS workflow_is_active
      FROM proc.bp_playbook p
      LEFT JOIN proc.bp_agent_workflow w ON w.workflow_id = p.agent_workflow_id
     WHERE p.playbook_status = 'active'
     ORDER BY p.playbook_id
"""


class PlaybookStoreUnavailable(RuntimeError):
    """``proc.bp_playbook`` could not be read.

    Raised for an outage, never for an empty table -- see the module docstring.
    """


@dataclass
class Playbook:
    """One playbook as the database holds it."""

    playbook_id: int
    playbook_name: str
    trigger_source: str
    trigger_match: Dict[str, Any] = field(default_factory=dict)
    agent_workflow_id: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


class PlaybookStore:
    """Load and cache active playbooks from ``proc.bp_playbook``."""

    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        playbook_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None
        self._playbooks: List[Playbook] = []
        self._load(playbook_rows)

    # -- loading ---------------------------------------------------------

    def _load(self, playbook_rows: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        rows = list(playbook_rows) if playbook_rows is not None else self._fetch_rows()
        loaded = [self._to_playbook(row) for row in rows]
        self._playbooks = [pb for pb in loaded if pb is not None]

    def _fetch_rows(self) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                if conn is None:
                    return []
                cursor = conn.cursor()
                try:
                    cursor.execute(_SELECT)
                    columns = [c[0] for c in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                finally:
                    cursor.close()
        except PlaybookStoreUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage
            raise PlaybookStoreUnavailable(
                f"could not read proc.bp_playbook: {exc}"
            ) from exc

    @contextmanager
    def _connect(self):
        factory = self._connection_factory
        if factory is None:
            yield None
            return
        resolved = factory() if callable(factory) else factory
        if resolved is None:
            yield None
            return
        if hasattr(resolved, "__enter__"):
            with resolved as conn:
                yield conn
        else:
            yield resolved

    # -- coercion --------------------------------------------------------

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        """JSONB arrives as a dict from psycopg2, as text from other drivers."""
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (str, bytes)):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    def _to_playbook(self, row: Dict[str, Any]) -> Optional[Playbook]:
        pid = row.get("playbook_id")
        status = str(row.get("playbook_status") or "").strip()
        if status and status != "active":
            return None
        source = str(row.get("trigger_source") or "").strip()
        if source not in MATCH_FIELDS:
            logger.error(
                "skipping proc.bp_playbook row %s: trigger_source %r is not a "
                "finding source", pid, source,
            )
            return None
        if "workflow_is_active" in row and not row["workflow_is_active"]:
            # The approve endpoint refuses a playbook whose workflow is
            # inactive, but a workflow can be deleted AFTER its playbook was
            # approved. Selecting it would queue a proposal that can only fail
            # at the moment somebody accepts it.
            logger.error(
                "skipping proc.bp_playbook row %s (%s): agent_workflow_id %s is "
                "missing or inactive, so it can propose nothing that could run",
                pid, row.get("playbook_name"), row.get("agent_workflow_id"),
            )
            return None
        match = self._as_mapping(row.get("trigger_match"))
        unknown = [key for key in match if key not in MATCH_FIELDS[source]]
        if unknown:
            # The endpoint validates, so this row was hand-written. Loading it
            # would give a playbook that never fires and nothing to say why.
            logger.error(
                "skipping proc.bp_playbook row %s (%s): trigger_match keys %s "
                "are not match fields for %s",
                pid, row.get("playbook_name"), sorted(unknown), source,
            )
            return None
        return Playbook(
            playbook_id=int(pid or 0),
            playbook_name=str(row.get("playbook_name") or f"playbook {pid}"),
            trigger_source=source,
            trigger_match=match,
            agent_workflow_id=int(row.get("agent_workflow_id") or 0),
            params=self._as_mapping(row.get("params")),
            version=int(row.get("version") or 1),
        )

    # -- reading ---------------------------------------------------------

    def active_playbooks(self) -> List[Playbook]:
        return list(self._playbooks)

    def for_source(self, source: str) -> List[Playbook]:
        return [pb for pb in self._playbooks if pb.trigger_source == source]

    def reload(self) -> None:
        self._load()


def load_playbook_store(
    agent_nick: Optional[Any] = None,
    playbook_rows: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[PlaybookStore]:
    """Build the store at startup, carrying failure rather than raising.

    Follows ``engines.rule_book.load_rule_book``: the blast radius of "the
    playbook table cannot be read" is playbooks. Raising from here would take
    the whole API down with it.
    """

    try:
        return PlaybookStore(agent_nick=agent_nick, playbook_rows=playbook_rows)
    except PlaybookStoreUnavailable:
        logger.exception(
            "playbook store unavailable -- no playbook will be proposed until "
            "it loads. Everything else still starts."
        )
        return None
