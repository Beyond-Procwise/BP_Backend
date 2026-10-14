"""Detection rules, loaded from ``proc.bp_rule``.

A **rule** reads facts and asserts that something is the case, producing a
finding. A **policy** reads a proposed action and answers allowed / needs
approval / forbidden. A rule can be wrong; it cannot forbid. A policy cannot
detect. They live in separate tables so that neither can quietly become the
other -- which is exactly what happened while detector configuration was
sitting in ``proc.bp_policy`` wearing a policy badge.

Mirrors ``PolicyEngine``: load once, cache in memory, index by slug, reload on
demand. It differs in one deliberate way. ``PolicyEngine`` returns ``[]`` when
its query fails, so a governance outage there is indistinguishable from "no
policy applies". The rule book refuses that trade: an unreadable store or an
empty rule set raises, because detection that silently finds nothing looks
exactly like a clean scan.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_SELECT = """
    SELECT rule_id, rule_name, detector_slug, finding_type, scope,
           required_fields, conditions, severity, rule_status, version
      FROM proc.bp_rule
     ORDER BY rule_id
"""


class RuleBookUnavailable(RuntimeError):
    """The rule store could not be read, or holds no active rules.

    Raised rather than returning an empty list: a sweep that runs zero
    detectors and reports zero findings is the same shape as a sweep that ran
    every detector and found nothing, and the difference matters.
    """


@dataclass
class Rule:
    """One detection rule as the database holds it."""

    rule_id: int
    rule_name: str
    detector_slug: str
    finding_type: str = "opportunity"
    scope: Optional[str] = None
    required_fields: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    severity: Optional[str] = None
    version: int = 1


class RuleBook:
    """Load and cache detection rules from ``proc.bp_rule``."""

    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        rule_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None
        self._rules: List[Rule] = []
        self._by_slug: Dict[str, Rule] = {}
        self._load(rule_rows)

    # -- loading ---------------------------------------------------------

    def _load(self, rule_rows: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        rows = list(rule_rows) if rule_rows is not None else self._fetch_rows()
        rules = [self._to_rule(row) for row in rows]
        active = [rule for rule in rules if rule is not None]
        if not active:
            raise RuleBookUnavailable(
                "proc.bp_rule holds no active rules -- refusing to run a sweep "
                "with no detectors, which would report zero findings and look "
                "like a clean scan"
            )
        self._rules = active
        self._by_slug = {rule.detector_slug: rule for rule in active}

    def _fetch_rows(self) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                if conn is None:
                    raise RuleBookUnavailable(
                        "no database connection available for proc.bp_rule"
                    )
                cursor = conn.cursor()
                try:
                    cursor.execute(_SELECT)
                    columns = [c[0] for c in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                finally:
                    cursor.close()
        except RuleBookUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage
            raise RuleBookUnavailable(
                f"could not read proc.bp_rule: {exc}"
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

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if isinstance(value, (str, bytes)):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return []
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        return []

    def _to_rule(self, row: Dict[str, Any]) -> Optional[Rule]:
        status = row.get("rule_status")
        if status is not None and int(status) != 1:
            return None
        slug = str(row.get("detector_slug") or "").strip()
        if not slug:
            logger.error("skipping proc.bp_rule row %s: no detector_slug", row.get("rule_id"))
            return None
        return Rule(
            rule_id=int(row.get("rule_id") or 0),
            rule_name=str(row.get("rule_name") or slug),
            detector_slug=slug,
            finding_type=str(row.get("finding_type") or "opportunity"),
            scope=(str(row["scope"]) if row.get("scope") else None),
            required_fields=self._as_list(row.get("required_fields")),
            conditions=self._as_mapping(row.get("conditions")),
            severity=(str(row["severity"]) if row.get("severity") else None),
            version=int(row.get("version") or 1),
        )

    # -- reading ---------------------------------------------------------

    def active_rules(self) -> List[Rule]:
        return list(self._rules)

    def rule_for(self, detector_slug: str) -> Optional[Rule]:
        return self._by_slug.get(str(detector_slug or "").strip())

    def slugs(self) -> List[str]:
        return [rule.detector_slug for rule in self._rules]

    def reload(self) -> None:
        self._load()


def load_rule_book(
    agent_nick: Optional[Any] = None,
    rule_rows: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[RuleBook]:
    """Build the rule book at startup, carrying failure rather than raising.

    The blast radius of "there are no detection rules" is detection. Raising
    from here would take the whole API down with it -- every unrelated
    endpoint included -- so an unreadable or empty rule store is logged and
    returned as ``None``. Whoever actually tries to detect something re-raises
    it then, which keeps the refusal loud without making it universal.
    """

    try:
        return RuleBook(agent_nick=agent_nick, rule_rows=rule_rows)
    except RuleBookUnavailable as exc:
        logger.error(
            "proc.bp_rule is unavailable, so opportunity detection will refuse "
            "to run until it is readable: %s",
            exc,
        )
        return None
