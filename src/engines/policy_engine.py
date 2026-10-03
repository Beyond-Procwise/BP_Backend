"""Database-backed policy loading utilities.

The original implementation read bundled JSON fixtures.  Runtime environments
now mandate that policy configuration is sourced directly from the PostgreSQL
``proc.bp_policy`` table so that agent behaviour reflects the latest governance
rules without requiring code deploys.  This module therefore provides a
lightweight repository for policy metadata with convenience helpers for common
lookups.

A **policy** answers "is this action allowed, does it need approval, is it
forbidden".  It is not the place for detection rules: what the system looks
for, and at what threshold, lives in ``proc.bp_rule`` behind
:class:`engines.rule_book.RuleBook`.  Detector configuration used to sit in
this table and be bound to detectors by fuzzy alias matching, which silently
mis-bound four of five rows; keeping the two apart is what stops that
recurring.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

# How long a degraded engine waits before trying the store again. base_agent
# builds ONE PolicyEngine during startup and never rebuilds it, so without a
# retry a database blip during boot left that agent with zero policies for the
# life of the process. With one, a read costs at most one connect attempt per
# cooldown window rather than one per authorization decision.
_RETRY_COOLDOWN_SECONDS = 30.0


class PolicyStoreUnavailable(RuntimeError):
    """``proc.bp_policy`` could not be read.

    Raised rather than answering with an empty policy set. An unreadable store
    and a store that holds no applicable policy produce the same empty result,
    and every caller here reads that result as consent: ``policies_for_action``
    returns ``[]`` and ``guardrail.authorize`` falls through to its
    default-allow branch; ``get_policy`` returns ``None`` and
    ``validate_and_apply`` answers "Validation successful (default weights)".
    An outage must not be able to say yes.

    Mirrors :class:`engines.rule_book.RuleBookUnavailable`, which already
    refuses this trade and names this module as the counter-example.
    """


class PolicyEngine:
    """Load and cache policy definitions from ``proc.bp_policy``."""

    SUPPLIER_POLICY_SLUGS = {
        "weight_allocation_policy",
        "categorical_scoring_policy",
        "normalization_direction_policy",
    }


    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        policy_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the engine and load policies.

        Parameters
        ----------
        agent_nick:
            When provided, the agent container is used to obtain a live
            database connection via ``get_db_connection``.
        connection_factory:
            Optional explicit factory returning a DB-API compatible
            connection.  This parameter is primarily intended for unit
            tests where lightweight stubs are preferable.
        policy_rows:
            Iterable of dictionaries mirroring the ``proc.bp_policy`` schema.
            When supplied the rows are used instead of querying the
            database, enabling deterministic fixtures in tests.
        """

        self.agent_nick = agent_nick
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None

        # Availability is tracked separately from emptiness. `policy_rows=[]` is
        # a legitimate "this store holds no policies" fixture; a query that
        # failed is not, and the two must not produce the same answer.
        self._store_available = True
        self._store_error: Optional[BaseException] = None
        self._last_attempt_at = 0.0
        self.retry_cooldown_seconds = _RETRY_COOLDOWN_SECONDS
        self._slug_index: Dict[str, Dict[str, Any]] = {}

        self._policies: List[Dict[str, Any]] = self._load_policies(policy_rows)
        self._rebuild_indexes()
        logger.info("PolicyEngine loaded %d policies", len(self._policies))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = str(value)
        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"(?<!^)(?=[A-Z0-9])", "_", text)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text).lower().strip("_")
        return slug or None

    @staticmethod
    def _coerce_details(payload: Any) -> Dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode(errors="ignore")
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except Exception:
                logger.debug("Unable to parse policy_details JSON: %s", text)
                return {"text": text}
            if isinstance(parsed, dict):
                return parsed
            return {}
        logger.debug("Unsupported policy_details payload: %r", payload)
        return {}

    @classmethod
    def _coerce_linked_agents(cls, payload: Any) -> List[str]:
        tokens: List[str] = []
        if payload is None:
            return tokens
        if isinstance(payload, str):
            payload = re.findall(r"[A-Za-z0-9_]+", payload)
        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                slug = cls._slugify(item)
                if slug:
                    tokens.append(slug)
        else:
            slug = cls._slugify(payload)
            if slug:
                tokens.append(slug)
        return tokens

    @contextmanager
    def _connect(self):
        factory = self._connection_factory
        if factory is None:
            yield None
            return
        resource = factory() if callable(factory) else factory
        if resource is None:
            yield None
            return
        if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
            with resource as conn:
                yield conn
            return
        try:
            yield resource
        finally:
            close = getattr(resource, "close", None)
            if callable(close):  # pragma: no cover - defensive cleanup
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close policy connection")

    def _fetch_policy_rows(self) -> List[Dict[str, Any]]:
        columns = [
            "policy_id",
            "policy_name",
            "policy_type",
            "policy_desc",
            "policy_details",
            "policy_linked_agents",
            "version",
        ]
        with self._connect() as conn:
            if conn is None:
                raise PolicyStoreUnavailable(
                    "no database connection available for proc.bp_policy"
                )
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT policy_id, policy_name, policy_type, policy_desc,
                               policy_details, policy_linked_agents, version
                        FROM proc.bp_policy
                        WHERE COALESCE(policy_status, 1) = 1
                        -- policy_id, NOT policy_name. Callers treat the first policy of a
                        -- type as the primary one, so a name-leading sort silently
                        -- reordered DISTINCT policies: for supplier_ranking it promoted
                        -- CategoricalScoringPolicy over WeightAllocationPolicy and the
                        -- ranking weights stopped being found. Ordering by id keeps
                        -- definition order, which is what "primary" has always meant here.
                        --
                        -- The name/version tiebreak this replaces existed to make DUPLICATE
                        -- (type, name) rows resolve deterministically. That is now enforced
                        -- in the database instead: ux_bp_policy_active_type_name makes a
                        -- second active row impossible, so ordering no longer has to
                        -- compensate for one.
                        ORDER BY policy_id
                        """
                    )
                    rows = cursor.fetchall()
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
            except PolicyStoreUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage
                raise PolicyStoreUnavailable(
                    f"could not read proc.bp_policy: {exc}"
                ) from exc
        return [dict(zip(columns, row)) for row in rows] if rows else []

    def _normalise_policy_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        record = dict(row)
        details = self._coerce_details(record.get("policy_details"))
        policy_id = record.get("policy_id")
        policy_name = record.get("policy_name")
        policy_desc = record.get("policy_desc")
        slug = (
            self._slugify(policy_name)
            or self._slugify(details.get("policy_name"))
            or self._slugify(details.get("identifier"))
            or self._slugify(policy_desc)
            or self._slugify(policy_id)
        )
        aliases = {
            value
            for value in (
                self._slugify(policy_name),
                self._slugify(policy_desc),
                self._slugify(policy_id),
                self._slugify(details.get("policy_name")),
                self._slugify(details.get("identifier")),
                self._slugify(details.get("policy_identifier")),
            )
            if value
        }
        rules = details.get("rules") if isinstance(details, dict) else {}
        if isinstance(rules, dict):
            aliases.update(
                {
                    alias
                    for alias in (
                        self._slugify(rules.get("name")),
                        self._slugify(rules.get("policy_name")),
                    )
                    if alias
                }
            )
        linked_agents = self._coerce_linked_agents(record.get("policy_linked_agents"))
        aliases.update(linked_agents)
        if slug:
            aliases.add(slug)
        identifier = (
            details.get("policy_identifier")
            or details.get("identifier")
            or policy_id
        )
        policy_identifier = str(identifier) if identifier is not None else None
        policy = {
            "policyId": policy_identifier,
            "policyName": policy_name,
            "policy_desc": policy_desc,
            "policy_type": record.get("policy_type"),
            "details": details if isinstance(details, dict) else {},
            "aliases": aliases,
            "slug": slug or self._slugify(policy_identifier) or "",
            "policy_linked_agents": linked_agents,
            "raw_row": record,
        }
        return policy

    def _load_policies(
        self, override_rows: Optional[Iterable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Load policies, recording an unreadable store rather than raising.

        Construction must survive a governance outage: ``base_agent`` builds a
        PolicyEngine during API startup, and raising here would turn a database
        blip into a boot failure -- a degraded boot being exactly when the
        audit trail matters most. The outage is recorded on the instance
        instead, and the read paths that make authorization decisions refuse.
        """

        if override_rows is not None:
            rows: List[Dict[str, Any]] = list(override_rows)
        else:
            try:
                rows = self._fetch_policy_rows()
            except PolicyStoreUnavailable as exc:
                self._store_available = False
                self._store_error = exc
                self._last_attempt_at = time.monotonic()
                logger.error(
                    "PolicyEngine could not read proc.bp_policy; every policy "
                    "lookup will refuse until it can: %s",
                    exc,
                )
                return []
        self._store_available = True
        self._store_error = None
        policies: List[Dict[str, Any]] = []
        for row in rows:
            policy = self._normalise_policy_row(row)
            if policy:
                policies.append(policy)
        return policies

    def _rebuild_indexes(self) -> None:
        self._slug_index = {}
        for policy in self._policies:
            for alias in policy.get("aliases", set()):
                if alias not in self._slug_index:
                    self._slug_index[alias] = policy
        self.supplier_policies = self._collect_supplier_policies()
        self._normalise_weight_policy()

    # ------------------------------------------------------------------
    # Store health
    # ------------------------------------------------------------------
    @property
    def policy_store_available(self) -> bool:
        """False when the last load could not read ``proc.bp_policy``.

        An empty policy list does NOT imply this is False -- a store can
        legitimately hold no policies.
        """

        return self._store_available

    @property
    def policy_store_error(self) -> Optional[BaseException]:
        """Why the store could not be read, or ``None``."""

        return self._store_error

    def _require_store(self) -> None:
        """Refuse the read when the store could not be loaded.

        Retries once per cooldown window first, so a blip during startup does
        not permanently strand a long-lived engine on an empty policy set.
        """

        if self._store_available:
            return
        waited = time.monotonic() - self._last_attempt_at
        if waited >= self.retry_cooldown_seconds:
            logger.info("PolicyEngine retrying proc.bp_policy after an outage")
            self._policies = self._load_policies(None)
            self._rebuild_indexes()
            if self._store_available:
                logger.info(
                    "PolicyEngine recovered: %d policies", len(self._policies)
                )
                return
        raise PolicyStoreUnavailable(
            f"proc.bp_policy could not be read: {self._store_error}"
        )

    def _collect_supplier_policies(self) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for slug in self.SUPPLIER_POLICY_SLUGS:
            # _lookup_policy, not get_policy: this runs during load, when the
            # store health check would either raise or recurse into a reload.
            policy = self._lookup_policy(slug)
            if policy:
                collected.append(policy)
        if collected:
            return collected
        for policy in self._policies:
            if any(token.startswith("supplier") for token in policy.get("aliases", [])):
                collected.append(policy)
        return collected


    def _normalise_weight_policy(self) -> None:
        """Ensure default supplier ranking weights sum to 1."""
        weight_policy = self._lookup_policy("weight_allocation_policy")
        if not weight_policy:
            return
        rules = weight_policy.get("details", {}).get("rules", {})
        weights = rules.get("default_weights", {}) if isinstance(rules, dict) else {}
        if not isinstance(weights, dict):
            return
        total = sum(float(value) for value in weights.values())
        if total and abs(total - 1.0) > 1e-6:
            for key, value in list(weights.items()):
                try:
                    weights[key] = round(float(value) / total, 4)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    weights[key] = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reload_policies(self) -> None:
        """Re-read policies from the database and rebuild internal caches."""
        self._policies = self._load_policies(None)
        self._rebuild_indexes()
        logger.info("PolicyEngine reloaded %d policies", len(self._policies))

    def list_policies(self) -> List[Dict[str, Any]]:
        return list(self._policies)

    def iter_policies(self) -> Iterator[Dict[str, Any]]:
        return iter(self._policies)

    def get_policy(self, slug: str) -> Optional[Dict[str, Any]]:
        """The policy named ``slug``, or ``None`` when no policy defines it.

        Raises :class:`PolicyStoreUnavailable` when the store could not be
        read. ``None`` means "no such policy", which callers are entitled to
        treat as "nothing restricts this"; an outage is not that.
        """

        self._require_store()
        return self._lookup_policy(slug)

    def _lookup_policy(self, slug: str) -> Optional[Dict[str, Any]]:
        """``get_policy`` without the store health check, for use during load."""

        key = self._slugify(slug)
        if not key:
            return None
        policy = self._slug_index.get(key)
        if policy:
            return policy
        for candidate in self._policies:
            if key in candidate.get("aliases", set()):
                return candidate
        return None

    def policies_for_action(self, action: str) -> List[Dict[str, Any]]:
        """Every active policy that declares it applies to ``action``.

        A policy opts in by listing the action in ``details.applies_to``. This
        is the gate's only lookup path, so policies continue to load from
        exactly one place.
        """

        self._require_store()
        wanted = str(action or "").strip()
        if not wanted:
            return []
        matched: List[Dict[str, Any]] = []
        for policy in self._policies:
            details = policy.get("details")
            if not isinstance(details, dict):
                continue
            applies = details.get("applies_to")
            if isinstance(applies, str):
                applies = [applies]
            if not isinstance(applies, (list, tuple, set)):
                continue
            if wanted in {str(a) for a in applies}:
                matched.append(policy)
        return matched

    def validate_workflow(self, workflow_name: str, user_id: str, input_data: dict) -> dict:
        """Validate a workflow against policy rules."""

        if workflow_name == "supplier_ranking":
            # Only this branch reads the policy store, so only this branch can
            # be wrong about it. The fall-through below consults nothing, so
            # gating it on store health would add denials without adding
            # safety.
            try:
                self._require_store()
            except PolicyStoreUnavailable as exc:
                logger.error(
                    "refusing supplier_ranking: the policy store is unreadable: %s",
                    exc,
                )
                return {
                    "allowed": False,
                    "reason": (
                        "Policy store unavailable; refusing to rank without "
                        f"the governing weights ({exc})"
                    ),
                }
            criteria = input_data.get("criteria")
            if not criteria:
                weight_policy = self._lookup_policy("weight_allocation_policy") or {}
                rules = weight_policy.get("details", {}).get("rules", {})
                default_weights = (
                    rules.get("default_weights", {}) if isinstance(rules, dict) else {}
                )
                criteria = list(default_weights.keys())
            if not criteria:
                # No WeightAllocationPolicy exists and no criteria provided;
                # allow the workflow to proceed with agent-level defaults.
                logger.info("No ranking criteria derived; allowing workflow with agent defaults")
                return {"allowed": True, "reason": "No weight policy; using agent defaults"}
            intent = {
                "template_id": "rank_by_criteria",
                "parameters": {"criteria": criteria},
            }
            allowed, reason, _ = self.validate_and_apply(intent)
            return {"allowed": allowed, "reason": reason}

        return {"allowed": True, "reason": "No policy checks"}

    def validate_and_apply(self, intent: dict) -> tuple[bool, str, dict]:
        logger.debug("PolicyEngine validating intent: %s", intent)
        if not intent or not intent.get("parameters"):
            return False, "Query intent could not be determined.", intent

        if intent.get("template_id") == "rank_by_criteria":
            criteria = intent.get("parameters", {}).get("criteria")
            if not criteria or not isinstance(criteria, list):
                reason = (
                    "Policy validation failed: Ranking requires at least one criterion"
                )
                return False, reason, intent

            try:
                weight_policy = self.get_policy("weight_allocation_policy")
            except PolicyStoreUnavailable as exc:
                # Distinct from the branch below: "not defined yet" is a
                # deliberate hole, "could not be read" is an outage, and
                # letting the outage take the hole is the fail-open bug.
                logger.error(
                    "refusing rank_by_criteria: the policy store is unreadable: %s",
                    exc,
                )
                reason = (
                    "Policy validation failed: the policy store is unavailable "
                    f"({exc})"
                )
                return False, reason, intent
            if not weight_policy:
                # When the weight allocation policy is not yet defined in the
                # database, allow the ranking workflow to proceed with the
                # criteria provided.  The agent will use its own defaults.
                logger.info("WeightAllocationPolicy not found; allowing ranking with provided criteria")
                return True, "Validation successful (default weights).", intent

            rules = weight_policy.get("details", {}).get("rules", {})
            weight_map = rules.get("default_weights", {}) if isinstance(rules, dict) else {}
            defined_weights = set(weight_map.keys())
            for criterion in criteria:
                if criterion not in defined_weights:
                    reason = (
                        f"Policy validation failed: Ranking criterion '{criterion}' "
                        "has no defined weight."
                    )
                    return False, reason, intent

        logger.debug("PolicyEngine intent validated successfully")
        return True, "Validation successful.", intent
