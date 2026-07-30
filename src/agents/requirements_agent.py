from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

_RECONSTRUCT_FIELDS = (
    "title", "category", "description", "quantity", "unit", "target_budget",
    "currency", "needed_by_date", "delivery_location", "priority",
    "specifications", "constraints",
)


def _jsonsafe(value: Any) -> Any:
    """Coerce DB-typed values (Decimal/date) to JSON-serialisable forms."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.requirement_session import RequirementSession
from services.redis_client import get_redis_client
from services import requirement_service
from services import requirement_scope

logger = logging.getLogger(__name__)

_DEFAULT_ELICITATION_PROMPT = (
    "You are a procurement requirements assistant. Given the current requirement "
    "(JSON) and the buyer's latest message, extract any NEW field values the "
    "message provides and ask ONE concise question for the single most important "
    "still-missing field. Never invent values not stated by the buyer.\n"
    "Allowed fields: title, category, description, quantity, unit, target_budget, "
    "currency, needed_by_date, delivery_location, priority.\n"
    "Current requirement: {requirement}\n"
    "Still missing: {missing}\n"
    "Buyer message: {message}\n"
    'Respond ONLY with JSON: {"updates": {<field>: <value>, ...}, '
    '"next_question": "<one question, or empty string if nothing missing>"}'
)

_DEFAULT_SCOPE_PROMPT = (
    "You are a senior procurement category manager drafting a supplier-ready "
    "requirement scope. The buyer has asked WHAT THE REQUIREMENTS SHOULD BE, so "
    "propose the scope — do not interrogate them.\n"
    "Requirement captured so far (JSON): {requirement}\n"
    "Commodity family: {family}\n"
    "Buyer's message: {message}\n"
    "Category history from our own spend (may be empty): {history}\n"
    "Draft scope areas to tailor (JSON): {areas}\n"
    "For each area you can genuinely make more specific to THIS need, write one "
    "clear, testable requirement statement. Omit areas you cannot improve — the "
    "generic wording will be used for those. You may add up to three extra areas "
    "unique to this commodity.\n"
    "Never invent volumes, budgets, dates, standards or supplier names that are "
    "not given above; where a figure is needed, say it is for the buyer to confirm.\n"
    'Respond ONLY with JSON: {"areas": [{"area": "<name>", '
    '"requirement": "<one sentence>", "why": "<short reason>"}]}'
)


class RequirementsAgent(BaseAgent):
    """Conversation-led scoping of a single procurement requirement.

    Two modes, chosen from what the buyer actually said:

    * **propose** — the buyer asked what the requirements should be ("give me a
      scope", "provide a list of requirements"). We draft the scope: a
      commodity-specific skeleton, tailored by one LLM call, followed by exactly
      ONE confirm-or-adjust question. This is the fix for the agent's original
      behaviour, which could only ask questions and so answered a request for a
      scope with yet another question.
    * **elicit** — the buyer is answering. We extract the stated fields and ask
      the single most valuable missing one, as before.

    A proposal is advice, never data: it is labelled ``template``/``tailored``
    and only enters ``specifications`` when the buyer explicitly accepts it.
    Acting (ranking/RFQ) is left to other agents.
    """

    AGENTIC_PLAN_STEPS = (
        "Load the requirement session and seed it with category history.",
        "Decide whether the buyer wants a proposed scope or is answering a question.",
        "Propose a commodity-specific scope, or extract stated fields and ask one question.",
        "Finalize and hand off when the requirement is complete.",
    )

    def _required_fields(self) -> List[str]:
        policy = None
        try:
            policy = self.governing_policy("requirement_required_fields")
        except Exception:  # pragma: no cover - defensive
            policy = None
        if policy:
            details = policy.get("policy_details") or policy.get("details")
            fields = None
            if isinstance(details, dict):
                fields = details.get("required_fields")
            elif isinstance(details, str):
                try:
                    fields = json.loads(details).get("required_fields")
                except Exception:
                    fields = None
            if isinstance(fields, list) and fields:
                return [str(f) for f in fields]
        return list(requirement_service.DEFAULT_REQUIRED_FIELDS)

    def _llm_json(self, prompt: str) -> Dict[str, Any]:
        """One JSON LLM call, tolerant of every response shape and of failure.

        ``call_ollama`` may return an ollama GenerateResponse (attribute access),
        a dict, or a raw string. A transport failure (Ollama down, timeout, model
        not loaded) must never fail the turn — the buyer's session is worth more
        than one model call, so we return {} and the caller degrades.
        """
        try:
            # think=False is required: AgentNick is a reasoning model and returns
            # an empty `response` without it (see Model Routing Policy).
            result = self.call_ollama(prompt=prompt, format="json", think=False)
            if isinstance(result, str):
                raw = result
            else:
                raw = getattr(result, "response", None)
                if raw is None and hasattr(result, "get"):
                    raw = result.get("response")
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.debug("requirements LLM call failed or returned non-JSON", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Propose mode
    # ------------------------------------------------------------------

    @staticmethod
    def _seed(session: RequirementSession) -> Dict[str, Any]:
        seed = session.requirement.get("_seed_context")
        if not isinstance(seed, dict):
            seed = {}
            session.requirement["_seed_context"] = seed
        return seed

    def _pending_scope(self, session: RequirementSession) -> Dict[str, Any]:
        """The proposal the buyer is currently looking at, if any.

        Stashed inside ``_seed_context`` rather than a new column: that dict is
        already persisted to (and reloaded from) ``bp_requirement.seed_context``,
        so a proposal survives a Redis miss and can still be accepted a turn
        later — without a migration. Acceptance must adopt the exact wording the
        buyer saw, so regenerating it on demand is not an option.
        """
        pending = self._seed(session).get("proposed_scope")
        return pending if isinstance(pending, dict) else {}

    def _propose_scope(
        self, session: RequirementSession, user_text: str = ""
    ) -> Dict[str, Any]:
        """Draft a scope for this requirement: skeleton first, LLM tailoring second.

        The skeleton is what guarantees the buyer gets a scope at all. The LLM
        only rewrites the areas it can make specific; anything it omits, garbles
        or (when Ollama is unreachable) never returns keeps its generic wording.
        """
        req = session.requirement
        family = requirement_scope.classify_family(
            req.get("title"), req.get("category"), req.get("description"), user_text
        )
        areas = requirement_scope.scope_skeleton(family, req)
        prompt_template = (
            self.resolve_prompt("requirements_scope_proposal") or _DEFAULT_SCOPE_PROMPT
        )
        # Explicit substitution, not str.format: templates carry literal JSON braces.
        prompt = (
            prompt_template
            .replace("{requirement}", json.dumps(
                {k: _jsonsafe(v) for k, v in req.items() if not k.startswith("_")}))
            .replace("{family}", requirement_scope.FAMILY_LABELS.get(family, family))
            .replace("{message}", user_text or "")
            .replace("{history}", json.dumps(self._seed(session).get("recent_suppliers") or []))
            .replace("{areas}", json.dumps([a["area"] for a in areas]))
        )
        tailored = self._llm_json(prompt).get("areas")
        merged = requirement_scope.merge_tailored(areas, tailored)
        if not any(a.get("source") == "tailored" for a in merged):
            logger.warning(
                "RequirementsAgent proposed a template-only scope for %s (family=%s); "
                "the tailoring call returned nothing usable",
                session.requirement_id, family,
            )
        scope = {
            "family": family,
            "family_label": requirement_scope.FAMILY_LABELS.get(family, family),
            "areas": merged,
            "open_points": requirement_scope.open_points(merged),
            # Honest label for the UI: did the model contribute, or is this the
            # generic template? Never present template text as bespoke advice.
            "basis": "tailored" if any(a.get("source") == "tailored" for a in merged)
                     else "template",
            "accepted": False,
        }
        self._seed(session)["proposed_scope"] = scope
        return scope

    def _adopt_scope(self, session: RequirementSession) -> Dict[str, Any]:
        """Buyer accepted the proposal: record it as the requirement's specifications.

        Written with ``source: agent_proposed`` and the accepting user/time, so
        downstream (RFQ drafting, supplier ranking) can tell agent-drafted scope
        from buyer-authored scope. Nothing else about the requirement changes —
        an accepted scope is not a substitute for the missing hard facts.
        """
        scope = dict(self._pending_scope(session))
        if not scope:
            return {}
        scope["accepted"] = True
        self._seed(session)["proposed_scope"] = scope
        session.requirement["specifications"] = {
            "source": "agent_proposed",
            "basis": scope.get("basis"),
            "family": scope.get("family"),
            "accepted_by": session.created_by or "",
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "scope_areas": [
                {k: a.get(k) for k in ("area", "requirement", "why", "source")}
                for a in scope.get("areas") or []
            ],
        }
        return scope

    @staticmethod
    def _open_point_question(scope: Dict[str, Any]) -> str:
        """ONE question after adoption — the first thing only the buyer can answer."""
        points = (scope or {}).get("open_points") or []
        if not points:
            return "Adopted. Anything else you want in the scope?"
        return f"Adopted — to firm the scope up, what is {points[0]}?"

    def _advance(
        self, session: RequirementSession, user_text: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """One elicitation call: apply what the buyer stated, return (question, rejected).

        Rejections are returned rather than stored on the agent — the registry
        hands the same instance to concurrent requests, so per-turn state must
        not live on ``self``.
        """
        missing = session.missing_fields or list(requirement_service.DEFAULT_REQUIRED_FIELDS)
        template = self.resolve_prompt("requirements_elicitation") or _DEFAULT_ELICITATION_PROMPT
        # Explicit placeholder substitution (NOT str.format): prompt templates
        # contain literal JSON braces, which str.format would parse as fields.
        prompt = (
            template
            .replace("{requirement}", json.dumps(session.requirement))
            .replace("{missing}", ", ".join(missing))
            .replace("{message}", user_text)
        )
        # Seeded before the try: the "no field updates" warning below reads both,
        # and a transport failure (Ollama down/timeout) never reaches the
        # assignments. Leaving them unbound turned every LLM blip into an
        # UnboundLocalError that failed the whole turn.
        raw = None
        result = None
        try:
            # think=False is required: AgentNick is a reasoning model and returns
            # an empty `response` without it (see Model Routing Policy).
            result = self.call_ollama(prompt=prompt, format="json", think=False)
            # call_ollama returns an ollama GenerateResponse object (attribute
            # access) OR a dict OR a raw string — extract the response text from
            # whichever shape we got.
            if isinstance(result, str):
                raw = result
            else:
                raw = getattr(result, "response", None)
                if raw is None and hasattr(result, "get"):
                    raw = result.get("response")
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except Exception:
            logger.debug("elicitation LLM parse failed", exc_info=True)
            parsed = {}
        if not (isinstance(parsed, dict) and parsed.get("updates")):
            logger.warning(
                "RequirementsAgent elicitation produced no field updates; raw=%r",
                (raw if isinstance(raw, str) else result),
            )
        if isinstance(parsed, dict):
            # Validate against the column types before anything is applied. A
            # plausible-sounding string in a typed column ("needed_by_date":
            # "3 years from contract start") used to reach the INSERT and fail
            # the entire turn, losing the buyer's answer with it.
            # source_text is the buyer's own words this turn: an extracted date,
            # place or figure has to trace back to them, or it is the model's
            # invention and is dropped.
            clean, rejected = requirement_service.coerce_updates(
                parsed.get("updates") or {}, source_text=user_text
            )
            session.apply_fields(clean)
            return str(parsed.get("next_question") or ""), rejected
        return "", []

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            data = context.input_data or {}
            created_by = str(data.get("created_by") or context.user_id or "")
            redis = get_redis_client()

            session_id = str(data.get("session_id") or uuid.uuid4().hex)
            session = RequirementSession.load(session_id, redis)
            if session is None and data.get("session_id"):
                # Redis miss (or disabled): reload the durable gathering row.
                row = requirement_service.get_by_session(session_id)
                if row:
                    session = self._session_from_row(session_id, row)
            if session is None:
                session = RequirementSession(
                    session_id=session_id,
                    requirement_id=requirement_service.mint_requirement_id(created_by),
                    created_by=created_by,
                )
                category = str(data.get("category") or "")
                if category:
                    session.apply_fields({"category": category})
                    session_seed = requirement_service.seed_context(category)
                else:
                    session_seed = {}
                session.requirement.setdefault("_seed_context", session_seed)

            user_text = " ".join(
                str(part) for part in (data.get("brief"), data.get("message")) if part
            ).strip()

            next_question = ""
            mode = "elicitation"
            scope: Dict[str, Any] = {}
            rejected: List[Dict[str, Any]] = []
            if user_text:
                session.add_turn("user", user_text)
                if requirement_scope.wants_scope(user_text):
                    # The buyer asked what the requirements should be. Answering
                    # that with a question is the defect this branch exists to
                    # fix: propose the scope, then ask ONE thing.
                    mode = "proposed_scope"
                    if data.get("brief") and not session.requirement.get("title"):
                        # Extract the brief's stated facts first, so the scope is
                        # written about "the managed cloud data platform" and its
                        # real term and value — not about a category label. The
                        # question this returns is discarded: we are proposing.
                        # Only on a session's first turn, so the extra LLM call
                        # is not paid on every later proposal.
                        _, rejected = self._advance(session, user_text)
                    scope = self._propose_scope(session, user_text)
                    next_question = requirement_scope.confirm_question(scope["areas"])
                elif self._pending_scope(session) and requirement_scope.is_acceptance(user_text):
                    mode = "scope_accepted"
                    scope = self._adopt_scope(session)
                    next_question = self._open_point_question(scope)
                else:
                    next_question, rejected = self._advance(session, user_text)

            required = self._required_fields()
            score, missing = requirement_service.evaluate_completeness(
                session.requirement, required
            )
            session.completeness_score = score
            session.missing_fields = missing

            requirement_out = {
                k: v for k, v in session.requirement.items() if not k.startswith("_")
            }

            complete = not missing
            status = "complete" if complete else "gathering"
            if complete:
                session.mark_complete()
            # Persist every turn: the bp_requirement row IS the durable session.
            # A storage failure must not swallow the answer the buyer is waiting
            # for, so it is reported (persisted=False) rather than raised — the
            # turn still returns, and Redis keeps the session alive meanwhile.
            record = self._build_record(session, requirement_out, score, missing, status)
            persisted = True
            try:
                requirement_service.persist(record)
            except Exception:
                persisted = False
                logger.exception("requirement %s could not be persisted", session.requirement_id)
            session.save(redis)

            if complete:
                # A "captured" message that leaves the buyer to write the
                # requirement themselves is the same failure in a different
                # place, so a completed requirement always carries a scope.
                mode = "complete"
                if not scope:
                    scope = self._pending_scope(session) or self._propose_scope(session, user_text)
                self._emit_handoff(session, requirement_out)
                summary = self._summary(session, requirement_out)
                # `query` is the sourcing handle downstream agents (supplier_ranking)
                # consume; surface it as an output field so a workflow can route it.
                query = requirement_out.get("title") or requirement_out.get("category") or ""
                data = {
                    "session_id": session.session_id,
                    "requirement_id": session.requirement_id,
                    "complete": True,
                    "mode": mode,
                    "persisted": persisted,
                    "requirement": requirement_out,
                    "query": query,
                    "completeness_score": score,
                    "summary": summary,
                    # Carried even when complete: a turn that proposed a scope
                    # still needs its confirm-or-adjust question, and a client
                    # that only reads next_question would otherwise show nothing.
                    "next_question": next_question,
                }
                if scope:
                    data["scope"] = scope
                return self._with_plan(context, AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=data,
                    next_agents=[],
                    confidence=score,
                ))

            data = {
                "session_id": session.session_id,
                "requirement_id": session.requirement_id,
                "complete": False,
                "mode": mode,
                "persisted": persisted,
                "next_question": next_question,
                "completeness_score": score,
                "missing_fields": missing,
                "requirement": requirement_out,
            }
            if scope:
                data["scope"] = scope
            if rejected:
                # Visible, not silent: the buyer said something we could not store
                # as typed data, and the field is still missing because of it.
                data["rejected_fields"] = rejected
            return self._with_plan(context, AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                next_agents=[],
                confidence=score,
            ))
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("RequirementsAgent.run failed")
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    def _session_from_row(self, session_id: str, row: Dict[str, Any]) -> RequirementSession:
        """Rebuild a session from its durable bp_requirement row (Redis-less path)."""
        session = RequirementSession(
            session_id=session_id,
            requirement_id=row.get("requirement_id") or "",
            created_by=row.get("created_by") or "",
            status=row.get("status") or "gathering",
        )
        req: Dict[str, Any] = {}
        for key in _RECONSTRUCT_FIELDS:
            value = row.get(key)
            if value is None:
                continue
            req[key] = _jsonsafe(value)
        seed = row.get("seed_context")
        if isinstance(seed, dict):
            req["_seed_context"] = seed
        session.requirement = req
        return session

    def _build_record(self, session, requirement_out, score, missing,
                      status: str = "complete") -> Dict[str, Any]:
        record = {
            "requirement_id": session.requirement_id,
            "session_id": session.session_id,
            "status": status,
            "created_by": session.created_by,
            "completeness_score": score,
            "missing_fields": missing,
            "seed_context": session.requirement.get("_seed_context", {}),
        }
        for key in (
            "title", "category", "description", "quantity", "unit",
            "target_budget", "currency", "needed_by_date",
            "delivery_location", "priority",
        ):
            if key in requirement_out:
                record[key] = requirement_out[key]
        specs = requirement_out.get("specifications")
        if isinstance(specs, dict):
            record["specifications"] = specs
        constraints = requirement_out.get("constraints")
        if isinstance(constraints, dict):
            record["constraints"] = constraints
        return record

    def _emit_handoff(self, session, requirement_out) -> None:
        query = requirement_out.get("title") or requirement_out.get("category") or ""
        payload = {
            "requirement_id": session.requirement_id,
            "requirement": requirement_out,
            "query": query,
        }
        self.emit_signal("SUGGEST_AGENT",
                         "Requirement ready for supplier sourcing",
                         {**payload, "agent": "supplier_ranking"})
        self.emit_signal("SUGGEST_AGENT",
                         "Requirement ready for RFQ drafting",
                         {**payload, "agent": "email_drafting"})

    def _summary(self, session, requirement_out) -> str:
        title = requirement_out.get("title", "requirement")
        qty = requirement_out.get("quantity")
        loc = requirement_out.get("delivery_location")
        by = requirement_out.get("needed_by_date")
        bits = [f"Requirement {session.requirement_id} captured: {title}"]
        if qty:
            bits.append(f"qty {qty}")
        if loc:
            bits.append(f"to {loc}")
        if by:
            bits.append(f"by {by}")
        return ", ".join(bits) + "."
