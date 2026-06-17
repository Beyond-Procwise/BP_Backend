from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.requirement_session import RequirementSession
from services.redis_client import get_redis_client
from services import requirement_service

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


class RequirementsAgent(BaseAgent):
    """Conversation-led elicitation of a single procurement requirement.

    Stateless per call: loads the session, merges the turn's input via one LLM
    call, recomputes completeness, then either asks the next question or
    finalizes and hands off. Acting (ranking/RFQ) is left to other agents.
    """

    AGENTIC_PLAN_STEPS = (
        "Load the requirement session and seed it with category history.",
        "Extract field values from the buyer's message or pasted brief.",
        "Ask the next question, or finalize and hand off when complete.",
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

    def _advance(self, session: RequirementSession, user_text: str) -> str:
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
        try:
            result = self.call_ollama(prompt=prompt, format="json")
            raw = result.get("response") if isinstance(result, dict) else result
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            logger.debug("elicitation LLM parse failed", exc_info=True)
            parsed = {}
        if isinstance(parsed, dict):
            session.apply_fields(parsed.get("updates") or {})
            return str(parsed.get("next_question") or "")
        return ""

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            data = context.input_data or {}
            created_by = str(data.get("created_by") or context.user_id or "")
            redis = get_redis_client()

            session_id = str(data.get("session_id") or uuid.uuid4().hex)
            session = RequirementSession.load(session_id, redis)
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
            if user_text:
                session.add_turn("user", user_text)
                next_question = self._advance(session, user_text)

            required = self._required_fields()
            score, missing = requirement_service.evaluate_completeness(
                session.requirement, required
            )
            session.completeness_score = score
            session.missing_fields = missing

            requirement_out = {
                k: v for k, v in session.requirement.items() if not k.startswith("_")
            }

            if not missing:
                session.mark_complete()
                record = self._build_record(session, requirement_out, score, missing)
                requirement_service.persist(record)
                session.save(redis)
                self._emit_handoff(session, requirement_out)
                summary = self._summary(session, requirement_out)
                return self._with_plan(context, AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={
                        "session_id": session.session_id,
                        "requirement_id": session.requirement_id,
                        "complete": True,
                        "requirement": requirement_out,
                        "completeness_score": score,
                        "summary": summary,
                    },
                    next_agents=[],
                    confidence=score,
                ))

            session.save(redis)
            return self._with_plan(context, AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "session_id": session.session_id,
                    "requirement_id": session.requirement_id,
                    "complete": False,
                    "next_question": next_question,
                    "completeness_score": score,
                    "missing_fields": missing,
                    "requirement": requirement_out,
                },
                next_agents=[],
                confidence=score,
            ))
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("RequirementsAgent.run failed")
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    def _build_record(self, session, requirement_out, score, missing) -> Dict[str, Any]:
        record = {
            "requirement_id": session.requirement_id,
            "session_id": session.session_id,
            "status": "complete",
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
