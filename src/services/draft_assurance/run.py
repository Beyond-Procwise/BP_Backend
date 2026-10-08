"""One assurance run for one draft: begin before the model writes, finish after.

``Env`` is everything the run needs from its host (the agent): a database connection, the policy
engine, a model, governed prompts, the supplier master. Nothing here reaches for a global, so a
test can hand in fakes -- including a fake model that returns BAD output.

Declared vs classified: the negotiation paths declare their family (it must still exist in
config); free text is classified. Either way the draft goes through fact resolution, validation
and the judge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import accountability, authority as authority_mod, stages, tone as tone_mod
from .assure import Inputs, prepare_inputs
from .brief import counter_brief
from .family import FamilyConfig, FamilyConfigUnavailable, list_families, load_family

logger = logging.getLogger(__name__)
FALLBACK_FAMILY = "free_prompt"


@dataclass
class Env:
    conn_factory: Callable[[], Any]                      # context manager yielding a connection
    policy_engine: Any
    ask: Optional[Callable[[str, str], str]] = None
    prompt: Callable[[str], Optional[str]] = lambda name: None
    master_emails: Callable[[Optional[str]], List[str]] = lambda sid: []
    agent_name: str = "EmailDraftingAgent"
    agent_ids: Iterable[str] = ()
    user_id: Optional[str] = None
    exemplars: Callable[[], Optional[Dict[str, Any]]] = lambda: None
    access: Dict[str, Any] = field(default_factory=dict)    # how the reads were made: set by the connection helper


@dataclass
class AssuranceRun:
    env: Env
    data: Dict[str, Any]
    family_source: str
    inputs: Optional[Inputs] = None
    error: Optional[str] = None
    tone: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    planned: Optional[Dict[str, Any]] = None          # planner result for free text
    instruction: Optional[str] = None
    request: Optional[str] = None
    repair_rejected: Optional[str] = None

    # -- after the model has written ------------------------------------------------------
    def check(self, body: str) -> List[Dict[str, str]]:
        return self.inputs.check_text(body) if self.inputs is not None else []

    def finalize(self, composed: str, recipients: Iterable[str], supplier_id: Optional[str], *,
                 repaired: bool = False) -> Dict[str, Any]:
        env = self.env
        initiated = accountability.initiated(env.user_id, env.agent_name, env.agent_ids)
        base: Dict[str, Any] = {"accountability": {"initiated_by": initiated["id"], "kind": initiated["kind"]},
                                "family_source": self.family_source, "user_instruction": self.instruction,
                                "exemplars": _safe(env.exemplars)}
        if self.inputs is None:
            return {"status": "unassured", "reason": self.error or "not run", **base,
                    "stage_status": {"all": {"status": "not_run", "reason": self.error or "not run"}},
                    "ready": False}
        inp = self.inputs
        fam = inp.family
        facts = {k: f.value for k, f in inp.facts.items()}
        # Stage 3: a planner's brief for free text, the negotiation agent's own plan for counters.
        if self.planned is not None:
            brief = self.planned.get("brief") if self.planned.get("status") == "captured" else self.planned
        else:
            brief = counter_brief(self.data, inp, self.tone)
        # Stage 4: the judge scores what was written; it never gates, and never invents a score.
        judge = self._judge(composed, brief if isinstance(brief, dict) else None, facts)
        extras = {**base, "tone": self.tone, "brief": brief, "judge": judge,
                  "authority": self._authority(fam),
                  "classification": self.classification,
                  "clarification": (self.classification or {}).get("clarification") or {}}
        record = inp.finalize(composed, recipients, env.master_emails(supplier_id), extras)
        record["repaired"] = repaired
        record["read_control"] = env.access.get("control")        # dedicated_role | interim_readonly_session | unenforced
        if self.repair_rejected:
            record["repair_rejected"] = self.repair_rejected
        return record

    def _judge(self, text: str, brief: Optional[Dict[str, Any]], facts: Dict[str, Any]) -> Dict[str, Any]:
        if self.env.ask is None:
            return {"status": "unavailable", "reason": "no model is available"}
        fam = self.inputs.family
        return stages.judge_draft(self.env.ask, self.env.prompt("email_draft_judge"), fam.rubric,
                                  text=text, brief=brief, facts=facts)

    def _authority(self, fam: FamilyConfig) -> Optional[Dict[str, Any]]:
        if not fam.authority_agent:
            return None
        cur = self.inputs.facts.get("currency")
        currency = cur.value if cur else (self.data.get("currency") or self.data.get("currency_code"))
        amount = authority_mod.commitment_amount(self.data)
        if amount is None:
            return None
        return authority_mod.check_commitment(self.env.policy_engine, fam.authority_agent, amount, currency)


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        logger.exception("stage input failed")
        return {"ids": [], "scope": "none", "status": "unavailable", "reason": "exemplar lookup failed"}


def begin(env: Env, data: Dict[str, Any], *, slug: Optional[str], workflow_id: Optional[str],
          instruction: Optional[str] = None, request: Optional[str] = None,
          classify: bool = False, lookup: Optional[Dict[str, Any]] = None) -> AssuranceRun:
    """Resolve the family, derive tone, read facts, and (for free text) classify and plan. Never raises."""

    run = AssuranceRun(env=env, data=data, family_source="declared", instruction=instruction, request=request)
    candidates: Dict[str, Any] = {}      # ids the classifier read out of the request: they fill gaps, nothing more
    try:
        if classify:
            families = list_families(env.policy_engine)
            res = stages.classify_request(env.ask, env.prompt("email_family_classify"), request or "", families) \
                if env.ask else {"status": "unavailable", "reason": "no model is available"}
            run.classification = res
            if res["status"] == "captured":
                cls = res["classification"]
                slug = f"email_family_{cls['family_id']}"
                run.family_source = "classified"
                run.instruction = cls["user_instruction"] or instruction
                candidates = cls["lookup_keys"]
            else:
                slug = f"email_family_{FALLBACK_FAMILY}"
                run.family_source = "fallback"
        family = load_family(slug, env.policy_engine)                 # a declared family must exist in config
        with env.conn_factory() as conn:
            # What the caller supplied always wins over a candidate the model proposed.
            lk = {**candidates, **data, **(lookup or {}), "workflow_id": workflow_id or data.get("workflow_id")}
            run.inputs = prepare_inputs(conn, family, data, lookup_keys=lk)
            try:
                rules = tone_mod.load_rules(env.policy_engine)
                run.tone = tone_mod.derive_tone(conn, rules, supplier_id=data.get("supplier_id"),
                                                workflow_id=lk.get("workflow_id"), instruction=run.instruction)
            except tone_mod.ToneRulesUnavailable as exc:
                run.tone = {"status": "unavailable", "reason": str(exc)}
        if classify and env.ask is not None and run.inputs is not None:
            facts = {k: f.value for k, f in run.inputs.facts.items()}
            run.planned = stages.plan_brief(env.ask, env.prompt("email_brief_plan"), family_id=family.family_id,
                                            tone=run.tone, instruction=run.instruction or "", facts=facts,
                                            context={k: data.get(k) for k in family.context if data.get(k) is not None},
                                            request=request or "")
    except FamilyConfigUnavailable as exc:
        run.error = f"unknown or unreadable family: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("draft assurance unavailable: %s", exc)
        run.error = f"{type(exc).__name__}: {exc}"
    return run
