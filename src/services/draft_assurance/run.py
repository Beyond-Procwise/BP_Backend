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

from . import accountability, authority as authority_mod, inbound as inbound_mod, payment_details, stages, steering as steering_mod, tone as tone_mod
from .assure import Inputs, prepare_inputs
from .brief import counter_brief
from .family import FamilyConfig, FamilyConfigUnavailable, list_families, list_labels, load_family

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
    store_factory: Optional[Callable[[], Any]] = None       # context manager yielding a connection that may read the email_agent tables


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
    repair_skipped: Optional[str] = None                     # why the repair pass was not run at all (the payment-details rule)
    steering: Optional[steering_mod.Steering] = None          # what steers the writer: tone, the author's style rules, exemplars
    inbound_block: Optional[Dict[str, Any]] = None            # {"state": clear|blocked|unknown|not_checked, "flag_ids": [...]}: a flagged reply on this thread

    def blocked_reason(self) -> Optional[str]:
        """Why an AGENT must not draft on this thread, or None. A flagged reply (a suspected request to change payment details) blocks;
        so does being unable to check, because 'could not look' must never read as 'nothing there'."""

        state = (self.inbound_block or {}).get("state")
        if state == "blocked":
            return (f"a reply from this supplier {(self.inbound_block or {}).get('phrase') or 'has been flagged'} and has not been reviewed by a "
                    "person; nothing is drafted on this thread until it is")
        if state == "unknown":
            return "whether this supplier's thread holds an unreviewed payment-detail request could not be checked, so nothing is drafted"
        return None

    def request_texts(self) -> List[str]:
        """What the person asked for, every way it arrives. The payment-details rule screens these as well as the draft."""

        seen: List[str] = []
        for t in (self.request, self.instruction, self.data.get("prompt"), self.data.get("instruction")):
            if isinstance(t, str) and t.strip() and t not in seen:
                seen.append(t)
        return seen

    def payment_hold(self, body: str) -> Optional[Dict[str, Any]]:
        return payment_details.hold(body, self.request_texts())

    def guidance(self) -> str:
        """The block appended to the writer's user message. Empty when nothing steers (the prompt is then unchanged)."""

        return self.steering.block() if self.steering is not None else ""

    # -- after the model has written ------------------------------------------------------
    def check(self, body: str) -> List[Dict[str, str]]:
        return self.inputs.check_text(body) if self.inputs is not None else []

    def finalize(self, composed: str, recipients: Iterable[str], supplier_id: Optional[str], *,
                 repaired: bool = False) -> Dict[str, Any]:
        env = self.env
        initiated = accountability.initiated(env.user_id, env.agent_name, env.agent_ids)
        blocked = self.blocked_reason()
        base: Dict[str, Any] = {"accountability": {"initiated_by": initiated["id"], "kind": initiated["kind"]},
                                "family_source": self.family_source, "user_instruction": self.instruction,
                                "exemplars": _safe(env.exemplars),
                                "steering": self.steering.record() if self.steering is not None else None}
        if self.inputs is None:
            # Nothing was checked, but a draft on a flagged thread is still marked: the mark does not depend on a family loading.
            marks = ({"violations": [{"kind": "inbound_flag_unreviewed", "severity": "fail", "detail": blocked}]}
                     if blocked and (self.inbound_block or {}).get("state") in ("blocked", "unknown") else {})
            # The payment-details rule does not depend on a family loading either.
            pay = payment_details.violations(composed, self.request_texts())
            if pay:
                marks = {"violations": list(marks.get("violations") or []) + pay,
                         "payment_details_hold": payment_details.hold(composed, self.request_texts())}
            return {"status": "unassured", "reason": self.error or "not run", **base, **marks,
                    **({"inbound_block": dict(self.inbound_block)} if self.inbound_block and self.inbound_block.get("state") != "not_checked" else {}),
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
        if (judge or {}).get("review_flag"):
            # Logged every time so the false-flag rate can be measured; the flag itself rides in the stored judgement.
            logger.warning("email judge flag: family=%s workflow=%s criteria=%s overall=%s", fam.family_id,
                           self.data.get("workflow_id"), judge["review_flag"]["criteria"], judge.get("overall"))
        extras = {**base, "tone": self.tone, "brief": brief, "judge": judge,
                  "authority": self._authority(fam),
                  "classification": self.classification,
                  "clarification": (self.classification or {}).get("clarification") or {}}
        record = inp.finalize(composed, recipients, env.master_emails(supplier_id), extras)
        record["repaired"] = repaired
        if blocked and self.inbound_block and self.inbound_block.get("state") in ("blocked", "unknown"):
            # A person may still write on this thread (they may be the one dealing with it) but the draft is marked, not ready, and the
            # send guard will refuse it while the flag stands.
            record["violations"] = list(record.get("violations") or []) + [
                {"kind": "inbound_flag_unreviewed", "severity": "fail", "detail": blocked}]
            record["status"], record["ready"] = "needs_review", False
        if self.inbound_block and self.inbound_block.get("state") != "not_checked":
            record["inbound_block"] = dict(self.inbound_block)
        record["read_control"] = env.access.get("control")        # dedicated_role | interim_readonly_session | unenforced
        if self.repair_rejected:
            record["repair_rejected"] = self.repair_rejected
        if self.repair_skipped:
            record["repair_skipped"] = self.repair_skipped
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


def _inbound_state(env: Env, workflow_id: Optional[str], supplier_id: Optional[str]) -> Dict[str, Any]:
    """Is there an unreviewed suspected payment-detail request on this thread? Never raises; failure is 'unknown', which blocks."""

    if env.store_factory is None:
        return {"state": "not_checked", "flag_ids": []}
    try:
        with env.store_factory() as conn:
            flags = inbound_mod.blocking_flags(conn, workflow_id, supplier_id)
    except Exception:  # noqa: BLE001 - the flag lookup failing, or the door itself failing
        logger.exception("could not check the thread for an unreviewed payment-detail request")
        return {"state": "unknown", "flag_ids": []}
    out = {"state": "blocked" if flags else "clear", "flag_ids": [f["id"] for f in flags]}
    if flags:
        out["phrase"] = inbound_mod.block_phrase(flags)
    return out


def begin(env: Env, data: Dict[str, Any], *, slug: Optional[str], workflow_id: Optional[str],
          instruction: Optional[str] = None, request: Optional[str] = None,
          classify: bool = False, lookup: Optional[Dict[str, Any]] = None) -> AssuranceRun:
    """Resolve the family, derive tone, read facts, and (for free text) classify and plan. Never raises."""

    run = AssuranceRun(env=env, data=data, family_source="declared", instruction=instruction, request=request)
    # First, and independent of everything below: a flagged thread must be seen even if the family cannot be loaded.
    run.inbound_block = _inbound_state(env, workflow_id or data.get("workflow_id"), data.get("supplier_id"))
    candidates: Dict[str, Any] = {}      # ids the classifier read out of the request: they fill gaps, nothing more
    try:
        if classify:
            families = list_families(env.policy_engine)
            res = stages.classify_request(env.ask, env.prompt("email_family_classify"), request or "", families,
                                    labels=list_labels(env.policy_engine)) \
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
            run.inputs.request_texts = run.request_texts()
            directives = None
            try:
                rules = tone_mod.load_rules(env.policy_engine)
                directives = rules.directives
                run.tone = tone_mod.derive_tone(conn, rules, supplier_id=data.get("supplier_id"),
                                                workflow_id=lk.get("workflow_id"), instruction=run.instruction)
            except tone_mod.ToneRulesUnavailable as exc:
                run.tone = {"status": "unavailable", "reason": str(exc)}
        run.steering = steering_mod.resolve(env.policy_engine, env.store_factory, family_id=family.family_id,
                                            author=env.user_id, tone=run.tone, directives=directives)
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
