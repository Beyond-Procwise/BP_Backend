"""Wrap a drafting path: resolve facts first, check the draft after.

``prepare_inputs`` runs BEFORE the draft is composed. It reads the family's facts
from Postgres, and where the payload disagrees Postgres wins: the payload value is
overwritten, so the model is never shown the wrong figure, and the disagreement is
recorded as a conflict.

``Inputs.finalize`` runs AFTER. It checks the composed text and the recipients and
returns one JSON-safe assurance record to store with the draft.

``recheck_facts`` re-reads the exact rows a draft used, for the send path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Set

from . import payment_details, validator as V
from .facts import FactResolver, ResolvedFact, Unresolved, coerce
from .family import FamilyConfig

SLUG = "email_family_negotiation_counter"


def _first(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if data.get(k) not in (None, ""):
            return data[k]
    return None


def _jsonable(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _same(a: Any, b: Any) -> bool:
    da, db = V.to_decimal(a), V.to_decimal(b)
    if da is not None and db is not None:
        return da == db
    return str(a).strip().casefold() == str(b).strip().casefold()


@dataclass
class Inputs:
    family: FamilyConfig
    facts: Dict[str, ResolvedFact] = field(default_factory=dict)
    unresolved: Dict[str, Unresolved] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    carried: Dict[str, Any] = field(default_factory=dict)       # no row, payload value used
    reasoned: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    never_state: Dict[str, Set[Decimal]] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    claims: List[Dict[str, Any]] = field(default_factory=list)   # facts read from Postgres that no person has vouched for
    request_texts: List[str] = field(default_factory=list)       # what the person asked for: the payment-details rule reads it too
    asserted_texts: List[str] = field(default_factory=list)      # the person's own request on a classified run: carried, never verified

    # -- allowed sets --------------------------------------------------------
    def _allowed(self):
        nums: Set[Decimal] = set()
        dates: Set = set()
        refs: Set[str] = set()
        for f in self.facts.values():
            nums |= V.numbers_in(f.value)
            dates |= V.dates_in(f.value)
            if f.key.endswith("rfq_id") or f.key.endswith("po_id"):
                refs.add(str(f.value))
        for key, val in self.carried.items():
            nums |= V.numbers_in(val)
            dates |= V.dates_in(val)
        for key, r in self.reasoned.items():
            nums |= V.numbers_in(r["value"])
            dates |= V.dates_in(r["value"])
        for extra in ("rfq_id", "rfq"):  # a reference the payload carries is reported, not trusted
            if self.data.get(extra) and "rfq_id" not in self.facts:
                refs.add(str(self.data[extra]))
        for key in self.family.carried_keys:
            nums |= V.numbers_in(self.data.get(key))
            dates |= V.dates_in(self.data.get(key))
            refs |= set(V._REF.findall(str(self.data.get(key) or "")))   # a reference the person typed is theirs
        for text in self.asserted_texts:     # the family a request was classified into may not carry the request itself
            nums |= V.numbers_in(text)
            dates |= V.dates_in(text)
            refs |= set(V._REF.findall(text))
        nums.add(Decimal(int(self.data.get("round") or 1)))
        return nums, dates, refs

    def _on_record(self) -> List[Any]:
        """Everything this run holds that a contact detail may legitimately come from: facts, carried and reasoned values,
        the payload (recipients included) and the person's own words."""

        vals: List[Any] = [f.value for f in self.facts.values()] + list(self.carried.values())
        vals += [r["value"] for r in self.reasoned.values()] + list(self.request_texts)
        for v in self.data.values():
            vals += [str(x) for x in v] if isinstance(v, (list, tuple, set)) else [v]
        return [v for v in vals if isinstance(v, (str, int, float, Decimal))]

    def _verified_numbers(self) -> Set[Decimal]:
        nums: Set[Decimal] = set()
        for f in self.facts.values():
            nums |= V.numbers_in(f.value)
        for r in self.reasoned.values():
            nums |= V.numbers_in(r["value"])
        return nums

    def _verified_dates(self) -> Set:
        dates: Set = set()
        for f in self.facts.values():
            dates |= V.dates_in(f.value)
        for r in self.reasoned.values():
            dates |= V.dates_in(r["value"])
        return dates

    def check_text(self, text: str) -> List[Dict[str, str]]:
        nums, dates, refs = self._allowed()
        out = V.check_figures(text, nums - set().union(*self.never_state.values())
                              if self.never_state else nums, dates, refs)
        out += V.check_leaks(text, self.never_state)
        out += V.check_patterns(text, self.family.forbidden_patterns)
        out += V.check_required(text, self.family.required_elements, self.data.get("asks") or [])
        out += V.check_length(text, self.family.length_target)
        out += payment_details.violations(text, self.request_texts)     # a hard rule: no family setting reaches it
        out += V.check_contact_details(text, self._on_record())        # no family setting reaches it either
        return out

    def finalize(self, text: str, recipients: Iterable[str],
                 master_emails: Iterable[str], extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        violations = self.check_text(text)
        master = {e.strip().lower() for e in master_emails if e}
        for r in recipients or []:
            if str(r).strip().lower() not in master:
                violations.append(V._v("recipient_not_on_master", str(r)))
        missing = [k for k in self.family.required_facts if k not in self.facts]
        carried_required = [k for k in missing if k in self.carried]
        hard = [x for x in violations if x["severity"] == "fail"]
        # Figures that rest only on what the request carried: allowed, never verified.
        verified = self._verified_numbers()
        verified |= {Decimal(int(self.data.get("round") or 1))}   # "round 2" is not a claim
        unverified = sorted(str(n) for n in V.figures_in(text) - verified)
        # Dates likewise: one that no fact or reasoned value holds is the person's own, and is listed as written.
        known_dates = self._verified_dates()
        unverified_dates = [raw for raw, k in V.dates_written(text) if not any(V._same_date(k, d) for d in known_dates)]
        # A warning as well as a list: the stored violations are what a reviewer's screen already shows.
        violations += [V._v("unverified_date", raw, "warn") for raw in unverified_dates]
        problems = bool(hard or self.conflicts or self.assumptions or unverified or unverified_dates or self.claims
                        or [k for k in missing if k not in self.carried] or carried_required)
        record = {
            "family_id": self.family.family_id,
            "family_version": self.family.version,
            "mode": self.family.mode,
            "status": "needs_review" if problems else "verified",
            "facts": {k: {"value": _jsonable(f.value), "label": self.family.facts[k].label, **f.provenance()}
                      for k, f in self.facts.items()},
            "unresolved": [{"fact": u.key, "reason": u.reason} for u in self.unresolved.values()],
            "carried_unverified": {k: _jsonable(v) for k, v in self.carried.items()},
            "claims": self.claims,
            "conflicts": self.conflicts,
            "reasoned": {k: {"value": _jsonable(r["value"]), "basis": r["basis"]}
                         for k, r in self.reasoned.items()},
            "assumptions": self.assumptions,
            "request_text": (self.data.get("prompt") or "")[:2000] if isinstance(self.data.get("prompt"), str) else None,
            "unverified_figures": unverified,
            "unverified_dates": unverified_dates,
            "violations": violations,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        hold = payment_details.hold(text, self.request_texts)
        if hold:
            record["payment_details_hold"] = hold
        record.update(stage_fields(extras or {}, record, self.reasoned))
        return record


def _claim_reason(origin: Optional[str]) -> str:
    if origin is None:
        return "The row does not say where this value came from (its origin was never recorded), so it is treated as a claim."
    if origin == "extracted_unverified":
        return "Read from the supplier's email by software and not yet confirmed by a person."
    if origin == "rejected":
        return "A person rejected this extracted value."
    return f"Its recorded origin ({origin}) is not a confirmed one."


def prepare_inputs(conn: Any, family: FamilyConfig, data: Dict[str, Any],
                   lookup_keys: Optional[Dict[str, Any]] = None) -> Inputs:
    """Resolve the family's facts and overwrite disagreeing payload values in ``data``.

    ``lookup_keys`` are candidates only (ids found in the request); they are used to
    find rows and never as values. Defaults to ``data``.
    """

    keys = lookup_keys if lookup_keys is not None else data
    inp = Inputs(family=family, data=data)
    resolver = FactResolver(conn)
    for key, src in family.facts.items():
        got = resolver.resolve(src, keys)
        supplied = _first(data, src.caller_keys)
        if isinstance(got, ResolvedFact):
            inp.facts[key] = got
            if got.claim:
                inp.claims.append({"fact": key, "label": src.label, "origin": got.origin or "not recorded",
                                   "reason": _claim_reason(got.origin)})
            if supplied is not None and not _same(supplied, got.value):
                inp.conflicts.append({"fact": key, "postgres": _jsonable(got.value),
                                      "supplied": _jsonable(supplied),
                                      "supplied_from": src.caller_keys[0] if src.caller_keys else None,
                                      "resolution": "postgres_wins"})
            for ck in src.caller_keys:  # Postgres wins, everywhere the payload carried it
                data[ck] = float(got.value) if isinstance(got.value, Decimal) else got.value
        else:
            inp.unresolved[key] = got
            if supplied is not None:
                inp.carried[key] = supplied
    for key, spec in family.reasoned.items():
        value = _first(data, spec.get("caller_keys") or [key])
        if value is None:
            continue
        basis = [b for b in (spec.get("basis_from") or []) if b in inp.facts]
        # A caller may declare what a value rests on, but only a real fact or a named
        # context item counts; anything else is dropped and the value stays an assumption.
        declared = (data.get("reasoned_basis") or {}).get(key) or []
        basis += [b for b in declared if isinstance(b, str) and b not in basis
                  and (b in inp.facts or b in family.context)]
        inp.reasoned[key] = {"value": value, "basis": basis}
        if not basis:
            inp.assumptions.append(f"{key} = {value} has no verified basis")
    for key, carriers in family.never_state.items():
        vals = V.numbers_in([data.get(c) for c in carriers])
        if vals:
            inp.never_state[key] = vals
    return inp


def recheck_facts(conn: Any, family: FamilyConfig, assurance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Facts whose source row now holds a different value than when the draft was made."""

    resolver = FactResolver(conn)
    changed = []
    for key, rec in (assurance.get("facts") or {}).items():
        src = family.facts.get(key)
        if src is None:
            changed.append({"fact": key, "was": rec.get("value"), "now": None,
                            "reason": "fact no longer in family"})
            continue
        fact = ResolvedFact(key, rec.get("value"), rec["table"], rec["column"],
                            str(rec["row_id"]), rec.get("retrieved_at", ""))
        now = resolver.reread(fact, src)
        if now is None or not _same(rec.get("value"), now):
            changed.append({"fact": key, "was": rec.get("value"), "now": _jsonable(now)})
    return changed


# --- Stage outputs ---------------------------------------------------------------------
#
# Convention, stored with every record and relied on by the learning job:
#   None / absent          NOT CAPTURED (the stage never ran for this record)
#   [] / {} / "none"       CAPTURED, and nothing applied
#   stage_status[stage]    captured | empty | not_run | unavailable | invalid  (+ reason)


def _stage(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    return {"status": status, **({"reason": reason} if reason else {})}


def _from_result(res: Optional[Dict[str, Any]], not_run: str) -> Dict[str, Any]:
    if not res:
        return _stage("not_run", not_run)
    st = res.get("status")
    if st in ("captured", "scored", "ready", "missing"):
        return _stage("captured")
    return _stage(st or "unavailable", res.get("reason"))


def _steering_stage(rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rec:
        return _stage("not_run", "steering did not run")
    st = rec.get("status")
    if st in ("captured", "empty"):
        return _stage(st, rec.get("reason"))
    if st == "unavailable":
        return _stage("unavailable", rec.get("reason"))
    return _stage("not_run", rec.get("reason") or "steering is off")


def stage_fields(extras: Dict[str, Any], record: Dict[str, Any], reasoned: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the stages add to a record, with explicit empties and a status for each stage."""

    source = extras.get("family_source") or "declared"
    cls = extras.get("classification")
    tone = extras.get("tone")
    brief = extras.get("brief")
    judge = extras.get("judge")
    authority = extras.get("authority")
    ex = extras.get("exemplars")
    clar = extras.get("clarification") or {}

    # structured assumptions: the brief's own, plus any reasoned value with no basis it missed
    items: List[Dict[str, Any]] = [dict(a) for a in ((brief or {}).get("assumptions") or [])]
    have = {a.get("key") for a in items if a.get("key")}
    for key, r in reasoned.items():
        if not r["basis"] and key not in have:
            items.append({"id": key, "key": key, "text": f"{key} = {r['value']} has no verified basis",
                          "resolution": None})
    # A value read from Postgres that no person has vouched for (a price taken from an email) is a claim: a person confirms it
    # before the draft counts as ready, through the same mechanism as every other assumption.
    for c in record.get("claims") or []:
        items.append({"id": f"claim:{c['fact']}", "key": f"claim.{c['fact']}",
                      "text": f"{c['label']} was read from the supplier's email by software and has not been confirmed by a person. "
                              "Confirm it, or give the right value.", "resolution": None})
    # New or changed bank/payment details: a person must look, whatever the family or its mode (ruling 2026-10-09).
    if record.get("payment_details_hold"):
        items.append({"id": "payment_details", "key": "payment_details",
                      "text": "This email or its request mentions bank or payment details. A person must review it before it is sent. "
                              + payment_details.PORTAL_GUIDANCE, "resolution": None})
    # Tone gaps a person must confirm: a variable whose rule is `assume` that fell back to its default,
    # and any tone word in the instruction that nothing understood. Same mechanism, same screen.
    if tone and tone.get("status") == "captured":
        for g in tone.get("gaps") or []:
            if g["on_gap"] == "assume":
                items.append({"id": f"tone:{g['variable']}", "key": f"tone.{g['variable']}",
                              "text": f"No stored data to set {g['variable'].replace('_', ' ')}, so it was assumed to be "
                                      f"{g['default']}. Confirm, or give the right value.",
                              "resolution": None})
        if tone.get("unmapped_instruction"):
            words = ", ".join(f"'{w}'" for w in tone["unmapped_instruction"])
            items.append({"id": "tone:instruction", "key": "tone.instruction",
                          "text": f"Your instruction used {words}, which no tone setting covers, so it had no effect. "
                                  "Confirm the tone as set, or edit.", "resolution": None})
    # The classifier's question is answered the same way: confirm the family chosen, or edit to the other.
    if clar and cls and cls.get("status") == "captured":
        items.append({"id": "clarification", "key": None, "text": clar.get("question"),
                      "options": clar.get("options"), "resolution": None})
    unresolved = [a for a in items if not a.get("resolution")]
    stages = {
        "classify": (_stage("not_run", "family declared by the calling path") if source == "declared"
                     else _from_result(cls, "no classification result")),
        "tone": _from_result(tone, "tone variables were not derived"),
        "exemplars": (_stage("not_run", "exemplar retrieval did not run") if ex is None else
                      _stage("captured" if ex.get("ids") else "empty", ex.get("reason"))
                      if ex.get("status") != "unavailable" else _stage("unavailable", ex.get("reason"))),
        "brief": _from_result(brief, "no brief was produced"),
        "judge": _from_result(judge, "the draft was not judged"),
        "authority": (_stage("captured") if authority else _stage("not_run", "no authority check ran")),
        "steering": _steering_stage(extras.get("steering")),
    }
    return {
        "family_source": source,
        "classification": cls.get("classification") if cls and cls.get("status") == "captured" else None,
        "clarification": (clar if cls and cls.get("status") == "captured" else None),
        "lookup_keys": (cls or {}).get("classification", {}).get("lookup_keys") if cls and cls.get("status") == "captured" else extras.get("lookup_keys"),
        "user_instruction": extras.get("user_instruction"),
        "tone": ({"variables": tone["values"], "sources": tone["sources"], "version": tone.get("version")}
                 if tone and tone.get("status") == "captured" else None),
        "exemplars": ({"ids": list(ex.get("ids") or []), "scope": ex.get("scope") or "none"} if ex is not None else None),
        "brief": brief if brief and brief.get("status") in ("ready", "missing") else None,
        "assumption_items": items,
        "judge": judge,
        "authority": authority,
        "accountability": extras.get("accountability"),
        "steering": extras.get("steering"),
        "stage_status": stages,
        "ready": not unresolved,
    }
