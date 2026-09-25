"""Building a Fact Pack: the registry, the audit event, and the provenance rule.

A Fact Pack is built by report-type-specific deterministic queries. There is no
generic builder and there is deliberately no way to add a fact from outside one:
``FactPackBuilder`` is a registry keyed by ``report_type_id``, and a report type
with no registered builder raises rather than producing an empty pack. An empty
pack renders as a report with no figures in it, which reads as "nothing to
report" rather than "nobody built this".

THE PROVENANCE RULE, AND WHERE IT IS ENFORCED

§2 says a fact without a ``provenance_id`` is not added and a Finding is raised
instead. Those are two different mechanisms and both exist:

  * *Not added* is enforced by the type — ``FactEntry`` will not construct
    without one (``services/rga/models``). A builder that tries gets a
    ValidationError at the point of the mistake.
  * *A Finding is raised* is enforced here, by ``FactBuilder.add``, which
    catches that failure and turns it into a ``FACT_WITHOUT_PROVENANCE``
    finding on the pack rather than letting the whole build die. One
    unevidenced figure should cost the report that figure, not the report.

CONFIDENCE, AND WHY AN AGGREGATE IS NOT ASSERTED

``services/formulas/unassessed`` states the rule this follows: a number derived
from other numbers can never be more trustworthy than its least trustworthy
input, so the ladder is taken a minimum over. Applied here:

  * ASSERTED     — read straight from the corpus, no derivation (a count).
  * CORROBORATED — derived or aggregated, and reconciled against an independent
                   source (a multi-currency total converted at published rates
                   with nothing excluded).
  * UNASSESSED   — could not be evaluated, or was evaluated over an incomplete
                   population. Never a zero.

This mapping is a decision, not a discovery; it is recorded in docs/rga/discovery.md §7 Q5.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    FactEntry,
    FactPack,
    Finding,
    FindingCode,
    FormatHint,
    NUMBER,
    Origin,
    Severity,
)

logger = logging.getLogger(__name__)

AGENT = "rga_factpack"
PHASE = "reporting"


class NoBuilderRegistered(KeyError):
    """No deterministic query set is registered for this report type."""


_BUILDERS: Dict[str, Callable[["FactBuilder"], None]] = {}
_SECTION_ORDERS: Dict[str, List[str]] = {}
_COMPOSER_NOTES: Dict[str, Any] = {}   # text, or a callable(pack) -> text
_COMPOSER_LABELS: Dict[str, Callable[[Any], str]] = {}
_TITLES: Dict[str, str] = {}
_PROSE_RULES: Dict[str, Callable[[Any], List[str]]] = {}
#: The title a report type draws with when it registered none (the first report type's).
DEFAULT_TITLE = "Executive procurement summary"


def register(report_type_id: str, *, section_order: Optional[List[str]] = None,
             composer_note: Any = None,
             composer_label: Optional[Callable[[Any], str]] = None,
             title: Optional[str] = None,
             prose_rules: Optional[Callable[[Any], List[str]]] = None
             ) -> Callable[[Callable[["FactBuilder"], None]], Callable]:
    """Register the deterministic query set for one report type, and -- if the report has
    one -- the order of its sections, a property of the report rather than a preference."""

    def _wrap(fn: Callable[["FactBuilder"], None]) -> Callable:
        _BUILDERS[report_type_id] = fn
        if section_order is not None:
            _SECTION_ORDERS[report_type_id] = list(section_order)
        if composer_note:
            _COMPOSER_NOTES[report_type_id] = composer_note
        if composer_label is not None:
            _COMPOSER_LABELS[report_type_id] = composer_label
        if title:
            _TITLES[report_type_id] = title
        if prose_rules is not None:
            _PROSE_RULES[report_type_id] = prose_rules
        return fn

    return _wrap


def prose_faults_for(report_type_id: str, ast: Any) -> List[str]:
    """What this report type forbids in the composer's sentences -- a fault sends the draft
    back for its one corrected attempt, like a schema fault."""
    rules = _PROSE_RULES.get(report_type_id)
    return rules(ast) if rules else []


def title_for(report_type_id: str) -> str:
    """The title a report of this type is drawn and stored under."""
    return _TITLES.get(report_type_id, DEFAULT_TITLE)


def composer_label_for(report_type_id: str) -> Callable[[Any], str]:
    """How the composer is shown each figure's label -- the label itself unless the report
    type says otherwise. Only the model's view: the pack, and the page, keep the real label."""
    return _COMPOSER_LABELS.get(report_type_id) or (lambda entry: entry.label)


def composer_note_for(report_type_id: str) -> Any:
    """What the composer must be told about this report type, if anything."""
    return _COMPOSER_NOTES.get(report_type_id)


def section_order_for(report_type_id: str) -> Optional[List[str]]:
    """The section order a report type registered, or None to take the platform default."""
    order = _SECTION_ORDERS.get(report_type_id)
    return list(order) if order is not None else None


def registered_types() -> List[str]:
    return sorted(_BUILDERS)


class FactBuilder:
    """The only way a fact gets into a pack.

    Hands out sequential fact ids, stamps the provenance every fact must carry,
    and converts a rejected fact into a Finding instead of an exception.
    """

    def __init__(self, *, pack_id: str, scope: Dict[str, Any], as_of: str) -> None:
        self.pack_id = pack_id
        self.scope = scope
        self.as_of = as_of
        self.facts: List[FactEntry] = []
        self.findings: List[Finding] = []
        self._next = 1

    # -- ids ---------------------------------------------------------------

    def _mint(self) -> str:
        fact_id = f"F{self._next:04d}"
        self._next += 1
        return fact_id

    def _finding_id(self) -> str:
        return f"{self.pack_id}-FND{len(self.findings) + 1:03d}"

    # -- facts -------------------------------------------------------------

    def add(
        self,
        *,
        label: str,
        value: Optional[Decimal],
        derivation: str,
        confidence: Confidence,
        origin: Origin = Origin.OBSERVED,
        format_hint: FormatHint = FormatHint.TEXT,
        currency: Optional[str] = None,
        unit: Optional[str] = None,
        provenance_id: Optional[str] = None,
    ) -> Optional[FactEntry]:
        """Add one fact, or raise a Finding and add nothing.

        ``provenance_id`` defaults to a reference that resolves: the audit-spine
        row for this pack build (written with ``trace_id = pack_id``) plus the
        fact's own id. ``derivation`` names the query that produced it, so the
        pair answers both "who computed this" and "from what".
        """
        fact_id = self._mint()
        pointer = provenance_id if provenance_id is not None else (
            f"bp_agent_actions:{self.pack_id}#{fact_id}")
        try:
            entry = FactEntry(
                fact_id=fact_id,
                label=label,
                value=value,
                unit=unit,
                currency=currency,
                format_hint=format_hint,
                confidence=confidence,
                origin=origin,
                provenance_id=pointer,
                derivation=derivation,
            )
        except ValidationError as exc:
            self.finding(
                code=FindingCode.FACT_WITHOUT_PROVENANCE,
                severity=Severity.HIGH,
                detail=(f"{label!r} was dropped: {exc.error_count()} validation "
                        f"error(s), first: {exc.errors()[0].get('msg', '')}"),
                fact_id=fact_id,
            )
            return None
        self.facts.append(entry)
        return entry

    def unmeasured(self, *, label: str, derivation: str, reason: str,
                   unit: Optional[str] = None) -> Optional[FactEntry]:
        """Record that a measure could not be taken.

        The fact is still added, with no value and UNASSESSED, because a report
        that silently omits a measure reads as though it were not relevant. A
        Finding is raised alongside so the omission is visible in the roll-up
        rather than only on the page.

        ``reason`` is shown to the composer, which may quote it, and a narrative
        may not carry a literal number -- so a reason with a digit in it makes a
        report that can never release. Refused here, where the mistake is.
        """
        if NUMBER.search(reason):
            raise ValueError(
                f"an unmeasured reason may not contain a number: {reason!r}. "
                "The composer may quote it, and a quoted number blocks release")
        entry = self.add(
            label=label, value=None, derivation=derivation,
            confidence=Confidence.UNASSESSED, format_hint=FormatHint.TEXT, unit=unit,
        )
        self.finding(
            code=FindingCode.MEASURE_UNAVAILABLE,
            severity=Severity.MEDIUM,
            detail=f"{label}: {reason}",
            fact_id=entry.fact_id if entry else None,
            blocks_release=False,
        )
        return entry

    # -- findings ----------------------------------------------------------

    def finding(self, *, code: FindingCode, severity: Severity, detail: str,
                fact_id: Optional[str] = None, blocks_release: bool = True,
                location: Optional[str] = None) -> Finding:
        found = Finding(
            finding_id=self._finding_id(), code=code, severity=severity,
            detail=detail, fact_id=fact_id, blocks_release=blocks_release,
            location=location,
        )
        self.findings.append(found)
        return found


def pack_id_for(report_type_id: str, scope: Dict[str, Any], as_of: str) -> str:
    """The id a pack for these inputs will have -- known before it is built, so a
    job can carry its run id from the start."""
    from src.services.rga.models import canonical_hash

    return "FP-" + canonical_hash({
        "report_type_id": report_type_id,
        "scope": scope,
        "as_of": as_of,
    })[:12]


def build_fact_pack(
    report_type_id: str,
    *,
    scope: Dict[str, Any],
    as_of: Optional[str] = None,
    generated_by: str = AGENT,
    now: Optional[datetime] = None,
    emit_audit: bool = True,
    writer: Any = None,
) -> FactPack:
    """Run the registered query set and return an immutable pack.

    ``pack_id`` is derived from the inputs rather than minted randomly, so two
    builds of the same scope at the same as-of produce the same id and therefore
    the same provenance pointers — which is what makes DoD3's identical-hash
    check meaningful rather than a comparison of two random strings.
    """
    if report_type_id not in _BUILDERS:
        raise NoBuilderRegistered(
            f"no Fact Pack builder registered for {report_type_id!r}; "
            f"registered types are {registered_types()}")

    resolved_as_of = as_of or date.today().isoformat()
    pack_id = pack_id_for(report_type_id, scope, resolved_as_of)

    builder = FactBuilder(pack_id=pack_id, scope=scope, as_of=resolved_as_of)
    _BUILDERS[report_type_id](builder)

    pack = FactPack(
        pack_id=pack_id,
        report_type_id=report_type_id,
        scope=scope,
        as_of=resolved_as_of,
        generated_at=now or datetime.now(timezone.utc),
        generated_by=generated_by,
        facts=builder.facts,
        findings=builder.findings,
    )

    if emit_audit:
        _audit_built(pack, writer=writer)
    return pack


def _audit_built(pack: FactPack, *, writer: Any = None) -> None:
    """Write ``report.factpack_built``.

    Goes through ``rga.audit`` rather than calling the spine directly, so every
    §7 event has one spelling and one injection point — a test can capture the
    whole run without a database, and without leaving rows behind in the live
    spine, which is what a direct ``record_action`` here used to do.

    This row is what every fact's ``provenance_id`` points at, keyed by
    ``trace_id = pack_id``.
    """
    from src.services.rga import audit

    audit.emit(
        audit.FACTPACK_BUILT,
        run_id=pack.pack_id,
        agent=AGENT,
        pack_hash=pack.hash,
        summary=f"{pack.report_type_id}: {len(pack.facts)} fact(s), "
                f"{len(pack.findings)} finding(s)",
        details={
            "pack_id": pack.pack_id,
            "report_type_id": pack.report_type_id,
            "scope": pack.scope,
            "as_of": pack.as_of,
            "facts": [
                {"fact_id": f.fact_id, "label": f.label,
                 "derivation": f.derivation, "confidence": f.confidence.value,
                 "origin": f.origin.value, "display": f.display}
                for f in pack.facts
            ],
            "findings": [f.model_dump(mode="json") for f in pack.findings],
        },
        writer=writer,
    )
