"""The Style Brief: what a report looks like, and where each value came from.

READ THIS BEFORE EXTENDING IT — this module is deliberately less than the brief
asks for.

§3 of the build brief defines a Style Brief as the output of the settings
resolver (D1): a value resolved down a scope chain of Global → Region → Legal
entity → BU → Category → Site/User, carrying locks and an inheritance chain.
**That resolver does not exist.** The discovery pass found no scope dimension
anywhere in this platform — not in code, not in the schema, and not in the
corpus. Two independent places in this codebase had to confront the same
absence and wrote it down:

    There is no tenant dimension in the corpus to scope by — no customer_id on
    the vector payloads or the bp_ tables — and the x-customer-id header the UI
    sends is the constant "001".
        -- src/api/auth.py

    RLS is deliberately NOT enabled: there is no second tenant and no tenant
    dimension anywhere else in proc, so a policy here would be theatre.
        -- deploy/sql/2026-08-07_commercial_fact.sql

So this module does exactly two things and refuses the third:

  1. It **registers the keys** with fail-closed defaults, because a renderer
     needs a palette, a font and a section order from somewhere, and the
     alternative is those values scattered as literals through the renderer.
  2. It **resolves at one level** — platform default — and says so on the
     object. Every key's provenance reads ``platform_default``, and
     ``unresolved_scopes`` names the six scopes it could not consult.
  3. It **does not implement scope resolution, locks or inheritance.** Those
     are D1's, and a half-built resolver here would be the substitute §0
     forbids: it would look like scoping works, and it would silently keep
     returning the default when someone set a value at BU level.

The consequence is stated rather than hidden: a StyleBrief from this module is
honest about being unscoped, and it will not quietly start working when D1
lands — wiring it up will be a deliberate edit. DoD4, DoD5 and DoD14 are NOT
MET and cannot be met from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

# The scopes D1 would resolve through, named so a reader of a rendered report
# can see what was not consulted rather than assuming it was.
UNRESOLVED_SCOPES: Tuple[str, ...] = (
    "region", "legal_entity", "business_unit", "category", "site", "user",
)

PLATFORM_DEFAULT = "platform_default"


@dataclass(frozen=True)
class StyleKey:
    """One governed appearance decision, and what it falls back to."""

    key: str
    kind: str
    default: Any
    notes: str = ""


# The design tokens are the UI's, not new ones. Ported verbatim from
# beyond_procwise_ui/src/components/ui/tokens.css so a deck and the product do
# not disagree about what the brand is. The chart set is fixed by RB6 ruling
# R19; the two print substitutions are R23's ("a screen value does not hold as
# a printed status") and apply to the docx/print path, not to a screen deck.
PALETTE: Dict[str, str] = {
    "ocean": "#0a7fb0",
    "ocean_dark": "#09608B",
    "teal": "#00BEA9",
    "violet": "#6E64FF",
    "ink": "#0A2A43",
    "mute": "#6C7D87",
    "rule": "#E4EAEC",
    "warm": "#C0572A",
    "ok": "#0E7A50",
    "paper": "#FFFFFF",
}

CHART_CATEGORICAL: Tuple[str, ...] = (
    "#09608B", "#0a7fb0", "#00BEA9", "#5FE0AA", "#B45C33", "#7392A0",
)

PRINT_SUBSTITUTIONS: Dict[str, str] = {"#09608B": "#0C6C9C", "#00BEA9": "#12897F"}


REGISTRY: Dict[str, StyleKey] = {
    k.key: k for k in (
        StyleKey("report.style.tone", "enum", "formal",
                 "formal | plain | editorial"),
        StyleKey("report.style.lead_with", "enum", "risk",
                 "risk | savings | coverage | actions"),
        StyleKey("report.style.exec_summary.max_bullets", "int", 5),
        StyleKey("report.style.section_order", "list",
                 ["exec_summary", "spend", "coverage", "opportunities", "findings"],
                 "overridden per report type by the builder's registration"),
        StyleKey("report.style.chart.preferred", "list", ["bar", "waterfall"]),
        StyleKey("report.style.show_confidence_badges", "bool", True,
                 "DORA-profile tenants lock this true — the lock needs D1"),
        StyleKey("report.style.show_provenance_footnotes", "bool", True,
                 "DORA-profile tenants lock this true — the lock needs D1"),
        StyleKey("report.style.palette", "ref", PALETTE),
        StyleKey("report.style.font.body", "ref", "DM Sans"),
        StyleKey("report.style.font.mono", "ref", "DM Mono",
                 "every figure is set in mono — RB6 ruling R5"),
        StyleKey("report.style.unassessed_treatment", "enum", "surface_as_finding",
                 "never 'hide'"),
    )
}


class UnknownStyleKey(KeyError):
    """Asked for a key nobody registered.

    Raising rather than returning a default: a typo that resolved to something
    plausible is how a report comes to be styled by a key that governs nothing
    — the same failure ``services/actions`` closes for policy action names.
    """


@dataclass(frozen=True)
class StyleBrief:
    """Resolved appearance values, with the provenance of each.

    ``provenance`` maps every key to the level it resolved at. Today that is
    ``platform_default`` for all of them, and the effective-configuration
    inspector DoD5 asks for cannot show more than that until D1 exists.
    """

    report_type_id: str
    values: Mapping[str, Any]
    provenance: Mapping[str, str]
    unresolved_scopes: Tuple[str, ...] = UNRESOLVED_SCOPES
    resolver: str = "rga.style.defaults_only"

    def get(self, key: str) -> Any:
        if key not in self.values:
            raise UnknownStyleKey(
                f"{key!r} is not a registered style key; add it to "
                "services/rga/style.REGISTRY with a default")
        return self.values[key]

    def chain(self, key: str) -> str:
        """Why this key has the value it has. Shown on the artefact."""
        if key not in self.provenance:
            raise UnknownStyleKey(key)
        return self.provenance[key]

    def version(self) -> str:
        """A stable identity for this brief, for the reproducibility record."""
        from src.services.rga.models import canonical_hash

        return canonical_hash({
            "resolver": self.resolver,
            "report_type_id": self.report_type_id,
            "values": dict(self.values),
        })[:16]

    def disclosure(self) -> str:
        """The one line a rendered report carries about its own styling.

        A report whose look was decided entirely by a default should say so,
        rather than implying somebody chose it.
        """
        return (
            f"Style: {self.resolver} · every key at {PLATFORM_DEFAULT} · "
            f"scopes not consulted: {', '.join(self.unresolved_scopes)}"
        )


def resolve_style_brief(
    report_type_id: str,
    *,
    section_order: List[str] | None = None,
) -> StyleBrief:
    """The Style Brief for a report type, resolved at platform default only.

    ``section_order`` is the one key a report type may set, because the order of
    a board paper's sections is a property of the report rather than a
    preference — it comes from the builder's registration, not from a scope.
    Everything else takes the registry default.

    This function takes no principal, no entity and no scope, and that is
    deliberate: accepting them and ignoring them is how a caller comes to
    believe scoping works.
    """

    values: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for key, spec in REGISTRY.items():
        values[key] = spec.default
        provenance[key] = PLATFORM_DEFAULT

    if section_order is None:
        # A report type may register its own order with its builder (factpack.register).
        from src.services.rga.factpack import section_order_for
        section_order = section_order_for(report_type_id)
    if section_order is not None:
        values["report.style.section_order"] = list(section_order)
        provenance["report.style.section_order"] = f"report_type:{report_type_id}"

    return StyleBrief(
        report_type_id=report_type_id,
        values=values,
        provenance=provenance,
    )
