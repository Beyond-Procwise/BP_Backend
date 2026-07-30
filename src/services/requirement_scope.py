"""Propose-first requirement scoping.

Why this exists
---------------
The requirements agent could only ever *elicit*: its governing prompt says
"extract new field values and ask ONE question for the most important missing
field". So a buyer who asked "tell me the requirements I should have for a
managed cloud platform" got another question back, then another — never a scope.

This module supplies the deterministic half of the fix:

* ``wants_scope``    — is this message a request for advice, or an answer?
* ``classify_family``— which commodity family is this (goods/services/SaaS/works)?
* ``scope_skeleton`` — a complete, usable draft scope for that family.
* ``merge_tailored`` — overlay the LLM's commodity-specific wording on it.

The skeleton is the safety net: the LLM tailors the wording, but if it is slow,
terse or unreachable the buyer still receives a scope rather than a question.

Honesty rules baked in
----------------------
Every area carries ``source`` (``template`` = generic best practice,
``tailored`` = written for this requirement) and ``confirm`` (the facts only the
buyer can supply). Nothing here is presented as a buyer-stated fact, and no
number, budget or volume is ever invented — placeholders stay as placeholders
until the buyer fills them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FAMILY_SAAS_IT = "saas_it"
FAMILY_SERVICES = "services"
FAMILY_GOODS = "goods"
FAMILY_WORKS = "works"

DEFAULT_FAMILY = FAMILY_SERVICES

FAMILY_LABELS: Dict[str, str] = {
    FAMILY_SAAS_IT: "technology / SaaS",
    FAMILY_SERVICES: "services",
    FAMILY_GOODS: "supply of goods",
    FAMILY_WORKS: "works & construction",
}

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

# A scope request needs BOTH an ask-cue and a scope-noun. Requiring both is what
# keeps a genuine answer that happens to contain "in scope" ("Migration is in
# scope; dashboards stay in-house") from being mistaken for a request for advice
# — which would throw away the buyer's stated facts.
_ASK_CUES = (
    r"tell me", r"give me", r"provide", r"list", r"draft", r"suggest", r"recommend",
    r"propose", r"produce", r"generate", r"write", r"show me", r"build me", r"create",
    # Bare "what"/"which" is enough as an ask-cue because a scope-noun is also
    # required, and the assertion guard below catches "the requirements are …".
    r"what\b", r"which", r"how should", r"do (?:i|we) need",
    r"help me (?:with|define|scope)",
    r"i need (?:a|the|some)", r"can you", r"could you", r"please",
)
_SCOPE_NOUNS = (
    r"requirements?", r"scope", r"specifications?", r"specs?", r"sow",
    r"statement of work", r"criteria", r"checklist", r"must[- ]haves?",
    r"scope of work", r"scoping", r"rfp", r"rfq", r"itt",
)

# Standalone overrides: the buyer is explicitly handing the drafting back to us,
# or telling us to stop interrogating. No scope-noun needed.
_HANDBACK_CUES = (
    r"you tell me", r"you decide", r"you suggest", r"you draft", r"you propose",
    r"i don'?t know", r"not sure what", r"no idea what", r"up to you",
    r"stop asking", r"skip the questions?", r"no more questions?",
    r"enough questions?", r"just give me", r"quit asking",
)

_ASK_RE = re.compile("|".join(_ASK_CUES), re.I)
_NOUN_RE = re.compile("|".join(r"\b" + n + r"\b" for n in _SCOPE_NOUNS), re.I)
_HANDBACK_RE = re.compile("|".join(_HANDBACK_CUES), re.I)

# "The requirements are already agreed" is a statement about requirements, not a
# request for them. These verbs mark the noun as the subject of an assertion.
_ASSERTION_RE = re.compile(
    r"\b(?:requirements?|scope|specs?|specifications?)\b\s+(?:are|is|was|were|have been|has been)\b",
    re.I,
)


def wants_scope(text: Optional[str]) -> bool:
    """True when the buyer is asking us to PROPOSE a scope, not answering a question."""
    if not text or not str(text).strip():
        return False
    message = str(text).strip()
    if _HANDBACK_RE.search(message):
        return True
    if _ASSERTION_RE.search(message):
        return False
    return bool(_ASK_RE.search(message) and _NOUN_RE.search(message))


_ACCEPT_RE = re.compile(
    r"^\s*(?:yes|yep|yeah|ok|okay|sure|agreed|approved?|accept(?:ed)?|confirm(?:ed)?|"
    r"looks? good|that works|sounds good|go ahead|use (?:it|that|this)|fine|perfect|"
    r"great|do it|proceed)\b",
    re.I,
)
_ACCEPT_ANYWHERE_RE = re.compile(
    r"\b(?:use (?:it|that|this)|accept that|approve that|that works|looks good|"
    r"happy with (?:that|it|this))\b",
    re.I,
)
_NEGATION_RE = re.compile(r"\b(?:not|no|don'?t|isn'?t|except|but|however|drop|remove|change)\b", re.I)


def is_acceptance(text: Optional[str]) -> bool:
    """True when the buyer is accepting a pending proposal wholesale.

    Deliberately strict: any qualifier ("yes but drop the training item") is NOT
    an acceptance, because adopting a scope the buyer partly rejected would
    record items they never agreed to.
    """
    if not text or not str(text).strip():
        return False
    message = str(text).strip()
    lead = _ACCEPT_RE.match(message)
    anywhere = _ACCEPT_ANYWHERE_RE.search(message)
    if not (lead or anywhere):
        return False
    # Strip the accepting phrase before hunting for qualifiers, so "that works"
    # is not read as containing a negation of itself.
    remainder = _ACCEPT_ANYWHERE_RE.sub(" ", message)
    if lead:
        remainder = remainder[lead.end():] if lead.end() <= len(remainder) else ""
    return not _NEGATION_RE.search(remainder)


# ---------------------------------------------------------------------------
# Commodity family classification
# ---------------------------------------------------------------------------

_FAMILY_KEYWORDS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    # Works first: "installation services" is works, not services.
    (FAMILY_WORKS, (
        "works", "construction", "civils", "build", "refurb", "fit-out", "fitout",
        "install", "installation", "capital project", "capex project", "robotics line",
        "plant", "mechanical & electrical", "m&e", "demolition", "groundworks",
        "infrastructure project",
    )),
    (FAMILY_SAAS_IT, (
        "saas", "software", "licence", "license", "subscription", "platform", "cloud",
        "it ", "it/", "/it", "information technology", "technology", "digital", "data",
        "application", "system", "hosting", "cyber", "network", "hardware", "laptop",
        "server", "api", "erp", "crm",
    )),
    (FAMILY_GOODS, (
        "supply of goods", "goods", "office supplies", "stationery", "consumables",
        "materials", "equipment", "spares", "parts", "products", "furniture",
        "office & facilities", "packaging", "chemicals", "food", "uniform", "ppe",
    )),
    (FAMILY_SERVICES, (
        "service", "services", "managed", "consultancy", "consulting", "advisory",
        "facilities", "cleaning", "security guarding", "catering", "logistics",
        "maintenance", "outsourc", "staffing", "recruitment", "training", "audit",
        "legal", "marketing", "print",
    )),
)


def classify_family(*signals: Optional[str]) -> str:
    """Pick a commodity family from any free-text signals (title, category, description).

    Later signals do not outrank earlier ones; the family with the most keyword
    hits wins, and ties break in _FAMILY_KEYWORDS order (works > IT > goods >
    services) so the more safety-critical scope is preferred when ambiguous.
    """
    blob = " ".join(str(s).lower() for s in signals if s)
    if not blob.strip():
        return DEFAULT_FAMILY
    best_family = DEFAULT_FAMILY
    best_score = 0
    for family, keywords in _FAMILY_KEYWORDS:
        score = sum(1 for kw in keywords if kw in blob)
        if score > best_score:
            best_family, best_score = family, score
    return best_family if best_score else DEFAULT_FAMILY


# ---------------------------------------------------------------------------
# Scope templates
# ---------------------------------------------------------------------------
# (area, requirement, why, confirm-with-buyer)
#
# `{subject}` is substituted with the requirement's title when known, else a
# neutral noun for the family. No other placeholder is ever filled from
# invention — anything the buyer must supply is listed in `confirm`.

_CORE_TEMPLATES: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("Scope boundaries",
     "Supplier delivers {subject}. The requirement states what is in scope and, "
     "explicitly, what is excluded, so bids are comparable and change control has a baseline.",
     "Unstated exclusions are the most common source of post-award variations.",
     ("which activities are explicitly excluded",)),
    ("Commercial model",
     "Pricing is submitted on a fixed, transparent basis with a rate card for any variable "
     "element, a stated price-review mechanism, and no undisclosed pass-through charges.",
     "Makes bids comparable and stops uplift arriving as 'admin' or 'indexation' later.",
     ("budget envelope", "preferred pricing model", "payment terms")),
    ("Governance & reporting",
     "Supplier attends a defined contract-review cadence, reports performance against the "
     "stated measures, and follows a named escalation route with response times at each tier.",
     "Performance you do not review is performance you cannot enforce.",
     ("review cadence", "escalation contacts")),
    ("Term & exit",
     "The requirement states the term, any extension options, notice periods, and the "
     "supplier's obligations on exit — handover, knowledge transfer and return of assets or data.",
     "Exit terms are cheapest to agree before award and most expensive to agree after.",
     ("initial term and options",)),
)

_FAMILY_TEMPLATES: Dict[str, Tuple[Tuple[str, str, str, Tuple[str, ...]], ...]] = {
    FAMILY_SAAS_IT: (
        ("Functional capability",
         "Supplier's platform delivers the stated business capability for {subject}, with each "
         "must-have capability demonstrated in a scripted evaluation rather than asserted in a bid.",
         "A demonstration separates a product that does this from a roadmap that might.",
         ("the must-have capability list",)),
        ("Service levels & availability",
         "Supplier commits to a measurable availability target with a defined measurement window, "
         "plus incident response and resolution targets by severity, and service credits when missed.",
         "Availability without a measurement method and a credit is a marketing number.",
         ("target availability", "core service hours")),
        ("Volume & scale",
         "Supplier's solution supports the stated peak user and data volumes with headroom for "
         "growth across the term, and pricing states what happens when volumes are exceeded.",
         "Volume-driven overage is the most common cause of SaaS budget overrun.",
         ("peak users", "peak data volume", "expected growth")),
        ("Licensing model",
         "The licence basis is stated (named-user, concurrent, enterprise or consumption), with "
         "true-up and audit rights, and the right to reduce as well as increase quantities.",
         "One-way licence counts turn attrition into stranded cost.",
         ("preferred licence basis", "expected user counts")),
        ("Security posture",
         "Supplier holds and maintains the required certifications, enforces MFA and role-based "
         "access, tests to an agreed penetration-test frequency, and notifies incidents within a stated period.",
         "Certification plus notification duty is what makes security enforceable, not just claimed.",
         ("mandatory certifications", "incident-notification window")),
        ("Data protection & residency",
         "Personal and business data is stored and processed only in the permitted jurisdictions, "
         "under a data-processing agreement naming sub-processors, with changes requiring notice.",
         "Residency and sub-processor drift are found in audits, not in bids.",
         ("permitted jurisdictions", "whether personal data is in scope")),
        ("Integration & interoperability",
         "Supplier integrates with the named upstream and downstream systems using documented, "
         "versioned APIs, with the data volumes and frequencies stated and no bespoke connector lock-in.",
         "Integration effort discovered after award is charged at change rates.",
         ("systems to integrate with", "interface volumes")),
        ("Implementation & migration",
         "Supplier delivers a milestone-based implementation with defined acceptance criteria, "
         "and migrates existing data with a reconciliation step proving completeness.",
         "Unreconciled migration is how the old tools stay alive and the saving disappears.",
         ("go-live date", "data to migrate")),
        ("Support & training",
         "Supplier provides support at the stated coverage hours through named channels, and "
         "trains the agreed user groups with materials the buyer may reuse.",
         "Adoption, not licences, is what delivers the business case.",
         ("support hours", "which groups need training")),
        ("Business continuity & recovery",
         "Supplier meets stated recovery time and recovery point objectives, tests its continuity "
         "plan at an agreed frequency, and shares the test results.",
         "An untested recovery plan is an assumption.",
         ("acceptable downtime", "acceptable data loss")),
        ("Exit & data portability",
         "On exit, supplier returns all buyer data in a documented, non-proprietary format within "
         "a stated period, provides migration support, and certifies deletion of residual copies.",
         "Data portability is the only real defence against renewal-time price rises.",
         ("required export formats",)),
    ),
    FAMILY_SERVICES: (
        ("Service description & deliverables",
         "Supplier delivers the stated services for {subject}, with each deliverable, its frequency "
         "and its acceptance criteria defined, and interfaces to buyer-retained activities described.",
         "Ambiguous deliverables become disputes at the first invoice.",
         ("the deliverable list and frequencies",)),
        ("Service levels & KPIs",
         "Performance is measured against a small set of outcome-based KPIs with defined method, "
         "measurement period and consequence for sustained failure.",
         "Too many KPIs measure nothing; outcome KPIs drive the behaviour you want.",
         ("the outcomes that matter most",)),
        ("Resourcing & key personnel",
         "Supplier provides suitably qualified resources, names key personnel who may not be "
         "substituted without approval, and states cover arrangements for absence.",
         "Bid teams and delivery teams are frequently not the same people.",
         ("required competencies", "on-site presence needed")),
        ("Health, safety & environment",
         "Supplier operates a documented HSE management system, reports incidents within a stated "
         "period, and complies with buyer site rules and inductions.",
         "HSE failure is the one performance failure that cannot be remedied commercially.",
         ("sites and access constraints",)),
        ("Compliance & insurance",
         "Supplier evidences the required registrations, accreditations and insurance limits, and "
         "maintains them for the term, notifying any lapse.",
         "Cover that lapses mid-term is discovered when it is needed.",
         ("required insurance limits", "mandatory accreditations")),
        ("Data protection",
         "Where supplier processes buyer or personal data, it does so under a data-processing "
         "agreement, in the permitted jurisdictions, with sub-processors disclosed.",
         "Service contracts carry data risk even when data is not the subject.",
         ("whether personal data is in scope",)),
        ("Transition & mobilisation",
         "Supplier mobilises to an agreed plan with milestones and a stated point of service "
         "acceptance, with no degradation of service during the handover from current arrangements.",
         "Transition is where continuity is won or lost.",
         ("current arrangement and end date", "go-live date")),
        ("Continuous improvement",
         "Supplier proposes and delivers documented efficiency improvements each year, with the "
         "benefit share stated.",
         "Multi-year service prices drift up unless improvement is contractual.",
         ("appetite for gainshare",)),
        ("Business continuity",
         "Supplier maintains and tests a continuity plan covering the stated critical activities, "
         "with recovery times agreed.",
         "Critical services need a plan for the day the supplier has a bad day.",
         ("which activities are critical",)),
    ),
    FAMILY_GOODS: (
        ("Specification & quality",
         "Goods supplied against {subject} meet the stated technical specification and applicable "
         "standards, with equivalents permitted only where demonstrated equal or better.",
         "'Or equivalent' without a test is how substitution enters unnoticed.",
         ("the technical specification or acceptable equivalents",)),
        ("Volume, delivery & lead time",
         "Supplier delivers the stated quantities to the named locations within the agreed lead "
         "time, with a committed on-time-in-full performance level and notice of any shortfall.",
         "Lead time and OTIF are the two measures that protect operations.",
         ("quantities", "delivery locations", "required lead time")),
        ("Inspection & acceptance",
         "Goods are subject to inspection on receipt against defined acceptance criteria; rejected "
         "goods are collected and replaced at supplier's cost within a stated period.",
         "Without an acceptance step, the buyer owns every defect.",
         ("inspection regime",)),
        ("Warranty & returns",
         "Supplier warrants the goods for a stated period, and operates a returns process with "
         "defined response times for defective or surplus items.",
         "Warranty terms are the whole-life cost that unit price hides.",
         ("required warranty period",)),
        ("Packaging & labelling",
         "Goods are packaged to prevent transit damage and labelled with the identifiers the "
         "buyer's receipting process requires, including any hazard information.",
         "Bad labelling stalls goods receipt and delays payment.",
         ("labelling and receipting requirements",)),
        ("Product compliance & safety",
         "Goods comply with the applicable safety, substance and conformity regimes, and supplier "
         "supplies the certification and safety data on request.",
         "Non-compliant goods become the buyer's liability on resale or use.",
         ("applicable regimes",)),
        ("Sustainability & origin",
         "Supplier discloses the origin of the goods and their material content, and evidences "
         "the buyer's responsible-sourcing requirements through the supply chain.",
         "Origin claims are the ones most often unsupported.",
         ("sustainability requirements",)),
        ("Ordering & invoicing",
         "Orders are placed and acknowledged through the agreed channel, and invoices quote the "
         "purchase-order reference and line detail so three-way matching succeeds.",
         "Unmatched invoices are the largest single cause of payment delay.",
         ("ordering channel",)),
        ("Continuity of supply",
         "Supplier states its capacity, holds or can access buffer stock for the critical lines, "
         "and notifies discontinuation or supply risk in advance.",
         "A cheap unit price with no continuity is an operational risk.",
         ("which lines are critical",)),
    ),
    FAMILY_WORKS: (
        ("Scope of works & design responsibility",
         "Supplier delivers the works for {subject} to the stated design, with design "
         "responsibility, interfaces and any buyer-retained works explicitly allocated.",
         "Unallocated design responsibility is the classic construction dispute.",
         ("design status and who owns it",)),
        ("Programme & milestones",
         "Supplier works to a programme with dated milestones, float and critical path stated, and "
         "notifies delay with cause and recovery plan.",
         "Delay found late costs multiples of delay found early.",
         ("required completion date", "access dates")),
        ("CDM, health & safety",
         "Duty holders are appointed under the applicable construction safety regime, supplier "
         "produces the required plans before starting on site, and reports incidents within a stated period.",
         "These duties are statutory — they are not negotiable commercial terms.",
         ("whether duty holders are appointed", "site hazards known")),
        ("Site conditions & logistics",
         "The requirement states known site conditions, existing structures, services and "
         "contamination risk, and the access, working hours and welfare arrangements available.",
         "Undisclosed ground and asbestos risk is the most expensive unknown in works.",
         ("existing structures", "contamination or asbestos surveys", "permitted working hours")),
        ("Standards, consents & specification",
         "Works comply with the stated standards and specification, and the party responsible for "
         "each consent, permit or approval is named with the date it is needed.",
         "A missing consent stops a site whatever the contract says.",
         ("applicable standards", "who holds the consents")),
        ("Quality, inspection & testing",
         "Supplier operates an inspection and test plan with hold points the buyer may witness, "
         "and provides records before covering up work.",
         "Covered-up defects are found at handover, when remedy is dearest.",
         ("required hold points",)),
        ("Commissioning & handover",
         "Supplier commissions the works, demonstrates performance against the stated criteria, "
         "and hands over as-built information, O&M manuals and training before completion is certified.",
         "Handover information missing at completion is rarely supplied later.",
         ("performance criteria at handover",)),
        ("Defects & warranties",
         "A defects liability period applies, with response times for making good, and product and "
         "workmanship warranties assigned to the buyer.",
         "Warranty assignment is what makes year-three failures someone else's cost.",
         ("required defects period",)),
        ("Insurance, bonds & payment",
         "Supplier evidences the required insurances and any bond or parent-company guarantee, and "
         "is paid against a valuation mechanism with retention as agreed.",
         "Contractor insolvency mid-works is the risk these terms exist for.",
         ("insurance limits", "whether a bond is required")),
        ("Variations & change control",
         "Changes are instructed and priced through a stated change-control procedure against the "
         "agreed rate card before the work is done.",
         "Unpriced verbal instructions are how works budgets are lost.",
         ("change-approval authority",)),
    ),
}

_SUBJECT_FALLBACKS: Dict[str, str] = {
    FAMILY_SAAS_IT: "the platform",
    FAMILY_SERVICES: "the service",
    FAMILY_GOODS: "the goods",
    FAMILY_WORKS: "the works",
}


def _subject(family: str, requirement: Optional[Dict[str, Any]]) -> str:
    req = requirement or {}
    for key in ("title", "category"):
        value = req.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return _SUBJECT_FALLBACKS.get(family, "the requirement")


def scope_skeleton(family: str, requirement: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """A complete draft scope for ``family``, ordered commodity-specific → core.

    Every item is marked ``source="template"``: generic professional practice,
    not a buyer-stated fact. ``confirm`` lists what only the buyer can supply,
    which is exactly what stops the template from inventing volumes or budgets.
    """
    family = family if family in _FAMILY_TEMPLATES else DEFAULT_FAMILY
    subject = _subject(family, requirement)
    areas: List[Dict[str, Any]] = []
    for area, text, why, confirm in tuple(_FAMILY_TEMPLATES[family]) + _CORE_TEMPLATES:
        areas.append({
            "area": area,
            "requirement": text.replace("{subject}", subject),
            "why": why,
            "confirm": list(confirm),
            "source": "template",
        })
    return areas


def open_points(areas: Iterable[Dict[str, Any]]) -> List[str]:
    """The distinct facts the buyer still needs to supply, in first-seen order."""
    seen: List[str] = []
    for area in areas or []:
        for point in area.get("confirm") or []:
            text = str(point).strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def merge_tailored(
    areas: Sequence[Dict[str, Any]],
    tailored: Any,
) -> List[Dict[str, Any]]:
    """Overlay LLM-written wording on the skeleton, keeping every skeleton area.

    Matching is by case-insensitive area name. Areas the model invents are kept
    (a commodity specialist may know an area the template does not), areas it
    omits keep their template wording — so a lazy or truncated model response
    degrades to the generic scope instead of a short one.
    """
    merged = [dict(area) for area in areas or []]
    if not isinstance(tailored, (list, tuple)):
        return merged
    index = {str(a.get("area", "")).strip().lower(): a for a in merged}
    for item in tailored:
        if not isinstance(item, dict):
            continue
        name = str(item.get("area") or "").strip()
        text = str(item.get("requirement") or "").strip()
        if not name or not text:
            continue
        target = index.get(name.lower())
        if target is not None:
            target["requirement"] = text
            target["source"] = "tailored"
            why = str(item.get("why") or "").strip()
            if why:
                target["why"] = why
        else:
            extra = {
                "area": name,
                "requirement": text,
                "why": str(item.get("why") or "").strip() or "Commodity-specific addition.",
                "confirm": [str(c) for c in (item.get("confirm") or []) if str(c).strip()],
                "source": "tailored",
            }
            merged.append(extra)
            index[name.lower()] = extra
    return merged


def confirm_question(areas: Sequence[Dict[str, Any]]) -> str:
    """The ONE question that follows a proposal.

    Deliberately singular: the whole defect being fixed is that the agent asked
    a queue of questions. After proposing, we ask the buyer to confirm or
    correct — and name at most two open points so the ask stays answerable.
    """
    points = open_points(areas)[:2]
    if points:
        joined = " and ".join(points)
        return (
            f"Does this scope look right — and can you confirm {joined} so I can firm it up?"
        )
    return "Does this scope look right, or is there anything you want changed?"
