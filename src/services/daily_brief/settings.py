"""What an administrator chooses about the daily brief, and who reads which one.

Everything here lives in one ``proc.bp_policy`` row (``daily_brief``), so it is
changed on a settings screen and never in code:

  * which VIEW a person gets — named for them, else by a group on their sign-in,
    else by their role, else the default;
  * what each view SHOWS — its sections, in order — and whether it covers
    everyone's records or only the reader's own;
  * the CUT-OFFS — how old an issue must be to count as aged, how far ahead
    "upcoming" looks, how many examples a section names.

Two rules, the same two the governed limits follow:

  1. **A missing or invalid setting refuses.** The brief is not served on a value
     the code made up. A chief procurement officer shown the buyer's brief
     because a setting failed to load would never know.
  2. **A saved setting is checked as a whole.** A view naming a section that does
     not exist, or a role pointed at a view nobody defined, is rejected at save
     with the reason, rather than discovered by whoever opens Home next.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.services import governed_limits

POLICY = "daily_brief"

SCOPE_EVERYONE = "everyone"
SCOPE_OWN = "own"
SCOPES = (SCOPE_EVERYONE, SCOPE_OWN)

#: Every section a view may show, and whether it can be narrowed to one person's
#: records. A section that cannot (it is a count across the whole company) may
#: only appear in a view that covers everyone: shown in a personal view it would
#: be company-wide figures under a heading that says "yours".
SECTIONS: Dict[str, bool] = {
    "momentum": False,
    "critical_aged": True,
    "upcoming": False,
    "approvals": False,
    "pipeline": True,
    "supplier_emails": True,
    "opportunities": True,
    "my_requests": True,
}

#: Whole-number settings and the range each may take.
NUMBERS: Dict[str, Tuple[int, int]] = {
    "momentum_window_days": (1, 90),
    "critical_aged_min_days": (0, 365),
    "critical_aged_top_n": (1, 25),
    "approval_aged_min_days": (0, 365),
    "upcoming_horizon_days": (1, 730),
    "upcoming_top_n": (1, 25),
    "pipeline_top_n": (1, 25),
    "supplier_email_window_days": (1, 90),
    "opportunity_top_n": (1, 25),
}

#: Settings that are a list of severity grades.
SEVERITY_LISTS = ("issue_severities", "critical_aged_severities")

#: Settings that map a name (a role, a sign-in group, a person) to a view.
VIEW_MAPS = ("view_by_role", "view_by_group", "view_by_person")

#: A money floor that may be left unset (null = no floor).
OPTIONAL_AMOUNTS = ("upcoming_min_value_gbp",)

_KNOWN = {"views", "view_order", "default_view", *VIEW_MAPS, *NUMBERS, *SEVERITY_LISTS,
          *OPTIONAL_AMOUNTS}


class BriefSettingsUnavailable(RuntimeError):
    """The settings could not be read, or do not hold together. The brief refuses."""


def validate(rules: Any) -> List[str]:
    """Everything wrong with ``rules``, in words an administrator can act on.

    An empty list means the settings can be used. Nothing is repaired or
    defaulted here: a setting that is wrong is reported, not corrected.
    """

    if not isinstance(rules, dict):
        return ["the settings are not a set of named values"]

    problems: List[str] = []
    for key in sorted(set(rules) - _KNOWN):
        problems.append(f"'{key}' is not a daily brief setting")
    for key in sorted(_KNOWN - set(rules)):
        problems.append(f"'{key}' is missing")

    views = rules.get("views")
    names: List[str] = []
    if "views" in rules:
        if not isinstance(views, dict) or not views:
            problems.append("'views' must define at least one view")
        else:
            names = [str(n) for n in views]
            for name, view in views.items():
                problems.extend(_view_problems(str(name), view))

    if "view_order" in rules:
        order = rules["view_order"]
        if not isinstance(order, list) or sorted(map(str, order)) != sorted(names):
            problems.append("'view_order' must list every view exactly once")

    if "default_view" in rules and names and rules["default_view"] not in names:
        problems.append(f"'default_view' names a view that is not defined: {rules['default_view']!r}")

    for key in VIEW_MAPS:
        if key not in rules:
            continue
        mapping = rules[key]
        if not isinstance(mapping, dict):
            problems.append(f"'{key}' must map names to views")
            continue
        for who, view in mapping.items():
            if not str(who).strip():
                problems.append(f"'{key}' has an entry with no name")
            if names and view not in names:
                problems.append(f"'{key}' sends {who!r} to a view that is not defined: {view!r}")

    for key, (low, high) in NUMBERS.items():
        if key not in rules:
            continue
        value = rules[key]
        # bool is an int in Python; "true days" is not a setting anyone chose.
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            problems.append(f"'{key}' must be a whole number from {low} to {high}")

    for key in SEVERITY_LISTS:
        if key not in rules:
            continue
        value = rules[key]
        if (not isinstance(value, list) or not value
                or any(not isinstance(s, str) or not s.strip() for s in value)):
            problems.append(f"'{key}' must list at least one severity grade")

    for key in OPTIONAL_AMOUNTS:
        if key not in rules or rules[key] is None:
            continue
        value = rules[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            problems.append(f"'{key}' must be an amount of zero or more, or left unset")

    return problems


def _view_problems(name: str, view: Any) -> List[str]:
    if not isinstance(view, dict):
        return [f"view '{name}' must state its sections and its scope"]
    problems: List[str] = []
    scope = view.get("scope")
    if scope not in SCOPES:
        problems.append(f"view '{name}' must cover '{SCOPE_EVERYONE}' or '{SCOPE_OWN}'")
    sections = view.get("sections")
    if not isinstance(sections, list) or not sections:
        return problems + [f"view '{name}' must show at least one section"]
    if len(set(map(str, sections))) != len(sections):
        problems.append(f"view '{name}' lists a section twice")
    for section in sections:
        if section not in SECTIONS:
            problems.append(f"view '{name}' shows a section that does not exist: {section!r}")
        elif scope == SCOPE_OWN and not SECTIONS[section]:
            problems.append(
                f"view '{name}' covers only the reader's own records, and '{section}' "
                f"is a company-wide figure that cannot be narrowed to one person")
    return problems


def load() -> Dict[str, Any]:
    """The settings as saved, read now, or raise.

    Read fresh on every call: a setting an administrator changes must apply to
    the next brief, without a restart.
    """

    try:
        rules = governed_limits.rules(POLICY, fresh=True)
    except governed_limits.LimitUnavailable as exc:
        raise BriefSettingsUnavailable(str(exc)) from exc
    problems = validate(rules)
    if problems:
        raise BriefSettingsUnavailable(
            "the daily brief settings do not hold together: " + "; ".join(problems))
    return rules


def resolve_view(rules: Dict[str, Any], *, role: Optional[str], groups: Iterable[str],
                 identities: Iterable[str]) -> Tuple[str, str]:
    """(view, basis) for one reader. ``basis`` says which setting decided it.

    Most specific wins: the person, then a group on their sign-in, then their
    role, then the default. A person in several groups that point at different
    views gets whichever of those ``view_order`` lists first.
    """

    by_person = {str(k).strip().lower(): v for k, v in rules["view_by_person"].items()}
    for identity in identities:
        view = by_person.get(str(identity or "").strip().lower())
        if view:
            return view, "person"

    by_group = rules["view_by_group"]
    wanted = {by_group[g] for g in groups if g in by_group}
    for view in rules["view_order"]:
        if view in wanted:
            return view, "group"

    view = rules["view_by_role"].get(str(role or ""))
    if view:
        return view, "role"
    return rules["default_view"], "default"


def catalogue() -> Dict[str, Any]:
    """What a settings screen may offer: the sections, the scopes and the ranges."""

    return {
        "sections": [{"key": key, "canBeOwn": own} for key, own in SECTIONS.items()],
        "scopes": list(SCOPES),
        "numbers": [{"key": key, "min": low, "max": high} for key, (low, high) in NUMBERS.items()],
        "severityLists": list(SEVERITY_LISTS),
        "optionalAmounts": list(OPTIONAL_AMOUNTS),
        "viewMaps": list(VIEW_MAPS),
    }
