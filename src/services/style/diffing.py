"""What would change if this profile were approved.

A recompiled profile arrives as a DRAFT and someone has to decide whether to activate it.
Showing them two JSON documents and asking them to spot the difference is not a decision
process — they will click approve. So the difference is computed and rendered in the same
plain language the rules use.

This is the "profile diff for approval" the feedback loop promises. It exists to make
approval a real gate rather than a formality: invariant 6 says nothing auto-activates, and
a review nobody can actually perform activates everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.style.profile import StyleProfile

# Plain-English names for the fields, so the diff does not read like a schema dump.
_LABELS = {
    "structural.subject_pattern": "Subject line pattern",
    "structural.greeting": "Greeting",
    "structural.opening_move": "Where the ask sits",
    "structural.body_form": "Body shape",
    "structural.target_words": "Typical length",
    "structural.sign_off": "Sign-off",
    "structural.signature_block": "Signature block",
    "register.formality": "Formality (1-5)",
    "register.directness": "Directness (1-5)",
    "register.hedging": "Hedging",
    "register.contractions": "Contractions",
    "register.person": "Voice",
    "lexical.preferred_terms": "Preferred terms",
    "lexical.banned_phrases": "Phrases to avoid",
    "lexical.number_format": "Number format",
    "lexical.date_format": "Date format",
    "behavioural.cta_form": "Call to action",
    "behavioural.deadline_phrasing": "Deadline phrasing",
    "behavioural.escalation_ladder": "Escalation ladder",
}


@dataclass
class FieldChange:
    """One field that differs between two profiles."""

    path: str
    label: str
    before: Any
    after: Any

    @property
    def is_list(self) -> bool:
        return isinstance(self.before, list) or isinstance(self.after, list)

    @property
    def added(self) -> List[Any]:
        if not self.is_list:
            return []
        return [x for x in (self.after or []) if x not in (self.before or [])]

    @property
    def removed(self) -> List[Any]:
        if not self.is_list:
            return []
        return [x for x in (self.before or []) if x not in (self.after or [])]

    def describe(self) -> str:
        if self.is_list:
            parts = []
            if self.added:
                parts.append("added " + ", ".join(f"“{x}”" for x in self.added))
            if self.removed:
                parts.append("removed " + ", ".join(f"“{x}”" for x in self.removed))
            return f"{self.label}: " + "; ".join(parts)
        return f"{self.label}: {_render(self.before)} → {_render(self.after)}"


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "none"
    return f"“{value}”" if isinstance(value, str) else str(value)


def _flatten(payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section, fields in (payload or {}).items():
        if isinstance(fields, dict):
            for name, value in fields.items():
                flat[f"{section}.{name}"] = value
        else:
            flat[section] = fields
    return flat


def diff_profiles(
    before: StyleProfile | Dict[str, Any],
    after: StyleProfile | Dict[str, Any],
) -> List[FieldChange]:
    """Every field that differs, in the schema's own order.

    Order matters for readability: structural changes first, because "your emails will get
    40 words longer" is a bigger deal to a reader than a reordered term list, and the
    schema already puts them in roughly that order.
    """

    before_json = before.to_json_dict() if isinstance(before, StyleProfile) else dict(before)
    after_json = after.to_json_dict() if isinstance(after, StyleProfile) else dict(after)

    flat_before, flat_after = _flatten(before_json), _flatten(after_json)

    changes: List[FieldChange] = []
    for path in list(flat_before) + [p for p in flat_after if p not in flat_before]:
        old, new = flat_before.get(path), flat_after.get(path)
        # Tuples and lists compare unequal despite meaning the same thing — target_words
        # round-trips through JSON as a list and comes back from Pydantic as a tuple.
        if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
            if list(old) == list(new):
                continue
        elif old == new:
            continue
        changes.append(
            FieldChange(path=path, label=_LABELS.get(path, path), before=old, after=new)
        )
    return changes


def render_diff(
    before: StyleProfile | Dict[str, Any],
    after: StyleProfile | Dict[str, Any],
) -> str:
    """The diff as text a person can read before approving."""

    changes = diff_profiles(before, after)
    if not changes:
        return "No changes — the recompiled profile is identical to the active one."
    return "\n".join(f"• {change.describe()}" for change in changes)
