"""Prompt templates: blanks, pinned fragments, and nothing else.

    {{name}}            a declared value
    {{deal.id}}         one field of a declared object
    {{> voice@3}}       the fragment `voice` at exactly version 3
    \\{{                 a literal "{{"

That is the whole language, on purpose. There are no conditions, loops,
filters or expressions, so a template cannot branch on anything and cannot run
code; and a fragment is always pinned to a version, so wording never changes
underneath a prompt that was checked with it.

A value is put into a blank and never read again: text a supplier wrote cannot
become part of the template, whatever it contains.

Refusals name the variable, never its value. Values are customer data and
these messages reach logs.

This module does no I/O. The registry supplies fragments; the renderer
supplies values.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

TYPES = ("string", "enum", "number", "boolean", "object")

_TOKEN = re.compile(r"\\\{\{|\{\{(.*?)\}\}", re.S)
_NAME = r"[a-z][a-z0-9_]*"
_BLANK = re.compile(rf"^\s*({_NAME})((?:\.{_NAME})*)\s*$")
_FRAGMENT = re.compile(r"^>\s*([a-z0-9][a-z0-9_.-]*)@([0-9]{1,9})\s*$")
_DECLARED_NAME = re.compile(rf"^{_NAME}$")

FragmentLookup = Callable[[str, int], Optional[Dict[str, Any]]]


class TemplateError(Exception):
    code = "TEMPLATE_ERROR"


class TemplateInvalid(TemplateError):
    """The template or its declarations are malformed. ``problems`` lists each."""

    code = "TEMPLATE_INVALID"

    def __init__(self, problems: Any) -> None:
        self.problems: List[str] = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


class VarMissing(TemplateError):
    code = "VAR_MISSING"


class VarType(TemplateError):
    code = "VAR_TYPE"


class VarUndeclared(TemplateError):
    code = "VAR_UNDECLARED"


class FragmentMissing(TemplateError):
    code = "FRAGMENT_MISSING"


@dataclass
class Parsed:
    #: ("text", str) | ("blank", root, (path, ...)) | ("fragment", key, version)
    segments: List[tuple] = field(default_factory=list)
    blanks: Set[str] = field(default_factory=set)
    fragment_refs: List[Dict[str, Any]] = field(default_factory=list)


def parse(template: str) -> Parsed:
    """Split a template into text, blanks and fragment references, or refuse it."""
    if not isinstance(template, str):
        raise TemplateInvalid("a template is text")
    out = Parsed()
    seen_fragments: Set[Tuple[str, int]] = set()
    at = 0

    def text(chunk: str) -> None:
        if "{{" in chunk:
            raise TemplateInvalid("'{{' is opened and never closed")
        if chunk:
            out.segments.append(("text", chunk))

    for match in _TOKEN.finditer(template):
        text(template[at:match.start()])
        at = match.end()
        if match.group(0) == "\\{{":
            out.segments.append(("text", "{{"))
            continue
        inner = match.group(1)
        blank = _BLANK.match(inner)
        fragment = _FRAGMENT.match(inner.strip())
        if blank:
            root, path = blank.group(1), tuple(p for p in blank.group(2).split(".") if p)
            out.segments.append(("blank", root, path))
            out.blanks.add(root)
        elif fragment:
            key, version = fragment.group(1), int(fragment.group(2))
            out.segments.append(("fragment", key, version))
            if (key, version) not in seen_fragments:
                seen_fragments.add((key, version))
                out.fragment_refs.append({"key": key, "version": version})
        else:
            raise TemplateInvalid(
                "'{{%s}}' is neither a blank nor a fragment pinned to a version"
                % inner.strip()[:40])
    text(template[at:])
    return out


# ---------------------------------------------------------------------------
# on save
# ---------------------------------------------------------------------------
def _declaration_problems(variables: List[Dict[str, Any]]) -> List[str]:
    problems: List[str] = []
    seen: Set[str] = set()
    for d in variables:
        name = d.get("name") if isinstance(d, dict) else None
        if not isinstance(name, str) or not _DECLARED_NAME.match(name):
            problems.append(f"{name!r} is not a usable name for a blank")
            continue
        if name in seen:
            problems.append(f"'{name}' is declared more than once")
        seen.add(name)
        kind = d.get("type")
        if kind not in TYPES:
            problems.append(f"'{name}' has an unknown type {kind!r}; use one of {', '.join(TYPES)}")
        elif kind == "enum" and not (isinstance(d.get("values"), list) and d["values"]):
            problems.append(f"'{name}' is an enum and needs a list of values")
        elif kind == "object" and "schema" in d and not isinstance(d["schema"], dict):
            problems.append(f"'{name}' has a schema that is not an object")
    return problems


def _declared(variables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {d["name"]: d for d in variables
            if isinstance(d, dict) and isinstance(d.get("name"), str)}


def problems_on_save(template: str, variables: List[Dict[str, Any]],
                     fragment: FragmentLookup, *, is_fragment: bool = False) -> List[str]:
    """Everything wrong with a template and its declarations. Empty means save it.

    ``fragment(key, version)`` returns that fragment version (its ``template``
    and ``variables``) or None.
    """
    try:
        parsed = parse(template)
    except TemplateInvalid as exc:
        return exc.problems

    variables = list(variables or [])
    problems = _declaration_problems(variables)
    declared = _declared(variables)
    used: Set[str] = set(parsed.blanks)

    if is_fragment and parsed.fragment_refs:
        problems.append("a fragment cannot use another fragment")

    for ref in ([] if is_fragment else parsed.fragment_refs):
        label = f"{ref['key']}@{ref['version']}"
        found = fragment(ref["key"], ref["version"])
        if found is None:
            problems.append(f"fragment '{label}' does not exist")
            continue
        try:
            inner = parse(found.get("template") or "")
        except TemplateInvalid:
            problems.append(f"fragment '{label}' is itself malformed")
            continue
        theirs = _declared(found.get("variables") or [])
        for name in sorted(inner.blanks):
            used.add(name)
            if name not in declared:
                problems.append(f"fragment '{label}' needs '{name}', which is not declared")
            elif name in theirs and theirs[name].get("type") != declared[name].get("type"):
                problems.append(
                    f"fragment '{label}' takes '{name}' as {theirs[name].get('type')}, "
                    f"declared here as {declared[name].get('type')}")

    for name in sorted(parsed.blanks - set(declared)):
        problems.append(f"blank '{name}' is not declared")
    for name in declared:
        if name not in used:
            problems.append(f"'{name}' is declared but never used")
    return problems


# ---------------------------------------------------------------------------
# on render
# ---------------------------------------------------------------------------
def _matches(declaration: Dict[str, Any], value: Any) -> bool:
    kind = declaration.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "enum":
        return value in (declaration.get("values") or [])
    if kind == "object":
        if not isinstance(value, dict):
            return False
        schema = declaration.get("schema")
        if schema:
            import jsonschema

            try:
                jsonschema.validate(value, schema)
            except jsonschema.ValidationError:
                return False
        return True
    return False


def validate_values(variables: List[Dict[str, Any]], values: Dict[str, Any]) -> None:
    """Refuse values that do not match the declarations. Names only, never values."""
    declared = _declared(list(variables or []))
    values = values or {}
    extra = sorted(set(values) - set(declared))
    if extra:
        raise VarUndeclared(f"not declared by this prompt: {', '.join(extra)}")
    missing = [n for n, d in declared.items()
               if d.get("required", True) and values.get(n) is None]
    if missing:
        raise VarMissing(f"no value for: {', '.join(missing)}")
    wrong = [f"{n} (expects {d.get('type')})" for n, d in declared.items()
             if values.get(n) is not None and not _matches(d, values[n])]
    if wrong:
        raise VarType(f"wrong type for: {', '.join(wrong)}")


class _Redacted:
    """Stands where a value was removed by a `redact` obligation."""

    def __repr__(self) -> str:
        return REDACTED_TEXT


REDACTED_TEXT = "[redacted]"
REDACTED = _Redacted()


def _shown(value: Any) -> str:
    if value is None:
        return ""
    if value is REDACTED:
        return REDACTED_TEXT
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda o: REDACTED_TEXT if o is REDACTED else str(o))


def _fill(parsed: Parsed, values: Dict[str, Any],
          fragments: Dict[Tuple[str, int], str], *, inside_fragment: bool) -> str:
    out: List[str] = []
    for segment in parsed.segments:
        if segment[0] == "text":
            out.append(segment[1])
        elif segment[0] == "blank":
            _, root, path = segment
            value = values.get(root)
            for step in path:
                if value is None or value is REDACTED:
                    break
                if not isinstance(value, dict) or step not in value:
                    raise VarMissing(f"no value for: {root}.{'.'.join(path)}")
                value = value[step]
            out.append(_shown(value))
        else:
            _, key, version = segment
            if inside_fragment:
                raise TemplateInvalid("a fragment cannot use another fragment")
            if (key, version) not in fragments:
                raise FragmentMissing(f"fragment '{key}@{version}' is not available")
            out.append(_fill(parse(fragments[(key, version)]), values, fragments,
                             inside_fragment=True))
    return "".join(out)


def render_text(template: str, variables: List[Dict[str, Any]], values: Dict[str, Any],
                fragments: Dict[Tuple[str, int], str]) -> str:
    """The template with its blanks filled and its fragments put in.

    The same inputs give the same text: nothing here reads a clock, and an
    object is written with its keys in order.
    """
    validate_values(variables, values)
    return _fill(parse(template), dict(values or {}), fragments, inside_fragment=False)


def fill(template: str, values: Dict[str, Any], fragments: Dict[Tuple[str, int], str]) -> str:
    """Fill a template with values that have ALREADY been validated.

    For the renderer, which validates what it was given and then masks what
    policy told it to redact: the masked values no longer match their declared
    types, and must not be validated a second time.
    """
    return _fill(parse(template), dict(values or {}), fragments, inside_fragment=False)
