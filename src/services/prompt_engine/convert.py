"""Turn wording imported as it was into a typed template, without changing a word.

The older prompts mark a blank as ``{name}`` and fill it with ``str.replace``.
A typed template marks it ``{{name}}`` and declares it. Converting is
mechanical, and is refused if it could change what the prompt says.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from services.prompt_engine import templates


class NotConvertible(Exception):
    pass


def braces_to_typed(text: str, names: Iterable[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """``(typed template, declarations)`` for ``text`` whose blanks are ``{name}``.

    Every blank becomes a required string, because that is what the older code
    put there: it turned each value into text before filling it in.
    """
    if "{{" in text:
        raise NotConvertible("the wording already contains '{{', which a typed "
                             "template would read as a blank")
    out = text
    declared: List[Dict[str, Any]] = []
    for name in names:
        token = "{" + name + "}"
        if token in out:
            out = out.replace(token, "{{" + name + "}}")
            declared.append({"name": name, "type": "string"})
    if not declared:
        raise NotConvertible("none of the named blanks appears in the wording")
    problems = templates.problems_on_save(out, declared, lambda key, version: None)
    if problems:
        raise NotConvertible("; ".join(problems))
    if typed_to_braces(out, [d["name"] for d in declared]) != text:
        raise NotConvertible("converting it back does not give the original wording")
    return out, declared


def typed_to_braces(template: str, names: Iterable[str]) -> str:
    """The same wording with its blanks marked the older way."""
    out = template
    for name in names:
        out = out.replace("{{" + name + "}}", "{" + name + "}")
    return out
