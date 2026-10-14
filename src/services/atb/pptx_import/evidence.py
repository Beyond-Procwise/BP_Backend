"""Where every value in a pack came from.

A fact in this codebase cannot be constructed without a provenance id, and a style token should
answer to the same standard: "#172033 is the ink" is not a finding, "#172033 paints 619 fills and
804 text runs" is.
"""
from __future__ import annotations

from typing import Any


class Evidence:
    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}
        self._incidental: list[dict[str, Any]] = []
        self._ignored: dict[str, dict[str, Any]] = {}
        self._assumed: list[dict[str, Any]] = []

    def record(self, path: str, value: Any, **facts: Any) -> None:
        self._values[path] = {'value': value, **facts}

    def assumed(self, path: str, value: Any, why: str) -> None:
        """A value the deck did not give us. Recorded AND surfaced as a problem.

        An evidence note alone is not enough: the review screen shows problems, and a pack built
        entirely of assumptions looked exactly like a measured one. Every fallback goes through
        here so "an absent measurement must be stated" is true where a human actually looks.
        """
        self._values[path] = {'value': value, 'assumed': True, 'why': why}
        self._assumed.append({'path': path, 'value': value, 'why': why})

    @property
    def assumptions(self) -> list[dict[str, Any]]:
        return list(self._assumed)

    def incidental(self, kind: str, value: Any, why: str) -> None:
        self._incidental.append({'kind': kind, 'value': value, 'why': why})

    def ignored(self, what: str, value: Any, why: str) -> None:
        self._ignored[what] = {'value': value, 'why': why}

    def as_dict(self) -> dict[str, Any]:
        return {
            'values': dict(self._values),
            'incidental': list(self._incidental),
            'ignored': dict(self._ignored),
            'assumed': list(self._assumed),
        }
