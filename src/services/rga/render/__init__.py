"""Renderers, and the record that makes one reproducible.

A renderer is deterministic and takes no decisions of its own: it is handed a
Fact Pack, a Style Brief and an AST, and everything it draws comes from one of
the three. Confidence badges and provenance footnotes are drawn HERE and not by
a model — §5 of the brief — because a badge a model can choose to omit is not a
disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class RenderedArtefact:
    """The bytes, plus everything needed to prove where they came from.

    The four hashes are the reproducibility record DoD12 rests on: given these,
    the same artefact can be rebuilt without calling a model. A field left blank
    is a post-check failure, not a shrug.
    """

    content: bytes
    media_type: str
    renderer: str
    renderer_version: str
    pack_id: str
    pack_hash: str
    style_version: str
    ast_hash: str
    style_provenance: Dict[str, str] = field(default_factory=dict)
    style_disclosure: str = ""
    # "3 of 8 measures not assessed" — a count of the report's own coverage.
    # Required on the page, and not a measurement of the corpus, so it is
    # recorded here to be stripped before the untraced-figure scan.
    coverage_disclosure: str = ""

    def recorded(self) -> bool:
        """Whether the reproducibility record is complete."""
        return all([
            self.pack_hash, self.style_version, self.ast_hash,
            self.renderer_version,
        ])

    def chrome_strings(self) -> tuple[str, ...]:
        """Text the renderer itself printed that is provenance, not measurement.

        The post-check removes these before hunting for untraced figures. A
        sha256 is mostly digits, and counting them as fabricated figures would
        fail every report for disclosing its own hash.
        """
        return tuple(s for s in (
            self.pack_hash, self.pack_hash[:12], self.style_version,
            self.ast_hash, self.ast_hash[:12], self.renderer_version,
            self.pack_id, self.style_disclosure, self.coverage_disclosure,
        ) if s)
