"""L1 — declarative PatternRegistry loaded from extraction_schemas/<doctype>.yaml.

Each field's patterns are compiled once per doc_type. The registry is read by
pattern_extractor.run_pattern_extractor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec, load_doc_schema


@dataclass(frozen=True)
class CompiledPattern:
    field: str
    name: str
    anchor_re: re.Pattern
    value_re: re.Pattern
    max_span_after_anchor_chars: int
    prior_confidence: float


@dataclass(frozen=True)
class FieldMeta:
    name: str
    type: str
    required: bool
    threshold: float
    ner_type_check: str
    db_column: str | None


class PatternRegistry:
    """Per-doc-type compiled regex registry plus field metadata."""

    def __init__(self, doc_type: str, *, _schema: DocSchema | None = None) -> None:
        self.doc_type = doc_type
        self._schema = _schema if _schema is not None else load_doc_schema(doc_type)
        self._by_field: dict[str, list[CompiledPattern]] = {}
        self._meta: dict[str, FieldMeta] = {}
        # (field, pattern_name) -> the prior_confidence the YAML author wrote. Every
        # apply_observed() is computed against THIS, never against whatever the last
        # application left behind — see apply_observed for why that matters.
        self._baseline: dict[tuple[str, str], float] = {}
        # (field, pattern_name) currently sitting below their baseline. Only used to tell
        # a true cold start (nothing measured, nothing to undo) from an empty map that
        # arrives AFTER a demotion and should restore it.
        self._demoted: set[tuple[str, str]] = set()
        self._compile()

    def _compile(self) -> None:
        for f in self._schema.fields:
            self._meta[f.name] = FieldMeta(
                name=f.name, type=f.type, required=f.required,
                threshold=f.confidence_threshold,
                ner_type_check=f.judge.ner_type_check,
                db_column=f.db_column,
            )
            patterns: list[CompiledPattern] = []
            for p in f.patterns:
                patterns.append(CompiledPattern(
                    field=f.name, name=p.name,
                    anchor_re=re.compile(p.anchor),
                    value_re=re.compile(p.value),
                    max_span_after_anchor_chars=p.max_span_after_anchor_chars,
                    prior_confidence=p.prior_confidence,
                ))
            # Order patterns by prior_confidence descending so the highest-prior
            # match is found first (pattern_extractor stops at first hit per pattern,
            # but caller iterates all patterns).
            patterns.sort(key=lambda cp: cp.prior_confidence, reverse=True)
            self._by_field[f.name] = patterns
            for cp in patterns:
                self._baseline[(f.name, cp.name)] = cp.prior_confidence

        # Compile line-item patterns the same way under "line_items[].<field>"
        if self._schema.line_items:
            for lf in self._schema.line_items.fields:
                key = f"line_items.{lf.name}"
                self._meta[key] = FieldMeta(
                    name=key, type=lf.type, required=lf.required,
                    threshold=lf.confidence_threshold,
                    ner_type_check=lf.judge.ner_type_check,
                    db_column=lf.db_column,
                )
                self._by_field[key] = [
                    CompiledPattern(
                        field=key, name=p.name,
                        anchor_re=re.compile(p.anchor),
                        value_re=re.compile(p.value),
                        max_span_after_anchor_chars=p.max_span_after_anchor_chars,
                        prior_confidence=p.prior_confidence,
                    )
                    for p in lf.patterns
                ]

    def fields(self) -> Iterable[str]:
        return self._by_field.keys()

    def header_fields(self) -> Iterable[str]:
        return [f for f in self._by_field if not f.startswith("line_items.")]

    def patterns_for(self, field: str) -> list[CompiledPattern]:
        return list(self._by_field.get(field, ()))

    def meta(self, field: str) -> FieldMeta:
        return self._meta[field]

    def threshold(self, field: str) -> float:
        return self._meta[field].threshold

    def apply_observed(self, accuracy: dict) -> int:
        """Replace hand-set priors with measured agreement rates. Returns patterns changed.

        A prior is somebody's opening guess at how much a rule deserves to be believed. Once
        there is a measurement — how often a human let this reader's answer stand — the
        measurement is simply better information.

        It may only ever lower trust. Raising a reader above the prior a human set would let
        learning promote documents that used to stop for review, which is the one failure
        mode this must not have: the pipeline may become more cautious on its own, never
        bolder.

        Every application starts from the YAML baseline, NOT from whatever the previous
        application left on the object. This registry is a process-wide singleton
        (get_registry) and the accuracy map behind it is refreshed on a timer, so comparing
        a fresh rate against an already-demoted prior would ratchet: a reader knocked down
        by one bad 15-minute window could never come back up (0.4 is not < 0.4), and two
        workers started at different times would read the same document with different
        priors. Against the baseline the result is a pure function of (YAML, measurement) —
        a reader whose rate recovers returns to exactly the prior a human wrote for it, and
        never one point above it.
        """
        if not accuracy and not self._demoted:
            return 0                       # true cold start: do not touch _by_field at all
        changed = 0
        demoted: set[tuple[str, str]] = set()
        for field, patterns in self._by_field.items():
            for i, pat in enumerate(patterns):
                baseline = self._baseline.get((field, pat.name), pat.prior_confidence)
                rate = accuracy.get((self.doc_type, field, pat.name))
                if rate is not None and float(rate) < baseline:
                    target = float(rate)
                    demoted.add((field, pat.name))
                else:
                    target = baseline      # unmeasured, or measured at/above the prior
                if target == pat.prior_confidence:
                    continue
                patterns[i] = replace(pat, prior_confidence=target)
                changed += 1
            # Order is the preference between readers — the extractor tries the highest
            # prior first — so a demoted reader has to actually move.
            patterns.sort(key=lambda cp: cp.prior_confidence, reverse=True)
        self._demoted = demoted
        return changed

    def is_required(self, field: str) -> bool:
        return self._meta[field].required

    @property
    def schema(self) -> DocSchema:
        return self._schema


# Process-wide cache: registries are pure functions of YAML on disk.
_CACHE: dict[str, PatternRegistry] = {}


def get_registry(doc_type: str) -> PatternRegistry:
    if doc_type not in _CACHE:
        _CACHE[doc_type] = PatternRegistry(doc_type)
    return _CACHE[doc_type]


def clear_cache() -> None:
    """Reset the registry cache. Used by tests after editing YAML on disk."""
    _CACHE.clear()
