"""The filter objects callers build, so the application needs no vendor client library for them.

They are named like the Qdrant models the callers were written against (``Filter``, ``FieldCondition``, ``MatchValue``, ``MatchAny``) so the
existing filter-building code did not have to change shape. ``to_store_filter`` (filters.py) turns them into the VectorStore's own filter mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional


@dataclass
class MatchValue:
    value: Any


@dataclass
class MatchAny:
    any: List[Any]


@dataclass
class FieldCondition:
    key: str
    match: Any


@dataclass
class Filter:
    must: Optional[List[Any]] = None
    must_not: Optional[List[Any]] = None
    should: Optional[List[Any]] = None


#: drop-in for ``from qdrant_client import models`` in code that only builds filters
models = SimpleNamespace(Filter=Filter, FieldCondition=FieldCondition, MatchValue=MatchValue, MatchAny=MatchAny)
