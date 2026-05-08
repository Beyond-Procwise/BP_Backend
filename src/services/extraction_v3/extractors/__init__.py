"""Extractor registry. Importing concrete extractor modules has the side effect
of registering them in yaml_schema.registry."""
from .base import Extractor

__all__ = ["Extractor"]
