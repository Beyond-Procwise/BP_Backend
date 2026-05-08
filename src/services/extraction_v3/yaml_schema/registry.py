from typing import Callable

_extractors: dict[str, Callable] = {}
_invariants: dict[str, Callable] = {}


def register_extractor(name: str):
    def deco(cls):
        _extractors[name] = cls
        return cls
    return deco


def register_invariant(name: str):
    def deco(fn):
        _invariants[name] = fn
        return fn
    return deco


def get_extractor(name: str): return _extractors[name]
def get_invariant(name: str): return _invariants[name]
def known_extractors() -> set[str]: return set(_extractors)
def known_invariants() -> set[str]: return set(_invariants)
