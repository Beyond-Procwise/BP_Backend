from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class DerivationRule:
    rule_id: str
    target_field: str
    inputs: list[str]
    compute: Callable[[dict], Any]


REGISTRY: list[DerivationRule] = []


def rule(rule_id: str, target_field: str, inputs: list[str]):
    def _decorator(fn: Callable[[dict], Any]):
        REGISTRY.append(DerivationRule(rule_id, target_field, inputs, fn))
        return fn
    return _decorator


def clear_registry():
    """For tests. Removes all registered rules."""
    REGISTRY.clear()


def reset_registry_to_builtins():
    """Re-import the rule modules to re-register builtin rules after a clear."""
    REGISTRY.clear()
    from src.services.structural_extractor import derivation_rules  # noqa: F401
