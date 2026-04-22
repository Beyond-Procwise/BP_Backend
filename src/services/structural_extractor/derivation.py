from dataclasses import dataclass
from typing import Any, Callable

from src.services.structural_extractor.types import ExtractedValue


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


def resolve_all(header: dict[str, ExtractedValue], doc_type: str,
                max_passes: int = 10) -> dict[str, ExtractedValue]:
    """Resolve all derivable fields in topological order. Mutates/extends header."""
    out = dict(header)
    for _pass in range(max_passes):
        progress = False
        for r in REGISTRY:
            if r.target_field in out:
                continue  # already set (extracted or previously derived)
            # Check all inputs are resolved
            inputs_resolved = {}
            missing = False
            for input_name in r.inputs:
                if input_name not in out:
                    missing = True
                    break
                inputs_resolved[input_name] = out[input_name].value
            if missing:
                continue
            try:
                result = r.compute(inputs_resolved)
            except Exception:
                continue
            if result is None:
                continue
            out[r.target_field] = ExtractedValue(
                value=result,
                provenance="derived",
                derivation_trace={"rule_id": r.rule_id, "inputs": inputs_resolved},
                source="derivation_registry",
                confidence=1.0,
                attempt=1,
            )
            progress = True
        if not progress:
            break
    return out
