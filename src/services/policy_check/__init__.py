"""A policy verdict computed in code, from the governed policy, never from a model's memory."""
from . import evaluate  # the module; call evaluate.evaluate(request, governance, config)

__all__ = ["evaluate"]
