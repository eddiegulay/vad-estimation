"""Importing this package registers every built-in architecture into
MODEL_REGISTRY (plan §2) -- add new architectures here as they're built.
"""

from vad.models import crnn  # noqa: F401
from vad.models.registry import MODEL_REGISTRY, build_model, register_model

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]
