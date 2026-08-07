"""Minimal YAML config loading and merging.

No Hydra/OmegaConf — plain dicts loaded from YAML, merged with a small
recursive rule. Per-consumer dataclasses (ModelConfig, TrainConfig, ...) are
defined next to the code that uses them, not here.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. `override` wins on conflicts.

    Dict values are merged key-by-key; any other value (including lists) is
    replaced wholesale by the override's value.
    """
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load and merge YAML configs in order; later paths override earlier ones."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    return merged
