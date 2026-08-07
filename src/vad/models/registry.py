"""Tiny model registry so new architectures (TCN, FSMN, ...) plug into the
same config-driven training/eval/export scaffolding without touching it —
just add `models/<arch>.py` + `configs/model/<arch>.yaml` (plan §2/§6).
"""

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def _decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def build_model(name: str, config: dict):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}', available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)
