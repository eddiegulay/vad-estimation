"""Checkpoint save/load: state_dict + config + registry key + git commit
hash + step/epoch, no arbitrary pickled objects (plan §6).
"""

import subprocess
from pathlib import Path

import torch


def _git_commit_hash(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(repo_root)
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_name: str,
    model_config: dict,
    epoch: int,
    step: int,
    repo_root: Path,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_name": model_name,
        "model_config": model_config,
        "epoch": epoch,
        "step": step,
        "git_commit": _git_commit_hash(repo_root),
        "extra": extra or {},
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    path = Path(path)
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {k: v for k, v in payload.items() if k not in ("model_state_dict", "optimizer_state_dict")}
