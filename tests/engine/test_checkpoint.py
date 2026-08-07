from pathlib import Path

import torch
from torch import nn

from vad.engine.checkpoint import load_checkpoint, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _tiny_model():
    return nn.Linear(4, 2)


def test_save_and_load_checkpoint_restores_weights(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "test.pt"

    save_checkpoint(
        ckpt_path, model, optimizer, model_name="tiny", model_config={"a": 1},
        epoch=2, step=100, repo_root=REPO_ROOT, extra={"train_loss": 0.5},
    )

    new_model = _tiny_model()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    meta = load_checkpoint(ckpt_path, new_model, new_optimizer)

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

    assert meta["model_name"] == "tiny"
    assert meta["model_config"] == {"a": 1}
    assert meta["epoch"] == 2
    assert meta["step"] == 100
    assert meta["extra"]["train_loss"] == 0.5


def test_load_checkpoint_without_optimizer(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(
        ckpt_path, model, None, model_name="tiny", model_config={}, epoch=0, step=0, repo_root=REPO_ROOT,
    )

    new_model = _tiny_model()
    meta = load_checkpoint(ckpt_path, new_model)
    assert meta["step"] == 0


def test_checkpoint_records_git_commit_field(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(
        ckpt_path, model, None, model_name="tiny", model_config={}, epoch=0, step=0, repo_root=REPO_ROOT,
    )
    new_model = _tiny_model()
    meta = load_checkpoint(ckpt_path, new_model)
    assert "git_commit" in meta  # may be None if repo has no commits yet -- field must still exist
