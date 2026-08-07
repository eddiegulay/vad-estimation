"""Phase 6 gate (plan §6): a short run on varying real batches shows a
downward loss trend over a few hundred steps, with no NaN/inf and a working
checkpoint save/reload -- distinct from test_overfit.py's fixed-batch check.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vad.config import load_yaml
from vad.data.collate import collate_batch
from vad.data.dataset import VADDataset
from vad.data.manifest import load_index
from vad.engine.checkpoint import load_checkpoint, save_checkpoint
from vad.engine.trainer import Trainer
from vad.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = REPO_ROOT / "data_cache"
TRAIN_MANIFEST = CACHE_ROOT / "manifests" / "train.jsonl"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "crnn_v1.yaml"

N_STEPS = 150
BATCH_SIZE = 16
MAX_TRAIN_DURATION_S = 10.0


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not yet generated")
def test_short_run_loss_trends_down_and_checkpoint_roundtrips(tmp_path):
    model_config = load_yaml(MODEL_CONFIG_PATH)
    sample_rate = 16000
    hop_s = model_config["frontend"]["chunk_samples"] / sample_rate
    chunk_samples = model_config["frontend"]["chunk_samples"]
    max_chunks = int(MAX_TRAIN_DURATION_S / hop_s)

    esc50_index = load_index(CACHE_ROOT / "index" / "esc50_index.json")
    rir_index = load_index(CACHE_ROOT / "index" / "rir_index.json")

    dataset = VADDataset(
        TRAIN_MANIFEST, CACHE_ROOT, sample_rate, hop_s, esc50_index, rir_index, run_seed=0
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collate_batch
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_model(model_config["architecture"], model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, device, optimizer, pos_weight=1.0, neg_weight=1.0, max_chunks=max_chunks)

    losses = []
    loader_iter = iter(loader)
    while len(losses) < N_STEPS:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            continue
        loss = trainer.train_step(batch, chunk_samples)
        if loss is not None:
            assert loss == loss and loss not in (float("inf"), float("-inf")), "NaN/inf loss"
            losses.append(loss)

    window = max(1, N_STEPS // 5)
    early_avg = sum(losses[:window]) / window
    late_avg = sum(losses[-window:]) / window
    print(f"\nshort-run loss: early_avg={early_avg:.4f} late_avg={late_avg:.4f} over {N_STEPS} steps")
    assert late_avg < early_avg, (
        f"loss did not trend down over {N_STEPS} varying-batch steps "
        f"(early={early_avg:.4f}, late={late_avg:.4f})"
    )

    ckpt_path = tmp_path / "smoke.pt"
    save_checkpoint(
        ckpt_path, model, optimizer, model_config["architecture"], model_config,
        epoch=0, step=N_STEPS, repo_root=REPO_ROOT, extra={"late_avg_loss": late_avg},
    )

    reloaded_model = build_model(model_config["architecture"], model_config).to(device)
    meta = load_checkpoint(ckpt_path, reloaded_model, map_location=str(device))
    assert meta["step"] == N_STEPS

    for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(p1, p2)
