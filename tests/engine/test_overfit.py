"""Phase 6 gate (plan §6): overfit ~8 real examples from the actual data
pipeline (VADDataset against the real train manifest, real CRNN model) to
near-zero loss before any full training run is attempted. This is a
pipeline-correctness check, not a model-quality check -- if it fails, the
bug is in data/labels/model wiring, not in "needs more training."
"""

from pathlib import Path

import pytest
import torch

from vad.config import load_yaml
from vad.data.collate import collate_batch
from vad.data.dataset import VADDataset
from vad.data.manifest import load_index
from vad.engine.trainer import Trainer, masked_bce_loss
from vad.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = REPO_ROOT / "data_cache"
TRAIN_MANIFEST = CACHE_ROOT / "manifests" / "train.jsonl"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "crnn_v1.yaml"

N_EXAMPLES = 8
N_STEPS = 400
LR = 3e-3


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not yet generated")
def test_overfit_eight_real_examples_reaches_near_zero_loss():
    model_config = load_yaml(MODEL_CONFIG_PATH)
    sample_rate = 16000
    hop_s = model_config["frontend"]["chunk_samples"] / sample_rate
    chunk_samples = model_config["frontend"]["chunk_samples"]

    esc50_index = load_index(CACHE_ROOT / "index" / "esc50_index.json")
    rir_index = load_index(CACHE_ROOT / "index" / "rir_index.json")

    dataset = VADDataset(
        TRAIN_MANIFEST, CACHE_ROOT, sample_rate, hop_s, esc50_index, rir_index, run_seed=0
    )
    dataset.set_epoch(0)

    # Fixed batch, built once -- fetched from the real pipeline, then reused
    # every step (this test targets wiring correctness, not augmentation
    # robustness, so the input must not change between steps).
    items = [dataset[i] for i in range(N_EXAMPLES)]
    batch = collate_batch(items)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_model(model_config["architecture"], model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    trainer = Trainer(model, device, optimizer, pos_weight=1.0, neg_weight=1.0)

    losses = []
    for _ in range(N_STEPS):
        loss = trainer.train_step(batch, chunk_samples)
        assert loss == loss, "NaN loss during overfit run"
        losses.append(loss)

    print(f"\noverfit loss: start={losses[0]:.4f} end={losses[-1]:.4f} min={min(losses):.4f}")

    assert losses[-1] < losses[0] * 0.1, (
        f"loss did not drop by >90% over {N_STEPS} steps on {N_EXAMPLES} fixed examples "
        f"(start={losses[0]:.4f}, end={losses[-1]:.4f}) -- likely a data/label/model wiring bug"
    )
    assert losses[-1] < 0.15, f"final overfit loss {losses[-1]:.4f} not near zero"


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not yet generated")
def test_overfit_predictions_match_labels_after_convergence():
    """A stronger correctness check than loss alone: after overfitting,
    predicted labels should match ground truth on the same fixed batch.
    """
    model_config = load_yaml(MODEL_CONFIG_PATH)
    sample_rate = 16000
    hop_s = model_config["frontend"]["chunk_samples"] / sample_rate
    chunk_samples = model_config["frontend"]["chunk_samples"]

    esc50_index = load_index(CACHE_ROOT / "index" / "esc50_index.json")
    rir_index = load_index(CACHE_ROOT / "index" / "rir_index.json")

    dataset = VADDataset(
        TRAIN_MANIFEST, CACHE_ROOT, sample_rate, hop_s, esc50_index, rir_index, run_seed=1
    )
    dataset.set_epoch(0)
    items = [dataset[i] for i in range(N_EXAMPLES)]
    batch = collate_batch(items)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_model(model_config["architecture"], model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    trainer = Trainer(model, device, optimizer, pos_weight=1.0, neg_weight=1.0)

    for _ in range(N_STEPS):
        trainer.train_step(batch, chunk_samples)

    model.eval()
    with torch.no_grad():
        result = trainer._forward_batch(batch, chunk_samples)
        probs, labels, mask = result
        preds = (probs > 0.5).long()
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total

    print(f"\noverfit frame accuracy on fixed batch: {accuracy:.4f}")
    assert accuracy > 0.95, f"frame accuracy {accuracy:.4f} too low after overfitting a fixed batch"
