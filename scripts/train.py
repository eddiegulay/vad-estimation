#!/usr/bin/env python3
"""Phase 6 — training entrypoint (plan §6).

Usage:
    python scripts/train.py                      # full run per configs/train/default.yaml
    python scripts/train.py --subset 300          # smoke run: stop after 300 optimizer steps
    python scripts/train.py --epochs 1 --subset 50
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vad.config import load_config, load_yaml  # noqa: E402
from vad.data.collate import collate_batch  # noqa: E402
from vad.data.dataset import VADDataset  # noqa: E402
from vad.data.manifest import load_index, read_jsonl  # noqa: E402
from vad.engine.checkpoint import save_checkpoint  # noqa: E402
from vad.engine.trainer import Trainer, compute_class_weights  # noqa: E402
from vad.models import build_model  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(name)


def build_loader(
    manifest_path, cache_root, sample_rate, hop_s, esc50_index, rir_index, run_seed,
    batch_size, num_workers, shuffle,
):
    dataset = VADDataset(manifest_path, cache_root, sample_rate, hop_s, esc50_index, rir_index, run_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
        collate_fn=collate_batch,
        shuffle=shuffle,
    )
    return dataset, loader


def main(epochs: int | None = None, run_name: str = "crnn_v1", subset: int | None = None) -> int:
    data_cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml", REPO_ROOT / "configs" / "data" / "default.yaml"
    )
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / "crnn_v1.yaml")
    train_cfg = load_yaml(REPO_ROOT / "configs" / "train" / "default.yaml")

    cache_root = REPO_ROOT / data_cfg["cache_root"]
    manifests_dir = REPO_ROOT / data_cfg["manifests_dir"]
    sample_rate = data_cfg["sample_rate"]
    hop_s = model_cfg["frontend"]["chunk_samples"] / sample_rate

    esc50_index = load_index(cache_root / "index" / "esc50_index.json")
    rir_index = load_index(cache_root / "index" / "rir_index.json")

    device = resolve_device(train_cfg["device"])
    print(f"device: {device}")

    dl_cfg = train_cfg["dataloader"]
    train_dataset, train_loader = build_loader(
        manifests_dir / "train.jsonl", cache_root, sample_rate, hop_s, esc50_index, rir_index,
        train_cfg["seed"], dl_cfg["batch_size"], dl_cfg["num_workers"], shuffle=True,
    )
    _, val_loader = build_loader(
        manifests_dir / "val.jsonl", cache_root, sample_rate, hop_s, esc50_index, rir_index,
        train_cfg["seed"], dl_cfg["batch_size"], dl_cfg["num_workers"], shuffle=False,
    )

    train_records = read_jsonl(manifests_dir / "train.jsonl")
    pos_weight, neg_weight = compute_class_weights(train_records)
    print(f"class weights: pos={pos_weight:.3f} neg={neg_weight:.3f}")

    model = build_model(model_cfg["architecture"], model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["optim"]["lr"], weight_decay=train_cfg["optim"]["weight_decay"]
    )
    chunk_samples = model_cfg["frontend"]["chunk_samples"]
    max_chunks = None
    if train_cfg.get("max_train_duration_s"):
        max_chunks = max(1, int(train_cfg["max_train_duration_s"] / hop_s))
    trainer = Trainer(model, device, optimizer, pos_weight, neg_weight, max_chunks=max_chunks)

    checkpoint_dir = REPO_ROOT / train_cfg["checkpoint"]["dir"] / run_name
    num_epochs = epochs if epochs is not None else train_cfg["schedule"]["epochs"]

    step = 0
    for epoch in range(num_epochs):
        train_dataset.set_epoch(epoch)
        epoch_losses = []
        t0 = time.time()
        for batch in train_loader:
            loss = trainer.train_step(batch, chunk_samples)
            if loss is not None:
                assert loss == loss, "NaN loss detected"
                epoch_losses.append(loss)
                step += 1
            if subset is not None and step >= subset:
                break
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        val_losses = []
        for batch in val_loader:
            loss = trainer.eval_step(batch, chunk_samples)
            if loss is not None:
                val_losses.append(loss)
        val_loss = sum(val_losses) / max(1, len(val_losses)) if val_losses else float("nan")

        elapsed = time.time() - t0
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"({elapsed:.1f}s, {step} steps)"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, model_cfg["architecture"], model_cfg,
            epoch, step, REPO_ROOT, extra={"train_loss": train_loss, "val_loss": val_loss},
        )

        if subset is not None and step >= subset:
            break

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--subset", type=int, default=None, help="stop after this many optimizer steps (smoke runs)"
    )
    parser.add_argument("--run-name", type=str, default="crnn_v1")
    args = parser.parse_args()
    raise SystemExit(main(epochs=args.epochs, run_name=args.run_name, subset=args.subset))
