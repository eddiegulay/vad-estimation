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
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vad.config import load_config, load_yaml  # noqa: E402
from vad.data.collate import collate_batch  # noqa: E402
from vad.data.dataset import VADDataset  # noqa: E402
from vad.data.manifest import load_index, read_jsonl  # noqa: E402
from vad.engine.checkpoint import save_checkpoint  # noqa: E402
from vad.engine.trainer import Trainer, build_lr_scheduler, compute_class_weights  # noqa: E402
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


def main(
    epochs: int | None = None, run_name: str = "crnn_v1", subset: int | None = None, model_config: str = "crnn_v1"
) -> int:
    data_cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml", REPO_ROOT / "configs" / "data" / "default.yaml"
    )
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / f"{model_config}.yaml")
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

    num_epochs = epochs if epochs is not None else train_cfg["schedule"]["epochs"]
    steps_per_epoch = len(train_loader) if subset is None else min(len(train_loader), subset)
    total_steps = num_epochs * steps_per_epoch
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=train_cfg["schedule"]["warmup_steps"],
        total_steps=total_steps,
        min_lr_ratio=train_cfg["schedule"].get("min_lr_ratio", 0.01),
    )

    trainer = Trainer(
        model, device, optimizer, pos_weight, neg_weight,
        max_chunks=max_chunks, grad_clip_norm=train_cfg.get("grad_clip_norm"), lr_scheduler=scheduler,
    )

    checkpoint_dir = REPO_ROOT / train_cfg["checkpoint"]["dir"] / run_name
    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tb"))
    best_val_f1 = float("-inf")

    epoch_progress = tqdm(range(num_epochs), desc="epochs", unit="epoch")
    step = 0
    for epoch in epoch_progress:
        train_dataset.set_epoch(epoch)
        epoch_losses = []
        t0 = time.time()
        train_progress = tqdm(
            train_loader, desc=f"epoch {epoch} train", unit="step", total=steps_per_epoch, leave=False
        )
        for batch in train_progress:
            loss = trainer.train_step(batch, chunk_samples)
            if loss is not None:
                assert loss == loss, "NaN loss detected"
                epoch_losses.append(loss)
                step += 1
                train_progress.set_postfix(loss=f"{loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if subset is not None and step >= subset:
                break
        train_progress.close()
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        val_progress = tqdm(val_loader, desc=f"epoch {epoch} val", unit="batch", leave=False)
        val_metrics = trainer.eval_epoch(val_progress, chunk_samples)
        val_progress.close()
        val_loss = val_metrics["val_loss"]
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}", val_f1=f"{val_metrics['val_f1']:.4f}", val_auroc=f"{val_metrics['val_auroc']:.4f}"
        )
        tqdm.write(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['val_f1']:.4f} val_auroc={val_metrics['val_auroc']:.4f} "
            f"lr={current_lr:.2e} ({elapsed:.1f}s, {step} steps)"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/val_f1", val_metrics["val_f1"], epoch)
        writer.add_scalar("metrics/val_auroc", val_metrics["val_auroc"], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        extra = {"train_loss": train_loss, "val_loss": val_loss, **{k: v for k, v in val_metrics.items() if k != "val_loss"}}
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, model_cfg["architecture"], model_cfg,
            epoch, step, REPO_ROOT, extra=extra,
        )
        if val_metrics["val_f1"] == val_metrics["val_f1"] and val_metrics["val_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1"]
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, model_cfg["architecture"], model_cfg,
                epoch, step, REPO_ROOT, extra=extra,
            )
            tqdm.write(f"  new best val_f1={best_val_f1:.4f} -> best.pt")

        if subset is not None and step >= subset:
            break

    epoch_progress.close()
    writer.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--subset", type=int, default=None, help="stop after this many optimizer steps (smoke runs)"
    )
    parser.add_argument("--run-name", type=str, default="crnn_v1")
    parser.add_argument(
        "--model-config", type=str, default="crnn_v1", help="name under configs/model/ (without .yaml)"
    )
    args = parser.parse_args()
    raise SystemExit(
        main(epochs=args.epochs, run_name=args.run_name, subset=args.subset, model_config=args.model_config)
    )
