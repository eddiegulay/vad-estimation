"""Training loop: masked BCE loss (padded regions excluded via
`label_mask`), optional balanced class weighting for the speech/non-speech
imbalance, one train/eval step per batch (plan §6).
"""

import math

import numpy as np
import torch

from vad.eval.metrics import frame_auroc, frame_precision_recall_f1


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over `warmup_steps`, then cosine decay to
    `min_lr_ratio` * base_lr over the remaining `total_steps - warmup_steps`.
    """
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_class_weights(train_records: list[dict]) -> tuple[float, float]:
    """Balanced positive/negative weights from a manifest's `label_intervals`
    (pure metadata, no audio decode needed).
    """
    speech_s = 0.0
    total_s = 0.0
    for rec in train_records:
        for start, end, label in rec["label_intervals"]:
            duration = end - start
            total_s += duration
            if label == 1:
                speech_s += duration

    if total_s <= 0 or speech_s <= 0 or speech_s >= total_s:
        return 1.0, 1.0

    speech_frac = speech_s / total_s
    pos_weight = 0.5 / speech_frac
    neg_weight = 0.5 / (1.0 - speech_frac)
    return pos_weight, neg_weight


def masked_bce_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    probs = probs.clamp(eps, 1.0 - eps)
    labels = labels.float()
    per_element = -(
        labels * torch.log(probs) * pos_weight + (1 - labels) * torch.log(1 - probs) * neg_weight
    )
    per_element = per_element * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return per_element.sum() / denom


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        max_chunks: int | None = None,
        grad_clip_norm: float | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        # Random-crop long sequences to at most `max_chunks` chunks before the
        # forward pass -- train_step only. Observed MPS throughput: nn.GRU's
        # per-timestep dispatch overhead dominates on long sequences (AMI's
        # 20s windows, LibriSpeech-concat examples up to ~60s); capping
        # sequence length is a training-time-only mitigation. eval_step never
        # crops, so val metrics reflect full-sequence behavior; see ROADMAP
        # known issues.
        self.max_chunks = max_chunks
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler

    def _crop_to_max_chunks(self, waveform, labels, mask, chunk_samples, num_chunks):
        if self.max_chunks is None or num_chunks <= self.max_chunks:
            return waveform, labels, mask, num_chunks
        start_chunk = int(torch.randint(0, num_chunks - self.max_chunks + 1, (1,)).item())
        start_sample = start_chunk * chunk_samples
        end_sample = start_sample + self.max_chunks * chunk_samples
        waveform = waveform[:, start_sample:end_sample]
        labels = labels[:, start_chunk : start_chunk + self.max_chunks]
        mask = mask[:, start_chunk : start_chunk + self.max_chunks]
        return waveform, labels, mask, self.max_chunks

    def _forward_batch(self, batch: dict, chunk_samples: int, crop: bool = False):
        waveform = batch["waveform"].to(self.device)
        labels = batch["labels"].to(self.device)
        mask = batch["label_mask"].to(self.device)

        total_len = waveform.shape[1]
        usable_len = (total_len // chunk_samples) * chunk_samples
        if usable_len == 0:
            return None
        waveform = waveform[:, :usable_len]
        num_chunks = usable_len // chunk_samples

        if crop:
            waveform, labels, mask, num_chunks = self._crop_to_max_chunks(
                waveform, labels, mask, chunk_samples, num_chunks
            )

        probs = self.model.forward_full(waveform)  # [B, num_chunks]

        # Source audio duration may not land on an exact chunk boundary --
        # trim probs/labels/mask to their shared length.
        n = min(num_chunks, labels.shape[1])
        if n == 0:
            return None
        return probs[:, :n], labels[:, :n], mask[:, :n]

    def train_step(self, batch: dict, chunk_samples: int) -> float | None:
        self.model.train()
        result = self._forward_batch(batch, chunk_samples, crop=True)
        if result is None:
            return None
        probs, labels, mask = result
        loss = masked_bce_loss(probs, labels, mask, self.pos_weight, self.neg_weight)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, batch: dict, chunk_samples: int) -> float | None:
        self.model.eval()
        result = self._forward_batch(batch, chunk_samples, crop=False)
        if result is None:
            return None
        probs, labels, mask = result
        loss = masked_bce_loss(probs, labels, mask, self.pos_weight, self.neg_weight)
        return loss.item()

    @torch.no_grad()
    def eval_epoch(self, val_loader, chunk_samples: int) -> dict:
        """One pass over `val_loader`: full-sequence loss plus masked
        frame-level F1/AUROC, in a single forward pass per batch (avoids a
        second full pass over the val set just to score it).
        """
        self.model.eval()
        losses = []
        all_probs, all_labels, all_mask = [], [], []
        for batch in val_loader:
            result = self._forward_batch(batch, chunk_samples, crop=False)
            if result is None:
                continue
            probs, labels, mask = result
            loss = masked_bce_loss(probs, labels, mask, self.pos_weight, self.neg_weight)
            losses.append(loss.item())
            all_probs.append(probs.reshape(-1).cpu().numpy())
            all_labels.append(labels.reshape(-1).cpu().numpy())
            all_mask.append(mask.reshape(-1).cpu().numpy())

        result = {"val_loss": sum(losses) / max(1, len(losses)) if losses else float("nan")}
        if not all_probs:
            result.update({"val_f1": float("nan"), "val_auroc": float("nan"), "val_accuracy": float("nan")})
            return result

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        mask = np.concatenate(all_mask).astype(bool)
        probs, labels = probs[mask], labels[mask]

        preds = (probs > 0.5).astype(int)
        frame_metrics = frame_precision_recall_f1(preds, labels)
        result["val_f1"] = frame_metrics["f1"]
        result["val_accuracy"] = frame_metrics["accuracy"]
        result["val_auroc"] = frame_auroc(probs, labels)
        return result
