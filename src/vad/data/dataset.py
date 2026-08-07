"""Streaming, memory-safe VADDataset (plan §3/§6).

Reads a manifest.jsonl at init (cheap — a few hundred bytes per example).
All audio decode, concatenation, and augmentation happens per-item inside
`__getitem__`, which runs in DataLoader worker processes — no corpus is
ever bulk-loaded into RAM. AMI windows are read via a partial `soundfile`
read (`start`/`frames`), never the whole meeting file.

Augmentation realization: if `manifest_seed` is set (val/test), the
per-item RNG is seeded from it and the pre-resolved concrete augmentation
values in the record are used directly (deterministic). If unset (train),
the per-item RNG is seeded from `(run_seed, epoch, index)` and a fresh
realization is resolved from the template each epoch (plan §3.4).
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from vad.augment.gain import random_gain
from vad.augment.gaps import make_gap_audio
from vad.augment.noise_mix import mix_at_snr
from vad.augment.reverb import convolve_rir
from vad.data.manifest import read_jsonl, resolve_augmentation
from vad.labels.intervals import to_frames
from vad.labels.synthetic import trim_internal_silence


class VADDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cache_root: str | Path,
        sample_rate: int,
        hop_s: float,
        esc50_index: list[dict] | None = None,
        rir_index: list[dict] | None = None,
        run_seed: int = 0,
        gain_range_db: tuple[float, float] = (-3.0, 3.0),
    ):
        self.records = read_jsonl(manifest_path)
        self.cache_root = Path(cache_root)
        self.sample_rate = sample_rate
        self.hop_s = hop_s
        self.esc50_index = esc50_index or []
        self.rir_index = rir_index or []
        self.run_seed = run_seed
        self.gain_range_db = gain_range_db
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch from the training loop — changes the train-split
        augmentation realization without touching which examples exist.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        seed_seq = np.random.SeedSequence([self.run_seed, self.epoch, index])
        return np.random.default_rng(seed_seq)

    def _read_source_window(self, source: dict) -> np.ndarray:
        path = source["path"]
        if "offset_s" in source:
            start = int(round(source["offset_s"] * self.sample_rate))
            frames = int(round(source["duration_s"] * self.sample_rate))
            data, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=False)
        else:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            # Every cached/AMI/TEN/FLEURS source is already at self.sample_rate by
            # construction (Phase 2) -- this is a defensive check, not expected to trigger.
            raise ValueError(f"{path}: unexpected sample rate {sr}, expected {self.sample_rate}")
        return data.astype(np.float32)

    def _assemble_waveform(self, record: dict, rng: np.random.Generator) -> np.ndarray:
        if record["kind"] == "concat_synthetic":
            gaps = record["assembly"]["gaps_s"]
            pieces = []
            for i, source in enumerate(record["sources"]):
                pieces.append(self._read_source_window(source))
                if i < len(gaps):
                    pieces.append(make_gap_audio(rng, gaps[i], self.sample_rate))
            return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        # ami_window, ten_eval: single source, direct slice
        return self._read_source_window(record["sources"][0])

    def _apply_augmentation(
        self, waveform: np.ndarray, record: dict, rng: np.random.Generator
    ) -> np.ndarray:
        policy = record.get("augmentation_policy")
        if policy is None:
            return waveform

        if record.get("manifest_seed") is not None:
            resolved = policy  # already concrete (val/test, plan §3.4)
        else:
            resolved = resolve_augmentation(rng, self.esc50_index, self.rir_index, policy)

        if resolved.get("rir_file"):
            rir = np.load(str(self.cache_root / resolved["rir_file"])).astype(np.float32)
            waveform = convolve_rir(waveform, rir)

        noise, _ = sf.read(
            str(self.cache_root / resolved["noise_file"]), dtype="float32", always_2d=False
        )
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        waveform = mix_at_snr(rng, waveform, noise.astype(np.float32), resolved["snr_db"])

        return random_gain(rng, waveform, self.gain_range_db)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        seed = record.get("manifest_seed")
        rng = np.random.default_rng(seed) if seed is not None else self._rng_for_index(index)

        waveform = self._assemble_waveform(record, rng)

        # Labels are derived from the clean, pre-augmentation waveform --
        # augmentation (noise mix, reverb, gain) preserves length but would
        # make energy-based silence detection unreliable (mixed-in noise
        # floor). Augmentation functions in this pipeline never change
        # sample count, so `labels` stays aligned with the post-augmentation
        # waveform below.
        num_frames = int(round(len(waveform) / self.sample_rate / self.hop_s))
        labels = to_frames(record["label_intervals"], num_frames, self.hop_s)
        if record["kind"] == "concat_synthetic":
            labels = trim_internal_silence(waveform, labels, self.sample_rate, self.hop_s)

        waveform = self._apply_augmentation(waveform, record, rng)

        return {
            "id": record["id"],
            "waveform": torch.from_numpy(waveform.astype(np.float32)),
            "labels": torch.from_numpy(labels.astype(np.int64)),
        }
