"""Fixed (non-trainable) STFT-magnitude frontend, implemented as a strided
Conv1d over a real cos/sin DFT basis -- deliberately avoids `torch.stft`'s
complex tensors (historically incomplete MPS support, and keeps the
frontend a plain op inside the exportable model graph). Matches Silero
VAD's traced architecture: a fixed-DFT Conv1d producing magnitude directly,
not a learned or mel-filtered representation.
"""

import math

import torch
from torch import nn


class DFTFrontend(nn.Module):
    def __init__(self, n_fft: int, hop_samples: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_samples = hop_samples
        self.n_bins = n_fft // 2 + 1

        n = torch.arange(n_fft, dtype=torch.float32)
        window = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (n_fft - 1))  # Hann
        freqs = torch.arange(self.n_bins, dtype=torch.float32).unsqueeze(1)  # [n_bins, 1]
        angles = 2 * math.pi * freqs * n.unsqueeze(0) / n_fft  # [n_bins, n_fft]
        cos_basis = torch.cos(angles) * window.unsqueeze(0)
        sin_basis = -torch.sin(angles) * window.unsqueeze(0)
        basis = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)  # [2*n_bins, 1, n_fft]

        self.register_buffer("basis", basis)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [B, 1, T] -> magnitude spectrum [B, n_bins, frames]."""
        spec = torch.nn.functional.conv1d(waveform, self.basis, stride=self.hop_samples)
        real, imag = spec[:, : self.n_bins], spec[:, self.n_bins :]
        return torch.sqrt(real**2 + imag**2 + 1e-9)
