"""CRNN (Silero-shape) VAD model: Conv1d-DFT frontend + small conv stack +
one GRU. Exposes an explicit streaming interface,
`forward(chunk, state) -> (prob, state')`, plus a vectorized
`forward_full(waveform)` convenience for training that is mathematically
identical to running the streaming path chunk-by-chunk with carried state
(verified by tests/models/test_crnn.py's streaming/full-forward equivalence
test — plan §6's Phase 5 gate).

State carried across streaming calls: a 64-sample waveform "context" tail
(left-context for the DFT frontend's analysis window, matching Silero's
traced context+reflect-pad scheme) and the GRU hidden state.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vad.models.frontend import DFTFrontend
from vad.models.registry import register_model


class ConvStack(nn.Module):
    def __init__(self, conv_blocks: list[dict]):
        super().__init__()
        layers = []
        for block in conv_blocks:
            layers.append(
                nn.Conv1d(
                    block["in_ch"],
                    block["out_ch"],
                    kernel_size=block["kernel"],
                    stride=block["stride"],
                    padding=block["kernel"] // 2,
                )
            )
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("crnn")
class CRNN(nn.Module):
    CONTEXT_SAMPLES = 64
    REFLECT_PAD = 64

    def __init__(self, config: dict):
        super().__init__()
        frontend_cfg = config["frontend"]
        self.chunk_samples = frontend_cfg["chunk_samples"]

        self.frontend = DFTFrontend(n_fft=frontend_cfg["n_fft"], hop_samples=frontend_cfg["hop_samples"])
        self.conv_stack = ConvStack(config["conv_blocks"])

        hidden_size = config["recurrent"]["hidden_size"]
        conv_out_ch = config["conv_blocks"][-1]["out_ch"]
        self.gru = nn.GRU(input_size=conv_out_ch, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def _encode(self, padded: torch.Tensor) -> torch.Tensor:
        """padded: [*, 640] -> per-chunk features [*, conv_out_ch]."""
        magnitude = self.frontend(padded.unsqueeze(1))  # [*, n_bins, frames]
        conv_out = self.conv_stack(magnitude)  # [*, conv_out_ch, 1]
        return conv_out.squeeze(-1)

    def _assemble_padded(self, context: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        """context: [B,64], chunk: [B,chunk_samples] -> [B, 640]."""
        assembled = torch.cat([context, chunk], dim=-1)
        return F.pad(assembled.unsqueeze(1), (0, self.REFLECT_PAD), mode="reflect").squeeze(1)

    def forward(
        self, chunk: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Streaming single-chunk step. chunk: [B, chunk_samples] -> (prob [B], new_state)."""
        batch_size = chunk.shape[0]
        if state is None:
            context = torch.zeros(batch_size, self.CONTEXT_SAMPLES, dtype=chunk.dtype, device=chunk.device)
            gru_hidden = None
        else:
            context, gru_hidden = state

        padded = self._assemble_padded(context, chunk)
        features = self._encode(padded).unsqueeze(1)  # [B, 1, conv_out_ch]
        gru_out, gru_hidden_new = self.gru(features, gru_hidden)  # [B, 1, hidden]
        logit = self.head(gru_out.transpose(1, 2)).squeeze(-1).squeeze(-1)  # [B]
        prob = torch.sigmoid(logit)

        new_context = chunk[:, -self.CONTEXT_SAMPLES :]
        return prob, (new_context, gru_hidden_new)

    def forward_full(self, waveform: torch.Tensor) -> torch.Tensor:
        """Non-streaming training convenience. waveform: [B, T], T a multiple
        of chunk_samples (pad the batch beforehand) -> probs [B, N].
        """
        batch_size, total_len = waveform.shape
        if total_len % self.chunk_samples != 0:
            raise ValueError("waveform length must be a multiple of chunk_samples")
        num_chunks = total_len // self.chunk_samples

        chunks = waveform.view(batch_size, num_chunks, self.chunk_samples)
        zero_tail = torch.zeros(
            batch_size, 1, self.CONTEXT_SAMPLES, dtype=waveform.dtype, device=waveform.device
        )
        prev_tail = torch.cat([zero_tail, chunks[:, :-1, -self.CONTEXT_SAMPLES :]], dim=1)  # [B,N,64]

        assembled = torch.cat([prev_tail, chunks], dim=-1)  # [B, N, 576]
        padded = F.pad(
            assembled.reshape(batch_size * num_chunks, 1, 576), (0, self.REFLECT_PAD), mode="reflect"
        ).squeeze(1)  # [B*N, 640]

        features = self._encode(padded).view(batch_size, num_chunks, -1)  # [B, N, conv_out_ch]
        gru_out, _ = self.gru(features)  # [B, N, hidden]
        logits = self.head(gru_out.transpose(1, 2)).squeeze(1)  # [B, N]
        return torch.sigmoid(logits)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
