"""TCN (MarbleNet-shape) VAD model: the *same* fixed Conv1d-DFT frontend
and per-chunk local conv encoder as `crnn.CRNN`, but cross-chunk temporal
context comes from a causal, dilated, depthwise-separable Conv1d stack
instead of a GRU -- isolates the architecture comparison to "how should the
model remember across chunks" (recurrent hidden state vs. a bounded
receptive field), everything else held constant, per note 08's
depthwise-separable "going bigger does not help" recipe.

Streaming state is a small per-layer activation cache (one small buffer per
temporal block, each zero-initialized), not a raw-audio or whole-window
recompute. This is a deliberate correctness choice, not just an efficiency
one: `forward_full` treats "no history before the sequence start" as a
*literal zero* at every layer's own input independently (each block's own
`F.pad`). A naive "recompute the whole receptive-field window from scratch"
streaming approach does *not* reproduce this -- feeding zero-valued inputs
through an earlier block (which has a bias term) produces a non-zero
output, and re-deriving a later block's boundary from that non-zero value
diverges from `forward_full`'s independent per-layer zero convention. Caching
each block's own true (possibly literal-zero) input history sidesteps this
entirely and is also the standard, efficient way real streaming causal
convs are implemented (O(1) per step, not O(receptive field)).
"""

import torch
import torch.nn.functional as F
from torch import nn

from vad.models.crnn import ConvStack
from vad.models.frontend import DFTFrontend
from vad.models.registry import register_model


class CausalDepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.channels = channels
        self.left_pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, C, T], same-length causal (left zero-padded)."""
        x = F.pad(x, (self.left_pad, 0))
        x = self.depthwise(x)
        return self.pointwise(x)

    def init_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, self.left_pad, dtype=dtype, device=device)

    def forward_step(self, x_new: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_new: [B, C] (one time step) + state: [B, C, left_pad] (this
        block's own cached input history, zero on the first call) ->
        (out: [B, C], new_state). No re-derivation through another block's
        bias -- state is this block's *own* raw input history, matching
        exactly what `forward`'s F.pad would supply at this position.
        """
        if self.left_pad == 0:
            window = x_new.unsqueeze(-1)
        else:
            window = torch.cat([state, x_new.unsqueeze(-1)], dim=-1)  # [B, C, left_pad+1]
        out = self.pointwise(self.depthwise(window)).squeeze(-1)  # [B, C]
        new_state = window[:, :, 1:] if self.left_pad > 0 else state
        return out, new_state


class TemporalStack(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CausalDepthwiseSeparableBlock(channels, kernel_size, d) for d in dilations]
        )
        self.receptive_field_chunks = 1 + sum((kernel_size - 1) * d for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + F.relu(block(x))  # constant channel count throughout -> residual always shape-safe
        return x

    def init_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        return [block.init_state(batch_size, dtype, device) for block in self.blocks]

    def forward_step(self, x_new: torch.Tensor, states: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x_new: [B, C] -> (out: [B, C], new_states). Exact single-step
        equivalent of `forward` at the newest time position.
        """
        x = x_new
        new_states = []
        for block, state in zip(self.blocks, states):
            block_out, new_state = block.forward_step(x, state)
            x = x + F.relu(block_out)
            new_states.append(new_state)
        return x, new_states


@register_model("tcn")
class TCN(nn.Module):
    CONTEXT_SAMPLES = 64
    REFLECT_PAD = 64

    def __init__(self, config: dict):
        super().__init__()
        frontend_cfg = config["frontend"]
        self.chunk_samples = frontend_cfg["chunk_samples"]

        self.frontend = DFTFrontend(n_fft=frontend_cfg["n_fft"], hop_samples=frontend_cfg["hop_samples"])
        self.local_conv = ConvStack(config["local_conv_blocks"])

        temporal_cfg = config["temporal"]
        self.channels = config["local_conv_blocks"][-1]["out_ch"]
        assert temporal_cfg["channels"] == self.channels, (
            "temporal.channels must match local_conv_blocks[-1].out_ch"
        )
        self.temporal = TemporalStack(self.channels, temporal_cfg["kernel_size"], temporal_cfg["dilations"])
        self.head = nn.Conv1d(self.channels, 1, kernel_size=1)

        self.receptive_field_chunks = self.temporal.receptive_field_chunks

    def _encode(self, padded: torch.Tensor) -> torch.Tensor:
        """padded: [*, 640] -> per-chunk features [*, conv_out_ch]."""
        magnitude = self.frontend(padded.unsqueeze(1))
        conv_out = self.local_conv(magnitude)
        return conv_out.squeeze(-1)

    def _assemble_padded(self, context: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        assembled = torch.cat([context, chunk], dim=-1)
        return F.pad(assembled.unsqueeze(1), (0, self.REFLECT_PAD), mode="reflect").squeeze(1)

    def forward_full(self, waveform: torch.Tensor) -> torch.Tensor:
        """Non-streaming training convenience. waveform: [B, T], T a
        multiple of chunk_samples -> probs [B, N]. Identical tail-carry
        convention to CRNN.forward_full (chunk 0 gets a zero tail).
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

        embeddings = self._encode(padded).view(batch_size, num_chunks, -1)  # [B, N, C]
        x = self.temporal(embeddings.transpose(1, 2))  # [B, C, N]
        logits = self.head(x).squeeze(1)  # [B, N]
        return torch.sigmoid(logits)

    def forward(
        self, chunk: torch.Tensor, state: tuple[torch.Tensor, list[torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """Streaming single-chunk step. chunk: [B, chunk_samples] -> (prob [B], new_state).
        state = (tail: [B, 64] DFT context, same convention as CRNN;
        temporal_states: per-block activation caches, see TemporalStack).
        """
        batch_size = chunk.shape[0]
        if state is None:
            tail = torch.zeros(batch_size, self.CONTEXT_SAMPLES, dtype=chunk.dtype, device=chunk.device)
            temporal_states = self.temporal.init_state(batch_size, chunk.dtype, chunk.device)
        else:
            tail, temporal_states = state

        padded = self._assemble_padded(tail, chunk)
        new_embedding = self._encode(padded)  # [B, C]
        x, new_temporal_states = self.temporal.forward_step(new_embedding, temporal_states)
        logit = self.head(x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [B]
        prob = torch.sigmoid(logit)

        new_tail = chunk[:, -self.CONTEXT_SAMPLES :]
        return prob, (new_tail, new_temporal_states)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
