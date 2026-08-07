"""ONNX export: an explicit streaming `(chunk, context, gru_hidden) ->
(prob, new_context, new_gru_hidden)` interface, fixed chunk size, batch-only
dynamic axis (plan §6/§8 — static shapes are what make ONNX Runtime's
memory planning fast; dynamic shapes are "the usual cause of a model that
benchmarks well and runs slow in an app").

Unlike the PyTorch-side `CRNN.forward(chunk, state=None)`, which accepts
`state=None` on the first call, the exported graph always takes concrete
tensor inputs — no Python `None`/tuples in an ONNX graph. The convention
(matching Silero's actual ONNX export) is that the *caller* seeds a
zero-initialized context/hidden state for the very first call.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from vad.models.crnn import CRNN
from vad.models.tcn import TCN


class StreamingStepWrapper(nn.Module):
    def __init__(self, model: CRNN):
        super().__init__()
        self.model = model

    def forward(
        self, chunk: torch.Tensor, context: torch.Tensor, gru_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = self.model._assemble_padded(context, chunk)
        features = self.model._encode(padded).unsqueeze(1)  # [B, 1, conv_out_ch]
        gru_out, new_gru_hidden = self.model.gru(features, gru_hidden)
        logit = self.model.head(gru_out.transpose(1, 2)).squeeze(-1).squeeze(-1)
        prob = torch.sigmoid(logit)
        new_context = chunk[:, -self.model.CONTEXT_SAMPLES :]
        return prob, new_context, new_gru_hidden


def export_onnx(
    model: CRNN, chunk_samples: int, hidden_size: int, out_path: str | Path, opset: int = 18
) -> None:
    # opset 18, not 17: the reflect-pad op used by the frontend has no
    # opset-17 version-converter adapter in this onnx/onnxscript release,
    # so requesting 17 just produces a noisy (non-fatal) fallback warning.
    model.eval()
    wrapper = StreamingStepWrapper(model)

    batch_size = 1
    dummy_chunk = torch.zeros(batch_size, chunk_samples)
    dummy_context = torch.zeros(batch_size, model.CONTEXT_SAMPLES)
    dummy_hidden = torch.zeros(1, batch_size, hidden_size)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_chunk, dummy_context, dummy_hidden),
        str(out_path),
        input_names=["chunk", "context", "gru_hidden"],
        output_names=["prob", "new_context", "new_gru_hidden"],
        dynamic_axes={
            "chunk": {0: "batch"},
            "context": {0: "batch"},
            "gru_hidden": {1: "batch"},
            "prob": {0: "batch"},
            "new_context": {0: "batch"},
            "new_gru_hidden": {1: "batch"},
        },
        opset_version=opset,
    )


def run_onnx_streaming(
    session, waveform: np.ndarray, chunk_samples: int, context_samples: int, hidden_size: int
) -> np.ndarray:
    """Run the exported ONNX model chunk-by-chunk over a full waveform
    ([T] numpy array), matching `CRNN.forward_full`'s semantics — used for
    PyTorch/ONNXRuntime parity checks.
    """
    context = np.zeros((1, context_samples), dtype=np.float32)
    gru_hidden = np.zeros((1, 1, hidden_size), dtype=np.float32)

    num_chunks = len(waveform) // chunk_samples
    probs = np.zeros(num_chunks, dtype=np.float32)
    for i in range(num_chunks):
        chunk = waveform[i * chunk_samples : (i + 1) * chunk_samples].reshape(1, -1).astype(np.float32)
        prob, context, gru_hidden = session.run(
            ["prob", "new_context", "new_gru_hidden"],
            {"chunk": chunk, "context": context, "gru_hidden": gru_hidden},
        )
        probs[i] = prob[0]
    return probs


class TCNStreamingStepWrapper(nn.Module):
    """Same streaming-step contract as `StreamingStepWrapper`, but TCN's
    state is a DFT tail plus one variable-length activation cache per
    temporal block (see `models/tcn.py`) instead of a single fixed-size GRU
    hidden state -- each block's cache gets its own named tensor input/
    output rather than a generic list, since ONNX graphs need concrete
    tensor-only I/O (no Python lists/tuples).
    """

    def __init__(self, model: TCN):
        super().__init__()
        self.model = model
        self.n_blocks = len(model.temporal.blocks)

    def forward(self, chunk: torch.Tensor, tail: torch.Tensor, *block_states: torch.Tensor):
        state = (tail, list(block_states))
        prob, (new_tail, new_block_states) = self.model(chunk, state)
        return (prob, new_tail, *new_block_states)


def export_onnx_tcn(model: TCN, chunk_samples: int, out_path: str | Path, opset: int = 18) -> None:
    """Unlike `export_onnx` (CRNN), batch is fixed at 1, not a dynamic axis:
    TCN's per-block state tensors have differing trailing dimensions
    (`left_pad` varies per dilation), which trips up the exporter's
    dynamic_axes-to-dynamic_shapes conversion. Streaming VAD inference is
    batch=1 in practice anyway (one live audio stream per model instance),
    so this isn't a real limitation.
    """
    model.eval()
    wrapper = TCNStreamingStepWrapper(model)
    wrapper.eval()

    batch_size = 1
    dummy_chunk = torch.zeros(batch_size, chunk_samples)
    dummy_tail = torch.zeros(batch_size, model.CONTEXT_SAMPLES)
    dummy_block_states = [
        torch.zeros(batch_size, model.channels, block.left_pad) for block in model.temporal.blocks
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_names = [f"state_{i}" for i in range(wrapper.n_blocks)]
    new_state_names = [f"new_state_{i}" for i in range(wrapper.n_blocks)]

    torch.onnx.export(
        wrapper,
        (dummy_chunk, dummy_tail, *dummy_block_states),
        str(out_path),
        input_names=["chunk", "tail", *state_names],
        output_names=["prob", "new_tail", *new_state_names],
        opset_version=opset,
    )


def run_onnx_streaming_tcn(session, waveform: np.ndarray, chunk_samples: int, model: TCN) -> np.ndarray:
    """TCN counterpart of `run_onnx_streaming` -- needs the PyTorch model
    only to read state shapes (context_samples, per-block left_pad), not
    for any computation.
    """
    tail = np.zeros((1, model.CONTEXT_SAMPLES), dtype=np.float32)
    block_states = [
        np.zeros((1, model.channels, block.left_pad), dtype=np.float32) for block in model.temporal.blocks
    ]
    state_names = [f"state_{i}" for i in range(len(block_states))]
    new_state_names = [f"new_state_{i}" for i in range(len(block_states))]

    num_chunks = len(waveform) // chunk_samples
    probs = np.zeros(num_chunks, dtype=np.float32)
    for i in range(num_chunks):
        chunk = waveform[i * chunk_samples : (i + 1) * chunk_samples].reshape(1, -1).astype(np.float32)
        inputs = {"chunk": chunk, "tail": tail, **dict(zip(state_names, block_states))}
        outputs = session.run(["prob", "new_tail", *new_state_names], inputs)
        prob, tail, *block_states = outputs
        probs[i] = prob[0]
    return probs
