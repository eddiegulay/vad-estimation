from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from vad.config import load_yaml
from vad.export.onnx_export import export_onnx, run_onnx_streaming
from vad.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "crnn_v1.yaml"


def _load_model():
    config = load_yaml(MODEL_CONFIG_PATH)
    model = build_model(config["architecture"], config)
    model.eval()
    return model, config


def test_export_produces_onnx_file(tmp_path):
    model, config = _load_model()
    out_path = tmp_path / "model.onnx"
    export_onnx(
        model, config["frontend"]["chunk_samples"], config["recurrent"]["hidden_size"], out_path
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_exported_model_loads_in_onnxruntime(tmp_path):
    model, config = _load_model()
    out_path = tmp_path / "model.onnx"
    export_onnx(
        model, config["frontend"]["chunk_samples"], config["recurrent"]["hidden_size"], out_path
    )
    session = ort.InferenceSession(str(out_path))
    input_names = {i.name for i in session.get_inputs()}
    output_names = {o.name for o in session.get_outputs()}
    assert input_names == {"chunk", "context", "gru_hidden"}
    assert output_names == {"prob", "new_context", "new_gru_hidden"}


def test_onnx_streaming_matches_pytorch_forward_full(tmp_path):
    model, config = _load_model()
    chunk_samples = config["frontend"]["chunk_samples"]
    hidden_size = config["recurrent"]["hidden_size"]

    out_path = tmp_path / "model.onnx"
    export_onnx(model, chunk_samples, hidden_size, out_path)
    session = ort.InferenceSession(str(out_path))

    torch.manual_seed(0)
    waveform = torch.randn(1, 6 * chunk_samples)

    with torch.no_grad():
        torch_probs = model.forward_full(waveform)[0].numpy()

    onnx_probs = run_onnx_streaming(
        session, waveform[0].numpy(), chunk_samples, model.CONTEXT_SAMPLES, hidden_size
    )

    assert torch_probs.shape == onnx_probs.shape
    max_diff = np.max(np.abs(torch_probs - onnx_probs))
    assert max_diff < 1e-3, f"pytorch/onnx parity diff {max_diff} exceeds tolerance"


def test_onnx_export_handles_different_batch_sizes(tmp_path):
    """Dynamic batch axis must actually work -- export with batch=1, run with batch=3."""
    model, config = _load_model()
    chunk_samples = config["frontend"]["chunk_samples"]
    hidden_size = config["recurrent"]["hidden_size"]

    out_path = tmp_path / "model.onnx"
    export_onnx(model, chunk_samples, hidden_size, out_path)
    session = ort.InferenceSession(str(out_path))

    batch_size = 3
    chunk = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    context = np.zeros((batch_size, model.CONTEXT_SAMPLES), dtype=np.float32)
    gru_hidden = np.zeros((1, batch_size, hidden_size), dtype=np.float32)

    prob, new_context, new_hidden = session.run(
        ["prob", "new_context", "new_gru_hidden"],
        {"chunk": chunk, "context": context, "gru_hidden": gru_hidden},
    )
    assert prob.shape == (batch_size,)
    assert new_context.shape == (batch_size, model.CONTEXT_SAMPLES)
    assert new_hidden.shape == (1, batch_size, hidden_size)
