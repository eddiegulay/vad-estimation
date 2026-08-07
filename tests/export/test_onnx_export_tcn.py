from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from vad.config import load_yaml
from vad.export.onnx_export import export_onnx_tcn, run_onnx_streaming_tcn
from vad.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "tcn_v1.yaml"


def _load_model():
    config = load_yaml(MODEL_CONFIG_PATH)
    model = build_model(config["architecture"], config)
    model.eval()
    return model, config


def test_export_produces_onnx_file(tmp_path):
    model, config = _load_model()
    out_path = tmp_path / "model.onnx"
    export_onnx_tcn(model, config["frontend"]["chunk_samples"], out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_exported_model_has_expected_io(tmp_path):
    model, config = _load_model()
    out_path = tmp_path / "model.onnx"
    export_onnx_tcn(model, config["frontend"]["chunk_samples"], out_path)
    session = ort.InferenceSession(str(out_path))

    n_blocks = len(model.temporal.blocks)
    expected_inputs = {"chunk", "tail"} | {f"state_{i}" for i in range(n_blocks)}
    expected_outputs = {"prob", "new_tail"} | {f"new_state_{i}" for i in range(n_blocks)}
    assert {i.name for i in session.get_inputs()} == expected_inputs
    assert {o.name for o in session.get_outputs()} == expected_outputs


def test_onnx_streaming_matches_pytorch_forward_full(tmp_path):
    model, config = _load_model()
    chunk_samples = config["frontend"]["chunk_samples"]

    out_path = tmp_path / "model.onnx"
    export_onnx_tcn(model, chunk_samples, out_path)
    session = ort.InferenceSession(str(out_path))

    torch.manual_seed(0)
    waveform = torch.randn(1, 40 * chunk_samples)

    with torch.no_grad():
        torch_probs = model.forward_full(waveform)[0].numpy()

    onnx_probs = run_onnx_streaming_tcn(session, waveform[0].numpy(), chunk_samples, model)

    assert torch_probs.shape == onnx_probs.shape
    max_diff = np.max(np.abs(torch_probs - onnx_probs))
    assert max_diff < 1e-3, f"pytorch/onnx parity diff {max_diff} exceeds tolerance"
