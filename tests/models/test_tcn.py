from pathlib import Path

import torch

from vad.config import load_yaml
from vad.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "model" / "tcn_v1.yaml"


def _load_model():
    config = load_yaml(CONFIG_PATH)
    return build_model(config["architecture"], config), config


def test_model_builds_from_config_via_registry():
    model, config = _load_model()
    assert model.chunk_samples == config["frontend"]["chunk_samples"]


def test_param_count_within_target_band():
    model, config = _load_model()
    lo, hi = config["target_param_count"]
    n_params = model.num_parameters(trainable_only=True)
    assert lo <= n_params <= hi, f"{n_params} params outside target band [{lo}, {hi}]"


def test_frontend_dft_basis_is_not_trainable():
    model, _ = _load_model()
    assert not model.frontend.basis.requires_grad
    assert not any(p is model.frontend.basis for p in model.parameters())


def test_forward_full_output_shape():
    model, config = _load_model()
    model.eval()
    chunk_samples = config["frontend"]["chunk_samples"]
    batch_size, num_chunks = 4, 10
    waveform = torch.randn(batch_size, num_chunks * chunk_samples)

    with torch.no_grad():
        probs = model.forward_full(waveform)

    assert probs.shape == (batch_size, num_chunks)
    assert torch.all((probs >= 0) & (probs <= 1))
    assert torch.all(torch.isfinite(probs))


def test_forward_full_rejects_non_multiple_length():
    model, config = _load_model()
    chunk_samples = config["frontend"]["chunk_samples"]
    waveform = torch.randn(2, chunk_samples + 1)
    try:
        model.forward_full(waveform)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_streaming_matches_full_forward_short_sequence():
    """Short sequence (fewer chunks than the receptive field) exercises the
    start-of-stream zero-embedding-history boundary case end to end.
    """
    model, config = _load_model()
    model.eval()
    chunk_samples = config["frontend"]["chunk_samples"]
    batch_size, num_chunks = 3, 8
    waveform = torch.randn(batch_size, num_chunks * chunk_samples)

    with torch.no_grad():
        full_probs = model.forward_full(waveform)

        streaming_probs = []
        state = None
        for i in range(num_chunks):
            chunk = waveform[:, i * chunk_samples : (i + 1) * chunk_samples]
            prob, state = model(chunk, state)
            streaming_probs.append(prob)
        streaming_probs = torch.stack(streaming_probs, dim=1)

    assert streaming_probs.shape == full_probs.shape
    max_diff = (streaming_probs - full_probs).abs().max().item()
    assert max_diff < 1e-4, f"streaming vs full-forward max diff {max_diff} exceeds tolerance"


def test_streaming_matches_full_forward_long_sequence():
    """Sequence longer than the receptive field exercises the steady-state
    (embedding-history fully real, no zero padding) regime too.
    """
    model, config = _load_model()
    model.eval()
    chunk_samples = config["frontend"]["chunk_samples"]
    assert model.receptive_field_chunks < 40, "test assumes RF < 40 chunks"
    batch_size, num_chunks = 2, 40
    waveform = torch.randn(batch_size, num_chunks * chunk_samples)

    with torch.no_grad():
        full_probs = model.forward_full(waveform)

        streaming_probs = []
        state = None
        for i in range(num_chunks):
            chunk = waveform[:, i * chunk_samples : (i + 1) * chunk_samples]
            prob, state = model(chunk, state)
            streaming_probs.append(prob)
        streaming_probs = torch.stack(streaming_probs, dim=1)

    assert streaming_probs.shape == full_probs.shape
    max_diff = (streaming_probs - full_probs).abs().max().item()
    assert max_diff < 1e-4, f"streaming vs full-forward max diff {max_diff} exceeds tolerance"


def test_streaming_state_resets_cleanly_with_none():
    model, config = _load_model()
    model.eval()
    chunk_samples = config["frontend"]["chunk_samples"]
    chunk = torch.randn(2, chunk_samples)

    with torch.no_grad():
        prob_a, _ = model(chunk, state=None)
        prob_b, _ = model(chunk, state=None)

    assert torch.allclose(prob_a, prob_b), "identical input+None state must be deterministic"


def test_gradients_flow_through_forward_full():
    model, config = _load_model()
    model.train()
    chunk_samples = config["frontend"]["chunk_samples"]
    waveform = torch.randn(2, 4 * chunk_samples, requires_grad=False)

    probs = model.forward_full(waveform)
    loss = probs.sum()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaN
