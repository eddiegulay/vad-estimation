from pathlib import Path

import numpy as np
import pytest

from vad.augment.reverb import convolve_rir

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "data_cache"


def test_convolve_rir_preserves_length():
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(16000).astype(np.float32)
    rir = rng.standard_normal(4000).astype(np.float32)
    wet = convolve_rir(waveform, rir)
    assert len(wet) == len(waveform)


def test_convolve_rir_empty_rir_is_noop():
    waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    wet = convolve_rir(waveform, np.zeros(0, dtype=np.float32))
    assert np.allclose(wet, waveform)


def test_convolve_rir_empty_waveform_is_empty():
    rir = np.array([1.0, 0.5], dtype=np.float32)
    wet = convolve_rir(np.zeros(0, dtype=np.float32), rir)
    assert len(wet) == 0


def test_convolve_rir_normalizes_to_original_peak():
    rng = np.random.default_rng(1)
    waveform = (rng.standard_normal(8000).astype(np.float32)) * 0.3
    rir = rng.standard_normal(2000).astype(np.float32)
    wet = convolve_rir(waveform, rir, normalize=True)
    assert np.max(np.abs(wet)) == pytest.approx(np.max(np.abs(waveform)), rel=1e-3)


def test_convolve_rir_no_nan_inf_no_clipping_across_fuzz():
    rng = np.random.default_rng(7)
    for _ in range(50):
        wav_len = int(rng.integers(1, 8000))
        rir_len = int(rng.integers(1, 4000))
        waveform = (rng.standard_normal(wav_len) * rng.uniform(0.0, 1.0)).astype(np.float32)
        rir = (rng.standard_normal(rir_len) * rng.uniform(0.0, 1.0)).astype(np.float32)

        wet = convolve_rir(waveform, rir, normalize=True)
        assert np.all(np.isfinite(wet))
        orig_peak = np.max(np.abs(waveform)) + 1e-10
        assert np.max(np.abs(wet)) <= orig_peak * 1.01  # normalized, should not exceed original peak


def test_convolve_rir_preserves_alignment_with_impulse_rir():
    """A unit-impulse RIR (delta at t=0) must be a no-op (identity convolution)."""
    waveform = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)
    rir = np.array([1.0], dtype=np.float32)
    wet = convolve_rir(waveform, rir, normalize=False)
    assert np.allclose(wet, waveform)


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_convolve_rir_with_real_cached_rir():
    rir_files = list((CACHE_ROOT / "rir").glob("*.npy"))
    if not rir_files:
        pytest.skip("no cached RIRs found")
    rir = np.load(str(rir_files[0]))
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(16000).astype(np.float32) * 0.2

    wet = convolve_rir(waveform, rir.astype(np.float32))
    assert len(wet) == len(waveform)
    assert np.all(np.isfinite(wet))
