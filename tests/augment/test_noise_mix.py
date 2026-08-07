import numpy as np
import pytest

from vad.augment.noise_mix import fit_noise_length, measured_snr_db, mix_at_snr, rms


def test_fit_noise_length_crops_when_longer():
    rng = np.random.default_rng(0)
    noise = np.arange(100, dtype=np.float32)
    fitted = fit_noise_length(rng, noise, target_len=10)
    assert len(fitted) == 10


def test_fit_noise_length_loops_when_shorter():
    rng = np.random.default_rng(0)
    noise = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fitted = fit_noise_length(rng, noise, target_len=10)
    assert len(fitted) == 10


def test_fit_noise_length_handles_empty_noise():
    rng = np.random.default_rng(0)
    fitted = fit_noise_length(rng, np.zeros(0, dtype=np.float32), target_len=10)
    assert len(fitted) == 10
    assert np.all(fitted == 0)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 5.0, 10.0, 20.0])
def test_mix_at_snr_achieves_target_snr(snr_db):
    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.5
    noise = rng.standard_normal(16000).astype(np.float32)

    mixed = mix_at_snr(rng, signal, noise, snr_db)
    noise_component = mixed - signal
    achieved_snr = measured_snr_db(signal, noise_component)

    assert abs(achieved_snr - snr_db) < 0.5  # dB tolerance


def test_mix_at_snr_no_nan_or_inf_across_fuzz():
    rng = np.random.default_rng(123)
    for _ in range(100):
        sig_len = int(rng.integers(1, 16000))
        noise_len = int(rng.integers(0, 32000))
        signal = (rng.standard_normal(sig_len) * rng.uniform(0.0, 1.0)).astype(np.float32)
        noise = (rng.standard_normal(noise_len) * rng.uniform(0.0, 1.0)).astype(np.float32)
        snr_db = rng.uniform(-10.0, 30.0)

        mixed = mix_at_snr(rng, signal, noise, snr_db)
        assert np.all(np.isfinite(mixed))


def test_mix_at_snr_handles_silent_signal():
    rng = np.random.default_rng(0)
    signal = np.zeros(1000, dtype=np.float32)
    noise = rng.standard_normal(1000).astype(np.float32)
    mixed = mix_at_snr(rng, signal, noise, snr_db=5.0)
    assert np.all(np.isfinite(mixed))


def test_mix_at_snr_handles_silent_noise():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(1000).astype(np.float32)
    noise = np.zeros(1000, dtype=np.float32)
    mixed = mix_at_snr(rng, signal, noise, snr_db=5.0)
    assert np.all(np.isfinite(mixed))
    assert np.allclose(mixed, signal)


def test_rms_of_zeros_is_near_zero():
    assert rms(np.zeros(100)) < 1e-4
