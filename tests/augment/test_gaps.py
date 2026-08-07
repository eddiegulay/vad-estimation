import numpy as np
import pytest

from vad.augment.gaps import make_gap_audio


def test_make_gap_audio_correct_length():
    rng = np.random.default_rng(0)
    audio = make_gap_audio(rng, duration_s=1.5, sample_rate=16000, gap_kind="silence")
    assert len(audio) == int(round(1.5 * 16000))


def test_make_gap_audio_zero_duration_is_empty():
    rng = np.random.default_rng(0)
    assert len(make_gap_audio(rng, duration_s=0.0, sample_rate=16000)) == 0


def test_make_gap_audio_silence_is_all_zeros():
    rng = np.random.default_rng(0)
    audio = make_gap_audio(rng, duration_s=0.5, sample_rate=16000, gap_kind="silence")
    assert np.all(audio == 0)


def test_make_gap_audio_low_level_noise_is_quiet_and_finite():
    rng = np.random.default_rng(0)
    audio = make_gap_audio(
        rng, duration_s=0.5, sample_rate=16000, gap_kind="low_level_noise", noise_floor_db=-50.0
    )
    assert np.all(np.isfinite(audio))
    assert not np.all(audio == 0)
    # -50dB amplitude -> rms should be small relative to full scale (1.0)
    assert np.sqrt(np.mean(audio**2)) < 0.01


def test_make_gap_audio_mixed_kind_respects_duration_bounds_across_fuzz():
    rng = np.random.default_rng(3)
    for _ in range(50):
        duration_s = rng.uniform(0.05, 5.0)
        sample_rate = 16000
        audio = make_gap_audio(
            rng, duration_s=duration_s, sample_rate=sample_rate, gap_kind="silence_or_lowlevel_noise"
        )
        expected_len = int(round(duration_s * sample_rate))
        assert len(audio) == expected_len
        assert np.all(np.isfinite(audio))


def test_make_gap_audio_unknown_kind_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_gap_audio(rng, duration_s=0.5, sample_rate=16000, gap_kind="not_a_real_kind")
