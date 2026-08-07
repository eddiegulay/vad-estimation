import numpy as np
import pytest

from vad.labels.synthetic import (
    build_concat_labels,
    sample_gap_s,
    sample_gaps,
    sample_num_utterances,
    total_duration,
    trim_internal_silence,
)


def test_build_concat_labels_alternates_speech_and_gaps():
    intervals = build_concat_labels(source_durations=[2.0, 3.0, 1.5], gaps_s=[0.5, 1.0])
    assert intervals == [
        (0.0, 2.0, 1),
        (2.0, 2.5, 0),
        (2.5, 5.5, 1),
        (5.5, 6.5, 0),
        (6.5, 8.0, 1),
    ]


def test_build_concat_labels_single_source_no_gaps():
    assert build_concat_labels(source_durations=[4.0], gaps_s=[]) == [(0.0, 4.0, 1)]


def test_build_concat_labels_empty_is_empty():
    assert build_concat_labels(source_durations=[], gaps_s=[]) == []


def test_build_concat_labels_wrong_gap_count_raises():
    with pytest.raises(ValueError):
        build_concat_labels(source_durations=[1.0, 2.0], gaps_s=[0.5, 0.5])


def test_total_duration_sums_sources_and_gaps():
    assert total_duration([1.0, 2.0, 3.0], [0.5, 0.5]) == pytest.approx(7.0)


def test_sample_num_utterances_within_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = sample_num_utterances(rng, (2, 6))
        assert 2 <= n <= 6


def test_sample_gap_s_within_typical_range_when_not_long_pause():
    rng = np.random.default_rng(0)
    gap = sample_gap_s(
        rng, typical_range=(0.2, 3.0), long_pause_prob=0.0, long_pause_range=(3.0, 8.0)
    )
    assert 0.2 <= gap <= 3.0


def test_sample_gap_s_uses_long_pause_range_when_forced():
    rng = np.random.default_rng(0)
    gap = sample_gap_s(
        rng, typical_range=(0.2, 3.0), long_pause_prob=1.0, long_pause_range=(3.0, 8.0)
    )
    assert 3.0 <= gap <= 8.0


def test_sample_gaps_returns_requested_count():
    rng = np.random.default_rng(0)
    gaps = sample_gaps(rng, n_gaps=5, typical_range=(0.2, 3.0), long_pause_prob=0.08, long_pause_range=(3.0, 8.0))
    assert len(gaps) == 5
    assert all(g > 0 for g in gaps)


def test_same_seed_is_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    gaps1 = sample_gaps(rng1, 10, (0.2, 3.0), 0.08, (3.0, 8.0))
    gaps2 = sample_gaps(rng2, 10, (0.2, 3.0), 0.08, (3.0, 8.0))
    assert gaps1 == gaps2


SR = 16000
HOP_S = 0.032
FRAME = int(round(HOP_S * SR))


def _tone(n_frames, amp=0.5):
    t = np.arange(n_frames * FRAME) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _near_silence(n_frames, amp=0.0001):
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(n_frames * FRAME)).astype(np.float32)


def test_trim_internal_silence_flips_sustained_low_energy_speech_frames():
    waveform = np.concatenate([_tone(10), _near_silence(8), _tone(10)])
    labels = np.ones(28, dtype=np.int64)
    trimmed = trim_internal_silence(waveform, labels, SR, HOP_S, min_run_frames=5)
    assert trimmed[:10].sum() == 10
    assert trimmed[10:18].sum() == 0
    assert trimmed[18:].sum() == 10


def test_trim_internal_silence_ignores_brief_dip_below_min_run():
    waveform = np.concatenate([_tone(10), _near_silence(2), _tone(10)])
    labels = np.ones(22, dtype=np.int64)
    trimmed = trim_internal_silence(waveform, labels, SR, HOP_S, min_run_frames=5)
    assert trimmed.sum() == 22


def test_trim_internal_silence_never_touches_already_gap_frames():
    waveform = _near_silence(10)
    labels = np.zeros(10, dtype=np.int64)
    trimmed = trim_internal_silence(waveform, labels, SR, HOP_S)
    assert np.array_equal(trimmed, labels)


def test_trim_internal_silence_empty_is_noop():
    trimmed = trim_internal_silence(np.array([], dtype=np.float32), np.array([], dtype=np.int64), SR, HOP_S)
    assert len(trimmed) == 0


def test_trim_internal_silence_uniform_loud_speech_untouched():
    waveform = _tone(20)
    labels = np.ones(20, dtype=np.int64)
    trimmed = trim_internal_silence(waveform, labels, SR, HOP_S)
    assert trimmed.sum() == 20
