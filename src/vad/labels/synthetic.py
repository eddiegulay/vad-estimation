"""Concat + gap-insertion recipe generation for corpora with no native
frame labels (LibriSpeech, FLEURS): whole utterances are treated as
label=1, separated by silence/low-level-noise gaps drawn from a
log-uniform "typical pause" distribution with a rare long-pause tail
(plan §5). Pure functions — no file I/O, no knowledge of any specific
corpus; callers (Phase 4 manifest builders) supply already-selected
source clip durations.
"""

import numpy as np

from vad.labels.intervals import Interval


def sample_num_utterances(rng: np.random.Generator, utterances_per_example: tuple[int, int]) -> int:
    lo, hi = utterances_per_example
    return int(rng.integers(lo, hi + 1))


def sample_gap_s(
    rng: np.random.Generator,
    typical_range: tuple[float, float],
    long_pause_prob: float,
    long_pause_range: tuple[float, float],
) -> float:
    """Draw one gap duration: log-uniform within `typical_range`, with
    probability `long_pause_prob` drawn log-uniform within `long_pause_range`
    instead.
    """
    lo, hi = long_pause_range if rng.random() < long_pause_prob else typical_range
    log_lo, log_hi = np.log(lo), np.log(hi)
    return float(np.exp(rng.uniform(log_lo, log_hi)))


def sample_gaps(
    rng: np.random.Generator,
    n_gaps: int,
    typical_range: tuple[float, float],
    long_pause_prob: float,
    long_pause_range: tuple[float, float],
) -> list[float]:
    return [
        sample_gap_s(rng, typical_range, long_pause_prob, long_pause_range)
        for _ in range(n_gaps)
    ]


def build_concat_labels(source_durations: list[float], gaps_s: list[float]) -> list[Interval]:
    """Walk sources and gaps arithmetically into a covering interval list:
    source -> label 1, gap -> label 0. `len(gaps_s)` must equal
    `len(source_durations) - 1`.
    """
    if len(source_durations) == 0:
        return []
    if len(gaps_s) != len(source_durations) - 1:
        raise ValueError(
            f"expected {len(source_durations) - 1} gaps for {len(source_durations)} "
            f"sources, got {len(gaps_s)}"
        )

    intervals: list[Interval] = []
    cursor = 0.0
    for i, duration in enumerate(source_durations):
        intervals.append((cursor, cursor + duration, 1))
        cursor += duration
        if i < len(gaps_s):
            gap = gaps_s[i]
            intervals.append((cursor, cursor + gap, 0))
            cursor += gap
    return intervals


def total_duration(source_durations: list[float], gaps_s: list[float]) -> float:
    return sum(source_durations) + sum(gaps_s)


def trim_internal_silence(
    waveform: np.ndarray,
    labels: np.ndarray,
    sample_rate: int,
    hop_s: float,
    rel_threshold_db: float = -35.0,
    min_run_frames: int = 5,
) -> np.ndarray:
    """Fix `build_concat_labels`'s blanket whole-utterance-is-speech labeling:
    flip already-label==1 frames to 0 wherever they sit in a sustained
    (>= min_run_frames, default ~160ms at a 32ms hop) low-energy run relative
    to that example's own typical speech level. Never touches label==0 (gap)
    frames -- this only removes label *noise* inside source utterances
    (leading/trailing/internal near-silence and breaths), it does not
    resegment. Operates on the clean (pre-augmentation) waveform, since a
    mixed-in noise floor would make energy an unreliable silence signal.
    """
    labels = labels.copy()
    n_frames = len(labels)
    if n_frames == 0:
        return labels

    speech_mask = labels == 1
    if not speech_mask.any():
        return labels

    frame_samples = max(1, int(round(hop_s * sample_rate)))
    frame_rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_samples
        seg = waveform[start : start + frame_samples]
        if len(seg):
            frame_rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))

    speech_rms = frame_rms[speech_mask]
    speech_rms = speech_rms[speech_rms > 0]
    if len(speech_rms) == 0:
        return labels

    reference = np.median(speech_rms)
    if reference <= 0:
        return labels
    threshold = reference * (10.0 ** (rel_threshold_db / 20.0))
    is_low = (frame_rms < threshold) & speech_mask

    i = 0
    while i < n_frames:
        if is_low[i]:
            j = i
            while j < n_frames and is_low[j]:
                j += 1
            if j - i >= min_run_frames:
                labels[i:j] = 0
            i = j
        else:
            i += 1

    return labels
