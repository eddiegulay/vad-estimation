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
