"""Interval-list label representation, shared by every corpus and every
architecture (see plan §3.2 — hop-agnostic storage, dense arrays only at
load time).

An "interval list" is a list of ``(start_s, end_s, label)`` tuples. Two
shapes are used:

- **covering**: sorted, contiguous, no gaps — spans ``[0, total_duration)``
  with an explicit 0/1 label on every sub-range. This is the manifest
  on-disk format (``label_intervals`` in every recipe).
- **speech-only** (sparse): a list of ``(start_s, end_s)`` pairs where
  label=1 is implicit and everything else is silence. Used transiently
  while merging multiple AMI channels before filling gaps into a covering
  list.
"""

import numpy as np

Interval = tuple[float, float, int]
SpanPair = tuple[float, float]


def merge_intervals(intervals: list[Interval], gap_tolerance_s: float = 0.0) -> list[Interval]:
    """Merge overlapping/near-adjacent same-label intervals.

    Sorts by start, then folds any interval into the previous one if it
    shares the same label and starts before ``prev_end + gap_tolerance_s``.
    Idempotent: ``merge_intervals(merge_intervals(x)) == merge_intervals(x)``.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: (iv[0], iv[1]))
    merged: list[list[float | int]] = [list(ordered[0])]
    for start, end, label in ordered[1:]:
        prev = merged[-1]
        if label == prev[2] and start <= prev[1] + gap_tolerance_s:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end, label])
    return [(s, e, int(l)) for s, e, l in merged]


def invert(intervals: list[Interval]) -> list[Interval]:
    """Flip every interval's binary label (0<->1). ``invert(invert(x)) == x``."""
    return [(start, end, 1 - label) for start, end, label in intervals]


def merge_speech_spans(spans: list[SpanPair], gap_tolerance_s: float = 0.0) -> list[SpanPair]:
    """Merge overlapping/near-adjacent sparse speech spans (label implicit)."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda sp: (sp[0], sp[1]))
    merged: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        prev = merged[-1]
        if start <= prev[1] + gap_tolerance_s:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def fill_gaps(speech_spans: list[SpanPair], total_duration: float) -> list[Interval]:
    """Turn sparse, non-overlapping speech spans into a covering interval
    list spanning ``[0, total_duration)``, filling gaps with label 0.

    ``speech_spans`` must already be merged/non-overlapping (see
    :func:`merge_speech_spans`) and sorted by start.
    """
    covering: list[Interval] = []
    cursor = 0.0
    for start, end in speech_spans:
        start = max(0.0, min(start, total_duration))
        end = max(0.0, min(end, total_duration))
        if start > cursor:
            covering.append((cursor, start, 0))
        if end > start:
            covering.append((start, end, 1))
        cursor = max(cursor, end)
    if cursor < total_duration:
        covering.append((cursor, total_duration, 0))
    return covering


def clip(intervals: list[Interval], offset_s: float, duration_s: float) -> list[Interval]:
    """Extract the ``[offset_s, offset_s + duration_s)`` window from a
    covering interval list and re-zero its time base.
    """
    window_end = offset_s + duration_s
    out: list[Interval] = []
    for start, end, label in intervals:
        clipped_start = max(start, offset_s)
        clipped_end = min(end, window_end)
        if clipped_end > clipped_start:
            out.append((clipped_start - offset_s, clipped_end - offset_s, label))
    return out


def to_frames(intervals: list[Interval], num_frames: int, hop_s: float) -> np.ndarray:
    """Materialize a covering interval list into a dense per-frame label
    array at the given hop, sampled at each frame's center time.
    """
    if num_frames == 0:
        return np.zeros(0, dtype=np.int8)
    centers = (np.arange(num_frames, dtype=np.float64) + 0.5) * hop_s
    frames = np.zeros(num_frames, dtype=np.int8)
    for start, end, label in intervals:
        mask = (centers >= start) & (centers < end)
        frames[mask] = label
    return frames


def from_frames(frame_labels: np.ndarray, hop_s: float) -> list[Interval]:
    """Run-length encode a dense per-frame label array back into a covering
    interval list.
    """
    n = len(frame_labels)
    if n == 0:
        return []
    intervals: list[Interval] = []
    run_label = int(frame_labels[0])
    run_start_idx = 0
    for i in range(1, n):
        label = int(frame_labels[i])
        if label != run_label:
            intervals.append((run_start_idx * hop_s, i * hop_s, run_label))
            run_label = label
            run_start_idx = i
    intervals.append((run_start_idx * hop_s, n * hop_s, run_label))
    return intervals


def speech_occupancy(intervals: list[Interval]) -> float:
    """Fraction of total covered duration with label 1."""
    total = sum(end - start for start, end, _ in intervals)
    if total <= 0:
        return 0.0
    speech = sum(end - start for start, end, label in intervals if label == 1)
    return speech / total
