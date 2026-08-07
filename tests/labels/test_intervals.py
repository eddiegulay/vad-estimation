import numpy as np

from vad.labels.intervals import (
    clip,
    fill_gaps,
    from_frames,
    invert,
    merge_intervals,
    merge_speech_spans,
    speech_occupancy,
    to_frames,
)


def test_invert_is_involution():
    x = [(0.0, 1.0, 1), (1.0, 2.5, 0), (2.5, 4.0, 1)]
    assert invert(invert(x)) == x


def test_invert_flips_labels():
    x = [(0.0, 1.0, 1), (1.0, 2.0, 0)]
    assert invert(x) == [(0.0, 1.0, 0), (1.0, 2.0, 1)]


def test_merge_intervals_is_idempotent():
    x = [(0.0, 1.0, 1), (0.9, 2.0, 1), (2.0, 3.0, 0), (3.0, 4.0, 0)]
    once = merge_intervals(x)
    twice = merge_intervals(once)
    assert once == twice


def test_merge_intervals_joins_overlapping_same_label():
    x = [(0.0, 1.0, 1), (0.5, 2.0, 1)]
    assert merge_intervals(x) == [(0.0, 2.0, 1)]


def test_merge_intervals_respects_gap_tolerance():
    x = [(0.0, 1.0, 1), (1.01, 2.0, 1)]
    assert merge_intervals(x, gap_tolerance_s=0.0) == [(0.0, 1.0, 1), (1.01, 2.0, 1)]
    assert merge_intervals(x, gap_tolerance_s=0.02) == [(0.0, 2.0, 1)]


def test_merge_intervals_does_not_join_different_labels():
    x = [(0.0, 1.0, 1), (1.0, 2.0, 0)]
    assert merge_intervals(x) == x


def test_merge_speech_spans_joins_across_channels():
    spans = [(0.0, 1.0), (5.0, 6.0), (0.9, 2.0)]
    assert merge_speech_spans(spans) == [(0.0, 2.0), (5.0, 6.0)]


def test_fill_gaps_produces_covering_list():
    spans = [(1.0, 2.0), (3.0, 3.5)]
    covering = fill_gaps(spans, total_duration=4.0)
    assert covering == [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0), (3.0, 3.5, 1), (3.5, 4.0, 0)]
    total = sum(e - s for s, e, _ in covering)
    assert abs(total - 4.0) < 1e-9


def test_fill_gaps_handles_speech_covering_full_duration():
    assert fill_gaps([(0.0, 5.0)], total_duration=5.0) == [(0.0, 5.0, 1)]


def test_fill_gaps_handles_no_speech():
    assert fill_gaps([], total_duration=3.0) == [(0.0, 3.0, 0)]


def test_clip_extracts_and_rezeroes_window():
    x = [(0.0, 2.0, 0), (2.0, 5.0, 1), (5.0, 8.0, 0)]
    windowed = clip(x, offset_s=1.0, duration_s=5.0)
    assert windowed == [(0.0, 1.0, 0), (1.0, 4.0, 1), (4.0, 5.0, 0)]


def test_clip_out_of_range_is_empty():
    x = [(0.0, 2.0, 1)]
    assert clip(x, offset_s=10.0, duration_s=1.0) == []


def test_to_frames_matches_expected_labels_at_hop():
    x = [(0.0, 1.0, 1), (1.0, 2.0, 0)]
    frames = to_frames(x, num_frames=4, hop_s=0.5)
    # frame centers: 0.25, 0.75, 1.25, 1.75 -> labels 1,1,0,0
    assert frames.tolist() == [1, 1, 0, 0]


def test_from_frames_round_trips_through_to_frames():
    x = [(0.0, 0.5, 1), (0.5, 1.0, 0), (1.0, 1.5, 1)]
    frames = to_frames(x, num_frames=3, hop_s=0.5)
    recovered = from_frames(frames, hop_s=0.5)
    frames_again = to_frames(recovered, num_frames=3, hop_s=0.5)
    assert np.array_equal(frames, frames_again)


def test_from_frames_run_length_encodes():
    frames = np.array([1, 1, 1, 0, 0, 1], dtype=np.int8)
    result = from_frames(frames, hop_s=0.1)
    expected = [(0.0, 0.3, 1), (0.3, 0.5, 0), (0.5, 0.6, 1)]
    assert len(result) == len(expected)
    for (s, e, label), (exp_s, exp_e, exp_label) in zip(result, expected):
        assert abs(s - exp_s) < 1e-9
        assert abs(e - exp_e) < 1e-9
        assert label == exp_label


def test_speech_occupancy():
    x = [(0.0, 1.0, 0), (1.0, 3.0, 1), (3.0, 4.0, 0)]
    assert abs(speech_occupancy(x) - 0.5) < 1e-9


def test_speech_occupancy_empty_is_zero():
    assert speech_occupancy([]) == 0.0
