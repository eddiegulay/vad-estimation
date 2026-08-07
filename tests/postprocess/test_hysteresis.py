import numpy as np

from vad.postprocess.hysteresis import apply_hysteresis, ms_to_frames


def test_all_below_theta_off_stays_silence():
    probs = np.array([0.1, 0.2, 0.15, 0.05])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35)
    assert np.array_equal(decisions, [0, 0, 0, 0])


def test_all_above_theta_on_becomes_speech():
    probs = np.array([0.9, 0.8, 0.95, 0.85])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35, min_speech_frames=1)
    assert np.array_equal(decisions, [1, 1, 1, 1])


def test_single_frame_spike_suppressed_by_onset_debounce():
    probs = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35, min_speech_frames=3)
    assert np.array_equal(decisions, [0, 0, 0, 0, 0])


def test_sustained_onset_commits_speech_retroactively():
    probs = np.array([0.1, 0.9, 0.9, 0.9, 0.1])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35, min_speech_frames=3)
    # frames 1-3 (the 3-frame run that crossed the debounce) should all be speech
    assert decisions[1] == 1 and decisions[2] == 1 and decisions[3] == 1
    assert decisions[0] == 0


def test_brief_dip_bridged_by_hangover():
    # speech, one low dip, speech again -- hangover should bridge the single dip
    probs = np.array([0.9, 0.9, 0.9, 0.2, 0.9, 0.9])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35, min_speech_frames=1, min_silence_frames=3)
    assert np.array_equal(decisions, [1, 1, 1, 1, 1, 1])


def test_sustained_offset_ends_speech():
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    decisions = apply_hysteresis(probs, theta_on=0.5, theta_off=0.35, min_speech_frames=1, min_silence_frames=3)
    assert decisions[0] == 1 and decisions[1] == 1
    assert decisions[-1] == 0


def test_empty_input_returns_empty():
    decisions = apply_hysteresis(np.array([]))
    assert len(decisions) == 0


def test_output_is_binary_int8():
    probs = np.random.default_rng(0).uniform(0, 1, 50)
    decisions = apply_hysteresis(probs, min_speech_frames=4, min_silence_frames=6)
    assert decisions.dtype == np.int8
    assert set(np.unique(decisions)).issubset({0, 1})


def test_ms_to_frames_rounds_and_floors_at_one():
    assert ms_to_frames(250, hop_s=0.032) == round(250 / 1000 / 0.032)
    assert ms_to_frames(1, hop_s=1.0) == 1  # never rounds down to 0
