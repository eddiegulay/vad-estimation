"""External (non-learned) hangover/hysteresis post-processing, applied
downstream of a model's raw per-frame probabilities -- never baked into the
model graph or its ONNX export. This matches how every production VAD in
the research teardown actually ships (Silero wraps its classifier in a
dual-threshold state machine; TEN VAD applies a 10-frame smoothing filter
before thresholding) and was the original architecture's stated design
intent ("external hysteresis/hangover") that the first training pass never
implemented.

Default thresholds follow Silero's own defaults (theta_on=0.5,
theta_off=0.35) and typical min-speech/min-silence ranges from the same
research (min_speech ~250ms onset debounce, min_silence ~100-800ms
hangover); see vault notes 04/07/08.
"""

import numpy as np


def apply_hysteresis(
    probs: np.ndarray,
    theta_on: float = 0.5,
    theta_off: float = 0.35,
    min_speech_frames: int = 1,
    min_silence_frames: int = 1,
) -> np.ndarray:
    """Dual-threshold (Schmitt-trigger) state machine with onset debounce
    and offset hangover, operating on a 1D sequence of per-frame speech
    probabilities. Returns a same-length int8 array of {0,1} decisions.

    - Enter the "candidate speech" state once a frame exceeds `theta_on`;
      only commit to speech (flip already-emitted frames to 1) once
      `min_speech_frames` consecutive candidate frames have accumulated --
      this suppresses single-frame clicks/coughs.
    - While in speech, drop below `theta_off` starts a "candidate silence"
      run; only commit to silence once `min_silence_frames` consecutive
      candidate frames have accumulated -- this is the hangover, it bridges
      brief mid-utterance dips instead of chopping speech into fragments.
    """
    n = len(probs)
    decisions = np.zeros(n, dtype=np.int8)
    if n == 0:
        return decisions

    in_speech = False
    run_len = 0  # length of the current same-direction candidate run

    for i in range(n):
        p = probs[i]
        if not in_speech:
            if p >= theta_on:
                run_len += 1
            else:
                run_len = 0
            if run_len >= min_speech_frames:
                in_speech = True
                decisions[i - run_len + 1 : i + 1] = 1
                run_len = 0  # reset so offset counting below starts fresh
            # else: stays 0 (already the array default)
        else:
            decisions[i] = 1
            if p < theta_off:
                run_len += 1
            else:
                run_len = 0
            if run_len >= min_silence_frames:
                in_speech = False
                decisions[i - run_len + 1 : i + 1] = 0
                run_len = 0  # reset so onset counting above starts fresh

    return decisions


def ms_to_frames(ms: float, hop_s: float) -> int:
    return max(1, round((ms / 1000.0) / hop_s))
