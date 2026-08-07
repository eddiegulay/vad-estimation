"""Gap-audio synthesis for the silence/low-level-noise segments inserted
between concatenated utterances in synthetic training examples (companion
to `labels/synthetic.py`, which handles the timing/label side of the same
recipe) — pure in-memory functions, no I/O.
"""

import numpy as np

GapKind = str  # "silence" | "low_level_noise" | "silence_or_lowlevel_noise"


def make_gap_audio(
    rng: np.random.Generator,
    duration_s: float,
    sample_rate: int,
    gap_kind: GapKind = "silence_or_lowlevel_noise",
    noise_floor_db: float = -50.0,
) -> np.ndarray:
    """Generate `duration_s` seconds of gap audio at `sample_rate`.

    - "silence": all zeros.
    - "low_level_noise": Gaussian noise at `noise_floor_db` amplitude.
    - "silence_or_lowlevel_noise": coin-flip between the two per call.
    """
    n = int(round(duration_s * sample_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    if gap_kind == "silence":
        is_silent = True
    elif gap_kind == "low_level_noise":
        is_silent = False
    elif gap_kind == "silence_or_lowlevel_noise":
        is_silent = rng.random() < 0.5
    else:
        raise ValueError(f"unknown gap_kind: {gap_kind}")

    if is_silent:
        return np.zeros(n, dtype=np.float32)

    amplitude = 10.0 ** (noise_floor_db / 20.0)
    return (rng.standard_normal(n).astype(np.float32) * amplitude)
