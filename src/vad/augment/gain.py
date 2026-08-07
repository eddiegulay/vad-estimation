"""Gain augmentation — pure in-memory functions, no I/O."""

import numpy as np


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return waveform * (10.0 ** (gain_db / 20.0))


def random_gain(
    rng: np.random.Generator, waveform: np.ndarray, range_db: tuple[float, float] = (-3.0, 3.0)
) -> np.ndarray:
    gain_db = rng.uniform(range_db[0], range_db[1])
    return apply_gain_db(waveform, gain_db)
