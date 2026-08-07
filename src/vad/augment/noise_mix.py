"""SNR-controlled additive noise mixing — pure in-memory functions, no I/O.

RMS-based gain solve: scale the noise clip so the mixed signal achieves the
requested SNR relative to the speech signal's RMS, matching noise length to
the signal by random-cropping (if longer) or looping (if shorter).
"""

import numpy as np

_EPS = 1e-10


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform)) + _EPS))


def fit_noise_length(
    rng: np.random.Generator, noise: np.ndarray, target_len: int
) -> np.ndarray:
    """Match `noise` to `target_len` samples: random-crop if longer, loop
    (tiled from a random phase) if shorter.
    """
    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) >= target_len:
        start = int(rng.integers(0, len(noise) - target_len + 1))
        return noise[start : start + target_len]

    reps = target_len // len(noise) + 1
    tiled = np.tile(noise, reps)
    start = int(rng.integers(0, len(tiled) - target_len + 1)) if len(tiled) > target_len else 0
    return tiled[start : start + target_len]


def mix_at_snr(
    rng: np.random.Generator, signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix `noise` into `signal` scaled to achieve `snr_db` (signal-to-noise
    ratio in dB, relative to RMS). `noise` is length-matched to `signal`
    first (crop/loop). Output is not renormalized — caller decides whether
    to peak-normalize afterward.
    """
    noise_matched = fit_noise_length(rng, noise, len(signal))

    signal_rms = rms(signal)
    noise_rms = rms(noise_matched)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / max(noise_rms, _EPS)

    return signal + noise_matched * scale


def measured_snr_db(signal: np.ndarray, noise_component: np.ndarray) -> float:
    """Recover the achieved SNR given the clean signal and the (scaled)
    noise component that was added to it — used by tests to verify
    `mix_at_snr` hit its target.
    """
    return 20.0 * np.log10(rms(signal) / max(rms(noise_component), _EPS))
