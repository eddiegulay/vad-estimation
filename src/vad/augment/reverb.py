"""Reverb simulation via RIR convolution — pure in-memory functions, no I/O.

Uses `scipy.signal.oaconvolve` (efficient for long IRs per plan §6), causally
trimmed to the input's original length so label timing stays aligned, with
peak-level normalization afterward to prevent clipping/level drift.
"""

import numpy as np
from scipy.signal import oaconvolve

_EPS = 1e-10


def convolve_rir(waveform: np.ndarray, rir: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convolve `waveform` with `rir`, keep only the first `len(waveform)`
    samples (causal trim — preserves alignment with the original label
    timeline), and peak-normalize back to the input's original peak.
    """
    if len(rir) == 0 or len(waveform) == 0:
        return waveform.copy()

    wet = oaconvolve(waveform, rir, mode="full")[: len(waveform)]

    if normalize:
        orig_peak = np.max(np.abs(waveform)) + _EPS
        wet_peak = np.max(np.abs(wet)) + _EPS
        wet = wet * (orig_peak / wet_peak)

    return wet.astype(waveform.dtype, copy=False)
