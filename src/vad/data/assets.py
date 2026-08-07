"""One-time asset preprocessing (plan §3.7): resample/mono/int16 conversion.

AMI stays uncached — already 16kHz mono, read directly from the vault at
load time. FLEURS stays uncached — cross-lingual sanity eval only, rarely
touched, never by parallel workers under load. ESC-50 (44.1kHz) and
LibriSpeech (FLAC) get resampled/decoded once to 16kHz mono int16 PCM WAV.
Aachen AIR RIRs (.mat, confirmed structure: `h_air` (1,N) float64 array,
`air_info.fs` sample rate) get converted once to 16kHz float32 .npy.
"""

import csv
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
import soundfile as sf


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Average channels if multi-channel; waveform shape (n,) or (n, ch)."""
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=1)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    gcd = np.gcd(orig_sr, target_sr)
    up, down = target_sr // gcd, orig_sr // gcd
    return scipy.signal.resample_poly(waveform, up, down)


def float_to_int16(waveform: np.ndarray) -> np.ndarray:
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def convert_to_16k_mono_int16(
    src_path: str | Path, dst_path: str | Path, target_sr: int = 16000
) -> float:
    """Read `src_path`, resample/mono/int16-convert, write to `dst_path`
    as PCM_16 WAV. Returns the resulting duration in seconds.
    """
    waveform, orig_sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    mono = to_mono(waveform)
    resampled = resample(mono, orig_sr, target_sr)
    int16_wave = float_to_int16(resampled)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), int16_wave, target_sr, subtype="PCM_16")
    return len(int16_wave) / target_sr


def read_esc50_meta(meta_csv_path: str | Path) -> list[dict]:
    with open(meta_csv_path, newline="") as f:
        return list(csv.DictReader(f))


def load_air_rir(mat_path: str | Path) -> tuple[np.ndarray, int]:
    """Load one Aachen AIR impulse response. Returns (ir_1d_float64, fs)."""
    mat = scipy.io.loadmat(str(mat_path))
    ir = np.asarray(mat["h_air"]).reshape(-1)
    fs = int(np.asarray(mat["air_info"][0, 0]["fs"]).reshape(-1)[0])
    return ir, fs


def convert_rir_mat_to_npy(
    mat_path: str | Path, dst_path: str | Path, target_sr: int = 16000
) -> float:
    """Load an Aachen AIR .mat impulse response, resample to `target_sr`,
    save as float32 .npy. Returns duration in seconds.
    """
    ir, orig_sr = load_air_rir(mat_path)
    resampled = resample(ir.astype(np.float64), orig_sr, target_sr).astype(np.float32)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(dst_path), resampled)
    return len(resampled) / target_sr
