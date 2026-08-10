#!/usr/bin/env python3
"""Impulse-response QC for the Aachen AIR pool.

For every cached RIR: direct-path position, direct-to-reverberant ratio,
Schroeder-integrated decay times (EDT / T20 / T30 -> RT60) with the decay
fit's linearity, and a contamination probe — non-decaying or modulated
energy in the tail, which is what a recording that captured speech (or
another source) instead of a clean impulse looks like.

Also reports each IR's per-octave RT60, because a pool selected only on
broadband RT60 can still contain rooms that are pathological in the
speech band.

Usage:  .venv/bin/python scripts/qc/qc_rir.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.signal
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402

SR = 16000
EPS = 1e-20
OUT = REPO_ROOT / "data_cache" / "qc" / "rir.jsonl"


def decay_time(edc_db: np.ndarray, start_db: float, end_db: float) -> tuple[float, float]:
    """Fit a line to the Schroeder curve between two levels; return
    (RT60 extrapolated from that slope in seconds, fit R^2)."""
    below_start = np.flatnonzero(edc_db <= start_db)
    below_end = np.flatnonzero(edc_db <= end_db)
    if below_start.size == 0 or below_end.size == 0:
        return float("nan"), float("nan")
    i0, i1 = int(below_start[0]), int(below_end[0])
    if i1 - i0 < 16:
        return float("nan"), float("nan")
    t = np.arange(i0, i1) / SR
    y = edc_db[i0:i1]
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + EPS
    if slope >= 0:
        return float("nan"), 1.0 - ss_res / ss_tot
    return float(-60.0 / slope), 1.0 - ss_res / ss_tot


def schroeder(h: np.ndarray) -> np.ndarray:
    energy = h.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    return 10.0 * np.log10(edc / (edc[0] + EPS) + EPS)


def analyse(path: Path) -> dict:
    h = np.load(str(path)).astype(np.float64)
    name = path.stem
    rec = {"id": name, "path": str(path), "n_samples": int(h.size),
           "duration_s": h.size / SR}

    m = re.match(r"air_(binaural|phone)_([a-z_]+?)_(\d+)_", name + "_")
    rec["room"] = m.group(2) if m else name
    rec["mic"] = m.group(1) if m else "unknown"

    if h.size == 0 or not np.isfinite(h).all():
        rec["error"] = "empty or non-finite"
        return rec

    peak_idx = int(np.argmax(np.abs(h)))
    rec["direct_idx"] = peak_idx
    rec["direct_delay_ms"] = 1000.0 * peak_idx / SR
    rec["peak"] = float(np.abs(h).max())

    # direct-to-reverberant ratio, ±2.5 ms window around the direct path
    w = int(0.0025 * SR)
    lo, hi = max(0, peak_idx - w), min(h.size, peak_idx + w)
    e_direct = float((h[lo:hi] ** 2).sum())
    e_total = float((h**2).sum())
    e_rev = max(e_total - e_direct, EPS)
    rec["drr_db"] = float(10.0 * np.log10(e_direct / e_rev))

    edc = schroeder(h[peak_idx:])
    edt, edt_r2 = decay_time(edc, 0.0, -10.0)
    t20, t20_r2 = decay_time(edc, -5.0, -25.0)
    t30, t30_r2 = decay_time(edc, -5.0, -35.0)
    rec.update({
        "edt_s": edt, "edt_r2": edt_r2,
        "t20_rt60_s": t20, "t20_r2": t20_r2,
        "t30_rt60_s": t30, "t30_r2": t30_r2,
        "rt60_s": t30 if np.isfinite(t30) else (t20 if np.isfinite(t20) else edt),
    })

    # per-octave RT60 over the speech band
    for fc in (250, 500, 1000, 2000, 4000):
        lo_f, hi_f = fc / np.sqrt(2), min(fc * np.sqrt(2), SR / 2 - 1)
        sos = scipy.signal.butter(4, [lo_f, hi_f], btype="band", fs=SR, output="sos")
        hb = scipy.signal.sosfilt(sos, h[peak_idx:])
        t, _ = decay_time(schroeder(hb), -5.0, -25.0)
        rec[f"rt60_{fc}hz_s"] = t

    # contamination probe: energy that arrives long after the decay,
    # and whether that late energy is modulated like a real source
    late_start = peak_idx + int(1.0 * SR)
    if h.size > late_start + SR // 2:
        late = h[late_start:]
        rec["late_energy_frac"] = float((late**2).sum() / (e_total + EPS))
        rec["late_peak_rel_db"] = float(
            20 * np.log10((np.abs(late).max() + EPS) / (rec["peak"] + EPS))
        )
        # 2-8 Hz modulation depth of the late envelope: speech-like
        env = np.abs(scipy.signal.hilbert(late))
        env = scipy.signal.decimate(env, 40, ftype="fir")  # -> 400 Hz
        env = env - env.mean()
        if env.size > 256:
            f, p = scipy.signal.welch(env, fs=SR / 40, nperseg=min(1024, env.size))
            band = (f >= 2) & (f <= 8)
            rest = (f > 8) & (f <= 50)
            rec["late_mod_2_8hz_db"] = float(
                10 * np.log10((p[band].mean() + EPS) / (p[rest].mean() + EPS))
            )
    else:
        rec["late_energy_frac"] = 0.0
        rec["late_peak_rel_db"] = -200.0
        rec["late_mod_2_8hz_db"] = 0.0

    return rec


def main() -> int:
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    cache_root = REPO_ROOT / paths_cfg["cache_root"]
    idx = json.load(open(cache_root / "index" / "rir_index.json"))

    recs = []
    for r in tqdm(idx, desc="RIR"):
        try:
            recs.append(analyse(cache_root / r["cached_path"]))
        except Exception as e:
            recs.append({"id": r["filename"], "error": f"{type(e).__name__}: {e}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT}  ({sum(1 for r in recs if 'error' in r)} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
