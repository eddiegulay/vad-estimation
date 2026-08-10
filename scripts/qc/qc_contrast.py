#!/usr/bin/env python3
"""Speech/non-speech contrast: does a training example look like a test file?

The single number that most directly predicts how hard a VAD problem is:
the level gap between speech-labelled frames and non-speech-labelled
frames, plus how much the two distributions overlap. A file whose silence
sits 60 dB below its speech is trivial; a file where it sits 10 dB below
is the real problem.

Measures that on the TEN test set (human labels, real audio), then on
simulated training examples under several augmentation policies, so the
policies can be compared against the target instead of argued about.
Also compares the *spectrum* of non-speech frames, because matching the
level while getting the spectral shape wrong still leaves a giveaway cue.

Usage:  .venv/bin/python scripts/qc/qc_contrast.py [n_sim]
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.augment.gain import random_gain  # noqa: E402
from vad.augment.gaps import make_gap_audio  # noqa: E402
from vad.augment.noise_mix import mix_at_snr  # noqa: E402
from vad.augment.reverb import convolve_rir  # noqa: E402
from vad.config import load_yaml  # noqa: E402
from vad.labels.ten import parse_scv  # noqa: E402

SR = 16000
FRAME = 512
EPS = 1e-12
OUT = REPO_ROOT / "data_cache" / "qc" / "contrast.json"
V2_ROOMS = ("meeting", "office", "booth", "lecture")


def snr_v1(rng):
    return rng.uniform(-5.0, 20.0)


def snr_v2(rng):
    """DESIGN-NOTES §7 mixture."""
    u = rng.random()
    if u < 0.55:
        return float(np.clip(rng.normal(18.0, 6.0), 10.0, 35.0))
    if u < 0.87:
        return rng.uniform(3.0, 10.0)
    return rng.uniform(-2.0, 3.0)


# gap_kind: what the silence between utterances is made of
# noise_prob: probability an ESC-50 event is mixed over the whole example
# clean_snr:  if set, "clean" examples still get noise, at this SNR range
POLICIES = {
    "v1": dict(snr=snr_v1, noise_prob=1.0, rir_prob=0.5, rooms=None,
               gain=(-3.0, 3.0), gap_kind="silence_or_lowlevel_noise",
               gap_range=(0.2, 3.0), long_p=0.08, long_range=(3.0, 8.0),
               clean_snr=None, trim_direct=False),
    "v2_spec": dict(snr=snr_v2, noise_prob=0.85, rir_prob=0.35, rooms=V2_ROOMS,
                    gain=(-18.0, 6.0), gap_kind="silence_or_lowlevel_noise",
                    gap_range=(0.15, 1.2), long_p=0.05, long_range=(1.5, 4.0),
                    clean_snr=None, trim_direct=True),
    "v2_floor": dict(snr=snr_v2, noise_prob=0.85, rir_prob=0.35, rooms=V2_ROOMS,
                     gain=(-18.0, 6.0), gap_kind="low_level_noise",
                     gap_range=(0.15, 1.2), long_p=0.05, long_range=(1.5, 4.0),
                     clean_snr=(25.0, 40.0), trim_direct=True),
}


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    fr = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt((fr**2).mean(axis=1)), EPS))


def nonspeech_spectrum(x: np.ndarray, lab: np.ndarray) -> np.ndarray | None:
    """Mean power spectrum of non-speech frames, normalised to unit sum."""
    n = min(len(x) // FRAME, len(lab))
    if n == 0:
        return None
    fr = x[: n * FRAME].reshape(n, FRAME)
    ns = fr[~lab[:n].astype(bool)]
    if len(ns) < 8:
        return None
    spec = (np.abs(np.fft.rfft(ns * np.hanning(FRAME), axis=1)) ** 2).mean(axis=0)
    return spec / (spec.sum() + EPS)


def contrast(x: np.ndarray, lab: np.ndarray) -> dict | None:
    db = frame_db(x)
    n = min(len(db), len(lab))
    db, lab = db[:n], lab[:n].astype(bool)
    if lab.sum() < 8 or (~lab).sum() < 8:
        return None
    sp, ns = db[lab], db[~lab]
    return {
        "contrast_db": float(np.median(sp) - np.median(ns)),
        "speech_p50": float(np.median(sp)),
        "nonspeech_p50": float(np.median(ns)),
        "overlap": float((ns > np.percentile(sp, 5)).mean()),
        "occupancy": float(lab.mean()),
    }


def simulate(name, pol, utts, noises, rirs, n_sim, seed=20260810):
    rng = np.random.default_rng(seed)
    pool = rirs if pol["rooms"] is None else [r for r in rirs
                                              if any(m in r.name for m in pol["rooms"])]
    recs, specs = [], []
    for _ in tqdm(range(n_sim), desc=name):
        k = int(rng.integers(2, 7))
        pieces, is_sp = [], []
        for i in range(k):
            w, _ = sf.read(str(rng.choice(utts)), dtype="float32", always_2d=False)
            pieces.append(w)
            is_sp.append(True)
            if i < k - 1:
                lo, hi = pol["gap_range"]
                gap = (rng.uniform(*pol["long_range"]) if rng.random() < pol["long_p"]
                       else rng.uniform(lo, hi))
                pieces.append(make_gap_audio(rng, gap, SR, pol["gap_kind"]))
                is_sp.append(False)

        x = np.concatenate(pieces)
        lab = np.zeros(len(x) // FRAME + 1, dtype=bool)
        pos = 0
        for piece, speech in zip(pieces, is_sp):
            if speech:
                lab[pos // FRAME: (pos + len(piece)) // FRAME] = True
            pos += len(piece)

        if rng.random() < pol["rir_prob"] and pool:
            h = np.load(str(rng.choice(pool))).astype(np.float32)
            if pol["trim_direct"]:
                h = h[int(np.argmax(np.abs(h))):]
            x = convolve_rir(x, h)

        if rng.random() < pol["noise_prob"]:
            snr = pol["snr"](rng)
        elif pol["clean_snr"] is not None:
            snr = rng.uniform(*pol["clean_snr"])
        else:
            snr = None
        if snr is not None:
            nz, _ = sf.read(str(rng.choice(noises)), dtype="float32", always_2d=False)
            x = mix_at_snr(rng, x, nz.astype(np.float32), snr)

        x = random_gain(rng, x, pol["gain"])

        c = contrast(x, lab)
        if c:
            recs.append(c)
            s = nonspeech_spectrum(x, lab)
            if s is not None:
                specs.append(s)
    return recs, specs


def summarise(name, recs):
    keys = ["contrast_db", "nonspeech_p50", "overlap", "occupancy"]
    return {"name": name, "n": len(recs),
            **{k: {"p10": float(np.percentile([r[k] for r in recs], 10)),
                   "p50": float(np.median([r[k] for r in recs])),
                   "p90": float(np.percentile([r[k] for r in recs], 90))}
               for k in keys}}


def main() -> int:
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    vault = Path(cfg["vault_data_root"])
    cache = REPO_ROOT / cfg["cache_root"]

    ten, ten_specs = [], []
    for p in sorted((vault / "ten-vad-testset").glob("*.scv")):
        _, iv = parse_scv(p)
        x, _ = sf.read(str(p.with_suffix(".wav")), dtype="float32", always_2d=False)
        lab = np.zeros(len(x) // FRAME + 1, dtype=bool)
        for a, b, l in iv:
            if l:
                lab[int(a * SR) // FRAME: int(b * SR) // FRAME] = True
        c = contrast(x, lab)
        if c:
            ten.append(c)
            s = nonspeech_spectrum(x, lab)
            if s is not None:
                ten_specs.append(s)

    ls = json.load(open(cache / "index" / "librispeech_index.json"))
    utts = [cache / r["cached_path"] for r in ls if r["split"].startswith("dev")]
    esc = json.load(open(cache / "index" / "esc50_index.json"))
    noises = [cache / r["cached_path"] for r in esc if r["fold"] in (1, 2, 3, 4)]
    rirs = sorted((cache / "rir").glob("*.npy"))

    report = {"ten": summarise("ten", ten)}
    ten_c = [r["contrast_db"] for r in ten]
    ten_mean_spec = np.mean(ten_specs, axis=0)

    for name, pol in POLICIES.items():
        recs, specs = simulate(name, pol, utts, noises, rirs, n_sim)
        rep = summarise(name, recs)
        ks = scipy.stats.ks_2samp([r["contrast_db"] for r in recs], ten_c)
        rep["ks_contrast_vs_ten"] = {"stat": float(ks.statistic), "p": float(ks.pvalue)}
        ks2 = scipy.stats.ks_2samp([r["nonspeech_p50"] for r in recs],
                                   [r["nonspeech_p50"] for r in ten])
        rep["ks_nonspeech_level_vs_ten"] = {"stat": float(ks2.statistic), "p": float(ks2.pvalue)}
        if specs:
            m = np.mean(specs, axis=0)
            # spectral divergence of the non-speech (i.e. "what silence sounds like")
            rep["nonspeech_spectral_kl"] = float(
                np.sum(ten_mean_spec * np.log((ten_mean_spec + EPS) / (m + EPS)))
            )
        report[name] = rep

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"\n{'policy':10s} {'contrast dB p10/p50/p90':>26s} {'non-speech dB p10/p50/p90':>28s}"
          f" {'overlap p50':>11s} {'KS(contrast)':>20s} {'KS(ns level)':>20s} {'specKL':>8s}")
    for k in ("ten",) + tuple(POLICIES):
        r = report[k]
        c, ns = r["contrast_db"], r["nonspeech_p50"]
        line = (f"{k:10s} {c['p10']:8.1f}/{c['p50']:7.1f}/{c['p90']:7.1f}"
                f"  {ns['p10']:8.1f}/{ns['p50']:7.1f}/{ns['p90']:7.1f}"
                f" {r['overlap']['p50']:11.3f}")
        if "ks_contrast_vs_ten" in r:
            line += (f"  D={r['ks_contrast_vs_ten']['stat']:.3f} p={r['ks_contrast_vs_ten']['p']:.1e}"
                     f"  D={r['ks_nonspeech_level_vs_ten']['stat']:.3f}"
                     f" p={r['ks_nonspeech_level_vs_ten']['p']:.1e}"
                     f" {r.get('nonspeech_spectral_kl', float('nan')):8.3f}")
        print(line)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
