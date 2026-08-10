#!/usr/bin/env python3
"""Is the noise corpus actually free of speech?

Every ESC-50 clip is scored with silero-vad's raw frame probabilities. Any
clip carrying speech is a clip that, mixed into a silence-labelled frame,
teaches the model that speech is silence. v1 selected "vocal confusers" by
ESC-50 category name; this measures the property directly.

Also reports each clip's usable-noise fraction: ESC-50 clips are 5 s but
many are mostly digital zero padding, which breaks SNR mixing (the noise
RMS is computed over the padding too, so the audible part lands louder
than the requested SNR).

Usage:  .venv/bin/python scripts/qc/qc_esc50_speech.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402

SR = 16000
WIN = 512          # silero v5+ expects 512-sample windows at 16 kHz
EPS = 1e-12
OUT = REPO_ROOT / "data_cache" / "qc" / "esc50_speech.jsonl"


def teacher_probs(model, x: np.ndarray) -> np.ndarray:
    """Raw per-window speech probabilities. State is reset per clip."""
    model.reset_states()
    n = len(x) // WIN
    probs = np.zeros(n, dtype=np.float32)
    t = torch.from_numpy(x[: n * WIN].reshape(n, WIN))
    for i in range(n):
        probs[i] = float(model(t[i: i + 1], SR).item())
    return probs


def main() -> int:
    import importlib.metadata as md
    from silero_vad import load_silero_vad

    version = md.version("silero-vad")
    model = load_silero_vad(onnx=False)

    cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    cache_root = REPO_ROOT / cfg["cache_root"]
    idx = json.load(open(cache_root / "index" / "esc50_index.json"))

    recs = []
    for r in tqdm(idx, desc="ESC-50 speech probe"):
        path = cache_root / r["cached_path"]
        x, _ = sf.read(str(path), dtype="float32", always_2d=False)
        p = teacher_probs(model, x)

        n = len(x) // WIN
        fr = x[: n * WIN].reshape(n, WIN).astype(np.float64)
        rms = np.sqrt((fr**2).mean(axis=1))
        db = 20 * np.log10(np.maximum(rms, EPS))
        # "usable" = frames within 40 dB of the clip's own loudest frame
        ref = float(db.max()) if n else -240.0
        usable = db > ref - 40.0
        active_rms = float(np.sqrt((fr[usable] ** 2).mean())) if usable.any() else 0.0
        whole_rms = float(np.sqrt((fr**2).mean())) if n else 0.0

        # longest contiguous speech run at 0.5
        sp = p >= 0.5
        longest = 0
        if sp.any():
            pad = np.concatenate(([False], sp, [False]))
            e = np.flatnonzero(pad[1:] != pad[:-1])
            longest = int((e[1::2] - e[0::2]).max())

        recs.append({
            "id": r["filename"], "category": r["category"], "fold": r["fold"],
            "esc10": r["esc10"],
            "sha1": hashlib.sha1(x.tobytes()).hexdigest(),
            "speech_frac_05": float((p >= 0.5).mean()),
            "speech_frac_03": float((p >= 0.3).mean()),
            "speech_frac_09": float((p >= 0.9).mean()),
            "max_prob": float(p.max()) if n else 0.0,
            "mean_prob": float(p.mean()) if n else 0.0,
            "longest_speech_run_s": longest * WIN / SR,
            "usable_frac": float(usable.mean()) if n else 0.0,
            "active_rms_db": 20 * np.log10(max(active_rms, EPS)),
            "whole_rms_db": 20 * np.log10(max(whole_rms, EPS)),
            # how much louder the audible part is than the whole-clip RMS
            # used by mix_at_snr -- i.e. the SNR error this clip induces
            "snr_error_db": 20 * np.log10(max(active_rms, EPS) / max(whole_rms, EPS)),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(json.dumps({"_meta": {"teacher": "silero-vad", "version": version,
                                      "threshold": 0.5, "window": WIN, "sr": SR}}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"-> {OUT}  (silero-vad {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
