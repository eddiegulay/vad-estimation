#!/usr/bin/env python3
"""Do the labels agree with the audio?

Two questions, both answered from energy alone (no model, no teacher):

1. **AMI** — the union-of-channels annotation says "someone is speaking".
   How much of that speech-labelled time is digitally silent, or below any
   plausible speech level? Channel dropouts labelled speech are the worst
   possible training example.
2. **LibriSpeech** — v1 labels every utterance 100% speech end to end. How
   much of each utterance is actually leading/trailing/internal silence
   carrying a speech label?

Frame grid is the project's own: 512-sample frames at a 512-sample hop.

Usage:  .venv/bin/python scripts/qc/qc_label_energy.py [ami|librispeech ...]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402
from vad.labels.ami import build_meeting_speech_intervals  # noqa: E402

SR = 16000
FRAME = 512
EPS = 1e-12
OUT_DIR = REPO_ROOT / "data_cache" / "qc"


def frame_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    fr = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms = np.sqrt((fr**2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, EPS)), (np.abs(fr).max(axis=1) == 0.0)


def ami_one(args) -> dict:
    mid, audio_path, seg_dir = args
    info = sf.info(audio_path)
    dur = info.frames / info.samplerate
    intervals = build_meeting_speech_intervals(seg_dir, mid, dur)

    x, _ = sf.read(audio_path, dtype="float32", always_2d=False)
    db, zero = frame_stats(x)
    n = len(db)
    lab = np.zeros(n, dtype=bool)
    for start, end, label in intervals:
        if label:
            lab[int(start * SR) // FRAME: min(int(end * SR) // FRAME, n)] = True

    sp, ns = lab.sum(), (~lab).sum()
    # a per-meeting "speech level" reference: median of speech-labelled frames
    ref = float(np.median(db[lab])) if sp else float("nan")
    out = {
        "corpus": "ami", "id": mid, "series": mid[:-1],
        "duration_s": dur, "n_frames": int(n), "occupancy": float(lab.mean()),
        "speech_ref_db": ref,
        "zero_in_speech": float((zero & lab).sum() / max(sp, 1)),
        "zero_in_nonspeech": float((zero & ~lab).sum() / max(ns, 1)),
    }
    for t in (-70, -60, -50, -40):
        out[f"speech_below_{-t}db"] = float(((db < t) & lab).sum() / max(sp, 1))
    for t in (-70, -60, -50, -40):
        out[f"nonspeech_below_{-t}db"] = float(((db < t) & ~lab).sum() / max(ns, 1))
    # non-speech frames that are LOUD relative to this meeting's speech
    out["nonspeech_above_ref_minus10"] = float(((db > ref - 10) & ~lab).sum() / max(ns, 1))
    # longest run of speech-labelled digital silence
    m = zero & lab
    if m.any():
        pad = np.concatenate(([False], m, [False]))
        e = np.flatnonzero(pad[1:] != pad[:-1])
        out["longest_zero_in_speech_s"] = float((e[1::2] - e[0::2]).max() * FRAME / SR)
    else:
        out["longest_zero_in_speech_s"] = 0.0
    return out


def ls_one(args) -> dict:
    path, meta = args
    x, _ = sf.read(path, dtype="float32", always_2d=False)
    db, zero = frame_stats(x)
    if len(db) == 0:
        return {**meta, "error": "too short"}
    # per-utterance adaptive floor: speech reference is the 90th pct frame
    ref = float(np.percentile(db, 90))
    out = {"corpus": "librispeech", **meta, "n_frames": int(len(db)),
           "duration_s": len(x) / SR, "ref_db": ref}
    # leading / trailing silence, at a -35 dB-below-reference gate
    gate = ref - 35.0
    active = np.flatnonzero(db > gate)
    if active.size:
        out["lead_silence_s"] = float(active[0] * FRAME / SR)
        out["trail_silence_s"] = float((len(db) - 1 - active[-1]) * FRAME / SR)
        interior = db[active[0]: active[-1] + 1]
        out["interior_silence_frac"] = float((interior <= gate).mean())
        # longest interior silence run
        m = interior <= gate
        if m.any():
            pad = np.concatenate(([False], m, [False]))
            e = np.flatnonzero(pad[1:] != pad[:-1])
            out["longest_interior_silence_s"] = float((e[1::2] - e[0::2]).max() * FRAME / SR)
        else:
            out["longest_interior_silence_s"] = 0.0
    else:
        out["lead_silence_s"] = out["trail_silence_s"] = len(x) / SR
        out["interior_silence_frac"] = 1.0
        out["longest_interior_silence_s"] = len(x) / SR
    out["frac_below_ref_minus35"] = float((db <= gate).mean())
    out["frac_below_ref_minus25"] = float((db <= ref - 25.0).mean())
    out["frac_zero"] = float(zero.mean())
    return out


def main() -> int:
    cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    vault = Path(cfg["vault_data_root"])
    cache_root = REPO_ROOT / cfg["cache_root"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["ami", "librispeech"]

    if "ami" in which:
        idx = json.load(open(cache_root / "index" / "ami_index.json"))
        seg = str(vault / "ami" / "annotations" / "segments")
        jobs = [(r["meeting_id"], r["audio_path"], seg) for r in idx]
        recs = []
        with ProcessPoolExecutor(max_workers=4) as ex:
            for r in tqdm(ex.map(ami_one, jobs), total=len(jobs), desc="ami labels"):
                recs.append(r)
        with open(OUT_DIR / "ami_labels.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"-> {OUT_DIR / 'ami_labels.jsonl'}")

    if "librispeech" in which:
        idx = json.load(open(cache_root / "index" / "librispeech_index.json"))
        jobs = []
        for r in idx:
            p = Path(r["cached_path"])
            jobs.append((str(cache_root / p),
                         {"id": p.stem, "split": r["split"], "speaker": p.parts[-3]}))
        recs = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            for r in tqdm(ex.map(ls_one, jobs, chunksize=16), total=len(jobs), desc="ls labels"):
                recs.append(r)
        with open(OUT_DIR / "librispeech_labels.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"-> {OUT_DIR / 'librispeech_labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
