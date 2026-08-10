#!/usr/bin/env python3
"""Waveform-level QC scan over every audio asset the project uses.

Computes, per file, the statistics that decide whether audio is usable:
integrity (finite, non-empty, decodable), level (peak/RMS/DC/crest),
pathology (clipping runs, digital silence runs, DC bias, mains hum),
dynamics (frame-RMS percentiles, noise floor, dynamic range), and
bandwidth (spectral rolloff, effective lowpass cutoff, HF energy share).

Streams long files in blocks, so AMI's 87-minute meetings never land in
RAM whole. Output is one JSONL record per file per corpus under
`data_cache/qc/`.

Usage:  .venv/bin/python scripts/qc/qc_scan.py [corpus ...]
        corpora: esc50 librispeech ami ten fleurs  (default: all)
"""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402

SR = 16000
FRAME = 512          # 32 ms — the project's chunk size
HOP = 256            # 16 ms
BLOCK = 1 << 22      # 4 M samples (~4.4 min) per streamed block
CLIP_THRESH = 0.9995
EPS = 1e-12
OUT_DIR = REPO_ROOT / "data_cache" / "qc"


# ---------------------------------------------------------------- streaming


class Accum:
    """Incremental waveform statistics over arbitrarily many blocks."""

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.sumsq = 0.0
        self.peak = 0.0
        self.n_nonfinite = 0
        self.n_zero = 0
        self.n_clip = 0
        self.max_zero_run = 0
        self.max_clip_run = 0
        self._zero_carry = 0
        self._clip_carry = 0
        self.frame_rms = []
        self.spec_acc = None
        self.spec_n = 0
        self.hasher = hashlib.sha1()
        self._tail = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _runs(mask: np.ndarray, carry: int) -> tuple[int, int]:
        """Longest True-run in `mask` given `carry` leading Trues from the
        previous block; returns (longest_including_carry, trailing_run)."""
        if mask.size == 0:
            return carry, carry
        # boundaries of True runs
        padded = np.concatenate(([False], mask, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        if starts.size == 0:
            return carry, 0
        lengths = ends - starts
        if starts[0] == 0:
            lengths[0] += carry
        longest = int(lengths.max())
        trailing = int(lengths[-1]) if ends[-1] == mask.size else 0
        return longest, trailing

    def push(self, x: np.ndarray) -> None:
        self.hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())

        finite = np.isfinite(x)
        self.n_nonfinite += int((~finite).sum())
        if not finite.all():
            x = np.where(finite, x, 0.0).astype(np.float32)

        self.n += x.size
        self.total += float(x.sum(dtype=np.float64))
        self.sumsq += float(np.dot(x.astype(np.float64), x.astype(np.float64)))
        self.peak = max(self.peak, float(np.abs(x).max(initial=0.0)))

        zero_mask = x == 0.0
        self.n_zero += int(zero_mask.sum())
        longest, self._zero_carry = self._runs(zero_mask, self._zero_carry)
        self.max_zero_run = max(self.max_zero_run, longest)

        clip_mask = np.abs(x) >= CLIP_THRESH
        self.n_clip += int(clip_mask.sum())
        longest, self._clip_carry = self._runs(clip_mask, self._clip_carry)
        self.max_clip_run = max(self.max_clip_run, longest)

        # frame RMS, carrying the sub-frame remainder across blocks
        buf = np.concatenate([self._tail, x]) if self._tail.size else x
        n_frames = 1 + (buf.size - FRAME) // HOP if buf.size >= FRAME else 0
        if n_frames > 0:
            idx = np.arange(n_frames) * HOP
            frames = np.lib.stride_tricks.as_strided(
                buf, shape=(n_frames, FRAME), strides=(buf.strides[0] * HOP, buf.strides[0])
            )
            self.frame_rms.append(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)))
            self._tail = buf[idx[-1] + HOP:].copy()
        else:
            self._tail = buf.copy()

        # average power spectrum, subsampled
        if buf.size >= 8192:
            n_seg = min(64, buf.size // 8192)
            starts = np.linspace(0, buf.size - 8192, n_seg).astype(int)
            segs = np.stack([buf[s:s + 8192] for s in starts])
            win = np.hanning(8192)
            spec = np.abs(np.fft.rfft(segs * win, axis=1)) ** 2
            s = spec.sum(axis=0)
            self.spec_acc = s if self.spec_acc is None else self.spec_acc + s
            self.spec_n += n_seg


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, EPS)))


def analyse(path: Path, meta: dict) -> dict:
    rec = {"path": str(path), **meta}
    try:
        info = sf.info(str(path))
    except Exception as e:  # unreadable header
        return {**rec, "error": f"info: {type(e).__name__}: {e}"}

    rec.update(
        {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "duration_s": info.frames / info.samplerate if info.samplerate else 0.0,
        }
    )

    acc = Accum()
    try:
        with sf.SoundFile(str(path)) as f:
            while True:
                block = f.read(BLOCK, dtype="float32", always_2d=False)
                if len(block) == 0:
                    break
                if block.ndim > 1:
                    block = block.mean(axis=1)
                acc.push(block.astype(np.float32))
    except Exception as e:
        return {**rec, "error": f"read: {type(e).__name__}: {e}"}

    if acc.n == 0:
        return {**rec, "error": "empty", "n_samples": 0}

    mean = acc.total / acc.n
    rms = float(np.sqrt(max(acc.sumsq / acc.n - 0.0, 0.0)))
    rec.update(
        {
            "n_samples": acc.n,
            "sha1": acc.hasher.hexdigest(),
            "n_nonfinite": acc.n_nonfinite,
            "peak": acc.peak,
            "peak_dbfs": _db(acc.peak),
            "rms": rms,
            "rms_dbfs": _db(rms),
            "crest_db": _db(acc.peak) - _db(rms),
            "dc_offset": mean,
            "dc_rel_db": _db(abs(mean)) - _db(rms),
            "clip_frac": acc.n_clip / acc.n,
            "max_clip_run": acc.max_clip_run,
            "zero_frac": acc.n_zero / acc.n,
            "max_zero_run_s": acc.max_zero_run / SR,
        }
    )

    if acc.frame_rms:
        fr = np.concatenate(acc.frame_rms)
        frdb = 20.0 * np.log10(np.maximum(fr, EPS))
        p = np.percentile(frdb, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        rec.update(
            {
                "n_frames": int(fr.size),
                "frame_db_p01": p[0], "frame_db_p05": p[1], "frame_db_p10": p[2],
                "frame_db_p25": p[3], "frame_db_p50": p[4], "frame_db_p75": p[5],
                "frame_db_p90": p[6], "frame_db_p95": p[7], "frame_db_p99": p[8],
                "noise_floor_db": p[1],
                "dynamic_range_db": p[7] - p[1],
                # fraction of frames within 3 dB of the file's own floor
                "frac_at_floor": float((frdb <= p[1] + 3.0).mean()),
                "frac_below_60db": float((frdb < -60.0).mean()),
                "frac_below_40db": float((frdb < -40.0).mean()),
            }
        )

    if acc.spec_acc is not None and acc.spec_n:
        psd = acc.spec_acc / acc.spec_n
        freqs = np.fft.rfftfreq(8192, 1.0 / SR)
        total = psd.sum() + EPS
        cum = np.cumsum(psd) / total
        rolloff99 = float(freqs[int(np.searchsorted(cum, 0.99))])
        rolloff995 = float(freqs[int(np.searchsorted(cum, 0.995))])
        psd_db = 10.0 * np.log10(psd + EPS)
        ref = float(psd_db.max())
        above = np.flatnonzero(psd_db > ref - 50.0)
        cutoff = float(freqs[above[-1]]) if above.size else 0.0

        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(psd[m].sum() / total)

        # mains hum: 50/60 Hz + harmonics vs local neighbourhood
        hum = {}
        for f0 in (50.0, 60.0):
            ratios = []
            for k in (1, 2, 3):
                fc = f0 * k
                tgt = (freqs >= fc - 3) & (freqs <= fc + 3)
                nb = ((freqs >= fc - 20) & (freqs < fc - 5)) | ((freqs > fc + 5) & (freqs <= fc + 20))
                if tgt.any() and nb.any():
                    ratios.append(10 * np.log10((psd[tgt].mean() + EPS) / (psd[nb].mean() + EPS)))
            hum[f"hum_{int(f0)}_db"] = float(max(ratios)) if ratios else 0.0

        rec.update(
            {
                "rolloff_99_hz": rolloff99,
                "rolloff_995_hz": rolloff995,
                "cutoff_50db_hz": cutoff,
                "band_0_100": band(0, 100),
                "band_100_300": band(100, 300),
                "band_300_3400": band(300, 3400),
                "band_3400_7000": band(3400, 7000),
                "band_7000_8000": band(7000, 8000),
                **hum,
            }
        )

    return rec


# ---------------------------------------------------------------- corpora


def jobs_esc50(cache_root: Path, vault: Path):
    idx = json.load(open(cache_root / "index" / "esc50_index.json"))
    return [
        (cache_root / r["cached_path"], {"corpus": "esc50", "id": r["filename"],
                                         "fold": r["fold"], "category": r["category"]})
        for r in idx
    ]


def jobs_librispeech(cache_root: Path, vault: Path):
    idx = json.load(open(cache_root / "index" / "librispeech_index.json"))
    out = []
    for r in idx:
        p = Path(r["cached_path"])
        speaker, chapter = p.parts[-3], p.parts[-2]
        out.append((cache_root / p, {"corpus": "librispeech", "id": p.stem,
                                     "split": r["split"], "speaker": speaker,
                                     "chapter": chapter}))
    return out


def jobs_ami(cache_root: Path, vault: Path):
    idx = json.load(open(cache_root / "index" / "ami_index.json"))
    return [
        (Path(r["audio_path"]), {"corpus": "ami", "id": r["meeting_id"],
                                 "series": r["meeting_id"][:-1]})
        for r in idx
    ]


def jobs_ten(cache_root: Path, vault: Path):
    d = vault / "ten-vad-testset"
    return [(p, {"corpus": "ten", "id": p.stem}) for p in sorted(d.glob("*.wav"))]


def jobs_fleurs(cache_root: Path, vault: Path):
    d = vault / "fleurs_sw_ke" / "audio"
    files = sorted(d.rglob("*.wav")) + sorted(d.rglob("*.flac"))
    return [(p, {"corpus": "fleurs", "id": p.stem, "split": p.parent.name}) for p in files]


BUILDERS = {
    "esc50": jobs_esc50,
    "librispeech": jobs_librispeech,
    "ami": jobs_ami,
    "ten": jobs_ten,
    "fleurs": jobs_fleurs,
}


def _work(args):
    path, meta = args
    try:
        return analyse(Path(path), meta)
    except Exception as e:  # never let one file kill the sweep
        return {"path": str(path), **meta, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    vault = Path(paths_cfg["vault_data_root"])
    cache_root = REPO_ROOT / paths_cfg["cache_root"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    which = sys.argv[1:] or list(BUILDERS)
    for corpus in which:
        jobs = BUILDERS[corpus](cache_root, vault)
        print(f"{corpus}: {len(jobs)} files")
        results = []
        workers = 2 if corpus == "ami" else 8
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rec in tqdm(ex.map(_work, jobs, chunksize=1 if corpus == "ami" else 8),
                            total=len(jobs), desc=corpus):
                results.append(rec)
        out = OUT_DIR / f"{corpus}.jsonl"
        with open(out, "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        n_err = sum(1 for r in results if "error" in r)
        print(f"  -> {out}  ({n_err} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
