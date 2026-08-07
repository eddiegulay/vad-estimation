#!/usr/bin/env python3
"""Phase 2 — one-time asset preprocessing (plan §3.7 / §6).

Converts ESC-50 (44.1kHz) and LibriSpeech (FLAC) to 16kHz mono int16 PCM
WAV under data_cache/audio/, and Aachen AIR RIRs (.mat) to 16kHz float32
.npy under data_cache/rir/. AMI (already 16kHz mono) and FLEURS (rarely
touched, eval-only) are indexed in place, never duplicated.

Idempotent: re-running skips files whose cached output already exists, so
an interrupted run can be resumed cheaply.
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vad.config import load_yaml  # noqa: E402
from vad.data.assets import (  # noqa: E402
    convert_rir_mat_to_npy,
    convert_to_16k_mono_int16,
    read_esc50_meta,
)

TARGET_SR = 16000
DURATION_TOLERANCE_S = 0.05


def source_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def convert_esc50(vault_root: Path, cache_root: Path, index_dir: Path) -> list[dict]:
    audio_dir = vault_root / "ESC-50-master" / "audio"
    meta_path = vault_root / "ESC-50-master" / "meta" / "esc50.csv"
    meta_rows = {row["filename"]: row for row in read_esc50_meta(meta_path)}

    out_dir = cache_root / "audio" / "esc50"
    records = []
    wav_files = sorted(audio_dir.glob("*.wav"))
    for src in tqdm(wav_files, desc="ESC-50"):
        dst = out_dir / src.name
        if dst.exists():
            duration = source_duration_s(dst)
        else:
            duration = convert_to_16k_mono_int16(src, dst, TARGET_SR)
        src_duration = source_duration_s(src)
        assert abs(duration - src_duration) < DURATION_TOLERANCE_S, (
            f"{src.name}: cached duration {duration:.3f}s vs source {src_duration:.3f}s"
        )
        meta = meta_rows.get(src.name, {})
        records.append(
            {
                "filename": src.name,
                "cached_path": str(dst.relative_to(cache_root)),
                "duration_s": duration,
                "fold": int(meta.get("fold", -1)),
                "category": meta.get("category", ""),
                "esc10": meta.get("esc10", "False") == "True",
            }
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "esc50_index.json", "w") as f:
        json.dump(records, f, indent=2)
    return records


def convert_librispeech(vault_root: Path, cache_root: Path, index_dir: Path) -> list[dict]:
    splits = ["dev-clean", "dev-other", "test-clean", "test-other"]
    ls_root = vault_root / "LibriSpeech"
    out_root = cache_root / "audio" / "librispeech"

    records = []
    for split in splits:
        split_dir = ls_root / split
        flac_files = sorted(split_dir.rglob("*.flac"))
        for src in tqdm(flac_files, desc=f"LibriSpeech {split}"):
            rel = src.relative_to(ls_root)
            dst = (out_root / rel).with_suffix(".wav")
            if dst.exists():
                duration = source_duration_s(dst)
            else:
                duration = convert_to_16k_mono_int16(src, dst, TARGET_SR)
            src_duration = source_duration_s(src)
            assert abs(duration - src_duration) < DURATION_TOLERANCE_S, (
                f"{rel}: cached duration {duration:.3f}s vs source {src_duration:.3f}s"
            )
            records.append(
                {
                    "split": split,
                    "cached_path": str(dst.relative_to(cache_root)),
                    "duration_s": duration,
                }
            )

    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "librispeech_index.json", "w") as f:
        json.dump(records, f, indent=2)
    return records


def convert_rirs(vault_root: Path, cache_root: Path, index_dir: Path) -> list[dict]:
    rir_dir = vault_root / "air_database" / "AIR_1_4"
    out_dir = cache_root / "rir"

    records = []
    mat_files = sorted(rir_dir.glob("*.mat"))
    for src in tqdm(mat_files, desc="Aachen AIR RIRs"):
        dst = (out_dir / src.stem).with_suffix(".npy")
        if dst.exists():
            duration = len(np.load(str(dst))) / TARGET_SR
        else:
            duration = convert_rir_mat_to_npy(src, dst, TARGET_SR)
        records.append(
            {
                "filename": src.name,
                "cached_path": str(dst.relative_to(cache_root)),
                "duration_s": duration,
            }
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "rir_index.json", "w") as f:
        json.dump(records, f, indent=2)
    return records


def index_ami(vault_root: Path, index_dir: Path) -> list[dict]:
    """AMI is already 16kHz mono — index only, never duplicated."""
    audio_dir = vault_root / "ami" / "audio"
    records = []
    for src in sorted(audio_dir.glob("*.Mix-Headset.wav")):
        meeting_id = src.name.split(".")[0]
        records.append(
            {
                "meeting_id": meeting_id,
                "audio_path": str(src),
                "duration_s": source_duration_s(src),
            }
        )
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "ami_index.json", "w") as f:
        json.dump(records, f, indent=2)
    return records


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    vault_root = Path(paths_cfg["vault_data_root"])
    cache_root = REPO_ROOT / paths_cfg["cache_root"]
    index_dir = cache_root / "index"

    t0 = time.time()
    esc50_records = convert_esc50(vault_root, cache_root, index_dir)
    librispeech_records = convert_librispeech(vault_root, cache_root, index_dir)
    rir_records = convert_rirs(vault_root, cache_root, index_dir)
    ami_records = index_ami(vault_root, index_dir)
    elapsed = time.time() - t0

    print(f"\nESC-50: {len(esc50_records)} files cached")
    print(f"LibriSpeech: {len(librispeech_records)} files cached")
    print(f"RIRs: {len(rir_records)} files cached")
    print(f"AMI: {len(ami_records)} meetings indexed (not cached, read in place)")

    cache_size_gb = dir_size_bytes(cache_root) / (1024**3)
    print(f"\ndata_cache/ total size: {cache_size_gb:.2f} GB")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
