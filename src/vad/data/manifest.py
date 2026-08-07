"""Manifest (training/eval example recipe) I/O and per-corpus builders
(plan §4). Manifests are JSONL, one recipe per line, `kind`-discriminated:
`concat_synthetic` (LibriSpeech/FLEURS), `ami_window`, `ten_eval`.

`sources[].path` is always an absolute filesystem path (into either
`data_cache/` for cached corpora or the vault directly for uncached ones —
callers don't need to branch on corpus to resolve it). Augmentation
`noise_file`/`rir_file` fields are cache-relative (always looked up against
`cache_root`, since noise/RIR pools are always cached corpora).

Augmentation policy has two shapes depending on split (plan §3.4):
- train: `{"noise_pool_folds": [...], "snr_db_range": [...], "rir_pool": [...],
  "rir_prob": ...}`, `manifest_seed: null` -> resolved fresh per epoch.
- val/test: `{"noise_file": ..., "snr_db": ..., "rir_file": ...}`,
  `manifest_seed: <int>` -> already resolved, deterministic.
"""

import json
from pathlib import Path

import numpy as np

from vad.labels import ami as ami_labels
from vad.labels import ten as ten_labels
from vad.labels.intervals import clip
from vad.labels.synthetic import build_concat_labels, sample_gaps, sample_num_utterances


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_index(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# --- augmentation policy templates / resolution (shared by every corpus) ---


def select_noise_pool(esc50_index: list[dict], folds: list[int]) -> list[dict]:
    return [r for r in esc50_index if r["fold"] in folds]


def select_rir_pool(rir_index: list[dict], pool_substrings: list[str]) -> list[dict]:
    if not pool_substrings:
        return list(rir_index)
    return [r for r in rir_index if any(s in r["filename"] for s in pool_substrings)]


def make_augmentation_template(
    noise_pool_folds: list[int], snr_db_range: list[float], rir_pool: list[str], rir_prob: float
) -> dict:
    return {
        "noise_pool_folds": noise_pool_folds,
        "snr_db_range": snr_db_range,
        "rir_pool": rir_pool,
        "rir_prob": rir_prob,
    }


def resolve_augmentation(
    rng: np.random.Generator,
    esc50_index: list[dict],
    rir_index: list[dict],
    template: dict,
) -> dict:
    """Sample one concrete augmentation realization from a template."""
    noise_candidates = select_noise_pool(esc50_index, template["noise_pool_folds"])
    noise_rec = noise_candidates[int(rng.integers(0, len(noise_candidates)))]
    snr_lo, snr_hi = template["snr_db_range"]
    snr_db = float(rng.uniform(snr_lo, snr_hi))

    rir_file = None
    if rng.random() < template["rir_prob"]:
        rir_candidates = select_rir_pool(rir_index, template["rir_pool"])
        if rir_candidates:
            rir_file = rir_candidates[int(rng.integers(0, len(rir_candidates)))]["cached_path"]

    return {"noise_file": noise_rec["cached_path"], "snr_db": snr_db, "rir_file": rir_file}


# --- concat_synthetic builder (shared by LibriSpeech and FLEURS) ---


def build_concat_synthetic_examples(
    pool: list[dict],  # [{"path": abs_path_str, "duration_s": float}, ...]
    concat_cfg: dict,
    aug_template: dict,
    rng: np.random.Generator,
    num_examples: int,
    split: str,
    id_prefix: str,
    corpus_name: str,
    esc50_index: list[dict] | None = None,
    rir_index: list[dict] | None = None,
    freeze_augmentation: bool = False,
) -> list[dict]:
    """Generate `num_examples` concat+gap-insertion recipes drawn from `pool`.

    `freeze_augmentation=True` (val/test) resolves one concrete augmentation
    realization per example at build time; otherwise (train) the template is
    stored as-is with `manifest_seed: null` for fresh per-epoch sampling.
    """
    records = []
    pool_size = len(pool)
    lo, hi = concat_cfg["utterances_per_example"]
    max_n = min(hi, pool_size)

    for i in range(num_examples):
        n = min(sample_num_utterances(rng, (lo, max_n)), pool_size)
        indices = rng.choice(pool_size, size=n, replace=False)
        sources = [pool[j] for j in indices]
        source_durations = [s["duration_s"] for s in sources]

        gap_cfg = concat_cfg["gap_s"]
        gaps = sample_gaps(
            rng, n - 1, gap_cfg["typical_range"], gap_cfg["long_pause_prob"], gap_cfg["long_pause_range"]
        )
        label_intervals = build_concat_labels(source_durations, gaps)

        if freeze_augmentation:
            augmentation_policy = resolve_augmentation(rng, esc50_index, rir_index, aug_template)
            manifest_seed = int(rng.integers(0, 2**31 - 1))
        else:
            augmentation_policy = dict(aug_template)
            manifest_seed = None

        records.append(
            {
                "id": f"{id_prefix}-{i:06d}",
                "kind": "concat_synthetic",
                "split": split,
                "sources": [
                    {"corpus": corpus_name, "path": s["path"], "duration_s": s["duration_s"]}
                    for s in sources
                ],
                "assembly": {"order": list(range(n)), "gaps_s": gaps},
                "label_intervals": label_intervals,
                "augmentation_policy": augmentation_policy,
                "manifest_seed": manifest_seed,
            }
        )
    return records


def build_librispeech_pool(librispeech_index: list[dict], cache_root: Path, splits: list[str]) -> list[dict]:
    return [
        {"path": str(cache_root / rec["cached_path"]), "duration_s": rec["duration_s"]}
        for rec in librispeech_index
        if rec["split"] in splits
    ]


def read_fleurs_tsv(tsv_path: str | Path) -> list[dict]:
    """Parse a FLEURS TSV: id, filename, raw_text, norm_text, char_tokens,
    num_samples, gender (tab-separated, no header, confirmed by inspection).
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "id": fields[0],
                    "filename": fields[1],
                    "num_samples": int(fields[5]),
                    "gender": fields[6],
                }
            )
    return rows


def build_fleurs_pool(vault_root: Path, split: str, sample_rate: int = 16000) -> list[dict]:
    """FLEURS is never cached (plan §3.7) — read directly from the vault."""
    tsv_path = vault_root / "fleurs_sw_ke" / f"{split}.tsv"
    audio_dir = vault_root / "fleurs_sw_ke" / "audio" / split
    rows = read_fleurs_tsv(tsv_path)
    return [
        {"path": str(audio_dir / row["filename"]), "duration_s": row["num_samples"] / sample_rate}
        for row in rows
    ]


# --- AMI builder ---


def build_ami_examples(
    segments_dir: Path,
    ami_audio_index: list[dict],
    ami_cfg: dict,
    aug_template: dict,
    rng: np.random.Generator,
    esc50_index: list[dict],
    rir_index: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Returns (train_records, val_records), split at the meeting level."""
    meetings = ami_labels.discover_meetings(segments_dir)
    val_meetings = set(ami_cfg["spontaneous_meetings_held_out_for_val"])
    remaining = [m for m in meetings if m not in val_meetings]

    extra_val_count = max(0, ami_cfg.get("val_meeting_count", 4) - len(val_meetings))
    if extra_val_count > 0 and remaining:
        chosen = rng.choice(len(remaining), size=min(extra_val_count, len(remaining)), replace=False)
        val_meetings.update(remaining[i] for i in chosen)

    train_meetings = sorted(m for m in meetings if m not in val_meetings)
    audio_by_id = {r["meeting_id"]: r for r in ami_audio_index}

    def build_for(meeting_ids: list[str], split_name: str, overlap: float, id_prefix: str) -> list[dict]:
        records = []
        for meeting_id in meeting_ids:
            audio_rec = audio_by_id.get(meeting_id)
            if audio_rec is None:
                continue
            duration = audio_rec["duration_s"]
            covering = ami_labels.build_meeting_speech_intervals(
                segments_dir, meeting_id, duration, gap_tolerance_s=ami_cfg["channel_merge_tolerance_s"]
            )
            windows = ami_labels.chunk_windows(duration, ami_cfg["window_s"], overlap)
            for i, (offset, win_dur) in enumerate(windows):
                clipped = clip(covering, offset, win_dur)

                if split_name == "train":
                    augmentation_policy = dict(aug_template)
                    manifest_seed = None
                else:
                    augmentation_policy = resolve_augmentation(rng, esc50_index, rir_index, aug_template)
                    manifest_seed = int(rng.integers(0, 2**31 - 1))

                records.append(
                    {
                        "id": f"{id_prefix}-{meeting_id}-{i:06d}",
                        "kind": "ami_window",
                        "split": split_name,
                        "sources": [
                            {
                                "corpus": "ami",
                                "path": audio_rec["audio_path"],
                                "offset_s": offset,
                                "duration_s": win_dur,
                            }
                        ],
                        "label_intervals": clipped,
                        "augmentation_policy": augmentation_policy,
                        "manifest_seed": manifest_seed,
                        "meeting_id": meeting_id,
                        "meeting_style": (
                            "spontaneous"
                            if meeting_id in ami_cfg["spontaneous_meetings_held_out_for_val"]
                            else "scripted"
                        ),
                    }
                )
        return records

    train_records = build_for(train_meetings, "train", ami_cfg["train_overlap"], "ami-train")
    val_records = build_for(sorted(val_meetings), "val", 0.0, "ami-val")
    return train_records, val_records


# --- TEN eval builder ---


def build_ten_examples(ten_dir: Path) -> list[dict]:
    records = []
    for scv_path in sorted(Path(ten_dir).glob("*.scv")):
        filename, intervals = ten_labels.parse_scv(scv_path)
        wav_path = Path(ten_dir) / f"{filename}.wav"
        duration = intervals[-1][1] if intervals else 0.0
        records.append(
            {
                "id": f"ten-{filename}",
                "kind": "ten_eval",
                "split": "test_ten",
                "sources": [
                    {"corpus": "ten", "path": str(wav_path), "offset_s": 0.0, "duration_s": duration}
                ],
                "label_intervals": intervals,
                "augmentation_policy": None,
                "manifest_seed": None,
            }
        )
    return records
