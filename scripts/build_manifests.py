#!/usr/bin/env python3
"""Phase 4 — build train/val/test_ten/sanity_fleurs manifests (plan §4/§6).

Requires Phase 2 (scripts/preprocess_assets.py) to have already populated
data_cache/index/{esc50,librispeech,rir,ami}_index.json.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from vad.config import load_config  # noqa: E402
from vad.data.manifest import (  # noqa: E402
    build_ami_examples,
    build_concat_synthetic_examples,
    build_fleurs_pool,
    build_librispeech_pool,
    build_ten_examples,
    load_index,
    make_augmentation_template,
    write_jsonl,
)

# Placeholder example counts for this first pipeline pass -- tune before scaling
# up real training (Phase 6). AMI window count is fixed by the corpus + window
# config, not by these constants.
NUM_LIBRISPEECH_TRAIN_EXAMPLES = 3000
NUM_LIBRISPEECH_VAL_EXAMPLES = 300
NUM_FLEURS_SANITY_EXAMPLES = 200


def main() -> int:
    cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml",
        REPO_ROOT / "configs" / "data" / "default.yaml",
    )
    vault_root = Path(cfg["vault_data_root"])
    cache_root = REPO_ROOT / cfg["cache_root"]
    manifests_dir = REPO_ROOT / cfg["manifests_dir"]
    index_dir = cache_root / "index"

    rng = np.random.default_rng(cfg["manifest_seed"])

    esc50_index = load_index(index_dir / "esc50_index.json")
    rir_index = load_index(index_dir / "rir_index.json")
    librispeech_index = load_index(index_dir / "librispeech_index.json")
    ami_audio_index = load_index(index_dir / "ami_index.json")

    aug_cfg = cfg["augmentation"]
    # rir_pool intentionally empty (-> whole RIR index); per-corpus room pools
    # are a tuning knob for later, not needed for pipeline correctness now.
    aug_template = make_augmentation_template(
        aug_cfg["esc50_train_folds"], aug_cfg["snr_db_range"], [], aug_cfg["rir_prob"]
    )
    concat_cfg = cfg["synthetic_concat"]

    # LibriSpeech: train pool = dev-clean/dev-other, val pool = test-clean/test-other,
    # kept disjoint so val isn't drawn from the same source utterances as train.
    train_pool = build_librispeech_pool(
        librispeech_index, cache_root, cfg["splits"]["librispeech"]["train"]
    )
    val_pool = build_librispeech_pool(
        librispeech_index, cache_root, cfg["splits"]["librispeech"]["test"]
    )

    librispeech_train = build_concat_synthetic_examples(
        train_pool,
        concat_cfg,
        aug_template,
        rng,
        NUM_LIBRISPEECH_TRAIN_EXAMPLES,
        split="train",
        id_prefix="ls-train",
        corpus_name="librispeech",
    )
    librispeech_val = build_concat_synthetic_examples(
        val_pool,
        concat_cfg,
        aug_template,
        rng,
        NUM_LIBRISPEECH_VAL_EXAMPLES,
        split="val",
        id_prefix="ls-val",
        corpus_name="librispeech",
        esc50_index=esc50_index,
        rir_index=rir_index,
        freeze_augmentation=True,
    )

    segments_dir = vault_root / cfg["corpora"]["ami_segments"]
    ami_train, ami_val = build_ami_examples(
        segments_dir, ami_audio_index, cfg["ami"], aug_template, rng, esc50_index, rir_index
    )

    ten_dir = vault_root / cfg["corpora"]["ten_testset"]
    ten_records = build_ten_examples(ten_dir)

    fleurs_pool = build_fleurs_pool(vault_root, "test")
    fleurs_sanity = build_concat_synthetic_examples(
        fleurs_pool,
        concat_cfg,
        aug_template,
        rng,
        NUM_FLEURS_SANITY_EXAMPLES,
        split="sanity_fleurs",
        id_prefix="fleurs-sanity",
        corpus_name="fleurs",
        esc50_index=esc50_index,
        rir_index=rir_index,
        freeze_augmentation=True,
    )

    train_records = librispeech_train + ami_train
    val_records = librispeech_val + ami_val

    write_jsonl(manifests_dir / "train.jsonl", train_records)
    write_jsonl(manifests_dir / "val.jsonl", val_records)
    write_jsonl(manifests_dir / "test_ten.jsonl", ten_records)
    write_jsonl(manifests_dir / "sanity_fleurs.jsonl", fleurs_sanity)

    print(
        f"train.jsonl: {len(train_records)} examples "
        f"({len(librispeech_train)} librispeech + {len(ami_train)} ami)"
    )
    print(
        f"val.jsonl: {len(val_records)} examples "
        f"({len(librispeech_val)} librispeech + {len(ami_val)} ami)"
    )
    print(f"test_ten.jsonl: {len(ten_records)} examples")
    print(f"sanity_fleurs.jsonl: {len(fleurs_sanity)} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
