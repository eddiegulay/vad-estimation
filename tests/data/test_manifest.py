import numpy as np
import pytest

from vad.data.manifest import (
    build_concat_synthetic_examples,
    build_ten_examples,
    make_augmentation_template,
    read_fleurs_tsv,
    read_jsonl,
    resolve_augmentation,
    select_noise_pool,
    select_rir_pool,
    write_jsonl,
)


def test_write_read_jsonl_roundtrip(tmp_path):
    records = [{"id": "a", "x": 1}, {"id": "b", "x": [1, 2, 3]}]
    path = tmp_path / "test.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_select_noise_pool_filters_by_fold():
    index = [{"filename": "a", "fold": 1}, {"filename": "b", "fold": 5}, {"filename": "c", "fold": 3}]
    result = select_noise_pool(index, [1, 3])
    assert {r["filename"] for r in result} == {"a", "c"}


def test_select_rir_pool_filters_by_substring():
    index = [
        {"filename": "air_binaural_office_1_1_1.mat", "cached_path": "rir/o.npy"},
        {"filename": "air_binaural_stairway_1_1_1.mat", "cached_path": "rir/s.npy"},
    ]
    result = select_rir_pool(index, ["office"])
    assert len(result) == 1
    assert result[0]["cached_path"] == "rir/o.npy"


def test_select_rir_pool_empty_substrings_returns_all():
    index = [{"filename": "a.mat"}, {"filename": "b.mat"}]
    assert select_rir_pool(index, []) == index


def test_resolve_augmentation_snr_within_range_and_valid_files():
    rng = np.random.default_rng(0)
    esc50_index = [
        {"filename": "n1.wav", "fold": 1, "cached_path": "audio/esc50/n1.wav"},
        {"filename": "n2.wav", "fold": 2, "cached_path": "audio/esc50/n2.wav"},
    ]
    rir_index = [{"filename": "air_x.mat", "cached_path": "rir/x.npy"}]
    template = make_augmentation_template([1, 2], [-5, 20], [], rir_prob=1.0)

    for _ in range(50):
        resolved = resolve_augmentation(rng, esc50_index, rir_index, template)
        assert -5 <= resolved["snr_db"] <= 20
        assert resolved["noise_file"] in {"audio/esc50/n1.wav", "audio/esc50/n2.wav"}
        assert resolved["rir_file"] == "rir/x.npy"  # rir_prob=1.0 -> always picked


def test_resolve_augmentation_rir_prob_zero_never_picks_rir():
    rng = np.random.default_rng(0)
    esc50_index = [{"filename": "n1.wav", "fold": 1, "cached_path": "audio/esc50/n1.wav"}]
    rir_index = [{"filename": "air_x.mat", "cached_path": "rir/x.npy"}]
    template = make_augmentation_template([1], [0, 0], [], rir_prob=0.0)
    for _ in range(20):
        resolved = resolve_augmentation(rng, esc50_index, rir_index, template)
        assert resolved["rir_file"] is None


def test_build_concat_synthetic_examples_train_shape():
    rng = np.random.default_rng(0)
    pool = [{"path": f"/fake/{i}.wav", "duration_s": 2.0 + i * 0.1} for i in range(20)]
    concat_cfg = {
        "utterances_per_example": [2, 4],
        "gap_s": {"typical_range": [0.2, 1.0], "long_pause_prob": 0.1, "long_pause_range": [1.0, 2.0]},
    }
    template = make_augmentation_template([1, 2, 3, 4], [-5, 20], [], 0.5)

    records = build_concat_synthetic_examples(
        pool, concat_cfg, template, rng, num_examples=10, split="train",
        id_prefix="test", corpus_name="fake",
    )

    assert len(records) == 10
    for rec in records:
        assert rec["kind"] == "concat_synthetic"
        assert rec["split"] == "train"
        assert rec["manifest_seed"] is None
        assert "snr_db_range" in rec["augmentation_policy"]
        n_sources = len(rec["sources"])
        assert 2 <= n_sources <= 4
        assert len(rec["assembly"]["gaps_s"]) == n_sources - 1
        # label_intervals total duration matches sources + gaps
        total_from_intervals = rec["label_intervals"][-1][1]
        expected_total = sum(s["duration_s"] for s in rec["sources"]) + sum(rec["assembly"]["gaps_s"])
        assert total_from_intervals == pytest.approx(expected_total)


def test_build_concat_synthetic_examples_frozen_augmentation_for_val():
    rng = np.random.default_rng(0)
    pool = [{"path": f"/fake/{i}.wav", "duration_s": 3.0} for i in range(10)]
    concat_cfg = {
        "utterances_per_example": [2, 3],
        "gap_s": {"typical_range": [0.2, 1.0], "long_pause_prob": 0.0, "long_pause_range": [1.0, 2.0]},
    }
    template = make_augmentation_template([1, 2, 3, 4], [-5, 20], [], 1.0)
    esc50_index = [{"filename": "n.wav", "fold": 1, "cached_path": "audio/esc50/n.wav"}]
    rir_index = [{"filename": "air_x.mat", "cached_path": "rir/x.npy"}]

    records = build_concat_synthetic_examples(
        pool, concat_cfg, template, rng, num_examples=5, split="val",
        id_prefix="test-val", corpus_name="fake",
        esc50_index=esc50_index, rir_index=rir_index, freeze_augmentation=True,
    )

    for rec in records:
        assert rec["manifest_seed"] is not None
        assert "snr_db" in rec["augmentation_policy"]
        assert "noise_file" in rec["augmentation_policy"]


def test_read_fleurs_tsv_parses_expected_columns(tmp_path):
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text(
        "249\t11410111868389236067.wav\traw text\tnorm text\tc h a r|\t205440\tFEMALE\n"
    )
    rows = read_fleurs_tsv(tsv_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "249"
    assert rows[0]["filename"] == "11410111868389236067.wav"
    assert rows[0]["num_samples"] == 205440
    assert rows[0]["gender"] == "FEMALE"


def test_build_ten_examples_parses_scv_and_pairs_wav(tmp_path):
    (tmp_path / "testset-audio-01.scv").write_text(
        "testset-audio-01,0.000,0.403,0,0.403,1.204,1"
    )
    (tmp_path / "testset-audio-01.wav").touch()

    records = build_ten_examples(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "ten_eval"
    assert rec["split"] == "test_ten"
    assert rec["sources"][0]["duration_s"] == pytest.approx(1.204)
    assert rec["augmentation_policy"] is None
