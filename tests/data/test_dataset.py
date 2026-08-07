import numpy as np
import soundfile as sf

from vad.data.dataset import VADDataset
from vad.data.manifest import write_jsonl

SR = 16000
HOP_S = 0.032  # matches crnn_v1's 512-sample chunk @16kHz


def _write_tone(path, duration_s, sr=SR, freq=220.0):
    t = np.arange(int(duration_s * sr)) / sr
    wave = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sr, subtype="PCM_16")


def _make_cache(tmp_path):
    cache_root = tmp_path / "cache"
    (cache_root / "audio" / "librispeech").mkdir(parents=True)
    (cache_root / "audio" / "esc50").mkdir(parents=True)
    (cache_root / "rir").mkdir(parents=True)

    utt1 = cache_root / "audio" / "librispeech" / "utt1.wav"
    utt2 = cache_root / "audio" / "librispeech" / "utt2.wav"
    _write_tone(utt1, 2.0, freq=220.0)
    _write_tone(utt2, 1.5, freq=330.0)

    noise = cache_root / "audio" / "esc50" / "n1.wav"
    _write_tone(noise, 5.0, freq=1000.0)

    rir = cache_root / "rir" / "r1.npy"
    np.save(str(rir), np.array([1.0, 0.1, 0.02], dtype=np.float32))

    return cache_root, utt1, utt2, noise


def _train_record(utt1, utt2):
    return {
        "id": "train-000",
        "kind": "concat_synthetic",
        "split": "train",
        "sources": [
            {"corpus": "librispeech", "path": str(utt1), "duration_s": 2.0},
            {"corpus": "librispeech", "path": str(utt2), "duration_s": 1.5},
        ],
        "assembly": {"order": [0, 1], "gaps_s": [0.5]},
        "label_intervals": [[0.0, 2.0, 1], [2.0, 2.5, 0], [2.5, 4.0, 1]],
        "augmentation_policy": {
            "noise_pool_folds": [1],
            "snr_db_range": [0, 10],
            "rir_pool": [],
            "rir_prob": 1.0,
        },
        "manifest_seed": None,
    }


def _val_record(utt1, utt2):
    return {
        "id": "val-000",
        "kind": "concat_synthetic",
        "split": "val",
        "sources": [
            {"corpus": "librispeech", "path": str(utt1), "duration_s": 2.0},
            {"corpus": "librispeech", "path": str(utt2), "duration_s": 1.5},
        ],
        "assembly": {"order": [0, 1], "gaps_s": [0.5]},
        "label_intervals": [[0.0, 2.0, 1], [2.0, 2.5, 0], [2.5, 4.0, 1]],
        "augmentation_policy": {
            "noise_file": "audio/esc50/n1.wav",
            "snr_db": 5.0,
            "rir_file": "rir/r1.npy",
        },
        "manifest_seed": 42,
    }


def _esc50_index():
    return [{"filename": "n1.wav", "fold": 1, "cached_path": "audio/esc50/n1.wav"}]


def _rir_index():
    return [{"filename": "r1.npy", "cached_path": "rir/r1.npy"}]


def test_getitem_shapes_are_consistent(tmp_path):
    cache_root, utt1, utt2, _ = _make_cache(tmp_path)
    manifest_path = tmp_path / "train.jsonl"
    write_jsonl(manifest_path, [_train_record(utt1, utt2)])

    ds = VADDataset(
        manifest_path, cache_root, sample_rate=SR, hop_s=HOP_S,
        esc50_index=_esc50_index(), rir_index=_rir_index(),
    )
    item = ds[0]
    expected_wave_len = int(round(4.0 * SR))
    expected_num_frames = int(round(4.0 / HOP_S))

    assert abs(item["waveform"].shape[0] - expected_wave_len) <= 1
    assert item["labels"].shape[0] == expected_num_frames
    assert item["waveform"].dtype.is_floating_point
    assert item["id"] == "train-000"


def test_train_item_augmentation_varies_across_epochs(tmp_path):
    cache_root, utt1, utt2, _ = _make_cache(tmp_path)
    manifest_path = tmp_path / "train.jsonl"
    write_jsonl(manifest_path, [_train_record(utt1, utt2)])

    ds = VADDataset(
        manifest_path, cache_root, sample_rate=SR, hop_s=HOP_S,
        esc50_index=_esc50_index(), rir_index=_rir_index(), run_seed=7,
    )
    ds.set_epoch(0)
    wave_epoch0 = ds[0]["waveform"].numpy().copy()
    ds.set_epoch(1)
    wave_epoch1 = ds[0]["waveform"].numpy().copy()

    assert not np.allclose(wave_epoch0, wave_epoch1)


def test_train_item_reproducible_within_same_epoch(tmp_path):
    cache_root, utt1, utt2, _ = _make_cache(tmp_path)
    manifest_path = tmp_path / "train.jsonl"
    write_jsonl(manifest_path, [_train_record(utt1, utt2)])

    ds = VADDataset(
        manifest_path, cache_root, sample_rate=SR, hop_s=HOP_S,
        esc50_index=_esc50_index(), rir_index=_rir_index(), run_seed=7,
    )
    ds.set_epoch(3)
    wave_a = ds[0]["waveform"].numpy().copy()
    wave_b = ds[0]["waveform"].numpy().copy()
    assert np.array_equal(wave_a, wave_b)


def test_val_item_deterministic_regardless_of_epoch(tmp_path):
    cache_root, utt1, utt2, _ = _make_cache(tmp_path)
    manifest_path = tmp_path / "val.jsonl"
    write_jsonl(manifest_path, [_val_record(utt1, utt2)])

    ds = VADDataset(manifest_path, cache_root, sample_rate=SR, hop_s=HOP_S)
    ds.set_epoch(0)
    wave_epoch0 = ds[0]["waveform"].numpy().copy()
    ds.set_epoch(5)
    wave_epoch5 = ds[0]["waveform"].numpy().copy()

    # val items use manifest_seed, not epoch -- must be identical regardless of epoch
    assert np.array_equal(wave_epoch0, wave_epoch5)


def test_getitem_output_has_no_nan_or_inf(tmp_path):
    cache_root, utt1, utt2, _ = _make_cache(tmp_path)
    manifest_path = tmp_path / "train.jsonl"
    write_jsonl(manifest_path, [_train_record(utt1, utt2)])

    ds = VADDataset(
        manifest_path, cache_root, sample_rate=SR, hop_s=HOP_S,
        esc50_index=_esc50_index(), rir_index=_rir_index(),
    )
    item = ds[0]
    assert np.all(np.isfinite(item["waveform"].numpy()))
