import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vad.data.assets import (
    convert_rir_mat_to_npy,
    convert_to_16k_mono_int16,
    float_to_int16,
    load_air_rir,
    resample,
    to_mono,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAULT_ROOT = Path(
    "/Users/eddiegulay/Documents/Obsidiam Vault/The vault/Voice Activity Research/data"
)


def test_to_mono_passes_through_1d():
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.array_equal(to_mono(x), x)


def test_to_mono_averages_channels():
    x = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert np.allclose(to_mono(x), [2.0, 3.0])


def test_resample_noop_when_same_rate():
    x = np.arange(10, dtype=np.float64)
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_resample_changes_length_proportionally():
    x = np.zeros(48000, dtype=np.float64)
    y = resample(x, 48000, 16000)
    assert len(y) == 16000


def test_float_to_int16_clips_and_scales():
    x = np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    y = float_to_int16(x)
    assert y.dtype == np.int16
    assert y[0] == -32767  # clipped from -2.0
    assert y[2] == 0
    assert y[4] == 32767  # clipped from 2.0


@pytest.mark.skipif(not VAULT_ROOT.exists(), reason="data kit not reachable in this environment")
def test_convert_to_16k_mono_int16_preserves_duration(tmp_path):
    src = VAULT_ROOT / "ESC-50-master" / "audio" / "1-100032-A-0.wav"
    dst = tmp_path / "out.wav"
    duration = convert_to_16k_mono_int16(src, dst, target_sr=16000)

    src_info = sf.info(str(src))
    src_duration = src_info.frames / src_info.samplerate
    assert abs(duration - src_duration) < 0.05

    out_info = sf.info(str(dst))
    assert out_info.samplerate == 16000
    assert out_info.channels == 1
    assert out_info.subtype == "PCM_16"


@pytest.mark.skipif(not VAULT_ROOT.exists(), reason="data kit not reachable in this environment")
def test_load_air_rir_returns_1d_array_and_fs():
    mat_path = VAULT_ROOT / "air_database" / "AIR_1_4" / "air_binaural_booth_1_1_1.mat"
    ir, fs = load_air_rir(mat_path)
    assert ir.ndim == 1
    assert len(ir) > 0
    assert fs > 0


@pytest.mark.skipif(not VAULT_ROOT.exists(), reason="data kit not reachable in this environment")
def test_convert_rir_mat_to_npy_resamples(tmp_path):
    mat_path = VAULT_ROOT / "air_database" / "AIR_1_4" / "air_binaural_booth_1_1_1.mat"
    dst = tmp_path / "out.npy"
    duration = convert_rir_mat_to_npy(mat_path, dst, target_sr=16000)

    saved = np.load(str(dst))
    assert saved.dtype == np.float32
    assert abs(len(saved) / 16000 - duration) < 1e-9


# --- Integration check on the actual generated data_cache/ (Phase 2 gate) ---

CACHE_ROOT = REPO_ROOT / "data_cache"


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_esc50_index_matches_expected_count():
    index_path = CACHE_ROOT / "index" / "esc50_index.json"
    if not index_path.exists():
        pytest.skip("esc50_index.json not generated yet")
    records = json.loads(index_path.read_text())
    assert len(records) == 2000
    for rec in records[:5]:
        cached_path = CACHE_ROOT / rec["cached_path"]
        assert cached_path.exists()
        info = sf.info(str(cached_path))
        assert info.samplerate == 16000
        assert abs(info.frames / info.samplerate - rec["duration_s"]) < 1e-6


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_librispeech_index_matches_expected_count():
    index_path = CACHE_ROOT / "index" / "librispeech_index.json"
    if not index_path.exists():
        pytest.skip("librispeech_index.json not generated yet")
    records = json.loads(index_path.read_text())
    assert len(records) == 2703 + 2864 + 2620 + 2939
    split_counts = {}
    for rec in records:
        split_counts[rec["split"]] = split_counts.get(rec["split"], 0) + 1
    assert split_counts == {
        "dev-clean": 2703,
        "dev-other": 2864,
        "test-clean": 2620,
        "test-other": 2939,
    }


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_rir_index_matches_expected_count():
    index_path = CACHE_ROOT / "index" / "rir_index.json"
    if not index_path.exists():
        pytest.skip("rir_index.json not generated yet")
    records = json.loads(index_path.read_text())
    assert len(records) == 214
    for rec in records[:5]:
        cached_path = CACHE_ROOT / rec["cached_path"]
        assert cached_path.exists()


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_ami_index_points_at_vault_not_cache():
    index_path = CACHE_ROOT / "index" / "ami_index.json"
    if not index_path.exists():
        pytest.skip("ami_index.json not generated yet")
    records = json.loads(index_path.read_text())
    assert len(records) == 33
    for rec in records:
        # AMI is never duplicated into data_cache/ — audio_path must point at the vault
        assert str(CACHE_ROOT) not in rec["audio_path"]
        assert Path(rec["audio_path"]).exists()


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_fleurs_was_never_cached():
    # FLEURS is intentionally never cached (plan §3.7) — no audio/fleurs dir should exist.
    assert not (CACHE_ROOT / "audio" / "fleurs").exists()


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="data_cache/ not yet generated")
def test_cache_size_within_expected_footprint():
    total_bytes = sum(f.stat().st_size for f in CACHE_ROOT.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024**3)
    assert total_gb < 5.0, f"data_cache/ is {total_gb:.2f} GB, larger than the ~2-3GB plan estimate"
