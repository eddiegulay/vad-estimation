import torch

from vad.data.collate import collate_batch


def test_collate_pads_to_max_length_in_batch():
    batch = [
        {"id": "a", "waveform": torch.ones(100), "labels": torch.ones(3, dtype=torch.int64)},
        {"id": "b", "waveform": torch.ones(50), "labels": torch.ones(2, dtype=torch.int64)},
    ]
    out = collate_batch(batch)

    assert out["waveform"].shape == (2, 100)
    assert out["labels"].shape == (2, 3)
    assert out["label_mask"].shape == (2, 3)


def test_collate_mask_marks_real_vs_padded():
    batch = [
        {"id": "a", "waveform": torch.ones(10), "labels": torch.ones(4, dtype=torch.int64)},
        {"id": "b", "waveform": torch.ones(10), "labels": torch.ones(2, dtype=torch.int64)},
    ]
    out = collate_batch(batch)

    assert out["label_mask"][0].tolist() == [True, True, True, True]
    assert out["label_mask"][1].tolist() == [True, True, False, False]
    # padded label region is zero-filled (mask, not value, is authoritative)
    assert out["labels"][1, 2:].sum().item() == 0


def test_collate_preserves_ids_in_order():
    batch = [
        {"id": "x", "waveform": torch.zeros(5), "labels": torch.zeros(1, dtype=torch.int64)},
        {"id": "y", "waveform": torch.zeros(5), "labels": torch.zeros(1, dtype=torch.int64)},
    ]
    out = collate_batch(batch)
    assert out["id"] == ["x", "y"]
