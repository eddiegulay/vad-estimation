"""Batch collation: pad variable-length waveform/label sequences and
produce a length mask so padded regions don't contribute to loss.
"""

import torch


def collate_batch(batch: list[dict]) -> dict:
    ids = [item["id"] for item in batch]
    waveforms = [item["waveform"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_wave_len = max(w.shape[0] for w in waveforms)
    max_label_len = max(label.shape[0] for label in labels)

    waveform_batch = torch.zeros(len(batch), max_wave_len, dtype=torch.float32)
    label_batch = torch.zeros(len(batch), max_label_len, dtype=torch.int64)
    label_mask = torch.zeros(len(batch), max_label_len, dtype=torch.bool)

    for i, (waveform, label) in enumerate(zip(waveforms, labels)):
        waveform_batch[i, : waveform.shape[0]] = waveform
        label_batch[i, : label.shape[0]] = label
        label_mask[i, : label.shape[0]] = True

    return {
        "id": ids,
        "waveform": waveform_batch,
        "labels": label_batch,
        "label_mask": label_mask,
    }
