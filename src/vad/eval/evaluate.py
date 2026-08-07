"""Evaluation harness: run a model over an eval manifest, compute
frame-level + event-level + RTF metrics, save qualitative waveform/label/
prediction plots for manual spot-checking (plan §6/§7 — this is deliberately
runnable against a randomly-initialized model as a pre-training plumbing
smoke test, independent of model quality, using TEN as the first target
since it needs zero label construction on our part).
"""

import time
from pathlib import Path

import numpy as np
import torch

from vad.eval.metrics import event_boundary_metrics, frame_auroc, frame_precision_recall_f1, real_time_factor
from vad.labels.intervals import from_frames
from vad.postprocess.hysteresis import apply_hysteresis


def save_qualitative_plot(
    example_id: str,
    waveform: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    sample_rate: int,
    hop_s: float,
    out_dir: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_wave = np.arange(len(waveform)) / sample_rate
    t_frame = (np.arange(len(labels)) + 0.5) * hop_s

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t_wave, waveform, linewidth=0.5, color="steelblue")
    axes[0].set_ylabel("waveform")
    axes[0].set_title(example_id)

    axes[1].step(t_frame, labels, where="mid", label="ground truth", color="black", linewidth=1.5)
    axes[1].step(t_frame, probs, where="mid", label="predicted p(speech)", color="crimson", alpha=0.7)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("p(speech)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{example_id}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def run_evaluation(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    chunk_samples: int,
    hop_s: float,
    onset_tolerance_s: float = 0.2,
    num_plot_examples: int = 10,
    plot_dir: str | Path | None = None,
    theta_on: float = 0.5,
    theta_off: float = 0.35,
    min_speech_frames: int = 1,
    min_silence_frames: int = 1,
) -> dict:
    """`theta_on`/`theta_off`/`min_speech_frames`/`min_silence_frames`
    configure the external hangover/hysteresis post-processing (plan/note
    08 -- applied per-example, since it's a stateful sequence op and must
    never carry state across separate audio files). Both raw (flat 0.5) and
    smoothed metrics are reported so the post-processing lift is visible.
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_smoothed_preds = []
    onset_precisions = []
    onset_recalls = []
    smoothed_onset_precisions = []
    smoothed_onset_recalls = []
    total_audio_s = 0.0
    total_wall_s = 0.0
    n_examples_scored = 0

    for i in range(len(dataset)):
        item = dataset[i]
        waveform = item["waveform"].unsqueeze(0).to(device)
        labels = item["labels"].numpy()

        total_len = waveform.shape[1]
        usable_len = (total_len // chunk_samples) * chunk_samples
        if usable_len == 0:
            continue
        waveform_trim = waveform[:, :usable_len]

        t0 = time.time()
        with torch.no_grad():
            probs = model.forward_full(waveform_trim).cpu().numpy()[0]
        wall = time.time() - t0

        n = min(len(probs), len(labels))
        if n == 0:
            continue
        probs, labels_trim = probs[:n], labels[:n]

        all_probs.append(probs)
        all_labels.append(labels_trim)
        n_examples_scored += 1

        total_audio_s += usable_len / dataset.sample_rate
        total_wall_s += wall

        true_intervals = from_frames(labels_trim, hop_s)

        pred_intervals = from_frames((probs > 0.5).astype(np.int8), hop_s)
        onset_metrics = event_boundary_metrics(pred_intervals, true_intervals, onset_tolerance_s)
        if not np.isnan(onset_metrics["onset_precision"]):
            onset_precisions.append(onset_metrics["onset_precision"])
        if not np.isnan(onset_metrics["onset_recall"]):
            onset_recalls.append(onset_metrics["onset_recall"])

        # Hysteresis is a stateful sequence op -- run per-example only,
        # never across the file boundary of a concatenated batch.
        smoothed_preds = apply_hysteresis(probs, theta_on, theta_off, min_speech_frames, min_silence_frames)
        all_smoothed_preds.append(smoothed_preds)
        smoothed_pred_intervals = from_frames(smoothed_preds, hop_s)
        smoothed_onset_metrics = event_boundary_metrics(smoothed_pred_intervals, true_intervals, onset_tolerance_s)
        if not np.isnan(smoothed_onset_metrics["onset_precision"]):
            smoothed_onset_precisions.append(smoothed_onset_metrics["onset_precision"])
        if not np.isnan(smoothed_onset_metrics["onset_recall"]):
            smoothed_onset_recalls.append(smoothed_onset_metrics["onset_recall"])

        if plot_dir is not None and i < num_plot_examples:
            save_qualitative_plot(
                item["id"],
                waveform_trim.cpu().numpy()[0],
                labels_trim,
                probs,
                dataset.sample_rate,
                hop_s,
                plot_dir,
            )

    if n_examples_scored == 0:
        return {"num_examples": 0, "error": "no scorable examples in dataset"}

    probs_concat = np.concatenate(all_probs)
    labels_concat = np.concatenate(all_labels)
    preds_concat = (probs_concat > 0.5).astype(int)
    smoothed_preds_concat = np.concatenate(all_smoothed_preds)

    metrics = frame_precision_recall_f1(preds_concat, labels_concat)
    metrics["auroc"] = frame_auroc(probs_concat, labels_concat)
    metrics["rtf"] = real_time_factor(total_wall_s, total_audio_s)
    metrics["mean_onset_precision"] = float(np.mean(onset_precisions)) if onset_precisions else float("nan")
    metrics["mean_onset_recall"] = float(np.mean(onset_recalls)) if onset_recalls else float("nan")
    metrics["num_examples"] = n_examples_scored
    metrics["num_frames"] = int(len(probs_concat))
    metrics["total_audio_s"] = total_audio_s

    smoothed = frame_precision_recall_f1(smoothed_preds_concat, labels_concat)
    metrics["smoothed"] = {
        "precision": smoothed["precision"],
        "recall": smoothed["recall"],
        "f1": smoothed["f1"],
        "accuracy": smoothed["accuracy"],
        "mean_onset_precision": float(np.mean(smoothed_onset_precisions)) if smoothed_onset_precisions else float("nan"),
        "mean_onset_recall": float(np.mean(smoothed_onset_recalls)) if smoothed_onset_recalls else float("nan"),
        "postprocess_params": {
            "theta_on": theta_on, "theta_off": theta_off,
            "min_speech_frames": min_speech_frames, "min_silence_frames": min_silence_frames,
        },
    }
    return metrics
