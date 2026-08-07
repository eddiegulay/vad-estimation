"""Evaluation metrics: frame-level P/R/F1/AUROC, event-level onset boundary
matching, and real-time factor (RTF) — plan §6/§7. AUROC is implemented via
the rank-sum (Mann-Whitney U) method using `scipy.stats.rankdata` for
correct tie handling, avoiding a new scikit-learn dependency.
"""

import numpy as np
from scipy.stats import rankdata

from vad.labels.intervals import Interval


def frame_precision_recall_f1(preds: np.ndarray, labels: np.ndarray) -> dict:
    preds = preds.astype(bool)
    labels = labels.astype(bool)

    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    fn = int(np.sum(~preds & labels))
    tn = int(np.sum(~preds & ~labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )
    accuracy = (tp + tn) / len(preds) if len(preds) > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def frame_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    all_scores = np.concatenate([pos, neg])
    ranks = rankdata(all_scores)
    n_pos, n_neg = len(pos), len(neg)
    sum_ranks_pos = ranks[:n_pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def event_boundary_metrics(
    pred_intervals: list[Interval], true_intervals: list[Interval], tolerance_s: float = 0.2
) -> dict:
    """Match predicted vs. true speech-onset times within `tolerance_s`."""
    pred_onsets = [start for start, _, label in pred_intervals if label == 1]
    true_onsets = [start for start, _, label in true_intervals if label == 1]

    matched_true: set[int] = set()
    n_matched = 0
    for p in pred_onsets:
        for j, t in enumerate(true_onsets):
            if j in matched_true:
                continue
            if abs(p - t) <= tolerance_s:
                matched_true.add(j)
                n_matched += 1
                break

    precision = n_matched / len(pred_onsets) if pred_onsets else float("nan")
    recall = n_matched / len(true_onsets) if true_onsets else float("nan")

    return {
        "onset_precision": precision,
        "onset_recall": recall,
        "n_pred_onsets": len(pred_onsets),
        "n_true_onsets": len(true_onsets),
        "n_matched_onsets": n_matched,
        "tolerance_s": tolerance_s,
    }


def real_time_factor(wall_seconds: float, audio_seconds: float) -> float:
    return wall_seconds / audio_seconds if audio_seconds > 0 else float("nan")
