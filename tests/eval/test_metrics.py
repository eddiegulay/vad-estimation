import numpy as np

from vad.eval.metrics import (
    event_boundary_metrics,
    frame_auroc,
    frame_precision_recall_f1,
    real_time_factor,
    sweep_thresholds,
)


def test_frame_precision_recall_f1_perfect_predictions():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    metrics = frame_precision_recall_f1(preds, labels)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_frame_precision_recall_f1_all_wrong():
    preds = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, 0, 0])
    metrics = frame_precision_recall_f1(preds, labels)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["accuracy"] == 0.0


def test_frame_precision_recall_f1_partial():
    preds = np.array([1, 1, 1, 0])
    labels = np.array([1, 0, 1, 0])
    metrics = frame_precision_recall_f1(preds, labels)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tn"] == 1
    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    assert metrics["recall"] == 1.0


def test_frame_auroc_perfect_separation_is_one():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert frame_auroc(probs, labels) == 1.0


def test_frame_auroc_inverted_is_zero():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1, 1, 0, 0])
    assert frame_auroc(probs, labels) == 0.0


def test_frame_auroc_random_is_near_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=10000)
    probs = rng.random(10000)  # uncorrelated with labels
    auc = frame_auroc(probs, labels)
    assert 0.45 < auc < 0.55


def test_frame_auroc_degenerate_single_class_is_nan():
    probs = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    assert np.isnan(frame_auroc(probs, labels))


def test_event_boundary_metrics_exact_match():
    pred = [(0.0, 1.0, 0), (1.0, 3.0, 1), (3.0, 4.0, 0)]
    true = [(0.0, 1.0, 0), (1.0, 3.0, 1), (3.0, 4.0, 0)]
    metrics = event_boundary_metrics(pred, true, tolerance_s=0.2)
    assert metrics["onset_precision"] == 1.0
    assert metrics["onset_recall"] == 1.0
    assert metrics["n_matched_onsets"] == 1


def test_event_boundary_metrics_within_tolerance():
    pred = [(0.0, 1.15, 0), (1.15, 3.0, 1)]  # onset at 1.15 vs true onset at 1.0
    true = [(0.0, 1.0, 0), (1.0, 3.0, 1)]
    metrics = event_boundary_metrics(pred, true, tolerance_s=0.2)
    assert metrics["onset_precision"] == 1.0
    assert metrics["onset_recall"] == 1.0


def test_event_boundary_metrics_outside_tolerance_no_match():
    pred = [(0.0, 2.0, 0), (2.0, 3.0, 1)]  # onset at 2.0, way off from true 1.0
    true = [(0.0, 1.0, 0), (1.0, 3.0, 1)]
    metrics = event_boundary_metrics(pred, true, tolerance_s=0.2)
    assert metrics["onset_precision"] == 0.0
    assert metrics["onset_recall"] == 0.0


def test_event_boundary_metrics_no_predicted_onsets_is_nan_precision():
    pred = [(0.0, 4.0, 0)]  # no speech at all
    true = [(0.0, 1.0, 0), (1.0, 3.0, 1), (3.0, 4.0, 0)]
    metrics = event_boundary_metrics(pred, true, tolerance_s=0.2)
    assert np.isnan(metrics["onset_precision"])
    assert metrics["onset_recall"] == 0.0


def test_real_time_factor_basic():
    assert real_time_factor(wall_seconds=1.0, audio_seconds=10.0) == 0.1


def test_real_time_factor_zero_audio_is_nan():
    assert np.isnan(real_time_factor(wall_seconds=1.0, audio_seconds=0.0))


def test_sweep_thresholds_finds_the_separating_threshold():
    probs = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    result = sweep_thresholds(probs, labels)
    assert result["best_f1"] == 1.0
    assert 0.4 <= result["best_threshold"] <= 0.6


def test_sweep_thresholds_returns_sweep_grid_results():
    probs = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    result = sweep_thresholds(probs, labels)
    assert len(result["sweep"]) > 1
    assert all("threshold" in r and "f1" in r for r in result["sweep"])
