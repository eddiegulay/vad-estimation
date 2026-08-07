import torch

from vad.engine.trainer import build_lr_scheduler, compute_class_weights, masked_bce_loss


def test_compute_class_weights_balanced_for_50_50():
    records = [{"label_intervals": [[0.0, 1.0, 1], [1.0, 2.0, 0]]}]
    pos_w, neg_w = compute_class_weights(records)
    assert pos_w == 1.0
    assert neg_w == 1.0


def test_compute_class_weights_upweights_minority_class():
    # 90% speech, 10% non-speech -> non-speech should get upweighted more
    records = [{"label_intervals": [[0.0, 9.0, 1], [9.0, 10.0, 0]]}]
    pos_w, neg_w = compute_class_weights(records)
    assert neg_w > pos_w


def test_compute_class_weights_degenerate_all_speech_falls_back_to_unweighted():
    records = [{"label_intervals": [[0.0, 5.0, 1]]}]
    assert compute_class_weights(records) == (1.0, 1.0)


def test_compute_class_weights_empty_manifest_falls_back():
    assert compute_class_weights([]) == (1.0, 1.0)


def test_masked_bce_loss_matches_manual_computation_no_mask_effect():
    probs = torch.tensor([[0.9, 0.1]])
    labels = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss = masked_bce_loss(probs, labels, mask)
    expected = -(torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.9))) / 2
    assert torch.allclose(loss, expected, atol=1e-5)


def test_masked_bce_loss_ignores_masked_out_positions():
    probs = torch.tensor([[0.9, 0.0001]])  # second position would be huge loss if counted
    labels = torch.tensor([[1, 1]])
    mask = torch.tensor([[True, False]])

    loss = masked_bce_loss(probs, labels, mask)
    expected = -torch.log(torch.tensor(0.9))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_masked_bce_loss_no_nan_at_extremes():
    probs = torch.tensor([[0.0, 1.0, 0.5]])
    labels = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[True, True, True]])

    loss = masked_bce_loss(probs, labels, mask)
    assert torch.isfinite(loss)


def test_masked_bce_loss_all_masked_out_does_not_divide_by_zero():
    probs = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([[1, 0]])
    mask = torch.tensor([[False, False]])

    loss = masked_bce_loss(probs, labels, mask)
    assert torch.isfinite(loss)


def test_masked_bce_loss_weighting_penalizes_minority_more():
    probs = torch.tensor([[0.1, 0.1]])  # wrong for both a speech and a silence frame
    labels = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    unweighted = masked_bce_loss(probs, labels, mask, pos_weight=1.0, neg_weight=1.0)
    pos_upweighted = masked_bce_loss(probs, labels, mask, pos_weight=5.0, neg_weight=1.0)
    assert pos_upweighted > unweighted


def test_lr_scheduler_warms_up_linearly_then_decays():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=10, total_steps=100, min_lr_ratio=0.01)

    lrs = []
    for _ in range(101):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[9], "lr should still be ramping up during warmup"
    assert abs(lrs[10] - 1e-3) < 1e-6, "lr should peak at base_lr right after warmup"
    assert lrs[-1] < lrs[10], "lr should have decayed by the end of the schedule"
    assert lrs[-1] >= 1e-3 * 0.01 - 1e-9, "lr should not decay below the configured floor"


def test_lr_scheduler_handles_total_steps_smaller_than_warmup():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # should not raise even in a degenerate config (e.g. a tiny smoke run)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=500, total_steps=10, min_lr_ratio=0.01)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
