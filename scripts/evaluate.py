#!/usr/bin/env python3
"""Phase 7 — evaluation entrypoint (plan §6/§7).

Usage:
    python scripts/evaluate.py --manifest test_ten
        # no --checkpoint -> randomly-initialized model; this is the
        # pre-training plumbing smoke test the plan calls for: proves the
        # eval/plotting harness works against real, already-correct ground
        # truth (TEN's .scv labels) before any model has been trained.
    python scripts/evaluate.py --manifest test_ten --checkpoint checkpoints/crnn_v1/last.pt
    python scripts/evaluate.py --manifest val --checkpoint checkpoints/crnn_v1/last.pt
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

import numpy as np  # noqa: E402

from vad.config import load_config, load_yaml  # noqa: E402
from vad.data.dataset import VADDataset  # noqa: E402
from vad.data.manifest import load_index  # noqa: E402
from vad.engine.checkpoint import load_checkpoint  # noqa: E402
from vad.eval.evaluate import run_evaluation  # noqa: E402
from vad.eval.metrics import sweep_thresholds  # noqa: E402
from vad.models import build_model  # noqa: E402
from vad.postprocess.hysteresis import ms_to_frames  # noqa: E402

MANIFEST_FILES = {
    "test_ten": "test_ten.jsonl",
    "val": "val.jsonl",
    "sanity_fleurs": "sanity_fleurs.jsonl",
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(name)


def calibrate_theta_on(model, val_dataset, device, chunk_samples: int, hop_s: float, grid: np.ndarray) -> dict:
    """F1-maximizing operating threshold on the val split (never the test
    split being reported on) -- see vad.eval.metrics.sweep_thresholds.
    """
    model.eval()
    all_probs, all_labels = [], []
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        waveform = item["waveform"].unsqueeze(0).to(device)
        labels = item["labels"].numpy()
        usable_len = (waveform.shape[1] // chunk_samples) * chunk_samples
        if usable_len == 0:
            continue
        with torch.no_grad():
            probs = model.forward_full(waveform[:, :usable_len]).cpu().numpy()[0]
        n = min(len(probs), len(labels))
        if n == 0:
            continue
        all_probs.append(probs[:n])
        all_labels.append(labels[:n])

    if not all_probs:
        return {"best_threshold": 0.5, "best_f1": float("nan"), "sweep": []}
    return sweep_thresholds(np.concatenate(all_probs), np.concatenate(all_labels), grid)


def main(manifest_name: str, checkpoint_path: str | None, plot_dir: str | None, model_config: str = "crnn_v1") -> int:
    data_cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml", REPO_ROOT / "configs" / "data" / "default.yaml"
    )
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / f"{model_config}.yaml")
    eval_cfg = load_yaml(REPO_ROOT / "configs" / "eval" / "default.yaml")
    train_cfg = load_yaml(REPO_ROOT / "configs" / "train" / "default.yaml")

    cache_root = REPO_ROOT / data_cfg["cache_root"]
    manifests_dir = REPO_ROOT / data_cfg["manifests_dir"]
    sample_rate = data_cfg["sample_rate"]
    chunk_samples = model_cfg["frontend"]["chunk_samples"]
    hop_s = chunk_samples / sample_rate

    manifest_path = manifests_dir / MANIFEST_FILES[manifest_name]
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path} (run scripts/build_manifests.py first)")
        return 1

    index_dir = cache_root / "index"
    esc50_index = load_index(index_dir / "esc50_index.json") if (index_dir / "esc50_index.json").exists() else []
    rir_index = load_index(index_dir / "rir_index.json") if (index_dir / "rir_index.json").exists() else []

    dataset = VADDataset(manifest_path, cache_root, sample_rate, hop_s, esc50_index, rir_index)

    device = resolve_device(train_cfg["device"])
    model = build_model(model_cfg["architecture"], model_cfg).to(device)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, map_location=str(device))
        print(f"loaded checkpoint: {checkpoint_path}")
    else:
        print("no --checkpoint given -- evaluating a randomly-initialized model "
              "(pre-training plumbing smoke test)")

    plot_dir_resolved = plot_dir or eval_cfg["qualitative_plots"]["out_dir"]
    num_plots = eval_cfg["qualitative_plots"]["num_examples"] if eval_cfg["qualitative_plots"]["enabled"] else 0

    pp_cfg = eval_cfg["postprocess"]
    theta_on = pp_cfg["theta_on"]
    calibration = None
    calib_cfg = eval_cfg.get("calibration", {})
    if checkpoint_path is not None and calib_cfg.get("enabled") and manifest_name != "val":
        val_path = manifests_dir / "val.jsonl"
        if val_path.exists():
            print("calibrating theta_on on the val split...")
            val_dataset = VADDataset(val_path, cache_root, sample_rate, hop_s, esc50_index, rir_index)
            grid = np.arange(calib_cfg["grid_min"], calib_cfg["grid_max"] + 1e-9, calib_cfg["grid_step"])
            calibration = calibrate_theta_on(model, val_dataset, device, chunk_samples, hop_s, grid)
            theta_on = calibration["best_threshold"]
            print(f"  calibrated theta_on={theta_on:.2f} (val f1={calibration['best_f1']:.4f})")

    metrics = run_evaluation(
        model,
        dataset,
        device,
        chunk_samples,
        hop_s,
        onset_tolerance_s=eval_cfg["boundary_tolerance_ms"] / 1000.0,
        num_plot_examples=num_plots,
        plot_dir=REPO_ROOT / plot_dir_resolved if num_plots > 0 else None,
        theta_on=theta_on,
        theta_off=pp_cfg["theta_off"],
        min_speech_frames=ms_to_frames(pp_cfg["min_speech_ms"], hop_s),
        min_silence_frames=ms_to_frames(pp_cfg["min_silence_ms"], hop_s),
    )
    if calibration is not None:
        metrics["calibration"] = calibration

    print(json.dumps(metrics, indent=2))

    report_dir = Path(checkpoint_path).parent if checkpoint_path else REPO_ROOT / "checkpoints" / "smoke_eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_report_{manifest_name}.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nreport written to {report_path}")
    if num_plots > 0:
        print(f"qualitative plots written to {REPO_ROOT / plot_dir_resolved}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", choices=list(MANIFEST_FILES), default="test_ten")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument(
        "--model-config", type=str, default="crnn_v1", help="name under configs/model/ (without .yaml)"
    )
    args = parser.parse_args()
    raise SystemExit(main(args.manifest, args.checkpoint, args.plot_dir, args.model_config))
