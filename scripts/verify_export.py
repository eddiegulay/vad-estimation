#!/usr/bin/env python3
"""Phase 8 — verify an ONNX export (plan §6/§8):
1. Numerical parity between PyTorch `forward_full` and the exported ONNX
   model run chunk-by-chunk (`run_onnx_streaming`), on real TEN audio.
2. Re-run Phase 7's frame-level metrics through the ONNX path on TEN and
   confirm they match the PyTorch checkpoint's `eval_report_test_ten.json`
   within tolerance -- proves export didn't silently change behavior.

Usage:
    python scripts/verify_export.py --checkpoint checkpoints/crnn_v1_smoke/last.pt \
        --onnx checkpoints/crnn_v1_smoke/model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from vad.config import load_config, load_yaml  # noqa: E402
from vad.data.dataset import VADDataset  # noqa: E402
from vad.engine.checkpoint import load_checkpoint  # noqa: E402
from vad.eval.metrics import frame_auroc, frame_precision_recall_f1  # noqa: E402
from vad.export.onnx_export import run_onnx_streaming, run_onnx_streaming_tcn  # noqa: E402
from vad.models import build_model  # noqa: E402

PARITY_TOLERANCE = 1e-3
METRIC_TOLERANCE = 0.02
N_PARITY_EXAMPLES = 5


def main(checkpoint_path: str, onnx_path: str, model_config: str = "crnn_v1") -> int:
    data_cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml", REPO_ROOT / "configs" / "data" / "default.yaml"
    )
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / f"{model_config}.yaml")
    is_tcn = model_cfg["architecture"] == "tcn"
    sample_rate = data_cfg["sample_rate"]
    chunk_samples = model_cfg["frontend"]["chunk_samples"]
    hidden_size = model_cfg["recurrent"]["hidden_size"] if not is_tcn else None
    context_samples = 64
    hop_s = chunk_samples / sample_rate

    model = build_model(model_cfg["architecture"], model_cfg)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()

    session = ort.InferenceSession(str(onnx_path))

    manifests_dir = REPO_ROOT / data_cfg["manifests_dir"]
    cache_root = REPO_ROOT / data_cfg["cache_root"]
    dataset = VADDataset(manifests_dir / "test_ten.jsonl", cache_root, sample_rate, hop_s)

    # --- 1. numerical parity on a handful of real examples ---
    max_diffs = []
    for i in range(min(N_PARITY_EXAMPLES, len(dataset))):
        item = dataset[i]
        waveform = item["waveform"].numpy()
        usable_len = (len(waveform) // chunk_samples) * chunk_samples
        if usable_len == 0:
            continue
        waveform = waveform[:usable_len]

        with torch.no_grad():
            torch_probs = model.forward_full(torch.from_numpy(waveform).unsqueeze(0))[0].numpy()
        if is_tcn:
            onnx_probs = run_onnx_streaming_tcn(session, waveform, chunk_samples, model)
        else:
            onnx_probs = run_onnx_streaming(session, waveform, chunk_samples, context_samples, hidden_size)

        diff = float(np.max(np.abs(torch_probs - onnx_probs)))
        max_diffs.append(diff)
        print(f"  {item['id']}: max diff = {diff:.6f}")

    overall_max_diff = max(max_diffs) if max_diffs else float("nan")
    parity_ok = overall_max_diff < PARITY_TOLERANCE
    print(f"\nparity check: max diff across {len(max_diffs)} examples = {overall_max_diff:.6f} "
          f"(tolerance {PARITY_TOLERANCE}) -- {'PASS' if parity_ok else 'FAIL'}")

    # --- 2. full TEN metrics via the ONNX path ---
    all_probs, all_labels = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        waveform = item["waveform"].numpy()
        labels = item["labels"].numpy()
        usable_len = (len(waveform) // chunk_samples) * chunk_samples
        if usable_len == 0:
            continue
        waveform = waveform[:usable_len]
        if is_tcn:
            probs = run_onnx_streaming_tcn(session, waveform, chunk_samples, model)
        else:
            probs = run_onnx_streaming(session, waveform, chunk_samples, context_samples, hidden_size)
        n = min(len(probs), len(labels))
        all_probs.append(probs[:n])
        all_labels.append(labels[:n])

    probs_concat = np.concatenate(all_probs)
    labels_concat = np.concatenate(all_labels)
    preds_concat = (probs_concat > 0.5).astype(int)

    onnx_metrics = frame_precision_recall_f1(preds_concat, labels_concat)
    onnx_metrics["auroc"] = frame_auroc(probs_concat, labels_concat)

    report_path = Path(checkpoint_path).parent / "eval_report_test_ten.json"
    metrics_match = True
    if report_path.exists():
        pytorch_metrics = json.loads(report_path.read_text())
        print(f"\nmetric comparison (PyTorch checkpoint report vs ONNX, tolerance {METRIC_TOLERANCE}):")
        for key in ("precision", "recall", "f1", "auroc"):
            py_val, onnx_val = pytorch_metrics[key], onnx_metrics[key]
            diff = abs(py_val - onnx_val)
            ok = diff < METRIC_TOLERANCE
            metrics_match = metrics_match and ok
            print(f"  {key}: pytorch={py_val:.4f} onnx={onnx_val:.4f} diff={diff:.4f} {'OK' if ok else 'MISMATCH'}")
    else:
        print(f"\nno PyTorch eval report found at {report_path} -- run scripts/evaluate.py first to compare")
        metrics_match = False

    print(f"\nONNX metrics: {json.dumps(onnx_metrics, indent=2)}")

    onnx_report_path = Path(checkpoint_path).parent / "eval_report_test_ten_onnx.json"
    onnx_report_path.write_text(json.dumps(onnx_metrics, indent=2))
    print(f"ONNX report written to {onnx_report_path}")

    success = parity_ok and metrics_match
    print(f"\n{'PASS' if success else 'FAIL'}: export verification {'succeeded' if success else 'failed'}")
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument(
        "--model-config", type=str, default="crnn_v1", help="name under configs/model/ (without .yaml)"
    )
    args = parser.parse_args()
    raise SystemExit(main(args.checkpoint, args.onnx, args.model_config))
