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

from vad.config import load_config, load_yaml  # noqa: E402
from vad.data.dataset import VADDataset  # noqa: E402
from vad.data.manifest import load_index  # noqa: E402
from vad.engine.checkpoint import load_checkpoint  # noqa: E402
from vad.eval.evaluate import run_evaluation  # noqa: E402
from vad.models import build_model  # noqa: E402

MANIFEST_FILES = {
    "test_ten": "test_ten.jsonl",
    "val": "val.jsonl",
    "sanity_fleurs": "sanity_fleurs.jsonl",
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(name)


def main(manifest_name: str, checkpoint_path: str | None, plot_dir: str | None) -> int:
    data_cfg = load_config(
        REPO_ROOT / "configs" / "data" / "paths.yaml", REPO_ROOT / "configs" / "data" / "default.yaml"
    )
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / "crnn_v1.yaml")
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

    metrics = run_evaluation(
        model,
        dataset,
        device,
        chunk_samples,
        hop_s,
        onset_tolerance_s=eval_cfg["boundary_tolerance_ms"] / 1000.0,
        num_plot_examples=num_plots,
        plot_dir=REPO_ROOT / plot_dir_resolved if num_plots > 0 else None,
    )

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
    args = parser.parse_args()
    raise SystemExit(main(args.manifest, args.checkpoint, args.plot_dir))
