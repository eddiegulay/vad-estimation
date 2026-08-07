#!/usr/bin/env python3
"""Phase 8 — ONNX export entrypoint (plan §6/§8).

Usage:
    python scripts/export.py --checkpoint checkpoints/crnn_v1_smoke/last.pt
    python scripts/export.py --checkpoint checkpoints/crnn_v1_smoke/last.pt --out checkpoints/crnn_v1_smoke/model.onnx
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402
from vad.engine.checkpoint import load_checkpoint  # noqa: E402
from vad.export.onnx_export import export_onnx  # noqa: E402
from vad.models import build_model  # noqa: E402


def main(checkpoint_path: str, out_path: str | None) -> int:
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model" / "crnn_v1.yaml")
    model = build_model(model_cfg["architecture"], model_cfg)
    meta = load_checkpoint(checkpoint_path, model, map_location="cpu")
    print(f"loaded checkpoint: {checkpoint_path} (step={meta.get('step')}, epoch={meta.get('epoch')})")

    resolved_out = Path(out_path) if out_path else Path(checkpoint_path).parent / "model.onnx"
    export_onnx(
        model,
        chunk_samples=model_cfg["frontend"]["chunk_samples"],
        hidden_size=model_cfg["recurrent"]["hidden_size"],
        out_path=resolved_out,
    )
    print(f"exported: {resolved_out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args.checkpoint, args.out))
