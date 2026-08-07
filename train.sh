#!/usr/bin/env bash
# Full pipeline: verify data, build manifests (idempotent), train both
# architectures end to end (full schedule), evaluate, export to ONNX,
# verify export parity. Meant to be started and left running unattended
# (e.g. overnight) -- self-caffeinates so macOS sleep doesn't pause it.
#
# Usage:
#   ./train.sh                    # run everything, log to train_run_<timestamp>.log
#   EPOCHS=10 ./train.sh          # override epoch count for a shorter run
#
# Safe to re-run: preprocessing/manifest steps are idempotent (skip work
# already done); training always starts a fresh model under a timestamped
# run name, it does not resume a previous run.

set -euo pipefail

# Re-exec under caffeinate so the run survives lid-close/display-sleep.
if [ -z "${TRAIN_SH_CAFFEINATED:-}" ]; then
  export TRAIN_SH_CAFFEINATED=1
  exec caffeinate -dims "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${REPO_ROOT}/train_run_${TIMESTAMP}.log"
EPOCHS="${EPOCHS:-}"  # empty -> use configs/train/default.yaml's schedule.epochs

exec > >(tee -a "$LOG_FILE") 2>&1

step() {
  echo ""
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] $1"
  echo "=================================================================="
}

on_error() {
  echo ""
  echo "FAILED at line $1 -- see $LOG_FILE for the full log."
  exit 1
}
trap 'on_error $LINENO' ERR

source .venv/bin/activate

EPOCH_ARGS=()
if [ -n "$EPOCHS" ]; then
  EPOCH_ARGS=(--epochs "$EPOCHS")
fi

step "0/8 Verify data paths reachable"
python scripts/check_data_paths.py

step "1/8 Preprocess assets (idempotent -- skips already-converted files)"
python scripts/preprocess_assets.py

step "2/8 Build manifests (idempotent)"
python scripts/build_manifests.py

step "3/8 Train CRNN (crnn_v1) -- full schedule"
python scripts/train.py --model-config crnn_v1 --run-name crnn_v1_full "${EPOCH_ARGS[@]}"

step "4/8 Evaluate CRNN on TEN testset"
python scripts/evaluate.py --manifest test_ten --checkpoint checkpoints/crnn_v1_full/best.pt --model-config crnn_v1

step "5/8 Export CRNN to ONNX + verify parity"
python scripts/export.py --checkpoint checkpoints/crnn_v1_full/best.pt --model-config crnn_v1
python scripts/verify_export.py --checkpoint checkpoints/crnn_v1_full/best.pt --onnx checkpoints/crnn_v1_full/model.onnx --model-config crnn_v1

step "6/8 Train TCN (tcn_v1) -- full schedule"
python scripts/train.py --model-config tcn_v1 --run-name tcn_v1_full "${EPOCH_ARGS[@]}"

step "7/8 Evaluate TCN on TEN testset"
python scripts/evaluate.py --manifest test_ten --checkpoint checkpoints/tcn_v1_full/best.pt --model-config tcn_v1

step "8/8 Export TCN to ONNX + verify parity"
python scripts/export.py --checkpoint checkpoints/tcn_v1_full/best.pt --model-config tcn_v1
python scripts/verify_export.py --checkpoint checkpoints/tcn_v1_full/best.pt --onnx checkpoints/tcn_v1_full/model.onnx --model-config tcn_v1

step "Done"
echo "CRNN report: checkpoints/crnn_v1_full/eval_report_test_ten.json"
echo "TCN report:  checkpoints/tcn_v1_full/eval_report_test_ten.json"
echo "Full log:    $LOG_FILE"
