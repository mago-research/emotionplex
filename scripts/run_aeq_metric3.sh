#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_aeq_metric3.sh JSON WAV_DIR OUTPUT_DIR [DEVICE] [MODEL_WEIGHT]

Arguments:
  JSON          AEQ-Bench dataset JSON
  WAV_DIR       Directory containing AEQ input WAV files
  OUTPUT_DIR    Inference and evaluation output directory
  DEVICE        Torch/FunASR device (default: cuda:0)
  MODEL_WEIGHT  Optional merged model.safetensors; omit for stock PersonaPlex

The OpenAI text judge runs only when OPENAI_API_KEY is set.
Set PERSONAPLEX_PYTHON and AEQ_EVAL_PYTHON when inference and evaluation use
different virtual environments (both default to python).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 3 || $# > 5 )); then
  usage >&2
  exit 2
fi

JSON_PATH=$1
WAV_DIR=$2
OUTPUT_DIR=$3
DEVICE=${4:-cuda:0}
MODEL_WEIGHT=${5:-}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREDICTIONS="$OUTPUT_DIR/predictions.csv"
EVAL_DIR="$OUTPUT_DIR/evaluation"
PERSONAPLEX_PYTHON=${PERSONAPLEX_PYTHON:-python}
AEQ_EVAL_PYTHON=${AEQ_EVAL_PYTHON:-python}

inference_args=(
  "$PERSONAPLEX_PYTHON" "$REPO_ROOT/inference/aeq_metric3/run_batch_metric3.py"
  --json "$JSON_PATH"
  --wav-dir "$WAV_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --skip-existing
)
if [[ -n "$MODEL_WEIGHT" ]]; then
  inference_args+=(--moshi-weight "$MODEL_WEIGHT")
fi

"${inference_args[@]}"
mkdir -p "$EVAL_DIR"

"$AEQ_EVAL_PYTHON" "$REPO_ROOT/evaluation/aeq_metric3/metric3_emotion2vec.py" \
  --input-csv "$PREDICTIONS" \
  --out-csv "$EVAL_DIR/metric3_audio.csv" \
  --device "$DEVICE"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  "$AEQ_EVAL_PYTHON" "$REPO_ROOT/evaluation/aeq_metric3/metric3_text_judge.py" \
    --input-csv "$PREDICTIONS" \
    --out-csv "$EVAL_DIR/metric3_text.csv"
else
  printf '%s\n' \
    "[AEQ] OPENAI_API_KEY is not set; skipping the Metric 3 text judge." >&2
fi
