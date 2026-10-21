#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_fdb.sh DATASET OUTPUT UPSTREAM_CHECKOUT DEVICE [MODEL_WEIGHT|-] [SPLIT ...]

Arguments:
  DATASET            FDB dataset root: <split>/<task>/<sample>/input.wav
  OUTPUT             Result root
  UPSTREAM_CHECKOUT  Pinned DanielLin94144/Full-Duplex-Bench checkout
  DEVICE             PersonaPlex device (for example cuda or cuda:0)
  MODEL_WEIGHT       Optional merged model.safetensors; use "-" for stock
  SPLIT              Optional split names (default: all discovered splits)

The script always copies missing benchmark companion files into OUTPUT, then
runs upstream ASR and evaluation task by task. If OPENAI_API_KEY is absent,
user_interruption evaluation is explicitly skipped; all non-OpenAI tasks run.
Set PERSONAPLEX_PYTHON and FDB_PYTHON when inference and upstream evaluation
use different virtual environments (both default to python).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 4 )); then
  usage >&2
  exit 2
fi

DATASET=$1
OUTPUT=$2
UPSTREAM_CHECKOUT=$3
DEVICE=$4
MODEL_WEIGHT=${5:--}
if (( $# >= 5 )); then
  shift 5
else
  shift 4
fi
SPLITS=("$@")

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_ROOT="$UPSTREAM_CHECKOUT/v1_v1.5"
ASR="$UPSTREAM_ROOT/get_transcript/asr.py"
EVALUATE="$UPSTREAM_ROOT/evaluation/evaluate.py"
GET_TIMING="$UPSTREAM_ROOT/evaluation/get_timing.py"
LOG_ROOT="$OUTPUT/logs"
PERSONAPLEX_PYTHON=${PERSONAPLEX_PYTHON:-python}
FDB_PYTHON=${FDB_PYTHON:-python}
EXPECTED_FDB_COMMIT=3e799c45a045256f47d5f1c9cda90157e2d2ec9e

for required_file in "$ASR" "$EVALUATE" "$GET_TIMING"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Required upstream file not found: %s\n' "$required_file" >&2
    exit 2
  fi
done

actual_commit=$(git -C "$UPSTREAM_CHECKOUT" rev-parse HEAD 2>/dev/null || true)
if [[ "$actual_commit" != "$EXPECTED_FDB_COMMIT" ]]; then
  printf 'FDB checkout must be pinned to %s (found %s).\n' \
    "$EXPECTED_FDB_COMMIT" "${actual_commit:-not a Git checkout}" >&2
  exit 2
fi

inference_args=(
  "$PERSONAPLEX_PYTHON" "$REPO_ROOT/inference/fdb/run_fdb.py"
  --dataset "$DATASET"
  --output "$OUTPUT"
  --device "$DEVICE"
  --copy-companions
  --resume
)
if [[ "$MODEL_WEIGHT" != "-" ]]; then
  inference_args+=(--model-weight "$MODEL_WEIGHT")
fi
if (( ${#SPLITS[@]} > 0 )); then
  inference_args+=(--splits "${SPLITS[@]}")
fi
"${inference_args[@]}"

split_dirs=()
if (( ${#SPLITS[@]} > 0 )); then
  for split in "${SPLITS[@]}"; do
    split_dirs+=("$OUTPUT/$split")
  done
else
  for candidate in "$OUTPUT"/*; do
    [[ -d "$candidate" && "$(basename "$candidate")" != "logs" ]] || continue
    split_dirs+=("$candidate")
  done
fi

for split_dir in "${split_dirs[@]}"; do
  [[ -d "$split_dir" ]] || continue
  split=$(basename "$split_dir")
  for task_dir in "$split_dir"/*; do
    [[ -d "$task_dir" ]] || continue
    task=$(basename "$task_dir")
    log_dir="$LOG_ROOT/$split"
    mkdir -p "$log_dir"

    asr_task=default
    if [[ "$split" != "v1_5" && "$task" == *user_interruption* ]]; then
      asr_task=user_interruption
    fi
    (
      cd "$log_dir"
      "$FDB_PYTHON" "$ASR" --root_dir "$task_dir" --task "$asr_task" 2>&1 |
        tee "$task.asr.log"
    )

    if [[ "$split" == "v1_5" ]]; then
      (
        cd "$log_dir"
        "$FDB_PYTHON" "$GET_TIMING" --root_dir "$task_dir" 2>&1 |
          tee "$task.timing.log"
        "$FDB_PYTHON" "$EVALUATE" \
          --task general_before_after \
          --root_dir "$task_dir" 2>&1 |
          tee "$task.evaluate.log"
      )
      continue
    fi

    case "$task" in
      *pause_handling*) evaluator_task=pause_handling ;;
      *backchannel*) evaluator_task=backchannel ;;
      *turn_taking*) evaluator_task=smooth_turn_taking ;;
      *user_interruption*) evaluator_task=user_interruption ;;
      *)
        printf '[FDB] No v1.0 evaluator mapping for %s/%s; skipping.\n' \
          "$split" "$task" >&2
        continue
        ;;
    esac

    if [[ "$evaluator_task" == "user_interruption" &&
          -z "${OPENAI_API_KEY:-}" ]]; then
      printf '%s\n' \
        "[FDB] OPENAI_API_KEY is not set; skipping user_interruption evaluation for $split/$task." >&2
      continue
    fi
    (
      cd "$log_dir"
      "$FDB_PYTHON" "$EVALUATE" \
        --task "$evaluator_task" \
        --root_dir "$task_dir" 2>&1 |
        tee "$task.evaluate.log"
    )
  done
done
