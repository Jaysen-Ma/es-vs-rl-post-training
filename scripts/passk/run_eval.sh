#!/bin/bash
# Evaluate one downloaded model across the four maths benchmarks, using the same
# generation settings for all arms: temperature=0.6, top_p=0.95, max_tokens=2048,
# qwen-boxed prompting and seed=1.
#
# Usage:
#   ./run_eval.sh [--config FILE] MODEL_ARM LABEL [RUN_TAG] [N_SAMPLING]
#   ./run_eval.sh --config config.toml 7B-base base my-run
#   ./run_eval.sh --config config.toml 7B-ES trained my-run
#   ./run_eval.sh --config config.toml 7B-RL rl my-run 32
#
# Install es-capacity in the maths evaluation environment and set MATH_EVAL_DIR
# to limit-of-RLVR/math/examples/math_eval. The optional config.sh supplies shell
# settings. --config takes precedence over ES_CAPACITY_CONFIG. Model paths are
# resolved locally before entering the external harness; no models are downloaded.
# Existing exported model-path variable names and direct local paths also work.
#
# N_SAMPLING overrides the budget for every benchmark; otherwise the standard
# per-benchmark budgets below apply. LABEL and RUN_TAG select results/RUN_TAG/LABEL.
set -euo pipefail

MODEL_CONFIG_ARGS=()
if [ "${1:-}" = "--config" ]; then
    if [ "$#" -lt 4 ]; then
        echo "Usage: $0 [--config FILE] MODEL_ARM LABEL [RUN_TAG] [N_SAMPLING]" >&2
        exit 2
    fi
    MODEL_CONFIG_ARGS=(--config "$2")
    shift 2
fi
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [--config FILE] MODEL_ARM LABEL [RUN_TAG] [N_SAMPLING]" >&2
    exit 2
fi

MODEL_DIR=$1
LABEL=$2
RUN_TAG=${3:-run1}
N_SAMPLING_OVERRIDE=${4:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -f "$REPO_ROOT/config.sh" ]; then
    source "$REPO_ROOT/config.sh"
fi
if [ -z "${MATH_EVAL_DIR:-}" ] && [ -n "${WORKSPACE:-}" ]; then
    MATH_EVAL_DIR="$WORKSPACE/limit-of-RLVR/math/examples/math_eval"
fi
if [ -z "${MATH_EVAL_DIR:-}" ]; then
    echo "ERROR: MATH_EVAL_DIR is not set." >&2
    echo "       Copy config.example.sh to config.sh and set it there, or:" >&2
    echo "       export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval" >&2
    exit 1
fi

if [[ "$MODEL_DIR" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -n "${!MODEL_DIR:-}" ]; then
    MODEL_DIR="${!MODEL_DIR}"
fi

# Resolve before changing directory to the external harness. All workflows use
# the installed package's local-only resolver; this does not download models.
MODEL_DIR=$(python -m es_capacity.cli.models path "$MODEL_DIR" "${MODEL_CONFIG_ARGS[@]}")

PROMPT_TYPE="qwen-boxed"
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=2048
SEED=1
# run_sharded_eval.sh's own default (0.9) assumes a GPU dedicated to this
# job, which doesn't hold on a shared/dev box. Override via env var
# (GPU_MEM_UTIL=0.3 ./run_eval.sh ...) if something else is using the GPU.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# benchmark:n_sampling pairs
PAIRS=(
    "aime24:512"
    "math500:128"
    "minerva_math:128"
    "olympiadbench:128"
)

OUT_ROOT="${REPO_ROOT}/results/${RUN_TAG}"
LABEL_OUT="${OUT_ROOT}/${LABEL}"
mkdir -p "$LABEL_OUT"

cd "$MATH_EVAL_DIR"

for pair in "${PAIRS[@]}"; do
    benchmark="${pair%%:*}"
    n_sampling="${pair##*:}"
    if [ -n "$N_SAMPLING_OVERRIDE" ]; then
        n_sampling="$N_SAMPLING_OVERRIDE"
    fi
    echo
    echo "########################################"
    echo "# Benchmark: $benchmark  (n_sampling=$n_sampling, model=$LABEL)"
    echo "########################################"
    ./run_sharded_eval.sh "$MODEL_DIR" "$LABEL_OUT" "$benchmark" "$n_sampling" \
        "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "$PROMPT_TYPE" "$SEED" "" "$GPU_MEM_UTIL"
done

echo
echo "All benchmarks complete for $LABEL. Outputs in ${LABEL_OUT}/{aime24,math500,minerva_math,olympiadbench}/"
echo "Analyze with: python ${SCRIPT_DIR}/analyze.py --run-tag ${RUN_TAG} --label ... --baseline ... --plot"
