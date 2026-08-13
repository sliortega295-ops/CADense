#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
PROMPT_ID=${2:?prompt id p0_s0..p7_s0 required}
OUTPUT_ROOT=${3:?shared output root required}
PLAN=${4:?frozen protocol JSON required}
SEED=${5:-0}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"

# GPU selection is deliberately external. The current launch allocation uses
# physical GPUs 1-4, one prompt process per GPU; never hard-code CUDA indices in
# this scientific launcher and never use DDP.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must select exactly one externally assigned GPU" >&2
  exit 2
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "one prompt job must see exactly one GPU, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
if [[ "$SEED" != "0" ]]; then
  echo "frozen screen requires seed 0" >&2
  exit 2
fi

python scripts/run_cross_step_endpoint_wan21.py \
  --plan "$PLAN" \
  --prompt "$PROMPT" \
  --prompt-id "$PROMPT_ID" \
  --model-id "${COFRAME_MODEL_ID:-$ROOT_DIR/models/Wan2.1-T2V-1.3B-Diffusers}" \
  --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --warmup-steps 5 \
  --guidance-scale 5 --flow-shift 3 \
  --local-files-only \
  --output-root "$OUTPUT_ROOT"
