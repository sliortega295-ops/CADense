#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}

python -m coframe.cli \
  --method coframe_ode \
  --prompt "$PROMPT" \
  --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 --steps 50 \
  --warmup-steps 5 --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode anchor_only \
  --interpolation-target delta \
  --decode --vae-tiling \
  --output-dir "$OUT_ROOT" \
  --run-name "coframe_ode_seed${SEED}"
