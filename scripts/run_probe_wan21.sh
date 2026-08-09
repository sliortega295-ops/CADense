#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A gymnast performs a fast cartwheel while a yellow ball rolls behind her.}"
OUTPUT_ROOT="${2:-outputs/wan21_probe}"
SEED="${3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

for KV_MODE in anchor_only full_kv; do
  python scripts/run_wan21_1_3b.py \
    --method coframe \
    --run-name "coframe_${KV_MODE}_seed${SEED}" \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --height 480 --width 832 --num-frames 81 --steps 50 \
    --guidance-scale 5.0 --flow-shift 3.0 \
    --warmup-steps 5 --num-anchors 9 \
    --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
    --kv-mode "$KV_MODE" \
    --interpolation-target delta --defect-target delta \
    --oracle-probe-steps 5,20,40 \
    --oracle-probe-blocks 8,14,20 \
    --oracle-probe-horizons 1,3 \
    --output-dir "$OUTPUT_ROOT"
done
