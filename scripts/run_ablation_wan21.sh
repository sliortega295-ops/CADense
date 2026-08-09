#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A red toy car turns sharply around a blue cube on a wooden table.}"
OUTPUT_ROOT="${2:-outputs/wan21_main}"
SEED="${3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --prompt "$PROMPT"
  --seed "$SEED"
  --height 480 --width 832 --num-frames 81 --steps 50
  --guidance-scale 5.0 --flow-shift 3.0
  --warmup-steps 5 --num-anchors 9
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3
  --kv-mode anchor_only
  --interpolation-target delta --defect-target delta
  --output-dir "$OUTPUT_ROOT"
)

for METHOD in dense fixed rhyme coframe; do
  python scripts/run_wan21_1_3b.py \
    --method "$METHOD" \
    --run-name "${METHOD}_seed${SEED}" \
    "${COMMON[@]}"
done

python scripts/compare_runs.py \
  --dense "$OUTPUT_ROOT/dense_seed${SEED}/latents.pt" \
  --candidate fixed "$OUTPUT_ROOT/fixed_seed${SEED}/latents.pt" \
  --candidate rhyme "$OUTPUT_ROOT/rhyme_seed${SEED}/latents.pt" \
  --candidate coframe "$OUTPUT_ROOT/coframe_seed${SEED}/latents.pt" \
  --output-json "$OUTPUT_ROOT/comparison_seed${SEED}.json"
