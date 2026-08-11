#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A gymnast performs a fast cartwheel while a yellow ball rolls behind her.}"
OUTPUT_ROOT="${2:-outputs/stage1c_curvature}"
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
  --kv-mode full_kv
  --interpolation-target delta --defect-target delta
  --oracle-probe-steps 5,20,40
  --oracle-probe-blocks 8,14,20
  --oracle-probe-horizons 1,3
  --probe-counterfactual-methods rhyme,fis,fixed
  --probe-curvature-signals
  --output-dir "$OUTPUT_ROOT"
)

# Two trajectories are enough for signal screening. We do not deploy the
# curvature policy yet; every probe ranks its hypothetical one-swap action
# against dense oracle truth from the exact same block input.
for SIGNAL in none gap_only; do
  python scripts/run_wan21_1_3b.py \
    --method coframe \
    --refresh-signal "$SIGNAL" \
    --run-name "stage1c_${SIGNAL}_full_kv_seed${SEED}" \
    "${COMMON[@]}"
done
