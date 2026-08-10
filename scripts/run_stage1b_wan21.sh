#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A gymnast performs a fast cartwheel while a yellow ball rolls behind her.}"
OUTPUT_ROOT="${2:-outputs/stage1b}"
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
  --output-dir "$OUTPUT_ROOT"
)

# These four runs attribute improvement to the refresh signal itself.  Every
# oracle probe also evaluates matched-input Rhyme/FIS/fixed operators.
for SIGNAL in none gap_only shuffled defect; do
  python scripts/run_wan21_1_3b.py \
    --method coframe \
    --refresh-signal "$SIGNAL" \
    --run-name "coframe_${SIGNAL}_full_kv_seed${SEED}" \
    "${COMMON[@]}"
done

python scripts/summarize_stage1b.py --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/stage1b_summary.json"
