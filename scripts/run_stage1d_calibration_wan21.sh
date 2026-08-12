#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}
FOLD_ID=${4:?fold id such as p0_s0 required}

mkdir -p "$OUT_ROOT"

# This is a uniform K=9 trajectory that computes leave-one-out defects at every
# sparse block group. Using adaptive_k with a singleton budget keeps K exactly 9
# while exercising the same defect-measurement path used by the adaptive runs.
python -m coframe.cli \
  --prompt "$PROMPT" --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --guidance-scale 5 --flow-shift 3 \
  --warmup-steps 5 \
  --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode full_kv \
  --interpolation-target delta --defect-target delta \
  --oracle-probe-steps 5,20,40 \
  --oracle-probe-blocks 8,14,20 \
  --oracle-probe-horizons 1,3 \
  --probe-counterfactual-methods fixed,fis \
  --method adaptive_k \
  --adaptive-k-policy mean_defect \
  --adaptive-k-values 9 \
  --adaptive-k-thresholds "" \
  --output-dir "$OUT_ROOT" \
  --run-name "${FOLD_ID}_calibration_k9"
