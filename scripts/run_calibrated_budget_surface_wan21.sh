#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}
FOLD_ID=${4:?fold id such as p0_s0 required}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# The deployed trajectory remains uniform K=9 because an absent step/group key
# falls back to num_anchors=9. Every three-block group is additionally branched
# counterfactually at K={6,9,12,21} from the exact same group input.
python -m coframe.cli \
  --prompt "$PROMPT" --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --guidance-scale 5 --flow-shift 3 \
  --warmup-steps 5 --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode full_kv \
  --interpolation-target delta --defect-target delta \
  --method adaptive_k \
  --adaptive-k-policy step_block \
  --adaptive-k-values 6,9,12,21 \
  --calibrated-budget-probe-mode surface \
  --local-files-only \
  --output-dir "$OUT_ROOT" \
  --run-name "${FOLD_ID}_surface_uniform_k9"
