#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A gymnast performs a fast cartwheel while a yellow ball rolls behind her.}"
OUTPUT_ROOT="${2:-outputs/entry_state_proxy_dp}"
SEED="${3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# A single static-Rhyme trajectory supplies matched block inputs. Fixed, FIS,
# Rhyme, one-swap gap-only, Entry-State Proxy-DP, and the exact dense oracle are
# all scored against the same dense block truth. Proxy-DP remains probe-only.
python scripts/run_wan21_1_3b.py \
  --method coframe \
  --refresh-signal none \
  --prompt "$PROMPT" \
  --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 --steps 50 \
  --guidance-scale 5.0 --flow-shift 3.0 \
  --warmup-steps 5 --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode full_kv \
  --interpolation-target delta --defect-target delta \
  --sketch-dim 64 \
  --oracle-probe-steps 5,20,40 \
  --oracle-probe-blocks 8,14,20 \
  --oracle-probe-horizons 1,3 \
  --probe-counterfactual-methods rhyme,fis,fixed \
  --probe-entry-state-proxy-dp \
  --local-files-only \
  --run-name "entry_state_proxy_dp_none_full_kv_seed${SEED}" \
  --output-dir "$OUTPUT_ROOT"
