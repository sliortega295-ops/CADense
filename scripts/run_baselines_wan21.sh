#!/usr/bin/env bash
set -euo pipefail
PROMPT="${1:-A red toy car turns sharply around a blue cube on a wooden table.}"
OUTPUT_ROOT="${2:-outputs/baselines}"
SEED="${3:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BASE=(--prompt "$PROMPT" --seed "$SEED" --height 480 --width 832 --num-frames 81 --steps 50 \
      --guidance-scale 5.0 --flow-shift 3.0 --num-anchors 9 --sparse-block-start 3 --sparse-block-end 27 \
      --block-group-size 3 --kv-mode anchor_only --output-dir "$OUTPUT_ROOT")

python scripts/run_wan21_1_3b.py --method dense --run-name "dense_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method fixed --warmup-steps 5 --interpolation-target delta --run-name "fixed_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method rhyme --warmup-steps 5 --interpolation-target delta --run-name "rhyme_selector_seed${SEED}" "${BASE[@]}"
# FIS uses block-interleaved anchors and state interpolation.  Dense tail is explicit; set FIS_DENSE_TAIL_STEPS if desired.
python scripts/run_wan21_1_3b.py --method fis --fis-dense-tail-steps "${FIS_DENSE_TAIL_STEPS:-0}" --interpolation-target state --run-name "fis_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method coframe --refresh-signal defect --warmup-steps 5 --interpolation-target delta --run-name "coframe_seed${SEED}" "${BASE[@]}"
