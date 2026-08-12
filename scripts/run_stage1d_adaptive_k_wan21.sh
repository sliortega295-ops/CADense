#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}
PLAN=${4:?budget plan json required}
FOLD_ID=${5:?fold id such as p0_s0 required}

mkdir -p "$OUT_ROOT"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

python - "$PLAN" "$FOLD_ID" "$TMP_DIR" <<'PY'
import json
import pathlib
import sys
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
fold_id = sys.argv[2]
out = pathlib.Path(sys.argv[3])
fold = plan["folds"][fold_id]
(out / "mean_thresholds.txt").write_text(",".join(str(x) for x in fold["mean_defect_thresholds"]))
(out / "max_thresholds.txt").write_text(",".join(str(x) for x in fold["max_defect_thresholds"]))
(out / "schedule.json").write_text(json.dumps(fold["step_block_schedule"], indent=2))
(out / "budgets.txt").write_text(",".join(str(x) for x in plan["budget_values"]))
PY

MEAN_THRESHOLDS=$(cat "$TMP_DIR/mean_thresholds.txt")
MAX_THRESHOLDS=$(cat "$TMP_DIR/max_thresholds.txt")
BUDGETS=$(cat "$TMP_DIR/budgets.txt")

COMMON=(
  --prompt "$PROMPT"
  --seed "$SEED"
  --height 480 --width 832 --num-frames 81
  --steps 50 --guidance-scale 5 --flow-shift 3
  --warmup-steps 5
  --num-anchors 9
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3
  --kv-mode full_kv
  --interpolation-target delta --defect-target delta
  --oracle-probe-steps 5,20,40
  --oracle-probe-blocks 8,14,20
  --oracle-probe-horizons 1,3
  --probe-counterfactual-methods fixed,fis
  --output-dir "$OUT_ROOT"
)

# Dense endpoint reference. No adaptive budget and no interpretation of its latency
# when GPUs are shared with unrelated jobs.
python -m coframe.cli \
  --prompt "$PROMPT" --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --guidance-scale 5 --flow-shift 3 \
  --method dense --output-dir "$OUT_ROOT" \
  --run-name "${FOLD_ID}_dense"

# Static exact-frame budget baseline.
python -m coframe.cli "${COMMON[@]}" \
  --method fixed \
  --run-name "${FOLD_ID}_fixed_k9"

# Prompt-independent schedule calibrated only from the other seven prompts.
python -m coframe.cli "${COMMON[@]}" \
  --method adaptive_k \
  --adaptive-k-policy step_block \
  --adaptive-k-values "$BUDGETS" \
  --adaptive-k-schedule-json "$TMP_DIR/schedule.json" \
  --run-name "${FOLD_ID}_step_block"

# Primary causal policy: previous completed group mean defect controls next-group K.
python -m coframe.cli "${COMMON[@]}" \
  --method adaptive_k \
  --adaptive-k-policy mean_defect \
  --adaptive-k-values "$BUDGETS" \
  --adaptive-k-thresholds "$MEAN_THRESHOLDS" \
  --run-name "${FOLD_ID}_mean_defect"

# Ablation: max defect uses the same causal timing and LOPO calibration.
python -m coframe.cli "${COMMON[@]}" \
  --method adaptive_k \
  --adaptive-k-policy max_defect \
  --adaptive-k-values "$BUDGETS" \
  --adaptive-k-thresholds "$MAX_THRESHOLDS" \
  --run-name "${FOLD_ID}_max_defect"
