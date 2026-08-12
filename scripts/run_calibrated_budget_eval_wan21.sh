#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}
SCHEDULE=${4:?held-out schedule json required}
FOLD_ID=${5:?fold id such as p0_s0 required}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python - "$SCHEDULE" "$FOLD_ID" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
fold = sys.argv[2]
schedule = json.loads(path.read_text(encoding="utf-8"))
expected = {f"{step}:{group}" for step in range(5, 50) for group in range(8)}
if set(schedule) != expected:
    raise SystemExit(f"{fold}: schedule keys differ from the 360 preregistered slots")
values = [int(schedule[key]) for key in sorted(schedule)]
if set(values) - {6, 9, 12, 21}:
    raise SystemExit(f"{fold}: schedule contains a non-preregistered K")
if sum(values) != 9 * len(values):
    raise SystemExit(f"{fold}: schedule does not have exact average K=9")
print(f"schedule_ok fold={fold} counts={dict(sorted(Counter(values).items()))}")
PY

mkdir -p "$OUT_ROOT"

# Dense endpoint reference under the identical prompt/seed/sampler contract.
python -m coframe.cli \
  --prompt "$PROMPT" --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --guidance-scale 5 --flow-shift 3 \
  --method dense \
  --local-files-only \
  --output-dir "$OUT_ROOT" \
  --run-name "${FOLD_ID}_dense"

# Existing Adaptive-K step/group runtime deploys the LOPO schedule. Group
# diagnostics evaluate only the assigned K from the same group input and are
# discarded; they do not select or change a budget.
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
  --adaptive-k-schedule-json "$SCHEDULE" \
  --calibrated-budget-probe-mode current \
  --local-files-only \
  --output-dir "$OUT_ROOT" \
  --run-name "${FOLD_ID}_calibrated_step_block"
